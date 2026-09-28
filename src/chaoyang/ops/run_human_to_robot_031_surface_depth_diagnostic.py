"""Bounded same-pixel MANO front-surface versus Stereo optical-Z diagnostic."""
from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json

TASK = "human_to_robot_031_surface_depth_diagnostic_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/motion/surface_depth_v1"
V3 = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
SOURCE = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
MASK_ROOT = V3 / "lanes/exact78/masks031_right_epoch7_v1/play_cards_0915_031/human"
DOMAIN = V3 / "lanes/exact78/prepare_full_v1/play_cards_0915_031/DOMAIN_MANIFEST.json"
DEPTH_ROOT = REPO_ROOT / ("_run/current/human_to_robot_quality_closure_s1_20260923/"
                         "attempts/attempt_0001/lanes/geometry_contact/depth_full_v1/play_cards_0915_031")
DEPTH_RESULT = DEPTH_ROOT / "RESULT.json"
FACES_LAUNCHER = REPO_ROOT / "src/chaoyang/ops/hawor_python.sh"
HAWOR_ROOT = REPO_ROOT / "vendor/HaWoR"
FRAMES = tuple(int(x) for x in np.rint(np.linspace(47, 148, 16)).astype(int))


def rasterize_optical_z(vertices: np.ndarray, faces: np.ndarray, k: np.ndarray,
                        shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Perspective-correct nearest positive optical Z for a triangle mesh."""
    height, width = shape
    zbuf = np.full(shape, np.inf, np.float32)
    vertices = np.asarray(vertices, np.float64)
    projected = (np.asarray(k, np.float64) @ vertices.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    for indices in np.asarray(faces, np.int32):
        triangle = uv[indices]
        depths = vertices[indices, 2]
        if not np.isfinite(triangle).all() or not np.isfinite(depths).all() or np.min(depths) <= 0:
            continue
        left = max(0, int(np.floor(np.min(triangle[:, 0]))))
        right = min(width - 1, int(np.ceil(np.max(triangle[:, 0]))))
        top = max(0, int(np.floor(np.min(triangle[:, 1]))))
        bottom = min(height - 1, int(np.ceil(np.max(triangle[:, 1]))))
        if right < left or bottom < top:
            continue
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
        px, py = xx + 0.5, yy + 0.5
        w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue
        inverse_z = w0 / depths[0] + w1 / depths[1] + w2 / depths[2]
        values = np.divide(1.0, inverse_z, out=np.full_like(inverse_z, np.inf), where=inverse_z > 0)
        target = zbuf[top:bottom + 1, left:right + 1]
        np.minimum(target, np.where(inside, values, np.inf), out=target)
    return zbuf


def admissible_pixels(model_z: np.ndarray, stereo: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    if model_z.shape != (480, 640) or mask.shape != (960, 1280):
        raise RuntimeError("IMAGE_DOMAIN_MISMATCH")
    hand = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    hand = cv2.erode(hand, np.ones((5, 5), np.uint8)) > 0
    depth = np.asarray(stereo["depth_m"], np.float32)
    valid = np.asarray(stereo["valid"], bool) & np.asarray(stereo["lr_consistent"], bool)
    finite = valid & np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(finite, depth, 0)
    local_min = cv2.erode(safe_depth, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(safe_depth, np.ones((3, 3), np.uint8))
    continuous = (local_min > 0) & (local_max - local_min < 0.025)
    return hand & finite & continuous & np.isfinite(model_z) & (model_z > 0)


def load_faces() -> np.ndarray:
    script = ("import io,os,sys,numpy as np;"
              "sys.path.insert(0,os.environ['CHA0YANG_HAWOR_ROOT']);"
              "from hawor.utils.process import get_mano_faces;"
              "buf=io.BytesIO();np.save(buf,get_mano_faces(),allow_pickle=False);"
              "sys.stdout.buffer.write(buf.getvalue())")
    done = subprocess.run([str(FACES_LAUNCHER), "-c", script], cwd=HAWOR_ROOT,
                          capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError("MANO_FACES_LOAD_FAILED:" + done.stderr.decode("utf-8", "replace")[-500:])
    faces = np.load(io.BytesIO(done.stdout), allow_pickle=False)
    if faces.shape != (1538, 3) or faces.min() < 0 or faces.max() >= 778:
        raise RuntimeError("MANO_TOPOLOGY_DRIFT")
    return faces


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    failure_path = ATTEMPT / "RUNTIME_FAILURE.json"
    retry_path = ATTEMPT / "RUNTIME_RETRY_SIGNATURE.json"
    if not failure_path.is_file() or retry_path.exists():
        raise RuntimeError("RUNTIME_RETRY_EVIDENCE_INVALID")
    atomic_json(retry_path, {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_RUNTIME_RETRY_V1",
                             "task_id": TASK, "prior_failure": artifact_ref(failure_path),
                             "runner": artifact_ref(Path(__file__)),
                             "reason": "Use BytesIO for numpy.save on non-seekable stdout pipe",
                             "model": artifact_ref(SOURCE), "domain": artifact_ref(DOMAIN),
                             "stereo_result": artifact_ref(DEPTH_RESULT),
                             "output_schema": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DIAGNOSTIC_V1"})
    domain = load_json(DOMAIN)
    depth_receipt = load_json(DEPTH_RESULT)
    if domain.get("frame_count") != 149 or domain.get("source_index") != 1:
        raise RuntimeError("SOURCE_DOMAIN_DRIFT")
    if depth_receipt.get("local_stereo_metric_dev") is not True or depth_receipt.get("execution_scope") != "FULL_SESSION":
        raise RuntimeError("STEREO_DEVELOPMENT_GATE_MISSING")
    with np.load(SOURCE, allow_pickle=False) as source:
        if (source["vertices_3d_camera"].shape != (2, 149, 778, 3)
                or list(source["anatomical_side_names"]) != ["left", "right"]
                or str(source["source_domain"]) != "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY"):
            raise RuntimeError("HAWOR_SOURCE_DRIFT")
        vertices = np.asarray(source["vertices_3d_camera"][1], np.float32)
        predicted = np.asarray(source["predicted_valid"][1], bool)
        timestamps = np.asarray(source["timestamp_ns"], np.int64)
        source_k = np.asarray(source["intrinsics"], np.float64)
    faces = load_faces()
    DEST.mkdir(parents=True)
    video = DEST / "031_MANO_STEREO_SURFACE_Z_16FRAME_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_NOT_OPEN")
    rows = []
    try:
        for frame in FRAMES:
            mask_path = MASK_ROOT / f"{frame:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (960, 1280) or mask.dtype != np.uint16 or mask.max() > 1:
                raise RuntimeError(f"MASK_DOMAIN_DRIFT:{frame}")
            raw_path = Path(domain["frames"][frame]["rgb"])
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (960, 1280, 3):
                raise RuntimeError(f"RAW_DOMAIN_DRIFT:{frame}")
            depth_path = DEPTH_ROOT / "frames" / f"{frame:06d}.npz"
            with np.load(depth_path, allow_pickle=False) as z:
                stereo = {key: np.asarray(z[key]) for key in ("depth_m", "valid", "lr_consistent")}
                k_depth = np.asarray(z["physical_left_intrinsics"], np.float64)
            if (not predicted[frame] or not np.isfinite(vertices[frame]).all()
                    or not np.allclose(k_depth[:2, :2], source_k[frame, :2, :2] / 2, atol=1e-3)):
                raise RuntimeError(f"MODEL_OR_K_DOMAIN_DRIFT:{frame}")
            model_z = rasterize_optical_z(vertices[frame], faces, k_depth)
            admitted = admissible_pixels(model_z, stereo, mask == 1)
            delta = stereo["depth_m"][admitted] - model_z[admitted]
            row = {"source_frame_id": frame, "timestamp_ns": int(timestamps[frame]),
                   "model_silhouette_px": int(np.isfinite(model_z).sum()),
                   "stereo_valid_in_model_silhouette_px": int((np.isfinite(model_z) & stereo["valid"]).sum()),
                   "same_pixel_surface_samples": int(delta.size),
                   "stereo_minus_mano_median_m": float(np.median(delta)) if delta.size else None,
                   "stereo_minus_mano_p10_m": float(np.percentile(delta, 10)) if delta.size else None,
                   "stereo_minus_mano_p90_m": float(np.percentile(delta, 90)) if delta.size else None,
                   "model": artifact_ref(SOURCE), "mask": artifact_ref(mask_path),
                   "stereo": artifact_ref(depth_path), "raw": artifact_ref(raw_path)}
            rows.append(row)
            left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            right = left.copy()
            paint = np.zeros((480, 640, 3), np.uint8)
            paint[admitted & (stereo["depth_m"] - model_z < -0.03)] = (255, 80, 50)
            paint[admitted & (np.abs(stereo["depth_m"] - model_z) <= 0.03)] = (50, 220, 50)
            paint[admitted & (stereo["depth_m"] - model_z > 0.03)] = (50, 80, 255)
            right[admitted] = cv2.addWeighted(right, 0.45, paint, 0.55, 0)[admitted]
            cv2.putText(left, f"031 f{frame} RAW | SAM ROI upstream", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            cv2.putText(right, f"Stereo - MANO surface Z | n={delta.size} | diagnostic only", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 2)
            writer.write(np.hstack((left, right)))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(video))
    count = 0
    while True:
        okay, _ = cap.read()
        if not okay:
            break
        count += 1
    cap.release()
    if count != len(FRAMES):
        raise RuntimeError(f"VIDEO_DECODE_COUNT_INVALID:{count}")
    medians = [row["stereo_minus_mano_median_m"] for row in rows if row["stereo_minus_mano_median_m"] is not None]
    result = {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DIAGNOSTIC_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": datetime.now().astimezone().isoformat(),
              "frame_ids": list(FRAMES), "frame_count": 149, "sampled_frames": len(FRAMES),
              "frames_with_surface_samples": len(medians), "total_surface_samples": sum(r["same_pixel_surface_samples"] for r in rows),
              "frame_median_of_signed_medians_m": float(np.median(medians)) if medians else None,
              "runtime_retry_signature": artifact_ref(retry_path),
              "rows": rows, "inputs": {"model": artifact_ref(SOURCE), "domain": artifact_ref(DOMAIN),
                                         "stereo_result": artifact_ref(DEPTH_RESULT)},
              "review_video": {**artifact_ref(video), "decoded_frames": count},
              "execution": "EXECUTED", "structure": "PASS", "quality": "DEVELOPMENT_DIAGNOSTIC_ONLY",
              "adoption": "NOT_ADOPTED", "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False,
              "claim_limit": "Same-pixel MANO/Stereo visible-surface optical-Z diagnostic; SAM is HaWoR ROI upstream, object occlusion and true correspondence are unverified. Not wrist-centre accuracy, correction, Contact or product authority."}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "frames_with_samples": len(medians),
                      "total_samples": result["total_surface_samples"],
                      "frame_median_of_signed_medians_m": result["frame_median_of_signed_medians_m"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
