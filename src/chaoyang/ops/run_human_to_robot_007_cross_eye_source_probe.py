"""Fixed four-frame physical-right to physical-left table-source diagnostic."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import fit_heldout_homography, table_mask
from chaoyang.ops.run_human_to_robot_007_late_donor_probe import target_input
from chaoyang.ops.run_human_to_robot_007_spatial_donor_check import matched_fit_inliers, spatial_support

TASK = "human_to_robot_007_cross_eye_source_probe_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/cross_eye_source_probe_v1"
FRAMES = (181, 184, 192, 196)
DOMAIN = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                      "lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/DOMAIN_MANIFEST.json")


def read_pair(cap: cv2.VideoCapture, frame: int) -> tuple[np.ndarray, np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    okay, sbs = cap.read()
    if not okay or sbs is None or sbs.shape != (1536, 4096, 3):
        raise RuntimeError(f"SBS_FRAME_OR_DOMAIN_MISSING:{frame}")
    right = cv2.resize(sbs[:, :2048], (1280, 960), interpolation=cv2.INTER_AREA)
    left = cv2.resize(sbs[:, 2048:], (1280, 960), interpolation=cv2.INTER_AREA)
    return left, right


def domain_residual(left: np.ndarray, pinned: np.ndarray) -> dict:
    if left.shape != pinned.shape or left.shape != (960, 1280, 3):
        raise ValueError("PINNED_LEFT_DOMAIN_MISMATCH")
    delta = np.abs(left.astype(np.int16) - pinned.astype(np.int16))
    return {"mean_abs_channel": float(delta.mean()),
            "p99_abs_channel": float(np.percentile(delta, 99)),
            "max_abs_channel": int(delta.max())}


def right_table_proxy(right: np.ndarray) -> np.ndarray:
    # A conservative RGB appearance proxy, not independently annotated surface truth.
    zero = np.zeros(right.shape[:2], dtype=bool)
    return table_mask(right, zero, donor_frame=184)


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    domain = load_json(DOMAIN)
    if domain.get("frame_count") != 378 or domain.get("source_index") != 1:
        raise RuntimeError("PINNED_DOMAIN_DRIFT")
    source_ref = domain["source_stereo"]
    source = Path(source_ref["path"])
    if artifact_ref(source) != source_ref:
        raise RuntimeError("SOURCE_STEREO_SHA_DRIFT")
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 378:
        raise RuntimeError("SOURCE_VIDEO_DECODE_OR_COUNT_FAILED")
    rows = []
    DEST.mkdir(parents=True)
    try:
        for frame in FRAMES:
            target = target_input(frame)
            decoded_left, right = read_pair(cap, frame)
            residual = domain_residual(decoded_left, target["raw"])
            # Decoding backends may differ by a few LSBs, but not by a crop, eye or frame.
            if residual["mean_abs_channel"] > 2.5 or residual["p99_abs_channel"] > 8:
                raise RuntimeError(f"LEFT_EYE_BINDING_FAILED:{frame}:{residual}")
            right_table = right_table_proxy(right)
            matrix, geometry = fit_heldout_homography(
                right, target["raw"], right_table, target["table"])
            row = {"source_frame_id": frame, "left_binding": residual,
                   "geometry": geometry, "eligible_write_pixels": int(target["eligible"].sum()),
                   "source_right_table_proxy_pixels": int(np.count_nonzero(right_table)),
                   "spatially_supported_eligible_pixels": 0,
                   "source_right_visible_proxy_only": True, "inputs": target["refs"]}
            if matrix is not None:
                replay, source_points, target_points = matched_fit_inliers(
                    right, target["raw"], right_table, target["table"])
                normalize = lambda value: value / value[2, 2]
                if not np.allclose(normalize(matrix), normalize(replay), rtol=1e-6, atol=1e-6):
                    raise RuntimeError(f"CROSS_EYE_MATCH_REPLAY_DRIFT:{frame}")
                support, target_hull, source_hull = spatial_support(
                    matrix, source_points, target_points, right_table, target["eligible"])
                row.update(fit_inliers=int(len(source_points)),
                           target_hull_pixels=int(target_hull.sum()),
                           warped_source_hull_pixels=int(source_hull.sum()),
                           spatially_supported_eligible_pixels=int(support.sum()),
                           status="GEOMETRY_PASSED_APPEARANCE_PROXY_ONLY")
            else:
                support = np.zeros((960, 1280), dtype=bool)
                row["status"] = "GEOMETRY_REJECTED"
            support_path = DEST / f"SUPPORT_{frame}.png"
            if not cv2.imwrite(str(support_path), support.astype(np.uint8) * 255):
                raise RuntimeError("SUPPORT_WRITE_FAILED")
            row["support"] = artifact_ref(support_path)
            rows.append(row)
    finally:
        cap.release()
    result = {"schema_version": "HUMAN_TO_ROBOT_007_CROSS_EYE_SOURCE_PROBE_V1",
              "task_id": TASK, "session_id": "get_potato_chips_0915_007",
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "domain": artifact_ref(DOMAIN), "source_stereo": source_ref,
              "frames": rows, "execution": "EXECUTED", "structure": "PASS",
              "quality": "SOURCE_DIAGNOSTIC_ONLY", "adoption": "NOT_ADOPTED",
              "claim_limit": "A single cross-eye appearance proxy with fit-inlier spatial support; no true hidden table, Clean, metric depth, training or product authority.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "frames": [{"frame": row["source_frame_id"],
          "status": row["status"], "supported": row["spatially_supported_eligible_pixels"]}
          for row in rows]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
