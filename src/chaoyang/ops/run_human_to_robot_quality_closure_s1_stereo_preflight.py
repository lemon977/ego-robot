#!/usr/bin/env python3
"""CPU-only encoded-domain stereo preflight for the S1 target sessions."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref
from chaoyang.ops.analyze_0915_stereo_domain_preflight_v1 import load_pico_module
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    aggregate_metrics,
    foundation_disparity_adapter,
    frame_metrics,
    resize_only,
    robust_correspondences,
    split_source_index_eyes,
)

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
MANIFEST_ROOT = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1"
SESSIONS = {
    "play_cards_0915_031": MANIFEST_ROOT / "play_cards_0915_031/DOMAIN_MANIFEST.json",
    "get_potato_chips_0915_007": MANIFEST_ROOT / "get_potato_chips_0915_007/DOMAIN_MANIFEST.json",
}
WIDTH, HEIGHT = 1280, 960


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def sample_ids(frame_count: int, count: int = 12) -> list[int]:
    return [int(x) for x in np.unique(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=np.int64))]


def review_panel(left: np.ndarray, right: np.ndarray, lp: np.ndarray, rp: np.ndarray, frame_id: int, metrics: dict) -> np.ndarray:
    left_small = cv2.resize(left, (640, 480), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (640, 480), interpolation=cv2.INTER_AREA)
    panel = np.concatenate([left_small, right_small], axis=1)
    if len(lp):
        choose = np.linspace(0, len(lp) - 1, min(80, len(lp)), dtype=np.int64)
        for index in choose:
            a = tuple(np.rint(lp[index] * 0.5).astype(int))
            bxy = np.rint(rp[index] * 0.5).astype(int); b = (int(bxy[0] + 640), int(bxy[1]))
            color = (0, 200, 0) if abs(float(lp[index, 1] - rp[index, 1])) <= 4.0 else (0, 0, 255)
            cv2.line(panel, a, b, color, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (1280, 58), (0, 0, 0), -1)
    text = f"frame {frame_id} | encoded resize-only | no lens remap | matches {metrics['robust_matches']} | median |dy| {metrics['median_abs_vertical_px']}"
    cv2.putText(panel, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def run_session(session: str, manifest_path: Path, root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["image_domain"] != "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY":
        raise RuntimeError("IMAGE_DOMAIN_DRIFT")
    if manifest["lens_undistortion_applied"] or manifest["distortion_coefficients_consumed"]:
        raise RuntimeError("LENS_REMAP_FORBIDDEN")
    stereo = Path(manifest["source_stereo"]["path"]).resolve(strict=True)
    camera = Path(manifest["source_camera"]["path"]).resolve(strict=True)
    params = json.loads(camera.read_text(encoding="utf-8"))
    pico = load_pico_module()
    _eyes, eye_width, eye_height, baseline_m = pico.load_camera_params(camera)
    source_indices = pico.load_source_indices(camera)
    if source_indices != (1, 0):
        raise RuntimeError(f"PHYSICAL_EYE_ROUTING_DRIFT:{source_indices}")
    frame_count = int(manifest["frame_count"])
    selected = sample_ids(frame_count)
    selected_set = set(selected)
    capture = cv2.VideoCapture(str(stereo))
    if not capture.isOpened():
        raise RuntimeError(f"VIDEO_OPEN_FAILED:{stereo}")
    rows, verticals, disparities, panels = [], [], [], []
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok: break
        if frame.shape[:2] != (eye_height, eye_width * 2):
            raise RuntimeError(f"SBS_GEOMETRY_DRIFT:{frame.shape}")
        if decoded in selected_set:
            left, right = split_source_index_eyes(
                frame, eye_width=eye_width, eye_height=eye_height,
                physical_left_source_index=source_indices[0],
                physical_right_source_index=source_indices[1],
            )
            left = resize_only(left, width=WIDTH, height=HEIGHT)
            right = resize_only(right, width=WIDTH, height=HEIGHT)
            lp, rp = robust_correspondences(left, right)
            metrics = frame_metrics(lp, rp, width=WIDTH, height=HEIGHT)
            rows.append({"frame_id": decoded, **metrics})
            verticals.append(np.abs(lp[:, 1] - rp[:, 1]))
            disparities.append(lp[:, 0] - rp[:, 0])
            panels.append(review_panel(left, right, lp, rp, decoded, metrics))
        decoded += 1
    capture.release()
    if decoded != frame_count or len(rows) != len(selected):
        raise RuntimeError(f"DECODE_OR_SAMPLE_MISMATCH:{decoded}:{len(rows)}")
    aggregate = aggregate_metrics(rows, verticals, disparities)
    adapter = foundation_disparity_adapter(aggregate)
    all_left = []
    for row, values in zip(rows, verticals):
        # Values are ordered exactly as the robust matches for each frame; the
        # spatial split is summarized at frame level when raw points are absent.
        if len(values): all_left.append(values)
    scale_x, scale_y = WIDTH / eye_width, HEIGHT / eye_height
    intrinsics = {}
    for name in ("left", "right"):
        src = params[name]["intrinsics"]
        intrinsics[name] = {
            "fx": float(src["fx"] * scale_x), "fy": float(src["fy"] * scale_y),
            "cx": float(src["cx"] * scale_x), "cy": float(src["cy"] * scale_y),
        }
    metric_closed = bool(
        baseline_m > 0 and all(np.isfinite(list(value.values())).all() for value in intrinsics.values())
    )
    session_root = root / session
    session_root.mkdir(parents=True, exist_ok=True)
    video_path = session_root / "ENCODED_STEREO_PREFLIGHT_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (1280, 480))
    if not writer.isOpened(): raise RuntimeError(f"VIDEO_WRITER_FAILED:{video_path}")
    for panel in panels: writer.write(panel)
    writer.release()
    verify = cv2.VideoCapture(str(video_path)); review_count = 0
    while True:
        ok, _ = verify.read()
        if not ok: break
        review_count += 1
    verify.release()
    if review_count != len(selected): raise RuntimeError("REVIEW_DECODE_FAILED")
    status = "PASS_LOCAL_STEREO_METRIC_DEV_PREFLIGHT" if adapter["authorized"] and metric_closed else "BLOCKED_STEREO_SUCCESSOR"
    result = {
        "schema_version": "S1_SAME_SESSION_ENCODED_STEREO_PREFLIGHT_V1",
        "task_id": TASK, "session_id": session, "created_at": now(), "status": status,
        "image_domain_check": {
            "passed": True, "domain": manifest["image_domain"], "source_indices": {"left": 1, "right": 0},
            "lens_undistortion_applied": False, "distortion_coefficients_consumed": False,
            "decoded_frames": decoded, "sample_frames": selected,
        },
        "stereo_geometry_check": aggregate,
        "metric_conversion_check": {
            "passed_internal_development": metric_closed,
            "output_geometry": [WIDTH, HEIGHT], "scaled_intrinsics": intrinsics,
            "baseline_m": float(baseline_m), "baseline_authority": "SAME_SESSION_FACTORY_CALIBRATION_INTERNAL_DEV",
            "external_metric_authority": False,
        },
        "foundation_input_adapter": adapter,
        "rows": rows,
        "review_video": {"path": str(video_path), "sha256": sha256(video_path), "decoded_frames": review_count},
        "inputs": {"manifest": artifact_ref(manifest_path), "source_stereo": artifact_ref(stereo), "camera_params": artifact_ref(camera)},
        "gpu_used": False, "model_inference_run": False, "weights": "ABSENT",
        "claim_limit": "Encoded-domain CPU evidence only; no lens remap, no FoundationStereo quality, no external metric accuracy, no Contact authority.",
    }
    write_once(session_root / "RESULT.json", result)
    return {"status": status, "result": artifact_ref(session_root / "RESULT.json"), "review_video": artifact_ref(video_path)}


def main() -> int:
    root = ATTEMPT / "lanes/geometry_contact/encoded_stereo_preflight_v1"
    if root.exists(): raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{root}")
    sessions = {session: run_session(session, path, root) for session, path in SESSIONS.items()}
    result = {
        "schema_version": "S1_SAME_SESSION_ENCODED_STEREO_PREFLIGHT_SUMMARY_V1",
        "task_id": TASK, "created_at": now(), "sessions": sessions,
        "status": "EXECUTED", "gpu_used": False, "new_model_invocations": 0,
    }
    write_once(root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
