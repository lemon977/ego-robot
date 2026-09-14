#!/usr/bin/env python3
"""Build future-2D projection candidates from a V5.2 visual Robot trajectory.

This is not a final Visual Aux bundle: the shared Raw/Robotized
``rgb_training_valid_mask`` is intentionally absent until a compositor has
passed the frozen contact/occlusion audit.  The output therefore remains a
projection/window upper bound and cannot start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "exact78-visual-retarget-projection-candidate-v52-v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    display = str(path.relative_to(relative_to.resolve())) if relative_to else str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def exact(root: Path, item: dict[str, Any], label: str) -> Path:
    p = Path(item["path"])
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.is_file() or p.is_symlink() or p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
        raise RuntimeError(f"{label}: exact ref mismatch: {p}")
    return p


def video_size(path: Path) -> tuple[int, int, int]:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path),
    ], text=True))["streams"][0]
    return int(data["width"]), int(data["height"]), int(data["nb_read_frames"])


def max_false_run(values: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        run = 0 if value else run + 1
        best = max(best, run)
    return best


def project(c2w: np.ndarray, intrinsics: np.ndarray, transforms: np.ndarray, valid: np.ndarray,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(c2w)
    uv = np.full((count, 2, 2), np.nan, dtype=np.float32)
    projected_valid = np.zeros((count, 2), dtype=bool)
    for frame in range(count):
        world_to_camera = np.linalg.inv(c2w[frame])
        for side in range(2):
            if not valid[frame, side] or not np.isfinite(transforms[frame, side]).all():
                continue
            world = transforms[frame, side, :3, 3]
            camera = world_to_camera[:3, :3] @ world + world_to_camera[:3, 3]
            if not np.isfinite(camera).all() or camera[2] <= 1e-6:
                continue
            pixel_h = intrinsics[frame] @ camera
            pixel = pixel_h[:2] / pixel_h[2]
            if not np.isfinite(pixel).all():
                continue
            if not (0.0 <= pixel[0] <= width - 1 and 0.0 <= pixel[1] <= height - 1):
                continue
            uv[frame, side] = pixel.astype(np.float32)
            projected_valid[frame, side] = True
    return uv, projected_valid


def future_arrays(uv: np.ndarray, point_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    count = len(uv)
    xy = np.full((count, 50, 2, 2), np.nan, dtype=np.float32)
    valid = np.zeros((count, 50, 2), dtype=bool)
    current = point_valid.all(axis=1)
    starts = []
    for frame in range(count):
        for horizon in range(50):
            source = frame + horizon + 1
            if source >= count:
                continue
            valid[frame, horizon] = point_valid[source]
            xy[frame, horizon, point_valid[source]] = uv[source, point_valid[source]]
        if current[frame] and all(
            int(valid[frame, :, side].sum()) >= 40 and max_false_run(valid[frame, :, side]) <= 5
            for side in range(2)
        ):
            starts.append(frame)
    return xy, valid, current, starts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-result", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    robot_result_path = args.robot_result.resolve()
    robot_root = robot_result_path.parent
    result = load(robot_result_path)
    if result.get("status") != "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY" or result.get("authority") is not False:
        raise RuntimeError("only V5.2 pose-only review candidates are accepted")
    if result.get("control_ground_truth") is not False or result.get("action_sidecar_published") is not False:
        raise RuntimeError("Robot candidate violates visual-only claim boundary")
    sidecar_path = exact(robot_root, result["outputs"]["trajectory_sidecar"], "trajectory sidecar")
    lineage_path = exact(robot_root, result["outputs"]["lineage"], "lineage")
    lineage = load(lineage_path)
    temporal_ref = lineage["input_manifest"]["hawor_chain"]["temporal_output"]
    hawor_path = exact(Path("/"), temporal_ref, "temporal HaWoR")
    source_video_ref = lineage["input_manifest"]["source_video"]
    source_video = exact(Path("/"), source_video_ref, "source video")

    with np.load(sidecar_path, allow_pickle=False) as z:
        transforms = np.asarray(z["T_actual_hand_root_world"], dtype=np.float64)
        robot_valid = np.asarray(z["valid_side_frame"], dtype=bool)
        if bool(z["control_ground_truth"]) or bool(z["metric_object_geometry"]):
            raise RuntimeError("forbidden truth bit enabled")
    with np.load(hawor_path, allow_pickle=False) as z:
        c2w = np.asarray(z["c2w"], dtype=np.float64)
        intrinsics = np.asarray(z["intrinsics"], dtype=np.float64)
        frames = np.asarray(z["original_frame_indices"], dtype=np.int64)
    count = int(result["frame_count"])
    width, height, decoded = video_size(source_video)
    if decoded != count or transforms.shape != (count, 2, 4, 4) or robot_valid.shape != (count, 2):
        raise RuntimeError("frame/trajectory closure mismatch")
    if c2w.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3) or not np.array_equal(frames, np.arange(count)):
        raise RuntimeError("HaWoR camera chain closure mismatch")
    uv, point_valid = project(c2w, intrinsics, transforms, robot_valid, width, height)
    original, future_valid, current, starts = future_arrays(uv, point_valid)
    normalized = original.copy()
    normalized[..., 0] /= max(width - 1, 1)
    normalized[..., 1] /= max(height - 1, 1)

    final = args.output_root.resolve() / result["session"]
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"immutable output exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{result['session']}.partial.", dir=final.parent))
    try:
        candidate = staging / "VISUAL_RETARGET_PROJECTION_CANDIDATE.npz"
        np.savez_compressed(
            candidate,
            schema_version_utf8=np.frombuffer(SCHEMA.encode(), dtype=np.uint8),
            frame_ids=np.arange(count, dtype=np.int64),
            current_2d_xy_original=uv,
            current_endpoint_valid=point_valid,
            future_2d_xy_original=original,
            future_2d_xy_normalized=normalized,
            future_2d_valid=future_valid,
            current_frame_valid=current,
            projection_upper_bound_h50_starts=np.asarray(starts, dtype=np.int64),
            image_width=np.asarray(width, dtype=np.int64),
            image_height=np.asarray(height, dtype=np.int64),
            label_source_utf8=np.frombuffer(b"VISUAL_RETARGET_PROJECTION", dtype=np.uint8),
            endpoint_definition_utf8=np.frombuffer(b"T_actual_hand_root_world translation; side_order=left,right", dtype=np.uint8),
            normalization_utf8=np.frombuffer(b"x/(W-1),y/(H-1); range=[0,1]", dtype=np.uint8),
            control_ground_truth=np.asarray(False, dtype=bool),
        )
        out_result = {
            "schema_version": "exact78-visual-retarget-projection-candidate-result-v52-v1",
            "created_at": now(),
            "status": "BLOCKED_PREREQ_ROBOTIZED_RGB_VALID_MASK_AND_GOLDSET",
            "task": result["task"], "session": result["session"], "frame_count": count,
            "image_size": [width, height],
            "coordinate_chain": "T_actual_hand_root_world -> inv(c2w[t]) -> camera -> intrinsics[t] -> selected-left pixels",
            "camera_motion_policy": "world-first trajectory; per-frame camera projection changes only observation",
            "label_source": "VISUAL_RETARGET_PROJECTION", "control_ground_truth": False,
            "projection": {
                "valid_current_endpoint_rows": int(point_valid.sum()),
                "valid_current_dual_frames": int(current.sum()),
                "future_valid_endpoint_steps": int(future_valid.sum()),
                "h50_window_upper_bound": len(starts),
            },
            "inputs": {
                "robot_result": ref(robot_result_path), "trajectory_sidecar": ref(sidecar_path),
                "lineage": ref(lineage_path), "temporal_hawor": ref(hawor_path), "source_video": ref(source_video),
            },
            "output": ref(candidate, relative_to=staging),
            "missing_for_final_bundle": [
                "GOLDSET_PASSED_ROBOTIZED_RGB", "SHARED_RAW_ROBOTIZED_RGB_TRAINING_VALID_MASK",
                "PAIRED_RGB_SELECTORS",
            ],
            "claim_limit": "Projection/H50 upper bound only; not an admitted training bundle, checkpoint, action, contact, or deployment authority.",
        }
        (staging / "RESULT.json").write_text(json.dumps(out_result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": out_result["status"], "session": result["session"], "h50_upper_bound": len(starts), "output": str(final)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
