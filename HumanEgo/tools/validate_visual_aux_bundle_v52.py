#!/usr/bin/env python3
"""Validate an exact78 V5.2 paired visual-aux session bundle.

This validator deliberately rejects legacy Robot-action/policy labels.  The
only target is future 2-D visual-retarget projection, shared byte-for-byte by
HUMAN_RAW_RGB and ROBOTIZED_RGB.  ``control_ground_truth`` must be false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import cv2


SCHEMA = "exact78-visual-aux-session-bundle-v52-v1"
LABEL_SCHEMA = "exact78-visual-aux-labels-v52-v1"
BRANCHES = ("HUMAN_RAW_RGB", "ROBOTIZED_RGB")
FORBIDDEN_LABEL_KEYS = {
    "q", "action", "actions", "robot_action", "robot_action_sidecar",
    "joint_lower", "joint_upper", "wrist_T_camera",
}


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def exact_ref(root: Path, item: dict[str, Any], label: str) -> Path:
    path = Path(str(item.get("path", "")))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{label}: regular non-symlink file required: {path}")
    if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
        raise ContractError(f"{label}: byte/SHA mismatch")
    return path


def max_false_run(values: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(values, dtype=bool).reshape(-1):
        run = 0 if bool(value) else run + 1
        best = max(best, run)
    return best


def eligible_starts(current: np.ndarray, future: np.ndarray, rgb_mask: np.ndarray) -> list[int]:
    starts = []
    for index in range(len(current)):
        if not bool(current[index]) or not bool(rgb_mask[index].any()):
            continue
        side_ok = []
        for side in range(2):
            validity = future[index, :, side]
            side_ok.append(int(validity.sum()) >= 40 and max_false_run(validity) <= 5)
        if all(side_ok):
            starts.append(index)
    return starts


def validate_labels(path: Path, frame_count: int, width: int, height: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        forbidden = sorted(keys & FORBIDDEN_LABEL_KEYS)
        if forbidden:
            raise ContractError(f"legacy action/control keys forbidden: {forbidden}")
        required = {
            "schema_version_utf8", "frame_ids", "future_2d_xy_original",
            "future_2d_xy_normalized", "future_2d_valid", "current_frame_valid",
            "rgb_training_valid_mask", "label_source_utf8", "control_ground_truth",
            "image_width", "image_height", "normalization_utf8",
        }
        missing = sorted(required - keys)
        if missing:
            raise ContractError(f"label arrays missing: {missing}")
        schema = bytes(np.asarray(z["schema_version_utf8"], dtype=np.uint8)).decode()
        source = bytes(np.asarray(z["label_source_utf8"], dtype=np.uint8)).decode()
        normalization = bytes(np.asarray(z["normalization_utf8"], dtype=np.uint8)).decode()
        if schema != LABEL_SCHEMA or source != "VISUAL_RETARGET_PROJECTION":
            raise ContractError("label schema/source mismatch")
        if bool(np.asarray(z["control_ground_truth"]).item()):
            raise ContractError("control_ground_truth must be false")
        if int(np.asarray(z["image_width"]).item()) != width or int(np.asarray(z["image_height"]).item()) != height:
            raise ContractError("label image dimensions differ from manifest")
        if normalization != "x/(W-1),y/(H-1); range=[0,1]":
            raise ContractError("normalization definition mismatch")
        frames = np.asarray(z["frame_ids"])
        original = np.asarray(z["future_2d_xy_original"])
        normalized = np.asarray(z["future_2d_xy_normalized"])
        valid = np.asarray(z["future_2d_valid"])
        current = np.asarray(z["current_frame_valid"])
        rgb_mask = np.asarray(z["rgb_training_valid_mask"])

    if frames.dtype != np.int64 or not np.array_equal(frames, np.arange(frame_count, dtype=np.int64)):
        raise ContractError("frame_ids must be exact int64 0..T-1")
    if original.shape != (frame_count, 50, 2, 2) or original.dtype != np.float32:
        raise ContractError("future_2d_xy_original must be float32 [T,50,2,2]")
    if normalized.shape != original.shape or normalized.dtype != np.float32:
        raise ContractError("future_2d_xy_normalized must be float32 [T,50,2,2]")
    if valid.shape != (frame_count, 50, 2) or valid.dtype != np.bool_:
        raise ContractError("future_2d_valid must be bool [T,50,2]")
    if current.shape != (frame_count,) or current.dtype != np.bool_:
        raise ContractError("current_frame_valid must be bool [T]")
    if rgb_mask.shape != (frame_count, height, width) or rgb_mask.dtype != np.bool_:
        raise ContractError("rgb_training_valid_mask must be bool [T,H,W]")
    finite_original = np.isfinite(original).all(axis=-1)
    finite_normalized = np.isfinite(normalized).all(axis=-1)
    if not np.array_equal(finite_original, valid) or not np.array_equal(finite_normalized, valid):
        raise ContractError("invalid future endpoint must be NaN; valid endpoint must be finite")
    expected = original.copy()
    expected[..., 0] /= max(width - 1, 1)
    expected[..., 1] /= max(height - 1, 1)
    if not np.allclose(normalized[valid], expected[valid], atol=1e-6, rtol=0):
        raise ContractError("normalized coordinates are not derived from original coordinates")
    if np.any((normalized[valid] < 0) | (normalized[valid] > 1)):
        raise ContractError("valid normalized endpoints must lie within [0,1]")
    if np.any(current & ~rgb_mask.reshape(frame_count, -1).any(axis=1)):
        raise ContractError("current_frame_valid cannot use an empty RGB-valid frame")
    starts = eligible_starts(current, valid, rgb_mask)
    return {
        "frame_count": frame_count,
        "valid_current_frames": int(current.sum()),
        "valid_future_endpoint_steps": int(valid.sum()),
        "eligible_h50_starts": starts,
        "eligible_h50_window_count": len(starts),
        "rgb_valid_pixel_fraction": float(rgb_mask.mean()),
    }


def validate_bundle(root: Path) -> dict[str, Any]:
    manifest_path = root / "VISUAL_AUX_SESSION_MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise ContractError("bundle schema mismatch")
    if manifest.get("task") not in {"chips", "poker"} or manifest.get("split") not in {"train", "validation", "test", "heldout"}:
        raise ContractError("task/split mismatch")
    if manifest.get("seed") != 7 or manifest.get("pred_horizon") != 50:
        raise ContractError("seed/H50 contract mismatch")
    if manifest.get("label_source") != "VISUAL_RETARGET_PROJECTION" or manifest.get("control_ground_truth") is not False:
        raise ContractError("visual-only claim boundary mismatch")
    if manifest.get("only_branch_variable") != "RGB_BYTES":
        raise ContractError("only branch variable must be RGB bytes")
    frame_count = int(manifest["frame_count"])
    width, height = int(manifest["image_width"]), int(manifest["image_height"])
    label_path = exact_ref(root, manifest["labels"], "labels")
    report = validate_labels(label_path, frame_count, width, height)
    branches = manifest.get("branches", {})
    if set(branches) != set(BRANCHES):
        raise ContractError("exactly HUMAN_RAW_RGB and ROBOTIZED_RGB branches are required")
    branch_records = {}
    for branch in BRANCHES:
        item = branches[branch]
        selector = exact_ref(root, item["selector"], f"{branch} selector")
        payload = load_json(selector)
        if payload.get("branch") != branch or payload.get("session_id") != manifest.get("session_id"):
            raise ContractError(f"{branch} selector identity mismatch")
        if payload.get("frame_ids") != list(range(frame_count)):
            raise ContractError(f"{branch} selector frame identity mismatch")
        records = payload.get("frames")
        if not isinstance(records, list) or len(records) != frame_count:
            raise ContractError(f"{branch} selector records mismatch")
        image_refs = []
        for index, row in enumerate(records):
            if row.get("frame_id") != index:
                raise ContractError(f"{branch} selector frame order mismatch")
            image_path = exact_ref(root, row["rgb"], f"{branch}:frame:{index}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ContractError(f"{branch}:frame:{index} decode/dimension mismatch")
            image_refs.append(image_path)
        branch_records[branch] = {
            "selector": selector, "images": image_refs,
            "frame_ids": [row["frame_id"] for row in records],
        }
    if branch_records[BRANCHES[0]]["frame_ids"] != branch_records[BRANCHES[1]]["frame_ids"]:
        raise ContractError("Raw/Robotized frame sets differ")
    declared = manifest.get("eligible_h50_starts")
    if declared != report["eligible_h50_starts"]:
        raise ContractError("declared and recomputed H50 windows differ")
    if manifest.get("labels_sha_shared_by_branches") != sha256(label_path):
        raise ContractError("shared label SHA binding mismatch")
    report.update({
        "status": "PASS_VISUAL_AUX_SESSION_BUNDLE",
        "task": manifest["task"], "session_id": manifest["session_id"], "split": manifest["split"],
        "labels_sha256": sha256(label_path), "manifest_sha256": sha256(manifest_path),
        "branch_frame_identity": True, "only_branch_variable": "RGB_BYTES",
        "control_ground_truth": False,
    })
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    args = ap.parse_args()
    report = validate_bundle(args.bundle.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
