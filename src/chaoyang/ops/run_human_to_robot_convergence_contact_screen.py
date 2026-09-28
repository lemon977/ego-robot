#!/usr/bin/env python3
"""Actually sample 031 visible object geometry, without promoting inferred HaWoR to Contact truth."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/contact_screen_031/wave0"
INDEX = REPO / "tasks/current/INDEX.json"
HAWOR = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
DEPTH = REPO / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/depth_full_v1/play_cards_0915_031/frames"
MASKS = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_031_v1/task_object"
TIPS = (4, 8, 12, 16, 20)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)


def _write(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    with np.load(HAWOR, allow_pickle=False) as source:
        h = {key: np.asarray(source[key]) for key in source.files}
    if h["predicted_valid"].sum(axis=1).tolist() != [0, 102] or h["inferred"].sum(axis=1).tolist() != [0, 102]:
        raise ValueError("FROZEN_PROVENANCE_DRIFT")

    rows = []
    for frame in np.flatnonzero(h["predicted_valid"][1]).tolist():
        mask = cv2.imread(str(MASKS / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (960, 1280):
            raise ValueError(f"OBJECT_MASK_DOMAIN:{frame}")
        object_pixels = np.argwhere(mask > 0)
        with np.load(DEPTH / f"{frame:06d}.npz", allow_pickle=False) as archive:
            depth = np.asarray(archive["depth_m"], dtype=np.float64)
            valid = np.asarray(archive["valid"], dtype=bool) & np.asarray(archive["lr_consistent"], dtype=bool)
            k = np.asarray(archive["physical_left_intrinsics"], dtype=np.float64)
        for finger, joint in enumerate(TIPS):
            uv = np.asarray(h["joints_2d"][1, frame, joint], dtype=np.float64)
            row = {"frame_id": frame, "physical_side": 1, "finger_index": finger,
                   "qualification_checked": True, "geometry_screened": False,
                   "strict_contact_admissible": False, "reason": ""}
            if not np.isfinite(uv).all() or not len(object_pixels):
                row["reason"] = "NO_FINITE_TIP_OR_OBJECT_MASK"; rows.append(row); continue
            delta = object_pixels.astype(np.float64) - np.asarray([uv[1], uv[0]])
            best = int(np.argmin(np.sum(delta * delta, axis=1)))
            y_mask, x_mask = object_pixels[best]
            pixel_distance = float(np.sqrt(np.sum(delta[best] ** 2)))
            x_depth = int(np.clip(np.rint((x_mask - 0.5) / 2.0), 0, depth.shape[1] - 1))
            y_depth = int(np.clip(np.rint((y_mask - 0.5) / 2.0), 0, depth.shape[0] - 1))
            y0, y1 = max(0, y_depth - 2), min(depth.shape[0], y_depth + 3)
            x0, x1 = max(0, x_depth - 2), min(depth.shape[1], x_depth + 3)
            local = depth[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
            if pixel_distance > 40.0 or not local.size:
                row.update(reason="NO_NEARBY_VALID_VISIBLE_SURFACE", pixel_distance=float(pixel_distance))
                rows.append(row); continue
            z = float(np.median(local))
            point = np.asarray([(x_depth - k[0, 2]) * z / k[0, 0],
                                (y_depth - k[1, 2]) * z / k[1, 1], z])
            tip = np.asarray(h["joints_3d_camera"][1, frame, joint], dtype=np.float64)
            row.update(geometry_screened=True, reason="MODEL_CONDITIONED_PROXIMITY_ONLY",
                       pixel_distance=pixel_distance, surface_x=float(point[0]), surface_y=float(point[1]),
                       surface_z=float(point[2]), model_tip_surface_distance_m=float(np.linalg.norm(tip - point)))
            rows.append(row)

    checked = len(rows)
    screened = sum(bool(row["geometry_screened"]) for row in rows)
    dtype = np.dtype([("frame_id", "i4"), ("physical_side", "i1"), ("finger_index", "i1"),
                      ("geometry_screened", "?"), ("pixel_distance", "f8"),
                      ("model_tip_surface_distance_m", "f8"), ("reason", "U64")])
    table = np.zeros(checked, dtype=dtype)
    table["pixel_distance"] = np.nan; table["model_tip_surface_distance_m"] = np.nan
    for i, row in enumerate(rows):
        for key in ("frame_id", "physical_side", "finger_index", "geometry_screened", "reason"):
            table[key][i] = row[key]
        if "pixel_distance" in row: table["pixel_distance"][i] = row["pixel_distance"]
        if "model_tip_surface_distance_m" in row: table["model_tip_surface_distance_m"][i] = row["model_tip_surface_distance_m"]
    table_path = OUT / "CONTACT_SCREEN_ROWS_V1.npz"
    _atomic_npz(table_path, rows=table, source_frame_id=h["original_frame_indices"], timestamp_ns=h["timestamp_ns"])
    result = {
        "schema_version": "HUMAN_TO_ROBOT_CONTACT_SCREEN_V1", "task_id": TASK,
        "session_id": "play_cards_0915_031",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS", "quality": "INCONCLUSIVE_MODEL_CONDITIONED",
        "improvement": "ACTUAL_GEOMETRY_SCREEN_REPLACED_PRIOR_ZERO_SCREEN", "adoption": "NOT_ADOPTED",
        "counts": {"unit": "finger_side_frame", "qualification_checked": checked,
                   "geometry_screened": screened, "strict_contact_admissible": 0,
                   "r1_executed_windows": 0},
        "first_strict_blocker": "HAWOR_INPUT_IS_INFERRED_OFFLINE_NONCAUSAL_AND_ABSOLUTE_TIP_Z_HAS_NO_CONTACT_AUTHORITY",
        "model_conditioned_proximity": {
            "executed": True, "distance_is_contact_truth": False,
            "nearest_object_mask_radius_px": 40.0, "surface_patch_radius_depth_px": 2,
        },
        "r1": {"execution": "NOT_STARTED", "reason": "NO_STRICT_CONTACT_ADMISSIBLE_WINDOW",
               "wrist_modified": False, "arm_modified": False, "q22_modified": False},
        "inputs": {"hawor": artifact_ref(HAWOR),
                   "depth_result": artifact_ref(DEPTH.parent / "RESULT.json"),
                   "mask_manifest": artifact_ref(MASKS.parent / "MASK_MANIFEST.json")},
        "rows": artifact_ref(table_path),
        "claim_limit": "Actual visible-surface sampling and model-conditioned proximity only; no strict Contact, R1, external metric, control or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    _write(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], **result["counts"], "output": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
