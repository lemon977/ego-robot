#!/usr/bin/env python3
"""Build a side-conservative 031 observability receipt from RGB-domain masks."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_root_cause_gated_r2_20260923"
ATTEMPT = REPO / "_run/current" / TASK / "attempts/attempt_0001"
MASK = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/masks031_right_epoch7_v1/play_cards_0915_031/MASK_MANIFEST.json"
MOTION = ATTEMPT / "lanes/lane2_motion/recovered_031_wave0/HAND_MOTION_V1.npz"


def ref(path: Path) -> dict:
    path = path.resolve(strict=True); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> int:
    manifest = json.loads(MASK.read_text())
    root = Path(manifest["mask_root"])
    role = manifest["roles"][0]["role"]
    rows = []
    for frame in range(int(manifest["frame_count"])):
        image = cv2.imread(str(root / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"missing independent proxy frame {frame}")
        mask = image > 0
        yy, xx = np.where(mask)
        rows.append({
            "frame_id": frame,
            "hand_region_visible": bool(mask.any()),
            "region_area_px": int(mask.sum()),
            "touches_image_boundary": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
            "anatomical_side": "UNKNOWN",
            "left_observability": "UNKNOWN",
            "right_observability": "UNKNOWN",
            "reason": "SAM_TEMPORAL_HAND_REGION_HAS_NO_INDEPENDENT_SIDE_AUTHORITY",
        })
    with np.load(MOTION, allow_pickle=False) as archive:
        roi = np.asarray(archive["roi_valid"], dtype=bool)
        prediction = np.asarray(archive["predicted_valid"], dtype=bool)
    visible = np.asarray([row["hand_region_visible"] for row in rows])
    result = {
        "schema_version": "AI2_INDEPENDENT_OBSERVABILITY_V32_COMPATIBLE_REGION_ONLY",
        "task_id": TASK, "session_id": "play_cards_0915_031",
        "execution": "EXECUTED", "structure": "PASS", "quality": "INCONCLUSIVE",
        "frame_count": len(rows), "region_visible_frames": int(visible.sum()),
        "region_unknown_frames": int((~visible).sum()),
        "roi_valid_left_right": roi.sum(axis=0).tolist(),
        "prediction_valid_left_right": prediction.sum(axis=0).tolist(),
        "left_visible_but_no_output": "NOT_COMPUTABLE_SIDE_UNKNOWN",
        "right_visible_but_no_output": "NOT_COMPUTABLE_SIDE_UNKNOWN",
        "session_strict_interpretation": "LEFT_FAILURE_IS_INCONCLUSIVE_OBSERVABILITY_NOT_PROOF_OF_HAWOR_GENERALIZATION_FAILURE",
        "roi_repair_authorized": False,
        "rows": rows, "inputs": {"independent_region_proxy": ref(MASK), "motion": ref(MOTION)},
        "claim_limit": "Independent hand-region evidence only; no per-finger, anatomical-side or human GT authority.",
        "training_eligible": False, "control_ground_truth": False,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output = ATTEMPT / "lanes/lane2_motion/observability_031_wave1/RESULT.json"
    write(output, result)
    print(json.dumps({key: result[key] for key in ("session_id", "region_visible_frames", "roi_valid_left_right", "session_strict_interpretation")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
