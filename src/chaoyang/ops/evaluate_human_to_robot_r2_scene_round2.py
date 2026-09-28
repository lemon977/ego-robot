#!/usr/bin/env python3
"""Evaluate the bounded second Scene repair without promoting it to Clean."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, TASK, ref, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    old_root = ATTEMPT / "lanes/lane1_scene/clean_candidate_031_wave4"
    new_root = ATTEMPT / "lanes/lane1_scene/clean_candidate_031_wave5"
    old, new = load(old_root / "RESULT.json"), load(new_root / "RESULT.json")
    prep = load(new_root / "SCENE_PREP_MANIFEST.json")
    if any(value.get("frame_count") != 149 for value in (old, new, prep)):
        raise ValueError("ROUND2_FRAME_SET_CHANGED")
    total_diff = old_changed_protect = new_changed_protect = 0
    changed_frames = 0
    per_frame = []
    for index, (before, after, source) in enumerate(zip(old["rows"], new["rows"], prep["rows"], strict=True)):
        raw = cv2.imread(source["raw"], cv2.IMREAD_COLOR)
        a = cv2.imread(before["clean"]["path"], cv2.IMREAD_COLOR)
        b = cv2.imread(after["clean"]["path"], cv2.IMREAD_COLOR)
        protect = cv2.imread(source["protect"], cv2.IMREAD_GRAYSCALE) > 0
        if any(value is None for value in (raw, a, b)):
            raise ValueError(f"ROUND2_DECODE:{index}")
        difference = int(np.any(a != b, axis=2).sum())
        before_damage = int((np.any(a != raw, axis=2) & protect).sum())
        after_damage = int((np.any(b != raw, axis=2) & protect).sum())
        total_diff += difference
        old_changed_protect += before_damage
        new_changed_protect += after_damage
        changed_frames += int(difference > 0)
        per_frame.append({"frame_id": index, "candidate_diff_px": difference,
                          "before_changed_protected_px": before_damage,
                          "after_changed_protected_px": after_damage})
    if new_changed_protect != 0:
        raise ValueError("ROUND2_PROTECTED_PIXELS_CHANGED")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_ROUND2_REVIEW_V1",
        "task_id": TASK,
        "session_id": "play_cards_0915_031",
        "created_at": now(),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED",
        "numeric": {
            "candidate_diff_px": total_diff,
            "changed_frames": changed_frames,
            "wave4_changed_current_object_protected_px": old_changed_protect,
            "wave5_changed_current_object_protected_px": new_changed_protect,
        },
        "visual_review": {
            "authority": "AI_REVIEW_PROXY",
            "frozen_frames": [60, 68, 90, 120, 147],
            "decision": "NO_MATERIAL_VISUAL_GAIN",
            "reason": "Direct-visible object interiors are preserved, but hand/object occlusion hallucination and incomplete device evidence remain; this is not sufficient for formal Clean.",
        },
        "before": ref(old_root / "RESULT.json"),
        "after": ref(new_root / "RESULT.json"),
        "after_review": ref(new_root / "SCENE_CLEAN_CANDIDATE_REVIEW.mp4"),
        "per_frame": per_frame,
        "next_session_decision": {
            "session_id": "get_potato_chips_0915_007",
            "round2_execution": "NOT_EVALUATED_QUALITY_PRECHECK",
            "reason": "The frozen 031 canary did not yield material visual gain and the shared device-role blocker remains; do not spend another GPU quality trial.",
            "unaffected": "The complete wave4 007 candidate remains available for review.",
        },
        "claim_limit": "Candidate comparison only; no Clean, object geometry, or human-removal authority.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    output = new_root / "QUALITY_REVIEW.json"
    write_json(output, result)
    write_json(ATTEMPT / "lanes/lane1_scene/clean_candidate_007_wave5/ROUND2_DECISION.json",
               result["next_session_decision"] | {"task_id": TASK, "created_at": now()})
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE5_OBJECT_PROTECTION_ROUND2",
        "observed_at": now(),
        "round2": ref(output),
        "scene_quality_pass": 0,
        "scene_adopted": 0,
        "next": "Stop this Scene repair direction; continue independent Motion, Product diagnostics and receipt closure.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE5.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
