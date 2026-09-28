"""Explicit per-frame AI RGB review of the frozen 031 second-hand sample."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/left_rgb_review_031/v1"
RAW = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/play_cards_0915_031/raw"
OLD = ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/lanes/motion_product/left_evidence_031/wave0/RESULT.json"
REASONS = {
    0: "NO_FOREGROUND_HAND_VISIBLE",
    10: "NO_FOREGROUND_HAND_VISIBLE",
    20: "NO_FOREGROUND_HAND_VISIBLE",
    30: "NO_FOREGROUND_HAND_VISIBLE",
    39: "NO_FOREGROUND_HAND_VISIBLE",
    49: "NO_SEPARATE_SECOND_HAND_VISIBLE",
    59: "NO_SEPARATE_SECOND_HAND_VISIBLE",
    69: "ONE_PARTIAL_HAND_AT_LOWER_RIGHT_ONLY",
    79: "ONE_HAND_AT_RIGHT_CARD_ONLY",
    89: "ONE_HAND_AT_RIGHT_CARD_ONLY",
    99: "ONE_HAND_AT_RIGHT_CARD_ONLY",
    109: "ONE_HAND_AND_ARM_EXITING_RIGHT_LOWER_BOUNDARY",
    118: "ONE_HAND_WITH_CONTINUOUS_PALM_FROM_LOWER_EDGE",
    128: "ONE_HAND_WITH_CONTINUOUS_PALM_FROM_LOWER_EDGE",
    138: "ONE_HAND_WITH_CONTINUOUS_PALM_FROM_LOWER_EDGE",
    148: "ONE_HAND_WITH_CONTINUOUS_PALM_FROM_LOWER_EDGE",
}


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    old = load_json(OLD)
    if old["sampled_frame_ids"] != list(REASONS):
        raise ValueError("FROZEN_SAMPLE_DRIFT")
    OUT.mkdir(parents=True)
    rows = []
    for frame, reason in REASONS.items():
        path = RAW / f"{frame:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (960, 1280):
            raise ValueError(f"RAW_DECODE:{frame}")
        rows.append({"source_frame_id": frame, "raw": artifact_ref(path),
                     "second_hand_independent_rgb_support": "NOT_ESTABLISHED_IN_THIS_FRAME",
                     "reason": reason, "reviewer": "AI_REVIEW_PROXY_NOT_HUMAN_GT",
                     "full_timeline_inference_allowed": False})
    result = {"schema_version": "HUMAN_TO_ROBOT_031_LEFT_RGB_PER_FRAME_REVIEW_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031", "sampled_frames": list(REASONS),
              "rows": rows, "source_old_review": artifact_ref(OLD),
              "new_hawor_invocation": False, "original_denominator": {"timeline": 149, "right": 102, "left": 0},
              "quality": "INCONCLUSIVE_FULL_TIMELINE_AI_SAMPLE_ONLY",
              "claim_limit": "16 independent raw RGB sample judgments only. No claim of full-session left absence, human GT, or model accuracy; no fill, mirror, hold or new IK.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    temp = OUT / f".RESULT.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, OUT / "RESULT.json")
    print(json.dumps({"status": result["quality"], "reviewed": len(rows),
                      "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
