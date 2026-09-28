"""Bounded 007 cable-intensity diagnostic and explicit no-full-expansion decision."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK, RAW, OLD_PREP

PREP = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/window_v1"
CLEAN = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/clean_window_v1"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/quality_review_v1"


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    clean = load_json(CLEAN / "RESULT.json")
    if clean["execution"] != "EXECUTED" or clean["source_frames"] != list(range(181, 197)):
        raise RuntimeError("NO_ACTUAL_FIXED_MODEL_OUTPUT")
    OUT.mkdir(parents=True)
    rows = []
    for local, frame in enumerate(range(181, 197)):
        raw = _image(RAW / f"{frame:06d}.png")
        old = _image(OLD_PREP.parent / "clean" / f"{frame:06d}.png")
        new = _image(CLEAN / "clean" / f"{local:06d}.png")
        mask = cv2.imread(str(PREP / "candidate" / f"{local:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(frame)
        candidate = mask > 0
        if not candidate.any():
            raise ValueError("EMPTY_COMPLAINT_PROXY")
        white = [int(np.count_nonzero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[candidate] > 190))
                 for image in (raw, old, new)]
        rows.append({"source_frame_id": frame, "candidate_pixels": int(candidate.sum()),
                     "raw_bright_proxy_px": white[0], "old_clean_bright_proxy_px": white[1],
                     "new_clean_bright_proxy_px": white[2],
                     "bright_proxy_threshold": "GRAY>190_DIAGNOSTIC_NOT_FROZEN_ACCEPTANCE_GATE"})
    result = {"schema_version": "HUMAN_TO_ROBOT_007_CABLE_QUALITY_REVIEW_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
              "rows": rows, "bright_proxy_reduced_frames": sum(row["new_clean_bright_proxy_px"] < row["old_clean_bright_proxy_px"] for row in rows),
              "independent_visual_proxy": {"sampled_frames": [181, 190, 196],
                                           "complaint_cable_region": "LOCAL_BRIGHT_CABLE_VISIBLY_REMOVED_IN_FROZEN_PROXY_REGION",
                                           "other_visible_residuals": "HAND_AND_LEFT_WHITE_LINE_REMAIN_IN_NEW_CLEAN",
                                           "reviewer": "AI_REVIEW_PROXY_NOT_HUMAN_GT"},
              "execution": "ACTUAL_MODEL_OUTPUT_REVIEWED", "structure": "PASS",
              "quality": "REJECTED_QUALITY_FULL_CLEAN", "adoption": "NOT_ADOPTED",
              "full_378_frame_expansion": "NOT_AUTHORIZED_BY_THIS_WINDOW",
              "reason": "LOCAL_CABLE_IMPROVEMENT_DOES_NOT_REMOVE_VISIBLE_HANDS_OR_PROVE_FULL_SESSION_INSTANCE_IDENTITY",
              "inputs": {"prep": artifact_ref(PREP / "RESULT.json"),
                         "model_output": artifact_ref(CLEAN / "RESULT.json")},
              "claim_limit": "Post-hoc brightness is a diagnostic proxy, not a frozen quality gate. One fixed-window cable improvement; no full Clean or product adoption.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    temp = OUT / f".RESULT.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, OUT / "RESULT.json")
    print(json.dumps({"status": result["quality"], "local_bright_proxy_reduced": result["bright_proxy_reduced_frames"],
                      "full_expansion": result["full_378_frame_expansion"],
                      "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
