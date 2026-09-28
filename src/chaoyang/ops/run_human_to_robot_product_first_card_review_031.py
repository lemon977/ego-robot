"""Bounded 031 card-edge review of the frozen, rejected ProPainter candidate."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

PRIOR = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_031_wave5"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/card_031/review_v1"


def _image(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    return data


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    prep = load_json(PRIOR / "SCENE_PREP_MANIFEST.json")
    invocation = load_json(PRIOR / "INVOCATION.json")
    prior_result = load_json(PRIOR / "RESULT.json")
    quality = load_json(PRIOR / "QUALITY_REVIEW.json")
    command = invocation["command"]
    if command[command.index("--mask") + 1] != str(PRIOR / "prep/model_masks"):
        raise RuntimeError("MODEL_MASK_COMMAND_MISMATCH")
    if int(prior_result["frame_count"]) != 149 or int(prior_result["model_returncode"]) != 0:
        raise RuntimeError("OLD_MODEL_OUTPUT_INCOMPLETE")
    rows = []
    for frame in range(66, 82):
        recorded = prep["rows"][frame]
        if int(recorded["frame_id"]) != frame:
            raise RuntimeError("FRAME_MAP_MISMATCH")
        raw = _image(Path(recorded["raw"]))
        clean = _image(PRIOR / f"clean/{frame:06d}.png")
        protect = _image(Path(recorded["protect"])) > 0
        write = _image(Path(recorded["write"])) > 0
        model = _image(PRIOR / f"prep/model_masks/{frame:06d}.png") > 0
        if raw.shape != clean.shape or raw.shape[:2] != write.shape:
            raise RuntimeError("DOMAIN_MISMATCH")
        changed = np.any(raw != clean, axis=2)
        if np.any(changed & ~write) or np.any(changed & protect):
            raise RuntimeError("WRITE_OR_PROTECT_VIOLATION")
        rows.append({"source_frame_id": frame, "write_pixels": int(write.sum()),
                     "protect_pixels": int(protect.sum()), "model_mask_pixels": int(model.sum()),
                     "changed_pixels": int(changed.sum()),
                     "protected_changed_pixels": int((changed & protect).sum()),
                     "outside_write_changed_pixels": int((changed & ~write).sum())})
    OUT.mkdir(parents=True)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_031_CARD_REVIEW_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "source_frames": [66, 81],
        "execution": "REUSED_FROZEN_MODEL_OUTPUT_RECHECKED", "new_model_invocations": 0,
        "structure": "PASS_FOR_FIXED_WINDOW", "quality": "REJECTED_QUALITY_CARD_EDGE_BLUR",
        "adoption": "NOT_ADOPTED", "rows": rows,
        "ai_visual_proxy": {"reviewed_frames": [66, 74, 81],
            "finding": "VISIBLE_CARD_BACK_AT_RIGHT_IS_BLURRED_AND_HIDDEN_CONTENT_HAS_NO_GROUND_TRUTH",
            "authority": "AI_REVIEW_PROXY_NOT_HUMAN_GROUND_TRUTH"},
        "deterministic_new_implementation_error": False,
        "retry_decision": "NO_NEW_GPU_RUN_WITHOUT_FALSIFIABLE_FIX",
        "claim_limit": "Existing write/protect invariants and model invocation confirmed; card-edge reconstruction remains poor. This does not prove hidden card appearance or authorise a new quality retry.",
        "inputs": {"prep": artifact_ref(PRIOR / "SCENE_PREP_MANIFEST.json"),
                   "invocation": artifact_ref(PRIOR / "INVOCATION.json"),
                   "model_result": artifact_ref(PRIOR / "RESULT.json"),
                   "quality_review": artifact_ref(PRIOR / "QUALITY_REVIEW.json")},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    path = OUT / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["quality"], "result": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
