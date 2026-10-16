"""Read-only T3 candidate: join six sealed action rows to sealed scene frontends.

Prints JSON only. It never follows latest paths, reads active successor partials,
registers a task, or promotes visual/product/training qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.governance import build_exact78_t3_six_matrix_v1 as action


ROOT = Path(__file__).resolve().parents[3]
STATE_REL = Path("docs/governance/LONG_HORIZON_TASK_STATE.json")
PREQUAL_REL = Path("docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/09_T3_GEOMETRY_PREQUAL_SIX_SESSIONS_ZH.md")
PREQUAL_SHA = "d7a67b6194744d970a7a271cfb29453baa07e10475efb8da51a892f6a6a38b12"
CORRECTION_REL = Path("_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/CORRECTED_CAPABILITY.json")
CORRECTION_SHA = "5577a80426333b5bd3eb26eb378f2332b8a2827027f01be5b439090087e00475"
FRONTENDS = {
    "get_potato_chips_0901_013": (
        "exact78_raw_to_robot_t2_chips013_frontend_v1_20260930",
        "4bf0438733b4d2eef9d2b8bd7639cea6cfc09a8cd6ceb0099dadd3be177d55f1"),
    "get_potato_chips_0902_103": (
        "exact78_raw_to_robot_t2_0902_chips103_frontend_v3_mp4_retry1_20261001",
        "7a6b70ce88a87cb21aee92e90a76966055921f6e49ba9c4d7a4ca83ae7738bf3"),
    "play_cards_0901_001": (
        "exact78_raw_to_robot_t2_0901_cards001_frontend_v1_20261001",
        "c4e182b47fc6c635785874bebcf121ec8c7b3b85b563cb4b40536590bc5eee5b"),
    "play_cards_0902_042": (
        "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930",
        "d065012d7d774dd5c7f8b4d42be0555dc7f5f3d9c28e29a950be65de7e027684"),
    "play_cards_0915_001": (
        "exact78_raw_to_robot_t2_0915_cards001_frontend_v1_20260930",
        "858299610bb876b07a4cf786dbbed735d6f7e7a64bb371540a68b7e6b15de5c3"),
    "play_cards_0915_002": (
        "exact78_raw_to_robot_t2_0915_cards002_frontend_v3_mp4_holdout_20261001",
        "0b1995fad6a6f992d041b53f6abec54fd64d198e16aed98c8b0e99bad4db8917"),
}
SOURCE = {sid: sid.split("_")[-2] for sid in FRONTENDS}
GEOMETRY_INPUT_PREQUAL = {
    "get_potato_chips_0901_013": (
        "BLOCKED_CALIBRATION_ABSENT", "BLOCKED_CALIBRATION_ABSENT_AND_FRESH_DEPTH_MASK_ABSENT"),
    "get_potato_chips_0902_103": (
        "BLOCKED_EYE_INDEX_AND_RECTIFICATION_UNVERIFIED",
        "BLOCKED_EYE_INDEX_AND_FRESH_DEPTH_MASK_ABSENT"),
    "play_cards_0901_001": (
        "BLOCKED_CALIBRATION_ABSENT", "BLOCKED_CALIBRATION_ABSENT_AND_FRESH_DEPTH_MASK_ABSENT"),
    "play_cards_0902_042": (
        "READY_TO_EVALUATE", "BLOCKED_FRESH_REGISTERED_DEPTH_AND_VISIBLE_INSTANCE_MASK_ABSENT"),
    "play_cards_0915_001": (
        "READY_TO_EVALUATE", "BLOCKED_FRESH_REGISTERED_DEPTH_AND_VISIBLE_INSTANCE_MASK_ABSENT"),
    "play_cards_0915_002": (
        "READY_TO_EVALUATE", "BLOCKED_FRESH_REGISTERED_DEPTH_AND_VISIBLE_INSTANCE_MASK_ABSENT"),
}
OCCLUSION_PREQUAL = "BLOCKED_REGISTERED_SCENE_DEPTH_AND_ROBOT_DEPTH_ABSENT"
NOT_EVALUATED = "NOT_EVALUATED"


def _terminal_frontend(root: Path, session_id: str, task: dict[str, Any] | None,
                       expected_frames: int) -> dict[str, Any]:
    task_id, pinned_sha = FRONTENDS[session_id]
    empty = {"status": NOT_EVALUATED, "task_id": task_id,
             "terminal_result": None, "mask": {"status": NOT_EVALUATED},
             "clean": {"status": NOT_EVALUATED}}
    if task is None or task.get("status") not in action.TERMINAL:
        return empty
    if task.get("task_id") != task_id or task.get("session") != session_id:
        raise ValueError("SCENE_TASK_IDENTITY_DRIFT")
    if task["status"] != "PASSED":
        empty["status"] = task["status"]
        return empty
    attempt = root / "_run/current" / task_id / "attempts/attempt_0001"
    ref = task.get("result")
    if not isinstance(ref, dict) or ref.get("sha256") != pinned_sha:
        raise ValueError("SCENE_TERMINAL_PINNED_SHA_DRIFT")
    result_path, result = action._verified(root, ref, required_parent=attempt)
    if result_path != attempt / "RESULT.json":
        raise ValueError("SCENE_TERMINAL_RESULT_LOCATION_DRIFT")
    receipt_ref = task.get("terminal_receipt")
    _, receipt = action._verified(root, receipt_ref, required_parent=root / "tasks/receipts")
    if (receipt.get("result") != ref or receipt.get("task_id") != task_id
            or receipt.get("status") != "PASSED"):
        raise ValueError("SCENE_TERMINAL_RECEIPT_CHAIN_DRIFT")
    if (result.get("task_id") != task_id or result.get("source_session") != session_id
            or result.get("status") != "PASSED"
            or result.get("execution_complete") is not True
            or result.get("source_frames_expected") != expected_frames
            or result.get("stage_seal_errors") != []
            or result.get("post_input_sha_errors") != []):
        raise ValueError("SCENE_TERMINAL_IDENTITY_OR_FRAMES_DRIFT")
    seals = result.get("stage_seals", {})
    children: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for name in ("MASK", "CLEAN"):
        seal_ref = seals.get(name)
        seal_path, seal = action._verified(root, seal_ref, required_parent=attempt)
        if (seal_path != attempt / "candidate" / f"SEAL_{name}.json"
                or seal.get("session_id") != session_id
                or seal.get("stage") != name
                or seal.get("frame_count") != expected_frames
                or seal.get("quality_adopted") is not False):
            raise ValueError(f"SCENE_{name}_SEAL_DRIFT")
        child_path, child = action._verified(root, seal.get("result"), required_parent=attempt)
        if child_path != attempt / "candidate" / f"{name}.json" or child.get("session_id") != session_id:
            raise ValueError(f"SCENE_{name}_RESULT_IDENTITY_DRIFT")
        rows = child.get("rows")
        if (not isinstance(rows, list) or len(rows) != expected_frames
                or [row.get("decode_display_index") for row in rows] != list(range(expected_frames))):
            raise ValueError(f"SCENE_{name}_ROW_MAP_DRIFT")
        children[name] = (seal_ref, seal["result"], child)
    mask_seal, mask_ref, mask = children["MASK"]
    clean_seal, clean_ref, clean = children["CLEAN"]
    if (result.get("mask_quality") != "NOT_YET_ADOPTED"
            or result.get("clean_quality") != "NOT_YET_ADOPTED"
            or result.get("user_review") != "PENDING"
            or mask.get("training_eligible") is not False
            or clean.get("training_eligible") is not False
            or clean.get("depth_object6d_contact_truth") is not False):
        raise ValueError("SCENE_QUALITY_OR_REVIEW_DRIFT")
    role_states = sorted({row.get("role_status", NOT_EVALUATED) for row in mask["rows"]})
    return {
        "status": "PASSED_EXECUTION_ONLY", "task_id": task_id,
        "terminal_result": ref, "terminal_receipt": receipt_ref,
        "source_frames_expected": expected_frames,
        "source_fps_in_terminal": result.get("source_fps", result.get("source_fps_nominal")),
        "source_time_mode_in_terminal": result.get("time_mode", NOT_EVALUATED),
        "qpc_head_sync": result.get("qpc_head_sync", NOT_EVALUATED),
        "mask": {"status": "SEALED_UNADOPTED", "seal": mask_seal, "result": mask_ref,
                 "frame_count": len(mask["rows"]), "quality": result["mask_quality"],
                 "role_review_states": role_states,
                 "visible_object_identity": mask.get("visible_object_identity", NOT_EVALUATED),
                 "physical_instance_tracking": mask.get("physical_instance_tracking", NOT_EVALUATED),
                 "human_review": "PENDING"},
        "clean": {"status": "SEALED_UNADOPTED_OFFLINE_VISUAL_ONLY",
                  "seal": clean_seal, "result": clean_ref,
                  "frame_count": len(clean["rows"]),
                  "quality": clean.get("quality", NOT_EVALUATED),
                  "human_review": clean.get("human_review", NOT_EVALUATED),
                  "source_fps": clean.get("source_fps", NOT_EVALUATED),
                  "time_mode": clean.get("time_mode", NOT_EVALUATED),
                  "depth_object6d_contact_truth": False},
        "visual_qualified": result.get("visual_qualified"),
        "product_qualified": result.get("product_qualified"),
        "training_eligible": result.get("training_eligible"),
        "user_review": result.get("user_review"),
    }


def _qualification(value: Any) -> str:
    if value is True:
        return "QUALIFIED_BY_SOURCE_CLAIM"
    if value is False:
        return "NOT_QUALIFIED"
    return NOT_EVALUATED


def build_matrix(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    prequal_ref = action._ref(root / PREQUAL_REL)
    correction_ref = action._ref(root / CORRECTION_REL)
    if (prequal_ref["sha256"] != PREQUAL_SHA or correction_ref["sha256"] != CORRECTION_SHA):
        raise ValueError("GEOMETRY_INPUT_PREQUAL_EVIDENCE_SHA_DRIFT")
    correction = action._json(root / CORRECTION_REL)
    base = action.build_matrix(root)
    if set(base["sessions"]) != set(FRONTENDS) or set(GEOMETRY_INPUT_PREQUAL) != set(FRONTENDS):
        raise ValueError("T3_SIX_SESSION_JOIN_SET_DRIFT")
    state_path = root / STATE_REL
    state_ref = action._ref(state_path)
    state = action._json(state_path)
    task_by_id = {task.get("task_id"): task for task in state.get("tasks", [])}
    if len(task_by_id) != len(state.get("tasks", [])):
        raise ValueError("TASK_LEDGER_DUPLICATE_IDS")
    rows: dict[str, Any] = {}
    for session_id, action_row in sorted(base["sessions"].items()):
        source = SOURCE[session_id]
        role = action_row["selection_role"]
        if role not in {"development", "holdout"}:
            raise ValueError("T3_ROLE_UNEXPECTED")
        task_id, _ = FRONTENDS[session_id]
        scene = _terminal_frontend(root, session_id, task_by_id.get(task_id), action_row["frames"])
        fps = action_row["nominal_fps"]
        if (scene["status"] == "PASSED_EXECUTION_ONLY"
                and scene["clean"]["source_fps"] != fps):
            raise ValueError("SCENE_FPS_T0_DRIFT")
        calibration = correction.get("calibration_states", {}).get(session_id)
        if not isinstance(calibration, dict):
            raise ValueError("T0_CORRECTED_CALIBRATION_MISSING")
        depth_input, object_input = GEOMETRY_INPUT_PREQUAL[session_id]
        stage_status = action_row["stage_qualification"]
        if any(stage_status.get(key) != NOT_EVALUATED for key in ("depth", "object6d", "occlusion")):
            raise ValueError("FRESH_GEOMETRY_RESULT_REQUIRES_SEPARATE_PROOF")
        if source == "0901" and calibration.get("metric_geometry") != "BLOCKED_CALIBRATION_ABSENT":
            raise ValueError("T0_0901_CALIBRATION_CLAIM_DRIFT")
        if source != "0901" and calibration.get("extrinsics") != "SOURCE_PRESENT":
            raise ValueError("T0_SOURCE_EXTRINSICS_DRIFT")
        action_status = action_row["execution"]["status"]
        execution = {
            "action_status": action_status,
            "action_complete": action_row["execution"].get("complete", False),
            "robot_video_complete_sealed_claim": action_row["robot"].get("video_complete_sealed_claim", False),
            "scene_frontend_status": scene["status"],
            "scene_mask_sealed_frames": scene["mask"].get("frame_count", NOT_EVALUATED),
            "scene_clean_sealed_frames": scene["clean"].get("frame_count", NOT_EVALUATED),
        }
        rows[session_id] = {
            "session_id": session_id, "source": source, "selection_role": role,
            "split_claim": action_row.get("split_claim"),
            "source_group": action_row["source_group"],
            "raw": {"frames": action_row["frames"], "nominal_fps": fps,
                    "physical_eye": action_row["physical_eye"],
                    "t0_time_mode": action_row["time_mode"],
                    "capture_to_decode_alignment": action_row["capture_to_decode_alignment"],
                    "head_pose_clock_domain": action_row["head_pose_clock_domain"],
                    "corrected_metric_geometry": calibration["metric_geometry"]},
            "scene_time": {"clean_fps": scene["clean"].get("source_fps", NOT_EVALUATED),
                           "clean_time_mode": scene["clean"].get("time_mode", NOT_EVALUATED),
                           "qpc_head_sync": scene.get("qpc_head_sync", NOT_EVALUATED)},
            "execution": execution,
            "hawor": action_row["hawor"],
            "robot": action_row["robot"],
            "mask": scene["mask"], "clean": scene["clean"],
            "geometry_prequalification": {
                "depth": {"input_status": depth_input, "fresh_result": NOT_EVALUATED,
                          "field_qualification": NOT_EVALUATED},
                "object6d": {"input_status": object_input, "fresh_result": NOT_EVALUATED,
                             "field_qualification": NOT_EVALUATED},
                "trusted_occlusion": {"input_status": OCCLUSION_PREQUAL,
                                      "geometry_decision_coverage": NOT_EVALUATED,
                                      "display_fallback_coverage": NOT_EVALUATED,
                                      "field_qualification": NOT_EVALUATED},
            },
            "visual_qualification": _qualification(action_row["visual_qualified"])
                                    if scene["visual_qualified"] is False else NOT_EVALUATED,
            "full_product_qualification": _qualification(action_row["product_qualified"])
                                          if scene["product_qualified"] is False else NOT_EVALUATED,
            "training_label_qualification": _qualification(action_row["training_eligible"])
                                             if scene["training_eligible"] is False else NOT_EVALUATED,
            "ai_review": NOT_EVALUATED,
            "human_review": "PENDING" if action_row["user_review"] == "PENDING"
                            and scene["user_review"] == "PENDING" else NOT_EVALUATED,
            "evidence": {"action": action_row["evidence"],
                         "scene_terminal_result": scene["terminal_result"],
                         "scene_terminal_receipt": scene.get("terminal_receipt"),
                         "scene_mask_result": scene["mask"].get("result"),
                         "scene_clean_result": scene["clean"].get("result")},
        }
    if action._ref(state_path) != state_ref:
        raise ValueError("TASK_STATE_CHANGED_DURING_SCENE_JOIN")
    development = [sid for sid, row in rows.items() if row["selection_role"] == "development"]
    holdout = [sid for sid, row in rows.items() if row["selection_role"] == "holdout"]
    if len(development) != 4 or len(holdout) != 2:
        raise ValueError("T3_4DEV_2HOLDOUT_SPLIT_DRIFT")
    return {
        "schema_version": "exact78-t3-six-session-scene-joined-cpu-snapshot-v1",
        "status": "READ_ONLY_CANDIDATE_NOT_T3_SEAL",
        "input_evidence": {"action_builder": action._ref(Path(action.__file__).resolve()),
                           "action_snapshot_inputs": base["inputs"],
                           "state_for_terminal_eligibility_only": state_ref,
                           "geometry_prequal_review": prequal_ref,
                           "t0_corrected_capability": correction_ref,
                           "generator_code": action._ref(Path(__file__).resolve())},
        "selection": {"development": development, "holdout": holdout,
                      "source_holdout_gap": {"source": "0902", "session_id": None,
                                             "status": "SPLIT_NOT_ESTABLISHED",
                                             "selection_claim": "NO_THIRD_HOLDOUT_SESSION_SELECTED"}},
        "sessions": rows,
        "claim_limit": "Four development and two pre-frozen cross-group holdouts; 0902 third split unestablished; sealed execution is not visual, geometric, product, training or human approval",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.repo_root), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
