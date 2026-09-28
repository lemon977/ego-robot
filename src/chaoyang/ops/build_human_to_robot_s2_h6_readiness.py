#!/usr/bin/env python3
"""Build the immutable evidence decision used by the S2 H6 checkpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUTPUT = ROOT / "h6_readiness/attempt_0001/RESULT.json"
POSITION = ROOT / "lanes/motion_product/position_first_031/attempt_0002/RESULT.json"
TWO_STAGE = ROOT / "lanes/motion_product/two_stage_031/attempt_0001/RESULT.json"
TWO_STAGE_CODE = REPO_ROOT / "src/chaoyang/ops/run_human_to_robot_s2_031_two_stage_candidate.py"
ATTACHMENT = REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json"
RESUME = ROOT / "formal_entry_resume/attempt_0002/RESULT.json"
STEREO = REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/encoded_stereo_preflight_v1/get_potato_chips_0915_007/RESULT.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"IMMUTABLE_H6_READINESS_EXISTS:{OUTPUT}")
    position = load_json(POSITION)
    two_stage = load_json(TWO_STAGE)
    attachment = load_json(ATTACHMENT)
    resume = load_json(RESUME)
    stereo = load_json(STEREO)
    code = TWO_STAGE_CODE.read_text(encoding="utf-8")
    if "stage2 = least_squares(\n            pose_residual" not in code:
        raise RuntimeError("TWO_STAGE_IMPLEMENTATION_NOT_RECOGNIZED")
    if "position_residual" in code.split("stage2 = least_squares", 1)[1].split("q[frame", 1)[0]:
        raise RuntimeError("TWO_STAGE_MAY_ALREADY_RETAIN_POSITION_CONSTRAINT")
    candidate_path = Path(position["candidate"]["path"])
    if artifact_ref(candidate_path) != position["candidate"]:
        raise RuntimeError("POSITION_CANDIDATE_REF_DRIFT")
    with np.load(candidate_path, allow_pickle=False) as archive:
        valid = np.asarray(archive["wrist_valid"], dtype=bool)
        success = np.asarray(archive["solver_success"], dtype=bool)
        residual = np.asarray(archive["position_residual_mm"], dtype=np.float64)
        rotation = np.asarray(archive["rotation_residual_deg"], dtype=np.float64)
        frame_id = np.asarray(archive["frame_id"], dtype=np.int64)
    numeric_position_rows = int(np.count_nonzero(valid & (residual <= 20.0)))
    solver_position_rows = int(np.count_nonzero(valid & success & (residual <= 20.0)))
    failed_rows = [
        {
            "frame_id": int(frame_id[t]),
            "side": int(side),
            "solver_success": bool(success[t, side]),
            "position_mm": float(residual[t, side]),
            "rotation_deg": float(rotation[t, side]),
        }
        for t, side in zip(*np.nonzero(valid & ~(success & (residual <= 20.0))))
    ]
    product_pairs = {
        "007": (
            ROOT / "lanes/motion_product/formal_product_007/attempt_0001/PRODUCT_RESULT.json",
            ROOT / "lanes/motion_product/formal_product_007/attempt_0002/PRODUCT_RESULT.json",
        ),
        "031": (
            ROOT / "lanes/motion_product/formal_product_031/attempt_0003/PRODUCT_RESULT.json",
            ROOT / "lanes/motion_product/formal_product_031/attempt_0004/PRODUCT_RESULT.json",
        ),
        "103": (
            ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0001/PRODUCT_RESULT.json",
            ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/PRODUCT_RESULT.json",
        ),
        "042": (
            ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0001/PRODUCT_RESULT.json",
            ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/PRODUCT_RESULT.json",
        ),
    }
    refresh = {}
    for short, (before_path, after_path) in product_pairs.items():
        before = load_json(before_path)
        after = load_json(after_path)
        same_video = (
            before["product_video"]["bytes"] == after["product_video"]["bytes"]
            and before["product_video"]["sha256"] == after["product_video"]["sha256"]
        )
        if not same_video:
            raise RuntimeError(f"CURRENT_SIGNATURE_REFRESH_CHANGED_VIDEO:{short}")
        if before["pipeline_signature"] == after["pipeline_signature"]:
            raise RuntimeError(f"CURRENT_SIGNATURE_REFRESH_DID_NOT_CHANGE_SIGNATURE:{short}")
        refresh[short] = {
            "byte_identical_video": True,
            "before_result": artifact_ref(before_path),
            "after_result": artifact_ref(after_path),
            "video": after["product_video"],
            "before_pipeline_signature": before["pipeline_signature"]["sha256"],
            "after_pipeline_signature": after["pipeline_signature"]["sha256"],
        }
    robust = int(stereo["stereo_geometry_check"]["metrics"]["total_robust_matches"])
    minimum = int(stereo["stereo_geometry_check"]["thresholds"]["minimum_total_robust_matches"])
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_H6_READINESS_V1",
        "task_id": TASK,
        "created_at": _now(),
        "status": "READY_FOR_H6_FREEZE_NO_NEW_FULL_SESSION_ALGORITHM",
        "formal_entry": {
            "current_signature_products": refresh,
            "resume": artifact_ref(RESUME),
            "resume_status": resume["status"],
            "resume_reused": resume["reused"],
            "resume_mutated": resume["mutated"],
        },
        "p031": {
            "valid_rows": int(valid.sum()),
            "position_numeric_within_20mm": numeric_position_rows,
            "position_solver_success_and_within_20mm": solver_position_rows,
            "position_candidate_reported_count": position["position_within_20mm_count"],
            "failed_solver_or_position_rows": failed_rows,
            "arm_limit_hits_position_candidate": position["arm_limit_hit_count"],
            "collision_scope_position_candidate": position["collision_scope"],
            "two_stage_kept_position_as_hard_constraint": False,
            "two_stage_behavior": (
                "Stage one solves position. Stage two uses stage-one q only as initialization and "
                "optimizes the weighted full-pose residual; it does not retain a position hard constraint."
            ),
            "repair_rounds_consumed": [
                {"round": 1, "candidate": artifact_ref(POSITION)},
                {"round": 2, "candidate": artifact_ref(TWO_STAGE)},
            ],
            "repair_budget": 2,
            "new_partial_direction_candidate_in_s2": "NOT_AUTHORIZED_REPAIR_BUDGET_EXHAUSTED",
            "interpretation": (
                "The old two-stage failure does not falsify a lexicographic partial-direction method. "
                "The position candidate supplies 94 solver-success rows within the frozen 20mm gate, "
                "but collision was not evaluated and eight rows fail solver-success-or-position; it is "
                "not a whole-window feasibility witness. A new method requires a successor task."
            ),
        },
        "c007": {
            "attachment_canary": artifact_ref(ATTACHMENT),
            "attachment_consumption_fraction": attachment["attachment_consumption_fraction"],
            "outside_write_changed_px": attachment["outside_write_changed_px"],
            "protected_changed_px": attachment["protected_changed_px"],
            "support_semantics": "CAPTURE_DEVICE_MASK_NOT_CABLE_INSTANCE_MASK",
            "visual_review_proxy": (
                "The fixed review shows long thin cable pixels outside the red write support and "
                "therefore unchanged in the local Clean. This is a support/identity gap, not evidence "
                "that ProPainter failed on a correctly supplied cable mask."
            ),
            "new_clean_invocation": "NOT_JUSTIFIED_WITHOUT_FROZEN_TOP1_CABLE_INSTANCE_SUPPORT",
            "stereo": {"robust_matches": robust, "minimum": minimum, "successor": False},
        },
        "occlusion": {
            "new_temporal_gate": "NOT_CREATED_AFTER_VIEWING_H3_WINDOW",
            "status": "INCONCLUSIVE",
            "reason": (
                "736/3269 is a raw same-pixel ownership-switch count, not an error rate. "
                "No frozen surface-correspondence evaluator exists in the current task."
            ),
        },
        "adapter_collision": {
            "status": "UNVERIFIED_VISUAL_GEOMETRY_ONLY",
            "new_collision_proxy": "NOT_CREATED_WITHOUT_APPROVED_COLLISION_ASSET_OR_COVERAGE_PROOF",
        },
        "h6_decision": {
            "new_full_session_algorithm_compute": False,
            "continue_only": [
                "formal artifact validation",
                "H9 component freeze",
                "H10-H12 media/reload/governance closure",
            ],
            "successor_candidates_not_executed_in_s2": [
                "P031_LEXICOGRAPHIC_POSITION_HARD_PLUS_SUPPORTED_DIRECTION",
                "C007_TOP1_CABLE_INSTANCE_SUPPORT_THEN_ONE_BOUNDED_CLEAN",
                "APPROVED_ADAPTER_COLLISION_GEOMETRY",
                "SURFACE_CORRESPONDENCE_OCCLUSION_TEMPORAL_AUDIT",
            ],
        },
        "authority": {
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True)
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "resume": "4/4", "p031_repair_budget": "2/2", "new_full_session_algorithm_compute": False, "output": artifact_ref(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
