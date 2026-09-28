#!/usr/bin/env python3
"""Publish the S2 H6 decision or H9 component freeze.

The nominal wall-clock gates remain the default.  ``--allow-early-close`` is
only for an explicitly authorized evidence-driven close: the same immutable
prerequisites, product structure, resume, readiness, and evidence checks still
run.  It does not authorize new algorithm work or quality promotion.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
)


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PLAN = REPO_ROOT / "docs/current/PLAN.md"
WORK_ENTRY = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
LANES = ("scene", "sensor", "motion_product", "compare")
PRODUCTS = {
    "get_potato_chips_0915_007": ROOT / "lanes/motion_product/formal_product_007/attempt_0002/PRODUCT_RESULT.json",
    "play_cards_0915_031": ROOT / "lanes/motion_product/formal_product_031/attempt_0004/PRODUCT_RESULT.json",
    "get_potato_chips_0902_103": ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/PRODUCT_RESULT.json",
    "play_cards_0902_042": ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/PRODUCT_RESULT.json",
}


def _checked(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return artifact_ref(path)


def _current_row(state: dict) -> dict:
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("S2_IS_NOT_CURRENT")
    return next(item for item in state["tasks"] if item.get("task_id") == TASK)


def _elapsed(row: dict) -> float:
    return (datetime.now().astimezone() - datetime.fromisoformat(str(row["t0"]))).total_seconds()


def _product_snapshot() -> dict[str, object]:
    result: dict[str, object] = {}
    for session, path in PRODUCTS.items():
        value = load_json(path)
        if value.get("execution") != "EXECUTED" or value.get("structure") != "PASS":
            raise RuntimeError(f"PRODUCT_STRUCTURE_NOT_PASS:{session}")
        result[session] = {
            "result": _checked(path),
            "frames": f"{value['decoded_frames']}/{value['expected_frames']}",
            "quality": value["quality"],
            "adoption": value["adoption"],
            "adapter_visible_frames": value["adapter_visible_frames"],
            "known_decision_coverage": value["known_decision_coverage"],
            "unknown_decision_ratio": value["unknown_decision_ratio"],
        }
    return result


def _lane_snapshot(path: Path) -> dict[str, object]:
    """Capture checkpoint semantics without pinning a later-mutated STATE.json."""
    value = load_json(path)
    return {
        key: value.get(key)
        for key in (
            "status",
            "execution",
            "structure",
            "quality",
            "adoption",
            "checkpoint",
            "updated_at",
        )
    }


def _write_navigation(checkpoint: str, checkpoint_path: Path) -> None:
    if checkpoint == "H6":
        next_step = (
            "H6已冻结为不启动新的全片算法；继续做H9组件组合冻结、H10最终结构/媒体验证和H12终态。"
        )
    else:
        next_step = (
            "H9已冻结本轮实际组件组合；H10后只进行完整解码、重载、SHA与终态发布，不再修改算法。"
        )
    PLAN.write_text(
        "# 当前计划入口：Human→Robot S2 执行中\n\n"
        f"当前唯一任务为 `{TASK}`；最新不可变检查点为 `{checkpoint}`。"
        "四条正式产品候选结构完整，但质量通过0/4、采用0/4。\n\n"
        f"{next_step}\n\n"
        "不得降低Scene、Depth、Contact、碰撞或动作门；不得把UNKNOWN、视频存在或任务终态化写成产品成功。\n\n"
        f"机器证据：[{checkpoint}_RESULT]({checkpoint_path})；"
        "执行解释：[H3外部复核后的决定](HUMAN_TO_ROBOT_S2_H3_REVIEW_DECISION_ZH.md)；"
        "[S2视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n",
        encoding="utf-8",
    )
    WORK_ENTRY.write_text(
        f"# 当前入口：Human→Robot S2 {checkpoint}\n\n"
        f"先运行治理校验，再读 [{checkpoint}结果]({checkpoint_path})、"
        "[当前状态](STATUS.json) 和 [视频索引](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)。\n\n"
        f"{next_step}\n\n"
        "当前仍是4/4结构候选、0/4质量通过、0/4采用；连接件碰撞、031时序遮挡、"
        "007长线缆、Contact/Robot R1和Local/HuRo独立真值均未获得额外权限。"
        "新的部分方向IK只允许作为后继候选，S2内不得改名重试。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("H6", "H9"), required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--allow-early-close", action="store_true")
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    row = _current_row(state)
    elapsed = _elapsed(row)
    minimum = {"H6": 6 * 3600, "H9": 9 * 3600}[args.checkpoint]
    if elapsed < minimum and not args.allow_early_close:
        raise RuntimeError(f"{args.checkpoint}_NOT_REACHED:{elapsed:.1f}")

    h3 = ROOT / "checkpoints/H3_RESULT.json"
    if not h3.is_file():
        raise RuntimeError("H3_MISSING")
    prerequisite = h3
    if args.checkpoint == "H9":
        prerequisite = ROOT / "checkpoints/H6_RESULT.json"
        if not prerequisite.is_file():
            raise RuntimeError("H6_MISSING")

    product_snapshot = _product_snapshot()
    lane_states = {
        lane: _lane_snapshot(ROOT / f"lanes/{lane}/STATE.json")
        for lane in LANES
    }
    common = {
        "schema_version": f"HUMAN_TO_ROBOT_S2_{args.checkpoint}_CHECKPOINT_V1",
        "task_id": TASK,
        "checkpoint": args.checkpoint,
        "created_at": now_iso(),
        "elapsed_seconds": elapsed,
        "nominal_time_gate_seconds": minimum,
        "early_close_authorized": bool(args.allow_early_close and elapsed < minimum),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE",
        "adoption": "NOT_ADOPTED",
        "prerequisite": _checked(prerequisite),
        "products": product_snapshot,
        "lane_states_before_checkpoint": lane_states,
        "authority": {
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
    }
    if args.checkpoint == "H6":
        common["decision"] = {
            "new_full_session_algorithm_compute": "NOT_JUSTIFIED_BY_CURRENT_EVIDENCE",
            "reasons": [
                "031_TWO_BOUNDED_IK_REPAIRS_REJECTED_AND_TARGET_ORIENTATION_AUTHORITY_WEAK",
                "031_WRIST_TARGET_FAULT_PACKAGE_EXHAUSTED_2_OF_2_PARTIAL_DIRECTION_CONTRACT_DEFERRED_TO_VERSIONED_SUCCESSOR",
                "007_BOUNDED_CLEAN_REPAIR_LEAVES_LONG_CABLE_AND_FULL_SESSION_ROLE_EVIDENCE_UNRESOLVED",
                "007_NO_FROZEN_UNIQUE_TOP1_CABLE_INSTANCE_SUPPORT_FOR_ANOTHER_CLEAN_INVOCATION",
                "007_STEREO_GATE_REMAINS_1021_OF_1500_NO_DEPTH_SUCCESSOR",
                "CONTACT_R1_SCREENING_NOT_COMPLETED_NO_ADMISSIBLE_WINDOW_ESTABLISHED",
                "031_OCCLUSION_H3_SWITCH_COUNTS_ARE_DIAGNOSTIC_NOT_AN_ERROR_RATE_AND_NO_RETROSPECTIVE_GATE_IS_ALLOWED",
            ],
            "continue": [
                "artifact_validation",
                "component_freeze",
                "documentation_and_terminal_evidence",
            ],
            "do_not_start": [
                "new_model",
                "new_removal_algorithm",
                "third_031_IK_retry_under_renamed_partial_direction_contract",
                "007_clean_retry_without_frozen_top1_cable_instance",
                "retrospective_occlusion_quality_threshold_from_H3_window",
                "quality_retry_same_signature",
                "fabricated_depth_or_contact",
            ],
            "single_successor_recommendation": {
                "status": "RECOMMENDED_AFTER_S2_NOT_EXECUTED_IN_S2",
                "priority": "031_POSITION_HARD_CONSTRAINT_WITH_PREFROZEN_DIRECTION_QUALIFICATION",
                "scope": "play_cards_0915_031_frames_66_81_right_arm_only",
                "budget_limit": "3_HOURS_MAXIMUM_FROM_SEPARATE_REGISTRATION",
                "input_freeze": [
                    "original_16_frame_indices_and_invalid_reasons",
                    "handmotion_and_wrist_targets",
                    "camera_base_placement_and_68p4mm_mount_contract",
                    "urdf_mesh_joint_axes_limits_and_finger_q",
                    "declared_existing_collision_scope",
                ],
                "preconditions": [
                    "position_gate_tau_p_traced_to_an_actual_quality_contract_not_a_loss_scale",
                    "direction_eligibility_frozen_before_solver_output_is_read",
                    "new_constraint_contract_has_a_counterexample_that_distinguishes_it_from_old_two_stage",
                    "position_feasible_witness_and_declared_collision_scope_reloaded_independently",
                ],
                "required_validation": [
                    "per_frame_position_constraint_and_violation_amount",
                    "qualified_direction_error_and_full_orientation_diagnostic",
                    "joint_limits_declared_collision_scope_and_real_timestamp_continuity",
                    "fixed_window_numeric_result_and_lossless_review",
                ],
                "claim_limit": (
                    "At most POSITION_FEASIBILITY_DIAGNOSTIC or "
                    "PARTIAL_DIRECTION_MOTION_CANDIDATE; not full-pose, product, "
                    "control, deployment or external-truth authority."
                ),
                "stop_conditions": [
                    "position_quality_contract_not_located",
                    "no_direction_has_prefrozen_evidence_qualification",
                    "solver_cannot_express_and_verify_a_true_position_constraint",
                    "declared_geometry_or_collision_precondition_is_not_satisfied",
                    "bounded_candidate_budget_exhausted",
                ],
                "fallback": "DO_NOT_AUTOMATICALLY_START_CABLE_OCCLUSION_CONTACT_OR_COLLISION_SUCCESSORS",
            },
        }
        resume = ROOT / "formal_entry_resume/attempt_0002/RESULT.json"
        resume_value = load_json(resume)
        if resume_value.get("status") != "PASS" or resume_value.get("reused") != 4:
            raise RuntimeError("FORMAL_ENTRY_RESUME_NOT_CLOSED")
        common["decision"]["formal_entry_resume"] = _checked(resume)
        common["decision"]["formal_entry_resume_status"] = "CLOSED_4_OF_4_NO_MUTATION"
        readiness = ROOT / "h6_readiness/attempt_0001/RESULT.json"
        readiness_value = load_json(readiness)
        if readiness_value.get("status") != "READY_FOR_H6_FREEZE_NO_NEW_FULL_SESSION_ALGORITHM":
            raise RuntimeError("H6_READINESS_NOT_CLOSED")
        common["decision"]["readiness"] = _checked(readiness)
        evidence_audit = ROOT / "evidence_ref_audit/attempt_0001/RESULT.json"
        evidence_audit_value = load_json(evidence_audit)
        if (
            evidence_audit_value.get("status") != "PASS"
            or evidence_audit_value.get("error_count") != 0
        ):
            raise RuntimeError("S2_EVIDENCE_REFERENCE_CLOSURE_NOT_PASS")
        common["decision"]["evidence_reference_audit"] = _checked(evidence_audit)
        common["decision"]["current_signature_refresh"] = (
            "4/4 products rerendered only because formal entry code signature changed; "
            "all four encoded videos are byte-identical to their H3 predecessors and "
            "the current-signature CLI reused all four without mutation."
        )
    else:
        frozen_paths = {
            "renderer": REPO_ROOT / "src/chaoyang/pipeline/v5_product.py",
            "compositor": REPO_ROOT / "src/chaoyang/pipeline/occlusion_compositor_v1.py",
            "formal_entry": REPO_ROOT / "src/chaoyang/ops/run_human_to_robot_baseline_v1.py",
            "mount": REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json",
            "compare_refresh": REPO_ROOT / "src/chaoyang/ops/run_human_to_robot_s2_local_huro_adapter_refresh.py",
        }
        common["component_freeze"] = {name: _checked(path) for name, path in frozen_paths.items()}
        common["freeze_decision"] = {
            "status": "FROZEN_FOR_S2_FINAL_VALIDATION",
            "formal_products": "4_STRUCTURAL_0_QUALITY_PASS",
            "scene": "NOT_ADOPTED",
            "sensor": "FROZEN_REUSE_MIXED_KINEMATIC_ONLY",
            "motion_product": "CANDIDATE_ONLY",
            "compare": "NO_WINNER",
            "robot_r1": "0_SCREENED_0_EXECUTED_0_ADOPTED_NO_ADMISSIBLE_WINDOW_ESTABLISHED",
            "partial_direction_ik": "NOT_RUN_IN_S2_DEFERRED_TO_VERSIONED_SUCCESSOR",
            "cable_top1_clean": "NOT_RUN_WITHOUT_FROZEN_INSTANCE_SUPPORT",
            "occlusion_temporal_quality": "INCONCLUSIVE_NO_RETROSPECTIVE_THRESHOLD",
            "new_algorithm_after_h9": False,
        }

    path = ROOT / f"checkpoints/{args.checkpoint}_RESULT.json"
    if path.exists():
        raise RuntimeError(f"IMMUTABLE_CHECKPOINT_EXISTS:{path}")
    atomic_json(path, common)
    checkpoint_ref = _checked(path)
    for lane in LANES:
        lane_path = ROOT / f"lanes/{lane}/STATE.json"
        value = load_json(lane_path)
        value["checkpoint"] = args.checkpoint
        value["checkpoint_result"] = checkpoint_ref
        value["updated_at"] = now_iso()
        atomic_json(lane_path, value)

    _write_navigation(args.checkpoint, path)

    timestamp = now_iso()
    row["updated_at"] = timestamp
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "PENDING",
        "created_at": timestamp,
        "message": f"S2 {args.checkpoint} checkpoint published without quality promotion.",
        "result": checkpoint_ref,
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"HUMAN_TO_ROBOT_S2_{args.checkpoint}_PUBLISHED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({
        "status": "PASS",
        "checkpoint": args.checkpoint,
        "governance_revision": published["governance_revision"],
        "result": checkpoint_ref,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
