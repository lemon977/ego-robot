"""Two explicit phase aliases; historical dependencies never become selections."""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref

PHASE_PATH = "docs/current/DUAL_BRANCH_BASELINE_V1.json"
PINS = {
    "docs/current/EXACT78_VISUAL_BASELINE_V1.json": "6b22fed92292ff7cb661cbdb8a8bda12ac319ca662ab297b4f1b2ec12854c225",
    "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json": "148204eec75694a2b3ebf7e8d05253b59537a20e57c450a0f210b64a27ae72dc",
}


def checked(ref):
    if errors := validate_artifact_ref(ref):
        raise RuntimeError("PHASE_REFERENCE_DRIFT:" + ";".join(errors))
    return ref


def build_phase(root: Path) -> dict:
    for name, sha in PINS.items():
        if artifact_ref(root / name)["sha256"] != sha:
            raise RuntimeError("PHASE_PARENT_DRIFT:" + name)
    visual = load_json(root / "docs/current/EXACT78_VISUAL_BASELINE_V1.json")
    manus = load_json(root / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json")
    receipt = load_json(root / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json")
    if receipt.get("status") != "PASSED" or receipt["baseline_pointer"] != artifact_ref(root / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"):
        raise RuntimeError("PHASE_ADOPTION_RECEIPT_DRIFT")
    if not manus["baseline_selected"] or manus["baseline_id"] != "0916_HARDWARE_WORLD_WRIST_V16":
        raise RuntimeError("PHASE_MANUS_SELECTION_MISMATCH")
    dependency_names = (
        "src/chaoyang/pipeline/huro_hand_frame_v2.py",
        "src/chaoyang/pipeline/huro_hand_only_retarget_v1.py",
        "_run/current/exact78_takeover_b2_full_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/exact78_takeover_b1_dev5_v1/attempts/attempt_0001/RESULT.json",
        "_run/current/robot100_v16_baseline_overlay_20260929/attempts/attempt_0001/DEPENDENCY_KEEP_SET.json",
    )
    return {
        "schema_version": "DUAL_BRANCH_PHASE_BASELINES_V1", "phase_id": "RAW_TO_ROBOT_PHASE_V1_20261008",
        "status": "USER_SELECTED_PHASE_START_NOT_NEW_QUALITY_CERTIFICATION", "selected_alias_count": 2,
        "aliases": {
            "EXACT78_PHASE_V1": {
                "display_name": "视觉支线 V1", "parent_implementation": "takeover_v13",
                "parent_baseline_id": visual["baseline_id"], "pointer": artifact_ref(root / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"),
                "adoption_receipt": artifact_ref(root / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"),
                "parent_result": checked(visual["v13_result"]), "fresh_replay": checked(visual["fresh_replay_result"]),
                "operation": visual["selected_operation"], "actual_motion_source": visual["actual_motion_source"],
                "validated_scope": visual["validated_scope"], "batch_visual_production_usable": True,
                "pure_rgb": False, "full_batch_human_review": "NOT_CLAIMED", "metric_occlusion_qualified": False,
                "training_eligible": False, "control_ground_truth": False,
            },
            "MANUS_PHASE_V1": {
                "display_name": "MANUS支线 V1", "parent_implementation": "robot100_v16",
                "parent_baseline_id": manus["baseline_id"], "pointer": artifact_ref(root / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json"),
                "parent_baseline": checked(manus["baseline"]), "reference_revision": checked(manus["reference_revision"]),
                "reviewed_video": checked(manus["reviewed_video"]), "operation": "run_robot100_v16_safe_replay",
                "actual_motion_source": "SAME_SESSION_CONTROLLER_HEAD_MANUS_INSTALLATION_PRIOR",
                "selection_scope": manus["selection_scope"],
                "validated_scope": {"session": "play_cards_0916_025", "frames": 176, "batch_100_validated": manus["batch_100_validated"]},
                "image_alignment_status": manus["image_alignment_status"], "physical_accuracy_verified": manus["physical_accuracy_verified"],
                "training_eligible": False, "control_ground_truth": False, "implementation_owner": "OTHER_BRANCH_UNCHANGED",
            },
        },
        "history_policy": {
            "default_selection": "ONLY_THE_TWO_EXPLICIT_ALIASES",
            "excluded_from_default": ["Cards042_V2", "P1_retries_virtual_camera", "takeover_v12_and_older_candidates",
                "HaWoR_0915_experiments", "0915_0918_input_adapter_experiments", "HuRo_comparison", "RuCo_user_named_comparison",
                "four_stream_campaign_and_old_156_ledger_as_visual_adoption_counts"],
            "dependency_only_examples": [{"artifact": artifact_ref(root / name), "selection_allowed": False} for name in dependency_names],
            "dependency_admission": "Actual signed Task Packet read_set plus SHA validation; old path/name alone gives no permission.",
            "conditional_references_not_default_reading": ["0915 cleaning/calibration/time schemas", "legacy geometry/H50 consumer contracts", "licenses"],
            "comparison_name_note": "HuRo is present in project references. The user label RuCo does not select an implementation; aruco is not a comparison branch.",
            "legacy_autopilot_execution_allowed": False, "files_deleted": 0, "files_moved": 0,
        },
        "execution_policy": {"new_finite_task_packet_required": True, "unique_live_fenced_writer_required": True,
                             "historical_packet_reactivation_allowed": False, "alias_is_execution_authorization": False},
        "future_pure_rgb_hawor": {"status": "FUTURE_TARGET_NOT_CURRENT_BASELINE", "current_runnable_operation": None},
        "claim_limit": "Two user-selected phase-V1 starting points, preserving original v13/v16 evidence and scopes; not new batch/human/geometry/product/label qualification.",
    }


def validate_phase(root: Path) -> dict:
    value = load_json(root / PHASE_PATH)
    if value != build_phase(root):
        raise RuntimeError("PHASE_MANIFEST_DRIFT_REQUIRES_NEW_PUBLICATION")
    return value


def resolve_alias(root: Path, name: str) -> dict:
    value = validate_phase(root)
    if name not in value["aliases"]:
        raise RuntimeError("NOT_A_PHASE_BASELINE:" + name)
    return value["aliases"][name]


def phase_contract(root: Path) -> dict:
    if not (root / PHASE_PATH).exists():
        return {}
    value = validate_phase(root)
    return {"dual_branch_phase_baselines_v1": {
        "contract": artifact_ref(root / PHASE_PATH), "selected_alias_count": 2,
        "aliases": {key: {k: row[k] for k in ("parent_implementation", "parent_baseline_id", "operation", "pointer")}
                    for key, row in value["aliases"].items()},
        "task_packet_required": True, "legacy_autopilot_execution_allowed": False,
        "historical_dependencies_are_not_selected_baselines": True, "training_steps_authorized": 0,
        "claim_limit": value["claim_limit"],
    }}
