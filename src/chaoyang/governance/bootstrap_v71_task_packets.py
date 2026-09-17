from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Publish immutable, bounded-context task packets for Chaoyang V7.1."""

import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import atomic_write, canonical_bytes, sha256_bytes
from chaoyang.governance.v52_contracts import atomic_write_new, validate_task_packet


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71"
PACKET_ROOT = RUN_ROOT / "task_packets"
PLAN = "docs/governance/EXACT78_V7_1_EXECUTION_PLAN_ZH.md"
BASE_READS = [
    "docs/governance/CURRENT_STATUS_RECEIPT.json",
    "docs/governance/CURRENT_PROJECT_STATUS_MIN.json",
    "docs/governance/PLAN_REVISION.json",
]
CREATED_AT = "2026-09-14T23:40:00+08:00"
VISUAL_AUX_UPSTREAM_STATE = (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/"
    "visual_tier_robot_R7_2/AUTOMATION_STATE.json"
)


def definition(
    task_id: str,
    objective: str,
    reads: list[str],
    writes: list[str],
    prerequisites: list[str],
    gates: list[str],
    outputs: list[str],
    claim_limit: str,
    *,
    cpu: int = 14_400,
    gpu: int = 0,
    wall: int = 28_800,
    attempts: int = 2,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "objective": objective,
        "reads": BASE_READS + reads,
        "writes": writes,
        "prerequisites": prerequisites,
        "gates": gates,
        "outputs": outputs,
        "claim_limit": claim_limit,
        "budgets": {"cpu_seconds": cpu, "gpu_seconds": gpu, "wall_seconds": wall},
        "attempt_max": attempts,
    }


TASKS = [
    definition(
        "g0_core_governance_v71",
        "Recover the core ledger and publish the V7.1 immutable execution graph.",
        [PLAN, "src/chaoyang/governance/common.py", "src/chaoyang/governance/v52_contracts.py"],
        ["docs/governance/", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/g0_core/"],
        [],
        ["receipt_pass_fresh", "no_dead_active_task", "single_aggregator", "packet_index_valid"],
        ["RESULT.json", "TASK_PACKET_INDEX.json"],
        "Governance only; no algorithm authority.",
        cpu=7_200,
        wall=14_400,
    ),
    definition(
        "g0_provenance_debt_v71",
        "Inventory unresolved weights, historical closure, CAD/TCP and real-action provenance without globally blocking unrelated work.",
        [PLAN, "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/provenance_debt/"],
        ["g0_core_governance_v71"],
        ["scope_local_only", "no_invented_sha", "terminal_for_each_debt"],
        ["PROVENANCE_DEBT_LEDGER.json", "RESULT.json"],
        "Local authority limitations only.",
    ),
    definition(
        "exact78_clean_r70_v71",
        "Close all 58 pinned R7_0 Wave0 Clean rows without consuming successor inputs.",
        [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json",
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53/",
        ],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53/"],
        ["g0_core_governance_v71"],
        ["pinned_r7_0", "58_unique_terminals", "clean_join_ready_separate"],
        ["EXACT78_WAVE0_CLEAN_TERMINAL_MATRIX.json", "RESULT.json"],
        "Synthetic visual Clean only; no geometry or physical-background truth.",
        cpu=86_400,
        gpu=43_200,
        wall=172_800,
        attempts=3,
    ),
    definition(
        "exact78_conversion_cause_ledger_v2",
        "Publish mutually exclusive first blockers, conditional yields and infrastructure/algorithm/provenance attribution.",
        [
            "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json",
            "docs/governance/EXACT78_CONVERSION_RATE_OPTIMIZATION_ZH.md",
        ],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/"],
        ["g0_core_governance_v71"],
        ["156_unique_sessions", "exclusive_first_blocker", "robot_eligible_denominator"],
        ["CONVERSION_CAUSE_LEDGER_V2.json", "CONVERSION_CAUSE_LEDGER_V2.csv", "RESULT.json"],
        "Diagnostic accounting; no automatic authority promotion.",
    ),
    definition(
        "successor_hawor_bounded_v71",
        "Run at most two bounded HaWoR successor rounds on the frozen failure cluster.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/hawor/"],
        ["g0_core_governance_v71", "exact78_conversion_cause_ledger_v2"],
        ["one_canary", "two_regressions", "r7_1_no_overwrite"],
        ["RESULT.json", "WAVE_DELTA.json"],
        "HaWoR monocular MANO remains non-ground-truth 3D.",
        gpu=7_200,
        wall=14_400,
    ),
    definition(
        "successor_role_mask_v71",
        "Run bounded Chips/Poker role-mask successor canaries with re-entry regression.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/role_mask/"],
        ["g0_core_governance_v71", "exact78_conversion_cause_ledger_v2"],
        ["one_canary", "two_regressions", "left_right_reentry", "r7_1_no_overwrite"],
        ["RESULT.json", "WAVE_DELTA.json"],
        "Pixel-role masks only.",
        gpu=7_200,
        wall=14_400,
    ),
    definition(
        "successor_object_identity_v71",
        "Run bounded Poker identity and Chips three-instance successor canaries.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/object_identity/"],
        ["g0_core_governance_v71", "exact78_conversion_cause_ledger_v2"],
        ["one_canary", "two_regressions", "no_chips_union", "r7_1_no_overwrite"],
        ["RESULT.json", "WAVE_DELTA.json"],
        "Task-object identity masks; no pose truth.",
        gpu=7_200,
        wall=14_400,
    ),
    definition(
        "object_pose_hypothesis_v2",
        "Build direct/tracked/attachment object-pose hypotheses while preserving observed-only Object6D.",
        [PLAN, "contracts/object_contact_evidence_v1.schema.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/"],
        ["g0_core_governance_v71"],
        ["direct_object6d_immutable", "evidence_dag_acyclic", "attachment_non_authoritative"],
        ["OBJECT_CONTACT_EVIDENCE_INDEX.json", "RESULT.json"],
        "Hypotheses may not promote formal Object6D or contact truth.",
        gpu=7_200,
        wall=28_800,
    ),
    definition(
        "contact_evidence_dag_v1",
        "Validate contact seeds and attachment propagation using an acyclic evidence graph.",
        [PLAN, "contracts/object_contact_evidence_v1.schema.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact/"],
        ["g0_core_governance_v71"],
        ["geometry_fixtures_pass", "no_attachment_self_proof", "unknown_is_legal"],
        ["CONTACT_EVIDENCE_RESULT.json", "RESULT.json"],
        "Contact hypothesis only unless independent evidence is present.",
        gpu=7_200,
        wall=28_800,
    ),
    definition(
        "occlusion_silver_v1",
        "Publish provenance/coverage/z-buffer/temporal occlusion Silver evidence without accuracy claims.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact/CONTACT_EVIDENCE_RESULT.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_silver/"],
        ["g0_core_governance_v71", "contact_evidence_dag_v1"],
        ["no_accuracy_without_gold", "provenance", "unknown_reported", "byte_exact_outside_band"],
        ["OCCLUSION_SILVER_REPORT.json", "RESULT.json"],
        "Silver visual consistency, not ground-truth occlusion accuracy.",
        gpu=7_200,
        wall=28_800,
    ),
    definition(
        "occlusion_goldset_v1",
        "Freeze and score an independent 120-frame-per-task human-reviewed ownership goldset, or close blocked.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_silver/OCCLUSION_SILVER_REPORT.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/occlusion_gold/"],
        ["g0_core_governance_v71"],
        ["frames_frozen_before_scoring", "independent_review", "blocked_if_absent"],
        ["CONTACT_OCCLUSION_GOLDSET_V1.json", "RESULT.json"],
        "Gold accuracy is unavailable without independent frozen labels.",
    ),
    definition(
        "robot_geometry_v1",
        "Produce pose-only Robot geometry and unified self-collision QA independently of Clean.",
        [PLAN, "configs/systems/robot/README.md", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/"],
        ["g0_core_governance_v71"],
        ["4_to_24_to_full", "unified_zbuffer", "self_collision", "30s_frame_total"],
        ["ROBOT_GEOMETRY_TERMINAL_MATRIX.json", "RESULT.json"],
        "Digital visual trajectory; control_ground_truth=false.",
        cpu=86_400,
        gpu=43_200,
        wall=172_800,
    ),
    definition(
        "metric_contact_robot_v1",
        "Evaluate Robot contact only where formal Object6D and legal contact evidence exist.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/ROBOT_GEOMETRY_TERMINAL_MATRIX.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/metric_contact_robot/"],
        ["g0_core_governance_v71", "robot_geometry_v1", "contact_evidence_dag_v1"],
        ["formal_object6d_only", "digital_residual_claim_limit", "no_attachment_self_proof"],
        ["METRIC_CONTACT_ROBOT_MATRIX.json", "RESULT.json"],
        "Digital contact residuals only; no physical contact accuracy.",
        cpu=43_200,
        wall=86_400,
    ),
    definition(
        "robotized_compositor_causal_v1",
        "Build causal Robotized RGB from Clean, Robot geometry and legal object appearance sources.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/ROBOT_GEOMETRY_TERMINAL_MATRIX.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robotized_causal/"],
        ["g0_core_governance_v71", "robot_geometry_v1"],
        ["causal_donor_only", "unified_zbuffer", "unknown_masked", "no_clean_as_object"],
        ["ROBOTIZED_RGB_LEDGER.json", "RESULT.json"],
        "Causal visual training input only; not physical rendering truth.",
        gpu=43_200,
        wall=172_800,
    ),
    definition(
        "visual_aux_chips_pair_v1",
        "Train matched Chips Raw and Robotized H50 future-2D branches independently of Poker.",
        [PLAN, "src/chaoyang/human_ego/README.md", VISUAL_AUX_UPSTREAM_STATE],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/chips/"],
        ["g0_core_governance_v71", "robotized_compositor_causal_v1"],
        ["paired_selector", "causal_input", "epoch0_both", "value_gate"],
        ["CHECKPOINT_PAIR_RESULT.json", "loss_curves/", "metrics/"],
        "Single-seed future-2D Visual Aux; not policy or action truth.",
        cpu=43_200,
        gpu=86_400,
        wall=172_800,
        attempts=3,
    ),
    definition(
        "visual_aux_poker_pair_v1",
        "Train matched Poker Raw and Robotized H50 future-2D branches independently of Chips.",
        [PLAN, "src/chaoyang/human_ego/README.md", VISUAL_AUX_UPSTREAM_STATE],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/poker/"],
        ["g0_core_governance_v71", "robotized_compositor_causal_v1"],
        ["paired_selector", "causal_input", "epoch0_both", "value_gate"],
        ["CHECKPOINT_PAIR_RESULT.json", "loss_curves/", "metrics/"],
        "Single-seed future-2D Visual Aux; not policy or action truth.",
        cpu=43_200,
        gpu=86_400,
        wall=172_800,
        attempts=3,
    ),
    definition(
        "sensor_h0_admission_v1",
        "Build an independent 0909/0910 sensor-session admission ledger.",
        [PLAN, "docs/reference/data/CHIPS_CARDS_HANDLE_RGB30_V1_PROCESSING_GUIDE_ZH.md"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/"],
        ["g0_core_governance_v71"],
        ["dynamic_denominator", "read_only_source", "branch_eligibility_independent"],
        ["SENSOR_PIPELINE_ADMISSION_LEDGER.json", "SENSOR_PIPELINE_ADMISSION_LEDGER.csv", "RESULT.json"],
        "Admission and provenance only.",
    ),
    definition(
        "sensor_h1_hand_v1",
        "Publish Controller-to-Wrist plus MANUS25 canonical hand observations.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/SENSOR_PIPELINE_ADMISSION_LEDGER.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/"],
        ["g0_core_governance_v71", "sensor_h0_admission_v1"],
        ["no_hawor_mix", "hand21_explicit_mapping", "camera_world_local_provenance"],
        ["CANONICAL_HAND_LEDGER.json", "RESULT.json"],
        "HAND21_FROM_MANUS_CONTROLLER is not MANO or anatomical ground truth.",
    ),
    definition(
        "sensor_h2_tactile_v1",
        "Extract synchronized five-finger tactile event sidecars from acquisition HDF5.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/SENSOR_PIPELINE_ADMISSION_LEDGER.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h2_tactile/"],
        ["g0_core_governance_v71", "sensor_h0_admission_v1"],
        ["frame_identity", "valid_and_age", "no_force_truth_claim"],
        ["TACTILE_SIDECAR_LEDGER.json", "RESULT.json"],
        "Contact-event timing evidence only; not force or object identity truth.",
    ),
    definition(
        "sensor_h3_stereo_v1",
        "Run same-session SBS rectification and FoundationStereo with metric internal checks.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/SENSOR_PIPELINE_ADMISSION_LEDGER.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/"],
        ["g0_core_governance_v71", "sensor_h0_admission_v1"],
        ["same_session_calibration", "z_fB_over_d", "registration", "no_native_confidence"],
        ["SENSOR_DEPTH_LEDGER.json", "RESULT.json"],
        "Stereo visible-surface optical-Z; not external metric truth or wrist truth.",
        gpu=43_200,
        wall=172_800,
    ),
    definition(
        "sensor_h4_mask_v1",
        "Segment glove, controller, forearm and task-object instances with a sensor-domain contract.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h0/SENSOR_PIPELINE_ADMISSION_LEDGER.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/"],
        ["g0_core_governance_v71", "sensor_h0_admission_v1"],
        ["separate_roles", "object_protection", "reentry", "no_chips_union"],
        ["SENSOR_MASK_LEDGER.json", "RESULT.json"],
        "Sensor-domain pixel masks only.",
        gpu=43_200,
        wall=172_800,
    ),
    definition(
        "cleanup_current_only_v71",
        "Continue current-only cleanup without touching live dependencies or protected data-side assets.",
        [PLAN, "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_cleanup_v6/task_packet/TASK_PACKET.json", "docs/governance/CURRENT_FILE_LAYOUT.json"],
        ["archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/cleanup/", "archive/"],
        ["g0_core_governance_v71"],
        ["reference_graph", "zero_active_fd_cwd", "explicit_commit_delete", "protected_data_unchanged"],
        ["CLEANUP_DELETION_RECEIPT.json", "RESULT.json"],
        "Cleanup only; proof gaps close as BLOCKED_REFERENCE_PROOF.",
        cpu=43_200,
        wall=86_400,
        attempts=3,
    ),
]


def packet_payload(item: dict[str, Any]) -> dict[str, Any]:
    packet = {
        "schema_version": "exact78-task-packet-v1",
        "plan_revision": "chaoyang-v7.1",
        "task_id": item["task_id"],
        "objective": item["objective"],
        "read_set": item["reads"],
        "write_set": item["writes"],
        "prerequisites": item["prerequisites"],
        "gates": item["gates"],
        "budgets": item["budgets"],
        "attempt_max": item["attempt_max"],
        "stop_condition": "Budget exhausted or one declared terminal is reached; never remain RUNNING indefinitely.",
        "output_contract": item["outputs"],
        "claim_limit": item["claim_limit"],
        "executor_epoch": 1,
        "fencing": {
            "pid_startticks_required": True,
            "immutable_final": True,
            "partial_attempt_is_never_successor_input": True,
        },
        "artifact_revision_contract": {
            "required_revision_status": "VALID_FOR_PINNED_REVISION",
            "forbid_in_place_overwrite": True,
        },
        "ai_io_limits": {
            "max_files_initial_read": 8,
            "max_search_results": 20,
            "max_log_tail_lines": 80,
            "max_directory_depth": 3,
            "full_log_read_requires_failure": True,
        },
        "initial_search_result_limit": 20,
        "initial_log_line_limit": 80,
        "created_at": CREATED_AT,
    }
    errors = validate_task_packet(packet)
    if errors:
        raise RuntimeError(f"invalid packet {item['task_id']}: {errors}")
    return packet


def main() -> int:
    index: list[dict[str, Any]] = []
    for item in TASKS:
        packet = packet_payload(item)
        task_root = PACKET_ROOT / item["task_id"]
        payload = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
        atomic_write_new(task_root / "TASK_PACKET.json", payload)
        card = (
            f"# {item['task_id']}\n\n"
            f"目标：{item['objective']}\n\n"
            f"前置：{', '.join(item['prerequisites']) or '无'}\n\n"
            f"证据边界：{item['claim_limit']}\n\n"
            "只读取 TASK_PACKET.read_set；失败必须在预算内进入明确终态。\n"
        ).encode()
        atomic_write_new(task_root / "CONTEXT_CARD.md", card)
        index.append(
            {
                "task_id": item["task_id"],
                "packet_path": str((task_root / "TASK_PACKET.json").relative_to(ROOT)),
                "packet_sha256": sha256_bytes(payload),
            }
        )
    result = {
        "schema_version": "chaoyang-v71-task-packet-index-v1",
        "plan_revision": "chaoyang-v7.1",
        "created_at": CREATED_AT,
        "status": "PASS",
        "task_packets": index,
        "claim_limit": "Execution routing only; no algorithm authority.",
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_new(RUN_ROOT / "TASK_PACKET_INDEX.json", payload)
    # Bounded current pointer for humans; this is not part of CURRENT_STATUS_RECEIPT.
    atomic_write(
        ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json",
        canonical_bytes(
            {
                "schema_version": "chaoyang-v71-task-packet-pointer-v1",
                "plan_revision": "chaoyang-v7.1",
                "index_path": str((RUN_ROOT / "TASK_PACKET_INDEX.json").relative_to(ROOT)),
                "index_sha256": sha256_bytes(payload),
            }
        ),
        mode=0o444,
    )
    print(RUN_ROOT / "TASK_PACKET_INDEX.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
