"""Register the finite product-first Human-to-Robot successor."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "human_to_robot_product_first_cleanup_20260923"
PREDECESSOR = "human_to_robot_baseline_v1_convergence_20260923"
PLAN_REVISION = "HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_20260923"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{PLAN_REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ("scene", "sensor", "motion_product", "compare")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    index = load_json(INDEX)
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    if any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("SUCCESSOR_ALREADY_REGISTERED")
    prior_path = REPO_ROOT / f"_run/current/{PREDECESSOR}/attempts/attempt_0001/RESULT.json"
    prior = load_json(prior_path)
    if prior.get("task_terminal_status") != "REJECTED_QUALITY" or prior.get("status") != "TERMINAL_WITH_QUALITY_GAPS":
        raise RuntimeError("PREDECESSOR_NOT_TERMINAL")
    if not PLAN.is_file() or ATTEMPT.exists():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")

    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=12)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Produce 007 product evidence, implement three geometric consumers, independently review four lanes, and actually purge audited obsolete payloads.",
        "phase": PLAN_REVISION, "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION, "route_id": ROUTE,
        "read_set": [
            "tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
            str(prior_path.relative_to(REPO_ROOT)),
            "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1/MASK_MANIFEST.json",
            "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json",
        ],
        "write_set": [
            f"_run/current/{TASK}", f"tasks/current/{TASK}",
            f"tasks/receipts/{TASK.upper()}_RESULT.json", "tasks/current/INDEX.json",
            "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md", "docs/current/STATUS.json",
            "docs/current/visuals/HUMAN_TO_ROBOT_BASELINE_V1_PRODUCT_FIRST",
        ],
        "prerequisites": [
            "governance_PASS_FRESH", "exclusive_current_index", "terminal_convergence_predecessor",
            "raw_processed_sealed_external_read_only", "future_condition_task_retention_frozen_before_delete",
            "single_publisher", "single_GPU_lease_owner", "cpu_total_soft_cap_8",
            "all_new_writes_inside_repo", "no_same_signature_quality_retry",
        ],
        "required_outputs": [
            "attempts/attempt_0001/RUN_SIGNATURE.json",
            *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
            "attempts/attempt_0001/cleanup/DELETE_RECEIPT.json",
            "attempts/attempt_0001/RESULT.json",
        ],
        "budgets": {
            "wall_seconds_max": 43200, "h3_checkpoint_seconds": 10800,
            "h9_freeze_seconds": 32400, "algorithm_change_cutoff_seconds": 36000,
            "cpu_threads_soft_cap": 8, "cpu_heavy_jobs_max": 2,
            "cleanup_io_workers_max": 1, "gpu_concurrent_owners_max": 1,
            "motion_031_root_cause_seconds_max": 5400, "new_031_ik_attempts_max": 0,
            "same_signature_quality_retry_max": 0,
        },
        "stop_conditions": [
            "all_registered_branches_terminal_or_wall_budget_exhausted",
            "ready_implementation_work_prevents_early_all_done_claim",
            "h9_component_and_safe_delete_list_frozen", "no_new_algorithm_after_h10",
            "quality_failure_does_not_auto_retry", "no_predecessor_or_protected_source_writes",
            "training_eligible_false", "control_ground_truth_false",
            "physical_deployable_false", "external_metric_authority_false",
        ],
        "target_sessions": {
            "primary": ["get_potato_chips_0915_007", "play_cards_0915_031"],
            "regression": ["get_potato_chips_0902_103", "play_cards_0902_042"],
            "sensor": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
        },
        "frozen_gates": {
            "position_euclidean_mm": 20.0, "supported_direction_deg": 15.0,
            "full_so3_rotation_deg": 15.0, "new_031_ik_attempts_max": 0,
            "scene_007_window": [181, 196], "scene_031_window": [66, 81],
            "motion_031_timeline_frames": 149, "motion_031_right_input_frames": 102,
            "motion_031_left_input_frames": 0,
        },
        "predecessor": artifact_ref(prior_path), "plan": artifact_ref(PLAN),
        "weights": "EXISTING_PINNED_ONLY",
        "calibration_or_absent": "DEVELOPMENT_MOUNT_PINNED;MEASURED_INSTALLATION_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "FOUR_ISOLATED_LANES_ONE_GPU_ONE_CLEANUP_IO_WORKER",
        "attempt_max": 1, "executor_epoch": 10,
        "fencing": {
            "pid_startticks_required": True, "unique_primary_writer": True,
            "immutable_final": True,
            "lane_writer_roots": {lane: f"_run/current/{TASK}/attempts/attempt_0001/lanes/{lane}" for lane in LANES},
        },
        "claim_limit": "Offline visual and audited cleanup only; no training, contact truth, external metric, control or deployment authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    for lane in LANES:
        root = ATTEMPT / "lanes" / lane
        root.mkdir(parents=True)
        atomic_json(root / "STATE.json", {
            "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_LANE_STATE_V1", "task_id": TASK,
            "lane": lane, "status": "PENDING", "execution": "NOT_STARTED",
            "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
            "improvement": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
            "blocker": None, "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 10},
            "updated_at": created, "training_eligible": False,
            "control_ground_truth": False, "physical_deployable": False,
            "external_metric_authority": False,
        })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan_revision": PLAN_REVISION,
        "predecessor": artifact_ref(prior_path), "plan": artifact_ref(PLAN),
        "mount_contract": artifact_ref(REPO_ROOT / packet["read_set"][4]),
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    atomic_json(REPO_ROOT / f"_run/current/{TASK}/registration_0001/RESULT.json", {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_REGISTRATION_V1",
        "task_id": TASK, "status": "REGISTERED", "t0": created,
        "deadline_at": deadline, "task_packet": packet_ref,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": PLAN_REVISION, "plan_execution_revision": PLAN_REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None,
        "session": "human_to_robot_product_first_cleanup", "pid": None,
        "proc_start_ticks": None, "gpu_id": None, "task_packet": packet_ref,
        "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK, "session": "human_to_robot_product_first_cleanup",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "All finite four-lane and audited cleanup branches terminal, or twelve-hour wall budget exhausted.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered finite product-first successor with independently auditable cleanup; predecessor immutable.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{PLAN_REVISION}_ROUTABLE", "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION, "status": "PASS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{
            "task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
            "weights": "EXISTING_PINNED_ONLY",
        }],
        "claim_limit": "Finite product-first routing only; execution, quality and cleanup remain receipt-bound.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor,
    )
    print(json.dumps({
        "status": "REGISTERED", "task_id": TASK,
        "governance_revision": published["governance_revision"],
        "t0": created, "deadline_at": deadline,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
