#!/usr/bin/env python3
"""Register the finite Human-to-Robot Baseline v1 R1 successor."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta
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
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "human_to_robot_completion_r1_20260922"
OLD_TASK_ID = "four_stream_visual_delivery_v5"
PLAN_REVISION = "HUMAN_TO_ROBOT_COMPLETION_R1_20260922"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REGISTRATION = REPO_ROOT / f"_run/current/{TASK_ID}/registration_0001"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
LANES = ("lane1_scene", "lane2_sensor", "lane3_product", "lane4_compare")
PLAN_ROOT = REPO_ROOT / "docs/plans/HUMAN_TO_ROBOT_COMPLETION_R1_20260922"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()

    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    if state.get("next_task") is not None:
        raise RuntimeError("an existing current task is still selected")
    if any(row.get("status") in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
           for row in state.get("tasks", [])):
        raise RuntimeError("an active task exists")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("current task packet index is not terminal empty")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("R1 task already registered")
    if REGISTRATION.exists() or ATTEMPT.exists():
        raise RuntimeError("R1 immutable registration/attempt already exists")

    old_result_path = REPO_ROOT / f"_run/current/{OLD_TASK_ID}/attempts/attempt_0001/RESULT.json"
    if not old_result_path.is_file():
        raise RuntimeError("missing sealed V5 predecessor result")
    old_result_ref = artifact_ref(old_result_path)
    contract_path = PLAN_ROOT / "DELIVERY_CONTRACT.json"
    if not contract_path.is_file():
        raise RuntimeError("missing R1 delivery contract")

    t0 = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=12)).isoformat(timespec="seconds")
    REGISTRATION.mkdir(parents=True)
    ATTEMPT.mkdir(parents=True)
    atomic_json(REGISTRATION / "PREDECESSOR_TASK_PACKET_INDEX.json", index)

    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": "Execute the user-reviewed Human-to-Robot Baseline v1 R1: four bounded lanes, real 007/031 products, 0902 regressions, three sensor backend consumers, and same-input Local R0 versus HuRo comparison.",
        "phase": "HUMAN_TO_ROBOT_BASELINE_V1_R1",
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/DOC_AUTHORITY_MAP.json",
            "docs/governance/ALGORITHM_CONTRACT.json",
            "tasks/current/INDEX.json",
            "docs/plans/HUMAN_TO_ROBOT_COMPLETION_R1_20260922/07_可执行任务重排_R1.md",
            "docs/plans/HUMAN_TO_ROBOT_COMPLETION_R1_20260922/01_AGENT_RULES_合并补充.md",
            "docs/plans/HUMAN_TO_ROBOT_COMPLETION_R1_20260922/09_任务依赖_R1.json",
            "docs/plans/HUMAN_TO_ROBOT_COMPLETION_R1_20260922/DELIVERY_CONTRACT.json",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            f"tasks/receipts/{TASK_ID.upper()}_RESULT.json",
            "docs/current/PLAN.md",
            "docs/current/README_ZH.md",
            "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/STATUS.json",
            "docs/current/visuals/FOUR_STREAM_V5/R1_20260922",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "exclusive_current_index",
            "sealed_v5_failure_bound_as_predecessor",
            "read_only_source_processed_archive_sealed",
            "single_publisher",
            "single_GPU_lease_owner",
            "cpu_total_soft_cap_8",
            "noncommercial_offline_visual_only",
        ],
        "required_outputs": [
            "attempts/attempt_0001/RUN_SIGNATURE.json",
            "attempts/attempt_0001/PROGRESS_2H.json",
            *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
            "attempts/attempt_0001/RESULT.json",
        ],
        "budgets": {
            "wall_seconds_max": 43200,
            "cpu_threads_soft_cap": 8,
            "gpu_concurrent_owners_max": 1,
            "max_evidence_driven_repairs_per_fault_package": 2,
            "same_signature_quality_retry_max": 0,
            "candidate_max_per_fault_package": 1,
        },
        "stop_conditions": [
            "terminal_with_gaps_at_deadline",
            "quality_failure_does_not_auto_retry",
            "writer_or_gpu_lease_violation_fails_closed",
            "no_source_processed_archive_or_sealed_writes",
            "training_eligible_false",
            "control_ground_truth_false",
            "physical_deployable_false",
        ],
        "target_sessions": {
            "main_products": ["get_potato_chips_0915_007", "play_cards_0915_031"],
            "regressions": ["get_potato_chips_0902_103", "play_cards_0902_042"],
            "sensor": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
            "frame_counts": {"get_potato_chips_0915_007": 378, "play_cards_0915_031": 149,
                             "get_potato_chips_0902_103": 284, "play_cards_0902_042": 171,
                             "play_cards_0916_097": 165, "play_cards_0916_098": 179,
                             "play_cards_0916_101": 122},
        },
        "fault_packages": {"F01": 2, "F02": 2, "F03": 2, "F04": 2,
                            "F05": 2, "F06": 2, "F07": 2, "F08": 2},
        "delivery_contract": artifact_ref(contract_path),
        "predecessor": {"task_id": OLD_TASK_ID, "result": old_result_ref,
                         "immutable": True, "revive_forbidden": True},
        "weights": "ABSENT_OR_EXISTING_PINNED_ONLY",
        "calibration_or_absent": "PER_LANE_PINNED_OR_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "FOUR_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE",
        "attempt_max": 1,
        "executor_epoch": 4,
        "fencing": {"pid_startticks_required": True, "unique_primary_writer": True,
                    "immutable_final": True,
                    "lane_writer_roots": {lane: f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/{lane}" for lane in LANES}},
        "claim_limit": "Offline visual candidate and reproducibility evidence only; no training, contact truth, physical metric authority, control or deployment.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid R1 packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    for lane in LANES:
        lane_path = ATTEMPT / "lanes" / lane
        lane_path.mkdir(parents=True)
        atomic_json(lane_path / "STATE.json", {
            "schema_version": "human-to-robot-r1-lane-state-v1", "task_id": TASK_ID,
            "lane": lane, "status": "PENDING", "updated_at": t0,
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 4},
            "execution": "NOT_STARTED", "integrity": "NOT_STARTED", "quality": "NOT_STARTED",
            "improvement": "NOT_STARTED", "training_eligible": False,
        })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "human-to-robot-r1-run-signature-v1", "task_id": TASK_ID,
        "plan_revision": PLAN_REVISION, "delivery_contract": packet["delivery_contract"],
        "predecessor_result": old_result_ref, "status": "REGISTERED_NOT_STARTED",
    })
    atomic_json(REGISTRATION / "RESULT.json", {
        "schema_version": "human-to-robot-r1-registration-v1", "task_id": TASK_ID,
        "status": "REGISTERED", "t0": t0, "deadline_at": deadline,
        "predecessor_terminal": old_result_ref, "task_packet": packet_ref,
    })
    authority = load_json(AUTHORITY_PATH)
    state["tasks"].append({
        "task_id": TASK_ID, "phase": packet["phase"], "plan_execution_revision": PLAN_REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": t0, "heartbeat_at": None,
        "session": "four_lane_r1", "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": t0,
    })
    state["next_task"] = {"task_id": TASK_ID, "session": "four_lane_r1",
                           "prerequisites": packet["prerequisites"],
                           "expected_resource": packet["expected_resource"],
                           "stop_condition": "12-hour bounded R1 execution or terminal_with_gaps."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID, "attempt": 0, "status": "PENDING", "created_at": t0,
        "message": "Registered user-reviewed Human-to-Robot Baseline v1 R1 successor; V5 preserved.",
        "task_packet": packet_ref, "predecessor": old_result_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{PLAN_REVISION}_ROUTABLE",
        "plan_revision": PLAN_REVISION, "execution_revision": PLAN_REVISION,
        "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{"task_id": TASK_ID,
                          "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                          "packet_sha256": packet_ref["sha256"],
                          "execution_class": "CURRENT_LEDGER_ROUTABLE",
                          "execution_allowed": True, "weights": "ABSENT_OR_EXISTING_PINNED_ONLY"}],
        "claim_limit": "R1 routing only; quality, product adoption and improvement remain receipt-bound.",
    }
    published = publish_bundle(authority, state, event_type="HUMAN_TO_ROBOT_COMPLETION_R1_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK_ID,
                      "governance_revision": published["governance_revision"],
                      "t0": t0, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
