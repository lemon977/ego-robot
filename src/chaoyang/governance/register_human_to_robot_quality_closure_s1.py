#!/usr/bin/env python3
"""Register the finite quality-closure successor after terminal R2."""
from __future__ import annotations

import argparse
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

TASK = "human_to_robot_quality_closure_s1_20260923"
PREDECESSOR = "human_to_robot_root_cause_gated_r2_20260923"
PLAN_REVISION = "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_20260923"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1_R2_QUALITY_CLOSURE_S1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{PLAN_REVISION}/00_EXECUTION.md"
RUN = REPO_ROOT / f"_run/current/{TASK}"
ATTEMPT = RUN / "attempts/attempt_0001"
LANES = ("scene_evidence", "geometry_contact", "assembly_product", "compare")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
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
    predecessor = REPO_ROOT / f"_run/current/{PREDECESSOR}/attempts/attempt_0001/RESULT.json"
    previous = load_json(predecessor)
    if previous.get("status") != "TERMINAL_WITH_QUALITY_GAPS":
        raise RuntimeError("PREDECESSOR_NOT_TERMINAL_WITH_GAPS")
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)

    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=12)).isoformat(timespec="seconds")
    ATTEMPT.mkdir(parents=True)
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK,
        "objective": "Close the remaining R2 Scene evidence, real geometry/Contact, assembly and fair-comparison quality gaps without same-signature retries.",
        "phase": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1",
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "route_id": ROUTE,
        "read_set": [
            "tasks/current/INDEX.json",
            str(PLAN.relative_to(REPO_ROOT)),
            str(predecessor.relative_to(REPO_ROOT)),
            f"_run/current/{PREDECESSOR}/attempts/attempt_0001/FINAL_VALIDATION.json",
        ],
        "write_set": [
            f"_run/current/{TASK}",
            f"tasks/receipts/{TASK.upper()}_RESULT.json",
            "docs/current/README_ZH.md",
            "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md",
            "docs/current/STATUS.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "exclusive_current_index",
            "terminal_r2_predecessor",
            "read_only_source_processed_archive_sealed_and_r2",
            "single_publisher",
            "single_GPU_lease_owner",
            "cpu_total_soft_cap_8",
            "no_same_signature_quality_retry",
        ],
        "required_outputs": [
            "attempts/attempt_0001/RUN_SIGNATURE.json",
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
            "all_registered_branches_terminal",
            "quality_failure_does_not_auto_retry",
            "evidence_absence_fails_closed_per_consumer",
            "no_source_processed_archive_sealed_or_r2_writes",
            "training_eligible_false",
            "control_ground_truth_false",
            "physical_deployable_false",
        ],
        "target_sessions": {
            "scene": [
                "play_cards_0915_031",
                "get_potato_chips_0915_007",
                "get_potato_chips_0902_103",
                "play_cards_0902_042",
            ],
            "sensor_read_only": [
                "play_cards_0916_097",
                "play_cards_0916_098",
                "play_cards_0916_101",
            ],
        },
        "fault_packages": {
            "DEVICE_CABLE_EVIDENCE": 2,
            "VISIBLE_OBJECT_PROTECTION": 2,
            "REAL_OCCLUSION_REGISTRATION": 2,
            "CONTACT_ROBOT_R1": 2,
            "ASSEMBLY_EVIDENCE": 2,
            "LOCAL_HURO_FAIR_COMPARE": 2,
        },
        "predecessor": artifact_ref(predecessor),
        "plan": artifact_ref(PLAN),
        "weights": "ABSENT_OR_EXISTING_PINNED_ONLY",
        "calibration_or_absent": "PER_CONSUMER_PINNED_OR_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "FOUR_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE",
        "attempt_max": 1,
        "executor_epoch": 6,
        "fencing": {
            "pid_startticks_required": True,
            "unique_primary_writer": True,
            "immutable_final": True,
            "lane_writer_roots": {
                lane: f"_run/current/{TASK}/attempts/attempt_0001/lanes/{lane}"
                for lane in LANES
            },
        },
        "claim_limit": "Offline evidence and candidate quality closure only; no training, contact truth, external metric, control or deployment authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    for lane in LANES:
        root = ATTEMPT / "lanes" / lane
        root.mkdir(parents=True)
        atomic_json(root / "STATE.json", {
            "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_LANE_STATE_V1",
            "task_id": TASK,
            "lane": lane,
            "status": "PENDING",
            "execution": "NOT_STARTED",
            "structure": "NOT_EVALUATED",
            "quality": "NOT_EVALUATED",
            "adoption": "NOT_ADOPTED",
            "blocker": None,
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 6},
            "updated_at": created,
            "training_eligible": False,
        })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_RUN_SIGNATURE_V1",
        "task_id": TASK,
        "plan_revision": PLAN_REVISION,
        "predecessor": artifact_ref(predecessor),
        "status": "REGISTERED_NOT_STARTED",
    })
    atomic_json(RUN / "registration_0001/RESULT.json", {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_REGISTRATION_V1",
        "task_id": TASK,
        "status": "REGISTERED",
        "t0": created,
        "deadline_at": deadline,
        "task_packet": packet_ref,
    })
    state["tasks"].append({
        "task_id": TASK,
        "phase": packet["phase"],
        "plan_execution_revision": PLAN_REVISION,
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": "human_to_robot_quality_closure_s1",
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
        "deadline_at": deadline,
        "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK,
        "session": "human_to_robot_quality_closure_s1",
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "All finite quality-closure branches terminal or 12-hour bounded checkpoint.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered finite R2 quality-closure successor; terminal R2 remains immutable.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{PLAN_REVISION}_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{
            "task_id": TASK,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": "ABSENT_OR_EXISTING_PINNED_ONLY",
        }],
        "claim_limit": "Quality-closure routing only; execution, quality and adoption remain receipt-bound.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    print(json.dumps({
        "status": "REGISTERED",
        "task_id": TASK,
        "governance_revision": published["governance_revision"],
        "t0": created,
        "deadline_at": deadline,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
