"""Register the bounded 007/031 source-coverage successor."""
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

TASK = "human_to_robot_source_coverage_recovery_20260923"
PREDECESSOR = "human_to_robot_product_first_cleanup_20260923"
REVISION = "HUMAN_TO_ROBOT_SOURCE_COVERAGE_RECOVERY_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ("scene", "motion_product")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    if any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    prior_path = REPO_ROOT / f"_run/current/{PREDECESSOR}/attempts/attempt_0001/RESULT.json"
    prior = load_json(prior_path)
    if prior.get("task_terminal_status") != "REJECTED_QUALITY":
        raise RuntimeError("PREDECESSOR_NOT_TERMINAL")
    if not PLAN.is_file() or ATTEMPT.exists():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=3)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Quantify and repair, where evidence permits, 007 missing hand-removal support and 031 target-source discontinuity without rerunning rejected candidates unchanged.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": [
            "tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
            str(prior_path.relative_to(REPO_ROOT)),
            "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/scene/cable_007/window_v1/RESULT.json",
            "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/lanes/motion_product/frame47_audit_031/v1/RESULT.json",
        ],
        "write_set": [
            f"_run/current/{TASK}", f"tasks/current/{TASK}",
            "tasks/current/INDEX.json", "docs/current/PLAN.md",
            "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/STATUS.json",
            "src/chaoyang/ops", "src/chaoyang/governance/current_r3_contracts.py",
            "tests/test_human_to_robot_source_coverage_recovery.py",
        ],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only", "all_writes_inside_repo", "one_GPU_owner"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json", "attempts/attempt_0001/lanes/scene/STATE.json", "attempts/attempt_0001/lanes/motion_product/STATE.json", "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 10800, "cpu_threads_soft_cap": 8, "gpu_concurrent_owners_max": 1, "same_signature_quality_retry_max": 0, "new_031_ik_attempts_max": 0},
        "stop_conditions": ["two_bound_diagnostics_terminal_or_budget_exhausted", "no_old_output_overwrite", "no_unproved_GPU_retry", "training_eligible_false", "control_ground_truth_false", "physical_deployable_false", "external_metric_authority_false"],
        "frozen_gates": {"scene_007_frames": [181, 196], "motion_031_timeline": 149, "motion_031_right_input": 102, "motion_031_left_input": 0},
        "predecessor": artifact_ref(prior_path), "plan": artifact_ref(PLAN),
        "weights": "EXISTING_PINNED_ONLY", "calibration_or_absent": "DEVELOPMENT_MOUNT_PINNED;MEASURED_INSTALLATION_ABSENT",
        "expected_resource": "TWO_FOCUSED_LANES_CPU_FIRST_OPTIONAL_SINGLE_GPU", "attempt_max": 1, "executor_epoch": 11,
        "claim_limit": "Source and input support recovery only; no automatic Clean, IK, Contact, metric, training, control or product adoption authority.",
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
            "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_LANE_STATE_V1", "task_id": TASK,
            "lane": lane, "status": "PENDING", "execution": "NOT_STARTED", "structure": "NOT_EVALUATED",
            "quality": "NOT_EVALUATED", "improvement": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 11},
            "updated_at": created, "training_eligible": False, "control_ground_truth": False,
            "physical_deployable": False, "external_metric_authority": False,
        })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan_revision": REVISION, "predecessor": artifact_ref(prior_path),
        "plan": artifact_ref(PLAN), "status": "REGISTERED_NOT_STARTED", "t0": created,
        "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None,
        "session": "human_to_robot_source_coverage_recovery", "pid": None,
        "proc_start_ticks": None, "gpu_id": None, "task_packet": packet_ref,
        "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK, "session": "human_to_robot_source_coverage_recovery",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "Both finite source-coverage branches terminal, or three-hour wall budget exhausted.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered bounded 007/031 source-coverage recovery; predecessor immutable.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
        "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{
            "task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True, "weights": "EXISTING_PINNED_ONLY",
        }],
        "claim_limit": "Bounded 007/031 source-coverage routing only; no product quality promotion.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_SOURCE_COVERAGE_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor,
    )
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
