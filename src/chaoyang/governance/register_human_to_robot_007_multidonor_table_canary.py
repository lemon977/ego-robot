"""Register one bounded CPU 007 multi-donor table diagnostic."""
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

TASK = "human_to_robot_007_multidonor_table_canary_20260923"
REVISION = "HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_CANARY_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    if any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    prior = REPO_ROOT / "_run/current/human_to_robot_007_attachment_clean_canary_20260923/attempts/attempt_0001/lanes/scene/attachment_clean_window_v1/RESULT.json"
    donors = REPO_ROOT / "_run/current/human_to_robot_007_table_donor_expansion_20260923/attempts/attempt_0001/lanes/scene/table_donor_expansion_v1/RESULT.json"
    if (load_json(prior).get("source_frames") != list(range(181, 197))
            or load_json(donors).get("geometry_pass_count") != 19):
        raise RuntimeError("FROZEN_INPUTS_NOT_READY")
    if not PLAN.is_file() or ATTEMPT.exists():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Fixed 16-frame same-session multi-donor table-only candidate and visual rejection/adoption evidence.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(prior.relative_to(REPO_ROOT)), str(donors.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/run_human_to_robot_007_device_donor_probe.py"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}",
                      "tasks/current/INDEX.json", "docs/current/PLAN.md",
                      "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/STATUS.json",
                      "src/chaoyang/ops/run_human_to_robot_007_multidonor_table_canary.py",
                      "src/chaoyang/ops/finalize_human_to_robot_007_multidonor_table_canary.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_007_multidonor_table_canary.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "CPU_TOTAL_SOFT_LIMIT_8"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             "attempts/attempt_0001/lanes/scene/multidonor_table_window_v1/RESULT.json",
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 3600, "gpu_seconds_max": 0,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1},
        "stop_conditions": ["fixed_16_frame_result_terminal_or_budget_exhausted",
                            "no_old_output_overwrite", "no_full_Clean_or_product_promotion",
                            "training_eligible_false", "control_ground_truth_false", "physical_deployable_false"],
        "frozen_gates": {"session": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
                         "donor_count": 19, "min_donors_per_pixel": 3,
                         "photometric_offset_bound_uint8": 25},
        "prior_clean": artifact_ref(prior), "donors": artifact_ref(donors), "plan": artifact_ref(PLAN),
        "weights": "ABSENT_CPU_ONLY", "calibration_or_absent": "ABSENT_PIXEL_ONLY",
        "expected_resource": "CPU_ONLY_ONE_HOUR", "attempt_max": 1, "executor_epoch": 17,
        "claim_limit": "Table-only 16-frame diagnostic; no object surface, complete Clean or product quality promotion.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    lane = ATTEMPT / "lanes/scene"
    lane.mkdir(parents=True)
    atomic_json(lane / "STATE.json", {
        "schema_version": "HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_LANE_STATE_V1",
        "task_id": TASK, "lane": "scene", "status": "PENDING",
        "execution": "NOT_STARTED", "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
        "adoption": "NOT_ADOPTED", "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 17},
        "updated_at": created, "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan": artifact_ref(PLAN), "prior_clean": artifact_ref(prior),
        "donors": artifact_ref(donors), "status": "REGISTERED_NOT_STARTED",
        "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created,
        "heartbeat_at": None, "session": "human_to_robot_007_multidonor_table_canary",
        "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK, "session": "human_to_robot_007_multidonor_table_canary",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "One fixed 16-frame table-only result or one-hour wall budget exhausted.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered same-session multi-donor table-only diagnostic.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
        "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                          "packet_sha256": packet_ref["sha256"],
                          "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                          "weights": "ABSENT_CPU_ONLY"}],
        "claim_limit": "Table-only CPU diagnostic; no Clean or product quality promotion.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_REGISTERED",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
