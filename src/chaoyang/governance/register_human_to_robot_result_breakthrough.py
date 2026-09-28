"""Register one bounded Robot/Clean/Data result cycle using the existing publisher."""
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

TASK = "human_to_robot_result_breakthrough_20260924"
PLAN = REPO_ROOT / "docs/current/RESULT_BREAKTHROUGH_CARD_ZH.md"
BASELINE = REPO_ROOT / "_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/RESULT.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
LANES = ("robot", "clean", "data", "cleanup")
ACTIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(x.get("status") in ACTIVE for x in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("task_packets") or any(x.get("task_id") == TASK for x in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    if ATTEMPT.exists():
        raise RuntimeError("ATTEMPT_ALREADY_EXISTS")
    baseline = load_json(BASELINE)
    if baseline.get("counts", {}).get("products_quality") != "0/4":
        raise RuntimeError("BASELINE_QUALITY_CHANGED")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=10)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Deliver full-session Robot and Clean quality attempts, Sensor097 field-qualified learning labels, and audited real cleanup.",
        "phase": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
        "plan_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
        "execution_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", "docs/current/RESULT_BREAKTHROUGH_CARD_ZH.md",
                     str(BASELINE.relative_to(REPO_ROOT)), "docs/current/STATUS.json"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current", "docs/archive", "docs/governance", "tests", "src/chaoyang",
                      "_run/cache", "_run/current", "tasks/receipts"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_receipts_read_only",
                          "all_writes_inside_repo", "one_GPU_owner", "cleanup_reference_proof"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 36000, "cpu_threads_soft_cap": 8,
                    "cpu_heavy_jobs_max": 2, "gpu_concurrent_owners_max": 1,
                    "cleanup_coordination_seconds_max": 7200, "new_031_ik_attempts_max": 0,
                    "new_huro_solver_runs_max": 0, "optimizer_steps_max": 0,
                    "same_signature_quality_retry_max": 0},
        "stop_conditions": ["all_authorized_actions_terminal_or_10h_deadline",
                            "first_cycle_all_candidates_rejected_skips_second_cycle",
                            "one_branch_failure_does_not_stop_other_ready_work"],
        "frozen_sessions": {"poker": "play_cards_0902_042", "poker_frames": 171,
                            "chips": "get_potato_chips_0915_042", "chips_frames": 363,
                            "sensor": "play_cards_0916_097", "clean_window": [76, 91],
                            "product_denominator": ["get_potato_chips_0915_007", "play_cards_0915_031",
                                                    "get_potato_chips_0902_103", "play_cards_0902_042"]},
        "baseline": artifact_ref(BASELINE), "plan": artifact_ref(PLAN),
        "weights": "EXISTING_PINNED_WEIGHTS_ONLY_NO_DOWNLOAD",
        "calibration_or_absent": "PINNED_DEVELOPMENT_MOUNT_OR_ABSENT_PER_FIELD",
        "expected_resource": "THREE_LANES_CPU8_GPU_SINGLE_OWNER_PLUS_ONE_IO_WORKER",
        "attempt_max": 1, "executor_epoch": 29,
        "claim_limit": "Offline Robot/Clean quality and field-scoped learning label eligibility; no training, control, deployment or external metric authority.",
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
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {"schema_version": "HUMAN_TO_ROBOT_RESULT_LANE_STATE_V1",
                           "task_id": TASK, "lane": lane, "status": "READY",
                           "execution": "NOT_STARTED", "structure": "NOT_EVALUATED",
                           "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
                           "next_action": "BIND_INPUT_AND_RUN", "evidence": [],
                           "blocker": None, "updated_at": created})
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_RESULT_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan": artifact_ref(PLAN), "baseline": artifact_ref(BASELINE),
        "task_packet": packet_ref, "t0": created, "deadline_at": deadline,
        "status": "REGISTERED_NOT_STARTED"})
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"],
                           "plan_execution_revision": packet["execution_revision"],
                           "attempt": 0, "status": "PENDING", "updated_at": created,
                           "heartbeat_at": None, "session": "human_to_robot_result_breakthrough",
                           "pid": None, "proc_start_ticks": None, "gpu_id": None,
                           "task_packet": packet_ref, "deadline_at": deadline, "t0": created})
    state["next_task"] = {"task_id": TASK, "session": "human_to_robot_result_breakthrough",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": packet["expected_resource"],
                          "stop_condition": "Full results or bounded deadline; local rejection does not stop independent ready work."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered three-result cycle and audited cleanup; old outputs remain immutable."}])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
                 "plan_revision": packet["plan_revision"], "execution_revision": packet["execution_revision"],
                 "status": "PASS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [{"task_id": TASK,
                                   "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                                   "weights": packet["weights"]}],
                 "claim_limit": packet["claim_limit"]}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
