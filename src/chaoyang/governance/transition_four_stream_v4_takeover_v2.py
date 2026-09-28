#!/usr/bin/env python3
"""Close the dead V4 recovery wrapper and CAS-register the bounded V2 takeover."""

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
    process_identity,
    publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


OLD = "four_stream_full_pipeline_v4_recovery_v1"
NEW = "four_stream_full_pipeline_v4_takeover_v2"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OLD_ATT = REPO_ROOT / f"_run/current/{OLD}/attempts/attempt_0001"
NEW_REG = REPO_ROOT / f"_run/current/{NEW}/registration_0001"
LANES = ("exact78", "controller_manus", "hawor_retarget", "huro")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == OLD)
    if row.get("status") != "PENDING" or row.get("pid") is not None:
        raise RuntimeError("old recovery is not an unowned PENDING task")
    if state.get("next_task", {}).get("task_id") != OLD:
        raise RuntimeError("old recovery is not the sole current task")
    index = load_json(INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1 or entries[0].get("task_id") != OLD or entries[0].get("execution_allowed") is not True:
        raise RuntimeError("current index does not exclusively route old recovery")
    running = [e for e in state.get("recent_events", []) if e.get("task_id") == OLD and e.get("status") == "RUNNING"]
    if not running:
        raise RuntimeError("no recorded prior recovery writer")
    previous = running[-1]
    ident = process_identity(int(previous["pid"]))
    if ident["alive"] and ident["start_ticks"] == previous.get("proc_start_ticks"):
        raise RuntimeError("previous recovery writer remains alive")
    if (OLD_ATT / "RESULT.json").exists():
        raise RuntimeError("old recovery has acquired a final RESULT; re-evaluate")
    for lane in LANES:
        lane_state = load_json(OLD_ATT / "lanes" / lane / "STATE.json")
        if lane_state.get("writer", {}).get("pid") is not None:
            raise RuntimeError(f"lane still has a writer: {lane}")
    if NEW_REG.exists():
        raise RuntimeError("takeover registration already exists")

    time = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=8)).isoformat(timespec="seconds")
    NEW_REG.mkdir(parents=True)
    atomic_json(NEW_REG / "PREDECESSOR_INDEX.json", index)
    terminal = {
        "schema_version": "chaoyang-v4-recovery-runtime-terminal-v1",
        "task_id": OLD,
        "status": "FAILED_RUNTIME_FINAL",
        "reason_code": "COORDINATOR_EXITED_WITHOUT_RESULT",
        "quality_evaluated": False,
        "partial_evidence_preserved": True,
        "previous_pid": previous["pid"],
        "previous_proc_start_ticks": previous.get("proc_start_ticks"),
        "previous_writer_alive": False,
        "exact78_pair_result": artifact_ref(OLD_ATT / "lanes/exact78/pair_production_0001/RESULT.json"),
        "controller_input_preflight": artifact_ref(OLD_ATT / "lanes/controller_manus/AI1_INPUT_PREFLIGHT.json"),
        "controller_blocker_correction": {
            "old_reason_code": "MISSING_PROCESSED_0916_SESSIONS",
            "finding": "WRONG_ROOT_USED_IN_PREFLIGHT",
            "correct_parent": "/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned/playing_cards",
            "not_an_M0_M1_quality_result": True,
        },
        "claim_limit": "Runtime wrapper failure only. Existing CPU evidence is preserved; no lane quality is passed or failed by this terminal.",
    }
    atomic_json(OLD_ATT / "RECOVERY_RUNTIME_TERMINAL_20260922.json", terminal)
    terminal_ref = artifact_ref(OLD_ATT / "RECOVERY_RUNTIME_TERMINAL_20260922.json")

    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": NEW,
        "objective": "Sole-writer V4 takeover: correct input/route facts, run one frozen bounded candidate per eligible lane, and produce evidence-linked full-session visuals or explicit fail-closed blockers.",
        "phase": "FOUR_STREAM_V4_TAKEOVER_V2",
        "plan_revision": "FOUR_STREAM_V4_TAKEOVER_V2",
        "execution_revision": "FOUR_STREAM_V4_TAKEOVER_V2",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/ALGORITHM_CONTRACT.json",
            "tasks/current/INDEX.json",
            f"tasks/current/{OLD}/TASK_PACKET.json",
            f"_run/current/{OLD}/attempts/attempt_0001/RECOVERY_RUNTIME_TERMINAL_20260922.json",
            f"_run/current/{OLD}/attempts/attempt_0001/lanes/exact78/pair_production_0001/RESULT.json",
            f"_run/current/{OLD}/attempts/attempt_0001/lanes/controller_manus/AI1_INPUT_PREFLIGHT.json",
            "docs/current/FOUR_STREAM_FULL_PIPELINE_V4_HANDOFF_ZH.md",
        ],
        "write_set": [
            f"_run/current/{NEW}",
            "docs/current/visuals/FOUR_STREAM_V4_TAKEOVER_V2",
            f"tasks/receipts/{NEW.upper()}_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH", "exclusive_current_index", "source_processed_archive_sealed_READ_ONLY",
            "registered_chaoyang_run_entry", "per_session_image_domain_and_input_SHA",
            "four_lane_writer_roots_ISOLATED", "single_GPU_lease_owner",
        ],
        "required_outputs": [
            "attempts/attempt_0001/ROUTE_AND_INPUT_AUDIT.json",
            *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
            "attempts/attempt_0001/RUN_SIGNATURE.json",
            "attempts/attempt_0001/RESULT.json",
        ],
        "budgets": {
            "wall_seconds_max": 28800, "cpu_threads_soft_cap": 8,
            "gpu_concurrent_owners_max": 1, "candidate_max_per_lane": 1,
            "same_signature_retry_max": 0,
        },
        "stop_conditions": [
            "deadline_or_budget_terminal_receipt", "quality_C_no_automatic_retry",
            "missing_input_or_producer_BLOCKED_NOT_READY", "no_source_or_sealed_writes",
            "no_clean_to_geometry", "control_ground_truth_false", "physical_deployable_false",
        ],
        "attempt_max": 1,
        "weights": "PER_LANE_PINNED_OR_ABSENT",
        "calibration_or_absent": "PER_LANE_PINNED_OR_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "CPU_GATE_FIRST_THEN_SINGLE_GOVERNED_GPU_LEASE",
        "executor_epoch": 2,
        "fencing": {
            "pid_startticks_required": True, "unique_primary_writer": True, "immutable_final": True,
            "lane_writer_roots": {lane: f"_run/current/{NEW}/attempts/attempt_0001/lanes/{lane}" for lane in LANES},
        },
        "target_sessions": {
            "exact78": "frozen_156_only_no_formal_training_claim",
            "controller_manus": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
            "hawor_retarget": ["get_potato_chips_0915_007", "play_cards_0915_031"],
            "huro": "same_frozen_input_and_robot_objective_as_comparator",
        },
        "image_domain_policy": "PER_SESSION_EXPLICIT_NO_GLOBAL_SOURCE_INDEX",
        "claim_limit": "Development-only visual and numeric evidence. No causal training, external metric truth, control ground truth or physical deployability.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid takeover packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / f"tasks/current/{NEW}/TASK_PACKET.json"
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)

    row.update(status="FAILED_RUNTIME_FINAL", phase="RECOVERY_RUNTIME_TERMINAL", updated_at=time,
               heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None,
               result=terminal_ref, last_attempt_terminal="FAILED_RUNTIME_FINAL",
               last_attempt_reason="COORDINATOR_EXITED_WITHOUT_RESULT")
    state["tasks"].append({
        "task_id": NEW, "phase": "FOUR_STREAM_V4_TAKEOVER_V2", "plan_execution_revision": "FOUR_STREAM_V4_TAKEOVER_V2",
        "attempt": 0, "status": "PENDING", "updated_at": time, "heartbeat_at": None,
        "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": time,
    })
    state["next_task"] = {
        "task_id": NEW, "session": "four_lane_takeover",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "Eight-hour bounded takeover with one candidate per lane or explicit blocker receipts.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [
        {"task_id": OLD, "attempt": 1, "status": "FAILED_RUNTIME_FINAL", "created_at": time,
         "message": "Recovery wrapper exited without RESULT; partial evidence preserved and false AI1 root blocker identified.", "result": terminal_ref},
        {"task_id": NEW, "attempt": 0, "status": "PENDING", "created_at": time,
         "message": "Registered sole-writer corrected V4 takeover with actual execution gate.", "task_packet": packet_ref},
    ])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "FOUR_STREAM_V4_TAKEOVER_V2_ROUTABLE",
        "plan_revision": "FOUR_STREAM_V4_TAKEOVER_V2",
        "execution_revision": "FOUR_STREAM_V4_TAKEOVER_V2",
        "status": "PASS", "supersedes_index": artifact_ref(NEW_REG / "PREDECESSOR_INDEX.json"),
        "task_packets": [{
            "task_id": NEW, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True, "weights": "PER_LANE_PINNED_OR_ABSENT",
        }],
        "claim_limit": "Only the corrected sole-writer V4 takeover is executable.",
    }
    atomic_json(NEW_REG / "RESULT.json", {
        "schema_version": "chaoyang-v4-takeover-registration-v1", "task_id": NEW,
        "status": "REGISTERED", "registered_at": time, "deadline_at": deadline,
        "old_terminal": terminal_ref, "task_packet": packet_ref,
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state, event_type="FOUR_STREAM_V4_TAKEOVER_V2_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor,
    )
    print(json.dumps({"status": "REGISTERED", "task_id": NEW, "revision": published["governance_revision"],
                      "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
