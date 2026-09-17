#!/usr/bin/env python3
"""CAS-register the RC1 T5 conversion report after T4 reaches a terminal."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    publish_bundle,
)


RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
PACKET_PATH = RUN / "task_packets/rc1_t5_batch_conversion/TASK_PACKET.json"
OUT = RUN / "scheduler_t5/attempts/attempt_0001"
EXACT_LEDGER = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
CAPACITY = RUN / "t0_freeze_capacity_split/attempts/attempt_0001/CAPACITY_REPORT.json"
T1 = RUN / "rc1_t1_sam31_mask_bounded/attempts/attempt_0004/RESULT.json"
T2 = RUN / "rc1_t2_causal_clean/attempts/attempt_0001/RESULT.json"
T3 = RUN / "rc1_t3_v77_causal_robot/attempts/attempt_0004/RESULT.json"
TERMINALS = {
    "PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE", "BLOCKED_EXTERNAL", "CANCELLED",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--t4-result", required=True, type=Path)
    parser.add_argument("--robot30-index", required=True, type=Path)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("scheduler CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    by_id = {str(row.get("task_id")): row for row in state.get("tasks", [])}
    t4 = by_id.get("rc1_t4_bundle_smoke")
    if t4 is None or t4.get("status") not in TERMINALS:
        raise RuntimeError("T4 has not reached a terminal")
    if "rc1_t5_batch_conversion" in by_id:
        raise RuntimeError("T5 already registered")

    t4_result = args.t4_result.resolve(strict=True)
    robot30_index = args.robot30_index.resolve(strict=True)
    reads = [
        REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json",
        EXACT_LEDGER, CAPACITY, T1, T2, T3, t4_result, robot30_index,
    ]
    attempt = RUN / "rc1_t5_batch_conversion/attempts/attempt_0001"
    command = (
        "python -m tools.build_rc1_t5_conversion_report "
        f"--exact-ledger {EXACT_LEDGER} --capacity-report {CAPACITY} "
        f"--t1-result {T1} --t2-result {T2} --t3-result {T3} "
        f"--t4-result {t4_result} --robot30-index {robot30_index} "
        f"--output-root {attempt}"
    )
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1",
        "task_id": "rc1_t5_batch_conversion",
        "stage": "RC1_T5_BATCH_CONVERSION",
        "objective": "Publish the truthful exact156, RC1 capacity, Robot30 and paired-window conversion snapshot after T4 terminal closure.",
        "non_goals": [
            "No authority promotion",
            "No selection-equals-pass claim",
            "No offline Robot result counted as causal input",
            "No RATE_FINALIZED while selected rows remain unevaluated",
        ],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": [f"rc1_t4_bundle_smoke={t4['status']}_TERMINAL", "governance=FRESH"],
        "read_set": [str(path) for path in reads],
        "write_set": [str(RUN / "rc1_t5_batch_conversion/attempts")],
        "commands": ["python -m tools.governance.validate_governance_state", command],
        "quality_gates": [
            "exact156_partition",
            "source_group_capacity_preserved",
            "robot30_selection_separate_from_pass",
            "offline_separate_from_causal",
            "paired_window_count_receipt_bound",
            "rate_finalized_false_when_pending",
        ],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 1800, "gpu_wait_seconds": 0},
        "stop_conditions": ["PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ"],
        "expected_outputs": ["RESULT.json", "QUALITY_SUMMARY.json", "CONVERSION_REPORT.md"],
        "claim_limit": "RC1 accounting only; no causal bundle, checkpoint, contact truth, control truth or physical authority is implied.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PACKET_PATH, packet)
    atomic_write(
        PACKET_PATH.parent / "CONTEXT_CARD.md",
        b"# rc1_t5_batch_conversion\n\nAggregate only the eight frozen receipts. Keep RATE_FINALIZED false when Robot30 causal/successor rows remain pending and keep both checkpoint pairs BLOCKED_DATA_VOLUME when source groups are insufficient.\n",
    )

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"]))
    if not source.is_absolute():
        source = REPO_ROOT / source
    old_index = load_json(source)
    entries = {str(row["task_id"]): row for row in old_index.get("task_packets", [])}
    entries["rc1_t5_batch_conversion"] = {
        "task_id": "rc1_t5_batch_conversion",
        "packet_path": str(PACKET_PATH.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET_PATH)["sha256"],
    }
    new_index = dict(old_index)
    new_index.update(
        packet_revision="RC1_FINAL_0006_READY_T5",
        supersedes_index=artifact_ref(source),
        task_packets=list(entries.values()),
    )
    index_path = OUT / "TASK_PACKET_INDEX.json"

    created = now_iso()
    state["tasks"].append({
        "task_id": "rc1_t5_batch_conversion", "phase": packet["stage"],
        "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING",
        "updated_at": created, "heartbeat_at": None, "session": None,
        "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": artifact_ref(PACKET_PATH),
    })
    state["next_task"] = {
        "task_id": "rc1_t5_batch_conversion", "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU receipt aggregation",
        "stop_condition": "Immutable truthful conversion snapshot.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "chaoyang-rc1-t5-scheduler-result-v1",
        "task_id": "rc1_schedule_t5_after_t4", "status": "PASSED",
        "created_at": created, "registered": ["rc1_t5_batch_conversion"],
        "t4_terminal": t4["status"],
        "robot30_index": artifact_ref(robot30_index),
    })
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "rc1_schedule_t5_after_t4", "attempt": 1, "status": "PASSED",
        "created_at": created,
        "message": "Registered T5 with frozen exact156, capacity, T1-T4 and Robot30 receipts.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        copy.deepcopy(load_json(AUTHORITY_PATH)), state,
        event_type="RC1_T5_REGISTERED", expected_revision=args.expected_revision,
        generator_path=Path(__file__), task_packet_index_path=index_path,
        task_packet_index_value=new_index,
    )
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": ["rc1_t5_batch_conversion"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
