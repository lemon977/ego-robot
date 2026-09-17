#!/usr/bin/env python3
"""CAS-register RC1 T4 after T2 and T3 have reached unique terminals."""
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
PACKET_PATH = RUN / "task_packets/rc1_t4_bundle_smoke/TASK_PACKET.json"
OUT = RUN / "scheduler_t4/attempts/attempt_0001"
T2_RESULT = RUN / "rc1_t2_causal_clean/attempts/attempt_0001/RESULT.json"
T3_RESULT = RUN / "rc1_t3_v77_causal_robot/attempts/attempt_0004/RESULT.json"
CAPACITY = RUN / "t0_freeze_capacity_split/attempts/attempt_0001/CAPACITY_REPORT.json"
TERMINALS = {
    "PASSED",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("scheduler CAS revision mismatch")

    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    by_id = {str(row.get("task_id")): row for row in state.get("tasks", [])}
    if "rc1_t4_bundle_smoke" in by_id:
        raise RuntimeError("T4 already registered")
    for task_id in ("rc1_t2_causal_clean", "rc1_t3_v77_causal_robot"):
        row = by_id.get(task_id)
        if row is None or row.get("status") not in TERMINALS:
            raise RuntimeError(f"dependency has not reached a terminal: {task_id}")

    reads = [
        REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json",
        REPO_ROOT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json",
        T2_RESULT,
        T3_RESULT,
        CAPACITY,
    ]
    attempt = RUN / "rc1_t4_bundle_smoke/attempts/attempt_0001"
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1",
        "task_id": "rc1_t4_bundle_smoke",
        "stage": "RC1_T4_BUNDLE_SMOKE",
        "objective": "Fail closed or build one legal causal paired bundle and 200-step smoke; never substitute offline Robot reviews or old Clean pixels.",
        "non_goals": [
            "No authority promotion",
            "No offline bidirectional input",
            "No review MP4 as model input",
            "No checkpoint when causal bundle prerequisites fail",
        ],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": [
            f"rc1_t2_causal_clean={by_id['rc1_t2_causal_clean']['status']}_TERMINAL",
            f"rc1_t3_v77_causal_robot={by_id['rc1_t3_v77_causal_robot']['status']}_TERMINAL",
            "governance=FRESH",
        ],
        "read_set": [str(path) for path in reads],
        "write_set": [str(RUN / "rc1_t4_bundle_smoke/attempts")],
        "commands": [
            "chaoyang validate-governance",
            (
                "chaoyang run run_rc1_t4_bundle_smoke_preflight "
                f"--t2-result {T2_RESULT} --t3-result {T3_RESULT} "
                f"--capacity-report {CAPACITY} --output-root {attempt}"
            ),
        ],
        "quality_gates": [
            "fresh_causal_clean_only",
            "full_session_causal_robot_only",
            "source_group_capacity",
            "offline_and_review_inputs_rejected",
            "no_checkpoint_on_block",
        ],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 1800, "gpu_wait_seconds": 1800},
        "stop_conditions": ["PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": [
            "RESULT.json",
            "ARTIFACT_MANIFEST.json",
            "METRICS.json",
            "RUN_RECEIPT.json",
            "DECISION.md",
            "NEXT_ACTION.json",
            "RESULT_SUMMARY.json",
        ],
        "claim_limit": "Paired-bundle/smoke prerequisite closure only; a negative terminal creates no RGB bundle or checkpoint.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PACKET_PATH, packet)
    atomic_write(
        PACKET_PATH.parent / "CONTEXT_CARD.md",
        b"# rc1_t4_bundle_smoke\n\nOnly consume the frozen T2/T3/T0 receipts. If causal Clean, full-session causal Robot, or source-group capacity is absent, publish BLOCKED_PREREQ without creating a bundle or checkpoint.\n",
    )

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"]))
    if not source.is_absolute():
        source = REPO_ROOT / source
    old_index = load_json(source)
    packet_entries = {str(row["task_id"]): row for row in old_index.get("task_packets", [])}
    packet_entries["rc1_t4_bundle_smoke"] = {
        "task_id": "rc1_t4_bundle_smoke",
        "packet_path": str(PACKET_PATH.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET_PATH)["sha256"],
    }
    new_index = dict(old_index)
    new_index.update(
        packet_revision="RC1_FINAL_0005_READY_T4",
        supersedes_index=artifact_ref(source),
        task_packets=list(packet_entries.values()),
    )
    index_path = OUT / "TASK_PACKET_INDEX.json"

    created = now_iso()
    state["tasks"].append({
        "task_id": "rc1_t4_bundle_smoke",
        "phase": packet["stage"],
        "plan_execution_revision": "RC1",
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": artifact_ref(PACKET_PATH),
    })
    state["next_task"] = {
        "task_id": "rc1_t4_bundle_smoke",
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU fail-closed paired-bundle/smoke preflight",
        "stop_condition": "Legal causal bundle path or explicit BLOCKED_PREREQ.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "chaoyang-rc1-t4-scheduler-result-v1",
        "task_id": "rc1_schedule_t4_after_t2_t3",
        "status": "PASSED",
        "created_at": created,
        "registered": ["rc1_t4_bundle_smoke"],
        "dependency_terminals": {
            "rc1_t2_causal_clean": by_id["rc1_t2_causal_clean"]["status"],
            "rc1_t3_v77_causal_robot": by_id["rc1_t3_v77_causal_robot"]["status"],
        },
    })
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "rc1_schedule_t4_after_t2_t3",
        "attempt": 1,
        "status": "PASSED",
        "created_at": created,
        "message": "Registered T4 after both dependencies reached terminals; negative upstream terminal is preserved fail-closed.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        copy.deepcopy(load_json(AUTHORITY_PATH)),
        state,
        event_type="RC1_T4_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=index_path,
        task_packet_index_value=new_index,
    )
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": ["rc1_t4_bundle_smoke"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
