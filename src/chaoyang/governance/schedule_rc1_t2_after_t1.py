#!/usr/bin/env python3
"""CAS-register RC1 T2 after the bounded T1 task reaches any terminal."""
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
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH, artifact_ref, atomic_json, atomic_write,
    load_json, now_iso, publish_bundle,
)


RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
PACKET_PATH = RUN / "task_packets/rc1_t2_causal_clean/TASK_PACKET.json"
OUT = RUN / "scheduler_t2/attempts/attempt_0001"
CLEAN_CONTRACT = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_clean20_cpu_contract_v1/attempts/attempt_0001/RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--mask-result", type=Path, required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("scheduler CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    t1 = next((x for x in state.get("tasks", []) if x.get("task_id") == "rc1_t1_sam31_mask_bounded"), None)
    terminals = {"PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE", "CANCELLED"}
    if not t1 or t1.get("status") not in terminals:
        raise RuntimeError("T1 has not reached a terminal")
    if any(x.get("task_id") == "rc1_t2_causal_clean" for x in state.get("tasks", [])):
        raise RuntimeError("T2 already registered")
    reads = [
        REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json",
        REPO_ROOT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json",
        args.mask_result.resolve(strict=True), CLEAN_CONTRACT,
    ]
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1", "task_id": "rc1_t2_causal_clean",
        "stage": "RC1_T2_CAUSAL_CLEAN",
        "objective": "Close causal Clean pilot prerequisites for Poker245 and Chips039 without reusing old Clean pixels or future donors.",
        "non_goals": ["No authority promotion", "No future donor", "No hidden object texture invention", "No automatic fallback to old Clean"],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": ["governance=FRESH", f"rc1_t1_sam31_mask_bounded={t1['status']}_TERMINAL"],
        "read_set": [str(x) for x in reads],
        "write_set": [str(RUN / "rc1_t2_causal_clean/attempts")],
        "commands": ["chaoyang validate-governance", f"chaoyang run run_rc1_t2_causal_clean_preflight --mask-result {args.mask_result.resolve()} --clean-contract-result {CLEAN_CONTRACT} --output-root {RUN / 'rc1_t2_causal_clean/attempts/attempt_0001'}"],
        "quality_gates": ["no_future_donor", "no_old_clean_rgb", "possible_task_object_fail_closed", "lossless_source_map", "unknown_not_relabelled_background"],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 1800, "gpu_wait_seconds": 1800},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json", "RESULT_SUMMARY.json"],
        "claim_limit": "Causal Clean pilot only; synthetic background is not observed truth and hidden object appearance remains UNKNOWN.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PACKET_PATH, packet)
    atomic_write(PACKET_PATH.parent / "CONTEXT_CARD.md", b"# rc1_t2_causal_clean\n\nClose only the two frozen causal Clean pilots. Do not run GPU work when semantic donor/atlas prerequisites remain open.\n")

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old = load_json(source); by_id = {str(x["task_id"]): x for x in old.get("task_packets", [])}
    by_id["rc1_t2_causal_clean"] = {"task_id": "rc1_t2_causal_clean", "packet_path": str(PACKET_PATH.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(PACKET_PATH)["sha256"]}
    new_index = dict(old); new_index.update(packet_revision="RC1_FINAL_0004_READY_T2", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    index_path = OUT / "TASK_PACKET_INDEX.json"

    created = now_iso()
    state["tasks"].append({"task_id": "rc1_t2_causal_clean", "phase": packet["stage"], "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(PACKET_PATH)})
    state["next_task"] = {"task_id": "rc1_t2_causal_clean", "session": None, "prerequisites": packet["prerequisites"], "expected_resource": "CPU fail-closed prerequisite closure", "stop_condition": "Pilot prerequisites pass or explicit BLOCKED_PREREQ."}
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "RESULT.json"
    atomic_json(result_path, {"schema_version": "chaoyang-rc1-t2-scheduler-result-v1", "task_id": "rc1_schedule_t2_after_t1", "status": "PASSED", "created_at": created, "t1_terminal": t1["status"], "registered": ["rc1_t2_causal_clean"]})
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_schedule_t2_after_t1", "attempt": 1, "status": "PASSED", "created_at": created, "message": "Registered RC1 T2 after bounded T1 terminal; negative T1 terminal is local rather than a global deadlock.", "result": artifact_ref(result_path)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_T2_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=new_index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": ["rc1_t2_causal_clean"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
