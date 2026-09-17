#!/usr/bin/env python3
"""Register one immutable current-READY Robot30 offline visual session."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import copy
import json
import re
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH, artifact_ref, atomic_json, atomic_write,
    load_json, now_iso, publish_bundle,
)

RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
SELECTION = RUN / "robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json"
MATRIX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
PREFIX = RUN / "robot30_prefix_schedule_pilot/attempts/attempt_0002/RESULT.json"
BUDGET = RUN / "robot30_budget_preflight/attempts/attempt_0001/RESULT.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--expected-revision", type=int, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.session):
        raise RuntimeError("unsafe session id")
    task_id = f"rc1_robot30_offline_visual_fresh_batch_{args.batch_id:03d}"
    stage = f"RC1_ROBOT30_OFFLINE_VISUAL_FRESH_BATCH_{args.batch_id:03d}"
    packet = RUN / "task_packets" / task_id / "TASK_PACKET.json"
    out = RUN / f"scheduler_robot30_offline_fresh_batch{args.batch_id:03d}/attempts/attempt_0001"
    write_root = RUN / f"robot30_offline_visual/fresh_batch_{args.batch_id:03d}"
    if out.exists(): raise RuntimeError(f"fresh scheduler attempt required: {out}")
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision: raise RuntimeError("CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if any(row.get("task_id") == task_id for row in state.get("tasks", [])): raise RuntimeError("task already registered")
    selected = {row["session_id"]: row for row in load_json(SELECTION)["rows"]}
    matrix_rows = {row["session_id"]: row for row in load_json(MATRIX)["rows"]}
    if args.session not in selected: raise RuntimeError("session outside frozen Robot30 selection")
    if matrix_rows.get(args.session, {}).get("robot_current_state") != "READY_FOR_ROBOT_CURRENT_DRAFT":
        raise RuntimeError("session is not current READY")
    reads = [REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json", SELECTION, MATRIX, PREFIX, BUDGET, REPO_ROOT / "src/chaoyang/ops/audit_robot_hard_soft_gate_v72.py"]
    value = {
        "schema_version": "chaoyang-rc1-task-packet-v1", "task_id": task_id, "stage": stage,
        "objective": f"Run one frozen current-READY v77 full-session offline visual terminal for {args.session}, using v72 schema-compatible hard-gate audit.",
        "non_goals": ["Not causal training input", "No Robot/contact/control/physical authority", "No selection-equals-pass claim"],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": ["governance=FRESH", "selection=frozen", "matrix.robot_current_state=READY_FOR_ROBOT_CURRENT_DRAFT"],
        "read_set": [str(path) for path in reads], "write_set": [str(write_root)],
        "commands": [f"chaoyang run run_rc1_robot30_offline_visual_batch --sessions {args.session} ... --contract-mode fresh"],
        "quality_gates": ["terminal_per_session", "v72_fail_closed_schema", "digital_collision", "full_review", "training_eligible_false"],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 14400, "gpu_wait_seconds": 1800},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json", "RESULT_SUMMARY.json"],
        "claim_limit": "One v77 offline visual terminal with v72 schema audit; rejected by RC1 causal loader.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    packet.parent.mkdir(parents=True, exist_ok=True); atomic_json(packet, value)
    atomic_write(packet.parent / "CONTEXT_CARD.md", f"# Robot30 fresh batch {args.batch_id:03d}\n\nRun exactly {args.session}. Offline review only; never causal training input.\n".encode())
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH); source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old = load_json(source); by_id = {str(row["task_id"]): row for row in old.get("task_packets", [])}
    by_id[task_id] = {"task_id": task_id, "packet_path": str(packet.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(packet)["sha256"]}
    index = dict(old); index.update(packet_revision=f"RC1_FINAL_ROBOT30_FRESH_{args.batch_id:03d}", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    created = now_iso(); index_path = out / "TASK_PACKET_INDEX.json"
    state["tasks"].append({"task_id": task_id, "phase": stage, "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": args.session, "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(packet)})
    state["next_task"] = {"task_id": task_id, "session": args.session, "prerequisites": value["prerequisites"], "expected_resource": "CPU plus serialized render resources", "stop_condition": "One immutable session terminal."}
    out.mkdir(parents=True, exist_ok=True); result = out / "RESULT.json"
    atomic_json(result, {"schema_version": "chaoyang-rc1-robot30-fresh-session-scheduler-v1", "task_id": f"schedule_{task_id}", "status": "PASSED", "created_at": created, "registered": [task_id], "session": args.session})
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": f"schedule_{task_id}", "attempt": 1, "status": "PASSED", "created_at": created, "message": f"Registered current-READY Robot30 fresh session {args.session}.", "result": artifact_ref(result)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_ROBOT30_FRESH_SESSION_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "task_id": task_id, "session": args.session}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
