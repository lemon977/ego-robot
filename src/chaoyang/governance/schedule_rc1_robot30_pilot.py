#!/usr/bin/env python3
"""CAS-register the frozen Robot30 causal scheduled-start pilot."""
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
TASK_ID = "rc1_robot30_prefix_schedule_pilot"
PACKET = RUN / "task_packets" / TASK_ID / "TASK_PACKET.json"
OUT = RUN / "scheduler_robot30_pilot" / "attempts" / "attempt_0001"
SELECTION = RUN / "robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json"
T3 = RUN / "rc1_t3_v77_causal_robot/attempts/attempt_0004/RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("scheduler CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError(f"already registered: {TASK_ID}")
    t3 = next((row for row in state.get("tasks", []) if row.get("task_id") == "rc1_t3_v77_causal_robot"), None)
    if not t3 or t3.get("status") != "PASSED":
        raise RuntimeError("T3 causal canaries must be PASSED")
    reads = [
        REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json",
        REPO_ROOT / "docs/governance/VISUAL_AUX_RC1_CONTRACT.json",
        SELECTION, T3,
    ]
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1", "task_id": TASK_ID,
        "stage": "RC1_ROBOT30_PREFIX_SCHEDULE_PILOT",
        "objective": "Freeze 30 Chips + 30 Poker Robot targets and measure causal prefix recomputation at scheduled starts before any batch expansion.",
        "non_goals": ["Selection is not 30 passes", "No Robot authority", "No training eligibility before digital collision/render/compositor gates", "No bidirectional trajectory adoption"],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": ["governance=FRESH", "rc1_t3_v77_causal_robot=PASSED", "robot30_selection=frozen_30_per_task"],
        "read_set": [str(path) for path in reads],
        "write_set": [str(RUN / "robot30_prefix_schedule_pilot/attempts")],
        "commands": [
            "python -m tools.governance.validate_governance_state",
            "python -m tools.run_rc1_robot_prefix_schedule_pilot ... --starts 15,20,25 (one frozen Chips and one frozen Poker session)",
        ],
        "quality_gates": ["scheduled_starts_frozen", "prefix_only", "finite_valid_sides", "joint_temporal_contract", "no_bidirectional_artifact", "collision_gate_explicit_not_evaluated"],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 7200, "gpu_wait_seconds": 0},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json", "RESULT_SUMMARY.json"],
        "claim_limit": "Causal Robot scheduled-start performance and terminal states only; no collision/compositor eligibility, control truth or physical authority.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PACKET, packet)
    atomic_write(PACKET.parent / "CONTEXT_CARD.md", b"# RC1 Robot30 prefix pilot\n\nFreeze selection and run only two sessions at starts 15/20/25. Selection count is not pass count. Collision remains a required downstream gate.\n")
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old = load_json(source); by_id = {str(row["task_id"]): row for row in old.get("task_packets", [])}
    by_id[TASK_ID] = {"task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(PACKET)["sha256"]}
    new_index = dict(old); new_index.update(packet_revision="RC1_FINAL_0005_ROBOT30_PILOT", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    index_path = OUT / "TASK_PACKET_INDEX.json"
    created = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": packet["stage"], "plan_execution_revision": "RC1", "attempt": 0,
        "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": None,
        "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(PACKET),
    })
    state["next_task"] = {"task_id": TASK_ID, "session": None, "prerequisites": packet["prerequisites"], "expected_resource": "CPU scheduled-prefix pilot", "stop_condition": "Two-session pilot terminal; then freeze P95 or stop."}
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "RESULT.json"
    atomic_json(result_path, {"schema_version": "chaoyang-rc1-robot30-pilot-scheduler-v1", "task_id": "rc1_schedule_robot30_pilot", "status": "PASSED", "created_at": created, "registered": [TASK_ID], "selection": artifact_ref(SELECTION)})
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_schedule_robot30_pilot", "attempt": 1, "status": "PASSED", "created_at": created, "message": "Registered frozen two-session scheduled-prefix pilot before Robot30 expansion.", "result": artifact_ref(result_path)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_ROBOT30_PILOT_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=new_index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": [TASK_ID]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
