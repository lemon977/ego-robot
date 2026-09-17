#!/usr/bin/env python3
"""CAS-register Robot30 offline visual microbatch 001."""
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
TASK_ID = "rc1_robot30_offline_visual_batch_001"
PACKET = RUN / "task_packets" / TASK_ID / "TASK_PACKET.json"
OUT = RUN / "scheduler_robot30_offline_batch001/attempts/attempt_0001"
SELECTION = RUN / "robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json"
MATRIX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
PREFIX = RUN / "robot30_prefix_schedule_pilot/attempts/attempt_0002/RESULT.json"
COLLISION_CHIPS = RUN / "robot30_prefix_schedule_pilot/attempts/attempt_0002/collision/chips_attempt2/RESULT.json"
COLLISION_POKER = RUN / "robot30_prefix_schedule_pilot/attempts/attempt_0002/collision/poker/RESULT.json"
SESSIONS = ["get_potato_chips_0903_052", "play_cards_0903_245"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--expected-revision", type=int, required=True); args = parser.parse_args()
    if OUT.exists(): raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision: raise RuntimeError("CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])): raise RuntimeError("task already registered")
    pilot = next((row for row in state.get("tasks", []) if row.get("task_id") == "rc1_robot30_prefix_schedule_pilot"), None)
    if not pilot or pilot.get("status") != "PASSED": raise RuntimeError("prefix pilot not passed")
    reads = [REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json", SELECTION, MATRIX, PREFIX, COLLISION_CHIPS, COLLISION_POKER]
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1", "task_id": TASK_ID,
        "stage": "RC1_ROBOT30_OFFLINE_VISUAL_BATCH_001",
        "objective": "Run one frozen Chips and one frozen Poker v77 full-session offline visual terminal before bounded Robot30 expansion.",
        "non_goals": ["Not causal training input", "No Robot authority", "No contact/control/physical truth", "No claim that 30 selected equals 30 passed"],
        "frozen_inputs": {str(i): artifact_ref(path) for i, path in enumerate(reads)},
        "prerequisites": ["governance=FRESH", "prefix_schedule_pilot=PASSED", "selection=30_per_task_frozen"],
        "read_set": [str(path) for path in reads],
        "write_set": [str(RUN / "robot30_offline_visual/batch_001")],
        "commands": ["python -m tools.governance.validate_governance_state", "python -m tools.run_rc1_robot30_offline_visual_batch --sessions get_potato_chips_0903_052,play_cards_0903_245 ..."],
        "quality_gates": ["terminal_per_session", "hard_geometry_separate_from_soft_pose", "digital_collision", "full_review", "offline_visual_only", "training_eligible_false"],
        "budgets": {"runtime_attempts": 2, "wall_seconds": 14400, "gpu_wait_seconds": 1800},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"],
        "expected_outputs": ["RESULT.json", "ARTIFACT_MANIFEST.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json", "RESULT_SUMMARY.json"],
        "claim_limit": "Pinned v77 offline visual terminal coverage only; the RC1 causal loader must reject these artifacts.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET.parent.mkdir(parents=True, exist_ok=True); atomic_json(PACKET, packet)
    atomic_write(PACKET.parent / "CONTEXT_CARD.md", b"# Robot30 offline visual batch 001\n\nRun exactly one Chips and one Poker session. These outputs are review-only and must remain rejected by the causal training loader.\n")
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH); source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old = load_json(source); by_id = {str(row["task_id"]): row for row in old.get("task_packets", [])}
    by_id[TASK_ID] = {"task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(PACKET)["sha256"]}
    new_index = dict(old); new_index.update(packet_revision="RC1_FINAL_0006_ROBOT30_OFFLINE_BATCH001", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    index_path = OUT / "TASK_PACKET_INDEX.json"; created = now_iso()
    state["tasks"].append({"task_id": TASK_ID, "phase": packet["stage"], "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(PACKET)})
    state["next_task"] = {"task_id": TASK_ID, "session": None, "prerequisites": packet["prerequisites"], "expected_resource": "CPU plus serialized render resources", "stop_condition": "Two immutable session terminals; then decide expansion."}
    OUT.mkdir(parents=True, exist_ok=True); result_path = OUT / "RESULT.json"
    atomic_json(result_path, {"schema_version": "chaoyang-rc1-robot30-offline-scheduler-v1", "task_id": "rc1_schedule_robot30_offline_batch001", "status": "PASSED", "created_at": created, "registered": [TASK_ID], "sessions": SESSIONS, "prefix_collision_snapshot": {"chips": artifact_ref(COLLISION_CHIPS), "poker": artifact_ref(COLLISION_POKER)}})
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_schedule_robot30_offline_batch001", "attempt": 1, "status": "PASSED", "created_at": created, "message": "Registered one Chips + one Poker offline visual v77 microbatch; causal expansion remains budget-gated.", "result": artifact_ref(result_path)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_ROBOT30_OFFLINE_BATCH001_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=new_index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": [TASK_ID]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
