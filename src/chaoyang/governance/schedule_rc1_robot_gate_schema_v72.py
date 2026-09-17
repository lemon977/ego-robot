#!/usr/bin/env python3
"""CAS-register the bounded Robot gate-schema audit successor."""
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
TASK_ID = "rc1_robot_gate_schema_v72_poker189"
PACKET = RUN / "task_packets" / TASK_ID / "TASK_PACKET.json"
OUT = RUN / "scheduler_robot_gate_schema_v72/attempts/attempt_0001"
BATCH = RUN / "robot30_offline_visual/fresh_batch_001/attempt_0002"
WORK = BATCH / "work_sessions/play_cards_0903_189"
COLLISION = BATCH / "hard_soft/sessions/play_cards_0903_189/play_cards_0903_189/collision_full/RESULT.json"
RESULT = RUN / "robot_gate_schema_v72/poker189/attempt_0001/RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"fresh scheduler attempt required: {OUT}")
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError("task already registered")
    reads = [
        REPO_ROOT / "docs/governance/CURRENT_RC1_STATUS_MIN.json",
        BATCH / "RESULT.json",
        WORK / "arm_round2/RESULT.json",
        WORK / "hand_round2/RESULT.json",
        WORK / "render/RESULT.json",
        COLLISION,
        REPO_ROOT / "src/chaoyang/ops/audit_robot_hard_soft_gate_v72.py",
        REPO_ROOT / "tests/test_audit_robot_hard_soft_gate_v72.py",
    ]
    packet = {
        "schema_version": "chaoyang-rc1-task-packet-v1",
        "task_id": TASK_ID,
        "stage": "RC1_ROBOT_GATE_SCHEMA_V72",
        "objective": "Recompute Poker189 hard/soft classification with a fail-closed alias map for the verified v77 arm gate schema.",
        "non_goals": ["No solver rerun", "No threshold change", "No Robot authority", "No causal training eligibility"],
        "frozen_inputs": {str(index): artifact_ref(path) for index, path in enumerate(reads)},
        "prerequisites": ["governance=FRESH", "v77_fullsession_terminal=COMPLETED", "schema_unit_tests=PASS"],
        "read_set": [str(path) for path in reads],
        "write_set": [str(RESULT.parent)],
        "commands": [
            "python -m pytest -q tests/test_audit_robot_hard_soft_gate_v72.py",
            "python -m tools.audit_robot_hard_soft_gate_v72 --batch-root ... --collision-root ... --output ...",
        ],
        "quality_gates": ["all_present_aliases_true", "missing_alias_fails_closed", "collision_unchanged", "state_limits_unchanged"],
        "budgets": {"runtime_attempts": 1, "wall_seconds": 600},
        "stop_conditions": ["PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL"],
        "expected_outputs": ["RESULT.json", "RESULT_SUMMARY.json", "RUN_RECEIPT.json", "DECISION.md"],
        "claim_limit": "Schema-compatible digital hard-gate audit only; no solver, contact, control, physical or causal-training authority.",
        "ai_io_limits": {"max_files_initial_read": 8, "max_search_results": 20, "max_log_tail_lines": 80, "max_directory_depth": 3},
    }
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PACKET, packet)
    atomic_write(PACKET.parent / "CONTEXT_CARD.md", b"# Robot gate schema v72\n\nRecompute Poker189 only. Resolve verified v77 arm gate aliases fail-closed; do not change thresholds or solver outputs.\n")
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    source = Path(str(pointer["index_path"])); source = source if source.is_absolute() else REPO_ROOT / source
    old = load_json(source); by_id = {str(row["task_id"]): row for row in old.get("task_packets", [])}
    by_id[TASK_ID] = {"task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(PACKET)["sha256"]}
    new_index = dict(old); new_index.update(packet_revision="RC1_FINAL_0008_ROBOT_GATE_SCHEMA_V72", supersedes_index=artifact_ref(source), task_packets=list(by_id.values()))
    index_path = OUT / "TASK_PACKET_INDEX.json"; created = now_iso()
    state["tasks"].append({"task_id": TASK_ID, "phase": packet["stage"], "plan_execution_revision": "RC1", "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None, "session": "play_cards_0903_189", "pid": None, "proc_start_ticks": None, "gpu_id": None, "task_packet": artifact_ref(PACKET)})
    state["next_task"] = {"task_id": TASK_ID, "session": "play_cards_0903_189", "prerequisites": packet["prerequisites"], "expected_resource": "CPU", "stop_condition": "One immutable successor audit terminal."}
    OUT.mkdir(parents=True, exist_ok=True); schedule_result = OUT / "RESULT.json"
    atomic_json(schedule_result, {"schema_version": "chaoyang-rc1-robot-gate-schema-v72-scheduler-v1", "task_id": "rc1_schedule_robot_gate_schema_v72", "status": "PASSED", "created_at": created, "registered": [TASK_ID]})
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": "rc1_schedule_robot_gate_schema_v72", "attempt": 1, "status": "PASSED", "created_at": created, "message": "Registered fail-closed Robot arm gate schema-alias successor for Poker189.", "result": artifact_ref(schedule_result)}])[-100:]
    published = publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="RC1_ROBOT_GATE_SCHEMA_V72_REGISTERED", expected_revision=args.expected_revision, generator_path=Path(__file__), task_packet_index_path=index_path, task_packet_index_value=new_index)
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "registered": [TASK_ID]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
