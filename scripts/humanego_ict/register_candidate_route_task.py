"""Bounded CPU route publication/recovery; never launches model or training.

Uses the existing governance publisher. Keep stdin open and send
``finish <immutable RESULT.json>`` after edits and CPU verification.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import secrets
import select
import sys
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "pick_place_humanego_ict_revo2_route_publication_v1_20261010"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PREDECESSORS = (
    "pick_place_humanego_ict_revo2_development_v1_20261010",
    "pick_place_humanego_ict_revo2_development_v2_20261010",
)


def publish(state: dict, index: dict, event: str) -> None:
    index["supersedes_index"] = artifact_ref(INDEX)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), copy.deepcopy(load_json(INDEX))
    if (state.get("next_task") or any(r.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for r in state["tasks"])
            or any(r.get("execution_allowed") for r in index["task_packets"])):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_ATTEMPT_REQUIRED")
    stat = Path("/proc/self/stat").read_text()
    writer = {"pid": int(stat.split(" ", 1)[0]),
              "proc_start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]),
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    repairs = []
    for task_id in PREDECESSORS:
        row = next(r for r in index["task_packets"] if r["task_id"] == task_id)
        ledger = next(r for r in state["tasks"] if r["task_id"] == task_id)
        if row["execution_class"] != "CPU_DEVELOPMENT_NO_ALGORITHM_EXECUTION" or row["execution_allowed"] is not False:
            raise RuntimeError("UNEXPECTED_PREDECESSOR_INDEX_STATE")
        if ledger["status"] not in {"PASSED", "BLOCKED_PREREQ"}:
            raise RuntimeError("PREDECESSOR_NOT_TERMINAL")
        repairs.append({"task_id": task_id, "before": copy.deepcopy(row),
                        "result": ledger["result"], "after_class": "TERMINAL_LEDGER_EVIDENCE"})
        row["execution_class"] = "TERMINAL_LEDGER_EVIDENCE"
    inputs = {"predecessors": repairs, "index_before": artifact_ref(INDEX),
              "baseline_before": artifact_ref(REPO_ROOT / "docs/current/DUAL_BRANCH_BASELINE_V1.json")}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Repair terminal engineering index classifications and publish a fail-closed HumanEgo ICT Revo2 candidate operation; no training",
        "phase": "HUMANEGO_ICT_REVO2_CPU_ROUTE_PUBLICATION", "plan_revision": "PICK_PLACE_HUMANEGO_ICT_REVO2_V1",
        "execution_revision": TASK.upper(), "execution_mode": "CPU_DEVELOPMENT_ONLY",
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
                     "tasks/current/INDEX.json", "docs/current/DUAL_BRANCH_BASELINE_V1.json",
                     "src/chaoyang/cli/__init__.py", "src/chaoyang/governance/current_r3_contracts.py",
                     "src/chaoyang/humanego_ict/runtime_gate.py", "docs/plans/PICK_PLACE_HUMANEGO_ICT_REVO2_V1/HANDOFF_ZH.md"],
        "read_set_expansions": ["new humanego_ict source/tests and sealed predecessor evidence", "baseline manifest and HumanEgo dependency code"],
        "write_set": ["src/chaoyang/ops/train_pick_place_humanego_ict_revo2_candidate_v1.py",
                      "src/chaoyang/governance/current_r3_contracts.py", "src/chaoyang/humanego_ict/contracts.py",
                      "src/chaoyang/humanego_ict/data.py", "src/chaoyang/humanego_ict/cli.py",
                      "src/chaoyang/humanego_ict/runtime_gate.py", "src/chaoyang/humanego_ict/training.py",
                      "tests/humanego_ict/test_data_readiness.py", "tests/humanego_ict/test_cli_runtime.py",
                      "tests/humanego_ict/test_training.py",
                      "scripts/humanego_ict", "tests/humanego_ict/test_candidate_route.py",
                      "docs/plans/PICK_PLACE_HUMANEGO_ICT_REVO2_V1/ROUTE_STATUS_ZH.md",
                      str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["PREDECESSORS_TERMINAL", "NO_ACTIVE_WRITER", "KNOWN_INDEX_CLASSIFICATION_REPAIR_ONLY"],
        "required_outputs": ["candidate_operation", "INDEX_REPAIR.json", "CPU_TEST_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": {"wall_seconds_max": 3600, "gpu_count": 0, "training_steps": 0, "model_calls": 0},
        "stop_conditions": ["SHA drift", "writer conflict", "unrelated baseline change", "deadline"],
        "claim_limit": "Candidate operation registration and CPU admission tests only; no data qualification, model forward, checkpoint or deployment",
        "inputs": inputs, "writer": writer, "runtime_code": [artifact_ref(Path(__file__))],
        "run_signature": hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=3600)).isoformat(),
        "commands": ["CPU editing, candidate publication and negative admission tests only"],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(ROOT / "INDEX_REPAIR.json", {"status": "PASSED", "repairs": repairs,
                                             "old_packets_unchanged": True, "execution_allowed": False})
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
                           "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
                           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
                           "gpu_id": None, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": TASK, "expected_resource": "CPU_ROUTE_ONLY"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_allowed": True,
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    publish(state, index, "HUMANEGO_ICT_REVO2_ROUTE_REGISTERED_INDEX_REPAIRED")
    print(json.dumps({"status": "REGISTERED", "writer": writer, "task_id": TASK}), flush=True)
    deadline, last = time.monotonic() + 3600, time.monotonic()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 10)
        if ready:
            command = sys.stdin.readline().strip()
            if command.startswith("finish "):
                path = Path(command[7:]).resolve(strict=True)
                if not path.is_relative_to(ROOT) or load_json(path).get("task_id") != TASK:
                    raise RuntimeError("RESULT_BINDING_MISMATCH")
                if artifact_ref(Path(inputs["baseline_before"]["path"])) != inputs["baseline_before"]:
                    raise RuntimeError("BASELINE_POINTER_DRIFT")
                state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
                row = next(r for r in state["tasks"] if r["task_id"] == TASK)
                row.update(status="PASSED", pid=None, proc_start_ticks=None,
                           result=artifact_ref(path), updated_at=now_iso(), heartbeat_at=now_iso())
                state["next_task"] = None
                next(r for r in index["task_packets"] if r["task_id"] == TASK)["execution_allowed"] = False
                receipt = REPO_ROOT / "tasks/receipts" / (TASK.upper() + ".json")
                atomic_json(receipt, {"task_id": TASK, "status": "PASSED", "result": artifact_ref(path),
                                      "claim_limit": packet["claim_limit"]})
                row["receipt"] = artifact_ref(receipt)
                publish(state, index, "HUMANEGO_ICT_REVO2_ROUTE_SEALED")
                print(json.dumps({"status": "PASSED", "receipt": str(receipt)}), flush=True)
                return 0
        if time.monotonic() - last >= 60:
            state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
            row = next(r for r in state["tasks"] if r["task_id"] == TASK)
            if row["status"] != "RUNNING" or row["pid"] != writer["pid"]:
                raise RuntimeError("WRITER_FENCE_DRIFT")
            row.update(updated_at=now_iso(), heartbeat_at=now_iso())
            publish(state, index, "HUMANEGO_ICT_REVO2_ROUTE_HEARTBEAT")
            last = time.monotonic()
    raise RuntimeError("CPU_ROUTE_DEADLINE_REACHED")


if __name__ == "__main__":
    raise SystemExit(main())
