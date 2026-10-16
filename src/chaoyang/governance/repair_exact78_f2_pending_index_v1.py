#!/usr/bin/env python3
"""One-field, separately receipted F2 pending-draft index repair."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys

from chaoyang.governance import execute_exact78_preconvergence_v1 as base
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_preconvergence_f2_pending_index_repair_v1_20260930"
OUTPUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_PRECONVERGENCE_F2_INDEX_REPAIR_V1.json"
F2_RESULT = base.root("F2") / "RESULT.json"
F2_POST = base.root("F2") / "POST_VALIDATION.json"


def corrected_index() -> dict:
    index = base.index_copy()
    rows = [row for row in index["task_packets"] if row["task_id"] == base.PENDING_ID]
    if len(rows) != 1 or rows[0]["execution_class"] != "FUTURE_BLOCKED_DRAFT" or rows[0]["execution_allowed"] is not False:
        raise RuntimeError("PENDING_INDEX_NOT_EXACTLY_EXPECTED")
    rows[0]["execution_class"] = "CATALOG_REFERENCE"
    return index


def main() -> int:
    base.verify_fixed_parent()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if state.get("next_task"):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(base.INDEX)["task_packets"]):
        raise RuntimeError("ACTIVE_PACKET_CONFLICT")
    if OUTPUT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("REPAIR_ALREADY_REGISTERED")
    post = load_json(F2_POST)
    if post["returncode"] == 0 or "indexed task lacks valid execution classification" not in post.get("stdout", ""):
        raise RuntimeError("UNEXPECTED_FAILURE_EVIDENCE")
    index = corrected_index()
    OUTPUT.mkdir(parents=True)
    writer = {
        "pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK,
    }
    inputs = {
        "f2_result": artifact_ref(F2_RESULT), "f2_failed_validation": artifact_ref(F2_POST),
        "index_before": artifact_ref(base.INDEX), "visual_parent": artifact_ref(base.PARENT),
    }
    code = artifact_ref(Path(__file__))
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": [code], "operation": "ONE_INDEX_CLASSIFICATION_REPAIR"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Correct invalid execution_class for non-runnable P1 draft without changing the packet or F2 evidence",
        "phase": "EXACT78_PRECONVERGENCE_F2_INDEX_REPAIR",
        "plan_revision": "EXACT78_PRECONVERGENCE_V1_20260930",
        "execution_revision": "EXACT78_PRECONVERGENCE_F2_INDEX_REPAIR_V1",
        "deadline_at": (datetime.now().astimezone() + timedelta(minutes=20)).isoformat(),
        "read_set": ["AGENTS.md", str(base.PLAN.relative_to(REPO_ROOT)), str(F2_RESULT.relative_to(REPO_ROOT))],
        "write_set": [str(OUTPUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["F2_EVIDENCE_IMMUTABLE", "NO_ACTIVE_TASK", "PENDING_DRAFT_NON_EXECUTABLE"],
        "required_outputs": ["corrected_catalog_classification", "governance_validation"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 1200, "session_attempts_max": 1},
        "stop_conditions": ["UNEXPECTED_INDEX_STATE", "SHA_DRIFT", "DOUBLE_WRITER", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One index classification repair only; P1 remains execution_allowed=false and F2 result unchanged.",
        "writer": writer, "inputs": inputs, "runtime_code": [code],
        "run_signature": signature,
        "commands": ["PYTHONPATH=src python -B -m chaoyang.governance.repair_exact78_f2_pending_index_v1"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_REPAIR_PACKET:" + ";".join(errors))
    atomic_json(OUTPUT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "Exact78_preconvergence", "updated_at": now, "heartbeat_at": now,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": TASK, "session": "Exact78_preconvergence",
                           "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
                           "stop_condition": "single index classification repair"}
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision="EXACT78_F2_INDEX_REPAIR_RUNNING", plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(base.INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_F2_INDEX_REPAIR_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=base.INDEX,
                   task_packet_index_value=index)
    result = {
        "schema_version": "exact78-one-field-index-repair-v1", "task_id": TASK,
        "status": "PASSED", "index_field": "execution_class",
        "before": "FUTURE_BLOCKED_DRAFT", "after": "CATALOG_REFERENCE",
        "pending_execution_allowed": False, "f2_result_unchanged": artifact_ref(F2_RESULT),
        "f2_post_validation_failure": artifact_ref(F2_POST),
        "files_deleted": False, "robot100_modified": False,
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-index-repair-receipt-v1", "task_id": TASK,
                          "status": "PASSED", "result": artifact_ref(OUTPUT / "RESULT.json")})
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUTPUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT), quality="GOVERNANCE_REPAIR_ONLY",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    index["task_packets"][-1]["execution_allowed"] = False
    index.update(packet_revision="EXACT78_F2_INDEX_REPAIR_TERMINAL", status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(base.INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_F2_INDEX_REPAIR_COMPLETE",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=base.INDEX,
                   task_packet_index_value=index)
    validation = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                                cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"),
                                                    "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True)
    atomic_json(OUTPUT / "POST_VALIDATION.json", {"returncode": validation.returncode,
                                                   "stdout": validation.stdout, "stderr": validation.stderr})
    if validation.returncode:
        raise RuntimeError("REPAIR_VALIDATION_FAILED:" + validation.stdout[-3000:])
    print(validation.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
