"""Unwired thin submit adapter for one known-format new Exact78 raw session.

The public CLI must explicitly call ``submit_one`` for dynamic ``--session``
after the algorithm contract route is published. This module does not change
the existing T0 submit path or automatically dispatch any session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chaoyang.governance.common import (
    REPO_ROOT, artifact_ref, load_json, process_identity, validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import DATA_ROOT, ColdStartInputError
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer


INDEX = REPO_ROOT / "tasks/current/INDEX.json"
STATE = REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
ACTIVE = frozenset({"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"})
TERMINAL = frozenset({"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
                      "FAILED_QUALITY_FINAL"})
CPU_ADMISSION_PHASE = "EXACT78_NEWRAW_CPU_ADMISSION"


def _is_cpu_admission_task(row: dict[str, Any]) -> bool:
    """Historical T2/scene/Robot tasks never occupy this producer namespace."""
    task_id = row.get("task_id")
    return (row.get("phase") == CPU_ADMISSION_PHASE
            or isinstance(task_id, str) and task_id.startswith("exact78_newraw_"))


def _active_packet(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    task_id = row.get("task_id")
    packet_path = REPO_ROOT / "tasks/current" / str(task_id) / "TASK_PACKET.json"
    if (row.get("packet_path") != str(packet_path.relative_to(REPO_ROOT))
            or not packet_path.is_file()
            or artifact_ref(packet_path)["sha256"] != row.get("packet_sha256")):
        raise ColdStartInputError("ACTIVE_PACKET_SHA_OR_PATH_DRIFT")
    packet = load_json(packet_path)
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    if (len(matches) != 1 or matches[0].get("status") != "RUNNING"
            or (state.get("next_task") or {}).get("task_id") != task_id
            or matches[0].get("run_signature") != packet.get("run_signature")
            or matches[0].get("task_packet") != artifact_ref(packet_path)):
        raise ColdStartInputError("ACTIVE_PACKET_STATE_DRIFT")
    writer = packet.get("writer")
    if (not isinstance(writer, dict) or type(writer.get("pid")) is not int
            or type(writer.get("proc_start_ticks")) is not int
            or writer.get("scope") != task_id):
        raise ColdStartInputError("ACTIVE_WRITER_ABSENT")
    attempt, _, _ = producer._paths(task_id)
    if (not (attempt / "WRITER.json").is_file()
            or load_json(attempt / "WRITER.json") != writer
            or matches[0].get("pid") != writer.get("pid")
            or matches[0].get("proc_start_ticks") != writer.get("proc_start_ticks")
            or process_identity(writer["pid"])["start_ticks"] != writer.get("proc_start_ticks")):
        raise ColdStartInputError("ACTIVE_WRITER_FENCE_DRIFT")
    return packet


def _terminal_attachment(packet: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    task_id = packet["task_id"]
    rows = [item for item in state.get("tasks", [])
            if (item.get("session") == packet["inputs"]["source_session"]
                and _is_cpu_admission_task(item))]
    if not rows:
        return None
    if len(rows) != 1 or rows[0].get("task_id") != task_id:
        raise ColdStartInputError("SESSION_PRIOR_DIFFERENT_SIGNATURE_REQUIRES_SUCCESSOR")
    row = rows[0]
    if row.get("status") not in TERMINAL:
        raise ColdStartInputError("SESSION_NONTERMINAL_WITHOUT_ACTIVE_PACKET")
    attempt, packet_path, receipt_path = producer._paths(task_id)
    for ref, expected in ((row.get("task_packet"), packet_path),
                          (row.get("result"), attempt / "RESULT.json"),
                          (row.get("terminal_receipt"), receipt_path)):
        if (not isinstance(ref, dict) or ref.get("path") != str(expected)
                or validate_artifact_ref(ref)):
            raise ColdStartInputError("TERMINAL_REF_SHA_OR_PATH_DRIFT")
    frozen = load_json(packet_path)
    result, receipt = load_json(attempt / "RESULT.json"), load_json(receipt_path)
    if (frozen.get("run_signature") != packet["run_signature"]
            or frozen.get("inputs", {}).get("descriptor_sha256")
               != packet["inputs"]["descriptor_sha256"]
            or result.get("task_id") != task_id
            or result.get("status") != row["status"]
            or result.get("run_signature") != packet["run_signature"]
            or receipt.get("task_id") != task_id
            or receipt.get("status") != row["status"]
            or receipt.get("result") != row["result"]):
        raise ColdStartInputError("TERMINAL_SIGNATURE_OR_RESULT_DRIFT")
    return {"status": "ATTACHED_TO_TERMINAL_CPU_ADMISSION", "task_id": task_id,
            "terminal_status": row["status"], "result": row["result"],
            "run_signature": packet["run_signature"], "execution_started_here": False,
            "visual_qualified": False, "training_eligible": False}


def plan_one(session: Path, *, allowed_root: Path = DATA_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only same-signature attach/first-run decision; never registers."""
    session = session.resolve(strict=True)
    initial_index = load_json(INDEX)
    initial_allowed = [row for row in initial_index.get("task_packets", [])
                       if row.get("execution_allowed") is True]
    if len(initial_allowed) > 1:
        raise ColdStartInputError("MULTIPLE_ACTIVE_PACKETS")
    if initial_allowed:
        state = load_json(STATE)
        active = _active_packet(initial_allowed[0], state)
        if active.get("inputs", {}).get("source_root") != str(session):
            raise ColdStartInputError("ACTIVE_DIFFERENT_WRITER_CONFLICT")
    prepared = producer.preflight(session, allowed_root=allowed_root)
    packet = producer.packet_value(prepared)
    index, state = load_json(INDEX), load_json(STATE)
    allowed = [row for row in index.get("task_packets", [])
               if row.get("execution_allowed") is True]
    if len(allowed) > 1:
        raise ColdStartInputError("MULTIPLE_ACTIVE_PACKETS")
    if allowed:
        active = _active_packet(allowed[0], state)
        if (active.get("task_id") != packet["task_id"]
                or active.get("run_signature") != packet["run_signature"]
                or active.get("inputs", {}).get("descriptor_sha256")
                   != packet["inputs"]["descriptor_sha256"]):
            raise ColdStartInputError("ACTIVE_DIFFERENT_SIGNATURE_OR_WRITER_CONFLICT")
        return {"status": "ATTACHED_TO_ACTIVE_CPU_ADMISSION", "task_id": packet["task_id"],
                "run_signature": packet["run_signature"],
                "task_packet": artifact_ref(producer._paths(packet["task_id"])[1]),
                "execution_started_here": False, "visual_qualified": False,
                "training_eligible": False}, packet
    if state.get("next_task") or any(row.get("status") in ACTIVE for row in state.get("tasks", [])):
        raise ColdStartInputError("ACTIVE_TASK_WITHOUT_PACKET_CONFLICT")
    terminal = _terminal_attachment(packet, state)
    if terminal is not None:
        return terminal, packet
    if any(path.exists() for path in producer._paths(packet["task_id"])):
        raise ColdStartInputError("UNSEALED_ATTEMPT_CONFLICT")
    return {"status": "FIRST_RUN_READY_NOT_REGISTERED", "task_id": packet["task_id"],
            "run_signature": packet["run_signature"],
            "descriptor_sha256": packet["inputs"]["descriptor_sha256"],
            "execution_started_here": False, "requires_algorithm_contract_route": True,
            "visual_qualified": False, "training_eligible": False}, packet


def submit_one(session: Path, *, allowed_root: Path = DATA_ROOT,
               preview: bool = False) -> dict[str, Any]:
    """Future ``chaoyang submit`` hook; never a direct algorithm entry."""
    decision, packet = plan_one(session, allowed_root=allowed_root)
    if preview or decision["status"] != "FIRST_RUN_READY_NOT_REGISTERED":
        return decision
    producer.register(packet)  # fail-closed until CLI operation is contracted
    return producer.run_registered_child(packet)


__all__ = ["plan_one", "submit_one"]
