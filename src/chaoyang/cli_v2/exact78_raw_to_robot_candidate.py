"""Unwired three-node candidate driver; no independent queue, packet or writer.

Each launch is an existing fenced executor in its own subprocess. No node may
consume a partial predecessor. This module never reads historic hand actions,
Robot priors, masks, Clean, or a different session's result as producer input.
"""

from __future__ import annotations

import json
import hmac
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from chaoyang.cli import exact78_submit as legacy_submit
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as cpu
from chaoyang.governance import execute_exact78_newraw_hawor_gpu_candidate_v4 as hawor
from chaoyang.governance import execute_exact78_newraw_robot_cpu_candidate_v1 as robot
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, load_json, process_identity,
    validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
TERMINAL = frozenset({"PASSED", "FAILED_RUNTIME_FINAL", "FAILED_QUALITY_FINAL",
                      "BLOCKED_RESOURCE"})
PHASES = ("EXACT78_NEWRAW_CPU_ADMISSION",
          "EXACT78_NEWRAW_HAWOR_GPU_CANDIDATE_V4",
          "EXACT78_NEWRAW_ROBOT_CPU_CANDIDATE")
SCHEMAS = ("exact78-first-submit-cpu-admission-result-v1",
           "exact78-newraw-hawor-gpu-execution-result-v4",
           "exact78-newraw-robot-cpu-execution-result-v1")
RECEIPT_SCHEMAS = ("exact78-first-submit-cpu-admission-receipt-v1",
                   "exact78-newraw-hawor-gpu-receipt-v4",
                   "exact78-newraw-robot-cpu-receipt-v1")
H20_STOP_HEAVY = datetime.fromisoformat("2026-10-01T14:47:25.116287+08:00")
H24_HARD_CUTOFF = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
STAGE_CLEARANCE = timedelta(seconds=120)
STAGE_WALL_SECONDS = {"CPU_ADMISSION": cpu.WALL_SECONDS_MAX,
                      "HAWOR_V4": hawor.WALL_SECONDS_MAX,
                      "ROBOT_V1": robot.WALL_SECONDS_MAX}
ACCEPTANCE_TASK_ID = "exact78_newraw_cli_v2_acceptance_play_cards_0902_002_v1"
ACCEPTANCE_PHASE = "EXACT78_NEWRAW_CLI_V2_CODE_ACCEPTANCE"
ACCEPTANCE_SESSION = "play_cards_0902_002"
ACCEPTANCE_SOURCE_ROOT = Path(
    "/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_002")
ACCEPTANCE_INPUT_SHA = "9e9de7fe6cc21209af6d9c96c1f6df356855d26c86b1b9227930a5a854223f63"
ACCEPTANCE_RESULT = (REPO_ROOT / "_run/current" / ACCEPTANCE_TASK_ID /
                     "attempts/attempt_0001/RESULT.json")
ACCEPTANCE_PACKET = REPO_ROOT / "tasks/current" / ACCEPTANCE_TASK_ID / "TASK_PACKET.json"
ACCEPTANCE_RECEIPT = (REPO_ROOT / "tasks/receipts" /
                      "EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_PLAY_CARDS_0902_002_V1.json")
ALGORITHM_CONTRACT = REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json"
CLI_CODE = ("src/chaoyang/cli_v2/__init__.py", "src/chaoyang/cli_v2/__main__.py",
            "src/chaoyang/cli_v2/exact78_raw_to_robot_candidate.py")
TEST_CODE = "tests/unit/test_exact78_newraw_submit_dag_candidate_v1.py"
ACCEPTANCE_OPERATIONS = (
    "exact78_newraw_cpu_admission_candidate_v1",
    "exact78_newraw_hawor_gpu_candidate_v4",
    "exact78_newraw_robot_cpu_candidate_v1",
)


def _ref(path: Path, expected: Any, label: str) -> dict[str, Any]:
    if (not isinstance(expected, dict) or expected.get("path") != str(path)
            or validate_artifact_ref(expected)):
        raise ColdStartInputError(f"{label}_REF_SHA_OR_PATH_DRIFT")
    return expected


def _accepted_code_for_execution(source: Path, expected_sha: str) -> dict[str, Any]:
    """Exact one-canary immutable code receipt; check-only never invokes this."""
    if source.resolve() != ACCEPTANCE_SOURCE_ROOT or expected_sha != ACCEPTANCE_INPUT_SHA:
        raise ColdStartInputError("CLI_V2_ACCEPTANCE_CANARY_ONLY")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state.get("tasks", [])
            if row.get("task_id") == ACCEPTANCE_TASK_ID]
    if (len(rows) != 1 or rows[0].get("phase") != ACCEPTANCE_PHASE
            or rows[0].get("session") != ACCEPTANCE_SESSION
            or rows[0].get("status") != "PASSED"):
        raise ColdStartInputError("CLI_V2_CODE_ACCEPTANCE_NOT_PASSED")
    row = rows[0]
    entries = [item for item in index.get("task_packets", [])
               if item.get("task_id") == ACCEPTANCE_TASK_ID]
    if (len(entries) != 1 or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(ACCEPTANCE_PACKET.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != row.get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V2_ACCEPTANCE_INDEX_OR_WRITER_DRIFT")
    _ref(ACCEPTANCE_PACKET, row.get("task_packet"), "CLI_V2_ACCEPTANCE_PACKET")
    _ref(ACCEPTANCE_RESULT, row.get("result"), "CLI_V2_ACCEPTANCE_RESULT")
    _ref(ACCEPTANCE_RECEIPT, row.get("terminal_receipt"), "CLI_V2_ACCEPTANCE_RECEIPT")
    packet, result, receipt = (load_json(ACCEPTANCE_PACKET), load_json(ACCEPTANCE_RESULT),
                               load_json(ACCEPTANCE_RECEIPT))
    signature = row.get("run_signature")
    if (packet.get("task_id") != ACCEPTANCE_TASK_ID
            or packet.get("phase") != ACCEPTANCE_PHASE
            or (packet.get("inputs") or {}).get("source_session") != ACCEPTANCE_SESSION
            or (packet.get("inputs") or {}).get("source_root") != str(source)
            or (packet.get("inputs") or {}).get("source_submission_input_sha256") != expected_sha
            or packet.get("run_signature") != signature
            or result.get("schema_version") != "exact78-newraw-cli-v2-acceptance-result-v1"
            or result.get("task_id") != ACCEPTANCE_TASK_ID
            or result.get("status") != "PASSED"
            or result.get("source_session") != ACCEPTANCE_SESSION
            or result.get("source_submission_input_sha256") != expected_sha
            or result.get("run_signature") != signature
            or result.get("task_packet") != row["task_packet"]
            or receipt.get("schema_version") != "exact78-newraw-cli-v2-acceptance-receipt-v1"
            or receipt.get("task_id") != ACCEPTANCE_TASK_ID
            or receipt.get("status") != "PASSED"
            or receipt.get("source_session") != ACCEPTANCE_SESSION
            or receipt.get("run_signature") != signature
            or receipt.get("task_packet") != row["task_packet"]
            or receipt.get("result") != row["result"]):
        raise ColdStartInputError("CLI_V2_ACCEPTANCE_IDENTITY_OR_SEAL_DRIFT")
    frozen_code = result.get("cli_code")
    if not isinstance(frozen_code, dict) or set(frozen_code) != set(CLI_CODE):
        raise ColdStartInputError("CLI_V2_ACCEPTED_CODE_SET_DRIFT")
    if ((packet.get("inputs") or {}).get("cli_code") != frozen_code
            or (packet.get("inputs") or {}).get("test_code") != result.get("test_code")
            or (packet.get("inputs") or {}).get("published_operations") != result.get("published_operations")
            or (packet.get("inputs") or {}).get("cli_check_only") != result.get("cli_check_only")):
        raise ColdStartInputError("CLI_V2_ACCEPTANCE_PACKET_INPUT_DRIFT")
    for relative in CLI_CODE:
        _ref(REPO_ROOT / relative, frozen_code[relative], "CLI_V2_ACCEPTED_CODE")
    _ref(REPO_ROOT / TEST_CODE, result.get("test_code"), "CLI_V2_ACCEPTED_TEST")
    check = result.get("cli_check_only") or {}
    if (check.get("status") != "READY" or check.get("node") != "CPU_ADMISSION"
            or check.get("source_session") != ACCEPTANCE_SESSION
            or not re.fullmatch(r"[0-9a-f]{64}", check.get("stdout_sha256") or "")):
        raise ColdStartInputError("CLI_V2_ACCEPTED_PREFLIGHT_DRIFT")
    published = result.get("published_operations")
    if not isinstance(published, dict) or set(published) != set(ACCEPTANCE_OPERATIONS):
        raise ColdStartInputError("CLI_V2_ACCEPTED_OPERATION_SET_DRIFT")
    contracts = load_json(ALGORITHM_CONTRACT).get("special_status_contracts") or {}
    for key in ACCEPTANCE_OPERATIONS:
        contract, sealed = contracts.get(key) or {}, published[key]
        candidate = contract.get("candidate_operation")
        operation = REPO_ROOT / "src/chaoyang/ops" / f"{candidate}.py"
        if (not isinstance(sealed, dict) or sealed.get("candidate_operation") != candidate
                or not isinstance(candidate, str)
                or not re.fullmatch(r"run_exact78_newraw_[a-z0-9_]+", candidate)
                or contract.get("task_packet_required") is not True
                or contract.get("current_runnable_operation") is not None
                or (contract.get("code_closure") or []).count(sealed.get("operation_ref")) != 1):
            raise ColdStartInputError("CLI_V2_ACCEPTED_OPERATION_ROUTE_DRIFT")
        _ref(operation, sealed["operation_ref"], "CLI_V2_ACCEPTED_OPERATION")
    return row["result"]


def _identity(packet: dict[str, Any], phase: str) -> tuple[str, str]:
    task_id = packet.get("task_id")
    session = (packet.get("inputs") or {}).get("source_session")
    if (packet.get("phase") != phase or not isinstance(task_id, str)
            or not task_id.startswith("exact78_newraw_")
            or not isinstance(session, str) or not session):
        raise ColdStartInputError("NODE_PACKET_IDENTITY_DRIFT")
    return task_id, session


def _live_writer(packet: dict[str, Any], row: dict[str, Any], attempt: Path) -> None:
    writer = packet.get("writer")
    if not isinstance(writer, dict) or writer.get("scope") != packet["task_id"]:
        raise ColdStartInputError("ACTIVE_WRITER_ABSENT")
    frozen = load_json(attempt / "WRITER.json")
    if frozen != writer or row.get("pid") != writer.get("pid") or row.get(
            "proc_start_ticks") != writer.get("proc_start_ticks"):
        raise ColdStartInputError("ACTIVE_WRITER_FENCE_DRIFT")
    pid, ticks = writer.get("pid"), writer.get("proc_start_ticks")
    if (type(pid) is not int or type(ticks) is not int or pid < 2
            or process_identity(pid)["start_ticks"] != ticks):
        raise ColdStartInputError("ACTIVE_WRITER_NOT_LIVE")


def _node(packet: dict[str, Any], phase: str, schema: str) -> dict[str, Any]:
    """Read-only ledger admission for exactly one matching node signature."""
    task_id, session = _identity(packet, phase)
    attempt = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    receipt_path = REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    rows = [row for row in state.get("tasks", [])
            if row.get("phase") == phase and row.get("session") == session]
    if len(rows) > 1 or any(row.get("task_id") != task_id for row in rows):
        raise ColdStartInputError("SAME_SESSION_DIFFERENT_SIGNATURE_REQUIRES_SUCCESSOR")
    active = [row for row in index.get("task_packets", [])
              if row.get("execution_allowed") is True]
    if len(active) > 1:
        raise ColdStartInputError("MULTIPLE_ACTIVE_WRITERS")
    successor_active = bool(active and active[0].get("task_id") != task_id)
    if successor_active:
        # A sealed predecessor may be inspected while its own same-session
        # successor is the unique live writer. That successor is checked again
        # against its exact expected packet/signature in the next _node call.
        live = [item for item in state.get("tasks", [])
                if item.get("task_id") == active[0].get("task_id")]
        later = PHASES[PHASES.index(phase) + 1:]
        if (len(live) != 1 or live[0].get("session") != session
                or live[0].get("phase") not in later
                or live[0].get("status") not in ACTIVE or not rows
                or rows[0].get("status") != "PASSED"
                or (state.get("next_task") or {}).get("task_id") != active[0]["task_id"]):
            raise ColdStartInputError("ACTIVE_DIFFERENT_WRITER_CONFLICT")
    if not rows:
        if (active or state.get("next_task") or any(row.get("status") in ACTIVE
                for row in state.get("tasks", []))):
            raise ColdStartInputError("ACTIVE_TASK_WITHOUT_MATCHING_PACKET")
        if any(path.exists() for path in (attempt, packet_path, receipt_path)):
            raise ColdStartInputError("UNSEALED_PARTIAL_ATTEMPT_CONFLICT")
        return {"status": "READY", "task_id": task_id, "phase": phase}
    row = rows[0]
    if row.get("status") not in ACTIVE | TERMINAL:
        raise ColdStartInputError("NODE_STATE_UNRECOGNIZED")
    if row.get("run_signature") != packet["run_signature"]:
        raise ColdStartInputError("NODE_LEDGER_SIGNATURE_DRIFT")
    _ref(packet_path, row.get("task_packet"), "NODE_PACKET")
    frozen = load_json(packet_path)
    for key in ("task_id", "phase", "execution_mode", "inputs", "runtime_code",
                "config", "run_signature", "commands"):
        if frozen.get(key) != packet.get(key):
            raise ColdStartInputError(f"NODE_SIGNATURE_DRIFT:{key}")
    if active and active[0].get("task_id") == task_id:
        allowed = active[0]
        if (allowed.get("packet_path") != str(packet_path.relative_to(REPO_ROOT))
                or allowed.get("packet_sha256") != row["task_packet"]["sha256"]
                or row.get("status") not in ACTIVE
                or (state.get("next_task") or {}).get("task_id") != task_id
                or row.get("run_signature") != packet["run_signature"]):
            raise ColdStartInputError("ACTIVE_NODE_PACKET_OR_STATE_DRIFT")
        _live_writer(frozen, row, attempt)
        return {"status": "ATTACHED_ACTIVE", "task_id": task_id, "phase": phase,
                "task_state": row["status"], "run_signature": packet["run_signature"]}
    if row.get("status") in ACTIVE or (state.get("next_task") and not successor_active):
        raise ColdStartInputError("NODE_NONTERMINAL_WITHOUT_LIVE_WRITER")
    result_path = attempt / "RESULT.json"
    _ref(result_path, row.get("result"), "NODE_RESULT")
    _ref(receipt_path, row.get("terminal_receipt"), "NODE_RECEIPT")
    result, receipt = load_json(result_path), load_json(receipt_path)
    if (result.get("schema_version") != schema or result.get("task_id") != task_id
            or result.get("status") != row["status"]
            or result.get("source_session") != session
            or receipt.get("schema_version") != RECEIPT_SCHEMAS[PHASES.index(phase)]
            or receipt.get("task_id") != task_id or receipt.get("status") != row["status"]
            or receipt.get("result") != row["result"]):
        raise ColdStartInputError("NODE_TERMINAL_RESULT_OR_RECEIPT_DRIFT")
    if phase == PHASES[0] and result.get("run_signature") != packet["run_signature"]:
        raise ColdStartInputError("CPU_TERMINAL_SIGNATURE_DRIFT")
    if row["status"] == "BLOCKED_RESOURCE":
        if result.get("algorithm_attempts") != 0:
            raise ColdStartInputError("GPU_WAIT_BURNED_ALGORITHM_ATTEMPT")
    return {"status": "ATTACHED_TERMINAL" if row["status"] == "PASSED" else "STOP_TERMINAL",
            "task_id": task_id, "phase": phase, "terminal_status": row["status"],
            "result": row["result"], "receipt": row["terminal_receipt"],
            "run_signature": packet["run_signature"]}


def _sealed_cpu(packet: dict[str, Any], terminal: dict[str, Any]) -> dict[str, Any]:
    """Reuse the existing field-level admission seal validator."""
    from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import (
        build_spec_from_sealed,
    )

    refs = packet["inputs"]
    return build_spec_from_sealed(
        packet=packet,
        descriptor_ref=refs["upstream_admission_descriptor"],
        frame_map_ref=refs["upstream_admission_frame_time_map"],
        seal_ref=refs["upstream_admission_seal"],
        producer_result_ref=terminal["result"],
        producer_receipt_ref=terminal["receipt"],
    )


def _budget(node: str, now: datetime | None = None) -> dict[str, Any] | None:
    """Conservative finite dispatch gate; never changes a child deadline."""
    instant = now or datetime.now(timezone.utc)
    if instant >= H24_HARD_CUTOFF:
        reason = "H24_HARD_CUTOFF"
    elif node != "CPU_ADMISSION" and instant >= H20_STOP_HEAVY:
        reason = "H20_STOP_NEW_HEAVY_STAGES"
    elif instant + timedelta(seconds=STAGE_WALL_SECONDS[node]) + STAGE_CLEARANCE > H24_HARD_CUTOFF:
        reason = "INSUFFICIENT_FULL_STAGE_WALL_BUDGET"
    else:
        return None
    return {"status": "NOT_EVALUATED_BUDGET", "node": node, "reason": reason,
            "h20_stop_heavy": H20_STOP_HEAVY.isoformat(),
            "h24_hard_cutoff": H24_HARD_CUTOFF.isoformat()}


def plan_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    """No writes. Returns one next node or an attached/terminal state."""
    source = legacy_submit._raw_session_path(session)
    if source in legacy_submit._frozen_t0_selected_paths():
        raise ColdStartInputError("T0_FROZEN_SESSION_NOT_NEWRAW_CANDIDATE")
    route = load_json(ROUTE)
    if (route.get("general_raw_to_robot") or {}).get("current_runnable_operation") is not None:
        raise ColdStartInputError("PRODUCTION_ROUTE_MUST_REMAIN_NULL")
    prepared = cpu.preflight(source)
    cpu_packet = cpu.packet_value(prepared)
    actual_sha = cpu_packet["inputs"]["submission_input_sha256"]
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_input_sha256)
            or not hmac.compare_digest(actual_sha, expected_input_sha256)):
        raise ColdStartInputError("PREFROZEN_SUBMISSION_INPUT_SHA256_DRIFT")
    cpu_state = _node(cpu_packet, PHASES[0], SCHEMAS[0])
    base = {"schema_version": "exact78-newraw-submit-dag-candidate-v1",
            "source_session": cpu_packet["inputs"]["source_session"],
            "submission_input_sha256": cpu_packet["inputs"]["submission_input_sha256"],
            "scene_clean": "NOT_WIRED", "general_operation": None,
            "visual_qualified": False, "product_qualified": False,
            "training_eligible": False,
            "code_acceptance": "NOT_CHECKED_CHECK_ONLY"}
    if cpu_state["status"] != "ATTACHED_TERMINAL":
        decision = {**base, "node": "CPU_ADMISSION", **cpu_state}
        return {**decision, **(_budget("CPU_ADMISSION") or {})} if decision["status"] == "READY" else decision
    cpu_result = Path(cpu_state["result"]["path"])
    hawor_packet = hawor.packet_value(cpu_result)
    _sealed_cpu(hawor_packet, cpu_state)
    hawor_state = _node(hawor_packet, PHASES[1], SCHEMAS[1])
    if hawor_state["status"] != "ATTACHED_TERMINAL":
        decision = {**base, "node": "HAWOR_V4", **hawor_state}
        return {**decision, **(_budget("HAWOR_V4") or {})} if decision["status"] == "READY" else decision
    hawor_result = Path(hawor_state["result"]["path"])
    robot_packet = robot.packet_value(hawor_result)  # verifies v4 receipt/native/seal
    robot_state = _node(robot_packet, PHASES[2], SCHEMAS[2])
    if robot_state["status"] != "ATTACHED_TERMINAL":
        decision = {**base, "node": "ROBOT_V1", **robot_state}
        return {**decision, **(_budget("ROBOT_V1") or {})} if decision["status"] == "READY" else decision
    frozen_packet = load_json(REPO_ROOT / "tasks/current" / robot_state["task_id"] /
                              "TASK_PACKET.json")
    result = load_json(Path(robot_state["result"]["path"]))
    if result.get("stage") != robot._verify_stage(frozen_packet):
        raise ColdStartInputError("ROBOT_SEALED_STAGE_DRIFT")
    return {**base, "status": "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
            "node": "ROBOT_V1", "task_id": robot_state["task_id"],
            "result": robot_state["result"], "receipt": robot_state["receipt"]}


def _launch(node: str, session: Path, upstream: Path | None,
            task_id: str) -> dict[str, Any]:
    """Supervise one fenced child to terminal, including after INT/TERM.

    A hard-killed submitter cannot supervise; in that case the child's ledger
    writer fence and PID remain the recovery authority. Never spawn another
    child from this process after an interrupted wait.
    """
    python = "/usr/local/bin/python"
    prefix = [python, "-B", "-m"]
    if node == "CPU_ADMISSION":
        command = [*prefix, "chaoyang.cli", "submit", "exact78-raw-to-robot",
                   "--session", str(session)]
    elif node == "HAWOR_V4" and upstream is not None:
        command = [*prefix, "chaoyang.governance.execute_exact78_newraw_hawor_gpu_candidate_v4",
                   "--producer-result", str(upstream), "--register-and-run"]
    elif node == "ROBOT_V1" and upstream is not None:
        command = [*prefix, "chaoyang.governance.execute_exact78_newraw_robot_cpu_candidate_v1",
                   "--hawor-result", str(upstream), "--register-and-run"]
    else:
        raise ColdStartInputError("DAG_NODE_COMMAND_INVALID")
    if not re.fullmatch(r"[a-z0-9_]+", task_id):
        raise ColdStartInputError("DAG_TASK_ID_UNSAFE")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1")
    interrupted: list[int] = []
    def defer_signal(signum: int, _frame: Any) -> None:
        interrupted.append(signum)
    previous = {kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)}
    for kind in previous:
        signal.signal(kind, defer_signal)
    child: subprocess.Popen[Any] | None = None
    try:
        if interrupted:
            raise ColdStartInputError("INTERRUPTED_BEFORE_CHILD_SPAWN")
        def event(kind: str, **details: Any) -> None:
            # Emitted to the caller only; all persistent stage logs belong to
            # the child Task Packet write_set, never to this thin wrapper.
            print(json.dumps({"event": kind, "at": datetime.now(timezone.utc).isoformat(),
                              "node": node, "task_id": task_id, **details},
                             sort_keys=True), file=sys.stderr, flush=True)
        event("PREPARED", command=command, source_session=session.name,
              upstream_result=str(upstream) if upstream else None)
        if interrupted:
            event("INTERRUPTED_BEFORE_CHILD_SPAWN", signals=interrupted)
            raise ColdStartInputError("INTERRUPTED_BEFORE_CHILD_SPAWN")
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        event("CHILD_STARTED", pid=child.pid,
              proc_start_ticks=identity.get("start_ticks"))
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        event("CHILD_TERMINAL", pid=child.pid, exit_code=code,
              deferred_signals=interrupted)
        return {"exit_code": code, "interrupted": bool(interrupted),
                "child_pid": child.pid, "child_start_ticks": identity.get("start_ticks")}
    finally:
        # If interrupted, waiting for the fenced executor is deliberate: a
        # signal must not leave a half-registered task or launch a second writer.
        if child is not None and child.poll() is None:
            child.wait()
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def submit_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    """Sequentially invoke existing writers; never register a driver task."""
    source = legacy_submit._raw_session_path(session)
    accepted_ref = _accepted_code_for_execution(source, expected_input_sha256)
    launched: list[str] = []
    for _ in PHASES:
        decision = plan_one(source, expected_input_sha256)
        if decision["status"] != "READY":
            return {**decision, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref,
                    "nodes_launched_here": launched}
        node = decision["node"]
        upstream = None
        if node == "HAWOR_V4":
            prepared = cpu.preflight(source)
            parent = cpu.packet_value(prepared)
            upstream = cpu._paths(parent["task_id"])[0] / "RESULT.json"
        elif node == "ROBOT_V1":
            prepared = cpu.preflight(source)
            parent = cpu.packet_value(prepared)
            hawor_id = hawor.packet_value(cpu._paths(parent["task_id"])[0] /
                                           "RESULT.json")["task_id"]
            upstream = hawor._paths(hawor_id)[0] / "RESULT.json"
        if budget := _budget(node):
            return {**decision, **budget, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref,
                    "nodes_launched_here": launched}
        launch = _launch(node, source, upstream, decision["task_id"])
        launched.append(node)
        after = plan_one(source, expected_input_sha256)  # seal before next node
        if launch["exit_code"] != 0 and after.get("status") not in {"STOP_TERMINAL", "ATTACHED_ACTIVE"}:
            raise ColdStartInputError(f"{node}_EXECUTOR_EXIT_{launch['exit_code']}_WITHOUT_TERMINAL")
        if launch["interrupted"]:
            return {**after, "status": "INTERRUPTED_AFTER_FENCED_CHILD_TERMINAL",
                    "last_node_status": after["status"], "nodes_launched_here": launched,
                    "code_acceptance": "PASSED", "code_acceptance_result": accepted_ref,
                    "child_pid": launch["child_pid"],
                    "child_start_ticks": launch["child_start_ticks"]}
        if after.get("node") == node:
            if after.get("status") == "READY":
                raise ColdStartInputError(f"{node}_UNSEALED_AFTER_EXECUTOR")
            return {**after, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref,
                    "nodes_launched_here": launched}
    return {**plan_one(source, expected_input_sha256), "code_acceptance": "PASSED",
            "code_acceptance_result": accepted_ref, "nodes_launched_here": launched}


__all__ = ["plan_one", "submit_one"]
