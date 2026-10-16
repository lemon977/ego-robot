"""Versioned one-submit successor with phase-exact CPU admission.

The existing CPU, HaWoR and Robot Task Packets remain their own fenced
writers. This coordinator has no queue, packet, persistent output or model
input. Execution is fail-closed until a separate v2 code-acceptance receipt.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any

from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as prior
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, load_json, process_identity, validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


TASK_ID = "exact78_newraw_cli_v2_acceptance_play_cards_0902_002_v2"
PHASE = "EXACT78_NEWRAW_CLI_V3_CODE_ACCEPTANCE"
SOURCE = prior.ACCEPTANCE_SOURCE_ROOT
INPUT_SHA = prior.ACCEPTANCE_INPUT_SHA
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RESULT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001/RESULT.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / "EXACT78_NEWRAW_CLI_V2_ACCEPTANCE_PLAY_CARDS_0902_002_V2.json"
CLI_CODE = tuple(f"src/chaoyang/cli_v3/{name}" for name in (
    "__init__.py", "__main__.py", "exact78_raw_to_robot_candidate_v2.py",
    "exact78_newraw_cpu_submit_phase_exact_v2.py"))
INHERITED_CODE = prior.CLI_CODE
TEST_CODE = "tests/unit/test_exact78_newraw_cpu_submit_phase_exact_v2.py"
LEDGER_FIXTURE = REPO_ROOT / "tests/fixtures/exact78_cards002_cli_v2_ledger_prefix_trap_v1.json"


def _ref(path: Path, value: Any, code: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get("path") != str(path)
            or validate_artifact_ref(value)):
        raise ColdStartInputError(code)
    return value


def accepted_code_for_execution(source: Path, expected_sha: str) -> dict[str, Any]:
    if source.resolve() != SOURCE or not hmac.compare_digest(expected_sha, INPUT_SHA):
        raise ColdStartInputError("CLI_V3_ACCEPTANCE_CANARY_ONLY")
    state, index = load_json(TASK_STATE_PATH), load_json(prior.INDEX)
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID]
    entries = [row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID]
    if (len(rows) != 1 or rows[0].get("phase") != PHASE
            or rows[0].get("session") != SOURCE.name or rows[0].get("status") != "PASSED"
            or len(entries) != 1 or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(PACKET.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != rows[0].get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V3_CODE_ACCEPTANCE_NOT_PASSED")
    row = rows[0]
    _ref(PACKET, row.get("task_packet"), "CLI_V3_ACCEPTANCE_PACKET_SHA_DRIFT")
    _ref(RESULT, row.get("result"), "CLI_V3_ACCEPTANCE_RESULT_SHA_DRIFT")
    _ref(RECEIPT, row.get("terminal_receipt"), "CLI_V3_ACCEPTANCE_RECEIPT_SHA_DRIFT")
    packet, result, receipt = load_json(PACKET), load_json(RESULT), load_json(RECEIPT)
    signature = row.get("run_signature")
    inputs = packet.get("inputs") or {}
    if (packet.get("task_id") != TASK_ID or packet.get("phase") != PHASE
            or packet.get("run_signature") != signature
            or inputs.get("source_session") != SOURCE.name
            or inputs.get("source_root") != str(SOURCE)
            or inputs.get("source_submission_input_sha256") != expected_sha
            or inputs.get("accepted_entry") != "chaoyang.cli_v3"
            or result.get("schema_version") != "exact78-newraw-cli-v2-acceptance-result-v2"
            or result.get("task_id") != TASK_ID or result.get("status") != "PASSED"
            or result.get("source_session") != SOURCE.name
            or result.get("source_submission_input_sha256") != expected_sha
            or result.get("accepted_entry") != "chaoyang.cli_v3"
            or result.get("run_signature") != signature
            or result.get("task_packet") != row["task_packet"]
            or receipt.get("schema_version") != "exact78-newraw-cli-v2-acceptance-receipt-v2"
            or receipt.get("task_id") != TASK_ID or receipt.get("status") != "PASSED"
            or receipt.get("source_session") != SOURCE.name
            or receipt.get("run_signature") != signature
            or receipt.get("task_packet") != row["task_packet"]
            or receipt.get("result") != row["result"]):
        raise ColdStartInputError("CLI_V3_ACCEPTANCE_IDENTITY_DRIFT")
    for key, names in (("cli_code", CLI_CODE), ("inherited_cli_v2_code", INHERITED_CODE)):
        refs = result.get(key)
        if (not isinstance(refs, dict) or set(refs) != set(names)
                or inputs.get(key) != refs):
            raise ColdStartInputError("CLI_V3_ACCEPTED_CODE_SET_DRIFT:" + key)
        for name in names:
            _ref(REPO_ROOT / name, refs[name], "CLI_V3_ACCEPTED_CODE_SHA_DRIFT")
    _ref(REPO_ROOT / TEST_CODE, result.get("test_code"), "CLI_V3_ACCEPTED_TEST_SHA_DRIFT")
    fixture = result.get("ledger_fixture")
    if (inputs.get("test_code") != result["test_code"]
            or inputs.get("ledger_fixture") != fixture
            or not isinstance(fixture, dict)):
        raise ColdStartInputError("CLI_V3_LEDGER_FIXTURE_REF_DRIFT")
    _ref(LEDGER_FIXTURE, fixture, "CLI_V3_LEDGER_FIXTURE_SHA_DRIFT")
    check = result.get("cli_check_only") or {}
    if (inputs.get("cli_check_only") != check
            or check.get("status") != "READY" or check.get("node") != "CPU_ADMISSION"
            or check.get("source_session") != SOURCE.name
            or not re.fullmatch(r"[0-9a-f]{64}", check.get("stdout_sha256") or "")):
        raise ColdStartInputError("CLI_V3_ACCEPTED_PREFLIGHT_DRIFT")
    published = result.get("published_operations")
    if (not isinstance(published, dict)
            or set(published) != set(prior.ACCEPTANCE_OPERATIONS)
            or inputs.get("published_operations") != published):
        raise ColdStartInputError("CLI_V3_ACCEPTED_OPERATION_SET_DRIFT")
    contracts = load_json(prior.ALGORITHM_CONTRACT).get("special_status_contracts") or {}
    for key in prior.ACCEPTANCE_OPERATIONS:
        contract, sealed = contracts.get(key) or {}, published[key]
        candidate = contract.get("candidate_operation")
        operation = REPO_ROOT / "src/chaoyang/ops" / f"{candidate}.py"
        if (not isinstance(sealed, dict) or sealed.get("candidate_operation") != candidate
                or not isinstance(candidate, str)
                or not re.fullmatch(r"run_exact78_newraw_[a-z0-9_]+", candidate)
                or contract.get("task_packet_required") is not True
                or contract.get("current_runnable_operation") is not None
                or (contract.get("code_closure") or []).count(sealed.get("operation_ref")) != 1):
            raise ColdStartInputError("CLI_V3_ACCEPTED_OPERATION_ROUTE_DRIFT")
        _ref(operation, sealed["operation_ref"], "CLI_V3_ACCEPTED_OPERATION_SHA_DRIFT")
    return row["result"]


def plan_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    decision = prior.plan_one(session, expected_input_sha256)
    return {**decision, "candidate_cli": "chaoyang.cli_v3"}


def _launch_cpu(session: Path, expected_sha: str, task_id: str) -> dict[str, Any]:
    command = ["/usr/local/bin/python", "-B", "-m",
               "chaoyang.cli_v3.exact78_newraw_cpu_submit_phase_exact_v2",
               "--session", str(session), "--expected-input-sha256", expected_sha,
               "--register-and-run"]
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
            raise ColdStartInputError("INTERRUPTED_BEFORE_CPU_CHILD_SPAWN")
        print(json.dumps({"event": "CPU_CHILD_PREPARED", "task_id": task_id,
                          "at": datetime.now(timezone.utc).isoformat()}, sort_keys=True),
              file=sys.stderr, flush=True)
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        print(json.dumps({"event": "CPU_CHILD_STARTED", "task_id": task_id,
                          "pid": child.pid, "proc_start_ticks": identity.get("start_ticks")},
                         sort_keys=True), file=sys.stderr, flush=True)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        print(json.dumps({"event": "CPU_CHILD_TERMINAL", "task_id": task_id,
                          "pid": child.pid, "exit_code": code,
                          "deferred_signals": interrupted}, sort_keys=True),
              file=sys.stderr, flush=True)
        return {"exit_code": code, "interrupted": bool(interrupted),
                "child_pid": child.pid, "child_start_ticks": identity.get("start_ticks")}
    finally:
        if child is not None and child.poll() is None:
            child.wait()
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def submit_one(session: Path, expected_input_sha256: str) -> dict[str, Any]:
    source = prior.legacy_submit._raw_session_path(session)
    accepted = accepted_code_for_execution(source, expected_input_sha256)
    launched: list[str] = []
    for _ in prior.PHASES:
        decision = plan_one(source, expected_input_sha256)
        if decision["status"] != "READY":
            return {**decision, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted, "nodes_launched_here": launched}
        node = decision["node"]
        if budget := prior._budget(node):
            return {**decision, **budget, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted, "nodes_launched_here": launched}
        upstream = None
        if node == "HAWOR_V4":
            prepared = prior.cpu.preflight(source)
            parent = prior.cpu.packet_value(prepared)
            upstream = prior.cpu._paths(parent["task_id"])[0] / "RESULT.json"
        elif node == "ROBOT_V1":
            prepared = prior.cpu.preflight(source)
            parent = prior.cpu.packet_value(prepared)
            hawor_id = prior.hawor.packet_value(prior.cpu._paths(parent["task_id"])[0] /
                                                  "RESULT.json")["task_id"]
            upstream = prior.hawor._paths(hawor_id)[0] / "RESULT.json"
        launch = (_launch_cpu(source, expected_input_sha256, decision["task_id"])
                  if node == "CPU_ADMISSION" else
                  prior._launch(node, source, upstream, decision["task_id"]))
        launched.append(node)
        after = plan_one(source, expected_input_sha256)
        if launch["exit_code"] != 0 and after.get("status") not in {"STOP_TERMINAL", "ATTACHED_ACTIVE"}:
            raise ColdStartInputError(f"{node}_EXECUTOR_EXIT_{launch['exit_code']}_WITHOUT_TERMINAL")
        if launch["interrupted"]:
            return {**after, "status": "INTERRUPTED_AFTER_FENCED_CHILD_TERMINAL",
                    "last_node_status": after["status"], "nodes_launched_here": launched,
                    "code_acceptance": "PASSED", "code_acceptance_result": accepted,
                    "child_pid": launch["child_pid"],
                    "child_start_ticks": launch["child_start_ticks"]}
        if after.get("node") == node:
            if after.get("status") == "READY":
                raise ColdStartInputError(f"{node}_UNSEALED_AFTER_EXECUTOR")
            return {**after, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted, "nodes_launched_here": launched}
    return {**plan_one(source, expected_input_sha256), "code_acceptance": "PASSED",
            "code_acceptance_result": accepted, "nodes_launched_here": launched}


__all__ = ["plan_one", "submit_one", "accepted_code_for_execution"]
