"""Fenced, finite Cards004 CPU→fresh HaWoR→Robot diagnostic candidate.

No task is registered by the driver itself. A separate, same-session v6 code
acceptance receipt with a fixed hard cutoff is mandatory before any child is
started. Each child is an existing governed Task Packet writer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hmac
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any

from chaoyang.cli_v5 import exact78_four_session_manifest_candidate_v1 as v5
from chaoyang.cli_v3 import exact78_raw_to_robot_candidate_v2 as v3
from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as prior
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, load_json, process_identity,
    validate_artifact_ref,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


SESSION = Path(f"{v5.VST_ROOT}/004")
SOURCE_SHA = v5.EXPECTED["play_cards_0915_004"][2]
ACCEPTANCE_TASK_ID = "exact78_newraw_cli_v6_cards004_code_acceptance_v1"
ACCEPTANCE_PHASE = "EXACT78_NEWRAW_CLI_V6_CARDS004_CODE_ACCEPTANCE"
ACCEPTANCE_PACKET = REPO_ROOT / "tasks/current" / ACCEPTANCE_TASK_ID / "TASK_PACKET.json"
ACCEPTANCE_RESULT = (REPO_ROOT / "_run/current" / ACCEPTANCE_TASK_ID /
                     "attempts/attempt_0001/RESULT.json")
ACCEPTANCE_RECEIPT = (REPO_ROOT / "tasks/receipts" /
                      f"{ACCEPTANCE_TASK_ID.upper()}.json")
CODE_FILES = tuple(f"src/chaoyang/cli_v6/{name}" for name in (
    "__init__.py", "__main__.py", "exact78_cards004_submit_candidate_v1.py",
    "exact78_cards004_cpu_child_v1.py"))
TEST_FILE = "tests/unit/test_exact78_cli_v6_cards004_candidate_v1.py"
INHERITED_FILES = tuple(f"src/chaoyang/cli_v5/{name}" for name in (
    "__init__.py", "__main__.py", "exact78_four_session_manifest_candidate_v1.py",
    "exact78_newraw_cpu_submit_manifest_v1.py"))
INHERITED_CODE_GROUPS = {
    "inherited_v5_code": INHERITED_FILES,
    "inherited_v4_code": v5.v4.CLI_CODE,
    "inherited_v3_code": v3.CLI_CODE,
    "inherited_v2_code": prior.CLI_CODE,
    "cpu_producer_code": (
        "src/chaoyang/governance/execute_exact78_first_submit_cpu_candidate_v1.py",
        "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
        "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
    ),
}
CLEARANCE = timedelta(seconds=120)
MAX_WINDOW = timedelta(hours=6, minutes=10)


def _ref(path: Path, value: Any, label: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get("path") != str(path)
            or validate_artifact_ref(value)):
        raise ColdStartInputError(label)
    return value


def accepted_code_for_execution(source: Path, expected_sha: str) -> tuple[dict[str, Any], datetime]:
    """Require an independent v6 acceptance terminal, never a v4/v5 receipt."""
    if source != SESSION or not hmac.compare_digest(expected_sha, SOURCE_SHA):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_CARDS004_ONLY")
    if not all(path.is_file() for path in (ACCEPTANCE_PACKET, ACCEPTANCE_RESULT, ACCEPTANCE_RECEIPT)):
        raise ColdStartInputError("CLI_V6_CODE_ACCEPTANCE_NOT_SEALED")
    state, index = load_json(TASK_STATE_PATH), load_json(prior.INDEX)
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == ACCEPTANCE_TASK_ID]
    entries = [row for row in index.get("task_packets", []) if row.get("task_id") == ACCEPTANCE_TASK_ID]
    if (len(rows) != 1 or len(entries) != 1
            or rows[0].get("phase") != ACCEPTANCE_PHASE
            or rows[0].get("session") != "play_cards_0915_004"
            or rows[0].get("status") != "PASSED"
            or entries[0].get("execution_allowed") is not False
            or entries[0].get("packet_path") != str(ACCEPTANCE_PACKET.relative_to(REPO_ROOT))
            or entries[0].get("packet_sha256") != rows[0].get("task_packet", {}).get("sha256")):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_LEDGER_DRIFT")
    row = rows[0]
    refs = {"task_packet": _ref(ACCEPTANCE_PACKET, row.get("task_packet"), "CLI_V6_PACKET_SHA_DRIFT"),
            "result": _ref(ACCEPTANCE_RESULT, row.get("result"), "CLI_V6_RESULT_SHA_DRIFT"),
            "receipt": _ref(ACCEPTANCE_RECEIPT, row.get("terminal_receipt"), "CLI_V6_RECEIPT_SHA_DRIFT")}
    packet, result, receipt = (load_json(ACCEPTANCE_PACKET), load_json(ACCEPTANCE_RESULT),
                               load_json(ACCEPTANCE_RECEIPT))
    entry, manifest_ref = v5._entry(source, expected_sha)
    inputs = packet.get("inputs") or {}
    signature = row.get("run_signature")
    if (packet.get("task_id") != ACCEPTANCE_TASK_ID
            or packet.get("phase") != ACCEPTANCE_PHASE
            or packet.get("execution_mode") != "EXACT78_GOVERNANCE_ONLY"
            or packet.get("run_signature") != signature
            or inputs.get("source_session") != entry["session_id"]
            or inputs.get("source_root") != str(source)
            or inputs.get("submission_input_sha256") != expected_sha
            or inputs.get("source_files") != entry["source_files"]
            or inputs.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or inputs.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or inputs.get("manifest") != manifest_ref
            or result.get("schema_version") != "exact78-newraw-cli-v6-cards004-acceptance-result-v1"
            or result.get("task_id") != ACCEPTANCE_TASK_ID
            or result.get("status") != "PASSED"
            or result.get("run_signature") != signature
            or result.get("task_packet") != refs["task_packet"]
            or result.get("manifest") != manifest_ref
            or result.get("source_session") != entry["session_id"]
            or result.get("source_files") != entry["source_files"]
            or result.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or result.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
            or result.get("visual_qualified") is not False
            or result.get("training_eligible") is not False
            or receipt.get("schema_version") != "exact78-newraw-cli-v6-cards004-acceptance-receipt-v1"
            or receipt.get("task_id") != ACCEPTANCE_TASK_ID
            or receipt.get("status") != "PASSED"
            or receipt.get("task_packet") != refs["task_packet"]
            or receipt.get("result") != refs["result"]):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_IDENTITY_DRIFT")
    codes = result.get("code")
    if (not isinstance(codes, dict) or set(codes) != set(CODE_FILES)
            or inputs.get("code") != codes):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_CODE_SET_DRIFT")
    for name in CODE_FILES:
        _ref(REPO_ROOT / name, codes[name], "CLI_V6_ACCEPTANCE_CODE_SHA_DRIFT")
    test = _ref(REPO_ROOT / TEST_FILE, result.get("test_code"), "CLI_V6_ACCEPTANCE_TEST_SHA_DRIFT")
    if inputs.get("test_code") != test:
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_TEST_INPUT_DRIFT")
    for group, names in INHERITED_CODE_GROUPS.items():
        inherited = result.get(group)
        if (not isinstance(inherited, dict) or set(inherited) != set(names)
                or inputs.get(group) != inherited):
            raise ColdStartInputError("CLI_V6_INHERITED_CODE_SET_DRIFT:" + group)
        for name in names:
            _ref(REPO_ROOT / name, inherited[name], "CLI_V6_INHERITED_CODE_SHA_DRIFT:" + group)
    published = result.get("published_operations")
    if (not isinstance(published, dict)
            or set(published) != set(prior.ACCEPTANCE_OPERATIONS)
            or inputs.get("published_operations") != published):
        raise ColdStartInputError("CLI_V6_OPERATION_SET_DRIFT")
    contracts = load_json(prior.ALGORITHM_CONTRACT).get("special_status_contracts") or {}
    for key in prior.ACCEPTANCE_OPERATIONS:
        contract, sealed = contracts.get(key) or {}, published[key]
        operation = contract.get("candidate_operation")
        if (not isinstance(sealed, dict) or not isinstance(operation, str)
                or not re.fullmatch(r"run_exact78_newraw_[a-z0-9_]+", operation)
                or sealed.get("candidate_operation") != operation
                or contract.get("task_packet_required") is not True
                or contract.get("current_runnable_operation") is not None
                or (contract.get("code_closure") or []).count(sealed.get("operation_ref")) != 1):
            raise ColdStartInputError("CLI_V6_OPERATION_ROUTE_DRIFT:" + key)
        _ref(REPO_ROOT / "src/chaoyang/ops" / f"{operation}.py",
             sealed.get("operation_ref"), "CLI_V6_OPERATION_SHA_DRIFT:" + key)
    check, tests = result.get("cli_check_only"), result.get("negative_tests")
    if (not isinstance(check, dict) or inputs.get("cli_check_only") != check
            or check.get("source_session") != "play_cards_0915_004"
            or check.get("status") != "READY"
            or check.get("node") != "CPU_ADMISSION"
            or check.get("registration_started") is not False
            or check.get("model_calls") != 0
            or not isinstance(tests, dict) or inputs.get("negative_tests") != tests
            or tests.get("status") != "PASSED" or tests.get("test_count", 0) < 9):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_PREFLIGHT_OR_TEST_DRIFT")
    # The new receipt pins a finite absolute cutoff; a future revision needs
    # its own Task Packet rather than silently extending this submission.
    raw_cutoff = result.get("hard_cutoff_at")
    try:
        cutoff = datetime.fromisoformat(raw_cutoff)
    except (TypeError, ValueError) as exc:
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_CUTOFF_INVALID") from exc
    now = datetime.now(timezone.utc)
    if (cutoff.tzinfo is None or inputs.get("hard_cutoff_at") != raw_cutoff
            or cutoff <= now or cutoff - now > MAX_WINDOW):
        raise ColdStartInputError("CLI_V6_ACCEPTANCE_CUTOFF_INVALID")
    return refs["result"], cutoff


def plan_one(session: Path, expected_sha: str) -> dict[str, Any]:
    source = prior.legacy_submit._raw_session_path(session)
    row, manifest_ref = v5._entry(source, expected_sha)
    if row["session_id"] != "play_cards_0915_004":
        raise ColdStartInputError("CLI_V6_CARDS004_ONLY")
    cpu_decision = v5.plan_one(source, expected_sha)
    base = {"schema_version": "exact78-newraw-cards004-submit-dag-candidate-v6",
            "source_session": row["session_id"], "submission_input_sha256": expected_sha,
            "manifest": manifest_ref, "time_mode": "TIMEBASE_UNVERIFIED",
            "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
            "physical_eye": "left_sourceIndex_1", "head_pose_source": "ABSENT_UNTIL_FIELD_PROJECTION_AND_CLOCK_PROOF",
            "control_ground_truth": False, "visual_qualified": False,
            "product_qualified": False, "training_eligible": False,
            "scene_clean": "NOT_WIRED", "general_operation": None,
            "code_acceptance": "NOT_CHECKED_CHECK_ONLY"}
    if cpu_decision["status"] != "ATTACHED_TERMINAL":
        return {**base, "node": "CPU_ADMISSION", **cpu_decision,
                "candidate_cli": "chaoyang.cli_v6",
                "code_acceptance": "NOT_CHECKED_CHECK_ONLY"}
    cpu_result = Path(cpu_decision["result"]["path"])
    hawor_packet = prior.hawor.packet_value(cpu_result)
    prior._sealed_cpu(hawor_packet, cpu_decision)
    hawor_decision = prior._node(hawor_packet, prior.PHASES[1], prior.SCHEMAS[1])
    if hawor_decision["status"] != "ATTACHED_TERMINAL":
        return {**base, "node": "HAWOR_V4", **hawor_decision}
    hawor_result = Path(hawor_decision["result"]["path"])
    robot_packet = prior.robot.packet_value(hawor_result)
    robot_decision = prior._node(robot_packet, prior.PHASES[2], prior.SCHEMAS[2])
    if robot_decision["status"] != "ATTACHED_TERMINAL":
        return {**base, "node": "ROBOT_V1", **robot_decision}
    frozen_packet = load_json(REPO_ROOT / "tasks/current" / robot_decision["task_id"] /
                              "TASK_PACKET.json")
    result = load_json(Path(robot_decision["result"]["path"]))
    if result.get("stage") != prior.robot._verify_stage(frozen_packet):
        raise ColdStartInputError("CLI_V6_ROBOT_SEALED_STAGE_DRIFT")
    return {**base, "status": "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
            "node": "ROBOT_V1", "task_id": robot_decision["task_id"],
            "result": robot_decision["result"], "receipt": robot_decision["receipt"]}


def _budget(node: str, cutoff: datetime) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    if now + timedelta(seconds=prior.STAGE_WALL_SECONDS[node]) + CLEARANCE > cutoff:
        return {"status": "NOT_EVALUATED_BUDGET", "node": node,
                "reason": "CLI_V6_INSUFFICIENT_PINNED_WINDOW",
                "hard_cutoff_at": cutoff.isoformat()}
    return None


def _launch_cpu(source: Path, expected_sha: str, task_id: str) -> dict[str, Any]:
    """Wait on the fenced CPU child; no replacement writer on interruption."""
    if not re.fullmatch(r"[a-z0-9_]+", task_id):
        raise ColdStartInputError("CLI_V6_TASK_ID_UNSAFE")
    command = ["/usr/local/bin/python", "-B", "-m",
               "chaoyang.cli_v6.exact78_cards004_cpu_child_v1",
               "--session", str(source), "--expected-input-sha256", expected_sha,
               "--register-and-run"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1")
    previous = {kind: signal.getsignal(kind) for kind in (signal.SIGINT, signal.SIGTERM)}
    interrupted: list[int] = []
    def defer(signum: int, _frame: Any) -> None:
        interrupted.append(signum)
    for kind in previous:
        signal.signal(kind, defer)
    child: subprocess.Popen[Any] | None = None
    try:
        if interrupted:
            raise ColdStartInputError("CLI_V6_INTERRUPTED_BEFORE_CHILD")
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        identity = process_identity(child.pid)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        return {"exit_code": code, "interrupted": bool(interrupted),
                "child_pid": child.pid, "child_start_ticks": identity.get("start_ticks")}
    finally:
        if child is not None and child.poll() is None:
            child.wait()
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def submit_one(session: Path, expected_sha: str) -> dict[str, Any]:
    source = prior.legacy_submit._raw_session_path(session)
    # The receipt gate runs before planning or launching. No v4/v5 acceptance
    # can satisfy this exact Cards004 task identity and schema.
    accepted_ref, cutoff = accepted_code_for_execution(source, expected_sha)
    launched: list[str] = []
    for _ in prior.PHASES:
        decision = plan_one(source, expected_sha)
        if decision["status"] != "READY":
            return {**decision, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref, "nodes_launched_here": launched}
        node = decision["node"]
        if budget := _budget(node, cutoff):
            return {**decision, **budget, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref, "nodes_launched_here": launched}
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
        launch = (_launch_cpu(source, expected_sha, decision["task_id"])
                  if node == "CPU_ADMISSION" else
                  prior._launch(node, source, upstream, decision["task_id"]))
        launched.append(node)
        after = plan_one(source, expected_sha)
        if (launch["exit_code"] != 0
                and after.get("status") not in {"STOP_TERMINAL", "ATTACHED_ACTIVE"}):
            raise ColdStartInputError(f"CLI_V6_{node}_EXIT_WITHOUT_TERMINAL")
        if launch["interrupted"]:
            return {**after, "status": "INTERRUPTED_AFTER_FENCED_CHILD_TERMINAL",
                    "nodes_launched_here": launched, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref,
                    "child_pid": launch["child_pid"],
                    "child_start_ticks": launch["child_start_ticks"]}
        if after.get("node") == node:
            if after.get("status") == "READY":
                raise ColdStartInputError("CLI_V6_NODE_UNSEALED_AFTER_EXECUTOR")
            return {**after, "code_acceptance": "PASSED",
                    "code_acceptance_result": accepted_ref,
                    "nodes_launched_here": launched}
    return {**plan_one(source, expected_sha), "code_acceptance": "PASSED",
            "code_acceptance_result": accepted_ref, "nodes_launched_here": launched}


__all__ = ["accepted_code_for_execution", "plan_one", "submit_one"]
