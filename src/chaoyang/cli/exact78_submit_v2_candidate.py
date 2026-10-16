"""Read-only candidate-first-submit planner for the six T0 Exact78 captures.

This is intentionally not wired to ``chaoyang submit`` and has no dispatch
function. The existing per-session executor owns registration, writer, GPU
lease and terminal publication. A future successor must pin this module in
its Task Packet before enabling the first-run path.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.common import REPO_ROOT, artifact_ref, validate_artifact_ref
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError
from chaoyang.governance.exact78_raw_two_source_executor_v1 import (
    T0, Target, _candidate_route_readiness, checked_inputs, packet_value,
)
from chaoyang.ops.run_exact78_raw_to_robot_v3 import _validate_t0_target


WRAPPERS = {
    "get_potato_chips_0901_013": "execute_exact78_raw_t2_0901_chips013_v1",
    "get_potato_chips_0902_103": "execute_exact78_raw_t2_0902_chips103_v1",
    "play_cards_0901_001": "execute_exact78_raw_t2_0901_cards001_holdout_v1",
    "play_cards_0902_042": "execute_exact78_raw_t2_0902_cards042_v1",
    "play_cards_0915_001": "execute_exact78_raw_t2_0915_cards001_v1",
    "play_cards_0915_002": "execute_exact78_raw_t2_0915_cards002_holdout_v1",
}
ACTIVE = frozenset({"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"})
TERMINAL = frozenset({"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE", "FAILED_QUALITY_FINAL"})


def _wrapper_name(session_id: str) -> str:
    name = WRAPPERS.get(session_id)
    if name is None:
        raise ColdStartInputError("T0_FROZEN_TARGET_NOT_AUTHORIZED")
    return name


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ColdStartInputError("JSON_OBJECT_REQUIRED")
    return value


def _verified_ref(ref: Any, *, expected: Path | None = None) -> dict[str, Any]:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ColdStartInputError("ARTIFACT_REF_REQUIRED")
    path = Path(ref["path"]).resolve(strict=True)
    if REPO_ROOT.resolve(strict=True) not in path.parents:
        raise ColdStartInputError("ARTIFACT_OUTSIDE_PROJECT")
    if expected is not None and path != expected.resolve(strict=True):
        raise ColdStartInputError("ARTIFACT_PATH_DRIFT")
    if errors := validate_artifact_ref(ref):
        raise ColdStartInputError("ARTIFACT_SHA_DRIFT:" + ";".join(errors))
    return _load(path)


def _live_writer(writer: Any, attempt: Path) -> None:
    if not isinstance(writer, dict):
        raise ColdStartInputError("WRITER_FENCE_ABSENT")
    frozen = _load(attempt / "WRITER.json")
    keys = ("pid", "proc_start_ticks", "executor_epoch", "fencing_token", "scope")
    if any(writer.get(key) != frozen.get(key) for key in keys):
        raise ColdStartInputError("WRITER_FENCE_DRIFT")
    pid, ticks = writer.get("pid"), writer.get("proc_start_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, int) or pid < 2:
        raise ColdStartInputError("WRITER_IDENTITY_INVALID")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        live_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise ColdStartInputError("WRITER_NOT_LIVE") from exc
    if live_ticks != ticks:
        raise ColdStartInputError("WRITER_NOT_LIVE")


def _expected_packet(session: Path) -> tuple[Target, dict[str, Any], dict[str, Any], Path]:
    request = build_request(session=session)
    if len(request["sources"]) != 1:
        raise ColdStartInputError("ONE_SESSION_PACKET_REQUIRED")
    source = request["sources"][0]
    session_id = source["session_id"]
    module_name = _wrapper_name(session_id)
    audit = _validate_t0_target(source, freeze=T0 / "INPUT_FREEZE.json",
                                frame_map=T0 / "FRAME_TIME_MAP.json")
    wrapper = importlib.import_module("chaoyang.governance." + module_name)
    target: Target = wrapper.TARGET
    script = Path(wrapper.__file__).resolve(strict=True)
    if (target.session_id != session_id or target.session.resolve(strict=True) != session.resolve(strict=True)
            or source["session_root"] != str(target.session)
            or target.frame_count != audit["frame_count"]
            or target.time_mode != audit["time_mode"]
            or target.physical_eye != audit["physical_eye"]):
        raise ColdStartInputError("T0_WRAPPER_IDENTITY_DRIFT")
    if wrapper.ROLE != audit["source_role"]:
        raise ColdStartInputError("T0_WRAPPER_ROLE_DRIFT")
    for relative, sha in wrapper.RECIPE_SHA.items():
        if artifact_ref(REPO_ROOT / relative)["sha256"] != sha:
            raise ColdStartInputError("FROZEN_RECIPE_SHA_DRIFT:" + relative)
    inputs, bound_request = checked_inputs(target)
    if bound_request["submission_input_sha256"] != request["submission_input_sha256"]:
        raise ColdStartInputError("SUBMISSION_INPUT_SHA_DRIFT")
    packet = packet_value(target, script, inputs)  # dry-run only; no registration or worker
    if (packet["inputs"]["submission_input_sha256"] != request["submission_input_sha256"]
            or packet["commands"] != [target.command]):
        raise ColdStartInputError("DRY_RUN_PACKET_BINDING_DRIFT")
    return target, request, packet, script


def _same_active(target: Target, request: dict[str, Any], expected: dict[str, Any],
                 allowed: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if allowed.get("task_id") != target.task_id:
        raise ColdStartInputError("ACTIVE_DIFFERENT_TASK_CONFLICT")
    packet_path = target.packet
    packet = _verified_ref({"path": str(packet_path),
                            "bytes": packet_path.stat().st_size,
                            "sha256": allowed.get("packet_sha256")}, expected=packet_path)
    if (packet.get("run_signature") != expected["run_signature"]
            or packet.get("task_id") != target.task_id
            or packet.get("inputs", {}).get("submission_input_sha256")
               != request["submission_input_sha256"]
            or packet.get("commands") != [target.command]
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or str(target.output.relative_to(REPO_ROOT)) not in packet.get("write_set", [])):
        raise ColdStartInputError("ACTIVE_PACKET_SIGNATURE_CONFLICT")
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == target.task_id]
    if (len(matches) != 1 or matches[0].get("status") not in ACTIVE
            or matches[0].get("session") != target.session_id
            or (state.get("next_task") or {}).get("task_id") != target.task_id
            or matches[0].get("run_signature") != packet["run_signature"]
            or matches[0].get("task_packet", {}).get("sha256") != allowed.get("packet_sha256")):
        raise ColdStartInputError("ACTIVE_TASK_STATE_DRIFT")
    task = matches[0]
    writer = packet.get("writer")
    if not isinstance(writer, dict) or writer.get("scope") != target.task_id:
        raise ColdStartInputError("WRITER_FENCE_ABSENT")
    if task.get("pid") != writer.get("pid") or task.get("proc_start_ticks") != writer.get("proc_start_ticks"):
        raise ColdStartInputError("ACTIVE_WRITER_STATE_DRIFT")
    _live_writer(writer, target.output)
    for ref in packet.get("runtime_code", []):
        if errors := validate_artifact_ref(ref):
            raise ColdStartInputError("ACTIVE_CODE_SHA_DRIFT:" + ";".join(errors))
    return {"status": "ATTACHED_TO_ACTIVE_CANDIDATE", "task_id": target.task_id,
            "task_state": task["status"], "run_signature": packet["run_signature"],
            "task_packet": artifact_ref(packet_path), "execution_started_here": False,
            "visual_qualified": False, "product_qualified": False, "training_eligible": False}


def _same_terminal(target: Target, request: dict[str, Any], expected: dict[str, Any],
                   state: dict[str, Any]) -> dict[str, Any] | None:
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == target.task_id]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].get("status") not in TERMINAL:
        raise ColdStartInputError("NONTERMINAL_TASK_WITHOUT_ACTIVE_PACKET")
    task = matches[0]
    packet = _verified_ref(task.get("task_packet"), expected=target.packet)
    if (packet.get("run_signature") != expected["run_signature"]
            or packet.get("task_id") != target.task_id
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("commands") != [target.command]
            or task.get("session") != target.session_id
            or packet.get("inputs", {}).get("submission_input_sha256")
               != request["submission_input_sha256"]):
        raise ColdStartInputError("TERMINAL_PACKET_SIGNATURE_CONFLICT")
    result = _verified_ref(task.get("result"), expected=target.output / "RESULT.json")
    receipt = _verified_ref(task.get("terminal_receipt"), expected=target.receipt)
    if (result.get("task_id") != target.task_id or result.get("status") != task["status"]
            or result.get("source_session") != target.session_id
            or receipt.get("task_id") != target.task_id or receipt.get("status") != task["status"]
            or receipt.get("result") != task["result"]):
        raise ColdStartInputError("TERMINAL_RESULT_RECEIPT_DRIFT")
    return {"status": "ATTACHED_TO_TERMINAL_CANDIDATE", "task_id": target.task_id,
            "terminal_status": task["status"], "run_signature": packet["run_signature"],
            "task_packet": task["task_packet"], "result": task["result"],
            "receipt": task["terminal_receipt"], "execution_started_here": False,
            "visual_qualified": result.get("visual_qualified"),
            "product_qualified": result.get("product_qualified"),
            "training_eligible": result.get("training_eligible")}


def plan_single(session: Path) -> dict[str, Any]:
    """Read-only decision. FIRST_RUN_READY is *not* a task registration."""
    target, request, expected, script = _expected_packet(session)
    routable, reason = _candidate_route_readiness()
    if not routable:
        raise ColdStartInputError("CANDIDATE_OPERATION_NOT_ROUTABLE:" + reason)
    index = _load(REPO_ROOT / "tasks/current/INDEX.json")
    state = _load(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    allowed = [item for item in index.get("task_packets", []) if item.get("execution_allowed") is True]
    if len(allowed) > 1:
        raise ColdStartInputError("MULTIPLE_ACTIVE_PACKETS")
    if allowed:
        if allowed[0].get("task_id") != target.task_id:
            # An unrelated live writer forbids a new launch, but it need not
            # prevent a read-only lookup of our already sealed terminal run.
            terminal = _same_terminal(target, request, expected, state)
            if terminal is not None:
                return terminal
            raise ColdStartInputError("ACTIVE_DIFFERENT_TASK_CONFLICT")
        return _same_active(target, request, expected, allowed[0], state)
    if state.get("next_task") or any(item.get("status") in ACTIVE for item in state.get("tasks", [])):
        raise ColdStartInputError("ACTIVE_TASK_WITHOUT_PACKET_CONFLICT")
    terminal = _same_terminal(target, request, expected, state)
    if terminal is not None:
        return terminal
    if any(path.exists() for path in (target.output, target.packet, target.receipt)):
        raise ColdStartInputError("UNSEALED_ATTEMPT_CONFLICT")
    return {"status": "FIRST_RUN_READY_NOT_REGISTERED", "task_id": target.task_id,
            "run_signature": expected["run_signature"],
            "submission_input_sha256": request["submission_input_sha256"],
            "fenced_executor": artifact_ref(script),
            "executor_command": ["/usr/local/bin/python", "-B", str(script)],
            "execution_started_here": False, "requires_fenced_successor_packet": True,
            "visual_qualified": False, "product_qualified": False, "training_eligible": False}


def plan_manifest(manifest: Path) -> dict[str, Any]:
    """Validate a frozen manifest, but never misroute it to one v3 packet."""
    request = build_request(manifest=manifest)
    for source in request["sources"]:
        _wrapper_name(source["session_id"])
        _validate_t0_target(source, freeze=T0 / "INPUT_FREEZE.json",
                            frame_map=T0 / "FRAME_TIME_MAP.json")
    return {"status": "BATCH_REQUIRES_PER_SESSION_TASK_PACKETS",
            "manifest": request["manifest"],
            "session_ids": [source["session_id"] for source in request["sources"]],
            "submission_input_sha256": request["submission_input_sha256"],
            "execution_started_here": False,
            "reason": "Existing fenced executor binds one --session input signature; aggregate manifest is not a packet"}


__all__ = ["plan_single", "plan_manifest"]
