"""Fail-closed authority check for 0902 Exact78 CLI operations.

This module deliberately has no imports from ``chaoyang.ops``.  A candidate
packet authorizes one exact command, not a production baseline promotion.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

from chaoyang.paths import get_paths


class Exact78GateDenied(SystemExit):
    """The current governance state does not authorize this operation."""


def _deny(reason: str) -> None:
    raise Exact78GateDenied(f"EXACT78_GATE_DENIED: {reason}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _deny(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _deny(f"not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _deny(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _inside_project(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _deny("missing project-local path")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        _deny(f"path escapes project: {raw}")
    return path


def _command_tail(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        _deny(f"invalid packet command: {exc}")
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens.pop(0)  # e.g. PYTHONPATH=src
    if len(tokens) >= 3 and tokens[:2] == ["chaoyang", "run"]:
        return tokens[2:]
    if tokens and Path(tokens[0]).name.startswith("python"):
        try:
            module_flag = tokens.index("-m")
        except ValueError:
            _deny("packet command is not a Chaoyang CLI invocation")
        if tokens[module_flag : module_flag + 3] == ["-m", "chaoyang.cli", "run"]:
            return tokens[module_flag + 3 :]
    _deny("packet command is not a Chaoyang CLI invocation")


def _proc_identity(pid: int) -> tuple[int, int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        return int(fields[1]), int(fields[19])  # PPID, starttime (field 22)
    except (OSError, ValueError, IndexError) as exc:
        _deny(f"writer process identity unavailable: {exc}")


def _caller_is_writer_descendant(writer_pid: int, writer_ticks: int, caller_pid: int) -> bool:
    _, actual_ticks = _proc_identity(writer_pid)
    if actual_ticks != writer_ticks:
        return False
    pid = caller_pid
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == writer_pid:
            return True
        seen.add(pid)
        pid, _ = _proc_identity(pid)
    return False


def _writer_file(root: Path, packet: dict[str, Any]) -> Path:
    task_id = packet.get("task_id")
    prefix = root / "_run" / "current" / str(task_id) / "attempts"
    matches: list[Path] = []
    for raw in packet.get("write_set", []):
        path = _inside_project(root, raw)
        if path.parent == prefix and path.name.startswith("attempt_"):
            matches.append(path / "WRITER.json")
    if len(matches) != 1:
        _deny("packet must designate exactly one Exact78 attempt writer")
    return matches[0]


def _same_writer(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    keys = ("pid", "proc_start_ticks", "executor_epoch", "fencing_token", "scope")
    return all(a.get(key) == b.get(key) for key in keys)


def authorize_exact78_operation(
    operation: str,
    args: list[str],
    *,
    repo_root: Path | None = None,
    caller_pid: int | None = None,
) -> None:
    """Allow only the exact active command of a fenced Exact78 task packet.

    ``repo_root`` and ``caller_pid`` are injectable for local unit tests; the
    public CLI always uses the actual project root and process ID.
    """

    root = (repo_root or get_paths().repo_root).resolve()
    index = _load_json(root / "tasks/current/INDEX.json")
    allowed = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if len(allowed) != 1:
        _deny("expected exactly one execution_allowed task packet")
    row = allowed[0]
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id.startswith("exact78_"):
        _deny("active task is not an Exact78 task")
    packet_path = _inside_project(root, row.get("packet_path"))
    if packet_path != root / "tasks" / "current" / task_id / "TASK_PACKET.json":
        _deny("task packet path does not match active task")
    if _sha256(packet_path) != row.get("packet_sha256"):
        _deny("task packet SHA drift")
    packet = _load_json(packet_path)
    if packet.get("task_id") != task_id:
        _deny("task packet identity mismatch")

    mode = packet.get("execution_mode")
    if mode not in ("EXACT78_CANDIDATE", "EXACT78_PRODUCTION"):
        _deny("packet lacks explicit Exact78 execution mode")
    commands = packet.get("commands")
    if not isinstance(commands, list) or len(commands) != 1 or not isinstance(commands[0], str):
        _deny("packet must designate exactly one command")
    if _command_tail(commands[0]) != [operation, *args]:
        _deny("CLI command differs from frozen task packet")

    state = _load_json(root / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if (state.get("next_task") or {}).get("task_id") != task_id:
        _deny("task is not the current next_task")
    matches = [task for task in state.get("tasks", []) if task.get("task_id") == task_id]
    if len(matches) != 1 or matches[0].get("status") != "RUNNING":
        _deny("task is not uniquely RUNNING")
    task = matches[0]
    if task.get("run_signature") != packet.get("run_signature"):
        _deny("run signature differs from current task")
    if (task.get("task_packet") or {}).get("sha256") != row.get("packet_sha256"):
        _deny("current task points to a different packet SHA")

    writer = packet.get("writer")
    if not _same_writer(writer, _load_json(_writer_file(root, packet))):
        _deny("WRITER.json differs from packet fencing identity")
    if task.get("pid") != writer.get("pid") or task.get("proc_start_ticks") != writer.get("proc_start_ticks"):
        _deny("current task writer identity mismatch")
    pid, ticks = writer.get("pid"), writer.get("proc_start_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, int):
        _deny("invalid writer PID/startticks")
    if not _caller_is_writer_descendant(pid, ticks, caller_pid or os.getpid()):
        _deny("caller is not a descendant of the live fenced writer")

    code_refs = packet.get("runtime_code")
    if not isinstance(code_refs, list) or not code_refs:
        _deny("runtime code SHA closure is absent")
    operation_path = root / "src" / "chaoyang" / "ops" / f"{operation}.py"
    pinned_operation = False
    for ref in code_refs:
        if not isinstance(ref, dict):
            _deny("invalid runtime code reference")
        path = _inside_project(root, ref.get("path"))
        if _sha256(path) != ref.get("sha256"):
            _deny(f"runtime code SHA drift: {path}")
        pinned_operation |= path == operation_path
    if not pinned_operation:
        _deny("operation module is absent from runtime code closure")

    if mode == "EXACT78_PRODUCTION":
        from chaoyang.governance.exact78_visual_selection_v1 import _policy
        contract_path = root / "docs/governance/ALGORITHM_CONTRACT.json"
        if contract_path.is_file():
            selection = _policy(_load_json(contract_path))
            if isinstance(selection, dict) and selection.get("enabled") is not False:
                _deny("selected V1 authorizes fenced batch visual tasks, not formal production; old V2/null routes cannot select it")
        route = _load_json(root / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json")
        if (route.get("general_raw_to_robot") or {}).get("current_runnable_operation") != operation:
            _deny("production operation is not selected by current route")


def authorize_operation_entry(operation: str, argv: list[str] | None = None) -> None:
    """Apply the same fence to direct module/function entry before any writes."""
    import sys
    from chaoyang.governance.operation_retirement import require_not_retired
    root = get_paths().repo_root
    require_not_retired(root, operation)
    authorize_exact78_operation(operation, list(sys.argv[1:] if argv is None else argv), repo_root=root)


__all__ = ["Exact78GateDenied", "authorize_exact78_operation", "authorize_operation_entry"]
