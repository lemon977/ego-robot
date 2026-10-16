"""The 0902 CLI must reject unrouted Exact78 code before algorithm import."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from chaoyang.cli import _is_exact78_invocation, _run_module
from chaoyang.governance.exact78_operation_gate import (
    Exact78GateDenied,
    _proc_identity,
    authorize_exact78_operation,
)


OPERATION = "run_exact78_unit_candidate"
TASK_ID = "exact78_unit_candidate_v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, *, mode: str = "EXACT78_CANDIDATE") -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    operation_path = root / "src/chaoyang/ops" / f"{OPERATION}.py"
    operation_path.parent.mkdir(parents=True, exist_ok=True)
    operation_path.write_text("def main(): return 0\n", encoding="utf-8")
    writer = {
        "pid": os.getpid(),
        "proc_start_ticks": _proc_identity(os.getpid())[1],
        "executor_epoch": 1,
        "fencing_token": "unit-fence",
        "scope": TASK_ID,
    }
    attempt_dir = root / "_run/current" / TASK_ID / "attempts/attempt_0001"
    _write_json(attempt_dir / "WRITER.json", writer)
    packet_path = root / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    packet = {
        "task_id": TASK_ID,
        "execution_mode": mode,
        "commands": [f"chaoyang run {OPERATION} --manifest 'input with spaces.json'"],
        "run_signature": "unit-signature",
        "runtime_code": [{"path": str(operation_path), "sha256": _sha(operation_path)}],
        "write_set": [str(attempt_dir)],
        "writer": writer,
    }
    _write_json(packet_path, packet)
    index_path = root / "tasks/current/INDEX.json"
    _write_json(
        index_path,
        {"task_packets": [{
            "task_id": TASK_ID,
            "packet_path": str(packet_path.relative_to(root)),
            "packet_sha256": _sha(packet_path),
            "execution_allowed": True,
        }]},
    )
    state_path = root / "docs/governance/LONG_HORIZON_TASK_STATE.json"
    _write_json(
        state_path,
        {
            "next_task": {"task_id": TASK_ID},
            "tasks": [{
                "task_id": TASK_ID,
                "status": "RUNNING",
                "run_signature": packet["run_signature"],
                "task_packet": {"sha256": _sha(packet_path)},
                "pid": writer["pid"],
                "proc_start_ticks": writer["proc_start_ticks"],
            }],
        },
    )
    route_path = root / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
    _write_json(route_path, {"general_raw_to_robot": {"current_runnable_operation": None}})
    return {
        "packet": packet_path,
        "index": index_path,
        "state": state_path,
        "writer": attempt_dir / "WRITER.json",
        "operation": operation_path,
        "route": route_path,
    }


def _authorize(root: Path) -> None:
    authorize_exact78_operation(
        OPERATION, ["--manifest", "input with spaces.json"], repo_root=root, caller_pid=os.getpid()
    )


def test_explicit_candidate_packet_allows_exact_command(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _authorize(tmp_path)


def test_missing_or_legacy_packet_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = json.loads(paths["index"].read_text())
    index["task_packets"][0]["execution_allowed"] = False
    _write_json(paths["index"], index)
    with pytest.raises(Exact78GateDenied, match="exactly one"):
        _authorize(tmp_path)

    index["task_packets"][0]["execution_allowed"] = True
    _write_json(paths["index"], index)
    packet = json.loads(paths["packet"].read_text())
    packet.pop("execution_mode")
    _write_json(paths["packet"], packet)
    index["task_packets"][0]["packet_sha256"] = _sha(paths["packet"])
    _write_json(paths["index"], index)
    with pytest.raises(Exact78GateDenied, match="execution mode"):
        _authorize(tmp_path)


@pytest.mark.parametrize("change", ["packet_sha", "command", "code_sha", "writer_ticks", "status"])
def test_identity_or_command_drift_is_rejected(tmp_path: Path, change: str) -> None:
    paths = _fixture(tmp_path)
    if change == "packet_sha":
        packet = json.loads(paths["packet"].read_text())
        packet["claim_limit"] = "changed after registration"
        _write_json(paths["packet"], packet)
    elif change == "command":
        with pytest.raises(Exact78GateDenied, match="CLI command"):
            authorize_exact78_operation(OPERATION, ["--manifest", "wrong.json"], repo_root=tmp_path)
        return
    elif change == "code_sha":
        paths["operation"].write_text("def main(): return 1\n", encoding="utf-8")
    elif change == "writer_ticks":
        writer = json.loads(paths["writer"].read_text())
        writer["proc_start_ticks"] += 1
        _write_json(paths["writer"], writer)
    elif change == "status":
        state = json.loads(paths["state"].read_text())
        state["tasks"][0]["status"] = "PASSED"
        _write_json(paths["state"], state)
    with pytest.raises(Exact78GateDenied):
        _authorize(tmp_path)


def test_production_requires_selected_route(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, mode="EXACT78_PRODUCTION")
    with pytest.raises(Exact78GateDenied, match="not selected"):
        _authorize(tmp_path)
    _write_json(paths["route"], {"general_raw_to_robot": {"current_runnable_operation": OPERATION}})
    _authorize(tmp_path)


def test_cli_denies_before_algorithm_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from chaoyang import cli
    from chaoyang.governance import exact78_operation_gate as gate

    imported = []
    monkeypatch.setattr(cli, "_maintained_operations", lambda: {OPERATION})
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: imported.append(name))
    monkeypatch.setattr(gate, "authorize_exact78_operation", lambda *args: (_ for _ in ()).throw(Exact78GateDenied("denied")))
    with pytest.raises(Exact78GateDenied):
        _run_module(OPERATION, [])
    assert imported == []


def test_historical_cards042_aliases_are_gated_without_gating_0916() -> None:
    assert _is_exact78_invocation("run_cards042_full_clean_robot_v3", [])
    assert _is_exact78_invocation(
        "run_role_mask_runtime_contract_repair_v71",
        ["--output-root", "/project/_run/current/exact78_cards042_role_mask_rebuild_v1"],
    )
    assert not _is_exact78_invocation(
        "run_role_mask_runtime_contract_repair_v71", ["--output-root", "/project/_run/current/robot100_0916"]
    )
