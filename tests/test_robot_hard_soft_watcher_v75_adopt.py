from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chaoyang.ops.run_robot_hard_soft_audit_watcher_v71 import adopted_source_batches


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def artifact_ref(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_v75_adopt_rows_resolve_to_source_batch_without_copying_tree(tmp_path: Path) -> None:
    preflight = tmp_path / "v74/batch_001/preflight/RESULT.json"
    write_json(preflight, {"status": "PASS"})
    robot_root = tmp_path / "v75"
    write_json(
        robot_root / "adopted/session_a/RESULT.json",
        {"session": "session_a", "source_artifacts": {"preflight_result": artifact_ref(preflight)}},
    )
    assert adopted_source_batches(robot_root, {"session_a"}) == {
        preflight.parent.parent.resolve(): {"session_a"}
    }


def test_v75_adopt_rows_reject_unselected_session(tmp_path: Path) -> None:
    preflight = tmp_path / "v74/batch_001/preflight/RESULT.json"
    write_json(preflight, {"status": "PASS"})
    robot_root = tmp_path / "v75"
    write_json(
        robot_root / "adopted/session_b/RESULT.json",
        {"session": "session_b", "source_artifacts": {"preflight_result": artifact_ref(preflight)}},
    )
    with pytest.raises(RuntimeError, match="unexpected adopted session"):
        adopted_source_batches(robot_root, {"session_a"})


def test_v75_runtime_failure_receipt_is_adjacent_to_run_root(tmp_path: Path) -> None:
    robot_root = tmp_path / "attempt_0001" / "run"
    robot_root.mkdir(parents=True)
    failure = robot_root.parent / "FAILED_RUNTIME_FINAL.json"
    write_json(failure, {"terminal_status": "FAILED_RUNTIME_FINAL"})
    receipts = sorted(robot_root.parent.glob("FAILED_RUNTIME_FINAL*.json"))
    assert receipts == [failure]
