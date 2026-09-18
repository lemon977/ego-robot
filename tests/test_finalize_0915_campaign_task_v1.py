from __future__ import annotations

import json
import hashlib
from pathlib import Path

from chaoyang.governance import finalize_0915_campaign_task_v1 as subject


def test_finalizer_has_explicit_bounded_terminal_sets() -> None:
    assert subject.TERMINAL == {
        "PASSED", "REJECTED_QUALITY", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE",
        "BLOCKED_EXTERNAL", "CANCELLED",
    }
    assert "RUNNING" in subject.LIVE
    assert not (subject.TERMINAL & subject.LIVE)


def test_successful_terminal_requires_exact_write_root_weights_and_outputs(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "task/attempts/attempt_0001"
    attempt.mkdir(parents=True)
    result_path = attempt / "RESULT.json"
    result = {"task_id": "task", "status": "PASSED", "weights": "ABSENT"}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    packet = {
        "task_id": "task",
        "write_set": ["task/attempts/attempt_0001"],
        "weights": "ABSENT",
        "required_outputs": ["RESULT.json", "SUMMARY.json"],
    }
    assert subject.validate_terminal_bundle(
        result_path, result, packet, repo_root=tmp_path,
    ) == ["required output is missing: SUMMARY.json"]
    (attempt / "SUMMARY.json").write_text("{}\n", encoding="utf-8")
    assert subject.validate_terminal_bundle(
        result_path, result, packet, repo_root=tmp_path,
    ) == []


def test_nonpass_terminal_does_not_fabricate_success_outputs(tmp_path: Path) -> None:
    attempt = tmp_path / "task/attempts/attempt_0001"
    attempt.mkdir(parents=True)
    result_path = attempt / "RESULT.json"
    result = {"task_id": "task", "status": "BLOCKED_RESOURCE"}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    packet = {
        "task_id": "task",
        "write_set": ["task/attempts/attempt_0001"],
        "weights": ["model.pt"],
        "required_outputs": ["RESULT.json", "GPU_COMMAND_RECEIPT.json"],
    }
    assert subject.validate_terminal_bundle(
        result_path, result, packet, repo_root=tmp_path,
    ) == []


def test_partial_terminal_index_recovers_only_from_sha_bound_active_predecessor(
    tmp_path: Path,
) -> None:
    predecessor_path = tmp_path / "PREDECESSOR_ACTIVE_INDEX.json"
    predecessor = {
        "task_packets": [
            {"task_id": "task", "execution_allowed": True, "packet_path": "task.json"}
        ]
    }
    payload = json.dumps(predecessor, sort_keys=True).encode()
    predecessor_path.write_bytes(payload)
    current = {
        "status": "PASS_NO_ACTIVE_TASKS",
        "packet_revision": "TASK_TERMINAL",
        "governance_revision": 8,
        "task_packets": [],
        "supersedes_index": {
            "path": str(predecessor_path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    recovered, source = subject.resolve_active_index_for_finalization(
        current,
        "task",
        expected_revision=7,
    )
    assert recovered == predecessor
    assert source == predecessor_path


def test_partial_terminal_index_rejects_unbound_predecessor(tmp_path: Path) -> None:
    predecessor_path = tmp_path / "PREDECESSOR_ACTIVE_INDEX.json"
    predecessor_path.write_text('{"task_packets": []}\n', encoding="utf-8")
    current = {
        "status": "PASS_NO_ACTIVE_TASKS",
        "packet_revision": "TASK_TERMINAL",
        "governance_revision": 8,
        "task_packets": [],
        "supersedes_index": {
            "path": str(predecessor_path),
            "bytes": predecessor_path.stat().st_size,
            "sha256": "0" * 64,
        },
    }
    try:
        subject.resolve_active_index_for_finalization(
            current,
            "task",
            expected_revision=7,
        )
    except RuntimeError as exc:
        assert "predecessor invalid" in str(exc)
    else:
        raise AssertionError("invalid predecessor must be rejected")
