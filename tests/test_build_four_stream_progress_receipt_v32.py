from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops.build_four_stream_progress_receipt_v32 import TASK_ID, build_receipt


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _attempt(tmp_path: Path) -> Path:
    root = tmp_path / "attempt_0001"
    _write(root / "RUN_SIGNATURE.json", {
        "task_id": TASK_ID,
        "t0": "2026-09-20T19:04:09+08:00",
    })
    for lane in ("exact78", "ai1", "ai2", "ai4_huro"):
        _write(root / "lanes" / lane / "STATE.json", {
            "lane": lane,
            "parent_task_id": TASK_ID,
            "status": "MILESTONE_COMPLETE",
            "machine_result_status": "PASSED_TEST",
            "blocker": None,
            "claims": {"TRAINING_COMPLETE": False},
            "latest_artifacts": [],
        })
    _write(root / "lanes" / "exact78" / "GPU_COMMAND_RECEIPT.json", {
        "attempt_id": "gpu_test",
        "status": "PASSED",
        "returncode": 0,
        "wait_seconds_observed": 12.5,
    })
    return root


def test_progress_receipt_is_not_allowed_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _attempt(tmp_path)
    monkeypatch.setattr(
        "chaoyang.ops.build_four_stream_progress_receipt_v32._git",
        lambda *args: "clean-head" if args == ("rev-parse", "HEAD") else "",
    )
    with pytest.raises(RuntimeError, match="before its scheduled time"):
        build_receipt(
            checkpoint_hour=2,
            observed_at="2026-09-20T20:00:00+08:00",
            attempt_root=root,
        )


def test_progress_receipt_collects_lane_and_gpu_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _attempt(tmp_path)

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("branch", "--show-current"):
            return "task/test"
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr("chaoyang.ops.build_four_stream_progress_receipt_v32._git", fake_git)
    value = build_receipt(
        checkpoint_hour=2,
        observed_at="2026-09-20T21:04:09+08:00",
        attempt_root=root,
    )
    assert value["status"] == "PROGRESS_SNAPSHOT_NOT_FINAL"
    assert value["elapsed_seconds"] == 7200
    assert set(value["lanes"]) == {"exact78", "ai1", "ai2", "ai4_huro"}
    assert value["resources"]["gpu_observed_wait_seconds_total"] == 12.5
    assert value["repository"]["tracked_and_untracked_clean"] is True
    assert value["authority"]["training_complete"] is False
