from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.governance import build_three_stream_status_v31 as subject


def test_unregistered_status_is_machine_derived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_state_path = tmp_path / "task_state.json"
    task_state_path.write_text(json.dumps({"tasks": [], "next_task": None}))
    monkeypatch.setattr(subject, "TASK_STATE_PATH", task_state_path)
    value = subject.build_status(
        task_state={"tasks": [], "next_task": None},
        parent_task_id="parent",
        run_root=tmp_path / "run",
        generated_at="2026-09-20T00:00:00+08:00",
    )
    assert value["parent"]["registered"] is False
    assert value["lanes"]["exact78"]["status"] == "NOT_REGISTERED"
    assert value["global_authority"]["control_ground_truth"] is False


def test_lane_state_cannot_change_global_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_state_path = tmp_path / "task_state.json"
    task_state_path.write_text(json.dumps({"tasks": [], "next_task": None}))
    monkeypatch.setattr(subject, "TASK_STATE_PATH", task_state_path)
    lane = tmp_path / "run/lanes/exact78"
    lane.mkdir(parents=True)
    (lane / "STATE.json").write_text(
        json.dumps(
            {
                "lane": "exact78",
                "status": "RUNNING",
                "authority": "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE",
                "latest_artifacts": [],
            }
        )
    )
    value = subject.build_status(
        task_state={
            "tasks": [{"task_id": "parent", "status": "RUNNING", "attempt": 1}],
            "next_task": {"task_id": "parent"},
        },
        parent_task_id="parent",
        run_root=tmp_path / "run",
        generated_at="2026-09-20T00:00:00+08:00",
    )
    assert value["parent"]["routable"] is True
    assert value["lanes"]["exact78"]["status"] == "RUNNING"
    assert value["global_authority"]["physical_deployable"] is False


def test_lane_identity_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_state_path = tmp_path / "task_state.json"
    task_state_path.write_text(json.dumps({"tasks": [], "next_task": None}))
    monkeypatch.setattr(subject, "TASK_STATE_PATH", task_state_path)
    lane = tmp_path / "run/lanes/ai1"
    lane.mkdir(parents=True)
    (lane / "STATE.json").write_text(json.dumps({"lane": "ai2", "latest_artifacts": []}))
    with pytest.raises(RuntimeError, match="lane identity mismatch"):
        subject.build_status(
            task_state={"tasks": [], "next_task": None},
            parent_task_id="parent",
            run_root=tmp_path / "run",
            generated_at="2026-09-20T00:00:00+08:00",
        )

