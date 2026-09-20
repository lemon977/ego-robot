from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.governance.build_four_stream_status_v32 import TASK_ID, build_status


def test_unregistered_v32_status_is_machine_derived(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"tasks": [], "next_task": None}))
    value = build_status(
        task_state={"tasks": [], "next_task": None},
        generated_at="2026-09-20T00:00:00+08:00",
        source_task_state_path=tmp_path / "state.json",
        run_root=tmp_path / "run",
    )
    assert value["parent"]["registered"] is False
    assert value["lanes"]["exact78"]["status"] == "NOT_REGISTERED"
    assert value["global_authority"]["physical_deployable"] is False


def test_lane_cannot_promote_global_authority(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(json.dumps({"tasks": [], "next_task": None}))
    lane = tmp_path / "run/lanes/ai4_huro"
    lane.mkdir(parents=True)
    (lane / "STATE.json").write_text(json.dumps({
        "lane": "ai4_huro",
        "parent_task_id": TASK_ID,
        "status": "MILESTONE_COMPLETE",
        "claims": {"PHYSICAL_DEPLOYABLE": True},
        "latest_artifacts": [],
    }))
    value = build_status(
        task_state={
            "tasks": [{"task_id": TASK_ID, "status": "PENDING", "attempt": 1}],
            "next_task": {"task_id": TASK_ID},
        },
        generated_at="2026-09-20T00:00:00+08:00",
        source_task_state_path=tmp_path / "state.json",
        run_root=tmp_path / "run",
    )
    assert value["lanes"]["ai4_huro"]["status"] == "MILESTONE_COMPLETE"
    assert value["global_authority"]["physical_deployable"] is False


def test_lane_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    lane = tmp_path / "run/lanes/ai1"
    lane.mkdir(parents=True)
    (lane / "STATE.json").write_text(json.dumps({
        "lane": "ai2", "parent_task_id": TASK_ID, "latest_artifacts": []
    }))
    with pytest.raises(RuntimeError, match="lane state identity mismatch"):
        build_status(
            task_state={"tasks": [], "next_task": None},
            generated_at="2026-09-20T00:00:00+08:00",
            source_task_state_path=tmp_path / "state.json",
            run_root=tmp_path / "run",
        )
