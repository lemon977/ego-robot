from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chaoyang.governance import finalize_three_stream_stable_baseline_v31 as finalize
from chaoyang.governance.common import artifact_ref, atomic_json


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _setup(tmp_path: Path) -> dict[str, Path]:
    paths = finalize._paths(tmp_path)
    _write(paths["authority"], {"schema_version": "test-authority"})
    _write(
        paths["task_state"],
        {
            "tasks": [
                {
                    "task_id": finalize.TASK_ID,
                    "status": "PENDING",
                    "attempt": 1,
                }
            ],
            "next_task": {"task_id": finalize.TASK_ID},
            "recent_events": [],
        },
    )
    _write(paths["receipt"], {"governance_revision": 10, "files": {}})
    _write(paths["task_packet"], {"task_id": finalize.TASK_ID})
    _write(
        paths["index"],
        {
            "status": "PASS",
            "task_packets": [
                {
                    "task_id": finalize.TASK_ID,
                    "packet_path": str(paths["task_packet"].relative_to(tmp_path)),
                    "packet_sha256": artifact_ref(paths["task_packet"])["sha256"],
                    "execution_allowed": True,
                }
            ],
        },
    )
    _write(
        paths["gpu_queue"],
        {
            "task_id": finalize.TASK_ID,
            "current_owner": None,
            "queue": [],
        },
    )
    statuses = {
        "exact78": ("BLOCKED_CPU_PREFLIGHT", "BLOCKED_E0_PAIR_INPUTS"),
        "ai1": ("CPU_PREFLIGHT_COMPLETE", "PASS_CPU_PREFLIGHT"),
        "ai2": ("BLOCKED_CPU_PREFLIGHT", "BLOCKED_AI2_EVIDENCE"),
    }
    for lane, (lane_status, machine_status) in statuses.items():
        lane_root = paths["attempt"] / "lanes" / lane
        machine = lane_root / "MACHINE_RESULT.json"
        _write(
            machine,
            {
                "status": machine_status,
                "training_ready": False,
                "gpu_ready": False,
                "training_eligible": False,
            },
        )
        _write(
            lane_root / "STATE.json",
            {
                "parent_task_id": finalize.TASK_ID,
                "lane": lane,
                "status": lane_status,
                "machine_result_status": machine_status,
                "latest_artifacts": [artifact_ref(machine)],
                "blocker": machine_status if lane_status != "CPU_PREFLIGHT_COMPLETE" else None,
                "claims": {
                    "PIPELINE_COMPLETE": False,
                    "TRAINING_ELIGIBLE": False,
                },
            },
        )
    return paths


def _status_builder(**kwargs: Any) -> dict[str, Any]:
    parent = next(
        row for row in kwargs["task_state"]["tasks"] if row["task_id"] == finalize.TASK_ID
    )
    return {
        "schema_version": "test-shallow-status",
        "parent_status": parent["status"],
        "parent_routable": kwargs["task_state"]["next_task"] is not None,
        "source_task_state_path": str(kwargs["source_task_state_path"]),
    }


def test_finite_blocked_closure_publishes_all_required_artifacts(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    calls: list[dict[str, Any]] = []

    def publisher(_authority: dict, task_state: dict, **kwargs: Any) -> dict[str, Any]:
        calls.append({"task_state": task_state, **kwargs})
        assert kwargs["expected_revision"] == 10
        assert task_state["next_task"] is None
        parent = next(row for row in task_state["tasks"] if row["task_id"] == finalize.TASK_ID)
        assert parent["status"] == "BLOCKED_PREREQ"
        index = dict(kwargs["task_packet_index_value"])
        index.update(governance_revision=11, generation_id="gov-test", generated_at="test")
        atomic_json(kwargs["task_packet_index_path"], index)
        return {"governance_revision": 11, "generation_id": "gov-test"}

    receipt = finalize.finalize(
        expected_revision=10,
        root=tmp_path,
        publisher=publisher,
        status_builder=_status_builder,
        created_at="2026-09-20T12:00:00+08:00",
    )

    assert len(calls) == 1
    assert receipt["status"] == "BLOCKED_PREREQ"
    assert receipt["finalization_status"] == "PASSED_CAS_PUBLISHED"
    assert receipt["pipeline_complete"] is False
    assert receipt["governance_revision"] == 11
    assert json.loads(paths["index"].read_text())["status"] == "PASS_NO_ACTIVE_TASKS"

    for path in (
        paths["attempt"] / "FINAL_AUDIT.json",
        paths["attempt"] / "RESULT.json",
        paths["attempt"] / "RUN_RECEIPT.json",
        paths["terminal_receipt"],
    ):
        assert path.is_file()
    audit = json.loads((paths["attempt"] / "FINAL_AUDIT.json").read_text())
    assert audit["status"] == "BLOCKED_PREREQ"
    assert audit["pipeline_complete"] is False
    assert audit["all_lanes_terminal_cpu_preflight"] is True
    assert audit["gpu_or_training_ready_signals"] == []
    assert audit["lane_blockers"] == {
        "exact78": "BLOCKED_E0_PAIR_INPUTS",
        "ai2": "BLOCKED_AI2_EVIDENCE",
    }
    result = json.loads((paths["attempt"] / "RESULT.json").read_text())
    assert result["pipeline_complete"] is False
    assert result["training_complete"] is False
    assert result["training_eligible"] is False
    assert result["lane_statuses"]["ai1"] == "CPU_PREFLIGHT_COMPLETE"
    shallow = json.loads(paths["shallow_status"].read_text())
    assert shallow["parent_status"] == "BLOCKED_PREREQ"
    assert shallow["parent_routable"] is False


def test_nonterminal_lane_prevents_any_final_artifact(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    state_path = paths["attempt"] / "lanes/ai2/STATE.json"
    state = json.loads(state_path.read_text())
    state["status"] = "READY_CPU_PREFLIGHT"
    _write(state_path, state)

    with pytest.raises(finalize.FinalizeError, match="not terminal CPU preflight"):
        finalize.finalize(
            expected_revision=10,
            root=tmp_path,
            publisher=lambda *_args, **_kwargs: {},
            status_builder=_status_builder,
        )
    assert not (paths["attempt"] / "FINAL_AUDIT.json").exists()
    assert not paths["terminal_receipt"].exists()


def test_gpu_or_training_ready_work_prevents_finite_closure(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    queue = json.loads(paths["gpu_queue"].read_text())
    queue["ready_jobs"] = [{"status": "READY_GPU", "lane": "exact78"}]
    _write(paths["gpu_queue"], queue)

    with pytest.raises(finalize.FinalizeError, match="GPU/training-ready work remains"):
        finalize.finalize(
            expected_revision=10,
            root=tmp_path,
            publisher=lambda *_args, **_kwargs: {},
            status_builder=_status_builder,
        )
    assert not (paths["attempt"] / "RESULT.json").exists()


def test_completed_lane_requires_explicit_no_ready_proof(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    state = json.loads((paths["attempt"] / "lanes/ai1/STATE.json").read_text())
    machine_path = Path(state["latest_artifacts"][0]["path"])
    _write(machine_path, {"status": "PASS_CPU_PREFLIGHT"})
    state["latest_artifacts"] = [artifact_ref(machine_path)]
    _write(paths["attempt"] / "lanes/ai1/STATE.json", state)

    with pytest.raises(finalize.FinalizeError, match="WITHOUT_EXPLICIT_NO_READY_PROOF"):
        finalize.finalize(
            expected_revision=10,
            root=tmp_path,
            publisher=lambda *_args, **_kwargs: {},
            status_builder=_status_builder,
        )
