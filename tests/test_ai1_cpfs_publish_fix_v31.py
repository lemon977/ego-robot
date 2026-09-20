from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chaoyang.governance import finalize_ai1_cpfs_publish_fix_v31 as finalize
from chaoyang.governance import register_ai1_cpfs_publish_fix_v31 as register
from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_ai1_cpfs_publish_fix_v31 as run


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _setup_governance(root: Path) -> bytes:
    for relative in register.CODE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    predecessor = root / register.PREDECESSOR_RELATIVE
    _write(
        predecessor,
        {
            "schema_version": "chaoyang-wiyh-ai1-lane-result-v31",
            "status": "FAILED_RUNTIME",
            "lane": "ai1",
        },
    )
    predecessor_bytes = predecessor.read_bytes()
    paths = register._paths(root)
    _write(paths["authority"], {"schema_version": "test-authority"})
    _write(paths["receipt"], {"governance_revision": 20, "files": {}})
    _write(paths["task_state"], {"tasks": [], "next_task": None, "recent_events": []})
    _write(
        paths["index"],
        {
            "schema_version": "chaoyang-v71-task-packet-index-v3",
            "status": "PASS_NO_ACTIVE_TASKS",
            "task_packets": [],
        },
    )
    index_ref = artifact_ref(paths["index"])
    _write(
        paths["pointer"],
        {
            "index_path": str(paths["index"].relative_to(root)),
            "index_sha256": index_ref["sha256"],
        },
    )
    return predecessor_bytes


def _publisher(root: Path):
    def publish(
        _authority: dict[str, Any],
        task_state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        revision = int(kwargs["expected_revision"]) + 1
        generation = f"gov-test-{revision}"
        index = dict(kwargs["task_packet_index_value"])
        index.update(governance_revision=revision, generation_id=generation)
        atomic_json(kwargs["task_packet_index_path"], index)
        atomic_json(root / "docs/governance/LONG_HORIZON_TASK_STATE.json", task_state)
        atomic_json(
            root / "docs/governance/CURRENT_STATUS_RECEIPT.json",
            {"governance_revision": revision, "files": {}},
        )
        return {"governance_revision": revision, "generation_id": generation}

    return publish


def _blocked_runner(calls: list[dict[str, Any]]):
    def runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        result = {
            "schema_version": "chaoyang-wiyh-ai1-lane-result-v31",
            "status": "BLOCKED_ADOPTION_OBSERVATIONS",
            "lane": "ai1",
            "blocker": "play_cards_0916_102:BLOCKED_MISSING_PINNED_OBSERVATION",
        }
        atomic_json(output / "RESULT.json", result)
        return result

    return runner


def test_narrow_successor_registers_runs_and_finalizes_without_overwriting_predecessor(
    tmp_path: Path,
) -> None:
    predecessor_bytes = _setup_governance(tmp_path)
    processed = tmp_path / "processed"
    experiments = tmp_path / "experiments"
    processed.mkdir()
    experiments.mkdir()
    publisher = _publisher(tmp_path)

    registration = register.register(
        expected_revision=20,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T13:00:00+08:00",
    )
    assert registration["status"] == "PASSED"
    assert registration["execution_started"] is False
    route = json.loads((tmp_path / "tasks/current/INDEX.json").read_text())
    assert route["task_packets"][0]["task_id"] == register.TASK_ID
    assert route["task_packets"][0]["execution_allowed"] is True

    calls: list[dict[str, Any]] = []
    result = run.run_successor(
        root=tmp_path,
        processed_root=processed,
        experiment_root=experiments,
        runner=_blocked_runner(calls),
    )
    assert result["status"] == "BLOCKED_PREREQ"
    assert result["machine_status"] == "BLOCKED_ADOPTION_OBSERVATIONS"
    assert result["runner_invocations"] == 1
    assert result["model_calls"] == 0
    assert result["gpu_calls"] == 0
    assert len(calls) == 1
    assert calls[0]["processed_root"] == processed
    assert calls[0]["experiment_root"] == experiments
    assert result["predecessor_preserved"] is True
    assert (tmp_path / register.PREDECESSOR_RELATIVE).read_bytes() == predecessor_bytes
    with pytest.raises(run.SuccessorRunError, match="fresh successor attempt"):
        run.run_successor(
            root=tmp_path,
            processed_root=processed,
            experiment_root=experiments,
            runner=_blocked_runner([]),
        )

    receipt = finalize.finalize(
        expected_revision=21,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T13:05:00+08:00",
    )
    assert receipt["status"] == "BLOCKED_PREREQ"
    assert receipt["machine_status"] == "BLOCKED_ADOPTION_OBSERVATIONS"
    assert receipt["current_index_status"] == "PASS_NO_ACTIVE_TASKS"
    terminal_index = json.loads((tmp_path / "tasks/current/INDEX.json").read_text())
    assert terminal_index["status"] == "PASS_NO_ACTIVE_TASKS"
    assert terminal_index["task_packets"] == []
    assert (tmp_path / register.PREDECESSOR_RELATIVE).read_bytes() == predecessor_bytes


def test_registration_requires_the_sealed_failed_runtime_predecessor(tmp_path: Path) -> None:
    _setup_governance(tmp_path)
    predecessor = tmp_path / register.PREDECESSOR_RELATIVE
    value = json.loads(predecessor.read_text(encoding="utf-8"))
    value["status"] = "BLOCKED_ADOPTION_OBSERVATIONS"
    _write(predecessor, value)
    with pytest.raises(register.RegisterError, match="FAILED_RUNTIME"):
        register.register(
            expected_revision=20,
            root=tmp_path,
            publisher=_publisher(tmp_path),
        )


def test_runner_requires_the_sole_execution_allowed_route(tmp_path: Path) -> None:
    _setup_governance(tmp_path)
    processed = tmp_path / "processed"
    experiments = tmp_path / "experiments"
    processed.mkdir()
    experiments.mkdir()
    with pytest.raises(run.SuccessorRunError, match="sole current route"):
        run.run_successor(
            root=tmp_path,
            processed_root=processed,
            experiment_root=experiments,
            runner=_blocked_runner([]),
        )
