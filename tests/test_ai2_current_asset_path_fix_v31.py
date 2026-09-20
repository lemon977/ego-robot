from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chaoyang.governance import finalize_ai2_current_asset_path_fix_v31 as finalize
from chaoyang.governance import register_ai2_current_asset_path_fix_v31 as register
from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_ai2_current_asset_path_fix_v31 as run
from chaoyang.ops import run_ai2_real_assets_cohort_audit_v31 as cohort


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _materialize_assets(root: Path) -> None:
    for spec in cohort.SESSION_SPECS:
        for path in cohort._paths(root, spec).values():
            if path is None:
                continue
            _write(path, {"fixture": str(path)})


def _blocked_audit(session_id: str) -> dict[str, Any]:
    return {
        "schema_version": "AI2_REAL_ASSETS_AUDIT_V31",
        "status": "COMPLETED_FAIL_CLOSED_AUDIT",
        "session_id": session_id,
        "blocker_codes": [
            "BLOCKED_INDEPENDENT_PART_OBSERVABILITY",
            "BLOCKED_INDEPENDENT_REPROJECTION",
            "BLOCKED_SUFFIX_PAIR_MATERIALIZATION",
        ],
        "kai22_tier": {
            "sides": {
                "left": {"highest_admitted_level": "KINEMATIC_ONLY"},
                "right": {"highest_admitted_level": "NONE"},
            }
        },
        "model_calls": 0,
        "gpu_calls": 0,
    }


def _setup_governance(root: Path) -> bytes:
    for relative in register.CODE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {relative}\n", encoding="utf-8")
    predecessor = root / register.PREDECESSOR_RELATIVE
    _write(
        predecessor,
        {
            "schema_version": "AI2_REAL_ASSETS_COHORT_AUDIT_V31",
            "status": "REJECTED_AI2_CURRENT_ASSET_AUDIT",
        },
    )
    predecessor_bytes = predecessor.read_bytes()
    paths = register._paths(root)
    _write(paths["authority"], {"schema_version": "test-authority"})
    _write(paths["receipt"], {"governance_revision": 10, "files": {}})
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


def test_narrow_successor_registers_runs_and_finalizes_without_overwriting_predecessor(
    tmp_path: Path,
) -> None:
    predecessor_bytes = _setup_governance(tmp_path)
    _materialize_assets(tmp_path)
    publisher = _publisher(tmp_path)

    registration = register.register(
        expected_revision=10,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T12:00:00+08:00",
    )
    assert registration["status"] == "PASSED"
    assert registration["execution_started"] is False
    route = json.loads((tmp_path / "tasks/current/INDEX.json").read_text())
    assert route["task_packets"][0]["task_id"] == register.TASK_ID
    assert route["task_packets"][0]["execution_allowed"] is True

    def builder(**kwargs: Any) -> dict[str, Any]:
        return _blocked_audit(str(kwargs["session_id"]))

    result = run.run_successor(root=tmp_path, audit_builder=builder)
    assert result["status"] == "BLOCKED_PREREQ"
    assert result["counts"] == {"total": 8, "pass": 0, "blocked": 8, "rejected": 0}
    assert result["all_existing_inputs_sha_bound"] is True
    assert result["poker_asset_directory"] == "playing_cards"
    assert all(
        row["asset_directory"] == "playing_cards"
        for row in result["sessions"]
        if row["task"] == "poker"
    )

    receipt = finalize.finalize(
        expected_revision=11,
        root=tmp_path,
        publisher=publisher,
        created_at="2026-09-20T12:05:00+08:00",
    )
    assert receipt["status"] == "BLOCKED_PREREQ"
    assert receipt["current_index_status"] == "PASS_NO_ACTIVE_TASKS"
    terminal_index = json.loads((tmp_path / "tasks/current/INDEX.json").read_text())
    assert terminal_index["status"] == "PASS_NO_ACTIVE_TASKS"
    assert terminal_index["task_packets"] == []
    assert (tmp_path / register.PREDECESSOR_RELATIVE).read_bytes() == predecessor_bytes


def test_successor_preserves_logical_poker_while_binding_playing_cards(tmp_path: Path) -> None:
    _materialize_assets(tmp_path)
    spec = next(row for row in cohort.SESSION_SPECS if row["task"] == "poker")
    paths = cohort._paths(tmp_path, spec)
    assert spec["task"] == "poker"
    assert "/sessions/playing_cards/" in str(paths["hawor_npz"])
    assert "/sessions/poker/" not in str(paths["hawor_npz"])
