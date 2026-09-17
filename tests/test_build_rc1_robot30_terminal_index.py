from __future__ import annotations

import json
from pathlib import Path

import chaoyang.ops.build_rc1_robot30_terminal_index as indexer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_offline_hard_pass_never_becomes_causal_training_input(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    selection = tmp_path / "selection.json"
    v77 = tmp_path / "v77.json"
    missing_poker189 = tmp_path / "missing-poker189.json"

    rows = []
    for rank in range(1, 31):
        rows.append(
            {
                "session_id": f"chips_{rank:03d}",
                "task": "chips",
                "target_rank": rank,
                "execution_route": "FRESH_CAUSAL_PREFIX_RUN",
            }
        )
    for rank in range(1, 31):
        rows.append(
            {
                "session_id": f"poker_{rank:03d}",
                "task": "poker",
                "target_rank": rank,
                "execution_route": (
                    "BOUNDED_SUCCESSOR_REQUIRED" if rank == 1 else "FRESH_CAUSAL_PREFIX_RUN"
                ),
            }
        )
    _write_json(selection, {"rows": rows})
    _write_json(v77, {"rows": []})

    result_path = run / "robot30_offline_visual/fresh_batch_001/attempt_0001/RESULT.json"
    _write_json(
        result_path,
        {
            "schema_version": "chaoyang-rc1-robot30-offline-visual-batch-v1",
            "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
            "training_eligible": False,
            "rows": [
                {
                    "session": "poker_001",
                    "task": "poker",
                    "terminal_status": "PASSED",
                    "hard_geometry_pass": True,
                }
            ],
        },
    )

    monkeypatch.setattr(indexer, "RUN", run)
    monkeypatch.setattr(indexer, "SELECTION", selection)
    monkeypatch.setattr(indexer, "V77_INDEX", v77)
    monkeypatch.setattr(indexer, "POKER189_V72", missing_poker189)

    indexed_rows, counts = indexer.build_rows()
    by_session = {row["session_id"]: row for row in indexed_rows}

    offline = by_session["poker_001"]
    assert offline["terminal_status"] == "PASSED_OFFLINE_VISUAL_HARD_GEOMETRY"
    assert offline["hard_geometry_pass"] is True
    assert offline["input_mode"] == "OFFLINE_BIDIRECTIONAL_VISUALIZATION"
    assert offline["training_eligible"] is False
    assert counts["poker"]["hard_geometry_pass_evidence"] == 1

    causal = by_session["chips_001"]
    assert causal["terminal_status"] == "PENDING_CAUSAL_PRODUCTION"
    assert causal["hard_geometry_pass"] is False
    assert causal["input_mode"] == "CAUSAL_TRAINING_INPUT_REQUIRED"
    assert causal["training_eligible"] is False


def test_offline_result_marked_train_eligible_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    run = tmp_path / "run"
    result_path = run / "robot30_offline_visual/fresh_batch_001/attempt_0001/RESULT.json"
    _write_json(
        result_path,
        {
            "schema_version": "chaoyang-rc1-robot30-offline-visual-batch-v1",
            "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
            "training_eligible": True,
            "rows": [],
        },
    )
    monkeypatch.setattr(indexer, "RUN", run)
    monkeypatch.setattr(indexer, "POKER189_V72", tmp_path / "missing.json")

    try:
        indexer._offline_results()
    except RuntimeError as exc:
        assert "incorrectly marked train eligible" in str(exc)
    else:
        raise AssertionError("offline training-eligible result must fail closed")
