from __future__ import annotations

from pathlib import Path

import pytest

from chaoyang.ops.finalize_rc1_robot30_causal_budget import close_rows


def _rows(*, pending_successor: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in ("chips", "poker"):
        for rank in range(1, 31):
            terminal = "PENDING_CAUSAL_PRODUCTION"
            hard = False
            mode = "CAUSAL_TRAINING_INPUT_REQUIRED"
            if rank <= 10:
                terminal = "PASSED_OFFLINE_VISUAL_HARD_GEOMETRY"
                hard = True
                mode = "OFFLINE_BIDIRECTIONAL_VISUALIZATION"
            if pending_successor and task == "poker" and rank == 30:
                terminal = "PENDING_BOUNDED_SUCCESSOR"
            rows.append(
                {
                    "session_id": f"{task}_{rank:03d}",
                    "task": task,
                    "target_rank": rank,
                    "execution_route": "FRESH_CAUSAL_PREFIX_RUN",
                    "terminal_status": terminal,
                    "hard_geometry_pass": hard,
                    "input_mode": mode,
                    "training_eligible": False,
                    "control_ground_truth": False,
                    "physical_deployment_authorized": False,
                    "result": None,
                }
            )
    return rows


def test_closes_only_causal_pending_as_budget_terminal(tmp_path: Path) -> None:
    progress = {
        "schema_version": "chaoyang-rc1-robot30-terminal-index-v1",
        "rows": _rows(),
    }
    rows, counts = close_rows(
        progress,
        output_root=tmp_path,
        prerequisite_refs={"t0_capacity": {"path": "/x", "bytes": 1, "sha256": "a"}},
    )
    assert len(rows) == 60
    assert counts["chips"]["not_evaluated_budget"] == 20
    assert counts["poker"]["not_evaluated_budget"] == 20
    assert counts["chips"]["hard_geometry_pass_evidence"] == 10
    assert counts["poker"]["hard_geometry_pass_evidence"] == 10
    assert counts["chips"]["training_eligible"] == 0
    terminal = next(row for row in rows if row["session_id"] == "chips_011")
    assert terminal["terminal_status"] == "NOT_EVALUATED_BUDGET"
    assert terminal["training_eligible"] is False
    assert Path(str(terminal["result"]["path"])).is_file()


def test_refuses_to_close_while_successor_is_pending(tmp_path: Path) -> None:
    progress = {
        "schema_version": "chaoyang-rc1-robot30-terminal-index-v1",
        "rows": _rows(pending_successor=True),
    }
    with pytest.raises(RuntimeError, match="bounded successors remain pending"):
        close_rows(progress, output_root=tmp_path, prerequisite_refs={})


def test_offline_evidence_is_never_promoted_to_training(tmp_path: Path) -> None:
    progress = {
        "schema_version": "chaoyang-rc1-robot30-terminal-index-v1",
        "rows": _rows(),
    }
    rows, _ = close_rows(progress, output_root=tmp_path, prerequisite_refs={})
    offline = next(row for row in rows if row["session_id"] == "poker_001")
    assert offline["input_mode"] == "OFFLINE_BIDIRECTIONAL_VISUALIZATION"
    assert offline["training_eligible"] is False
