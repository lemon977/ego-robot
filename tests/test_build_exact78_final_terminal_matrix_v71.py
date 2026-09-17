import json
from pathlib import Path

import pytest

from chaoyang.ops.build_exact78_final_terminal_matrix_v71 import (
    CLEAN_PASS_STATES,
    assert_clean_fact_ledger_closed,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact78_final_matrix_has_78_plus_78_and_no_duplicate() -> None:
    path = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal/EXACT78_FINAL_TERMINAL_MATRIX.json"
    if not path.exists():
        return  # published only after the active Clean queue reaches zero
    value = json.loads(path.read_text())
    assert value["counts"] == {"total": 156, "chips": 78, "poker": 78, "wave0": 58}
    assert len(value["rows"]) == 156
    assert len({row["session_id"] for row in value["rows"]}) == 156


def test_current_clean_pass_name_is_terminal_and_join_ready() -> None:
    assert "PASSED_GRADE_B" in CLEAN_PASS_STATES


def test_final_matrix_refuses_pending_or_active_clean_fact_state() -> None:
    with pytest.raises(RuntimeError, match="still pending"):
        assert_clean_fact_ledger_closed({"waves": {"wave0_clean_pending": 1}})
    with pytest.raises(RuntimeError, match="active Clean"):
        assert_clean_fact_ledger_closed({
            "waves": {"wave0_clean_pending": 0},
            "active_tasks": [{"task_id": "exact78_clean_r70_v71"}],
        })
    assert_clean_fact_ledger_closed({"waves": {"wave0_clean_pending": 0}, "active_tasks": []})
