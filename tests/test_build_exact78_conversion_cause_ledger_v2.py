from __future__ import annotations

import csv
import json
from pathlib import Path

import jsonschema
import pytest

from chaoyang.ops.build_exact78_conversion_cause_ledger_v2 import build


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
COHORT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
SCHEMA = ROOT / "contracts/exact78_conversion_cause_ledger_v2.schema.json"


def test_real_exact78_ledger_has_dynamic_exclusive_partition(tmp_path: Path) -> None:
    result = build(
        matrix_path=MATRIX,
        cohort_path=COHORT,
        output_root=tmp_path,
        created_at="2026-09-14T23:59:00+08:00",
    )
    ledger = json.loads((tmp_path / "CONVERSION_CAUSE_LEDGER_V2.json").read_text())
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(ledger, schema)

    assert result["status"] == "PASSED"
    blockers = ledger["counts"]["exclusive_first_blockers"]
    assert blockers == {
        "HAWOR_C": 12,
        "ROLE_AFTER_HAWOR": 20,
        "OBJECT_AFTER_HAWOR_ROLE": 23,
        "CALIBRATION_MISSING_AFTER_TRIPLE": 43,
        "METRIC_READY": 58,
    }
    assert sum(blockers.values()) == 156
    assert len({row["session_id"] for row in ledger["rows"]}) == 156
    current_matrix = json.loads(MATRIX.read_text())
    expected_eligible = sum(
        bool(row["metric_ready_wave0"]) and row.get("clean_state") == "PASSED_GRADE_B"
        for row in current_matrix["rows"]
    )
    assert ledger["counts"]["robot_eligible"] == expected_eligible
    assert ledger["counts"]["robot_candidates"] == 8
    candidate_rate = {r["stage"]: r for r in ledger["conditional_conversion_rates"]}[
        "ROBOT_ELIGIBLE_TO_ROBOT_CANDIDATE"
    ]
    assert candidate_rate == {
        "stage": "ROBOT_ELIGIBLE_TO_ROBOT_CANDIDATE",
        "numerator": 8,
        "denominator": expected_eligible,
        "rate_percent": round(100.0 * 8 / expected_eligible, 4),
    }
    expected_infrastructure = sum(
        row["current_primary_cause_category"] == "INFRASTRUCTURE_FAILURE"
        for row in ledger["rows"]
    )
    assert ledger["failure_cause_attribution"]["categories"]["INFRASTRUCTURE_FAILURE"]["count"] == expected_infrastructure
    assert len(ledger["breakdowns"]["by_task"]) == 2
    assert len(ledger["breakdowns"]["by_date"]) == 3
    assert len(ledger["breakdowns"]["by_acquisition_contract"]) == 3
    assert ledger["successor_new_passes"]["newly_passed_sessions"] == 0

    with (tmp_path / "CONVERSION_CAUSE_LEDGER_V2.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 156
    assert {row["first_blocker"] for row in csv_rows} == set(blockers)


def test_successor_delta_is_counted_without_overwriting_baseline(tmp_path: Path) -> None:
    baseline = json.loads(MATRIX.read_text())
    changed = next(row for row in baseline["rows"] if row["hawor"]["grade"] == "C")
    successor = json.loads(json.dumps(baseline))
    successor_row = next(row for row in successor["rows"] if row["session_id"] == changed["session_id"])
    successor_row["hawor"]["grade"] = "B"
    successor_path = tmp_path / "successor.json"
    successor_path.write_text(json.dumps(successor))

    output = tmp_path / "out"
    build(
        matrix_path=MATRIX,
        cohort_path=COHORT,
        successor_path=successor_path,
        output_root=output,
        created_at="2026-09-14T23:59:00+08:00",
    )
    ledger = json.loads((output / "CONVERSION_CAUSE_LEDGER_V2.json").read_text())
    delta = ledger["successor_new_passes"]
    assert delta["newly_passed_sessions"] == 1
    assert delta["newly_passed_by_stage"]["hawor"] == 1
    assert changed["hawor"]["grade"] == "C"
    assert json.loads(MATRIX.read_text())["rows"][changed["position"]]["hawor"]["grade"] == "C"


def test_refuses_duplicate_sessions_and_different_immutable_overwrite(tmp_path: Path) -> None:
    matrix = json.loads(MATRIX.read_text())
    matrix["rows"][1]["session_id"] = matrix["rows"][0]["session_id"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="unique session_id"):
        build(matrix_path=invalid, cohort_path=COHORT, output_root=tmp_path / "invalid-out")

    output = tmp_path / "valid-out"
    build(
        matrix_path=MATRIX,
        cohort_path=COHORT,
        output_root=output,
        created_at="2026-09-14T23:59:00+08:00",
    )
    with pytest.raises(FileExistsError, match="immutable artifact"):
        build(
            matrix_path=MATRIX,
            cohort_path=COHORT,
            output_root=output,
            created_at="2026-09-15T00:00:00+08:00",
        )
