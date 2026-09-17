from __future__ import annotations

import pytest

from chaoyang.ops.build_rc1_t5_conversion_report import summarize


def _exact() -> dict:
    return {
        "counts": {
            "sessions": 156,
            "first_blocker": {"HAWOR_C": 12, "ROLE_MASK_C": 20, "OBJECT_MASK_C": 23, "CALIBRATION_MISSING": 43, "METRIC_GEOMETRY_READY": 58},
        },
        "conditional_conversion_rates": {},
    }


def _capacity(ready: bool = False) -> dict:
    return {
        "tasks": {
            task: {
                "independent_source_groups_total": 1 if task == "chips" else 3,
                "train_source_groups": 1 if task == "chips" else 3,
                "validation_source_groups": 0,
                "scheduled_train_windows_upper_bound": 100,
                "scheduled_validation_windows_upper_bound": 0,
                "capacity_ready_under_rc1": ready,
                "pair_terminal": "PASS" if ready else "BLOCKED_DATA_VOLUME",
            }
            for task in ("chips", "poker")
        }
    }


def _robot(pending: int = 7) -> dict:
    return {
        "counts": {
            task: {
                "selected": 30,
                "hard_geometry_pass_evidence": 10,
                "failed_quality_c": 0,
                "failed_runtime_final": 0,
                "pending_successor": pending,
                "pending_causal_production": 15,
            }
            for task in ("chips", "poker")
        }
    }


def test_current_rc1_state_is_incomplete_and_does_not_train() -> None:
    value = summarize(
        _exact(), _capacity(False), {"status": "FAILED_QUALITY_C"},
        {"status": "BLOCKED_PREREQ"}, {"status": "PASSED"},
        {"status": "BLOCKED_PREREQ", "paired_bundle_generated": False, "smoke_training_started": False},
        _robot(),
    )
    flags = value["release_flags"]
    assert flags["RATE_FINALIZED"] is False
    assert flags["PIPELINE_OPERATIONALLY_CLOSED"] is False
    assert flags["FOUR_CHECKPOINTS_TRAINED"] is False
    assert flags["RC1_RELEASE_STATUS"] == "INCOMPLETE"
    assert value["tasks"]["chips"]["checkpoint_pair_status"] == "BLOCKED_DATA_VOLUME"
    assert value["tasks"]["poker"]["paired_eligible_windows"] == 0


def test_exact_partition_must_cover_156_once() -> None:
    broken = _exact()
    broken["counts"]["first_blocker"]["HAWOR_C"] = 11
    with pytest.raises(RuntimeError, match="do not sum"):
        summarize(
            broken, _capacity(), {"status": "X"}, {"status": "X"}, {"status": "X"},
            {"status": "X"}, _robot(),
        )


def test_offline_hard_evidence_never_closes_pipeline() -> None:
    robot = _robot(pending=0)
    for task in robot["counts"].values():
        task["pending_causal_production"] = 0
        task["hard_geometry_pass_evidence"] = 30
    value = summarize(
        _exact(), _capacity(True), {"status": "PASSED"}, {"status": "PASSED"}, {"status": "PASSED"},
        {"status": "PASSED", "paired_bundle_generated": False, "smoke_training_started": False}, robot,
    )
    assert value["release_flags"]["RATE_FINALIZED"] is True
    assert value["release_flags"]["PIPELINE_OPERATIONALLY_CLOSED"] is False
    assert value["boundaries"]["offline_robot_hard_geometry_counts_as_training_input"] is False


def test_budget_terminals_keep_rate_unfinalized() -> None:
    robot = _robot(pending=0)
    for task in robot["counts"].values():
        task["pending_causal_production"] = 0
        task["not_evaluated_budget"] = 15
    value = summarize(
        _exact(), _capacity(False), {"status": "FAILED_QUALITY_C"},
        {"status": "BLOCKED_PREREQ"}, {"status": "PASSED"},
        {"status": "BLOCKED_PREREQ", "paired_bundle_generated": False, "smoke_training_started": False},
        robot,
    )
    assert value["release_flags"]["RATE_FINALIZED"] is False
    assert value["tasks"]["chips"]["robot30_not_evaluated_budget"] == 15
