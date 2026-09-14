from __future__ import annotations

import pytest

from tools import run_exact78_robot_arm_placement_sweep_v52 as sweep


def metrics(**updates):
    value = {
        "failed_rows": 0,
        "position_mm_max": 10.0,
        "rotation_deg_max": 5.0,
        "branch_margin_m_min": 0.0,
        "velocity_rad_per_frame_max_contiguous": 0.12,
        "acceleration_rad_per_frame2_max_contiguous": 0.06,
    }
    value.update(updates)
    return value


def test_frozen_candidate_order_and_labels() -> None:
    values = [0.2, 0.24, 0.258, 0.2595, 0.26, 0.28, 0.3]
    assert sorted(values, key=lambda x: (abs(x - sweep.PRIOR_M), x)) == [
        0.26,
        0.2595,
        0.258,
        0.24,
        0.28,
        0.3,
        0.2,
    ]
    assert sweep.backoff_label(0.2595) == "0p2595"
    assert sweep.backoff_label(0.2) == "0p2"


def test_numeric_score_is_fail_closed() -> None:
    passed = {"status": "PASS_NUMERIC_CANARY_NO_AUTHORITY", "metrics": metrics()}
    assert sweep.numeric_score(passed, 0.258)[0] is False
    status_hold = {"status": "HOLD_NUMERIC_CANARY", "metrics": metrics()}
    assert sweep.numeric_score(status_hold, 0.26)[0] is True
    failed_row = {
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY",
        "metrics": metrics(failed_rows=1),
    }
    assert sweep.numeric_score(failed_row, 0.26)[0] is True
    excess = {
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY",
        "metrics": metrics(position_mm_max=10.001),
    }
    assert sweep.numeric_score(excess, 0.26)[0] is True


def test_only_exact_empty_bounds_exception_is_deterministic_nonpass() -> None:
    exact = "tools.render.TemporalReviewError: empty previous-accepted bounds: max_excess=0.03"
    assert sweep.deterministic_bounds_failure(1, False, exact)
    assert not sweep.deterministic_bounds_failure(0, False, exact)
    assert not sweep.deterministic_bounds_failure(1, True, exact)
    assert not sweep.deterministic_bounds_failure(1, False, "MemoryError")


def test_forward_tool_override_must_be_in_preflight_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    override = {"path": "/abs/v3.py", "bytes": 3, "sha256": "a" * 64}
    monkeypatch.setattr(sweep, "verify_ref", lambda value, label: sweep.Path("/abs/v3.py"))
    monkeypatch.setattr(
        sweep,
        "ref",
        lambda path: {"path": str(path), "bytes": 3, "sha256": "a" * 64},
    )
    assert sweep.select_arm_tool(
        {"arm_forward_tool": override, "programs": {"v3.py": override}}
    ) == sweep.Path("/abs/v3.py")
    with pytest.raises(ValueError, match="program closure"):
        sweep.select_arm_tool({"arm_forward_tool": override, "programs": {}})


def test_deterministic_prior_status_is_admitted_for_remaining_placements() -> None:
    assert (
        "HOLD_ARM_PRIOR_DETERMINISTIC_INFEASIBLE_ELIGIBLE_PLACEMENT_SWEEP"
        in sweep.ALLOWED_PRIOR_STATUSES
    )
