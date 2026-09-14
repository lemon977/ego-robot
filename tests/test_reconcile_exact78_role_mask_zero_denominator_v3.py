import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools/reconcile_exact78_role_mask_zero_denominator_v3.py"
SPEC = importlib.util.spec_from_file_location("reconcile_v3", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def row(expected, present, full=10, drift=0, empty=None):
    if empty is None:
        empty = full - expected
    return {
        "full_frame_denominator": full,
        "expected_visible_denominator": expected,
        "present_on_expected_visible": present,
        "background_drift_on_not_visible_frames": drift,
        "offscreen_or_unobserved_empty_frames": empty,
    }


def result(left, right):
    return {
        "hard_gates": {
            "human_visible_frame_pass_fraction_at_least_0p95": True,
            "tracker_expected_visible_coverage_fraction_at_least_0p80": False,
            "tracker_offscreen_or_unobserved_is_empty": True,
            "tracker_current_wrist_roi_bounded": True,
            "full_frame_denominator_recorded": True,
        },
        "metrics": {"tracker_reentry_summary": {"left_tracker": left, "right_tracker": right}},
    }


def test_zero_side_is_vacuous_only_when_fully_empty():
    ok, modes = module.eligible(result(row(0, 0, empty=10), row(10, 8, empty=0)))
    assert ok
    assert "NOT_APPLICABLE" in modes["left_tracker"]


def test_nonzero_coverage_threshold_is_not_lowered():
    ok, _ = module.eligible(result(row(0, 0, empty=10), row(10, 7, empty=0)))
    assert not ok


def test_zero_side_must_have_no_drift_and_full_empty_count():
    assert not module.eligible(result(row(0, 0, drift=1, empty=10), row(10, 10, empty=0)))[0]
    assert not module.eligible(result(row(0, 0, empty=9), row(10, 10, empty=0)))[0]
