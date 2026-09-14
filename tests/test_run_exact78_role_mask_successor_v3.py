import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools/run_exact78_role_mask_successor_v3.py"
SPEC = importlib.util.spec_from_file_location("role_v3", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def meta(expected, present, full=10, drift=0, empty=None):
    if empty is None:
        empty = full - expected
    return {
        "full_frame_denominator": full,
        "expected_visible_denominator": expected,
        "present_on_expected_visible": present,
        "visible_coverage_fraction": present / max(expected, 1),
        "offscreen_or_unobserved_empty_frames": empty,
        "background_drift_on_not_visible_frames": drift,
        "records": [],
    }


def detailed():
    return {"human_fixed_denominator": {
        "left_human": [{"joint_visible_denominator": 1, "pass": True}],
        "right_human": [{"joint_visible_denominator": 1, "pass": True}],
    }}


def test_zero_expected_is_not_applicable_only_if_empty_fail_closed():
    result = module.adaptive_gates(
        {"indices": list(range(10))}, detailed(),
        {"left_tracker": meta(0, 0, empty=10), "right_tracker": meta(10, 8, empty=0)},
    )
    assert result["hard_gates"]["tracker_expected_visible_coverage_fraction_at_least_0p80"]
    assert result["tracker_expected_visible_coverage_fractions"]["left_tracker"] is None


def test_nonzero_threshold_remains_0p80():
    result = module.adaptive_gates(
        {"indices": list(range(10))}, detailed(),
        {"left_tracker": meta(0, 0, empty=10), "right_tracker": meta(10, 7, empty=0)},
    )
    assert not result["hard_gates"]["tracker_expected_visible_coverage_fraction_at_least_0p80"]


def test_zero_expected_with_drift_fails():
    result = module.adaptive_gates(
        {"indices": list(range(10))}, detailed(),
        {"left_tracker": meta(0, 0, drift=1, empty=10), "right_tracker": meta(10, 10, empty=0)},
    )
    assert not result["hard_gates"]["tracker_expected_visible_coverage_fraction_at_least_0p80"]
