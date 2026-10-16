"""Synthetic CPU fixtures only; these rows are not 0901 observations."""

import numpy as np
import pytest

from chaoyang.pipeline.exact78_chips001_first_anchor_display_safety_v1 import apply_display_policy


def rows(statuses, sources=None, edges=None):
    sources = sources or ["RGB_HAWOR_PREDICTED"] * len(statuses)
    edges = edges or ["NONE"] * len(statuses)
    return [
        {"frame": frame, "side": side, "motion_source": sources[frame],
         "edge_gap_kind": edges[frame],
         "arm": {"frame": frame, "side": side,
                 "motion_source": sources[frame], "status": statuses[frame],
                 "pass_numeric": statuses[frame].startswith("PASS")}}
        for frame in range(len(statuses)) for side in ("left", "right")
    ]


def test_f47_first_legal_anchor_fills_prefix_without_teleport():
    q = np.zeros((49, 2, 7))
    q[47:, :, 0] = 1.5708
    q[48, :, 0] = 1.6
    statuses = ["HOLD_REACH"] * 47 + ["PASS_NUMERIC_CAMERA_DIAGNOSTIC"] * 2
    result, sidecar = apply_display_policy(
        q, rows(statuses), np.zeros((2, 7)), np.ones((2, 49), dtype=bool), 25.0)
    assert np.all(result[:48, :, 0] == 1.5708)
    assert sidecar[92]["display_status"] == "EDGE_HOLD_OVERLONG_DIAGNOSTIC"
    assert sidecar[92]["display_edge_kind"] == "PREFIX"
    assert sidecar[92]["display_right_anchor_frame"] == 47
    assert sidecar[92]["motion_source"] == "RGB_HAWOR_PREDICTED"
    assert sidecar[92]["display_source"] == "FIRST_LEGAL_ARM_ANCHOR_DIAGNOSTIC"
    assert sidecar[92]["diagnostic_future_donor"] is True
    assert sidecar[94]["display_status"] == "LEGAL_NUMERIC_ANCHOR_DIAGNOSTIC"
    assert sidecar[94]["display_minus_previous_display_max_abs_rad"] == 0
    assert sidecar[94]["arm_numeric_pass"] is True
    assert sidecar[96]["display_step_existing_velocity_gate"] == "PASS"
    assert all(r["display_arm_visible"] and r["robot_drawable"] for r in sidecar)
    assert all(not r["motion_evidence_eligible"] and not r["visual_qualified"]
               and not r["training_eligible"] for r in sidecar)


def test_long_internal_gap_two_end_transition_and_long_suffix_hold():
    q = np.zeros((20, 2, 7))
    q[10:, :, 0] = 1.0
    sources = (["RGB_HAWOR_PREDICTED"] + ["DIAGNOSTIC_LONG_GAP"] * 9
               + ["RGB_HAWOR_PREDICTED"] + ["DIAGNOSTIC_LONG_GAP"] * 9)
    edges = ["NONE"] * 11 + ["SUFFIX"] * 9
    valid = np.zeros((2, 20), dtype=bool)
    valid[:, [0, 10]] = True
    statuses = ["PASS_NUMERIC_CAMERA_DIAGNOSTIC"] + ["HOLD_REACH"] * 9
    statuses += ["PASS_NUMERIC_CAMERA_DIAGNOSTIC"] + ["HOLD_REACH"] * 9
    result, sidecar = apply_display_policy(
        q, rows(statuses, sources, edges),
        np.zeros((2, 7)), valid, 25.0)
    assert result[1, 0, 0] == pytest.approx(0.1)
    assert result[5, 0, 0] == pytest.approx(0.5)
    assert result[9, 0, 0] == pytest.approx(0.9)
    assert sidecar[2]["display_status"] == "TWO_END_LONG_INTERPOLATION_DIAGNOSTIC"
    assert sidecar[2]["display_edge_kind"] == "NONE"
    assert sidecar[2]["display_left_anchor_frame"] == 0
    assert sidecar[2]["display_right_anchor_frame"] == 10
    assert sidecar[2]["display_anchor_gap_within_short_limit"] is False
    assert sidecar[22]["display_status"] == "EDGE_HOLD_OVERLONG_DIAGNOSTIC"
    assert sidecar[22]["display_edge_kind"] == "SUFFIX"
    assert sidecar[22]["direct_hawor_prediction"] is False
    assert all(r["display_arm_visible"] for r in sidecar)
    assert all(r["motion_observation_qualified"] is False for r in sidecar)


def test_no_legal_arm_anchor_draws_neutral_complete_robot():
    q = np.zeros((2, 2, 7))
    neutral = np.full((2, 7), 0.4)
    result, sidecar = apply_display_policy(
        q, rows(["HOLD_REACH"] * 2), neutral, np.ones((2, 2), dtype=bool), 25.0)
    assert np.array_equal(result, np.full_like(q, 0.4))
    assert all(r["display_status"] == "NEUTRAL_NO_LEGAL_ARM_ANCHOR_DIAGNOSTIC"
               and r["display_arm_visible"] for r in sidecar)


def test_short_internal_gap_interpolates_and_retains_source_label():
    q = np.zeros((10, 2, 7))
    q[9, :, 0] = 0.9
    valid = np.zeros((2, 10), dtype=bool)
    valid[:, [0, 9]] = True
    sources = ["RGB_HAWOR_PREDICTED"] + ["DETERMINISTIC_INTERPOLATED"] * 8
    sources += ["RGB_HAWOR_PREDICTED"]
    statuses = ["PASS_NUMERIC_CAMERA_DIAGNOSTIC"] + ["HOLD_REACH"] * 8
    statuses += ["PASS_NUMERIC_CAMERA_DIAGNOSTIC"]
    result, sidecar = apply_display_policy(
        q, rows(statuses, sources), np.zeros((2, 7)), valid, 25.0)
    assert result[5, 0, 0] == pytest.approx(0.5)
    assert sidecar[2]["display_status"] == "TWO_END_SHORT_INTERPOLATION_DIAGNOSTIC"
    assert sidecar[2]["motion_source"] == "DETERMINISTIC_INTERPOLATED"
    assert sidecar[2]["motion_evidence_eligible"] is False


def test_large_two_end_step_explicitly_fails_existing_gate():
    q = np.zeros((3, 2, 7))
    q[2, :, 0] = 1.0
    valid = np.ones((2, 3), dtype=bool)
    result, sidecar = apply_display_policy(
        q, rows(["PASS_NUMERIC_CAMERA_DIAGNOSTIC", "HOLD_REACH",
                 "PASS_NUMERIC_CAMERA_DIAGNOSTIC"]), np.zeros((2, 7)), valid, 25.0)
    assert result[1, 0, 0] == pytest.approx(0.5)
    assert sidecar[2]["display_step_existing_velocity_gate"] == "FAIL"
    assert sidecar[2]["visual_qualified"] is False


def test_short_gap_and_edge_limit_reject_mislabeled_overlong_source():
    q = np.zeros((10, 2, 7))
    sources = ["RGB_HAWOR_PREDICTED"] + ["EDGE_HELD"] * 9
    edges = ["NONE"] + ["SUFFIX"] * 9
    valid = np.zeros((2, 10), dtype=bool)
    valid[:, 0] = True
    with pytest.raises(ValueError, match="FRAME_ROW_DRIFT"):
        apply_display_policy(q, rows(["HOLD_REACH"] * 10, sources, edges),
                             np.zeros((2, 7)), valid, 25.0)


@pytest.mark.parametrize("fps,limit", [(25.0, 8), (30.0, 10)])
def test_frozen_short_edge_limit(fps, limit):
    q = np.zeros((limit + 1, 2, 7))
    sources = ["RGB_HAWOR_PREDICTED"] + ["EDGE_HELD"] * limit
    edges = ["NONE"] + ["SUFFIX"] * limit
    valid = np.zeros((2, len(q)), dtype=bool)
    valid[:, 0] = True
    _, sidecar = apply_display_policy(
        q, rows(["PASS_NUMERIC_CAMERA_DIAGNOSTIC"] + ["HOLD_REACH"] * limit,
                sources, edges), np.zeros((2, 7)), valid, fps)
    assert sidecar[-1]["short_gap_limit_frames"] == limit
    assert sidecar[-1]["motion_source"] == "EDGE_HELD"
    assert sidecar[-1]["display_status"] == "EDGE_HOLD_SHORT_DIAGNOSTIC"
    assert sidecar[-1]["motion_observation_qualified"] is False


def test_incomplete_or_inconsistent_grid_fails_closed():
    q = np.zeros((1, 2, 7))
    with pytest.raises(ValueError, match="INCOMPLETE_FRAME_GRID"):
        apply_display_policy(q, [], np.zeros((2, 7)), np.ones((2, 1), dtype=bool), 25.0)
    bad = rows(["PASS_NUMERIC_CAMERA_DIAGNOSTIC"])
    bad[0]["arm"]["pass_numeric"] = False
    with pytest.raises(ValueError, match="FRAME_ROW_DRIFT"):
        apply_display_policy(q, bad, np.zeros((2, 7)), np.ones((2, 1), dtype=bool), 25.0)
