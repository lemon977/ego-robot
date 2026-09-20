from __future__ import annotations

import numpy as np

from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    HUMAN_TO_PHYSICAL,
    keypoints_from_q,
    load_hand_model,
    solve_frame,
    solve_sequence,
)
from chaoyang.ops.run_huro_derived_hand_only_v1 import (
    KAI_URDFS,
    load_local_q22_comparison,
)


def _hands():
    return (
        load_hand_model(KAI_URDFS[0], "left"),
        load_hand_model(KAI_URDFS[1], "right"),
    )


def test_neutral_target_is_finite_and_within_limits() -> None:
    hand = _hands()[0]
    neutral = 0.5 * (hand.lower + hand.upper)
    target = keypoints_from_q(hand, neutral)
    q, points, diagnostics = solve_frame(hand, target, max_evaluations=10)
    assert diagnostics.success
    assert np.all(np.isfinite(q))
    assert np.all(q >= hand.lower)
    assert np.all(q <= hand.upper)
    assert points.shape == (21, 3)


def test_missing_observation_stays_missing_and_side_mapping_is_explicit() -> None:
    hands = _hands()
    frame_count = 2
    joints = np.zeros((2, frame_count, 21, 3), dtype=np.float64)
    right_neutral = 0.5 * (hands[0].lower + hands[0].upper)
    joints[1, 0] = keypoints_from_q(hands[0], right_neutral)
    observed = np.zeros((2, frame_count), dtype=bool)
    observed[1, 0] = True
    result = solve_sequence(hands, joints, observed, max_evaluations=10)
    assert np.array_equal(result["human_to_physical"], HUMAN_TO_PHYSICAL)
    assert result["valid"][0, 0]
    assert not result["valid"][0, 1]
    assert not result["valid"][1].any()
    assert np.isnan(result["q22"][1]).all()


def test_solver_never_turns_invalid_frame_into_observed() -> None:
    hands = _hands()
    joints = np.ones((2, 3, 21, 3), dtype=np.float64)
    observed = np.zeros((2, 3), dtype=bool)
    result = solve_sequence(hands, joints, observed, max_evaluations=2)
    assert not result["valid"].any()
    assert np.isnan(result["q22"]).all()
    assert np.isnan(result["fk21_root_relative"]).all()


def test_local_comparison_accepts_explicit_kinematic_only_fields() -> None:
    q22 = np.zeros((3, 2, 22), dtype=np.float64)
    valid = np.asarray([[True, False], [False, True], [True, True]])
    q22[~valid] = np.nan
    loaded_q, loaded_valid, q_key, valid_key = load_local_q22_comparison(
        {"q22_frozen_postclip": q22, "q22_valid_physical": valid}, q22.shape
    )
    assert q_key == "q22_frozen_postclip"
    assert valid_key == "q22_valid_physical"
    assert np.array_equal(loaded_valid, valid)
    assert np.array_equal(loaded_q[valid], q22[valid])
