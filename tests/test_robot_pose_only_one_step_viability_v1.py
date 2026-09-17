"""The causal one-step envelope keeps hard acceleration and URDF gates feasible."""

import numpy as np

from chaoyang.ops.run_robot_pose_only_independent_arm_v5 import _one_step_viable_limits
from chaoyang.ops.run_robot_pose_only_independent_arm_v6 import _fill_free


def test_upper_joint_limit_is_anticipated_without_future_frames():
    lower = np.array([0.0])
    upper = np.array([1.0])
    previous = np.array([0.9])
    previous_previous = np.array([0.85])
    lo, hi = _one_step_viable_limits(lower, upper, previous, previous_previous, 0.12)
    np.testing.assert_allclose(hi, [0.98], atol=1e-12)
    assert lo[0] <= hi[0]
    # At the tightest admitted state, next q=1.0 meets the original 0.06
    # acceleration limit exactly and remains inside the URDF joint limit.
    acceleration = 1.0 - 2 * hi[0] + previous[0]
    assert abs(acceleration) <= 0.06 + 1e-12


def test_lower_joint_limit_is_symmetric():
    lo, hi = _one_step_viable_limits(np.array([0.0]), np.array([1.0]),
                                     np.array([0.1]), np.array([0.15]), 0.12)
    np.testing.assert_allclose(lo, [0.02], atol=1e-12)
    assert lo[0] <= hi[0]


def test_collapsed_joint_axis_expands_only_free_coordinates():
    original = np.arange(7, dtype=float)
    free = np.array([True, True, False, True, True, True, True])
    expanded = _fill_free(original, free, np.full(6, 9.0))
    np.testing.assert_array_equal(expanded, [9, 9, 2, 9, 9, 9, 9])
