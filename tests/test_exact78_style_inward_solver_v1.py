import numpy as np

from chaoyang.pipeline.exact78_style_inward_solver_v1 import bounded_joint_limits


def test_bounded_limits_apply_step_and_acceleration():
    lower = np.full(2, -2.0)
    upper = np.full(2, 2.0)
    previous_previous = np.array([0.0, 0.0])
    previous = np.array([0.08, -0.08])
    lo, hi, acceleration = bounded_joint_limits(
        lower,
        upper,
        previous,
        previous_previous,
        step_limit=0.12,
        acceleration_limit=0.06,
    )
    assert acceleration
    np.testing.assert_allclose(lo, [0.10, -0.20])
    np.testing.assert_allclose(hi, [0.20, -0.10])


def test_bounded_limits_never_cross_hard_limits():
    lower = np.array([-1.0, -0.5])
    upper = np.array([1.0, 0.5])
    lo, hi, acceleration = bounded_joint_limits(
        lower,
        upper,
        np.array([0.95, -0.45]),
        step_limit=0.12,
    )
    assert not acceleration
    np.testing.assert_allclose(lo, [0.83, -0.5])
    np.testing.assert_allclose(hi, [1.0, -0.33])
