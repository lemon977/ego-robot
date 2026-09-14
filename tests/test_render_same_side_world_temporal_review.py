from __future__ import annotations

import numpy as np

from tools.render_same_side_world_temporal_review import (
    ACCELERATION_LIMIT,
    ARM_SOLVER_STEP_LIMIT,
    ARM_STEP_LIMIT,
    EXPECTED,
    FRAME0_CAMERA_BASE_ATOL,
    HAND_SOLVER_STEP_LIMIT,
    HAND_STEP_LIMIT,
    bounded_limits,
)


def test_both_tasks_require_same_side_accepted_decisions() -> None:
    assert EXPECTED["poker"]["decision"].startswith("ACCEPT_POKER")
    assert EXPECTED["chips"]["decision"].startswith("ACCEPT_CHIPS")
    assert HAND_SOLVER_STEP_LIMIT <= HAND_STEP_LIMIT
    assert ARM_SOLVER_STEP_LIMIT <= ARM_STEP_LIMIT
    assert FRAME0_CAMERA_BASE_ATOL == 1e-12


def test_bounded_limits_intersect_step_and_acceleration() -> None:
    lower = np.asarray([-2.0])
    upper = np.asarray([2.0])
    previous_previous = np.asarray([0.0])
    previous = np.asarray([0.05])
    bounded_lower, bounded_upper = bounded_limits(
        lower, upper, previous, previous_previous, ARM_STEP_LIMIT
    )
    predicted = 2.0 * previous - previous_previous
    assert bounded_lower[0] >= previous[0] - ARM_STEP_LIMIT
    assert bounded_upper[0] <= previous[0] + ARM_STEP_LIMIT
    assert bounded_lower[0] >= predicted[0] - ACCELERATION_LIMIT
    assert bounded_upper[0] <= predicted[0] + ACCELERATION_LIMIT


def test_bounded_limits_preserve_valid_single_point_intersection() -> None:
    lower = np.asarray([0.0])
    upper = np.asarray([1.0])
    previous_previous = np.asarray([0.0])
    previous = np.asarray([0.06])
    bounded_lower, bounded_upper = bounded_limits(
        lower, upper, previous, previous_previous, 0.06
    )
    assert bounded_lower[0] <= bounded_upper[0]


def test_world_first_render_chain_closes() -> None:
    c2w = np.eye(4)
    c2w[:3, 3] = [0.2, -0.4, 1.0]
    world_base = np.eye(4)
    world_base[:3, 3] = [-0.3, 0.5, 0.8]
    camera_base = np.linalg.inv(c2w) @ world_base
    assert np.allclose(c2w @ camera_base, world_base)
