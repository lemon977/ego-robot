import numpy as np
import pytest

from chaoyang.pipeline.full_robot_review_inward_v3 import (
    head_relative_targets,
    world_first_targets,
)


def _rz(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(rotation=None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.eye(3) if rotation is None else rotation
    result[:3, 3] = translation
    return result


def test_shared_head_axes_do_not_rotate_translation_through_each_wrist():
    wrists = np.zeros((2, 2, 4, 4), dtype=float)
    wrists[0, 0] = _pose(_rz(90), (0.1, 0.2, 0.3))
    wrists[0, 1] = _pose(_rz(-90), (-0.1, -0.2, 0.4))
    wrists[1, 0] = _pose(_rz(110), (0.4, 0.0, 0.5))
    wrists[1, 1] = _pose(_rz(-70), (0.2, -0.4, 0.6))
    valid = np.ones((2, 2), dtype=bool)
    references = np.stack(
        [_pose(_rz(15), (0.5, 0.2, 1.0)), _pose(_rz(-15), (0.5, -0.2, 1.0))]
    )

    targets = head_relative_targets(
        wrists,
        valid,
        references,
        robot_from_head_R=np.eye(3),
        translation_scale=0.75,
    )

    np.testing.assert_allclose(targets[0], references, atol=1e-12)
    expected_delta = np.array([0.225, -0.15, 0.15])
    np.testing.assert_allclose(targets[1, 0, :3, 3] - targets[0, 0, :3, 3], expected_delta)
    np.testing.assert_allclose(targets[1, 1, :3, 3] - targets[0, 1, :3, 3], expected_delta)


def test_invalid_head_axis_or_scale_is_rejected():
    wrists = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    valid = np.ones((1, 2), dtype=bool)
    references = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    with pytest.raises(ValueError, match="RIGHT_HANDED"):
        head_relative_targets(
            wrists,
            valid,
            references,
            robot_from_head_R=np.diag([1.0, 1.0, -1.0]),
            translation_scale=1.0,
        )
    with pytest.raises(ValueError, match="SCALE"):
        head_relative_targets(
            wrists,
            valid,
            references,
            robot_from_head_R=np.eye(3),
            translation_scale=0.0,
        )


def test_world_first_is_unit_gain_in_shared_world_axes():
    wrists = np.zeros((2, 2, 4, 4), dtype=float)
    wrists[0, 0] = _pose(_rz(80), (1.0, 2.0, 3.0))
    wrists[0, 1] = _pose(_rz(-60), (-1.0, -2.0, 2.0))
    wrists[1, 0] = _pose(_rz(90), (1.2, 1.7, 3.4))
    wrists[1, 1] = _pose(_rz(-50), (-0.8, -2.3, 2.4))
    references = np.stack([_pose(translation=(0.4, 0.3, 1.1)), _pose(translation=(0.4, -0.3, 1.1))])
    targets = world_first_targets(
        wrists,
        np.ones((2, 2), dtype=bool),
        references,
        robot_from_world_R=np.eye(3),
    )
    np.testing.assert_allclose(targets[0], references, atol=1e-12)
    np.testing.assert_allclose(targets[1, 0, :3, 3] - targets[0, 0, :3, 3], [0.2, -0.3, 0.4])
    np.testing.assert_allclose(targets[1, 1, :3, 3] - targets[0, 1, :3, 3], [0.2, -0.3, 0.4])
