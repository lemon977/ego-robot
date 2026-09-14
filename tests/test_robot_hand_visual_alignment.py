from __future__ import annotations

import numpy as np
import pytest

from pipeline.robot_hand_visual_alignment import (
    HandVisualAlignmentError,
    angle_deg,
    kai_palmar_axis_local,
    project_camera,
)


def test_signed_kai_palmar_axes_are_chirality_specific() -> None:
    assert np.array_equal(kai_palmar_axis_local("left"), [0.0, -1.0, 0.0])
    assert np.array_equal(kai_palmar_axis_local("right"), [0.0, 1.0, 0.0])
    with pytest.raises(HandVisualAlignmentError):
        kai_palmar_axis_local("screen-left")


def test_angle_and_projection() -> None:
    assert angle_deg(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == pytest.approx(90.0)
    points = np.asarray([[1.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    k = np.asarray([[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
    assert np.allclose(project_camera(points, k), [[60.0, 220.0], [10.0, 20.0]])


def test_projection_rejects_points_behind_camera() -> None:
    with pytest.raises(HandVisualAlignmentError):
        project_camera(np.asarray([[0.0, 0.0, -1.0]]), np.eye(3))
