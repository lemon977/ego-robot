from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops.run_human_to_robot_sensor_display_correction import (
    fixed_local_scale, metric_world_xy, run_case,
)


def test_world_x_and_z_have_exactly_equal_pixels_per_metre() -> None:
    center = np.array([1.2, -0.4])
    origin = np.array([1.2, 0.0, -0.4])
    x = np.array([1.3, 0.0, -0.4])
    z = np.array([1.2, 0.0, -0.3])
    a, b, c = (metric_world_xy(v, center, 200.0) for v in (origin, x, z))
    assert b[0] - a[0] == 20
    assert a[1] - c[1] == 20


def test_local_scale_is_shared_across_frames_and_manus_robot() -> None:
    human = np.zeros((2, 2, 21, 3), np.float64)
    human[1, 0, 4, 0] = .20
    robot = np.zeros_like(human)
    robot[0, 1, 4, 1] = .05
    motion = {"manus_local_21_m": human,
              "manus_local_joint_valid": np.ones((2, 2, 21), bool)}
    robot_data = {"fk21_root_relative": robot, "valid": np.ones((2, 2), bool)}
    assert np.isclose(fixed_local_scale(motion, robot_data), .22)


def test_renderer_consumes_saved_arrays_and_keeps_clipped_bones() -> None:
    source = inspect.getsource(run_case)
    assert 'root / "HAND_MOTION_V1.npz"' in source
    assert 'root / "KAI22_COMMON_BACKEND_V1.npz"' in source
    assert "_draw_clipped_skeleton" in source
    assert "draw_world" in source and "draw_local" in source
    assert "source_video" in source
