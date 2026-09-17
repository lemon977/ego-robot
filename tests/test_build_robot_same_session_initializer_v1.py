"""No foreign-session placement or future frame is silently admitted."""

from __future__ import annotations

import numpy as np
import pytest

from chaoyang.ops import build_robot_same_session_initializer_v1 as placement


def test_base_translation_uses_bilateral_same_session_anchors() -> None:
    neutral = np.array([[0.0, 1.0, 1.2], [0.0, -1.0, 1.2]])
    observed = np.array([[0.3, 0.2, 0.8], [0.3, -0.2, 0.8]])
    assert np.allclose(placement.independent_base_translation(observed, neutral), [0.3, 0.0, -0.4])


def test_first_observed_is_not_frame_count_or_foreign_template() -> None:
    joints = np.full((2, 6, 21, 3), np.nan)
    joints[0, 2] = 0.0
    joints[1, 4] = 1.0
    anchors, valid = placement.first_observed(joints, 5)
    assert anchors.tolist() == [2, 4]
    assert valid[0].tolist() == [False, False, True, False, False, False]
    with pytest.raises(RuntimeError, match="no valid observation inside frozen prefix"):
        placement.first_observed(joints, 4)


def test_prefix_excludes_future_inputs() -> None:
    full = {
        "c2w": np.arange(6 * 16).reshape(6, 4, 4),
        "joints_3d_world": np.arange(2 * 6 * 21 * 3).reshape(2, 6, 21, 3),
        "original_frame_indices": np.arange(6),
        "mano_joint_names": np.arange(21),
    }
    result = placement.prefix_arrays(full, 3)
    assert result["c2w"].shape[0] == 3
    assert result["joints_3d_world"].shape[1] == 3
    assert np.array_equal(result["mano_joint_names"], full["mano_joint_names"])
    full["joints_3d_world"][:, 3:] = -999
    assert np.array_equal(result["joints_3d_world"], placement.prefix_arrays(full, 3)["joints_3d_world"])


def test_invalid_base_geometry_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        placement.independent_base_translation(np.full((2, 3), np.nan), np.zeros((2, 3)))
