"""Small deterministic guards for the offline independent placement successor."""

import numpy as np
import pytest

from chaoyang.ops.run_robot_pose_only_independent_placement_pilot_v1 import _proper, _world_base


def test_first_prefix_world_base_maps_observed_midpoint_to_frozen_target():
    c2w = np.eye(4)
    c2w[:3, 3] = [0.1, -0.2, 0.3]
    midpoint = np.array([0.2, 0.1, 0.4])
    # Robot +X follows camera +Z, +Y follows camera -X, +Z follows camera -Y.
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    target = np.array([0.45, 0, 1.0])
    world_base = _world_base(c2w, midpoint, rotation, target)
    np.testing.assert_allclose(world_base[:3, :3] @ target + world_base[:3, 3], midpoint)
    _proper(world_base, 'test')


def test_invalid_rotation_is_rejected():
    transform = np.eye(4)
    transform[0, 0] = -1
    with pytest.raises(RuntimeError, match='improper rotation'):
        _proper(transform, 'mirrored')


def test_nonfinite_transform_is_rejected():
    transform = np.eye(4)
    transform[0, 3] = np.nan
    with pytest.raises(RuntimeError, match='nonfinite'):
        _proper(transform, 'nan')
