import numpy as np

from chaoyang.ops.adapt_object6d_rectified_to_selected_v71 import adapt_poses


def test_rectified_pose_is_rotated_before_selected_c2w() -> None:
    pose = np.tile(np.eye(4), (2, 1, 1))
    pose[:, 2, 3] = 1.0
    angle = np.deg2rad(90.0)
    registration = np.eye(4)
    registration[:3, :3] = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[:, 0, 3] = 2.0
    selected, world = adapt_poses(
        pose, registration, c2w, np.asarray([0, 1]), np.asarray([True, True])
    )
    np.testing.assert_allclose(selected[:, :3, 3], [[1.0, 0.0, 0.0]] * 2, atol=1e-8)
    np.testing.assert_allclose(world[:, :3, 3], [[3.0, 0.0, 0.0]] * 2, atol=1e-8)


def test_absolute_frame_index_is_used_for_full_c2w() -> None:
    pose = np.tile(np.eye(4), (2, 1, 1))
    c2w = np.tile(np.eye(4), (5, 1, 1))
    c2w[:, 1, 3] = np.arange(5)
    _, world = adapt_poses(
        pose, np.eye(4), c2w, np.asarray([2, 4]), np.asarray([True, True])
    )
    np.testing.assert_allclose(world[:, 1, 3], [2.0, 4.0])
