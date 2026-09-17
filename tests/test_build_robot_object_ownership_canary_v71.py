import numpy as np

from chaoyang.ops.build_robot_object_ownership_canary_v71 import ellipsoid_triangles, ownership_from_depth, rectified_object_to_selected


def test_ellipsoid_mesh_is_metric_and_finite():
    size = np.asarray([0.05, 0.038, 0.003])
    triangles = ellipsoid_triangles(size, rings=6, sectors=12)
    assert triangles.shape == (144, 3, 3)
    assert np.isfinite(triangles).all()
    assert np.max(np.abs(triangles), axis=(0, 1)).tolist() == [0.025, 0.019, 0.0015]


def test_depth_ownership_and_missing_appearance_fail_closed():
    inf = np.inf
    robot = np.asarray([[0.5, 0.4, inf, 0.5, 0.5]])
    obj = np.asarray([[0.4, 0.5, 0.4, 0.501, 0.4]])
    visible = np.asarray([[True, True, True, True, False]], dtype=bool)
    owner = ownership_from_depth(robot, obj, visible)
    assert owner.tolist() == [[2, 3, 2, 4, 4]]


def test_robot_only_and_background_are_explicit():
    inf = np.inf
    owner = ownership_from_depth(
        np.asarray([[0.5, inf]]), np.asarray([[inf, inf]]), np.zeros((1, 2), dtype=bool)
    )
    assert owner.tolist() == [[3, 0]]


def test_rectified_pose_is_explicitly_mapped_to_selected_camera():
    correction = np.eye(4)
    correction[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    result = rectified_object_to_selected(correction, pose)
    assert np.allclose(result[:3, 3], [-2, 1, 3])
