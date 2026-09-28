import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chaoyang.pipeline.huro_wrist_local_shape_v1 import (
    numpy_wrist_local_direction_deltas,
    validate_local_wrist_indices,
    wrist_local_shape_core_source,
)


def rz(degrees):
    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


def test_valid_wrist_rotation_removes_rigid_pose_conflict():
    local = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.04, 0.03, 0.0]])
    target_r, robot_r = rz(80), rz(-35)
    target = local @ target_r.T + np.array([0.2, -0.1, 0.5])
    robot = local @ robot_r.T + np.array([-0.3, 0.4, 0.2])
    dt, dr = numpy_wrist_local_direction_deltas(
        target, robot, target_r[None], robot_r[None], [0, 0, 0], [1]
    )
    np.testing.assert_allclose(dt, dr, atol=1e-12)
    assert not np.allclose(target[:, None] - target[None, :], robot[:, None] - robot[None, :])


def test_unknown_rotation_uses_original_base_frame_not_identity_observation():
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    robot = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dt, dr = numpy_wrist_local_direction_deltas(
        target, robot, rz(90)[None], rz(-90)[None], [0, 0], [0]
    )
    np.testing.assert_array_equal(dt, target[:, None] - target[None, :])
    np.testing.assert_array_equal(dr, robot[:, None] - robot[None, :])


def test_hands_use_independent_wrist_frames():
    local = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    target_r = np.stack([rz(40), rz(-70)])
    robot_r = np.stack([rz(-10), rz(25)])
    target = np.concatenate((local @ target_r[0].T, local @ target_r[1].T + [1, 0, 0]))
    robot = np.concatenate((local @ robot_r[0].T, local @ robot_r[1].T + [0, 1, 0]))
    dt, dr = numpy_wrist_local_direction_deltas(
        target, robot, target_r, robot_r, [0, 0, 1, 1], [1, 1]
    )
    np.testing.assert_allclose(dt[0, 1], dr[0, 1], atol=1e-12)
    np.testing.assert_allclose(dt[2, 3], dr[2, 3], atol=1e-12)


def test_schema_and_rotation_fail_closed():
    with pytest.raises(ValueError, match="LOCAL_WRIST_INDEX"):
        validate_local_wrist_indices([0, 2], 2, 2)
    with pytest.raises(ValueError, match="WRIST_ROTATION_NOT_ORTHONORMAL"):
        numpy_wrist_local_direction_deltas(
            np.zeros((2, 3)), np.zeros((2, 3)), np.ones((1, 3, 3)),
            np.eye(3)[None], [0, 0], [1]
        )


def test_source_transform_is_exact_and_idempotence_rejected():
    source = '''def solve_retargeting_with_projected_joints(
    local_kpt_mask: jnp.ndarray,
):
    @jaxls.Cost.factory
    def local_alignment_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
        kpt_mask: jnp.ndarray,
    ) -> jax.Array:
        T_root_link = fk()
        robot_pos = pos()
        delta_target = keypoints[:, None] - keypoints[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]
        return delta_target
    costs = [
        local_alignment_cost(var_joints, local_keypoints, local_kpt_mask),
    ]
'''
    transformed = wrist_local_shape_core_source(source)
    assert "local_wrist_indices" in transformed
    assert "_completion_wrist_local_deltas" in transformed
    assert "target_wrist_se3, wrist_rot_mask" in transformed
    with pytest.raises(ValueError, match="WRIST_LOCAL_SOURCE_DRIFT"):
        wrist_local_shape_core_source(transformed)
