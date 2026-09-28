"""Opt-in HuRo hand-shape adapter that separates wrist pose from articulation.

The upstream local-direction residual is expressed in the robot base frame. If
an explicit wrist rotation target is active, that residual can fight the wrist
pose residual through the same arm joints. This module expresses connected hand
directions in the corresponding target/robot wrist frames when, and only when,
the target wrist rotation is valid. Unknown rotations retain the upstream
base-frame behavior; an identity placeholder is never promoted to observation.
"""
from __future__ import annotations

import ast
import numpy as np


def validate_local_wrist_indices(indices, point_count: int, wrist_count: int) -> np.ndarray:
    result = np.asarray(indices)
    if result.shape != (point_count,) or result.dtype.kind not in "iu":
        raise ValueError("LOCAL_WRIST_INDEX_SHAPE_OR_DTYPE")
    if wrist_count < 1 or np.any(result < 0) or np.any(result >= wrist_count):
        raise ValueError("LOCAL_WRIST_INDEX_RANGE")
    return result.astype(np.int32, copy=False)


def wrist_local_direction_deltas(
    target_points,
    robot_points,
    target_wrist_rotation,
    robot_wrist_rotation,
    local_wrist_indices,
    wrist_rotation_valid,
):
    """Return pairwise deltas in wrist-local or fail-closed base coordinates.

    Arrays are for one time step. Pair ``i,j`` uses the wrist assigned to point
    ``i``; cross-hand pairs must remain disabled by the caller's connection mask.
    """
    import jax.numpy as jnp

    point_count = target_points.shape[0]
    target_delta = target_points[:, None] - target_points[None, :]
    robot_delta = robot_points[:, None] - robot_points[None, :]
    target_inverse = jnp.swapaxes(target_wrist_rotation, -1, -2)
    robot_inverse = jnp.swapaxes(robot_wrist_rotation, -1, -2)
    target_local = jnp.einsum(
        "iab,ijb->ija", target_inverse[local_wrist_indices], target_delta
    )
    robot_local = jnp.einsum(
        "iab,ijb->ija", robot_inverse[local_wrist_indices], robot_delta
    )
    use_local = wrist_rotation_valid[local_wrist_indices] > 0.5
    if use_local.shape != (point_count,):
        raise ValueError("WRIST_ROTATION_MASK_SHAPE")
    use_local = use_local[:, None, None]
    return (
        jnp.where(use_local, target_local, target_delta),
        jnp.where(use_local, robot_local, robot_delta),
    )


def numpy_wrist_local_direction_deltas(
    target_points,
    robot_points,
    target_wrist_rotation,
    robot_wrist_rotation,
    local_wrist_indices,
    wrist_rotation_valid,
):
    """Validated CPU oracle wrapper used by deterministic tests and audits."""
    points_t = np.asarray(target_points, dtype=float)
    points_r = np.asarray(robot_points, dtype=float)
    rotations_t = np.asarray(target_wrist_rotation, dtype=float)
    rotations_r = np.asarray(robot_wrist_rotation, dtype=float)
    valid = np.asarray(wrist_rotation_valid, dtype=float)
    if points_t.ndim != 2 or points_t.shape[-1] != 3 or points_r.shape != points_t.shape:
        raise ValueError("POINT_SHAPE")
    if rotations_t.ndim != 3 or rotations_t.shape[1:] != (3, 3) or rotations_r.shape != rotations_t.shape:
        raise ValueError("WRIST_ROTATION_SHAPE")
    if valid.shape != (len(rotations_t),) or not np.isin(valid, [0.0, 1.0]).all():
        raise ValueError("WRIST_ROTATION_MASK")
    if not all(np.isfinite(value).all() for value in (points_t, points_r, rotations_t, rotations_r, valid)):
        raise ValueError("NONFINITE_INPUT")
    indices = validate_local_wrist_indices(local_wrist_indices, len(points_t), len(rotations_t))
    for rotations in (rotations_t, rotations_r):
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-7):
            raise ValueError("WRIST_ROTATION_NOT_ORTHONORMAL")
        if np.any(np.linalg.det(rotations) < 0.999999):
            raise ValueError("WRIST_ROTATION_NOT_PROPER")
    target_delta = points_t[:, None] - points_t[None, :]
    robot_delta = points_r[:, None] - points_r[None, :]
    target_local = np.einsum("iab,ijb->ija", rotations_t.transpose(0, 2, 1)[indices], target_delta)
    robot_local = np.einsum("iab,ijb->ija", rotations_r.transpose(0, 2, 1)[indices], robot_delta)
    use_local = valid[indices, None, None].astype(bool)
    return np.where(use_local, target_local, target_delta), np.where(use_local, robot_local, robot_delta)


def wrist_local_shape_core_source(projected_source: str) -> str:
    """Transform only the pinned projected+wrist solver and fail on drift."""
    source = projected_source

    def once(before: str, after: str) -> None:
        nonlocal source
        if source.count(before) != 1:
            raise ValueError("WRIST_LOCAL_SOURCE_DRIFT: " + before[:70])
        source = source.replace(before, after, 1)

    once(
        "    local_kpt_mask: jnp.ndarray,\n",
        "    local_kpt_mask: jnp.ndarray,\n    local_wrist_indices: jnp.ndarray,\n",
    )
    once(
        "    def local_alignment_cost(\n"
        "        var_values: jaxls.VarValues,\n"
        "        var_robot_cfg: jaxls.Var[jnp.ndarray],\n"
        "        keypoints: jnp.ndarray,\n"
        "        kpt_mask: jnp.ndarray,\n"
        "    ) -> jax.Array:",
        "    def local_alignment_cost(\n"
        "        var_values: jaxls.VarValues,\n"
        "        var_robot_cfg: jaxls.Var[jnp.ndarray],\n"
        "        keypoints: jnp.ndarray,\n"
        "        kpt_mask: jnp.ndarray,\n"
        "        target_wrist: jaxlie.SE3,\n"
        "        target_wrist_rot_mask: jnp.ndarray,\n"
        "    ) -> jax.Array:",
    )
    once(
        "        delta_target = keypoints[:, None] - keypoints[None, :]\n"
        "        delta_robot = robot_pos[:, None] - robot_pos[None, :]\n",
        "        actual_wrist = jaxlie.SE3(T_root_link.wxyz_xyz[wrist_link_indices])\n"
        "        delta_target, delta_robot = _completion_wrist_local_deltas(\n"
        "            keypoints, robot_pos, target_wrist.rotation().as_matrix(),\n"
        "            actual_wrist.rotation().as_matrix(), local_wrist_indices,\n"
        "            target_wrist_rot_mask)\n",
    )
    once(
        "        local_alignment_cost(var_joints, local_keypoints, local_kpt_mask),\n",
        "        local_alignment_cost(var_joints, local_keypoints, local_kpt_mask,\n"
        "                             target_wrist_se3, wrist_rot_mask),\n",
    )
    ast.parse(source)
    return source
