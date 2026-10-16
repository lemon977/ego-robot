"""Exact78-style bounded multiseed arm solver with an inward-forearm gate."""
from __future__ import annotations

import numpy as np

from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES


def bounded_joint_limits(lower, upper, previous, previous_previous=None, *,
                         step_limit=0.12, acceleration_limit=0.06):
    """Return deterministic per-frame bounds without crossing hard limits."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    lo = np.maximum(lower, previous - float(step_limit))
    hi = np.minimum(upper, previous + float(step_limit))
    acceleration_applied = previous_previous is not None
    if previous_previous is not None:
        predicted = 2.0 * previous - np.asarray(previous_previous, dtype=np.float64)
        accel_lo = predicted - float(acceleration_limit)
        accel_hi = predicted + float(acceleration_limit)
        candidate_lo = np.maximum(lo, accel_lo)
        candidate_hi = np.minimum(hi, accel_hi)
        if np.all(candidate_lo <= candidate_hi):
            lo, hi = candidate_lo, candidate_hi
        else:
            acceleration_applied = False
    if np.any(lo > hi):
        raise RuntimeError("EMPTY_BOUNDED_JOINT_INTERVAL")
    return lo, hi, acceleration_applied


def _pose_error(assets, mounts, side, q, target):
    actual = arm._tool_fk(assets, side, q) @ mounts[side]
    delta = np.linalg.inv(target) @ actual
    position_mm = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    rotation_deg = float(np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3]))))
    return actual, position_mm, rotation_deg


def solve_inward_multiseed(source, assets, mounts, inward_margin_fn, *,
                           minimum_inward_m=0.01, max_nfev=500,
                           step_limit=0.12, acceleration_limit=0.06,
                           continuity_weight=0.01):
    """Resolve frozen targets using bounded multiseed IK and an inward gate."""
    from scipy.optimize import least_squares

    result = dict(source)
    target = np.asarray(source["T_target_root"], dtype=np.float64)
    valid = np.asarray(source["wrist_valid"], dtype=bool)
    edge = np.asarray(source["time_edge_valid"], dtype=bool)
    lower, upper = arm._arm_limits(assets)
    neutral = np.asarray(source["neutral_q_arm"], dtype=np.float64)
    reference = np.asarray(source["reference_q_arm"], dtype=np.float64)
    tool_mounts = np.asarray(source["T_tool_hand"], dtype=np.float64)
    count = len(valid)

    q_arm = np.full((count, 2, 7), np.nan, dtype=np.float64)
    actual = np.full_like(target, np.nan)
    position = np.full((count, 2), np.nan, dtype=np.float64)
    rotation = np.full((count, 2), np.nan, dtype=np.float64)
    posture = np.full((count, 2), np.nan, dtype=np.float64)
    solver_success = np.zeros((count, 2), dtype=bool)
    solver_status = np.full((count, 2), -999, dtype=np.int32)
    solver_nfev = np.zeros((count, 2), dtype=np.int32)
    chosen_seed = np.full((count, 2), -1, dtype=np.int32)
    acceleration_bound = np.zeros((count, 2), dtype=bool)
    reset_reason = np.full((count, 2), "UNSET", dtype="<U32")

    for side in range(2):
        previous = None
        previous_previous = None
        for frame in range(count):
            if not valid[frame, side]:
                previous = None
                previous_previous = None
                reset_reason[frame, side] = "INVALID_INPUT"
                continue
            continuous = bool(frame > 0 and edge[frame] and valid[frame - 1, side])
            if not continuous or previous is None:
                lo = lower[side].copy()
                hi = upper[side].copy()
                seeds = [reference[side], neutral[side], 0.5 * (lo + hi)]
                reset_reason[frame, side] = "SEGMENT_START_GLOBAL_MULTISEED"
                prior = None
            else:
                lo, hi, accel_used = bounded_joint_limits(
                    lower[side], upper[side], previous, previous_previous,
                    step_limit=step_limit, acceleration_limit=acceleration_limit,
                )
                acceleration_bound[frame, side] = accel_used
                seeds = [
                    np.clip(previous, lo, hi),
                    np.clip(reference[side], lo, hi),
                    0.5 * (lo + hi),
                ]
                reset_reason[frame, side] = "CONTIGUOUS_BOUNDED_MULTISEED"
                prior = previous
            target_tool = target[frame, side] @ np.linalg.inv(tool_mounts[side])
            attempts = []
            for seed_index, seed in enumerate(seeds):
                seed = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)

                def residual(q):
                    pose = arm._pose_residual(arm._tool_fk(assets, side, q), target_tool)
                    margin = float(inward_margin_fn(assets, mounts, neutral, side, q))
                    inward = np.asarray([min(margin - minimum_inward_m, 0.0) / 0.001])
                    continuity = (
                        np.empty(0, dtype=np.float64)
                        if prior is None
                        else float(continuity_weight) * (q - prior)
                    )
                    return np.concatenate((pose, inward, continuity))

                solved = least_squares(
                    residual,
                    seed,
                    bounds=(lo, hi),
                    max_nfev=int(max_nfev),
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                pose, pos_mm, rot_deg = _pose_error(
                    assets, tool_mounts, side, solved.x, target[frame, side]
                )
                margin = float(inward_margin_fn(assets, mounts, neutral, side, solved.x))
                failed = not (
                    solved.success
                    and pos_mm <= 20.0
                    and rot_deg <= 15.0
                    and margin >= minimum_inward_m - 1e-9
                )
                score = (
                    failed,
                    max(0.0, minimum_inward_m - margin),
                    max(0.0, pos_mm - 20.0) + max(0.0, rot_deg - 15.0),
                    pos_mm + rot_deg,
                )
                attempts.append((score, solved, pose, pos_mm, rot_deg, margin, seed_index))
            score, selected, selected_pose, pos_mm, rot_deg, margin, seed_index = min(
                attempts, key=lambda item: item[0]
            )
            q_arm[frame, side] = selected.x
            actual[frame, side] = selected_pose
            position[frame, side] = pos_mm
            rotation[frame, side] = rot_deg
            posture[frame, side] = margin
            solver_success[frame, side] = bool(selected.success)
            solver_status[frame, side] = int(selected.status)
            solver_nfev[frame, side] = int(selected.nfev)
            chosen_seed[frame, side] = int(seed_index)
            previous_previous, previous = previous, selected.x.copy()

    tolerance = (
        valid
        & solver_success
        & (position <= 20.0)
        & (rotation <= 15.0)
        & (posture >= minimum_inward_m - 1e-9)
    )
    result.update(
        q_arm=q_arm,
        T_actual_root=actual,
        position_residual_mm=position,
        rotation_residual_deg=rotation,
        posture_margin_m=posture,
        posture_minimum_m=np.asarray(minimum_inward_m),
        solver_success=solver_success,
        solver_status=solver_status,
        solver_nfev=solver_nfev,
        multiseed_chosen_index=chosen_seed,
        acceleration_bound_applied=acceleration_bound,
        reset_reason=reset_reason,
        tolerance_pass=tolerance,
        quality_pass=tolerance.copy(),
        continuity_weight=np.asarray(continuity_weight),
        joint_step_limit_rad=np.asarray(step_limit),
        joint_acceleration_limit_rad=np.asarray(acceleration_limit),
        solver_policy=np.asarray("EXACT78_STYLE_BOUNDED_MULTISEED_INWARD"),
    )
    times = (np.asarray(source["timestamp_ns"]) - source["timestamp_ns"][0]).astype(float) * 1e-9
    derivatives = motion_derivatives(q_arm, valid, times, edge)
    for name, values in derivatives.items():
        result[f"arm_{name}"] = values
    return result
