"""Deterministic development-only palm-frame adapter; no wrist calibration claim."""
from __future__ import annotations

import numpy as np

from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    HUMAN_TO_PHYSICAL, HuroHandOnlyError, keypoints_from_q, solve_frame,
    _local_directions, _root_relative_scaled_targets,
)

METHOD_ID = "HURO_DERIVED_HAND_ONLY_PALM_FRAME_V2"


def palm_basis(points):
    """Columns: index-minus-pinky, orthogonal middle ray, proper cross product.

    This is an inferred hand-shape coordinate convention, not a measured wrist
    orientation. The fixed MANO21/Kai link mapping is inherited from V1.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise HuroHandOnlyError("palm requires finite MANO21")
    x = p[5] - p[17]
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        raise HuroHandOnlyError("degenerate palm width")
    x = x / nx
    y = p[9] - p[0]
    y = y - x * np.dot(x, y)
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        raise HuroHandOnlyError("degenerate palm ray")
    y = y / ny
    basis = np.column_stack((x, y, np.cross(x, y)))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10) or np.linalg.det(basis) < .999999:
        raise HuroHandOnlyError("invalid proper palm frame")
    return basis


def canonical_target(points, neutral):
    """Remove camera rigid rotation before comparing with robot-local FK."""
    source_basis, robot_basis = palm_basis(points), palm_basis(neutral)
    relative = np.asarray(points, dtype=np.float64) - points[0]
    mapped = relative @ source_basis @ robot_basis.T
    return _root_relative_scaled_targets(mapped, neutral), robot_basis @ source_basis.T


def solve_aligned_sequence(hands, joints, observed, frame_ids, *, max_evaluations=80, progress=None):
    joints = np.asarray(joints, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    frames = np.asarray(frame_ids)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise HuroHandOnlyError("expected anatomical (2,T,21,3)")
    count = joints.shape[1]
    if observed.shape != (2, count) or frames.shape != (count,) or np.any(np.diff(frames) <= 0):
        raise HuroHandOnlyError("invalid observed/frame axis")
    data = dict(q22=np.full((count, 2, 22), np.nan),
                fk21_root_relative=np.full((count, 2, 21, 3), np.nan),
                target21_robot_frame=np.full((count, 2, 21, 3), np.nan),
                inferred_camera_to_robot_rotation=np.full((count, 2, 3, 3), np.nan),
                valid=np.zeros((count, 2), bool), solver_success=np.zeros((count, 2), bool),
                solver_evaluations=np.zeros((count, 2), np.int32),
                solver_cost=np.full((count, 2), np.nan),
                target_valid=np.zeros((count, 2), bool),
                failure_reason=np.full((count, 2), "UNOBSERVED", dtype="U80"),
                source_observed_physical=observed[::-1].T.copy(),
                source_frame_id=frames.copy(), human_to_physical=HUMAN_TO_PHYSICAL.copy(),
                joint_names=np.asarray([h.joint_names for h in hands]),
                method_id=np.asarray(METHOD_ID),
                authority=np.asarray("DEVELOPMENT_ONLY_NOT_OFFICIAL_FULL_HURO_REPRODUCTION"))
    neutral = [keypoints_from_q(h, .5 * (h.lower + h.upper)) for h in hands]
    previous = [None, None]
    for t in range(count):
        if t and frames[t] != frames[t-1] + 1:
            previous = [None, None]
        for anatomical, physical in enumerate(HUMAN_TO_PHYSICAL):
            physical = int(physical)
            if not observed[anatomical, t]:
                previous[physical] = None
                continue
            try:
                target, rotation = canonical_target(joints[anatomical, t], neutral[physical])
                data['target21_robot_frame'][t, physical] = target
                data['inferred_camera_to_robot_rotation'][t, physical] = rotation
                data['target_valid'][t, physical] = True
                q, fk, diag = solve_frame(hands[physical], target,
                                          previous_q=previous[physical], max_evaluations=max_evaluations)
                data['q22'][t, physical] = q
                data['fk21_root_relative'][t, physical] = fk - fk[0]
                data['solver_success'][t, physical] = diag.success
                data['valid'][t, physical] = diag.success
                data['solver_evaluations'][t, physical] = diag.evaluations
                data['solver_cost'][t, physical] = diag.cost
                data['failure_reason'][t, physical] = "NONE" if diag.success else "SOLVER_NOT_CONVERGED"
                previous[physical] = q.copy() if diag.success else None
            except HuroHandOnlyError as exc:
                data['failure_reason'][t, physical] = str(exc)[:80]
                previous[physical] = None
        if progress and (t % 25 == 0 or t == count-1):
            progress(t+1, count)
    return data


def evaluate_q(hands, q22, valid, targets, target_valid, collision_checker=None):
    q22 = np.asarray(q22, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    shape = valid.shape
    if q22.shape != (*shape, 22) or targets.shape != (*shape, 21, 3):
        raise HuroHandOnlyError("evaluation shape mismatch")
    out = {key: np.full(shape, np.nan) for key in (
        'tip_rms_mm', 'direction_rms', 'limit_violation_rad', 'collision_count', 'penetration_m')}
    out['evaluated'] = np.zeros(shape, bool)
    out['collision_known'] = np.zeros(shape, bool)
    for t, side in np.argwhere(valid):
        q = q22[t, side]
        if not np.isfinite(q).all():
            raise HuroHandOnlyError("valid nonfinite q")
        hand = hands[side]
        out['limit_violation_rad'][t, side] = max(0., float(np.max(hand.lower-q)), float(np.max(q-hand.upper)))
        if collision_checker is not None:
            c = collision_checker.check(int(side), q)
            out['collision_known'][t, side] = c.known
            out['collision_count'][t, side] = c.illegal_contact_count
            out['penetration_m'][t, side] = c.max_penetration_m
        if not target_valid[t, side]:
            continue
        fk = keypoints_from_q(hand, q)
        fk = fk - fk[0]
        target = targets[t, side]
        out['tip_rms_mm'][t, side] = 1000 * np.sqrt(np.mean((fk[[4,8,12,16,20]]-target[[4,8,12,16,20]])**2))
        out['direction_rms'][t, side] = np.sqrt(np.mean((_local_directions(fk)-_local_directions(target))**2))
        out['evaluated'][t, side] = True
    return out


def temporal_metrics(q22, valid, times, frame_ids):
    """Actual-dt differences only on uninterrupted valid edges; never bridge gaps."""
    times = np.asarray(times, float)
    frame_ids = np.asarray(frame_ids)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise HuroHandOnlyError("nonmonotonic time")
    dt = np.diff(times)
    contiguous = (np.diff(frame_ids) == 1) & (dt <= 2.5 * np.median(dt))
    edge = valid[1:] & valid[:-1] & contiguous[:, None]
    velocity = np.full_like(q22, np.nan)
    velocity[1:] = np.where(edge[..., None], np.diff(q22, axis=0)/dt[:, None, None], np.nan)
    acceleration = np.full_like(q22, np.nan)
    midpoint_dt = .5 * (dt[1:] + dt[:-1])
    adjacent = edge[1:] & edge[:-1]
    acceleration[2:] = np.where(adjacent[..., None], np.diff(velocity[1:], axis=0)/midpoint_dt[:, None, None], np.nan)
    jerk = np.full_like(q22, np.nan)
    velocity_times = .5 * (times[1:] + times[:-1])
    acceleration_times = .5 * (velocity_times[1:] + velocity_times[:-1])
    if len(times) > 3:
        jerk[3:] = np.diff(acceleration[2:], axis=0) / np.diff(acceleration_times)[:,None,None]
    return dict(velocity_rad_s=velocity, acceleration_rad_s2=acceleration,
                jerk_rad_s3=jerk, edge_valid=np.vstack([np.zeros((1,2),bool),edge]))
