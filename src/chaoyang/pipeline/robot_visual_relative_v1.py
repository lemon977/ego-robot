"""Development-only relative-motion Robot Visual conversion for 0915.

This module intentionally does not consume or invent a measured TCP, a
tool-to-KaiHand installation transform, or a camera/world-to-base
calibration.  It maps each observed HaWoR wrist trajectory into a bounded
virtual robot workspace relative to the first observed wrist, then solves the
actual pinned Tianji arm URDF.  KaiHand joints are retargeted from MANO finger
flexion while respecting the pinned URDF limits.

The result is useful for conversion-loss accounting and visual review only.
It is never a control trajectory or physical deployment authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chaoyang.pipeline import robot_scene_state_cpu as arm_solver
from chaoyang.pipeline import robot_wrist_kai_adapter as wrist_adapter
from chaoyang.pipeline.robot_renderer_eevee_fullchain import (
    PinnedRobotAssets,
    SIDES,
    load_pinned_robot_assets,
)


MANO_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
HAND_GROUPS = (
    tuple(range(0, 6)),
    tuple(range(6, 10)),
    tuple(range(10, 14)),
    tuple(range(14, 18)),
    tuple(range(18, 22)),
)
SCALE_CANDIDATES = (0.55, 0.75, 0.95)
YAW_CANDIDATES_DEG = (-15.0, 0.0, 15.0)


class RobotVisualRelativeError(ValueError):
    pass


@dataclass(frozen=True)
class HandLimits:
    lower: np.ndarray
    upper: np.ndarray
    neutral: np.ndarray


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _proper_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    delta = np.linalg.inv(target) @ actual
    return float(np.degrees(np.linalg.norm(arm_solver._rotation_vector(delta[:3, :3]))))


def palm_transforms(joints_world: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Build proper MANO palm transforms without filling missing observations."""

    joints = np.asarray(joints_world, dtype=np.float64)
    valid = np.asarray(observed, dtype=bool)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise RobotVisualRelativeError("joints_world must be [2,T,21,3]")
    if valid.shape != joints.shape[:2]:
        raise RobotVisualRelativeError("observed shape mismatch")
    transforms = np.full((2, joints.shape[1], 4, 4), np.nan, dtype=np.float64)
    for side in range(2):
        for frame in np.flatnonzero(valid[side]):
            points = joints[side, frame]
            if not np.isfinite(points).all():
                continue
            rotation = wrist_adapter.final_v3_mano_palm_basis(
                points, handedness=SIDES[side],
            )
            transforms[side, frame] = _proper_transform(rotation, points[0])
    return transforms


def relative_tool_targets(
    palm: np.ndarray,
    observed: np.ndarray,
    neutral_tools: np.ndarray,
    *,
    scale: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map hand motion into a virtual robot workspace with no calibration claim."""

    transforms = np.asarray(palm, dtype=np.float64)
    valid = np.asarray(observed, dtype=bool).copy()
    tools = np.asarray(neutral_tools, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[:2] != valid.shape or transforms.shape[2:] != (4, 4):
        raise RobotVisualRelativeError("palm/observed geometry mismatch")
    if tools.shape != (2, 4, 4):
        raise RobotVisualRelativeError("neutral_tools must be [2,4,4]")
    if not np.isfinite(scale) or not 0.25 <= scale <= 1.25:
        raise RobotVisualRelativeError("relative motion scale outside bounded range")
    yaw = _rotation_z(yaw_deg)
    targets = np.full((transforms.shape[1], 2, 4, 4), np.nan, dtype=np.float64)
    for side in range(2):
        finite = np.isfinite(transforms[side]).all(axis=(1, 2))
        valid[side] &= finite
        candidates = np.flatnonzero(valid[side])
        if candidates.size == 0:
            valid[side] = False
            continue
        anchor = transforms[side, int(candidates[0])]
        alignment = tools[side, :3, :3] @ yaw @ anchor[:3, :3].T
        for frame in candidates:
            rotation = alignment @ transforms[side, frame, :3, :3]
            translation = tools[side, :3, 3] + scale * alignment @ (
                transforms[side, frame, :3, 3] - anchor[:3, 3]
            )
            targets[frame, side] = _proper_transform(rotation, translation)
    return targets, valid.T


def hand_limits(assets: PinnedRobotAssets) -> tuple[HandLimits, HandLimits]:
    values: list[HandLimits] = []
    for model in (assets.left_hand, assets.right_hand):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
        upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
        if lower.shape != (22,) or upper.shape != (22,):
            raise RobotVisualRelativeError("KaiHand moving-joint identity drift")
        values.append(HandLimits(lower, upper, 0.5 * (lower + upper)))
    return values[0], values[1]


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))


def mano_flexion(points: np.ndarray) -> np.ndarray:
    """Return three bounded bend angles for each MANO finger."""

    value = np.asarray(points, dtype=np.float64)
    if value.shape != (21, 3) or not np.isfinite(value).all():
        raise RobotVisualRelativeError("one finite MANO21 frame is required")
    result = np.zeros((5, 3), dtype=np.float64)
    for finger, chain in enumerate(MANO_CHAINS):
        bones = np.diff(value[np.asarray(chain)], axis=0)
        result[finger] = [_angle(bones[index], bones[index + 1]) for index in range(3)]
    return result


def retarget_kaihand_frame(points: np.ndarray, limits: HandLimits) -> tuple[np.ndarray, float]:
    """Map MANO flexion to pinned KaiHand limits and report normalized loss."""

    flexion = mano_flexion(points)
    q = limits.neutral.copy()
    errors: list[float] = []
    for finger, group in enumerate(HAND_GROUPS):
        indices = np.asarray(group, dtype=np.int64)
        lower, upper = limits.lower[indices], limits.upper[indices]
        if finger == 0:
            desired = np.asarray([
                q[indices[0]], flexion[finger, 0], flexion[finger, 1],
                0.5 * flexion[finger, 2], flexion[finger, 2], flexion[finger, 2],
            ])
        else:
            desired = np.asarray([
                q[indices[0]], flexion[finger, 0], flexion[finger, 1], flexion[finger, 2],
            ])
        selected = np.clip(desired, lower, upper)
        q[indices] = selected
        span = np.maximum(upper - lower, 1e-6)
        errors.extend((np.abs(selected - desired) / span).tolist())
    if not np.all((q >= limits.lower - 1e-9) & (q <= limits.upper + 1e-9)):
        raise RobotVisualRelativeError("KaiHand retarget escaped URDF limits")
    return q, float(np.mean(errors))


def _neutral_tools(assets: PinnedRobotAssets, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    neutral = 0.5 * (lower + upper)
    return np.stack([
        arm_solver._tool_fk(assets, side, neutral[side]) for side in range(2)
    ])


def _sample_frames(valid: np.ndarray, maximum: int = 8) -> np.ndarray:
    rows = np.flatnonzero(np.any(valid, axis=1))
    if rows.size <= maximum:
        return rows
    return rows[np.unique(np.linspace(0, rows.size - 1, maximum, dtype=np.int64))]


def select_workspace_candidate(
    *,
    assets: PinnedRobotAssets,
    palm: np.ndarray,
    observed: np.ndarray,
    solve: Callable[..., tuple[np.ndarray, int]] = arm_solver._solve_one_arm,
) -> dict[str, Any]:
    """Choose one finite session-static virtual mapping on a frozen grid."""

    lower, upper = arm_solver._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    neutral_tools = _neutral_tools(assets, lower, upper)
    rows: list[dict[str, Any]] = []
    for scale in SCALE_CANDIDATES:
        for yaw in YAW_CANDIDATES_DEG:
            targets, valid = relative_tool_targets(
                palm, observed, neutral_tools, scale=scale, yaw_deg=yaw,
            )
            position: list[float] = []
            rotation: list[float] = []
            evaluations = 0
            for frame in _sample_frames(valid):
                for side in range(2):
                    if not valid[frame, side]:
                        continue
                    q, count = solve(
                        assets, side=side, base=np.eye(4),
                        target_tool=targets[frame, side], initial_q=neutral[side],
                        lower=lower[side], upper=upper[side],
                    )
                    actual = arm_solver._tool_fk(assets, side, q)
                    position.append(float(np.linalg.norm(
                        actual[:3, 3] - targets[frame, side, :3, 3]
                    ) * 1000.0))
                    rotation.append(_rotation_error_deg(actual, targets[frame, side]))
                    evaluations += int(count)
            if not position:
                score = float("inf")
            else:
                score = float(np.percentile(position, 90) + 0.5 * np.percentile(rotation, 90))
            rows.append({
                "scale": scale, "yaw_deg": yaw, "score": score,
                "position_p90_mm": float(np.percentile(position, 90)) if position else None,
                "rotation_p90_deg": float(np.percentile(rotation, 90)) if rotation else None,
                "solver_evaluations": evaluations,
            })
    selected = min(rows, key=lambda row: (row["score"], row["scale"], abs(row["yaw_deg"])))
    if not np.isfinite(selected["score"]):
        raise RobotVisualRelativeError("no observed wrist can initialize Robot Visual")
    return {
        "policy": "FINITE_SESSION_STATIC_RELATIVE_WORKSPACE_GRID_V1",
        "selected": selected,
        "candidates": rows,
        "calibration_interpretation": "NONE_DEVELOPMENT_VIRTUAL_MAPPING_ONLY",
    }


def solve_robot_visual(
    *, project_root: Path, joints_world: np.ndarray, observed: np.ndarray,
    fps: float, assets: PinnedRobotAssets | None = None,
) -> dict[str, Any]:
    """Solve one complete development Robot Visual session."""

    if not np.isfinite(fps) or fps <= 0:
        raise RobotVisualRelativeError("fps must be positive")
    robot = assets or load_pinned_robot_assets(project_root)
    joints = np.asarray(joints_world, dtype=np.float64)
    source_valid = np.asarray(observed, dtype=bool)
    palm = palm_transforms(joints, source_valid)
    search = select_workspace_candidate(
        assets=robot, palm=palm, observed=source_valid,
    )
    lower, upper = arm_solver._arm_limits(robot)
    neutral = 0.5 * (lower + upper)
    targets, valid = relative_tool_targets(
        palm, source_valid, _neutral_tools(robot, lower, upper),
        scale=float(search["selected"]["scale"]),
        yaw_deg=float(search["selected"]["yaw_deg"]),
    )
    frames = joints.shape[1]
    q_arm = np.full((frames, 2, 7), np.nan, dtype=np.float64)
    q_hand = np.full((frames, 2, 22), np.nan, dtype=np.float64)
    arm_position = np.full((frames, 2), np.nan, dtype=np.float64)
    arm_rotation = np.full((frames, 2), np.nan, dtype=np.float64)
    hand_loss = np.full((frames, 2), np.nan, dtype=np.float64)
    limits = hand_limits(robot)
    evaluations = 0
    for side in range(2):
        previous = neutral[side].copy()
        for frame in range(frames):
            if not valid[frame, side]:
                continue
            positioned, count_position = arm_solver._solve_one_arm_position_only(
                robot, side=side, base=np.eye(4), target_tool=targets[frame, side],
                initial_q=previous, lower=lower[side], upper=upper[side],
            )
            solved, count_pose = arm_solver._solve_one_arm(
                robot, side=side, base=np.eye(4), target_tool=targets[frame, side],
                initial_q=positioned, lower=lower[side], upper=upper[side],
            )
            evaluations += int(count_position + count_pose)
            q_arm[frame, side] = solved
            previous = solved
            actual = arm_solver._tool_fk(robot, side, solved)
            arm_position[frame, side] = float(np.linalg.norm(
                actual[:3, 3] - targets[frame, side, :3, 3]
            ) * 1000.0)
            arm_rotation[frame, side] = _rotation_error_deg(actual, targets[frame, side])
            q_hand[frame, side], hand_loss[frame, side] = retarget_kaihand_frame(
                joints[side, frame], limits[side],
            )

    selected = valid & np.isfinite(q_arm).all(axis=2) & np.isfinite(q_hand).all(axis=2)
    joint_limit_violations = int(np.count_nonzero([
        np.any(q_arm[:, side][selected[:, side]] < lower[side] - 1e-9)
        or np.any(q_arm[:, side][selected[:, side]] > upper[side] + 1e-9)
        or np.any(q_hand[:, side][selected[:, side]] < limits[side].lower - 1e-9)
        or np.any(q_hand[:, side][selected[:, side]] > limits[side].upper + 1e-9)
        for side in range(2)
    ]))
    velocities: list[float] = []
    accelerations: list[float] = []
    tool_distances: list[float] = []
    for side in range(2):
        ids = np.flatnonzero(selected[:, side])
        for segment in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
            if len(segment) > 1:
                velocities.extend(np.abs(np.diff(q_arm[segment, side], axis=0)).ravel())
                velocities.extend(np.abs(np.diff(q_hand[segment, side], axis=0)).ravel())
            if len(segment) > 2:
                accelerations.extend(np.abs(np.diff(q_arm[segment, side], n=2, axis=0)).ravel())
                accelerations.extend(np.abs(np.diff(q_hand[segment, side], n=2, axis=0)).ravel())
    for frame in np.flatnonzero(np.all(selected, axis=1)):
        positions = [arm_solver._tool_fk(robot, side, q_arm[frame, side])[:3, 3] for side in range(2)]
        tool_distances.append(float(np.linalg.norm(positions[0] - positions[1])))
    valid_rows = int(selected.sum())
    source_rows = int(source_valid.sum())
    position_p95 = float(np.nanpercentile(arm_position, 95)) if valid_rows else None
    rotation_p95 = float(np.nanpercentile(arm_rotation, 95)) if valid_rows else None
    hand_mean = float(np.nanmean(hand_loss)) if valid_rows else None
    velocity_max = float(max(velocities, default=0.0))
    acceleration_max = float(max(accelerations, default=0.0))
    close_tool_fraction = float(np.mean(np.asarray(tool_distances) < 0.12)) if tool_distances else None
    gates = {
        "observed_rows_solved": valid_rows == source_rows and valid_rows > 0,
        "arm_position_p95_le_20mm": position_p95 is not None and position_p95 <= 20.0,
        "arm_rotation_p95_le_15deg": rotation_p95 is not None and rotation_p95 <= 15.0,
        "joint_limits": joint_limit_violations == 0,
        "velocity_rad_per_frame_le_0_12": velocity_max <= 0.12 + 1e-9,
        "acceleration_rad_per_frame2_le_0_06": acceleration_max <= 0.06 + 1e-9,
    }
    return {
        "status": "PASS" if all(gates.values()) else "REJECTED_QUALITY",
        "q_arm": q_arm,
        "q_hand": q_hand,
        "valid_side_frame": selected,
        "virtual_tool_targets": targets,
        "workspace_search": search,
        "metrics": {
            "arm_ik": None if position_p95 is None or rotation_p95 is None else position_p95 + rotation_p95,
            "kaihand_retarget": hand_mean,
            "collision": close_tool_fraction,
            "joint_limit": float(joint_limit_violations),
            "velocity": velocity_max,
            "acceleration": acceleration_max,
        },
        "diagnostics": {
            "source_observed_side_rows": source_rows,
            "solved_side_rows": valid_rows,
            "arm_position_p95_mm": position_p95,
            "arm_rotation_p95_deg": rotation_p95,
            "minimum_bilateral_tool_origin_distance_m": min(tool_distances, default=None),
            "collision_metric_scope": "TOOL_ORIGIN_PROXIMITY_PROXY_NOT_COLLISION_CERTIFICATION",
            "solver_evaluations": evaluations,
            "fps": float(fps),
        },
        "gates": gates,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
        "calibration_evidence": {
            "robot_tcp": "ABSENT",
            "tool_to_kaihand_root": "PRESENT_CANDIDATE_GEOMETRY",
            "camera_world_to_base": "ABSENT",
        },
        "claim_limit": (
            "Actual pinned-URDF IK and bounded relative-motion conversion only. "
            "The session-static workspace mapping is not TCP, mount, camera/world-to-base "
            "calibration, collision certification, control ground truth, or deployment authority."
        ),
    }
