"""HuRo-derived, root-relative KaiHand retargeting.

This module ports the *objective structure* used by HuRo Stage 8 (local
keypoint-direction alignment, a rest prior and temporal joint smoothness) to
the two pinned hand-only KaiHand URDFs.  It intentionally does not claim to be
the official HuRo whole-robot Stage 8 implementation: there is no arm,
camera-link, EEF frame or world-to-base solve in this adapter.

Missing human observations remain missing.  In particular, the official HuRo
``fill_missing`` behaviour is not used because it would turn inferred frames
into apparently observed training labels in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from chaoyang.pipeline.robot_renderer_cycles import (
    UrdfModel,
    forward_kinematics,
    parse_urdf,
)


SCHEMA_VERSION = "HURO_DERIVED_HAND_ONLY_RESULT_V1"
METHOD_ID = "HURO_DERIVED_HAND_ONLY_RETARGET"
AUTHORITY = "DEVELOPMENT_ONLY_NOT_OFFICIAL_FULL_HURO_REPRODUCTION"
HUMAN_TO_PHYSICAL = np.asarray([1, 0], dtype=np.int32)
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class HuroHandOnlyError(RuntimeError):
    """A hand-only input or solver invariant failed closed."""


@dataclass(frozen=True)
class HandModel:
    side: str
    urdf_path: Path
    model: UrdfModel
    joint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    keypoint_links: tuple[str, ...]


@dataclass(frozen=True)
class SolveDiagnostics:
    success: bool
    evaluations: int
    cost: float
    local_direction_rms: float
    tip_position_rms_m: float
    temporal_delta_rms_rad: float


def _moving_joints(model: UrdfModel):
    return tuple(joint for joint in model.joints if joint.joint_type != "fixed")


def _keypoint_links(side: str) -> tuple[str, ...]:
    prefix = "hand_l" if side == "left" else "hand_r"
    links = [f"{prefix}_base_link"]
    # KaiHand has six thumb links for four MANO bones.  The spaced mapping
    # preserves the base, intermediate and distal semantics without inventing
    # extra MANO observations.
    links.extend(
        [
            f"{prefix}_thumb_link1",
            f"{prefix}_thumb_link2",
            f"{prefix}_thumb_link4",
            f"{prefix}_thumb_link6",
        ]
    )
    for finger in FINGER_NAMES[1:]:
        links.extend(f"{prefix}_{finger}_link{index}" for index in range(1, 5))
    return tuple(links)


def load_hand_model(path: Path, side: str) -> HandModel:
    if side not in {"left", "right"}:
        raise HuroHandOnlyError(f"unsupported physical side: {side}")
    model = parse_urdf(path)
    moving = _moving_joints(model)
    if len(moving) != 22:
        raise HuroHandOnlyError(f"KaiHand must have 22 moving joints, got {len(moving)}")
    lower = np.asarray(
        [joint.lower if joint.lower is not None else -np.pi for joint in moving],
        dtype=np.float64,
    )
    upper = np.asarray(
        [joint.upper if joint.upper is not None else np.pi for joint in moving],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise HuroHandOnlyError("KaiHand limits must be finite")
    if np.any(lower >= upper):
        raise HuroHandOnlyError("KaiHand limits are invalid")
    links = _keypoint_links(side)
    missing = sorted(set(links) - set(model.links))
    if missing:
        raise HuroHandOnlyError(f"KaiHand keypoint links missing: {missing}")
    return HandModel(
        side=side,
        urdf_path=path.resolve(),
        model=model,
        joint_names=tuple(joint.name for joint in moving),
        lower=lower,
        upper=upper,
        keypoint_links=links,
    )


def keypoints_from_q(hand: HandModel, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (22,) or not np.all(np.isfinite(q)):
        raise HuroHandOnlyError("q must be finite with shape (22,)")
    transforms = forward_kinematics(
        hand.model, dict(zip(hand.joint_names, q.tolist(), strict=True))
    )
    points = np.asarray([transforms[name][:3, 3] for name in hand.keypoint_links])
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise HuroHandOnlyError("FK keypoint extraction failed")
    return points


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _local_directions(points: np.ndarray) -> np.ndarray:
    directions: list[np.ndarray] = []
    for chain in FINGER_CHAINS:
        previous = 0
        for current in chain:
            directions.append(_unit(points[current] - points[previous]))
            previous = current
    return np.asarray(directions, dtype=np.float64)


def _root_relative_scaled_targets(target: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (21, 3) or not np.all(np.isfinite(target)):
        raise HuroHandOnlyError("target must be finite with shape (21,3)")
    target_rel = target - target[0]
    neutral_rel = neutral - neutral[0]
    mcp = np.asarray([1, 5, 9, 13, 17], dtype=np.int64)
    target_scale = float(np.median(np.linalg.norm(target_rel[mcp], axis=1)))
    robot_scale = float(np.median(np.linalg.norm(neutral_rel[mcp], axis=1)))
    if target_scale < 1e-6 or robot_scale < 1e-6:
        raise HuroHandOnlyError("degenerate hand scale")
    return target_rel * (robot_scale / target_scale)


def solve_frame(
    hand: HandModel,
    target: np.ndarray,
    *,
    previous_q: np.ndarray | None = None,
    max_evaluations: int = 80,
) -> tuple[np.ndarray, np.ndarray, SolveDiagnostics]:
    neutral_q = 0.5 * (hand.lower + hand.upper)
    neutral_points = keypoints_from_q(hand, neutral_q)
    target_scaled = _root_relative_scaled_targets(target, neutral_points)
    target_dirs = _local_directions(target_scaled)
    initial = neutral_q if previous_q is None else np.clip(previous_q, hand.lower, hand.upper)
    previous = initial.copy()
    tip_indices = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)

    # These are the same objective families as HuRo Stage 8, specialized to a
    # fixed hand root.  Numeric values remain development configuration, not a
    # paper reproduction or physical accuracy threshold.
    direction_weight = 20.0
    tip_weight = 8.0
    temporal_weight = np.sqrt(10.0)
    rest_weight = np.sqrt(0.15)

    def residual(q: np.ndarray) -> np.ndarray:
        points = keypoints_from_q(hand, q)
        points_rel = points - points[0]
        local = (_local_directions(points_rel) - target_dirs).reshape(-1)
        tip = (points_rel[tip_indices] - target_scaled[tip_indices]).reshape(-1)
        temporal = q - previous
        rest = q - neutral_q
        return np.concatenate(
            [
                direction_weight * local,
                tip_weight * tip,
                temporal_weight * temporal,
                rest_weight * rest,
            ]
        )

    solved = least_squares(
        residual,
        initial,
        bounds=(hand.lower, hand.upper),
        max_nfev=max_evaluations,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    q = np.asarray(solved.x, dtype=np.float64)
    points = keypoints_from_q(hand, q)
    points_rel = points - points[0]
    local_rms = float(np.sqrt(np.mean((_local_directions(points_rel) - target_dirs) ** 2)))
    tip_rms = float(
        np.sqrt(np.mean((points_rel[tip_indices] - target_scaled[tip_indices]) ** 2))
    )
    temporal_rms = float(np.sqrt(np.mean((q - previous) ** 2)))
    diagnostics = SolveDiagnostics(
        success=bool(solved.success and np.all(np.isfinite(q))),
        evaluations=int(solved.nfev),
        cost=float(solved.cost),
        local_direction_rms=local_rms,
        tip_position_rms_m=tip_rms,
        temporal_delta_rms_rad=temporal_rms,
    )
    return q, points, diagnostics


def solve_sequence(
    hands: tuple[HandModel, HandModel],
    joints_anatomical: np.ndarray,
    observed_anatomical: np.ndarray,
    *,
    max_evaluations: int = 80,
) -> dict[str, np.ndarray]:
    joints = np.asarray(joints_anatomical, dtype=np.float64)
    observed = np.asarray(observed_anatomical, dtype=bool)
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise HuroHandOnlyError("joints must have shape (2,T,21,3)")
    if observed.shape != joints.shape[:2]:
        raise HuroHandOnlyError("observed must have shape (2,T)")
    frame_count = joints.shape[1]
    q22 = np.full((frame_count, 2, 22), np.nan, dtype=np.float64)
    fk21 = np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float64)
    valid = np.zeros((frame_count, 2), dtype=bool)
    success = np.zeros((frame_count, 2), dtype=bool)
    evaluations = np.zeros((frame_count, 2), dtype=np.int32)
    cost = np.full((frame_count, 2), np.nan, dtype=np.float64)
    local_rms = np.full((frame_count, 2), np.nan, dtype=np.float64)
    tip_rms_m = np.full((frame_count, 2), np.nan, dtype=np.float64)
    temporal_delta = np.full((frame_count, 2), np.nan, dtype=np.float64)
    previous_by_physical: dict[int, np.ndarray] = {}
    for frame in range(frame_count):
        for anatomical in range(2):
            if not observed[anatomical, frame]:
                continue
            physical = int(HUMAN_TO_PHYSICAL[anatomical])
            target = joints[anatomical, frame]
            if not np.all(np.isfinite(target)):
                continue
            q, points, diag = solve_frame(
                hands[physical],
                target,
                previous_q=previous_by_physical.get(physical),
                max_evaluations=max_evaluations,
            )
            q22[frame, physical] = q
            fk21[frame, physical] = points
            valid[frame, physical] = diag.success
            success[frame, physical] = diag.success
            evaluations[frame, physical] = diag.evaluations
            cost[frame, physical] = diag.cost
            local_rms[frame, physical] = diag.local_direction_rms
            tip_rms_m[frame, physical] = diag.tip_position_rms_m
            temporal_delta[frame, physical] = diag.temporal_delta_rms_rad
            if diag.success:
                previous_by_physical[physical] = q
    return {
        "q22": q22,
        "fk21_root_relative": fk21,
        "valid": valid,
        "solver_success": success,
        "solver_evaluations": evaluations,
        "solver_cost": cost,
        "local_direction_rms": local_rms,
        "tip_position_rms_m": tip_rms_m,
        "temporal_delta_rms_rad": temporal_delta,
        "human_to_physical": HUMAN_TO_PHYSICAL.copy(),
        "joint_names": np.asarray([hand.joint_names for hand in hands]),
        "method_id": np.asarray(METHOD_ID),
        "authority": np.asarray(AUTHORITY),
    }

