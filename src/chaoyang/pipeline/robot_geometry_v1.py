"""V7.1 Robot geometry contracts independent of Clean and metric contact.

This module deliberately owns neither RGB compositing nor physical authority.
It validates one-frame solve budgets, a single scene z-buffer for both robots
and all object instances, and collision evidence that is distinct from visual
self-occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "ROBOT_GEOMETRY_V1"


class RobotGeometryContractError(ValueError):
    pass


class FrameMode(str, Enum):
    NON_CONTACT_FRAME = "NON_CONTACT_FRAME"
    CONTACT_WINDOW = "CONTACT_WINDOW"
    HARD_CONTACT_FRAME = "HARD_CONTACT_FRAME"


STAGE_BUDGET_S: dict[str, float] = {
    "WRIST_ARM_IK": 4.0,
    "HUMAN_POSE_HAND_RETARGET": 5.0,
    "CONTACT_FINGER_REFINEMENT": 8.0,
    "COLLISION_CLEANUP": 6.0,
    "TEMPORAL_REFINEMENT": 5.0,
    "FINAL_AUDIT": 2.0,
}
FRAME_TOTAL_BUDGET_S = 30.0


def selected_object_world_for_robot(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Accept only an explicit rectified-to-selected Object6D adapter.

    Legacy ``T_object_to_world`` is deliberately rejected because the former
    producer multiplied a rectified-depth-camera pose by selected-camera c2w.
    """

    required = {
        "T_object_to_selected_camera",
        "T_object_to_world_corrected",
        "source_coordinate_domain",
        "target_coordinate_domain",
        "control_ground_truth",
    }
    missing = required - set(arrays)
    if missing:
        raise RobotGeometryContractError(
            f"Object6D selected-camera adapter keys missing: {sorted(missing)}"
        )
    source = str(np.asarray(arrays["source_coordinate_domain"]).item())
    target = str(np.asarray(arrays["target_coordinate_domain"]).item())
    if source != "STEREO_RECTIFIED_DEPTH_CAMERA" or target != "SELECTED_LEFT_RGB_CAMERA":
        raise RobotGeometryContractError("Object6D coordinate-domain identity mismatch")
    if bool(np.asarray(arrays["control_ground_truth"]).item()):
        raise RobotGeometryContractError("development Object6D cannot be control ground truth")
    world = np.asarray(arrays["T_object_to_world_corrected"], dtype=np.float64)
    selected = np.asarray(arrays["T_object_to_selected_camera"], dtype=np.float64)
    if world.ndim != 3 or world.shape[1:] != (4, 4) or selected.shape != world.shape:
        raise RobotGeometryContractError("corrected Object6D pose shape mismatch")
    if not np.isfinite(world).all() or not np.isfinite(selected).all():
        raise RobotGeometryContractError("corrected Object6D poses contain non-finite values")
    return world


@dataclass(frozen=True)
class SceneLayer:
    instance_id: str
    link_id: str
    part_id: str
    triangle_id: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class UnifiedZBuffer:
    depth_m: np.ndarray
    instance_id: np.ndarray
    part_id: np.ndarray
    link_id: np.ndarray
    triangle_id: np.ndarray
    depth_tie_mask: np.ndarray


@dataclass(frozen=True)
class UnifiedZBufferAudit:
    covered_pixel_count: int
    depth_tie_ratio: float
    invalid_triangle_ratio: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CollisionObservation:
    link_a: str
    link_b: str
    minimum_distance_m: float
    penetration_depth_m: float
    penetration_volume_m3: float


@dataclass(frozen=True)
class CollisionAudit:
    self_collision_count: int
    minimum_non_adjacent_link_distance_m: float | None
    max_penetration_depth_m: float
    penetration_volume_m3: float
    failures: tuple[str, ...]


def validate_frame_budget(
    mode: FrameMode,
    stage_budget_s: Mapping[str, float] = STAGE_BUDGET_S,
) -> None:
    if set(stage_budget_s) != set(STAGE_BUDGET_S):
        raise RobotGeometryContractError("all six Robot geometry stages are required")
    values = [float(stage_budget_s[key]) for key in STAGE_BUDGET_S]
    if any(not np.isfinite(v) or v < 0.0 for v in values):
        raise RobotGeometryContractError("stage budgets must be finite and non-negative")
    if sum(values) > FRAME_TOTAL_BUDGET_S + 1e-9:
        raise RobotGeometryContractError("per-frame total budget exceeds 30 seconds")
    if mode is FrameMode.NON_CONTACT_FRAME and stage_budget_s["CONTACT_FINGER_REFINEMENT"] != 0:
        raise RobotGeometryContractError("non-contact frame must not run contact refinement")


def budget_for_mode(mode: FrameMode) -> dict[str, float]:
    if mode is FrameMode.NON_CONTACT_FRAME:
        # Fast pose-only path, total 3 seconds.
        return {
            "WRIST_ARM_IK": 1.0,
            "HUMAN_POSE_HAND_RETARGET": 1.0,
            "CONTACT_FINGER_REFINEMENT": 0.0,
            "COLLISION_CLEANUP": 0.5,
            "TEMPORAL_REFINEMENT": 0.3,
            "FINAL_AUDIT": 0.2,
        }
    if mode is FrameMode.CONTACT_WINDOW:
        # Ordinary contact target, total 8 seconds.
        return {
            "WRIST_ARM_IK": 1.0,
            "HUMAN_POSE_HAND_RETARGET": 1.5,
            "CONTACT_FINGER_REFINEMENT": 2.5,
            "COLLISION_CLEANUP": 1.5,
            "TEMPORAL_REFINEMENT": 1.0,
            "FINAL_AUDIT": 0.5,
        }
    return dict(STAGE_BUDGET_S)


def rasterize_unified_zbuffer(
    layers: Iterable[SceneLayer], *, tie_epsilon_m: float = 1e-6
) -> UnifiedZBuffer:
    # Sorting makes the winner at an exact depth tie independent of caller or
    # renderer draw order.  The tie remains explicitly visible to QA.
    layers = tuple(sorted(layers, key=lambda item: (item.instance_id, item.link_id, item.part_id)))
    if not layers:
        raise RobotGeometryContractError("unified z-buffer requires scene layers")
    shape = np.asarray(layers[0].depth_m).shape
    if len(shape) != 2:
        raise RobotGeometryContractError("scene depth must be HxW")
    depth = np.full(shape, np.inf, dtype=np.float64)
    instance = np.full(shape, "", dtype=object)
    part = np.full(shape, "", dtype=object)
    link = np.full(shape, "", dtype=object)
    triangle = np.full(shape, -1, dtype=np.int64)
    depth_tie = np.zeros(shape, dtype=np.bool_)
    if not np.isfinite(tie_epsilon_m) or tie_epsilon_m < 0.0:
        raise RobotGeometryContractError("tie_epsilon_m must be finite and non-negative")
    for layer in layers:
        z = np.asarray(layer.depth_m, dtype=np.float64)
        valid = np.asarray(layer.valid)
        tri = np.asarray(layer.triangle_id)
        if z.shape != shape or valid.shape != shape or tri.shape != shape or valid.dtype != np.bool_:
            raise RobotGeometryContractError("all scene layer arrays must share HxW shape")
        if not layer.instance_id or not layer.part_id or not layer.link_id:
            raise RobotGeometryContractError("instance/part/link identity is required")
        eligible = valid & np.isfinite(z) & (z > 0.0)
        tied = eligible & np.isfinite(depth) & (np.abs(z - depth) <= tie_epsilon_m)
        depth_tie[tied] = True
        choose = eligible & (z < depth - tie_epsilon_m)
        depth[choose] = z[choose]
        instance[choose] = layer.instance_id
        part[choose] = layer.part_id
        link[choose] = layer.link_id
        triangle[choose] = tri[choose]
    depth[~np.isfinite(depth)] = np.nan
    return UnifiedZBuffer(depth, instance, part, link, triangle, depth_tie)


def audit_unified_zbuffer(
    zbuffer: UnifiedZBuffer,
    *,
    max_depth_tie_ratio: float = 0.01,
    max_invalid_triangle_ratio: float = 0.0,
) -> UnifiedZBufferAudit:
    """Audit renderer provenance without treating it as physical truth."""

    covered = np.isfinite(np.asarray(zbuffer.depth_m, dtype=np.float64))
    shape = covered.shape
    arrays = (
        np.asarray(zbuffer.instance_id),
        np.asarray(zbuffer.part_id),
        np.asarray(zbuffer.link_id),
        np.asarray(zbuffer.triangle_id),
        np.asarray(zbuffer.depth_tie_mask),
    )
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise RobotGeometryContractError("z-buffer audit arrays must share HxW shape")
    if arrays[-1].dtype != np.bool_:
        raise RobotGeometryContractError("depth_tie_mask must be bool")
    denominator = int(np.count_nonzero(covered))
    ties = int(np.count_nonzero(arrays[-1] & covered))
    invalid_triangles = int(np.count_nonzero(covered & (arrays[3] < 0)))
    tie_ratio = ties / denominator if denominator else 0.0
    invalid_ratio = invalid_triangles / denominator if denominator else 0.0
    failures: list[str] = []
    if tie_ratio > max_depth_tie_ratio:
        failures.append("DEPTH_TIE_RATIO")
    if invalid_ratio > max_invalid_triangle_ratio:
        failures.append("INVALID_TRIANGLE_RATIO")
    if np.any(covered & ((arrays[0] == "") | (arrays[1] == "") | (arrays[2] == ""))):
        failures.append("MISSING_PIXEL_IDENTITY")
    return UnifiedZBufferAudit(
        denominator, tie_ratio, invalid_ratio, not failures, tuple(failures)
    )


def audit_self_collision(
    observations: Iterable[CollisionObservation],
    *,
    adjacent_allowlist: set[frozenset[str]],
    penetration_tolerance_m: float = 0.0,
) -> CollisionAudit:
    illegal: list[CollisionObservation] = []
    distances: list[float] = []
    for row in observations:
        if not all(np.isfinite(v) for v in (
            row.minimum_distance_m,
            row.penetration_depth_m,
            row.penetration_volume_m3,
        )):
            raise RobotGeometryContractError("collision evidence must be finite")
        if row.link_a == row.link_b:
            raise RobotGeometryContractError("collision pair must contain distinct links")
        pair = frozenset((row.link_a, row.link_b))
        if pair in adjacent_allowlist:
            continue
        distances.append(row.minimum_distance_m)
        if row.penetration_depth_m > penetration_tolerance_m:
            illegal.append(row)
    max_depth = max((r.penetration_depth_m for r in illegal), default=0.0)
    volume = sum((r.penetration_volume_m3 for r in illegal), start=0.0)
    failures = ("SELF_INTERSECTION",) if illegal else ()
    return CollisionAudit(
        len(illegal), min(distances) if distances else None, max_depth, volume, failures
    )
