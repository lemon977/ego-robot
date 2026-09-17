"""Fail-closed table-plane drift and re-anchoring primitives.

The plane convention is ``normal @ point + offset == 0`` in the frozen
world coordinate system.  Re-anchoring is deliberately limited to the scalar
offset: it cannot change the normal or conceal tangential camera/world drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np


class TablePlaneError(RuntimeError):
    """Raised when a plane operation would exceed its evidence."""


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset_m: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise TablePlaneError("plane normal must be one finite 3-vector")
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(self.offset_m) or abs(norm - 1.0) > 1e-8:
            raise TablePlaneError("plane normal must be unit length and offset finite")
        object.__setattr__(self, "normal", normal)


@dataclass(frozen=True)
class ReprojectionFrame:
    rgb: np.ndarray
    support: np.ndarray
    intrinsic: np.ndarray
    camera_to_world: np.ndarray

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        support = np.asarray(self.support)
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        c2w = np.asarray(self.camera_to_world, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise TablePlaneError("RGB must be HxWx3 uint8")
        if support.shape != rgb.shape[:2] or support.dtype != np.bool_:
            raise TablePlaneError("support must be a boolean image plane")
        if intrinsic.shape != (3, 3) or c2w.shape != (4, 4):
            raise TablePlaneError("camera matrices have invalid shape")
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(c2w)):
            raise TablePlaneError("camera matrices contain non-finite values")
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "camera_to_world", c2w)


def sign_consistent_axes(axes: np.ndarray) -> np.ndarray:
    result = np.asarray(axes, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise TablePlaneError("axes must be a non-empty Nx3 array")
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise TablePlaneError("axis contains a zero or non-finite vector")
    result /= norms[:, None]
    reference = result[0]
    result[np.einsum("ij,j->i", result, reference) < 0] *= -1
    return result


def cylinder_bottom_points(
    object_to_world: np.ndarray, height_m: float, *, axis_index: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.asarray(object_to_world, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise TablePlaneError("object transforms must be Nx4x4")
    if axis_index not in (0, 1, 2) or not np.isfinite(height_m) or height_m <= 0:
        raise TablePlaneError("cylinder geometry is invalid")
    axes = sign_consistent_axes(transforms[:, :3, axis_index])
    centers = transforms[:, :3, 3]
    if not np.all(np.isfinite(centers)):
        raise TablePlaneError("object centers contain non-finite values")
    return centers - 0.5 * float(height_m) * axes, axes


def fit_axis_constrained_plane(bottom_points: np.ndarray, axes: np.ndarray) -> Plane:
    points = np.asarray(bottom_points, dtype=np.float64)
    aligned = sign_consistent_axes(axes)
    if points.shape != aligned.shape or not np.all(np.isfinite(points)):
        raise TablePlaneError("plane fit inputs must be matched finite Nx3 arrays")
    normal = np.mean(aligned, axis=0)
    normal /= np.linalg.norm(normal)
    offset = -float(np.median(points @ normal))
    return Plane(normal, offset)


def signed_distances(plane: Plane, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
        raise TablePlaneError("points must have a finite final dimension of three")
    return values @ plane.normal + plane.offset_m


def reanchor_normal_offset(plane: Plane, anchor: np.ndarray, *, contact_verified: bool) -> Plane:
    """Return a same-normal plane through ``anchor`` only after contact proof."""

    point = np.asarray(anchor, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TablePlaneError("anchor must be one finite 3-vector")
    if contact_verified is not True:
        raise TablePlaneError("normal-offset reanchor requires independent contact evidence")
    return Plane(plane.normal.copy(), -float(plane.normal @ point))


def plane_basis(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,) or abs(float(np.linalg.norm(n)) - 1.0) > 1e-8:
        raise TablePlaneError("basis normal must be unit length")
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = seed - n * float(seed @ n)
    first /= np.linalg.norm(first)
    second = np.cross(n, first)
    return np.stack((first, second), axis=0)


def tangential_coordinates(normal: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise TablePlaneError("points must end in three coordinates")
    return values @ plane_basis(normal).T


def axis_angles_deg(normal: np.ndarray, axes: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    aligned = sign_consistent_axes(axes)
    cosine = np.clip(aligned @ n, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def wrist_table_physical_gate(
    plane: Plane,
    wrist_points_world: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_clearance_m: float = 0.0,
    high_confidence_threshold: float = 0.8,
) -> dict[str, np.ndarray]:
    """Evaluate wrist/table compliance without trusting confidence as geometry."""

    points = np.asarray(wrist_points_world, dtype=np.float64)
    score = np.asarray(confidence, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or score.shape != (len(points),):
        raise TablePlaneError("wrist gate input shape drift")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(score)):
        raise TablePlaneError("wrist gate input contains non-finite values")
    if minimum_clearance_m < 0 or not 0.0 <= high_confidence_threshold <= 1.0:
        raise TablePlaneError("wrist gate threshold is invalid")
    distance = signed_distances(plane, points)
    below = distance < float(minimum_clearance_m)
    return {
        "signed_distance_m": distance,
        "wrist_below_table": below,
        "high_confidence_below_table": below & (score > high_confidence_threshold),
    }


def finite_cylinder_sdf(
    points_world: np.ndarray,
    object_to_world: np.ndarray,
    radius_m: float,
    height_m: float,
) -> np.ndarray:
    """Exact capped-cylinder SDF for a local-Y analytic cylinder."""

    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(object_to_world, dtype=np.float64)
    if points.shape[-1:] != (3,) or transform.shape != (4, 4):
        raise TablePlaneError("cylinder SDF input shape drift")
    if radius_m <= 0 or height_m <= 0 or not np.all(np.isfinite(points)):
        raise TablePlaneError("cylinder SDF geometry/input invalid")
    inverse = np.linalg.inv(transform)
    flat = points.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    local = (homogeneous @ inverse.T)[:, :3]
    radial = np.sqrt(local[:, 0] ** 2 + local[:, 2] ** 2) - float(radius_m)
    axial = np.abs(local[:, 1]) - 0.5 * float(height_m)
    q = np.stack((radial, axial), axis=1)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return (outside + inside).reshape(points.shape[:-1])


def _grid(width: int, height: int, columns: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    if min(width, height, columns, rows) <= 0:
        raise TablePlaneError("grid dimensions must be positive")
    xs = np.linspace(0, width - 1, columns).round().astype(np.int64)
    ys = np.linspace(0, height - 1, rows).round().astype(np.int64)
    x, y = np.meshgrid(xs, ys)
    return x.ravel(), y.ravel()


def _target_world_points(
    frame: ReprojectionFrame, plane: Plane, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.stack((x, y, np.ones_like(x)), axis=1).astype(np.float64)
    rays_camera = pixels @ np.linalg.inv(frame.intrinsic).T
    rotation = frame.camera_to_world[:3, :3]
    origin = frame.camera_to_world[:3, 3]
    rays_world = rays_camera @ rotation.T
    denominator = rays_world @ plane.normal
    numerator = -(float(origin @ plane.normal) + plane.offset_m)
    valid = np.abs(denominator) > 1e-10
    distance = np.full(len(x), np.nan, dtype=np.float64)
    distance[valid] = numerator / denominator[valid]
    valid &= distance > 0
    points = origin[None, :] + distance[:, None] * rays_world
    return points, valid


def _nearest_donor_samples(
    frame: ReprojectionFrame, points_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(frame.camera_to_world)
    homogeneous = np.concatenate((points_world, np.ones((len(points_world), 1))), axis=1)
    camera = homogeneous @ world_to_camera.T
    depth = camera[:, 2]
    projected = camera[:, :3] @ frame.intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = projected[:, 0] / depth
        v = projected[:, 1] / depth
    finite_projection = np.isfinite(u) & np.isfinite(v)
    ui = np.zeros(len(u), dtype=np.int64)
    vi = np.zeros(len(v), dtype=np.int64)
    ui[finite_projection] = np.rint(u[finite_projection]).astype(np.int64)
    vi[finite_projection] = np.rint(v[finite_projection]).astype(np.int64)
    height, width = frame.support.shape
    valid = (
        finite_projection
        & (depth > 1e-8)
        & (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
    )
    colors = np.zeros((len(points_world), 3), dtype=np.float64)
    indices = np.flatnonzero(valid)
    valid[indices] &= frame.support[vi[indices], ui[indices]]
    accepted = np.flatnonzero(valid)
    colors[accepted] = frame.rgb[vi[accepted], ui[accepted]].astype(np.float64)
    return colors, valid


def leave_one_out_reprojection(
    frames: Sequence[ReprojectionFrame],
    planes: Sequence[Plane],
    *,
    grid_columns: int = 160,
    grid_rows: int = 120,
    min_donors: int = 3,
) -> list[dict[str, float | int]]:
    """Photometric diagnostics on caller-supplied conservative support.

    The function never infers table identity.  A caller must supply a support
    mask, and any formal gate must additionally freeze the meaning of that mask.
    """

    if len(frames) != len(planes) or len(frames) < 2 or min_donors < 1:
        raise TablePlaneError("LOO inputs are inconsistent")
    height, width = frames[0].support.shape
    if any(frame.support.shape != (height, width) for frame in frames):
        raise TablePlaneError("LOO image dimensions drift")
    x, y = _grid(width, height, grid_columns, grid_rows)
    results: list[dict[str, float | int]] = []
    for target_index, (target, plane) in enumerate(zip(frames, planes)):
        target_support = target.support[y, x]
        points, ray_valid = _target_world_points(target, plane, x, y)
        eligible = target_support & ray_valid
        donor_colors = []
        donor_validity = []
        safe_points = np.where(np.isfinite(points), points, 0.0)
        for donor_index, donor in enumerate(frames):
            if donor_index == target_index:
                continue
            colors, valid = _nearest_donor_samples(donor, safe_points)
            donor_colors.append(colors)
            donor_validity.append(valid & eligible)
        color_stack = np.stack(donor_colors, axis=1)
        valid_stack = np.stack(donor_validity, axis=1)
        valid_count = np.sum(valid_stack, axis=1)
        accepted = eligible & (valid_count >= min_donors)
        masked_colors = np.where(valid_stack[:, :, None], color_stack, np.nan)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prediction = np.nanmedian(masked_colors, axis=1)
        target_rgb = target.rgb[y, x].astype(np.float64)
        error = np.mean(np.abs(prediction - target_rgb), axis=1)
        accepted_error = error[accepted]
        results.append(
            {
                "eligible_samples": int(np.count_nonzero(eligible)),
                "accepted_samples": int(np.count_nonzero(accepted)),
                "coverage_ratio": float(np.count_nonzero(accepted) / max(1, np.count_nonzero(eligible))),
                "mae_rgb_0_255": float(np.mean(accepted_error)) if len(accepted_error) else float("nan"),
                "p95_rgb_0_255": float(np.percentile(accepted_error, 95)) if len(accepted_error) else float("nan"),
            }
        )
    return results
