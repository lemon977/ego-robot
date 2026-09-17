"""Observed-only Object6D geometry on registered physical-left depth pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class VisibleSurfaceError(ValueError):
    pass


@dataclass(frozen=True)
class VisiblePoints:
    points_depth_camera_m: np.ndarray
    depth_pixels_xy: np.ndarray
    primary_pixels_xy: np.ndarray


def _homogeneous_map(points_xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy))))
    mapped = homogeneous @ transform.T
    return mapped[:, :2] / mapped[:, 2:3]


def backproject_registered_visible_surface(
    *, mask_primary: np.ndarray, depth_m: np.ndarray, depth_valid: np.ndarray,
    depth_intrinsics: np.ndarray, h_depth_to_primary: np.ndarray,
) -> VisiblePoints:
    mask = np.asarray(mask_primary, bool)
    depth = np.asarray(depth_m, np.float64)
    valid = np.asarray(depth_valid, bool)
    intrinsics = np.asarray(depth_intrinsics, np.float64)
    homography = np.asarray(h_depth_to_primary, np.float64)
    if mask.ndim != 2 or depth.ndim != 2 or valid.shape != depth.shape:
        raise VisibleSurfaceError("mask/depth geometry is invalid")
    if intrinsics.shape != (3, 3) or homography.shape != (3, 3):
        raise VisibleSurfaceError("registration/intrinsics must be 3x3")
    yy, xx = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
    depth_xy = np.column_stack((xx, yy)).astype(np.float64)
    primary_xy = _homogeneous_map(depth_xy, homography)
    primary_index = np.rint(primary_xy).astype(np.int64)
    in_bounds = (
        np.isfinite(primary_xy).all(axis=1)
        & (primary_index[:, 0] >= 0) & (primary_index[:, 0] < mask.shape[1])
        & (primary_index[:, 1] >= 0) & (primary_index[:, 1] < mask.shape[0])
    )
    selected = np.zeros(len(depth_xy), bool)
    selected[in_bounds] = mask[
        primary_index[in_bounds, 1], primary_index[in_bounds, 0]
    ]
    depth_xy = depth_xy[selected]
    primary_xy = primary_xy[selected]
    z = depth[yy[selected], xx[selected]]
    inverse = np.linalg.inv(intrinsics)
    rays = np.column_stack((depth_xy, np.ones(len(depth_xy)))) @ inverse.T
    points = rays * z[:, None]
    return VisiblePoints(points, depth_xy, primary_xy)


def fit_visible_plane(points: np.ndarray) -> dict[str, object]:
    value = np.asarray(points, np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 100:
        raise VisibleSurfaceError("at least 100 visible 3-D points are required")
    if not np.isfinite(value).all():
        raise VisibleSurfaceError("visible points contain non-finite values")
    centroid = value.mean(axis=0)
    centered = value - centroid
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0:
        normal = -normal
    residual = np.abs(centered @ normal)
    tangent = vh[0]
    bitangent = np.cross(normal, tangent)
    return {
        "centroid_m": centroid.tolist(),
        "normal_depth_camera": normal.tolist(),
        "tangent_depth_camera": tangent.tolist(),
        "bitangent_depth_camera": bitangent.tolist(),
        "visible_point_count": len(value),
        "plane_residual_median_m": float(np.median(residual)),
        "plane_residual_p90_m": float(np.percentile(residual, 90)),
        "singular_values": singular.tolist(),
        "geometry_authority": "DIRECTLY_VISIBLE_SURFACE_ONLY",
        "hidden_shape_inferred": False,
    }


def estimate_visible_object(
    *, task: str, mask_primary: np.ndarray, depth_m: np.ndarray,
    depth_valid: np.ndarray, depth_intrinsics: np.ndarray,
    h_depth_to_primary: np.ndarray,
) -> dict[str, object]:
    visible = backproject_registered_visible_surface(
        mask_primary=mask_primary, depth_m=depth_m, depth_valid=depth_valid,
        depth_intrinsics=depth_intrinsics,
        h_depth_to_primary=h_depth_to_primary,
    )
    if len(visible.points_depth_camera_m) < 100:
        return {
            "status": "INVALID_INSUFFICIENT_VISIBLE_SURFACE",
            "task": task,
            "visible_point_count": len(visible.points_depth_camera_m),
            "hidden_shape_inferred": False,
        }
    plane = fit_visible_plane(visible.points_depth_camera_m)
    if task == "playing_cards":
        kind = "VISIBLE_CARD_OR_STACK_PLANE"
    elif task == "potato_chips":
        kind = "VISIBLE_CHIP_SURFACE_TANGENT_HYPOTHESIS"
    else:
        raise VisibleSurfaceError(f"unsupported task: {task}")
    return {
        "status": "PASS_VISIBLE_SURFACE",
        "task": task,
        "geometry_kind": kind,
        **plane,
        "claim_limit": (
            "Visible-surface optical-Z geometry only; occluded pixels and "
            "hidden object shape remain invalid."
        ),
    }
