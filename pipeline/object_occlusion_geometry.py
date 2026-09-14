#!/usr/bin/env python3
"""Shape-dispatched object depth for robot compositing.

The returned ray parameter is optical-axis camera Z because camera rays are
parameterised as ``[(u-cx)/fx, (v-cy)/fy, 1]``.  The module intentionally has no
shape guessing: a caller must supply an explicit geometry kind and authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pipeline.depth_occlusion_v3 import (
    CylinderDepth,
    DepthV3Error,
    SUPERSAMPLE,
    project_amodal_cylinder_depth,
)


ShapeKind = Literal["cylinder_y", "box_xyz", "mask_depth"]


@dataclass(frozen=True)
class ObjectGeometry:
    kind: ShapeKind
    transform_object_to_camera: np.ndarray | None = None
    cylinder_radius_m: float | None = None
    cylinder_height_m: float | None = None
    box_size_xyz_m: np.ndarray | None = None
    near_depth_m_2x: np.ndarray | None = None
    far_depth_m_2x: np.ndarray | None = None


def _camera_rays(
    intrinsics: np.ndarray, image_height: int, image_width: int
) -> np.ndarray:
    camera = np.asarray(intrinsics, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise DepthV3Error("camera intrinsics must be finite 3x3")
    if camera[0, 0] <= 0.0 or camera[1, 1] <= 0.0:
        raise DepthV3Error("camera focal length must be positive")
    if image_height < 1 or image_width < 1:
        raise DepthV3Error("image dimensions must be positive")
    out_h, out_w = image_height * SUPERSAMPLE, image_width * SUPERSAMPLE
    u = (np.arange(out_w, dtype=np.float64) + 0.5) / SUPERSAMPLE - 0.5
    v = (np.arange(out_h, dtype=np.float64) + 0.5) / SUPERSAMPLE - 0.5
    uu, vv = np.meshgrid(u, v)
    return np.stack(
        (
            (uu - camera[0, 2]) / camera[0, 0],
            (vv - camera[1, 2]) / camera[1, 1],
            np.ones_like(uu),
        ),
        axis=-1,
    )


def project_amodal_box_depth(
    transform_object_to_camera: np.ndarray,
    size_xyz_m: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    """Ray-cast a closed object-local axis-aligned box at the fixed 2x grid."""

    transform = np.asarray(transform_object_to_camera, dtype=np.float64)
    size = np.asarray(size_xyz_m, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise DepthV3Error("box pose must be finite 4x4")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise DepthV3Error("box pose has invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise DepthV3Error("box pose is not right-handed rigid SE(3)")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise DepthV3Error("box size must be three positive finite metres")

    rays_camera = _camera_rays(intrinsics, image_height, image_width)
    translation = transform[:3, 3]
    origin_object = -rotation.T @ translation
    rays_object = rays_camera @ rotation
    half = size / 2.0

    near = np.full(rays_object.shape[:2], -np.inf, dtype=np.float64)
    far = np.full(rays_object.shape[:2], np.inf, dtype=np.float64)
    valid = np.ones(rays_object.shape[:2], dtype=bool)
    for axis in range(3):
        origin = float(origin_object[axis])
        direction = rays_object[..., axis]
        parallel = np.abs(direction) <= 1e-15
        valid &= ~(parallel & ((origin < -half[axis]) | (origin > half[axis])))
        safe = np.where(parallel, 1.0, direction)
        first = (-half[axis] - origin) / safe
        second = (half[axis] - origin) / safe
        axis_near = np.minimum(first, second)
        axis_far = np.maximum(first, second)
        near = np.maximum(near, np.where(parallel, -np.inf, axis_near))
        far = np.minimum(far, np.where(parallel, np.inf, axis_far))

    mask = valid & (far >= near) & (far > 0.0)
    near = np.where(mask, np.maximum(near, 0.0), np.nan)
    far = np.where(mask, far, np.nan)
    if np.any(near[mask] <= 0.0) or np.any(far[mask] < near[mask]):
        raise DepthV3Error("analytic box produced invalid near/far depth")
    return CylinderDepth(near_m=near, far_m=far, amodal_mask=mask)


def _validated_external_depth(
    near: np.ndarray,
    far: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    expected = (image_height * SUPERSAMPLE, image_width * SUPERSAMPLE)
    near_value = np.asarray(near, dtype=np.float64)
    far_value = np.asarray(far, dtype=np.float64)
    if near_value.shape != expected or far_value.shape != expected:
        raise DepthV3Error("external mask+depth arrays have wrong 2x shape")
    mask = np.isfinite(near_value)
    if not np.array_equal(mask, np.isfinite(far_value)):
        raise DepthV3Error("external near/far validity differs")
    if np.any(near_value[mask] <= 0.0) or np.any(far_value[mask] < near_value[mask]):
        raise DepthV3Error("external mask+depth is not positive ordered depth")
    return CylinderDepth(near_m=near_value, far_m=far_value, amodal_mask=mask)


def project_object_depth(
    geometry: ObjectGeometry,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    """Dispatch only an explicitly-authorised object geometry kind."""

    if geometry.kind == "cylinder_y":
        if (
            geometry.transform_object_to_camera is None
            or geometry.cylinder_radius_m is None
            or geometry.cylinder_height_m is None
        ):
            raise DepthV3Error("cylinder authority is incomplete")
        return project_amodal_cylinder_depth(
            geometry.transform_object_to_camera,
            float(geometry.cylinder_radius_m),
            float(geometry.cylinder_height_m),
            intrinsics,
            image_height,
            image_width,
        )
    if geometry.kind == "box_xyz":
        if geometry.transform_object_to_camera is None or geometry.box_size_xyz_m is None:
            raise DepthV3Error("box/card authority is incomplete")
        return project_amodal_box_depth(
            geometry.transform_object_to_camera,
            geometry.box_size_xyz_m,
            intrinsics,
            image_height,
            image_width,
        )
    if geometry.kind == "mask_depth":
        if geometry.near_depth_m_2x is None or geometry.far_depth_m_2x is None:
            raise DepthV3Error("external mask+depth authority is incomplete")
        return _validated_external_depth(
            geometry.near_depth_m_2x,
            geometry.far_depth_m_2x,
            image_height,
            image_width,
        )
    raise DepthV3Error(f"unsupported object geometry kind: {geometry.kind!r}")
