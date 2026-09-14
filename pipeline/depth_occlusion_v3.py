#!/usr/bin/env python3
"""Depth-occlusion-v3 compositor reconstructed from the frozen specification.

This module is fail-closed and has no production writer.  Object depth is the
analytic, amodal Y-axis cylinder from Object6D; robot Range is converted to
camera Z before the frozen 3 mm comparison.  Decisions happen on a fixed 2x2
subpixel grid and ObjectIndex is returned only as QA metadata.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

OCCLUSION_EPSILON_M = 0.003
SUPERSAMPLE = 2
FRAME_BUNDLE_SCHEMA = "s8-depth-occlusion-v3-frame-bundle-v1"
SOURCE_CLEAN_BG = np.uint8(0)
SOURCE_ROBOT = np.uint8(1)
SOURCE_OBJECT = np.uint8(2)
SOURCE_NAMES = ("clean_bg", "robot", "object")
ALLOWED_OBJECT_TEXTURE_SOURCES = frozenset({"raw_observation", "object_donor"})


class DepthV3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Object6DCylinderFrame:
    frame_index: int
    transform_object_to_camera: np.ndarray
    radius_m: float
    height_m: float
    confidence: float


@dataclass(frozen=True)
class CylinderDepth:
    near_m: np.ndarray
    far_m: np.ndarray
    amodal_mask: np.ndarray


@dataclass(frozen=True)
class CompositeResult:
    rgb: np.ndarray
    source_map: np.ndarray
    source_map_2x: np.ndarray
    source_fractions: np.ndarray
    object_alpha: np.ndarray
    robot_alpha: np.ndarray
    clean_bg_alpha: np.ndarray
    robot_z_m_2x: np.ndarray
    object_depth_near_m_2x: np.ndarray
    object_depth_far_m_2x: np.ndarray
    object_index_qa: dict[str, int] | None


@dataclass(frozen=True)
class S8FrameBundle:
    session_id: str
    frame_name: str
    frame_index: int
    clean_rgb: np.ndarray
    object_rgb_2x: np.ndarray
    object_texture_source: str
    robot_rgb_2x: np.ndarray
    robot_range_m_2x: np.ndarray
    robot_alpha_2x: np.ndarray
    camera_intrinsics: np.ndarray
    object_index_2x: np.ndarray | None


def _validate_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(intrinsics, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise DepthV3Error("camera intrinsics must be a finite 3x3 matrix")
    if value[0, 0] <= 0 or value[1, 1] <= 0 or not np.allclose(
        value[2], (0.0, 0.0, 1.0), atol=1e-9
    ):
        raise DepthV3Error("camera intrinsics are not a positive pinhole model")
    return value


def _validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise DepthV3Error("Object6D transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise DepthV3Error("Object6D transform has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise DepthV3Error("Object6D rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise DepthV3Error("Object6D rotation is not right-handed")
    return value


def decode_object6d_cylinder_frame(
    payload: bytes, frame_index: int
) -> Object6DCylinderFrame:
    """Decode one valid Object6D frame from already identity-verified bytes."""
    required = {
        "T_object_to_camera",
        "confidence",
        "valid",
        "cylinder_radius_m",
        "cylinder_height_m",
    }
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise DepthV3Error(f"Object6D is missing keys: {missing}")
            frame_count = int(archive["T_object_to_camera"].shape[0])
            if frame_index < 0 or frame_index >= frame_count:
                raise DepthV3Error(
                    f"Object6D frame index {frame_index} outside [0,{frame_count})"
                )
            if not bool(archive["valid"][frame_index]):
                raise DepthV3Error(f"Object6D frame is invalid: {frame_index}")
            transform = _validate_rigid_transform(
                archive["T_object_to_camera"][frame_index]
            )
            radius = float(archive["cylinder_radius_m"])
            height = float(archive["cylinder_height_m"])
            confidence = float(archive["confidence"][frame_index])
    except OSError as exc:
        raise DepthV3Error("cannot decode Object6D bytes") from exc
    if radius <= 0 or height <= 0 or not np.isfinite((radius, height)).all():
        raise DepthV3Error("Object6D cylinder dimensions are invalid")
    if not np.isfinite(confidence):
        raise DepthV3Error("Object6D confidence is invalid")
    return Object6DCylinderFrame(
        frame_index=frame_index,
        transform_object_to_camera=transform,
        radius_m=radius,
        height_m=height,
        confidence=confidence,
    )


def load_object6d_cylinder_frame(path: Path, frame_index: int) -> Object6DCylinderFrame:
    """T0 compatibility reader; T1 runners consume verified bytes directly."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return decode_object6d_cylinder_frame(b"".join(chunks), frame_index)


def range_to_metric_z(
    range_buffer: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Convert Euclidean camera Range to optical-axis Z in metres."""

    value = np.asarray(range_buffer, dtype=np.float64)
    if value.ndim != 2 or fx <= 0 or fy <= 0:
        raise DepthV3Error("invalid Range buffer or intrinsics")
    height, width = value.shape
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    ray_norm = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
    return value / ray_norm


def supersampled_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """Return the fixed pixel-centre preserving 2x camera matrix."""

    camera = _validate_intrinsics(intrinsics).copy()
    camera[0, 0] *= SUPERSAMPLE
    camera[1, 1] *= SUPERSAMPLE
    camera[0, 2] = (camera[0, 2] + 0.5) * SUPERSAMPLE - 0.5
    camera[1, 2] = (camera[1, 2] + 0.5) * SUPERSAMPLE - 0.5
    return camera


def _scalar_string(archive: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(archive.get(name))
    if value.shape != ():
        raise DepthV3Error(f"frame bundle scalar missing: {name}")
    return str(value.item())


def _scalar_integer(archive: Mapping[str, np.ndarray], name: str) -> int:
    value = np.asarray(archive.get(name))
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise DepthV3Error(f"frame bundle integer scalar missing: {name}")
    return int(value.item())


def decode_s8_frame_bundle(payload: bytes) -> S8FrameBundle:
    """Decode the exact per-frame S8 handoff without path or fallback semantics."""

    required = {
        "schema_version",
        "session_id",
        "frame_name",
        "frame_index",
        "object_texture_source",
        "clean_rgb",
        "object_rgb_2x",
        "robot_rgb_2x",
        "robot_range_m_2x",
        "robot_alpha_2x",
        "camera_intrinsics",
    }
    optional = {"object_index_2x"}
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            names = tuple(archive.files)
            if len(names) != len(set(names)):
                raise DepthV3Error("S8 frame bundle contains duplicate array names")
            missing = sorted(required - set(names))
            extra = sorted(set(names) - required - optional)
            if missing or extra:
                raise DepthV3Error(
                    f"S8 frame bundle key closure differs: missing={missing}, extra={extra}"
                )
            # Validate the key closure before materializing arrays, so a bundle
            # cannot smuggle labels or fallback pixels through an unused field.
            values = {name: np.asarray(archive[name]) for name in names}
    except (OSError, ValueError, KeyError) as exc:
        raise DepthV3Error(f"invalid S8 frame bundle: {exc}") from exc
    if _scalar_string(values, "schema_version") != FRAME_BUNDLE_SCHEMA:
        raise DepthV3Error("unsupported S8 frame bundle schema")
    session_id = _scalar_string(values, "session_id")
    lowered = session_id.lower()
    if any(token in lowered for token in ("labels", "025", "blind", "processed")):
        raise DepthV3Error("forbidden S8 session identity")
    frame_name = _scalar_string(values, "frame_name")
    frame_index = _scalar_integer(values, "frame_index")
    if frame_index < 0 or frame_name != f"{frame_index:05d}":
        raise DepthV3Error("S8 frame name/index identity mismatch")
    texture_source = _scalar_string(values, "object_texture_source")
    if texture_source not in ALLOWED_OBJECT_TEXTURE_SOURCES:
        raise DepthV3Error("S8 object texture source is not RAW/donor")
    object_index = values.get("object_index_2x")
    return S8FrameBundle(
        session_id=session_id,
        frame_name=frame_name,
        frame_index=frame_index,
        clean_rgb=values["clean_rgb"],
        object_rgb_2x=values["object_rgb_2x"],
        object_texture_source=texture_source,
        robot_rgb_2x=values["robot_rgb_2x"],
        robot_range_m_2x=values["robot_range_m_2x"],
        robot_alpha_2x=values["robot_alpha_2x"],
        camera_intrinsics=_validate_intrinsics(values["camera_intrinsics"]),
        object_index_2x=object_index,
    )


def compose_s8_frame_bundle(
    bundle: S8FrameBundle, object6d: Object6DCylinderFrame
) -> CompositeResult:
    """Project the amodal cylinder and apply the frozen v3 comparison."""

    if bundle.frame_index != object6d.frame_index:
        raise DepthV3Error("S8 bundle/Object6D frame identity mismatch")
    clean = np.asarray(bundle.clean_rgb)
    if clean.ndim != 3 or clean.shape[2] != 3:
        raise DepthV3Error("S8 clean image has invalid shape")
    height, width = clean.shape[:2]
    depth = project_amodal_cylinder_depth(
        object6d.transform_object_to_camera,
        object6d.radius_m,
        object6d.height_m,
        bundle.camera_intrinsics,
        height,
        width,
    )
    return compose_depth_v3(
        clean_rgb=bundle.clean_rgb,
        object_rgb_2x=bundle.object_rgb_2x,
        object_texture_source=bundle.object_texture_source,
        object_depth_near_m_2x=depth.near_m,
        object_depth_far_m_2x=depth.far_m,
        robot_rgb_2x=bundle.robot_rgb_2x,
        robot_range_m_2x=bundle.robot_range_m_2x,
        robot_alpha_2x=bundle.robot_alpha_2x,
        robot_intrinsics_2x=supersampled_intrinsics(bundle.camera_intrinsics),
        object_index_2x=bundle.object_index_2x,
    )


def project_amodal_cylinder_depth(
    transform_object_to_camera: np.ndarray,
    radius_m: float,
    height_m: float,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
) -> CylinderDepth:
    """Ray-cast the complete closed Y-axis cylinder at a fixed 2x grid."""
    transform = _validate_rigid_transform(transform_object_to_camera)
    camera = _validate_intrinsics(intrinsics)
    if radius_m <= 0 or height_m <= 0 or image_height < 1 or image_width < 1:
        raise DepthV3Error("invalid cylinder dimensions or image size")
    if not np.isfinite((radius_m, height_m)).all():
        raise DepthV3Error("cylinder dimensions must be finite")

    output_height = image_height * SUPERSAMPLE
    output_width = image_width * SUPERSAMPLE
    u = (np.arange(output_width, dtype=np.float64) + 0.5) / SUPERSAMPLE - 0.5
    v = (np.arange(output_height, dtype=np.float64) + 0.5) / SUPERSAMPLE - 0.5
    uu, vv = np.meshgrid(u, v)
    rays_camera = np.stack(
        (
            (uu - camera[0, 2]) / camera[0, 0],
            (vv - camera[1, 2]) / camera[1, 1],
            np.ones_like(uu),
        ),
        axis=-1,
    )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    origin_object = -rotation.T @ translation
    rays_object = rays_camera @ rotation

    ox, oy, oz = origin_object
    dx = rays_object[..., 0]
    dy = rays_object[..., 1]
    dz = rays_object[..., 2]
    radial_a = dx * dx + dz * dz
    radial_b = 2.0 * (ox * dx + oz * dz)
    radial_c = ox * ox + oz * oz - radius_m * radius_m
    discriminant = radial_b * radial_b - 4.0 * radial_a * radial_c
    valid_side = (radial_a > 1e-15) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = np.where(valid_side, 2.0 * radial_a, 1.0)
    side_low = (-radial_b - root) / denominator
    side_high = (-radial_b + root) / denominator
    half_height = height_m / 2.0
    candidates: list[np.ndarray] = []
    for side in (side_low, side_high):
        within_height = np.abs(oy + side * dy) <= half_height + 1e-12
        candidates.append(
            np.where(valid_side & within_height & (side > 0.0), side, np.nan)
        )
    for cap_y in (-half_height, half_height):
        cap = np.divide(
            cap_y - oy,
            dy,
            out=np.full_like(dy, np.nan),
            where=np.abs(dy) > 1e-15,
        )
        radial_at_cap = (ox + cap * dx) ** 2 + (oz + cap * dz) ** 2
        candidates.append(
            np.where(
                (cap > 0.0) & (radial_at_cap <= radius_m * radius_m + 1e-12),
                cap,
                np.nan,
            )
        )

    intersections = np.stack(candidates, axis=0)
    finite = np.isfinite(intersections)
    amodal_mask = finite.any(axis=0)
    near = np.min(np.where(finite, intersections, np.inf), axis=0)
    far = np.max(np.where(finite, intersections, -np.inf), axis=0)
    near = np.where(amodal_mask, near, np.nan)
    far = np.where(amodal_mask, far, np.nan)
    if np.any(near[amodal_mask] <= 0.0) or np.any(far[amodal_mask] < near[amodal_mask]):
        raise DepthV3Error("analytic cylinder produced an invalid near/far interval")
    return CylinderDepth(near_m=near, far_m=far, amodal_mask=amodal_mask)


def _validate_rgb(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (*shape, 3) or not np.isfinite(array).all():
        raise DepthV3Error(f"{name} must be a finite HxWx3 image")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise DepthV3Error(f"{name} has an unsupported dtype")
    if np.any((array < 0.0) | (array > 255.0)):
        raise DepthV3Error(f"{name} values must be in [0,255]")
    return array.astype(np.float64)


def compose_depth_v3(
    *,
    clean_rgb: np.ndarray,
    object_rgb_2x: np.ndarray,
    object_texture_source: str,
    object_depth_near_m_2x: np.ndarray,
    object_depth_far_m_2x: np.ndarray,
    robot_rgb_2x: np.ndarray,
    robot_range_m_2x: np.ndarray,
    robot_alpha_2x: np.ndarray,
    robot_intrinsics_2x: np.ndarray,
    object_index_2x: np.ndarray | None = None,
) -> CompositeResult:
    """Composite object/robot/clean at 2x, then average each 2x2 cell."""
    if object_texture_source not in ALLOWED_OBJECT_TEXTURE_SOURCES:
        raise DepthV3Error(
            "object texture must come from RAW observation or an object donor"
        )
    clean = np.asarray(clean_rgb)
    if clean.ndim != 3 or clean.shape[2] != 3:
        raise DepthV3Error("clean_rgb must be HxWx3")
    base_shape = clean.shape[:2]
    high_shape = (base_shape[0] * SUPERSAMPLE, base_shape[1] * SUPERSAMPLE)
    clean_float = _validate_rgb("clean_rgb", clean, base_shape)
    object_rgb = _validate_rgb("object_rgb_2x", object_rgb_2x, high_shape)
    robot_rgb = _validate_rgb("robot_rgb_2x", robot_rgb_2x, high_shape)

    near = np.asarray(object_depth_near_m_2x, dtype=np.float64)
    far = np.asarray(object_depth_far_m_2x, dtype=np.float64)
    robot_range = np.asarray(robot_range_m_2x, dtype=np.float64)
    robot_alpha_input = np.asarray(robot_alpha_2x, dtype=np.float64)
    if any(value.shape != high_shape for value in (near, far, robot_range, robot_alpha_input)):
        raise DepthV3Error("all depth/range/alpha buffers must match the fixed 2x grid")
    object_mask = np.isfinite(near)
    if not np.array_equal(object_mask, np.isfinite(far)):
        raise DepthV3Error("Object6D near/far validity differs")
    if np.any(near[object_mask] <= 0.0) or np.any(far[object_mask] < near[object_mask]):
        raise DepthV3Error("Object6D near/far depth is invalid")
    if not np.isfinite(robot_alpha_input).all() or np.any(
        (robot_alpha_input < 0.0) | (robot_alpha_input > 1.0)
    ):
        raise DepthV3Error("robot alpha must be finite and in [0,1]")
    robot_mask = robot_alpha_input > 0.0
    camera = _validate_intrinsics(robot_intrinsics_2x)
    robot_z = range_to_metric_z(
        robot_range, camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2]
    )
    if np.any(~np.isfinite(robot_z[robot_mask])) or np.any(robot_z[robot_mask] <= 0.0):
        raise DepthV3Error("robot Range is invalid wherever robot alpha is nonzero")

    object_front = object_mask & (
        (~robot_mask) | (near <= robot_z + OCCLUSION_EPSILON_M)
    )
    robot_front = robot_mask & ~object_front
    source_map_2x = np.full(high_shape, SOURCE_CLEAN_BG, dtype=np.uint8)
    source_map_2x[robot_front] = SOURCE_ROBOT
    source_map_2x[object_front] = SOURCE_OBJECT

    clean_2x = np.repeat(
        np.repeat(clean_float, SUPERSAMPLE, axis=0), SUPERSAMPLE, axis=1
    )
    selected = clean_2x
    selected = np.where(robot_front[..., None], robot_rgb, selected)
    selected = np.where(object_front[..., None], object_rgb, selected)
    rgb = np.rint(
        selected.reshape(
            base_shape[0], SUPERSAMPLE, base_shape[1], SUPERSAMPLE, 3
        ).mean(axis=(1, 3))
    ).astype(np.uint8)

    one_hot = np.stack(
        [source_map_2x == code for code in (SOURCE_CLEAN_BG, SOURCE_ROBOT, SOURCE_OBJECT)],
        axis=-1,
    ).astype(np.float64)
    fractions = one_hot.reshape(
        base_shape[0], SUPERSAMPLE, base_shape[1], SUPERSAMPLE, 3
    ).mean(axis=(1, 3))
    source_map = np.argmax(fractions, axis=-1).astype(np.uint8)

    object_index_qa = None
    if object_index_2x is not None:
        object_index = np.asarray(object_index_2x)
        if object_index.shape != high_shape or not np.isfinite(object_index).all():
            raise DepthV3Error("ObjectIndex QA buffer must be finite and match 2x grid")
        object_index_qa = {
            "nonzero_subpixels": int(np.count_nonzero(object_index)),
            "unique_roles": int(np.unique(object_index).size),
        }
    return CompositeResult(
        rgb=rgb,
        source_map=source_map,
        source_map_2x=source_map_2x,
        source_fractions=fractions,
        object_alpha=fractions[..., int(SOURCE_OBJECT)],
        robot_alpha=fractions[..., int(SOURCE_ROBOT)],
        clean_bg_alpha=fractions[..., int(SOURCE_CLEAN_BG)],
        robot_z_m_2x=robot_z,
        object_depth_near_m_2x=near.copy(),
        object_depth_far_m_2x=far.copy(),
        object_index_qa=object_index_qa,
    )


def _plane_range(z_m: float, shape: tuple[int, int], intrinsics: np.ndarray) -> np.ndarray:
    height, width = shape
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    ray_norm = np.sqrt(
        1.0
        + ((u - intrinsics[0, 2]) / intrinsics[0, 0]) ** 2
        + ((v - intrinsics[1, 2]) / intrinsics[1, 1]) ** 2
    )
    return z_m * ray_norm


def write_synthetic_evidence(output_dir: Path, object6d_path: Path) -> dict[str, Any]:
    """Write T0 synthetic evidence; this is deliberately not a production entry."""
    project_root = Path(__file__).resolve().parents[1]
    audit_root = (project_root / "audits").resolve()
    output = output_dir.resolve()
    if output != audit_root and not output.is_relative_to(audit_root):
        raise DepthV3Error("synthetic evidence may only be written below archive/audits/")
    output.mkdir(parents=True, exist_ok=True)

    height, width = 48, 72
    high_shape = (height * SUPERSAMPLE, width * SUPERSAMPLE)
    camera = np.array(
        [[72.0, 0.0, 35.5], [0.0, 72.0, 23.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 1.0
    depth = project_amodal_cylinder_depth(transform, 0.22, 0.6, camera, height, width)
    camera_2x = camera.copy()
    camera_2x[0, 0] *= SUPERSAMPLE
    camera_2x[1, 1] *= SUPERSAMPLE
    camera_2x[0, 2] = (camera[0, 2] + 0.5) * SUPERSAMPLE - 0.5
    camera_2x[1, 2] = (camera[1, 2] + 0.5) * SUPERSAMPLE - 0.5
    robot_alpha = depth.amodal_mask.astype(np.float64)
    robot_range = _plane_range(1.5, high_shape, camera_2x)
    clean = np.full((height, width, 3), (80, 90, 100), dtype=np.uint8)
    robot = np.full((*high_shape, 3), (30, 80, 230), dtype=np.uint8)
    object_rgb = np.full((*high_shape, 3), (230, 120, 30), dtype=np.uint8)
    result = compose_depth_v3(
        clean_rgb=clean,
        object_rgb_2x=object_rgb,
        object_texture_source="object_donor",
        object_depth_near_m_2x=depth.near_m,
        object_depth_far_m_2x=depth.far_m,
        robot_rgb_2x=robot,
        robot_range_m_2x=robot_range,
        robot_alpha_2x=robot_alpha,
        robot_intrinsics_2x=camera_2x,
        object_index_2x=np.where(robot_alpha > 0.0, 7, 0),
    )
    if np.any(result.source_map_2x == SOURCE_ROBOT):
        raise DepthV3Error("counterexample failed: behind-object robot remains visible")

    clean_2x = np.repeat(np.repeat(clean, 2, axis=0), 2, axis=1)
    before_2x = np.where(robot_alpha[..., None] > 0.0, robot, clean_2x)
    before = np.rint(
        before_2x.reshape(height, 2, width, 2, 3).mean(axis=(1, 3))
    ).astype(np.uint8)
    source_colors = np.array(
        [[80, 90, 100], [30, 80, 230], [230, 120, 30]], dtype=np.uint8
    )
    source_2x = source_colors[result.source_map_2x]
    source_preview = np.rint(
        source_2x.reshape(height, 2, width, 2, 3).mean(axis=(1, 3))
    ).astype(np.uint8)
    comparison_rgb = np.concatenate((before, result.rgb, source_preview), axis=1)
    try:
        import cv2
    except ImportError as exc:
        raise DepthV3Error("OpenCV is required only for the audit preview") from exc
    preview_path = output / "counterexample_before_after_source.png"
    if not cv2.imwrite(str(preview_path), cv2.cvtColor(comparison_rgb, cv2.COLOR_RGB2BGR)):
        raise DepthV3Error(f"cannot write audit preview: {preview_path}")

    row_counts = depth.amodal_mask.sum(axis=1)
    row = int(np.argmax(row_counts))
    columns = np.flatnonzero(depth.amodal_mask[row])
    scan_start = max(0, int(columns[0]) - 2)
    scan_stop = min(high_shape[1], int(columns[-1]) + 3)
    scanline = [
        {
            "x_2x": int(column),
            "z_object_m": (
                round(float(depth.near_m[row, column]), 6)
                if depth.amodal_mask[row, column]
                else None
            ),
            "z_robot_m": round(float(result.robot_z_m_2x[row, column]), 6),
            "source": SOURCE_NAMES[int(result.source_map_2x[row, column])],
        }
        for column in range(scan_start, scan_stop)
    ]

    actual = load_object6d_cylinder_frame(object6d_path, 240)
    actual_camera = np.array(
        [[160.0, 0.0, 159.5], [0.0, 160.0, 119.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    actual_depth = project_amodal_cylinder_depth(
        actual.transform_object_to_camera,
        actual.radius_m,
        actual.height_m,
        actual_camera,
        240,
        320,
    )
    report = {
        "status": "SYNTHETIC_T0_NOT_PRODUCTION",
        "degraded": True,
        "degraded_reason": (
            "D2 full arm is HOLD: q_arm/calibration/base placement unavailable"
        ),
        "epsilon_m": OCCLUSION_EPSILON_M,
        "supersample": SUPERSAMPLE,
        "counterexample": {
            "robot_subpixels_before": int(np.count_nonzero(robot_alpha)),
            "robot_subpixels_after": int(
                np.count_nonzero(result.source_map_2x == SOURCE_ROBOT)
            ),
            "object_subpixels_after": int(
                np.count_nonzero(result.source_map_2x == SOURCE_OBJECT)
            ),
            "preview": str(preview_path),
        },
        "scanline_y_2x": row,
        "scanline": scanline,
        "object_index_role": "QA_ONLY_NOT_USED_IN_DECISION",
        "actual_004_frame_240": {
            "valid": True,
            "confidence": actual.confidence,
            "radius_m": actual.radius_m,
            "height_m": actual.height_m,
            "amodal_subpixels_2x": int(np.count_nonzero(actual_depth.amodal_mask)),
            "near_min_m": float(np.nanmin(actual_depth.near_m)),
            "far_max_m": float(np.nanmax(actual_depth.far_m)),
        },
    }
    report_path = output / "synthetic_depth_v3_evidence.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-evidence-dir", type=Path, required=True)
    parser.add_argument("--object6d", type=Path, required=True)
    args = parser.parse_args()
    report = write_synthetic_evidence(args.synthetic_evidence_dir, args.object6d)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
