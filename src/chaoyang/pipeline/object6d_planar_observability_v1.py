"""Observed-only planar Object6D quantities with independent observability.

The estimator deliberately does not emit a six-degree-of-freedom pose.  It
reports three quantities independently:

* the centroid of the directly visible surface (not the object centre),
* the normal of a plane fitted to directly visible depth samples, and
* a pi-periodic in-plane major axis when the visible support is anisotropic.

Masks are sampled through an explicit depth-pixel to mask-pixel map.  Identity
registration is never assumed.  Unknown/occluded inputs and weak geometry are
returned as explicit unobservable records; no hidden pixels are completed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "object6d-planar-observability-v1"
DEPTH_REFERENCE = "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"
MASK_REGISTRATION_AUTHORITY = "PROVEN_DEPTH_TO_SAM_RESIZE_PIXEL_MAP"
DIRECT_VISIBILITY = "DIRECT_VISIBLE_PIXELS_ONLY"
BLOCKED_VISIBILITY_STATES = frozenset({"OCCLUDED", "UNKNOWN"})
ADMISSIBLE_MASK_STATES = frozenset({"seeded", "tracked", "reseeded"})

MIN_TRANSLATION_POINTS = 24
MIN_PLANE_POINTS = 100
MIN_ROTATION_POINTS = 160
MAX_PLANARITY_RATIO = 0.25
MIN_IN_PLANE_ANISOTROPY = 1.20


class PlanarObservabilityError(ValueError):
    """Raised when the declared input contract is internally inconsistent."""


@dataclass(frozen=True)
class PlanarFrameInput:
    """One frame in the explicit mask/depth join domain.

    ``depth_to_mask_xy[y, x]`` contains the floating point SAM resize-only
    pixel centre corresponding to depth pixel ``(x, y)``.  It is sampled with
    nearest-neighbour semantics because masks are categorical evidence.
    """

    frame_index: int
    mask: np.ndarray
    mask_state: str
    visibility_state: str
    depth_m: np.ndarray
    depth_valid: np.ndarray
    depth_intrinsics: np.ndarray
    depth_to_mask_xy: np.ndarray
    registration_valid: np.ndarray
    depth_reference: str = DEPTH_REFERENCE
    registration_authority: str = MASK_REGISTRATION_AUTHORITY


def _component(
    observability: str,
    *,
    estimate: Any = None,
    residual: dict[str, float] | None = None,
    reason: str | None = None,
    semantics: str,
) -> dict[str, Any]:
    return {
        "observability": observability,
        "estimate": estimate,
        "residual": residual,
        "reason": reason,
        "semantics": semantics,
    }


def _unobservable_record(
    *, frame_index: int, mask_state: str, visibility_state: str, reason: str,
    mask_pixel_count: int = 0, registered_valid_depth_count: int = 0,
) -> dict[str, Any]:
    common = {
        "frame_index": int(frame_index),
        "mask_state": mask_state,
        "visibility_state": visibility_state,
        "mask_pixel_count": int(mask_pixel_count),
        "registered_valid_depth_count": int(registered_valid_depth_count),
        "registered_valid_depth_fraction": 0.0,
    }
    return {
        **common,
        "translation": _component(
            "UNOBSERVABLE", reason=reason,
            semantics="DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_CENTER",
        ),
        "plane_normal": _component(
            "UNOBSERVABLE", reason=reason,
            semantics="DIRECT_VISIBLE_SURFACE_PLANE_NORMAL",
        ),
        "in_plane_rotation": _component(
            "UNOBSERVABLE", reason=reason,
            semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
        ),
        "hidden_geometry_inferred": False,
    }


def _validate_input(value: PlanarFrameInput) -> tuple[np.ndarray, ...]:
    mask = np.asarray(value.mask, dtype=bool)
    depth = np.asarray(value.depth_m, dtype=np.float64)
    valid = np.asarray(value.depth_valid, dtype=bool)
    intrinsics = np.asarray(value.depth_intrinsics, dtype=np.float64)
    mapping = np.asarray(value.depth_to_mask_xy, dtype=np.float64)
    registration_valid = np.asarray(value.registration_valid, dtype=bool)
    if value.frame_index < 0:
        raise PlanarObservabilityError("frame_index must be non-negative")
    if mask.ndim != 2:
        raise PlanarObservabilityError("mask must be a two-dimensional array")
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise PlanarObservabilityError("depth and depth_valid geometry differ")
    if mapping.shape != (*depth.shape, 2):
        raise PlanarObservabilityError("depth_to_mask_xy must be HxWx2")
    if registration_valid.shape != depth.shape:
        raise PlanarObservabilityError("registration_valid geometry differs")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise PlanarObservabilityError("depth_intrinsics must be finite 3x3")
    if abs(float(intrinsics[2, 2]) - 1.0) > 1e-9:
        raise PlanarObservabilityError("depth_intrinsics homogeneous scale is invalid")
    if float(intrinsics[0, 0]) <= 0.0 or float(intrinsics[1, 1]) <= 0.0:
        raise PlanarObservabilityError("depth focal lengths must be positive")
    if value.depth_reference != DEPTH_REFERENCE:
        raise PlanarObservabilityError("depth must be rectified-left optical-Z metres")
    if value.registration_authority != MASK_REGISTRATION_AUTHORITY:
        raise PlanarObservabilityError(
            "mask/depth join requires a proven explicit pixel map"
        )
    if value.mask_state not in ADMISSIBLE_MASK_STATES | {"unknown"}:
        raise PlanarObservabilityError(f"unsupported mask_state: {value.mask_state}")
    if value.visibility_state not in BLOCKED_VISIBILITY_STATES | {DIRECT_VISIBILITY}:
        raise PlanarObservabilityError(
            f"unsupported visibility_state: {value.visibility_state}"
        )
    return mask, depth, valid, intrinsics, mapping, registration_valid


def _registered_points(
    *, mask: np.ndarray, depth: np.ndarray, valid: np.ndarray,
    intrinsics: np.ndarray, mapping: np.ndarray,
    registration_valid: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    finite_map = np.isfinite(mapping).all(axis=2)
    base = valid & registration_valid & finite_map & np.isfinite(depth) & (depth > 0.0)
    source_y, source_x = np.nonzero(base)
    if not len(source_x):
        return np.empty((0, 3), np.float64), int(mask.sum()), 0
    mapped = np.rint(mapping[source_y, source_x]).astype(np.int64)
    in_bounds = (
        (mapped[:, 0] >= 0) & (mapped[:, 0] < mask.shape[1])
        & (mapped[:, 1] >= 0) & (mapped[:, 1] < mask.shape[0])
    )
    selected = np.zeros(len(source_x), dtype=bool)
    selected[in_bounds] = mask[mapped[in_bounds, 1], mapped[in_bounds, 0]]
    xx = source_x[selected].astype(np.float64)
    yy = source_y[selected].astype(np.float64)
    z = depth[source_y[selected], source_x[selected]]
    x = (xx - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.column_stack((x, y, z))
    return points, int(mask.sum()), int(len(points))


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64).copy()
    for coordinate in result:
        if abs(float(coordinate)) > 1e-12:
            if coordinate < 0.0:
                result *= -1.0
            break
    return result


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.92:
        reference = np.asarray([1.0, 0.0, 0.0])
    first = reference - normal * float(np.dot(reference, normal))
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def estimate_planar_frame(value: PlanarFrameInput) -> dict[str, Any]:
    """Estimate independently observable quantities for one object/frame."""

    mask, depth, valid, intrinsics, mapping, registration_valid = _validate_input(value)
    if value.mask_state == "unknown":
        return _unobservable_record(
            frame_index=value.frame_index, mask_state=value.mask_state,
            visibility_state=value.visibility_state, reason="MASK_STATE_UNKNOWN",
            mask_pixel_count=int(mask.sum()),
        )
    if value.visibility_state in BLOCKED_VISIBILITY_STATES:
        return _unobservable_record(
            frame_index=value.frame_index, mask_state=value.mask_state,
            visibility_state=value.visibility_state,
            reason=f"VISIBILITY_{value.visibility_state}",
            mask_pixel_count=int(mask.sum()),
        )
    points, mask_count, point_count = _registered_points(
        mask=mask, depth=depth, valid=valid, intrinsics=intrinsics,
        mapping=mapping, registration_valid=registration_valid,
    )
    denominator = max(mask_count, 1)
    fraction = min(float(point_count / denominator), 1.0)
    if point_count < MIN_TRANSLATION_POINTS:
        record = _unobservable_record(
            frame_index=value.frame_index, mask_state=value.mask_state,
            visibility_state=value.visibility_state,
            reason="INSUFFICIENT_REGISTERED_VISIBLE_DEPTH",
            mask_pixel_count=mask_count,
            registered_valid_depth_count=point_count,
        )
        record["registered_valid_depth_fraction"] = fraction
        return record

    translation = np.median(points, axis=0)
    radial = np.linalg.norm(points - translation, axis=1)
    translation_record = _component(
        "OBSERVABLE_VISIBLE_SURFACE_CENTROID",
        estimate={"xyz_m": translation.tolist()},
        residual={
            "median_radial_spread_m": float(np.median(radial)),
            "p90_radial_spread_m": float(np.quantile(radial, 0.90)),
        },
        semantics="DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_CENTER",
    )
    if point_count < MIN_PLANE_POINTS:
        normal_record = _component(
            "UNOBSERVABLE", reason="INSUFFICIENT_POINTS_FOR_PLANE",
            semantics="DIRECT_VISIBLE_SURFACE_PLANE_NORMAL",
        )
        rotation_record = _component(
            "UNOBSERVABLE", reason="PLANE_NORMAL_UNOBSERVABLE",
            semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
        )
    else:
        centered = points - translation
        _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if normal[2] > 0.0:
            normal *= -1.0
        plane_distance = np.abs(centered @ normal)
        planar_ratio = float(singular[-1] / max(singular[-2], 1e-12))
        if not np.isfinite(planar_ratio) or planar_ratio > MAX_PLANARITY_RATIO:
            normal_record = _component(
                "UNOBSERVABLE", reason="VISIBLE_SUPPORT_NOT_PLANAR_ENOUGH",
                residual={
                    "median_plane_distance_m": float(np.median(plane_distance)),
                    "p90_plane_distance_m": float(np.quantile(plane_distance, 0.90)),
                    "small_to_middle_singular_ratio": planar_ratio,
                },
                semantics="DIRECT_VISIBLE_SURFACE_PLANE_NORMAL",
            )
            rotation_record = _component(
                "UNOBSERVABLE", reason="PLANE_NORMAL_UNOBSERVABLE",
                semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
            )
        else:
            normal_record = _component(
                "OBSERVABLE_DIRECT_VISIBLE_PLANE",
                estimate={
                    "unit_xyz": normal.tolist(),
                    "sign_convention": "Z_NONPOSITIVE_FACES_CAMERA",
                },
                residual={
                    "median_plane_distance_m": float(np.median(plane_distance)),
                    "p90_plane_distance_m": float(np.quantile(plane_distance, 0.90)),
                    "small_to_middle_singular_ratio": planar_ratio,
                },
                semantics="DIRECT_VISIBLE_SURFACE_PLANE_NORMAL",
            )
            if point_count < MIN_ROTATION_POINTS:
                rotation_record = _component(
                    "UNOBSERVABLE", reason="INSUFFICIENT_POINTS_FOR_IN_PLANE_AXIS",
                    semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
                )
            else:
                basis_x, basis_y = _plane_basis(normal)
                coordinates = np.column_stack((centered @ basis_x, centered @ basis_y))
                covariance = coordinates.T @ coordinates / max(len(coordinates) - 1, 1)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                major = eigenvectors[:, -1]
                anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], 1e-15))
                axis_3d = _canonical_axis(major[0] * basis_x + major[1] * basis_y)
                orthogonal = np.abs(centered @ np.cross(normal, axis_3d))
                if not np.isfinite(anisotropy) or anisotropy < MIN_IN_PLANE_ANISOTROPY:
                    rotation_record = _component(
                        "UNOBSERVABLE", reason="VISIBLE_SUPPORT_AXIS_AMBIGUOUS",
                        residual={
                            "major_to_minor_variance_ratio": anisotropy,
                            "median_orthogonal_spread_m": float(np.median(orthogonal)),
                            "p90_orthogonal_spread_m": float(np.quantile(orthogonal, 0.90)),
                        },
                        semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
                    )
                else:
                    angle = math.atan2(
                        float(np.dot(axis_3d, basis_y)),
                        float(np.dot(axis_3d, basis_x)),
                    ) % math.pi
                    rotation_record = _component(
                        "OBSERVABLE_PI_PERIODIC_MAJOR_AXIS",
                        estimate={
                            "axis_unit_xyz": axis_3d.tolist(),
                            "angle_about_plane_normal_rad_mod_pi": float(angle),
                            "symmetry_period_rad": float(math.pi),
                        },
                        residual={
                            "major_to_minor_variance_ratio": anisotropy,
                            "median_orthogonal_spread_m": float(np.median(orthogonal)),
                            "p90_orthogonal_spread_m": float(np.quantile(orthogonal, 0.90)),
                        },
                        semantics="PI_PERIODIC_VISIBLE_MAJOR_AXIS_NOT_SIGNED_OBJECT_YAW",
                    )

    return {
        "frame_index": int(value.frame_index),
        "mask_state": value.mask_state,
        "visibility_state": value.visibility_state,
        "mask_pixel_count": mask_count,
        "registered_valid_depth_count": point_count,
        "registered_valid_depth_fraction": fraction,
        "translation": translation_record,
        "plane_normal": normal_record,
        "in_plane_rotation": rotation_record,
        "hidden_geometry_inferred": False,
    }


def build_observability_document(
    *, session_id: str, object_frames: dict[str, list[dict[str, Any]]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build the contract document without deriving a unified confidence."""

    if tuple(object_frames) != ("playing_card_00", "playing_card_01"):
        raise PlanarObservabilityError(
            "the bounded canary accepts only playing_card_00 and playing_card_01"
        )
    counts: dict[str, dict[str, int]] = {}
    frame_count: int | None = None
    for instance_id, frames in object_frames.items():
        if frame_count is None:
            frame_count = len(frames)
        elif frame_count != len(frames):
            raise PlanarObservabilityError("object frame axes differ")
        counts[instance_id] = {
            component: sum(
                row[component]["observability"] != "UNOBSERVABLE" for row in frames
            )
            for component in ("translation", "plane_normal", "in_plane_rotation")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_DEVELOPMENT_OBSERVABILITY",
        "session_id": session_id,
        "frame_count": int(frame_count or 0),
        "coordinate_domain": DEPTH_REFERENCE,
        "translation_semantics": "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_CENTER",
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_confidence_emitted": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
        "inputs": inputs,
        "objects": [
            {
                "instance_id": instance_id,
                "geometry_class": "PLAYING_CARD_VISIBLE_PLANE",
                "frames": frames,
            }
            for instance_id, frames in object_frames.items()
        ],
        "observable_frame_counts": counts,
        "claim_limit": (
            "Directly visible rectified-left optical-Z surface evidence only. "
            "No object-centre translation, hidden geometry, signed yaw, unified "
            "confidence, external accuracy, Contact truth or Robot authority."
        ),
    }
