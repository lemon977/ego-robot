"""Session-static table-plane plumbing primitives.

The producer fits exactly one plane from geometry-bound AprilTag corner sets.
It cannot expand table support or re-anchor per frame without independent,
content-addressed evidence.  Physical gates deliberately have no confidence
input, keeping geometric violations separate from estimator quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Sequence

import numpy as np

from pipeline.table_plane_drift_reanchor import (
    Plane,
    TablePlaneError,
    finite_cylinder_sdf,
    signed_distances,
)


LOCAL_SUPPORT_SEMANTICS = "APRILTAG_QUAD_ERODED_3PX_ONLY"
REQUIRED_ANCHOR_FRAMES = 15
REQUIRED_ANCHOR_POINTS = 60
TAG_REPROJECTION_MAX_PX = 1.0
RESIDUAL_P95_MAX_M = 0.01
RESIDUAL_MAX_M = 0.015
HEX64_PATTERN = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class StaticPlaneFit:
    plane: Plane
    residual_mean_abs_m: float
    residual_p95_abs_m: float
    residual_max_abs_m: float
    loo_normal_delta_max_deg: float
    loo_offset_delta_max_m: float
    per_frame_residuals: tuple[dict[str, int | float], ...]


def tag_corners_local(tag_size_m: float) -> np.ndarray:
    if not np.isfinite(tag_size_m) or tag_size_m <= 0:
        raise TablePlaneError("tag size must be positive and finite")
    half = float(tag_size_m) * 0.5
    return np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if matrix.shape != (4, 4) or values.ndim != 2 or values.shape[1] != 3:
        raise TablePlaneError("point transform shape drift")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise TablePlaneError("point transform contains non-finite values")
    homogeneous = np.concatenate((values, np.ones((len(values), 1))), axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def _fit_plane(points: np.ndarray) -> Plane:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise TablePlaneError("static plane requires at least three 3D points")
    if not np.all(np.isfinite(values)):
        raise TablePlaneError("static plane points contain non-finite values")
    center = np.mean(values, axis=0)
    _, singular_values, right = np.linalg.svd(values - center, full_matrices=False)
    if len(singular_values) != 3 or singular_values[1] <= 1e-10:
        raise TablePlaneError("static plane anchors are degenerate")
    normal = right[-1].copy()
    normal /= np.linalg.norm(normal)
    if normal[1] > 0:
        normal *= -1.0
    return Plane(normal, -float(normal @ center))


def fit_session_static_plane(
    frame_indices: Sequence[int], corner_points_world: np.ndarray
) -> StaticPlaneFit:
    """Fit one plane to all frames and quantify leave-one-frame-out stability."""

    frames = tuple(int(item) for item in frame_indices)
    points = np.asarray(corner_points_world, dtype=np.float64)
    if points.shape != (len(frames), 4, 3) or len(frames) < 3:
        raise TablePlaneError("static-plane anchor shape drift")
    if len(set(frames)) != len(frames) or not np.all(np.isfinite(points)):
        raise TablePlaneError("static-plane anchors are duplicate or non-finite")
    plane = _fit_plane(points.reshape(-1, 3))
    residuals = signed_distances(plane, points)
    absolute = np.abs(residuals)
    per_frame = tuple(
        {
            "frame_index": frame,
            "corner_count": 4,
            "signed_mean_m": float(np.mean(residuals[index])),
            "abs_max_m": float(np.max(absolute[index])),
        }
        for index, frame in enumerate(frames)
    )
    normal_deltas = []
    offset_deltas = []
    for excluded in range(len(frames)):
        keep = np.arange(len(frames)) != excluded
        loo = _fit_plane(points[keep].reshape(-1, 3))
        dot = float(np.clip(loo.normal @ plane.normal, -1.0, 1.0))
        normal_deltas.append(float(np.degrees(np.arccos(dot))))
        offset_deltas.append(abs(float(loo.offset_m - plane.offset_m)))
    return StaticPlaneFit(
        plane=plane,
        residual_mean_abs_m=float(np.mean(absolute)),
        residual_p95_abs_m=float(np.percentile(absolute, 95)),
        residual_max_abs_m=float(np.max(absolute)),
        loo_normal_delta_max_deg=float(np.max(normal_deltas)),
        loo_offset_delta_max_m=float(np.max(offset_deltas)),
        per_frame_residuals=per_frame,
    )


def require_static_fit_pass(
    fit: StaticPlaneFit,
    *,
    anchor_frame_count: int,
    anchor_point_count: int,
    tag_corner_reprojection_max_px: float,
) -> None:
    exact_ints = type(anchor_frame_count) is int and type(anchor_point_count) is int
    if not exact_ints:
        raise TablePlaneError("anchor counts must be exact integers")
    if anchor_frame_count != REQUIRED_ANCHOR_FRAMES or anchor_point_count != REQUIRED_ANCHOR_POINTS:
        raise TablePlaneError("static plane requires exactly 15 frames / 60 corner anchors")
    if (
        not np.isfinite(tag_corner_reprojection_max_px)
        or tag_corner_reprojection_max_px > TAG_REPROJECTION_MAX_PX
        or fit.residual_p95_abs_m > RESIDUAL_P95_MAX_M
        or fit.residual_max_abs_m > RESIDUAL_MAX_M
    ):
        raise TablePlaneError("static table-plane quality gate failed")


def _validate_content_ref(evidence_ref: dict[str, Any], payload: bytes) -> None:
    if not isinstance(evidence_ref, dict) or set(evidence_ref) != {"path", "bytes", "sha256"}:
        raise TablePlaneError("evidence ref fields drift")
    if (
        not isinstance(evidence_ref["path"], str)
        or not evidence_ref["path"]
        or type(evidence_ref["bytes"]) is not int
        or evidence_ref["bytes"] <= 0
        or not isinstance(evidence_ref["sha256"], str)
        or not HEX64_PATTERN.fullmatch(evidence_ref["sha256"])
        or not isinstance(payload, bytes)
        or len(payload) != evidence_ref["bytes"]
        or hashlib.sha256(payload).hexdigest() != evidence_ref["sha256"]
    ):
        raise TablePlaneError("evidence ref is not bound to supplied ordinary-file bytes")


def validate_support_scope(
    semantics: str,
    *,
    expansion_requested: bool,
    expansion_evidence_ref: dict[str, Any] | None,
    expansion_evidence_payload: bytes | None,
) -> None:
    if semantics != LOCAL_SUPPORT_SEMANTICS:
        raise TablePlaneError("unknown table support semantics")
    if type(expansion_requested) is not bool:
        raise TablePlaneError("support expansion flag must be strict bool")
    if expansion_requested and (
        expansion_evidence_ref is None or expansion_evidence_payload is None
    ):
        raise TablePlaneError("table support expansion requires independent geometry evidence")
    if expansion_requested:
        _validate_content_ref(expansion_evidence_ref, expansion_evidence_payload)
    if not expansion_requested and (
        expansion_evidence_ref is not None or expansion_evidence_payload is not None
    ):
        raise TablePlaneError("unused table support expansion evidence is forbidden")


def per_frame_reanchor_allowed(
    *,
    contact_evidence_valid: bool,
    evidence_ref: dict[str, Any] | None,
    evidence_payload: bytes | None,
) -> bool:
    if type(contact_evidence_valid) is not bool:
        raise TablePlaneError("contact evidence validity must be strict bool")
    if not contact_evidence_valid:
        if evidence_ref is not None or evidence_payload is not None:
            raise TablePlaneError("invalid contact evidence cannot authorize reanchor")
        return False
    if evidence_ref is None or evidence_payload is None:
        raise TablePlaneError("reanchor requires a content-addressed evidence ref")
    _validate_content_ref(evidence_ref, evidence_payload)
    return True


def evaluate_physical_gates(
    plane: Plane,
    *,
    wrist_points_world: np.ndarray | None,
    hand_points_world: np.ndarray | None,
    fingertip_points_world: np.ndarray | None,
    object_to_world: np.ndarray | None,
    cylinder_radius_m: float | None,
    cylinder_height_m: float | None,
) -> dict[str, Any]:
    """Evaluate geometry-only gates; confidence is intentionally absent."""

    supplied = (
        wrist_points_world,
        hand_points_world,
        fingertip_points_world,
        object_to_world,
        cylinder_radius_m,
        cylinder_height_m,
    )
    if all(item is None for item in supplied):
        return {
            "status": "NOT_EVALUATED_MISSING_REVIEWED_ROBOT_GEOMETRY",
            "confidence_consumed": False,
            "wrist_table_signed_distance_m": None,
            "wrist_below_table": None,
            "hand_table_signed_distance_m": None,
            "hand_point_below_table": None,
            "fingertip_cylinder_sdf_m": None,
            "fingertip_inside_cylinder": None,
        }
    if any(item is None for item in supplied):
        raise TablePlaneError("physical gates require a complete reviewed geometry bundle")
    wrists = np.asarray(wrist_points_world, dtype=np.float64)
    hands = np.asarray(hand_points_world, dtype=np.float64)
    tips = np.asarray(fingertip_points_world, dtype=np.float64)
    if wrists.ndim != 2 or hands.ndim != 2 or tips.ndim != 2:
        raise TablePlaneError("physical gate point arrays must be 2D")
    if wrists.shape[1:] != (3,) or hands.shape[1:] != (3,) or tips.shape[1:] != (3,):
        raise TablePlaneError("physical gate points must be Nx3")
    wrist_distance = signed_distances(plane, wrists)
    hand_distance = signed_distances(plane, hands)
    sdf = finite_cylinder_sdf(
        tips,
        np.asarray(object_to_world, dtype=np.float64),
        float(cylinder_radius_m),
        float(cylinder_height_m),
    )
    wrist_below = wrist_distance < 0.0
    hand_below = hand_distance < 0.0
    inside = sdf < 0.0
    return {
        "status": "FAIL_PHYSICAL_GEOMETRY" if np.any(wrist_below) or np.any(hand_below) or np.any(inside) else "PASS_PHYSICAL_GEOMETRY",
        "confidence_consumed": False,
        "wrist_table_signed_distance_m": wrist_distance.tolist(),
        "wrist_below_table": wrist_below.tolist(),
        "hand_table_signed_distance_m": hand_distance.tolist(),
        "hand_point_below_table": hand_below.tolist(),
        "fingertip_cylinder_sdf_m": sdf.tolist(),
        "fingertip_inside_cylinder": inside.tolist(),
    }
