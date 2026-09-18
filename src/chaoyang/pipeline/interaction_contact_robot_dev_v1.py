"""Evidence-bounded Interaction, Contact, and Kai22 development helpers.

This module deliberately separates observations from derived hypotheses.  Stereo
samples are visible surface points associated with a finger in image space; they
are not anatomical fingertip ground truth.  Contact uses a finite observed object
patch and never relaxes its distance gate when uncertainty grows.  Robot helpers
only construct a local, tapered development delta and do not create deployment
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


DEPTH_SHAPE = (480, 640)
MASK_SHAPE = (960, 1280)
DEPTH_TO_MASK_SCALE = 2.0
DEPTH_TO_MASK_OFFSET = 0.5
CONTACT_DISTANCE_M = 0.005
MAX_ALIGNMENT_SCALE_DELTA = 0.20
MAX_ALIGNMENT_OFFSET_M = 0.15
MAX_WRIST_TRANSLATION_M = 0.030
MAX_WRIST_ROTATION_DEG = 15.0


class InteractionContactError(ValueError):
    """Raised when an evidence contract would otherwise be silently weakened."""


@dataclass(frozen=True)
class FiniteSurfacePatch:
    """One directly observed planar surface patch in camera coordinates."""

    center_xyz: np.ndarray
    normal_xyz: np.ndarray
    axis_u_xyz: np.ndarray
    axis_v_xyz: np.ndarray
    hull_uv_m: np.ndarray
    plane_residual_p90_m: float
    registered_valid_depth_fraction: float
    source_mask_pixel_count: int


def mask_to_depth_uv(mask_uv: Sequence[float]) -> np.ndarray:
    """Map a 1280x960 SAM pixel centre into the 640x480 Depth domain."""

    value = np.asarray(mask_uv, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise InteractionContactError("one finite mask-domain pixel is required")
    return (value - DEPTH_TO_MASK_OFFSET) / DEPTH_TO_MASK_SCALE


def depth_to_mask_uv(depth_uv: Sequence[float]) -> np.ndarray:
    value = np.asarray(depth_uv, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise InteractionContactError("one finite depth-domain pixel is required")
    return DEPTH_TO_MASK_SCALE * value + DEPTH_TO_MASK_OFFSET


def backproject_pixels(
    xy: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray,
) -> np.ndarray:
    """Back-project pixel-centre coordinates to optical-Z camera coordinates."""

    pixels = np.asarray(xy, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    k = np.asarray(intrinsics, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] != depth.size:
        raise InteractionContactError("xy/depth shape mismatch")
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise InteractionContactError("finite 3x3 intrinsics required")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise InteractionContactError("positive finite optical-Z required")
    x = (pixels[:, 0] - k[0, 2]) * depth / k[0, 0]
    y = (pixels[:, 1] - k[1, 2]) * depth / k[1, 1]
    return np.column_stack((x, y, depth))


def canonicalize_plane_normal(normal: Sequence[float]) -> np.ndarray:
    """Resolve the n/-n plane equivalence by forcing the Z component non-positive."""

    value = np.asarray(normal, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise InteractionContactError("finite 3-vector normal required")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise InteractionContactError("zero plane normal")
    value = value / norm
    if value[2] > 0 or (abs(value[2]) <= 1e-12 and tuple(value) < tuple(-value)):
        value = -value
    return value


def canonicalize_normal_sequence(normals: np.ndarray) -> np.ndarray:
    """Canonicalize valid normals without filling unavailable frames."""

    raw = np.asarray(normals, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise InteractionContactError("normal sequence must be [T,3]")
    result = np.full_like(raw, np.nan)
    for index, normal in enumerate(raw):
        if np.isfinite(normal).all() and np.linalg.norm(normal) > 1e-12:
            result[index] = canonicalize_plane_normal(normal)
    return result


def temporal_geometry_diagnostics(
    centers_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    *,
    comparable_visibility: np.ndarray,
    camera_motion_compensation_available: bool,
) -> dict[str, Any]:
    """Report non-authoritative temporal motion diagnostics.

    A visible-surface centroid is not an object-fixed centre.  Consequently these
    statistics never contain a pass/fail decision.  Missing camera-motion evidence
    explicitly downgrades the cross-frame interpretation.
    """

    centers = np.asarray(centers_xyz, dtype=np.float64)
    normals = canonicalize_normal_sequence(normals_xyz)
    comparable = np.asarray(comparable_visibility, dtype=bool)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise InteractionContactError("center sequence must be [T,3]")
    if normals.shape != centers.shape or comparable.shape != (centers.shape[0],):
        raise InteractionContactError("temporal diagnostic axes differ")
    center_steps: list[float] = []
    normal_steps: list[float] = []
    for index in range(1, centers.shape[0]):
        if not (comparable[index - 1] and comparable[index]):
            continue
        if np.isfinite(centers[index - 1:index + 1]).all():
            center_steps.append(float(np.linalg.norm(centers[index] - centers[index - 1])))
        if np.isfinite(normals[index - 1:index + 1]).all():
            cosine = float(np.clip(np.dot(normals[index], normals[index - 1]), -1.0, 1.0))
            normal_steps.append(float(np.degrees(np.arccos(cosine))))
    return {
        "visible_surface_center_semantics": "NOT_OBJECT_FIXED_CENTER",
        "interpretation": (
            "DIAGNOSTIC_CAMERA_MOTION_COMPENSATED"
            if camera_motion_compensation_available
            else "DIAGNOSTIC_ONLY_NO_TRUSTED_CAMERA_MOTION_COMPENSATION"
        ),
        "failure_gate_applied": False,
        "comparable_step_count": len(center_steps),
        "camera_space_visible_center_step_p95_m": (
            float(np.percentile(center_steps, 95)) if center_steps else None
        ),
        "sign_canonicalized_normal_step_p95_deg": (
            float(np.percentile(normal_steps, 95)) if normal_steps else None
        ),
    }


def _orthonormal_plane_basis(normal: np.ndarray, axis_hint: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n = canonicalize_plane_normal(normal)
    if axis_hint is not None:
        u = np.asarray(axis_hint, dtype=np.float64)
        if u.shape == (3,) and np.isfinite(u).all():
            u = u - n * float(np.dot(u, n))
        else:
            u = np.zeros(3, dtype=np.float64)
    else:
        u = np.zeros(3, dtype=np.float64)
    if np.linalg.norm(u) <= 1e-9:
        candidate = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(candidate, n))) > 0.9:
            candidate = np.asarray([0.0, 1.0, 0.0])
        u = candidate - n * float(np.dot(candidate, n))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def build_finite_surface_patch(
    *,
    center_xyz: Sequence[float],
    normal_xyz: Sequence[float],
    axis_hint_xyz: Sequence[float] | None,
    mask_1280: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    intrinsics: np.ndarray,
    plane_residual_p90_m: float,
    registered_valid_depth_fraction: float,
) -> FiniteSurfacePatch | None:
    """Reconstruct a finite visible patch hull; never extend it to an infinite plane."""

    mask = np.asarray(mask_1280, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.asarray(depth_valid, dtype=bool)
    if mask.shape != MASK_SHAPE or depth.shape != DEPTH_SHAPE or valid.shape != DEPTH_SHAPE:
        raise InteractionContactError("finite patch input domain mismatch")
    center = np.asarray(center_xyz, dtype=np.float64)
    normal = canonicalize_plane_normal(normal_xyz)
    axis = None if axis_hint_xyz is None else np.asarray(axis_hint_xyz, dtype=np.float64)
    u, v = _orthonormal_plane_basis(normal, axis)
    # Nearest-neighbour centre mapping is exact for the frozen analytic resize relation.
    mask_depth = cv2.resize(mask.astype(np.uint8), (DEPTH_SHAPE[1], DEPTH_SHAPE[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    selected = mask_depth & valid & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(selected)
    if xx.size < 12:
        return None
    points = backproject_pixels(np.column_stack((xx, yy)), depth[yy, xx], intrinsics)
    relative = points - center
    plane_distance = relative @ normal
    inliers = np.abs(plane_distance) <= max(0.003, 3.0 * float(plane_residual_p90_m))
    if int(np.count_nonzero(inliers)) < 12:
        return None
    local = np.column_stack((relative[inliers] @ u, relative[inliers] @ v)).astype(np.float32)
    hull = cv2.convexHull(local.reshape(-1, 1, 2)).reshape(-1, 2)
    if hull.shape[0] < 3 or abs(float(cv2.contourArea(hull.reshape(-1, 1, 2)))) <= 1e-8:
        return None
    return FiniteSurfacePatch(
        center_xyz=center,
        normal_xyz=normal,
        axis_u_xyz=u,
        axis_v_xyz=v,
        hull_uv_m=np.asarray(hull, dtype=np.float64),
        plane_residual_p90_m=float(plane_residual_p90_m),
        registered_valid_depth_fraction=float(registered_valid_depth_fraction),
        source_mask_pixel_count=int(np.count_nonzero(mask)),
    )


def point_to_finite_patch(point_xyz: Sequence[float], patch: FiniteSurfacePatch) -> dict[str, Any]:
    """Distance from a 3D point to the observed finite patch, not its infinite plane."""

    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise InteractionContactError("finite 3D point required")
    delta = point - patch.center_xyz
    signed_plane = float(np.dot(delta, patch.normal_xyz))
    local = np.asarray([
        float(np.dot(delta, patch.axis_u_xyz)),
        float(np.dot(delta, patch.axis_v_xyz)),
    ])
    hull = patch.hull_uv_m.astype(np.float32).reshape(-1, 1, 2)
    signed_inside_distance = float(cv2.pointPolygonTest(hull, tuple(local.astype(float)), True))
    outside = max(0.0, -signed_inside_distance)
    finite_distance = float(np.hypot(signed_plane, outside))
    return {
        "object_plane_signed_distance_m": signed_plane,
        "object_plane_local_coordinate_m": local.tolist(),
        "inside_visible_patch": signed_inside_distance >= 0.0,
        "distance_to_visible_patch_boundary_m": abs(signed_inside_distance),
        "finite_patch_distance_m": finite_distance,
    }


def sample_finger_associated_visible_surface(
    *,
    source_pixel_uv_1280: Sequence[float],
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    lr_consistent: np.ndarray,
    lr_residual_px: np.ndarray,
    intrinsics: np.ndarray,
    hand_mask_1280: np.ndarray,
    sleeve_mask_1280: np.ndarray | None,
    object_union_1280: np.ndarray,
    associated_hand: str,
    associated_finger: str,
    radius_depth_px: int = 4,
    anatomical_tip_evidence: bool = False,
) -> dict[str, Any]:
    """Sample a visible surface near a projected finger with conservative qualification."""

    pixel = np.asarray(source_pixel_uv_1280, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.asarray(depth_valid, dtype=bool)
    lr_ok = np.asarray(lr_consistent, dtype=bool)
    residual = np.asarray(lr_residual_px, dtype=np.float64)
    hand = np.asarray(hand_mask_1280, dtype=bool)
    objects = np.asarray(object_union_1280, dtype=bool)
    sleeve = None if sleeve_mask_1280 is None else np.asarray(sleeve_mask_1280, dtype=bool)
    if depth.shape != DEPTH_SHAPE or valid.shape != DEPTH_SHAPE or lr_ok.shape != DEPTH_SHAPE or residual.shape != DEPTH_SHAPE:
        raise InteractionContactError("Depth evidence domain mismatch")
    if hand.shape != MASK_SHAPE or objects.shape != MASK_SHAPE or (sleeve is not None and sleeve.shape != MASK_SHAPE):
        raise InteractionContactError("mask evidence domain mismatch")
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return {
            "status": "UNKNOWN", "reason": "NONFINITE_PROJECTED_FINGER",
            "associated_hand": associated_hand, "associated_finger": associated_finger,
            "fingertip_surface_observation": False,
        }
    depth_uv = mask_to_depth_uv(pixel)
    x0, y0 = np.rint(depth_uv).astype(int)
    if not (0 <= x0 < DEPTH_SHAPE[1] and 0 <= y0 < DEPTH_SHAPE[0]):
        return {
            "status": "UNKNOWN", "reason": "PROJECTED_FINGER_OUT_OF_FRAME",
            "source_pixel_uv": pixel.tolist(), "associated_hand": associated_hand,
            "associated_finger": associated_finger, "fingertip_surface_observation": False,
        }
    yy, xx = np.ogrid[:DEPTH_SHAPE[0], :DEPTH_SHAPE[1]]
    roi = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius_depth_px ** 2
    # Sample semantics at the exact analytic corresponding SAM pixel centres.
    ys, xs = np.nonzero(roi)
    sx = np.clip(np.rint(DEPTH_TO_MASK_SCALE * xs + DEPTH_TO_MASK_OFFSET).astype(int), 0, MASK_SHAPE[1] - 1)
    sy = np.clip(np.rint(DEPTH_TO_MASK_SCALE * ys + DEPTH_TO_MASK_OFFSET).astype(int), 0, MASK_SHAPE[0] - 1)
    semantic_hand = hand[sy, sx]
    semantic_object = objects[sy, sx]
    admitted = valid[ys, xs] & lr_ok[ys, xs] & np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0) & semantic_hand & ~semantic_object
    roi_count = int(xs.size)
    admitted_count = int(np.count_nonzero(admitted))
    hand_purity = float(np.mean(semantic_hand)) if roi_count else 0.0
    object_fraction = float(np.mean(semantic_object)) if roi_count else 1.0
    if admitted_count < 6 or hand_purity < 0.45 or object_fraction > 0.20:
        return {
            "status": "UNKNOWN", "reason": "LOCAL_ROLE_OR_DEPTH_ADMISSION_FAILED",
            "source_pixel_uv": pixel.tolist(), "associated_hand": associated_hand,
            "associated_finger": associated_finger, "surface_role": "unknown",
            "association_quality": {
                "admitted_pixel_count": admitted_count, "roi_pixel_count": roi_count,
                "hand_mask_purity": hand_purity, "object_mask_fraction": object_fraction,
            },
            "fingertip_surface_observation": False,
        }
    selected_x, selected_y = xs[admitted], ys[admitted]
    selected_depth = depth[selected_y, selected_x]
    median_depth = float(np.median(selected_depth))
    median_pixel = np.asarray([float(np.median(selected_x)), float(np.median(selected_y))])
    point = backproject_pixels(median_pixel.reshape(1, 2), np.asarray([median_depth]), intrinsics)[0]
    mad = float(1.4826 * np.median(np.abs(selected_depth - median_depth)))
    depth_span_p90_p10 = float(
        np.percentile(selected_depth, 90) - np.percentile(selected_depth, 10)
    )
    lr_values = residual[selected_y, selected_x]
    finite_lr = lr_values[np.isfinite(lr_values)]
    surface_role = "skin"
    if sleeve is not None and float(np.mean(sleeve[sy[admitted], sx[admitted]])) >= 0.50:
        surface_role = "sleeve"
    qualified_tip = bool(
        anatomical_tip_evidence
        and surface_role == "skin"
        and hand_purity >= 0.70
        and object_fraction <= 0.05
        and admitted_count >= 10
        and mad <= 0.008
        and depth_span_p90_p10 <= 0.015
    )
    return {
        "status": "OBSERVED_VISIBLE_SURFACE",
        "reason": None,
        "surface_point_xyz": point.tolist(),
        "source_pixel_uv": pixel.tolist(),
        "sample_pixel_uv_depth_domain": median_pixel.tolist(),
        "associated_hand": associated_hand,
        "associated_finger": associated_finger,
        "surface_role": surface_role,
        "association_quality": {
            "admitted_pixel_count": admitted_count, "roi_pixel_count": roi_count,
            "hand_mask_purity": hand_purity, "object_mask_fraction": object_fraction,
        },
        "depth_quality": {
            "local_depth_median_m": median_depth,
            "local_depth_robust_sigma_m": mad,
            "local_depth_p90_p10_m": depth_span_p90_p10,
            "lr_residual_median_px": float(np.median(finite_lr)) if finite_lr.size else None,
            "lr_consistent_fraction": 1.0,
        },
        "fingertip_surface_observation": qualified_tip,
    }


def fit_human_stereo_ray_depth_alignment(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixed_bias_mm: float | None = None,
    minimum_train_rows: int = 30,
    minimum_holdout_rows: int = 8,
) -> dict[str, Any]:
    """Fit a bounded session-static ray-depth scale/offset on non-contact rows only."""

    if fixed_bias_mm is not None:
        raise InteractionContactError("fixed depth bias, including 48 mm, is forbidden")
    admitted: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("contact_or_object_fit_used") is True or row.get("near_task_object") is True:
            continue
        if row.get("source_kind") != "NON_CONTACT_VISIBLE_HAND_SURFACE":
            continue
        hz, sz = row.get("hawor_ray_depth_m"), row.get("stereo_surface_depth_m")
        if isinstance(hz, (int, float)) and isinstance(sz, (int, float)) and np.isfinite([hz, sz]).all() and hz > 0 and sz > 0:
            admitted.append(row)
    train = [row for row in admitted if int(row["frame_id"]) % 5 != 0]
    holdout = [row for row in admitted if int(row["frame_id"]) % 5 == 0]
    base = {
        "schema_version": "HUMAN_STEREO_ALIGNMENT_CHECK_V1",
        "fit_scope": "SESSION_STATIC_RAY_DEPTH_SCALE_OFFSET",
        "fit_data_policy": "NON_CONTACT_VISIBLE_HAND_SURFACE_ONLY",
        "contact_or_object_fit_used": False,
        "fixed_48mm_bias_used": False,
        "train_row_count": len(train),
        "holdout_row_count": len(holdout),
    }
    if len(train) < minimum_train_rows or len(holdout) < minimum_holdout_rows:
        return {
            **base, "status": "BLOCKED_INSUFFICIENT_NON_CONTACT_ALIGNMENT_EVIDENCE",
            "metric_translation_authorized": False, "fit": None,
            "heldout": None,
        }
    x = np.asarray([float(row["hawor_ray_depth_m"]) for row in train])
    y = np.asarray([float(row["stereo_surface_depth_m"]) for row in train])
    keep = np.ones(x.size, dtype=bool)
    scale, offset = 1.0, 0.0
    for _ in range(4):
        design = np.column_stack((x[keep], np.ones(int(np.count_nonzero(keep)))))
        estimate, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        scale = float(np.clip(estimate[0], 1.0 - MAX_ALIGNMENT_SCALE_DELTA, 1.0 + MAX_ALIGNMENT_SCALE_DELTA))
        offset = float(np.clip(estimate[1], -MAX_ALIGNMENT_OFFSET_M, MAX_ALIGNMENT_OFFSET_M))
        residual = y - (scale * x + offset)
        robust = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        new_keep = np.abs(residual - np.median(residual)) <= max(0.010, 3.0 * robust)
        if np.array_equal(new_keep, keep) or int(np.count_nonzero(new_keep)) < minimum_train_rows:
            break
        keep = new_keep
    hx = np.asarray([float(row["hawor_ray_depth_m"]) for row in holdout])
    hy = np.asarray([float(row["stereo_surface_depth_m"]) for row in holdout])
    hres = np.abs(hy - (scale * hx + offset))
    heldout = {
        "median_abs_residual_m": float(np.median(hres)),
        "p90_abs_residual_m": float(np.percentile(hres, 90)),
        "frame_ids": sorted({int(row["frame_id"]) for row in holdout}),
    }
    passed = bool(
        0.8 <= scale <= 1.2
        and abs(offset) <= MAX_ALIGNMENT_OFFSET_M
        and heldout["median_abs_residual_m"] <= 0.015
        and heldout["p90_abs_residual_m"] <= 0.030
    )
    return {
        **base,
        "status": "PASS_DEVELOPMENT_ALIGNMENT" if passed else "REJECTED_HELDOUT_ALIGNMENT",
        "metric_translation_authorized": passed,
        "fit": {"scale": scale, "offset_m": offset, "retained_train_rows": int(np.count_nonzero(keep))},
        "heldout": heldout,
    }


def apply_ray_depth_alignment(point_xyz: Sequence[float], alignment: Mapping[str, Any]) -> np.ndarray:
    """Apply the admitted scalar ray-depth transform without changing the camera ray."""

    point = np.asarray(point_xyz, dtype=np.float64)
    fit = alignment.get("fit")
    if alignment.get("metric_translation_authorized") is not True or not isinstance(fit, Mapping):
        raise InteractionContactError("alignment does not authorize metric translation")
    z = float(point[2])
    if point.shape != (3,) or not np.isfinite(point).all() or z <= 0:
        raise InteractionContactError("positive finite camera point required")
    target_z = float(fit["scale"]) * z + float(fit["offset_m"])
    if target_z <= 0:
        raise InteractionContactError("aligned ray depth became non-positive")
    return point * (target_z / z)


def classify_contact_candidate(
    *,
    finite_patch_distance_m: float | None,
    inside_visible_patch: bool,
    local_depth_robust_sigma_m: float | None,
    plane_residual_p90_m: float | None,
    lr_residual_median_px: float | None,
    approach_supported: bool,
    co_motion_supported: bool,
    tactile_supported: bool,
    uncertainty_limit_m: float = 0.005,
) -> dict[str, Any]:
    """Classify Contact without allowing uncertainty to enlarge the 5 mm gate."""

    terms = (local_depth_robust_sigma_m, plane_residual_p90_m, lr_residual_median_px)
    uncertainty_complete = all(value is not None and np.isfinite(value) for value in terms)
    uncertainty_m = None
    if uncertainty_complete:
        uncertainty_m = float(np.hypot(float(local_depth_robust_sigma_m), float(plane_residual_p90_m)))
    geometric_proximity = bool(
        finite_patch_distance_m is not None
        and np.isfinite(finite_patch_distance_m)
        and float(finite_patch_distance_m) <= CONTACT_DISTANCE_M
        and inside_visible_patch
    )
    uncertainty_admitted = bool(
        uncertainty_complete
        and uncertainty_m is not None
        and uncertainty_m <= uncertainty_limit_m
        and float(lr_residual_median_px) <= 1.0
    )
    interval_could_cover_contact = bool(
        finite_patch_distance_m is not None
        and uncertainty_m is not None
        and float(finite_patch_distance_m) - 3.0 * uncertainty_m <= CONTACT_DISTANCE_M
        and inside_visible_patch
    )
    if finite_patch_distance_m is None or not uncertainty_complete:
        state = "UNKNOWN"
    elif not inside_visible_patch:
        state = "NO_EVIDENCE"
    elif geometric_proximity and uncertainty_admitted:
        state = "CONTACT_CANDIDATE"
    elif interval_could_cover_contact and not uncertainty_admitted:
        state = "NEAR_UNCERTAIN"
    elif approach_supported:
        state = "APPROACH"
    else:
        state = "NO_EVIDENCE"
    if state == "CONTACT_CANDIDATE" and co_motion_supported:
        state = "CO_MOTION_SUPPORTED"
    evidence = 0.0
    evidence += 0.45 if geometric_proximity else 0.0
    evidence += 0.25 if uncertainty_admitted else 0.0
    evidence += 0.15 if approach_supported else 0.0
    evidence += 0.10 if co_motion_supported else 0.0
    evidence += 0.05 if tactile_supported else 0.0
    return {
        "state": state,
        "geometric_proximity_pass": geometric_proximity,
        "uncertainty_admission_pass": uncertainty_admitted,
        "uncertainty_complete": uncertainty_complete,
        "combined_internal_uncertainty_m": uncertainty_m,
        "distance_gate_m": CONTACT_DISTANCE_M,
        "distance_gate_was_uncertainty_expanded": False,
        "support_score": round(evidence, 6),
        "support_score_semantics": "UNCALIBRATED_HEURISTIC_NOT_PROBABILITY_NOT_GROUND_TRUTH",
        "tactile_effect": "SUPPORT_ONLY" if tactile_supported else "NONE",
    }


def build_pair_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    accepted_states: Iterable[str] = ("CONTACT_CANDIDATE", "GRASP_SUPPORTED", "CO_MOTION_SUPPORTED", "TRANSPORT"),
    minimum_frames: int = 5,
    maximum_dt_s: float = 0.050,
) -> list[dict[str, Any]]:
    """Build continuous windows without crossing identity or frame/time gaps."""

    accepted = set(accepted_states)
    selected = [row for row in rows if row.get("state") in accepted]
    selected.sort(key=lambda row: (str(row.get("hand_id")), str(row.get("finger_id")), str(row.get("object_id")), int(row.get("frame_id", -1))))
    windows: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    last_key: tuple[str, str, str] | None = None
    last_frame: int | None = None
    last_time: float | None = None
    for row in selected:
        key = (str(row["hand_id"]), str(row["finger_id"]), str(row["object_id"]))
        frame, timestamp = int(row["frame_id"]), float(row["timestamp_s"])
        continuous = (
            current and key == last_key and frame == int(last_frame) + 1
            and 0.0 < timestamp - float(last_time) <= maximum_dt_s
        )
        if not continuous:
            if current:
                windows.append(current)
            current = []
        current.append(row)
        last_key, last_frame, last_time = key, frame, timestamp
    if current:
        windows.append(current)
    return [{
        "hand_id": window[0]["hand_id"],
        "finger_id": window[0]["finger_id"],
        "object_id": window[0]["object_id"],
        "start_frame": int(window[0]["frame_id"]),
        "end_frame": int(window[-1]["frame_id"]),
        "frame_ids": [int(row["frame_id"]) for row in window],
        "timestamp_start_s": float(window[0]["timestamp_s"]),
        "timestamp_end_s": float(window[-1]["timestamp_s"]),
        "terminal": "ADMITTED_LOCAL_WINDOW" if len(window) >= minimum_frames else "BLOCKED_SHORT_WINDOW",
    } for window in windows]


def transition_taper_weights(
    timestamps_s: Sequence[float], windows: Sequence[Mapping[str, Any]], *, transition_s: float = 0.2,
) -> np.ndarray:
    """Return bounded taper weights without generating new Contact labels."""

    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise InteractionContactError("strictly increasing finite timestamps required")
    weights = np.zeros(times.size, dtype=np.float64)
    for window in windows:
        if window.get("terminal") != "ADMITTED_LOCAL_WINDOW":
            continue
        start, end = float(window["timestamp_start_s"]), float(window["timestamp_end_s"])
        core = (times >= start) & (times <= end)
        before = (times < start) & (times >= start - transition_s)
        after = (times > end) & (times <= end + transition_s)
        weights[core] = 1.0
        if transition_s > 0:
            weights[before] = np.maximum(weights[before], (times[before] - (start - transition_s)) / transition_s)
            weights[after] = np.maximum(weights[after], ((end + transition_s) - times[after]) / transition_s)
    return np.clip(weights, 0.0, 1.0)


def timestamp_motion_diagnostics(values: np.ndarray, timestamps_s: Sequence[float], valid: np.ndarray) -> dict[str, Any]:
    """Compute physical-time derivatives only within contiguous valid segments."""

    q = np.asarray(values, dtype=np.float64)
    t = np.asarray(timestamps_s, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if q.shape[0] != t.size or mask.shape != (t.size,):
        raise InteractionContactError("motion diagnostic axis mismatch")
    speeds: list[float] = []
    accelerations: list[float] = []
    ids = np.flatnonzero(mask & np.isfinite(q).all(axis=tuple(range(1, q.ndim))))
    for segment in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
        if segment.size < 2:
            continue
        dt = np.diff(t[segment])
        if np.any(dt <= 0):
            raise InteractionContactError("timestamps must increase")
        velocity = np.diff(q[segment], axis=0) / dt.reshape((-1,) + (1,) * (q.ndim - 1))
        speeds.extend(np.abs(velocity).reshape(-1).tolist())
        if segment.size >= 3:
            mid_dt = 0.5 * (dt[1:] + dt[:-1])
            acceleration = np.diff(velocity, axis=0) / mid_dt.reshape((-1,) + (1,) * (q.ndim - 1))
            accelerations.extend(np.abs(acceleration).reshape(-1).tolist())
    return {
        "max_abs_velocity_rad_s": max(speeds, default=None),
        "p95_abs_velocity_rad_s": float(np.percentile(speeds, 95)) if speeds else None,
        "max_abs_acceleration_rad_s2": max(accelerations, default=None),
        "p95_abs_acceleration_rad_s2": float(np.percentile(accelerations, 95)) if accelerations else None,
        "legacy_per_frame_metric_authority": "DIAGNOSTIC_ONLY",
    }


def evaluate_frozen_frames(
    before: Mapping[int, float | None], after: Mapping[int, float | None], frozen_frame_ids: Sequence[int],
) -> dict[str, Any]:
    """Fair before/after metric: the exact frozen frame set must remain evaluable."""

    ids = tuple(int(value) for value in frozen_frame_ids)
    before_valid = [frame for frame in ids if before.get(frame) is not None and np.isfinite(before[frame])]
    after_valid = [frame for frame in ids if after.get(frame) is not None and np.isfinite(after[frame])]
    same = before_valid == after_valid == list(ids)
    return {
        "frozen_frame_ids": list(ids),
        "before_valid_frame_ids": before_valid,
        "after_valid_frame_ids": after_valid,
        "coverage_preserved": same,
        "comparison_admitted": same,
        "before_median": float(np.median([before[frame] for frame in ids])) if same else None,
        "after_median": float(np.median([after[frame] for frame in ids])) if same else None,
    }


def collision_scope_result(
    *, robot_self_collision: str, observed_object_patch: str,
) -> dict[str, str]:
    """Keep verified local collision scopes separate from unknown full geometry."""

    return {
        "collision_scope": "ROBOT_SELF_PLUS_OBSERVED_OBJECT_PATCH",
        "robot_self_collision": robot_self_collision,
        "observed_object_patch": observed_object_patch,
        "full_object_collision": "UNVERIFIED",
        "full_object_environment": "UNVERIFIED",
    }
