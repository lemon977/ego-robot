"""Fail-closed wrist calibration and dual-representation primitives.

The transform convention in this module is always ``T_A_B``: a point expressed
in frame B is mapped into frame A.  In particular::

    T_camera_wrist = T_camera_controller @ T_controller_wrist

An anatomical wrist centre and a currently visible wrist surface point are
different observations.  The surface helper below therefore refuses to create
a 3-D point unless every independent admission flag is true.  It can still
publish a region-only registration result without inventing geometry.

All fusion is relative to the static-calibrated prior for the *same frame*.
There is deliberately no previous-frame correction state, so the bounded
development correction cannot accumulate and silently replace calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


SIDES = ("left", "right")
TEMPORAL_AUTHORITIES = {
    "CAUSAL_CURRENT",
    "OFFLINE_NONCAUSAL",
    "INFERRED",
    "UNKNOWN_TEMPORAL_AUTHORITY",
}
UNKNOWN_UNCERTAINTY = "UNKNOWN"


class WristDualRepresentationError(ValueError):
    """Raised when wrist evidence is malformed or semantically unsafe."""


def _as_transform(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise WristDualRepresentationError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise WristDualRepresentationError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise WristDualRepresentationError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise WristDualRepresentationError(f"{name} rotation is not proper")
    return matrix


def _as_transform_series(value: Any, *, name: str) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.ndim < 3 or matrices.shape[-2:] != (4, 4):
        raise WristDualRepresentationError(f"{name} must end in [4,4]")
    flat = matrices.reshape((-1, 4, 4))
    for index, matrix in enumerate(flat):
        _as_transform(matrix, name=f"{name}[{index}]")
    return matrices


def compose_camera_wrist(
    T_camera_controller: Any,
    T_controller_wrist: Any,
) -> np.ndarray:
    """Compose transforms using the repository-wide ``T_A_B`` convention."""

    controller = _as_transform_series(
        T_camera_controller, name="T_camera_controller"
    )
    wrist = np.asarray(T_controller_wrist, dtype=np.float64)
    if wrist.shape == (4, 4):
        _as_transform(wrist, name="T_controller_wrist")
    else:
        _as_transform_series(wrist, name="T_controller_wrist")
    try:
        result = np.matmul(controller, wrist)
    except ValueError as error:
        raise WristDualRepresentationError(
            "T_camera_controller and T_controller_wrist are not broadcast compatible"
        ) from error
    _as_transform_series(result, name="T_camera_wrist")
    return result


def invert_transform(value: Any) -> np.ndarray:
    """Invert one or more validated rigid transforms."""

    matrices = _as_transform_series(value, name="transform")
    result = np.empty_like(matrices)
    rotation = matrices[..., :3, :3]
    translation = matrices[..., :3, 3]
    result[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", result[..., :3, :3], translation
    )
    result[..., 3, :] = (0.0, 0.0, 0.0, 1.0)
    return result


def lever_arm_residuals(
    T_camera_controller: Any,
    T_controller_wrist: Any,
    observed_anatomical_center_camera: Any,
) -> dict[str, np.ndarray]:
    """Return the same positional residual in camera and controller frames."""

    controller = _as_transform_series(
        T_camera_controller, name="T_camera_controller"
    )
    predicted = compose_camera_wrist(controller, T_controller_wrist)[..., :3, 3]
    observed = np.asarray(observed_anatomical_center_camera, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise WristDualRepresentationError(
            "observed anatomical centre must match predicted wrist centre shape"
        )
    residual_camera = observed - predicted
    rotation = controller[..., :3, :3]
    residual_controller = np.einsum(
        "...ji,...j->...i", rotation, residual_camera
    )
    return {
        "predicted_center_camera": predicted,
        "residual_camera": residual_camera,
        "residual_controller": residual_controller,
    }


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise WristDualRepresentationError("rotation mean requires Nx3x3 values")
    left, _singular, right = np.linalg.svd(np.sum(values, axis=0))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _controller_rotation_span_deg(transforms: np.ndarray) -> float:
    rotations = transforms[:, :3, :3]
    reference = rotations[0]
    return float(max(_rotation_distance_deg(reference, value) for value in rotations))


def fit_m1_translation(
    T_camera_controller: Any,
    observed_anatomical_center_camera: Any,
    nominal_T_controller_wrist: Any,
    valid: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one controller-local translation while preserving nominal rotation."""

    controller = _as_transform_series(
        T_camera_controller, name="T_camera_controller"
    )
    if controller.ndim != 3:
        raise WristDualRepresentationError("M1 expects T_camera_controller [N,4,4]")
    nominal = _as_transform(nominal_T_controller_wrist, name="nominal_T_controller_wrist")
    observed = np.asarray(observed_anatomical_center_camera, dtype=np.float64)
    if observed.shape != (len(controller), 3):
        raise WristDualRepresentationError("M1 observed centres must be [N,3]")
    admitted = np.isfinite(observed).all(axis=1)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (len(controller),):
            raise WristDualRepresentationError("M1 valid mask must be [N]")
        admitted &= mask
    if int(admitted.sum()) < 3:
        raise WristDualRepresentationError("M1 requires at least three valid frames")
    local = np.einsum(
        "nij,nj->ni",
        np.swapaxes(controller[admitted, :3, :3], 1, 2),
        observed[admitted] - controller[admitted, :3, 3],
    )
    fitted_translation = np.median(local, axis=0)
    candidate = nominal.copy()
    candidate[:3, 3] = fitted_translation
    correction = fitted_translation - nominal[:3, 3]
    residual = local - fitted_translation
    return candidate, {
        "model": "M1_CONTROLLER_LOCAL_TRANSLATION",
        "fit_frames": int(admitted.sum()),
        "translation_correction_m": correction.tolist(),
        "local_residual_norm_mm_p50": float(
            np.percentile(np.linalg.norm(residual, axis=1), 50) * 1000.0
        ),
        "local_residual_norm_mm_p95": float(
            np.percentile(np.linalg.norm(residual, axis=1), 95) * 1000.0
        ),
    }


def fit_m2_se3(
    T_camera_controller: Any,
    observed_T_camera_wrist: Any,
    *,
    valid: Any | None = None,
    orientation_valid: Any | None = None,
    minimum_orientation_samples: int = 8,
    minimum_controller_rotation_span_deg: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one static controller-to-wrist SE(3) when orientation is identifiable."""

    controller = _as_transform_series(
        T_camera_controller, name="T_camera_controller"
    )
    observed = np.asarray(observed_T_camera_wrist, dtype=np.float64)
    if controller.ndim != 3 or observed.shape != controller.shape:
        raise WristDualRepresentationError("M2 transform inputs must both be [N,4,4]")
    admitted = np.ones(len(controller), dtype=bool)
    if valid is not None:
        values = np.asarray(valid, dtype=bool)
        if values.shape != admitted.shape:
            raise WristDualRepresentationError("M2 valid mask must be [N]")
        admitted &= values
    if orientation_valid is not None:
        values = np.asarray(orientation_valid, dtype=bool)
        if values.shape != admitted.shape:
            raise WristDualRepresentationError("M2 orientation mask must be [N]")
        admitted &= values
    admitted &= np.isfinite(observed).all(axis=(1, 2))
    count = int(admitted.sum())
    if count < minimum_orientation_samples:
        raise WristDualRepresentationError(
            f"M2 orientation evidence is insufficient: {count} < {minimum_orientation_samples}"
        )
    span = _controller_rotation_span_deg(controller[admitted])
    if span < minimum_controller_rotation_span_deg:
        raise WristDualRepresentationError(
            "M2 controller rotation span is insufficient: "
            f"{span:.3f} < {minimum_controller_rotation_span_deg:.3f} deg"
        )
    for index, matrix in zip(np.flatnonzero(admitted), observed[admitted], strict=True):
        _as_transform(matrix, name=f"observed_T_camera_wrist[{index}]")
    local = invert_transform(controller[admitted]) @ observed[admitted]
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, 3] = np.median(local[:, :3, 3], axis=0)
    candidate[:3, :3] = _mean_rotation(local[:, :3, :3])
    translation_residual = np.linalg.norm(
        local[:, :3, 3] - candidate[:3, 3], axis=1
    )
    rotation_residual = np.asarray(
        [_rotation_distance_deg(candidate[:3, :3], value) for value in local[:, :3, :3]]
    )
    return candidate, {
        "model": "M2_STATIC_SE3",
        "fit_frames": count,
        "controller_rotation_span_deg": span,
        "translation_residual_mm_p95": float(
            np.percentile(translation_residual, 95) * 1000.0
        ),
        "rotation_residual_deg_p95": float(np.percentile(rotation_residual, 95)),
    }


def evaluate_static_calibration(
    T_camera_controller: Any,
    T_controller_wrist: Any,
    observed_T_camera_wrist: Any,
    valid: Any | None = None,
) -> dict[str, Any]:
    """Evaluate position and, when supplied, frame-orientation residuals."""

    controller = _as_transform_series(
        T_camera_controller, name="T_camera_controller"
    )
    observed = np.asarray(observed_T_camera_wrist, dtype=np.float64)
    predicted = compose_camera_wrist(controller, T_controller_wrist)
    if predicted.shape != observed.shape or predicted.ndim != 3:
        raise WristDualRepresentationError("calibration evaluation expects matching [N,4,4]")
    admitted = np.ones(len(predicted), dtype=bool)
    if valid is not None:
        values = np.asarray(valid, dtype=bool)
        if values.shape != admitted.shape:
            raise WristDualRepresentationError("evaluation valid mask must be [N]")
        admitted &= values
    admitted &= np.isfinite(observed).all(axis=(1, 2))
    if not admitted.any():
        return {
            "valid_frames": 0,
            "translation_residual_mm_p50": None,
            "translation_residual_mm_p95": None,
            "rotation_residual_deg_p50": None,
            "rotation_residual_deg_p95": None,
        }
    for index, matrix in zip(np.flatnonzero(admitted), observed[admitted], strict=True):
        _as_transform(matrix, name=f"observed_T_camera_wrist[{index}]")
    translation = np.linalg.norm(
        observed[admitted, :3, 3] - predicted[admitted, :3, 3], axis=1
    ) * 1000.0
    rotation = np.asarray(
        [
            _rotation_distance_deg(first, second)
            for first, second in zip(
                predicted[admitted, :3, :3],
                observed[admitted, :3, :3],
                strict=True,
            )
        ]
    )
    return {
        "valid_frames": int(admitted.sum()),
        "translation_residual_mm_p50": float(np.percentile(translation, 50)),
        "translation_residual_mm_p95": float(np.percentile(translation, 95)),
        "rotation_residual_deg_p50": float(np.percentile(rotation, 50)),
        "rotation_residual_deg_p95": float(np.percentile(rotation, 95)),
    }


@dataclass(frozen=True)
class SurfaceAdmission:
    point_camera: np.ndarray
    source_pixel_uv: np.ndarray
    surface_point_valid: bool
    region_registration_only: bool
    blocker: str | None
    surface_role: str


def admit_visible_wrist_surface(
    *,
    source_pixel_uv: Any,
    surface_patch_xyz_camera: Any | None,
    rgb_wrist_region_valid: bool,
    source_pixel_traceable: bool,
    mask_purity_valid: bool,
    depth_rgb_registration_valid: bool,
    local_depth_continuity_valid: bool,
    contaminant_free: bool,
    surface_role: str,
) -> SurfaceAdmission:
    """Admit a visible surface point only when all independent evidence passes."""

    pixel = np.asarray(source_pixel_uv, dtype=np.float64).reshape(-1)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        pixel = np.full(2, np.nan, dtype=np.float64)
        source_pixel_traceable = False
    checks = (
        (rgb_wrist_region_valid, "RGB_WRIST_REGION_INVALID"),
        (source_pixel_traceable, "SOURCE_PIXEL_UNTRACEABLE"),
        (mask_purity_valid, "MASK_PURITY_INVALID"),
        (depth_rgb_registration_valid, "DEPTH_RGB_REGISTRATION_INVALID"),
        (local_depth_continuity_valid, "LOCAL_DEPTH_DISCONTINUITY"),
        (contaminant_free, "SURFACE_ROLE_CONTAMINATED"),
    )
    blocker = next((reason for passed, reason in checks if not passed), None)
    patch = (
        np.asarray(surface_patch_xyz_camera, dtype=np.float64)
        if surface_patch_xyz_camera is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    if patch.ndim != 2 or patch.shape[1:] != (3,):
        blocker = blocker or "SURFACE_PATCH_SHAPE_INVALID"
    elif not len(patch) or not np.isfinite(patch).all():
        blocker = blocker or "SURFACE_PATCH_INVALID"
    point = (
        np.median(patch, axis=0)
        if blocker is None
        else np.full(3, np.nan, dtype=np.float64)
    )
    return SurfaceAdmission(
        point_camera=point,
        source_pixel_uv=pixel,
        surface_point_valid=blocker is None,
        region_registration_only=bool(rgb_wrist_region_valid and blocker is not None),
        blocker=blocker,
        surface_role=str(surface_role),
    )


def bounded_nonaccumulating_fusion(
    static_T_camera_wrist: Any,
    measured_anatomical_center_camera: Any,
    measurement_valid: Any,
    *,
    heuristic_weight: float,
    correction_bound_mm: float = 30.0,
) -> dict[str, np.ndarray]:
    """Fuse centres against each frame's static prior without temporal accumulation."""

    prior = _as_transform_series(
        static_T_camera_wrist, name="static_T_camera_wrist"
    )
    measured = np.asarray(measured_anatomical_center_camera, dtype=np.float64)
    if measured.shape != prior.shape[:-2] + (3,):
        raise WristDualRepresentationError("measurement centre shape does not match prior")
    valid = np.asarray(measurement_valid, dtype=bool)
    if valid.shape != prior.shape[:-2]:
        raise WristDualRepresentationError("measurement valid shape does not match prior")
    if not (0.0 <= heuristic_weight <= 1.0):
        raise WristDualRepresentationError("heuristic weight must be in [0,1]")
    if not (0.0 < correction_bound_mm <= 30.0):
        raise WristDualRepresentationError("development correction bound must be in (0,30] mm")

    valid &= np.isfinite(measured).all(axis=-1)
    innovation = measured - prior[..., :3, 3]
    correction = innovation * heuristic_weight
    norm = np.linalg.norm(correction, axis=-1)
    bound_m = correction_bound_mm / 1000.0
    scale = np.minimum(1.0, np.divide(bound_m, norm, out=np.ones_like(norm), where=norm > 0))
    correction *= scale[..., None]
    fused = np.full_like(prior, np.nan)
    flat_fused = fused.reshape((-1, 4, 4))
    flat_prior = prior.reshape((-1, 4, 4))
    flat_valid = valid.reshape(-1)
    flat_correction = correction.reshape((-1, 3))
    flat_fused[flat_valid] = flat_prior[flat_valid]
    flat_fused[flat_valid, :3, 3] = (
        flat_prior[flat_valid, :3, 3] + flat_correction[flat_valid]
    )
    applied = np.zeros_like(correction)
    applied[valid] = correction[valid]
    return {
        "fused_T_camera_wrist": fused,
        "fusion_valid": valid,
        "innovation_camera_m": innovation,
        "applied_correction_camera_m": applied,
        "correction_clipped": valid & (norm > bound_m),
        "uncertainty_status": np.full(valid.shape, UNKNOWN_UNCERTAINTY, dtype="U16"),
        "heuristic_weight": np.full(valid.shape, heuristic_weight, dtype=np.float64),
    }


def audit_suffix_invariance(
    full_fields: Mapping[str, Any],
    prefix_fields: Mapping[str, Any],
    *,
    target_frame: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare a current input made from a full sequence with a prefix-only run.

    ``full_fields[name][target_frame]`` is compared with
    ``prefix_fields[name][-1]``.  Non-floating arrays must match byte-for-byte.
    A mismatching field is explicitly labelled ``OFFLINE_NONCAUSAL``.
    """

    if set(full_fields) != set(prefix_fields):
        raise WristDualRepresentationError("suffix audit field sets differ")
    rows: dict[str, Any] = {}
    all_passed = True
    for name in sorted(full_fields):
        full = np.asarray(full_fields[name])
        prefix = np.asarray(prefix_fields[name])
        if full.ndim == 0 or prefix.ndim == 0:
            raise WristDualRepresentationError(f"suffix audit field is scalar: {name}")
        if target_frame < 0 or target_frame >= len(full) or not len(prefix):
            raise WristDualRepresentationError(f"suffix audit index is invalid: {name}")
        first = np.asarray(full[target_frame])
        second = np.asarray(prefix[-1])
        if first.shape != second.shape:
            passed = False
            maximum = None
        elif np.issubdtype(first.dtype, np.floating) or np.issubdtype(
            second.dtype, np.floating
        ):
            difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
            passed = bool(np.allclose(first, second, atol=atol, rtol=rtol, equal_nan=True))
            finite = difference[np.isfinite(difference)]
            maximum = float(np.max(finite)) if finite.size else 0.0
        else:
            passed = bool(first.dtype == second.dtype and first.tobytes() == second.tobytes())
            maximum = None
        all_passed &= passed
        rows[name] = {
            "passed": passed,
            "temporal_authority": (
                "CAUSAL_CURRENT" if passed else "OFFLINE_NONCAUSAL"
            ),
            "maximum_absolute_difference": maximum,
        }
    return {
        "status": "PASS_SUFFIX_INVARIANCE" if all_passed else "FAIL_SUFFIX_INVARIANCE",
        "target_frame": int(target_frame),
        "atol": float(atol),
        "rtol": float(rtol),
        "fields": rows,
    }


def validate_temporal_authority(value: str) -> str:
    authority = str(value)
    if authority not in TEMPORAL_AUTHORITIES:
        raise WristDualRepresentationError(f"invalid temporal authority: {authority}")
    return authority
