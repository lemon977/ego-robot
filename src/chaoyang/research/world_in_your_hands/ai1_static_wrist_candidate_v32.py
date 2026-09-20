"""AI1 V3.2 static-wrist diagnostics with fail-closed adoption semantics.

The diagnostic order is part of the interface.  None of the helpers searches
over coordinate systems, axes, or lags to select the visually best answer.
``selected_model`` means only which candidate is rendered/replayed; adoption is
an independent evidence decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chaoyang.research.world_in_your_hands.wrist_dual_representation_v1 import (
    WristDualRepresentationError,
    compose_camera_wrist,
    lever_arm_residuals,
)


SCHEMA_VERSION = "AI1_STATIC_WRIST_CANDIDATE_V32"
DIAGNOSTIC_ORDER = (
    "IMAGE_DOMAIN_EYE_SIDE",
    "TRANSFORM_DIRECTION_AND_UNIT",
    "MANUS_ROOT_AND_AXIS",
    "TIMESTAMP_AND_FINITE_LAG",
    "CONTROLLER_LOCAL_RESIDUAL",
    "HAWOR_2D_RELATIVE_SHAPE_ABSOLUTE_DEPTH",
)
ALLOWED_MODELS = ("M0_LEGACY", "M1_CONTROLLER_LOCAL_TRANSLATION", "M2_STATIC_SE3")
ADOPTION_AUTHORITY = "MODEL_ALIGNMENT_ADAPTER_POSITION_ONLY"
DEVELOPMENT_AUTHORITY = "DEVELOPMENT_ONLY"


class Ai1StaticWristCandidateError(ValueError):
    """Raised when a diagnostic would compare incompatible evidence."""


def require_same_wrist_semantics(first: str, second: str) -> None:
    """Reject an error computation unless both sources name the same wrist frame."""

    left = str(first).strip()
    right = str(second).strip()
    if not left or not right or left != right:
        raise Ai1StaticWristCandidateError(
            "wrist semantics differ; direct residual is forbidden: "
            f"{left or 'UNSPECIFIED'} != {right or 'UNSPECIFIED'}"
        )


def summarize_norm_mm(value: Any, valid: Any | None = None) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise Ai1StaticWristCandidateError("residual vectors must be [N,3]")
    admitted = np.isfinite(array).all(axis=1)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != admitted.shape:
            raise Ai1StaticWristCandidateError("residual valid mask must be [N]")
        admitted &= mask
    norms = np.linalg.norm(array[admitted], axis=1) * 1000.0
    return {
        "valid_frames": int(len(norms)),
        "norm_mm_p50": float(np.percentile(norms, 50)) if len(norms) else None,
        "norm_mm_p95": float(np.percentile(norms, 95)) if len(norms) else None,
    }


def controller_local_residual_diagnostic(
    T_camera_controller: Any,
    T_controller_wrist: Any,
    observed_wrist_camera_m: Any,
    *,
    candidate_wrist_semantics: str,
    observed_wrist_semantics: str,
    valid: Any | None = None,
) -> dict[str, Any]:
    """Return paired camera/controller residuals after a semantic-frame check."""

    require_same_wrist_semantics(candidate_wrist_semantics, observed_wrist_semantics)
    diagnostic = lever_arm_residuals(
        T_camera_controller,
        T_controller_wrist,
        observed_wrist_camera_m,
    )
    admitted = np.ones(len(diagnostic["residual_camera"]), dtype=bool)
    if valid is not None:
        admitted = np.asarray(valid, dtype=bool)
        if admitted.shape != (len(diagnostic["residual_camera"]),):
            raise Ai1StaticWristCandidateError("controller-local valid mask must be [N]")
    return {
        **diagnostic,
        "camera_summary": summarize_norm_mm(diagnostic["residual_camera"], admitted),
        "controller_local_summary": summarize_norm_mm(
            diagnostic["residual_controller"], admitted
        ),
        "wrist_semantics": candidate_wrist_semantics,
    }


def finite_lag_diagnostics(
    predicted_camera_m: Any,
    observed_camera_m: Any,
    valid: Any,
    *,
    prescribed_lags: Sequence[int] = (-2, -1, 0, 1, 2),
    group_id: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a fixed finite lag list without ranking or selecting a winner."""

    predicted = np.asarray(predicted_camera_m, dtype=np.float64)
    observed = np.asarray(observed_camera_m, dtype=np.float64)
    admitted = np.asarray(valid, dtype=bool)
    if predicted.shape != observed.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise Ai1StaticWristCandidateError("finite-lag points must be matching [N,3]")
    if admitted.shape != (len(predicted),):
        raise Ai1StaticWristCandidateError("finite-lag valid mask must be [N]")
    groups = None if group_id is None else np.asarray(group_id)
    if groups is not None and groups.shape != (len(predicted),):
        raise Ai1StaticWristCandidateError("finite-lag group IDs must be [N]")
    lags = tuple(int(value) for value in prescribed_lags)
    if not lags or len(set(lags)) != len(lags) or any(abs(value) > 2 for value in lags):
        raise Ai1StaticWristCandidateError("prescribed lags must be unique and within [-2,2]")
    rows: list[dict[str, Any]] = []
    for lag in lags:
        if lag > 0:
            predicted_slice = predicted[:-lag]
            observed_slice = observed[lag:]
            valid_slice = admitted[:-lag] & admitted[lag:]
            if groups is not None:
                valid_slice &= groups[:-lag] == groups[lag:]
        elif lag < 0:
            offset = -lag
            predicted_slice = predicted[offset:]
            observed_slice = observed[:-offset]
            valid_slice = admitted[offset:] & admitted[:-offset]
            if groups is not None:
                valid_slice &= groups[offset:] == groups[:-offset]
        else:
            predicted_slice = predicted
            observed_slice = observed
            valid_slice = admitted.copy()
        rows.append(
            {
                "lag_frames": lag,
                **summarize_norm_mm(observed_slice - predicted_slice, valid_slice),
            }
        )
    return rows


def project_points(points_camera_m: Any, camera_K: Any) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera_m, dtype=np.float64)
    intrinsics = np.asarray(camera_K, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or intrinsics.shape != (len(points), 3, 3):
        raise Ai1StaticWristCandidateError("projection expects points [N,3] and K [N,3,3]")
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(intrinsics).all(axis=(1, 2)) & (depth > 0)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    homogeneous = np.einsum("nij,nj->ni", intrinsics[valid], points[valid])
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, valid


def two_dimensional_diagnostic(
    predicted_camera_m: Any,
    observed_uv: Any,
    camera_K: Any,
    valid: Any,
) -> dict[str, Any]:
    projected, projectable = project_points(predicted_camera_m, camera_K)
    observed = np.asarray(observed_uv, dtype=np.float64)
    admitted = np.asarray(valid, dtype=bool)
    if observed.shape != projected.shape or admitted.shape != (len(projected),):
        raise Ai1StaticWristCandidateError("2D diagnostic shape mismatch")
    admitted &= projectable & np.isfinite(observed).all(axis=1)
    residual = np.linalg.norm(projected[admitted] - observed[admitted], axis=1)
    return {
        "valid_frames": int(len(residual)),
        "residual_px_p50": float(np.percentile(residual, 50)) if len(residual) else None,
        "residual_px_p95": float(np.percentile(residual, 95)) if len(residual) else None,
        "selection_authority": False,
    }


def absolute_depth_diagnostic(
    candidate_depth_m: Any,
    observed_depth_m: Any,
    valid: Any,
    *,
    candidate_wrist_semantics: str,
    observed_wrist_semantics: str,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_depth_m, dtype=np.float64)
    observed = np.asarray(observed_depth_m, dtype=np.float64)
    admitted = np.asarray(valid, dtype=bool)
    if candidate.shape != observed.shape or candidate.ndim != 1 or admitted.shape != candidate.shape:
        raise Ai1StaticWristCandidateError("absolute-depth inputs must be matching [N]")
    admitted &= np.isfinite(candidate) & np.isfinite(observed)
    raw = {
        "valid_frames": int(admitted.sum()),
        "candidate_depth_m_p50": (
            float(np.percentile(candidate[admitted], 50)) if admitted.any() else None
        ),
        "observed_depth_m_p50": (
            float(np.percentile(observed[admitted], 50)) if admitted.any() else None
        ),
    }
    try:
        require_same_wrist_semantics(candidate_wrist_semantics, observed_wrist_semantics)
    except Ai1StaticWristCandidateError as error:
        return {
            **raw,
            "status": "REJECTED_INCOMPATIBLE_WRIST_SEMANTICS",
            "residual_mm_p50": None,
            "residual_mm_p95": None,
            "reason": str(error),
        }
    residual = np.abs(candidate[admitted] - observed[admitted]) * 1000.0
    return {
        **raw,
        "status": "COMPARABLE",
        "residual_mm_p50": float(np.percentile(residual, 50)) if len(residual) else None,
        "residual_mm_p95": float(np.percentile(residual, 95)) if len(residual) else None,
        "reason": None,
    }


def decide_candidate_authority(
    *,
    selected_model: str,
    deterministic_error_fixed: bool,
    leave_one_recording_out_passed: bool,
    regression_101_passed: bool,
    wrist_semantics_proven_equivalent: bool,
    independent_rotation_evidence: bool,
) -> dict[str, Any]:
    """Separate diagnostic selection, adoption, and 102/103 opening authority."""

    if selected_model not in ALLOWED_MODELS:
        raise Ai1StaticWristCandidateError(f"unsupported candidate: {selected_model}")
    if selected_model == "M2_STATIC_SE3" and not independent_rotation_evidence:
        raise Ai1StaticWristCandidateError("M2 cannot open without independent rotation evidence")
    nonfitting_gate = leave_one_recording_out_passed and regression_101_passed
    opening_evidence = deterministic_error_fixed or nonfitting_gate
    adopted = bool(opening_evidence and wrist_semantics_proven_equivalent)
    terminal = (
        "ADOPTED_POSITION_ONLY_DEVELOPMENT_CANDIDATE"
        if adopted
        else "NO_ADMISSIBLE_STATIC_CANDIDATE"
    )
    return {
        "selected_model": selected_model,
        "selected_for_diagnostics": True,
        "adopted": adopted,
        "terminal_status": terminal,
        "opening_gate_102_103": "OPEN" if adopted else "CLOSED",
        "opening_evidence": {
            "deterministic_error_fixed": bool(deterministic_error_fixed),
            "leave_one_recording_out_passed": bool(leave_one_recording_out_passed),
            "regression_101_passed": bool(regression_101_passed),
            "wrist_semantics_proven_equivalent": bool(wrist_semantics_proven_equivalent),
            "independent_rotation_evidence": bool(independent_rotation_evidence),
        },
        "authority": ADOPTION_AUTHORITY if adopted else "NONE",
        "scope": DEVELOPMENT_AUTHORITY,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }


def validate_diagnostic_order(rows: Sequence[Mapping[str, Any]]) -> None:
    actual = tuple(str(row.get("stage")) for row in rows)
    if actual != DIAGNOSTIC_ORDER:
        raise Ai1StaticWristCandidateError(
            f"diagnostics must follow fixed order: {actual} != {DIAGNOSTIC_ORDER}"
        )


def camera_wrist_series(T_camera_controller: Any, T_controller_wrist: Any) -> np.ndarray:
    """Explicit wrapper retained in V3.2 outputs to document composition direction."""

    try:
        return compose_camera_wrist(T_camera_controller, T_controller_wrist)
    except WristDualRepresentationError as error:
        raise Ai1StaticWristCandidateError(str(error)) from error
