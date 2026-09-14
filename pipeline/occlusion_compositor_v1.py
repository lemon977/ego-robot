"""Occlusion-only compositor primitives with explicit RGB provenance.

This module does not infer human contact and does not modify Robot poses.  It
decides image ownership from frozen depths/masks, refuses to invent hidden
object texture, and reports coverage beside known-decision accuracy.

FoundationStereo currently has no native confidence field.  The API therefore
accepts a boolean validity mask plus named quality evidence only; passing a
``depth_confidence`` keyword is intentionally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


SCHEMA_VERSION = "OCCLUSION_COMPOSITOR_V1"
OCCLUSION_EPSILON_M = 0.003


class OcclusionContractError(ValueError):
    """Raised when compositor evidence is malformed or self-contradictory."""


class Ownership(IntEnum):
    BACKGROUND = 0
    HUMAN_FRONT = 1
    OBJECT_FRONT = 2
    ROBOT_FRONT = 3
    TIE_UNKNOWN = 4


class ObjectPixelSource(IntEnum):
    NONE_UNKNOWN = 0
    RAW_VISIBLE = 1
    TEMPORAL_OBJECT_DONOR = 2
    TEXTURED_OBJECT_RENDERER = 3


@dataclass(frozen=True)
class DepthQualityEvidence:
    disparity_finite: np.ndarray
    registration_pass: np.ndarray
    away_from_occlusion_edge: np.ndarray
    texture_support: np.ndarray
    local_consistency_pass: np.ndarray
    in_valid_depth_range: np.ndarray
    depth_confidence_present: bool = False

    def combined(self) -> np.ndarray:
        if self.depth_confidence_present:
            raise OcclusionContractError(
                "FoundationStereo native confidence is unavailable; depth_confidence_present must be false"
            )
        arrays = [
            np.asarray(self.disparity_finite),
            np.asarray(self.registration_pass),
            np.asarray(self.away_from_occlusion_edge),
            np.asarray(self.texture_support),
            np.asarray(self.local_consistency_pass),
            np.asarray(self.in_valid_depth_range),
        ]
        shape = arrays[0].shape
        if not shape or any(value.shape != shape or value.dtype != np.bool_ for value in arrays):
            raise OcclusionContractError("depth quality evidence must be matching bool images")
        return np.logical_and.reduce(arrays)


@dataclass(frozen=True)
class FrameOcclusionResult:
    ownership: np.ndarray
    training_valid_mask: np.ndarray
    object_pixel_source: np.ndarray
    contact_decision_mask: np.ndarray
    object_rgb: np.ndarray


@dataclass(frozen=True)
class FrameAudit:
    known_decision_coverage: float
    unknown_pixel_ratio: float
    accuracy_on_known: float | None
    protected_retention: float | None
    contact_pixels: int
    unknown_contact_pixels: int
    wrong_known_pixels: int | None


@dataclass(frozen=True)
class SessionAudit:
    known_decision_coverage: float
    unknown_pixel_ratio: float
    accuracy_on_known: float | None
    protected_retention: float | None
    unknown_contact_frame_ratio: float
    max_unknown_run: int
    max_wrong_known_run: int | None
    passed: bool
    failed_gates: tuple[str, ...]


def _bool_image(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise OcclusionContractError(f"{label} must be bool with shape {shape}")
    return result


def _float_image(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise OcclusionContractError(f"{label} must have shape {shape}")
    return result


def choose_object_pixels(
    *,
    raw_rgb: np.ndarray,
    raw_visible_mask: np.ndarray,
    temporal_donor_rgb: np.ndarray | None = None,
    temporal_donor_valid: np.ndarray | None = None,
    renderer_rgb: np.ndarray | None = None,
    renderer_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select object appearance in the frozen legal precedence order."""

    raw = np.asarray(raw_rgb)
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise OcclusionContractError("raw_rgb must be uint8 HxWx3")
    shape = raw.shape[:2]
    raw_valid = _bool_image(raw_visible_mask, shape, "raw_visible_mask")
    output = np.zeros_like(raw)
    source = np.full(shape, int(ObjectPixelSource.NONE_UNKNOWN), dtype=np.uint8)
    output[raw_valid] = raw[raw_valid]
    source[raw_valid] = int(ObjectPixelSource.RAW_VISIBLE)

    def add(rgb: np.ndarray | None, valid: np.ndarray | None, kind: ObjectPixelSource, label: str) -> None:
        if rgb is None and valid is None:
            return
        if rgb is None or valid is None:
            raise OcclusionContractError(f"{label} RGB and valid mask must be supplied together")
        image = np.asarray(rgb)
        if image.shape != raw.shape or image.dtype != np.uint8:
            raise OcclusionContractError(f"{label} RGB must match raw uint8 image")
        mask = _bool_image(valid, shape, f"{label}_valid")
        select = mask & (source == int(ObjectPixelSource.NONE_UNKNOWN))
        output[select] = image[select]
        source[select] = int(kind)

    add(temporal_donor_rgb, temporal_donor_valid, ObjectPixelSource.TEMPORAL_OBJECT_DONOR, "temporal donor")
    add(renderer_rgb, renderer_valid, ObjectPixelSource.TEXTURED_OBJECT_RENDERER, "renderer")
    return output, source


def resolve_ownership(
    *,
    human_mask: np.ndarray,
    object_amodal_mask: np.ndarray,
    object_depth_m: np.ndarray,
    object_depth_valid: np.ndarray,
    robot_alpha_mask: np.ndarray,
    robot_depth_m: np.ndarray,
    robot_depth_valid: np.ndarray,
    stereo_depth_valid: np.ndarray,
    depth_quality_evidence: DepthQualityEvidence,
    object_rgb: np.ndarray,
    object_pixel_source: np.ndarray,
    contact_decision_mask: np.ndarray,
    occlusion_epsilon_m: float = OCCLUSION_EPSILON_M,
) -> FrameOcclusionResult:
    """Resolve one frame; missing legal object RGB forces ``TIE_UNKNOWN``."""

    object_depth = np.asarray(object_depth_m, dtype=np.float64)
    if object_depth.ndim != 2:
        raise OcclusionContractError("object_depth_m must be a 2D image")
    shape = object_depth.shape
    human = _bool_image(human_mask, shape, "human_mask")
    object_mask = _bool_image(object_amodal_mask, shape, "object_amodal_mask")
    object_valid = _bool_image(object_depth_valid, shape, "object_depth_valid")
    robot = _bool_image(robot_alpha_mask, shape, "robot_alpha_mask")
    robot_depth = _float_image(robot_depth_m, shape, "robot_depth_m")
    robot_valid = _bool_image(robot_depth_valid, shape, "robot_depth_valid")
    stereo_valid = _bool_image(stereo_depth_valid, shape, "stereo_depth_valid")
    decision = _bool_image(contact_decision_mask, shape, "contact_decision_mask")
    quality = depth_quality_evidence.combined()
    if quality.shape != shape:
        raise OcclusionContractError("depth quality evidence shape mismatch")
    pixels = np.asarray(object_rgb)
    if pixels.shape != (*shape, 3) or pixels.dtype != np.uint8:
        raise OcclusionContractError("object_rgb must be uint8 and match depth resolution")
    provenance = np.asarray(object_pixel_source)
    legal_values = {int(value) for value in ObjectPixelSource}
    if provenance.shape != shape or not np.issubdtype(provenance.dtype, np.integer):
        raise OcclusionContractError("object_pixel_source must be an integer image")
    if not set(np.unique(provenance).tolist()).issubset(legal_values):
        raise OcclusionContractError("object_pixel_source contains an unknown value")
    epsilon = float(occlusion_epsilon_m)
    if not np.isfinite(epsilon) or epsilon != OCCLUSION_EPSILON_M:
        raise OcclusionContractError("V1 occlusion epsilon is frozen at 0.003 m")
    if not np.isfinite(object_depth[object_valid]).all() or np.any(object_depth[object_valid] <= 0.0):
        raise OcclusionContractError("valid object depth must be finite and positive")
    if not np.isfinite(robot_depth[robot_valid]).all() or np.any(robot_depth[robot_valid] <= 0.0):
        raise OcclusionContractError("valid Robot depth must be finite and positive")

    ownership = np.full(shape, int(Ownership.BACKGROUND), dtype=np.uint8)
    ownership[human] = int(Ownership.HUMAN_FRONT)
    legal_object_rgb = provenance != int(ObjectPixelSource.NONE_UNKNOWN)
    object_evidence = object_mask & object_valid & stereo_valid & quality
    robot_evidence = robot & robot_valid

    object_only = object_evidence & ~robot
    ownership[object_only & legal_object_rgb] = int(Ownership.OBJECT_FRONT)
    ownership[object_only & ~legal_object_rgb] = int(Ownership.TIE_UNKNOWN)
    robot_only = robot_evidence & ~object_mask
    ownership[robot_only] = int(Ownership.ROBOT_FRONT)

    overlap = object_mask & robot
    comparable = overlap & object_evidence & robot_evidence
    object_front = comparable & (object_depth + epsilon < robot_depth)
    robot_front = comparable & (robot_depth + epsilon < object_depth)
    ownership[object_front & legal_object_rgb] = int(Ownership.OBJECT_FRONT)
    ownership[robot_front] = int(Ownership.ROBOT_FRONT)
    unresolved = overlap & ~(object_front | robot_front)
    unresolved |= object_front & ~legal_object_rgb
    ownership[unresolved] = int(Ownership.TIE_UNKNOWN)
    # In the contact decision band, an amodal object or Robot layer lacking
    # depth evidence cannot be ordered.  Do not inherit HUMAN/BACKGROUND from
    # a prior assignment and call that a decision.
    missing_order_evidence = decision & (
        (object_mask & ~object_evidence) | (robot & ~robot_evidence)
    )
    ownership[missing_order_evidence] = int(Ownership.TIE_UNKNOWN)

    # Any decision pixel claiming object ownership without appearance evidence
    # is a contract violation, not permission to use Clean/background pixels.
    illegal = (ownership == int(Ownership.OBJECT_FRONT)) & ~legal_object_rgb
    if np.any(illegal):
        raise OcclusionContractError("OBJECT_FRONT lacks legal object pixel provenance")
    training_valid = ownership != int(Ownership.TIE_UNKNOWN)
    return FrameOcclusionResult(ownership, training_valid, provenance.copy(), decision.copy(), pixels.copy())


def audit_frame(
    result: FrameOcclusionResult,
    *,
    reference_ownership: np.ndarray | None = None,
    protected_raw_object_mask: np.ndarray | None = None,
) -> FrameAudit:
    decision = np.asarray(result.contact_decision_mask, dtype=np.bool_)
    ownership = np.asarray(result.ownership)
    unknown = decision & (ownership == int(Ownership.TIE_UNKNOWN))
    known = decision & ~unknown
    total = int(np.count_nonzero(decision))
    known_count = int(np.count_nonzero(known))
    unknown_count = int(np.count_nonzero(unknown))
    coverage = known_count / total if total else 1.0
    unknown_ratio = unknown_count / total if total else 0.0
    accuracy: float | None = None
    wrong: int | None = None
    if reference_ownership is not None:
        reference = np.asarray(reference_ownership)
        if reference.shape != ownership.shape:
            raise OcclusionContractError("reference ownership shape mismatch")
        wrong = int(np.count_nonzero(known & (ownership != reference)))
        accuracy = 1.0 - wrong / known_count if known_count else 0.0
    retention: float | None = None
    if protected_raw_object_mask is not None:
        protected = _bool_image(protected_raw_object_mask, ownership.shape, "protected_raw_object_mask")
        denominator = int(np.count_nonzero(protected))
        retained = protected & (ownership == int(Ownership.OBJECT_FRONT)) & (
            result.object_pixel_source == int(ObjectPixelSource.RAW_VISIBLE)
        )
        retention = int(np.count_nonzero(retained)) / denominator if denominator else 1.0
    return FrameAudit(coverage, unknown_ratio, accuracy, retention, total, unknown_count, wrong)


def _max_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def audit_session(frames: Sequence[FrameAudit]) -> SessionAudit:
    if not frames:
        raise OcclusionContractError("session audit requires at least one frame")
    total = sum(frame.contact_pixels for frame in frames)
    unknown = sum(frame.unknown_contact_pixels for frame in frames)
    known = total - unknown
    coverage = known / total if total else 1.0
    unknown_ratio = unknown / total if total else 0.0
    labelled = [frame for frame in frames if frame.wrong_known_pixels is not None]
    accuracy = None
    if labelled:
        labelled_known = sum(frame.contact_pixels - frame.unknown_contact_pixels for frame in labelled)
        labelled_wrong = sum(int(frame.wrong_known_pixels or 0) for frame in labelled)
        accuracy = 1.0 - labelled_wrong / labelled_known if labelled_known else 0.0
    retained = [frame.protected_retention for frame in frames if frame.protected_retention is not None]
    retention = min(retained) if retained else None
    unknown_frames = [
        frame.contact_pixels > 0 and frame.unknown_contact_pixels / frame.contact_pixels > 0.30
        for frame in frames
    ]
    unknown_frame_ratio = sum(unknown_frames) / len(frames)
    wrong_frames = [bool(frame.wrong_known_pixels) for frame in frames if frame.wrong_known_pixels is not None]
    failures: list[str] = []
    if coverage < 0.70:
        failures.append("KNOWN_DECISION_COVERAGE")
    if unknown_ratio > 0.30:
        failures.append("UNKNOWN_PIXEL_RATIO")
    if unknown_frame_ratio > 0.20:
        failures.append("UNKNOWN_CONTACT_FRAME_RATIO")
    if _max_true_run(unknown_frames) > 5:
        failures.append("MAX_UNKNOWN_RUN")
    if accuracy is None:
        failures.append("KNOWN_ACCURACY_UNMEASURED")
    elif accuracy < 0.95:
        failures.append("ACCURACY_ON_KNOWN")
    if wrong_frames and _max_true_run(wrong_frames) > 2:
        failures.append("MAX_WRONG_KNOWN_RUN")
    if retention is None:
        failures.append("PROTECTED_RETENTION_UNMEASURED")
    elif retention < 0.99:
        failures.append("PROTECTED_RETENTION")
    return SessionAudit(
        coverage, unknown_ratio, accuracy, retention, unknown_frame_ratio,
        _max_true_run(unknown_frames), _max_true_run(wrong_frames) if wrong_frames else None,
        not failures, tuple(failures),
    )
