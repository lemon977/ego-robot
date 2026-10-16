"""Candidate-only, offline reference selection and Clean evidence checks.

These CPU helpers do not load ProPainter, infer a background, or read another
session. The candidate inference adapter may replace only its ``refs`` list at
the existing ProtectedProPainter selection site; all model settings, neighbors,
output ownership and averaging remain frozen. No model adapter is enabled here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index


def _mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"MASK_DOMAIN_OR_TYPE:{name}")
    return array


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"SOURCE_SHA_REQUIRED:{name}")


@dataclass(frozen=True)
class ReferenceEvidence:
    """Raw donor support already warped into the target's native image domain.

    ``visible_background`` means actually visible *raw* background, not Clean or
    a hidden-object proposal. ``flow_error`` is an existing raw forward/backward
    flow residual, not a newly invented confidence. Nonfinite residual pixels
    carry no usable support. A producer must archive the arrays and their SHAs;
    these fields do not themselves prove that a producer ran correctly.
    """

    session_id: str
    target_frame: int
    reference_frame: int
    visible_background: np.ndarray
    flow_error: np.ndarray
    raw_frame_sha256: str
    raw_support_sha256: str
    raw_flow_sha256: str
    support_source: str = "RAW_VISIBLE_BACKGROUND"
    flow_source: str = "RAW_RGB_FORWARD_BACKWARD_FLOW"
    coordinate_domain: str = "TARGET_NATIVE_RAW_IMAGE"
    evidence_metadata: Mapping[str, object] | None = None


def select_reference_indices(
    mid: int,
    neighbors: Sequence[int],
    length: int,
    *,
    session_id: str,
    effective_delete_mask: np.ndarray,
    evidence: Sequence[ReferenceEvidence],
    settings: Mapping[str, object] = SETTINGS,
    time_mode: str = "OFFLINE_VISUAL",
) -> dict:
    """Rank donors in the unchanged context; preserve the old reference count.

    Ranking is lexicographic: more raw-supported hole pixels, then lower median
    existing flow residual, then frame index. There are no new score weights or
    thresholds. Selected model references remain in chronological order. If the
    frozen count cannot be filled with supported donors, retain the complete
    original list and report that ranking was not applied. Never shrink context
    or claim that a generated fill is an observed hidden-object texture.
    """
    if time_mode != "OFFLINE_VISUAL":
        raise ValueError("OFFLINE_REFERENCE_SELECTOR_NOT_CAUSAL")
    if dict(settings) != SETTINGS:
        raise ValueError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("SESSION_ID_REQUIRED")
    if type(length) is not int or type(mid) is not int or length < 2 or not 0 <= mid < length:
        raise ValueError("REFERENCE_TIME_DOMAIN")
    neighbors = list(neighbors)
    if (len(set(neighbors)) != len(neighbors)
            or any(type(i) is not int or not 0 <= i < length for i in neighbors)
            or mid not in neighbors):
        raise ValueError("REFERENCE_NEIGHBOR_DOMAIN")
    hole = np.asarray(effective_delete_mask)
    if hole.ndim != 2 or not hole.size:
        raise ValueError("REFERENCE_HOLE_DOMAIN")
    hole = _mask(hole, hole.shape, "effective_delete_mask")
    sub, stride = int(settings["subvideo_length"]), int(settings["ref_stride"])
    ref_num = sub // stride if length > sub else -1
    original = _get_ref_index(mid, neighbors, length, stride, ref_num)
    if ref_num == -1:
        allowed = set(range(length)) - set(neighbors)
    else:
        start = max(0, mid - stride * (ref_num // 2))
        end = min(length, mid + stride * (ref_num // 2))
        allowed = set(range(start, end)) - set(neighbors)
    rows, seen = [], set()
    for item in evidence:
        if item.session_id != session_id:
            raise ValueError("CROSS_SESSION_REFERENCE_FORBIDDEN")
        if item.target_frame != mid or item.coordinate_domain != "TARGET_NATIVE_RAW_IMAGE":
            raise ValueError("REFERENCE_TARGET_ALIGNMENT_UNPROVEN")
        if (item.support_source not in {"RAW_VISIBLE_BACKGROUND", "RAW_UNREMOVED_VISIBLE_SUPPORT"}
                or item.flow_source != "RAW_RGB_FORWARD_BACKWARD_FLOW"):
            raise ValueError("REFERENCE_REQUIRES_RAW_BACKGROUND_AND_RAW_FLOW")
        for name in ("raw_frame_sha256", "raw_support_sha256", "raw_flow_sha256"):
            _sha(getattr(item, name), name)
        if item.reference_frame in seen:
            raise ValueError("DUPLICATE_REFERENCE_EVIDENCE")
        if type(item.reference_frame) is not int or not 0 <= item.reference_frame < length:
            raise ValueError("REFERENCE_FRAME_DOMAIN")
        seen.add(item.reference_frame)
        support = _mask(item.visible_background, hole.shape, "visible_background")
        error = np.asarray(item.flow_error)
        if error.shape != hole.shape or not np.issubdtype(error.dtype, np.floating):
            raise ValueError("REFERENCE_FLOW_ERROR_DOMAIN_OR_TYPE")
        if np.any(np.isfinite(error) & (error < 0)):
            raise ValueError("REFERENCE_FLOW_ERROR_NEGATIVE")
        supported = hole & support & np.isfinite(error)
        count = int(supported.sum())
        if item.reference_frame not in allowed or count == 0:
            continue
        rows.append({
            "frame": item.reference_frame,
            "supported_hole_pixels": count,
            "median_raw_flow_error": float(np.median(error[supported])),
            "raw_frame_sha256": item.raw_frame_sha256,
            "raw_support_sha256": item.raw_support_sha256,
            "raw_flow_sha256": item.raw_flow_sha256,
        })
        if item.support_source == "RAW_UNREMOVED_VISIBLE_SUPPORT" and item.evidence_metadata is not None:
            rows[-1]["evidence_metadata"] = dict(item.evidence_metadata)
    rows.sort(key=lambda row: (-row["supported_hole_pixels"],
                               row["median_raw_flow_error"], row["frame"]))
    count = len(original)
    if not hole.any():
        status, selected = "BASELINE_NO_HOLES", original
    elif not count:
        status, selected = "BASELINE_NO_REFERENCE_SLOTS", original
    elif len(rows) < count:
        status, selected = "BASELINE_INSUFFICIENT_RAW_SUPPORTED_REFERENCES", original
    else:
        status = "RANKED_RAW_BACKGROUND_AND_FLOW"
        selected = sorted(row["frame"] for row in rows[:count])
    result = {
        "status": status,
        "original_references": original,
        "references": selected,
        "reference_count": count,
        "ranked_evidence": rows,
        "neighbors_unchanged": True,
        "frozen_model_settings_unchanged": True,
        "time_mode": time_mode,
        "causal_training_input": False,
        "fill_source": "MODEL_PROPAGATED_OR_GENERATED_NOT_HIDDEN_OBJECT_TRUTH",
    }
    if any(item.support_source == "RAW_UNREMOVED_VISIBLE_SUPPORT" for item in evidence):
        if status == "RANKED_RAW_BACKGROUND_AND_FLOW":
            result["status"] = "RANKED_RAW_VISIBLE_SUPPORT_AND_FLOW"
        result.update(support_sources=sorted({item.support_source for item in evidence}),
                      background_geometry_qualified=False, static_background_truth=False)
    return result


def validate_clean_preservation(
    raw: np.ndarray,
    clean: np.ndarray,
    effective_delete_mask: np.ndarray,
    protected_visible_mask: np.ndarray,
) -> dict:
    """Measure existing zero-change invariants without a Robot overlay or repair."""
    raw, clean = np.asarray(raw), np.asarray(clean)
    if (raw.shape != clean.shape or raw.ndim not in (3, 4)
            or raw.shape[-1] != 3 or raw.dtype != np.uint8 or clean.dtype != np.uint8
            or not raw.size):
        raise ValueError("CLEAN_RGB_DOMAIN_OR_TYPE")
    shape = raw.shape[:-1]
    effective = _mask(effective_delete_mask, shape, "effective_delete_mask")
    protected = _mask(protected_visible_mask, shape, "protected_visible_mask")
    if np.any(effective & protected):
        raise ValueError("PROTECTED_MASK_ENTERED_EFFECTIVE_DELETE_DOMAIN")
    changed = np.any(raw != clean, axis=-1)
    outside, protection = changed & ~effective, changed & protected
    axes = (1, 2) if changed.ndim == 3 else (0, 1)
    return {
        "status": "PASS" if not outside.any() and not protection.any() else "FAIL",
        "outside_delete_changed_pixels": int(outside.sum()),
        "protected_changed_pixels": int(protection.sum()),
        "outside_delete_changed_pixels_per_frame": np.atleast_1d(outside.sum(axis=axes)).tolist(),
        "protected_changed_pixels_per_frame": np.atleast_1d(protection.sum(axis=axes)).tolist(),
        "clean_visual_quality": "NOT_EVALUATED",
        "robot_coverage_used_for_clean_score": False,
    }


def classify_residual_pixels(
    raw: np.ndarray,
    clean: np.ndarray,
    reviewed_visible_residual: np.ndarray,
    effective_delete_mask: np.ndarray,
    protected_visible_mask: np.ndarray,
) -> dict:
    """Partition evaluator-supported residue, not invent a detector or causal proof.

    A residue outside E diagnoses insufficient deletion support. A residue inside
    E is a Clean-domain issue, but the fact alone does not prove regeneration:
    propagation, copying and generation require subsequent evidence review.
    Protected conflicts remain explicit and are not pasted back as a fix.
    """
    validation = validate_clean_preservation(raw, clean, effective_delete_mask,
                                           protected_visible_mask)
    shape = np.asarray(raw).shape[:-1]
    residue = _mask(reviewed_visible_residual, shape, "reviewed_visible_residual")
    effective = _mask(effective_delete_mask, shape, "effective_delete_mask")
    protected = _mask(protected_visible_mask, shape, "protected_visible_mask")
    changed = np.any(np.asarray(raw) != np.asarray(clean), axis=-1)
    outside, inside = residue & ~effective, residue & effective
    return {
        "outside_effective_delete": outside,
        "inside_effective_delete": inside,
        "protected_residual_conflict": residue & protected,
        "outside_effective_delete_pixels": int(outside.sum()),
        "inside_effective_delete_pixels": int(inside.sum()),
        "inside_changed_pixels": int((inside & changed).sum()),
        "inside_unchanged_pixels": int((inside & ~changed).sum()),
        "mask_issue_code": "VISIBLE_RESIDUE_OUTSIDE_DELETE_SUPPORT" if outside.any() else None,
        "clean_issue_code": "VISIBLE_RESIDUE_INSIDE_DELETE_DOMAIN" if inside.any() else None,
        "inside_cause": "PROPAGATION_COPY_OR_GENERATION_NOT_YET_ATTRIBUTED" if inside.any() else None,
        "validation": validation,
        "producer_may_read_review_annotations": False,
        "clean_visual_quality": "NOT_SCORED_BY_ROBOT_COVERAGE",
    }
