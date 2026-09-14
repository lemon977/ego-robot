"""Evidence-only CLEAN frame assembly.

There is intentionally no inpainting implementation and no missing-pixel
fallback in this module.  Source RGB is retained only outside the reveal
target.  Every target pixel comes from its declared object/temporal donor or is
written as an explicit diagnostic sentinel and marked HOLD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .reveal_labels import (
    EvidenceIntegrityError,
    EvidenceRef,
    ObjectTextureEvidence,
    RevealFrameResult,
    RevealLabel,
    TemporalBackgroundEvidence,
    _rgb_plane,
    load_verified_reveal_policy,
)


UNSUPPORTED_SENTINEL_RGB = np.array([255, 0, 255], dtype=np.uint8)


class FormalCleanBlocked(RuntimeError):
    """This tooling revision has no authority to produce real/formal CLEAN."""


@dataclass(frozen=True)
class CleanFrameResult:
    artifact_state: str
    formal_clean_status: str
    clean_rgb: np.ndarray
    object_reveal_rgb: np.ndarray
    background_reveal_rgb: np.ndarray
    unsupported_mask: np.ndarray
    donor_provenance_refs: tuple[EvidenceRef, ...]
    route_ids: tuple[str, ...]
    coverage: Mapping[str, float | int | None]
    hold: bool
    hidden_fallback_used: bool = False


def _coverage(required: np.ndarray, supplied: np.ndarray) -> float:
    count = int(np.count_nonzero(required))
    if count == 0:
        return 1.0
    return float(np.count_nonzero(required & supplied) / count)


def _quality_pass(
    *,
    claimed_coverage: float | None,
    actual_coverage: float,
    coverage_min: float,
    flow: float | None,
    flow_min: float,
    photometric: float | None,
    photometric_max: float,
) -> bool:
    if claimed_coverage is None or flow is None or photometric is None:
        return False
    if not all(math.isfinite(value) for value in (claimed_coverage, flow, photometric)):
        return False
    return (
        math.isclose(claimed_coverage, actual_coverage, rel_tol=0, abs_tol=1e-9)
        and actual_coverage >= coverage_min
        and flow >= flow_min
        and photometric <= photometric_max
    )


def assemble_clean_frame(
    *,
    source_rgb: np.ndarray,
    reveal: RevealFrameResult,
    policy_ref: EvidenceRef,
    object_texture: ObjectTextureEvidence | None,
    background_donor: TemporalBackgroundEvidence | None,
    execution_mode: str,
) -> CleanFrameResult:
    """Assemble CLEAN pixels with exact reveal-class donor boundaries."""

    if execution_mode != "SYNTHETIC_DRY_RUN":
        raise FormalCleanBlocked(
            "FORMAL_CLEAN_BLOCKED: no authenticated real atlas/manifests and no formal authorization"
        )
    policy = load_verified_reveal_policy(policy_ref)
    if reveal.policy_ref != policy_ref:
        raise EvidenceIntegrityError("CLEAN policy does not equal reveal policy")

    source = _rgb_plane(source_rgb, name="source RGB")
    if source.shape[:2] != reveal.labels.shape:
        raise ValueError("source RGB dimensions differ from reveal labels")
    labels = np.asarray(reveal.labels)
    target = np.asarray(reveal.target_mask, dtype=bool)
    declared_unsupported = np.asarray(reveal.unsupported_mask, dtype=bool)
    if target.shape != labels.shape or declared_unsupported.shape != labels.shape:
        raise ValueError("reveal target/unsupported dimensions differ from labels")
    if not np.isin(labels, [label.value for label in RevealLabel]).all():
        raise ValueError("reveal labels contain an unknown class")
    if np.any(target & (labels == RevealLabel.KEEP_SOURCE)):
        raise ValueError("reveal target may not use KEEP_SOURCE")
    if np.any(~target & (labels != RevealLabel.KEEP_SOURCE)):
        raise ValueError("non-target pixels may only use KEEP_SOURCE")
    if not np.array_equal(declared_unsupported, labels == RevealLabel.HOLD_UNSUPPORTED):
        raise ValueError("unsupported mask and HOLD labels disagree")
    if reveal.fallback_used:
        raise ValueError("CLEAN refuses reveal results that report fallback use")

    object_required = labels == RevealLabel.REVEAL_OBJECT
    background_required = labels == RevealLabel.REVEAL_BACKGROUND
    allowed_routes = set(policy.route_registry)
    if any(route not in allowed_routes for route in reveal.route_ids):
        raise ValueError("reveal result contains an unregistered CLEAN route")
    if np.any(object_required) and not any(
        policy.route_registry.get(route) == "REVEAL_OBJECT" for route in reveal.route_ids
    ):
        raise ValueError("object reveal has no declared Object6D/analytic route")
    if np.any(background_required) and not any(
        policy.route_registry.get(route) == "REVEAL_BACKGROUND" for route in reveal.route_ids
    ):
        raise ValueError("background reveal has no declared temporal donor route")
    unsupported = declared_unsupported.copy()

    object_supplied = np.zeros_like(object_required)
    if object_texture is not None:
        if object_texture.provenance_ref != reveal.object_texture_ref:
            raise EvidenceIntegrityError("CLEAN object donor does not equal reveal object donor")
        object_texture.provenance_ref.verified_json()
        if object_texture.rgb.shape != source.shape:
            raise ValueError("object donor RGB dimensions differ from source")
        object_candidate = object_required & object_texture.support_mask
        if _quality_pass(
            claimed_coverage=object_texture.coverage_ratio,
            actual_coverage=_coverage(object_required, object_candidate),
            coverage_min=policy.object_donor_coverage_min,
            flow=object_texture.flow_consistency,
            flow_min=policy.donor_flow_consistency_min,
            photometric=object_texture.photometric_residual_normalized,
            photometric_max=policy.donor_photometric_residual_normalized_max,
        ):
            object_supplied = object_candidate
    unsupported |= object_required & ~object_supplied

    background_supplied = np.zeros_like(background_required)
    if background_donor is not None:
        if background_donor.provenance_ref != reveal.background_donor_ref:
            raise EvidenceIntegrityError("CLEAN background donor does not equal reveal background donor")
        background_donor.provenance_ref.verified_json()
        if background_donor.rgb.shape != source.shape:
            raise ValueError("background donor RGB dimensions differ from source")
        background_candidate = background_required & background_donor.support_mask
        if _quality_pass(
            claimed_coverage=background_donor.coverage_ratio,
            actual_coverage=_coverage(background_required, background_candidate),
            coverage_min=policy.background_donor_coverage_min,
            flow=background_donor.flow_consistency,
            flow_min=policy.donor_flow_consistency_min,
            photometric=background_donor.photometric_residual_normalized,
            photometric_max=policy.donor_photometric_residual_normalized_max,
        ):
            background_supplied = background_candidate
    unsupported |= background_required & ~background_supplied

    clean = source.copy()
    object_layer = np.zeros_like(source)
    background_layer = np.zeros_like(source)
    if object_texture is not None:
        clean[object_supplied] = object_texture.rgb[object_supplied]
        object_layer[object_supplied] = object_texture.rgb[object_supplied]
    if background_donor is not None:
        clean[background_supplied] = background_donor.rgb[background_supplied]
        background_layer[background_supplied] = background_donor.rgb[background_supplied]

    # Unsupported reveal targets are deliberately conspicuous and never copied
    # from source RGB.  The explicit mask remains the authority for HOLD.
    clean[unsupported] = UNSUPPORTED_SENTINEL_RGB

    provenance: list[EvidenceRef] = []
    if np.any(object_supplied) and object_texture is not None:
        provenance.append(object_texture.provenance_ref)
    if np.any(background_supplied) and background_donor is not None:
        provenance.append(background_donor.provenance_ref)

    coverage: dict[str, float | int | None] = {
        "object_reveal_coverage": _coverage(object_required, object_supplied),
        "background_reveal_coverage": _coverage(background_required, background_supplied),
        "unsupported_pixel_count": int(np.count_nonzero(unsupported)),
        "unsupported_ratio": float(
            np.count_nonzero(unsupported) / max(1, np.count_nonzero(reveal.target_mask))
        ),
        "flow_consistency": (
            background_donor.flow_consistency
            if background_donor is not None and np.any(background_supplied)
            else None
        ),
        "photometric_residual": (
            background_donor.photometric_residual_normalized
            if background_donor is not None and np.any(background_supplied)
            else None
        ),
    }
    return CleanFrameResult(
        artifact_state="SYNTHETIC_TEST_ONLY",
        formal_clean_status="FORMAL_CLEAN_BLOCKED",
        clean_rgb=clean,
        object_reveal_rgb=object_layer,
        background_reveal_rgb=background_layer,
        unsupported_mask=unsupported,
        donor_provenance_refs=tuple(provenance),
        route_ids=reveal.route_ids,
        coverage=coverage,
        hold=bool(np.any(unsupported)),
    )
