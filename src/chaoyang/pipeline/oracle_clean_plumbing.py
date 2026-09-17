"""Oracle-mask CLEAN plumbing contract for public development frames only.

This module cannot create a CLEAN candidate.  It either applies pixels from
explicitly supplied, source-bound object/table donors or leaves those target
pixels unresolved.  The first project run intentionally exercises the latter
path because no verified object-atlas or table-plane donor exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


SOURCE_KEEP = 0
SOURCE_OBJECT_ATLAS = 1
SOURCE_TABLE_REPROJ = 2
SOURCE_UNRESOLVED = 3
SOURCE_NAMES = {
    SOURCE_KEEP: "keep_source",
    SOURCE_OBJECT_ATLAS: "object_atlas",
    SOURCE_TABLE_REPROJ: "table_reproj",
    SOURCE_UNRESOLVED: "unresolved",
}

REQUIRED_GOVERNANCE = {
    "mask_source": "HUMAN_ORACLE_DEVELOPMENT_LABEL",
    "plumbing_validation_only": True,
    "is_clean_candidate": False,
    "candidate_requires_human_review": True,
    "advancement_authorized": False,
    "formal_consumer_allowed": False,
    "labels_read_splits": ["development"],
}


class OraclePlumbingError(RuntimeError):
    """Raised on label, donor, output or governance drift."""


@dataclass(frozen=True)
class DonorPlane:
    rgb: np.ndarray
    support: np.ndarray
    confidence: np.ndarray
    source_code: int
    provenance_sha256: str


@dataclass(frozen=True)
class OracleCleanFrame:
    clean_rgb: np.ndarray
    source_map: np.ndarray
    confidence_map: np.ndarray
    residual_mask: np.ndarray
    metrics: Mapping[str, int | float]
    hold: bool
    hold_reasons: tuple[str, ...]


def validate_governance(record: Mapping[str, object]) -> None:
    for key, expected in REQUIRED_GOVERNANCE.items():
        observed = record.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise OraclePlumbingError(f"oracle plumbing governance drift: {key}")
    forbidden = {
        "same_session_blind",
        "cross_session_blind",
        "grap_a_cap_025",
        "grap_a_cap_149",
        "mask_producer_score",
        "mask_candidate_ref",
        "processed_output",
        "hidden_fallback",
    }
    if forbidden & set(record):
        raise OraclePlumbingError("oracle plumbing contains a forbidden consumer/input")


def validate_houb(
    human: np.ndarray,
    object_: np.ndarray,
    uncertain: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    planes = tuple(np.asarray(x) for x in (human, object_, uncertain, background))
    if any(x.ndim != 2 or x.dtype != np.bool_ for x in planes):
        raise OraclePlumbingError("H/O/U/B must be 2D boolean planes")
    if len({x.shape for x in planes}) != 1:
        raise OraclePlumbingError("H/O/U/B shape drift")
    membership = sum(x.astype(np.uint8) for x in planes)
    if np.any(membership != 1):
        raise OraclePlumbingError("H/O/U/B must be exhaustive and mutually exclusive")
    return planes


def _validate_donor(donor: DonorPlane, shape: tuple[int, int], code: int) -> None:
    if donor.source_code != code or code not in (SOURCE_OBJECT_ATLAS, SOURCE_TABLE_REPROJ):
        raise OraclePlumbingError("donor source code drift")
    if donor.rgb.shape != (*shape, 3) or donor.rgb.dtype != np.uint8:
        raise OraclePlumbingError("donor RGB shape/type drift")
    if donor.support.shape != shape or donor.support.dtype != np.bool_:
        raise OraclePlumbingError("donor support shape/type drift")
    if donor.confidence.shape != shape or not np.issubdtype(donor.confidence.dtype, np.floating):
        raise OraclePlumbingError("donor confidence shape/type drift")
    if not donor.provenance_sha256 or len(donor.provenance_sha256) != 64:
        raise OraclePlumbingError("donor provenance is not digest-bound")
    if not np.all(np.isfinite(donor.confidence)) or np.any(
        (donor.confidence < 0) | (donor.confidence > 1)
    ):
        raise OraclePlumbingError("donor confidence outside [0,1]")


def assemble_oracle_clean_frame(
    *,
    raw_rgb: np.ndarray,
    human: np.ndarray,
    object_: np.ndarray,
    uncertain: np.ndarray,
    background: np.ndarray,
    object_atlas: DonorPlane | None,
    table_reprojection: DonorPlane | None,
    confidence_min: float,
) -> OracleCleanFrame:
    """Assemble one isolated oracle-plumbing frame without any fallback."""

    h, o, u, _ = validate_houb(human, object_, uncertain, background)
    raw = np.asarray(raw_rgb)
    if raw.shape != (*h.shape, 3) or raw.dtype != np.uint8:
        raise OraclePlumbingError("RAW shape/type drift")
    if not np.isfinite(confidence_min) or not 0.0 <= confidence_min <= 1.0:
        raise OraclePlumbingError("confidence threshold outside [0,1]")

    target = h | u
    clean = raw.copy()
    source = np.full(h.shape, SOURCE_KEEP, dtype=np.uint8)
    confidence = np.zeros(h.shape, dtype=np.float32)

    if object_atlas is not None:
        _validate_donor(object_atlas, h.shape, SOURCE_OBJECT_ATLAS)
        accepted = (
            target
            & ~o
            & object_atlas.support
            & (object_atlas.confidence >= confidence_min)
        )
        clean[accepted] = object_atlas.rgb[accepted]
        source[accepted] = SOURCE_OBJECT_ATLAS
        confidence[accepted] = object_atlas.confidence[accepted]

    if table_reprojection is not None:
        _validate_donor(table_reprojection, h.shape, SOURCE_TABLE_REPROJ)
        accepted = (
            target
            & ~o
            & (source == SOURCE_KEEP)
            & table_reprojection.support
            & (table_reprojection.confidence >= confidence_min)
        )
        clean[accepted] = table_reprojection.rgb[accepted]
        source[accepted] = SOURCE_TABLE_REPROJ
        confidence[accepted] = table_reprojection.confidence[accepted]

    residual = target & (source == SOURCE_KEEP)
    source[residual] = SOURCE_UNRESOLVED
    changed = np.any(clean != raw, axis=2)
    filled = target & ~residual
    sourced = np.isin(source, (SOURCE_OBJECT_ATLAS, SOURCE_TABLE_REPROJ))
    without_source = filled & (~sourced | (confidence <= 0))

    metrics: dict[str, int | float] = {
        "outside_mask_changed_px": int(np.count_nonzero(changed & ~target)),
        "visible_object_changed_px": int(np.count_nonzero(changed & o)),
        "filled_without_source_px": int(np.count_nonzero(without_source)),
        "u_band_filled_without_source_px": int(np.count_nonzero(without_source & u)),
        "h_core_filled_px": int(np.count_nonzero(filled & h)),
        "h_core_residual_px": int(np.count_nonzero(residual & h)),
        "u_band_filled_px": int(np.count_nonzero(filled & u)),
        "u_band_residual_px": int(np.count_nonzero(residual & u)),
        "object_atlas_px": int(np.count_nonzero(source == SOURCE_OBJECT_ATLAS)),
        "table_reproj_px": int(np.count_nonzero(source == SOURCE_TABLE_REPROJ)),
        "unresolved_px": int(np.count_nonzero(source == SOURCE_UNRESOLVED)),
        "target_px": int(np.count_nonzero(target)),
        "unresolved_ratio": float(np.count_nonzero(residual) / max(1, np.count_nonzero(target))),
    }
    hard_gate_names = (
        "outside_mask_changed_px",
        "visible_object_changed_px",
        "filled_without_source_px",
        "u_band_filled_without_source_px",
    )
    if any(metrics[name] != 0 for name in hard_gate_names):
        raise OraclePlumbingError("oracle plumbing hard gate violated")
    reasons = []
    if object_atlas is None:
        reasons.append("VERIFIED_OBJECT_ATLAS_UNAVAILABLE")
    if table_reprojection is None:
        reasons.append("VERIFIED_TABLE_PLANE_REPROJECTION_UNAVAILABLE")
    if np.any(residual):
        reasons.append("UNRESOLVED_ORACLE_TARGET_PIXELS")
    return OracleCleanFrame(
        clean,
        source,
        confidence,
        residual,
        metrics,
        bool(reasons),
        tuple(reasons),
    )
