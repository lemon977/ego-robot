from __future__ import annotations

import numpy as np
import pytest

from pipeline.oracle_clean_plumbing import (
    DonorPlane,
    OraclePlumbingError,
    SOURCE_OBJECT_ATLAS,
    SOURCE_TABLE_REPROJ,
    SOURCE_UNRESOLVED,
    assemble_oracle_clean_frame,
    validate_governance,
)


def labels():
    h = np.zeros((4, 5), dtype=bool)
    o = np.zeros_like(h)
    u = np.zeros_like(h)
    b = np.ones_like(h)
    h[1, 1] = True
    u[1, 2] = True
    o[2, 2] = True
    b[h | o | u] = False
    return h, o, u, b


def governance():
    return {
        "mask_source": "HUMAN_ORACLE_DEVELOPMENT_LABEL",
        "plumbing_validation_only": True,
        "is_clean_candidate": False,
        "candidate_requires_human_review": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read_splits": ["development"],
    }


def donor(code: int, support: np.ndarray, confidence: float = 0.9) -> DonorPlane:
    rgb = np.full((*support.shape, 3), 77 if code == SOURCE_OBJECT_ATLAS else 99, np.uint8)
    return DonorPlane(
        rgb,
        support,
        np.full(support.shape, confidence, np.float32),
        code,
        "a" * 64,
    )


def test_no_verified_donor_yields_explicit_residual_not_raw_fallback():
    raw = np.full((4, 5, 3), 13, np.uint8)
    h, o, u, b = labels()
    result = assemble_oracle_clean_frame(
        raw_rgb=raw, human=h, object_=o, uncertain=u, background=b,
        object_atlas=None, table_reprojection=None, confidence_min=0.8,
    )
    np.testing.assert_array_equal(result.clean_rgb, raw)
    assert np.all(result.source_map[h | u] == SOURCE_UNRESOLVED)
    assert result.metrics["u_band_filled_px"] == 0
    assert result.metrics["u_band_residual_px"] == 1
    assert result.metrics["filled_without_source_px"] == 0
    assert result.hold


def test_u_can_be_filled_only_with_source_and_confidence():
    raw = np.zeros((4, 5, 3), np.uint8)
    h, o, u, b = labels()
    support = np.zeros_like(h)
    support[u] = True
    result = assemble_oracle_clean_frame(
        raw_rgb=raw, human=h, object_=o, uncertain=u, background=b,
        object_atlas=donor(SOURCE_OBJECT_ATLAS, support),
        table_reprojection=None, confidence_min=0.8,
    )
    assert result.metrics["u_band_filled_px"] == 1
    assert result.metrics["u_band_filled_without_source_px"] == 0
    assert result.source_map[u].item() == SOURCE_OBJECT_ATLAS
    assert result.confidence_map[u].item() == pytest.approx(0.9)


def test_low_confidence_is_unresolved_and_never_written():
    raw = np.zeros((4, 5, 3), np.uint8)
    h, o, u, b = labels()
    support = h | u
    result = assemble_oracle_clean_frame(
        raw_rgb=raw, human=h, object_=o, uncertain=u, background=b,
        object_atlas=None,
        table_reprojection=donor(SOURCE_TABLE_REPROJ, support, confidence=0.1),
        confidence_min=0.8,
    )
    assert np.all(result.source_map[h | u] == SOURCE_UNRESOLVED)
    assert not np.any(result.clean_rgb)


def test_visible_object_and_outside_target_are_never_changed():
    raw = np.full((4, 5, 3), 11, np.uint8)
    h, o, u, b = labels()
    all_support = np.ones_like(h)
    result = assemble_oracle_clean_frame(
        raw_rgb=raw, human=h, object_=o, uncertain=u, background=b,
        object_atlas=donor(SOURCE_OBJECT_ATLAS, all_support),
        table_reprojection=donor(SOURCE_TABLE_REPROJ, all_support),
        confidence_min=0.8,
    )
    assert result.metrics["outside_mask_changed_px"] == 0
    assert result.metrics["visible_object_changed_px"] == 0
    np.testing.assert_array_equal(result.clean_rgb[o | b], raw[o | b])


def test_non_one_hot_labels_and_governance_drift_fail_closed():
    raw = np.zeros((4, 5, 3), np.uint8)
    h, o, u, b = labels()
    o[1, 1] = True
    with pytest.raises(OraclePlumbingError, match="exhaustive"):
        assemble_oracle_clean_frame(
            raw_rgb=raw, human=h, object_=o, uncertain=u, background=b,
            object_atlas=None, table_reprojection=None, confidence_min=0.8,
        )
    record = governance()
    validate_governance(record)
    record["formal_consumer_allowed"] = True
    with pytest.raises(OraclePlumbingError, match="governance drift"):
        validate_governance(record)
    record = governance()
    record["grap_a_cap_025"] = True
    with pytest.raises(OraclePlumbingError, match="forbidden"):
        validate_governance(record)
