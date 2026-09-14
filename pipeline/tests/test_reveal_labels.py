from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.reveal_labels import (
    EvidenceIntegrityError,
    EvidenceRef,
    RevealLabel,
    classify_reveal_pixels,
)
from pipeline.tests.evidence_test_utils import (
    background_evidence,
    geometry_evidence,
    object_texture_evidence,
    policy_ref,
)


def _complete_evidence(root: Path, geometry_mask: np.ndarray, background_mask: np.ndarray):
    object_rgb = np.full((*geometry_mask.shape, 3), 101, np.uint8)
    background_rgb = np.full((*geometry_mask.shape, 3), 202, np.uint8)
    return (
        policy_ref(root),
        geometry_evidence(root, geometry_mask),
        object_texture_evidence(
            root, object_rgb, geometry_mask, coverage=1.0, flow=0.95, photometric=0.03
        ),
        background_evidence(
            root,
            background_rgb,
            background_mask,
            background_mask,
            coverage=1.0,
            flow=0.96,
            photometric=0.02,
        ),
    )


def test_verified_routes_separate_identity_and_conflicts_hold(tmp_path: Path) -> None:
    shape = (3, 4)
    human = np.zeros(shape, bool)
    human[0, :] = True
    human[1, 0] = True
    uncertain = np.zeros(shape, bool)
    uncertain[2, 0] = True
    visible = np.zeros(shape, bool)
    visible[1, 0] = True
    geometry_mask = np.zeros(shape, bool)
    geometry_mask[0, 0] = geometry_mask[0, 3] = True
    geometry_mask[1, 0] = geometry_mask[2, 0] = True
    background_mask = np.zeros(shape, bool)
    background_mask[0, 1] = background_mask[0, 3] = True
    policy, geometry, texture, background = _complete_evidence(
        tmp_path, geometry_mask, background_mask
    )

    result = classify_reveal_pixels(
        h_core=human,
        o_visible_core=visible,
        u_contact=uncertain,
        policy_ref=policy,
        object_geometry=geometry,
        object_texture=texture,
        background_donor=background,
    )
    assert result.labels[0, 0] == RevealLabel.REVEAL_OBJECT
    assert result.labels[2, 0] == RevealLabel.KEEP_SOURCE
    assert result.labels[0, 1] == RevealLabel.REVEAL_BACKGROUND
    assert result.labels[0, 2] == RevealLabel.HOLD_UNSUPPORTED
    assert result.labels[0, 3] == RevealLabel.HOLD_UNSUPPORTED
    assert result.labels[1, 0] == RevealLabel.HOLD_UNSUPPORTED
    assert result.route_ids == ("ANALYTIC_GEOMETRY_TEXTURE", "TEMPORAL_DONOR_ATLAS")
    assert result.counts["u_contact_keep_source"] == 1
    assert not result.target_mask[2, 0]
    assert "CONTRADICTORY_REVEAL_IDENTITY" in result.hold_reasons
    assert "VISIBLE_OBJECT_MASK_CONFLICT" in result.hold_reasons
    assert result.fallback_used is False


@pytest.mark.parametrize(
    "quality_override",
    [
        {"coverage": None, "flow": 0.95, "photometric": 0.03},
        {"coverage": 1.0, "flow": None, "photometric": 0.03},
        {"coverage": 1.0, "flow": 0.2, "photometric": 0.03},
        {"coverage": 1.0, "flow": 0.95, "photometric": 0.9},
    ],
)
def test_missing_or_failed_donor_quality_is_hold(
    tmp_path: Path, quality_override: dict[str, float | None]
) -> None:
    mask = np.ones((2, 2), bool)
    policy = policy_ref(tmp_path)
    geometry = geometry_evidence(tmp_path, mask)
    texture = object_texture_evidence(
        tmp_path,
        np.full((2, 2, 3), 5, np.uint8),
        mask,
        coverage=quality_override["coverage"],
        flow=quality_override["flow"],
        photometric=quality_override["photometric"],
    )
    result = classify_reveal_pixels(
        h_core=mask,
        o_visible_core=np.zeros_like(mask),
        u_contact=np.zeros_like(mask),
        policy_ref=policy,
        object_geometry=geometry,
        object_texture=texture,
        background_donor=None,
    )
    assert np.all(result.labels == RevealLabel.HOLD_UNSUPPORTED)
    assert "OBJECT_DONOR_QUALITY_UNVERIFIED_OR_BELOW_THRESHOLD" in result.hold_reasons


def test_missing_policy_threshold_holds_instead_of_enabling_defaults(tmp_path: Path) -> None:
    mask = np.ones((1, 1), bool)
    policy = policy_ref(
        tmp_path,
        thresholds={"donor_flow_consistency_min": None},
    )
    geometry = geometry_evidence(tmp_path, mask)
    texture = object_texture_evidence(
        tmp_path,
        np.full((1, 1, 3), 7, np.uint8),
        mask,
        coverage=1.0,
        flow=1.0,
        photometric=0.0,
    )
    result = classify_reveal_pixels(
        h_core=mask,
        o_visible_core=np.zeros_like(mask),
        u_contact=np.zeros_like(mask),
        policy_ref=policy,
        object_geometry=geometry,
        object_texture=texture,
        background_donor=None,
    )
    assert result.hold
    assert "VERIFIED_POLICY_INVALID" in result.hold_reasons
    assert result.route_ids == ()


def test_background_donor_without_flow_metric_is_hold(tmp_path: Path) -> None:
    mask = np.ones((1, 2), bool)
    background = background_evidence(
        tmp_path,
        np.full((1, 2, 3), 8, np.uint8),
        mask,
        mask,
        coverage=1.0,
        flow=None,
        photometric=0.0,
    )
    result = classify_reveal_pixels(
        h_core=mask,
        o_visible_core=np.zeros_like(mask),
        u_contact=np.zeros_like(mask),
        policy_ref=policy_ref(tmp_path),
        object_geometry=None,
        object_texture=None,
        background_donor=background,
    )
    assert result.hold
    assert np.all(result.labels == RevealLabel.HOLD_UNSUPPORTED)
    assert "BACKGROUND_DONOR_QUALITY_UNVERIFIED_OR_BELOW_THRESHOLD" in result.hold_reasons


def test_current_registry_does_not_admit_object6d_route(tmp_path: Path) -> None:
    mask = np.ones((1, 1), bool)
    policy = policy_ref(tmp_path)
    geometry = geometry_evidence(tmp_path, mask, source_kind="OBJECT6D_GEOMETRY")
    texture = object_texture_evidence(
        tmp_path,
        np.full((1, 1, 3), 7, np.uint8),
        mask,
        coverage=1.0,
        flow=1.0,
        photometric=0.0,
    )
    result = classify_reveal_pixels(
        h_core=mask,
        o_visible_core=np.zeros_like(mask),
        u_contact=np.zeros_like(mask),
        policy_ref=policy,
        object_geometry=geometry,
        object_texture=texture,
        background_donor=None,
    )
    assert result.hold
    assert "OBJECT_ROUTE_NOT_IN_VERIFIED_REGISTRY" in result.hold_reasons


def test_evidence_ref_rejects_tampered_or_symlinked_file(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"status": "VERIFIED"}), encoding="utf-8")
    ref = EvidenceRef.from_file(str(path), producer="x", schema_version="x-v1")
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(EvidenceIntegrityError, match="byte count|digest mismatch"):
        ref.verified_bytes()

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    linked_ref = EvidenceRef(
        str(link),
        EvidenceRef.from_file(str(target), producer="x").sha256,
        "x",
        target.stat().st_size,
    )
    with pytest.raises(EvidenceIntegrityError, match="without symlinks"):
        linked_ref.verified_bytes()
