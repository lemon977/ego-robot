from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pipeline.reveal_labels import (
    EvidenceRef,
    ObjectGeometryEvidence,
    ObjectTextureEvidence,
    TemporalBackgroundEvidence,
    array_sha256,
)


def write_json_ref(
    root: Path,
    name: str,
    payload: Mapping[str, object],
    *,
    producer: str,
    schema_version: str,
) -> EvidenceRef:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return EvidenceRef.from_file(
        str(path), producer=producer, schema_version=schema_version
    )


def write_bytes_ref(root: Path, name: str, payload: bytes, *, producer: str) -> EvidenceRef:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return EvidenceRef.from_file(str(path), producer=producer)


def policy_ref(root: Path, *, thresholds: Mapping[str, object] | None = None) -> EvidenceRef:
    values: dict[str, object] = {
        "object_donor_coverage_min": 0.8,
        "background_donor_coverage_min": 0.8,
        "donor_flow_consistency_min": 0.8,
        "donor_photometric_residual_normalized_max": 0.1,
    }
    if thresholds:
        values.update(thresholds)
    return write_json_ref(
        root,
        "policy.json",
        {
            "schema_version": "reveal-policy-v1",
            "producer": "reveal_policy_owner",
            "status": "VERIFIED",
            "route_registry": {
                "ANALYTIC_GEOMETRY_TEXTURE": "REVEAL_OBJECT",
                "TEMPORAL_DONOR_ATLAS": "REVEAL_BACKGROUND",
            },
            "thresholds": values,
        },
        producer="reveal_policy_owner",
        schema_version="reveal-policy-v1",
    )


def geometry_evidence(
    root: Path, mask: np.ndarray, *, source_kind: str = "ANALYTIC_GEOMETRY"
) -> ObjectGeometryEvidence:
    ref = write_json_ref(
        root,
        "geometry.json",
        {
            "schema_version": "object-geometry-evidence-v1",
            "producer": "object_geometry_producer",
            "status": "VERIFIED",
            "source_kind": source_kind,
            "support_mask_sha256": array_sha256(mask),
        },
        producer="object_geometry_producer",
        schema_version="object-geometry-evidence-v1",
    )
    return ObjectGeometryEvidence(mask, source_kind, ref)


def object_texture_evidence(
    root: Path,
    rgb: np.ndarray,
    support: np.ndarray,
    *,
    coverage: float | None,
    flow: float | None,
    photometric: float | None,
) -> ObjectTextureEvidence:
    ref = write_json_ref(
        root,
        "object_texture.json",
        {
            "schema_version": "object-texture-donor-v1",
            "producer": "object_geometry_producer",
            "status": "VERIFIED",
            "rgb_sha256": array_sha256(rgb),
            "support_mask_sha256": array_sha256(support),
            "coverage_ratio": coverage,
            "flow_consistency": flow,
            "photometric_residual_normalized": photometric,
        },
        producer="object_geometry_producer",
        schema_version="object-texture-donor-v1",
    )
    return ObjectTextureEvidence(rgb, support, ref, coverage, flow, photometric)


def background_evidence(
    root: Path,
    rgb: np.ndarray,
    identity: np.ndarray,
    support: np.ndarray,
    *,
    coverage: float | None,
    flow: float | None,
    photometric: float | None,
) -> TemporalBackgroundEvidence:
    ref = write_json_ref(
        root,
        "background.json",
        {
            "schema_version": "temporal-background-donor-v1",
            "producer": "temporal_donor_atlas",
            "status": "VERIFIED",
            "rgb_sha256": array_sha256(rgb),
            "background_identity_mask_sha256": array_sha256(identity),
            "support_mask_sha256": array_sha256(support),
            "coverage_ratio": coverage,
            "flow_consistency": flow,
            "photometric_residual_normalized": photometric,
        },
        producer="temporal_donor_atlas",
        schema_version="temporal-background-donor-v1",
    )
    return TemporalBackgroundEvidence(
        rgb, identity, support, ref, coverage, flow, photometric
    )
