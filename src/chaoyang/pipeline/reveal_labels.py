"""Digest-bound, fail-closed reveal identity routing.

Every policy and evidence plane is authenticated by an ordinary-file JSON
manifest. The router consumes a finite verified registry and normalized donor
thresholds; it never invents an enabled route or fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import math
import os
import re
import stat
from typing import Any, Mapping

import numpy as np


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class EvidenceIntegrityError(RuntimeError):
    """An evidence path, digest or typed manifest is not trustworthy."""


def _read_ordinary_no_symlinks(path: str) -> bytes:
    if not path:
        raise EvidenceIntegrityError("evidence path is empty")
    absolute = os.path.isabs(path)
    parts = [part for part in path.split(os.sep) if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise EvidenceIntegrityError(f"unsafe evidence path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep if absolute else ".", directory_flags)
    try:
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise EvidenceIntegrityError(f"evidence is not an ordinary file: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise EvidenceIntegrityError(f"cannot read evidence without symlinks: {path}: {error}") from error
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str
    producer: str
    bytes: int
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if not self.path or not self.producer:
            raise ValueError("evidence path and producer must be non-empty")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("evidence sha256 must be 64 lowercase hex characters")
        if self.bytes <= 0:
            raise ValueError("evidence bytes must be positive")

    @classmethod
    def from_file(
        cls, path: str, *, producer: str, schema_version: str | None = None
    ) -> "EvidenceRef":
        payload = _read_ordinary_no_symlinks(path)
        return cls(
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            producer=producer,
            bytes=len(payload),
            schema_version=schema_version,
        )

    def verified_bytes(self) -> bytes:
        payload = _read_ordinary_no_symlinks(self.path)
        if len(payload) != self.bytes:
            raise EvidenceIntegrityError(f"evidence byte count mismatch: {self.path}")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise EvidenceIntegrityError(f"evidence digest mismatch: {self.path}")
        return payload

    def verified_json(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.verified_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvidenceIntegrityError(f"invalid evidence JSON: {self.path}: {error}") from error
        if not isinstance(value, dict):
            raise EvidenceIntegrityError(f"evidence JSON root is not an object: {self.path}")
        return value

    def to_dict(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {
            "path": self.path,
            "sha256": self.sha256,
            "producer": self.producer,
            "bytes": self.bytes,
        }
        if self.schema_version is not None:
            value["schema_version"] = self.schema_version
        return value


def _bool_plane(value: np.ndarray, *, name: str) -> np.ndarray:
    plane = np.asarray(value)
    if plane.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional plane")
    if plane.dtype != np.bool_:
        if not np.issubdtype(plane.dtype, np.integer) or not np.isin(plane, (0, 1)).all():
            raise ValueError(f"{name} must contain only boolean/0/1 values")
    return plane.astype(bool, copy=False)


def _rgb_plane(value: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be an HxWx3 uint8 RGB image")
    return image


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _typed_manifest(
    ref: EvidenceRef, *, expected_schema: str, expected_producer: str
) -> Mapping[str, Any]:
    if ref.schema_version != expected_schema or ref.producer != expected_producer:
        raise EvidenceIntegrityError("EvidenceRef producer/schema does not match required type")
    payload = ref.verified_json()
    if payload.get("schema_version") != expected_schema:
        raise EvidenceIntegrityError("evidence manifest schema_version mismatch")
    if payload.get("producer") != expected_producer:
        raise EvidenceIntegrityError("evidence manifest producer mismatch")
    if payload.get("status") != "VERIFIED":
        raise EvidenceIntegrityError("evidence manifest status is not VERIFIED")
    return payload


@dataclass(frozen=True)
class VerifiedRevealPolicy:
    ref: EvidenceRef
    route_registry: Mapping[str, str]
    object_donor_coverage_min: float
    background_donor_coverage_min: float
    donor_flow_consistency_min: float
    donor_photometric_residual_normalized_max: float


def load_verified_reveal_policy(ref: EvidenceRef) -> VerifiedRevealPolicy:
    payload = _typed_manifest(
        ref, expected_schema="reveal-policy-v1", expected_producer="reveal_policy_owner"
    )
    registry = payload.get("route_registry")
    thresholds = payload.get("thresholds")
    if not isinstance(registry, dict) or not registry:
        raise EvidenceIntegrityError("verified reveal policy has no finite route registry")
    if any(
        not isinstance(route, str)
        or reveal_class not in {"REVEAL_OBJECT", "REVEAL_BACKGROUND"}
        for route, reveal_class in registry.items()
    ):
        raise EvidenceIntegrityError("verified reveal policy has invalid route registry")
    if not isinstance(thresholds, dict):
        raise EvidenceIntegrityError("verified reveal policy has no threshold object")
    names = (
        "object_donor_coverage_min",
        "background_donor_coverage_min",
        "donor_flow_consistency_min",
        "donor_photometric_residual_normalized_max",
    )
    values: dict[str, float] = {}
    for name in names:
        value = thresholds.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise EvidenceIntegrityError(f"verified policy threshold missing/invalid: {name}")
        values[name] = float(value)
    if any(not 0 <= values[name] <= 1 for name in names[:3]):
        raise EvidenceIntegrityError("verified reveal policy ratio threshold outside [0,1]")
    if values[names[3]] < 0:
        raise EvidenceIntegrityError("photometric residual threshold must be non-negative")
    return VerifiedRevealPolicy(ref=ref, route_registry=dict(registry), **values)


def _verified_array_manifest(
    ref: EvidenceRef,
    *,
    expected_schema: str,
    expected_producer: str,
    arrays: Mapping[str, np.ndarray],
    expected_fields: Mapping[str, object],
) -> Mapping[str, Any]:
    payload = _typed_manifest(
        ref, expected_schema=expected_schema, expected_producer=expected_producer
    )
    for name, array in arrays.items():
        if payload.get(f"{name}_sha256") != array_sha256(array):
            raise EvidenceIntegrityError(f"{name} array is not bound by evidence manifest")
    for name, expected in expected_fields.items():
        if payload.get(name) != expected:
            raise EvidenceIntegrityError(f"evidence field mismatch: {name}")
    return payload


class RevealLabel(IntEnum):
    KEEP_SOURCE = 0
    REVEAL_OBJECT = 1
    REVEAL_BACKGROUND = 2
    HOLD_UNSUPPORTED = 3


@dataclass(frozen=True)
class ObjectGeometryEvidence:
    support_mask: np.ndarray
    source_kind: str
    provenance_ref: EvidenceRef

    def __post_init__(self) -> None:
        mask = _bool_plane(self.support_mask, name="object geometry support").copy()
        mask.setflags(write=False)
        object.__setattr__(self, "support_mask", mask)
        if self.source_kind not in {"OBJECT6D_GEOMETRY", "ANALYTIC_GEOMETRY"}:
            raise ValueError("object geometry kind is not Object6D/analytic")
        self.verify()

    def verify(self) -> None:
        _verified_array_manifest(
            self.provenance_ref,
            expected_schema="object-geometry-evidence-v1",
            expected_producer="object_geometry_producer",
            arrays={"support_mask": self.support_mask},
            expected_fields={"source_kind": self.source_kind},
        )

    @property
    def route_id(self) -> str:
        return (
            "OBJECT6D_GEOMETRY_TEXTURE"
            if self.source_kind == "OBJECT6D_GEOMETRY"
            else "ANALYTIC_GEOMETRY_TEXTURE"
        )


@dataclass(frozen=True)
class ObjectTextureEvidence:
    rgb: np.ndarray
    support_mask: np.ndarray
    provenance_ref: EvidenceRef
    coverage_ratio: float | None
    flow_consistency: float | None
    photometric_residual_normalized: float | None

    def __post_init__(self) -> None:
        rgb = _rgb_plane(self.rgb, name="object texture donor").copy()
        mask = _bool_plane(self.support_mask, name="object texture support").copy()
        if rgb.shape[:2] != mask.shape:
            raise ValueError("object texture RGB and support dimensions differ")
        rgb.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "support_mask", mask)
        self.verify()

    def verify(self) -> None:
        _verified_array_manifest(
            self.provenance_ref,
            expected_schema="object-texture-donor-v1",
            expected_producer="object_geometry_producer",
            arrays={"rgb": self.rgb, "support_mask": self.support_mask},
            expected_fields={
                "coverage_ratio": self.coverage_ratio,
                "flow_consistency": self.flow_consistency,
                "photometric_residual_normalized": self.photometric_residual_normalized,
            },
        )


@dataclass(frozen=True)
class TemporalBackgroundEvidence:
    rgb: np.ndarray
    background_identity_mask: np.ndarray
    support_mask: np.ndarray
    provenance_ref: EvidenceRef
    coverage_ratio: float | None
    flow_consistency: float | None
    photometric_residual_normalized: float | None

    def __post_init__(self) -> None:
        rgb = _rgb_plane(self.rgb, name="temporal background donor").copy()
        identity = _bool_plane(self.background_identity_mask, name="background identity").copy()
        support = _bool_plane(self.support_mask, name="temporal background support").copy()
        if rgb.shape[:2] != identity.shape or identity.shape != support.shape:
            raise ValueError("temporal background evidence dimensions differ")
        if np.any(support & ~identity):
            raise ValueError("background donor support exceeds proven background identity")
        rgb.setflags(write=False)
        identity.setflags(write=False)
        support.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "background_identity_mask", identity)
        object.__setattr__(self, "support_mask", support)
        self.verify()

    def verify(self) -> None:
        _verified_array_manifest(
            self.provenance_ref,
            expected_schema="temporal-background-donor-v1",
            expected_producer="temporal_donor_atlas",
            arrays={
                "rgb": self.rgb,
                "background_identity_mask": self.background_identity_mask,
                "support_mask": self.support_mask,
            },
            expected_fields={
                "coverage_ratio": self.coverage_ratio,
                "flow_consistency": self.flow_consistency,
                "photometric_residual_normalized": self.photometric_residual_normalized,
            },
        )


@dataclass(frozen=True)
class RevealFrameResult:
    labels: np.ndarray
    target_mask: np.ndarray
    unsupported_mask: np.ndarray
    route_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    policy_ref: EvidenceRef | None
    object_texture_ref: EvidenceRef | None
    background_donor_ref: EvidenceRef | None
    counts: Mapping[str, int]
    quality_metrics: Mapping[str, float | None]
    hold_reasons: tuple[str, ...]
    hold: bool
    fallback_used: bool = False


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    count = int(np.count_nonzero(denominator))
    return 1.0 if count == 0 else float(np.count_nonzero(numerator & denominator) / count)


def _donor_quality_pass(
    *,
    actual_coverage: float,
    claimed_coverage: float | None,
    flow_consistency: float | None,
    photometric_residual: float | None,
    coverage_min: float,
    policy: VerifiedRevealPolicy,
) -> bool:
    values = (claimed_coverage, flow_consistency, photometric_residual)
    if any(value is None or not math.isfinite(value) for value in values):
        return False
    assert claimed_coverage is not None and flow_consistency is not None
    assert photometric_residual is not None
    return (
        math.isclose(claimed_coverage, actual_coverage, rel_tol=0, abs_tol=1e-9)
        and actual_coverage >= coverage_min
        and flow_consistency >= policy.donor_flow_consistency_min
        and photometric_residual <= policy.donor_photometric_residual_normalized_max
    )


def classify_reveal_pixels(
    *,
    h_core: np.ndarray,
    o_visible_core: np.ndarray,
    u_contact: np.ndarray,
    policy_ref: EvidenceRef | None,
    object_geometry: ObjectGeometryEvidence | None,
    object_texture: ObjectTextureEvidence | None,
    background_donor: TemporalBackgroundEvidence | None,
) -> RevealFrameResult:
    human = _bool_plane(h_core, name="h_core")
    visible_object = _bool_plane(o_visible_core, name="o_visible_core")
    uncertain = _bool_plane(u_contact, name="u_contact")
    if human.shape != visible_object.shape or human.shape != uncertain.shape:
        raise ValueError("H_core, O_visible_core and U_contact dimensions differ")
    # U is an observed hand/object ambiguity band.  CLEAN must retain the RAW
    # observation there; it is not a reveal target and never consumes a donor.
    target = human
    reasons: list[str] = []
    try:
        policy = load_verified_reveal_policy(policy_ref) if policy_ref is not None else None
    except EvidenceIntegrityError:
        policy = None
        reasons.append("VERIFIED_POLICY_INVALID")
    if policy is None:
        reasons.append("VERIFIED_POLICY_MISSING")

    geometry_support = np.zeros_like(target)
    texture_support = np.zeros_like(target)
    background_identity = np.zeros_like(target)
    background_support = np.zeros_like(target)
    evidence_refs: list[EvidenceRef] = []
    object_route: str | None = None
    if object_geometry is not None:
        object_geometry.verify()
        if object_geometry.support_mask.shape != target.shape:
            raise ValueError("object geometry dimensions differ from MASK evidence")
        geometry_support = object_geometry.support_mask
        object_route = object_geometry.route_id
        evidence_refs.append(object_geometry.provenance_ref)
    if object_texture is not None:
        object_texture.verify()
        if object_texture.support_mask.shape != target.shape:
            raise ValueError("object texture dimensions differ from MASK evidence")
        texture_support = object_texture.support_mask
        evidence_refs.append(object_texture.provenance_ref)
    if background_donor is not None:
        background_donor.verify()
        if background_donor.support_mask.shape != target.shape:
            raise ValueError("background donor dimensions differ from MASK evidence")
        background_identity = background_donor.background_identity_mask
        background_support = background_donor.support_mask
        evidence_refs.append(background_donor.provenance_ref)

    object_identity = target & geometry_support
    background_identity = target & background_identity
    contradictory = object_identity & background_identity
    protected_conflict = target & visible_object
    object_actual_coverage = _ratio(texture_support, object_identity)
    background_actual_coverage = _ratio(background_support, background_identity)
    object_route_enabled = bool(
        policy is not None
        and object_route is not None
        and policy.route_registry.get(object_route) == "REVEAL_OBJECT"
    )
    background_route_enabled = bool(
        policy is not None
        and policy.route_registry.get("TEMPORAL_DONOR_ATLAS") == "REVEAL_BACKGROUND"
    )
    object_quality = bool(
        policy is not None
        and object_texture is not None
        and _donor_quality_pass(
            actual_coverage=object_actual_coverage,
            claimed_coverage=object_texture.coverage_ratio,
            flow_consistency=object_texture.flow_consistency,
            photometric_residual=object_texture.photometric_residual_normalized,
            coverage_min=policy.object_donor_coverage_min,
            policy=policy,
        )
    )
    background_quality = bool(
        policy is not None
        and background_donor is not None
        and _donor_quality_pass(
            actual_coverage=background_actual_coverage,
            claimed_coverage=background_donor.coverage_ratio,
            flow_consistency=background_donor.flow_consistency,
            photometric_residual=background_donor.photometric_residual_normalized,
            coverage_min=policy.background_donor_coverage_min,
            policy=policy,
        )
    )
    if np.any(object_identity) and not object_route_enabled:
        reasons.append("OBJECT_ROUTE_NOT_IN_VERIFIED_REGISTRY")
    if np.any(object_identity) and not object_quality:
        reasons.append("OBJECT_DONOR_QUALITY_UNVERIFIED_OR_BELOW_THRESHOLD")
    if np.any(background_identity) and not background_route_enabled:
        reasons.append("BACKGROUND_ROUTE_NOT_IN_VERIFIED_REGISTRY")
    if np.any(background_identity) and not background_quality:
        reasons.append("BACKGROUND_DONOR_QUALITY_UNVERIFIED_OR_BELOW_THRESHOLD")

    object_ready = (
        object_identity & texture_support & object_route_enabled & object_quality
        & ~contradictory & ~protected_conflict
    )
    background_ready = (
        background_identity & background_support & background_route_enabled & background_quality
        & ~geometry_support & ~protected_conflict
    )
    labels = np.full(target.shape, RevealLabel.KEEP_SOURCE, dtype=np.uint8)
    labels[object_ready] = RevealLabel.REVEAL_OBJECT
    labels[background_ready] = RevealLabel.REVEAL_BACKGROUND
    unsupported = target & ~(object_ready | background_ready)
    labels[unsupported] = RevealLabel.HOLD_UNSUPPORTED
    if np.any(contradictory):
        reasons.append("CONTRADICTORY_REVEAL_IDENTITY")
    if np.any(protected_conflict):
        reasons.append("VISIBLE_OBJECT_MASK_CONFLICT")
    if np.any(unsupported) and not reasons:
        reasons.append("REVEAL_EVIDENCE_MISSING")

    route_ids: list[str] = []
    if np.any(object_ready) and object_route is not None:
        route_ids.append(object_route)
    if np.any(background_ready):
        route_ids.append("TEMPORAL_DONOR_ATLAS")
    counts = {
        "target": int(np.count_nonzero(target)),
        "u_contact_keep_source": int(np.count_nonzero(uncertain)),
        "object": int(np.count_nonzero(object_ready)),
        "background": int(np.count_nonzero(background_ready)),
        "unsupported": int(np.count_nonzero(unsupported)),
        "contradictory_identity": int(np.count_nonzero(contradictory)),
        "visible_object_conflict": int(np.count_nonzero(protected_conflict)),
    }
    return RevealFrameResult(
        labels=labels,
        target_mask=target,
        unsupported_mask=unsupported,
        route_ids=tuple(route_ids),
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
        policy_ref=policy.ref if policy is not None else None,
        object_texture_ref=object_texture.provenance_ref if object_texture is not None else None,
        background_donor_ref=background_donor.provenance_ref if background_donor is not None else None,
        counts=counts,
        quality_metrics={
            "object_donor_coverage": object_actual_coverage,
            "background_donor_coverage": background_actual_coverage,
            "object_flow_consistency": object_texture.flow_consistency if object_texture else None,
            "background_flow_consistency": background_donor.flow_consistency if background_donor else None,
            "object_photometric_residual": (
                object_texture.photometric_residual_normalized if object_texture else None
            ),
            "background_photometric_residual": (
                background_donor.photometric_residual_normalized if background_donor else None
            ),
        },
        hold_reasons=tuple(dict.fromkeys(reasons)),
        hold=bool(np.any(unsupported)),
    )
