"""CPU-only, fail-closed contract for an Object6D direct-observation canary.

This module validates bytes that already exist.  It does not detect an object,
estimate depth, track pixels, fit a pose, load a model, use a GPU, or authorize a
formal consumer.  In particular, it cannot manufacture a missing H5(b)
decision: every unresolved semantic/input choice is represented by a required,
byte-bound authority record.

The initial scope is deliberately one row: row 0 maps to
``grap_a_cap_059`` frame 371.  A different frame, a second row, or any temporal
source is outside this canary contract.  A structurally honest invalid row is
accepted as an encoding but its terminal status remains ``CARD_FAILED``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Literal
import zipfile

import numpy as np


CONCRETE_PATH_TYPE = type(Path(os.path.sep))
EXACT_NUMPY_INTEGER_TYPES = frozenset(
    np.dtype(name).type
    for name in (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "longlong",
        "ulonglong",
    )
)
EXACT_NUMPY_FLOAT_TYPES = frozenset(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)
EXACT_INTEGER_SCALAR_TYPES = frozenset({int, *EXACT_NUMPY_INTEGER_TYPES})
EXACT_REAL_SCALAR_TYPES = frozenset(
    {int, float, *EXACT_NUMPY_INTEGER_TYPES, *EXACT_NUMPY_FLOAT_TYPES}
)


POSE_KEYS: tuple[str, ...] = (
    "T_object_to_world",
    "T_object_to_camera",
    "T_object_to_world_raw_center_corrected",
)
NPZ_KEYS: tuple[str, ...] = (
    *POSE_KEYS,
    "confidence",
    "valid",
    "visual_observed",
    "timestamp_ns",
    "cylinder_radius_m",
    "cylinder_height_m",
)

MINIMUM_3D_CORRESPONDENCES = 8
MINIMUM_RANSAC_INLIERS = 8
MINIMUM_RANSAC_INLIER_RATIO = 0.45
MAXIMUM_RESIDUAL_M = 0.012
SE3_ABSOLUTE_TOLERANCE = 1e-8
RATIO_ABSOLUTE_TOLERANCE = 1e-12
MAX_CAPTURED_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_NPZ_MEMBER_BYTES = 64 * 1024
MAX_NPZ_HEADER_BYTES = 1024
MAX_EXACT_INT64 = (1 << 63) - 1

CANARY_SESSION_ID = "grap_a_cap_059"
CANARY_FRAME_INDEX = 371
CANARY_ROWS = 1

CANONICAL_FRAME_YAW_ROLE = "CANONICAL_OBJECT_FRAME_AND_YAW"
REDETECTION_POLICY_ROLE = "PER_FRAME_REDETECTION_POLICY"
TEMPORAL_POLICY_ROLE = "TEMPORAL_OBSERVATION_AND_WRITER_POLICY"
PER_FRAME_REDETECTION_POLICY = "PER_FRAME_DIRECT_REDETECTION"
DIRECT_OBSERVATION_TEMPORAL_POLICY = (
    "DIRECT_OBSERVATION_ONLY_NO_NEIGHBOR_NO_INTERPOLATION_"
    "NO_PROPAGATION_NO_FALLBACK_NO_SMOOTHING"
)
AUTHORITY_RECORD_SCHEMA = "object6d-direct-observation-authority-v1"
AUTHORITY_RECORD_KEYS = frozenset(
    {
        "schema",
        "status",
        "role",
        "authority_id",
        "decision",
        "source_artifact",
        "h5b_decision_authorized",
        "label_independent",
        "canary_only",
        "formal_consumer_allowed",
    }
)

# These roles are intentionally explicit.  Supplying a model name while omitting
# its weight bytes, a RANSAC threshold while omitting seed/tie-break semantics,
# or a camera pose while omitting stereo/time conventions is not authority.
REQUIRED_AUTHORITY_ROLES: tuple[str, ...] = (
    CANONICAL_FRAME_YAW_ROLE,
    "OBJECT_DETECTION_PROMPT_CONTRACT",
    "OBJECT_DETECTION_MODEL_AND_WEIGHT",
    REDETECTION_POLICY_ROLE,
    "FOUNDATION_STEREO_SOURCE",
    "FOUNDATION_STEREO_CHECKPOINT",
    "COTRACKER_SOURCE",
    "COTRACKER_CHECKPOINT",
    "STEREO_LEFT_RIGHT_LAYOUT",
    "STEREO_EXTRINSICS_FRAME_AXES_AND_DIRECTION",
    "MP4_PTS_TO_SLAM_TO_LEFT_CAMERA_MAPPING",
    "CAMERA_TO_WORLD_TRAJECTORY",
    "DIRECT_RAW_OBJECT_MASK_AND_CANONICAL_TRACKS",
    "MATCHER_SAMPLING_POLICY",
    "RANSAC_SEED_AND_TIEBREAK_POLICY",
    TEMPORAL_POLICY_ROLE,
    "CYLINDER_GEOMETRY_REFERENCE_AND_REUSE_POLICY",
    "SUCCESSOR_WRITER_AND_NPZ_LINEAGE",
)

CardStatus = Literal["CARD_VALIDATED_CPU_ONLY", "CARD_FAILED"]


class SuccessorContractError(RuntimeError):
    """The successor schema, evidence, authority, or mathematics is invalid."""


def _is_exact_bool(value: object) -> bool:
    return type(value) is bool


def _require_exact_int(
    value: object,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in EXACT_INTEGER_SCALAR_TYPES:
        raise SuccessorContractError(f"{field} must be an exact integer")
    try:
        result = int(value)
    except MemoryError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            f"{field} cannot be represented as an exact integer"
        ) from error
    if minimum is not None and result < minimum:
        raise SuccessorContractError(f"{field} is below its minimum")
    if maximum is not None and result > maximum:
        raise SuccessorContractError(f"{field} exceeds its maximum")
    return result


def _require_finite_real(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if type(value) not in EXACT_REAL_SCALAR_TYPES:
        raise SuccessorContractError(f"{field} must be a finite real scalar")
    value_type = type(value)
    try:
        result = float(value)
    except MemoryError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            f"{field} cannot be represented as a finite real scalar"
        ) from error
    if not math.isfinite(result):
        raise SuccessorContractError(f"{field} must be finite")
    try:
        if value_type in EXACT_INTEGER_SCALAR_TYPES:
            lossless = int(result) == int(value)
        elif value_type is float:
            lossless = True
        else:
            lossless = bool(value_type(result) == value)
    except MemoryError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            f"{field} cannot be checked for lossless binary64 normalization"
        ) from error
    if not lossless:
        raise SuccessorContractError(
            f"{field} must normalize losslessly to a binary64 real"
        )
    if minimum is not None and (
        result < minimum or (not minimum_inclusive and result == minimum)
    ):
        raise SuccessorContractError(f"{field} is below its minimum")
    if maximum is not None and (
        result > maximum or (not maximum_inclusive and result == maximum)
    ):
        raise SuccessorContractError(f"{field} exceeds its maximum")
    return result


def _snapshot_camera_to_world(value: object) -> np.ndarray:
    if value is None:
        raise SuccessorContractError("camera-to-world authority array is absent")
    try:
        snapshot = np.asarray(value).copy()
    except MemoryError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            "camera-to-world authority cannot be snapshotted as an array"
        ) from error
    if (
        type(snapshot) is not np.ndarray
        or snapshot.shape != (CANARY_ROWS, 4, 4)
        or snapshot.dtype != np.dtype(np.float64)
    ):
        raise SuccessorContractError(
            "camera-to-world authority must be an exact float64 ndarray [1,4,4]"
        )
    return snapshot


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_non_placeholder(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
    forbidden_phrases = {
        "tbd",
        "todo",
        "pending",
        "placeholder",
        "unknown",
        "none",
        "null",
        "unfrozen",
        "not decided",
        "not authorized",
        "unauthorized",
    }
    forbidden_tokens = {
        "tbd",
        "todo",
        "pending",
        "placeholder",
        "unknown",
        "unfrozen",
    }
    return normalized not in forbidden_phrases and not (
        set(normalized.split()) & forbidden_tokens
    )


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    sha256: str


@dataclass(frozen=True)
class _BoundFileRequest:
    path: Path
    canonical_path: str
    expected_bytes: int
    expected_sha256: str
    label: str


@dataclass
class _HeldRegularFile:
    request: _BoundFileRequest
    descriptor: int
    leaf_stat: os.stat_result
    ancestor_descriptors: tuple[int, ...]
    ancestor_stats: tuple[os.stat_result, ...]

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)
        for ancestor_descriptor in reversed(self.ancestor_descriptors):
            os.close(ancestor_descriptor)
        self.ancestor_descriptors = ()


@dataclass(frozen=True)
class AuthorityEvidence:
    """A byte-bound, explicitly authorized H5(b) decision or input contract."""

    role: str
    authority_id: str
    authority_path: Path
    authority_bytes: int
    authority_sha256: str
    source_path: Path
    source_bytes: int
    source_sha256: str
    decision: str
    artifact_exists: bool
    h5b_decision_authorized: bool
    label_independent: bool

    def file_requests(
        self, expected_role: str
    ) -> tuple[_BoundFileRequest, _BoundFileRequest]:
        snapshot = _snapshot_authority_evidence(self)
        if type(snapshot.role) is not str or snapshot.role != expected_role:
            raise SuccessorContractError(
                f"authority role mismatch: expected exact string {expected_role}"
            )
        if not _is_non_placeholder(snapshot.authority_id):
            raise SuccessorContractError(f"authority id is absent for {expected_role}")
        if type(snapshot.authority_path) is not CONCRETE_PATH_TYPE:
            raise SuccessorContractError(
                f"authority record {expected_role} path must have exact concrete Path type"
            )
        authority_bytes = _require_exact_int(
            snapshot.authority_bytes,
            field=f"authority record bytes for {expected_role}",
            minimum=1,
            maximum=MAX_CAPTURED_CONTRACT_BYTES,
        )
        if not _is_sha256(snapshot.authority_sha256):
            raise SuccessorContractError(
                f"authority record SHA-256 is invalid for {expected_role}"
            )
        if type(snapshot.source_path) is not CONCRETE_PATH_TYPE:
            raise SuccessorContractError(
                f"authority source {expected_role} path must have exact concrete Path type"
            )
        source_bytes = _require_exact_int(
            snapshot.source_bytes,
            field=f"authority source bytes for {expected_role}",
            minimum=1,
            maximum=MAX_EXACT_INT64,
        )
        if not _is_sha256(snapshot.source_sha256):
            raise SuccessorContractError(
                f"authority SHA-256 is invalid for {expected_role}"
            )
        if not _is_non_placeholder(snapshot.decision):
            raise SuccessorContractError(
                f"authority decision is unresolved for {expected_role}"
            )
        if not _is_exact_bool(snapshot.artifact_exists) or not snapshot.artifact_exists:
            raise SuccessorContractError(
                f"authority artifact does not exist for {expected_role}"
            )
        if (
            not _is_exact_bool(snapshot.h5b_decision_authorized)
            or not snapshot.h5b_decision_authorized
        ):
            raise SuccessorContractError(
                f"H5(b) decision is not authorized for {expected_role}"
            )
        if (
            not _is_exact_bool(snapshot.label_independent)
            or not snapshot.label_independent
        ):
            raise SuccessorContractError(
                f"authority is not label-independent for {expected_role}"
            )
        return (
            _bound_file_request(
                snapshot.authority_path,
                expected_bytes=authority_bytes,
                expected_sha256=snapshot.authority_sha256,
                label=f"authority record {expected_role}",
            ),
            _bound_file_request(
                snapshot.source_path,
                expected_bytes=source_bytes,
                expected_sha256=snapshot.source_sha256,
                label=f"authority source artifact {expected_role}",
            ),
        )

    def require_valid(
        self, expected_role: str, held_files: "_HeldFileSet"
    ) -> tuple[_FileIdentity, _FileIdentity]:
        snapshot = _snapshot_authority_evidence(self)
        authority_request, source_request = snapshot.file_requests(expected_role)
        payload, _, identity = _read_bound_regular_file(
            authority_request,
            held_files=held_files,
            capture=True,
        )
        if payload is None:
            raise SuccessorContractError(
                f"authority record bytes were not captured for {expected_role}"
            )
        record = _load_authority_record(payload, expected_role=expected_role)
        expected_fields = {
            "role": snapshot.role,
            "authority_id": snapshot.authority_id,
            "decision": snapshot.decision,
            "h5b_decision_authorized": snapshot.h5b_decision_authorized,
            "label_independent": snapshot.label_independent,
        }
        for key, expected in expected_fields.items():
            if type(record[key]) is not type(expected) or record[key] != expected:
                raise SuccessorContractError(
                    f"authority record field mismatch for {expected_role}: {key}"
                )
        source_record = record["source_artifact"]
        expected_source_record = {
            "path": source_request.canonical_path,
            "bytes": source_request.expected_bytes,
            "sha256": snapshot.source_sha256,
        }
        if (
            type(source_record) is not dict
            or set(source_record) != set(expected_source_record)
            or any(
                type(source_record[key]) is not type(expected)
                or source_record[key] != expected
                for key, expected in expected_source_record.items()
            )
        ):
            raise SuccessorContractError(
                f"authority source artifact reference mismatch for {expected_role}"
            )
        _, _, source_identity = _read_bound_regular_file(
            source_request,
            held_files=held_files,
            capture=False,
        )
        _require_distinct_files(
            (identity, source_identity),
            label=f"authority record/source {expected_role}",
        )
        return identity, source_identity


@dataclass(frozen=True)
class EvidenceDigest:
    """Digest identity for direct evidence already verified by its producer."""

    kind: str
    source_path: Path
    source_bytes: int
    source_sha256: str

    def file_request(
        self, expected_kind: str, *, capture: bool = False
    ) -> _BoundFileRequest:
        snapshot = _snapshot_evidence_digest(self)
        if type(snapshot.kind) is not str or snapshot.kind != expected_kind:
            raise SuccessorContractError(
                f"direct evidence kind mismatch: expected exact string {expected_kind}"
            )
        if type(snapshot.source_path) is not CONCRETE_PATH_TYPE:
            raise SuccessorContractError(
                f"direct evidence {expected_kind} path must have exact concrete Path type"
            )
        source_bytes = _require_exact_int(
            snapshot.source_bytes,
            field=f"direct evidence bytes for {expected_kind}",
            minimum=1,
            maximum=(MAX_CAPTURED_CONTRACT_BYTES if capture else MAX_EXACT_INT64),
        )
        if not _is_sha256(snapshot.source_sha256):
            raise SuccessorContractError(
                f"direct evidence SHA-256 invalid: {expected_kind}"
            )
        return _bound_file_request(
            snapshot.source_path,
            expected_bytes=source_bytes,
            expected_sha256=snapshot.source_sha256,
            label=f"direct evidence {expected_kind}",
        )

    def require_valid(
        self,
        expected_kind: str,
        *,
        held_files: "_HeldFileSet",
        capture: bool = False,
    ) -> tuple[bytes | None, _FileIdentity]:
        request = self.file_request(expected_kind, capture=capture)
        payload, _, identity = _read_bound_regular_file(
            request,
            held_files=held_files,
            capture=capture,
        )
        return payload, identity


@dataclass(frozen=True)
class CanaryRowMapping:
    row_index: int
    session_id: str
    frame_index: int
    timestamp_ns: int
    cylinder_radius_m: float
    cylinder_height_m: float


@dataclass(frozen=True)
class OperationCounters:
    """Auditable counts for every forbidden temporal/fallback operation."""

    neighbor_pose_reads: int
    interpolation_calls: int
    propagation_calls: int
    fallback_calls: int
    temporal_smoothing_calls: int

    def require_all_zero(self) -> None:
        snapshot = _snapshot_operation_counters(self)
        for name in _OPERATION_COUNTER_FIELD_NAMES:
            value = getattr(snapshot, name)
            try:
                parsed = _require_exact_int(value, field=name, minimum=0, maximum=0)
            except SuccessorContractError as error:
                raise SuccessorContractError(
                    f"forbidden operation counter must be exact zero: {name}"
                ) from error
            if parsed != 0:
                raise SuccessorContractError(
                    f"forbidden operation counter must be exact zero: {name}"
                )


@dataclass(frozen=True)
class DirectObservationEvidence:
    """Per-row evidence and fixed-gate inputs; no pose is sourced from this audit."""

    row_index: int
    session_id: str
    frame_index: int
    timestamp_ns: int
    direct_observation_attempted: bool
    source_frame_indices: tuple[int, ...]
    positive_depth_correspondences: int
    ransac_inliers: int
    ransac_inlier_ratio: float
    residual_m: float
    raw_mask: EvidenceDigest | None
    tracks: EvidenceDigest | None
    positive_depth: EvidenceDigest | None
    camera_to_world: EvidenceDigest | None
    operation_counters: OperationCounters | None = None
    used_neighbor_pose: bool = False
    used_interpolation: bool = False
    used_propagation: bool = False
    used_fallback: bool = False
    used_temporal_smoothing: bool = False


_AUTHORITY_EVIDENCE_FIELD_NAMES: tuple[str, ...] = (
    "role",
    "authority_id",
    "authority_path",
    "authority_bytes",
    "authority_sha256",
    "source_path",
    "source_bytes",
    "source_sha256",
    "decision",
    "artifact_exists",
    "h5b_decision_authorized",
    "label_independent",
)
_EVIDENCE_DIGEST_FIELD_NAMES: tuple[str, ...] = (
    "kind",
    "source_path",
    "source_bytes",
    "source_sha256",
)
_CANARY_ROW_MAPPING_FIELD_NAMES: tuple[str, ...] = (
    "row_index",
    "session_id",
    "frame_index",
    "timestamp_ns",
    "cylinder_radius_m",
    "cylinder_height_m",
)
_OPERATION_COUNTER_FIELD_NAMES: tuple[str, ...] = (
    "neighbor_pose_reads",
    "interpolation_calls",
    "propagation_calls",
    "fallback_calls",
    "temporal_smoothing_calls",
)
_DIRECT_OBSERVATION_EVIDENCE_FIELD_NAMES: tuple[str, ...] = (
    "row_index",
    "session_id",
    "frame_index",
    "timestamp_ns",
    "direct_observation_attempted",
    "source_frame_indices",
    "positive_depth_correspondences",
    "ransac_inliers",
    "ransac_inlier_ratio",
    "residual_m",
    "raw_mask",
    "tracks",
    "positive_depth",
    "camera_to_world",
    "operation_counters",
    "used_neighbor_pose",
    "used_interpolation",
    "used_propagation",
    "used_fallback",
    "used_temporal_smoothing",
)


def _snapshot_exact_dataclass_fields(
    value: object,
    *,
    expected_type: type,
    expected_fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not expected_type:
        raise SuccessorContractError(f"{label} has wrong concrete dataclass type")
    try:
        state = value.__dict__
        if type(state) is not dict:
            raise SuccessorContractError(f"{label} state must be an exact dict")
        items = tuple(state.items())
    except MemoryError:
        raise
    except SuccessorContractError:
        raise
    except Exception as error:
        raise SuccessorContractError(f"{label} state cannot be snapshotted") from error
    expected = frozenset(expected_fields)
    snapshot: dict[str, object] = {}
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise SuccessorContractError(f"{label} state has malformed fields")
        key, field_value = item
        if type(key) is not str or key not in expected or key in snapshot:
            raise SuccessorContractError(f"{label} state has unexpected fields")
        snapshot[key] = field_value
    if len(snapshot) != len(expected) or any(
        field not in snapshot for field in expected_fields
    ):
        raise SuccessorContractError(f"{label} state must contain exact fields")
    return snapshot


def _snapshot_authority_evidence(value: object) -> AuthorityEvidence:
    fields = _snapshot_exact_dataclass_fields(
        value,
        expected_type=AuthorityEvidence,
        expected_fields=_AUTHORITY_EVIDENCE_FIELD_NAMES,
        label="authority evidence",
    )
    return AuthorityEvidence(**fields)


def _snapshot_evidence_digest(value: object) -> EvidenceDigest:
    fields = _snapshot_exact_dataclass_fields(
        value,
        expected_type=EvidenceDigest,
        expected_fields=_EVIDENCE_DIGEST_FIELD_NAMES,
        label="direct evidence digest",
    )
    return EvidenceDigest(**fields)


def _snapshot_canary_row_mapping(value: object) -> CanaryRowMapping:
    fields = _snapshot_exact_dataclass_fields(
        value,
        expected_type=CanaryRowMapping,
        expected_fields=_CANARY_ROW_MAPPING_FIELD_NAMES,
        label="canary row mapping",
    )
    return CanaryRowMapping(**fields)


def _snapshot_operation_counters(value: object) -> OperationCounters:
    fields = _snapshot_exact_dataclass_fields(
        value,
        expected_type=OperationCounters,
        expected_fields=_OPERATION_COUNTER_FIELD_NAMES,
        label="operation counters",
    )
    return OperationCounters(**fields)


def _snapshot_direct_observation_evidence(
    value: object,
) -> DirectObservationEvidence:
    fields = _snapshot_exact_dataclass_fields(
        value,
        expected_type=DirectObservationEvidence,
        expected_fields=_DIRECT_OBSERVATION_EVIDENCE_FIELD_NAMES,
        label="direct-observation evidence",
    )
    for name in ("raw_mask", "tracks", "positive_depth", "camera_to_world"):
        record = fields[name]
        if record is not None:
            fields[name] = _snapshot_evidence_digest(record)
    counters = fields["operation_counters"]
    if counters is not None:
        fields["operation_counters"] = _snapshot_operation_counters(counters)
    return DirectObservationEvidence(**fields)


@dataclass(frozen=True)
class ValidationCard:
    """Non-authorizing result of CPU contract validation."""

    status: CardStatus
    failure_reasons: tuple[str, ...]
    authorities_valid: bool
    schema_valid: bool
    mathematics_valid: bool
    direct_observation_only: bool
    rows_total: int
    rows_valid: int
    failed_row_indices: tuple[int, ...]
    session_id: str
    frame_index: int
    npz_bytes: int | None
    npz_sha256: str | None
    minimum_3d_correspondences: int = MINIMUM_3D_CORRESPONDENCES
    minimum_ransac_inliers: int = MINIMUM_RANSAC_INLIERS
    minimum_ransac_inlier_ratio: float = MINIMUM_RANSAC_INLIER_RATIO
    maximum_residual_m: float = MAXIMUM_RESIDUAL_M
    canary_only: bool = True
    gpu_executed: bool = False
    model_loaded: bool = False
    formal_consumer_allowed: bool = False


def _failed_card(
    reason: str,
    *,
    authorities_valid: bool = False,
    schema_valid: bool = False,
    mathematics_valid: bool = False,
    direct_observation_only: bool = False,
    rows_valid: int = 0,
    failed_row_indices: tuple[int, ...] = (),
    npz_bytes: int | None = None,
    npz_sha256: str | None = None,
) -> ValidationCard:
    return ValidationCard(
        status="CARD_FAILED",
        failure_reasons=(reason,),
        authorities_valid=authorities_valid,
        schema_valid=schema_valid,
        mathematics_valid=mathematics_valid,
        direct_observation_only=direct_observation_only,
        rows_total=CANARY_ROWS,
        rows_valid=rows_valid,
        failed_row_indices=failed_row_indices,
        session_id=CANARY_SESSION_ID,
        frame_index=CANARY_FRAME_INDEX,
        npz_bytes=npz_bytes,
        npz_sha256=npz_sha256,
    )


def _prepare_authorities(
    authorities: Mapping[str, AuthorityEvidence] | None,
) -> dict[str, AuthorityEvidence]:
    if authorities is None:
        raise SuccessorContractError(
            "missing required authorities: " + ", ".join(REQUIRED_AUTHORITY_ROLES)
        )
    try:
        if not isinstance(authorities, Mapping):
            raise SuccessorContractError("authorities must be a mapping")
        items = tuple(authorities.items())
    except MemoryError:
        raise
    except SuccessorContractError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            "authority mapping cannot be snapshotted"
        ) from error
    snapshot: dict[str, AuthorityEvidence] = {}
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise SuccessorContractError(
                "authority mapping items must be exact key/value pairs"
            )
        key, value = item
        if type(key) is not str:
            raise SuccessorContractError("authority mapping keys must be exact strings")
        if type(value) is not AuthorityEvidence:
            raise SuccessorContractError(
                f"authority record has wrong type for mapping key: {key}"
            )
        if key in snapshot:
            raise SuccessorContractError(f"authority mapping has duplicate key: {key}")
        snapshot[key] = _snapshot_authority_evidence(value)
    supplied = set(snapshot)
    required = set(REQUIRED_AUTHORITY_ROLES)
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing:
        raise SuccessorContractError(
            "missing required authorities: " + ", ".join(missing)
        )
    if extra:
        raise SuccessorContractError("unexpected authority roles: " + ", ".join(extra))
    for role in REQUIRED_AUTHORITY_ROLES:
        record = snapshot[role]
        record.file_requests(role)
    if snapshot[REDETECTION_POLICY_ROLE].decision != PER_FRAME_REDETECTION_POLICY:
        raise SuccessorContractError(
            "per-frame re-detection authority does not require direct re-detection"
        )
    if snapshot[TEMPORAL_POLICY_ROLE].decision != DIRECT_OBSERVATION_TEMPORAL_POLICY:
        raise SuccessorContractError(
            "temporal authority permits a neighbor, interpolation, propagation, "
            "fallback, or smoothing path"
        )
    return snapshot


def _authority_file_requests(
    authorities: Mapping[str, AuthorityEvidence],
) -> tuple[_BoundFileRequest, ...]:
    requests: list[_BoundFileRequest] = []
    for role in REQUIRED_AUTHORITY_ROLES:
        requests.extend(authorities[role].file_requests(role))
    return tuple(requests)


def _require_authorities(
    authorities: Mapping[str, AuthorityEvidence], held_files: "_HeldFileSet"
) -> tuple[_FileIdentity, ...]:
    identities: list[_FileIdentity] = []
    for role in REQUIRED_AUTHORITY_ROLES:
        identities.extend(authorities[role].require_valid(role, held_files))
    _require_distinct_files(identities, label="authority records")
    return tuple(identities)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SuccessorContractError(f"authority record has duplicate key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise SuccessorContractError(
        f"authority record contains forbidden non-finite JSON constant: {value}"
    )


def _load_authority_record(payload: bytes, *, expected_role: str) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        record = json.loads(
            decoded,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except SuccessorContractError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise SuccessorContractError(
            f"authority record is not strict UTF-8 JSON for {expected_role}: {error}"
        ) from error
    if not isinstance(record, dict) or set(record) != AUTHORITY_RECORD_KEYS:
        raise SuccessorContractError(
            f"authority record schema mismatch for {expected_role}"
        )
    fixed_fields: tuple[tuple[str, object], ...] = (
        ("schema", AUTHORITY_RECORD_SCHEMA),
        ("status", "ARTIFACT_EXISTS"),
        ("role", expected_role),
        ("h5b_decision_authorized", True),
        ("label_independent", True),
        ("canary_only", True),
        ("formal_consumer_allowed", False),
    )
    for key, expected in fixed_fields:
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise SuccessorContractError(
                f"authority record has invalid {key} for {expected_role}"
            )
    if not _is_non_placeholder(record["authority_id"]):
        raise SuccessorContractError(
            f"authority record id is absent for {expected_role}"
        )
    if not _is_non_placeholder(record["decision"]):
        raise SuccessorContractError(
            f"authority record decision is unresolved for {expected_role}"
        )
    return record


def _canonical_absolute_path(path: Path, *, label: str) -> tuple[str, tuple[str, ...]]:
    if type(path) is not CONCRETE_PATH_TYPE:
        raise SuccessorContractError(f"{label} path must have exact concrete Path type")
    raw_path = os.fspath(path)
    if type(raw_path) is not str or not raw_path or "\x00" in raw_path:
        raise SuccessorContractError(f"{label} path is absent or invalid")
    if not raw_path.startswith(os.path.sep):
        raise SuccessorContractError(f"{label} path must be absolute")
    components = tuple(raw_path.split(os.path.sep)[1:])
    if (
        not components
        or any(value in ("", ".", "..") for value in components)
        or raw_path != os.path.sep + os.path.sep.join(components)
    ):
        raise SuccessorContractError(
            f"{label} path must be one canonical absolute leaf path"
        )
    return raw_path, components


def _bound_file_request(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    label: str,
) -> _BoundFileRequest:
    parsed_bytes = _require_exact_int(
        expected_bytes,
        field=f"expected {label} byte count",
        minimum=1,
        maximum=MAX_EXACT_INT64,
    )
    if not _is_sha256(expected_sha256):
        raise SuccessorContractError(f"expected {label} SHA-256 is absent or invalid")
    canonical_path, _ = _canonical_absolute_path(path, label=label)
    return _BoundFileRequest(
        path=Path(canonical_path),
        canonical_path=canonical_path,
        expected_bytes=parsed_bytes,
        expected_sha256=expected_sha256,
        label=label,
    )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_held_regular_file(request: _BoundFileRequest) -> _HeldRegularFile:
    _, components = _canonical_absolute_path(request.path, label=request.label)
    directory_flags = _directory_open_flags()
    ancestor_descriptors: list[int] = []
    ancestor_stats: list[os.stat_result] = []
    descriptor = -1
    try:
        root_descriptor = os.open(os.path.sep, directory_flags)
        ancestor_descriptors.append(root_descriptor)
        root_stat = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise SuccessorContractError(f"{request.label} root is not a directory")
        ancestor_stats.append(root_stat)
        for component in components[:-1]:
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=ancestor_descriptors[-1],
                )
            except OSError as error:
                raise SuccessorContractError(
                    f"{request.label} ancestor no-follow open failed: {error}"
                ) from error
            ancestor_descriptors.append(child_descriptor)
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise SuccessorContractError(
                    f"{request.label} ancestor is not a directory"
                )
            ancestor_stats.append(child_stat)
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(
                components[-1], file_flags, dir_fd=ancestor_descriptors[-1]
            )
        except OSError as error:
            raise SuccessorContractError(
                f"{request.label} leaf no-follow open failed: {error}"
            ) from error
        leaf_stat = os.fstat(descriptor)
        if not stat.S_ISREG(leaf_stat.st_mode):
            raise SuccessorContractError(
                f"{request.label} target is not a regular file"
            )
        if leaf_stat.st_nlink != 1:
            raise SuccessorContractError(f"{request.label} target must have nlink=1")
        if leaf_stat.st_size != request.expected_bytes:
            raise SuccessorContractError(
                f"{request.label} byte count does not match expected evidence"
            )
        return _HeldRegularFile(
            request=request,
            descriptor=descriptor,
            leaf_stat=leaf_stat,
            ancestor_descriptors=tuple(ancestor_descriptors),
            ancestor_stats=tuple(ancestor_stats),
        )
    except SuccessorContractError:
        if descriptor >= 0:
            os.close(descriptor)
        for ancestor_descriptor in reversed(ancestor_descriptors):
            os.close(ancestor_descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        for ancestor_descriptor in reversed(ancestor_descriptors):
            os.close(ancestor_descriptor)
        raise SuccessorContractError(
            f"{request.label} no-follow held open failed: {error}"
        ) from error


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_held_snapshot(held: _HeldRegularFile) -> None:
    try:
        current_leaf = os.fstat(held.descriptor)
        if _stat_identity(current_leaf) != _stat_identity(held.leaf_stat):
            raise SuccessorContractError(
                f"{held.request.label} leaf identity changed during held-FD validation"
            )
        if len(held.ancestor_descriptors) != len(held.ancestor_stats):
            raise SuccessorContractError(
                f"{held.request.label} held ancestor chain is inconsistent"
            )
        for descriptor, expected in zip(
            held.ancestor_descriptors, held.ancestor_stats, strict=True
        ):
            if _stat_identity(os.fstat(descriptor)) != _stat_identity(expected):
                raise SuccessorContractError(
                    f"{held.request.label} ancestor identity changed during held-FD validation"
                )
    except SuccessorContractError:
        raise
    except OSError as error:
        raise SuccessorContractError(
            f"{held.request.label} held-FD identity verification failed: {error}"
        ) from error


class _HeldFileSet:
    def __init__(self, requests: Sequence[_BoundFileRequest]) -> None:
        self._held: dict[str, _HeldRegularFile] = {}
        self._requests = tuple(requests)
        canonical_paths = [request.canonical_path for request in self._requests]
        if len(canonical_paths) != len(set(canonical_paths)):
            raise SuccessorContractError(
                "all NPZ, authority, source, and direct-evidence paths must be "
                "globally unique canonical absolute paths for distinct files"
            )

    def open_all(self) -> None:
        try:
            for request in self._requests:
                self._held[request.canonical_path] = _open_held_regular_file(request)
            inode_keys = {
                (held.leaf_stat.st_dev, held.leaf_stat.st_ino)
                for held in self._held.values()
            }
            if len(inode_keys) != len(self._held):
                raise SuccessorContractError(
                    "all held paths must identify globally distinct physical files"
                )
        except (SuccessorContractError, OSError):
            self.close()
            raise

    def get(self, request: _BoundFileRequest) -> _HeldRegularFile:
        held = self._held.get(request.canonical_path)
        if held is None or held.request != request:
            raise SuccessorContractError(
                f"{request.label} is absent from the immutable held-file set"
            )
        return held

    def verify_all(self) -> None:
        for held in self._held.values():
            _require_held_snapshot(held)
        for held in self._held.values():
            reopened = _open_held_regular_file(held.request)
            try:
                if _stat_identity(reopened.leaf_stat) != _stat_identity(held.leaf_stat):
                    raise SuccessorContractError(
                        f"{held.request.label} logical path identity changed"
                    )
                if len(reopened.ancestor_stats) != len(held.ancestor_stats) or any(
                    _stat_identity(current) != _stat_identity(expected)
                    for current, expected in zip(
                        reopened.ancestor_stats, held.ancestor_stats, strict=True
                    )
                ):
                    raise SuccessorContractError(
                        f"{held.request.label} logical ancestor identity changed"
                    )
            finally:
                reopened.close()
        for held in self._held.values():
            _require_held_snapshot(held)

    def close(self) -> None:
        for held in reversed(tuple(self._held.values())):
            held.close()
        self._held.clear()


def _read_bound_regular_file(
    request: _BoundFileRequest,
    *,
    held_files: _HeldFileSet,
    capture: bool,
) -> tuple[bytes | None, str, _FileIdentity]:
    held = held_files.get(request)
    descriptor = held.descriptor
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(held.leaf_stat):
            raise SuccessorContractError(
                f"{request.label} identity changed before same-FD read"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        hasher = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture else None
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            bytes_read += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            after
        ) != _stat_identity(held.leaf_stat):
            raise SuccessorContractError(
                f"{request.label} identity changed during same-FD read"
            )
    except SuccessorContractError:
        raise
    except OSError as error:
        raise SuccessorContractError(
            f"{request.label} same-FD read failed: {error}"
        ) from error
    if bytes_read != request.expected_bytes:
        raise SuccessorContractError(
            f"{request.label} byte count does not match expected evidence"
        )
    digest = hasher.hexdigest()
    if digest != request.expected_sha256:
        raise SuccessorContractError(
            f"{request.label} SHA-256 does not match expected evidence"
        )
    payload = b"".join(chunks) if chunks is not None else None
    return (
        payload,
        digest,
        _FileIdentity(before.st_dev, before.st_ino, before.st_size, digest),
    )


def _require_distinct_files(identities: Sequence[_FileIdentity], *, label: str) -> None:
    inode_keys = {(value.device, value.inode) for value in identities}
    digests = {value.sha256 for value in identities}
    if len(inode_keys) != len(identities) or len(digests) != len(identities):
        raise SuccessorContractError(
            f"{label} must bind distinct files with distinct bytes"
        )


def _read_npz_bytes(
    request: _BoundFileRequest,
    *,
    held_files: _HeldFileSet,
) -> tuple[bytes, str, _FileIdentity]:
    if request.expected_bytes > MAX_CAPTURED_CONTRACT_BYTES:
        raise SuccessorContractError("successor NPZ is unreasonably large for N=1")
    payload, digest, identity = _read_bound_regular_file(
        request,
        held_files=held_files,
        capture=True,
    )
    if payload is None:
        raise SuccessorContractError("successor NPZ bytes were not captured")
    return payload, digest, identity


def _expected_npz_array_schema(key: str) -> tuple[tuple[int, ...], np.dtype]:
    if key in POSE_KEYS:
        return (CANARY_ROWS, 4, 4), np.dtype(np.float64)
    if key == "confidence":
        return (CANARY_ROWS,), np.dtype(np.float64)
    if key in ("valid", "visual_observed"):
        return (CANARY_ROWS,), np.dtype(np.bool_)
    if key == "timestamp_ns":
        return (CANARY_ROWS,), np.dtype(np.int64)
    if key in ("cylinder_radius_m", "cylinder_height_m"):
        return (), np.dtype(np.float64)
    raise SuccessorContractError(f"unexpected successor NPZ member key: {key}")


def _decode_prevalidated_npy_member(payload: bytes, *, key: str) -> np.ndarray:
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise SuccessorContractError(f"successor NPZ member is not NPY: {key}")
    version = (payload[6], payload[7])
    if version == (1, 0):
        length_bytes = 2
        header_reader = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        length_bytes = 4
        header_reader = np.lib.format.read_array_header_2_0
    else:
        raise SuccessorContractError(
            f"successor NPZ member has unsupported NPY version: {key}"
        )
    prefix_bytes = 8 + length_bytes
    if len(payload) < prefix_bytes:
        raise SuccessorContractError(
            f"successor NPZ member has truncated header: {key}"
        )
    header_bytes = int.from_bytes(payload[8:prefix_bytes], "little")
    if header_bytes > MAX_NPZ_HEADER_BYTES:
        raise SuccessorContractError(
            f"successor NPZ member header is unreasonably large: {key}"
        )
    header_end = prefix_bytes + header_bytes
    if header_end > len(payload):
        raise SuccessorContractError(
            f"successor NPZ member has truncated header: {key}"
        )
    stream = io.BytesIO(payload)
    parsed_version = np.lib.format.read_magic(stream)
    if parsed_version != version:
        raise SuccessorContractError(
            f"successor NPZ member NPY version is inconsistent: {key}"
        )
    shape, fortran_order, dtype = header_reader(
        stream, max_header_size=MAX_NPZ_HEADER_BYTES
    )
    expected_shape, expected_dtype = _expected_npz_array_schema(key)
    if shape != expected_shape:
        raise SuccessorContractError(
            f"successor NPZ member shape does not match exact schema: {key}"
        )
    if fortran_order is not False:
        raise SuccessorContractError(f"successor NPZ member must use C order: {key}")
    if dtype != expected_dtype or dtype.hasobject:
        raise SuccessorContractError(
            f"successor NPZ member dtype does not match exact schema: {key}"
        )
    expected_count = math.prod(expected_shape)
    expected_data_bytes = expected_count * expected_dtype.itemsize
    if stream.tell() != header_end or header_end + expected_data_bytes != len(payload):
        raise SuccessorContractError(
            f"successor NPZ member byte length does not match exact schema: {key}"
        )
    return (
        np.frombuffer(
            payload,
            dtype=dtype,
            count=expected_count,
            offset=header_end,
        )
        .reshape(expected_shape)
        .copy()
    )


def _load_exact_nine_arrays(payload: bytes) -> dict[str, np.ndarray]:
    expected_members = {f"{key}.npy" for key in NPZ_KEYS}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member_info = archive.infolist()
            members = [value.filename for value in member_info]
            if len(members) != len(set(members)):
                raise SuccessorContractError("NPZ has duplicate members")
            if len(members) != len(NPZ_KEYS) or set(members) != expected_members:
                raise SuccessorContractError(
                    "successor NPZ must contain exactly nine keys"
                )
            if any(value.file_size > MAX_NPZ_MEMBER_BYTES for value in member_info):
                raise SuccessorContractError(
                    "successor NPZ member is unreasonably large for N=1"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise SuccessorContractError(f"NPZ CRC failure: {bad_member}")
            member_payloads = {
                value.filename: archive.read(value) for value in member_info
            }
        arrays = {
            key: _decode_prevalidated_npy_member(member_payloads[f"{key}.npy"], key=key)
            for key in NPZ_KEYS
        }
    except SuccessorContractError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise SuccessorContractError(
            f"successor NPZ cannot be fully decoded: {error}"
        ) from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise SuccessorContractError("successor NPZ object arrays are forbidden")
    return arrays


def _require_array_schema(arrays: Mapping[str, np.ndarray]) -> None:
    for key in POSE_KEYS:
        value = arrays[key]
        if value.shape != (CANARY_ROWS, 4, 4) or value.dtype != np.dtype(np.float64):
            raise SuccessorContractError(f"{key} must be float64 [1,4,4]")
    confidence = arrays["confidence"]
    if confidence.shape != (CANARY_ROWS,) or confidence.dtype != np.dtype(np.float64):
        raise SuccessorContractError("confidence must be float64 [1]")
    for key in ("valid", "visual_observed"):
        value = arrays[key]
        if value.shape != (CANARY_ROWS,) or value.dtype != np.dtype(np.bool_):
            raise SuccessorContractError(f"{key} must be bool [1]")
    timestamp = arrays["timestamp_ns"]
    if timestamp.shape != (CANARY_ROWS,) or timestamp.dtype != np.dtype(np.int64):
        raise SuccessorContractError("timestamp_ns must be int64 [1]")
    if int(timestamp[0]) <= 0:
        raise SuccessorContractError("timestamp_ns must be positive")
    for key in ("cylinder_radius_m", "cylinder_height_m"):
        value = arrays[key]
        if value.shape != () or value.dtype != np.dtype(np.float64):
            raise SuccessorContractError(f"{key} must be a float64 scalar")
        _require_finite_real(value[()], field=key, minimum=0.0, minimum_inclusive=False)


def _prepare_canary_inputs(
    row_mapping: Sequence[CanaryRowMapping] | None,
    evidence_rows: Sequence[DirectObservationEvidence] | None,
) -> tuple[tuple[CanaryRowMapping, ...], tuple[DirectObservationEvidence, ...]]:
    if row_mapping is None:
        raise SuccessorContractError("canary requires exactly one row mapping")
    if evidence_rows is None:
        raise SuccessorContractError(
            "canary requires exactly one direct-observation audit"
        )
    invalid_sequence_types = (str, bytes, bytearray)
    try:
        if not isinstance(row_mapping, Sequence) or isinstance(
            row_mapping, invalid_sequence_types
        ):
            raise SuccessorContractError("canary row mapping must be a finite sequence")
        if not isinstance(evidence_rows, Sequence) or isinstance(
            evidence_rows, invalid_sequence_types
        ):
            raise SuccessorContractError(
                "direct-observation audit must be a finite sequence"
            )
        mapping_snapshot = tuple(row_mapping)
        evidence_snapshot = tuple(evidence_rows)
    except MemoryError:
        raise
    except SuccessorContractError:
        raise
    except Exception as error:
        raise SuccessorContractError(
            "canary row inputs are not finite sequences"
        ) from error
    if len(mapping_snapshot) != CANARY_ROWS:
        raise SuccessorContractError("canary requires exactly one row mapping")
    if len(evidence_snapshot) != CANARY_ROWS:
        raise SuccessorContractError(
            "canary requires exactly one direct-observation audit"
        )
    if type(mapping_snapshot[0]) is not CanaryRowMapping:
        raise SuccessorContractError("canary row mapping has wrong type")
    if type(evidence_snapshot[0]) is not DirectObservationEvidence:
        raise SuccessorContractError("direct-observation audit has wrong type")
    mapping_snapshot = (_snapshot_canary_row_mapping(mapping_snapshot[0]),)
    evidence_snapshot = (_snapshot_direct_observation_evidence(evidence_snapshot[0]),)
    for record, kind in (
        (evidence_snapshot[0].raw_mask, "DIRECT_RAW_OBJECT_MASK"),
        (evidence_snapshot[0].tracks, "DIRECT_CANONICAL_TRACKS"),
        (evidence_snapshot[0].positive_depth, "POSITIVE_STEREO_DEPTH"),
        (evidence_snapshot[0].camera_to_world, "CAMERA_TO_WORLD"),
    ):
        if record is not None:
            if type(record) is not EvidenceDigest:
                raise SuccessorContractError(f"direct evidence has wrong type: {kind}")
            record.file_request(kind, capture=kind == "CAMERA_TO_WORLD")
    return mapping_snapshot, evidence_snapshot


def _direct_evidence_file_requests(
    evidence: DirectObservationEvidence,
) -> tuple[_BoundFileRequest, ...]:
    requests: list[_BoundFileRequest] = []
    for record, kind in (
        (evidence.raw_mask, "DIRECT_RAW_OBJECT_MASK"),
        (evidence.tracks, "DIRECT_CANONICAL_TRACKS"),
        (evidence.positive_depth, "POSITIVE_STEREO_DEPTH"),
        (evidence.camera_to_world, "CAMERA_TO_WORLD"),
    ):
        if record is not None:
            requests.append(
                record.file_request(kind, capture=kind == "CAMERA_TO_WORLD")
            )
    return tuple(requests)


def _require_canary_mapping(
    arrays: Mapping[str, np.ndarray],
    row_mapping: Sequence[CanaryRowMapping] | None,
    evidence_rows: Sequence[DirectObservationEvidence] | None,
) -> tuple[CanaryRowMapping, DirectObservationEvidence]:
    if row_mapping is None or len(row_mapping) != CANARY_ROWS:
        raise SuccessorContractError("canary requires exactly one row mapping")
    if evidence_rows is None or len(evidence_rows) != CANARY_ROWS:
        raise SuccessorContractError(
            "canary requires exactly one direct-observation audit"
        )
    mapping = row_mapping[0]
    evidence = evidence_rows[0]
    if type(mapping) is not CanaryRowMapping:
        raise SuccessorContractError("canary row mapping has wrong type")
    if type(evidence) is not DirectObservationEvidence:
        raise SuccessorContractError("direct-observation audit has wrong type")
    mapping_row_index = _require_exact_int(
        mapping.row_index, field="canary mapping row_index"
    )
    mapping_frame_index = _require_exact_int(
        mapping.frame_index,
        field="canary mapping frame_index",
    )
    if (
        mapping_row_index != 0
        or type(mapping.session_id) is not str
        or mapping.session_id != CANARY_SESSION_ID
        or mapping_frame_index != CANARY_FRAME_INDEX
    ):
        raise SuccessorContractError("N=1 canary row 0 must map to 059 frame 371")
    mapping_timestamp = _require_exact_int(
        mapping.timestamp_ns, field="canary mapping timestamp_ns", minimum=1
    )
    if int(arrays["timestamp_ns"][0]) != mapping_timestamp:
        raise SuccessorContractError("NPZ/mapping timestamp mismatch")
    for field, expected in (
        ("cylinder_radius_m", mapping.cylinder_radius_m),
        ("cylinder_height_m", mapping.cylinder_height_m),
    ):
        expected_float = _require_finite_real(
            expected,
            field=f"canary mapping {field}",
            minimum=0.0,
            minimum_inclusive=False,
        )
        if float(arrays[field]) != expected_float:
            raise SuccessorContractError(f"NPZ/mapping {field} mismatch")
    evidence_row_index = _require_exact_int(
        evidence.row_index, field="direct evidence row_index"
    )
    evidence_frame_index = _require_exact_int(
        evidence.frame_index,
        field="direct evidence frame_index",
    )
    evidence_timestamp = _require_exact_int(
        evidence.timestamp_ns, field="direct evidence timestamp_ns", minimum=1
    )
    if type(evidence.session_id) is not str:
        raise SuccessorContractError("direct-evidence session_id has wrong type")
    evidence_identity = (
        evidence_row_index,
        evidence.session_id,
        evidence_frame_index,
        evidence_timestamp,
    )
    mapping_identity = (
        mapping_row_index,
        mapping.session_id,
        mapping_frame_index,
        mapping_timestamp,
    )
    if evidence_identity != mapping_identity:
        raise SuccessorContractError(
            "NPZ/mapping/direct-evidence row identity mismatch"
        )
    return mapping, evidence


def _require_invalid_row_encoding(arrays: Mapping[str, np.ndarray]) -> None:
    """Treat the mandated invalid representation as part of the nine-key schema."""

    if bool(arrays["valid"][0]):
        return
    for key in POSE_KEYS:
        if not np.isnan(arrays[key][0]).all():
            raise SuccessorContractError(
                f"invalid row requires all-NaN pose matrix: {key}"
            )
    if float(arrays["confidence"][0]) != 0.0:
        raise SuccessorContractError("invalid row requires confidence=0")
    if bool(arrays["visual_observed"][0]):
        raise SuccessorContractError("invalid row requires visual_observed=false")


def _require_direct_observation_only(
    mapping: CanaryRowMapping,
    evidence: DirectObservationEvidence,
) -> None:
    flags = (
        evidence.used_neighbor_pose,
        evidence.used_interpolation,
        evidence.used_propagation,
        evidence.used_fallback,
        evidence.used_temporal_smoothing,
    )
    if not all(_is_exact_bool(value) for value in flags):
        raise SuccessorContractError("temporal/fallback audit flags must be exact bool")
    if any(flags):
        raise SuccessorContractError(
            "direct-observation-only contract forbids neighbor pose, interpolation, "
            "propagation, fallback, and temporal smoothing"
        )
    if type(evidence.operation_counters) is not OperationCounters:
        raise SuccessorContractError("forbidden operation counters are absent")
    evidence.operation_counters.require_all_zero()
    if type(evidence.source_frame_indices) is not tuple:
        raise SuccessorContractError(
            "source frame indices must be an exact integer tuple"
        )
    for value in evidence.source_frame_indices:
        _require_exact_int(value, field="source frame index")
    if evidence.source_frame_indices not in ((), (mapping.frame_index,)):
        raise SuccessorContractError(
            "direct-observation-only contract forbids neighbor source frames"
        )


def _require_se3(value: np.ndarray, *, field: str) -> None:
    transform = np.asarray(value, dtype=np.float64)
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SuccessorContractError(f"{field} is not a finite right-handed SE(3)")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if (
        not np.array_equal(transform[3], expected_bottom)
        or not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=SE3_ABSOLUTE_TOLERANCE,
        )
        or determinant <= 0.0
        or not math.isclose(
            determinant, 1.0, rel_tol=0.0, abs_tol=SE3_ABSOLUTE_TOLERANCE
        )
    ):
        raise SuccessorContractError(f"{field} is not a finite right-handed SE(3)")


def _load_bound_camera_to_world(payload: bytes, expected: np.ndarray) -> None:
    try:
        decoded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        raise SuccessorContractError(
            f"camera-to-world evidence cannot be fully decoded: {error}"
        ) from error
    if not isinstance(decoded, np.ndarray):
        if hasattr(decoded, "close"):
            decoded.close()
        raise SuccessorContractError("camera-to-world evidence must be one NPY array")
    value = np.asarray(decoded)
    if value.shape != (CANARY_ROWS, 4, 4) or value.dtype != np.dtype(np.float64):
        raise SuccessorContractError("camera-to-world evidence must be float64 [1,4,4]")
    if not np.array_equal(value, expected, equal_nan=True):
        raise SuccessorContractError(
            "camera-to-world array does not match its bound evidence bytes"
        )


def _require_evidence_metrics(
    evidence: DirectObservationEvidence,
    *,
    camera_to_world: np.ndarray,
    reserved_identities: Sequence[_FileIdentity],
    held_files: _HeldFileSet,
) -> bool:
    if not _is_exact_bool(evidence.direct_observation_attempted):
        raise SuccessorContractError("direct_observation_attempted must be exact bool")
    correspondences = _require_exact_int(
        evidence.positive_depth_correspondences,
        field="positive_depth_correspondences",
        minimum=0,
        maximum=MAX_EXACT_INT64,
    )
    inliers = _require_exact_int(
        evidence.ransac_inliers,
        field="ransac_inliers",
        minimum=0,
        maximum=MAX_EXACT_INT64,
    )
    if inliers > correspondences:
        raise SuccessorContractError(
            "direct-observation correspondence counts are invalid"
        )
    ratio = _require_finite_real(
        evidence.ransac_inlier_ratio,
        field="RANSAC inlier ratio",
        minimum=0.0,
        maximum=1.0,
    )
    residual = _require_finite_real(
        evidence.residual_m, field="RANSAC residual", minimum=0.0
    )
    expected_ratio = inliers / correspondences if correspondences else 0.0
    if not math.isclose(
        ratio,
        expected_ratio,
        rel_tol=0.0,
        abs_tol=RATIO_ABSOLUTE_TOLERANCE,
    ):
        raise SuccessorContractError("RANSAC ratio/count mathematics disagree")
    refs = (
        (evidence.raw_mask, "DIRECT_RAW_OBJECT_MASK"),
        (evidence.tracks, "DIRECT_CANONICAL_TRACKS"),
        (evidence.positive_depth, "POSITIVE_STEREO_DEPTH"),
        (evidence.camera_to_world, "CAMERA_TO_WORLD"),
    )
    complete_refs = all(record is not None for record, _ in refs)
    evidence_identities: list[_FileIdentity] = []
    for record, kind in refs:
        if record is not None:
            if type(record) is not EvidenceDigest:
                raise SuccessorContractError(f"direct evidence has wrong type: {kind}")
            payload, identity = record.require_valid(
                kind,
                held_files=held_files,
                capture=kind == "CAMERA_TO_WORLD",
            )
            evidence_identities.append(identity)
            if kind == "CAMERA_TO_WORLD":
                if payload is None:
                    raise SuccessorContractError(
                        "camera-to-world evidence bytes were not captured"
                    )
                _load_bound_camera_to_world(payload, camera_to_world)
    _require_distinct_files(evidence_identities, label="direct evidence records")
    _require_distinct_files(
        (*reserved_identities, *evidence_identities),
        label="NPZ, authority, and direct evidence records",
    )
    return (
        evidence.direct_observation_attempted
        and evidence.source_frame_indices == (CANARY_FRAME_INDEX,)
        and complete_refs
        and correspondences >= MINIMUM_3D_CORRESPONDENCES
        and inliers >= MINIMUM_RANSAC_INLIERS
        and ratio >= MINIMUM_RANSAC_INLIER_RATIO
        and residual <= MAXIMUM_RESIDUAL_M
    )


def _validate_row_mathematics(
    arrays: Mapping[str, np.ndarray],
    camera_to_world: np.ndarray | None,
    evidence: DirectObservationEvidence,
    *,
    reserved_identities: Sequence[_FileIdentity],
    held_files: _HeldFileSet,
) -> bool:
    if camera_to_world is None:
        raise SuccessorContractError("camera-to-world authority array is absent")
    camera = camera_to_world
    _require_se3(camera[0], field="T_W_C")
    passes_gates = _require_evidence_metrics(
        evidence,
        camera_to_world=camera,
        reserved_identities=reserved_identities,
        held_files=held_files,
    )
    valid = bool(arrays["valid"][0])
    observed = bool(arrays["visual_observed"][0])
    confidence = float(arrays["confidence"][0])
    if not valid:
        for key in POSE_KEYS:
            if not np.isnan(arrays[key][0]).all():
                raise SuccessorContractError(
                    f"invalid row requires all-NaN pose matrix: {key}"
                )
        if confidence != 0.0:
            raise SuccessorContractError("invalid row requires confidence=0")
        if observed:
            raise SuccessorContractError("invalid row requires visual_observed=false")
        if passes_gates:
            raise SuccessorContractError(
                "invalid row disagrees with complete evidence that passes fixed gates"
            )
        return False
    if not observed:
        raise SuccessorContractError("valid row requires visual_observed=true")
    if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
        raise SuccessorContractError("valid row confidence must be finite in (0,1]")
    if not passes_gates:
        raise SuccessorContractError("valid row fails fixed direct-observation gates")
    for key in POSE_KEYS:
        _require_se3(arrays[key][0], field=key)
    expected_world = camera[0] @ arrays["T_object_to_camera"][0]
    if not np.allclose(
        arrays["T_object_to_world"][0],
        expected_world,
        rtol=0.0,
        atol=SE3_ABSOLUTE_TOLERANCE,
    ):
        raise SuccessorContractError("T_W_O != T_W_C @ T_C_O")
    return True


def validate_canary_npz(
    npz_path: Path,
    *,
    expected_npz_bytes: int | None = None,
    expected_npz_sha256: str | None = None,
    camera_to_world: np.ndarray | None = None,
    row_mapping: Sequence[CanaryRowMapping] | None = None,
    direct_observation_evidence: Sequence[DirectObservationEvidence] | None = None,
    authorities: Mapping[str, AuthorityEvidence] | None = None,
) -> ValidationCard:
    """Validate the exact one-row successor canary without running a producer.

    Every ordinary contract failure is returned as ``CARD_FAILED``.  Unexpected
    programming errors are not hidden.  A successful card is CPU-only evidence
    that the supplied bytes obey this contract; it is never formal admission.
    """

    authorities_valid = False
    schema_valid = False
    mathematics_valid = False
    direct_only = False
    npz_bytes: int | None = None
    npz_sha256: str | None = None
    held_files: _HeldFileSet | None = None
    try:
        npz_request = _bound_file_request(
            npz_path,
            expected_bytes=expected_npz_bytes,
            expected_sha256=expected_npz_sha256,
            label="NPZ",
        )
        if npz_request.expected_bytes > MAX_CAPTURED_CONTRACT_BYTES:
            raise SuccessorContractError("successor NPZ is unreasonably large for N=1")
        authority_snapshot = _prepare_authorities(authorities)
        mapping_snapshot, evidence_snapshot = _prepare_canary_inputs(
            row_mapping, direct_observation_evidence
        )
        camera_snapshot = _snapshot_camera_to_world(camera_to_world)
        requests = (
            npz_request,
            *_authority_file_requests(authority_snapshot),
            *_direct_evidence_file_requests(evidence_snapshot[0]),
        )
        held_files = _HeldFileSet(requests)
        held_files.open_all()
    except SuccessorContractError as error:
        if held_files is not None:
            held_files.close()
        return _failed_card(str(error))
    try:
        try:
            authority_identities = _require_authorities(authority_snapshot, held_files)
            authorities_valid = True
        except SuccessorContractError as error:
            return _failed_card(str(error))
        try:
            payload, npz_sha256, npz_identity = _read_npz_bytes(
                npz_request,
                held_files=held_files,
            )
            npz_bytes = len(payload)
            arrays = _load_exact_nine_arrays(payload)
            _require_array_schema(arrays)
            mapping, evidence = _require_canary_mapping(
                arrays, mapping_snapshot, evidence_snapshot
            )
            _require_invalid_row_encoding(arrays)
            schema_valid = True
        except SuccessorContractError as error:
            return _failed_card(
                str(error),
                authorities_valid=authorities_valid,
                npz_bytes=npz_bytes,
                npz_sha256=npz_sha256,
            )
        try:
            _require_direct_observation_only(mapping, evidence)
            direct_only = True
        except SuccessorContractError as error:
            return _failed_card(
                str(error),
                authorities_valid=authorities_valid,
                schema_valid=schema_valid,
                npz_bytes=npz_bytes,
                npz_sha256=npz_sha256,
            )
        try:
            row_valid = _validate_row_mathematics(
                arrays,
                camera_snapshot,
                evidence,
                reserved_identities=(*authority_identities, npz_identity),
                held_files=held_files,
            )
            mathematics_valid = True
        except SuccessorContractError as error:
            return _failed_card(
                str(error),
                authorities_valid=authorities_valid,
                schema_valid=schema_valid,
                direct_observation_only=direct_only,
                npz_bytes=npz_bytes,
                npz_sha256=npz_sha256,
            )
        if not row_valid:
            return _failed_card(
                "row 0 did not satisfy all fixed direct-observation gates",
                authorities_valid=authorities_valid,
                schema_valid=schema_valid,
                mathematics_valid=mathematics_valid,
                direct_observation_only=direct_only,
                failed_row_indices=(0,),
                npz_bytes=npz_bytes,
                npz_sha256=npz_sha256,
            )
        try:
            held_files.verify_all()
        except SuccessorContractError as error:
            return _failed_card(
                str(error),
                authorities_valid=authorities_valid,
                schema_valid=schema_valid,
                mathematics_valid=mathematics_valid,
                direct_observation_only=direct_only,
                npz_bytes=npz_bytes,
                npz_sha256=npz_sha256,
            )
        return ValidationCard(
            status="CARD_VALIDATED_CPU_ONLY",
            failure_reasons=(),
            authorities_valid=True,
            schema_valid=True,
            mathematics_valid=True,
            direct_observation_only=True,
            rows_total=CANARY_ROWS,
            rows_valid=CANARY_ROWS,
            failed_row_indices=(),
            session_id=CANARY_SESSION_ID,
            frame_index=CANARY_FRAME_INDEX,
            npz_bytes=npz_bytes,
            npz_sha256=npz_sha256,
        )
    finally:
        held_files.close()
