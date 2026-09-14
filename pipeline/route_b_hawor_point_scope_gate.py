"""Fail-closed Route-B HaWoR point identity, scope, and GPU admission gate.

This module is deliberately independent from the Route-A/A-prime selector.  It
does not accept text recall, raw instance pools, selector decisions, or Route-A
scores.  CPU code may validate immutable HaWoR point records and the future GPU
admission bundle; it does not read H/O/U/B pixels or produce masks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Literal, Mapping, Sequence


Side = Literal["left", "right"]
PointState = Literal["AVAILABLE", "OUTSIDE_IMAGE", "MISSING", "UNVERIFIED_IDENTITY"]
GateStatus = Literal["PASS", "HOLD"]
ScopeStatus = Literal["COMPLETE_REQUIRED_SCOPE", "SYSTEMATIC_SCOPE_TOO_SMALL"]

SIDES: tuple[Side, Side] = ("left", "right")
ROUTE_OF_EVIDENCE = "ROUTE_B_HAWOR_POINT_PROMPT_ONLY"
U_SEMANTICS = "ASYMMETRIC_FALSE_NEGATIVE_ONLY"
SCOPE_COMPONENTS = (
    "hand",
    "wrist_cuff",
    "sleeve",
    "forearm_to_required_image_extent",
)
HAWOR_SOURCE_STATES = (
    "invalid",
    "observed_gated_smoothed",
    "short_optical_flow_or_interpolation",
    "mid_bidirectional_interpolation",
    "reinitialized",
)
ADMITTED_HAWOR_SOURCE_STATE = "observed_gated_smoothed"

QUALITY_POLICY_SHA256 = "0754ec83a22c84e39df1783b435f9693801adcb871b2b2da733c6d5895856506"
SOURCE_MANIFEST_SHA256 = "1dc476360d567894ba629dd74dae608f5d8fd80b0991767611259e47929342a6"
ASSET_PIN_SHA256 = "7b03a85cdecf5e71ef7b1d65dd38fdbf73deea0e9ddfc2bbe41ddc111d7b11c0"
MODEL_CONFIG_SHA256 = "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636"
MODEL_WEIGHT_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
IMPLEMENTATION_INVENTORY = (
    38,
    1_733_828,
    "97e9ab4200d741d5ea8c5c4e63b46793a6a9b886c87abea6954f21dc3520025c",
)
# Owner directive 2026-08-28: CPU gate work may continue, but Route-B GPU is
# explicitly HOLD until a later authorization is supplied and this constant is
# frozen to that authorization's digest.  ``None`` is intentionally fail-closed.
OWNER_GPU_RELEASE_AUTHORIZATION_SHA256: str | None = None
# No real Route-B HaWoR point manifest has been authorized yet.  Keeping this
# unset makes every real point gate fail closed.  CPU tests temporarily pin a
# synthetic manifest digest to exercise membership semantics without reading
# real prompts.  A future value must be frozen before labels and independently
# reviewed; a caller cannot supply its own expected digest.
HAWOR_AUTHORITY_MANIFEST_SHA256: str | None = None
HAWOR_AUTHORITY_MANIFEST_PATH = "contracts/manifests/route_b_hawor_point_authority_v4.json"
CANONICAL_QA_PATH = "archive/audits/INDEPENDENT_QA_ROUTE_B_POINT_SCOPE_GATE_T1_V4.json"
CANONICAL_OWNER_RELEASE_PATH = "contracts/authorizations/ROUTE_B_GPU_RELEASE_AUTHORIZATION.json"
HEX64 = re.compile(r"[a-f0-9]{64}")

FORBIDDEN_ROUTE_A_KEYS = frozenset(
    {
        "a_prime_evidence",
        "an_arm_prompt",
        "background_rejections",
        "raw_instance_evidence",
        "recalled_instance",
        "selected_formal_raw_instances",
        "selector_decision",
        "selector_pass_rate",
        "text_recall_evidence",
    }
)
FORBIDDEN_OVERRIDE_KEYS = frozenset(
    {
        "session_overrides",
        "frame_overrides",
        "session_id_thresholds",
        "per_frame_thresholds",
        "legacy_fallback",
        "hidden_fallback",
        "geometry_fill",
        "hard_crop",
        "corridor_fill",
        "mask_union",
        "morphology",
    }
)
FORBIDDEN_ROUTE_A_VALUES = frozenset(
    {
        "A_PRIME_SELECTOR_ONLY",
        "ROUTE_A",
        "ROUTE_A_SELECTOR",
        "SELECTION_RULE_REJECTED_RECALLED_INSTANCE",
        "PROMPT_RECALL_INSUFFICIENT",
    }
)
SEALED_TOKENS = (
    "grap_a_cap_025",
    "cross_session_blind",
    "same_session_blind",
)

GPU_FROZEN_REF_KINDS = {
    "task": "ROUTE_B_GATE_TASK",
    "identity_scope_module": "ROUTE_B_GATE_MODULE",
    "identity_scope_tests": "ROUTE_B_GATE_TESTS",
    "route_b_contract": "ROUTE_B_CONTRACT_MODULE",
    "route_b_contract_tests": "ROUTE_B_CONTRACT_TESTS",
    "implementation_freeze": "ROUTE_B_IMPLEMENTATION_FREEZE",
    "hyperparameter_freeze": "ROUTE_B_HYPERPARAMETER_FREEZE",
    "input_freeze": "ROUTE_B_INPUT_FREEZE",
    "hawor_authority_manifest": "HAWOR_AUTHORITY_MANIFEST",
    "prompt_manifest": "ROUTE_B_SEALED_PROMPT_MANIFEST",
    "raw_input_manifest": "ROUTE_B_RAW_INPUT_MANIFEST",
    "asset_pin": "SAM21_ASSET_PIN",
    "independent_qa": "ROUTE_B_INDEPENDENT_QA",
    "owner_gpu_release": "ROUTE_B_GPU_RELEASE_AUTHORIZATION",
}


class RouteBPointScopeContractError(RuntimeError):
    """Raised whenever Route-B identity/scope/admission evidence is incomplete."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceRef:
    """A root-contained, no-follow, regular-file byte identity."""

    path: str
    bytes: int
    sha256: str
    source_kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "source_kind": self.source_kind,
        }

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> EvidenceRef:
        if not isinstance(value, Mapping) or set(value) != {
            "path", "bytes", "sha256", "source_kind"
        }:
            raise RouteBPointScopeContractError(f"{label} EvidenceRef fields drift")
        return cls(
            value["path"], value["bytes"], value["sha256"], value["source_kind"]
        )

    def read_verified(self, *, allowed_root: Path) -> bytes:
        if type(self.bytes) is not int or self.bytes <= 0:
            raise RouteBPointScopeContractError("evidence byte count is not a positive integer")
        if not isinstance(self.sha256, str) or not HEX64.fullmatch(self.sha256):
            raise RouteBPointScopeContractError("evidence SHA256 is malformed")
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise RouteBPointScopeContractError("evidence source kind is missing")
        root = allowed_root.resolve(strict=True)
        target = Path(self.path)
        if not target.is_absolute():
            target = root / target
        normalized = Path(os.path.abspath(target))
        try:
            relative = normalized.relative_to(root)
        except ValueError as exc:
            raise RouteBPointScopeContractError("evidence path escapes allowed root") from exc
        if not relative.parts:
            raise RouteBPointScopeContractError("evidence path resolves to allowed root")
        directory_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(root, directory_flags)
        except OSError as exc:
            raise RouteBPointScopeContractError("evidence root open failed") from exc
        try:
            for component in relative.parts[:-1]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise RouteBPointScopeContractError(
                        "evidence path contains symlink or non-directory component"
                    ) from exc
                os.close(directory_fd)
                directory_fd = child_fd
            try:
                file_fd = os.open(
                    relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise RouteBPointScopeContractError("evidence open failed") from exc
            try:
                metadata = os.fstat(file_fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise RouteBPointScopeContractError("evidence is not a regular file")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
            finally:
                os.close(file_fd)
        finally:
            os.close(directory_fd)
        if len(data) != self.bytes:
            raise RouteBPointScopeContractError("evidence byte count mismatch")
        if sha256_bytes(data) != self.sha256:
            raise RouteBPointScopeContractError("evidence SHA256 mismatch")
        return data


@dataclass(frozen=True)
class HaworPointPrompt:
    """One side's label-independent point and immutable provenance."""

    side: Side
    state: PointState
    session_id: str
    frame_index: int
    lineage_id: str
    track_id: str
    normalized_xy: tuple[float, float] | None
    quality_value: float | None
    source_slot: int
    source_track_index: int | None
    hawor_source_state: str
    hawor_source_valid: bool
    hawor_measurement_accepted: bool
    hawor_source: EvidenceRef
    evidence: EvidenceRef


@dataclass(frozen=True)
class SidePointGate:
    side: Side
    status: GateStatus
    reason: str | None
    lineage_id: str
    evidence_sha256: str | None


@dataclass(frozen=True)
class PointIdentityGateResult:
    left: SidePointGate
    right: SidePointGate
    route_b_point_identity_ready: bool


def _json_object(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteBPointScopeContractError(f"{label} is not JSON") from exc
    if not isinstance(value, Mapping):
        raise RouteBPointScopeContractError(f"{label} is not a JSON object")
    return value


def _canonical_json_value(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise RouteBPointScopeContractError(
            f"{label} is not a finite canonical JSON value"
        ) from exc


def _iter_key_values(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _iter_key_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_key_values(child)


def _iter_scalars(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_scalars(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_scalars(child)
    else:
        yield value


def reject_route_a_or_override_evidence(value: Any) -> None:
    """Reject cross-route evidence and hidden semantic overrides recursively."""

    forbidden_keys = FORBIDDEN_ROUTE_A_KEYS | FORBIDDEN_OVERRIDE_KEYS
    hits = sorted({key for key, _ in _iter_key_values(value) if key in forbidden_keys})
    if hits:
        raise RouteBPointScopeContractError(f"forbidden Route-A/override keys: {hits}")
    for child in _iter_scalars(value):
        if isinstance(child, str) and child in FORBIDDEN_ROUTE_A_VALUES:
            raise RouteBPointScopeContractError(f"forbidden Route-A evidence value: {child}")


def _require_canonical_ref_path(
    ref: EvidenceRef, *, expected_relative_path: str
) -> None:
    if Path(ref.path).is_absolute() or Path(ref.path).as_posix() != expected_relative_path:
        raise RouteBPointScopeContractError(
            f"evidence ref is not canonical project-relative path: {expected_relative_path}"
        )


def _validate_policy_and_manifest_refs(
    *, quality_policy_ref: EvidenceRef, source_manifest_ref: EvidenceRef, evidence_root: Path
) -> None:
    if quality_policy_ref.source_kind != "HAWOR_POINT_QUALITY_POLICY":
        raise RouteBPointScopeContractError("point quality policy source kind mismatch")
    if source_manifest_ref.source_kind != "VERIFIED_SOURCE_MANIFEST":
        raise RouteBPointScopeContractError("point source manifest kind mismatch")
    if quality_policy_ref.sha256 != QUALITY_POLICY_SHA256:
        raise RouteBPointScopeContractError("point quality policy SHA mismatch")
    if source_manifest_ref.sha256 != SOURCE_MANIFEST_SHA256:
        raise RouteBPointScopeContractError("point source manifest SHA mismatch")
    policy = _json_object(
        quality_policy_ref.read_verified(allowed_root=evidence_root), "point quality policy"
    )
    expected_policy = {
        "schema_version": "hawor-point-quality-policy-v1",
        "document_status": "FROZEN_ROUTE_B_CPU_ADMISSION_POLICY",
        "policy_id": "hawor-point-quality-v1",
        "quality_domain": "FINITE_UNIT_INTERVAL",
        "minimum_quality_rule": "STRICTLY_GREATER_THAN_ZERO",
        "minimum_quality_exclusive": 0.0,
        "label_independent_required": True,
        "source_manifest_path": "manifests/verified_source_manifest_v1.json",
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "a_prime_selector_evidence_allowed": False,
        "advancement_authorized": False,
    }
    if policy != expected_policy:
        raise RouteBPointScopeContractError("point quality policy payload mismatch")
    # This RAW authority contains no HaWoR point identities.  It is still bound
    # because the quality policy names it, while the separate typed HaWoR
    # authority below supplies the required per-point membership relation.
    source = _json_object(
        source_manifest_ref.read_verified(allowed_root=evidence_root),
        "verified RAW source manifest",
    )
    if (
        source.get("schema_version") != "verified-source-manifest-v1"
        or source.get("immutable") is not True
        or source.get("no_fallback") is not True
        or "grap_a_cap_004" not in source.get("authorized_calibration_sessions", [])
    ):
        raise RouteBPointScopeContractError("verified RAW source manifest semantics drift")


def _load_hawor_authority_manifest(
    ref: EvidenceRef, *, evidence_root: Path
) -> dict[tuple[str, int, Side], Mapping[str, Any]]:
    """Load the one digest-pinned upstream HaWoR membership manifest."""

    if HAWOR_AUTHORITY_MANIFEST_SHA256 is None:
        raise RouteBPointScopeContractError(
            "real HaWoR authority manifest remains unpinned; point identity HOLD"
        )
    if ref.source_kind != "HAWOR_AUTHORITY_MANIFEST":
        raise RouteBPointScopeContractError("HaWoR authority manifest kind mismatch")
    _require_canonical_ref_path(ref, expected_relative_path=HAWOR_AUTHORITY_MANIFEST_PATH)
    if ref.sha256 != HAWOR_AUTHORITY_MANIFEST_SHA256:
        raise RouteBPointScopeContractError("HaWoR authority manifest SHA mismatch")
    payload = _json_object(
        ref.read_verified(allowed_root=evidence_root), "HaWoR authority manifest"
    )
    reject_route_a_or_override_evidence(payload)
    expected_top = {
        "schema_version",
        "document_status",
        "route_of_evidence",
        "label_independent",
        "entries",
    }
    if set(payload) != expected_top:
        raise RouteBPointScopeContractError("HaWoR authority manifest fields drift")
    if (
        payload.get("schema_version") != "route-b-hawor-authority-manifest-v4"
        or payload.get("document_status") != "FROZEN_UPSTREAM_HAWOR_POINT_AUTHORITY"
        or payload.get("route_of_evidence") != ROUTE_OF_EVIDENCE
        or payload.get("label_independent") is not True
        or not isinstance(payload.get("entries"), list)
        or not payload["entries"]
    ):
        raise RouteBPointScopeContractError("HaWoR authority manifest semantics drift")
    index: dict[tuple[str, int, Side], Mapping[str, Any]] = {}
    entry_fields = {
        "session_id",
        "frame_index",
        "side",
        "state",
        "lineage_id",
        "track_id",
        "source_slot",
        "source_track_index",
        "hawor_source_state",
        "hawor_source_valid",
        "hawor_measurement_accepted",
        "hawor_source_ref",
        "normalized_xy",
        "quality_value",
    }
    for entry in payload["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != entry_fields:
            raise RouteBPointScopeContractError("HaWoR authority entry fields drift")
        side = entry.get("side")
        session = entry.get("session_id")
        frame = entry.get("frame_index")
        state = entry.get("state")
        if (
            side not in SIDES
            or not isinstance(session, str)
            or not session
            or type(frame) is not int
            or frame < 0
            or state not in {"AVAILABLE", "OUTSIDE_IMAGE"}
            or not isinstance(entry.get("lineage_id"), str)
            or not entry["lineage_id"]
            or not isinstance(entry.get("track_id"), str)
            or not entry["track_id"]
        ):
            raise RouteBPointScopeContractError("HaWoR authority entry identity invalid")
        if type(entry.get("source_slot")) is not int or entry["source_slot"] not in (0, 1):
            raise RouteBPointScopeContractError("HaWoR authority entry slot invalid")
        if type(entry.get("source_track_index")) is not int or entry["source_track_index"] < -1:
            raise RouteBPointScopeContractError("HaWoR authority entry track invalid")
        if (
            entry.get("hawor_source_state") not in HAWOR_SOURCE_STATES
            or type(entry.get("hawor_source_valid")) is not bool
            or type(entry.get("hawor_measurement_accepted")) is not bool
        ):
            raise RouteBPointScopeContractError("HaWoR authority source state invalid")
        source_ref = EvidenceRef.from_mapping(
            entry.get("hawor_source_ref"), label="HaWoR source"
        )
        if source_ref.source_kind != "FROZEN_HAWOR_GEOMETRY":
            raise RouteBPointScopeContractError("HaWoR authority source kind invalid")
        xy = entry.get("normalized_xy")
        quality = entry.get("quality_value")
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(type(value) not in (int, float) or isinstance(value, bool) for value in xy)
            or any(not math.isfinite(float(value)) for value in xy)
            or type(quality) not in (int, float)
            or isinstance(quality, bool)
            or not math.isfinite(float(quality))
            or not 0.0 < float(quality) <= 1.0
        ):
            raise RouteBPointScopeContractError("HaWoR authority point/quality invalid")
        inside = all(0.0 <= float(value) <= 1.0 for value in xy)
        if (state == "AVAILABLE") is not inside:
            raise RouteBPointScopeContractError("HaWoR authority point state mismatch")
        key = (session, frame, side)
        if key in index:
            raise RouteBPointScopeContractError("duplicate HaWoR authority membership")
        index[key] = entry
    return index


def _validate_authority_left_right_pair_independence(
    authority_index: Mapping[tuple[str, int, Side], Mapping[str, Any]],
) -> None:
    """Reject one HaWoR identity presented as both sides of a frozen frame.

    ``source_slot`` is deliberately not used as identity evidence here: valid pairs
    must already carry left=0/right=1, and the V3 attack preserved that difference
    while aliasing the underlying identity.  This reproduces the pair rule used by
    :func:`evaluate_hawor_point_identity_gate` in the GPU input-freeze path.
    """

    frame_keys = sorted({(session, frame) for session, frame, _ in authority_index})
    for session, frame in frame_keys:
        left = authority_index.get((session, frame, "left"))
        right = authority_index.get((session, frame, "right"))
        if left is None or right is None:
            raise RouteBPointScopeContractError(
                "HaWoR authority left/right pair membership incomplete"
            )
        left_source = EvidenceRef.from_mapping(
            left.get("hawor_source_ref"), label="left HaWoR authority source"
        )
        right_source = EvidenceRef.from_mapping(
            right.get("hawor_source_ref"), label="right HaWoR authority source"
        )
        same_geometry_track = (
            left_source.sha256 == right_source.sha256
            and left.get("source_track_index") == right.get("source_track_index")
        )
        if (
            left.get("lineage_id") == right.get("lineage_id")
            or left.get("track_id") == right.get("track_id")
            or same_geometry_track
        ):
            raise RouteBPointScopeContractError(
                "HaWoR authority left/right identity alias"
            )


def _validate_authority_source_admission(
    authority_index: Mapping[tuple[str, int, Side], Mapping[str, Any]],
) -> None:
    """Require every frozen training point to be a direct accepted observation."""

    for (_, _, side), entry in authority_index.items():
        expected_slot = 0 if side == "left" else 1
        if entry.get("source_slot") != expected_slot:
            raise RouteBPointScopeContractError(
                "HaWoR authority source slot side-collapse or alias"
            )
        if entry.get("hawor_source_state") != ADMITTED_HAWOR_SOURCE_STATE:
            raise RouteBPointScopeContractError(
                "HaWoR authority source state not admitted"
            )
        if entry.get("hawor_source_valid") is not True:
            raise RouteBPointScopeContractError("HaWoR authority source invalid")
        if entry.get("hawor_measurement_accepted") is not True:
            raise RouteBPointScopeContractError(
                "HaWoR authority source measurement not accepted"
            )
        if (
            type(entry.get("source_track_index")) is not int
            or entry["source_track_index"] < 0
        ):
            raise RouteBPointScopeContractError(
                "HaWoR authority source track not admitted"
            )


def _evaluate_one_prompt(
    prompt: HaworPointPrompt,
    *,
    evidence_root: Path,
    authority_entry: Mapping[str, Any] | None,
) -> SidePointGate:
    if prompt.side not in SIDES:
        raise RouteBPointScopeContractError("invalid prompt side")
    if type(prompt.frame_index) is not int or prompt.frame_index < 0:
        raise RouteBPointScopeContractError("prompt frame index is invalid")
    if not prompt.session_id or not prompt.lineage_id or not prompt.track_id:
        return SidePointGate(prompt.side, "HOLD", "POINT_IDENTITY_LINEAGE_MISSING", prompt.lineage_id, None)
    expected_slot = 0 if prompt.side == "left" else 1
    if type(prompt.source_slot) is not int or prompt.source_slot != expected_slot:
        return SidePointGate(
            prompt.side, "HOLD", "POINT_SOURCE_SLOT_IDENTITY_MISMATCH", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state in {"AVAILABLE", "OUTSIDE_IMAGE"} and (
        type(prompt.source_track_index) is not int or prompt.source_track_index < 0
    ):
        return SidePointGate(
            prompt.side, "HOLD", "POINT_SOURCE_TRACK_MISSING_OR_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.source_track_index is not None and (
        type(prompt.source_track_index) is not int or prompt.source_track_index < 0
    ):
        return SidePointGate(
            prompt.side, "HOLD", "POINT_SOURCE_TRACK_MISSING_OR_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.hawor_source.source_kind != "FROZEN_HAWOR_GEOMETRY":
        return SidePointGate(
            prompt.side, "HOLD", "POINT_HAWOR_SOURCE_KIND_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state == "AVAILABLE":
        if prompt.hawor_source_state != ADMITTED_HAWOR_SOURCE_STATE:
            return SidePointGate(
                prompt.side, "HOLD", "POINT_SOURCE_STATE_NOT_ADMITTED",
                prompt.lineage_id, prompt.evidence.sha256,
            )
        if prompt.hawor_source_valid is not True:
            return SidePointGate(
                prompt.side, "HOLD", "POINT_SOURCE_INVALID",
                prompt.lineage_id, prompt.evidence.sha256,
            )
        if prompt.hawor_measurement_accepted is not True:
            return SidePointGate(
                prompt.side, "HOLD", "POINT_SOURCE_MEASUREMENT_NOT_ACCEPTED",
                prompt.lineage_id, prompt.evidence.sha256,
            )
    if prompt.state in {"AVAILABLE", "OUTSIDE_IMAGE"}:
        if authority_entry is None:
            return SidePointGate(
                prompt.side, "HOLD", "POINT_NOT_IN_PINNED_HAWOR_AUTHORITY",
                prompt.lineage_id, prompt.evidence.sha256,
            )
        expected_authority = {
            "session_id": prompt.session_id,
            "frame_index": prompt.frame_index,
            "side": prompt.side,
            "state": prompt.state,
            "lineage_id": prompt.lineage_id,
            "track_id": prompt.track_id,
            "source_slot": prompt.source_slot,
            "source_track_index": prompt.source_track_index,
            "hawor_source_state": prompt.hawor_source_state,
            "hawor_source_valid": prompt.hawor_source_valid,
            "hawor_measurement_accepted": prompt.hawor_measurement_accepted,
            "hawor_source_ref": prompt.hawor_source.as_dict(),
            "normalized_xy": None if prompt.normalized_xy is None else [
                float(prompt.normalized_xy[0]), float(prompt.normalized_xy[1])
            ],
            "quality_value": None if prompt.quality_value is None else float(prompt.quality_value),
        }
        if dict(authority_entry) != expected_authority:
            return SidePointGate(
                prompt.side, "HOLD", "POINT_PINNED_HAWOR_MEMBERSHIP_MISMATCH",
                prompt.lineage_id, prompt.evidence.sha256,
            )
        prompt.hawor_source.read_verified(allowed_root=evidence_root)
    if prompt.evidence.source_kind != "HAWOR_ROUTE_B_POINT_PROMPT":
        return SidePointGate(
            prompt.side, "HOLD", "POINT_EVIDENCE_SOURCE_KIND_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    xy = prompt.normalized_xy
    quality = prompt.quality_value
    state_reason = {
        "AVAILABLE": None,
        "OUTSIDE_IMAGE": "POINT_OUTSIDE_IMAGE",
        "MISSING": "POINT_MISSING",
        "UNVERIFIED_IDENTITY": "POINT_IDENTITY_UNVERIFIED",
    }.get(prompt.state, "UNKNOWN_POINT_STATE")
    if state_reason == "UNKNOWN_POINT_STATE":
        return SidePointGate(
            prompt.side, "HOLD", state_reason, prompt.lineage_id, prompt.evidence.sha256
        )
    coordinate_well_formed = (
        xy is not None
        and len(xy) == 2
        and all(type(value) in (float, int) and not isinstance(value, bool) for value in xy)
        and all(float("-inf") < float(value) < float("inf") for value in xy)
    )
    quality_well_formed = (
        type(quality) in (float, int)
        and not isinstance(quality, bool)
        and float("-inf") < float(quality) < float("inf")
        and 0.0 < float(quality) <= 1.0
    )
    if prompt.state == "AVAILABLE" and (
        not coordinate_well_formed
        or any(not 0.0 <= float(value) <= 1.0 for value in xy or ())
    ):
        return SidePointGate(
            prompt.side, "HOLD", "POINT_OUTSIDE_IMAGE", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state == "AVAILABLE" and not quality_well_formed:
        return SidePointGate(
            prompt.side, "HOLD", "POINT_QUALITY_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state == "OUTSIDE_IMAGE" and (
        not coordinate_well_formed
        or not any(not 0.0 <= float(value) <= 1.0 for value in xy or ())
    ):
        return SidePointGate(
            prompt.side, "HOLD", "POINT_OUTSIDE_STATE_EVIDENCE_INVALID",
            prompt.lineage_id, prompt.evidence.sha256,
        )
    if prompt.state == "OUTSIDE_IMAGE" and not quality_well_formed:
        return SidePointGate(
            prompt.side, "HOLD", "POINT_QUALITY_INVALID", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state in {"MISSING", "UNVERIFIED_IDENTITY"} and (
        xy is not None or quality is not None
    ):
        return SidePointGate(
            prompt.side, "HOLD", "POINT_UNAVAILABLE_STATE_HAS_MEASUREMENT",
            prompt.lineage_id, prompt.evidence.sha256,
        )
    payload = _json_object(prompt.evidence.read_verified(allowed_root=evidence_root), "point evidence")
    reject_route_a_or_override_evidence(payload)
    expected = {
        "schema_version": "route-b-hawor-point-v2",
        "source_kind": "HAWOR_ROUTE_B_POINT_PROMPT",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "session_id": prompt.session_id,
        "frame_index": prompt.frame_index,
        "side": prompt.side,
        "state": prompt.state,
        "lineage_id": prompt.lineage_id,
        "track_id": prompt.track_id,
        "source_slot": prompt.source_slot,
        "source_track_index": prompt.source_track_index,
        "hawor_source_state": prompt.hawor_source_state,
        "hawor_source_valid": prompt.hawor_source_valid,
        "hawor_measurement_accepted": prompt.hawor_measurement_accepted,
        "hawor_source": {
            "bytes": prompt.hawor_source.bytes,
            "sha256": prompt.hawor_source.sha256,
        },
        "normalized_xy": None if xy is None else [float(xy[0]), float(xy[1])],
        "quality_value": None if quality is None else float(quality),
        "quality_policy_sha256": QUALITY_POLICY_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "hawor_authority_manifest_sha256": HAWOR_AUTHORITY_MANIFEST_SHA256,
        "label_independent": True,
        "route_a_selector_evidence_consumed": False,
    }
    if payload != expected:
        return SidePointGate(
            prompt.side, "HOLD", "POINT_EVIDENCE_PAYLOAD_MISMATCH", prompt.lineage_id,
            prompt.evidence.sha256,
        )
    if prompt.state != "AVAILABLE":
        return SidePointGate(
            prompt.side, "HOLD", state_reason, prompt.lineage_id, prompt.evidence.sha256
        )
    return SidePointGate(prompt.side, "PASS", None, prompt.lineage_id, prompt.evidence.sha256)


def evaluate_hawor_point_identity_gate(
    prompts: Mapping[Side, HaworPointPrompt],
    *,
    quality_policy_ref: EvidenceRef,
    source_manifest_ref: EvidenceRef,
    hawor_authority_manifest_ref: EvidenceRef,
    evidence_root: Path,
) -> PointIdentityGateResult:
    """Validate left/right point identity independently; never frame-abort one side."""

    if set(prompts) != set(SIDES):
        raise RouteBPointScopeContractError("point gate requires exact left/right identities")
    _validate_policy_and_manifest_refs(
        quality_policy_ref=quality_policy_ref,
        source_manifest_ref=source_manifest_ref,
        evidence_root=evidence_root,
    )
    authority = _load_hawor_authority_manifest(
        hawor_authority_manifest_ref, evidence_root=evidence_root
    )
    observed = {side: prompts[side] for side in SIDES}
    if observed["left"].side != "left" or observed["right"].side != "right":
        raise RouteBPointScopeContractError("point mapping identity mismatch")
    results: dict[Side, SidePointGate] = {}
    for side in SIDES:
        try:
            point = observed[side]
            results[side] = _evaluate_one_prompt(
                point,
                evidence_root=evidence_root,
                authority_entry=authority.get((point.session_id, point.frame_index, side)),
            )
        except (OSError, ValueError, TypeError, RouteBPointScopeContractError) as exc:
            results[side] = SidePointGate(
                side, "HOLD", f"POINT_SIDE_VERIFICATION_FAILED:{type(exc).__name__}",
                observed[side].lineage_id, observed[side].evidence.sha256,
            )
    pair_mismatch = (
        observed["left"].session_id != observed["right"].session_id
        or observed["left"].frame_index != observed["right"].frame_index
    )
    if pair_mismatch:
        results = {
            side: SidePointGate(
                side, "HOLD", "LEFT_RIGHT_POINT_PAIR_FRAME_MISMATCH",
                observed[side].lineage_id, observed[side].evidence.sha256,
            )
            for side in SIDES
        }
    both_have_lineage = all(observed[side].lineage_id for side in SIDES)
    evidence_alias = (
        observed["left"].evidence.sha256 == observed["right"].evidence.sha256
    )
    if not pair_mismatch and both_have_lineage and (
        observed["left"].lineage_id == observed["right"].lineage_id
        or observed["left"].track_id == observed["right"].track_id
        or (
            observed["left"].hawor_source.sha256 == observed["right"].hawor_source.sha256
            and observed["left"].source_track_index is not None
            and observed["left"].source_track_index == observed["right"].source_track_index
        )
        or evidence_alias
    ):
        results = {
            side: SidePointGate(
                side, "HOLD", "LEFT_RIGHT_POINT_IDENTITY_ALIAS",
                observed[side].lineage_id,
                observed[side].evidence.sha256,
            )
            for side in SIDES
        }
    return PointIdentityGateResult(
        results["left"], results["right"],
        results["left"].status == "PASS" and results["right"].status == "PASS",
    )


@dataclass(frozen=True)
class ScopeObservation:
    """A review-only scope record; it can never modify decoder pixels."""

    side: Side
    session_id: str
    frame_index: int
    prompt_evidence_ref: EvidenceRef
    decoder_mask_ref: EvidenceRef
    decoder_mask_record_ref: EvidenceRef
    hand: bool
    wrist_cuff: bool
    sleeve: bool
    forearm_to_required_image_extent: bool
    route_of_evidence: str = ROUTE_OF_EVIDENCE
    evaluator_role: str = "REQUIRED_SCOPE_REVIEW_ONLY"
    modifies_pixels: bool = False


@dataclass(frozen=True)
class SideScopeResult:
    side: Side
    status: ScopeStatus
    missing_components: tuple[str, ...]


@dataclass(frozen=True)
class ScopePanelResult:
    status: ScopeStatus
    left_total: int
    left_too_small: int
    right_total: int
    right_too_small: int
    observations: tuple[SideScopeResult, ...]


def evaluate_scope_observation(
    observation: ScopeObservation, *, evidence_root: Path
) -> SideScopeResult:
    if observation.side not in SIDES:
        raise RouteBPointScopeContractError("scope side invalid")
    if type(observation.frame_index) is not int or observation.frame_index < 0:
        raise RouteBPointScopeContractError("scope frame invalid")
    if observation.route_of_evidence != ROUTE_OF_EVIDENCE:
        raise RouteBPointScopeContractError("scope cross-route evidence forbidden")
    if observation.evaluator_role != "REQUIRED_SCOPE_REVIEW_ONLY":
        raise RouteBPointScopeContractError("scope evaluator role drift")
    if type(observation.modifies_pixels) is not bool or observation.modifies_pixels:
        raise RouteBPointScopeContractError("scope evaluator may not modify pixels")
    if observation.prompt_evidence_ref.source_kind != "HAWOR_ROUTE_B_POINT_PROMPT":
        raise RouteBPointScopeContractError("scope prompt evidence kind mismatch")
    prompt_payload = _json_object(
        observation.prompt_evidence_ref.read_verified(allowed_root=evidence_root),
        "scope point prompt",
    )
    reject_route_a_or_override_evidence(prompt_payload)
    for key, expected in {
        "schema_version": "route-b-hawor-point-v2",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "session_id": observation.session_id,
        "frame_index": observation.frame_index,
        "side": observation.side,
    }.items():
        if type(prompt_payload.get(key)) is not type(expected) or prompt_payload.get(key) != expected:
            raise RouteBPointScopeContractError(f"scope prompt identity mismatch: {key}")
    if observation.decoder_mask_ref.source_kind != "SAM21_RAW_DECODER_MASK":
        raise RouteBPointScopeContractError("scope decoder mask kind mismatch")
    observation.decoder_mask_ref.read_verified(allowed_root=evidence_root)
    if observation.decoder_mask_record_ref.source_kind != "SAM21_RAW_DECODER_MASK_RECORD":
        raise RouteBPointScopeContractError("scope decoder mask record kind mismatch")
    mask_record = _json_object(
        observation.decoder_mask_record_ref.read_verified(allowed_root=evidence_root),
        "scope decoder mask record",
    )
    expected_record = {
        "schema_version": "route-b-raw-decoder-mask-record-v1",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "session_id": observation.session_id,
        "frame_index": observation.frame_index,
        "side": observation.side,
        "prompt_evidence_ref": observation.prompt_evidence_ref.as_dict(),
        "decoder_mask_ref": observation.decoder_mask_ref.as_dict(),
        "postprocessing_applied": False,
    }
    if mask_record != expected_record:
        raise RouteBPointScopeContractError("scope decoder mask record identity mismatch")
    values = {
        component: getattr(observation, component)
        for component in SCOPE_COMPONENTS
    }
    if any(type(value) is not bool for value in values.values()):
        raise RouteBPointScopeContractError("scope component must be strict Boolean")
    missing = tuple(component for component, present in values.items() if not present)
    return SideScopeResult(
        observation.side,
        "SYSTEMATIC_SCOPE_TOO_SMALL" if missing else "COMPLETE_REQUIRED_SCOPE",
        missing,
    )


def aggregate_scope_panel(
    observations: Sequence[ScopeObservation], *, evidence_root: Path
) -> ScopePanelResult:
    """Fail if any side misses a required part; aggregate coverage cannot hide it."""

    if not observations:
        raise RouteBPointScopeContractError("scope panel is empty")
    results = tuple(
        evaluate_scope_observation(value, evidence_root=evidence_root)
        for value in observations
    )
    totals = {side: sum(value.side == side for value in results) for side in SIDES}
    if any(totals[side] == 0 for side in SIDES):
        raise RouteBPointScopeContractError("scope panel must report both sides")
    short = {
        side: sum(
            value.side == side and value.status == "SYSTEMATIC_SCOPE_TOO_SMALL"
            for value in results
        )
        for side in SIDES
    }
    return ScopePanelResult(
        "SYSTEMATIC_SCOPE_TOO_SMALL" if any(short.values()) else "COMPLETE_REQUIRED_SCOPE",
        totals["left"], short["left"], totals["right"], short["right"], results,
    )


def implementation_inventory(path: Path) -> tuple[int, int, str]:
    """Use the exact ASSET_PIN inventory algorithm and reject cache artifacts."""

    root = path.resolve(strict=True)
    if not root.is_dir() or path.is_symlink():
        raise RouteBPointScopeContractError("implementation root invalid")
    rows: list[str] = []
    regular_bytes = 0
    for value in sorted(root.rglob("*")):
        relative = value.relative_to(root).as_posix()
        if value.name == "__pycache__" or value.suffix == ".pyc":
            raise RouteBPointScopeContractError("implementation bytecode/cache pollution")
        if value.is_symlink():
            rows.append(f"{relative}\0symlink\0{os.readlink(value)}\n")
        elif value.is_file():
            size = value.stat().st_size
            regular_bytes += size
            rows.append(f"{relative}\0file\0{size}\0{sha256_file(value)}\n")
    return len(rows), regular_bytes, sha256_bytes("".join(rows).encode("utf-8"))


def _require_directory_nofollow(path: Path, *, allowed_root: Path) -> None:
    root = allowed_root.resolve(strict=True)
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(root)
    except ValueError as exc:
        raise RouteBPointScopeContractError("asset directory escapes project root") from exc
    descriptor = os.open(
        root, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        for component in relative.parts:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RouteBPointScopeContractError(
                    "asset directory contains symlink or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _regular_file_identity_nofollow(path: Path, *, allowed_root: Path) -> tuple[int, str]:
    """Hash a file through a component-wise no-follow descriptor walk."""

    root = allowed_root.resolve(strict=True)
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(root)
    except ValueError as exc:
        raise RouteBPointScopeContractError("asset file escapes project root") from exc
    if not relative.parts:
        raise RouteBPointScopeContractError("asset file resolves to project root")
    directory_fd = os.open(
        root, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise RouteBPointScopeContractError(
                    "asset file path contains symlink or non-directory component"
                ) from exc
            os.close(directory_fd)
            directory_fd = child
        try:
            descriptor = os.open(
                relative.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise RouteBPointScopeContractError("asset file open failed") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RouteBPointScopeContractError("asset is not a regular file")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    return total, digest.hexdigest()


def require_bytecode_disabled() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RouteBPointScopeContractError("PYTHONDONTWRITEBYTECODE must equal 1")
    if sys.dont_write_bytecode is not True:
        raise RouteBPointScopeContractError("sys.dont_write_bytecode must be True")


def verify_sam21_asset_pin(
    *, project_root: Path, asset_pin_ref: EvidenceRef,
    expected_asset_pin_sha256: str = ASSET_PIN_SHA256,
) -> Mapping[str, Any]:
    """Bind pin/config/checkpoint/implementation inventory before GPU admission."""

    require_bytecode_disabled()
    if asset_pin_ref.source_kind != "SAM21_ASSET_PIN":
        raise RouteBPointScopeContractError("asset pin source kind mismatch")
    if asset_pin_ref.sha256 != expected_asset_pin_sha256:
        raise RouteBPointScopeContractError("asset pin SHA mismatch")
    pin = _json_object(asset_pin_ref.read_verified(allowed_root=project_root), "asset pin")
    if pin.get("schema_version") != "local-model-asset-pin-v1":
        raise RouteBPointScopeContractError("asset pin schema mismatch")
    if pin.get("model_identifier") != "SAM2_1_HIERA_LARGE":
        raise RouteBPointScopeContractError("asset pin model mismatch")
    local = pin.get("local")
    if not isinstance(local, Mapping):
        raise RouteBPointScopeContractError("asset pin local section missing")
    implementation = project_root / str(local.get("implementation_ref", ""))
    config = implementation / str(local.get("config_path", ""))
    weight = project_root / str(local.get("weight_path", ""))
    expected_inventory = (
        local.get("implementation_inventory_entries"),
        local.get("implementation_regular_bytes"),
        local.get("implementation_inventory_sha256"),
    )
    _require_directory_nofollow(implementation, allowed_root=project_root)
    inventory = implementation_inventory(implementation)
    if expected_asset_pin_sha256 == ASSET_PIN_SHA256 and inventory != IMPLEMENTATION_INVENTORY:
        raise RouteBPointScopeContractError("registered implementation inventory constant drift")
    if inventory != expected_inventory:
        raise RouteBPointScopeContractError("implementation inventory/pin drift")
    config_identity = _regular_file_identity_nofollow(config, allowed_root=project_root)
    if config_identity[1] != local.get("config_sha256"):
        raise RouteBPointScopeContractError("model config digest drift")
    if expected_asset_pin_sha256 == ASSET_PIN_SHA256 and local.get("config_sha256") != MODEL_CONFIG_SHA256:
        raise RouteBPointScopeContractError("registered model config constant drift")
    weight_identity = _regular_file_identity_nofollow(weight, allowed_root=project_root)
    if weight_identity[0] != local.get("weight_bytes") or weight_identity[1] != local.get("weight_sha256"):
        raise RouteBPointScopeContractError("model checkpoint byte identity drift")
    if expected_asset_pin_sha256 == ASSET_PIN_SHA256 and local.get("weight_sha256") != MODEL_WEIGHT_SHA256:
        raise RouteBPointScopeContractError("registered model checkpoint constant drift")
    return {
        "asset_pin_sha256": asset_pin_ref.sha256,
        "implementation_inventory_entries": inventory[0],
        "implementation_regular_bytes": inventory[1],
        "implementation_inventory_sha256": inventory[2],
        "config_sha256": local.get("config_sha256"),
        "weight_sha256": local.get("weight_sha256"),
    }


def validate_cpu_preflight_governance(record: Mapping[str, Any]) -> None:
    expected = {
        "AUTH_TIER": "T1_CPU_PREFLIGHT_ONLY",
        "WHY_NOT_BLOCKED": "AUTHORIZED_ROUTE_B_CPU_GATE_NO_LABEL_GPU_OR_MASK_ACCESS",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "gpu_started": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "route_a_evidence_consumed": False,
    }
    if set(record) != set(expected):
        raise RouteBPointScopeContractError("CPU preflight governance key set drift")
    for key, expected_value in expected.items():
        observed = record.get(key)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise RouteBPointScopeContractError(f"CPU preflight governance drift: {key}")
    reject_route_a_or_override_evidence(record)


def _contains_sealed_token(value: Any) -> bool:
    if isinstance(value, str):
        return any(token in value for token in SEALED_TOKENS)
    if isinstance(value, Mapping):
        # Exact schemas contain explicit false-valued audit fields such as
        # ``grap_a_cap_025_references_in_freeze``.  Scan referenced values, not
        # those required governance field names.
        return any(_contains_sealed_token(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sealed_token(child) for child in value)
    return False


def _live_implementation_binding(project_root: Path) -> Mapping[str, str]:
    paths = {
        "task_sha256": project_root / "archive/legacy/task_cards/23_ROUTE_B_HAWOR_POINT_IDENTITY_AND_SCOPE_GATE_T1.md",
        "identity_scope_module_sha256": project_root / "pipeline/route_b_hawor_point_scope_gate.py",
        "identity_scope_tests_sha256": project_root / "tests/test_route_b_hawor_point_scope_gate.py",
        "route_b_contract_sha256": project_root / "pipeline/sam21_route_b_contract.py",
        "route_b_contract_tests_sha256": project_root / "tests/test_sam21_route_b_contract.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_hyperparameter_freeze(value: Mapping[str, Any]) -> None:
    """Require every archive/legacy/task_cards/16 result-sensitive training choice before labels."""

    expected_keys = {
        "schema_version", "route_of_evidence", "frozen_before_labels",
        "one_configuration_only", "no_sweep", "seed", "optimizer",
        "learning_rate", "weight_decay", "loss_formula", "u_semantics",
        "batch_size", "gradient_accumulation_steps", "max_steps",
        "prompt_generation", "augmentation", "mask_logit_threshold",
        "checkpoint_every_steps", "stop_rule", "trainable_prefix",
        "trainable_tensor_count", "trainable_parameter_count", "oof_folds",
    }
    if set(value) != expected_keys:
        raise RouteBPointScopeContractError("hyperparameter freeze key set incomplete or expanded")
    exact = {
        "schema_version": "route-b-hyperparameter-freeze-v1",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "frozen_before_labels": True,
        "one_configuration_only": True,
        "no_sweep": True,
        "optimizer": "AdamW",
        "loss_formula": "ASYMMETRIC_HOUB_V1",
        "u_semantics": U_SEMANTICS,
        "prompt_generation": "FROZEN_LABEL_INDEPENDENT_HAWOR_PER_SIDE_POINT",
        "mask_logit_threshold": 0.0,
        "stop_rule": "FIXED_MAX_STEPS_NO_RESULT_DRIVEN_EARLY_STOP",
        "trainable_prefix": "sam_mask_decoder.*",
        "trainable_tensor_count": 131,
        "trainable_parameter_count": 4_215_109,
        "oof_folds": {
            "F1": [228, 231, 234, 237, 240],
            "F2": [229, 232, 235, 238, 241],
            "F3": [230, 233, 236, 239, 242],
        },
    }
    for key, expected in exact.items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise RouteBPointScopeContractError(f"hyperparameter freeze drift: {key}")
    for key in ("seed", "batch_size", "gradient_accumulation_steps", "max_steps", "checkpoint_every_steps"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise RouteBPointScopeContractError(f"hyperparameter must be positive integer: {key}")
    for key, lower, inclusive in (("learning_rate", 0.0, False), ("weight_decay", 0.0, True)):
        observed = value.get(key)
        if type(observed) not in (int, float) or isinstance(observed, bool):
            raise RouteBPointScopeContractError(f"hyperparameter must be numeric: {key}")
        if not math.isfinite(float(observed)):
            raise RouteBPointScopeContractError(f"hyperparameter must be finite: {key}")
        if (observed < lower) if inclusive else (observed <= lower):
            raise RouteBPointScopeContractError(f"hyperparameter range invalid: {key}")
    augmentation = value.get("augmentation")
    if not isinstance(augmentation, list) or any(not isinstance(item, str) for item in augmentation):
        raise RouteBPointScopeContractError("augmentation freeze must be a string list")
    reject_route_a_or_override_evidence(value)


def _validate_prompt_manifest(
    ref: EvidenceRef,
    *,
    evidence_root: Path,
    hawor_authority_manifest_ref: EvidenceRef,
    authority_index: Mapping[tuple[str, int, Side], Mapping[str, Any]],
) -> None:
    if ref.source_kind != "ROUTE_B_SEALED_PROMPT_MANIFEST":
        raise RouteBPointScopeContractError("prompt manifest source kind mismatch")
    payload = _json_object(ref.read_verified(allowed_root=evidence_root), "prompt manifest")
    reject_route_a_or_override_evidence(payload)
    expected_keys = {
        "schema_version", "route_of_evidence", "sealed_before_labels",
        "labels_read_before_seal", "supervised_session", "development_frames",
        "hawor_authority_manifest_ref", "entries",
    }
    if set(payload) != expected_keys:
        raise RouteBPointScopeContractError("prompt manifest fields drift")
    if (
        payload.get("schema_version") != "route-b-sealed-point-prompt-manifest-v4"
        or payload.get("route_of_evidence") != ROUTE_OF_EVIDENCE
        or payload.get("sealed_before_labels") is not True
        or type(payload.get("labels_read_before_seal")) is not int
        or payload.get("labels_read_before_seal") != 0
        or payload.get("supervised_session") != "grap_a_cap_004"
        or payload.get("development_frames") != list(range(228, 243))
        or payload.get("hawor_authority_manifest_ref")
        != hawor_authority_manifest_ref.as_dict()
        or not isinstance(payload.get("entries"), list)
    ):
        raise RouteBPointScopeContractError("prompt manifest semantics drift")
    expected_keys_set = {
        ("grap_a_cap_004", frame, side)
        for frame in range(228, 243)
        for side in SIDES
    }
    if set(authority_index) != expected_keys_set:
        raise RouteBPointScopeContractError(
            "HaWoR authority manifest does not cover exact dev15 sides"
        )
    seen: set[tuple[str, int, Side]] = set()
    for entry in payload["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "session_id", "frame_index", "side", "point_evidence_ref"
        }:
            raise RouteBPointScopeContractError("prompt manifest entry fields drift")
        side = entry.get("side")
        key = (entry.get("session_id"), entry.get("frame_index"), side)
        if side not in SIDES or key not in expected_keys_set or key in seen:
            raise RouteBPointScopeContractError("prompt manifest entry identity drift")
        point_ref = EvidenceRef.from_mapping(
            entry.get("point_evidence_ref"), label="prompt manifest point"
        )
        if point_ref.source_kind != "HAWOR_ROUTE_B_POINT_PROMPT":
            raise RouteBPointScopeContractError("prompt manifest point kind mismatch")
        point = _json_object(
            point_ref.read_verified(allowed_root=evidence_root), "sealed point prompt"
        )
        reject_route_a_or_override_evidence(point)
        if (
            point.get("schema_version") != "route-b-hawor-point-v2"
            or point.get("route_of_evidence") != ROUTE_OF_EVIDENCE
            or point.get("session_id") != key[0]
            or point.get("frame_index") != key[1]
            or point.get("side") != key[2]
            or point.get("hawor_authority_manifest_sha256")
            != hawor_authority_manifest_ref.sha256
            or point.get("label_independent") is not True
            or point.get("route_a_selector_evidence_consumed") is not False
        ):
            raise RouteBPointScopeContractError("sealed point prompt identity drift")
        authority_entry = authority_index.get(key)  # type: ignore[arg-type]
        if authority_entry is None:
            raise RouteBPointScopeContractError(
                "sealed point prompt is absent from pinned HaWoR authority"
            )
        joined_fields = (
            "session_id",
            "frame_index",
            "side",
            "state",
            "lineage_id",
            "track_id",
            "source_slot",
            "source_track_index",
            "hawor_source_state",
            "hawor_source_valid",
            "hawor_measurement_accepted",
            "normalized_xy",
            "quality_value",
        )
        for field in joined_fields:
            point_value = _canonical_json_value(
                point.get(field), label=f"sealed point {field}"
            )
            authority_value = _canonical_json_value(
                authority_entry.get(field),
                label=f"HaWoR authority {field}",
            )
            if point_value != authority_value:
                raise RouteBPointScopeContractError(
                    f"sealed point/HaWoR authority membership mismatch: {field}"
                )
        authority_source = EvidenceRef.from_mapping(
            authority_entry.get("hawor_source_ref"), label="HaWoR authority source"
        )
        if authority_source.source_kind != "FROZEN_HAWOR_GEOMETRY":
            raise RouteBPointScopeContractError(
                "sealed point authority geometry source kind mismatch"
            )
        expected_point_source = {
            "bytes": authority_source.bytes,
            "sha256": authority_source.sha256,
        }
        if _canonical_json_value(
            point.get("hawor_source"), label="sealed point HaWoR geometry"
        ) != _canonical_json_value(
            expected_point_source, label="HaWoR authority geometry"
        ):
            raise RouteBPointScopeContractError(
                "sealed point/HaWoR authority geometry bytes/SHA mismatch"
            )
        authority_source.read_verified(allowed_root=evidence_root)
        seen.add(key)  # type: ignore[arg-type]
    if seen != expected_keys_set:
        raise RouteBPointScopeContractError("prompt manifest does not cover exact dev15 sides")


def _validate_raw_input_manifest(ref: EvidenceRef, *, evidence_root: Path) -> None:
    if ref.source_kind != "ROUTE_B_RAW_INPUT_MANIFEST":
        raise RouteBPointScopeContractError("RAW input manifest source kind mismatch")
    payload = _json_object(ref.read_verified(allowed_root=evidence_root), "RAW input manifest")
    reject_route_a_or_override_evidence(payload)
    if set(payload) != {
        "schema_version", "route_of_evidence", "frozen_before_labels",
        "labels_read_before_freeze", "supervised_session", "development_frames",
        "entries",
    }:
        raise RouteBPointScopeContractError("RAW input manifest fields drift")
    if (
        payload.get("schema_version") != "route-b-raw-input-manifest-v2"
        or payload.get("route_of_evidence") != ROUTE_OF_EVIDENCE
        or payload.get("frozen_before_labels") is not True
        or type(payload.get("labels_read_before_freeze")) is not int
        or payload.get("labels_read_before_freeze") != 0
        or payload.get("supervised_session") != "grap_a_cap_004"
        or payload.get("development_frames") != list(range(228, 243))
        or not isinstance(payload.get("entries"), list)
    ):
        raise RouteBPointScopeContractError("RAW input manifest semantics drift")
    seen: set[int] = set()
    for entry in payload["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "session_id", "frame_index", "raw_image_ref"
        }:
            raise RouteBPointScopeContractError("RAW input entry fields drift")
        frame = entry.get("frame_index")
        if (
            entry.get("session_id") != "grap_a_cap_004"
            or type(frame) is not int
            or frame not in range(228, 243)
            or frame in seen
        ):
            raise RouteBPointScopeContractError("RAW input entry identity drift")
        image_ref = EvidenceRef.from_mapping(entry.get("raw_image_ref"), label="RAW image")
        if image_ref.source_kind != "FROZEN_RAW_RGB":
            raise RouteBPointScopeContractError("RAW image source kind mismatch")
        image_ref.read_verified(allowed_root=evidence_root)
        seen.add(frame)
    if seen != set(range(228, 243)):
        raise RouteBPointScopeContractError("RAW input manifest does not cover exact dev15")


def validate_input_freeze(
    value: Mapping[str, Any],
    *,
    evidence_root: Path,
    prompt_manifest_ref: EvidenceRef,
    raw_input_manifest_ref: EvidenceRef,
    hawor_authority_manifest_ref: EvidenceRef,
) -> None:
    expected = {
        "schema_version": "route-b-input-freeze-v2",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "supervised_session": "grap_a_cap_004",
        "supervised_split": "development",
        "development_frames": list(range(228, 243)),
        "oof_fold_ids": ["F1", "F2", "F3"],
        "labels_read_before_freeze": 0,
        "point_prompts_sealed_before_labels": True,
        "point_identity_gate": "PASS_LEFT_RIGHT_INDEPENDENT",
        "route_a_evidence_consumed": False,
        "blind_references_in_freeze": False,
        "grap_a_cap_025_references_in_freeze": False,
        "free_visual_sessions_after_candidate": [
            "grap_a_cap_002", "grap_a_cap_005", "grap_a_cap_012"
        ],
    }
    ref_keys = {
        "prompt_manifest_ref",
        "raw_input_manifest_ref",
        "hawor_authority_manifest_ref",
    }
    if set(value) != set(expected) | ref_keys:
        raise RouteBPointScopeContractError("input freeze key set incomplete or expanded")
    for key, expected_value in expected.items():
        observed = value.get(key)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise RouteBPointScopeContractError(f"input freeze drift: {key}")
    frozen_ref_values = {
        "prompt_manifest_ref": prompt_manifest_ref,
        "raw_input_manifest_ref": raw_input_manifest_ref,
        "hawor_authority_manifest_ref": hawor_authority_manifest_ref,
    }
    for key, ref in frozen_ref_values.items():
        if value.get(key) != ref.as_dict():
            raise RouteBPointScopeContractError(f"input freeze EvidenceRef mismatch: {key}")
    authority_index = _load_hawor_authority_manifest(
        hawor_authority_manifest_ref, evidence_root=evidence_root
    )
    _validate_prompt_manifest(
        prompt_manifest_ref,
        evidence_root=evidence_root,
        hawor_authority_manifest_ref=hawor_authority_manifest_ref,
        authority_index=authority_index,
    )
    _validate_authority_left_right_pair_independence(authority_index)
    _validate_authority_source_admission(authority_index)
    _validate_raw_input_manifest(raw_input_manifest_ref, evidence_root=evidence_root)


def require_route_b_gpu_admission(
    record: Mapping[str, Any],
    *,
    frozen_refs: Mapping[str, EvidenceRef],
    evidence_root: Path,
    project_root: Path,
    project_run_root: Path,
    run_root: Path,
) -> EvidenceRef:
    """Verify exact QA/freeze/asset gates, then create a fresh Route-B run."""

    if OWNER_GPU_RELEASE_AUTHORIZATION_SHA256 is None:
        raise RouteBPointScopeContractError(
            "GPU blocked by current owner directive; no Route-B GPU release is registered"
        )
    if evidence_root.resolve(strict=True) != project_root.resolve(strict=True):
        raise RouteBPointScopeContractError(
            "GPU admission evidence root must be the exact project root"
        )

    required = {
        "AUTH_TIER": "T1",
        "WHY_NOT_BLOCKED": "AUTHORIZED_TASK16_AFTER_INDEPENDENT_P0_ZERO",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "round_index": 1,
        "previous_completed_route_b_rounds": 0,
        "point_identity_gate": "PASS_LEFT_RIGHT_INDEPENDENT",
        "scope_gate": "PENDING_RUNTIME_REQUIRED_SCOPE_EVALUATION",
        "u_semantics": U_SEMANTICS,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read_before_run_freeze": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "route_a_evidence_consumed": False,
        "gpu_started_before_admission": False,
    }
    if set(record) != set(required):
        raise RouteBPointScopeContractError("GPU admission record key set mismatch")
    for key, expected in required.items():
        if key in {"round_index", "previous_completed_route_b_rounds"}:
            continue
        observed = record.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise RouteBPointScopeContractError(f"GPU admission blocked: {key}")
    round_index = record.get("round_index")
    previous_rounds = record.get("previous_completed_route_b_rounds")
    if type(round_index) is not int or type(previous_rounds) is not int:
        raise RouteBPointScopeContractError("GPU admission round counters must be strict integers")
    if (round_index, previous_rounds) not in {(1, 0), (2, 1)}:
        raise RouteBPointScopeContractError("GPU admission permits exactly Route-B round 1 or 2")
    reject_route_a_or_override_evidence(record)
    if set(frozen_refs) != set(GPU_FROZEN_REF_KINDS):
        raise RouteBPointScopeContractError("GPU admission frozen ref set mismatch")
    payloads: dict[str, bytes] = {}
    for name, expected_kind in GPU_FROZEN_REF_KINDS.items():
        ref = frozen_refs[name]
        if ref.source_kind != expected_kind:
            raise RouteBPointScopeContractError(f"GPU admission source kind mismatch: {name}")
        payloads[name] = ref.read_verified(allowed_root=evidence_root)
    live = _live_implementation_binding(project_root)
    live_ref_bindings = {
        "task": "task_sha256",
        "identity_scope_module": "identity_scope_module_sha256",
        "identity_scope_tests": "identity_scope_tests_sha256",
        "route_b_contract": "route_b_contract_sha256",
        "route_b_contract_tests": "route_b_contract_tests_sha256",
    }
    for ref_name, live_name in live_ref_bindings.items():
        if frozen_refs[ref_name].sha256 != live[live_name]:
            raise RouteBPointScopeContractError(f"GPU admission live ref SHA mismatch: {ref_name}")
    implementation = _json_object(payloads["implementation_freeze"], "implementation freeze")
    reject_route_a_or_override_evidence(implementation)
    expected_implementation = {
        "schema_version": "route-b-point-scope-implementation-freeze-v4",
        "task_path": "archive/legacy/task_cards/23_ROUTE_B_HAWOR_POINT_IDENTITY_AND_SCOPE_GATE_T1.md",
        "module_path": "pipeline/route_b_hawor_point_scope_gate.py",
        "tests_path": "tests/test_route_b_hawor_point_scope_gate.py",
        "route_b_contract_path": "pipeline/sam21_route_b_contract.py",
        "route_b_contract_tests_path": "tests/test_sam21_route_b_contract.py",
        **live,
    }
    if implementation != expected_implementation:
        raise RouteBPointScopeContractError("GPU admission implementation/live SHA mismatch")
    qa = _json_object(payloads["independent_qa"], "independent QA")
    reject_route_a_or_override_evidence(qa)
    qa_binding_names = {
        "task",
        "identity_scope_module",
        "identity_scope_tests",
        "route_b_contract",
        "route_b_contract_tests",
        "implementation_freeze",
        "hyperparameter_freeze",
        "input_freeze",
        "asset_pin",
        "hawor_authority_manifest",
        "prompt_manifest",
        "raw_input_manifest",
    }
    expected_qa = {
        "schema_version": "route-b-point-scope-independent-qa-v4",
        "producer": "independent_cpu_qa",
        "qa_scope": "ROUTE_B_POINT_IDENTITY_SCOPE_GPU_ADMISSION_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "status": "PASS_CPU_ADMISSION_EXACT",
        "p0_findings": 0,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "real_hawor_prompts_read": 0,
        "gpu_started": False,
        **live,
        "frozen_authority_refs": {
            name: frozen_refs[name].as_dict() for name in sorted(qa_binding_names)
        },
    }
    if qa != expected_qa:
        raise RouteBPointScopeContractError("GPU admission independent QA/binding mismatch")
    _require_canonical_ref_path(
        frozen_refs["independent_qa"], expected_relative_path=CANONICAL_QA_PATH
    )
    release = _json_object(payloads["owner_gpu_release"], "owner GPU release")
    reject_route_a_or_override_evidence(release)
    if frozen_refs["owner_gpu_release"].sha256 != OWNER_GPU_RELEASE_AUTHORIZATION_SHA256:
        raise RouteBPointScopeContractError("GPU admission owner release SHA mismatch")
    _require_canonical_ref_path(
        frozen_refs["owner_gpu_release"],
        expected_relative_path=CANONICAL_OWNER_RELEASE_PATH,
    )
    expected_release = {
        "schema_version": "route-b-gpu-release-authorization-v3",
        "authorization_scope": "TASK16_ROUTE_B_GPU_ROUND_1_OR_2_ONLY",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "gpu_release_authorized": True,
        "maximum_route_b_rounds": 2,
        **live,
        "independent_qa_ref": frozen_refs["independent_qa"].as_dict(),
        "implementation_freeze_ref": frozen_refs["implementation_freeze"].as_dict(),
        "hyperparameter_freeze_ref": frozen_refs["hyperparameter_freeze"].as_dict(),
        "input_freeze_ref": frozen_refs["input_freeze"].as_dict(),
        "hawor_authority_manifest_ref": frozen_refs["hawor_authority_manifest"].as_dict(),
        "prompt_manifest_ref": frozen_refs["prompt_manifest"].as_dict(),
        "raw_input_manifest_ref": frozen_refs["raw_input_manifest"].as_dict(),
    }
    if release != expected_release:
        raise RouteBPointScopeContractError("GPU admission owner release payload/binding mismatch")
    if frozen_refs["asset_pin"].sha256 != ASSET_PIN_SHA256:
        raise RouteBPointScopeContractError("GPU admission asset pin SHA mismatch")
    hyperparameters = _json_object(payloads["hyperparameter_freeze"], "hyperparameter freeze")
    inputs = _json_object(payloads["input_freeze"], "input freeze")
    reject_route_a_or_override_evidence(hyperparameters)
    reject_route_a_or_override_evidence(inputs)
    if _contains_sealed_token(hyperparameters) or _contains_sealed_token(inputs):
        raise RouteBPointScopeContractError("GPU admission sealed 025/blind reference")
    validate_hyperparameter_freeze(hyperparameters)
    validate_input_freeze(
        inputs,
        evidence_root=evidence_root,
        prompt_manifest_ref=frozen_refs["prompt_manifest"],
        raw_input_manifest_ref=frozen_refs["raw_input_manifest"],
        hawor_authority_manifest_ref=frozen_refs["hawor_authority_manifest"],
    )
    verify_sam21_asset_pin(
        project_root=project_root,
        asset_pin_ref=frozen_refs["asset_pin"],
    )
    expected_runs = project_root / "_run"
    if Path(os.path.abspath(project_run_root)) != Path(os.path.abspath(expected_runs)):
        raise RouteBPointScopeContractError("GPU project run root is not canonical _run")
    _require_directory_nofollow(project_run_root, allowed_root=project_root)
    project_runs = project_run_root.resolve(strict=True)
    if Path(os.path.abspath(run_root.parent)) != Path(os.path.abspath(project_runs)):
        raise RouteBPointScopeContractError("GPU run must be a direct child of project _run")
    if not run_root.name.startswith("route_b_"):
        raise RouteBPointScopeContractError("GPU run name outside Route-B namespace")
    run_parent_fd = os.open(
        project_runs,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        try:
            os.mkdir(run_root.name, 0o750, dir_fd=run_parent_fd)
        except FileExistsError as exc:
            raise RouteBPointScopeContractError("GPU run already exists; O_EXCL required") from exc
        run_fd = os.open(
            run_root.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0),
            dir_fd=run_parent_fd,
        )
    finally:
        os.close(run_parent_fd)
    owner_payload = json.dumps(
        {
            "schema_version": "route-b-owner-v1",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "round_index": record["round_index"],
            "frozen_refs": {name: frozen_refs[name].sha256 for name in sorted(frozen_refs)},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    marker = run_root / "OWNER.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open("OWNER.json", flags, 0o440, dir_fd=run_fd)
    except OSError as exc:
        raise RouteBPointScopeContractError("GPU owner marker O_EXCL creation failed") from exc
    try:
        os.write(descriptor, owner_payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        os.close(run_fd)
    return EvidenceRef(
        str(marker), len(owner_payload), sha256_bytes(owner_payload), "ROUTE_B_RUN_OWNER"
    )
