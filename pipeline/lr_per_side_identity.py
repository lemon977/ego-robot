"""Fail-closed left/right identity selection contract.

This module is a CPU-only structural preflight.  It does not call SAM, read
labels, create pixels, or join masks.  Each physical side owns a distinct
candidate pool, authority lineage, and accept/reject decision.  A failure in
one authority path therefore cannot erase the other side's valid decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


Side = Literal["left", "right"]
AuthorityState = Literal["AVAILABLE", "OUTSIDE_IMAGE", "MISSING", "UNVERIFIED_IDENTITY"]
DecisionStatus = Literal["ACCEPT", "REJECT", "HOLD"]

SIDES: tuple[Side, Side] = ("left", "right")
ROUTE_OF_SELECTOR_EVIDENCE = "A_PRIME_SELECTOR_ONLY"
ROUTE_B_RELEVANT_GATE = "HAWOR_POINT_PROMPT_IDENTITY_AND_QUALITY_ONLY"
MIN_JOINT_SUPPORT_RATIO = 0.20
MIN_SIDE_MARGIN = 0.05
MAX_OBJECT_OVERLAP_OVER_INSTANCE = 0.12

FORBIDDEN_OVERRIDE_KEYS = frozenset(
    {
        "session_overrides",
        "frame_overrides",
        "session_id_thresholds",
        "per_frame_thresholds",
        "color_classifier",
        "hidden_fallback",
        "geometry_fill",
        "hard_crop",
    }
)
T0_DECISION_SHA256 = "80af258db543521a1a79d738ad44b6e411bc44b498a3abe90eee61999ce8d987"
HAWOR_POINT_QUALITY_POLICY_SHA256 = "0754ec83a22c84e39df1783b435f9693801adcb871b2b2da733c6d5895856506"
VERIFIED_SOURCE_MANIFEST_SHA256 = "1dc476360d567894ba629dd74dae608f5d8fd80b0991767611259e47929342a6"
GPU_FROZEN_REF_KINDS = {
    "t0_decision": "LR_CAUSAL_T0_JSON",
    "independent_qa": "INDEPENDENT_QA_JSON",
    "frozen_config": "A_PRIME_FROZEN_CONFIG",
    "implementation": "A_PRIME_IMPLEMENTATION",
    "input_manifest": "A_PRIME_INPUT_MANIFEST",
}


class PerSideIdentityContractError(RuntimeError):
    """Raised when the structural identity contract is violated."""


@dataclass(frozen=True)
class EvidenceRef:
    """Digest-bound regular-file evidence; free-form path strings are insufficient."""

    path: str
    bytes: int
    sha256: str
    source_kind: str

    def read_verified(self, *, allowed_root: Path) -> bytes:
        if not self.source_kind or self.source_kind == "A_PRIME_SELECTOR_OUTPUT":
            raise PerSideIdentityContractError("forbidden or missing evidence source kind")
        root = allowed_root.resolve(strict=True)
        target = Path(self.path)
        if not target.is_absolute():
            target = root / target
        normalized = Path(os.path.abspath(target))
        try:
            relative = normalized.relative_to(root)
        except ValueError as exc:
            raise PerSideIdentityContractError("evidence path escapes allowed root") from exc
        if not relative.parts:
            raise PerSideIdentityContractError("evidence path resolves to allowed root")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            directory_fd = os.open(root, directory_flags)
        except OSError as exc:
            raise PerSideIdentityContractError("evidence root open failed") from exc
        try:
            for component in relative.parts[:-1]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise PerSideIdentityContractError(
                        "evidence path contains symlink or non-directory component"
                    ) from exc
                os.close(directory_fd)
                directory_fd = child_fd
            file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise PerSideIdentityContractError("evidence open failed") from exc
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode):
                    raise PerSideIdentityContractError("evidence is not a regular file")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
            finally:
                os.close(fd)
        finally:
            os.close(directory_fd)
        if len(data) != self.bytes:
            raise PerSideIdentityContractError("evidence byte count mismatch")
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise PerSideIdentityContractError("evidence SHA256 mismatch")
        return data


@dataclass(frozen=True)
class PointPromptEvidence:
    """Typed HaWoR point-prompt record consumed only by Route B."""

    identity: Side
    lineage_id: str
    normalized_xy: tuple[float, float]
    quality_value: float
    quality_policy_ref: str
    evidence: EvidenceRef
    quality_policy: EvidenceRef
    source_manifest: EvidenceRef

    def verify(self, *, allowed_root: Path) -> str | None:
        if self.evidence.source_kind != "HAWOR_POINT_PROMPT":
            return "POINT_PROMPT_SOURCE_KIND_INVALID"
        if self.quality_policy.source_kind != "HAWOR_POINT_QUALITY_POLICY":
            return "POINT_PROMPT_QUALITY_POLICY_KIND_INVALID"
        if self.source_manifest.source_kind != "VERIFIED_SOURCE_MANIFEST":
            return "POINT_PROMPT_SOURCE_MANIFEST_KIND_INVALID"
        if self.quality_policy.sha256 != HAWOR_POINT_QUALITY_POLICY_SHA256:
            return "POINT_PROMPT_QUALITY_POLICY_SHA_MISMATCH"
        if self.source_manifest.sha256 != VERIFIED_SOURCE_MANIFEST_SHA256:
            return "POINT_PROMPT_SOURCE_MANIFEST_SHA_MISMATCH"
        policy_data = self.quality_policy.read_verified(allowed_root=allowed_root)
        source_manifest_data = self.source_manifest.read_verified(allowed_root=allowed_root)
        try:
            policy = json.loads(policy_data)
            source_manifest = json.loads(source_manifest_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PerSideIdentityContractError("point prompt policy/manifest is not JSON") from exc
        expected_policy = {
            "schema_version": "hawor-point-quality-policy-v1",
            "document_status": "FROZEN_ROUTE_B_CPU_ADMISSION_POLICY",
            "policy_id": "hawor-point-quality-v1",
            "quality_domain": "FINITE_UNIT_INTERVAL",
            "minimum_quality_rule": "STRICTLY_GREATER_THAN_ZERO",
            "minimum_quality_exclusive": 0.0,
            "label_independent_required": True,
            "source_manifest_path": "manifests/verified_source_manifest_v1.json",
            "source_manifest_sha256": VERIFIED_SOURCE_MANIFEST_SHA256,
            "route_of_evidence": "ROUTE_B_HAWOR_POINT_PROMPT_ONLY",
            "a_prime_selector_evidence_allowed": False,
            "advancement_authorized": False,
        }
        if policy != expected_policy:
            return "POINT_PROMPT_QUALITY_POLICY_PAYLOAD_MISMATCH"
        if self.quality_policy_ref != policy["policy_id"]:
            return "POINT_PROMPT_QUALITY_POLICY_ID_MISMATCH"
        if (
            source_manifest.get("schema_version") != "verified-source-manifest-v1"
            or source_manifest.get("document_status") != "VERIFIED_G0_CALIBRATION_INPUT"
            or source_manifest.get("raw_readiness") != "PASS_G0_SOURCE_READABLE"
            or source_manifest.get("immutable") is not True
        ):
            return "POINT_PROMPT_SOURCE_MANIFEST_PAYLOAD_INVALID"
        data = self.evidence.read_verified(allowed_root=allowed_root)
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PerSideIdentityContractError("point prompt evidence is not JSON") from exc
        expected = {
            "source_kind": "HAWOR_POINT_PROMPT",
            "side": self.identity,
            "lineage_id": self.lineage_id,
            "normalized_xy": list(self.normalized_xy),
            "quality_value": self.quality_value,
            "quality_policy_ref": self.quality_policy_ref,
            "quality_policy_sha256": self.quality_policy.sha256,
            "source_manifest_sha256": self.source_manifest.sha256,
            "label_independent": True,
        }
        if payload != expected:
            return "POINT_PROMPT_EVIDENCE_PAYLOAD_MISMATCH"
        if not self.lineage_id or not self.quality_policy_ref:
            return "POINT_PROMPT_PROVENANCE_MISSING"
        if (
            len(self.normalized_xy) != 2
            or not all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in self.normalized_xy)
        ):
            return "POINT_PROMPT_OUTSIDE_IMAGE"
        if not np.isfinite(self.quality_value) or not 0.0 < self.quality_value <= 1.0:
            return "POINT_PROMPT_QUALITY_INVALID"
        return None


@dataclass(frozen=True)
class SelectorThresholds:
    """The three frozen A-prime numeric thresholds.

    The fields remain explicit for manifest/report serialization, but values
    other than the frozen A-prime numbers are rejected rather than treated as
    configuration.
    """

    min_joint_support_ratio: float = MIN_JOINT_SUPPORT_RATIO
    min_side_margin: float = MIN_SIDE_MARGIN
    max_object_overlap_over_instance: float = MAX_OBJECT_OVERLAP_OVER_INSTANCE

    def validate(self) -> None:
        expected = (
            MIN_JOINT_SUPPORT_RATIO,
            MIN_SIDE_MARGIN,
            MAX_OBJECT_OVERLAP_OVER_INSTANCE,
        )
        observed = (
            self.min_joint_support_ratio,
            self.min_side_margin,
            self.max_object_overlap_over_instance,
        )
        if observed != expected:
            raise PerSideIdentityContractError(
                f"frozen selector threshold drift: {observed!r} != {expected!r}"
            )


@dataclass(frozen=True)
class SideAuthority:
    """One side's authority evidence and immutable source lineage."""

    identity: Side
    lineage_id: str
    state: AuthorityState = "AVAILABLE"
    lineage_verified: bool = False
    evidence: EvidenceRef | None = None
    quality_verified: bool = False
    label_independent_provenance: bool = False

    def hold_reason(self, *, allowed_root: Path) -> str | None:
        if self.evidence is None or self.evidence.source_kind != "HAWOR_SIDE_AUTHORITY":
            return "SIDE_AUTHORITY_PROVENANCE_MISSING"
        data = self.evidence.read_verified(allowed_root=allowed_root)
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PerSideIdentityContractError("side authority evidence is not JSON") from exc
        expected = {
            "source_kind": "HAWOR_SIDE_AUTHORITY",
            "side": self.identity,
            "lineage_id": self.lineage_id,
            "state": self.state,
            "lineage_verified": self.lineage_verified,
            "quality_verified": self.quality_verified,
            "label_independent": self.label_independent_provenance,
        }
        if payload != expected:
            return "SIDE_AUTHORITY_EVIDENCE_PAYLOAD_MISMATCH"
        if self.state == "OUTSIDE_IMAGE":
            return "SIDE_AUTHORITY_EVIDENCE_OUTSIDE_IMAGE"
        if self.state == "MISSING":
            return "SIDE_AUTHORITY_EVIDENCE_MISSING"
        if self.state == "UNVERIFIED_IDENTITY" or not self.lineage_verified:
            return "SIDE_IDENTITY_LINEAGE_UNVERIFIED"
        if not self.quality_verified:
            return "SIDE_AUTHORITY_QUALITY_UNVERIFIED"
        if not self.label_independent_provenance:
            return "SIDE_AUTHORITY_LABEL_INDEPENDENCE_UNVERIFIED"
        if self.state != "AVAILABLE":
            raise PerSideIdentityContractError(f"unknown authority state: {self.state!r}")
        if not self.lineage_id:
            return "SIDE_IDENTITY_LINEAGE_UNVERIFIED"
        return None


@dataclass(frozen=True)
class SideRawCandidate:
    """One unmodified raw SAM instance assigned to exactly one side identity."""

    identity: Side
    authority_lineage_id: str
    frame_id: str
    raw_source: EvidenceRef
    raw_instance_offset: int
    instance_id: int
    mask: np.ndarray
    score: float
    joint_support_ratio: float
    non_target_joint_support_ratio: float
    wrist_supported: bool
    boundary_supported: bool
    object6d_overlap_over_instance: float
    area_ratio: float


@dataclass(frozen=True)
class CandidateAudit:
    raw_source_sha256: str
    frame_id: str
    raw_instance_offset: int
    instance_id: int
    eligible: bool
    rejection_reasons: tuple[str, ...]
    joint_support_ratio: float
    side_difference: float
    object6d_overlap_over_instance: float
    boundary_supported_diagnostic: bool


@dataclass(frozen=True)
class SideDecision:
    identity: Side
    authority_lineage_id: str
    status: DecisionStatus
    raw_instance_offset: int | None
    instance_id: int | None
    mask: np.ndarray | None
    failure_reason: str | None
    candidate_audit: tuple[CandidateAudit, ...]


@dataclass(frozen=True)
class PerSideFrameSelection:
    left: SideDecision
    right: SideDecision
    frame_complete: bool
    frame_status: Literal["COMPLETE", "INCOMPLETE", "PARTIAL_SIDE_HOLD", "IDENTITY_HOLD"]
    pixels_created_or_edited: int = 0
    union_operations: int = 0
    crop_operations: int = 0
    fill_operations: int = 0
    morphology_operations: int = 0


@dataclass(frozen=True)
class PromptSideGate:
    identity: Side
    authority_lineage_id: str
    evidence_sha256: str
    status: Literal["PASS", "HOLD"]
    hold_reason: str | None


@dataclass(frozen=True)
class RouteBPromptIdentityGate:
    """Route-B-only gate; deliberately contains no A-prime selector result."""

    left: PromptSideGate
    right: PromptSideGate
    route_b_prompt_identity_ready: bool
    route_b_gate: str = ROUTE_B_RELEVANT_GATE
    a_prime_selector_evidence_consumed: Literal[False] = False


def _mask_sha256(mask: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(mask, dtype=np.bool_)
    header = f"{contiguous.shape[0]}x{contiguous.shape[1]}:bool:".encode("ascii")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _validate_candidate(
    candidate: SideRawCandidate,
    authority: SideAuthority,
    *,
    evidence_root: Path,
) -> np.ndarray:
    if candidate.identity != authority.identity:
        raise PerSideIdentityContractError(
            f"candidate identity crossed pool: {candidate.identity!r} -> {authority.identity!r}"
        )
    if candidate.authority_lineage_id != authority.lineage_id:
        raise PerSideIdentityContractError("candidate/authority lineage mismatch")
    if candidate.raw_instance_offset < 0:
        raise PerSideIdentityContractError("negative raw instance offset")
    mask = np.asarray(candidate.mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise PerSideIdentityContractError("raw candidate mask must be 2D bool")
    if candidate.raw_source.source_kind != "SAM_RAW_INSTANCE":
        raise PerSideIdentityContractError("raw candidate source kind mismatch")
    raw_payload = candidate.raw_source.read_verified(allowed_root=evidence_root)
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerSideIdentityContractError("raw instance evidence is not JSON") from exc
    expected_payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": candidate.frame_id,
        "instance_id": candidate.instance_id,
        "mask_sha256": _mask_sha256(mask),
    }
    if payload != expected_payload:
        raise PerSideIdentityContractError("raw instance evidence payload mismatch")
    scalar_ratios = (
        candidate.score,
        candidate.joint_support_ratio,
        candidate.non_target_joint_support_ratio,
        candidate.object6d_overlap_over_instance,
        candidate.area_ratio,
    )
    if not all(np.isfinite(value) for value in scalar_ratios):
        raise PerSideIdentityContractError("non-finite candidate evidence")
    for value in (
        candidate.joint_support_ratio,
        candidate.non_target_joint_support_ratio,
        candidate.object6d_overlap_over_instance,
        candidate.area_ratio,
    ):
        if not 0.0 <= value <= 1.0:
            raise PerSideIdentityContractError("candidate ratio outside [0,1]")
    return mask


def _rejection_reasons(
    candidate: SideRawCandidate,
    thresholds: SelectorThresholds,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.joint_support_ratio < thresholds.min_joint_support_ratio:
        reasons.append("INSUFFICIENT_EGO_JOINT_SUPPORT")
    if not candidate.wrist_supported:
        reasons.append("EGO_WRIST_NOT_SUPPORTED")
    if (
        candidate.joint_support_ratio
        < candidate.non_target_joint_support_ratio + thresholds.min_side_margin
    ):
        reasons.append("NOT_SIDE_SPECIFIC")
    if (
        candidate.object6d_overlap_over_instance
        > thresholds.max_object_overlap_over_instance
    ):
        reasons.append("OBJECT6D_PROTECTION_REJECTION")
    return tuple(reasons)


def _rank(candidate: SideRawCandidate) -> tuple[float, ...]:
    return (
        candidate.joint_support_ratio,
        candidate.joint_support_ratio - candidate.non_target_joint_support_ratio,
        candidate.score,
        -candidate.object6d_overlap_over_instance,
        -candidate.area_ratio,
        -float(candidate.raw_instance_offset),
    )


def select_side_identity(
    authority: SideAuthority,
    candidates: Sequence[SideRawCandidate],
    *,
    evidence_root: Path,
    thresholds: SelectorThresholds = SelectorThresholds(),
) -> SideDecision:
    """Evaluate one identity without reading or mutating the other side."""

    thresholds.validate()
    if authority.identity not in SIDES:
        raise PerSideIdentityContractError(f"invalid side: {authority.identity!r}")
    hold_reason = authority.hold_reason(allowed_root=evidence_root)
    if hold_reason is not None:
        return SideDecision(
            authority.identity,
            authority.lineage_id,
            "HOLD",
            None,
            None,
            None,
            hold_reason,
            (),
        )

    audited: list[tuple[SideRawCandidate, np.ndarray, CandidateAudit]] = []
    eligible: list[tuple[SideRawCandidate, np.ndarray]] = []
    for candidate in candidates:
        mask = _validate_candidate(candidate, authority, evidence_root=evidence_root)
        reasons = _rejection_reasons(candidate, thresholds)
        audit = CandidateAudit(
            candidate.raw_source.sha256,
            candidate.frame_id,
            candidate.raw_instance_offset,
            candidate.instance_id,
            not reasons,
            reasons,
            candidate.joint_support_ratio,
            candidate.joint_support_ratio - candidate.non_target_joint_support_ratio,
            candidate.object6d_overlap_over_instance,
            candidate.boundary_supported,
        )
        audited.append((candidate, mask, audit))
        if not reasons:
            eligible.append((candidate, mask))

    candidate_audit = tuple(item[2] for item in audited)
    if not eligible:
        recalled = any(
            candidate.joint_support_ratio >= thresholds.min_joint_support_ratio
            and candidate.wrist_supported
            for candidate, _, _ in audited
        )
        failure = (
            "SELECTION_RULE_REJECTED_RECALLED_INSTANCE"
            if recalled
            else "PROMPT_RECALL_INSUFFICIENT"
        )
        return SideDecision(
            authority.identity,
            authority.lineage_id,
            "REJECT",
            None,
            None,
            None,
            failure,
            candidate_audit,
        )

    selected, selected_mask = max(eligible, key=lambda item: _rank(item[0]))
    return SideDecision(
        authority.identity,
        authority.lineage_id,
        "ACCEPT",
        selected.raw_instance_offset,
        selected.instance_id,
        selected_mask.copy(),
        None,
        candidate_audit,
    )


def select_per_side_identities(
    authorities: Mapping[Side, SideAuthority],
    candidate_pools: Mapping[Side, Sequence[SideRawCandidate]],
    *,
    evidence_root: Path,
    thresholds: SelectorThresholds = SelectorThresholds(),
) -> PerSideFrameSelection:
    """Select left and right independently; never perform a frame-global abort."""

    if set(authorities) != set(SIDES) or set(candidate_pools) != set(SIDES):
        raise PerSideIdentityContractError("exact left/right authority and candidate pools required")
    left_authority = authorities["left"]
    right_authority = authorities["right"]
    if left_authority.identity != "left" or right_authority.identity != "right":
        raise PerSideIdentityContractError("authority mapping identity mismatch")

    # A raw SAM instance is one physical pixel producer identity.  Renaming
    # its side/lineage cannot make it eligible for both pools.
    left_raw_ids = {
        (candidate.frame_id, candidate.instance_id, candidate.raw_source.sha256)
        for candidate in candidate_pools["left"]
    }
    right_raw_ids = {
        (candidate.frame_id, candidate.instance_id, candidate.raw_source.sha256)
        for candidate in candidate_pools["right"]
    }
    if left_raw_ids & right_raw_ids:
        hold_left = SideDecision(
            "left", left_authority.lineage_id, "HOLD", None, None, None,
            "RAW_INSTANCE_CROSS_POOL_ALIAS", (),
        )
        hold_right = SideDecision(
            "right", right_authority.lineage_id, "HOLD", None, None, None,
            "RAW_INSTANCE_CROSS_POOL_ALIAS", (),
        )
        return PerSideFrameSelection(hold_left, hold_right, False, "IDENTITY_HOLD")

    # Aliased source lineage is not resolved by picking whichever side scores
    # higher.  Both identities stay explicit and fail closed.
    if (
        left_authority.lineage_id
        and left_authority.lineage_id == right_authority.lineage_id
    ):
        hold_left = SideDecision(
            "left", left_authority.lineage_id, "HOLD", None, None, None,
            "SIDE_IDENTITY_LINEAGE_UNVERIFIED", (),
        )
        hold_right = SideDecision(
            "right", right_authority.lineage_id, "HOLD", None, None, None,
            "SIDE_IDENTITY_LINEAGE_UNVERIFIED", (),
        )
        return PerSideFrameSelection(hold_left, hold_right, False, "IDENTITY_HOLD")

    left = select_side_identity(
        left_authority, candidate_pools["left"], evidence_root=evidence_root,
        thresholds=thresholds,
    )
    right = select_side_identity(
        right_authority, candidate_pools["right"], evidence_root=evidence_root,
        thresholds=thresholds,
    )
    complete = left.status == "ACCEPT" and right.status == "ACCEPT"
    if complete:
        frame_status = "COMPLETE"
    elif left.failure_reason == "SIDE_IDENTITY_LINEAGE_UNVERIFIED" and right.failure_reason == "SIDE_IDENTITY_LINEAGE_UNVERIFIED":
        frame_status = "IDENTITY_HOLD"
    elif left.status == "HOLD" or right.status == "HOLD":
        frame_status = "PARTIAL_SIDE_HOLD"
    else:
        frame_status = "INCOMPLETE"
    return PerSideFrameSelection(left, right, complete, frame_status)


def evaluate_route_b_prompt_identity_gate(
    prompts: Mapping[Side, PointPromptEvidence],
    *,
    evidence_root: Path,
) -> RouteBPromptIdentityGate:
    """Check only HaWoR point-prompt identity/quality for Route B.

    This gate does not accept A-prime raw candidates, selector decisions,
    boundary support, or any of the A-prime numeric thresholds as inputs.
    """

    if set(prompts) != set(SIDES):
        raise PerSideIdentityContractError("Route-B prompt gate requires exact left/right prompts")
    observed = {side: prompts[side] for side in SIDES}
    if observed["left"].identity != "left" or observed["right"].identity != "right":
        raise PerSideIdentityContractError("Route-B prompt mapping identity mismatch")
    aliased = (
        bool(observed["left"].lineage_id)
        and observed["left"].lineage_id == observed["right"].lineage_id
    ) or observed["left"].evidence.sha256 == observed["right"].evidence.sha256
    gates: dict[Side, PromptSideGate] = {}
    for side in SIDES:
        prompt = observed[side]
        reason = prompt.verify(allowed_root=evidence_root)
        if aliased:
            reason = "SIDE_IDENTITY_LINEAGE_ALIAS"
        gates[side] = PromptSideGate(
            side,
            prompt.lineage_id,
            prompt.evidence.sha256,
            "HOLD" if reason is not None else "PASS",
            reason,
        )
    return RouteBPromptIdentityGate(
        gates["left"],
        gates["right"],
        gates["left"].status == "PASS" and gates["right"].status == "PASS",
    )


def validate_cpu_preflight_governance(record: Mapping[str, Any]) -> None:
    """Reject CPU-preflight records that could silently authorize execution."""

    expected = {
        "AUTH_TIER": "T1_CPU_PREFLIGHT_ONLY",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "gpu_started": False,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }
    for key, value in expected.items():
        observed = record.get(key)
        if type(observed) is not type(value) or observed != value:
            raise PerSideIdentityContractError(f"CPU preflight governance drift: {key}")
    forbidden = sorted(_find_forbidden_keys(record))
    if forbidden:
        raise PerSideIdentityContractError(f"forbidden override keys: {forbidden}")
    unexpected = sorted(set(record) - set(expected))
    if unexpected:
        raise PerSideIdentityContractError(f"CPU preflight unexpected keys: {unexpected}")


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_OVERRIDE_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def require_a_prime_gpu_admission(
    record: Mapping[str, Any],
    *,
    frozen_refs: Mapping[str, EvidenceRef],
    evidence_root: Path,
    project_run_root: Path,
    run_root: Path,
) -> EvidenceRef:
    """Validate digest-bound gates and atomically open a new A-prime run.

    A Boolean claim that O_EXCL happened is deliberately not an input.  This
    function itself creates both the direct-child run directory and owner
    marker, and returns the marker identity for the caller's freeze manifest.
    """

    required = {
        "route_of_evidence": ROUTE_OF_SELECTOR_EVIDENCE,
        "t0_status": "COMPLETE_DECISION_BRANCH_D_PER_SIDE_IDENTITY_REQUIRED_BEFORE_ROUTE_B",
        "t0_decision_branch": "d",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "labels_read_before_run_freeze": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
    }
    for key, value in required.items():
        observed = record.get(key)
        if type(observed) is not type(value) or observed != value:
            raise PerSideIdentityContractError(f"GPU admission blocked: {key}")
    forbidden = sorted(_find_forbidden_keys(record))
    if forbidden:
        raise PerSideIdentityContractError(f"GPU admission forbidden override keys: {forbidden}")
    allowed_record_keys = set(required) | {
        "min_joint_support_ratio", "min_side_margin", "max_object_overlap_over_instance"
    }
    unexpected = sorted(set(record) - allowed_record_keys)
    if unexpected:
        raise PerSideIdentityContractError(f"GPU admission unexpected keys: {unexpected}")
    SelectorThresholds(
        record.get("min_joint_support_ratio"),
        record.get("min_side_margin"),
        record.get("max_object_overlap_over_instance"),
    ).validate()

    if set(frozen_refs) != set(GPU_FROZEN_REF_KINDS):
        raise PerSideIdentityContractError("GPU admission frozen ref set mismatch")
    ref_bytes: dict[str, bytes] = {}
    for name, expected_kind in GPU_FROZEN_REF_KINDS.items():
        ref = frozen_refs[name]
        if ref.source_kind != expected_kind:
            raise PerSideIdentityContractError(f"GPU admission source kind mismatch: {name}")
        ref_bytes[name] = ref.read_verified(allowed_root=evidence_root)
    if frozen_refs["t0_decision"].sha256 != T0_DECISION_SHA256:
        raise PerSideIdentityContractError("GPU admission T0 SHA mismatch")
    try:
        qa = json.loads(ref_bytes["independent_qa"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerSideIdentityContractError("GPU admission independent QA is not JSON") from exc
    if type(qa.get("p0_findings")) is not int or qa.get("p0_findings") != 0:
        raise PerSideIdentityContractError("GPU admission independent QA P0 is not zero")
    if qa.get("status") != "PASS_CPU_ADMISSION_EXACT":
        raise PerSideIdentityContractError("GPU admission independent QA status is not exact PASS")
    required_qa_identity = {
        "schema_version": "lr-per-side-independent-qa-v3",
        "producer": "independent_cpu_qa",
        "qa_scope": "CPU_ADMISSION_ONLY",
        "route_of_evidence": ROUTE_OF_SELECTOR_EVIDENCE,
        "labels_read": 0,
        "blind_scores_read": 0,
        "grap_a_cap_025_frames_read": 0,
        "gpu_started": False,
    }
    if any(qa.get(key) != value for key, value in required_qa_identity.items()):
        raise PerSideIdentityContractError("GPU admission independent QA identity mismatch")
    if Path(frozen_refs["independent_qa"].path).name != "INDEPENDENT_QA_LR_PER_SIDE_IDENTITY_T1_V3.json":
        raise PerSideIdentityContractError("GPU admission independent QA canonical path mismatch")

    project_source_root = Path(__file__).resolve().parents[1]
    live_paths = {
        "task_sha256": project_source_root / "archive/legacy/task_cards/18_LR_PER_SIDE_IDENTITY_T1.md",
        "module_sha256": project_source_root / "pipeline/lr_per_side_identity.py",
        "tests_sha256": project_source_root / "tests/test_lr_per_side_identity.py",
    }
    live_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in live_paths.items()
    }
    try:
        implementation = json.loads(ref_bytes["implementation"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerSideIdentityContractError("GPU admission implementation is not JSON") from exc
    expected_implementation = {
        "schema_version": "lr-per-side-implementation-v1",
        "task_path": "archive/legacy/task_cards/18_LR_PER_SIDE_IDENTITY_T1.md",
        "task_sha256": live_hashes["task_sha256"],
        "module_path": "pipeline/lr_per_side_identity.py",
        "module_sha256": live_hashes["module_sha256"],
        "tests_path": "tests/test_lr_per_side_identity.py",
        "tests_sha256": live_hashes["tests_sha256"],
    }
    if implementation != expected_implementation:
        raise PerSideIdentityContractError("GPU admission implementation/live SHA mismatch")
    expected_qa_bindings = {
        "task_sha256": live_hashes["task_sha256"],
        "module_sha256": live_hashes["module_sha256"],
        "tests_sha256": live_hashes["tests_sha256"],
        "implementation_ref_sha256": frozen_refs["implementation"].sha256,
    }
    if any(qa.get(key) != value for key, value in expected_qa_bindings.items()):
        raise PerSideIdentityContractError("GPU admission QA/live SHA binding mismatch")

    project_root = project_run_root.resolve(strict=True)
    if run_root.parent.resolve(strict=True) != project_root:
        raise PerSideIdentityContractError("GPU run must be a direct child of project _run")
    if not run_root.name.startswith("lr_per_side_"):
        raise PerSideIdentityContractError("GPU run name is outside task namespace")
    try:
        os.mkdir(run_root, 0o750)
    except FileExistsError as exc:
        raise PerSideIdentityContractError("GPU run already exists; O_EXCL required") from exc
    owner_payload = json.dumps(
        {
            "schema_version": "lr-per-side-owner-v1",
            "route_of_evidence": ROUTE_OF_SELECTOR_EVIDENCE,
            "frozen_refs": {name: frozen_refs[name].sha256 for name in sorted(frozen_refs)},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    marker = run_root / "OWNER.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(marker, flags, 0o440)
    except OSError as exc:
        raise PerSideIdentityContractError("GPU owner marker O_EXCL creation failed") from exc
    try:
        os.write(fd, owner_payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return EvidenceRef(
        str(marker), len(owner_payload), hashlib.sha256(owner_payload).hexdigest(),
        "A_PRIME_RUN_OWNER",
    )


SelectorThresholds().validate()
