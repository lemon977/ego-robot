"""CPU-only A-prime distributed-side evidence contract.

This versioned module implements exactly the two semantic changes frozen by
``archive/legacy/task_cards/26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md``:

* wrist support is diagnostic-only; and
* an out-of-image authority wrist is diagnostic-only while side routing and
  acceptance use the distribution of *in-image* HaWoR joints.

It never calls a model, changes a mask pixel, combines masks, reads labels, or
authorizes GPU execution.  Route-B evidence is deliberately outside its API.
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
ROUTE_OF_EVIDENCE = "A_PRIME_SELECTOR_ONLY"
PRODUCER_ID = "a_prime_distributed_side_evidence_v1"
TASK26_RELATIVE_PATH = "archive/legacy/task_cards/26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md"
TASK26_SHA256 = "ac58a593b4642e562686f018ba75f0bc75a9c8563ad943f61c2d9bd4363a2843"
TEXT_PROMPT = "an arm"
MIN_JOINT_SUPPORT_RATIO = 0.20
MIN_SIDE_MARGIN = 0.05
MAX_OBJECT_OVERLAP_OVER_INSTANCE = 0.12
GPU_EXECUTION_AUTHORIZED = False
FORBIDDEN_SESSION = "grap_a_cap_025"
FIXED_FRAME_INDICES: Mapping[str, tuple[int, ...]] = {
    "grap_a_cap_004": tuple(range(228, 243)),
    "grap_a_cap_002": (20, 70, 121, 171, 221, 271, 322, 372),
    "grap_a_cap_005": (20, 70, 120, 170, 220, 270, 320, 370),
    "grap_a_cap_012": (19, 65, 112, 158, 205, 251, 298, 344),
}
WRONG_SIDE_SENTINELS: tuple[tuple[str, int, int, Side], ...] = (
    ("grap_a_cap_005", 220, 1, "right"),
    ("grap_a_cap_005", 270, 1, "right"),
)
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
        "morphology",
        "prompt_override",
    }
)


class DistributedSideEvidenceError(RuntimeError):
    """Raised when the frozen distributed-side contract is violated."""


def canonical_json(value: Any) -> bytes:
    """Return deterministic JSON bytes used by every digest-bound record."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o440) -> None:
    """Create one regular file with O_EXCL and fsync its contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, mode)
    except OSError as exc:
        raise DistributedSideEvidenceError(f"O_EXCL creation failed: {path}") from exc
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class EvidenceRef:
    """Digest-bound regular file opened without following any path symlink."""

    path: str
    bytes: int
    sha256: str
    source_kind: str

    def read_verified(self, *, allowed_root: Path) -> bytes:
        if not self.source_kind:
            raise DistributedSideEvidenceError("evidence source_kind is required")
        root = allowed_root.resolve(strict=True)
        target = Path(self.path)
        if not target.is_absolute():
            target = root / target
        normalized = Path(os.path.abspath(target))
        try:
            relative = normalized.relative_to(root)
        except ValueError as exc:
            raise DistributedSideEvidenceError("evidence path escapes allowed root") from exc
        if not relative.parts:
            raise DistributedSideEvidenceError("evidence path resolves to allowed root")

        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            directory_fd = os.open(root, directory_flags)
        except OSError as exc:
            raise DistributedSideEvidenceError("evidence root open failed") from exc
        try:
            for component in relative.parts[:-1]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise DistributedSideEvidenceError(
                        "evidence path contains symlink or non-directory component"
                    ) from exc
                os.close(directory_fd)
                directory_fd = child_fd
            file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise DistributedSideEvidenceError("evidence open failed") from exc
            try:
                status = os.fstat(fd)
                if not stat.S_ISREG(status.st_mode):
                    raise DistributedSideEvidenceError("evidence is not a regular file")
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
            raise DistributedSideEvidenceError("evidence byte count mismatch")
        if sha256_bytes(data) != self.sha256:
            raise DistributedSideEvidenceError("evidence SHA256 mismatch")
        return data


@dataclass(frozen=True)
class SelectorThresholds:
    min_joint_support_ratio: float = MIN_JOINT_SUPPORT_RATIO
    min_side_margin: float = MIN_SIDE_MARGIN
    max_object_overlap_over_instance: float = MAX_OBJECT_OVERLAP_OVER_INSTANCE

    def validate(self) -> None:
        observed = (
            self.min_joint_support_ratio,
            self.min_side_margin,
            self.max_object_overlap_over_instance,
        )
        expected = (
            MIN_JOINT_SUPPORT_RATIO,
            MIN_SIDE_MARGIN,
            MAX_OBJECT_OVERLAP_OVER_INSTANCE,
        )
        if observed != expected:
            raise DistributedSideEvidenceError(
                f"frozen selector threshold drift: {observed!r} != {expected!r}"
            )


@dataclass(frozen=True)
class SideAuthority:
    identity: Side
    session_id: str
    frame_index: int
    lineage_id: str
    state: AuthorityState
    wrist_xy: tuple[float, float] | None
    source_frame_selection: EvidenceRef
    label_independent: bool = True

    def validate(
        self,
        *,
        evidence_root: Path,
        image_shape: tuple[int, int],
    ) -> str | None:
        if self.identity not in SIDES:
            raise DistributedSideEvidenceError(f"invalid authority side: {self.identity!r}")
        if self.source_frame_selection.source_kind != "A_PRIME_V3_FRAME_SELECTION":
            raise DistributedSideEvidenceError("authority source kind mismatch")
        data = self.source_frame_selection.read_verified(allowed_root=evidence_root)
        try:
            payload = json.loads(data)
            source_side = payload["sides"][self.identity]
            source_authority = source_side["authority"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DistributedSideEvidenceError("authority source payload invalid") from exc
        expected_wrist = None if self.wrist_xy is None else list(self.wrist_xy)
        expected = {
            "frame_index": self.frame_index,
            "identity": self.identity,
            "lineage_id": self.lineage_id,
            "state": self.state,
            "wrist_xy": expected_wrist,
        }
        observed = {
            "frame_index": payload.get("frame_index"),
            "identity": source_authority.get("identity"),
            "lineage_id": source_authority.get("lineage_id"),
            "state": source_authority.get("state"),
            "wrist_xy": source_authority.get("wrist_xy"),
        }
        if observed != expected:
            raise DistributedSideEvidenceError("authority/source payload mismatch")
        if not self.session_id or not self.lineage_id or not self.label_independent:
            return "SIDE_IDENTITY_OR_PROVENANCE_UNVERIFIED"
        if self.state == "MISSING":
            return "SIDE_AUTHORITY_EVIDENCE_MISSING"
        if self.state == "UNVERIFIED_IDENTITY":
            return "SIDE_IDENTITY_OR_PROVENANCE_UNVERIFIED"
        if self.wrist_xy is None or not np.isfinite(self.wrist_xy).all():
            return "SIDE_AUTHORITY_COORDINATE_INVALID"
        height, width = image_shape
        x, y = self.wrist_xy
        inside = 0.0 <= x < width and 0.0 <= y < height
        if self.state == "AVAILABLE" and not inside:
            raise DistributedSideEvidenceError("AVAILABLE authority wrist is outside image")
        if self.state == "OUTSIDE_IMAGE" and inside:
            raise DistributedSideEvidenceError("OUTSIDE_IMAGE authority wrist is inside image")
        if self.state not in ("AVAILABLE", "OUTSIDE_IMAGE"):
            raise DistributedSideEvidenceError(f"unknown authority state: {self.state!r}")
        # OUTSIDE_IMAGE is diagnostic-only by archive/legacy/task_cards/26 S2.
        return None

    @property
    def wrist_outside_image_diagnostic(self) -> bool:
        return self.state == "OUTSIDE_IMAGE"


@dataclass(frozen=True)
class JointSupportEvidence:
    in_image_joint_count: int
    supported_in_image_joint_count: int
    joint_support_ratio: float
    support_radius_px: float
    wrist_supported_diagnostic: bool

    def validate(self) -> None:
        if not 0 <= self.in_image_joint_count <= 21:
            raise DistributedSideEvidenceError("in-image joint count outside [0,21]")
        if not 0 <= self.supported_in_image_joint_count <= self.in_image_joint_count:
            raise DistributedSideEvidenceError("supported joint count is inconsistent")
        expected = (
            self.supported_in_image_joint_count / self.in_image_joint_count
            if self.in_image_joint_count
            else 0.0
        )
        if not np.isclose(self.joint_support_ratio, expected, rtol=0.0, atol=1e-12):
            raise DistributedSideEvidenceError("joint support ratio/count mismatch")
        if not np.isfinite(self.support_radius_px) or self.support_radius_px < 1.0:
            raise DistributedSideEvidenceError("invalid support radius")


def point_supported(mask: np.ndarray, point_xy: Sequence[float], radius_px: float) -> bool:
    """Test raw-mask support in a disk without creating or changing pixels."""

    binary = np.asarray(mask)
    if binary.ndim != 2 or binary.dtype != np.bool_:
        raise DistributedSideEvidenceError("raw mask must be 2D bool")
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return False
    radius = max(float(radius_px), 1.0)
    height, width = binary.shape
    x, y = point
    x0 = max(int(np.floor(x - radius)), 0)
    x1 = min(int(np.ceil(x + radius)) + 1, width)
    y0 = max(int(np.floor(y - radius)), 0)
    y1 = min(int(np.ceil(y + radius)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return bool(binary[y0:y1, x0:x1][disk].any())


def measure_in_image_joint_support(
    mask: np.ndarray,
    joints_xy: np.ndarray,
    *,
    frozen_hand_scale_px: float,
) -> JointSupportEvidence:
    """Measure support using only finite joints inside the raw image domain.

    The hand scale remains the frozen v3 diagnostic value so S1/S2 are the
    only semantic changes.  The scale does not route or accept a side; it only
    preserves the existing point-support radius.
    """

    binary = np.asarray(mask)
    joints = np.asarray(joints_xy, dtype=np.float64)
    if binary.ndim != 2 or binary.dtype != np.bool_:
        raise DistributedSideEvidenceError("raw mask must be 2D bool")
    if joints.shape != (21, 2):
        raise DistributedSideEvidenceError("HaWoR joints must have shape [21,2]")
    if not np.isfinite(frozen_hand_scale_px) or frozen_hand_scale_px <= 1.0:
        raise DistributedSideEvidenceError("frozen hand scale is invalid")
    height, width = binary.shape
    finite = np.isfinite(joints).all(axis=1)
    inside = (
        finite
        & (joints[:, 0] >= 0.0)
        & (joints[:, 0] < width)
        & (joints[:, 1] >= 0.0)
        & (joints[:, 1] < height)
    )
    radius = max(0.055 * float(frozen_hand_scale_px), 1.0)
    supported = sum(point_supported(binary, point, radius) for point in joints[inside])
    wrist_supported = point_supported(
        binary,
        joints[0],
        max(0.12 * float(frozen_hand_scale_px), 1.0),
    )
    result = JointSupportEvidence(
        int(inside.sum()),
        int(supported),
        float(supported / inside.sum()) if inside.any() else 0.0,
        radius,
        wrist_supported,
    )
    result.validate()
    return result


def mask_sha256(mask: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(mask, dtype=np.bool_)
    header = f"{contiguous.shape[0]}x{contiguous.shape[1]}:bool:".encode("ascii")
    return sha256_bytes(header + contiguous.tobytes())


@dataclass(frozen=True)
class RawInstance:
    session_id: str
    frame_index: int
    raw_instance_offset: int
    instance_id: int
    raw_source: EvidenceRef
    mask: np.ndarray
    score: float
    support_by_side: Mapping[Side, JointSupportEvidence]
    boundary_supported_diagnostic: Mapping[Side, bool]
    object6d_overlap_over_instance: float
    area_ratio: float


@dataclass(frozen=True)
class RoutingAudit:
    raw_instance_offset: int
    instance_id: int
    assigned_pool: Side | None
    routing_reason: str
    left_joint_support_ratio: float
    right_joint_support_ratio: float


@dataclass(frozen=True)
class CandidateAudit:
    raw_source_sha256: str
    raw_instance_offset: int
    instance_id: int
    assigned_pool: Side | None
    considered_for_side: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]
    in_image_joint_count: int
    supported_in_image_joint_count: int
    joint_support_ratio: float
    non_target_joint_support_ratio: float
    side_difference: float
    wrist_supported_diagnostic: bool
    authority_wrist_outside_image_diagnostic: bool
    boundary_supported_diagnostic: bool
    object6d_overlap_over_instance: float


@dataclass(frozen=True)
class SideDecision:
    identity: Side
    authority_lineage_id: str
    authority_state: AuthorityState
    status: DecisionStatus
    raw_instance_offset: int | None
    instance_id: int | None
    mask: np.ndarray | None
    failure_reason: str | None
    authority_wrist_outside_image_diagnostic: bool
    candidate_audit: tuple[CandidateAudit, ...]


@dataclass(frozen=True)
class FrameSelection:
    session_id: str
    frame_index: int
    left: SideDecision
    right: SideDecision
    frame_complete: bool
    frame_status: Literal["COMPLETE", "INCOMPLETE", "PARTIAL_SIDE_HOLD", "IDENTITY_HOLD"]
    routing_audit: tuple[RoutingAudit, ...]
    pixels_created_or_edited: int = 0
    union_operations: int = 0
    crop_operations: int = 0
    fill_operations: int = 0
    morphology_operations: int = 0
    object6d_subtraction_operations: int = 0


@dataclass(frozen=True)
class WrongSideAcceptanceCheck:
    session_id: str
    frame_index: int
    raw_instance_offset: int
    expected_side: Side
    observed_accepted_sides: tuple[Side, ...]
    status: Literal["PASS", "FAIL_WRONG_SIDE_ACCEPTANCE"]


def _validate_raw(raw: RawInstance, *, evidence_root: Path) -> np.ndarray:
    if raw.raw_instance_offset < 0 or raw.instance_id < 0:
        raise DistributedSideEvidenceError("negative raw instance identity")
    if set(raw.support_by_side) != set(SIDES):
        raise DistributedSideEvidenceError("raw instance requires exact left/right support")
    if set(raw.boundary_supported_diagnostic) != set(SIDES):
        raise DistributedSideEvidenceError("raw instance requires exact boundary diagnostics")
    mask = np.asarray(raw.mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise DistributedSideEvidenceError("raw mask must be 2D bool")
    if raw.raw_source.source_kind != "SAM_RAW_INSTANCE":
        raise DistributedSideEvidenceError("raw source kind mismatch")
    data = raw.raw_source.read_verified(allowed_root=evidence_root)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributedSideEvidenceError("raw source is not JSON") from exc
    expected = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": f"{raw.session_id}:{raw.frame_index}",
        "instance_id": raw.instance_id,
        "mask_sha256": mask_sha256(mask),
    }
    if payload != expected:
        raise DistributedSideEvidenceError("raw source/mask payload mismatch")
    for evidence in raw.support_by_side.values():
        evidence.validate()
    scalars = (raw.score, raw.object6d_overlap_over_instance, raw.area_ratio)
    if not all(np.isfinite(value) for value in scalars):
        raise DistributedSideEvidenceError("non-finite raw instance scalar")
    if not 0.0 <= raw.object6d_overlap_over_instance <= 1.0:
        raise DistributedSideEvidenceError("Object6D overlap outside [0,1]")
    if not 0.0 <= raw.area_ratio <= 1.0:
        raise DistributedSideEvidenceError("area ratio outside [0,1]")
    return mask


def validate_raw_instance(raw: RawInstance, *, evidence_root: Path) -> np.ndarray:
    """Revalidate a raw instance at the point where its pixels are consumed.

    Selection extensions call this immediately before copying an accepted raw
    mask.  The second check deliberately binds the copied bytes to the same
    ``SAM_RAW_INSTANCE`` record that was validated at frame admission.
    """

    return _validate_raw(raw, evidence_root=evidence_root)


def route_raw_instance(raw: RawInstance) -> RoutingAudit:
    """Route only by the in-image joint-support distribution; wrist is absent."""

    left = raw.support_by_side["left"].joint_support_ratio
    right = raw.support_by_side["right"].joint_support_ratio
    if left == right:
        assigned: Side | None = None
        reason = "NO_SIDE_SUPPORT" if left == 0.0 else "TIED_IN_IMAGE_JOINT_SUPPORT"
    elif left > right:
        assigned = "left"
        reason = "LEFT_MAX_IN_IMAGE_JOINT_SUPPORT"
    else:
        assigned = "right"
        reason = "RIGHT_MAX_IN_IMAGE_JOINT_SUPPORT"
    return RoutingAudit(
        raw.raw_instance_offset,
        raw.instance_id,
        assigned,
        reason,
        left,
        right,
    )


def _rejection_reasons(
    own: JointSupportEvidence,
    other: JointSupportEvidence,
    object_overlap: float,
    thresholds: SelectorThresholds,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if own.joint_support_ratio < thresholds.min_joint_support_ratio:
        reasons.append("INSUFFICIENT_IN_IMAGE_JOINT_SUPPORT")
    if own.joint_support_ratio < other.joint_support_ratio + thresholds.min_side_margin:
        reasons.append("NOT_SIDE_SPECIFIC_IN_IMAGE_JOINT_DISTRIBUTION")
    if object_overlap > thresholds.max_object_overlap_over_instance:
        reasons.append("OBJECT6D_PROTECTION_REJECTION")
    return tuple(reasons)


def _rank(raw: RawInstance, side: Side) -> tuple[float, ...]:
    other: Side = "right" if side == "left" else "left"
    own_ratio = raw.support_by_side[side].joint_support_ratio
    other_ratio = raw.support_by_side[other].joint_support_ratio
    return (
        own_ratio,
        own_ratio - other_ratio,
        raw.score,
        -raw.object6d_overlap_over_instance,
        -raw.area_ratio,
        -float(raw.raw_instance_offset),
    )


def _failure_reason(
    side: Side,
    raws: Sequence[RawInstance],
    routing: Sequence[RoutingAudit],
    audits: Sequence[CandidateAudit],
) -> str:
    if not raws:
        return "NO_RAW_INSTANCE_IN_SIDE_POOL"
    own_support = [raw.support_by_side[side].joint_support_ratio for raw in raws]
    if max(own_support, default=0.0) == 0.0:
        return "NO_INSTANCE_SUPPORTS_SIDE"
    assigned_to_other = any(
        route.assigned_pool not in (None, side)
        and raws[index].support_by_side[side].joint_support_ratio
        >= MIN_JOINT_SUPPORT_RATIO
        for index, route in enumerate(routing)
    )
    considered = [audit for audit in audits if audit.considered_for_side]
    if assigned_to_other and not considered:
        return "AUTHORITY_SIDE_ALIAS"
    reason_sets = [set(audit.rejection_reasons) for audit in considered]
    if reason_sets and all(
        reasons == {"INSUFFICIENT_IN_IMAGE_JOINT_SUPPORT"} for reasons in reason_sets
    ):
        return "WEAK_IN_IMAGE_JOINT_SUPPORT"
    if reason_sets and all(
        "NOT_SIDE_SPECIFIC_IN_IMAGE_JOINT_DISTRIBUTION" in reasons
        for reasons in reason_sets
    ):
        return "AUTHORITY_SIDE_ALIAS"
    if reason_sets and all(reasons == {"OBJECT6D_PROTECTION_REJECTION"} for reasons in reason_sets):
        return "OBJECT6D_PROTECTION_REJECTION"
    if assigned_to_other:
        return "AUTHORITY_SIDE_ALIAS_OR_MULTIPLE_NUMERIC_REJECTIONS"
    return "MULTIPLE_NUMERIC_REJECTIONS"


def _hold_decision(authority: SideAuthority, reason: str) -> SideDecision:
    return SideDecision(
        authority.identity,
        authority.lineage_id,
        authority.state,
        "HOLD",
        None,
        None,
        None,
        reason,
        authority.wrist_outside_image_diagnostic,
        (),
    )


def select_side(
    authority: SideAuthority,
    raws: Sequence[RawInstance],
    routing: Sequence[RoutingAudit],
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    thresholds: SelectorThresholds,
) -> SideDecision:
    """Select one side independently from the shared immutable raw inventory."""

    hold = authority.validate(evidence_root=evidence_root, image_shape=image_shape)
    if hold is not None:
        return _hold_decision(authority, hold)
    side = authority.identity
    other: Side = "right" if side == "left" else "left"
    audits: list[CandidateAudit] = []
    eligible: list[RawInstance] = []
    for raw, route in zip(raws, routing, strict=True):
        own = raw.support_by_side[side]
        other_support = raw.support_by_side[other]
        considered = route.assigned_pool == side
        if considered:
            reasons = _rejection_reasons(
                own,
                other_support,
                raw.object6d_overlap_over_instance,
                thresholds,
            )
        elif route.assigned_pool is None:
            reasons = ("UNASSIGNED_BY_IN_IMAGE_JOINT_DISTRIBUTION",)
        else:
            reasons = (f"ROUTED_TO_{route.assigned_pool.upper()}_BY_IN_IMAGE_JOINT_DISTRIBUTION",)
        audit = CandidateAudit(
            raw.raw_source.sha256,
            raw.raw_instance_offset,
            raw.instance_id,
            route.assigned_pool,
            considered,
            considered and not reasons,
            reasons,
            own.in_image_joint_count,
            own.supported_in_image_joint_count,
            own.joint_support_ratio,
            other_support.joint_support_ratio,
            own.joint_support_ratio - other_support.joint_support_ratio,
            own.wrist_supported_diagnostic,
            authority.wrist_outside_image_diagnostic,
            bool(raw.boundary_supported_diagnostic[side]),
            raw.object6d_overlap_over_instance,
        )
        audits.append(audit)
        if audit.eligible:
            eligible.append(raw)
    if not eligible:
        return SideDecision(
            side,
            authority.lineage_id,
            authority.state,
            "REJECT",
            None,
            None,
            None,
            _failure_reason(side, raws, routing, audits),
            authority.wrist_outside_image_diagnostic,
            tuple(audits),
        )
    selected = max(eligible, key=lambda raw: _rank(raw, side))
    return SideDecision(
        side,
        authority.lineage_id,
        authority.state,
        "ACCEPT",
        selected.raw_instance_offset,
        selected.instance_id,
        np.asarray(selected.mask, dtype=np.bool_).copy(),
        None,
        authority.wrist_outside_image_diagnostic,
        tuple(audits),
    )


def select_frame(
    authorities: Mapping[Side, SideAuthority],
    raws: Sequence[RawInstance],
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    thresholds: SelectorThresholds = SelectorThresholds(),
) -> FrameSelection:
    """Select left/right independently; never globally abort on one wrist."""

    thresholds.validate()
    if set(authorities) != set(SIDES):
        raise DistributedSideEvidenceError("exact left/right authorities required")
    left_authority = authorities["left"]
    right_authority = authorities["right"]
    if left_authority.identity != "left" or right_authority.identity != "right":
        raise DistributedSideEvidenceError("authority mapping identity mismatch")
    if (
        left_authority.session_id != right_authority.session_id
        or left_authority.frame_index != right_authority.frame_index
    ):
        raise DistributedSideEvidenceError("left/right authority frame mismatch")
    if left_authority.lineage_id == right_authority.lineage_id:
        left = _hold_decision(left_authority, "SIDE_IDENTITY_LINEAGE_ALIAS")
        right = _hold_decision(right_authority, "SIDE_IDENTITY_LINEAGE_ALIAS")
        return FrameSelection(
            left_authority.session_id,
            left_authority.frame_index,
            left,
            right,
            False,
            "IDENTITY_HOLD",
            (),
        )

    observed_ids: set[tuple[int, int, str]] = set()
    for raw in raws:
        if raw.session_id != left_authority.session_id or raw.frame_index != left_authority.frame_index:
            raise DistributedSideEvidenceError("raw instance frame mismatch")
        _validate_raw(raw, evidence_root=evidence_root)
        identity = (raw.raw_instance_offset, raw.instance_id, raw.raw_source.sha256)
        if identity in observed_ids:
            raise DistributedSideEvidenceError("duplicate raw instance identity")
        observed_ids.add(identity)
    routing = tuple(route_raw_instance(raw) for raw in raws)
    left = select_side(
        left_authority,
        raws,
        routing,
        evidence_root=evidence_root,
        image_shape=image_shape,
        thresholds=thresholds,
    )
    right = select_side(
        right_authority,
        raws,
        routing,
        evidence_root=evidence_root,
        image_shape=image_shape,
        thresholds=thresholds,
    )
    complete = left.status == "ACCEPT" and right.status == "ACCEPT"
    if complete:
        frame_status: Literal[
            "COMPLETE", "INCOMPLETE", "PARTIAL_SIDE_HOLD", "IDENTITY_HOLD"
        ] = "COMPLETE"
    elif left.status == "HOLD" or right.status == "HOLD":
        frame_status = "PARTIAL_SIDE_HOLD"
    else:
        frame_status = "INCOMPLETE"
    return FrameSelection(
        left_authority.session_id,
        left_authority.frame_index,
        left,
        right,
        complete,
        frame_status,
        routing,
    )


@dataclass(frozen=True)
class WristReviewSlice:
    pixels: np.ndarray
    anchor_xy: tuple[int, int]
    anchor_kind: Literal["AUTHORITY_WRIST_CENTER", "NEAREST_VALID_IMAGE_EDGE"]
    original_wrist_xy: tuple[float, float]
    authority_state: AuthorityState
    algorithm_consumed: Literal[False] = False
    authority_wrist_repaired: Literal[False] = False


def extract_wrist_review_slice(
    mask: np.ndarray,
    wrist_xy: tuple[float, float],
    *,
    authority_state: AuthorityState,
    size: int = 41,
) -> WristReviewSlice:
    """Create an exact padded review crop; it is never selector input."""

    binary = np.asarray(mask)
    if binary.ndim != 2 or binary.dtype != np.bool_:
        raise DistributedSideEvidenceError("review mask must be 2D bool")
    if size != 41:
        raise DistributedSideEvidenceError("archive/legacy/task_cards/26 review slice must be exactly 41x41")
    if not np.isfinite(wrist_xy).all():
        raise DistributedSideEvidenceError("review wrist must be finite")
    height, width = binary.shape
    x, y = wrist_xy
    inside = 0.0 <= x < width and 0.0 <= y < height
    if authority_state == "AVAILABLE" and not inside:
        raise DistributedSideEvidenceError("AVAILABLE review wrist outside image")
    if authority_state == "OUTSIDE_IMAGE" and inside:
        raise DistributedSideEvidenceError("OUTSIDE_IMAGE review wrist inside image")
    if authority_state not in ("AVAILABLE", "OUTSIDE_IMAGE"):
        raise DistributedSideEvidenceError("review requires an observed finite wrist")
    anchor_x = min(max(int(np.rint(x)), 0), width - 1)
    anchor_y = min(max(int(np.rint(y)), 0), height - 1)
    half = size // 2
    output = np.zeros((size, size), dtype=np.bool_)
    source_x0 = max(anchor_x - half, 0)
    source_y0 = max(anchor_y - half, 0)
    source_x1 = min(anchor_x + half + 1, width)
    source_y1 = min(anchor_y + half + 1, height)
    target_x0 = source_x0 - (anchor_x - half)
    target_y0 = source_y0 - (anchor_y - half)
    output[
        target_y0 : target_y0 + (source_y1 - source_y0),
        target_x0 : target_x0 + (source_x1 - source_x0),
    ] = binary[source_y0:source_y1, source_x0:source_x1]
    return WristReviewSlice(
        output,
        (anchor_x, anchor_y),
        "AUTHORITY_WRIST_CENTER" if authority_state == "AVAILABLE" else "NEAREST_VALID_IMAGE_EDGE",
        (float(x), float(y)),
        authority_state,
    )


def summarize_frames(frames: Sequence[FrameSelection]) -> dict[str, Any]:
    """Return both required frame-level and side-level denominators."""

    session_ids = sorted({frame.session_id for frame in frames})
    per_session: dict[str, Any] = {}
    for session_id in session_ids:
        subset = [frame for frame in frames if frame.session_id == session_id]
        per_session[session_id] = {
            "frame_passed": sum(frame.frame_complete for frame in subset),
            "frame_total": len(subset),
            "left_passed": sum(frame.left.status == "ACCEPT" for frame in subset),
            "right_passed": sum(frame.right.status == "ACCEPT" for frame in subset),
            "side_passed": sum(
                frame.left.status == "ACCEPT" for frame in subset
            )
            + sum(frame.right.status == "ACCEPT" for frame in subset),
            "side_total": 2 * len(subset),
        }
    return {
        "frame_passed": sum(frame.frame_complete for frame in frames),
        "frame_total": len(frames),
        "left_passed": sum(frame.left.status == "ACCEPT" for frame in frames),
        "right_passed": sum(frame.right.status == "ACCEPT" for frame in frames),
        "side_passed": sum(frame.left.status == "ACCEPT" for frame in frames)
        + sum(frame.right.status == "ACCEPT" for frame in frames),
        "side_total": 2 * len(frames),
        "per_session": per_session,
    }


def check_wrong_side_acceptance(
    frames: Sequence[FrameSelection],
    *,
    sentinels: Sequence[tuple[str, int, int, Side]] = WRONG_SIDE_SENTINELS,
) -> tuple[WrongSideAcceptanceCheck, ...]:
    """Hard-fail a sentinel accepted for the wrong physical side.

    This audit consumes only the selected raw-instance offsets.  In
    particular, it has no wrist-supported or authority-state input, so the
    archive/legacy/task_cards/26 diagnostic downgrades cannot weaken wrong-side protection.
    """

    by_key = {(frame.session_id, frame.frame_index): frame for frame in frames}
    checks: list[WrongSideAcceptanceCheck] = []
    for session_id, frame_index, offset, expected_side in sentinels:
        try:
            frame = by_key[(session_id, frame_index)]
        except KeyError as exc:
            raise DistributedSideEvidenceError("wrong-side sentinel frame missing") from exc
        accepted: list[Side] = []
        if frame.left.status == "ACCEPT" and frame.left.raw_instance_offset == offset:
            accepted.append("left")
        if frame.right.status == "ACCEPT" and frame.right.raw_instance_offset == offset:
            accepted.append("right")
        wrong = any(side != expected_side for side in accepted)
        checks.append(
            WrongSideAcceptanceCheck(
                session_id,
                frame_index,
                offset,
                expected_side,
                tuple(accepted),
                "FAIL_WRONG_SIDE_ACCEPTANCE" if wrong else "PASS",
            )
        )
    return tuple(checks)


def resolve_cpu_preflight_status(
    *,
    wrong_side_failed: bool,
    development_frame_passed: int,
    preregistered_arithmetic_matches: bool,
) -> str:
    """Resolve disposition with wrong-side acceptance as the highest priority."""

    if wrong_side_failed:
        return "FAIL_WRONG_SIDE_ACCEPTANCE"
    if development_frame_passed < 8:
        return "HOLD_DEVELOPMENT_STOP_RULE"
    if not preregistered_arithmetic_matches:
        return "HOLD_PREREGISTERED_ARITHMETIC_MISMATCH"
    return "CPU_PREFLIGHT_MATCHES_PREREGISTRATION_AWAITING_INDEPENDENT_QA_GPU_BLOCKED"


def validate_input_covariate_freeze(record: Mapping[str, Any]) -> None:
    """Validate the pre-replay, numeric-only 39-frame covariate freeze."""

    required_top = {
        "schema_version",
        "document_status",
        "task26_sha256",
        "created_before_selector_replay",
        "analysis_performed",
        "frame_count",
        "frames",
    }
    if set(record) != required_top:
        raise DistributedSideEvidenceError("input covariate freeze top-level schema drift")
    expected_header = {
        "schema_version": "a-prime-input-covariates-freeze-v1",
        "document_status": "FROZEN_PURE_INPUT_NUMBERS_NO_ANALYSIS",
        "task26_sha256": TASK26_SHA256,
        "created_before_selector_replay": True,
        "analysis_performed": False,
        "frame_count": 39,
    }
    for key, expected in expected_header.items():
        if record.get(key) != expected:
            raise DistributedSideEvidenceError(f"input covariate freeze drift: {key}")
    frames = record["frames"]
    if not isinstance(frames, list) or len(frames) != 39:
        raise DistributedSideEvidenceError("input covariate freeze must contain 39 frames")
    expected_keys = {
        (session_id, frame_index)
        for session_id, indices in FIXED_FRAME_INDICES.items()
        for frame_index in indices
    }
    observed_keys: set[tuple[str, int]] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise DistributedSideEvidenceError("input covariate frame must be an object")
        required_frame = {
            "session_id",
            "frame_index",
            "joint_source",
            "left",
            "right",
            "wrist_pixel_distance",
            "left_over_right_hand_scale_ratio",
        }
        if set(frame) != required_frame:
            raise DistributedSideEvidenceError("input covariate frame schema drift")
        key = (frame["session_id"], frame["frame_index"])
        if key in observed_keys:
            raise DistributedSideEvidenceError("duplicate input covariate frame")
        observed_keys.add(key)
        source = frame["joint_source"]
        if not isinstance(source, Mapping) or set(source) != {"path", "bytes", "sha256"}:
            raise DistributedSideEvidenceError("input covariate source ref schema drift")
        if (
            not isinstance(source["path"], str)
            or type(source["bytes"]) is not int
            or source["bytes"] <= 0
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
        ):
            raise DistributedSideEvidenceError("input covariate source ref invalid")
        for side in SIDES:
            value = frame[side]
            if not isinstance(value, Mapping) or set(value) != {
                "source_track_index",
                "validity_state",
                "wrist_xy",
                "hand_scale_pixels",
            }:
                raise DistributedSideEvidenceError("input covariate side schema drift")
            if value["validity_state"] not in {
                "VALID_IN_IMAGE_WRIST",
                "VALID_OUTSIDE_IMAGE_WRIST",
                "MISSING_TRACK",
                "INVALID_TRACK",
                "NONFINITE_JOINTS",
            }:
                raise DistributedSideEvidenceError("input covariate validity state invalid")
            scale = value["hand_scale_pixels"]
            if scale is not None and (not np.isfinite(scale) or scale <= 1.0):
                raise DistributedSideEvidenceError("input covariate hand scale invalid")
        for numeric_key in ("wrist_pixel_distance", "left_over_right_hand_scale_ratio"):
            value = frame[numeric_key]
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise DistributedSideEvidenceError(f"input covariate invalid: {numeric_key}")
    if observed_keys != expected_keys:
        raise DistributedSideEvidenceError("input covariate frame set drift")


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


def validate_cpu_config(record: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "a-prime-distributed-side-cpu-config-v1",
        "route_of_evidence": ROUTE_OF_EVIDENCE,
        "producer_id": PRODUCER_ID,
        "prompt": TEXT_PROMPT,
        "wrist_supported_role": "DIAGNOSTIC_ONLY",
        "authority_wrist_outside_image_role": "DIAGNOSTIC_ONLY",
        "side_routing_basis": "IN_IMAGE_HAWOR_JOINT_SUPPORT_DISTRIBUTION_ONLY",
        "min_joint_support_ratio": MIN_JOINT_SUPPORT_RATIO,
        "min_side_margin": MIN_SIDE_MARGIN,
        "max_object_overlap_over_instance": MAX_OBJECT_OVERLAP_OVER_INSTANCE,
        "boundary_supported_role": "DIAGNOSTIC_ONLY",
        "object6d_role": "REJECTION_PROTECTION_ONLY",
        "raw_pixel_semantics": "COPY_ONE_UNMODIFIED_RAW_SAM_INSTANCE_PER_ACCEPTED_SIDE",
        "forbidden_pixel_operations": [
            "union",
            "fill",
            "crop",
            "hard_crop",
            "morphology",
            "object6d_subtraction",
        ],
        "fixed_frame_indices": {
            session: list(indices) for session, indices in FIXED_FRAME_INDICES.items()
        },
        "forbidden_session": FORBIDDEN_SESSION,
    }
    if dict(record) != expected:
        raise DistributedSideEvidenceError("CPU config is not the exact archive/legacy/task_cards/26 freeze")
    forbidden = sorted(_find_forbidden_keys(record))
    if forbidden:
        raise DistributedSideEvidenceError(f"CPU config forbidden override keys: {forbidden}")
    SelectorThresholds(
        record["min_joint_support_ratio"],
        record["min_side_margin"],
        record["max_object_overlap_over_instance"],
    ).validate()


def validate_cpu_governance(record: Mapping[str, Any]) -> None:
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
        "auditor_outputs_read": 0,
    }
    if dict(record) != expected:
        raise DistributedSideEvidenceError("CPU governance drift")


def verify_task26(project_root: Path) -> EvidenceRef:
    path = project_root / TASK26_RELATIVE_PATH
    data = path.read_bytes()
    if sha256_bytes(data) != TASK26_SHA256:
        raise DistributedSideEvidenceError("archive/legacy/task_cards/26 SHA drift")
    return EvidenceRef(str(path), len(data), TASK26_SHA256, "TASK26_FROZEN_CARD")


def claim_cpu_audit_root(
    output_root: Path,
    *,
    project_root: Path,
    freeze_payload: Mapping[str, Any],
) -> EvidenceRef:
    """Atomically claim a direct-child archive/audits directory and owner marker."""

    verify_task26(project_root)
    audit_parent = (project_root / "archive/audits").resolve(strict=True)
    if output_root.parent.resolve(strict=True) != audit_parent:
        raise DistributedSideEvidenceError("CPU audit root must be a direct child of archive/audits/")
    if not output_root.name.startswith("lr_distributed_side_evidence_cpu_"):
        raise DistributedSideEvidenceError("CPU audit root is outside task namespace")
    freeze_bytes = canonical_json(freeze_payload)
    try:
        os.mkdir(output_root, 0o750)
    except FileExistsError as exc:
        raise DistributedSideEvidenceError("CPU audit root already exists; O_EXCL required") from exc
    owner_payload = canonical_json(
        {
            "schema_version": "a-prime-distributed-side-cpu-owner-v1",
            "route_of_evidence": ROUTE_OF_EVIDENCE,
            "task26_sha256": TASK26_SHA256,
            "freeze_payload_sha256": sha256_bytes(freeze_bytes),
            "cpu_only": True,
            "gpu_execution_authorized": False,
        }
    )
    owner_path = output_root / "OWNER.json"
    exclusive_bytes(owner_path, owner_payload)
    return EvidenceRef(
        str(owner_path),
        len(owner_payload),
        sha256_bytes(owner_payload),
        "A_PRIME_DISTRIBUTED_SIDE_CPU_OWNER",
    )


def require_gpu_execution() -> None:
    """Fail closed until a future explicit owner release is separately implemented."""

    raise DistributedSideEvidenceError("GPU execution is not authorized by CPU preflight")


SelectorThresholds().validate()
