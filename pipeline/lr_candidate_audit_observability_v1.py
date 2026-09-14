"""Pure observability wrapper for A-prime distributed-side selection.

This module does not implement a selector. It calls the pinned selector once, returns
its exact decision object, and constructs a separately serializable rejection audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pipeline import lr_distributed_side_evidence as selector


SCHEMA_VERSION = "a-prime-candidate-audit-observability-v1"
PINNED_SELECTOR_SHA256 = "bca51f866de925a2e795c5257540abc9c06fb8d298f26dd049196fda14fb9ff3"
SIDES = ("left", "right")
Stage = Literal[
    "NOT_SCORED_BEFORE_CANDIDATE_ENUMERATION",
    "CANDIDATES_ENUMERATED",
]


class CandidateAuditObservabilityError(RuntimeError):
    """Raised for observability schema, path, or selector-pin drift."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _mask_identity(mask: np.ndarray | None) -> str | None:
    return None if mask is None else selector.mask_sha256(np.asarray(mask, dtype=np.bool_))


def decision_projection(value: selector.FrameSelection) -> dict[str, Any]:
    """Projection of every decision/pixel-operation bit, independent of D3 audit."""

    sides: dict[str, Any] = {}
    for side in SIDES:
        decision = getattr(value, side)
        sides[side] = {
            "identity": decision.identity,
            "authority_lineage_id": decision.authority_lineage_id,
            "authority_state": decision.authority_state,
            "status": decision.status,
            "raw_instance_offset": decision.raw_instance_offset,
            "instance_id": decision.instance_id,
            "mask_sha256": _mask_identity(decision.mask),
            "failure_reason": decision.failure_reason,
            "authority_wrist_outside_image_diagnostic": decision.authority_wrist_outside_image_diagnostic,
            "candidate_audit": [asdict(item) for item in decision.candidate_audit],
        }
    return {
        "session_id": value.session_id,
        "frame_index": value.frame_index,
        "sides": sides,
        "frame_complete": value.frame_complete,
        "frame_status": value.frame_status,
        "routing_audit": [asdict(item) for item in value.routing_audit],
        "pixels_created_or_edited": value.pixels_created_or_edited,
        "union_operations": value.union_operations,
        "crop_operations": value.crop_operations,
        "fill_operations": value.fill_operations,
        "morphology_operations": value.morphology_operations,
        "object6d_subtraction_operations": value.object6d_subtraction_operations,
    }


def decision_sha256(value: selector.FrameSelection) -> str:
    return hashlib.sha256(canonical_json(decision_projection(value))).hexdigest()


def _side_candidate_record(
    raw: selector.RawInstance,
    side: str,
    decision: selector.SideDecision,
) -> dict[str, Any]:
    other = "right" if side == "left" else "left"
    own = raw.support_by_side[side]
    non_target = raw.support_by_side[other]
    matches = [
        item for item in decision.candidate_audit
        if item.raw_instance_offset == raw.raw_instance_offset
        and item.raw_source_sha256 == raw.raw_source.sha256
    ]
    if len(matches) > 1:
        raise CandidateAuditObservabilityError("duplicate selector candidate audit identity")
    if matches:
        item = matches[0]
        veto = list(item.rejection_reasons)
        if decision.raw_instance_offset == raw.raw_instance_offset:
            disposition = "SELECTED"
        elif item.eligible:
            disposition = "ELIGIBLE_NOT_SELECTED_BY_FROZEN_RANK"
            veto = ["NOT_SELECTED_BY_FROZEN_RANK"]
        else:
            disposition = "REJECTED_OR_ROUTED_AWAY"
    elif decision.status == "HOLD":
        disposition = "SIDE_PRECHECK_HOLD_AFTER_ENUMERATION"
        veto = [f"SIDE_PRECHECK_HOLD:{decision.failure_reason}"]
    else:
        raise CandidateAuditObservabilityError("enumerated raw lacks selector audit")
    return {
        "side": side,
        "joint_support_ratio": own.joint_support_ratio,
        "non_target_joint_support_ratio": non_target.joint_support_ratio,
        "side_margin": own.joint_support_ratio - non_target.joint_support_ratio,
        "in_image_joint_count": own.in_image_joint_count,
        "supported_in_image_joint_count": own.supported_in_image_joint_count,
        "wrist_supported_diagnostic": own.wrist_supported_diagnostic,
        "authority_wrist_outside_image_diagnostic": decision.authority_wrist_outside_image_diagnostic,
        "boundary_supported_diagnostic": bool(raw.boundary_supported_diagnostic[side]),
        "object6d_overlap_over_instance": raw.object6d_overlap_over_instance,
        "final_disposition": disposition,
        "final_veto_items": veto,
    }


def build_enumerated_audit(
    selection: selector.FrameSelection,
    raws: Sequence[selector.RawInstance],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in raws:
        records.append({
            "raw_instance_offset": raw.raw_instance_offset,
            "instance_id": raw.instance_id,
            "raw_source_ref": asdict(raw.raw_source),
            "raw_mask_sha256": selector.mask_sha256(raw.mask),
            "score": raw.score,
            "area_ratio": raw.area_ratio,
            "object6d_overlap_over_instance": raw.object6d_overlap_over_instance,
            "side_audit": {
                side: _side_candidate_record(raw, side, getattr(selection, side))
                for side in SIDES
            },
        })
    return _base_record(
        selection.session_id,
        selection.frame_index,
        stage="CANDIDATES_ENUMERATED",
        abort_reason=None,
        selection=selection,
        candidate_records=records,
    )


def build_pre_enumeration_record(
    session_id: str,
    frame_index: int,
    *,
    abort_reason: str,
    selection: selector.FrameSelection | None = None,
) -> dict[str, Any]:
    if not abort_reason:
        raise CandidateAuditObservabilityError("pre-enumeration abort reason required")
    return _base_record(
        session_id,
        frame_index,
        stage="NOT_SCORED_BEFORE_CANDIDATE_ENUMERATION",
        abort_reason=abort_reason,
        selection=selection,
        candidate_records=[],
    )


def _base_record(
    session_id: str,
    frame_index: int,
    *,
    stage: Stage,
    abort_reason: str | None,
    selection: selector.FrameSelection | None,
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "AUTH_TIER": "T0",
        "WHY_NOT_BLOCKED": "CPU-only pure observability over already enumerated A-prime evidence",
        "stage": stage,
        "session_id": session_id,
        "frame_index": frame_index,
        "abort_reason": abort_reason,
        "candidate_count": len(candidate_records),
        "candidate_audit": candidate_records,
        "selection_decision_sha256": None if selection is None else decision_sha256(selection),
        "selector_sha256": PINNED_SELECTOR_SHA256,
        "access": {
            "labels_read": 0,
            "blind_scores_read": 0,
            "grap_a_cap_025_frames_read": 0,
            "gpu_calls": 0,
            "mask_pixels_modified": 0,
        },
        "semantic_changes": {
            "prompt": 0,
            "threshold": 0,
            "selection": 0,
            "pixels": 0,
            "status": 0,
        },
    }


@dataclass(frozen=True)
class ObservedSelection:
    selection: selector.FrameSelection
    audit: Mapping[str, Any]


def select_frame_observed(
    authorities: Mapping[str, selector.SideAuthority],
    raws: Sequence[selector.RawInstance],
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    selector_path: Path,
    thresholds: selector.SelectorThresholds = selector.SelectorThresholds(),
) -> ObservedSelection:
    if sha256_file(selector_path) != PINNED_SELECTOR_SHA256:
        raise CandidateAuditObservabilityError("pinned selector SHA drift")
    selection = selector.select_frame(
        authorities,
        raws,
        evidence_root=evidence_root,
        image_shape=image_shape,
        thresholds=thresholds,
    )
    if selection.frame_status == "IDENTITY_HOLD" and not selection.routing_audit:
        audit = build_pre_enumeration_record(
            selection.session_id,
            selection.frame_index,
            abort_reason="SIDE_IDENTITY_LINEAGE_ALIAS_BEFORE_CANDIDATE_ENUMERATION",
            selection=selection,
        )
    else:
        audit = build_enumerated_audit(selection, raws)
    return ObservedSelection(selection, audit)


def _assert_safe_destination(destination: Path, output_root: Path) -> None:
    root = output_root.resolve(strict=True)
    if not root.is_dir() or output_root.is_symlink():
        raise CandidateAuditObservabilityError("audit output root invalid")
    try:
        relative = destination.relative_to(output_root)
    except ValueError as exc:
        raise CandidateAuditObservabilityError("audit output escapes root") from exc
    if len(relative.parts) != 1:
        raise CandidateAuditObservabilityError("audit output must be direct child")
    for path in (output_root, destination):
        if path.exists() and stat.S_ISLNK(path.lstat().st_mode):
            raise CandidateAuditObservabilityError("audit output symlink rejected")


def write_audit_exclusive(destination: Path, output_root: Path, record: Mapping[str, Any]) -> str:
    _assert_safe_destination(destination, output_root)
    data = canonical_json(record) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(destination, flags, 0o640)
    except OSError as exc:
        raise CandidateAuditObservabilityError("exclusive audit write failed") from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        raise
    return hashlib.sha256(data).hexdigest()
