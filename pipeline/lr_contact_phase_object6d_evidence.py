"""Production D4 contact-phase selector layered on the task26 selector.

The task26 selector remains the source of routing, thresholds and every normal
decision.  This module changes exactly one case: when the current side is
rejected only by the frozen Object6D overlap rule, exactly one routed candidate
otherwise passes the unchanged .20/.05 thresholds, and at least one of the 21
HaWoR joint centres lies strictly inside the frozen finite Object6D cylinder, that
one raw instance is accepted.  Positive distance, invalid geometry, ambiguity,
and every other rejection path retain the task26 result.

No tolerance, hand thickness, session/frame branch, pixel operation or temporal
pixel transport exists in this API.  ``previous_intersecting`` names only the
diagnostic transition phase and cannot affect acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pipeline import lr_distributed_side_evidence as base


Side = base.Side
SIDES = base.SIDES
SIDE_AXIS: Mapping[Side, int] = {"left": 0, "right": 1}
CONTACT_BOUNDARY_M = 0.0
SEMANTIC_DELTA_ID = "D4_CONTACT_PHASE_OBJECT6D_STRICT_NEGATIVE_V2"
PRODUCER_ID = "a_prime_distributed_side_evidence_plus_d4_contact_v2"
ROUTE_OF_EVIDENCE = "A_PRIME_SELECTOR_PLUS_HAWOR_OBJECT6D_CONTACT_PHASE"
GPU_EXECUTION_AUTHORIZED = False
SHARED_IO_RELATIVE_PATH = "tools/immutable_artifact_io.py"
SHARED_IO_SHA256 = "e9af8d38146dee0813d415330fb626a295482740b160435fac544a045d74e57e"
OBJECT_REASON = "OBJECT6D_PROTECTION_REJECTION"

ContactState = Literal["INTERSECTING", "SEPARATED", "UNMEASURED"]
ContactTransition = Literal[
    "INITIAL_INTERSECTING",
    "INITIAL_SEPARATED",
    "INTERSECTION_ENTRY",
    "INTERSECTION_CONTINUING",
    "INTERSECTION_EXIT",
    "SEPARATED_CONTINUING",
    "UNMEASURED",
]
ApplicationStatus = Literal[
    "APPLIED_OBJECT_OVERLAP_AS_CONTACT_EVIDENCE",
    "NO_CHANGE_SEPARATED",
    "NO_CHANGE_UNMEASURED",
    "NO_CHANGE_NOT_EXACT_OBJECT_ONLY_REJECTION",
]


class ContactPhaseEvidenceError(RuntimeError):
    """The D4 evidence or exact semantic boundary is inconsistent."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def finite_y_cylinder_signed_distance(
    points_object: np.ndarray,
    radius_m: float,
    height_m: float,
) -> np.ndarray:
    """Exact point SDF for the frozen finite Y-axis cylinder."""

    points = np.asarray(points_object, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ContactPhaseEvidenceError("contact points must be finite [21,3]")
    if not math.isfinite(radius_m) or not math.isfinite(height_m):
        raise ContactPhaseEvidenceError("cylinder dimensions must be finite")
    if radius_m <= 0.0 or height_m <= 0.0:
        raise ContactPhaseEvidenceError("cylinder dimensions must be positive")
    radial = np.hypot(points[:, 0], points[:, 2]) - radius_m
    axial = np.abs(points[:, 1]) - height_m / 2.0
    q = np.stack((radial, axial), axis=1)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def camera_points_to_object(
    points_camera: np.ndarray,
    transform_object_to_camera: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(transform_object_to_camera, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ContactPhaseEvidenceError("HaWoR points must be finite [21,3]")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ContactPhaseEvidenceError("Object6D transform must be finite [4,4]")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise ContactPhaseEvidenceError("Object6D transform is singular") from error
    homogeneous = np.concatenate((points, np.ones((21, 1))), axis=1)
    result = (inverse @ homogeneous.T).T
    if not np.array_equal(result[:, 3], np.ones(21)):
        raise ContactPhaseEvidenceError("Object6D homogeneous transform drift")
    return result[:, :3]


def transition_phase(current: bool | None, previous: bool | None) -> ContactTransition:
    if current is None:
        return "UNMEASURED"
    if previous is None:
        return "INITIAL_INTERSECTING" if current else "INITIAL_SEPARATED"
    if current and not previous:
        return "INTERSECTION_ENTRY"
    if current and previous:
        return "INTERSECTION_CONTINUING"
    if not current and previous:
        return "INTERSECTION_EXIT"
    return "SEPARATED_CONTINUING"


@dataclass(frozen=True)
class ContactGeometryInput:
    session_id: str
    frame_index: int
    identity: Side
    physical_hand_axis: int
    joints_3d_camera: np.ndarray
    transform_object_to_camera: np.ndarray
    cylinder_radius_m: float
    cylinder_height_m: float
    hawor_valid: bool
    object6d_valid: bool
    previous_intersecting: bool | None
    hawor_source_sha256: str
    object6d_source_sha256: str
    cylinder_source_sha256: str
    label_independent: bool = True

    def validate_identity(self) -> None:
        if self.identity not in SIDES:
            raise ContactPhaseEvidenceError("invalid contact-evidence side")
        if self.physical_hand_axis != SIDE_AXIS[self.identity]:
            raise ContactPhaseEvidenceError("physical hand axis/side mismatch")
        if not self.session_id or self.frame_index < 0:
            raise ContactPhaseEvidenceError("invalid contact-evidence frame identity")
        if type(self.hawor_valid) is not bool or type(self.object6d_valid) is not bool:
            raise ContactPhaseEvidenceError("geometry validity must be exact bool")
        if self.previous_intersecting is not None and type(self.previous_intersecting) is not bool:
            raise ContactPhaseEvidenceError("previous intersection must be bool or None")
        if not self.label_independent:
            raise ContactPhaseEvidenceError("contact evidence must be label-independent")
        for value in (
            self.hawor_source_sha256,
            self.object6d_source_sha256,
            self.cylinder_source_sha256,
        ):
            if not _is_sha256(value):
                raise ContactPhaseEvidenceError("contact source SHA is invalid")
        if np.asarray(self.joints_3d_camera).shape != (21, 3):
            raise ContactPhaseEvidenceError("HaWoR point shape drift")
        if np.asarray(self.transform_object_to_camera).shape != (4, 4):
            raise ContactPhaseEvidenceError("Object6D transform shape drift")
        if not math.isfinite(float(self.cylinder_radius_m)) or not math.isfinite(
            float(self.cylinder_height_m)
        ):
            raise ContactPhaseEvidenceError("cylinder dimension is non-finite")
        if self.cylinder_radius_m <= 0.0 or self.cylinder_height_m <= 0.0:
            raise ContactPhaseEvidenceError("cylinder dimension is non-positive")


@dataclass(frozen=True)
class ContactPhaseMeasurement:
    session_id: str
    frame_index: int
    identity: Side
    physical_hand_axis: int
    state: ContactState
    transition: ContactTransition
    minimum_joint_center_signed_distance_m: float | None
    joint_centers_inside_or_on_count: int | None
    analytic_intersection: bool | None
    input_binding_sha256: str
    boundary_m: float = CONTACT_BOUNDARY_M
    new_adjustable_parameters: int = 0


def _geometry_binding(value: ContactGeometryInput) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "session_id": value.session_id,
                "frame_index": value.frame_index,
                "identity": value.identity,
                "physical_hand_axis": value.physical_hand_axis,
                "hawor_valid": value.hawor_valid,
                "object6d_valid": value.object6d_valid,
                "previous_intersecting": value.previous_intersecting,
                "hawor_source_sha256": value.hawor_source_sha256,
                "object6d_source_sha256": value.object6d_source_sha256,
                "cylinder_source_sha256": value.cylinder_source_sha256,
                "cylinder_radius_m": float(value.cylinder_radius_m),
                "cylinder_height_m": float(value.cylinder_height_m),
            }
        )
    )
    for array in (value.joints_3d_camera, value.transform_object_to_camera):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(canonical_json({"shape": list(contiguous.shape), "dtype": "float64"}))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def measure_contact_phase(value: ContactGeometryInput) -> ContactPhaseMeasurement:
    value.validate_identity()
    binding = _geometry_binding(value)
    if not value.hawor_valid or not value.object6d_valid:
        return ContactPhaseMeasurement(
            value.session_id,
            value.frame_index,
            value.identity,
            value.physical_hand_axis,
            "UNMEASURED",
            "UNMEASURED",
            None,
            None,
            None,
            binding,
        )
    points_object = camera_points_to_object(
        value.joints_3d_camera, value.transform_object_to_camera
    )
    distances = finite_y_cylinder_signed_distance(
        points_object,
        float(value.cylinder_radius_m),
        float(value.cylinder_height_m),
    )
    minimum = float(np.min(distances))
    # Owner 2026-08-29 23:00: contact is strict negative SDF.  Exact zero is
    # non-contact and therefore retains the unchanged Object6D .12 veto.
    inside = int(np.count_nonzero(distances < CONTACT_BOUNDARY_M))
    intersects = inside > 0
    return ContactPhaseMeasurement(
        value.session_id,
        value.frame_index,
        value.identity,
        value.physical_hand_axis,
        "INTERSECTING" if intersects else "SEPARATED",
        transition_phase(intersects, value.previous_intersecting),
        minimum,
        inside,
        intersects,
        binding,
    )


@dataclass(frozen=True)
class ObjectOnlyCandidateFacts:
    raw_instance_offset: int
    instance_id: int
    raw_source_sha256: str
    joint_support_ratio: float
    side_margin: float
    object6d_overlap_over_instance: float
    rejection_reasons: tuple[str, ...]

    def qualifies(self) -> bool:
        values = (
            self.joint_support_ratio,
            self.side_margin,
            self.object6d_overlap_over_instance,
        )
        if self.raw_instance_offset < 0 or self.instance_id < 0:
            raise ContactPhaseEvidenceError("negative raw candidate identity")
        if not _is_sha256(self.raw_source_sha256):
            raise ContactPhaseEvidenceError("raw candidate source SHA is invalid")
        if not all(math.isfinite(value) for value in values):
            raise ContactPhaseEvidenceError("non-finite candidate metric")
        return (
            self.joint_support_ratio >= base.MIN_JOINT_SUPPORT_RATIO
            and self.side_margin >= base.MIN_SIDE_MARGIN
            and self.object6d_overlap_over_instance
            > base.MAX_OBJECT_OVERLAP_OVER_INSTANCE
            and self.rejection_reasons == (OBJECT_REASON,)
        )


def counterfactual_flip_from_facts(
    *,
    current_status: str,
    current_failure_reason: str | None,
    measurement: ContactPhaseMeasurement,
    candidates: Sequence[ObjectOnlyCandidateFacts],
) -> bool:
    qualified = [candidate for candidate in candidates if candidate.qualifies()]
    if len(qualified) > 1:
        raise ContactPhaseEvidenceError("multiple object-only candidates are outside D4 evidence")
    return (
        current_status == "REJECT"
        and current_failure_reason == OBJECT_REASON
        and measurement.analytic_intersection is True
        and len(qualified) == 1
    )


@dataclass(frozen=True)
class ContactApplicationAudit:
    identity: Side
    status: ApplicationStatus
    original_status: base.DecisionStatus
    original_failure_reason: str | None
    output_status: base.DecisionStatus
    selected_raw_instance_offset: int | None
    selected_raw_source_sha256: str | None
    object6d_overlap_over_instance: float | None
    contact_input_binding_sha256: str
    pixel_source: Literal["UNCHANGED_TASK26", "EXACT_ACCEPTED_RAW_INSTANCE"]
    pixels_created_or_edited: Literal[0] = 0
    threshold_changes: Literal[0] = 0


@dataclass(frozen=True)
class ContactAwareSelection:
    base_selection: base.FrameSelection
    selection: base.FrameSelection
    contact_measurements: Mapping[Side, ContactPhaseMeasurement]
    application_audits: Mapping[Side, ContactApplicationAudit]
    semantic_delta_id: str = SEMANTIC_DELTA_ID
    pixels_created_or_edited: Literal[0] = 0


def _facts_from_audit(audit: base.CandidateAudit) -> ObjectOnlyCandidateFacts:
    return ObjectOnlyCandidateFacts(
        audit.raw_instance_offset,
        audit.instance_id,
        audit.raw_source_sha256,
        audit.joint_support_ratio,
        audit.side_difference,
        audit.object6d_overlap_over_instance,
        audit.rejection_reasons,
    )


def _unchanged_application(
    side: Side,
    decision: base.SideDecision,
    measurement: ContactPhaseMeasurement,
) -> ContactApplicationAudit:
    if measurement.state == "UNMEASURED":
        status: ApplicationStatus = "NO_CHANGE_UNMEASURED"
    elif measurement.state == "SEPARATED":
        status = "NO_CHANGE_SEPARATED"
    else:
        status = "NO_CHANGE_NOT_EXACT_OBJECT_ONLY_REJECTION"
    return ContactApplicationAudit(
        side,
        status,
        decision.status,
        decision.failure_reason,
        decision.status,
        decision.raw_instance_offset,
        None,
        None,
        measurement.input_binding_sha256,
        "UNCHANGED_TASK26",
    )


def _apply_side(
    side: Side,
    decision: base.SideDecision,
    raws: Sequence[base.RawInstance],
    measurement: ContactPhaseMeasurement,
    *,
    evidence_root: Path,
) -> tuple[base.SideDecision, ContactApplicationAudit]:
    candidate_facts = [_facts_from_audit(audit) for audit in decision.candidate_audit]
    should_apply = counterfactual_flip_from_facts(
        current_status=decision.status,
        current_failure_reason=decision.failure_reason,
        measurement=measurement,
        candidates=candidate_facts,
    )
    if not should_apply:
        return decision, _unchanged_application(side, decision, measurement)

    qualified = [
        (audit, facts)
        for audit, facts in zip(decision.candidate_audit, candidate_facts, strict=True)
        if facts.qualifies()
    ]
    if len(qualified) != 1:
        raise ContactPhaseEvidenceError("D4 application lost unique candidate")
    selected_audit, selected_facts = qualified[0]
    matching = [
        raw
        for raw in raws
        if raw.raw_instance_offset == selected_facts.raw_instance_offset
        and raw.instance_id == selected_facts.instance_id
        and raw.raw_source.sha256 == selected_facts.raw_source_sha256
    ]
    if len(matching) != 1:
        raise ContactPhaseEvidenceError("D4 candidate/raw identity mismatch")
    selected = matching[0]
    # Geometry evaluation happens after the base selector's admission read.
    # Rebind the in-memory pixels to the frozen raw record at the copy boundary
    # so a mutation between those two phases fails closed.
    base.validate_raw_instance(selected, evidence_root=evidence_root)
    revised_audits = tuple(
        replace(audit, eligible=True, rejection_reasons=())
        if audit.raw_instance_offset == selected.raw_instance_offset
        else audit
        for audit in decision.candidate_audit
    )
    revised = base.SideDecision(
        side,
        decision.authority_lineage_id,
        decision.authority_state,
        "ACCEPT",
        selected.raw_instance_offset,
        selected.instance_id,
        np.asarray(selected.mask, dtype=np.bool_).copy(),
        None,
        decision.authority_wrist_outside_image_diagnostic,
        revised_audits,
    )
    if base.mask_sha256(revised.mask) != base.mask_sha256(selected.mask):
        raise ContactPhaseEvidenceError("D4 accepted output is not the exact raw mask")
    application = ContactApplicationAudit(
        side,
        "APPLIED_OBJECT_OVERLAP_AS_CONTACT_EVIDENCE",
        decision.status,
        decision.failure_reason,
        "ACCEPT",
        selected.raw_instance_offset,
        selected.raw_source.sha256,
        selected.object6d_overlap_over_instance,
        measurement.input_binding_sha256,
        "EXACT_ACCEPTED_RAW_INSTANCE",
    )
    return revised, application


def _frame_status(left: base.SideDecision, right: base.SideDecision) -> str:
    if left.status == "ACCEPT" and right.status == "ACCEPT":
        return "COMPLETE"
    if left.status == "HOLD" or right.status == "HOLD":
        return "PARTIAL_SIDE_HOLD"
    return "INCOMPLETE"


def _validate_contact_pair(
    authorities: Mapping[Side, base.SideAuthority],
    geometry: Mapping[Side, ContactGeometryInput],
) -> None:
    if set(geometry) != set(SIDES):
        raise ContactPhaseEvidenceError("exact left/right contact geometry required")
    for side in SIDES:
        value = geometry[side]
        value.validate_identity()
        authority = authorities[side]
        if (
            value.identity != side
            or value.session_id != authority.session_id
            or value.frame_index != authority.frame_index
        ):
            raise ContactPhaseEvidenceError("contact/authority frame identity mismatch")
    left = geometry["left"]
    right = geometry["right"]
    if left.hawor_source_sha256 != right.hawor_source_sha256:
        raise ContactPhaseEvidenceError("left/right HaWoR source mismatch")
    if left.object6d_source_sha256 != right.object6d_source_sha256:
        raise ContactPhaseEvidenceError("left/right Object6D source mismatch")
    if left.cylinder_source_sha256 != right.cylinder_source_sha256:
        raise ContactPhaseEvidenceError("left/right cylinder source mismatch")
    shared_object_is_invalid = (
        left.object6d_valid is False and right.object6d_valid is False
    )
    if (
        left.object6d_valid != right.object6d_valid
        or left.cylinder_radius_m != right.cylinder_radius_m
        or left.cylinder_height_m != right.cylinder_height_m
        or not np.array_equal(
            np.asarray(left.transform_object_to_camera),
            np.asarray(right.transform_object_to_camera),
            equal_nan=shared_object_is_invalid,
        )
    ):
        raise ContactPhaseEvidenceError("left/right Object6D geometry mismatch")


def select_frame_contact_phase(
    authorities: Mapping[Side, base.SideAuthority],
    raws: Sequence[base.RawInstance],
    contact_geometry: Mapping[Side, ContactGeometryInput],
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    thresholds: base.SelectorThresholds = base.SelectorThresholds(),
) -> ContactAwareSelection:
    """Apply the D4 delta after an exact frozen task26 decision."""

    thresholds.validate()
    _validate_contact_pair(authorities, contact_geometry)
    base_selection = base.select_frame(
        authorities,
        raws,
        evidence_root=evidence_root,
        image_shape=image_shape,
        thresholds=thresholds,
    )
    measurements = {
        side: measure_contact_phase(contact_geometry[side]) for side in SIDES
    }
    left, left_audit = _apply_side(
        "left",
        base_selection.left,
        raws,
        measurements["left"],
        evidence_root=evidence_root,
    )
    right, right_audit = _apply_side(
        "right",
        base_selection.right,
        raws,
        measurements["right"],
        evidence_root=evidence_root,
    )
    if left is base_selection.left and right is base_selection.right:
        selection = base_selection
    else:
        status = _frame_status(left, right)
        selection = base.FrameSelection(
            base_selection.session_id,
            base_selection.frame_index,
            left,
            right,
            status == "COMPLETE",
            status,
            base_selection.routing_audit,
            pixels_created_or_edited=0,
            union_operations=0,
            crop_operations=0,
            fill_operations=0,
            morphology_operations=0,
            object6d_subtraction_operations=0,
        )
    return ContactAwareSelection(
        base_selection,
        selection,
        measurements,
        {"left": left_audit, "right": right_audit},
    )


# The production entry is intentionally named like the base selector.  Callers
# select the D4 module and invoke ``select_frame``; the required geometry input
# makes accidental fallback to the non-contact selector impossible.
select_frame = select_frame_contact_phase
