import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import lr_contact_phase_object6d_evidence as d4
from pipeline import lr_distributed_side_evidence as base


SHA = "a" * 64


def write_json(path: Path, value: dict) -> base.EvidenceRef:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return base.EvidenceRef(str(path), len(payload), base.sha256_bytes(payload), value["source_kind"])


def authority(tmp_path: Path, side: str, frame: int = 0) -> base.SideAuthority:
    wrist = [10.0, 10.0] if side == "left" else [80.0, 10.0]
    payload = {
        "source_kind": "A_PRIME_V3_FRAME_SELECTION",
        "frame_index": frame,
        "sides": {
            side: {
                "authority": {
                    "identity": side,
                    "lineage_id": f"lineage-{side}",
                    "state": "AVAILABLE",
                    "wrist_xy": wrist,
                }
            }
        },
    }
    ref = write_json(tmp_path / f"authority-{side}-{frame}.json", payload)
    return base.SideAuthority(
        side,
        "grap_a_cap_012",
        frame,
        f"lineage-{side}",
        "AVAILABLE",
        tuple(wrist),
        ref,
        True,
    )


def support(own: float) -> base.JointSupportEvidence:
    supported = int(round(own * 20))
    return base.JointSupportEvidence(20, supported, supported / 20, 3.0, True)


def raw(
    tmp_path: Path,
    *,
    offset: int,
    instance_id: int,
    left: float,
    right: float,
    overlap: float,
    pixel: tuple[int, int],
) -> base.RawInstance:
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[pixel] = True
    payload = {
        "source_kind": "SAM_RAW_INSTANCE",
        "frame_id": "grap_a_cap_012:0",
        "instance_id": instance_id,
        "mask_sha256": base.mask_sha256(mask),
    }
    ref = write_json(tmp_path / f"raw-{offset}.json", payload)
    return base.RawInstance(
        "grap_a_cap_012",
        0,
        offset,
        instance_id,
        ref,
        mask,
        0.9,
        {"left": support(left), "right": support(right)},
        {"left": False, "right": False},
        overlap,
        0.01,
    )


def points(intersecting: bool) -> np.ndarray:
    result = np.full((21, 3), [3.0, 0.0, 0.0], dtype=np.float64)
    if intersecting:
        result[0] = [0.5, 0.0, 0.0]
    return result


def geometry(
    side: str,
    *,
    intersecting: bool,
    previous: bool | None = None,
    valid: bool = True,
    frame: int = 0,
) -> d4.ContactGeometryInput:
    return d4.ContactGeometryInput(
        "grap_a_cap_012",
        frame,
        side,
        0 if side == "left" else 1,
        points(intersecting),
        np.eye(4),
        1.0,
        4.0,
        valid,
        valid,
        previous,
        SHA,
        "b" * 64,
        "c" * 64,
        True,
    )


def inputs(tmp_path: Path, *, left_contact: bool):
    authorities = {side: authority(tmp_path, side) for side in d4.SIDES}
    raws = [
        raw(
            tmp_path,
            offset=0,
            instance_id=10,
            left=0.9,
            right=0.0,
            overlap=0.2,
            pixel=(20, 20),
        ),
        raw(
            tmp_path,
            offset=1,
            instance_id=11,
            left=0.0,
            right=0.9,
            overlap=0.0,
            pixel=(20, 80),
        ),
    ]
    contact = {
        "left": geometry("left", intersecting=left_contact),
        "right": geometry("right", intersecting=False),
    }
    return authorities, raws, contact


def select(tmp_path: Path, *, left_contact: bool):
    authorities, raws, contact = inputs(tmp_path, left_contact=left_contact)
    return d4.select_frame_contact_phase(
        authorities,
        raws,
        contact,
        evidence_root=tmp_path,
        image_shape=(100, 100),
    ), raws


def test_contact_applies_exact_object_only_delta_and_raw_pixels(tmp_path):
    result, raws = select(tmp_path, left_contact=True)
    assert result.base_selection.left.status == "REJECT"
    assert result.base_selection.left.failure_reason == d4.OBJECT_REASON
    assert result.selection.left.status == "ACCEPT"
    assert result.selection.left.raw_instance_offset == 0
    assert np.array_equal(result.selection.left.mask, raws[0].mask)
    assert not np.shares_memory(result.selection.left.mask, raws[0].mask)
    assert result.application_audits["left"].pixel_source == "EXACT_ACCEPTED_RAW_INSTANCE"
    assert result.selection.pixels_created_or_edited == 0
    assert result.selection.union_operations == 0
    assert result.selection.object6d_subtraction_operations == 0


def test_noncontact_is_exact_task26_object_and_pixels(tmp_path):
    result, _ = select(tmp_path, left_contact=False)
    assert result.selection is result.base_selection
    assert result.selection.left.status == "REJECT"
    assert result.application_audits["left"].status == "NO_CHANGE_SEPARATED"


def test_positive_arbitrarily_small_distance_does_not_apply(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=False)
    tiny = points(False)
    tiny[0] = [1.0 + np.finfo(np.float64).eps, 0.0, 0.0]
    contact["left"] = d4.ContactGeometryInput(
        **{**contact["left"].__dict__, "joints_3d_camera": tiny}
    )
    result = d4.select_frame_contact_phase(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert result.contact_measurements["left"].minimum_joint_center_signed_distance_m > 0.0
    assert result.selection is result.base_selection


def test_exact_surface_is_noncontact_and_retains_object_veto(tmp_path):
    value = geometry("left", intersecting=False)
    exact = points(False)
    exact[0] = [1.0, 0.0, 0.0]
    value = d4.ContactGeometryInput(**{**value.__dict__, "joints_3d_camera": exact})
    measured = d4.measure_contact_phase(value)
    assert measured.minimum_joint_center_signed_distance_m == 0.0
    assert measured.state == "SEPARATED"
    assert measured.analytic_intersection is False
    assert measured.boundary_m == 0.0
    assert measured.new_adjustable_parameters == 0

    authorities, raws, contact = inputs(tmp_path, left_contact=False)
    contact["left"] = value
    result = d4.select_frame(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert result.selection.left.status == "REJECT"
    assert result.selection.left.failure_reason == d4.OBJECT_REASON


def test_invalid_nan_object_geometry_is_unmeasured_and_cannot_apply(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    invalid_transform = np.full((4, 4), np.nan, dtype=np.float64)
    for side in d4.SIDES:
        contact[side] = d4.ContactGeometryInput(
            **{
                **contact[side].__dict__,
                "transform_object_to_camera": invalid_transform,
                "object6d_valid": False,
            }
        )
    result = d4.select_frame_contact_phase(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert all(
        result.contact_measurements[side].state == "UNMEASURED" for side in d4.SIDES
    )
    assert result.selection is result.base_selection
    assert all(
        result.application_audits[side].status == "NO_CHANGE_UNMEASURED"
        for side in d4.SIDES
    )
    assert all(
        result.application_audits[side].pixel_source == "UNCHANGED_TASK26"
        for side in d4.SIDES
    )
    assert result.selection.pixels_created_or_edited == 0


def test_previous_phase_is_diagnostic_only(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    contact_entry = dict(contact)
    contact_stay = dict(contact)
    contact_entry["left"] = geometry("left", intersecting=True, previous=False)
    contact_stay["left"] = geometry("left", intersecting=True, previous=True)
    a = d4.select_frame_contact_phase(
        authorities, raws, contact_entry, evidence_root=tmp_path, image_shape=(100, 100)
    )
    b = d4.select_frame_contact_phase(
        authorities, raws, contact_stay, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert a.contact_measurements["left"].transition == "INTERSECTION_ENTRY"
    assert b.contact_measurements["left"].transition == "INTERSECTION_CONTINUING"
    assert base.mask_sha256(a.selection.left.mask) == base.mask_sha256(b.selection.left.mask)


def test_other_numeric_rejection_is_not_overridden(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    raws[0] = raw(
        tmp_path,
        offset=2,
        instance_id=12,
        left=0.15,
        right=0.0,
        overlap=0.2,
        pixel=(21, 20),
    )
    result = d4.select_frame_contact_phase(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert result.selection.left.status == "REJECT"
    assert result.application_audits["left"].status == "NO_CHANGE_NOT_EXACT_OBJECT_ONLY_REJECTION"


def test_overlap_at_exact_existing_limit_needs_no_override(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    raws[0] = raw(
        tmp_path,
        offset=2,
        instance_id=12,
        left=0.9,
        right=0.0,
        overlap=0.12,
        pixel=(21, 20),
    )
    result = d4.select_frame_contact_phase(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert result.base_selection.left.status == "ACCEPT"
    assert result.selection is result.base_selection


def test_wrong_side_routing_cannot_be_changed_by_contact(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    raws[0] = raw(
        tmp_path,
        offset=2,
        instance_id=12,
        left=0.5,
        right=0.9,
        overlap=0.2,
        pixel=(21, 20),
    )
    result = d4.select_frame_contact_phase(
        authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
    )
    assert result.selection.left.status == "REJECT"
    assert result.selection.left.raw_instance_offset is None


def test_multiple_object_only_candidates_fail_closed(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    raws.insert(
        1,
        raw(
            tmp_path,
            offset=2,
            instance_id=12,
            left=0.8,
            right=0.0,
            overlap=0.21,
            pixel=(21, 21),
        ),
    )
    with pytest.raises(d4.ContactPhaseEvidenceError, match="multiple object-only"):
        d4.select_frame_contact_phase(
            authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
        )


def test_side_axis_alias_and_cross_source_mismatch_fail_closed(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    contact["left"] = d4.ContactGeometryInput(
        **{**contact["left"].__dict__, "physical_hand_axis": 1}
    )
    with pytest.raises(d4.ContactPhaseEvidenceError, match="axis/side"):
        d4.select_frame_contact_phase(
            authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
        )
    contact = inputs(tmp_path, left_contact=True)[2]
    contact["right"] = d4.ContactGeometryInput(
        **{**contact["right"].__dict__, "object6d_source_sha256": "d" * 64}
    )
    with pytest.raises(d4.ContactPhaseEvidenceError, match="Object6D source mismatch"):
        d4.select_frame_contact_phase(
            authorities, raws, contact, evidence_root=tmp_path, image_shape=(100, 100)
        )


def test_threshold_drift_remains_forbidden(tmp_path):
    authorities, raws, contact = inputs(tmp_path, left_contact=True)
    with pytest.raises(base.DistributedSideEvidenceError, match="threshold drift"):
        d4.select_frame_contact_phase(
            authorities,
            raws,
            contact,
            evidence_root=tmp_path,
            image_shape=(100, 100),
            thresholds=base.SelectorThresholds(max_object_overlap_over_instance=0.13),
        )


def test_fact_counterfactual_exact_population_boundary():
    measurement = d4.ContactPhaseMeasurement(
        "grap_a_cap_012",
        75,
        "left",
        0,
        "INTERSECTING",
        "INTERSECTION_ENTRY",
        -0.001,
        1,
        True,
        SHA,
    )
    candidate = d4.ObjectOnlyCandidateFacts(
        3, 4, SHA, 0.9, 0.9, 0.2, (d4.OBJECT_REASON,)
    )
    assert d4.counterfactual_flip_from_facts(
        current_status="REJECT",
        current_failure_reason=d4.OBJECT_REASON,
        measurement=measurement,
        candidates=[candidate],
    )
    noncontact = replace_measurement(measurement, analytic_intersection=False)
    assert not d4.counterfactual_flip_from_facts(
        current_status="REJECT",
        current_failure_reason=d4.OBJECT_REASON,
        measurement=noncontact,
        candidates=[candidate],
    )


def replace_measurement(value, **changes):
    return d4.ContactPhaseMeasurement(**{**value.__dict__, **changes})
