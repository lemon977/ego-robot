import numpy as np

from tools.diagnose_d4_contact_counterfactual_t0 import (
    camera_points_to_object,
    compress_intervals,
    counterfactual_flip,
    finite_y_cylinder_signed_distance,
    otherwise_eligible_object_only_candidates,
    transition_phase,
)


def audit(*, overlap=0.2, joint=0.9, margin=0.8, reasons=None, considered=True):
    if reasons is None:
        reasons = ["OBJECT6D_PROTECTION_REJECTION"]
    return {
        "raw_instance_offset": 3,
        "instance_id": 7,
        "raw_source_sha256": "a" * 64,
        "considered_for_side": considered,
        "joint_support_ratio": joint,
        "side_margin": margin,
        "object6d_overlap_over_instance": overlap,
        "ordered_rejection_reasons": reasons,
        "final_veto": None,
    }


def side_record(*audits, status="REJECT", reason="OBJECT6D_PROTECTION_REJECTION"):
    return {
        "status": status,
        "failure_reason": reason,
        "observability_candidate_audit": list(audits),
    }


def test_finite_y_cylinder_signed_distance_has_exact_zero_boundary():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.1, 2.1, 0.0],
        ]
    )
    distance = finite_y_cylinder_signed_distance(points, radius_m=1.0, height_m=4.0)
    assert np.allclose(distance, [-1.0, 0.0, 0.1, 0.0, np.hypot(0.1, 0.1)])


def test_camera_to_object_uses_object_to_camera_inverse():
    transform = np.eye(4)
    transform[:3, 3] = [3.0, -2.0, 5.0]
    camera = np.array([[4.0, 0.0, 8.0]])
    assert np.allclose(camera_points_to_object(camera, transform), [[1.0, 2.0, 3.0]])


def test_phase_is_raw_zero_crossing_without_smoothing():
    assert transition_phase(False, None) == "INITIAL_SEPARATED"
    assert transition_phase(True, None) == "INITIAL_INTERSECTING"
    assert transition_phase(True, False) == "INTERSECTION_ENTRY"
    assert transition_phase(True, True) == "INTERSECTION_CONTINUING"
    assert transition_phase(False, True) == "INTERSECTION_EXIT"
    assert transition_phase(False, False) == "SEPARATED_CONTINUING"
    assert transition_phase(None, True) == "UNMEASURED"


def test_counterfactual_flips_only_exact_object_only_intersection():
    row = side_record(audit())
    flip, candidates = counterfactual_flip(row, analytic_intersection=True)
    assert flip
    assert len(candidates) == 1
    assert not counterfactual_flip(row, analytic_intersection=False)[0]
    assert not counterfactual_flip(row, analytic_intersection=None)[0]


def test_existing_thresholds_are_not_lowered_or_bypassed():
    assert otherwise_eligible_object_only_candidates(side_record(audit(joint=0.199999))) == []
    assert otherwise_eligible_object_only_candidates(side_record(audit(margin=0.049999))) == []
    assert otherwise_eligible_object_only_candidates(side_record(audit(overlap=0.12))) == []
    assert otherwise_eligible_object_only_candidates(
        side_record(audit(reasons=["OBJECT6D_PROTECTION_REJECTION", "OTHER"]))
    ) == []
    assert otherwise_eligible_object_only_candidates(side_record(audit(considered=False))) == []


def test_non_object_reject_and_existing_accept_do_not_flip():
    candidate = audit()
    assert not counterfactual_flip(
        side_record(candidate, reason="WEAK_IN_IMAGE_JOINT_SUPPORT"),
        analytic_intersection=True,
    )[0]
    assert not counterfactual_flip(
        side_record(candidate, status="ACCEPT", reason=None),
        analytic_intersection=True,
    )[0]


def test_multiple_candidates_fail_closed_in_counterfactual():
    first = audit()
    second = {**audit(), "raw_instance_offset": 4, "instance_id": 8}
    assert not counterfactual_flip(
        side_record(first, second), analytic_intersection=True
    )[0]


def test_interval_compression():
    assert compress_intervals([5, 2, 3, 3, 9]) == [[2, 3], [5, 5], [9, 9]]
