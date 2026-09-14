import inspect
import hashlib

import numpy as np
import pytest

from pipeline.table_plane_drift_reanchor import Plane, TablePlaneError
from pipeline.table_plane_static_producer import (
    LOCAL_SUPPORT_SEMANTICS,
    evaluate_physical_gates,
    fit_session_static_plane,
    per_frame_reanchor_allowed,
    require_static_fit_pass,
    validate_support_scope,
)


def synthetic_corner_sets(frames: int = 15) -> np.ndarray:
    base = np.array(
        [[-0.05, 0.0, -0.05], [0.05, 0.0, -0.05], [0.05, 0.0, 0.05], [-0.05, 0.0, 0.05]],
        dtype=np.float64,
    )
    return np.stack([base + [0.002 * index, 0.0, 0.001 * index] for index in range(frames)])


def test_static_fit_emits_one_negative_y_plane_and_passes_frozen_gate() -> None:
    frames = tuple(range(228, 243))
    fit = fit_session_static_plane(frames, synthetic_corner_sets())
    assert np.array_equal(fit.plane.normal, np.array([0.0, -1.0, 0.0]))
    assert fit.plane.offset_m == pytest.approx(0.0)
    assert fit.residual_max_abs_m == pytest.approx(0.0)
    assert len(fit.per_frame_residuals) == 15
    require_static_fit_pass(
        fit,
        anchor_frame_count=15,
        anchor_point_count=60,
        tag_corner_reprojection_max_px=0.5,
    )


def test_static_fit_gate_rejects_wrong_cardinality_and_bad_residual() -> None:
    fit = fit_session_static_plane(range(15), synthetic_corner_sets())
    with pytest.raises(TablePlaneError, match="15 frames / 60"):
        require_static_fit_pass(
            fit,
            anchor_frame_count=14,
            anchor_point_count=56,
            tag_corner_reprojection_max_px=0.1,
        )
    noisy = synthetic_corner_sets()
    noisy[0, 0, 1] = 0.2
    bad = fit_session_static_plane(range(15), noisy)
    with pytest.raises(TablePlaneError, match="quality"):
        require_static_fit_pass(
            bad,
            anchor_frame_count=15,
            anchor_point_count=60,
            tag_corner_reprojection_max_px=0.1,
        )


def test_support_cannot_expand_without_geometry_evidence() -> None:
    validate_support_scope(
        LOCAL_SUPPORT_SEMANTICS,
        expansion_requested=False,
        expansion_evidence_ref=None,
        expansion_evidence_payload=None,
    )
    with pytest.raises(TablePlaneError, match="requires independent"):
        validate_support_scope(
            LOCAL_SUPPORT_SEMANTICS,
            expansion_requested=True,
            expansion_evidence_ref=None,
            expansion_evidence_payload=None,
        )


def test_per_frame_reanchor_fails_closed_without_independent_contact() -> None:
    assert not per_frame_reanchor_allowed(
        contact_evidence_valid=False,
        evidence_ref=None,
        evidence_payload=None,
    )
    with pytest.raises(TablePlaneError, match="invalid contact"):
        per_frame_reanchor_allowed(
            contact_evidence_valid=False,
            evidence_ref={"path": "x", "bytes": 1, "sha256": "0" * 64},
            evidence_payload=b"x",
        )
    payload = b"independent-contact-evidence"
    assert per_frame_reanchor_allowed(
        contact_evidence_valid=True,
        evidence_ref={
            "path": "x",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        evidence_payload=payload,
    )


def test_reanchor_rejects_forged_sha_or_unbound_payload() -> None:
    payload = b"real-evidence"
    with pytest.raises(TablePlaneError, match="bound"):
        per_frame_reanchor_allowed(
            contact_evidence_valid=True,
            evidence_ref={"path": "x", "bytes": len(payload), "sha256": "0" * 64},
            evidence_payload=payload,
        )


def test_physical_gates_are_geometry_only_and_independent_of_confidence() -> None:
    assert "confidence" not in inspect.signature(evaluate_physical_gates).parameters
    plane = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    result = evaluate_physical_gates(
        plane,
        wrist_points_world=np.array([[0.0, -0.1, 0.0], [0.0, 0.2, 0.0]]),
        hand_points_world=np.array([[0.0, 0.1, 0.0]]),
        fingertip_points_world=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        object_to_world=np.eye(4),
        cylinder_radius_m=0.1,
        cylinder_height_m=1.0,
    )
    assert result["confidence_consumed"] is False
    assert result["wrist_below_table"] == [True, False]
    assert result["hand_point_below_table"] == [False]
    assert result["fingertip_inside_cylinder"] == [True, False]
    assert result["status"] == "FAIL_PHYSICAL_GEOMETRY"


def test_missing_reviewed_geometry_is_not_mislabeled_pass() -> None:
    result = evaluate_physical_gates(
        Plane(np.array([0.0, -1.0, 0.0]), 0.0),
        wrist_points_world=None,
        hand_points_world=None,
        fingertip_points_world=None,
        object_to_world=None,
        cylinder_radius_m=None,
        cylinder_height_m=None,
    )
    assert result["status"] == "NOT_EVALUATED_MISSING_REVIEWED_ROBOT_GEOMETRY"
    assert result["confidence_consumed"] is False
