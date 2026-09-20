from __future__ import annotations

import numpy as np
import pytest

from chaoyang.research.world_in_your_hands.ai1_static_wrist_candidate_v32 import (
    Ai1StaticWristCandidateError,
    controller_local_residual_diagnostic,
    decide_candidate_authority,
    finite_lag_diagnostics,
    require_same_wrist_semantics,
)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return value


def test_fixed_controller_lever_arm_rotates_camera_residual_with_pose() -> None:
    controller = np.stack([_rotation_z(0.0), _rotation_z(np.pi / 2.0), _rotation_z(np.pi)])
    nominal = np.eye(4, dtype=np.float64)
    nominal[:3, 3] = (0.1, 0.0, 0.0)
    true_local_offset = np.asarray((0.02, -0.01, 0.0), dtype=np.float64)
    observed = (controller @ nominal)[:, :3, 3] + np.einsum(
        "nij,j->ni", controller[:, :3, :3], true_local_offset
    )

    result = controller_local_residual_diagnostic(
        controller,
        nominal,
        observed,
        candidate_wrist_semantics="ANATOMICAL_WRIST_CENTER",
        observed_wrist_semantics="ANATOMICAL_WRIST_CENTER",
    )

    expected_camera = np.einsum("nij,j->ni", controller[:, :3, :3], true_local_offset)
    assert np.allclose(result["residual_camera"], expected_camera)
    assert not np.allclose(result["residual_camera"][0], result["residual_camera"][1])
    assert np.allclose(
        result["residual_controller"],
        np.broadcast_to(true_local_offset, (3, 3)),
    )


def test_different_wrist_semantics_cannot_form_direct_error() -> None:
    with pytest.raises(Ai1StaticWristCandidateError, match="wrist semantics differ"):
        require_same_wrist_semantics(
            "CONTROLLER_DERIVED_STATIC_WRIST_POSITION_ONLY",
            "HAWOR_MODEL_DEFINED_ANATOMICAL_WRIST",
        )
    controller = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    with pytest.raises(Ai1StaticWristCandidateError, match="wrist semantics differ"):
        controller_local_residual_diagnostic(
            controller,
            np.eye(4),
            np.zeros((3, 3)),
            candidate_wrist_semantics="CONTROLLER_WRIST",
            observed_wrist_semantics="ANATOMICAL_WRIST",
        )


def test_selected_m1_is_not_automatically_adopted_or_opening_102_103() -> None:
    decision = decide_candidate_authority(
        selected_model="M1_CONTROLLER_LOCAL_TRANSLATION",
        deterministic_error_fixed=False,
        leave_one_recording_out_passed=False,
        regression_101_passed=False,
        wrist_semantics_proven_equivalent=False,
        independent_rotation_evidence=False,
    )
    assert decision["selected_for_diagnostics"] is True
    assert decision["adopted"] is False
    assert decision["terminal_status"] == "NO_ADMISSIBLE_STATIC_CANDIDATE"
    assert decision["opening_gate_102_103"] == "CLOSED"
    assert decision["authority"] == "NONE"


def test_m2_requires_independent_rotation_evidence() -> None:
    with pytest.raises(Ai1StaticWristCandidateError, match="independent rotation"):
        decide_candidate_authority(
            selected_model="M2_STATIC_SE3",
            deterministic_error_fixed=True,
            leave_one_recording_out_passed=True,
            regression_101_passed=True,
            wrist_semantics_proven_equivalent=True,
            independent_rotation_evidence=False,
        )


def test_finite_lag_diagnostics_reports_fixed_order_without_selection() -> None:
    points = np.stack([np.arange(6), np.zeros(6), np.ones(6)], axis=1).astype(np.float64)
    rows = finite_lag_diagnostics(points, points, np.ones(6, dtype=bool))
    assert [row["lag_frames"] for row in rows] == [-2, -1, 0, 1, 2]
    assert set(rows[0]) == {"lag_frames", "valid_frames", "norm_mm_p50", "norm_mm_p95"}
