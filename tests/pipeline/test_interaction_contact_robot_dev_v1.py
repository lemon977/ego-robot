from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.pipeline.interaction_contact_robot_dev_v1 import (
    CONTACT_DISTANCE_M,
    FiniteSurfacePatch,
    InteractionContactError,
    build_pair_windows,
    canonicalize_normal_sequence,
    classify_contact_candidate,
    collision_scope_result,
    evaluate_frozen_frames,
    fit_human_stereo_ray_depth_alignment,
    point_to_finite_patch,
    sample_finger_associated_visible_surface,
    temporal_geometry_diagnostics,
    timestamp_motion_diagnostics,
    transition_taper_weights,
)


def test_visible_centroid_motion_is_diagnostic_not_fixed_center_failure() -> None:
    centers = np.asarray([[0.0, 0.0, 0.5], [0.03, 0.0, 0.5]])
    normals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    result = temporal_geometry_diagnostics(
        centers, normals, comparable_visibility=np.ones(2, bool),
        camera_motion_compensation_available=False,
    )
    assert result["camera_space_visible_center_step_p95_m"] == pytest.approx(0.03)
    assert result["failure_gate_applied"] is False
    assert result["visible_surface_center_semantics"] == "NOT_OBJECT_FIXED_CENTER"
    assert "NO_TRUSTED_CAMERA_MOTION" in result["interpretation"]


def test_plane_normal_sign_equivalence() -> None:
    result = canonicalize_normal_sequence(np.asarray([
        [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
    ]))
    assert np.allclose(result[0], result[1])
    diagnostics = temporal_geometry_diagnostics(
        np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]), result,
        comparable_visibility=np.ones(2, bool),
        camera_motion_compensation_available=True,
    )
    assert diagnostics["sign_canonicalized_normal_step_p95_deg"] == pytest.approx(0.0)


def _surface_inputs() -> dict:
    depth = np.full((480, 640), 0.4, np.float64)
    valid = np.ones((480, 640), bool)
    lr = np.ones((480, 640), bool)
    residual = np.full((480, 640), 0.2, np.float64)
    hand = np.zeros((960, 1280), bool)
    hand[470:530, 610:670] = True
    objects = np.zeros_like(hand)
    k = np.asarray([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    return dict(
        source_pixel_uv_1280=[640.5, 500.5], depth_m=depth,
        depth_valid=valid, lr_consistent=lr, lr_residual_px=residual,
        intrinsics=k, hand_mask_1280=hand, sleeve_mask_1280=None,
        object_union_1280=objects, associated_hand="left",
        associated_finger="index",
    )


def test_visible_surface_is_not_automatically_anatomical_fingertip() -> None:
    inputs = _surface_inputs()
    inputs["hand_mask_1280"][:] = False
    inputs["hand_mask_1280"][490:510, 630:650] = True
    result = sample_finger_associated_visible_surface(**inputs)
    assert result["fingertip_surface_observation"] is False
    assert result["status"] == "OBSERVED_VISIBLE_SURFACE"


@pytest.mark.parametrize("failure", ["sleeve", "object", "depth_boundary"])
def test_sleeve_object_or_depth_boundary_cannot_get_fingertip_authority(failure: str) -> None:
    inputs = _surface_inputs()
    if failure == "sleeve":
        inputs["sleeve_mask_1280"] = inputs["hand_mask_1280"].copy()
    elif failure == "object":
        inputs["object_union_1280"][470:530, 610:670] = True
    else:
        inputs["depth_m"][:, 320:] = 0.8
    result = sample_finger_associated_visible_surface(**inputs)
    assert result["fingertip_surface_observation"] is False


def _alignment_rows(count: int = 60) -> list[dict]:
    rows = []
    for index in range(count):
        source = 0.25 + 0.002 * index
        rows.append({
            "frame_id": index,
            "source_kind": "NON_CONTACT_VISIBLE_HAND_SURFACE",
            "near_task_object": False,
            "contact_or_object_fit_used": False,
            "hawor_ray_depth_m": source,
            "stereo_surface_depth_m": 1.02 * source + 0.004,
        })
    return rows


def test_alignment_excludes_contact_fit_and_freezes_holdout() -> None:
    rows = _alignment_rows()
    rows.extend({
        "frame_id": index,
        "source_kind": "NON_CONTACT_VISIBLE_HAND_SURFACE",
        "near_task_object": True,
        "contact_or_object_fit_used": True,
        "hawor_ray_depth_m": 0.4,
        "stereo_surface_depth_m": 0.8,
    } for index in range(60, 70))
    result = fit_human_stereo_ray_depth_alignment(rows)
    assert result["status"] == "PASS_DEVELOPMENT_ALIGNMENT"
    assert result["contact_or_object_fit_used"] is False
    assert result["train_row_count"] == 48
    assert result["holdout_row_count"] == 12
    assert result["heldout"]["frame_ids"] == list(range(0, 60, 5))


def test_alignment_forbids_fixed_48mm_bias() -> None:
    with pytest.raises(InteractionContactError, match="48 mm"):
        fit_human_stereo_ray_depth_alignment(_alignment_rows(), fixed_bias_mm=48.0)


def test_large_uncertainty_never_expands_contact_gate() -> None:
    result = classify_contact_candidate(
        finite_patch_distance_m=0.030, inside_visible_patch=True,
        local_depth_robust_sigma_m=0.012, plane_residual_p90_m=0.002,
        lr_residual_median_px=0.5, approach_supported=False,
        co_motion_supported=False, tactile_supported=True,
    )
    assert result["distance_gate_m"] == CONTACT_DISTANCE_M
    assert result["distance_gate_was_uncertainty_expanded"] is False
    assert result["geometric_proximity_pass"] is False
    assert result["state"] == "NEAR_UNCERTAIN"


def test_missing_uncertainty_never_defaults_to_zero() -> None:
    result = classify_contact_candidate(
        finite_patch_distance_m=0.003, inside_visible_patch=True,
        local_depth_robust_sigma_m=None, plane_residual_p90_m=0.001,
        lr_residual_median_px=0.3, approach_supported=False,
        co_motion_supported=False, tactile_supported=False,
    )
    assert result["uncertainty_complete"] is False
    assert result["uncertainty_admission_pass"] is False
    assert result["state"] == "UNKNOWN"


def _patch() -> FiniteSurfacePatch:
    return FiniteSurfacePatch(
        center_xyz=np.asarray([0.0, 0.0, 0.5]),
        normal_xyz=np.asarray([0.0, 0.0, -1.0]),
        axis_u_xyz=np.asarray([1.0, 0.0, 0.0]),
        axis_v_xyz=np.asarray([0.0, -1.0, 0.0]),
        hull_uv_m=np.asarray([[-0.02, -0.03], [0.02, -0.03], [0.02, 0.03], [-0.02, 0.03]]),
        plane_residual_p90_m=0.001,
        registered_valid_depth_fraction=0.9,
        source_mask_pixel_count=1000,
    )


def test_infinite_plane_hit_outside_finite_patch_is_not_contact() -> None:
    distance = point_to_finite_patch([0.10, 0.0, 0.5], _patch())
    assert distance["object_plane_signed_distance_m"] == pytest.approx(0.0)
    assert distance["inside_visible_patch"] is False
    result = classify_contact_candidate(
        finite_patch_distance_m=distance["finite_patch_distance_m"],
        inside_visible_patch=distance["inside_visible_patch"],
        local_depth_robust_sigma_m=0.001, plane_residual_p90_m=0.001,
        lr_residual_median_px=0.2, approach_supported=False,
        co_motion_supported=False, tactile_supported=False,
    )
    assert result["geometric_proximity_pass"] is False
    assert result["state"] == "NO_EVIDENCE"


def test_static_contact_and_partial_sequence_can_form_window() -> None:
    rows = [{
        "hand_id": "left", "finger_id": "index", "object_id": "card00",
        "frame_id": frame, "timestamp_s": frame / 30.0,
        "state": "CONTACT_CANDIDATE",
    } for frame in range(10, 15)]
    windows = build_pair_windows(rows)
    assert len(windows) == 1
    assert windows[0]["terminal"] == "ADMITTED_LOCAL_WINDOW"


def test_continuity_never_crosses_identity_or_frame_gap() -> None:
    rows = []
    for finger, frames in (("index", [1, 2, 4, 5, 6]), ("middle", [7, 8, 9, 10, 11])):
        rows.extend({
            "hand_id": "left", "finger_id": finger, "object_id": "card00",
            "frame_id": frame, "timestamp_s": frame / 30.0,
            "state": "CONTACT_CANDIDATE",
        } for frame in frames)
    windows = build_pair_windows(rows)
    assert [row["frame_ids"] for row in windows] == [[1, 2], [4, 5, 6], [7, 8, 9, 10, 11]]
    assert [row["terminal"] for row in windows] == ["BLOCKED_SHORT_WINDOW", "BLOCKED_SHORT_WINDOW", "ADMITTED_LOCAL_WINDOW"]


def test_transition_smooths_delta_but_does_not_add_contact_labels() -> None:
    times = np.arange(20) / 10.0
    windows = [{
        "terminal": "ADMITTED_LOCAL_WINDOW", "timestamp_start_s": 0.8,
        "timestamp_end_s": 1.0,
    }]
    weights = transition_taper_weights(times, windows, transition_s=0.2)
    assert weights[8:11].tolist() == [1.0, 1.0, 1.0]
    assert 0 < weights[7] < 1
    assert 0 < weights[11] < 1
    assert weights[5] == 0
    assert "state" not in windows[0]


def test_timestamp_derivatives_are_physical_units() -> None:
    values = np.asarray([[0.0], [0.1], [0.2]])
    result = timestamp_motion_diagnostics(values, [0.0, 0.1, 0.2], np.ones(3, bool))
    assert result["max_abs_velocity_rad_s"] == pytest.approx(1.0)
    assert result["max_abs_acceleration_rad_s2"] == pytest.approx(0.0)


def test_fair_comparison_rejects_coverage_drop() -> None:
    before = {0: 0.01, 1: 0.02}
    after = {0: 0.005, 1: None}
    result = evaluate_frozen_frames(before, after, [0, 1])
    assert result["coverage_preserved"] is False
    assert result["comparison_admitted"] is False
    assert result["after_median"] is None


def test_collision_scopes_stay_separate() -> None:
    result = collision_scope_result(
        robot_self_collision="PASS_PINNED_GEOMETRY",
        observed_object_patch="PASS_NO_VISIBLE_PATCH_CROSSING",
    )
    assert result["collision_scope"] == "ROBOT_SELF_PLUS_OBSERVED_OBJECT_PATCH"
    assert result["full_object_collision"] == "UNVERIFIED"
    assert result["full_object_environment"] == "UNVERIFIED"


def test_published_r0_survives_missing_contact_and_r2_does_not_block() -> None:
    root = Path(__file__).resolve().parents[2]
    metrics = json.loads((
        root / "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1/METRICS.json"
    ).read_text(encoding="utf-8"))
    assert metrics["admitted_contact_window_count"] == 0
    assert metrics["r0_status"] == "COMPLETED_DEVELOPMENT_BASELINE"
    assert metrics["r1_status"] == "BLOCKED_LOCAL_EVIDENCE"
    assert metrics["r2_status"] == "NOT_RUN_OPTIONAL_R1_NOT_CLOSED"


@pytest.mark.parametrize("filename", [
    "OBJECT6D_GEOMETRY_REVIEW.mp4",
    "INTERACTION_CONTACT_REVIEW.mp4",
    "KAI22_R0_VS_R1_REVIEW.mp4",
])
def test_required_review_video_decodes_all_150_frames(filename: str) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1" / filename
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    count = 0
    geometry = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        geometry = frame.shape[:2]
        count += 1
    capture.release()
    assert count == 150
    assert geometry == (480, 1280)
