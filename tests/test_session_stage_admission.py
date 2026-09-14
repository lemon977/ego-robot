from copy import deepcopy

from pipeline.session_stage_admission import MANO21_ORDER, REQUIRED_SEGMENTS, evaluate_session


def passing_metrics():
    side = {
        "expected_active_frame_count": 720,
        "expected_active_definition_verified": True,
        "visual_overlay_canary_verified": True,
        "valid_fraction": 0.99,
        "max_invalid_gap_frames": 2,
        "confidence_median": 0.8,
        "confidence_p05": 0.6,
        "joint_overlay_p05_fraction": 0.95,
        "reprojection_p95_px": 8.0,
        "bone_length_cv_max": 0.03,
        "positive_depth_fraction": 1.0,
        "identity_switch_count": 0,
        "duplicate_track_frame_count": 0,
        "root_rotation_orthogonality_max": 1e-6,
        "direct_observation_fraction": 0.95,
        "contact_canary_direct_fraction": 1.0,
        "contact_canary_frame_count": 24,
        "contact_canary_definition_verified": True,
        "wrist_step_p99_mm": 30.0,
        "wrist_step_max_mm": 60.0,
        "named_joint_mapping_verified": True,
    }
    return {
        "session_id": "fit_001",
        "task_id": "chips",
        "source": {
            "decode_ok": True,
            "same_session_clean_plate": True,
            "metric_measurement_authority": True,
            "peripheral_fiducials_visible": True,
            "reset_count": 0,
            "frame_count": 900,
            "timestamp_count": 900,
            "intrinsics_count": 900,
            "c2w_count": 900,
            "segments": list(REQUIRED_SEGMENTS),
        },
        "hawor": {
            "joint_order": MANO21_ORDER,
            "identity_assignment_verified": True,
            "dual_hand_collapse_frame_count": 0,
            "required_sides": ["left", "right"],
            "sides": {"left": side, "right": deepcopy(side)},
        },
        "mask": {
            "required_role_canary_recall": 1.0,
            "human_object_overlap_pixels": 0,
            "fixture_in_human_area_fraction": 0.0,
            "flow_aligned_iou_median": 0.9,
            "identity_switch_count": 0,
            "published_frame_fraction": 1.0,
            "object_protection_verified": True,
        },
        "clean": {
            "same_session_same_camera_donor": True,
            "donor_archive_crc_verified": True,
            "donor_purity_verified": True,
            "changed_pixels_outside_authorized_domain": 0,
            "changed_protected_object_pixels": 0,
            "source_map_known_fraction": 1.0,
            "per_pixel_source_map_verified": True,
            "residual_area_fraction": 0.0,
            "human_residual_ratio_max": 0.0,
            "tracker_residual_ratio_max": 0.0,
            "stable_background_changed_ratio_max": 0.0,
            "yellow_green_chroma_ghost_ratio_max": 0.0,
            "boundary_to_context_gradient_ratio_max": 1.0,
            "boundary_luma_halo_delta_max": 2.0,
            "illumination_mismatch_lab_p95_max": 0.0,
            "shadow_residual_area_fraction": 0.0,
            "flow_stabilized_temporal_flicker_lab_p95": 0.0,
            "consecutive_temporal_windows_verified": True,
            "object_codec_abs_error_p99": 0.0,
            "codec_audit_verified": True,
            "lossless_audit_master_available": True,
            "visual_review_no_human_or_seam": True,
        },
        "robot": {
            "session_constant_base": True,
            "required_pose_position_max_mm": 9.0,
            "required_pose_rotation_max_deg": 4.0,
            "joint_velocity_max_rad_per_frame_30fps": 0.1,
            "joint_acceleration_max_rad_per_frame2_30fps": 0.05,
            "dense_signed_distance_min_mm": -0.5,
            "required_contact_signed_distance_min_mm": 0.2,
            "required_contact_signed_distance_max_mm": 2.0,
            "exact_self_collision_count": 0,
            "joint_limits_pass": True,
            "object_depth_occlusion_pass": True,
        },
    }


def test_full_pass():
    result = evaluate_session(passing_metrics())
    assert result["first_failure"] is None
    assert result["routing"]["mask_auto_seed_allowed"] is True
    assert result["routing"]["robot_auto_seed_allowed"] is True


def test_single_session_constant_intrinsics_is_accepted():
    metrics = passing_metrics()
    metrics["source"]["intrinsics_count"] = 1
    metrics["source"]["intrinsics_session_constant"] = True
    result = evaluate_session(metrics)
    assert result["stages"]["source"]["status"] == "PASS"


def test_missing_clean_plate_is_capture_failure():
    metrics = passing_metrics()
    metrics["source"]["same_session_clean_plate"] = False
    result = evaluate_session(metrics)
    assert result["first_failure"]["owner"] == "CAPTURE"
    assert result["routing"]["manual_mask_route_allowed"] is True
    assert result["routing"]["clean_capture_ready"] is False


def test_missing_metric_geometry_blocks_robot_but_not_clean():
    metrics = passing_metrics()
    metrics["source"]["metric_measurement_authority"] = False
    result = evaluate_session(metrics)
    assert result["routing"]["clean_capture_ready"] is True
    assert result["routing"]["robot_capture_ready"] is False
    assert result["routing"]["robot_auto_seed_allowed"] is False


def test_wrong_joint_order_blocks_hawor_not_mask():
    metrics = passing_metrics()
    metrics["hawor"]["joint_order"] = "HUMANEGO_WRIST5"
    result = evaluate_session(metrics)
    assert result["first_failure"]["owner"] == "HAWOR"
    assert result["routing"]["mask_auto_seed_allowed"] is False


def test_missing_active_denominator_provenance_blocks_hawor():
    metrics = passing_metrics()
    metrics["hawor"]["sides"]["right"]["expected_active_definition_verified"] = False
    result = evaluate_session(metrics)
    assert result["stages"]["hawor_mask"]["status"] == "FAIL"
    assert result["stages"]["hawor_robot"]["status"] == "FAIL"


def test_mask_and_robot_use_different_reprojection_limits():
    metrics = passing_metrics()
    metrics["hawor"]["sides"]["right"]["reprojection_p95_px"] = 15.0
    result = evaluate_session(metrics)
    assert result["stages"]["hawor_mask"]["status"] == "PASS"
    assert result["stages"]["hawor_robot"]["status"] == "FAIL"


def test_dual_hand_collapse_blocks_both_hawor_routes():
    metrics = passing_metrics()
    metrics["hawor"]["dual_hand_collapse_frame_count"] = 1
    result = evaluate_session(metrics)
    assert result["stages"]["hawor_mask"]["status"] == "FAIL"
    assert result["stages"]["hawor_robot"]["status"] == "FAIL"


def test_mask_can_pass_while_robot_hawor_gate_fails():
    metrics = passing_metrics()
    metrics["hawor"]["sides"]["left"]["max_invalid_gap_frames"] = 6
    result = evaluate_session(metrics)
    assert result["stages"]["hawor_mask"]["status"] == "PASS"
    assert result["stages"]["hawor_robot"]["status"] == "FAIL"


def test_mask_failure_is_attributed_after_hawor_pass():
    metrics = passing_metrics()
    metrics["mask"]["human_object_overlap_pixels"] = 1
    result = evaluate_session(metrics)
    assert result["first_failure"]["owner"] == "MASK"


def test_clean_cross_camera_donor_is_rejected():
    metrics = passing_metrics()
    metrics["clean"]["same_session_same_camera_donor"] = False
    result = evaluate_session(metrics)
    assert result["first_failure"]["owner"] == "CLEAN"


def test_clean_old_v3_visual_summary_is_rejected():
    metrics = passing_metrics()
    metrics["clean"].update(
        {
            "donor_archive_crc_verified": False,
            "donor_purity_verified": False,
            "per_pixel_source_map_verified": False,
            "human_residual_ratio_max": 0.048660110408364717,
            "tracker_residual_ratio_max": 0.15116014558689717,
            "stable_background_changed_ratio_max": 0.7801680390905381,
            "yellow_green_chroma_ghost_ratio_max": 0.00034334763948497857,
            "boundary_to_context_gradient_ratio_max": 1.433046948775052,
            "boundary_luma_halo_delta_max": 5.0,
            "illumination_mismatch_lab_p95_max": 3.0,
            "shadow_residual_area_fraction": 0.2907676636725623,
            "flow_stabilized_temporal_flicker_lab_p95": 32.015621185302734,
            "consecutive_temporal_windows_verified": False,
            "object_codec_abs_error_p99": 26.0,
            "codec_audit_verified": False,
            "lossless_audit_master_available": False,
        }
    )
    result = evaluate_session(metrics)
    assert result["stages"]["clean"]["status"] == "FAIL"
    assert result["first_failure"]["owner"] == "CLEAN"


def test_clean_replay_calibrated_summary_passes():
    result = evaluate_session(passing_metrics())
    assert result["stages"]["clean"]["status"] == "PASS"


def test_clean_missing_new_quality_field_fails_closed():
    metrics = passing_metrics()
    del metrics["clean"]["flow_stabilized_temporal_flicker_lab_p95"]
    result = evaluate_session(metrics)
    assert result["stages"]["clean"]["status"] == "FAIL"
    assert any(
        "flow_stabilized_temporal_flicker_lab_p95" in reason
        for reason in result["stages"]["clean"]["failures"]
    )


def test_robot_penetration_is_robot_failure():
    metrics = passing_metrics()
    metrics["robot"]["dense_signed_distance_min_mm"] = -1.01
    result = evaluate_session(metrics)
    assert result["first_failure"]["owner"] == "ROBOT"


def test_missing_downstream_is_not_silently_passed():
    metrics = passing_metrics()
    del metrics["mask"]
    result = evaluate_session(metrics)
    assert result["stages"]["mask"]["status"] == "NOT_EVALUATED"
    assert result["first_failure"]["owner"] == "MASK"
