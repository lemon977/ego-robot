from copy import deepcopy

import pytest

from pipeline.mask_successor_protocol import (
    ProtocolError,
    temporal_failures,
    tracker_spatial_failures,
    validate_plan,
    validate_shared,
)


SHARED = {
    "schema_version": "shared-mask-spatial-temporal-successor-v1",
    "shared_model": {"checkpoint_write_forbidden": True},
    "evaluation_design": {
        "fit": ["chips:c1"],
        "independent_eval": ["chips:c8", "poker:p1"],
    },
    "spatial_canary": {
        "frame_count": 12,
        "execution": "one_independent_single_frame_model_state_per_source_frame",
        "previous_or_next_sample_visible_to_model": False,
        "cross_sample_flow_or_propagation_forbidden": True,
    },
    "temporal_canary": {
        "window_count": 3,
        "frames_per_window": 12,
        "source_indices_must_be_unit_stride": True,
        "propagation_between_windows_forbidden": True,
    },
    "tracker_spatial_gate": {
        "positive_anchor_coverage_fraction_min": 1.0,
        "negative_anchor_exclusion_fraction_min": 1.0,
        "own_pico_wrist_min_distance_px_max": 12.0,
        "centroid_to_own_pico_wrist_px_max": 120.0,
        "opposing_wrist_centroid_margin_px_min": 20.0,
        "frame_area_fraction_min": 0.0002,
        "frame_area_fraction_max": 0.02,
        "connected_components_over_64px_max": 2,
        "largest_component_area_fraction_min": 0.9,
        "protected_object_overlap_pixels_max": 0,
        "non_wrist_authorized_image_edge_contact_pixels_max": 0,
    },
}


PLAN = {
    "schema_version": "task-mask-successor-evaluation-plan-v1",
    "split": "FIT",
    "task_id": "chips",
    "session_id": "c1",
    "frame_count": 100,
    "spatial_source_frames": list(range(0, 24, 2)),
    "temporal_windows": [
        {"source_frames": list(range(30, 42))},
        {"source_frames": list(range(50, 62))},
        {"source_frames": list(range(70, 82))},
    ],
    "threshold_override": None,
}


def test_protocol_and_plan_accept_frozen_structure():
    validate_shared(SHARED)
    validate_plan(PLAN, SHARED)


def test_plan_rejects_cross_sample_temporal_jump():
    plan = deepcopy(PLAN)
    plan["temporal_windows"][1]["source_frames"][5] += 3
    with pytest.raises(ProtocolError, match="unit-stride"):
        validate_plan(plan, SHARED)


def test_eval_rejects_fit_result_prompt_leakage():
    plan = deepcopy(PLAN)
    plan.update({"split": "EVAL", "session_id": "c8", "fit_result_access_for_prompt_design": True})
    with pytest.raises(ProtocolError, match="may not inspect FIT"):
        validate_plan(plan, SHARED)


def test_tracker_gate_catches_background_leak_and_hand_absorption():
    gate = SHARED["tracker_spatial_gate"]
    good = {
        "present": True,
        "positive_anchor_coverage_fraction": 1.0,
        "negative_anchor_exclusion_fraction": 1.0,
        "own_pico_wrist_min_distance_px": 0.0,
        "centroid_to_own_pico_wrist_px": 35.0,
        "opposing_minus_own_wrist_centroid_px": 300.0,
        "frame_area_fraction": 0.008,
        "connected_components_over_64px": 1,
        "largest_component_area_fraction": 1.0,
        "protected_object_overlap_pixels": 0,
        "non_wrist_authorized_image_edge_contact_pixels": 0,
        "same_wrist_adjacency": True,
    }
    assert tracker_spatial_failures(good, gate) == []
    leaked = {**good, "non_wrist_authorized_image_edge_contact_pixels": 91}
    assert "background_edge_leak" in tracker_spatial_failures(leaked, gate)
    absorbed_hand = {**good, "negative_anchor_exclusion_fraction": 0.5, "frame_area_fraction": 0.03}
    failures = tracker_spatial_failures(absorbed_hand, gate)
    assert "negative_anchor" in failures and "area" in failures


def test_temporal_gate_requires_clean_contiguous_window_metrics():
    gate = {
        "flow_aligned_iou_median_min": 0.85,
        "flow_cycle_error_p90_px_half_max": 5.0,
        "adjacent_area_ratio_min": 0.65,
        "adjacent_area_ratio_max": 1.35,
        "side_swap_count_max": 0,
        "background_leak_frame_count_max": 0,
        "unexplained_role_disappearance_count_max": 0,
    }
    good = {
        "all_frames_pass_spatial": True,
        "flow_aligned_iou_median": 0.9,
        "flow_cycle_error_p90_px_half": 2.0,
        "adjacent_area_ratio_min": 0.8,
        "adjacent_area_ratio_max": 1.2,
        "side_swap_count": 0,
        "background_leak_frame_count": 0,
        "unexplained_role_disappearance_count": 0,
    }
    assert temporal_failures(good, gate) == []
    bad = {**good, "background_leak_frame_count": 1, "flow_aligned_iou_median": 0.2}
    assert temporal_failures(bad, gate) == ["flow_iou", "background_leak"]
