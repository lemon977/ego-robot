import numpy as np

from pipeline.object_state_layout_ab import compare_binary_masks, summarize_equivalence


GATE = {
    "per_role_per_frame_binary_mask_iou_min": 1.0,
    "per_role_per_frame_area_ratio_min": 1.0,
    "per_role_per_frame_area_ratio_max": 1.0,
    "per_role_per_frame_centroid_delta_px_max": 0.0,
    "cross_id_bleed_frame_count_max": 0,
    "internal_nonoverlap_or_suppression_applied": False,
    "speedup_ratio_min": 1.25,
}


def test_strict_equivalence_requires_no_internal_cross_object_modification():
    mask = np.zeros((8, 9), bool)
    mask[2:5, 3:6] = True
    row = compare_binary_masks(mask, mask.copy(), frame_index=0, role="chip_0")
    result = summarize_equivalence(
        [row],
        expected_role_frame_count=1,
        gate=GATE,
        per_role_gate_vectors_exact=True,
        cross_instance_gate_vectors_exact=True,
        final_status_exact=True,
        cross_id_bleed_frame_count=0,
        internal_nonoverlap_or_suppression_applied=True,
        wall_seconds_a=10.0,
        wall_seconds_b=5.0,
    )
    assert result["pass"] is False
    assert result["gates"]["binary_mask_exact"] is True
    assert result["gates"]["no_internal_nonoverlap_or_suppression"] is False


def test_one_pixel_difference_fails_exact_equivalence():
    left = np.zeros((8, 9), bool)
    left[2:5, 3:6] = True
    right = left.copy()
    right[2, 3] = False
    row = compare_binary_masks(left, right, frame_index=3, role="card_1")
    result = summarize_equivalence(
        [row],
        expected_role_frame_count=1,
        gate=GATE,
        per_role_gate_vectors_exact=True,
        cross_instance_gate_vectors_exact=True,
        final_status_exact=True,
        cross_id_bleed_frame_count=0,
        internal_nonoverlap_or_suppression_applied=False,
        wall_seconds_a=10.0,
        wall_seconds_b=5.0,
    )
    assert result["pass"] is False
    assert result["gates"]["binary_mask_exact"] is False
