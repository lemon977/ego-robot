from chaoyang.ops.build_rc1_expandable_candidates_v1 import connected_components, expansion_route


def test_connected_components_merge_cross_source_clip():
    rows = [
        {"session_id": "a", "claimed_source_sessions": ["source_1"]},
        {"session_id": "b", "claimed_source_sessions": ["source_2", "source_3"]},
        {"session_id": "c", "claimed_source_sessions": ["source_1", "source_2"]},
        {"session_id": "unknown", "claimed_source_sessions": []},
    ]
    component, count = connected_components(rows)
    assert count == 1
    assert component["a"] == component["b"] == component["c"]
    assert "unknown" not in component


def test_calibration_only_is_visual_review_not_metric_authority():
    assert expansion_route("CALIBRATION_MISSING") == "VISUAL_TIER_REVIEW_NO_METRIC_CONTACT"
    assert expansion_route("OBJECT_MASK_C") == "BOUNDED_SAM31_OBJECT_IDENTITY_SUCCESSOR"
