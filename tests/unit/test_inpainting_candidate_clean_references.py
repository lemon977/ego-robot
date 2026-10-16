from dataclasses import replace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import SETTINGS
from chaoyang.inpainting_candidate.clean_references import (
    ReferenceEvidence,
    classify_residual_pixels,
    select_reference_indices,
    validate_clean_preservation,
)


SHAPE = (2, 3)


def evidence(frame, support=None, error=1.0, **kwargs):
    return ReferenceEvidence(
        session_id="raw_session", target_frame=10, reference_frame=frame,
        visible_background=np.ones(SHAPE, bool) if support is None else support,
        flow_error=np.full(SHAPE, error, np.float32),
        raw_frame_sha256="1" * 64, raw_support_sha256="2" * 64,
        raw_flow_sha256="3" * 64, **kwargs,
    )


def select(items, **kwargs):
    return select_reference_indices(
        10, [8, 9, 10, 11, 12], 21,
        session_id="raw_session", effective_delete_mask=np.ones(SHAPE, bool),
        evidence=items, **kwargs,
    )


def test_references_change_only_selection_and_preserve_count_order_settings():
    small = np.zeros(SHAPE, bool)
    small[0, :2] = True
    result = select([evidence(3, small, 0.01), evidence(4, error=8), evidence(2, error=1)])
    assert result["original_references"] == [0, 20]
    assert result["references"] == [2, 4]
    assert result["reference_count"] == 2
    assert [row["frame"] for row in result["ranked_evidence"]] == [2, 4, 3]
    assert result["neighbors_unchanged"] and result["frozen_model_settings_unchanged"]
    assert result["causal_training_input"] is False
    assert "NOT_HIDDEN_OBJECT_TRUTH" in result["fill_source"]


def test_missing_evidence_does_not_reduce_reference_count():
    result = select([evidence(3)])
    assert result["references"] == [0, 20]
    assert result["status"] == "BASELINE_INSUFFICIENT_RAW_SUPPORTED_REFERENCES"


def test_flow_unknown_pixels_are_not_counted_as_support():
    item = evidence(3)
    item.flow_error[:] = np.nan
    result = select([item, evidence(4)])
    assert result["references"] == [0, 20]
    assert [row["frame"] for row in result["ranked_evidence"]] == [4]


def test_long_clip_never_expands_frozen_context_or_changes_neighbor_list():
    items = [replace(evidence(i), target_frame=50) for i in range(10, 20)]
    items.append(replace(evidence(95, error=0), target_frame=50))
    result = select_reference_indices(
        50, list(range(45, 56)), 101, session_id="raw_session",
        effective_delete_mask=np.ones(SHAPE, bool), evidence=items,
    )
    assert len(result["references"]) == len(result["original_references"]) == 7
    assert 95 not in result["references"]
    assert all(10 <= frame < 90 and frame not in range(45, 56) for frame in result["references"])


@pytest.mark.parametrize("field,value,code", [
    ("session_id", "different_session", "CROSS_SESSION"),
    ("target_frame", 11, "TARGET_ALIGNMENT"),
    ("coordinate_domain", "SOURCE_FRAME", "TARGET_ALIGNMENT"),
    ("support_source", "CLEAN_GENERATED_BACKGROUND", "REQUIRES_RAW"),
    ("support_source", "HIDDEN_OBJECT_GUESS", "REQUIRES_RAW"),
    ("flow_source", "COMPLETED_MASK_FLOW", "REQUIRES_RAW"),
    ("raw_flow_sha256", "absent", "SOURCE_SHA_REQUIRED"),
])
def test_reference_negative_provenance(field, value, code):
    with pytest.raises(ValueError, match=code):
        select([replace(evidence(3), **{field: value}), evidence(4)])


def test_duplicate_evidence_and_causal_use_are_rejected():
    with pytest.raises(ValueError, match="DUPLICATE"):
        select([evidence(3), evidence(3)])
    with pytest.raises(ValueError, match="NOT_CAUSAL"):
        select([evidence(3), evidence(4)], time_mode="CAUSAL_TRAINING_INPUT")


def test_frozen_settings_cannot_be_changed():
    with pytest.raises(ValueError, match="SETTINGS_DRIFT"):
        select([evidence(3), evidence(4)], settings={**SETTINGS, "ref_stride": 2})


def test_reference_selection_does_not_mutate_inputs():
    first, second = evidence(3), evidence(4)
    support, error = first.visible_background.copy(), first.flow_error.copy()
    select([first, second])
    np.testing.assert_array_equal(first.visible_background, support)
    np.testing.assert_array_equal(first.flow_error, error)


def test_clean_zero_change_checks_are_independent_of_robot():
    raw = np.zeros((2, 2, 3, 3), np.uint8)
    clean = raw.copy()
    effective = np.zeros(raw.shape[:-1], bool)
    effective[:, 0, 0] = True
    protected = np.zeros_like(effective)
    protected[:, 1, 2] = True
    clean[:, 0, 0] = 255
    result = validate_clean_preservation(raw, clean, effective, protected)
    assert result["status"] == "PASS"
    assert result["clean_visual_quality"] == "NOT_EVALUATED"
    assert result["robot_coverage_used_for_clean_score"] is False
    clean[1, 1, 2] = 255
    result = validate_clean_preservation(raw, clean, effective, protected)
    assert result["status"] == "FAIL"
    assert result["outside_delete_changed_pixels_per_frame"] == [0, 1]
    assert result["protected_changed_pixels_per_frame"] == [0, 1]
    assert clean[1, 1, 2, 0] == 255  # The evaluator did not paste away the failure.


def test_clean_effective_domain_must_exclude_protection():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    with pytest.raises(ValueError, match="PROTECTED_MASK_ENTERED"):
        validate_clean_preservation(raw, raw.copy(), np.ones(SHAPE, bool), np.ones(SHAPE, bool))


def test_inside_and_outside_residue_are_separate_without_causal_overclaim():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    clean = raw.copy()
    effective, residue, protected = [np.zeros(SHAPE, bool) for _ in range(3)]
    effective[0, :2] = True
    residue[0, :] = True
    protected[0, 2] = True
    clean[0, 0] = 255
    result = classify_residual_pixels(raw, clean, residue, effective, protected)
    assert result["inside_effective_delete_pixels"] == 2
    assert result["outside_effective_delete_pixels"] == 1
    assert result["inside_changed_pixels"] == result["inside_unchanged_pixels"] == 1
    assert result["protected_residual_conflict"].sum() == 1
    assert result["mask_issue_code"] and result["clean_issue_code"]
    assert result["inside_cause"] == "PROPAGATION_COPY_OR_GENERATION_NOT_YET_ATTRIBUTED"
    assert result["producer_may_read_review_annotations"] is False
    assert result["clean_visual_quality"] == "NOT_SCORED_BY_ROBOT_COVERAGE"
