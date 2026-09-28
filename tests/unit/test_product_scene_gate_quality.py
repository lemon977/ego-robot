import pytest

from chaoyang.ops.run_human_to_robot_baseline_v1 import validate_scene_for_product
from chaoyang.ops.run_human_to_robot_quality_acceptance import assess_product_gates


def test_technical_clean_pass_does_not_require_prior_user_adoption():
    validate_scene_for_product(
        {"quality": "PASS", "adoption": "NOT_ADOPTED"},
        {"product_mode": "STRICT_PRODUCT"},
    )


def test_rejected_clean_only_explicit_offline_candidate():
    with pytest.raises(ValueError):
        validate_scene_for_product({"quality": "REJECTED_QUALITY"},
                                   {"product_mode": "STRICT_PRODUCT"})
    with pytest.raises(ValueError):
        validate_scene_for_product({"quality": "REJECTED_QUALITY"},
                                   {"product_mode": "CANDIDATE_ONLY"})
    validate_scene_for_product({"quality": "REJECTED_QUALITY"},
                               {"product_mode": "CANDIDATE_ONLY", "allow_rejected_scene": True})


def test_unknown_scene_quality_does_not_count_as_pass():
    with pytest.raises(ValueError):
        validate_scene_for_product({"quality": "INCONCLUSIVE"},
                                   {"product_mode": "STRICT_PRODUCT"})


def test_technical_pass_is_possible_but_never_automatically_adopted():
    gates = {name: True for name in (
        "source_time_domain", "motion_tracking", "finger_motion",
        "scene_clean", "occlusion", "full_decode_and_sha", "visual_review")}
    result = assess_product_gates(gates, product_sha="specified_sha")
    assert result["quality"] == "PASS"
    assert result["adoption"] == "NOT_ADOPTED"
    assert assess_product_gates(gates, product_sha="specified_sha",
                                user_confirmed_sha="specified_sha")["adoption"] == "ADOPTED"


@pytest.mark.parametrize("missing", ["scene_clean", "occlusion", "motion_tracking"])
def test_missing_quality_evidence_cannot_be_silently_skipped(missing):
    gates = {name: True for name in (
        "source_time_domain", "motion_tracking", "finger_motion",
        "scene_clean", "occlusion", "full_decode_and_sha", "visual_review")}
    gates[missing] = False
    result = assess_product_gates(gates, product_sha="specified_sha",
                                  user_confirmed_sha="specified_sha")
    assert result == {"quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
                      "failed_or_missing_gates": [missing]}
