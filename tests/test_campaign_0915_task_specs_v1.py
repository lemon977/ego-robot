from __future__ import annotations

import copy

import pytest

from chaoyang.governance.campaign_0915_task_specs_v1 import (
    TASK_ORDER,
    build_packet,
    predecessor_task,
    validate_packet_policy,
)


def test_campaign_has_finite_serial_order() -> None:
    assert TASK_ORDER == (
        "0915_input_prepare_cad_v1",
        "0915_input_prepare_cad_v2",
        "0915_hawor_full_v1",
        "0915_vst_image_domain_ab_v1",
        "0915_hawor_resize_only_canary_v1",
        "0915_sam31_strict_role_canary_v1",
        "0915_sam31_strict_role_canary_v2",
        "0915_sam31_strict_role_canary_v3",
        "0915_sam31_strict_role_canary_v4",
        "0915_sam31_strict_role_canary_v5",
        "0915_stereo_interaction_cpu_canary_v1",
        "0915_sam31_weak_role_canary_v1",
        "0915_removal_envelope_single_session_canary_v1",
        "0915_foundationstereo_single_session_canary_v1",
        "0915_planar_object6d_single_session_canary_v1",
        "0915_sam31_mask_full_v1",
        "0915_foundationstereo_full_v1",
        "0915_post_geometry_robot_v1",
        "0915_stereo_encoded_domain_preflight_v1",
        "0915_removal_envelope_v2_real_canary_v1",
        "0915_foundationstereo_encoded_domain_canary_v1",
        "0915_planar_object6d_observability_canary_v2",
        "0915_interaction_contact_robot_dev_v1",
        "0915_interaction_contact_robot_dev_v2",
        "0915_interaction_contact_robot_dev_v3",
    )
    assert predecessor_task(TASK_ORDER[0]) == "0915_0916_input_audit_clean_v1"
    assert predecessor_task("0915_stereo_encoded_domain_preflight_v1") == (
        "0915_foundationstereo_single_session_canary_v1"
    )
    assert predecessor_task("0915_foundationstereo_encoded_domain_canary_v1") == (
        "0915_stereo_encoded_domain_preflight_v1"
    )
    assert predecessor_task("0915_planar_object6d_observability_canary_v2") == (
        "0915_foundationstereo_encoded_domain_canary_v1"
    )
    assert predecessor_task(TASK_ORDER[-1]) == "0915_interaction_contact_robot_dev_v2"


def test_every_algorithm_packet_has_exactly_one_logical_weight() -> None:
    for task_id in TASK_ORDER:
        packet = build_packet(task_id)
        if packet["weights"] == "ABSENT":
            assert task_id in {
                TASK_ORDER[0], TASK_ORDER[1],
                "0915_vst_image_domain_ab_v1",
                "0915_stereo_interaction_cpu_canary_v1",
                "0915_removal_envelope_single_session_canary_v1",
                "0915_planar_object6d_single_session_canary_v1",
                "0915_post_geometry_robot_v1",
                "0915_stereo_encoded_domain_preflight_v1",
                "0915_removal_envelope_v2_real_canary_v1",
                "0915_planar_object6d_observability_canary_v2",
                "0915_interaction_contact_robot_dev_v1",
                "0915_interaction_contact_robot_dev_v2",
                "0915_interaction_contact_robot_dev_v3",
            }
        else:
            assert len(packet["weights"]) == 1
        assert len(packet["read_set"]) <= 8


def test_mask_packet_is_sam31_only_without_selection_branch() -> None:
    packet = build_packet("0915_sam31_mask_full_v1")
    assert packet["weights"] == ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"]
    encoded = str(packet).lower()
    assert "sam2" not in encoded
    assert "cutie" not in encoded
    assert "sam3.1_only_user_locked" in encoded
    assert "winner" in encoded  # explicitly forbidden in the claim boundary


def test_mask_packet_rejects_added_competitor() -> None:
    packet = build_packet("0915_sam31_mask_full_v1")
    bad = copy.deepcopy(packet)
    bad["weights"].append("competitor.pt")
    with pytest.raises(ValueError, match="exactly one"):
        validate_packet_policy(bad)
    bad = copy.deepcopy(packet)
    bad["objective"] += " Cutie"
    with pytest.raises(ValueError, match="alternate model"):
        validate_packet_policy(bad)


def test_bounded_canaries_separate_routing_from_algorithm_dependencies() -> None:
    cpu = build_packet("0915_stereo_interaction_cpu_canary_v1")
    weak = build_packet("0915_sam31_weak_role_canary_v1")
    removal = build_packet("0915_removal_envelope_single_session_canary_v1")
    depth = build_packet("0915_foundationstereo_single_session_canary_v1")
    object6d = build_packet("0915_planar_object6d_single_session_canary_v1")
    assert cpu["weights"] == removal["weights"] == object6d["weights"] == "ABSENT"
    assert weak["weights"] == [
        "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
    ]
    assert depth["weights"] == [
        "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
    ]
    assert "not_dependencies" in weak["algorithm_prerequisites"]
    assert "not_dependencies" in depth["algorithm_prerequisites"]
    assert predecessor_task(removal["task_id"]) == weak["task_id"]
    assert predecessor_task(depth["task_id"]) == removal["task_id"]
    assert depth["algorithm_prerequisites"]["depth"] == [
        "stereo_preflight_PASS_GPU_DEPTH_ADMISSION"
    ]


def test_vst_image_domain_packet_is_single_session_weightless() -> None:
    packet = build_packet("0915_vst_image_domain_ab_v1")
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"]["gpu_hours"] == 0
    assert "play_cards_0915_001" in str(packet)
    assert "single-session" in packet["claim_limit"].lower()


def test_resize_only_hawor_canary_is_single_session_one_weight() -> None:
    packet = build_packet("0915_hawor_resize_only_canary_v1")
    assert packet["budgets"]["gpu_hours"] == 1
    assert packet["weights"] == [
        "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"
    ]
    encoded = str(packet)
    assert "play_cards_0915_001" in encoded
    assert "PICO26_NOT_CONSUMED" in encoded


def test_encoded_object6d_v2_packet_is_cpu_only_and_entity_locked() -> None:
    packet = build_packet("0915_planar_object6d_observability_canary_v2")
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"]["gpu_hours"] == 0
    assert predecessor_task(packet["task_id"]) == (
        "0915_foundationstereo_encoded_domain_canary_v1"
    )
    encoded = str(packet)
    assert "three_cards_independent" in encoded
    assert "black_tray_separate_support_entity" in encoded
    assert "card_set_semantic_only" in encoded
    assert "old_rectified_depth" in encoded
