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
        "0915_sam31_mask_full_v1",
        "0915_foundationstereo_full_v1",
        "0915_post_geometry_robot_v1",
    )
    assert predecessor_task(TASK_ORDER[0]) == "0915_0916_input_audit_clean_v1"
    assert predecessor_task(TASK_ORDER[-1]) == TASK_ORDER[-2]


def test_every_algorithm_packet_has_exactly_one_logical_weight() -> None:
    for task_id in TASK_ORDER:
        packet = build_packet(task_id)
        if packet["weights"] == "ABSENT":
            assert task_id in {
                TASK_ORDER[0], TASK_ORDER[1],
                "0915_vst_image_domain_ab_v1", TASK_ORDER[-1],
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


def test_vst_image_domain_packet_is_single_session_weightless() -> None:
    packet = build_packet("0915_vst_image_domain_ab_v1")
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"]["gpu_hours"] == 0
    assert "play_cards_0915_001" in str(packet)
    assert "single-session" in packet["claim_limit"].lower()
