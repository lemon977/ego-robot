from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.pipeline.removal_envelope_v2 import (
    CableInstanceProfileV2,
    RemovalEnvelopeV2Builder,
    RemovalEnvelopeV2Config,
    RemovalEnvelopeV2Error,
    enforce_consumer_firewall,
)


ROOT = Path(__file__).resolve().parents[2]


def hand(center_x: float = 60.0, wrist_y: float = 90.0) -> np.ndarray:
    points = np.zeros((21, 2), np.float64)
    points[0] = (center_x, wrist_y)
    for chain, offset in zip(
        ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
         (13, 14, 15, 16), (17, 18, 19, 20)),
        (-18, -9, 0, 9, 18),
    ):
        for depth, index in enumerate(chain):
            points[index] = (center_x + offset, wrist_y - 14 - depth * 10)
    return points


def semantic(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, bool)
        for name in ("hand", "forearm", "equipment")
    }


def settings(**kwargs: object) -> RemovalEnvelopeV2Config:
    profile = CableInstanceProfileV2(
        profile_id="test_yellow_instance",
        hsv_lower=(20, 150, 100),
        hsv_upper=(35, 255, 255),
        minimum_component_pixels=8,
        maximum_component_pixels=2_000,
        maximum_instances=1,
        minimum_total_score=0.35,
        minimum_anchor_score=0.05,
        anchor_distance_px=45,
        output_dilation_px=1,
    )
    return RemovalEnvelopeV2Config(cable_profile=profile, **kwargs)


def run(
    builder: RemovalEnvelopeV2Builder,
    *,
    masks: dict[str, np.ndarray] | None = None,
    proposals: np.ndarray | None = None,
    sleeve: np.ndarray | None = None,
    cable: np.ndarray | None = None,
    reverse_cable: np.ndarray | None = None,
    frame: np.ndarray | None = None,
    hands: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    background: np.ndarray | None = None,
) -> dict[str, object]:
    shape = builder.shape
    return builder.step(
        frame_bgr=np.zeros((*shape, 3), np.uint8) if frame is None else frame,
        semantic_masks=semantic(shape) if masks is None else masks,
        foreground_proposals=np.zeros(shape, bool) if proposals is None else proposals,
        sleeve_candidate_mask=np.zeros(shape, bool) if sleeve is None else sleeve,
        cable_candidate_mask=np.zeros(shape, bool) if cable is None else cable,
        reverse_cable_support_mask=reverse_cable,
        task_object_mask=np.zeros(shape, bool),
        joints_2d=np.stack((hand(45), hand(85))) if hands is None else hands,
        observed=np.ones(2, bool) if observed is None else observed,
        stable_background_mask=background,
    )


def test_unknown_weak_evidence_cannot_create_removal_without_sam_base() -> None:
    shape = (130, 130)
    builder = RemovalEnvelopeV2Builder(shape, settings())
    result = run(builder, background=np.zeros(shape, bool))
    assert not result["semantic_base"].any()
    assert not result["mano_gap_repair"].any()
    assert not result["selected_forearm"].any()
    assert not result["removal_envelope"].any()
    assert result["quality"]["status"] == "UNKNOWN_FAIL_CLOSED"


def test_mano_only_repairs_short_local_sam_gap() -> None:
    shape = (130, 130)
    masks = semantic(shape)
    joints = np.stack((hand(45), hand(85)))
    # Endpoint islands around one MANO phalanx leave a short, closable gap.
    for index in (5, 6):
        xy = tuple(np.rint(joints[0, index]).astype(np.int32))
        cv2.circle(masks["hand"].view(np.uint8), xy, 3, 1, -1)
    builder = RemovalEnvelopeV2Builder(shape, settings())
    result = run(
        builder, masks=masks, hands=joints,
        background=np.zeros(shape, bool),
    )
    repair = result["mano_gap_repair"]
    assert repair.any()
    assert np.all(~repair | cv2.dilate(
        masks["hand"].astype(np.uint8), np.ones((33, 33), np.uint8),
    ).astype(bool))
    assert int(repair.sum()) < 300
    assert result["provenance"]["policy"]["full_mano_capsule_generation"] is False


def test_forearm_is_selected_from_real_wrist_connected_proposal_only() -> None:
    shape = (130, 130)
    masks = semantic(shape)
    masks["hand"][70:100, 35:55] = True
    masks["forearm"][88:122, 38:53] = True
    joints = np.stack((hand(45, 90), hand(90, 90)))
    no_proposal = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks, hands=joints,
        background=np.zeros(shape, bool),
    )
    assert not no_proposal["selected_forearm"].any()

    proposal = np.zeros(shape, bool)
    proposal[88:122, 38:53] = True
    selected = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks,
        proposals=proposal, hands=joints, background=np.zeros(shape, bool),
    )
    assert np.array_equal(selected["selected_forearm"], proposal)
    assert selected["provenance"]["forearm"]["state"] == "SEEDED_WRIST_CONNECTED_PROPOSAL"
    assert selected["provenance"]["policy"]["wrist_to_image_boundary_generation"] is False


def test_missing_forearm_proposal_is_unknown_and_not_forward_filled() -> None:
    shape = (130, 130)
    masks = semantic(shape)
    masks["hand"][70:100, 35:55] = True
    masks["forearm"][88:122, 38:53] = True
    proposal = np.zeros(shape, bool)
    proposal[88:122, 38:53] = True
    builder = RemovalEnvelopeV2Builder(shape, settings())
    first = run(
        builder, masks=masks, proposals=proposal,
        hands=np.stack((hand(45, 90), hand(90, 90))),
        background=np.zeros(shape, bool),
    )
    assert first["selected_forearm"].any()
    missing = run(
        builder, masks=masks,
        hands=np.stack((hand(45, 90), hand(90, 90))),
        background=np.zeros(shape, bool),
    )
    assert not missing["selected_forearm"].any()
    assert missing["provenance"]["forearm"]["state"].startswith("UNKNOWN")


def test_sleeve_must_attach_to_hand_boundary_and_mano_direction() -> None:
    shape = (140, 140)
    masks = semantic(shape)
    masks["hand"][45:95, 30:70] = True
    joints = np.stack((hand(50, 95), hand(100, 95)))
    attached = np.zeros(shape, bool)
    attached[48:69, 25:35] = True
    result = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks,
        sleeve=attached, hands=joints, background=np.zeros(shape, bool),
    )
    assert result["attached_sleeve_repair"].any()

    detached = np.zeros(shape, bool)
    detached[5:20, 110:125] = True
    rejected = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks,
        sleeve=detached, hands=joints, background=np.zeros(shape, bool),
    )
    assert not rejected["attached_sleeve_repair"].any()
    assert rejected["provenance"]["sleeve"]["state"].startswith("UNKNOWN")


def test_cable_selects_top_one_scored_instance_not_all_yellow_blobs() -> None:
    shape = (150, 180)
    masks = semantic(shape)
    masks["hand"][70:115, 25:65] = True
    frame = np.zeros((*shape, 3), np.uint8)
    cv2.line(frame, (55, 85), (105, 100), (0, 255, 255), 3)
    cv2.line(frame, (135, 10), (170, 15), (0, 255, 255), 3)
    cv2.circle(frame, (150, 70), 8, (0, 255, 255), -1)
    result = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks, frame=frame,
        hands=np.stack((hand(45, 100), hand(90, 100))),
        background=np.zeros(shape, bool),
    )
    cable = result["cable_instance"]
    record = result["provenance"]["cable"]
    assert cable.any()
    assert record["raw_candidate_components"] >= 3
    assert record["accepted_components"] == 1
    assert not cable[10:20, 130:179].any()
    assert not cable[60:80, 140:160].any()
    assert set(record["accepted"][0]["scores"]) == {
        "anchor", "thinness", "path", "endpoint", "forward_backward_temporal",
    }
    assert result["provenance"]["policy"]["all_appearance_components_accepted"] is False


def test_missing_candidate_is_unknown_and_never_forward_filled() -> None:
    shape = (130, 150)
    masks = semantic(shape)
    masks["hand"][65:105, 20:60] = True
    builder = RemovalEnvelopeV2Builder(shape, settings())
    frame = np.zeros((*shape, 3), np.uint8)
    cv2.line(frame, (50, 80), (100, 90), (0, 255, 255), 3)
    seeded = run(builder, masks=masks, frame=frame, background=np.zeros(shape, bool))
    assert seeded["cable_instance"].any()
    missing = run(builder, masks=masks, background=np.zeros(shape, bool))
    assert not missing["cable_instance"].any()
    assert missing["provenance"]["cable"]["state"] == "UNKNOWN_NO_VALID_INSTANCE"
    assert missing["provenance"]["cable"]["unbounded_hold_used"] is False


def test_cable_tracking_requires_reverse_support_after_seed() -> None:
    shape = (130, 150)
    masks = semantic(shape)
    masks["hand"][65:105, 20:60] = True
    builder = RemovalEnvelopeV2Builder(shape, settings())
    first_frame = np.zeros((*shape, 3), np.uint8)
    cv2.line(first_frame, (50, 80), (100, 90), (0, 255, 255), 3)
    assert run(
        builder, masks=masks, frame=first_frame,
        background=np.zeros(shape, bool),
    )["cable_instance"].any()

    moved_frame = np.zeros((*shape, 3), np.uint8)
    cv2.line(moved_frame, (52, 80), (102, 90), (0, 255, 255), 3)
    forward_only = run(
        builder, masks=masks, frame=moved_frame,
        background=np.zeros(shape, bool),
    )
    assert not forward_only["cable_instance"].any()
    reverse = np.zeros(shape, np.uint8)
    cv2.line(reverse, (52, 80), (102, 90), 1, 3)
    bidirectional = run(
        builder, masks=masks, frame=moved_frame,
        reverse_cable=reverse.astype(bool), background=np.zeros(shape, bool),
    )
    assert bidirectional["cable_instance"].any()
    record = bidirectional["provenance"]["cable"]
    assert record["state"] == "TRACKED_TOP_K_IDENTITY"
    assert record["reverse_support_supplied"] is True
    temporal = record["accepted"][0]["scores"]["forward_backward_temporal"]
    assert temporal is not None and temporal > 0.5


def test_quality_reports_contributions_and_fails_on_background_spill() -> None:
    shape = (100, 100)
    masks = semantic(shape)
    masks["hand"][30:70, 30:70] = True
    result = run(
        RemovalEnvelopeV2Builder(shape, settings()), masks=masks,
        hands=np.stack((hand(45, 80), hand(70, 80))),
        background=np.ones(shape, bool),
    )
    quality = result["quality"]
    assert quality["status"] == "REJECTED_QUALITY"
    assert quality["gates"]["background_spill"]["status"] == "FAIL"
    assert set(quality["metrics"]) == {
        "semantic_base_pixels", "removal_pixels", "repair_added_pixels",
        "area_inflation", "background_spill_ratio",
        "temporal_area_derivative", "repair_contribution_ratio",
        "protected_object_core_pixels", "protected_object_core_damage_pixels",
        "protected_object_core_damage_ratio",
    }
    assert set(result["provenance"]["exclusive_repair_contribution_pixels"]) == {
        "validated_mano_gap_repair", "validated_attached_sleeve_repair",
        "validated_cable_instance",
    }
    assert np.array_equal(result["source_bits"] != 0, result["removal_envelope"])


def test_visible_object_core_is_protected_without_preserving_hand_boundary() -> None:
    shape = (140, 140)
    masks = semantic(shape)
    masks["hand"][55:95, 30:75] = True
    masks["equipment"][60:105, 60:110] = True
    task_object = np.zeros(shape, bool)
    task_object[65:115, 70:125] = True
    builder = RemovalEnvelopeV2Builder(shape, settings())
    result = builder.step(
        frame_bgr=np.zeros((*shape, 3), np.uint8),
        semantic_masks=masks,
        foreground_proposals=np.zeros(shape, bool),
        sleeve_candidate_mask=np.zeros(shape, bool),
        cable_candidate_mask=np.zeros(shape, bool),
        reverse_cable_support_mask=None,
        task_object_mask=task_object,
        joints_2d=np.stack((hand(45), hand(95))),
        observed=np.ones(2, bool),
        stable_background_mask=np.zeros(shape, bool),
    )
    core = result["protected_object_core"]
    assert core.any()
    assert not np.any(result["removal_envelope"] & core)
    # The hand/object interaction boundary remains removable; only a visible
    # interior away from the hand is protected.
    assert np.any(result["removal_envelope"] & task_object)
    gate = result["quality"]["gates"]["protected_object_core_damage"]
    assert gate["status"] == "PASS"
    assert gate["value"] == 0.0


def test_missing_background_hook_keeps_quality_unknown() -> None:
    shape = (100, 100)
    masks = semantic(shape)
    masks["hand"][30:70, 30:70] = True
    result = run(RemovalEnvelopeV2Builder(shape, settings()), masks=masks)
    assert result["quality"]["status"] == "UNKNOWN_FAIL_CLOSED"
    assert result["quality"]["gates"]["background_spill"]["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "consumer", ["DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY", "CONTROL_GROUND_TRUTH"],
)
def test_visual_only_outputs_are_firewalled(consumer: str) -> None:
    with pytest.raises(RemovalEnvelopeV2Error, match="visual-only"):
        enforce_consumer_firewall("removal_envelope_v2", consumer)
    enforce_consumer_firewall("feather_alpha_v2", "VISUAL_INPAINT")


def test_schema_and_config_encode_conservative_v2_contract() -> None:
    contract = json.loads((ROOT / "contracts/removal_envelope_v2.schema.json").read_text())
    config = json.loads((
        ROOT / "configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json"
    ).read_text())
    assert contract["properties"]["design"]["const"] == "SAM_BASE_PLUS_VALIDATED_LOCAL_REPAIRS"
    assert config["unknown_behavior"] == "EMPTY_REPAIR_FAIL_CLOSED"
    assert set(config["forbidden_generators"]) == {
        "FULL_MANO_CAPSULE", "WRIST_TO_IMAGE_BOUNDARY", "ALL_HSV_COMPONENT_UNION",
    }
    assert config["cable_profile"]["maximum_instances"] <= 2
    forbidden = config["consumer_firewall"]["forbidden"]
    assert {"DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY"} <= set(forbidden)
