from __future__ import annotations

import numpy as np
import pytest

from chaoyang.pipeline.removal_envelope_v1 import (
    RemovalEnvelopeBuilder,
    RemovalEnvelopeConfig,
    RemovalEnvelopeError,
    CableAppearanceProfile,
    bounded_internal_geometry,
    enforce_consumer_firewall,
)


def hand(center_x: float = 50.0, wrist_y: float = 75.0, scale: float = 1.0) -> np.ndarray:
    points = np.zeros((21, 2), np.float64)
    points[0] = (center_x, wrist_y)
    chains = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
              (13, 14, 15, 16), (17, 18, 19, 20))
    x_offsets = (-18, -10, 0, 10, 18)
    for chain, offset in zip(chains, x_offsets):
        for depth, index in enumerate(chain):
            points[index] = (
                center_x + offset * scale,
                wrist_y - (12 + depth * 10) * scale,
            )
    return points


def empty_semantic(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {name: np.zeros(shape, bool) for name in ("hand", "forearm", "sleeve", "cable")}


def config(**kwargs: object) -> RemovalEnvelopeConfig:
    profile = CableAppearanceProfile(
        profile_id="test_no_match", hsv_lower=(1, 255, 255),
        hsv_upper=(1, 255, 255),
    )
    return RemovalEnvelopeConfig(cable_profile=profile, **kwargs)


def test_bounded_geometry_only_fills_two_sided_short_verified_gaps() -> None:
    joints = np.stack([[hand() for _ in range(7)] for _ in range(2)])
    observed = np.ones((2, 7), bool)
    observed[0, 2:4] = False
    observed[1, :2] = False
    confidence = np.ones((2, 7), np.float64) * 0.8
    filled, states = bounded_internal_geometry(joints, observed, confidence)
    assert states[0, 2:4].tolist() == ["BOUNDED_INTERNAL_HOLD"] * 2
    assert states[1, :2].tolist() == ["UNKNOWN", "UNKNOWN"]
    assert np.isfinite(filled[0, 2:4]).all()
    assert not observed[0, 2:4].any()

    confidence[0, 1] = 0.1
    _, states = bounded_internal_geometry(joints, observed, confidence)
    assert states[0, 2:4].tolist() == ["UNKNOWN", "UNKNOWN"]


def test_invalid_geometry_does_not_expand_or_forward_fill() -> None:
    shape = (120, 120)
    builder = RemovalEnvelopeBuilder(shape, config())
    frame = np.zeros((*shape, 3), np.uint8)
    direct = builder.step(
        frame_bgr=frame,
        joints_2d=np.stack((hand(42), hand(78))),
        observed=np.asarray([True, True]),
        detector_confidence=np.asarray([0.8, 0.8]),
        geometry_states=np.asarray(["OBSERVED", "OBSERVED"]),
        semantic_masks=empty_semantic(shape),
        task_object_mask=np.zeros(shape, bool),
    )
    unknown = builder.step(
        frame_bgr=frame,
        joints_2d=np.full((2, 21, 2), np.nan),
        observed=np.asarray([False, False]),
        detector_confidence=np.asarray([0.0, 0.0]),
        geometry_states=np.asarray(["UNKNOWN", "UNKNOWN"]),
        semantic_masks=empty_semantic(shape),
        task_object_mask=np.zeros(shape, bool),
    )
    assert direct["mano_finger_capsules"].any()
    assert not unknown["mano_finger_capsules"].any()
    assert not unknown["palm_wrist_forearm_corridors"].any()


def test_low_confidence_radius_cannot_grow() -> None:
    shape = (180, 180)
    settings = config(maximum_finger_radius_px=50.0)
    builder = RemovalEnvelopeBuilder(shape, settings)
    frame = np.zeros((*shape, 3), np.uint8)
    first = builder.step(
        frame_bgr=frame,
        joints_2d=np.stack((hand(60, 130, 0.8), hand(120, 130, 0.8))),
        observed=np.ones(2, bool), detector_confidence=np.ones(2) * 0.9,
        geometry_states=np.asarray(["OBSERVED", "OBSERVED"]),
        semantic_masks=empty_semantic(shape), task_object_mask=np.zeros(shape, bool),
    )
    second = builder.step(
        frame_bgr=frame,
        joints_2d=np.stack((hand(60, 130, 1.5), hand(120, 130, 1.5))),
        observed=np.ones(2, bool), detector_confidence=np.ones(2) * 0.1,
        geometry_states=np.asarray(["OBSERVED", "OBSERVED"]),
        semantic_masks=empty_semantic(shape), task_object_mask=np.zeros(shape, bool),
    )
    first_radii = [row["finger_radius_px"] for row in first["provenance"]["hands"]]
    second_radii = [row["finger_radius_px"] for row in second["provenance"]["hands"]]
    assert all(after <= before for before, after in zip(first_radii, second_radii))


def test_optional_sleeve_absence_does_not_block_and_object_core_is_protected() -> None:
    shape = (140, 140)
    profile = CableAppearanceProfile(
        profile_id="synthetic_yellow_device", hsv_lower=(20, 200, 200),
        hsv_upper=(30, 255, 255),
    )
    builder = RemovalEnvelopeBuilder(shape, RemovalEnvelopeConfig(cable_profile=profile))
    frame = np.zeros((*shape, 3), np.uint8)
    semantic = empty_semantic(shape)
    semantic["hand"][95:125, 20:45] = True
    task_object = np.zeros(shape, bool)
    task_object[10:35, 90:125] = True
    result = builder.step(
        frame_bgr=frame,
        joints_2d=np.stack((hand(35, 105, 0.65), hand(65, 105, 0.65))),
        observed=np.ones(2, bool), detector_confidence=np.ones(2) * 0.8,
        geometry_states=np.asarray(["OBSERVED", "OBSERVED"]),
        semantic_masks=semantic, task_object_mask=task_object,
    )
    assert result["removal_envelope"].any()
    assert not np.any(result["removal_envelope"] & result["protected_visible_object"])
    assert np.array_equal(result["source_bits"] != 0, result["removal_envelope"])
    assert np.all(result["feather_alpha"][result["removal_envelope"]] == 1.0)


def test_cable_appearance_profile_is_replaceable_and_not_semantic_evidence() -> None:
    shape = (120, 120)
    profile = CableAppearanceProfile(
        profile_id="synthetic_green_device", hsv_lower=(50, 180, 100),
        hsv_upper=(80, 255, 255), hand_anchor_radius_px=40,
    )
    builder = RemovalEnvelopeBuilder(shape, RemovalEnvelopeConfig(cable_profile=profile))
    frame = np.zeros((*shape, 3), np.uint8)
    frame[65:72, 30:85] = (0, 255, 0)
    result = builder.step(
        frame_bgr=frame,
        joints_2d=np.stack((hand(45, 90, 0.55), hand(75, 90, 0.55))),
        observed=np.ones(2, bool), detector_confidence=np.ones(2) * 0.8,
        geometry_states=np.asarray(["OBSERVED", "OBSERVED"]),
        semantic_masks=empty_semantic(shape), task_object_mask=np.zeros(shape, bool),
    )
    assert result["cable_tracked_region"].any()
    assert result["provenance"]["cable"]["appearance_profile_id"] == "synthetic_green_device"


@pytest.mark.parametrize("consumer", [
    "DEPTH", "OBJECT6D", "CONTACT", "ROBOT_GEOMETRY", "CONTROL_GROUND_TRUTH",
])
def test_visual_only_artifacts_are_rejected_by_geometry_consumers(consumer: str) -> None:
    with pytest.raises(RemovalEnvelopeError, match="visual-only"):
        enforce_consumer_firewall("removal_envelope", consumer)
    enforce_consumer_firewall("feather_alpha", "VISUAL_INPAINT")
