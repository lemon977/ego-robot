from dataclasses import replace

import numpy as np
import pytest

from chaoyang.inpainting_candidate.automatic_masks import (
    AutomaticMaskContractError, RGBProposal, frames_sha256,
)
from chaoyang.inpainting_candidate.automatic_v1_seeds import (
    SEED_AUTHORITY, generate_v1_seed_bundle,
)
from chaoyang.pick_place_1008.sam_point_adapter import _validated_inputs

SHA = "a" * 64


def fixture():
    rgb = np.zeros((4, 48, 64, 3), np.uint8)
    left = np.zeros(rgb.shape[:3], bool)
    right = left.copy()
    obj = left.copy()
    left[:, 16:35, 3:18] = True
    right[:, 16:35, 44:59] = True
    obj[:, 2:10, 28:36] = True
    hints = np.full((2, 4, 1, 2), np.nan)
    hints[0, :, 0] = [10, 25]
    hints[1, :, 0] = [51, 25]
    props = [RGBProposal(role, track, mask, frames_sha256(rgb), SHA, SHA, SHA)
             for role, track, mask in (("glove", 1, left), ("glove", 2, right), ("object", 3, obj))]
    return rgb, props, hints


def generate(rgb, props, hints):
    return generate_v1_seed_bundle(rgb, props, hints, raw_source_sha256=SHA,
                                   frame_map_sha256=SHA, model_sha256=SHA)


def test_multi_positive_negative_and_v1_adapter_contract():
    rgb, props, hints = fixture()
    out = generate(rgb, props, hints)
    assert out["status"] == "READY_MODEL_CANDIDATE"
    assert len(out["objects"]) == 2
    for obj in out["objects"]:
        assert obj["point_labels"].count(1) == 3
        assert 1 <= obj["point_labels"].count(0) <= 8
        mask = next(p.masks[obj["frame_idx"]] for p in props if p.track_id == obj["proposal_track_id"])
        for (x, y), label in zip(obj["points_px"], obj["point_labels"]):
            assert bool(mask[int(y), int(x)]) == bool(label)
        assert obj["human_visible_confirmed"] is False
    _validated_inputs(rgb, out["objects"], out["anchor_frame"])
    assert out["seed_authority"] == SEED_AUTHORITY
    assert out["manual_seed_reads"] == out["historical_mask_reads"] == 0


def test_no_identity_hints_is_unknown_not_guessed_side():
    rgb, props, _ = fixture()
    out = generate(rgb, props, None)
    assert out["status"] == "IDENTITY_UNKNOWN_NO_SEEDS"
    assert not out["objects"]


def test_equal_side_evidence_abstains():
    rgb, props, hints = fixture()
    hints[:] = [10, 25]
    out = generate(rgb, props[:1], hints)
    assert not out["objects"]


def test_unique_highest_side_rate_uses_existing_gate():
    rgb, props, hints = fixture()
    hints[1, :1] = hints[0, :1]
    out = generate(rgb, props[:1], hints)
    assert out["objects"][0]["side"] == "left"
    assert out["instance_association"][0]["side_hit_rates"] == [1.0, .25]


@pytest.mark.parametrize("source", ["SAM31_RGB_POINT_AUTO", "SAM31_RGB_TEXT_POINT_WEARER_UNION"])
def test_historical_point_or_merge_sources_rejected(source):
    rgb, props, hints = fixture()
    with pytest.raises(AutomaticMaskContractError, match="ONLY_RGB_TEXT"):
        generate(rgb, [replace(props[0], source=source)], hints)


def test_raw_rgb_drift_rejected():
    rgb, props, hints = fixture()
    rgb[0, 0, 0, 0] = 1
    with pytest.raises(AutomaticMaskContractError, match="SIGNATURE"):
        generate(rgb, props, hints)


def test_input_binding_drift_rejected():
    rgb, props, hints = fixture()
    with pytest.raises(AutomaticMaskContractError, match="SIGNATURE"):
        generate(rgb, [replace(props[0], raw_source_sha256="b" * 64)], hints)


def test_unsupported_native_frame_cannot_supply_anchor():
    rgb, props, hints = fixture()
    props[0] = replace(props[0], unsupported_frames=np.asarray([True, True, False, False]))
    out = generate(rgb, props[:1], hints)
    assert out["objects"][0]["frame_idx"] == 2
    assert out["instance_association"][0]["selected_pixels_per_native_frame"][:2] == [0, 0]


def test_proposal_order_independent_and_inputs_not_mutated():
    rgb, props, hints = fixture()
    before = [p.masks.copy() for p in props]
    assert generate(rgb, props, hints) == generate(rgb, list(reversed(props)), hints)
    assert all(np.array_equal(p.masks, old) for p, old in zip(props, before))


def test_projection_cannot_create_foreground_pixels():
    rgb, props, hints = fixture()
    props[0] = replace(props[0], masks=np.zeros_like(props[0].masks))
    out = generate(rgb, props[:1], hints)
    assert not out["objects"]
    assert sum(out["instance_association"][0]["selected_pixels_per_native_frame"]) == 0


def test_unknown_background_instance_can_supply_only_negative():
    rgb, props, hints = fixture()
    bg = np.zeros(rgb.shape[:3], bool)
    bg[:, 1:9, 1:9] = True
    props.append(RGBProposal("glove", 8, bg, frames_sha256(rgb), SHA, SHA, SHA))
    out = generate(rgb, props, hints)
    assert not any(obj["proposal_track_id"] == 8 for obj in out["objects"])
    assert any(source["track_id"] == 8 for obj in out["objects"]
               for source in obj["negative_proposal_sources"])


def test_worn_equipment_association_is_current_rgb_only():
    rgb, props, hints = fixture()
    mask = np.zeros(rgb.shape[:3], bool)
    mask[:, 35:38, 8:15] = True
    props.append(RGBProposal("attachment", 9, mask, frames_sha256(rgb), SHA, SHA, SHA))
    out = generate(rgb, props, hints)
    part = next(obj for obj in out["objects"] if obj["proposal_track_id"] == 9)
    assert part["side"] == "left"
    for (x, y), label in zip(part["points_px"], part["point_labels"]):
        if label:
            assert mask[part["frame_idx"], int(y), int(x)]


def test_duplicate_role_track_rejected():
    rgb, props, hints = fixture()
    with pytest.raises(AutomaticMaskContractError, match="DUPLICATE"):
        generate(rgb, [props[0], props[0]], hints)


def test_invalid_hint_domain_rejected():
    rgb, props, hints = fixture()
    with pytest.raises(AutomaticMaskContractError, match="PROJECTION_HINTS"):
        generate(rgb, props, hints[0])
