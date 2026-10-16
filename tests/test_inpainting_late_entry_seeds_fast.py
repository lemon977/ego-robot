"""Synthetic CPU checks; the provider is a fake and never loads a model."""
from dataclasses import replace
import inspect

import numpy as np
import pytest

from chaoyang.inpainting_candidate.automatic_masks import (
    AutomaticMaskContractError, RGBProposal, frames_sha256,
)
from chaoyang.inpainting_candidate.automatic_v1_seeds import generate_v1_seed_bundle
from chaoyang.inpainting_candidate.late_entry_seeds import (
    refresh_controller_proposals, select_device_requery_anchors,
)


SHA = "a" * 64
BINDINGS = {
    "raw_source_sha256": SHA,
    "frame_map_sha256": SHA,
    "model_sha256": SHA,
}


def proposal(rgb, role, track_id, masks, **kwargs):
    return RGBProposal(role, track_id, masks, frames_sha256(rgb), SHA, SHA, SHA,
                       **kwargs)


def fixture():
    rgb = np.zeros((6, 48, 64, 3), np.uint8)
    rgb[:, 0, 0, 0] = np.arange(len(rgb))
    left = np.zeros(rgb.shape[:3], bool)
    right = left.copy()
    left[:, 16:30, 4:16] = True
    left[2:4, 16:34, 3:19] = True
    right[:, 16:30, 46:58] = True
    right[4, 14:36, 43:61] = True
    hints = np.full((2, len(rgb), 1, 2), np.nan)
    hints[0, :, 0] = [10, 25]
    hints[1, :, 0] = [51, 25]
    props = [proposal(rgb, "glove", 1, left),
             proposal(rgb, "glove", 2, right)]
    return rgb, props, hints


def packed_tracks(masks, track_id=7):
    return {track_id: {frame: np.packbits(mask)
                       for frame, mask in enumerate(masks) if mask.any()}}


class FakeSam:
    def __init__(self, outputs=None):
        self.outputs = {} if outputs is None else outputs
        self.calls = []
        self.aborted = []

    def track_text(self, frames_rgb, prompt, anchor):
        frames = np.asarray(frames_rgb)
        self.calls.append((frames.copy(), prompt, anchor))
        return self.outputs(anchor) if callable(self.outputs) else self.outputs


def select(rgb, props, hints):
    return select_device_requery_anchors(rgb, props, hints, **BINDINGS)


def refresh(sam, rgb, props, hints):
    return refresh_controller_proposals(sam, rgb, props, hints, **BINDINGS)


def bundle(rgb, props, hints):
    return generate_v1_seed_bundle(rgb, props, hints, **BINDINGS)


def test_anchor_uses_same_side_union_earliest_tie_and_excludes_frame_zero():
    rgb, _, hints = fixture()
    first = np.zeros(rgb.shape[:3], bool)
    second = first.copy()
    first[:, 20:30, 5:15] = True
    second[:, 20:30, 5:15] = True
    first[0, 18:35, 2:19] = True
    second[0] = first[0]
    for frame in (1, 3):
        first[frame, 20:30, 3:16] = True
        second[frame, 20:30, 13:26] = True
    first[2, 16:30, 3:19] = True
    second[2] = first[2]
    # Summing instance areas would select 2; the union ties at 1 and 3.
    props = [proposal(rgb, "glove", 1, first),
             proposal(rgb, "human", 9, second)]
    anchors, receipt = select(rgb, props, hints)
    assert anchors == [1]
    assert isinstance(receipt, dict)


def test_two_native_side_anchors_are_deduplicated_and_order_independent():
    rgb, props, hints = fixture()
    assert select(rgb, props, hints)[0] == [2, 4]
    assert select(rgb, list(reversed(props)), hints)[0] == [2, 4]
    right = props[1].masks.copy()
    right[2] = right[4]
    right[4] = right[1]
    assert select(rgb, [props[0], replace(props[1], masks=right)], hints)[0] == [2]


def test_existing_nonempty_controller_skips_all_device_requeries():
    rgb, props, hints = fixture()
    controller = np.zeros(rgb.shape[:3], bool)
    controller[5, 1, 1] = True
    props.append(proposal(rgb, "controller", 7, controller))
    sam = FakeSam()
    assert select(rgb, props, hints)[0] == []
    refreshed, receipt = refresh(sam, rgb, props, hints)
    assert not sam.calls
    assert len(refreshed) == len(props)
    assert all(result is original for result, original in zip(refreshed, props))
    assert isinstance(receipt, dict)


def test_empty_controller_proposal_still_allows_one_call_per_anchor():
    rgb, props, hints = fixture()
    unsupported_only = np.zeros(rgb.shape[:3], bool)
    unsupported_only[0, 25, 10] = True
    for controller in (
        proposal(rgb, "controller", 7, np.zeros(rgb.shape[:3], bool)),
        proposal(rgb, "controller", 7, unsupported_only,
                 unsupported_frames=np.ones(len(rgb), bool)),
    ):
        original = [*props, controller]
        sam = FakeSam()
        refreshed, _ = refresh(sam, rgb, original, hints)
        assert [call[2] for call in sam.calls] == [2, 4]
        assert all(call[1] == "VR controller" for call in sam.calls)
        assert len(refreshed) == len(original)


def test_missing_or_unknown_identity_hints_do_not_guess_requery_anchors():
    rgb, props, hints = fixture()
    ambiguous = hints.copy()
    ambiguous[:] = [10, 25]
    for unavailable in (None, np.full_like(hints, np.nan), ambiguous):
        sam = FakeSam()
        # Restrict to the left track, so equal hints cannot assign a side.
        assert select(rgb, props[:1], unavailable)[0] == []
        refreshed, _ = refresh(sam, rgb, props[:1], unavailable)
        assert not sam.calls
        assert len(refreshed) == 1
    empty = replace(props[0], masks=np.zeros(rgb.shape[:3], bool))
    sam = FakeSam()
    assert select(rgb, [empty], hints)[0] == []
    refresh(sam, rgb, [empty], hints)
    assert not sam.calls


def test_unsupported_native_frame_cannot_supply_maximum_area_anchor():
    rgb, props, hints = fixture()
    unsupported = np.zeros(len(rgb), bool)
    unsupported[2:4] = True
    left = replace(props[0], unsupported_frames=unsupported)
    before = left.masks.copy()
    assert select(rgb, [left], hints)[0] == [1]
    assert np.array_equal(left.masks, before)
    assert np.array_equal(left.unsupported_frames, unsupported)


def test_raw_sha_and_decoded_rgb_drift_fail_before_any_provider_call():
    rgb, props, hints = fixture()
    changed_rgb = rgb.copy()
    changed_rgb[0, 0, 0, 0] ^= 1
    for frames, proposals in (
        (rgb, [replace(props[0], raw_source_sha256="b" * 64)]),
        (changed_rgb, props),
    ):
        sam = FakeSam()
        with pytest.raises(AutomaticMaskContractError, match="SIGNATURE"):
            refresh(sam, frames, proposals, hints)
        assert not sam.calls


def test_original_v1_validation_rejects_non_native_or_non_text_proposals():
    rgb, props, hints = fixture()
    for invalid in (
        replace(props[0], masks=props[0].masks[:, ::2]),
        replace(props[0], masks=props[0].masks.astype(np.uint8)),
        replace(props[0], source="SAM31_RGB_POINT_AUTO"),
    ):
        sam = FakeSam()
        with pytest.raises(AutomaticMaskContractError):
            refresh(sam, rgb, [invalid], hints)
        assert not sam.calls


def test_session_ids_are_not_inputs_to_anchor_or_refresh_api():
    rgb, props, hints = fixture()
    sam = FakeSam()
    for function in (select_device_requery_anchors, refresh_controller_proposals):
        assert "session_id" not in inspect.signature(function).parameters
    for session_id in ("001", "050"):
        with pytest.raises(TypeError):
            select_device_requery_anchors(rgb, props, hints,
                                         session_id=session_id, **BINDINGS)
        with pytest.raises(TypeError):
            refresh_controller_proposals(sam, rgb, props, hints,
                                         session_id=session_id, **BINDINGS)
    assert not sam.calls


def test_late_device_uses_fixed_prompt_native_packed_rgb_and_v1_association():
    rgb, props, hints = fixture()
    late = np.zeros(rgb.shape[:3], bool)
    late[2:, 27:30, 8:14] = True
    background = np.zeros_like(late)
    background[2:, 0:3, 30:33] = True
    tracks = {**packed_tracks(late, 7), **packed_tracks(background, 8)}
    sam = FakeSam(tracks)
    before_rgb = rgb.copy()
    before_masks = [entry.masks.copy() for entry in props]
    refreshed, receipt = refresh(sam, rgb, props[:1], hints)
    assert len(sam.calls) == 1
    frames, prompt, anchor = sam.calls[0]
    assert prompt == "VR controller" and anchor == 2
    assert np.array_equal(frames, before_rgb)
    controllers = [entry for entry in refreshed if entry.role == "controller"]
    assert len(controllers) == 2
    assert all(entry.source == "SAM31_RGB_TEXT" for entry in controllers)
    assert all(entry.masks.dtype == np.bool_ and entry.masks.shape == rgb.shape[:3]
               for entry in controllers)
    assert any(np.array_equal(entry.masks, late) for entry in controllers)
    assert any(np.array_equal(entry.masks, background) for entry in controllers)
    seeds = bundle(rgb, refreshed, hints)
    device = next(entry for entry in controllers if np.array_equal(entry.masks, late))
    unrelated = next(entry for entry in controllers
                     if np.array_equal(entry.masks, background))
    assert any(seed["proposal_track_id"] == device.track_id and seed["role"] == "controller"
               and seed["side"] == "left" and seed["frame_idx"] == 2
               for seed in seeds["objects"])
    assert not any(seed["role"] == "controller"
                   and seed["proposal_track_id"] == unrelated.track_id
                   for seed in seeds["objects"])
    assert np.array_equal(rgb, before_rgb)
    assert all(np.array_equal(entry.masks, before)
               for entry, before in zip(props, before_masks))
    assert isinstance(receipt, dict)


def test_vendor_track_ids_are_remapped_across_anchors_and_existing_proposals():
    rgb, props, hints = fixture()
    props.append(proposal(rgb, "controller", 7, np.zeros(rgb.shape[:3], bool)))
    masks = np.zeros(rgb.shape[:3], bool)
    masks[2:, 27:30, 8:14] = True
    sam = FakeSam(packed_tracks(masks, 7))
    refreshed, _ = refresh(sam, rgb, props, hints)
    assert [call[2] for call in sam.calls] == [2, 4]
    controller_ids = [entry.track_id for entry in refreshed if entry.role == "controller"]
    assert len(controller_ids) == 3 and len(set(controller_ids)) == 3
    assert 7 in controller_ids
    for entry in refreshed:
        assert entry.frames_sha256 == frames_sha256(rgb)
        assert entry.raw_source_sha256 == SHA
        assert entry.frame_map_sha256 == SHA
        assert entry.model_sha256 == SHA
    # The normal POINT seed association remains the final identity gate.
    assert bundle(rgb, refreshed, hints)["manual_seed_reads"] == 0


def test_malformed_sam_native_tracks_or_packed_masks_fail_closed():
    rgb, props, hints = fixture()
    mask = np.zeros(rgb.shape[1:3], bool)
    mask[25, 10] = True
    packed = np.packbits(mask)
    for malformed in (
        [],
        {True: {2: packed}},
        {7: {len(rgb): packed}},
        {7: {2: packed.astype(np.float32)}},
        {7: {2: packed[:-1]}},
        {7: {2: packed.reshape(48, 8)}},
    ):
        sam = FakeSam(malformed)
        with pytest.raises(AutomaticMaskContractError):
            refresh(sam, rgb, props[:1], hints)
        assert len(sam.calls) == 1
