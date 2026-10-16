import copy
import numpy as np
import pytest

from chaoyang.pick_place_1008 import visual
from test_point_visual import _fixture, _run


SHAPE = (480, 640)


def _packed(mask):
    return np.packbits(mask)


def _mask(packed):
    return np.unpackbits(packed, count=np.prod(SHAPE)).reshape(SHAPE).astype(bool)


def _seed(obj_id, side="left"):
    return {"obj_id": obj_id, "role": "glove", "side": side, "frame_idx": 0}


def _veto(tracks, seeds, count=1):
    by_id = {seed["obj_id"]: seed for seed in seeds}
    meta = {"objects": {str(obj_id): {"anchor_mask_present": True} for obj_id in by_id}}
    return visual._tiny_component_veto(tracks, by_id, meta, count, SHAPE)


def _main():
    mask = np.zeros(SHAPE, bool)
    mask[88:120, 103] = True  # Exactly 32 observed pixels.
    return mask


def test_detached_tiny_veto_is_strict_subset_and_preserves_raw_packed_evidence():
    raw = _main()
    raw[200:204, 260:264] = True
    packed = _packed(raw)
    snapshot = packed.copy()
    tracks = {1: {0: packed}}
    filtered, metadata = _veto(tracks, [_seed(1)])
    assert np.array_equal(packed, snapshot)
    assert tracks[1][0] is packed
    result = _mask(filtered[1][0])
    assert np.array_equal(result, _main())
    assert not (result & ~raw).any()
    assert metadata["removed_pixels_total"] == 16
    info = metadata["objects"]["1"]
    assert info["original_pixels_per_frame"] == [48]
    assert info["filtered_pixels_per_frame"] == [32]
    assert info["removed_pixels_per_frame"] == [16]
    assert info["removed_components"][0]["bbox_xywh"] == [260, 200, 4, 4]
    assert metadata["raw_tracks_mutated"] is False
    assert metadata["geometry_authority"] is False
    assert metadata["cup_authority"] is False


def test_exact_64_pixel_distance_is_retained_but_65_is_vetoed():
    first, second = _main(), _main()
    first[103, 167] = True
    second[103, 168] = True
    filtered, metadata = _veto({1: {0: _packed(first), 1: _packed(second)}},
                               [_seed(1)], count=2)
    assert _mask(filtered[1][0])[103, 167]
    assert not _mask(filtered[1][1])[103, 168]
    assert metadata["objects"]["1"]["removed_pixels_per_frame"] == [0, 1]
    assert metadata["objects"]["1"]["removed_components"][0][
        "minimum_same_side_large_component_distance_px"] == pytest.approx(65.)


@pytest.mark.parametrize("area, removed", [(16, 16), (17, 0)])
def test_component_area_bound_preserves_larger_components(area, removed):
    raw = _main()
    raw[200:204, 260:264] = True
    if area == 17:
        raw[204, 260] = True
    filtered, metadata = _veto({1: {0: _packed(raw)}}, [_seed(1)])
    assert metadata["removed_pixels_total"] == removed
    assert int(_mask(filtered[1][0]).sum()) == int(raw.sum()) - removed


def test_near_current_same_side_component_support_retains_real_small_parts():
    main = _main()
    small = np.zeros(SHAPE, bool)
    small[103, 135] = True
    filtered, metadata = _veto({1: {0: _packed(main)}, 2: {0: _packed(small)}},
                               [_seed(1), _seed(2)])
    assert _mask(filtered[2][0])[103, 135]
    assert metadata["removed_pixels_total"] == 0


def test_other_hand_cannot_supply_support_or_veto_authority():
    small = np.zeros(SHAPE, bool)
    small[103, 168] = True
    other_hand = np.zeros(SHAPE, bool)
    other_hand[100:110, 170:180] = True
    tracks = {1: {0: _packed(small)}, 5: {0: _packed(other_hand)}}
    filtered, metadata = _veto(tracks, [_seed(1), _seed(5, "right")])
    assert _mask(filtered[1][0])[103, 168]  # No left-hand large component exists.
    assert metadata["no_same_side_large_component_raw_frames"]["left"] == [0]
    tracks[2] = {0: _packed(_main())}
    filtered, metadata = _veto(tracks, [_seed(1), _seed(2), _seed(5, "right")])
    assert not _mask(filtered[1][0])[103, 168]  # Right proximity does not grant support.
    assert metadata["cross_side_authority"] is False


def test_support_cannot_be_borrowed_from_another_frame():
    small = np.zeros(SHAPE, bool)
    small[103, 168] = True
    filtered, metadata = _veto({1: {0: _packed(_main()), 1: _packed(small)}},
                               [_seed(1)], count=2)
    assert _mask(filtered[1][1])[103, 168]
    assert metadata["no_same_side_large_component_raw_frames"]["left"] == [1]
    assert metadata["removed_pixels_total"] == 0


def test_31_pixel_union_has_no_authority_but_32_pixel_union_can_veto():
    raw31 = _main()
    raw31[88, 103] = False
    raw31[103, 168] = True
    raw32 = _main()
    raw32[103, 168] = True
    filtered, metadata = _veto({1: {0: _packed(raw31), 1: _packed(raw32)}},
                               [_seed(1)], count=2)
    assert _mask(filtered[1][0])[103, 168]
    assert not _mask(filtered[1][1])[103, 168]
    assert metadata["objects"]["1"]["removed_pixels_per_frame"] == [0, 1]


def test_large_connected_background_error_is_explicitly_outside_this_veto():
    raw = _main()
    raw[200:220, 260:310] = True
    filtered, metadata = _veto({1: {0: _packed(raw)}}, [_seed(1)])
    assert np.array_equal(_mask(filtered[1][0]), raw)
    assert metadata["removed_pixels_total"] == 0
    assert metadata["status"] == "CANDIDATE_REQUIRES_ACTUAL_GPU_AND_IMAGE_REVIEW"


def test_mask_builder_returns_model_raw_per_object_tracks_before_veto(monkeypatch):
    fixture = _fixture(monkeypatch)
    before = copy.deepcopy(fixture.tracks)
    out = _run(fixture)
    assert out["point_object_tracks"] is fixture.tracks
    for obj_id, frames in before.items():
        for frame, packed in frames.items():
            assert np.array_equal(out["point_object_tracks"][obj_id][frame], packed)
    assert out["metadata"]["tiny_component_veto"]["removed_pixels_total"] == 0


def test_same_id_correction_exact_evidence_fields_pass_visual_selection_unchanged():
    correction = {
        "raw_frame_index": 2, "frame_idx": 2, "qpc_ns": 123456,
        "image": {"path": "/original/raw_0002.png", "bytes": 123, "sha256": "a"*64},
        "device_serial": "device", "session_id": "001",
        "points_px": [[12, 13], [18, 19]], "point_labels": [1, 0],
        "visible_confirmed": True,
        "seed_semantics": "VISIBLE_FOREGROUND_PIXEL_NOT_JOINT_OR_CAD_POINT",
        "native_evidence_validation": "NATIVE_QPC_ORIGINAL_RGB_AND_SOURCE_IMAGE_ROW_EXACT_VERIFIED",
    }
    seed = {**_seed(1), "points_px": [[20, 20]], "point_labels": [1],
            "correction_points": [correction]}
    before = copy.deepcopy(seed)
    selected, anchor = visual._visible_seed_objects({"objects": [seed]}, 3, SHAPE)
    assert selected[0]["correction_points"] == [correction]
    assert seed == before
    assert anchor == 0  # Own-anchor stays explicit; corrections refine the same object.


@pytest.mark.parametrize("removed_frame, expected_code", [
    (0, "VISIBLE_ROLE_ANCHOR_SUPPORT_REMOVED_BY_TINY_COMPONENT_VETO"),
    (1, "VISIBLE_CORRECTION_SUPPORT_REMOVED_BY_TINY_COMPONENT_VETO"),
])
def test_required_native_prompt_removed_by_veto_is_quality_failure_not_false_support(monkeypatch, removed_frame, expected_code):
    fixture = _fixture(monkeypatch)
    fixture.h, fixture.w = 120, 160
    fixture.rgb = np.zeros((fixture.n, fixture.h, fixture.w, 3), np.uint8)
    fixture.session.width, fixture.session.height = fixture.w, fixture.h
    regions = {
        1: (slice(100, 110), slice(20, 30)),
        2: (slice(10, 11), slice(145, 146)),
        5: (slice(100, 110), slice(110, 120)),
        6: (slice(80, 85), slice(120, 125)),
    }
    fixture.tracks.clear()
    for seed in fixture.seeds:
        obj_id = seed["obj_id"]
        mask = np.zeros((fixture.n, fixture.h, fixture.w), bool)
        mask[:, regions[obj_id][0], regions[obj_id][1]] = True
        fixture.tracks[obj_id] = {frame: np.packbits(mask[frame]) for frame in range(fixture.n)}
        seed["points_px"] = [[regions[obj_id][1].start, regions[obj_id][0].start], [80, 30]]
    if removed_frame == 1:
        own = np.unpackbits(fixture.tracks[2][0], count=fixture.h*fixture.w).reshape(fixture.h, fixture.w).astype(bool)
        own[12:18, 140:146] = True
        fixture.tracks[2][0] = np.packbits(own)
        fixture.seeds[1]["correction_points"] = [{"frame_idx": 1, "raw_frame_index": 1,
            "points_px": [[145,10], [80,30]], "point_labels": [1,0]}]
    cup = np.zeros((fixture.n, fixture.h, fixture.w), bool)
    cup[:, 30:35, 80:85] = True
    class Sam:
        aborted = []
        def track_text(self, frames, prompt, anchor):
            return {99: {frame: np.packbits(cup[frame]) for frame in range(fixture.n)}}
    fixture.sam = Sam()
    out = _run(fixture)
    original = np.unpackbits(out["point_object_tracks"][2][removed_frame],
                             count=fixture.h*fixture.w).reshape(fixture.h, fixture.w)
    assert original[10, 145]
    assert not out["core"][removed_frame, 10, 145]
    problem = next(row for row in out["metadata"]["quality_problems"]
                   if row["code"] == expected_code)
    assert problem["severity"] == "QUALITY_FAILURE" and problem["obj_id"] == 2
    info = out["metadata"]["foreground_role_support"]["2"]
    assert info["anchor_mask_present"] is True  # Official SAM support is preserved as evidence.
    assert info["anchor_filtered_mask_present"] is (removed_frame == 1)
    assert info["original_pixels_per_frame"] == ([37,1,1,1] if removed_frame else [1]*fixture.n)
    assert info["tiny_veto_removed_pixels_per_frame"] == ([0,1,1,1] if removed_frame else [1]*fixture.n)
