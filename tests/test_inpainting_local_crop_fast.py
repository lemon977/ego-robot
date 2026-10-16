"""Pure synthetic NumPy tests. No model, GPU, or filesystem output."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from chaoyang.inpainting_candidate.local_crop import (
    build_local_crop_manifest, flatten_roi_manifest,
    native_rgb_crop, remap_native_crop_masks,
)


def fixture_data(masks, roles=None, unsupported=None, prior=None):
    n, h, w = masks[0].shape
    packed, proposals = {}, []
    roles = roles or ["glove"] * len(masks)
    for i, (mask, role) in enumerate(zip(masks, roles)):
        key = f"p{i}"
        packed[key] = np.packbits(mask.reshape(n, -1), axis=1)
        p = {"array_key": key, "role": role, "track_id": i * 41 - 3,
             "source": "SAM31_RGB_TEXT", "native_shape": [n, h, w],
             "unsupported_frames_array_key": None,
             "prior_unsupported_frames_array_key": None,
             "model_sha256": "f" * 64, "raw_source_sha256": "a" * 64}
        for name, values in (("unsupported_frames", unsupported),
                             ("prior_unsupported_frames", prior)):
            if values is not None:
                flag_key = key + "_" + name
                p[name + "_array_key"] = flag_key
                packed[flag_key] = np.asarray(values[i], dtype=bool)
        proposals.append(p)
    meta = {"schema": "inpainting-sam-native-proposal-diagnostic-v1",
            "native_shape": [n, h, w], "stage": "TEXT",
            "coordinate_domain": "NATIVE_RAW_RGB",
            "packing": "NUMPY_PACKBITS_C_ORDER_AXIS1_BIG_BITORDER",
            "proposals": proposals}
    rgb = np.arange(n * h * w * 3, dtype=np.uint32).astype(np.uint8).reshape(n, h, w, 3)
    return rgb, meta, packed


def plan_for(masks, anchors=(1,), **kwargs):
    rgb, meta, packed = fixture_data(masks, **kwargs)
    return rgb, build_local_crop_manifest(rgb.shape, meta, packed, anchors)


def test_boundary_windows_and_duplicate_anchors():
    mask = np.zeros((5, 10, 12), bool)
    mask[:, 3:5, 4:6] = True
    _, plan = plan_for([mask], anchors=(0, 1, 0, 4))
    assert [w["native_frames"] for w in plan["windows"]] == [[0, 1, 2], [2, 3, 4]]
    assert plan["windows"][0]["anchors"] == [0, 1]
    assert plan["crop_count"] == 2


def test_window_union_uses_real_neighbors_and_empty_middle():
    mask = np.zeros((5, 12, 14), bool)
    mask[0, 4:6, 5:7] = True
    mask[2, 6:8, 7:9] = True
    mask[4] = True
    _, plan = plan_for([mask])
    crop = plan["windows"][0]["crops"][0]
    assert crop["rect_xyxy"] == [4, 3, 10, 9]
    assert crop["provenance"][0]["valid_mask_pixels_per_frame"] == [4, 0, 4]


def test_unsupported_excludes_current_but_prior_is_history():
    mask = np.zeros((3, 12, 14), bool)
    mask[0, 4:6, 5:7] = True
    mask[1] = True
    mask[2, 4:6, 5:7] = True
    _, plan = plan_for([mask], unsupported=[[False, True, False]],
                      prior=[[True, False, True]])
    crop = plan["windows"][0]["crops"][0]
    assert crop["rect_xyxy"] == [4, 3, 8, 7]
    provenance = crop["provenance"][0]
    assert provenance["valid_mask_pixels_per_frame"] == [4, 0, 4]
    assert provenance["unsupported_native_frames"] == [1]
    assert provenance["prior_unsupported_native_frames"] == [0, 2]


def test_empty_and_all_unsupported_proposals_generate_no_crop():
    mask = np.ones((3, 8, 9), bool)
    _, plan = plan_for([mask, np.zeros_like(mask)],
                      unsupported=[[True] * 3, [False] * 3])
    assert plan["crop_count"] == 0


def test_only_exact_rectangle_dedup_preserves_all_sources():
    a = np.zeros((3, 20, 20), bool)
    b = np.zeros_like(a)
    a[:, 5:9, 5:9] = True
    b[:, 5:9, 6:10] = True
    _, plan = plan_for([a, a.copy(), b], roles=["human", "glove", "object"])
    crops = plan["windows"][0]["crops"]
    assert len(crops) == 2
    assert [p["proposal"]["role"] for p in crops[0]["provenance"]] == ["human", "glove"]
    assert crops[0]["rect_xyxy"] == [4, 4, 10, 10]
    assert crops[1]["rect_xyxy"] == [5, 4, 11, 10]


def test_role_selection_does_not_select_track_ids():
    a = np.zeros((3, 10, 12), bool)
    a[:, 2:4, 3:5] = True
    _, plan = plan_for([a] * 5, roles=["human", "glove", "object", "attachment", "cable"])
    source_rows = plan["windows"][0]["crops"][0]["provenance"]
    assert len(source_rows) == 4
    assert [p["proposal"]["track_id"] for p in source_rows] == [-3, 38, 79, 120]


def test_outward_rounding_and_native_boundary_clipping():
    a = np.zeros((3, 9, 13), bool)
    a[:, :3, :2] = True
    _, plan = plan_for([a])
    assert plan["windows"][0]["crops"][0]["rect_xyxy"] == [0, 0, 3, 4]


def test_native_crop_and_translation_only_remap():
    a = np.zeros((3, 12, 14), bool)
    a[:, 4:6, 5:7] = True
    rgb, plan = plan_for([a])
    window = plan["windows"][0]
    crop = window["crops"][0]
    pixels = native_rgb_crop(rgb, plan, window, crop)
    assert np.array_equal(pixels, rgb[:, 3:7, 4:8])
    local = np.arange(48).reshape(3, 4, 4)
    native = remap_native_crop_masks(local, plan, crop)
    assert np.array_equal(native[:, 3:7, 4:8], local)
    assert np.count_nonzero(native[:, :3]) == 0
    with pytest.raises(ValueError):
        remap_native_crop_masks(local[:, :-1], plan, crop)


def test_flatten_schema_sha_and_pixel_content_binding():
    a = np.zeros((3, 12, 14), bool)
    a[:, 4:6, 5:7] = True
    rgb, plan = plan_for([a])
    root = flatten_roi_manifest({"A": (rgb, plan), "B": (rgb.copy(), plan)})
    assert root["threshold_changes"] is False
    assert len(root["crops"]) == 2
    row = root["crops"][0]
    assert set(row) == {"crop_id", "session_key", "native_frames", "rect_xyxy",
                        "crop_shape", "frame_pixel_sha256", "provenance"}
    assert row["crop_shape"] == [3, 4, 4, 3]
    assert row["frame_pixel_sha256"][0] == hashlib.sha256(rgb[0, 3:7, 4:8].tobytes()).hexdigest()
    assert row["crop_id"] != root["crops"][1]["crop_id"]
    changed = rgb.copy()
    changed[0, 3, 4, 0] ^= 1
    other = flatten_roi_manifest({"A": (changed, plan)})["crops"][0]
    assert other["frame_pixel_sha256"][1:] == row["frame_pixel_sha256"][1:]
    assert other["frame_pixel_sha256"][0] != row["frame_pixel_sha256"][0]
    assert other["crop_id"] != row["crop_id"]


def test_inputs_are_not_mutated_and_provenance_is_independent():
    a = np.zeros((3, 9, 13), bool)
    a[:, 2:4, 3:5] = True
    rgb, meta, packed = fixture_data([a], unsupported=[[False, True, False]])
    old_meta, old_packed = deepcopy(meta), {k: v.copy() for k, v in packed.items()}
    plan = build_local_crop_manifest(rgb.shape, meta, packed, [1])
    assert meta == old_meta
    assert all(np.array_equal(v, old_packed[k]) for k, v in packed.items())
    row = flatten_roi_manifest({"A": (rgb, plan)})["crops"][0]
    row["provenance"][0]["proposal"]["model_sha256"] = "changed"
    assert plan["windows"][0]["crops"][0]["provenance"][0]["proposal"]["model_sha256"] == "f" * 64


def test_structural_contract_errors_do_not_add_quality_thresholds():
    a = np.ones((3, 9, 13), bool)
    rgb, meta, packed = fixture_data([a])
    with pytest.raises(ValueError):
        build_local_crop_manifest((2, 9, 13, 3), meta, packed, [0])
    with pytest.raises(ValueError):
        build_local_crop_manifest(rgb.shape, meta, packed, [3])
    with pytest.raises(ValueError):
        build_local_crop_manifest(rgb.shape, meta, packed, [True])
    bad = dict(packed, p0=packed["p0"].astype(np.int16))
    with pytest.raises(ValueError):
        build_local_crop_manifest(rgb.shape, meta, bad, [1])
