"""Synthetic CPU contracts: no SAM inference, no file or session evidence."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate.v1_mask_repair import (
    REPAIR_AUTHORITY, V1MaskRepairContractError, make_repair_prompts, merge_repair_support,
    qualify_identity_support,
)
from chaoyang.pick_place_1008.sam_point_adapter import _validated_inputs


SHAPE = (3, 16, 16)
BINDINGS = {"frames_sha256": "a" * 64, "raw_source_sha256": "b" * 64,
            "frame_map_sha256": "c" * 64, "model_sha256": "d" * 64}


def original():
    result = {key: np.zeros(SHAPE, bool) for key in (
        "core", "sam_core", "equipment", "removal", "protected_visible",
        "cup_observed", "held", "unknown", "semantic_conflict")}
    result["core"][:, 0, 0] = True
    result["sam_core"][:, 0, 0] = True
    result["equipment"][:, 0, 1] = True
    result["core"] |= result["equipment"]
    result["removal"] |= result["core"]
    result["metadata"] = {"removal": {"observed_segmentation_dilation_px": 0,
                                       "temporal_bracket_frames": 0}}
    result["input_bindings"] = dict(BINDINGS)
    return result


def proposal(role="human", track=0, masks=None):
    mask = np.zeros(SHAPE, bool) if masks is None else masks.copy()
    if masks is None:
        mask[0, 6:9, 6:9] = True
        mask[1, 6:11, 6:11] = True
        mask[2, 6:11, 6:11] = True
    return SimpleNamespace(role=role, track_id=track, masks=mask,
                           source="SAM31_RGB_TEXT", unsupported_frames=None, **BINDINGS)


def assignment(role="human", track=0, side="right", obj=2):
    return {"role": role, "track_id": track, "side": side, "obj_id": obj,
            "association": "UNIQUE_MAX_SAME_CAPTURE_PROJECTION_HIT_RATE"}


def make(base=None, proposals=None, accepted=None):
    return make_repair_prompts(original() if base is None else base,
                               [proposal()] if proposals is None else proposals,
                               [assignment()] if accepted is None else accepted)


def test_largest_gap_earliest_frame_and_distance_point():
    prompts, meta = make()
    assert len(prompts) == 1
    assert prompts[0]["frame_idx"] == 1
    assert prompts[0]["points_px"] == [[8.0, 8.0]]
    assert prompts[0]["obj_id"] == 2
    assert meta["automatic_points"] == 1
    assert meta["manual_points_added"] == 0
    assert prompts[0]["seed_authority"] == REPAIR_AUTHORITY
    assert prompts[0]["human_visible_confirmed"] is False
    assert "visible_confirmed" not in prompts[0]


def test_equal_distance_is_y_then_x():
    mask = np.zeros(SHAPE, bool)
    mask[0, 4:6, 4:6] = True
    prompts, _ = make(proposals=[proposal(masks=mask)])
    assert prompts[0]["points_px"] == [[4.0, 4.0]]


def test_original_coverage_protection_and_opposite_rgb_excluded():
    base = original()
    base["protected_visible"][1:, 6:11, 6:11] = True
    other_mask = np.zeros(SHAPE, bool)
    other_mask[0, 6:9, 6:9] = True
    prompts, meta = make(base, [proposal(), proposal(track=1, masks=other_mask)],
                         [assignment(), assignment(track=1, side="left", obj=1)])
    assert prompts == []
    assert meta["status"] == "NO_RGB_GAP"


def test_original_side_support_excludes_cross_side_pixel():
    base = original()
    base["side_support"] = np.zeros((2, *SHAPE), bool)
    base["side_support"][0] = proposal().masks
    assert make(base)[0] == []


def test_already_effectively_deleted_region_not_prompted():
    base = original()
    base["effective_removal"] = proposal().masks | base["removal"]
    assert make(base)[0] == []


def test_unsupported_text_frame_cannot_be_anchor():
    item = proposal()
    item.unsupported_frames = np.asarray([False, True, True])
    prompts, _ = make(proposals=[item])
    assert prompts[0]["frame_idx"] == 0


@pytest.mark.parametrize("role", ["object", "controller", "attachment", "cable"])
def test_nonwearer_track_never_promoted(role):
    prompts, _ = make(proposals=[proposal(role=role)], accepted=[assignment(role=role)])
    assert prompts == []


@pytest.mark.parametrize("key", list(BINDINGS))
def test_source_signature_mismatch_rejected(key):
    item = proposal()
    setattr(item, key, "e" * 64)
    with pytest.raises(V1MaskRepairContractError, match="SIGNATURE_MISMATCH"):
        make(proposals=[item])


def test_nontext_proposal_rejected():
    item = proposal()
    item.source = "SAM31_RGB_POINT_AUTO"
    with pytest.raises(V1MaskRepairContractError, match="ONLY_SAM31"):
        make(proposals=[item])


@pytest.mark.parametrize("field,value", [("side", "UNKNOWN"), ("obj_id", True), ("obj_id", 17),
                                       ("association", "IDENTITY_UNKNOWN"), ("accepted", False)])
def test_unaccepted_identity_rejected(field, value):
    accepted = assignment()
    accepted[field] = value
    with pytest.raises(V1MaskRepairContractError, match="LEGAL_ORIGINAL"):
        make(accepted=[accepted])


def test_duplicate_track_rejected():
    with pytest.raises(V1MaskRepairContractError, match="DUPLICATE_ROLE_TRACK"):
        make(proposals=[proposal(), proposal()])


def test_ambiguous_multiple_tracks_one_object_rejected():
    with pytest.raises(V1MaskRepairContractError, match="AMBIGUOUS_MULTIPLE"):
        make(proposals=[proposal(), proposal(track=1)],
             accepted=[assignment(), assignment(track=1)])


def test_multiple_tracks_uniquely_linked_same_object_choose_largest():
    mask = np.zeros(SHAPE, bool)
    mask[0, 3:12, 3:12] = True
    accepted = [assignment(), assignment(track=1)]
    for value in accepted:
        value["identity_unique"] = True
    prompts, _ = make(proposals=[proposal(), proposal(track=1, masks=mask)], accepted=accepted)
    assert len(prompts) == 1
    assert prompts[0]["track_id"] == 1


def test_seed_is_legal_original_adapter_anchor_not_manual_correction():
    prompts, _ = make()
    rgb = np.zeros((*SHAPE, 3), np.uint8)
    _, prepared = _validated_inputs(rgb, prompts, prompts[0]["frame_idx"])
    assert prepared[0]["obj_id"] == 2
    assert prepared[0]["correction_points"] == []


def test_text_role_does_not_change_original_v1_object_role():
    accepted = assignment()
    accepted["original_role"] = "glove"
    prompts, _ = make(accepted=[accepted])
    assert prompts[0]["role"] == "glove"
    assert prompts[0]["proposal_role"] == "human"
    arrays, report = merge_repair_support(original(), {2: proposal().masks}, [proposal()], prompts, [accepted])
    assert arrays["core"][1, 8, 8]
    assert report["objects"][0]["role"] == "glove"
    assert report["objects"][0]["proposal_role"] == "human"


def test_merge_only_point_intersect_same_text_preserves_all_old_support():
    base = original()
    base["protected_visible"][:, 9, 9] = True
    saved = copy.deepcopy(base)
    item = proposal()
    prompts, _ = make(base, [item])
    point = np.ones(SHAPE, bool)  # A deliberately overbroad point result.
    arrays, report = merge_repair_support(base, {2: point}, [item], prompts, [assignment()])
    expected = item.masks & ~base["protected_visible"]
    assert np.array_equal(arrays["core"], base["core"] | expected)
    assert np.array_equal(arrays["equipment"], base["equipment"])
    assert np.array_equal(arrays["protected_visible"], base["protected_visible"])
    assert not np.any(arrays["effective_removal"] & base["protected_visible"])
    assert report["original_core_pixels_removed"] == 0
    for key in saved:
        if isinstance(saved[key], np.ndarray):
            assert np.array_equal(base[key], saved[key])
        else:
            assert base[key] == saved[key]


def test_packed_point_tracks_and_native_binding():
    base, item = original(), proposal()
    prompts, _ = make(base, [item])
    packed = {frame: np.packbits(mask) for frame, mask in enumerate(item.masks)}
    arrays, report = merge_repair_support(base, {2: packed}, [item], prompts, [assignment()])
    assert arrays["core"][1, 8, 8]
    assert report["objects"][0]["anchor_point_supported"]


def test_missing_anchor_rejects_all_additions():
    base, item = original(), proposal()
    prompts, _ = make(base, [item])
    point = item.masks.copy()
    point[1, 8, 8] = False
    arrays, report = merge_repair_support(base, {2: point}, [item], prompts, [assignment()])
    assert np.array_equal(arrays["core"], base["core"])
    assert not report["objects"][0]["anchor_point_supported"]


def test_point_only_pixels_not_promoted_without_text_support():
    base, item = original(), proposal()
    prompts, _ = make(base, [item])
    point = np.ones(SHAPE, bool)
    arrays, _ = merge_repair_support(base, {2: point}, [item], prompts, [assignment()])
    assert not arrays["core"][1, 15, 15]


def test_proposal_pixels_without_point_support_not_promoted():
    base, item = original(), proposal()
    prompts, _ = make(base, [item])
    point = np.zeros(SHAPE, bool)
    point[1, 8, 8] = True
    arrays, _ = merge_repair_support(base, {2: point}, [item], prompts, [assignment()])
    assert arrays["core"][1, 8, 8]
    assert not arrays["core"][1, 6, 6]


def test_unprompted_point_id_rejected():
    prompts, _ = make()
    with pytest.raises(V1MaskRepairContractError, match="UNPROMPTED"):
        merge_repair_support(original(), {9: proposal().masks}, [proposal()], prompts, [assignment()])


def test_tampered_prompt_rejected_before_merge():
    prompts, _ = make()
    prompts[0]["points_px"] = [[0.0, 0.0]]
    with pytest.raises(V1MaskRepairContractError, match="NOT_DETERMINISTIC"):
        merge_repair_support(original(), {2: proposal().masks}, [proposal()], prompts, [assignment()])


def test_frozen_original_dilation_and_bracket_reused():
    base = original()
    base["metadata"]["removal"] = {"observed_segmentation_dilation_px": 1,
                                       "temporal_bracket_frames": 1}
    mask = np.zeros(SHAPE, bool)
    mask[[0, 2], 8, 8] = True
    item = proposal(masks=mask)
    prompts, _ = make(base, [item])
    arrays, report = merge_repair_support(base, {2: mask}, [item], prompts, [assignment()])
    assert arrays["removal"][1, 8, 8]
    assert arrays["removal"][0, 8, 9]
    assert report["removal_expansion_rule"] == base["metadata"]["removal"]


def test_missing_original_rules_rejected():
    base = original()
    base["metadata"] = {}
    prompts, _ = make(base)
    with pytest.raises(V1MaskRepairContractError, match="FROZEN_ORIGINAL"):
        merge_repair_support(base, {2: proposal().masks}, [proposal()], prompts, [assignment()])


def test_original_masks_json_nested_metadata_supported():
    base = original()
    base["metadata"] = {"schema": "pick_place_1008_protected_clean_v1", "metadata": base["metadata"]}
    prompts, _ = make(base)
    arrays, _ = merge_repair_support(base, {2: proposal().masks}, [proposal()], prompts, [assignment()])
    assert arrays["core"][1, 8, 8]


@pytest.mark.parametrize("bad", [np.zeros(SHAPE, np.uint8), np.zeros((2, 16, 16), bool)])
def test_bad_proposal_mask_domain_rejected(bad):
    item = proposal()
    item.masks = bad
    with pytest.raises(V1MaskRepairContractError, match="NATIVE_DOMAIN"):
        make(proposals=[item])


def test_qualification_vetoes_only_opposite_supported_component():
    mask = np.zeros(SHAPE, bool)
    mask[:, 1, 1] = True
    mask[:, 14, 14] = True
    item = proposal(masks=mask)
    hints = np.zeros((2, SHAPE[0], 1, 2), np.float64)
    hints[0, :, 0] = [1, 1]
    hints[1, :, 0] = [14, 14]
    qualified, report = qualify_identity_support([item], [assignment(side='left')], hints, BINDINGS)
    assert qualified[0].masks[:, 1, 1].all()
    assert not qualified[0].masks[:, 14, 14].any()
    assert np.array_equal(item.masks, mask)
    assert report['pixels_excluded_total'] == 3
    assert report['foreground_pixels_added'] == 0
    assert report['existing_search_radius_px'] == 8
    assert all(row['search_hits_left_right'] == [False, True]
               for row in report['objects'][0]['explicit_opposite_identity_conflicts'])


def test_qualification_both_side_and_no_side_hits_remain_explicitly_uncertain():
    item = proposal()
    hints = np.full((2, SHAPE[0], 1, 2), np.nan, np.float64)
    hints[:, 1, 0] = [8, 8]
    qualified, report = qualify_identity_support([item], [assignment()], hints, BINDINGS)
    assert np.array_equal(qualified[0].masks, item.masks)
    reasons = {row['decision'] for row in report['objects'][0]['retained_identity_uncertainty']}
    assert reasons == {'BOTH_SIDES_HIT_EXISTING_GLOBAL_IDENTITY_ONLY', 'NO_LOCAL_HINT_HIT_EXISTING_GLOBAL_IDENTITY_ONLY'}
    assert not report['projection_is_pixel_evidence']


def test_qualification_removes_bad_component_before_prompt_and_merge():
    base = original()
    mask = np.zeros(SHAPE, bool)
    mask[:, 1, 1] = True
    mask[:, 14, 14] = True
    base['effective_removal'] = np.zeros(SHAPE, bool)
    base['effective_removal'][:, 1, 1] = True
    item = proposal(masks=mask)
    hints = np.zeros((2, SHAPE[0], 1, 2), np.float64)
    hints[0, :, 0] = [1, 1]
    hints[1, :, 0] = [14, 14]
    accepted = [assignment(side='left')]
    qualified, _ = qualify_identity_support([item], accepted, hints, BINDINGS)
    prompts, _ = make(base, qualified, accepted)
    assert prompts == []
    arrays, _ = merge_repair_support(base, {}, qualified, prompts, accepted)
    assert not arrays['core'][:, 14, 14].any()


def test_qualification_current_source_bindings_checked():
    item = proposal()
    item.frames_sha256 = 'e' * 64
    with pytest.raises(V1MaskRepairContractError, match='SIGNATURE_MISMATCH'):
        qualify_identity_support([item], [assignment()], np.zeros((2, 3, 1, 2), np.float64), BINDINGS)


def test_qualification_no_new_projected_foreground_from_absent_rgb():
    item = proposal(masks=np.zeros(SHAPE, bool))
    qualified, report = qualify_identity_support([item], [assignment()], np.ones((2, 3, 1, 2), np.float64), BINDINGS)
    assert not qualified[0].masks.any()
    assert report['objects'][0]['components_examined'] == 0


def test_qualification_hint_frame_domain_rejected():
    with pytest.raises(V1MaskRepairContractError, match='HINT_FRAME_DOMAIN'):
        qualify_identity_support([proposal()], [assignment()], np.zeros((2, 1, 1, 2), np.float64), BINDINGS)
