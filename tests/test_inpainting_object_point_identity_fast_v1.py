"""Eight bounded CPU identity fixtures; no model, GPU, file producer or truth claim."""
from dataclasses import fields, replace
import inspect

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automatic_masks as am


SIG = {"raw_source_sha256": "a" * 64, "frame_map_sha256": "b" * 64, "model_sha256": "c" * 64}
STATUS = "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE"


def fixture():
    raw = np.arange(4 * 48 * 64 * 3, dtype=np.uint8).reshape(4, 48, 64, 3)
    wearer = np.zeros(raw.shape[:3], bool)
    wearer[:, 12:20, 12:20] = True
    empty = np.zeros_like(wearer)

    def proposal(role, track, mask, *, point=False, **kwargs):
        return am.RGBProposal(role, track, mask.copy(), am.frames_sha256(raw), **SIG,
            source="SAM31_RGB_POINT_AUTO" if point else "SAM31_RGB_TEXT",
            refinement_receipt_payload_sha256="d" * 64 if point else None,
            refinement_status=STATUS if point else None, **kwargs)

    text = [proposal("glove", 0, wearer), proposal("object", 0, empty)]
    point = [proposal("glove", 0, empty, point=True), proposal("object", 0, wearer, point=True)]
    return raw, wearer, empty, proposal, text, point


def merge(raw, text, point, **kwargs):
    return am.merge_text_wearer_support_with_object_point_identity(raw, text, point, **SIG,
        object_point_identity_policy=am.OBJECT_POINT_TEXT_IDENTITY, **kwargs)


def build(raw, proposals):
    return am.build_automatic_masks_with_source_aware_veto(raw, proposals, **SIG,
        source_aware_retention_veto=am.SOURCE_AWARE_RETENTION_VETO)


def same_proposals(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for field in fields(am.RGBProposal):
            x, y = getattr(a, field.name), getattr(b, field.name)
            if isinstance(x, np.ndarray):
                assert np.array_equal(x, y), field.name
            else:
                assert x == y, field.name


def test_disabled_wrapper_is_exact_existing_merge_and_old_abi():
    raw, _, _, _, text, point = fixture()
    old = am.merge_text_wearer_support(raw, text, point, **SIG)
    actual, meta = am.merge_text_wearer_support_with_object_point_identity(raw, text, point, **SIG)
    same_proposals(old, actual)
    assert meta["status"] == "DISABLED_EXACT_EXISTING_MERGE"
    assert set(inspect.signature(am.merge_text_wearer_support).parameters) == {
        "frames_rgb", "text_proposals", "point_proposals", "raw_source_sha256", "frame_map_sha256",
        "model_sha256", "projection_hints", "text_wearer_retention"}


def test_point_only_intrusion_released_in_both_merge_and_final_build():
    raw, wearer, _, _, text, point = fixture()
    old = build(raw, am.merge_text_wearer_support(raw, text, point, **SIG))
    actual, meta = merge(raw, text, point)
    new = build(raw, actual)
    assert old["retention_veto"][wearer].all() and not old["effective_removal"][wearer].any()
    assert meta["identity_rejected_object_point"][wearer].all()
    assert not actual[1].masks.any(), "The filtered object must reach final build, not only the merge OR"
    assert new["effective_removal"][wearer].all()
    assert not new["protected_visible"][wearer].any() and not new["retention_veto"][wearer].any()
    assert meta["valid_text_object_pixels_rejected"] == 0 and meta["model_calls"] == 0


def test_same_track_real_text_object_conflict_stays_unknown():
    raw, wearer, _, _, text, point = fixture()
    text[1] = replace(text[1], masks=wearer.copy())
    actual, meta = merge(raw, text, point)
    result = build(raw, actual)
    assert not meta["identity_rejected_object_point"].any()
    assert result["unknown"][wearer].all() and result["retention_veto"][wearer].all()
    assert not result["effective_removal"][wearer].any()


def test_other_track_real_text_object_also_blocks_universal_wearer_wins():
    raw, wearer, empty, proposal, text, point = fixture()
    text.append(proposal("object", 1, wearer))
    point.append(proposal("object", 1, empty, point=True))
    actual, meta = merge(raw, text, point)
    result = build(raw, actual)
    assert not meta["identity_rejected_object_point"].any()
    assert np.array_equal(actual[1].masks, point[1].masks)
    assert result["unknown"][wearer].all() and not result["effective_removal"][wearer].any()


def test_rule_is_per_pixel_and_preserves_object_point_outside_wearer():
    raw, wearer, empty, _, text, point = fixture()
    supported = empty.copy(); supported[:, 12:20, 16:20] = True
    outside = empty.copy(); outside[:, 28:32, 28:32] = True
    text[1] = replace(text[1], masks=supported)
    point[1] = replace(point[1], masks=wearer | outside)
    actual, meta = merge(raw, text, point)
    assert np.array_equal(meta["identity_rejected_object_point"], wearer & ~supported)
    assert np.array_equal(actual[1].masks, supported | outside)
    assert meta["valid_text_object_pixels_rejected"] == 0


def test_unsupported_text_cannot_support_rejection_or_fill_missing_point():
    raw, wearer, _, _, text, point = fixture()
    text[0] = replace(text[0], unsupported_frames=np.array([False, True, False, False]))
    point[1] = replace(point[1], unsupported_frames=np.array([False, False, True, False]))
    actual, meta = merge(raw, text, point)
    assert not meta["identity_rejected_object_point"][1].any()
    assert not meta["identity_rejected_object_point"][2].any()
    assert actual[1].masks[1][wearer[1]].all()
    assert actual[1].unsupported_frames[2]


def test_actual_source_legal_text_and_native_flags_bindings_rejected():
    raw, wearer, _, _, text, point = fixture()
    built = am.build_automatic_masks(raw, text, **SIG)
    bad_meta = {**built, "metadata": {**built["metadata"], "frame_map_sha256": "e" * 64}}
    with pytest.raises(ValueError, match="LEGAL_TEXT_BINDING"):
        merge(raw, text, point, built_text=bad_meta)
    with pytest.raises(ValueError, match="SOURCE_OR_NATIVE"):
        merge(raw, text, [point[0], replace(point[1], raw_source_sha256="e" * 64)])
    with pytest.raises(ValueError, match="UNSUPPORTED_DOMAIN"):
        merge(raw, text, [point[0], replace(point[1], unsupported_frames=np.zeros(3, bool))])
    with pytest.raises(ValueError, match="TRACK_ORDER"):
        merge(raw, text, list(reversed(point)))
    with pytest.raises(ValueError, match="UNKNOWN_OBJECT"):
        am.merge_text_wearer_support_with_object_point_identity(raw, text, point, **SIG,
            object_point_identity_policy="UNREGISTERED_RULE")


def test_cached_union_requires_raw_point_origin_and_inputs_not_mutated():
    raw, _, _, _, text, point = fixture()
    before_text = [replace(p, masks=p.masks.copy()) for p in text]
    before_point = [replace(p, masks=p.masks.copy()) for p in point]
    built = am.build_automatic_masks(raw, text, **SIG)
    selected_before = built["equipment_roles"]["glove"].copy()
    actual, meta = merge(raw, text, point, built_text=built)
    same_proposals(text, before_text); same_proposals(point, before_point)
    assert np.array_equal(built["equipment_roles"]["glove"], selected_before)
    assert actual[1].track_id == point[1].track_id and actual[1].source == point[1].source
    assert meta["thresholds_changed"] is False and meta["human_review"] == "PENDING"
    with pytest.raises(ValueError, match="SOURCE_OR_NATIVE"):
        merge(raw, text, [replace(point[0], source=am.TEXT_POINT_WEARER_UNION), point[1]])
