import numpy as np
import pytest

from chaoyang.inpainting_candidate.occlusion import (
    DisplayPolicy,
    GeometryDecision,
    compose_with_occlusion,
    resolve_pixel_occlusion,
    resolve_robot_depth_chain,
)


NAMES = ("arm", "flange", "palm", "finger")
KINDS = dict(zip(NAMES, ("ARM", "FLANGE", "PALM", "FINGER")))
SHAPE = (2, 4)


def chain():
    supports, depths = {}, {}
    for col, name in enumerate(NAMES):
        mask = np.zeros(SHAPE, bool)
        mask[:, col] = True
        supports[name] = mask
        depths[name] = np.full(SHAPE, 2.0, np.float32)
    return resolve_robot_depth_chain(depths, supports, required_links=NAMES, kind_by_link=KINDS)


def resolve(*, visible=None, human=None, equipment=None, delete=None, alpha=None, **kwargs):
    zeros = np.zeros(SHAPE, bool)
    return resolve_pixel_occlusion(
        np.ones(SHAPE, bool) if alpha is None else alpha, chain(),
        zeros if visible is None else visible, zeros if human is None else human,
        zeros if equipment is None else equipment, zeros if delete is None else delete,
        object_support_source="RAW_VISIBLE_OBJECT_SUPPORT", **kwargs,
    )


def test_complete_robot_links_resolve_nearest_pixel_not_single_component():
    supports = {name: np.ones(SHAPE, bool) for name in NAMES}
    depths = {name: np.full(SHAPE, index + 1.0, np.float32) for index, name in enumerate(NAMES)}
    depths["finger"][0, 0] = 0.5
    result = resolve_robot_depth_chain(depths, supports, required_links=NAMES, kind_by_link=KINDS)
    assert result["robot_link_index"][0, 0] == 3
    assert result["robot_link_index"][1, 0] == 0
    assert result["robot_depth"][0, 0] == 0.5


@pytest.mark.parametrize("missing", NAMES)
def test_missing_robot_asset_link_rejected(missing):
    names = tuple(name for name in NAMES if name != missing)
    depths = {name: np.ones(SHAPE, np.float32) for name in names}
    supports = {name: np.ones(SHAPE, bool) for name in names}
    with pytest.raises(ValueError, match="LINK_CLOSURE_MISMATCH"):
        resolve_robot_depth_chain(depths, supports, required_links=NAMES, kind_by_link=KINDS)


def test_truncated_manifest_cannot_omit_flange_or_whole_arm():
    names = ("palm", "finger")
    with pytest.raises(ValueError, match="MISSING_COMPLETE_COMPONENTS"):
        resolve_robot_depth_chain(
            {name: np.ones(SHAPE, np.float32) for name in names},
            {name: np.ones(SHAPE, bool) for name in names},
            required_links=names, kind_by_link={name: KINDS[name] for name in names},
        )


def test_missing_depth_keeps_unknown_and_fallback_only_on_visible_overlap():
    visible = np.zeros(SHAPE, bool)
    visible[0, 3] = True
    result = resolve(visible=visible)
    assert np.all(result["geometry_decision"] == GeometryDecision.UNKNOWN)
    assert result["object_first_fallback"].sum() == 1
    assert result["display_policy"][0, 3] == DisplayPolicy.OBJECT_FIRST_FALLBACK
    assert result["display_policy"][1, 3] == DisplayPolicy.ROBOT_WITH_UNKNOWN_ORDER
    assert result["link_summary"]["finger"]["object_first_fallback_pixels"] == 1
    assert result["physical_contact_ground_truth"] is False


def test_fingertip_front_does_not_hide_whole_finger():
    visible = np.zeros(SHAPE, bool)
    visible[:, 3] = True
    scene = np.full(SHAPE, 3.0, np.float32)
    scene[0, 3] = 1.0
    result = resolve(visible=visible, scene_depth=scene, scene_depth_valid=np.ones(SHAPE, bool),
                     scene_depth_source="RAW_REGISTERED_SCENE_DEPTH", depths_comparable=True)
    assert result["geometry_decision"][0, 3] == GeometryDecision.SCENE_FRONT
    assert result["geometry_decision"][1, 3] == GeometryDecision.ROBOT_FRONT
    assert result["link_summary"]["finger"]["scene_front_pixels"] == 1
    assert result["link_summary"]["finger"]["robot_front_pixels"] == 1
    assert result["object_first_fallback"].sum() == 0


def test_equal_invalid_and_unverified_depth_are_unknown_not_fabricated_order():
    visible = np.ones(SHAPE, bool)
    scene = np.full(SHAPE, 2.0, np.float32)
    scene[0, 0], scene[0, 1] = np.nan, -1
    result = resolve(visible=visible, scene_depth=scene, scene_depth_valid=np.ones(SHAPE, bool),
                     scene_depth_source="RAW_REGISTERED_SCENE_DEPTH", depths_comparable=True)
    assert np.all(result["geometry_decision"] == GeometryDecision.UNKNOWN)
    assert result["object_first_fallback"].sum() == 8
    scene[:] = 1.0
    result = resolve(visible=visible, scene_depth=scene, scene_depth_valid=np.ones(SHAPE, bool),
                     scene_depth_source="RAW_REGISTERED_SCENE_DEPTH", depths_comparable=False)
    assert np.all(result["geometry_decision"] == GeometryDecision.UNKNOWN)


def test_original_human_devices_and_delete_depth_cannot_cover_robot():
    visible = np.ones(SHAPE, bool)
    human, equipment, delete = [np.zeros(SHAPE, bool) for _ in range(3)]
    human[:, 0], equipment[:, 1], delete[:, 2] = True, True, True
    result = resolve(visible=visible, human=human, equipment=equipment, delete=delete,
                     scene_depth=np.ones(SHAPE, np.float32), scene_depth_valid=np.ones(SHAPE, bool),
                     scene_depth_source="RAW_REGISTERED_SCENE_DEPTH", depths_comparable=True)
    assert result["trusted_object_overlap"].sum() == 2
    assert result["excluded_raw_foreground"].sum() == 6
    assert np.all(result["geometry_decision"][:, :3] == GeometryDecision.UNKNOWN)
    assert np.all(result["display_policy"][:, :3] == DisplayPolicy.ROBOT_WITH_UNKNOWN_ORDER)


def test_clean_generated_object_or_depth_is_never_foreground_evidence():
    with pytest.raises(ValueError, match="VISIBLE_RAW_EVIDENCE"):
        resolve_pixel_occlusion(
            np.ones(SHAPE, bool), chain(), *[np.zeros(SHAPE, bool) for _ in range(4)],
            object_support_source="CLEAN_HIDDEN_OBJECT_TEXTURE",
        )
    with pytest.raises(ValueError, match="MUST_NOT_COME_FROM_CLEAN"):
        resolve(visible=np.ones(SHAPE, bool), scene_depth=np.ones(SHAPE, np.float32),
                scene_depth_valid=np.ones(SHAPE, bool), scene_depth_source="CLEAN_PREDICTED_DEPTH")


def test_scene_depth_evidence_requires_validity_and_actual_array():
    with pytest.raises(ValueError, match="VALIDITY_REQUIRED"):
        resolve(scene_depth=np.ones(SHAPE, np.float32), scene_depth_source="RAW_REGISTERED_SCENE_DEPTH")
    with pytest.raises(ValueError, match="EVIDENCE_ABSENT"):
        resolve(depths_comparable=True)


def test_soft_alpha_edges_enter_depth_chain_and_display_decision():
    alpha = np.zeros(SHAPE, np.float32)
    alpha[0, 0] = 0.25
    result = resolve(visible=np.ones(SHAPE, bool), alpha=alpha)
    assert result["trusted_object_overlap"].sum() == 1
    assert result["object_first_fallback"][0, 0]
    bad = chain()
    bad["robot_link_index"][0, 0] = -1
    with pytest.raises(ValueError, match="OUTSIDE_COMPLETE_DEPTH_CHAIN"):
        resolve_pixel_occlusion(alpha, bad, *[np.zeros(SHAPE, bool) for _ in range(4)],
                                object_support_source="RAW_VISIBLE_OBJECT_SUPPORT")


def test_composite_restores_only_trusted_visible_raw_pixels_and_does_not_mutate():
    raw = np.full((*SHAPE, 3), 20, np.uint8)
    clean = raw.copy()
    robot = np.full_like(raw, 200)
    visible = np.zeros(SHAPE, bool)
    visible[0, 3] = True
    alpha = np.ones(SHAPE, np.float32)
    alpha[1, 0] = 0.25
    decision = resolve(visible=visible, alpha=alpha)
    zeros = np.zeros(SHAPE, bool)
    result = compose_with_occlusion(raw, clean, robot, alpha, decision, zeros, visible)
    np.testing.assert_array_equal(result["image"][0, 3], raw[0, 3])
    assert np.all(result["image"][1, 3] == 200)
    assert np.all(result["image"][1, 0] == 65)
    assert result["draw"][0, 3] == 0 and result["draw"][1, 3] == 1
    assert result["clean_visual_quality"] == "NOT_EVALUATED_BY_ROBOT_OVERLAY"
    assert np.all(raw == 20) and np.all(clean == 20) and np.all(robot == 200)


def test_robot_overlay_cannot_hide_failed_clean_preservation():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    clean = np.ones_like(raw)
    with pytest.raises(ValueError, match="ZERO_CHANGE_INVARIANT_FAILED"):
        compose_with_occlusion(raw, clean, raw, np.ones(SHAPE, bool), resolve(),
                               np.zeros(SHAPE, bool), np.zeros(SHAPE, bool))


def test_forged_whole_link_object_first_is_rejected():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    decision = resolve()
    decision["display_policy"][:, 3] = DisplayPolicy.OBJECT_FIRST_FALLBACK
    with pytest.raises(ValueError, match="OUTSIDE_TRUSTED_RAW_OVERLAP"):
        compose_with_occlusion(raw, raw.copy(), raw.copy(), np.ones(SHAPE, bool), decision,
                               np.zeros(SHAPE, bool), np.zeros(SHAPE, bool))


def test_composite_cannot_apply_a_decision_from_different_robot_support():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    decision = resolve(alpha=np.zeros(SHAPE, bool))
    with pytest.raises(ValueError, match="ROBOT_SUPPORT_CHANGED"):
        compose_with_occlusion(raw, raw.copy(), raw.copy(), np.ones(SHAPE, bool), decision,
                               np.zeros(SHAPE, bool), np.zeros(SHAPE, bool))


def test_composite_cannot_silently_relabel_unknown_geometry_as_depth_ordered():
    raw = np.zeros((*SHAPE, 3), np.uint8)
    decision = resolve()
    decision["display_policy"][0, 0] = DisplayPolicy.DEPTH_SCENE_FRONT
    with pytest.raises(ValueError, match="GEOMETRY_DISPLAY_MISMATCH"):
        compose_with_occlusion(raw, raw.copy(), raw.copy(), np.ones(SHAPE, bool), decision,
                               np.zeros(SHAPE, bool), np.zeros(SHAPE, bool))
