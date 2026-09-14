from __future__ import annotations

import inspect

import numpy as np
import pytest

from pipeline.sam31_ego_human_producer_p2 import (
    HUMAN_PROMPTS,
    OBJECT_PROMPT,
    ProducerP2Config,
    ProducerP2ContractError,
    SamInstances,
    closest_boundary_point,
    project_cylinder_mask,
    select_frame_candidate,
)
from tools.run_sam31_ego_human_producer_p2 import (
    prepare_owned_run,
    prepare_session_output_tree,
)


HEIGHT = 100
WIDTH = 100


def _joints():
    left = np.zeros((21, 2), dtype=np.float64)
    right = np.zeros((21, 2), dtype=np.float64)
    left[0] = [10, 50]
    right[0] = [90, 70]
    for index in range(1, 21):
        left[index] = [14 + index, 44 + (index % 5) * 3]
        right[index] = [86 - index, 64 + (index % 5) * 3]
    return np.stack([left, right])


def _empty_instances():
    return SamInstances(
        masks=np.zeros((0, HEIGHT, WIDTH), dtype=bool),
        scores=np.zeros(0, dtype=np.float64),
        instance_ids=np.zeros(0, dtype=np.int64),
    )


def _instances(masks, scores=None, ids=None):
    array = np.stack(masks).astype(bool) if masks else np.zeros((0, HEIGHT, WIDTH), bool)
    count = array.shape[0]
    return SamInstances(
        masks=array,
        scores=np.asarray(scores if scores is not None else [0.9] * count),
        instance_ids=np.asarray(ids if ids is not None else list(range(count))),
    )


def _rectangle(x0, y0, x1, y1):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _good_prompt_instances(include_background=True, object_overlap=False):
    left_hand = _rectangle(7, 39, 37, 65)
    right_hand = _rectangle(63, 58, 94, 84)
    background_person = _rectangle(43, 4, 58, 70)
    hand_masks = [left_hand, right_hand]
    if include_background:
        hand_masks.append(background_person)
    left_arm = _rectangle(0, 43, 16, 58)
    right_arm = _rectangle(86, 64, 100, 78)
    object_mask = (
        _rectangle(28, 44, 43, 63)
        if object_overlap
        else _rectangle(44, 20, 58, 40)
    )
    return {
        "a hand": _instances(hand_masks, ids=[10, 20, 99][: len(hand_masks)]),
        "a human forearm": _instances([left_arm, right_arm], ids=[30, 40]),
        "a sleeve": _empty_instances(),
        OBJECT_PROMPT: _instances([object_mask, background_person], ids=[50, 60]),
    }, object_mask


def test_complete_candidate_uses_seeded_hands_and_boundary_arms_only():
    prompts, object_mask = _good_prompt_instances()
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert result.sufficient
    assert result.selected_ids["a hand"] == {"left": 10, "right": 20}
    assert result.selected_ids["a human forearm"] == {"left": 30, "right": 40}
    assert 99 not in result.selected_ids["a hand"].values()
    assert result.diagnostics["pixel_lineage"]["geometry_created_human_pixels"] is False
    assert result.diagnostics["post_object_overlap_pixels"] == 0


def test_background_person_without_authoritative_ego_seed_is_rejected():
    prompts, object_mask = _good_prompt_instances(include_background=True)
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    hand_diagnostics = result.diagnostics["prompt_diagnostics"]["a hand"]["instances"]
    background = next(item for item in hand_diagnostics if item["instance_id"] == 99)
    assert background["eligible_sides"] == []
    assert not np.logical_and(result.final_mask, prompts["a hand"].masks[2]).any()


def test_object_protect_can_only_subtract_and_records_pre_post_overlap():
    prompts, object_mask = _good_prompt_instances(object_overlap=True)
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert result.diagnostics["pre_object_overlap_pixels"] > 0
    assert result.diagnostics["post_object_overlap_pixels"] == 0
    assert np.logical_and(result.final_mask, ~result.human_union_before_object_protect).sum() == 0
    assert np.logical_and(result.final_mask, result.object_protect_mask).sum() == 0


def test_geometry_and_seeds_cannot_create_human_pixels():
    _, object_mask = _good_prompt_instances()
    prompts = {prompt: _empty_instances() for prompt in HUMAN_PROMPTS}
    prompts[OBJECT_PROMPT] = _instances([object_mask], ids=[50])
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert result.final_mask.sum() == 0
    assert not result.sufficient
    assert any(reason.startswith("missing_seeded_hand_instance") for reason in result.hold_reasons)


def test_missing_object_instance_holds_without_inventing_protect():
    prompts, object_mask = _good_prompt_instances()
    prompts[OBJECT_PROMPT] = _empty_instances()
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert "missing_object6d_selected_visible_object_instance" in result.hold_reasons
    assert result.object_protect_mask.sum() == 0


def test_ambiguous_seeded_instance_is_not_assigned_to_both_sides():
    prompts, object_mask = _good_prompt_instances()
    ambiguous = np.logical_or(prompts["a hand"].masks[0], prompts["a hand"].masks[1])
    prompts["a hand"] = _instances([ambiguous], ids=[77])
    result = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert result.selected_ids["a hand"] == {}
    assert not result.sufficient


def test_temporal_regression_and_identity_change_hold():
    prompts, object_mask = _good_prompt_instances()
    first = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    changed = np.zeros_like(first.final_mask)
    prompts["a hand"] = _instances(
        [prompts["a hand"].masks[0], prompts["a hand"].masks[1]], ids=[11, 21]
    )
    second = select_frame_candidate(
        prompt_instances=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
        previous_final_mask=changed,
        previous_selected_ids=first.selected_ids,
    )
    assert "temporal_iou_below_dimensionless_limit" in second.hold_reasons
    assert "same_prompt_identity_changed" in second.hold_reasons


def test_prompt_set_is_exact_and_has_no_hidden_route():
    prompts, object_mask = _good_prompt_instances()
    prompts["an arm"] = _empty_instances()
    with pytest.raises(ProducerP2ContractError, match="exactly"):
        select_frame_candidate(
            prompt_instances=prompts,
            joints_by_side=_joints(),
            cad_object_mask=object_mask,
        )


def test_selector_api_has_no_session_or_frame_branch_inputs():
    parameters = inspect.signature(select_frame_candidate).parameters
    assert "session_id" not in parameters
    assert "frame_index" not in parameters
    assert HUMAN_PROMPTS == ("a hand", "a human forearm", "a sleeve")
    assert OBJECT_PROMPT == "a beverage can"


def test_closest_boundary_is_evidence_only_and_deterministic():
    assert closest_boundary_point(np.array([10.0, 50.0]), WIDTH, HEIGHT) == (0.0, 50.0)
    assert closest_boundary_point(np.array([90.0, 70.0]), WIDTH, HEIGHT) == (99.0, 70.0)


def test_analytic_cylinder_projection_is_nonempty_object_evidence():
    transform = np.eye(4)
    transform[2, 3] = 1.0
    mask = project_cylinder_mask(
        transform,
        np.array([100.0, 100.0, 50.0, 50.0]),
        0.1,
        0.3,
        WIDTH,
        HEIGHT,
    )
    assert mask.dtype == bool
    assert 0 < mask.sum() < WIDTH * HEIGHT


@pytest.mark.parametrize(
    "transform,radius",
    [(np.eye(3), 0.1), (np.eye(4), -0.1)],
)
def test_invalid_object6d_projection_fails_closed(transform, radius):
    with pytest.raises(ProducerP2ContractError):
        project_cylinder_mask(
            transform,
            np.array([100.0, 100.0, 50.0, 50.0]),
            radius,
            0.3,
            WIDTH,
            HEIGHT,
        )


def test_p2_run_root_and_owner_are_exclusive(tmp_path):
    (tmp_path / "_run").mkdir()
    component_root, marker = prepare_owned_run(
        tmp_path, "sam31_p2_candidate_v1", "b" * 64
    )
    assert component_root.is_dir()
    assert marker.is_file()
    with pytest.raises(FileExistsError):
        prepare_owned_run(tmp_path, "sam31_p2_candidate_v1", "b" * 64)


@pytest.mark.parametrize(
    "run_id,token",
    [("../escape", "b" * 64), ("valid_run", "invalid-token")],
)
def test_p2_run_ownership_inputs_fail_closed(tmp_path, run_id, token):
    (tmp_path / "_run").mkdir()
    with pytest.raises(ValueError):
        prepare_owned_run(tmp_path, run_id, token)


def test_session_output_tree_explicitly_creates_missing_sessions_parent(tmp_path):
    component_root = tmp_path / "component"
    component_root.mkdir()
    paths = prepare_session_output_tree(component_root, "grap_a_cap_004")
    assert (component_root / "sessions").is_dir()
    assert all(path.is_dir() for path in paths.values())
    with pytest.raises(FileExistsError):
        prepare_session_output_tree(component_root, "grap_a_cap_004")
