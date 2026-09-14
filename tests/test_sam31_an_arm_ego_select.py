from __future__ import annotations

import numpy as np

from pipeline.sam31_an_arm_ego_select import (
    DEVELOPMENT_FRAMES,
    PRODUCER_ID,
    TEXT_PROMPT,
    SelectorConfig,
    select_raw_instances,
)


def _joints() -> np.ndarray:
    left = np.stack([np.linspace(10, 30, 21), np.linspace(15, 35, 21)], axis=1)
    right = np.stack([np.linspace(70, 90, 21), np.linspace(60, 80, 21)], axis=1)
    return np.stack([left, right])


def test_fixed_identity_and_development_scope() -> None:
    assert PRODUCER_ID == "sam31_an_arm_ego_select_v1"
    assert TEXT_PROMPT == "an arm"
    assert DEVELOPMENT_FRAMES == tuple(range(228, 243))


def test_selects_whole_separate_raw_instances_without_editing() -> None:
    masks = np.zeros((3, 100, 100), dtype=bool)
    masks[0, 0:40, 0:40] = True
    masks[1, 55:100, 55:100] = True
    masks[2, 40:50, 40:50] = True
    before = masks.copy()
    selection = select_raw_instances(
        masks,
        np.array([0.8, 0.9, 0.99]),
        np.array([5, 6, 7]),
        joints_by_side=_joints(),
        cad_object_mask=np.zeros((100, 100), bool),
    )
    assert selection.frame_complete
    assert selection.left.instance_id == 5
    assert selection.right.instance_id == 6
    assert np.array_equal(selection.left.mask, masks[0])
    assert np.array_equal(selection.right.mask, masks[1])
    assert np.array_equal(masks, before)
    assert selection.pixels_created_or_edited == 0
    assert selection.background_rejections == (
        {"raw_instance_offset": 2, "instance_id": 7, "reason": "NO_AUTHORITY_EGO_HAND_WRIST_SUPPORT"},
    )


def test_object6d_is_rejection_only_and_never_subtracted() -> None:
    masks = np.zeros((2, 100, 100), dtype=bool)
    masks[0, 0:45, 0:45] = True
    masks[1, 55:100, 55:100] = True
    cad = masks[0].copy()
    selection = select_raw_instances(
        masks,
        np.array([0.9, 0.9]),
        np.array([0, 1]),
        joints_by_side=_joints(),
        cad_object_mask=cad,
        config=SelectorConfig(max_object_overlap_over_instance=0.12),
    )
    assert selection.left.mask is None
    assert selection.left.failure_reason == "SELECTION_RULE_REJECTED_RECALLED_INSTANCE"
    assert selection.right.mask is not None
    assert np.array_equal(selection.right.mask, masks[1])


def test_no_instance_is_prompt_recall_failure() -> None:
    selection = select_raw_instances(
        np.zeros((0, 100, 100), bool),
        np.zeros(0),
        np.zeros(0, int),
        joints_by_side=_joints(),
        cad_object_mask=np.zeros((100, 100), bool),
    )
    assert not selection.frame_complete
    assert selection.left.failure_reason == "PROMPT_RECALL_INSUFFICIENT"
    assert selection.right.failure_reason == "PROMPT_RECALL_INSUFFICIENT"
