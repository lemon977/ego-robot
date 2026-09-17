from __future__ import annotations

import numpy as np
import pytest

from chaoyang.pipeline.sam31_0915_prompt_contract_v2 import (
    Sam31PromptContractError,
    build_prompt_plan,
    select_bilateral_anchor,
)


def fixture() -> tuple[np.ndarray, np.ndarray]:
    joints = np.full((2, 5, 21, 2), np.nan, np.float64)
    observed = np.zeros((2, 5), bool)
    observed[:, 1] = True
    observed[:, 3] = True
    for side in range(2):
        joints[side, 1, :10, 0] = 100 + side * 600 + np.arange(10)
        joints[side, 1, :10, 1] = 300 + np.arange(10)
        joints[side, 3, :, 0] = 120 + side * 600 + np.arange(21)
        joints[side, 3, :, 1] = 320 + np.arange(21)
    return joints, observed


def test_anchor_is_selected_from_dynamic_hawor_coverage() -> None:
    joints, observed = fixture()
    assert select_bilateral_anchor(joints, observed, width=1280, height=960) == 3


def test_plan_is_sam31_only_and_has_no_tracker_controller_or_pico() -> None:
    joints, observed = fixture()
    value = build_prompt_plan(joints, observed, task="playing_cards")
    assert value["model_identity"] == "SAM3.1"
    assert value["anchor_frame"] == 3
    assert value["tracker_role_created"] is False
    assert value["controller_role_created"] is False
    assert value["pico26_consumed"] is False
    assert len(value["hand_spatial_prompts"]) == 2
    assert all(prompt["source"] == "HAWOR_PROJECTED_JOINTS"
               for prompt in value["hand_spatial_prompts"])


def test_no_bilateral_hawor_fails_closed_without_filling() -> None:
    joints, observed = fixture()
    observed[1] = False
    with pytest.raises(Sam31PromptContractError, match="no frame"):
        build_prompt_plan(joints, observed, task="potato_chips")
