from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops.run_human_to_robot_007_wearable_seed_refine import (
    PROMPT, SPECS, choose_seed, main,
)


def test_prompt_and_six_frozen_devices() -> None:
    assert PROMPT == "finger-mounted sensor"
    assert [row[0] for row in SPECS] == [201, 202, 203, 204, 205, 206]


def test_seed_must_contain_frozen_positive() -> None:
    masks = np.zeros((2, 960, 1280), dtype=bool)
    _, _, (x, y, _, _), (px, py), _ = SPECS[0]
    masks[0, y, x] = True
    masks[1, py, px] = True
    assert choose_seed(masks, np.array([71, 72]), SPECS[0]) == 72
    assert choose_seed(masks[:1], np.array([71]), SPECS[0]) is None


def test_seed_tie_is_deterministic_and_raw_id_not_fabricated() -> None:
    masks = np.zeros((2, 960, 1280), dtype=bool)
    _, _, _, (px, py), _ = SPECS[0]
    masks[:, py, px] = True
    assert choose_seed(masks, np.array([9, 5]), SPECS[0]) == 5


def test_runner_uses_two_separate_prompts_and_same_raw_id() -> None:
    source = inspect.getsource(main)
    assert 'text_str=PROMPT' in source
    assert 'box_labels=[1]' in source
    assert 'text_str=None' in source
    assert 'obj_id=seed_id' in source
    assert 'clear_old_boxes=False' in source
    assert 'evaluate_instance(refined_mask, spec)' in source
    assert 'if DEST.exists()' in source
