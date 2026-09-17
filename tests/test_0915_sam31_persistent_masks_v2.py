from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops import run_0915_sam31_persistent_masks_v2 as subject


def test_human_selection_uses_dynamic_hawor_points_and_distinct_ids() -> None:
    masks = np.zeros((2, 100, 120), bool)
    masks[0, 20:80, 5:55] = True
    masks[1, 20:80, 65:115] = True
    ids = np.asarray([41, 72])
    prompts = [
        {"role": "left_human_skin_forearm",
         "positive_points_xy": [(20.0, 40.0), (25.0, 45.0), (30.0, 50.0)]},
        {"role": "right_human_skin_forearm",
         "positive_points_xy": [(80.0, 40.0), (85.0, 45.0), (90.0, 50.0)]},
    ]
    chosen, _evidence = subject.select_distinct_human_ids(masks, ids, prompts)
    assert chosen == {
        "left_human_skin_forearm": 41,
        "right_human_skin_forearm": 72,
    }


def test_seed_candidates_reject_human_leakage() -> None:
    masks = np.zeros((2, 40, 50), bool)
    masks[0, 5:25, 5:25] = True
    masks[1, 10:30, 25:45] = True
    human = masks[0].copy()
    rows = subject._seed_candidates(
        masks, np.asarray([0.9, 0.8]), np.asarray([1, 2]),
        human_union=human, minimum_area=80, maximum_fraction=0.25,
        maximum_human_overlap=0.35,
    )
    assert rows[0]["eligible"] is False
    assert rows[1]["eligible"] is True


def test_finger_sleeve_can_overlap_hand_without_weakening_object_gate() -> None:
    masks = np.zeros((1, 40, 50), bool)
    masks[0, 5:25, 5:25] = True
    human = masks[0].copy()
    object_rows = subject._seed_candidates(
        masks, np.asarray([0.9]), np.asarray([1]),
        human_union=human, minimum_area=80, maximum_fraction=0.25,
        maximum_human_overlap=0.35,
    )
    sleeve_rows = subject._seed_candidates(
        masks, np.asarray([0.9]), np.asarray([1]),
        human_union=human, minimum_area=80, maximum_fraction=0.25,
        maximum_human_overlap=1.0,
    )
    assert object_rows[0]["eligible"] is False
    assert sleeve_rows[0]["eligible"] is True
    assert sleeve_rows[0]["maximum_human_overlap"] == 1.0


def test_worker_is_sam31_only_without_fixed_sample_geometry() -> None:
    source = inspect.getsource(subject)
    assert "SAM3.1_ONLY_USER_LOCKED" in source
    assert "SAM2" not in source
    assert "Cutie" not in source
    assert "frame_count != 379" not in source
    assert "frozen_points" not in source
    assert '"tracker_role_created": False' in source
