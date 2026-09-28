import numpy as np

from chaoyang.ops.run_human_to_robot_031_surface_depth_temporal_audit import (
    load_visible_cards, summary,
)


def test_unknown_card_is_not_treated_as_absent():
    empty = np.zeros((149, 153600), np.uint8)
    states = [[{"semantic_admitted": True} for _ in range(149)] for _ in range(3)]
    states[1][80]["semantic_admitted"] = False
    guard, complete, admitted = load_visible_cards(80, [empty] * 3, states)
    assert not complete
    assert admitted == [True, False, True]
    assert not guard.any()


def test_visible_card_pixels_are_guarded_and_empty_summary_unknown():
    packed = np.zeros((149, 153600), np.uint8)
    mask = np.zeros((960, 1280), np.uint8)
    mask[400:420, 600:620] = 1
    packed[80] = np.packbits(mask.reshape(-1), bitorder="big")
    states = [[{"semantic_admitted": True} for _ in range(149)] for _ in range(3)]
    guard, complete, _ = load_visible_cards(80, [packed] * 3, states)
    assert complete and guard[205, 305]
    assert not guard[100, 100]
    assert summary(np.array([], np.float32))["median_m"] is None
