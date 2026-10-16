"""Synthetic, CPU-only spread-prompt contracts; no real input or inference."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from chaoyang.inpainting_candidate.v1_mask_repair import (
    SINGLE_MAX_GAP, SPREAD_MAX_GAP_THREE, SPREAD_POINT_RULE,
    V1MaskRepairContractError, make_repair_prompts, merge_repair_support,
)
from chaoyang.pick_place_1008.sam_point_adapter import _validated_inputs


SHAPE = (3, 16, 16)
BINDINGS = {"frames_sha256": "a" * 64, "raw_source_sha256": "b" * 64,
            "frame_map_sha256": "c" * 64, "model_sha256": "d" * 64}


def inputs(mask=None):
    base = {key: np.zeros(SHAPE, bool) for key in (
        "core", "sam_core", "equipment", "removal", "protected_visible",
        "cup_observed", "held", "unknown", "semantic_conflict")}
    base["equipment"][:, 0, 0] = True
    base["core"] |= base["equipment"]
    base["sam_core"] |= base["core"]
    base["removal"] |= base["core"]
    base["metadata"] = {"removal": {"observed_segmentation_dilation_px": 0,
                                       "temporal_bracket_frames": 0}}
    base["input_bindings"] = dict(BINDINGS)
    if mask is None:
        mask = np.zeros(SHAPE, bool)
        mask[0, 6:9, 6:9] = True
        mask[1:, 6:11, 6:11] = True
    proposal = SimpleNamespace(role="human", track_id=0, masks=mask.copy(),
                              source="SAM31_RGB_TEXT", unsupported_frames=None, **BINDINGS)
    assignment = {"role": "human", "original_role": "glove", "track_id": 0,
                  "obj_id": 5, "side": "right", "association": "UNIQUE_MAX_SAME_CAPTURE_PROJECTION_HIT_RATE"}
    return base, [proposal], [assignment]


def make(args, policy=SPREAD_MAX_GAP_THREE):
    return make_repair_prompts(*args, prompt_policy=policy)


def test_default_exactly_preserves_single_point_prompts_and_metadata():
    args = inputs()
    implicit = make_repair_prompts(*args)
    explicit = make(args, SINGLE_MAX_GAP)
    assert implicit == explicit
    assert implicit[0][0]["points_px"] == [[8.0, 8.0]]
    assert implicit[1]["automatic_points"] == 1
    assert implicit[1]["maximum_points_per_original_object"] == 1
    assert "prompt_policy" not in implicit[1]
    assert "spread_point_rule" not in implicit[1]


def test_spread_preserves_object_anchor_and_legacy_first_point():
    args = inputs()
    original, _ = make_repair_prompts(*args)
    spread, meta = make(args)
    before, after = original[0], spread[0]
    assert {key: before[key] for key in before if key not in ("points_px", "point_labels")} == {
        key: after[key] for key in after if key not in ("points_px", "point_labels")}
    assert after["points_px"][0] == before["points_px"][0]
    assert len(after["points_px"]) == 3
    assert after["point_labels"] == [1, 1, 1]
    assert meta["automatic_points"] == 3
    assert meta["prompted_objects"] == 1
    assert meta["maximum_points_per_original_object"] == 3
    assert meta["spread_point_rule"] == SPREAD_POINT_RULE
    assert after["human_visible_confirmed"] is False
    assert "correction_points" not in after


def test_three_points_inside_one_connected_component_and_deterministic():
    args = inputs()
    prompts, _ = make(args)
    chosen = prompts[0]
    mask = args[1][0].masks[chosen["frame_idx"]]
    assert cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] == 2
    assert len({tuple(p) for p in chosen["points_px"]}) == 3
    assert all(mask[int(y), int(x)] for x, y in chosen["points_px"])
    assert make(args)[0] == prompts


def test_spread_score_matches_independent_squared_distance_oracle():
    args = inputs()
    prompts, _ = make(args)
    chosen = prompts[0]
    mask = args[1][0].masks[chosen["frame_idx"]]
    boundary = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    prior = [chosen["points_px"][0]]
    for actual in chosen["points_px"][1:]:
        # Iterating y then x and using max's first tie is an independent oracle.
        scored = [(float(boundary[y, x]) * min((x-px)**2 + (y-py)**2 for px, py in prior),
                   [float(x), float(y)])
                  for y, x in zip(*np.nonzero(mask))]
        expected = max(scored, key=lambda value: value[0])[1]
        assert actual == expected
        prior.append(actual)


def test_line_spread_ties_use_earliest_y_then_x():
    mask = np.zeros(SHAPE, bool)
    mask[0, 4, 4:9] = True
    prompts, _ = make(inputs(mask))
    assert prompts[0]["points_px"] == [[4.0, 4.0], [8.0, 4.0], [6.0, 4.0]]


@pytest.mark.parametrize("pixels,expected", [([(4, 4)], 1), ([(4, 4), (5, 4)], 2)])
def test_tiny_gap_has_only_distinct_available_points(pixels, expected):
    mask = np.zeros(SHAPE, bool)
    for x, y in pixels:
        mask[0, y, x] = True
    prompts, meta = make(inputs(mask))
    assert len(prompts[0]["points_px"]) == expected
    assert meta["automatic_points"] == expected


def test_no_gap_keeps_empty_result_under_spread_policy():
    prompts, meta = make(inputs(np.zeros(SHAPE, bool)))
    assert prompts == []
    assert meta["automatic_points"] == 0
    assert meta["status"] == "NO_RGB_GAP"


def test_spread_points_never_enter_protection_or_opposite_support():
    base, proposals, assignments = inputs()
    base["protected_visible"][:, 6:11, 8] = True
    base["side_support"] = np.zeros((2, *SHAPE), bool)
    base["side_support"][0, :, 6:11, 6] = True
    prompts, _ = make((base, proposals, assignments))
    for point in prompts[0]["points_px"]:
        x, y = map(int, point)
        frame = prompts[0]["frame_idx"]
        assert not base["protected_visible"][frame, y, x]
        assert not base["side_support"][0, frame, y, x]


def test_spread_seed_accepted_by_unchanged_point_adapter():
    prompts, _ = make(inputs())
    _, prepared = _validated_inputs(np.zeros((*SHAPE, 3), np.uint8), prompts, 1)
    assert prepared[0]["points_px"] == prompts[0]["points_px"]
    assert prepared[0]["correction_points"] == []


def test_merge_first_point_gate_not_relaxed_when_other_points_supported():
    base, proposals, assignments = inputs()
    prompts, _ = make((base, proposals, assignments))
    point = proposals[0].masks.copy()
    first_x, first_y = map(int, prompts[0]["points_px"][0])
    point[prompts[0]["frame_idx"], first_y, first_x] = False
    arrays, report = merge_repair_support(base, {5: point}, proposals, prompts, assignments,
                                          prompt_policy=SPREAD_MAX_GAP_THREE)
    assert np.array_equal(arrays["core"], base["core"])
    obj = report["objects"][0]
    assert not obj["anchor_point_supported"]
    assert [p["point_supported"] for p in obj["automatic_point_support_at_anchor"]] == [False, True, True]
    assert report["anchor_support_rule"] == "LEGACY_FIRST_POINT_ONLY_UNCHANGED"


def test_merge_other_point_failures_only_diagnostic_and_intersection_unchanged():
    base, proposals, assignments = inputs()
    prompts, _ = make((base, proposals, assignments))
    point = np.zeros(SHAPE, bool)
    x, y = map(int, prompts[0]["points_px"][0])
    point[prompts[0]["frame_idx"], y, x] = True
    point[:, 15, 15] = True  # Unsupported RGB remains excluded.
    saved = copy.deepcopy(base)
    arrays, report = merge_repair_support(base, {5: point}, proposals, prompts, assignments,
                                          prompt_policy=SPREAD_MAX_GAP_THREE)
    assert arrays["core"][1, y, x]
    assert not arrays["core"][:, 15, 15].any()
    assert [p["point_supported"] for p in report["objects"][0]["automatic_point_support_at_anchor"]] == [True, False, False]
    assert np.array_equal(arrays["protected_visible"], saved["protected_visible"])
    assert np.array_equal(arrays["equipment"], saved["equipment"])
    assert np.array_equal(base["core"], saved["core"])


def test_merge_requires_same_spread_policy_and_detects_later_point_tampering():
    base, proposals, assignments = inputs()
    prompts, _ = make((base, proposals, assignments))
    with pytest.raises(V1MaskRepairContractError, match="NOT_DETERMINISTIC"):
        merge_repair_support(base, {5: proposals[0].masks}, proposals, prompts, assignments)
    changed = copy.deepcopy(prompts)
    changed[0]["points_px"][2] = [0.0, 0.0]
    with pytest.raises(V1MaskRepairContractError, match="NOT_DETERMINISTIC"):
        merge_repair_support(base, {5: proposals[0].masks}, proposals, changed, assignments,
                             prompt_policy=SPREAD_MAX_GAP_THREE)


@pytest.mark.parametrize("policy", ["OTHER", None, [], 3])
def test_unknown_policy_fail_closed_in_make_and_merge(policy):
    base, proposals, assignments = inputs()
    with pytest.raises(V1MaskRepairContractError, match="UNKNOWN_PROMPT_POLICY"):
        make_repair_prompts(base, proposals, assignments, prompt_policy=policy)
    with pytest.raises(V1MaskRepairContractError, match="UNKNOWN_PROMPT_POLICY"):
        merge_repair_support(base, {}, proposals, [], assignments, prompt_policy=policy)
