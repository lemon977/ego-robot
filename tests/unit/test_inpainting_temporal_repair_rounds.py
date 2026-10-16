"""Synthetic CPU evidence for the bounded same-rule temporal repair wrapper."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate.automation_mask import repair_current_masks


@pytest.fixture
def repair_case(monkeypatch):
    from chaoyang.exact78_baseline import mask as parent_mask
    from chaoyang.pick_place_1008 import visual
    from chaoyang.inpainting_candidate import automatic_masks, automatic_v1_seeds, v1_mask_repair

    shape = (4, 48, 64)
    rgb = np.zeros((*shape, 3), np.uint8)
    region = np.zeros(shape, bool)
    region[:2, 10:16, 10:16] = True
    region[3, 18:30, 10:30] = True
    base = {key: np.zeros(shape, bool) for key in
            ("core", "sam_core", "equipment", "removal", "protected_visible")}
    base["core"][:, 1:3, 1:3] = True
    base["sam_core"][:] = base["core"]
    base["equipment"][:, 40, 50] = True
    base["removal"][:] = base["core"] | base["equipment"]
    base["protected_visible"][:2, 10:12, 10:12] = True
    base["side_support"] = np.zeros((2, *shape), bool)
    base["side_support"][1, :2, 14, 14] = True
    base["effective_removal"], _ = visual.effective_clean_masks(base["removal"], base["protected_visible"])
    base["metadata"] = {"removal": {"observed_segmentation_dilation_px": 0, "temporal_bracket_frames": 0}}
    binding = {"frames_sha256": automatic_masks.frames_sha256(rgb), "raw_source_sha256": "a" * 64,
               "frame_map_sha256": "b" * 64, "model_sha256": "c" * 64}
    proposal = automatic_masks.RGBProposal(role="human", track_id=7, masks=region, **binding)
    calls = {"collect": 0, "associate": 0, "qualify": 0}

    def collect(sam, pixels, **kwargs):
        calls["collect"] += 1
        assert pixels is rgb
        return [proposal], {"status": "SYNTHETIC_RGB_PROPOSAL"}

    def seed_bundle(pixels, proposals, hints, **kwargs):
        calls["associate"] += 1
        return {"instance_association": [{"role": "human", "track_id": 7, "side": "left",
            "association": "SAME_CAPTURE_CURRENT_RGB_HINT", "side_hit_rates": [1.0, 0.0]}]}

    original_qualify = v1_mask_repair.qualify_identity_support

    def qualify(*args, **kwargs):
        calls["qualify"] += 1
        return original_qualify(*args, **kwargs)

    hints = np.empty((2, shape[0], 1, 2), float)
    hints[0] = [15.0, 15.0]
    hints[1] = [60.0, 44.0]
    geometry = {key: hints.copy() for key in ("hands", "forearm", "tracker")}
    monkeypatch.setattr(parent_mask, "pico_geometry", lambda session, scale: geometry)
    monkeypatch.setattr(visual, "_gate_diagnostic_geometry", lambda value, session: (value, []))
    monkeypatch.setattr(automatic_masks, "collect_sam_rgb_proposals", collect)
    monkeypatch.setattr(automatic_v1_seeds, "generate_v1_seed_bundle", seed_bundle)
    monkeypatch.setattr(v1_mask_repair, "qualify_identity_support", qualify)

    def invoke(**kwargs):
        return repair_current_masks(sam=kwargs.pop("sam", object()), rgb=rgb,
            session=SimpleNamespace(width=64), original_masks=base,
            original_objects=[{"role": "glove", "side": "left", "obj_id": 2}],
            raw_source_sha256=binding["raw_source_sha256"], frame_map_sha256=binding["frame_map_sha256"],
            model_sha256=binding["model_sha256"], **kwargs)

    def anchor_tracker(sam, pixels, prompts, anchor):
        assert len(prompts) == 1 and len(prompts[0]["points_px"]) == 3
        output = region[anchor].copy()
        # Unsupported background, protection, and opposite-side pixels are
        # deliberately supplied by POINT; only the legitimate RGB gap survives.
        output[35, 35] = True
        return {2: {anchor: output}}, {"status": "COMPLETE", "errors": [], "aborted": []}

    return SimpleNamespace(base=base, rgb=rgb, region=region, calls=calls, invoke=invoke, tracker=anchor_tracker)


def test_second_round_selects_remaining_maximum_and_keeps_first_round_gain(repair_case):
    case = repair_case
    before = deepcopy(case.base)
    final, audit, proposals, hints, first = case.invoke(repair_rounds=2, return_round_masks=True, tracker=case.tracker)
    assert case.calls == {"collect": 1, "associate": 1, "qualify": 1}
    assert [row["prompts"][0]["frame_idx"] for row in audit["rounds"]] == [3, 0]
    assert [row["input_mask_basis"] for row in audit["rounds"]] == ["FRESH_ORIGINAL_V1", "PREVIOUS_ROUND_CANDIDATE"]
    assert audit["point_tracking_calls"] == audit["repair_rounds"] == 2
    assert audit["prompts"] == audit["rounds"][0]["prompts"]
    assert audit["remaining_prompts"][0]["frame_idx"] == 1  # bounded: no third round
    assert np.any(first["core"][3] & ~before["core"][3])
    assert np.any(final["core"][0] & ~first["core"][0])
    for key in ("core", "sam_core", "equipment", "removal", "effective_removal"):
        assert not np.any(before[key] & ~final[key])
        assert not np.any(first[key] & ~final[key])
        assert np.array_equal(case.base[key], before[key])
        assert not np.shares_memory(first[key], final[key])
    assert np.array_equal(first["core"][3], final["core"][3])
    assert np.array_equal(final["protected_visible"], before["protected_visible"])
    added = final["core"] & ~before["core"]
    assert not np.any(added & ~case.region)
    assert not np.any(added & before["protected_visible"])
    assert not np.any(added & before["side_support"][1])
    assert audit["round_1_support_preservation"]["support_lost_pixels"] == {
        key: 0 for key in ("core", "sam_core", "equipment", "removal", "effective_removal")}
    assert len(proposals) == 1 and hints.shape == (2, 4, 3, 2)


def test_default_stays_four_values_and_opt_in_snapshot_is_independent(repair_case):
    case = repair_case
    legacy = case.invoke(tracker=case.tracker)
    assert len(legacy) == 4 and legacy[1]["repair_rounds"] == 1
    final, audit, _, _, first = case.invoke(return_round_masks=True, tracker=case.tracker)
    assert len(audit["rounds"]) == 1
    assert np.array_equal(final["core"], first["core"])
    first["core"][:] = False
    first["metadata"]["removal"]["observed_segmentation_dilation_px"] = 99
    assert final["core"].any()
    assert final["metadata"]["removal"]["observed_segmentation_dilation_px"] == 0


def test_exhausted_gap_skips_second_point_call(repair_case):
    case = repair_case
    calls = []

    def tracker(sam, pixels, prompts, anchor):
        calls.append(anchor)
        return {2: case.region.copy()}, {"status": "COMPLETE", "errors": [], "aborted": []}

    _, audit, _, _, _ = case.invoke(repair_rounds=2, return_round_masks=True, tracker=tracker)
    assert calls == [3]
    assert audit["rounds"][1]["point_tracking"]["status"] == "NO_RGB_GAP"
    assert audit["rounds"][1]["new_core_pixels_per_native_frame"] == [0, 0, 0, 0]
    assert audit["remaining_prompts"] == []


@pytest.mark.parametrize("report", [
    {"status": "PARTIAL_SUPPORT", "errors": [], "aborted": []},
    {"status": "COMPLETE", "errors": [{"message": "failure"}], "aborted": []},
    {"status": "COMPLETE", "errors": [], "aborted": [{"direction": "reverse"}]},
])
def test_incomplete_second_round_fails_without_mutating_input(repair_case, report):
    case = repair_case
    before, calls = deepcopy(case.base), []

    def tracker(sam, pixels, prompts, anchor):
        calls.append(anchor)
        if len(calls) == 2:
            return {}, report
        return case.tracker(sam, pixels, prompts, anchor)

    with pytest.raises(ValueError, match="AUTOMATION_REPAIR_POINT_TRACKING_INCOMPLETE"):
        case.invoke(repair_rounds=2, tracker=tracker)
    assert calls == [3, 0]
    assert all(np.array_equal(case.base[key], value) for key, value in before.items() if isinstance(value, np.ndarray))


@pytest.mark.parametrize("value", [0, 4, True, 1.0, "2", None])
def test_round_budget_rejected_before_proposals(repair_case, value):
    with pytest.raises(ValueError, match="AUTOMATION_REPAIR_ROUNDS_MUST_BE_ONE_TO_THREE"):
        repair_case.invoke(repair_rounds=value)
    assert repair_case.calls["collect"] == 0


def test_snapshot_option_requires_bool_before_proposals(repair_case):
    with pytest.raises(ValueError, match="AUTOMATION_RETURN_ROUND_MASKS_BOOL_REQUIRED"):
        repair_case.invoke(return_round_masks=1)
    assert repair_case.calls["collect"] == 0


def test_each_round_uses_new_adapter_state_and_clears_it(repair_case):
    case = repair_case

    class SyntheticModel:
        def __init__(self):
            self.states = []
            self.anchors = []

        def init_state(self, **kwargs):
            state = {}
            self.states.append(state)
            return state

        def _cache_frame_outputs(self, state, native_frame, outputs):
            state.setdefault("cached_frame_outputs", {})[native_frame] = outputs

        def add_prompt(self, *, inference_state, frame_idx, **kwargs):
            assert len(inference_state["cached_frame_outputs"]) == len(case.rgb)
            inference_state["anchor"] = frame_idx
            self.anchors.append(frame_idx)
            return frame_idx, {"out_binary_masks": case.region[frame_idx][None], "out_obj_ids": np.array([2])}

        def propagate_in_video(self, *, inference_state, start_frame_idx, reverse, **kwargs):
            indices = range(start_frame_idx) if reverse else range(start_frame_idx, len(case.rgb))
            for frame in indices:
                yield frame, ({"out_binary_masks": case.region[frame][None], "out_obj_ids": np.array([2])}
                              if frame == inference_state["anchor"] else {})

    model = SyntheticModel()
    final, audit, _, _, first = case.invoke(sam=SimpleNamespace(model=model), repair_rounds=2, return_round_masks=True)
    assert model.anchors == [3, 0]
    assert len(model.states) == 2 and model.states[0] is not model.states[1]
    assert all(state == {} for state in model.states)
    assert audit["point_tracking_calls"] == 2
    assert all(row["point_tracking"]["status"] == "COMPLETE" for row in audit["rounds"])
    assert not np.any(first["removal"] & ~final["removal"])
