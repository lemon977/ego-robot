"""Synthetic CPU contracts; these tests are not a real SAM/visual evaluation."""

from dataclasses import replace
import inspect
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate.automatic_masks import (
    AutomaticMaskContractError, PROMPTS, RGBProposal, build_automatic_masks,
    collect_sam_rgb_proposals, frames_sha256, refine_sam_rgb_proposals,
)


SIGNATURES = {"raw_source_sha256": "a" * 64, "frame_map_sha256": "b" * 64,
              "model_sha256": "c" * 64}


@pytest.fixture(autouse=True)
def fake_point_tensor_api(monkeypatch):
    """No Torch/CUDA loading is needed to verify a fake model's API contract."""
    class FakeTensor:
        def __init__(self, values, dtype):
            self.value = np.asarray(values, dtype=dtype)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def tolist(self):
            return self.value.tolist()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=lambda values, dtype: FakeTensor(values, dtype),
        float32=np.float32, int64=np.int64))


def rgb(count=4):
    # Isotropically scaled native candidate domain, no file/model access.
    return np.zeros((count, 48, 64, 3), np.uint8)


def rectangle(frames, x0, y0, x1, y1):
    mask = np.zeros(frames.shape[:3], bool)
    mask[:, y0:y1, x0:x1] = True
    return mask


def proposal(frames, role, masks, track_id=1, **kwargs):
    return RGBProposal(role, track_id, masks, frames_sha256(frames),
                       **SIGNATURES, **kwargs)


def run(frames, proposals, **kwargs):
    return build_automatic_masks(frames, proposals, **SIGNATURES, **kwargs)


def test_api_has_no_session_task_configuration_seed_or_external_point_arguments():
    fields = set(inspect.signature(build_automatic_masks).parameters)
    assert fields == {"frames_rgb", "proposals", "raw_source_sha256", "frame_map_sha256",
                      "model_sha256", "projection_hints"}
    fields = set(inspect.signature(collect_sam_rgb_proposals).parameters)
    assert fields == {"sam", "frames_rgb", "raw_source_sha256", "frame_map_sha256",
                      "model_sha256", "controller_text_prompt"}
    assert PROMPTS["object"] == "object"
    assert "paper cup" not in repr(PROMPTS)
    assert "white" not in PROMPTS["controller"]
    assert "MANUS" not in PROMPTS["glove"]


def test_human_and_device_visible_support_separated_from_object_protection():
    frames = rgb()
    hand = rectangle(frames, 18, 12, 28, 29)
    controller = rectangle(frames, 27, 20, 33, 27)
    obj = rectangle(frames, 28, 25, 38, 33)
    output = run(frames, [proposal(frames, "human", hand),
                          proposal(frames, "controller", controller),
                          proposal(frames, "object", obj)])
    assert np.array_equal(output["human"], hand)
    assert np.array_equal(output["equipment"], controller)
    assert np.array_equal(output["protected_visible"], obj & ~(hand | controller))
    assert np.array_equal(output["semantic_conflict"], obj & (hand | controller))
    assert not (output["protected_visible"] & output["effective_removal"]).any()
    assert not (output["protected_visible"] & output["core"]).any()
    assert output["metadata"]["conflict_is_protection"] is False
    assert output["unknown"][output["semantic_conflict"]].all()


def test_internal_points_are_inside_current_proposal_and_reproducible():
    frames = rgb()
    hand = rectangle(frames, 10, 10, 30, 34)
    hand[0] = False
    output = run(frames, [proposal(frames, "human", hand)])
    prompt, = output["automatic_points"]
    frame = prompt["native_frame_index"]
    x, y = map(int, prompt["points_px"][0])
    assert frame == 1
    assert hand[frame, y, x]
    assert prompt["source"] == "CURRENT_RGB_MODEL_MASK_INTERIOR_AUTO"
    assert prompt["points_px"] == run(frames, [proposal(frames, "human", hand)])["automatic_points"][0]["points_px"]
    assert output["metadata"]["automatic_point_refinement_executed"] is False


def test_projection_alone_never_creates_foreground():
    frames = rgb()
    hints = np.full((2, len(frames), 1, 2), np.nan)
    hints[0, :, 0] = [24, 24]
    output = run(frames, [], projection_hints=hints)
    assert not output["core"].any()
    assert not output["removal"].any()
    assert not output["automatic_points"]
    assert output["metadata"]["projection_is_pixel_evidence"] is False


def test_projection_can_assign_side_without_borrowing_hardware_pixels():
    frames = rgb()
    hand = rectangle(frames, 15, 10, 30, 34)
    hints = np.full((2, len(frames), 1, 2), np.nan)
    hints[0, :, 0] = [20, 20]
    output = run(frames, [proposal(frames, "human", hand)], projection_hints=hints)
    assert output["automatic_points"][0]["side"] == "LEFT"
    assert np.array_equal(output["core"], hand)


def test_remote_device_component_is_not_removed_even_if_track_touches_wearer():
    frames = rgb()
    hand = rectangle(frames, 10, 10, 20, 30)
    device = rectangle(frames, 19, 12, 25, 20)
    remote = rectangle(frames, 50, 35, 58, 42)
    output = run(frames, [proposal(frames, "human", hand),
                          proposal(frames, "controller", device | remote)])
    assert np.array_equal(output["equipment"], device)
    assert output["unknown"][remote].all()
    assert not output["core"][remote].any()


def test_device_without_visible_human_is_unknown_not_geometry_recovered():
    frames = rgb()
    device = rectangle(frames, 20, 20, 30, 30)
    output = run(frames, [proposal(frames, "controller", device)])
    assert not output["equipment"].any()
    assert output["unknown"][device].all()


def test_explicit_low_support_frame_stays_unknown_not_protected():
    frames = rgb()
    hand = rectangle(frames, 20, 10, 30, 28)
    unsupported = np.array([False, True, False, False])
    output = run(frames, [proposal(frames, "human", hand, unsupported_frames=unsupported)])
    assert not output["human"][1].any()
    assert output["unknown"][1][hand[1]].all()
    # The frozen temporal bracket can widen deletion; it does not increase
    # actual human proposal support or reclassify UNKNOWN as observed.
    assert output["metadata"]["proposal_tracks"][0]["selected_pixels_per_native_frame"][1] == 0
    assert output["metadata"]["proposal_tracks"][0]["unsupported_native_frames"] == [1]
    assert output["metadata"]["wearer_support_status_by_native_frame"][1] == "UNKNOWN"


def test_large_object_keeps_existing_shape_rule_and_is_explicitly_unknown():
    frames = rgb()
    hand = rectangle(frames, 20, 10, 30, 28)
    large = rectangle(frames, 3, 3, 60, 44)
    output = run(frames, [proposal(frames, "human", hand),
                          proposal(frames, "object", large, source="RGB_CLASS_AGNOSTIC_INSTANCE")])
    assert not output["protected_visible"].any()
    assert output["unknown"][large].all()
    assert output["metadata"]["object_shape_rule"].endswith("LARGE_OBJECTS_UNKNOWN")


@pytest.mark.parametrize("field", ["frames_sha256", "raw_source_sha256", "frame_map_sha256", "model_sha256"])
def test_proposal_signature_drift_rejected(field):
    frames = rgb()
    value = proposal(frames, "human", rectangle(frames, 10, 10, 20, 20))
    with pytest.raises(AutomaticMaskContractError, match="SHA_MISMATCH"):
        run(frames, [replace(value, **{field: "d" * 64})])


def test_manual_seed_dicts_and_geometry_sources_rejected():
    frames = rgb()
    with pytest.raises(AutomaticMaskContractError, match="NOT_MANUAL_SEED"):
        run(frames, [{"role": "human", "points_px": [[20, 20]]}])
    value = proposal(frames, "human", rectangle(frames, 10, 10, 20, 20))
    with pytest.raises(AutomaticMaskContractError, match="SOURCE_OR_ID"):
        run(frames, [replace(value, source="HARDWARE_PROJECTION")])


def test_proposal_native_domain_and_duplicate_identity_rejected():
    frames = rgb()
    value = proposal(frames, "human", rectangle(frames, 10, 10, 20, 20))
    with pytest.raises(AutomaticMaskContractError, match="DUPLICATE"):
        run(frames, [value, value])
    with pytest.raises(AutomaticMaskContractError, match="MASK_DOMAIN"):
        run(frames, [replace(value, masks=value.masks[:2])])


def test_non_native_rgb_and_wrong_aspect_are_rejected():
    with pytest.raises(AutomaticMaskContractError, match="RAW_RGB_UINT8"):
        run(rgb().astype(np.float32), [])
    with pytest.raises(AutomaticMaskContractError, match="ISOTROPIC"):
        run(np.zeros((4, 30, 64, 3), np.uint8), [])


def test_sam_bridge_uses_fixed_prompts_and_native_frame_zero_without_task_or_points():
    frames = rgb()
    mask = rectangle(frames, 10, 10, 20, 20)

    class FakeSam:
        aborted = []

        def __init__(self):
            self.calls = []

        def track_text(self, source, prompt, anchor):
            self.calls.append((prompt, anchor, len(source)))
            return {7: {0: np.packbits(mask[0])}} if prompt == PROMPTS["human"] else {}

    sam = FakeSam()
    proposals, receipt = collect_sam_rgb_proposals(sam, frames, **SIGNATURES)
    assert sam.calls == [(prompt, 0, len(frames)) for prompt in PROMPTS.values()]
    assert len(proposals) == 1
    assert proposals[0].masks[0].any()
    assert not proposals[0].masks[1:].any()
    assert receipt["object_provider"] == "SAM31_GENERIC_OBJECT_CONCEPT_NOT_ALL_INSTANCES"
    output = run(frames, proposals)
    assert output["metadata"]["producer_manual_seed_reads"] == 0
    assert output["metadata"]["per_session_overrides"] == 0
    assert output["metadata"]["training_eligible"] is False


def test_sam_bridge_fails_explicitly_on_bad_frame_or_absent_model_api():
    frames = rgb()
    with pytest.raises(AutomaticMaskContractError, match="PROVIDER_REQUIRED"):
        collect_sam_rgb_proposals(object(), frames, **SIGNATURES)

    class FakeSam:
        def track_text(self, source, prompt, anchor):
            return {0: {len(source): np.zeros(384, np.uint8)}}

    with pytest.raises(AutomaticMaskContractError, match="FRAME_OUT_OF_DOMAIN"):
        collect_sam_rgb_proposals(FakeSam(), frames, **SIGNATURES)


class PromptOnlyFakeSam:
    """Record text calls only; no checkpoint, Torch or neural inference."""
    aborted = []

    def __init__(self):
        self.calls = []

    def track_text(self, frames, prompt, anchor):
        self.calls.append((prompt, anchor, len(frames)))
        return {}


def test_explicit_default_controller_prompt_is_complete_noop():
    omitted, explicit = PromptOnlyFakeSam(), PromptOnlyFakeSam()
    result = collect_sam_rgb_proposals(omitted, rgb(), **SIGNATURES)
    assert result == collect_sam_rgb_proposals(
        explicit, rgb(), **SIGNATURES, controller_text_prompt=None)
    assert omitted.calls == explicit.calls == [(prompt, 0, 4) for prompt in PROMPTS.values()]
    assert PROMPTS["controller"] == "a handheld or hand-mounted VR controller"


def test_controller_candidate_changes_exactly_one_global_text_without_mutating_defaults():
    defaults = dict(PROMPTS)
    baseline, candidate = PromptOnlyFakeSam(), PromptOnlyFakeSam()
    _, old = collect_sam_rgb_proposals(baseline, rgb(), **SIGNATURES)
    _, new = collect_sam_rgb_proposals(candidate, rgb(), **SIGNATURES,
                                     controller_text_prompt="VR controller")
    assert len(candidate.calls) == len(baseline.calls) == 6
    changed = [i for i, pair in enumerate(zip(old["calls"], new["calls"])) if pair[0] != pair[1]]
    assert changed == [list(PROMPTS).index("controller")]
    controller = new["calls"][changed[0]]
    assert controller == {**old["calls"][changed[0]], "prompt": "VR controller"}
    assert all(call[1:] == (0, 4) for call in candidate.calls)
    assert PROMPTS == defaults


@pytest.mark.parametrize("bad", ["", "vr controller", "custom controller", PROMPTS["controller"],
                                0, True, [], {}, np.str_("VR controller")])
def test_controller_override_invalid_value_rejected_before_any_sam_call(bad):
    sam = PromptOnlyFakeSam()
    with pytest.raises(AutomaticMaskContractError, match="CONTROLLER_TEXT_PROMPT"):
        collect_sam_rgb_proposals(sam, rgb(), **SIGNATURES, controller_text_prompt=bad)
    assert sam.calls == []


def test_other_prompt_overrides_are_not_accepted_as_hidden_algorithm_controls():
    sam = PromptOnlyFakeSam()
    with pytest.raises(TypeError, match="unexpected keyword"):
        collect_sam_rgb_proposals(sam, rgb(), **SIGNATURES, glove_text_prompt="glove")
    assert sam.calls == []


def test_missing_source_signature_never_produces_credible_masks():
    with pytest.raises(AutomaticMaskContractError, match="RAW_SOURCE_SHA_REQUIRED"):
        build_automatic_masks(rgb(), [], **{**SIGNATURES, "raw_source_sha256": "ABSENT"})


class FakeOfficialPointModel:
    """Shape/API fake only: it loads no checkpoint and performs no inference."""

    def __init__(self, masks, retract_frame=None):
        self.masks = masks
        self.retract_frame = retract_frame
        self.calls = []
        self.directions = []
        self.state = None

    def init_state(self, *, resource_path, offload_video_to_cpu, async_loading_frames):
        assert len(resource_path) == len(self.masks)
        assert offload_video_to_cpu is True
        assert async_loading_frames is False
        self.state = {"cached_frame_outputs": {}, "ids": []}
        return self.state

    def _cache_frame_outputs(self, state, frame, outputs):
        assert outputs == {}
        state["cached_frame_outputs"][frame] = outputs

    def add_prompt(self, *, inference_state, frame_idx, text_str, points, point_labels,
                   obj_id, clear_old_points, rel_coordinates, output_prob_thresh):
        assert text_str is None
        assert clear_old_points is True and rel_coordinates is True
        assert output_prob_thresh == .5
        width, height = self.masks.shape[2], self.masks.shape[1]
        pixels = points.detach().cpu().numpy() * [width, height]
        x, y = np.rint(pixels[0]).astype(int)
        assert self.masks[frame_idx, y, x]
        assert point_labels.tolist() == [1]
        self.calls.append({"native_frame": frame_idx, "points_px": [x, y], "obj_id": obj_id})
        if obj_id not in inference_state["ids"]:
            inference_state["ids"].append(obj_id)
        return frame_idx, {"out_obj_ids": np.array([obj_id]),
                           "out_binary_masks": self.masks[frame_idx][None]}

    def propagate_in_video(self, *, inference_state, start_frame_idx,
                           max_frame_num_to_track, reverse, output_prob_thresh):
        assert output_prob_thresh == .5
        self.directions.append((start_frame_idx, max_frame_num_to_track, reverse))
        frames = (range(start_frame_idx, -1, -1) if reverse
                  else range(start_frame_idx, len(self.masks)))
        ids = np.asarray(inference_state["ids"])
        for frame in list(frames)[:max_frame_num_to_track]:
            mask = (np.zeros_like(self.masks[frame]) if frame == self.retract_frame
                    else self.masks[frame])
            yield frame, {"out_obj_ids": ids,
                          "out_binary_masks": np.stack([mask for _ in ids])}


def test_auto_points_really_feed_official_api_without_inventing_side_or_manual_evidence():
    frames = rgb(5)
    mask = rectangle(frames, 10, 10, 20, 20)
    mask[2, 10:35, 10:30] = True
    model = FakeOfficialPointModel(mask)
    sam = type("FakeSam", (), {"model": model})()
    original = proposal(frames, "human", mask)
    refined, receipt = refine_sam_rgb_proposals(sam, frames, [original], **SIGNATURES)
    assert [call["native_frame"] for call in model.calls] == [0, 2, 4]
    assert receipt["automatic_points_submitted"] == 3
    assert receipt["manual_points_submitted"] == 0
    assert receipt["manual_visible_confirmation_claim"] is False
    assert receipt["objects"][0]["side"] == "UNKNOWN"
    assert receipt["status"] == "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE"
    assert refined[0].source == "SAM31_RGB_POINT_AUTO"
    assert refined[0].track_id == original.track_id
    assert np.array_equal(refined[0].masks, mask)
    assert model.state == {}
    final_metadata = run(frames, refined)["metadata"]
    assert final_metadata["automatic_point_refinement_executed"] is True
    assert final_metadata["producer_scope"] == "BUILD_ONLY_NO_MODEL_EXECUTION"
    assert final_metadata["automatic_point_refinement_executed_by_this_builder"] is False
    assert final_metadata["upstream_point_refinement_status"] == "RECEIPT_PAYLOAD_BOUND"
    assert final_metadata["upstream_point_refinement_receipts"][0]["payload_sha256"] == receipt["receipt_payload_sha256"]
    body = {key: value for key, value in receipt.items()
            if key not in {"receipt_payload_sha256", "receipt_signature_scope"}}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest() == receipt["receipt_payload_sha256"]


def test_point_refinement_reverse_includes_native_frame_zero_without_synthesizing_it():
    frames = rgb()
    mask = rectangle(frames, 10, 10, 20, 20)
    mask[0] = False
    model = FakeOfficialPointModel(mask)
    sam = type("FakeSam", (), {"model": model})()
    refined, receipt = refine_sam_rgb_proposals(
        sam, frames, [proposal(frames, "human", mask)], **SIGNATURES)
    assert model.directions == [(1, 3, False), (1, 2, True)]
    assert not refined[0].masks[0].any()
    assert receipt["required_support_missing"] == []


def test_explicit_empty_point_output_retracts_anchor_and_preserves_unknown():
    frames = rgb()
    mask = rectangle(frames, 10, 10, 20, 20)
    model = FakeOfficialPointModel(mask, retract_frame=3)
    sam = type("FakeSam", (), {"model": model})()
    refined, receipt = refine_sam_rgb_proposals(
        sam, frames, [proposal(frames, "human", mask)], **SIGNATURES)
    assert receipt["status"] == "INCOMPLETE_RGB_PROMPT_SUPPORT"
    assert receipt["required_support_missing"] == [
        {"role": "human", "track_id": 1, "native_frame_index": 3}]
    assert refined[0].unsupported_frames[3]
    assert not refined[0].masks[3].any()
    assert run(frames, refined)["metadata"]["wearer_support_status_by_native_frame"][3] == "UNKNOWN"


def test_no_rgb_points_does_not_allocate_or_call_model():
    refined, receipt = refine_sam_rgb_proposals(object(), rgb(), [], **SIGNATURES)
    assert refined == []
    assert receipt["status"] == "NO_RGB_SUPPORTED_AUTOMATIC_POINTS"
    assert receipt["model_execution_performed"] is False


def test_point_refinement_validates_model_api_and_existing_object_capacity_before_allocation():
    frames = rgb()
    mask = rectangle(frames, 10, 10, 20, 20)
    with pytest.raises(AutomaticMaskContractError, match="MODEL_API_REQUIRED"):
        refine_sam_rgb_proposals(object(), frames, [proposal(frames, "human", mask)], **SIGNATURES)
    many = [proposal(frames, "human", mask, track_id=index) for index in range(17)]
    with pytest.raises(AutomaticMaskContractError, match="CAPACITY_EXCEEDED"):
        refine_sam_rgb_proposals(object(), frames, many, **SIGNATURES)


def test_point_refinement_rejects_external_manual_point_arguments():
    fields = set(inspect.signature(refine_sam_rgb_proposals).parameters)
    assert fields == {"sam", "frames_rgb", "proposals", "raw_source_sha256", "frame_map_sha256",
                      "model_sha256", "projection_hints"}


def test_point_model_recovery_does_not_inherit_old_text_failure_flags():
    frames = rgb()
    mask = rectangle(frames, 10, 10, 20, 20)
    failed_text = np.array([False, True, False, False])
    original = proposal(frames, "human", mask, unsupported_frames=failed_text)
    assert not run(frames, [original])["human"][1].any()
    model = FakeOfficialPointModel(mask)
    sam = type("FakeSam", (), {"model": model})()
    refined, receipt = refine_sam_rgb_proposals(sam, frames, [original], **SIGNATURES)
    assert not refined[0].unsupported_frames[1]
    assert refined[0].prior_unsupported_frames[1]
    output = run(frames, refined)
    assert output["human"][1].any()
    assert output["metadata"]["wearer_support_status_by_native_frame"][1] == "RGB_MODEL_PROPOSAL"
    row = output["metadata"]["proposal_tracks"][0]
    assert row["prior_stage_unsupported_native_frames"] == [1]
    assert row["unsupported_native_frames"] == []
    assert receipt["objects"][0]["point_model_supported_native_frames"] == [0, 1, 2, 3]


def test_point_source_without_receipt_never_claims_verified_upstream_execution():
    frames = rgb()
    original = proposal(frames, "human", rectangle(frames, 10, 10, 20, 20))
    unbound = replace(original, source="SAM31_RGB_POINT_AUTO")
    output = run(frames, [unbound])
    metadata = output["metadata"]
    assert metadata["automatic_point_refinement_executed"] is None
    assert metadata["upstream_point_refinement_status"] == "UNBOUND_POINT_SOURCE_DECLARATION"
    assert {"code": "POINT_REFINEMENT_RECEIPT_UNBOUND", "status": "UNKNOWN"} in metadata["quality_problems"]


def test_rgb_glove_seeds_wearer_without_renaming_it_human():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    output = run(frames, [proposal(frames, "glove", glove)])
    assert not output["human"].any() and not output["sam_core"].any()
    for name in ("equipment", "wearer_support", "core"):
        assert np.array_equal(output[name], glove)
    assert np.array_equal(output["equipment_roles"]["glove"], glove)
    point, = output["automatic_points"]
    assert point["role"] == "glove" and point["side"] == "UNKNOWN"
    metadata = output["metadata"]
    assert metadata["human_support_status_by_native_frame"] == ["UNKNOWN"] * len(frames)
    assert metadata["wearer_support_sources_by_native_frame"] == [["glove"]] * len(frames)
    assert metadata["training_eligible"] is False


def test_same_glove_rule_is_used_before_and_after_actual_point_api_bridge():
    frames = rgb(5)
    glove = rectangle(frames, 20, 12, 30, 28)
    model = FakeOfficialPointModel(glove)
    sam = type("FakeSam", (), {"model": model})()
    refined, receipt = refine_sam_rgb_proposals(
        sam, frames, [proposal(frames, "glove", glove, track_id=9)], **SIGNATURES)
    assert [call["native_frame"] for call in model.calls] == [0, 4]
    assert receipt["objects"][0]["role"] == "glove"
    assert receipt["objects"][0]["track_id"] == 9
    assert refined[0].role == "glove" and refined[0].track_id == 9
    assert refined[0].source == "SAM31_RGB_POINT_AUTO"
    output = run(frames, refined)
    assert np.array_equal(output["equipment_roles"]["glove"], glove)
    assert not output["human"].any()
    assert output["metadata"]["automatic_point_refinement_executed"] is True


def test_glove_unsupported_frame_stays_unknown_despite_temporal_removal_bracket():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    unsupported = np.array([False, True, False, False])
    output = run(frames, [proposal(frames, "glove", glove, unsupported_frames=unsupported)])
    assert not output["equipment_roles"]["glove"][1].any()
    assert not output["wearer_support"][1].any()
    assert output["unknown"][1][glove[1]].all()
    assert output["metadata"]["wearer_support_status_by_native_frame"][1] == "UNKNOWN"
    assert output["metadata"]["wearer_support_sources_by_native_frame"][1] == []
    assert output["metadata"]["proposal_tracks"][0]["unsupported_native_frames"] == [1]


def test_glove_absent_frame_is_not_invented_or_counted_as_supported():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    glove[1] = False
    output = run(frames, [proposal(frames, "glove", glove)])
    assert not output["wearer_support"][1].any()
    assert output["metadata"]["wearer_support_status_by_native_frame"][1] == "UNKNOWN"
    assert output["metadata"]["proposal_tracks"][0]["selected_pixels_per_native_frame"][1] == 0


def test_point_glove_retraction_remains_missing_not_restored_from_text_mask():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    model = FakeOfficialPointModel(glove, retract_frame=3)
    refined, receipt = refine_sam_rgb_proposals(
        type("FakeSam", (), {"model": model})(), frames,
        [proposal(frames, "glove", glove)], **SIGNATURES)
    assert receipt["status"] == "INCOMPLETE_RGB_PROMPT_SUPPORT"
    output = run(frames, refined)
    assert not output["equipment_roles"]["glove"][3].any()
    assert output["metadata"]["wearer_support_status_by_native_frame"][3] == "UNKNOWN"
    assert refined[0].unsupported_frames[3]


def test_controller_can_attach_to_glove_but_remote_components_stay_unknown():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    near = rectangle(frames, 29, 15, 35, 23)
    remote = rectangle(frames, 50, 35, 58, 43)
    output = run(frames, [proposal(frames, "controller", near | remote),
                          proposal(frames, "glove", glove)])
    assert np.array_equal(output["equipment_roles"]["controller"], near)
    assert output["unknown"][remote].all()
    assert not output["core"][remote].any()
    # Frozen role order, not caller proposal order, decides downstream stages.
    reverse = run(frames, [proposal(frames, "glove", glove),
                           proposal(frames, "controller", near | remote)])
    assert np.array_equal(output["core"], reverse["core"])


def test_objects_and_devices_cannot_circularly_bootstrap_wearer():
    frames = rgb()
    device = rectangle(frames, 20, 12, 30, 28)
    obj = rectangle(frames, 29, 15, 35, 23)
    output = run(frames, [proposal(frames, "controller", device),
                          proposal(frames, "object", obj)])
    assert not output["wearer_support"].any()
    assert not output["core"].any()
    assert not output["protected_visible"].any()
    assert output["unknown"][device].all()


def test_object_overlap_does_not_relabel_glove_as_protected_scene_pixels():
    frames = rgb()
    glove = rectangle(frames, 20, 12, 30, 28)
    output = run(frames, [proposal(frames, "glove", glove),
                          proposal(frames, "object", glove)])
    assert np.array_equal(output["equipment_roles"]["glove"], glove)
    assert not output["protected_visible"].any()
    assert output["unknown"][glove].all()
    assert output["metadata"]["object_shape_rule"].endswith("LARGE_OBJECTS_UNKNOWN")


def test_empty_glove_and_projection_alone_cannot_allocate_point_model():
    frames = rgb()
    empty = np.zeros(frames.shape[:3], bool)
    hints = np.full((2, len(frames), 1, 2), np.nan)
    hints[0, :, 0] = [25, 20]
    refined, receipt = refine_sam_rgb_proposals(
        object(), frames, [proposal(frames, "glove", empty)],
        **SIGNATURES, projection_hints=hints)
    assert receipt["model_execution_performed"] is False
    output = run(frames, refined, projection_hints=hints)
    assert not output["wearer_support"].any() and not output["automatic_points"]
