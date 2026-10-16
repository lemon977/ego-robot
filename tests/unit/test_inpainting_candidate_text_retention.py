"""Pure CPU fake-SAM regression; not fresh Mask or visual qualification."""
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automatic_masks as a
from chaoyang.inpainting_candidate import model_stages as stages
from chaoyang.inpainting_candidate.contracts import file_ref, verify_sealed_stage


SIG = {"raw_source_sha256": "a" * 64, "frame_map_sha256": "b" * 64, "model_sha256": "c" * 64}


@pytest.fixture(autouse=True)
def fake_tensor_api(monkeypatch):
    class Tensor:
        def __init__(self, value, dtype): self.value = np.asarray(value, dtype=dtype)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
        def tolist(self): return self.value.tolist()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=lambda value, dtype: Tensor(value, dtype), float32=np.float32, int64=np.int64))


def frames(): return np.zeros((4, 48, 64, 3), np.uint8)


def rect(rgb, x0=10, y0=10, x1=30, y1=30):
    value = np.zeros(rgb.shape[:3], bool)
    value[:, y0:y1, x0:x1] = True
    return value


def proposal(rgb, role="human", mask=None, track=7, **kwargs):
    return a.RGBProposal(role, track, rect(rgb) if mask is None else mask,
                         a.frames_sha256(rgb), **SIG, **kwargs)


class FakePointModel:
    def __init__(self, output): self.output, self.calls = output, []
    def init_state(self, **kwargs):
        self.calls.append(("init", len(kwargs["resource_path"])))
        return {"cached_frame_outputs": {}, "ids": []}
    def _cache_frame_outputs(self, state, frame, output): state["cached_frame_outputs"][frame] = output
    def _out(self, state, index):
        return {"out_obj_ids": np.asarray(state["ids"], np.int64),
                "out_binary_masks": np.stack([self.output[index]] * len(state["ids"]))}
    def add_prompt(self, **kwargs):
        assert kwargs["output_prob_thresh"] == .5 and kwargs["text_str"] is None
        assert kwargs["point_labels"].tolist() == [1]
        state, index, obj = kwargs["inference_state"], kwargs["frame_idx"], kwargs["obj_id"]
        self.calls.append(("prompt", index, obj, kwargs["points"].tolist()))
        if obj not in state["ids"]: state["ids"].append(obj)
        return index, self._out(state, index)
    def propagate_in_video(self, **kwargs):
        self.calls.append(("propagate", kwargs["start_frame_idx"], kwargs["max_frame_num_to_track"], kwargs["reverse"]))
        indices = (range(kwargs["start_frame_idx"], -1, -1) if kwargs["reverse"] else
                   range(kwargs["start_frame_idx"], len(self.output)))
        for index in list(indices)[:kwargs["max_frame_num_to_track"]]:
            yield index, self._out(kwargs["inference_state"], index)


def refine(rgb, proposals, output=None, policy=a.TEXT_WEARER_RETENTION):
    model = FakePointModel(np.zeros(rgb.shape[:3], bool) if output is None else output)
    result, receipt = a.refine_sam_rgb_proposals_with_text_retention(
        SimpleNamespace(model=model), rgb, proposals, **SIG, text_wearer_retention=policy)
    return result, receipt, model


def build(rgb, proposals): return a.build_automatic_masks(rgb, proposals, **SIG)


def test_legacy_abi_and_none_are_exact_default_noop():
    assert set(inspect.signature(a.refine_sam_rgb_proposals).parameters) == {
        "sam", "frames_rgb", "proposals", "raw_source_sha256", "frame_map_sha256",
        "model_sha256", "projection_hints"}
    rgb, originals = frames(), [proposal(frames())]
    model = FakePointModel(np.zeros(rgb.shape[:3], bool))
    old, old_receipt = a.refine_sam_rgb_proposals(SimpleNamespace(model=model), rgb, originals, **SIG)
    new, new_receipt, new_model = refine(rgb, originals, policy=None)
    assert old_receipt == new_receipt and model.calls == new_model.calls
    assert old[0].source == new[0].source == "SAM31_RGB_POINT_AUTO"
    assert np.array_equal(old[0].masks, new[0].masks)
    assert build(rgb, old)["metadata"] == build(rgb, new)["metadata"]
    assert not any(key.startswith("wearer_origin__") for key in build(rgb, new))


@pytest.mark.parametrize("role", ["human", "glove"])
def test_text_only_support_retained_with_pixel_origin_not_point_measurement(role):
    rgb = frames()
    original = proposal(rgb, role)
    result, receipt, model = refine(rgb, [original])
    default, _, default_model = refine(rgb, [original], policy=None)
    assert model.calls == default_model.calls
    assert len(result) == 1 and result[0].track_id == original.track_id and result[0].role == role
    assert result[0].source == a.TEXT_POINT_WEARER_UNION
    assert np.array_equal(result[0].retained_text_masks, original.masks)
    assert not result[0].point_masks.any()
    assert np.array_equal(result[0].masks, original.masks)
    assert not default[0].masks.any()
    built = build(rgb, result)
    assert np.array_equal(built["wearer_origin__" + role + "__retained_text_masks"], original.masks)
    assert not built["wearer_origin__" + role + "__point_masks"].any()
    assert built["metadata"]["retained_text_is_point_observation"] is False
    assert built["metadata"]["proposal_tracks"][0]["point_stage_unsupported_native_frames"] == [0, 1, 2, 3]
    assert receipt["objects"][0]["point_model_supported_native_frames"] == []
    assert receipt["model_execution_performed"] is True
    assert built["metadata"]["training_eligible"] is False


def test_union_has_disjoint_text_only_and_actual_point_origins():
    rgb, text = frames(), rect(frames())
    point = rect(rgb, 20, 20, 35, 35)
    result, _, _ = refine(rgb, [proposal(rgb, mask=text)], output=point)
    assert np.array_equal(result[0].point_masks, point)
    assert np.array_equal(result[0].retained_text_masks, text & ~point)
    assert np.array_equal(result[0].masks, text | point)
    assert not (result[0].retained_text_masks & result[0].point_masks).any()


@pytest.mark.parametrize("field", list(SIG) + ["frames_sha256"])
def test_signature_drift_rejected_before_fake_model(field):
    rgb, original = frames(), proposal(frames())
    model = FakePointModel(np.zeros(rgb.shape[:3], bool))
    with pytest.raises(a.AutomaticMaskContractError, match="SHA_MISMATCH"):
        a.refine_sam_rgb_proposals_with_text_retention(SimpleNamespace(model=model), rgb,
            [replace(original, **{field: "e" * 64})], **SIG, text_wearer_retention=a.TEXT_WEARER_RETENTION)
    assert model.calls == []


@pytest.mark.parametrize("value", ["", "PRESERVE", 0, False, [], {}])
def test_invalid_strategy_rejected_before_model(value):
    model = FakePointModel(np.zeros(frames().shape[:3], bool))
    with pytest.raises(a.AutomaticMaskContractError, match="FROZEN_POLICY"):
        a.refine_sam_rgb_proposals_with_text_retention(SimpleNamespace(model=model), frames(),
            [], **SIG, text_wearer_retention=value)
    assert model.calls == []


def test_text_unsupported_and_absent_frames_never_gain_retained_support():
    rgb, text = frames(), rect(frames())
    text[0] = False
    text[3] = False
    original = proposal(rgb, mask=text, unsupported_frames=np.asarray([False, True, False, False]))
    result, _, _ = refine(rgb, [original])
    assert result[0].retained_text_masks.any(axis=(1, 2)).tolist() == [False, False, True, False]
    assert build(rgb, result)["metadata"]["wearer_support_status_by_native_frame"] == ["UNKNOWN", "UNKNOWN", "RGB_MODEL_PROPOSAL", "UNKNOWN"]


def test_entire_side_without_text_support_no_model_or_generated_pixels():
    rgb = frames()
    result, receipt = a.refine_sam_rgb_proposals_with_text_retention(
        object(), rgb, [proposal(rgb, "glove", np.zeros(rgb.shape[:3], bool))],
        **SIG, text_wearer_retention=a.TEXT_WEARER_RETENTION)
    assert receipt["model_execution_performed"] is False
    assert not build(rgb, result)["wearer_support"].any()


@pytest.mark.parametrize("role", ["controller", "attachment", "cable", "object"])
def test_non_wearer_roles_and_empty_point_outputs_not_text_retained(role):
    rgb = frames()
    originals = [proposal(rgb), proposal(rgb, role, rect(rgb, 25, 22, 32, 29), track=8)]
    result, _, _ = refine(rgb, originals)
    role_proposal = next(x for x in result if x.role == role)
    assert role_proposal.source == "SAM31_RGB_POINT_AUTO"
    assert not role_proposal.masks.any()
    assert role_proposal.retained_text_masks is None
    assert [(x.role, x.track_id) for x in result] == [(x.role, x.track_id) for x in originals]


def test_text_only_object_conflict_unknown_neither_protected_nor_dilated_removal():
    rgb = frames()
    text, obj = rect(rgb), rect(rgb, 22, 22, 29, 29)
    result, _, _ = refine(rgb, [proposal(rgb, mask=text), proposal(rgb, "object", obj, track=9)])
    human = next(x for x in result if x.role == "human")
    assert np.array_equal(human.retention_conflict_masks, obj)
    assert not human.retained_text_masks[obj].any()
    built = build(rgb, result)
    assert built["unknown"][obj].all() and built["semantic_conflict"][obj].all()
    assert not built["protected_visible"][obj].any()
    assert not built["effective_removal"][obj].any()
    assert not built["removal"][obj].any()


@pytest.mark.parametrize("role", ["object", "controller"])
def test_union_cannot_change_role_to_nonwearer(role):
    rgb = frames()
    result, _, _ = refine(rgb, [proposal(rgb)])
    with pytest.raises(a.AutomaticMaskContractError, match="ORIGIN_OR_UNION"):
        build(rgb, [replace(result[0], role=role)])


@pytest.mark.parametrize("field", ["retained_text_masks", "point_masks", "retention_conflict_masks"])
def test_origin_missing_or_wrong_domain_rejected(field):
    rgb = frames()
    result, _, _ = refine(rgb, [proposal(rgb)])
    with pytest.raises(a.AutomaticMaskContractError, match="ORIGIN_OR_UNION"):
        build(rgb, [replace(result[0], **{field: None})])


def test_payload_still_binds_actual_point_calls_and_policy():
    _, receipt, _ = refine(frames(), [proposal(frames())])
    body = {k: v for k, v in receipt.items() if k not in {"receipt_payload_sha256", "receipt_signature_scope"}}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() == receipt["receipt_payload_sha256"]
    assert body["text_wearer_retention"] == a.TEXT_WEARER_RETENTION


def test_pure_counterfactual_reuses_fresh_merge_exactly_without_model():
    rgb, originals = frames(), [proposal(frames())]
    default, _, _ = refine(rgb, originals, policy=None)
    fresh, _, _ = refine(rgb, originals)
    replay = a.merge_text_wearer_support(rgb, originals, default, **SIG)
    assert np.array_equal(fresh[0].masks, replay[0].masks)
    for field in ("retained_text_masks", "point_masks", "retention_conflict_masks"):
        assert np.array_equal(getattr(fresh[0], field), getattr(replay[0], field))
    assert replay[0].refinement_receipt_payload_sha256 == default[0].refinement_receipt_payload_sha256


def test_pure_counterfactual_cannot_join_different_tracks():
    rgb, originals = frames(), [proposal(frames())]
    points, _, _ = refine(rgb, originals, policy=None)
    with pytest.raises(a.AutomaticMaskContractError, match="TRACK_IDENTITIES"):
        a.merge_text_wearer_support(rgb, originals, [replace(points[0], track_id=99)], **SIG)


def test_pure_counterfactual_cannot_use_unbound_point_receipt():
    rgb, originals = frames(), [proposal(frames())]
    points, _, _ = refine(rgb, originals, policy=None)
    with pytest.raises(a.AutomaticMaskContractError, match="REPLACEMENT_RECEIPT"):
        a.merge_text_wearer_support(rgb, originals,
            [replace(points[0], refinement_receipt_payload_sha256=None)], **SIG)


def test_stage_seals_policy_actual_parameter_and_provenance_arrays(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-fake")
    raw, weight = tmp_path / "raw.h264", tmp_path / "weight.pt"
    raw.write_bytes(b"fake native raw"); weight.write_bytes(b"not loaded")
    rgb = frames(); text = rect(rgb)
    point = FakePointModel(np.zeros(rgb.shape[:3], bool))
    calls = []
    def track_text(frames, prompt, anchor):
        calls.append((prompt, anchor))
        return {7: {i: np.packbits(text[i]) for i in range(4)}} if prompt == a.PROMPTS["human"] else {}
    sam = SimpleNamespace(model=point, track_text=track_text, aborted=[])
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True, "task_id": "cpu-fake",
        "run_signature": "a" * 64, "fencing_token": "fence", "gpu_uuid": "GPU-cpu-fake", "worker_logical_index": 0}
    receipt = stages.run_automatic_mask_stage(rgb, sam=sam, output_dir=tmp_path / "Mask",
        raw_source_ref=file_ref(raw), frame_map=[{"native_frame_index": i, "decoded_display_index": i} for i in range(4)],
        model_ref=file_ref(weight), component_signature="d" * 64, authorization=lambda: owner,
        text_wearer_retention=a.TEXT_WEARER_RETENTION)
    verify_sealed_stage(receipt, "d" * 64, output_root=tmp_path / "Mask")
    assert receipt["actual_inputs"]["text_wearer_retention"] == a.TEXT_WEARER_RETENTION
    assert calls == [(p, 0) for p in a.PROMPTS.values()]
    assert receipt["manual_points_submitted"] == 0 and receipt["text_prompt_invocations"] == 6
    with np.load(tmp_path / "Mask/MASK_ARRAYS.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["wearer_origin__human__retained_text_masks"], text)
        assert not archive["wearer_origin__human__point_masks"].any()
    point_receipt = json.loads((tmp_path / "Mask/SAM_POINT.json").read_text())
    assert point_receipt["objects"][0]["point_model_supported_native_frames"] == []
    archive_index = json.loads((tmp_path / "Mask/SAM_POINT_PROPOSALS.json").read_text())
    row = archive_index["proposals"][0]
    assert row["source"] == a.TEXT_POINT_WEARER_UNION
    with np.load(tmp_path / "Mask/SAM_POINT_PROPOSALS.npz", allow_pickle=False) as archive:
        retained = np.unpackbits(archive[row["retained_text_masks_array_key"]], axis=1,
            count=48 * 64).reshape(text.shape).astype(bool)
        point = np.unpackbits(archive[row["point_masks_array_key"]], axis=1,
            count=48 * 64).reshape(text.shape).astype(bool)
        assert np.array_equal(retained, text) and not point.any()


def test_invalid_stage_policy_refused_before_source_or_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cpu-fake")
    owner = {"execution_allowed": True, "algorithm_execution_allowed": True, "task_id": "cpu-fake",
        "run_signature": "a" * 64, "fencing_token": "fence", "gpu_uuid": "GPU-cpu-fake", "worker_logical_index": 0}
    with pytest.raises(a.AutomaticMaskContractError, match="FROZEN_POLICY"):
        stages.run_automatic_mask_stage(frames(), sam=object(), output_dir=tmp_path / "Mask",
            raw_source_ref={}, frame_map=[], model_ref={}, component_signature="a" * 64,
            authorization=lambda: owner, text_wearer_retention="UNKNOWN")
    assert not (tmp_path / "Mask").exists()
