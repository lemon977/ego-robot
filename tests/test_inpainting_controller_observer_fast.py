"""Fake-only observer checks; no SAM vendor import or actual model call."""
import json

import numpy as np
import pytest

from chaoyang.inpainting_candidate.controller_observer import Sam31ControllerObserver


def nms_masks(pred_probs, pred_masks, prob_threshold, iou_threshold,
              nms_use_iom=False, **kwargs):
    keep = pred_probs > prob_threshold
    keep[..., 1] = False
    return keep


class FakeDetector:
    rank = 0

    def forward_grounding(self, backbone_out=None, find_input=None, **kwargs):
        batch = len(find_input) if find_input is not None else 1
        logits = np.tile(np.asarray([1.4, .1, -.4], np.float32)[None, :, None],
                         (batch, 1, 1))
        masks = np.full((batch, 3, 4, 4), -1.0, np.float32)
        masks[:, :, 1:3, 1:3] = 1
        self.last_raw = {"pred_logits": logits, "pred_masks": masks,
                         "pred_boxes_xyxy": np.zeros((batch, 3, 4), np.float32),
                         "presence_logit_dec": np.ones((batch, 1), np.float32)}
        return self.last_raw

    def _apply_nms(self, out):
        probs = 1 / (1 + np.exp(-out["pred_logits"][..., 0]))
        keep = nms_masks(probs, out["pred_masks"], .1, .8, nms_use_iom=True)
        out["pred_logits"][..., 0] -= 1e4 * (~keep)
        return out

    def _build_multigpu_buffer_next_chunk(self, frame_idx_begin, frame_idx_end,
                                         num_frames, **kwargs):
        return self._apply_nms(self.forward_grounding(find_input=[frame_idx_begin]))

    def _process_grounding_chunk_batched(self, chunk_start, chunk_end, **kwargs):
        return self._apply_nms(self.forward_grounding(find_input=list(range(chunk_start, chunk_end))))


class FakeModel:
    score_threshold_detection = .4
    new_det_thresh = .65
    postprocess_batch_size = 16
    use_batched_grounding = False
    det_nms_use_iom = True
    hotstart_delay = 15

    def __init__(self):
        self.detector = FakeDetector()

    def run_backbone_and_detection(self, frame_idx, num_frames, reverse=False, **kwargs):
        out = self.detector._build_multigpu_buffer_next_chunk(
            frame_idx_begin=frame_idx, frame_idx_end=frame_idx + 1, num_frames=num_frames)
        scores = np.exp(-np.logaddexp(0, -out["pred_logits"][..., 0]))
        return {"scores": scores, "mask": out["pred_masks"],
                "bbox": out["pred_boxes_xyxy"]}, scores > self.score_threshold_detection

    def run_tracker_update_planning_phase(self, frame_idx, det_out, det_keep,
                                         reverse=False, **kwargs):
        return {"new_det_fa_inds": np.asarray([1]), "new_det_obj_ids": np.asarray([7]),
                "det_to_matched_trk_obj_ids": {2: [9]},
                "num_obj_dropped_due_to_limit": 0}, {
            "obj_ids_all_gpu": np.asarray([7]),
            "rank0_metadata": {"removed_obj_ids": set(), "suppressed_obj_ids": {frame_idx: set()}}}

    def _run_single_frame_inference(self, inference_state, frame_idx, reverse=False):
        det, keep = self.run_backbone_and_detection(frame_idx, inference_state["num_frames"], reverse)
        permutation = np.argsort(-keep[0].astype(np.int8), kind="stable")
        self.run_tracker_update_planning_phase(frame_idx,
            {key: value[0][permutation] for key, value in det.items()}, keep[0][permutation], reverse)
        mask = np.zeros((1, 4, 4), bool)
        mask[0, 1, 1] = True
        return {"obj_id_to_mask": {7: mask}, "obj_id_to_score": {7: .8},
                "obj_id_to_sam2_score": {7: .9}, "removed_obj_ids": set(),
                "suppressed_obj_ids": set(), "unconfirmed_obj_ids": []}

    @staticmethod
    def final(out, removed_obj_ids=None, suppressed_obj_ids=None, unconfirmed_obj_ids=None):
        hidden = set(removed_obj_ids or ()) | set(suppressed_obj_ids or ()) | set(unconfirmed_obj_ids or ())
        ids = [key for key in sorted(out["obj_id_to_mask"]) if key not in hidden]
        masks = np.concatenate([out["obj_id_to_mask"][key] for key in ids]) if ids else np.zeros((0, 4, 4), bool)
        return {"out_obj_ids": np.asarray(ids, np.int64), "out_probs": np.asarray([out["obj_id_to_score"][key] for key in ids]),
                "out_binary_masks": masks}

    def _postprocess_output(self, inference_state, out, removed_obj_ids=None,
                            suppressed_obj_ids=None, unconfirmed_obj_ids=None):
        return self.final(out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids)

    def _postprocess_output_batched(self, H_video, W_video, batched_outs):
        return [self.final(*row) for row in batched_outs]


STATE = {"num_frames": 6, "orig_height": 4, "orig_width": 4}


def observer(model, samples=(0, 2, 4)):
    return Sam31ControllerObserver(model, native_frames=6, sample_native_frames=samples,
                                   bindings={"raw_source_sha256": "a" * 64})


def run_frame(model, frame):
    out = model._run_single_frame_inference(STATE, frame, False)
    return model._postprocess_output(STATE, out)


def test_observer_preserves_outputs_and_original_runtime_gates():
    expected = run_frame(FakeModel(), 2)
    model = FakeModel()
    with observer(model) as obs:
        actual = run_frame(model, 2)
        assert model.score_threshold_detection == .4 and model.new_det_thresh == .65
    for key in expected:
        assert np.array_equal(actual[key], expected[key])
    assert obs.summary()["outputs_returned_without_replacement"] is True
    assert obs.summary()["model_calls_added"] == 0
    assert obs.summary()["parameters"]["hotstart_delay"] == 15


def test_wrappers_return_the_exact_original_result_objects():
    model = FakeModel()
    original = model.run_backbone_and_detection
    results = []
    def recorded(*args, **kwargs):
        value = original(*args, **kwargs)
        results.append(value)
        return value
    model.run_backbone_and_detection = recorded
    with observer(model):
        actual = model.run_backbone_and_detection(frame_idx=2, num_frames=6)
    assert actual is results[0]
    assert model.run_backbone_and_detection is recorded


def test_raw_logits_and_masks_survive_nms_and_reused_buffer_mutation():
    model = FakeModel()
    with observer(model) as obs:
        run_frame(model, 2)
        row = next(row for row in obs.records if row["phase"] == "raw_detector")
        assert row["pred_logits"][1] == pytest.approx(.1)
        saved = obs.masks[row["mask"]["array_key"]].copy()
        assert model.detector.last_raw["pred_logits"][0, 1, 0] < -9000
        model.detector.last_raw["pred_masks"][:] = -10
        assert np.array_equal(obs.masks[row["mask"]["array_key"]], saved)
        assert obs.masks[row["mask"]["array_key"]].dtype == np.bool_
        assert row["mask"]["domain"] == "DETECTOR_NATIVE_RESOLUTION"


def test_nms_and_post_score_gates_and_original_query_permutation_are_explicit():
    model = FakeModel()
    with observer(model) as obs:
        run_frame(model, 2)
    nms = next(row for row in obs.records if row["phase"] == "nms")
    detection = next(row for row in obs.records if row["phase"] == "detection")
    planning = next(row for row in obs.records if row["phase"] == "planning")
    assert nms["keep"] == [True, False, True]
    assert nms["prob_threshold"] == .1 and nms["nms_use_iom"] is True
    assert detection["query_permutation"] == [0, 2, 1]
    assert detection["score_gate_keep"] == [[True, False, True]]
    assert planning["new_det_original_query_indices"] == [2]
    assert planning["det_to_matched_original_query_indices"] == {"1": [9]}
    assert {row["stage"] for row in obs.summary()["loss_stages"]} == {"NMS", "POST_SCORE_AND_BOUNDARY"}


def test_masks_only_cover_requested_frames_and_summary_covers_all_native_frames():
    model = FakeModel()
    with observer(model, (2, 4)) as obs:
        for frame in range(6):
            run_frame(model, frame)
    assert all(key.startswith(("f000002_", "f000004_")) for key in obs.masks)
    summary = obs.summary()
    assert summary["observed_native_frames"] == list(range(6))
    assert summary["coverage"]["complete"] is True
    assert summary["coverage"]["missing_native_frames"] == []
    assert summary["map_errors"] == []


def test_delayed_postprocess_maps_returned_out_identity_not_latest_frame():
    model = FakeModel()
    with observer(model) as obs:
        older = model._run_single_frame_inference(STATE, 2)
        newer = model._run_single_frame_inference(STATE, 4)
        model._postprocess_output(STATE, older)
        model._postprocess_output(STATE, newer)
    assert [row["native_frame_index"] for row in obs.records
            if row["phase"] == "object_postprocess"] == [2, 4]


def test_batched_postprocess_maps_each_original_out_and_vendor_hidden_ids():
    model = FakeModel()
    with observer(model) as obs:
        older = model._run_single_frame_inference(STATE, 2)
        newer = model._run_single_frame_inference(STATE, 4)
        results = model._postprocess_output_batched(4, 4,
            [(newer, set(), {7}, []), (older, set(), set(), [])])
    assert not results[0]["out_obj_ids"].size
    rows = [row for row in obs.records if row["phase"] == "object_postprocess"]
    assert [row["native_frame_index"] for row in rows] == [4, 2]
    assert rows[0]["suppressed_obj_ids"] == [7]


def test_raw_batched_chunk_uses_native_start_end_and_single_chunk_uses_rank():
    model = FakeModel()
    with observer(model) as obs:
        model.detector._process_grounding_chunk_batched(chunk_start=2, chunk_end=4)
        model.detector.rank = 1
        model.detector._build_multigpu_buffer_next_chunk(frame_idx_begin=3, frame_idx_end=6, num_frames=6)
    assert [row["native_frame_index"] for row in obs.records if row["phase"] == "raw_detector"] == [2, 3, 4]
    assert obs.summary()["map_errors"] == []


def test_unmapped_raw_and_postprocess_report_errors_without_guessing():
    model = FakeModel()
    with observer(model) as obs:
        raw = model.detector.forward_grounding()
        out = {"obj_id_to_mask": {}, "obj_id_to_score": {}}
        result = model._postprocess_output(STATE, out)
    assert raw["pred_logits"].shape == (1, 3, 1)
    assert result["out_obj_ids"].size == 0
    assert not obs.records
    assert {row["code"] for row in obs.map_errors} == {
        "RAW_BATCH_NATIVE_FRAME_MAPPING_MISSING", "POSTPROCESS_OUT_NATIVE_FRAME_MAPPING_MISSING"}


def test_context_restores_class_methods_and_module_nms_alias_after_exception():
    model = FakeModel()
    before_nms = nms_masks
    before_method = model.run_backbone_and_detection.__func__
    with pytest.raises(RuntimeError, match="synthetic"):
        with observer(model):
            run_frame(model, 2)
            raise RuntimeError("synthetic")
    assert nms_masks is before_nms
    assert model.run_backbone_and_detection.__func__ is before_method
    assert "run_backbone_and_detection" not in model.__dict__
    assert not hasattr(model, "_chaoyang_controller_observer")


def test_json_records_are_finite_serializable_and_native_frames_are_not_renumbered():
    model = FakeModel()
    state = {**STATE, "num_frames": 238}
    with Sam31ControllerObserver(model, native_frames=238,
            sample_native_frames=(0, 103, 104, 119, 132), bindings={"source": "frozen"}) as obs:
        out = model._run_single_frame_inference(state, 103)
        model._postprocess_output(state, out)
    json.dumps(obs.records, allow_nan=False)
    json.dumps(obs.summary(), allow_nan=False)
    assert obs.summary()["observed_native_frames"] == [103]
    assert {row["native_frame_index"] for row in obs.records} == {103}
    assert obs.summary()["coverage"]["complete"] is False


def test_invalid_frame_domain_and_nested_observer_are_rejected():
    model = FakeModel()
    with pytest.raises(ValueError, match="FRAME_COUNT"):
        Sam31ControllerObserver(model, native_frames=0, sample_native_frames=())
    with pytest.raises(ValueError, match="FRAME_DOMAIN"):
        observer(model, (True,))
    with observer(model):
        with pytest.raises(ValueError, match="ALREADY_ACTIVE"):
            with observer(model):
                pass


def test_bfloat16_snapshot_and_json_preserve_original_cpu_tensors():
    torch = pytest.importorskip("torch")
    from chaoyang.inpainting_candidate.controller_observer import _array, _json
    source = torch.tensor([1.4, .1, -.4], dtype=torch.bfloat16)
    before = source.clone()
    array = _array(source)
    assert array.dtype == np.float32
    assert source.dtype == torch.bfloat16 and torch.equal(source, before)
    array[:] = 99
    assert torch.equal(source, before)
    json.dumps(_json({"scores": source}), allow_nan=False)
