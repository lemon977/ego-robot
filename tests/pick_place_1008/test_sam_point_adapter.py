import copy
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.pick_place_1008 import sam_point_adapter as adapter


class _State(dict):
    def __init__(self, model):
        super().__init__(active=True)
        self.model = model

    def clear(self):
        self.model.events.append("clear")
        self.model.cleared = True
        super().clear()


class FakeModel:
    def __init__(self, *, unsupported=(), abort=False, fail_at=None, bad_output=None,
                 skip_frame=None, torch_outputs=False):
        self.unsupported = set(unsupported)
        self.abort, self.fail_at, self.bad_output = abort, fail_at, bad_output
        self.skip_frame, self.torch_outputs = skip_frame, torch_outputs
        self.events, self.calls, self.ids = [], [], []
        self.cache_calls = []
        self.cleared = False

    def init_state(self, **kwargs):
        self.events.append("init")
        if self.fail_at == "init":
            raise RuntimeError("fixture init error")
        images = kwargs["resource_path"]
        self.count = len(images)
        self.width, self.height = images[0].size
        self.buffer = np.zeros((16, self.height, self.width), bool)
        self.state = _State(self)
        assert kwargs["offload_video_to_cpu"] is True
        assert kwargs["async_loading_frames"] is False
        return self.state

    def _cache_frame_outputs(self, state, frame_idx, obj_id_to_mask, **kwargs):
        assert state is self.state and obj_id_to_mask == {} and not kwargs
        self.cache_calls.append(frame_idx)
        state.setdefault("cached_frame_outputs", {})[frame_idx] = {}

    def _output(self, frame, *, seed_phase=False):
        self.buffer.fill(False)
        for index, obj in enumerate(self.ids):
            if not (seed_phase and obj in self.unsupported):
                self.buffer[index, frame % self.height, obj % self.width] = True
        ids = np.asarray(self.ids, np.int64)
        masks = self.buffer[:len(ids)]
        if self.bad_output == "float_masks":
            masks = masks.astype(float)
        elif self.bad_output == "unknown_id":
            ids = ids.copy()
            ids[0] = 999
        elif self.bad_output == "id_count":
            ids = ids[:-1]
        if self.torch_outputs:
            import torch
            masks = torch.from_numpy(masks)[:, None]
            ids = torch.from_numpy(ids)
        return {"out_binary_masks": masks, "out_obj_ids": ids}

    def add_prompt(self, **kwargs):
        obj = kwargs["obj_id"]
        self.events.append(f"add:{obj}")
        self.calls.append(kwargs)
        assert kwargs["inference_state"] is self.state
        assert kwargs["text_str"] is None
        assert kwargs["clear_old_points"] is True
        assert kwargs["rel_coordinates"] is True
        assert kwargs["points"].device.type == "cpu"
        assert kwargs["point_labels"].device.type == "cpu"
        if self.fail_at == f"add:{obj}":
            raise RuntimeError("fixture add error")
        self.ids.append(obj)
        return kwargs["frame_idx"], self._output(kwargs["frame_idx"], seed_phase=True)

    def propagate_in_video(self, **kwargs):
        reverse = kwargs["reverse"]
        self.events.append("prop:reverse" if reverse else "prop:forward")
        assert kwargs["inference_state"] is self.state
        if self.abort:
            raise RuntimeError(adapter.NO_POINTS_MESSAGE)
        if self.fail_at == "prop":
            raise RuntimeError("fixture propagation error")
        anchor = kwargs["start_frame_idx"]
        maximum = kwargs["max_frame_num_to_track"]
        end = max(anchor - maximum, 0) if reverse else min(anchor + maximum, self.count - 1)
        order = range(anchor - 1, end - 1, -1) if reverse else range(anchor, end + 1)
        assert kwargs["max_frame_num_to_track"] == len(order)
        for frame in order:
            if self.skip_frame == frame:
                continue
            if self.bad_output == "frame":
                yield self.count, self._output(frame)
            else:
                yield frame, self._output(frame)
        # Deliberately overwrite the one shared output buffer after the last yield.
        self.buffer.fill(False)


def _rgb():
    return np.arange(4 * 4 * 8 * 3, dtype=np.uint8).reshape(4, 4, 8, 3)


def _seeds():
    return [
        {"obj_id": 1, "role": "glove", "side": "left",
         "points_px": [[1., 1.], [7., 3.]], "point_labels": [1, 0]},
        {"obj_id": 2, "role": "controller", "side": "right",
         "points_px": [[6., 2.], [0., 3.]], "point_labels": [1, 0]},
    ]


def _run(model, anchor=0, seeds=None):
    return adapter.track_visible_points(SimpleNamespace(model=model), _rgb(),
                                        _seeds() if seeds is None else seeds, anchor)


def _unpack(value):
    return np.unpackbits(value, count=4 * 8).reshape(4, 8).astype(bool)


def test_one_state_adds_all_ids_before_one_pass_and_preserves_shared_output_buffers():
    model = FakeModel()
    rgb, seeds = _rgb(), _seeds()
    original_rgb, original_seeds = rgb.copy(), copy.deepcopy(seeds)
    tracks, metadata = adapter.track_visible_points(SimpleNamespace(model=model), rgb, seeds, 0)
    assert metadata["status"] == "COMPLETE"
    assert model.events == ["init", "add:1", "add:2", "prop:forward", "clear"]
    assert set(tracks) == {1, 2}
    assert np.allclose(model.calls[0]["points"].numpy(), [[1 / 8, 1 / 4], [7 / 8, 3 / 4]])
    assert model.calls[0]["point_labels"].tolist() == [1, 0]
    assert not model.buffer.any()
    for obj in [1, 2]:
        assert set(tracks[obj]) == set(range(4))
        for frame, packed in tracks[obj].items():
            expected = np.zeros((4, 8), bool)
            expected[frame, obj] = True
            assert np.array_equal(_unpack(packed), expected)
    assert metadata["objects"]["1"]["side"] == "left"
    assert metadata["objects"]["2"]["role"] == "controller"
    assert metadata["physical_segmentation_ground_truth"] is False
    assert metadata["hidden_pixel_observation_claim"] is False
    assert metadata["sam_weights_modified"] is False
    assert metadata["control_ground_truth"] is False
    assert np.array_equal(rgb, original_rgb) and seeds == original_seeds


def test_middle_anchor_runs_both_directions_with_exact_native_frame_coverage():
    model = FakeModel()
    tracks, metadata = _run(model, anchor=2)
    assert metadata["status"] == "COMPLETE"
    assert model.events == ["init", "add:1", "add:2", "prop:forward", "prop:reverse", "clear"]
    assert [row["yielded_raw_frame_indices"] for row in metadata["propagation"]] == [[2, 3], [0, 1]]
    assert all(record["track_frame_indices"] == [0, 1, 2, 3] for record in metadata["objects"].values())
    assert all(set(values) == set(range(4)) for values in tracks.values())


def test_cpu_tensor_n_one_h_w_outputs_are_immediately_copied_and_packed():
    model = FakeModel(torch_outputs=True)
    tracks, metadata = _run(model)
    assert metadata["status"] == "COMPLETE"
    assert _unpack(tracks[1][0])[0, 1]
    assert _unpack(tracks[2][3])[3, 2]
    assert not model.buffer.any()


def test_no_anchor_support_cannot_gain_authority_from_later_predicted_masks():
    model = FakeModel(unsupported=[2])
    tracks, metadata = _run(model)
    assert metadata["status"] == "PARTIAL_SUPPORT"
    assert set(tracks) == {1}
    assert metadata["objects"]["2"]["anchor_mask_present"] is False
    assert metadata["objects"]["2"]["status"] == "NO_ANCHOR_MASK_SUPPORT"
    assert metadata["objects"]["2"]["track_frame_indices"] == []
    assert model.cleared


def test_no_supported_points_skips_propagation_and_reports_no_support():
    model = FakeModel(unsupported=[1, 2])
    tracks, metadata = _run(model)
    assert tracks == {} and metadata["status"] == "NO_SUPPORTED_TRACKS"
    assert model.events == ["init", "add:1", "add:2", "clear"]


def test_empty_seeds_does_not_allocate_video_state():
    model = FakeModel()
    tracks, metadata = _run(model, seeds=[])
    assert tracks == {} and metadata["status"] == "NO_VISIBLE_SEEDS"
    assert model.events == []


@pytest.mark.parametrize("failure", ["init", "add:2", "prop"])
def test_model_errors_are_explicit_and_own_state_is_always_cleared(failure):
    model = FakeModel(fail_at=failure)
    tracks, metadata = _run(model)
    assert metadata["status"] == "TRACKING_ERROR"
    assert len(metadata["errors"]) == 1 and metadata["errors"][0]["exception_type"] == "RuntimeError"
    assert model.cleared is (failure != "init")
    if failure == "add:2":
        assert set(tracks) == {1}


def test_known_no_points_abort_is_distinct_from_success_or_unrelated_error():
    model = FakeModel(abort=True)
    _, metadata = _run(model)
    assert metadata["status"] == "PROPAGATION_ABORTED"
    assert metadata["errors"] == []
    assert metadata["aborted"] == [{"direction": "forward", "reason": adapter.NO_POINTS_MESSAGE}]
    assert model.cleared


@pytest.mark.parametrize("bad_output", ["float_masks", "unknown_id", "id_count", "frame"])
def test_wrong_output_type_identity_or_frame_is_rejected_in_metadata(bad_output):
    model = FakeModel(bad_output=bad_output)
    _, metadata = _run(model)
    assert metadata["status"] == "TRACKING_ERROR"
    assert metadata["errors"][0]["exception_type"] == "SamPointContractError"
    assert model.cleared


def test_incomplete_native_generator_is_not_declared_complete():
    model = FakeModel(skip_frame=1)
    _, metadata = _run(model)
    assert metadata["status"] == "TRACKING_ERROR"
    assert metadata["propagation"][0]["missing_raw_frame_indices"] == [1]
    assert metadata["errors"][0]["message"] == "INCOMPLETE_NATIVE_PROPAGATION_FRAME_COVERAGE"
    assert model.cleared


@pytest.mark.parametrize("mutation", [
    {"obj_id": -1}, {"obj_id": True}, {"role": "unknown"}, {"side": "hidden"},
    {"points_px": []}, {"points_px": [[8., 1.], [7., 3.]]},
    {"points_px": [[1., 4.], [7., 3.]]}, {"points_px": [[-1., 1.], [7., 3.]]},
    {"points_px": [[float("nan"), 1.], [7., 3.]]},
    {"point_labels": [2, 0]}, {"point_labels": [0, 0]}, {"point_labels": [1.]},
    {"point_labels": [1., 0.]}, {"frame_idx": 4},
    {"visibility": "OCCLUDED"}, {"raw_observed": False},
])
def test_bad_or_hidden_seed_is_rejected_before_model_state_allocation(mutation):
    seeds, model = _seeds(), FakeModel()
    seeds[0].update(mutation)
    with pytest.raises(adapter.SamPointContractError):
        _run(model, seeds=seeds)
    assert model.events == []


def test_duplicate_ids_or_more_than_sixteen_objects_are_rejected_before_allocation():
    model = FakeModel()
    duplicate = _seeds()
    duplicate[1]["obj_id"] = 1
    with pytest.raises(adapter.SamPointContractError, match="UNIQUE_OBJECT"):
        _run(model, seeds=duplicate)
    many = [{**_seeds()[0], "obj_id": obj} for obj in range(17)]
    with pytest.raises(adapter.SamPointContractError, match="AT_MOST_16"):
        _run(model, seeds=many)
    assert model.events == []


@pytest.mark.parametrize("anchor", [-1, 4, True])
def test_invalid_anchor_is_rejected_before_model_state_allocation(anchor):
    model = FakeModel()
    with pytest.raises(adapter.SamPointContractError, match="ANCHOR"):
        _run(model, anchor=anchor)
    assert model.events == []


def test_wrong_rgb_type_is_rejected_before_model_state_allocation():
    model = FakeModel()
    with pytest.raises(adapter.SamPointContractError, match="RAW_RGB"):
        adapter.track_visible_points(SimpleNamespace(model=model), _rgb().astype(float), _seeds(), 0)
    assert model.events == []


class MissingFrameMergeModel(FakeModel):
    """Faithful pinned-vendor 1244 early-return then 3041/2462 cache write."""
    def _output(self, frame, *, seed_phase=False):
        refined = super()._output(frame, seed_phase=seed_phase)
        cache = self.state.setdefault("cached_frame_outputs", {})
        if frame not in cache:
            cache[frame] = {}
            return {"out_binary_masks": np.zeros((0, self.height, self.width), bool),
                    "out_obj_ids": np.zeros(0, np.int64)}
        cache[frame] = {int(obj): mask.copy()
                        for obj, mask in zip(refined["out_obj_ids"], refined["out_binary_masks"])}
        return refined


def test_empty_frame_bootstrap_fixes_first_id_and_later_real_tracker_merge():
    broken = MissingFrameMergeModel()
    broken.init_state(resource_path=[__import__("PIL.Image", fromlist=["Image"]).fromarray(f)
                                    for f in _rgb()],
                      offload_video_to_cpu=True, async_loading_frames=False)
    broken.ids = [1]
    assert broken._output(0, seed_phase=True)["out_obj_ids"].size == 0
    assert broken._output(0, seed_phase=True)["out_obj_ids"].tolist() == [1]
    assert broken._output(1)["out_obj_ids"].size == 0
    fixed = MissingFrameMergeModel()
    tracks, metadata = _run(fixed)
    assert metadata["status"] == "COMPLETE"
    assert metadata["objects"]["1"]["anchor_mask_present"] is True
    assert fixed.cache_calls == [0, 1, 2, 3]
    assert metadata["cache_bootstrap_native_frame_indices"] == [0, 1, 2, 3]
    assert metadata["cache_bootstrap_policy"] == "EMPTY_CACHE_BOOTSTRAP_NO_MASKS"
    assert metadata["cache_bootstrap_contains_observations"] is False
    for obj in [1, 2]:
        assert set(tracks[obj]) == {0, 1, 2, 3}
        assert all(_unpack(tracks[obj][f])[f, obj] for f in range(4))


def test_empty_cache_bootstrap_preserves_preexisting_nonempty_predictions():
    class ExistingCache(FakeModel):
        def init_state(self, **kwargs):
            state = super().init_state(**kwargs)
            self.original_prediction = {"retained": np.ones((1, 4, 8), bool)}
            state["cached_frame_outputs"] = {1: self.original_prediction}
            return state
        def add_prompt(self, **kwargs):
            assert self.state["cached_frame_outputs"][1] is self.original_prediction
            return super().add_prompt(**kwargs)
    model = ExistingCache()
    _, metadata = _run(model)
    assert metadata["status"] == "COMPLETE"
    assert model.cache_calls == [0, 2, 3]
    assert model.original_prediction["retained"].all()


def test_missing_official_cache_writer_fails_closed_without_point_or_propagation_calls():
    model = FakeModel()
    model._cache_frame_outputs = None
    tracks, metadata = _run(model)
    assert tracks == {} and metadata["status"] == "TRACKING_ERROR"
    assert metadata["errors"][0]["phase"] == "empty_cache_bootstrap"
    assert metadata["errors"][0]["message"] == "OFFICIAL_EMPTY_FRAME_CACHE_WRITER_REQUIRED"
    assert model.events == ["init", "clear"]


def test_per_object_native_anchors_are_honored_and_global_propagation_uses_earliest():
    seeds = _seeds()
    seeds[0].update(frame_idx=3, raw_frame_index=3)
    seeds[1].update(frame_index=1)
    model = MissingFrameMergeModel()
    tracks, metadata = _run(model, anchor=2, seeds=seeds)
    assert metadata["status"] == "COMPLETE"
    assert [call["frame_idx"] for call in model.calls] == [3, 1]
    assert metadata["objects"]["1"]["anchor_raw_frame_index"] == 3
    assert metadata["objects"]["2"]["anchor_raw_frame_index"] == 1
    assert metadata["anchor_raw_frame_index"] == 1
    assert metadata["caller_default_anchor_raw_frame_index"] == 2
    assert [row["start_raw_frame_index"] for row in metadata["propagation"]] == [1, 1]
    assert [row["yielded_raw_frame_indices"] for row in metadata["propagation"]] == [[1, 2, 3], [0]]
    assert all(set(values) == {0, 1, 2, 3} for values in tracks.values())


@pytest.mark.parametrize("fields", [{"frame_idx":1,"frame_index":2},
                                   {"raw_frame_index":True},
                                   {"frame_idx":-1},
                                   {"frame_index":4}])
def test_conflicting_or_non_native_per_object_anchor_fields_fail_before_allocation(fields):
    seeds, model = _seeds(), FakeModel()
    seeds[0].update(fields)
    with pytest.raises(adapter.SamPointContractError, match="MUST_AGREE_AND_BE_BOUNDED"):
        _run(model, seeds=seeds)
    assert model.events == []


def test_support_is_decided_on_each_own_native_seed_anchor():
    seeds = _seeds()
    seeds[0]["frame_idx"] = 3
    seeds[1]["frame_idx"] = 1
    model = MissingFrameMergeModel(unsupported=[2])
    tracks, metadata = _run(model, seeds=seeds)
    assert metadata["status"] == "PARTIAL_SUPPORT"
    assert set(tracks) == {1}
    assert metadata["objects"]["1"]["anchor_mask_present"] is True
    assert metadata["objects"]["2"]["anchor_mask_present"] is False
    assert metadata["objects"]["2"]["track_frame_indices"] == []

@pytest.mark.parametrize("anchor", [0, 1, 2, 3])
def test_pinned_multiplex_reverse_excludes_anchor_and_native_union_is_complete(anchor):
    model = FakeModel()
    tracks, meta = _run(model, anchor=anchor)
    assert meta["status"] == "COMPLETE"
    assert meta["propagation"][0]["expected_raw_frame_indices"] == list(range(anchor, 4))
    if anchor:
        reverse = meta["propagation"][1]
        assert reverse["max_frame_num_to_track"] == anchor
        assert reverse["expected_raw_frame_indices"] == list(range(anchor))
        assert reverse["yielded_raw_frame_indices"] == list(range(anchor))
    else:
        assert len(meta["propagation"]) == 1
    assert set().union(*(set(p["yielded_raw_frame_indices"]) for p in meta["propagation"])) == set(range(4))
    assert all(set(values) == set(range(4)) for values in tracks.values())

def test_pinned_single_native_frame_has_no_reverse_pass():
    model = FakeModel()
    tracks, meta = adapter.track_visible_points(
        SimpleNamespace(model=model), np.zeros((1, 4, 8, 3), np.uint8), _seeds(), 0)
    assert meta["status"] == "COMPLETE"
    assert len(meta["propagation"]) == 1
    assert all(set(values) == {0} for values in tracks.values())

@pytest.mark.parametrize("missing", [0, 1, 2])
def test_pinned_reverse_boundary_fix_does_not_admit_missing_native_frames(missing):
    model = FakeModel(skip_frame=missing)
    _, meta = _run(model, anchor=2)
    assert meta["status"] == "TRACKING_ERROR"
    assert meta["errors"][0]["message"] == "INCOMPLETE_NATIVE_PROPAGATION_FRAME_COVERAGE"
    failed = next(p for p in meta["propagation"] if p["missing_raw_frame_indices"])
    assert failed["direction"] == ("forward" if missing == 2 else "reverse")
    assert failed["missing_raw_frame_indices"] == [missing]
    assert model.cleared

def test_pinned_reverse_boundary_does_not_admit_extra_anchor_output():
    class ExtraReverseAnchor(FakeModel):
        def propagate_in_video(self, **kwargs):
            yield from super().propagate_in_video(**kwargs)
            if kwargs["reverse"]:
                anchor = kwargs["start_frame_idx"]
                yield anchor, self._output(anchor)
    model = ExtraReverseAnchor()
    _, meta = _run(model, anchor=2)
    assert meta["status"] == "TRACKING_ERROR"
    assert meta["errors"][0]["message"] == "INCOMPLETE_NATIVE_PROPAGATION_FRAME_COVERAGE"
    assert meta["propagation"][1]["missing_raw_frame_indices"] == []
    assert model.cleared
