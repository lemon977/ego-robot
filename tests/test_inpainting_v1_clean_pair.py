"""CPU tensors and synthetic providers only; no weights, CUDA or governance."""
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chaoyang.exact78_baseline.clean import SETTINGS
from chaoyang.inpainting_candidate import v1_clean_pair as pair


class SyntheticFeature:
    def __init__(self):
        self.pixel_calls = 0
        self.forward_shapes = []

    def img_propagation(self, frames, flows, masks, interpolation="nearest"):
        self.pixel_calls += 1
        b, t, c, h, w = frames.shape
        return (torch.ones_like(frames).reshape(b * t, c, h, w),
                torch.zeros_like(masks).reshape(b * t, 1, h, w))

    def __call__(self, frames, flows, masks, updated_masks, local):
        self.forward_shapes.append((tuple(frames.shape), local))
        return frames[:, :local].clone()


def synthetic_parent():
    records = []
    def raft(frames, iters):
        records.append((int(frames.shape[1]), iters, random.random(), np.random.random(),
                        torch.rand(1).item()))
        b, t, _, h, w = frames.shape
        flow = torch.zeros((b, t - 1, 2, h, w), dtype=frames.dtype)
        return flow, flow.clone()
    flow = SimpleNamespace(forward_bidirect_flow=lambda flows, masks: (flows, None),
                           combine_flow=lambda gt, pred, masks: pred)
    return SimpleNamespace(torch=torch, settings=dict(SETTINGS), device=torch.device("cpu"),
        model=SyntheticFeature(), raft=raft, flow=flow, weights={"synthetic": "NO_WEIGHT_LOADED"},
        raft_records=records)


def fixture_arrays(length=128):
    raw = np.arange(length * 24 * 32 * 3, dtype=np.uint32).reshape(length, 24, 32, 3).astype(np.uint8)
    removal = np.zeros(raw.shape[:3], bool)
    removal[:, 10:13, 17:20] = True
    protected = np.zeros_like(removal)
    protected[:, 11, 18] = True
    return raw, removal, protected


def test_full128_fresh_pair_shared_models_reset_rng_and_zero_protection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a: pytest.fail("CPU test must not touch CUDA RNG"))
    parent = synthetic_parent()
    arrays = fixture_arrays()
    copies = [value.copy() for value in arrays]
    calls = []
    # Even a budget callback that consumes randomness must finish before reset.
    def on_call(variant):
        calls.append(variant)
        random.random()
        np.random.random()
        torch.rand(1)
    result = pair.run_fresh_clean_pair(parent, *arrays, authority=lambda: None, on_call=on_call)
    receipt = result["receipt"]
    assert calls == ["A", "B"] and receipt["model_calls"] == 2
    assert receipt["fresh_A"] is receipt["fresh_B"] is True
    assert receipt["completed_frame_indices"] == list(range(128))
    assert receipt["calls"][0]["random_state_before"] == receipt["calls"][1]["random_state_before"]
    assert receipt["calls"][0]["actual_inputs"] == receipt["calls"][1]["actual_inputs"]
    split = len(parent.raft_records) // 2
    assert parent.raft_records[:split] == parent.raft_records[split:]
    assert all(row[1] == SETTINGS["raft_iter"] for row in parent.raft_records)
    assert parent.model.pixel_calls == 2
    assert receipt["B_ablation"]["img_propagation_calls"] == 2
    assert receipt["B_ablation"]["forward_calls"] == len(parent.model.forward_shapes) // 2
    half = len(parent.model.forward_shapes) // 2
    assert parent.model.forward_shapes[:half] == parent.model.forward_shapes[half:]
    assert any(receipt["A_B_difference_pixels_per_frame"])
    raw, _, protected = arrays
    for variant in ("A", "B"):
        validation = result["validations"][variant]
        assert validation["status"] == "PASS"
        assert validation["protected_changed_pixels_per_frame"] == [0] * 128
        assert validation["outside_delete_changed_pixels_per_frame"] == [0] * 128
        assert np.array_equal(result[variant][protected], raw[protected])
        assert np.array_equal(result[variant][~result["effective_removal"]], raw[~result["effective_removal"]])
    assert all(np.array_equal(before, after) for before, after in zip(copies, arrays))
    assert receipt["historical_backend_inferred"] is False
    assert receipt["visual_quality_pass"] is receipt["training_eligible"] is False
    assert set(result["rng_arrays"]) == {"python_mt_state", "numpy_mt_state", "torch_cpu_rng"}


def test_random_state_roundtrip_preserves_actual_cpu_generators():
    state = pair.RandomState(torch, "cpu")
    signature = state.metadata()
    draws = (random.random(), np.random.random(), torch.rand(1).item())
    state.restore()
    assert pair.RandomState(torch, "cpu").metadata() == signature
    assert (random.random(), np.random.random(), torch.rand(1).item()) == draws


def test_backend_mismatch_before_call_stops_with_no_model_work():
    parent = synthetic_parent()
    backend = pair.runtime_backend(torch, "cpu")
    backend["flags"]["cudnn_benchmark"] = not backend["flags"]["cudnn_benchmark"]
    with pytest.raises(ValueError, match="BACKEND_DRIFT"):
        pair.run_fresh_clean_pair(parent, *fixture_arrays(4), authority=lambda: None,
            on_call=lambda variant: pytest.fail("must stop before call"), backend_before_load=backend)
    assert parent.raft_records == [] and parent.model.pixel_calls == 0


def test_backend_drift_after_A_stops_before_B(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    parent = synthetic_parent()
    original_backend = pair.runtime_backend
    reads = []
    def changed_backend(*args):
        record = deepcopy(original_backend(*args))
        reads.append(record)
        if len(reads) >= 3:
            record["flags"]["cudnn_benchmark"] = not record["flags"]["cudnn_benchmark"]
        return record
    monkeypatch.setattr(pair, "runtime_backend", changed_backend)
    calls = []
    with pytest.raises(ValueError, match="BACKEND_DRIFT"):
        pair.run_fresh_clean_pair(parent, *fixture_arrays(4), authority=lambda: None, on_call=calls.append)
    assert calls == ["A"]


def test_input_mutation_after_A_stops_before_B(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    parent = synthetic_parent()
    arrays = fixture_arrays(4)
    counter = []
    def authority():
        counter.append(None)
        if len(counter) == 2:
            arrays[0][0, 0, 0, 0] ^= 1
    calls = []
    with pytest.raises(ValueError, match="ACTUAL_INPUT_DRIFT"):
        pair.run_fresh_clean_pair(parent, *arrays, authority=authority, on_call=calls.append)
    assert calls == ["A"]


def test_budget_failure_stops_B_without_reusing_old_A(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    parent = synthetic_parent()
    def on_call(variant):
        if variant == "B":
            raise ValueError("SYNTHETIC_BUDGET_EXHAUSTED")
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        pair.run_fresh_clean_pair(parent, *fixture_arrays(4), authority=lambda: None, on_call=on_call)
    assert parent.model.pixel_calls == 1


@pytest.mark.parametrize("columns_count", [2, 3])
def test_comparison_title_outside_image_preserves_all_pixels(columns_count):
    columns = [np.full((4, 24, 32, 3), 41 + i, np.uint8) for i in range(columns_count)]
    def label(header, title):
        header[:] = 254
        return header
    out = pair.comparison_outside_titles(columns, ["标题"] * columns_count, [0, 1, 2, 3], "050", label)
    assert out.shape == (4, 56, 32 * columns_count, 3)
    assert np.all(out[:, :32] == 254)
    for i, pixels in enumerate(columns):
        assert np.array_equal(out[:, 32:, i * 32:(i + 1) * 32], pixels)


def test_array_signature_covers_dtype_shape_and_pixels():
    base = np.arange(8, dtype=np.uint8)
    assert pair.array_signature(base) != pair.array_signature(base.reshape(2, 4))
    assert pair.array_signature(base) != pair.array_signature(base.astype(np.uint16))
    changed = base.copy()
    changed[-1] ^= 1
    assert pair.array_signature(base) != pair.array_signature(changed)
    with pytest.raises(ValueError, match="OBJECT_ARRAY"):
        pair.array_signature(np.array([object()], dtype=object))
