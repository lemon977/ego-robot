"""Synthetic CPU providers exercise the unchanged original ProtectedProPainter."""
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chaoyang.exact78_baseline.clean import SETTINGS
from chaoyang.inpainting_candidate import v1_mask_clean_causal as causal


class SyntheticFeature:
    def __init__(self, drift=False):
        self.pixel_calls = 0
        self.forward_calls = 0
        self.drift = drift

    def img_propagation(self, frames, flows, masks, interpolation="nearest"):
        self.pixel_calls += 1
        b, t, c, h, w = frames.shape
        value = 0.2 + 0.05 * self.pixel_calls if self.drift else 0.5
        return (torch.full_like(frames, value).reshape(b * t, c, h, w),
                torch.zeros_like(masks).reshape(b * t, 1, h, w))

    def __call__(self, frames, flows, masks, updated_masks, local):
        self.forward_calls += 1
        return frames[:, :local].clone()


def synthetic_parent(drift=False):
    records = []

    def raft(frames, iters):
        records.append((int(frames.shape[1]), iters, random.random(),
                        np.random.random(), torch.rand(1).item()))
        b, t, _, h, w = frames.shape
        flow = torch.zeros((b, t - 1, 2, h, w), dtype=frames.dtype)
        return flow, flow.clone()

    flow = SimpleNamespace(forward_bidirect_flow=lambda flows, masks: (flows, None),
                           combine_flow=lambda gt, pred, masks: pred)
    return SimpleNamespace(torch=torch, settings=dict(SETTINGS), device=torch.device("cpu"),
        model=SyntheticFeature(drift), raft=raft, flow=flow,
        weights={"synthetic": "NO_WEIGHT_LOADED"}, raft_records=records)


def fixture_arrays(length=4):
    raw = np.arange(length * 24 * 32 * 3, dtype=np.uint32).reshape(length, 24, 32, 3).astype(np.uint8)
    base = np.zeros(raw.shape[:3], bool)
    base[:, 10:13, 17:20] = True
    repaired = base.copy()
    repaired[-2:, 3:5, 3:5] = True
    protected = np.zeros_like(base)
    protected[:, 11, 18] = True
    protected[:, 4, 4] = True
    return raw, base, repaired, protected


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a: pytest.fail("No CUDA RNG in CPU test"))


def test_full128_three_original_calls_fixed_inputs_rng_and_protection():
    parent = synthetic_parent()
    arrays = fixture_arrays(128)
    originals = [array.copy() for array in arrays]
    calls, authorizations = [], []

    def on_call(variant):
        calls.append(variant)
        random.random()
        np.random.random()
        torch.rand(1)

    result = causal.run_mask_clean_causal(parent, *arrays,
        authority=lambda: authorizations.append(True), on_call=on_call)
    receipt = result["receipt"]
    assert calls == ["A", "B", "A_repeat"]
    assert len(authorizations) == 9
    assert receipt["model_calls"] == 3 and receipt["model_loads_by_pair"] == 0
    assert receipt["single_change"] == "REMOVAL_MASK_ONLY_ORIGINAL_PIXEL_PROPAGATION"
    assert receipt["original_image_pixel_propagation_all_calls"] is True
    # Full128 uses two original image propagation chunks on every call.
    assert parent.model.pixel_calls == 6
    records = receipt["calls"]
    assert records[0]["random_state_before"] == records[1]["random_state_before"] == records[2]["random_state_before"]
    assert records[0]["actual_inputs"] == records[2]["actual_inputs"]
    assert records[0]["actual_inputs"]["removal"] != records[1]["actual_inputs"]["removal"]
    assert records[0]["actual_inputs"]["effective_removal"] != records[1]["actual_inputs"]["effective_removal"]
    for key in ("raw_rgb", "protected_visible"):
        assert records[0]["actual_inputs"][key] == records[1]["actual_inputs"][key] == records[2]["actual_inputs"][key]
    one = len(parent.raft_records) // 3
    assert parent.raft_records[:one] == parent.raft_records[one:one * 2] == parent.raft_records[one * 2:]
    assert all(row[1] == SETTINGS["raft_iter"] for row in parent.raft_records)
    assert receipt["A_repeat_reproducibility"]["exact_equal"] is True
    assert receipt["A_repeat_reproducibility"]["max_abs_channel_delta"] == 0
    # The synthetic gain is confined to the changed tail; unchanged earlier
    # frames are neither required to improve nor labelled as visual passes.
    assert receipt["A_B_difference_pixels_per_frame"][:-2] == [0] * 126
    assert all(receipt["A_B_difference_pixels_per_frame"][-2:])
    assert receipt["A_B_difference_outside_effective_union_per_frame"] == [0] * 128
    assert receipt["A_B_protected_difference_pixels_per_frame"] == [0] * 128
    assert receipt["local_improvement_requires_all_frames_improved"] is False
    assert receipt["visual_quality_pass"] is receipt["candidate_adopted"] is False
    raw, _, _, protected = arrays
    for variant in causal.VARIANTS:
        validation = result["validations"][variant]
        assert validation["status"] == "PASS"
        assert validation["independent_protected_changed_pixels_per_frame"] == [0] * 128
        assert validation["independent_outside_effective_changed_pixels_per_frame"] == [0] * 128
        assert np.array_equal(result[variant][protected], raw[protected])
        effective = result["effective_removal"][variant]
        assert np.array_equal(result[variant][~effective], raw[~effective])
    assert all(np.array_equal(first, second) for first, second in zip(originals, arrays))
    assert set(result["rng_arrays"]) == {"python_mt_state", "numpy_mt_state", "torch_cpu_rng"}


def test_repeat_numerical_difference_is_recorded_without_discarding_B():
    result = causal.run_mask_clean_causal(synthetic_parent(drift=True), *fixture_arrays(),
        authority=lambda: None, on_call=lambda variant: None)
    repeat = result["receipt"]["A_repeat_reproducibility"]
    assert repeat["status"] == "NUMERICAL_DIFFERENCE"
    assert repeat["exact_equal"] is False
    assert repeat["max_abs_channel_delta"] > 0
    assert repeat["mean_abs_channel_delta"] > 0
    assert any(repeat["difference_pixels_per_frame"])
    assert result["receipt"]["status"] == "PAIRED_COMPLETE"
    assert result["receipt"]["local_improvement_requires_repeat_exact_match"] is False
    assert "B" in result


@pytest.mark.parametrize("stop", ["B", "A_repeat"])
def test_budget_failure_stops_before_next_original_call(stop):
    parent = synthetic_parent()
    calls = []

    def on_call(variant):
        calls.append(variant)
        if variant == stop:
            raise ValueError("SYNTHETIC_BUDGET_EXHAUSTED")

    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        causal.run_mask_clean_causal(parent, *fixture_arrays(), authority=lambda: None, on_call=on_call)
    assert parent.model.pixel_calls == (1 if stop == "B" else 2)
    assert calls[-1] == stop


def test_authority_failure_prevents_any_model_call():
    parent = synthetic_parent()

    def authority():
        raise ValueError("SYNTHETIC_AUTHORITY_REVOKED")

    with pytest.raises(ValueError, match="AUTHORITY_REVOKED"):
        causal.run_mask_clean_causal(parent, *fixture_arrays(), authority=authority,
            on_call=lambda variant: pytest.fail("No budget reservation without authority"))
    assert parent.model.pixel_calls == 0


@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_budget_callback_input_mutation_is_detected_before_inference(which):
    parent = synthetic_parent()
    arrays = fixture_arrays()

    def on_call(variant):
        value = arrays[which]
        value.flat[0] = int(value.flat[0]) ^ 1

    with pytest.raises(ValueError, match="ACTUAL_INPUT_DRIFT"):
        causal.run_mask_clean_causal(parent, *arrays, authority=lambda: None, on_call=on_call)
    assert parent.model.pixel_calls == 0


def test_model_state_drift_after_A_stops_before_B():
    parent = synthetic_parent()
    count = []

    def authority():
        count.append(True)
        if len(count) == 3:
            parent.weights["synthetic"] = "CHANGED"

    with pytest.raises(ValueError, match="MODEL_STATE_DRIFT"):
        causal.run_mask_clean_causal(parent, *fixture_arrays(), authority=authority,
            on_call=lambda variant: None)
    assert parent.model.pixel_calls == 1


def test_backend_mismatch_before_call_does_no_model_work():
    parent = synthetic_parent()
    backend = deepcopy(causal.runtime_backend(torch, "cpu"))
    backend["flags"]["cudnn_benchmark"] = not backend["flags"]["cudnn_benchmark"]
    with pytest.raises(ValueError, match="BACKEND_DRIFT"):
        causal.run_mask_clean_causal(parent, *fixture_arrays(), authority=lambda: None,
            on_call=lambda variant: pytest.fail("No reservation on backend mismatch"),
            backend_before_load=backend)
    assert parent.model.pixel_calls == 0


def test_output_protection_is_checked_independently_of_painter_claim(monkeypatch):
    from chaoyang.pick_place_1008.visual import ProtectedProPainter

    original = ProtectedProPainter.inpaint

    def broken(self, raw, removal, protected):
        output = original(self, raw, removal, protected)
        output[protected] ^= 1
        return output

    monkeypatch.setattr(ProtectedProPainter, "inpaint", broken)
    with pytest.raises(ValueError, match="ORIGINAL_PIXEL_PROTECTION_FAILED"):
        causal.run_mask_clean_causal(synthetic_parent(), *fixture_arrays(),
            authority=lambda: None, on_call=lambda variant: None)


def test_non_boolean_mask_is_rejected_before_budget():
    arrays = list(fixture_arrays())
    arrays[2] = arrays[2].astype(np.uint8)
    with pytest.raises(ValueError, match="MASK_DOMAIN"):
        causal.run_mask_clean_causal(synthetic_parent(), *arrays, authority=lambda: None,
            on_call=lambda variant: pytest.fail("No reservation on invalid input"))
