"""CPU synthetic tests only: no loaded weights or neural model execution."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.inpainting_candidate.v1_clean_ablation import (
    PixelPropagationBypass, make_pixel_propagation_ablation,
)
from chaoyang.pick_place_1008.visual import ProtectedProPainter


class FakeModel:
    marker = "frozen_feature_forward"

    def __init__(self):
        self.pixel_calls = 0
        self.forward_records = []

    def img_propagation(self, frames, flows, masks, interpolation="nearest"):
        self.pixel_calls += 1
        b, t, c, h, w = frames.shape
        return (torch.ones_like(frames).reshape(b * t, c, h, w),
                torch.zeros_like(masks).reshape(b * t, 1, h, w))

    def __call__(self, frames, flows, masks, updated_masks, local, **kwargs):
        self.forward_records.append((tuple(frames.shape), local, kwargs,
                                     torch.equal(masks, updated_masks)))
        return frames[:, :local].clone()


def parent():
    flow_calls = []
    raft_calls = []

    def raft(frames, iters):
        raft_calls.append((int(frames.shape[1]), iters))
        b, t, _, h, w = frames.shape
        flow = torch.zeros((b, t - 1, 2, h, w), dtype=frames.dtype)
        return flow, flow.clone()

    def forward_flow(flows, masks):
        flow_calls.append(("forward", int(masks.shape[1])))
        return flows, None

    def combine_flow(gt, pred, masks):
        flow_calls.append(("combine", int(masks.shape[1])))
        return pred

    return SimpleNamespace(torch=torch, settings=dict(SETTINGS), device=torch.device("cpu"),
        model=FakeModel(), raft=raft,
        flow=SimpleNamespace(forward_bidirect_flow=forward_flow, combine_flow=combine_flow),
        weights={"frozen": "no-loaded-model"}, raft_calls=raft_calls, flow_calls=flow_calls)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch,length", [(1, 2), (2, 5)])
def test_bypass_returns_flattened_original_tensors_without_writes(dtype, batch, length):
    original = FakeModel()
    proxy = PixelPropagationBypass(original)
    frames = torch.arange(batch * length * 3 * 8 * 16).reshape(batch, length, 3, 8, 16).to(dtype)
    mask = torch.ones((batch, length, 1, 8, 16), dtype=dtype)
    before_frames, before_mask = frames.clone(), mask.clone()
    out, holes = proxy.img_propagation(frames, (object(), object()), mask)
    assert out.shape == (batch * length, 3, 8, 16)
    assert holes.shape == (batch * length, 1, 8, 16)
    assert out.dtype == holes.dtype == dtype
    assert torch.equal(out.reshape_as(frames), before_frames)
    assert torch.equal(holes.reshape_as(mask), before_mask)
    assert torch.equal(frames, before_frames) and torch.equal(mask, before_mask)
    assert original.pixel_calls == 0 and proxy.img_propagation_calls == 1
    assert proxy.metadata()["updated_mask_source"] == "ORIGINAL_UNFILLED_MASK"


def test_noncontiguous_inputs_keep_values():
    proxy = PixelPropagationBypass(FakeModel())
    frames = torch.arange(1 * 6 * 3 * 8 * 8).float().reshape(1, 6, 3, 8, 8)[:, ::2]
    mask = torch.ones((1, 6, 1, 8, 8))[:, ::2]
    result, holes = proxy.img_propagation(frames, None, mask)
    assert torch.equal(result.reshape_as(frames), frames)
    assert torch.equal(holes.reshape_as(mask), mask)


def test_parent_isolated_settings_copied_and_weights_untouched():
    frozen = parent()
    state = dict(frozen.__dict__)
    candidate, metadata = make_pixel_propagation_ablation(frozen)
    assert candidate is not frozen
    assert frozen.__dict__ == state
    assert frozen.model is state["model"] and candidate.model.wrapped_model is frozen.model
    assert candidate.settings == frozen.settings and candidate.settings is not frozen.settings
    assert candidate.raft is frozen.raft and candidate.flow is frozen.flow
    assert candidate.weights is frozen.weights
    assert candidate.model.marker == frozen.model.marker
    candidate.settings["raft_iter"] = -1
    assert frozen.settings == SETTINGS
    assert metadata["visual_quality_pass"] is False
    assert metadata["training_eligible"] is False


def test_forward_positional_and_keyword_delegation():
    frozen = FakeModel()
    proxy = PixelPropagationBypass(frozen)
    frames = torch.zeros((1, 3, 3, 8, 8))
    masks = torch.zeros((1, 3, 1, 8, 8))
    result = proxy(frames, None, masks, masks, 2, interpolation="bilinear")
    assert result.shape[1] == 2
    assert frozen.forward_records[-1][2] == {"interpolation": "bilinear"}
    assert proxy.forward_calls == 1 and frozen.pixel_calls == 0


def test_factory_does_not_switch_original_parent_inference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    frozen = parent()
    candidate, _ = make_pixel_propagation_ablation(frozen)
    raw = np.full((4, 24, 32, 3), 80, np.uint8)
    removal = np.zeros((4, 24, 32), bool)
    removal[:, 10:13, 17:20] = True
    protected = np.zeros_like(removal)
    ProtectedProPainter(frozen).inpaint(raw, removal, protected)
    assert frozen.model.pixel_calls == 1
    assert candidate.model.img_propagation_calls == 0
    assert any(not record[-1] for record in frozen.model.forward_records)


@pytest.mark.parametrize("length", [4, 81, 128])
def test_unchanged_protected_caller_short_and_chunked_paths(monkeypatch, length):
    # The original function calls empty_cache; replace that no-op for CPU-only
    # synthetic testing, rather than touching/initializing CUDA.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    frozen = parent()
    candidate, _ = make_pixel_propagation_ablation(frozen)
    raw = np.arange(length * 24 * 32 * 3, dtype=np.uint32).reshape(length, 24, 32, 3).astype(np.uint8)
    removal = np.zeros((length, 24, 32), bool)
    removal[:, 10:13, 17:20] = True
    protected = np.zeros_like(removal)
    protected[:, 11, 18] = True
    originals = (raw.copy(), removal.copy(), protected.copy())
    painter = ProtectedProPainter(candidate)
    result = painter.inpaint(raw, removal, protected)
    assert result.shape == raw.shape and result.dtype == np.uint8
    assert painter.last_validation["status"] == "PASS"
    assert not any(painter.last_validation["protected_changed_pixels_per_frame"])
    assert not any(painter.last_validation["outside_delete_changed_pixels_per_frame"])
    assert all(n > 0 for n in painter.last_validation["output_frame_visits"])
    assert np.array_equal(result[protected], raw[protected])
    assert np.array_equal(result[~painter.effective_masks], raw[~painter.effective_masks])
    assert all(np.array_equal(a, b) for a, b in zip(originals, (raw, removal, protected)))
    assert frozen.model.pixel_calls == 0
    assert candidate.model.img_propagation_calls == (1 if length <= 80 else 2)
    assert frozen.raft_calls and all(iters == SETTINGS["raft_iter"] for _, iters in frozen.raft_calls)
    assert frozen.flow_calls
    expected = []
    stride = SETTINGS["neighbor_length"] // 2
    for mid in range(0, length, stride):
        neighbors = list(range(max(0, mid - stride), min(length, mid + stride + 1)))
        ref_num = SETTINGS["subvideo_length"] // SETTINGS["ref_stride"] if length > 80 else -1
        refs = _get_ref_index(mid, neighbors, length, SETTINGS["ref_stride"], ref_num)
        expected.append(((1, len(neighbors + refs), 3, 24, 32), len(neighbors), {}, True))
    assert frozen.model.forward_records == expected
    assert candidate.model.forward_calls == len(expected)
    assert frozen.model is not candidate.model and frozen.settings == SETTINGS


@pytest.mark.parametrize("variant", ["rank", "channel", "time", "spatial", "dtype", "interpolation"])
def test_invalid_tensor_domains_fail_before_delegated_model(variant):
    frozen = FakeModel()
    proxy = PixelPropagationBypass(frozen)
    frames = torch.zeros((1, 2, 3, 8, 8))
    mask = torch.ones((1, 2, 1, 8, 8))
    interpolation = "nearest"
    if variant == "rank": frames = frames[0]
    if variant == "channel": frames = frames[:, :, :1]
    if variant == "time": mask = mask[:, :1]
    if variant == "spatial": mask = mask[..., :4]
    if variant == "dtype": mask = mask.half()
    if variant == "interpolation": interpolation = "bilinear"
    with pytest.raises(ValueError, match="V1_CLEAN_"):
        proxy.img_propagation(frames, None, mask, interpolation)
    assert proxy.img_propagation_calls == 0 and frozen.pixel_calls == 0


def test_frozen_settings_and_model_interface_rejected():
    frozen = parent()
    frozen.settings["ref_stride"] = 1
    with pytest.raises(ValueError, match="SETTINGS_DRIFT"):
        make_pixel_propagation_ablation(frozen)
    with pytest.raises(ValueError, match="INTERFACE_REQUIRED"):
        PixelPropagationBypass(object())
