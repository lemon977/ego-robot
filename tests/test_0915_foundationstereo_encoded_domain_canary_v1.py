from __future__ import annotations

import inspect

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import run_0915_foundationstereo_encoded_domain_canary_v1 as subject


def test_mirrored_intrinsic_uses_pixel_index_reflection() -> None:
    intrinsic = np.asarray([
        [300.0, 0.0, 317.25],
        [0.0, 301.0, 240.5],
        [0.0, 0.0, 1.0],
    ])
    mirrored = subject.mirrored_intrinsics(intrinsic, 640)
    assert mirrored[0, 2] == 639.0 - 317.25
    assert mirrored[0, 0] == intrinsic[0, 0]
    assert mirrored[1, 2] == intrinsic[1, 2]
    assert subject.coordinate_roundtrip_error(640) == 0.0


def test_unflip_restores_physical_left_disparity_and_depth() -> None:
    physical = np.tile(np.arange(1, 641, dtype=np.float32), (480, 1)) / 10.0
    model = cv2.flip(physical, 1)
    disparity, depth, valid = subject.unflip_disparity_and_depth(
        model, focal_px=300.0, baseline_m=0.064,
    )
    assert np.array_equal(disparity, physical)
    sample = valid & np.isclose(disparity, 20.0)
    assert sample.any()
    assert np.allclose(depth[sample], 300.0 * 0.064 / 20.0)


def test_rgb_roundtrip_is_pixel_exact() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(960, 1280, 3), dtype=np.uint8)
    metrics = subject.rgb_roundtrip_metrics(image)
    assert metrics["maximum_absolute_channel_error"] == 0
    assert metrics["mismatched_pixels"] == 0
    assert metrics["physical_left_depth_rgb_sha256"] == metrics["unflipped_model_left_sha256"]


def test_runner_contains_no_lens_remap_or_camera_swap() -> None:
    source = inspect.getsource(subject.run_worker)
    assert "remap_pair" not in source
    assert "make_map" not in source
    assert "cv2.remap" not in source
    assert "split_resize_physical_eyes" in source
    assert "source_indices=(1, 0)" in source
    assert "cv2.flip(left, 1), cv2.flip(right, 1)" in source


def test_task_packet_binds_one_weight_and_required_adapter_receipts() -> None:
    packet = build_packet(subject.TASK_ID)
    assert packet["weights"] == [subject.MODEL_WEIGHT]
    assert "ADAPTER_CONTRACT.json" in packet["required_outputs"]
    assert "RGB_ALIGNMENT_QA.json" in packet["required_outputs"]
    assert any("OBJECT6D_USER_CONFIRMATION" in item for item in packet["read_set"])
