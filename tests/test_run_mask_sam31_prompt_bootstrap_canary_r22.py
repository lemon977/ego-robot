from __future__ import annotations

import pytest
import torch

from chaoyang.ops.run_mask_sam31_prompt_bootstrap_canary_r22 import tracker_point_tensors


def test_tracker_point_tensor_contract() -> None:
    points, labels = tracker_point_tensors([640, 480], 1280, 960)
    assert points.shape == (1, 2)
    assert labels.shape == (1,)
    assert points.dtype == torch.float32
    assert labels.dtype == torch.int32
    assert torch.allclose(points, torch.tensor([[0.5, 0.5]], dtype=torch.float32))
    assert labels.tolist() == [1]


@pytest.mark.parametrize("point", [[-1, 0], [0, -1], [1280, 10], [10, 960]])
def test_tracker_point_tensor_rejects_out_of_bounds(point: list[int]) -> None:
    with pytest.raises(RuntimeError, match="outside image"):
        tracker_point_tensors(point, 1280, 960)
