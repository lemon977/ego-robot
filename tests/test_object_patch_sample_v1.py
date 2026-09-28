from __future__ import annotations
import numpy as np
from chaoyang.pipeline.object_patch_sample_v1 import sample_object_patch


def test_invalid_center_cannot_be_replaced_by_neighbor_median() -> None:
    mask = np.ones((10, 10), dtype=np.uint8)
    depth = np.ones((5, 5), dtype=np.float32)
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    result = sample_object_patch(mask, depth, valid, np.eye(3), (5, 5))
    assert result["reason"] == "INVALID_CENTER_DEPTH"


def test_same_instance_and_center_depth_are_enforced() -> None:
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[:, 6:] = 2
    depth = np.ones((5, 5), dtype=np.float32)
    depth[2, 2] = 1.01
    result = sample_object_patch(mask, depth, np.ones((5, 5), bool), np.eye(3), (5, 5))
    assert result["valid"] is True
    assert result["object_id"] == 1
    assert result["center_depth_m"] == float(depth[2, 2])
    assert result["neighborhood_used_for_depth"] is False
