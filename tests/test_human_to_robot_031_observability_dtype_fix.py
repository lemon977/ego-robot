"""The label-1 uint16 mask must not silently vanish through grayscale conversion."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from chaoyang.ops.run_human_to_robot_031_observability_dtype_fix import read_region


def test_uint16_label_one_is_visible(tmp_path):
    path = tmp_path / "mask.png"
    source = np.zeros((960, 1280), np.uint16)
    source[800:810, 700:710] = 1
    assert cv2.imwrite(str(path), source)
    region = read_region(path)
    assert int(region.sum()) == 100
    assert not (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0).any()


def test_wrong_dtype_or_label_rejected(tmp_path):
    path = tmp_path / "mask.png"
    assert cv2.imwrite(str(path), np.zeros((960, 1280), np.uint8))
    with pytest.raises(RuntimeError, match="MASK_DOMAIN_OR_DTYPE_INVALID"):
        read_region(path)
    source = np.zeros((960, 1280), np.uint16)
    source[10, 10] = 2
    assert cv2.imwrite(str(path), source)
    with pytest.raises(RuntimeError, match="MASK_LABEL_UNEXPECTED"):
        read_region(path)
