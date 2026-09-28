import cv2
import numpy as np

from chaoyang.pipeline.attachment_tracker_v1 import (
    AttachmentTrackGateV1,
    admit_bidirectional_candidate,
    propagate_one_frame,
    read_binary_mask,
    seed_brackets,
)


def test_uint16_one_foreground_is_not_lost(tmp_path):
    path = tmp_path / "mask.png"
    value = np.zeros((8, 8), dtype=np.uint16)
    value[3:5, 2:6] = 1
    assert cv2.imwrite(str(path), value)
    loaded = read_binary_mask(str(path))
    assert loaded.dtype == np.bool_
    assert int(loaded.sum()) == 8


def test_only_short_seed_brackets_are_eligible():
    gate = AttachmentTrackGateV1(maximum_seed_gap=6)
    assert seed_brackets([1, 2, 8, 20], gate=gate) == [(2, 8)]


def test_bidirectional_disagreement_stays_unknown():
    left = np.zeros((12, 12), dtype=bool); left[2:5, 2:5] = True
    right = np.zeros((12, 12), dtype=bool); right[8:11, 8:11] = True
    passed, metrics = admit_bidirectional_candidate(left, right, left_seed_area=9, right_seed_area=9)
    assert not passed
    assert metrics["bidirectional_iou"] == 0.0


def test_adjacent_translation_warp_tracks_mask():
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    target = np.zeros_like(source)
    source[20:35, 18:33] = 255
    target[20:35, 21:36] = 255
    mask = np.zeros((64, 64), dtype=bool); mask[20:35, 18:33] = True
    moved = propagate_one_frame(source, target, mask)
    expected = np.zeros_like(mask); expected[20:35, 21:36] = True
    intersection = np.count_nonzero(moved & expected)
    union = np.count_nonzero(moved | expected)
    assert intersection / union > 0.80
