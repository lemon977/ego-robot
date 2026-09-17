import cv2
import numpy as np

from chaoyang.ops.audit_poker245_donor_observability import track_order


CONFIG = {
    "hue_half_width": 7, "sat_min": 65, "value_min": 70,
    "value_max": 205, "table_roi_y_min": 350,
    "min_component_px": 20, "min_order_gap_px": 60,
    "max_center_step_px": 30, "min_unoccluded_area_px": 900,
    "min_rightmost_visible_px": 300,
}


def fixture_frames() -> tuple[list[np.ndarray], np.ndarray]:
    frames = []
    for _ in range(16):
        image = np.zeros((960, 1280, 3), np.uint8)
        for x, width, height in ((660, 40, 40), (790, 40, 40), (930, 30, 25)):
            cv2.rectangle(image, (x, 450), (x + width, 450 + height), (128, 0, 128), -1)
        frames.append(image)
    mask = np.zeros(frames[12].shape[:2], bool)
    mask[450:476, 930:961] = True
    return frames, mask


def test_observability_uses_only_prefix():
    frames, mask = fixture_frames()
    original = track_order(frames, mask, CONFIG, 13)[1]
    assert original["all_prefix_cue_pass"]
    frames[14][:] = (0, 255, 0)
    frames[15][:] = (0, 255, 0)
    assert track_order(frames, mask, CONFIG, 13)[1] == original


def test_rightmost_back_disappearance_rejected():
    frames, mask = fixture_frames()
    frames[13][450:476, 930:961] = 0
    ledger, summary = track_order(frames, mask, CONFIG, 13)
    assert not summary["all_prefix_cue_pass"]
    assert not ledger[-1]["three_visible_back_components"]
