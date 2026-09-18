from __future__ import annotations

import numpy as np

from chaoyang.ops.render_0915_hawor_batch_review_v1 import overlay_frame


def test_overlay_marks_only_observed_skeletons() -> None:
    image = np.zeros((120, 160, 3), np.uint8)
    joints = np.full((2, 21, 2), 1000, np.float32)
    joints[0, :, 0] = np.arange(21) + 20
    joints[0, :, 1] = np.arange(21) + 90
    boxes = np.full((2, 4), np.nan, np.float32)
    rendered = overlay_frame(
        image, joints, np.asarray([True, False]), boxes, 0, "session",
    )
    assert rendered.shape == image.shape
    assert np.count_nonzero(rendered[82:]) > 0
