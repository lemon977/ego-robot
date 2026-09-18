from __future__ import annotations

import numpy as np

from chaoyang.ops import run_0915_hawor_resize_only_canary_v1 as subject


def test_scaled_intrinsics_uses_physical_left_without_remap() -> None:
    camera = {
        "width": 2048, "height": 1536,
        "left": {
            "sourceIndex": 1,
            "intrinsics": {"fx": 960.0, "fy": 944.0, "cx": 1024.0, "cy": 768.0},
        },
    }
    k = subject.scaled_intrinsics(camera)
    assert np.allclose(
        k, [[600.0, 0.0, 640.0], [0.0, 590.0, 480.0], [0.0, 0.0, 1.0]],
    )


def test_physical_left_c2w_changes_eye_without_touching_pixels() -> None:
    selected = np.eye(4)
    right = np.eye(4)
    right[0, 3] = 0.03
    left = np.eye(4)
    left[0, 3] = -0.03
    camera = {
        "extrinsic_convention": "head_to_camera_4x4_row_major",
        "extrinsics": {"left": left.tolist(), "right": right.tolist()},
    }
    result = subject.physical_left_c2w(selected, camera)
    assert np.isclose(result[0, 3], 0.06)


def test_canary_contract_has_no_sam_or_depth_execution() -> None:
    assert subject.TASK_ID == "0915_hawor_resize_only_canary_v1"
    assert subject.SESSION_ID == "play_cards_0915_001"
    assert subject.TARGET_SIZE == (1280, 960)
