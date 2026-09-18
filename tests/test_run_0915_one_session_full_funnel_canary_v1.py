from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops import run_0915_one_session_full_funnel_canary_v1 as canary
from chaoyang.ops import run_0915_sam31_persistent_masks_v2 as sam_worker


def test_canary_identity_and_stage_outputs_are_bounded(tmp_path: Path) -> None:
    assert canary.TASK == "playing_cards"
    assert canary.SESSION_ID == "play_cards_0915_001"

    resolved = canary.paths(tmp_path)
    for key in ("sam_session", "depth_session", "post_session"):
        assert resolved[key].is_relative_to(tmp_path)
    assert not resolved["video"].is_relative_to(tmp_path)
    assert not resolved["hawor"].is_relative_to(tmp_path)


def test_canary_source_keeps_gpu_weights_in_separate_stages() -> None:
    source = inspect.getsource(canary)
    for stage in ("preflight", "sam31", "sam_visualize", "depth", "post", "visualize"):
        assert f'"{stage}"' in source
    assert "SAM3.1_ONLY_USER_LOCKED" in source
    assert "FOUNDATIONSTEREO_PINNED_BASELINE" in source
    assert "SAM2" not in source
    assert "Cutie" not in source
    assert '"tracker": "ABSENT_NO_ROLE_CREATED"' in source
    assert '"pico26": "PRESENT_PRESERVED_NOT_CONSUMED"' in source


def test_draw_hawor_renders_only_observed_side() -> None:
    frame = np.zeros((64, 96, 3), np.uint8)
    joints = np.full((2, 1, 21, 2), np.nan, np.float64)
    joints[0, 0, :, 0] = np.linspace(10, 80, 21)
    joints[0, 0, :, 1] = np.linspace(10, 50, 21)
    joints[1, 0] = 5.0
    observed = np.asarray([[True], [False]])

    rendered = canary._draw_hawor(frame, joints, observed, 0)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
    assert not np.any(np.all(rendered == np.asarray((30, 70, 255)), axis=2))


def test_depth_panel_marks_invalid_pixels_black(tmp_path: Path) -> None:
    depth = np.asarray([[0.4, np.nan], [0.8, 1.2]], np.float32)
    valid = np.asarray([[True, False], [True, True]])
    artifact = tmp_path / "depth.npz"
    np.savez_compressed(artifact, depth_m=depth, valid=valid)

    panel = canary._depth_panel(artifact)

    assert panel.shape == (960, 1280, 3)
    # The upper-right source pixel is invalid and nearest-neighbour resize
    # keeps the corresponding quadrant black.
    assert np.array_equal(panel[100, 1000], np.zeros(3, np.uint8))
    assert np.any(panel[100, 100] != 0)


def test_union_text_hand_fallback_uses_valid_box_then_point_api(
    tmp_path: Path, monkeypatch,
) -> None:
    height, width = 100, 120
    left = np.zeros((height, width), bool)
    right = np.zeros((height, width), bool)
    left[20:70, 5:45] = True
    right[20:70, 75:115] = True
    union = left | right

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def init_state(self, **_kwargs):
            return {}

        def add_prompt(self, **kwargs):
            self.calls.append(kwargs)
            text = kwargs.get("text_str")
            boxes = kwargs.get("boxes_xywh")
            points = kwargs.get("points")
            obj_id = kwargs.get("obj_id")
            if text is not None and boxes is None:
                return 0, {
                    "out_binary_masks": union[None],
                    "out_probs": np.asarray([0.9]),
                    "out_obj_ids": np.asarray([0]),
                }
            if text is not None and boxes is not None:
                assert points is None and obj_id is None
                is_left = float(boxes[0][0]) < 0.5
                raw_id = 11 if is_left else 22
                mask = left if is_left else right
                return 0, {
                    "out_binary_masks": mask[None],
                    "out_probs": np.asarray([0.95]),
                    "out_obj_ids": np.asarray([raw_id]),
                }
            assert text is None and boxes is None
            assert points is not None and obj_id in {11, 22}
            mask = left if obj_id == 11 else right
            return 0, {
                "out_binary_masks": mask[None],
                "out_probs": np.asarray([0.97]),
                "out_obj_ids": np.asarray([obj_id]),
            }

    monkeypatch.setattr(sam_worker, "_write_mask", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sam_worker,
        "_propagate_to_directories",
        lambda _model, _state, **kwargs: {
            raw_id: 80 for raw_id in kwargs["id_to_directory"]
        },
    )
    plan = {
        "anchor_frame": 0,
        "hand_spatial_prompts": [
            {
                "role": "left_human_skin_forearm",
                "positive_points_xy": [(15.0, 30.0), (20.0, 40.0)],
                "negative_points_xy": [(90.0, 30.0)],
                "box_xywh": [5.0, 20.0, 40.0, 50.0],
            },
            {
                "role": "right_human_skin_forearm",
                "positive_points_xy": [(85.0, 30.0), (90.0, 40.0)],
                "negative_points_xy": [(20.0, 30.0)],
                "box_xywh": [75.0, 20.0, 40.0, 50.0],
            },
        ],
    }
    model = FakeModel()

    instances, evidence = sam_worker.track_hands(
        model, tmp_path, plan, tmp_path / "out", 100, height, width,
    )

    assert [row["role"] for row in instances] == [
        "left_human_skin_forearm", "right_human_skin_forearm",
    ]
    assert {row["physical_identity_policy"] for row in instances} == {
        "SIDE_LOCKED_HUMAN_ROLE"
    }
    assert evidence["route"] == (
        "INDEPENDENT_SAM31_SPATIAL_STATE_PER_SIDE_AFTER_TEXT_UNION"
    )
    fallback_calls = model.calls[1:]
    assert len(fallback_calls) == 4
    for box_call, point_call in zip(
        fallback_calls[::2], fallback_calls[1::2], strict=True,
    ):
        assert box_call["text_str"] is not None
        assert box_call["boxes_xywh"] is not None
        assert box_call.get("points") is None
        assert point_call["text_str"] is None
        assert point_call["boxes_xywh"] is None
        assert point_call["points"] is not None
        assert point_call["obj_id"] in {11, 22}
