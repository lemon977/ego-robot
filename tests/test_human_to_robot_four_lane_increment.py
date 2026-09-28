"""Frozen input and negative-control checks for the CPU four-lane increment."""
from __future__ import annotations

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, load_json
from chaoyang.ops import run_human_to_robot_four_lane_increment as task
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


def test_frozen_scene_complaint_points_reach_actual_model_or_remain_unknown() -> None:
    old = REPO_ROOT / ("_run/current/human_to_robot_007_device_donor_probe_20260923/"
                       "attempts/attempt_0001/lanes/scene/device_donor_probe_v1/RESULT.json")
    points = load_json(old)["complaint_points"]
    folder = task.PRIOR / "lanes/scene/context_007_v1/input"
    model = cv2.imread(str(folder / "model_masks/000024.png"), cv2.IMREAD_GRAYSCALE)
    write = cv2.imread(str(folder / "write/000024.png"), cv2.IMREAD_GRAYSCALE)
    assert model is not None and write is not None
    supported = [bool(model[int(p["y"] * .75), int(p["x"] * .75)]) and bool(write[p["y"], p["x"]])
                 for p in points]
    assert supported == [True] * 6 + [False]
    raw = task.REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                            "lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw/000184.png")
    assert cv2.imread(str(raw)).shape == (960, 1280, 3)
    assert cv2.imread(str(folder / "frames/000024.png")).shape == (720, 960, 3)


def test_fixed_huro_urdf_has_actual_violations_not_a_renamed_pass() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    source = task.V5 / "lanes/huro/full_0001/get_potato_chips_0915_007/HURO_CORE_V1.npz"
    with np.load(source, allow_pickle=False) as archive:
        q, valid = archive["q_arm"], archive["wrist_valid"]
    violation = np.maximum(np.maximum(lower[None] - q, q - upper[None]), 0)
    assert np.count_nonzero(valid & np.any(violation > 1e-9, axis=-1)) > 0
    assert upper[0, 5] < np.max(q[:, 0, 5][valid[:, 0]])


def test_pose_error_does_not_make_a_displaced_robot_pass() -> None:
    target = np.eye(4)
    actual = np.eye(4)
    actual[0, 3] = .025
    position_mm, rotation_deg = task._pose_error(actual, target)
    assert position_mm == 25.0 and rotation_deg == 0.0


def test_031_invalid_left_side_remains_invalid_in_motion_input() -> None:
    source = task.R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    with np.load(source, allow_pickle=False) as archive:
        valid = archive["wrist_valid"]
        frame_id = archive["frame_id"]
    assert frame_id.tolist() == list(range(149))
    assert valid.sum(axis=0).tolist() == [0, 102]


def test_031_jump_is_already_in_saved_hand_motion_not_in_target_builder() -> None:
    folder = task.R2 / "lanes/lane2_motion/recovered_031_wave0"
    with np.load(folder / "HAND_MOTION_V1.npz", allow_pickle=False) as archive:
        hand = archive["T_camera_wrist"]
        valid = archive["position_valid"]
    with np.load(folder / "ROBOT_R0_V1.npz", allow_pickle=False) as archive:
        target = archive["T_target_root_cam"]
    assert np.array_equal(hand[..., :3, 3][valid], target[..., :3, 3][valid])
    assert not np.array_equal(hand[..., :3, :3][valid], target[..., :3, :3][valid])
    assert np.linalg.norm(hand[48, 1, :3, 3] - hand[47, 1, :3, 3]) > 1.0


def test_031_first_valid_source_box_is_tiny_and_source_wrist_is_preserved() -> None:
    source = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                          "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz")
    with np.load(source, allow_pickle=False) as archive:
        valid = archive["predicted_valid"][1]
        boxes = archive["detector_boxes_xyxy"][1]
        wrist = archive["joints_3d_camera"][1, :, 0]
    with np.load(task.R2 / "lanes/lane2_motion/recovered_031_wave0/HAND_MOTION_V1.npz",
                 allow_pickle=False) as archive:
        hand_wrist = archive["T_camera_wrist"][:, 1, :3, 3]
    assert np.flatnonzero(valid)[0] == 47
    assert (boxes[47, 2:] - boxes[47, :2]).tolist() == [22.0, 12.0]
    assert boxes[47, 3] == 960
    assert np.array_equal(wrist[valid], hand_wrist[valid])


def test_sensor_wave_paths_are_actual_published_arrays() -> None:
    for _, folder, _ in task.SENSOR_CASES:
        assert (task.R2 / "lanes/lane3_sensor" / folder / "HAND_MOTION_V1.npz").is_file()


def test_review_curve_does_not_bridge_invalid_frames() -> None:
    image = task._timeline_canvas("test", np.arange(4))
    task._draw_curve(image, np.array([10.0, np.nan, 10.0, 10.0]), 20.0, (0, 0, 255))
    y = 440 - int(10 * 340 / 20)
    # The invalid second frame must not connect the first and third samples.
    assert tuple(image[y, 75 + int(1145 / 3)]) != (0, 0, 255)
    assert tuple(image[y, 75]) == (0, 0, 255)


def test_derived_huro_review_inputs_are_bounded_without_mutating_source() -> None:
    result = task.ATTEMPT / "lanes/compare/huro_limit_repair_window_v1/RESULT.json"
    rows = load_json(result)["sessions"]
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    for row in rows:
        with np.load(row["result_array"]["path"], allow_pickle=False) as archive:
            q = archive["q_derived"]
            valid = archive["source_valid"]
        assert np.all(np.isfinite(q[valid]))
        assert np.all(q[valid] >= lower[np.broadcast_to(np.arange(2), valid.shape)[valid]] - 1e-8)
        assert np.all(q[valid] <= upper[np.broadcast_to(np.arange(2), valid.shape)[valid]] + 1e-8)
