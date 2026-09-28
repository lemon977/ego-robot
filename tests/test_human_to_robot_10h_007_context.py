from __future__ import annotations

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops import run_human_to_robot_10h_007_context as candidate


def test_frozen_full_schedule_has_unique_output_and_preserves_canary_window() -> None:
    windows = candidate.full_windows()
    assert [(row[0], row[1]) for row in windows if row[0] == 181] == [(181, 196)]
    assert windows[0] == (0, 4, 0, 24)
    assert windows[-1] == (373, 377, 352, 377)
    assert [frame for first, last, _, _ in windows for frame in range(first, last + 1)] == list(range(378))
    assert all(context_first == max(0, first - 21) and context_last == min(377, last + 20)
               for first, last, context_first, context_last in windows)


def test_actual_vendor_schedule_consumes_added_context_for_center_frames() -> None:
    from chaoyang.ops.run_human_to_robot_10h_007_context import model_reference_schedule

    center = [row for row in model_reference_schedule() if 21 <= row["model_mid_local_index"] <= 36]
    assert center
    assert any(any(frame < 181 or frame > 196 for frame in row["reference_source_frames"])
               for row in center)
    assert all(not set(row["neighbor_local_indices"]) & set(row["reference_local_indices"])
               for row in model_reference_schedule())


def test_frozen_center_inputs_are_identical_to_old_canary_and_context_is_real() -> None:
    baseline = load_json(candidate.BASELINE / "input/RESULT.json")
    assert baseline["source_frames"] == list(range(181, 197))
    for frame in range(160, 217):
        sources = candidate._sources(frame)
        values = {kind: cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for kind, path in sources.items()}
        assert all(value is not None for value in values.values()), frame
        assert values["frames"].shape == (720, 960, 3)
        assert values["model_masks"].shape == (720, 960)
        assert values["write"].shape == values["protect"].shape == (960, 1280)
        assert values["model_masks"].any() and values["write"].any()
        assert not np.any((values["write"] > 0) & (values["protect"] > 0))
        if 181 <= frame <= 196:
            old = baseline["rows"][frame - 181]["prepared"]
            assert all(artifact_ref(path) == old[kind] for kind, path in sources.items())


def test_motion_review_uses_actual_saved_robot_and_does_not_project_invalid_side() -> None:
    for short, (session, frames, product_part) in candidate.MOTION_SESSIONS.items():
        product = candidate.REPO_ROOT / ("_run/current/human_to_robot_evidence_unlock_s2_20260923/"
                                         f"attempts/attempt_0001/lanes/motion_product/{product_part}/PRODUCT_RESULT.json")
        value = load_json(product)
        assert value["session_id"] == session and value["expected_frames"] == frames
        assert "ROBOT_R0_V1.npz" in value["robot_r0"]["path"]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[:, 2] = 1
    k = np.array([[20., 0., 50.], [0., 20., 50.], [0., 0., 1.]])
    candidate._project_hand(image, joints, np.zeros(21, dtype=bool), k, (0, 0, 255))
    assert not image.any()
    candidate._project_hand(image, joints, np.ones(21, dtype=bool), k, (0, 0, 255))
    assert image.any()


def test_independent_fk_reproduces_saved_valid_root_without_invalid_side_promotion() -> None:
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    path = candidate.REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
                                  "attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key][:50] if archive[key].ndim and len(archive[key]) == 149 else archive[key]
                for key in archive.files}
    actual, delta, status = candidate._independent_root_fk(load_pinned_robot_assets(candidate.REPO_ROOT), data)
    assert np.isnan(actual[47, 0]).all() and np.isnan(delta[47, 0])
    assert status[47, 0] == "SOURCE_INVALID"
    assert np.isfinite(actual[47, 1]).all() and delta[47, 1] < 1e-6 and status[47, 1] == "PASS"
    values = np.array([[1., np.nan], [23., 4.]])
    report = candidate._metric_summary(values, np.array([[True, False], [False, True]]), 20.)
    assert report["count"] == 2 and report["gate_fail_count"] == 0


def test_huro_hard_limit_is_a_side_local_rejection_not_silent_clipping() -> None:
    from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    path = candidate.REPO_ROOT / ("_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/"
                                  "lanes/huro/full_0001/get_potato_chips_0915_007/HURO_CORE_V1.npz")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key][:1] if archive[key].ndim and len(archive[key]) == 378 else archive[key]
                for key in archive.files}
    actual, _, status = candidate._independent_root_fk(load_pinned_robot_assets(candidate.REPO_ROOT), data)
    assert status[0, 0].startswith("FIXED_URDF_HARD_LIMIT")
    assert status[0, 1] == "PASS"
    assert np.isnan(actual[0, 0]).all() and np.isfinite(actual[0, 1]).all()


def test_sensor_review_is_bound_to_newest_display_not_old_solver_video() -> None:
    source = candidate.REPO_ROOT / ("_run/current/human_to_robot_sensor_display_correction_20260923/"
                                    "attempts/attempt_0001/lanes/sensor/review_metric_v2/RESULT.json")
    value = load_json(source)
    assert [row["frames"] for row in value["sessions"]] == [165, 179, 122]
    for row in value["sessions"]:
        assert "SENSOR_METRIC_REVIEW_V2.mp4" in row["video"]["path"]
        assert artifact_ref(candidate.Path(row["video"]["path"]))["sha256"] == row["video"]["sha256"]


def test_current_status_prefers_active_10h_delivery_over_old_terminal(monkeypatch, tmp_path) -> None:
    from chaoyang.governance import build_human_to_robot_r2_terminal_status as status_builder

    monkeypatch.setattr(status_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(status_builder, "OUTPUT", tmp_path / "docs/current/STATUS.json")
    state = {
        "governance_revision": 14073,
        "next_task": {"task_id": status_builder.TEN_HOUR_DELIVERY_TASK},
        "tasks": [
            {"task_id": status_builder.SURFACE_TEMPORAL_TASK, "status": "PASSED"},
            {"task_id": status_builder.TEN_HOUR_DELIVERY_TASK, "status": "RUNNING",
             "task_packet": {"path": "task-packet", "sha256": "test"}},
        ],
    }
    assert status_builder.publish_navigation_if_human_to_robot_r2(tmp_path, state, "test-time")
    current = load_json(status_builder.OUTPUT)
    assert current["latest_task"] == status_builder.TEN_HOUR_DELIVERY_TASK
    assert current["active_tasks"] == [status_builder.TEN_HOUR_DELIVERY_TASK]
    assert current["counts"]["products_quality"] == "0/4"


def test_geometry_review_terminal_correspondence_is_not_fabricated() -> None:
    base = candidate.REPO_ROOT / ("_run/current/human_to_robot_product_first_cleanup_20260923/"
                                  "attempts/attempt_0001/lanes")
    robot = base / "motion_product/robot_correspondence_031/window_v1"
    patch = base / "scene/object_patch_031/window_v1"
    assert load_json(robot / "RESULT.json")["source_frames"][-1] == 81
    assert not (robot / "correspondence_000081.png").exists()
    assert (patch / "overlay_000081.png").is_file()
