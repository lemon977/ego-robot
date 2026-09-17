from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops import prepare_0915_physical_left_batch_v1 as prepare
from chaoyang.ops import run_0915_hawor_persistent_worker_v1 as worker


def test_physical_left_c2w_uses_same_session_extrinsics() -> None:
    selected = np.eye(4)
    left = np.eye(4)
    left[0, 3] = -0.03
    right = np.eye(4)
    right[0, 3] = 0.04
    camera = {
        "extrinsic_convention": "head_to_camera_4x4_row_major",
        "extrinsics": {"left": left.tolist(), "right": right.tolist()},
    }
    actual = prepare.physical_left_c2w(selected, camera)
    np.testing.assert_allclose(actual, selected @ right @ np.linalg.inv(left))


def test_preparation_is_dynamic_and_does_not_consume_pico_sidecars() -> None:
    source = inspect.getsource(prepare)
    assert "frame_count != 379" not in source
    assert "sourceIndex" in source
    assert '"pico26": "PRESENT_PRESERVED_NOT_CONSUMED"' in source
    assert "controller_poses" not in source
    assert 'glob("trackingData' not in source
    assert "trackingData_hand\": \"NOT_CONSUMED" in source


def test_hawor_worker_caches_models_and_resets_tracker_per_session() -> None:
    source = inspect.getsource(worker.install_caches)
    assert 'cache["hawor"] is None' in source
    assert 'cache["yolo"] is None' in source
    assert 'cache["yolo"].predictor = None' in source
    assert "hawor_load_count" in source
    assert "detector_load_count" in source


def test_hawor_quality_keeps_missing_frames_missing(tmp_path) -> None:
    observed = np.zeros((2, 3), dtype=bool)
    observed[0, 0] = True
    joints_2d = np.full((2, 3, 21, 2), np.nan, dtype=np.float32)
    joints_3d = np.full((2, 3, 21, 3), np.nan, dtype=np.float32)
    joints_2d[0, 0] = 10
    joints_3d[0, 0] = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
    path = tmp_path / "result.npz"
    np.savez(path, observed=observed, joints_2d=joints_2d,
             joints_3d_camera=joints_3d)
    metrics = worker.quality(path)
    assert metrics["observed_frames"] == {"left": 1, "right": 0, "bilateral": 0}
    assert metrics["observed_fraction"]["right"] == 0.0


def test_staging_reference_records_final_atomic_path(tmp_path) -> None:
    staging = tmp_path / ".session.tmp" / "video.mp4"
    staging.parent.mkdir()
    staging.write_bytes(b"video")
    final = tmp_path / "session" / "video.mp4"
    value = prepare.ref_as(staging, final)
    assert value["path"] == str(final.resolve())
    assert ".session.tmp" not in value["path"]
    assert value["bytes"] == 5
