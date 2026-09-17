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


def test_hawor_worker_gives_upstream_a_fresh_child_and_publishes_npz_at_session_root() -> None:
    source = inspect.getsource(worker.main)
    assert 'upstream_output = staging / "upstream_hawor"' in source
    assert 'upstream_npz = upstream_output / "HAWOR_RAW_MANO21.npz"' in source
    assert 'npz = staging / "HAWOR_RAW_MANO21.npz"' in source
    assert 'staging / "HAWOR_RAW_REVIEW.mp4"' not in source


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
    assert metrics["numeric_mask_gate_pass"] is False


def test_hawor_numeric_gate_requires_frozen_coverage_and_provenance(tmp_path) -> None:
    frame_count = 20
    observed = np.ones((2, frame_count), dtype=bool)
    joints_2d = np.zeros((2, frame_count, 21, 2), dtype=np.float32)
    joints_3d = np.zeros((2, frame_count, 21, 3), dtype=np.float32)
    for side in range(2):
        joints_2d[side, ..., 0] = 200 + side * 400 + np.arange(21)
        joints_2d[side, ..., 1] = 300 + np.arange(21)
        joints_3d[side, ..., 0] = np.arange(21) * 0.01
        joints_3d[side, ..., 1] = side * 0.02
        joints_3d[side, ..., 2] = 0.5
    rotations = np.broadcast_to(np.eye(3), (2, frame_count, 3, 3)).copy()
    confidence = np.full((2, frame_count), 0.8, np.float32)
    path = tmp_path / "pass.npz"
    np.savez(
        path, observed=observed, joints_2d=joints_2d,
        joints_3d_camera=joints_3d, joints_3d_world=joints_3d,
        root_orient_camera=rotations, detector_confidence=confidence,
        provenance=np.full((2, frame_count), "OBSERVED"), fps=np.asarray(30.0),
    )
    metrics = worker.quality(path)
    assert metrics["numeric_mask_gate_pass"] is True
    assert metrics["sides"]["left"]["bone_length_cv_max"] == 0.0


def test_staging_reference_records_final_atomic_path(tmp_path) -> None:
    staging = tmp_path / ".session.tmp" / "video.mp4"
    staging.parent.mkdir()
    staging.write_bytes(b"video")
    final = tmp_path / "session" / "video.mp4"
    value = prepare.ref_as(staging, final)
    assert value["path"] == str(final.resolve())
    assert ".session.tmp" not in value["path"]
    assert value["bytes"] == 5
