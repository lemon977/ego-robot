from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hawor_optimizer_job72",
    ROOT / "NOW/daemon/jobs/72_hawor_apply_optimizer_all59.py",
)
assert SPEC is not None and SPEC.loader is not None
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)

SPEC_131 = importlib.util.spec_from_file_location(
    "hawor_onboard_job131",
    ROOT / "NOW/daemon/jobs/131_onboard_0902.py",
)
assert SPEC_131 is not None and SPEC_131.loader is not None
JOB_131 = importlib.util.module_from_spec(SPEC_131)
SPEC_131.loader.exec_module(JOB_131)


def test_camera_to_world_transforms_every_side_frame_and_joint() -> None:
    joints = np.zeros((2, 3, 21, 3), dtype=np.float64)
    joints[..., 0] = 0.25
    joints[..., 2] = 0.5
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    c2w[:, :3, 3] = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    )

    world = JOB.camera_to_world(joints, c2w)

    assert world.shape == joints.shape
    np.testing.assert_allclose(world[1, 2, 7], [3.25, 4.0, 5.5])


def test_camera_to_world_rejects_frame_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        JOB.camera_to_world(np.zeros((2, 3, 21, 3)), np.zeros((2, 4, 4)))


def test_camera_to_world_preserves_nan_and_supports_rotation() -> None:
    joints = np.asarray([[[[1.0, 2.0, 3.0], [np.nan, 0.0, 1.0]]]])
    c2w = np.eye(4, dtype=np.float64)[None]
    # +90 degrees around camera Z followed by translation.
    c2w[0, :3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    c2w[0, :3, 3] = [10.0, 20.0, 30.0]

    world = JOB.camera_to_world(joints, c2w)

    np.testing.assert_allclose(world[0, 0, 0], [8.0, 21.0, 33.0])
    assert np.isnan(world[0, 0, 1]).any()


def test_onboard_internal_optimizer_regenerates_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "chips" / "sample" / "hawor" / "run" / "raw.npz"
    raw.parent.mkdir(parents=True)
    camera = np.zeros((2, 31, 21, 3), dtype=np.float64)
    camera[..., 2] = 0.5
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 31, axis=0)
    c2w[:, 0, 3] = 1.0
    np.savez_compressed(
        raw,
        joints_3d_camera=camera,
        joints_3d_world=np.zeros_like(camera),
        joints_2d=np.zeros((2, 31, 21, 2)),
        intrinsics=np.repeat(np.eye(3)[None], 31, axis=0),
        c2w=c2w,
    )

    class FakeJob72:
        np = np
        SMOOTH_WEIGHT = 12.0

        @staticmethod
        def newest_raw_npz(_hawor: Path) -> Path:
            return raw

        @staticmethod
        def optimize(joints: np.ndarray, _valid: np.ndarray) -> np.ndarray:
            result = joints.copy()
            result[..., 1] += 0.25
            return result

        @staticmethod
        def project(joints: np.ndarray, _intrinsics: np.ndarray) -> np.ndarray:
            return np.zeros(joints.shape[:-1] + (2,))

        camera_to_world = staticmethod(JOB.camera_to_world)

    monkeypatch.setattr(JOB_131, "PROCESSED", tmp_path)
    monkeypatch.setattr(JOB_131, "load_module", lambda *_args: FakeJob72)

    assert JOB_131.internal_optimize("chips", "sample") == 0
    output = raw.parent / JOB_131.OPT_NPZ
    with np.load(output, allow_pickle=True) as data:
        expected = JOB.camera_to_world(data["joints_3d_camera"], data["c2w"])
        np.testing.assert_allclose(data["joints_3d_world"], expected)
        assert np.all(data["joints_3d_camera"][..., 1] == 0.25)
