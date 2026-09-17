from pathlib import Path

import numpy as np

from chaoyang.pipeline.robot_visual_relative_v1 import (
    HandLimits,
    mano_flexion,
    relative_tool_targets,
    retarget_kaihand_frame,
)


def _mano() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
              (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for finger, chain in enumerate(chains):
        for step, joint in enumerate(chain[1:], 1):
            points[joint] = (0.01 * finger, 0.02 * step, 0.005 * step * step)
    return points


def test_relative_targets_keep_missing_unknown_and_are_not_calibration() -> None:
    palm = np.full((2, 3, 4, 4), np.nan)
    observed = np.asarray([[True, False, True], [True, True, False]])
    for side, frame in zip(*np.nonzero(observed)):
        palm[side, frame] = np.eye(4)
        palm[side, frame, 0, 3] = 0.01 * frame
    tools = np.stack((np.eye(4), np.eye(4)))
    tools[:, 2, 3] = 0.5
    targets, valid = relative_tool_targets(
        palm, observed, tools, scale=0.75, yaw_deg=15.0,
    )
    assert valid.shape == (3, 2)
    assert np.array_equal(valid.T, observed)
    assert np.isnan(targets[1, 0]).all()
    assert np.isnan(targets[2, 1]).all()
    assert np.isfinite(targets[valid]).all()


def test_flexion_retarget_is_bounded_by_kaihand_limits() -> None:
    points = _mano()
    flexion = mano_flexion(points)
    assert flexion.shape == (5, 3)
    limits = HandLimits(
        lower=np.full(22, -0.2), upper=np.full(22, 0.8),
        neutral=np.full(22, 0.3),
    )
    q, loss = retarget_kaihand_frame(points, limits)
    assert q.shape == (22,)
    assert np.all(q >= limits.lower)
    assert np.all(q <= limits.upper)
    assert 0.0 <= loss <= 1.0


def test_module_has_no_archive_or_unit_calibration_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src/chaoyang/pipeline/robot_visual_relative_v1.py"
    ).read_text(encoding="utf-8").lower()
    assert "archive/" not in source
    assert "camera_world_to_base\": \"absent" in source
    assert "robot_tcp\": \"absent" in source
