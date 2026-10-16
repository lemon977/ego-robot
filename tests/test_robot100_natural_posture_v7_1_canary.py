import numpy as np
import pytest

from chaoyang.ops.run_robot100_natural_posture_v7_1_canary import temporal_majorant


def test_temporal_majorant_preserves_requirement_and_limits_change() -> None:
    required = np.zeros((12, 2), dtype=float)
    required[5, 0] = 0.42
    valid = np.ones((12, 2), dtype=bool)
    edges = np.ones(12, dtype=bool)
    actual = temporal_majorant(required, valid, edges, 0.03)
    assert np.all(actual >= required)
    assert np.max(np.abs(np.diff(actual, axis=0))) <= 0.03 + 1e-12
    assert actual[4, 0] > 0 and actual[6, 0] > 0
    assert np.all(actual[:, 1] == 0)


def test_temporal_majorant_does_not_cross_gap_or_invalid_frame() -> None:
    required = np.zeros((8, 1), dtype=float)
    required[2, 0] = 0.5
    valid = np.ones((8, 1), dtype=bool)
    edges = np.ones(8, dtype=bool)
    edges[4] = False
    actual = temporal_majorant(required, valid, edges, 0.1)
    assert np.any(actual[:4, 0] > 0)
    assert np.all(actual[4:, 0] == 0)
    valid[3, 0] = False
    actual = temporal_majorant(required, valid, np.ones(8, dtype=bool), 0.1)
    assert actual[3, 0] == 0
    assert np.all(actual[4:, 0] == 0)


def test_temporal_majorant_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="PROJECTION_SHAPE"):
        temporal_majorant(np.zeros(3), np.zeros(3, dtype=bool), np.ones(3, dtype=bool), 0.1)
    with pytest.raises(ValueError, match="PROJECTION_SLOPE"):
        temporal_majorant(np.zeros((3, 1)), np.ones((3, 1), dtype=bool), np.ones(3, dtype=bool), 0.0)
