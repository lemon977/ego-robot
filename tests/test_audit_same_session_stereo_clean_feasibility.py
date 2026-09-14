from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/audit_same_session_stereo_clean_feasibility.py"
SPEC = importlib.util.spec_from_file_location("stereo_clean_audit_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_union_fraction_counts_union_on_fixed_denominator() -> None:
    existing = np.array([True, False, False, True])
    candidate = np.array([False, True, False, True])
    assert MODULE.union_fraction(existing, candidate) == pytest.approx(0.75)


def test_union_fraction_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        MODULE.union_fraction(np.zeros(2, dtype=bool), np.zeros(3, dtype=bool))


def test_array_hash_binds_dtype_shape_and_bytes() -> None:
    value = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert MODULE.sha256_array(value) == MODULE.sha256_array(value.copy())
    assert MODULE.sha256_array(value) != MODULE.sha256_array(value.reshape(3, 2))
    assert MODULE.sha256_array(value) != MODULE.sha256_array(value.astype(np.int16))


@pytest.mark.parametrize(
    ("coverage", "consistency", "expected_status", "expected_grade", "expected_authorized"),
    [
        (0.49, 0.99, "HOLD_STEREO_INSUFFICIENT_GEOMETRIC_COVERAGE", "C", False),
        (0.75, 0.79, "HOLD_STEREO_OCCLUSION_REGISTRATION_UNTRUSTED", "C", False),
        (0.75, 0.80, "PASS_STEREO_SUCCESSOR_FEASIBLE", "B", True),
    ],
)
def test_output_decision_is_fail_closed(
    coverage: float,
    consistency: float,
    expected_status: str,
    expected_grade: str,
    expected_authorized: bool,
) -> None:
    assert MODULE.output_decision(coverage, consistency) == (
        expected_status,
        expected_grade,
        expected_authorized,
    )


class _DummyEye:
    pass


class _DummyPico:
    @staticmethod
    def project_equidis62(points: np.ndarray, eye: object) -> np.ndarray:
        assert isinstance(eye, _DummyEye)
        return points[:, :2]


def test_projection_applies_right_camera_baseline_before_unrectification() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = MODULE.project_rectified_to_raw_right(
        points,
        baseline=0.25,
        right_rotation=np.eye(3),
        right_eye=_DummyEye(),
        pico=_DummyPico(),
    )
    np.testing.assert_allclose(actual, [[0.75, 2.0], [3.75, 5.0]])
    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
