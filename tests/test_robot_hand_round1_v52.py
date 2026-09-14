from __future__ import annotations

import numpy as np
import pytest

from tools import run_exact78_robot_hand_round1_v52 as hand
from tools import run_newtask_robot_shared_v4_hand as handfit


class TemporalV2Stub:
    @staticmethod
    def difference_matrices(delta_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = int(delta_t.size + 1)
        d1 = np.zeros((max(0, count - 1), count), dtype=np.float64)
        for row, dt in enumerate(delta_t):
            d1[row, row] = -1.0 / dt
            d1[row, row + 1] = 1.0 / dt
        if count < 3:
            return d1, np.empty((0, count)), np.empty((0, count))
        acceleration_dt = 0.5 * (delta_t[:-1] + delta_t[1:])
        d2 = np.zeros((count - 2, count), dtype=np.float64)
        for row, dt in enumerate(acceleration_dt):
            d2[row] = (d1[row + 1] - d1[row]) / dt
        if count < 4:
            return d1, d2, np.empty((0, count))
        jerk_dt = 0.5 * (acceleration_dt[:-1] + acceleration_dt[1:])
        d3 = np.zeros((count - 3, count), dtype=np.float64)
        for row, dt in enumerate(jerk_dt):
            d3[row] = (d2[row + 1] - d2[row]) / dt
        return d1, d2, d3

    @staticmethod
    def trajectory_metrics(
        values: np.ndarray, d1: np.ndarray, d2: np.ndarray, d3: np.ndarray
    ) -> dict[str, float]:
        def maximum(operator: np.ndarray) -> float:
            if not operator.shape[0]:
                return 0.0
            return float(np.max(np.abs(np.einsum("ij,jpk->ipk", operator, values))))

        return {
            "velocity_max_rad_per_second": maximum(d1),
            "acceleration_max_rad_per_second2": maximum(d2),
            "jerk_max_rad_per_second3": maximum(d3),
        }


def temporal_contracts(joints: int = 22) -> list[dict[str, np.ndarray]]:
    return [
        {
            "lower": np.full(joints, -1.0, dtype=np.float64),
            "upper": np.full(joints, 1.0, dtype=np.float64),
        }
        for _ in range(2)
    ]


def test_temporal_project_preserves_singleton_observed_segment() -> None:
    raw = np.linspace(-0.5, 0.5, 44, dtype=np.float64).reshape(1, 2, 22)
    projected, metrics, solvers = handfit.temporal_project(
        TemporalV2Stub(), raw, temporal_contracts(), np.asarray([17]), 30.0
    )

    np.testing.assert_array_equal(projected, raw)
    assert metrics["delta_t_seconds"] == []
    assert metrics["velocity_max_rad_per_second"] == 0.0
    assert metrics["acceleration_max_rad_per_second2"] == 0.0
    assert metrics["jerk_max_rad_per_second3"] == 0.0
    assert metrics["pass"] is True
    assert len(solvers) == 44
    assert {row["optimizer_message"] for row in solvers} == {
        "SINGLE_FRAME_NO_TEMPORAL_CONSTRAINTS"
    }


def test_temporal_project_accepts_two_frames_without_empty_reduction() -> None:
    raw = np.zeros((2, 2, 22), dtype=np.float64)
    projected, metrics, solvers = handfit.temporal_project(
        TemporalV2Stub(), raw, temporal_contracts(), np.asarray([17, 18]), 30.0
    )

    assert projected.shape == raw.shape
    assert metrics["acceleration_max_rad_per_second2"] == 0.0
    assert metrics["jerk_max_rad_per_second3"] == 0.0
    assert len(solvers) == 44


def test_temporal_project_rejects_non_increasing_source_frames() -> None:
    raw = np.zeros((2, 2, 22), dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        handfit.temporal_project(
            TemporalV2Stub(), raw, temporal_contracts(), np.asarray([17, 17]), 30.0
        )


def test_arm_lineage_accepts_only_final_arm_contract() -> None:
    method1_result = {"path": "/tmp/m1.json", "bytes": 1, "sha256": "a" * 64}
    method1_states = {"path": "/tmp/m1.npz", "bytes": 2, "sha256": "b" * 64}
    assert hand.arm_lineage(
        {
            "status": "PASS_ARM_METHOD1_CARRIED_NO_ROUND2",
            "method1_result": method1_result,
            "method1_states": method1_states,
        },
        "s1",
    ) == (method1_result, method1_states, 1, True)

    round2_result = {"path": "/tmp/m2.json", "bytes": 3, "sha256": "c" * 64}
    round2_states = {"path": "/tmp/m2.npz", "bytes": 4, "sha256": "d" * 64}
    assert hand.arm_lineage(
        {
            "status": "PASS_ARM_ROUND2_BIDIRECTIONAL",
            "result": round2_result,
            "states": round2_states,
        },
        "s2",
    ) == (round2_result, round2_states, 2, True)
    assert hand.arm_lineage(
        {
            "status": "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C",
            "result": round2_result,
            "states": round2_states,
        },
        "s3",
    ) == (round2_result, round2_states, 2, False)

    with pytest.raises(ValueError, match="unexpected final arm status"):
        hand.arm_lineage({"status": "FAILED_RUNTIME_RETRYABLE"}, "s4")


def test_unique_sessions_rejects_duplicates_and_missing_identity() -> None:
    assert hand.unique_sessions([{"session": "a"}, {"session": "b"}], "rows") == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate"):
        hand.unique_sessions([{"session": "a"}, {"session": "a"}], "rows")
    with pytest.raises(ValueError, match="non-empty"):
        hand.unique_sessions([{"session": ""}], "rows")
    with pytest.raises(ValueError, match="non-empty"):
        hand.unique_sessions([{}], "rows")
