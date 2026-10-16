import numpy as np
import pytest

from chaoyang.governance.common import REPO_ROOT
from chaoyang.governance import finalize_robot100_inward_successor_v6 as finalizer
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v5 as v5
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


def test_posture_reference_is_bounded_and_inward() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    assert np.all(v6.POSTURE_REFERENCE_Q >= lower)
    assert np.all(v6.POSTURE_REFERENCE_Q <= upper)
    for side in range(2):
        assert v6.inward_margin_m(
            assets, mounts, neutral, side, v6.POSTURE_REFERENCE_Q[side]
        ) >= v6.MIN_INWARD_M


def test_projection_is_identity_for_valid_reference() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    for side in range(2):
        actual, alpha = v6.project_one(
            assets, mounts, neutral, side, v6.POSTURE_REFERENCE_Q[side]
        )
        assert alpha == 0.0
        assert np.array_equal(actual, v6.POSTURE_REFERENCE_Q[side])


def test_projection_repairs_old_midpoint_without_leaving_limits() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    for side in range(2):
        assert v6.inward_margin_m(assets, mounts, neutral, side, neutral[side]) < v6.MIN_INWARD_M
        actual, alpha = v6.project_one(assets, mounts, neutral, side, neutral[side])
        assert 0.0 < alpha <= 1.0
        assert np.all(actual >= lower[side])
        assert np.all(actual <= upper[side])
        assert v6.inward_margin_m(assets, mounts, neutral, side, actual) >= v6.MIN_INWARD_M - 1e-8


def test_v6_keeps_v5_selection_and_output_is_separate() -> None:
    assert v6.OUTPUT != v5.OUTPUT
    assert v6.OUTPUT.name == "successor_sessions"
    assert v6.OUTPUT.parent.name == "inward_forearm_posture_v6"


def test_raw_predecessor_evidence_must_be_identical() -> None:
    predecessor = {
        name: np.array([1.0, np.nan]) for name in finalizer.RAW_PRESERVED_ARRAYS
    }
    current = {name: value.copy() for name, value in predecessor.items()}
    finalizer.verify_raw_predecessor_preserved(current, predecessor)
    current["q_arm_raw"][0] = 2.0
    with pytest.raises(RuntimeError, match="RAW_PREDECESSOR_DRIFT:q_arm_raw"):
        finalizer.verify_raw_predecessor_preserved(current, predecessor)
