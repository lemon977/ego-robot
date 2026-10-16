import numpy as np

from chaoyang.governance.common import REPO_ROOT
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.ops import run_robot100_natural_posture_v7_canary as v7
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


def test_reference_satisfies_complete_arm_shape() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    for side in range(2):
        metrics = v7.arm_shape_metrics(assets, mounts, neutral, side, v6.POSTURE_REFERENCE_Q[side])
        assert v7.shape_pass(metrics)


def test_projection_repairs_above_shoulder_pose() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    raised = np.array([0.0, 1.4, 0.0, -0.5, 0.0, 0.0, 0.0])
    repaired, alpha = v7.project_one(assets, mounts, neutral, 0, raised)
    assert 0.0 < alpha <= 1.0
    assert v7.shape_pass(v7.arm_shape_metrics(assets, mounts, neutral, 0, repaired))
    assert np.all(repaired >= lower[0])
    assert np.all(repaired <= upper[0])


def test_valid_reference_is_identity() -> None:
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    actual, alpha = v7.project_one(assets, mounts, neutral, 0, v6.POSTURE_REFERENCE_Q[0])
    assert alpha == 0.0
    assert np.array_equal(actual, v6.POSTURE_REFERENCE_Q[0])
