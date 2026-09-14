from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
CANARY_PATH = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
CONTRACT_PATH = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "robot_chips034_gradeb_cross_chirality_successor_v1/CONTRACT.json"
)
VIABILITY_CONTRACT_PATH = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "robot_chips034_gradeb_viability_successor_v2/CONTRACT.json"
)
FORMAL_METHOD_CONTRACT_PATH = (
    PROJECT
    / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    / "robot_formal_method_contracts_v1/chips034_v2.json"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


canary = load(CANARY_PATH, "test_robot_gradeb_cross_chirality_canary")


def invocation() -> argparse.Namespace:
    return argparse.Namespace(
        task_id="chips",
        session_id="get_potato_chips_0902_034",
        window_frames=24,
        prefer_earliest_window=True,
        strict_previous_accepted=True,
        fixed_previous_ik_branch=True,
        first_frame_static_exception=True,
        fast_hand_retarget=False,
        skip_expensive_self_collision=False,
        arm_step_limit=0.12,
        hand_step_limit=0.08,
    )


def test_frozen_contract_accepts_only_exact_bounded_invocation() -> None:
    contract = canary.load_gradeb_successor_contract(CONTRACT_PATH, invocation())
    assert contract["scope"]["formal_deployment_authorized"] is False
    bad = invocation()
    bad.fast_hand_retarget = True
    with pytest.raises(RuntimeError, match="invocation or frozen contract mismatch"):
        canary.load_gradeb_successor_contract(CONTRACT_PATH, bad)


def test_formal_method_contract_accepts_full_session_without_gate_relaxation() -> None:
    args = invocation()
    args.window_frames = 293
    contract = canary.load_gradeb_successor_contract(
        FORMAL_METHOD_CONTRACT_PATH, args
    )
    assert contract["scope"]["formal_robot_execution_requires_fresh_preflight"] is True
    assert contract["scope"]["formal_deployment_authorized"] is False
    assert contract["temporal_hard_gates"]["full_fixed_denominator_per_link_collision_required_in_postprocessor"] is True
    assert contract["hand_retarget"]["hundred_degree_relaxation_allowed"] is False


def test_temporal_bounds_intersect_step_acceleration_and_urdf() -> None:
    lower = np.asarray([-1.0, -0.2])
    upper = np.asarray([1.0, 0.2])
    previous = np.asarray([0.20, 0.18])
    previous_previous = np.asarray([0.15, 0.10])
    bounded_lower, bounded_upper = canary.bounded_temporal_limits(
        lower, upper, previous, previous_previous, 0.12, 0.06
    )
    assert np.allclose(bounded_lower, [0.19, 0.20])
    assert np.allclose(bounded_upper, [0.31, 0.20])
    assert np.all(bounded_lower >= lower)
    assert np.all(bounded_upper <= upper)


def test_viability_velocity_cap_prevents_next_frame_empty_domain() -> None:
    lower = np.asarray([-1.0])
    upper = np.asarray([1.0])
    with pytest.raises(RuntimeError, match="empty previous-accepted temporal bounds"):
        canary.bounded_temporal_limits(
            lower,
            upper,
            np.asarray([1.0]),
            np.asarray([0.92]),
            0.08,
            0.06,
        )
    bounded_lower, bounded_upper = canary.bounded_temporal_limits(
        lower,
        upper,
        np.asarray([1.0]),
        np.asarray([0.94]),
        0.06,
        0.06,
    )
    assert bounded_lower[0] == pytest.approx(1.0)
    assert bounded_upper[0] == pytest.approx(1.0)
    contract = canary.load_gradeb_successor_contract(
        VIABILITY_CONTRACT_PATH, invocation()
    )
    temporal = contract["temporal_hard_gates"]
    assert temporal["arm_solver_effective_step_max_rad"] == 0.06
    assert temporal["hand_solver_effective_step_max_rad"] == 0.06
    assert temporal["one_step_stop_viability_required"] is True


def test_hand_gate_accepts_morphology_outlier_but_rejects_100_degree_error() -> None:
    contract = canary.load_gradeb_successor_contract(CONTRACT_PATH, invocation())
    detail = {
        finger: {
            "bone_errors_deg": [53.0, 10.0, 10.0, 20.0],
            "mean_bone_error_deg": 23.25,
            "max_bone_error_deg": 53.0,
        }
        for finger in canary.FINGERS
    }
    assert canary.gradeb_hand_shape_gate(detail, contract)["pass"] is True
    detail["pinky"]["bone_errors_deg"] = [99.0, 10.0, 10.0, 20.0]
    detail["pinky"]["mean_bone_error_deg"] = 34.75
    detail["pinky"]["max_bone_error_deg"] = 99.0
    assert canary.gradeb_hand_shape_gate(detail, contract)["pass"] is False


class RotationStub:
    @staticmethod
    def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
        angle = np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
        return np.asarray([angle, 0.0, 0.0])


def test_frame0_visual_placement_is_exact_and_cap_bounded() -> None:
    tool_world = np.eye(4)
    target = np.eye(4)
    angle = np.deg2rad(20.0)
    target[:3, :3] = [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    target[0, 3] = 0.04
    effective, audit = canary.derive_frame0_static_visual_placement(
        tool_world, target, np.eye(4), RotationStub, 0.05, 35.0
    )
    assert np.allclose(tool_world @ effective, target)
    assert audit["post_position_mm"] == pytest.approx(0.0)
    assert audit["post_rotation_deg"] == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="translation cap"):
        canary.derive_frame0_static_visual_placement(
            tool_world, target, np.eye(4), RotationStub, 0.039, 35.0
        )
