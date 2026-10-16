"""Three bounded CPU test families; all providers/data are synthetic fixtures."""
from copy import deepcopy

import numpy as np
import pytest

from chaoyang.humanego_ict.assumed_deployment_v1 import (
    AssumedDeploymentError, MODE, PROFILE_SCHEMA, ROT6_DECLARATION, bridge_assumed_action_chunk,
    build_assumed_ict,
)
from chaoyang.humanego_ict.bridge import IKSolution
from chaoyang.humanego_ict.deployment_adapter import DeploymentAdapterError, build_measured_ict

NORMALIZATION = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
FALSE_FLAGS = ("measured", "geometry_qualified", "geometry_verified", "timebase_verified",
               "feedback_qualified", "hardware_ack_verified", "control_ground_truth",
               "physical_deployment_authorized")


def profile(*, with_limits=True):
    # Match the root-owned conceptual ASSUMPTIONS.json's schema/list structure.
    p = {"schema": PROFILE_SCHEMA, "mode": MODE, "position_unit": "meters",
         "rotation_convention": ROT6_DECLARATION,
         "hand_channel_order": ["thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky"],
         "device_ack_status": "UNKNOWN", **{key: False for key in FALSE_FLAGS},
         "assumptions": [
             {"id": "WORLD_CAMERA_IDENTITY_FOR_OFFLINE_MATH", "matrix_direction": "T_world_from_camera"},
             {"id": "BASE_TRANSFORMS_CALLER_SUPPLIED_OR_IDENTITY", "matrix_direction": "T_world_from_arm_base"},
             {"id": "MODEL_TOOL_EQUALS_TRAINING_WRIST_PRIOR", "matrix_direction": "T_model_tool_from_training_wrist_prior"},
             {"id": "NOMINAL_TIMESTAMPS_ONLY_WHEN_SOURCE_TIME_ABSENT", "nominal_hz": 30},
         ]}
    for item in p["assumptions"]:
        item.update(source_class="ASSUMED_NOT_MEASURED", physical_qualification=False)
        if "matrix_direction" in item:
            item["default_matrix"] = np.eye(4).tolist()
    if with_limits:
        # Explicit synthetic numeric limits, not proposed production thresholds.
        p["numerical_limits"] = {"joint_lower_rad": [-2.] * 14, "joint_upper_rad": [2.] * 14,
            "joint_speed_max_rad_s": [1.] * 14, "joint_step_max_rad": [.1] * 14,
            "position_tolerance_m": .0001, "rotation_tolerance_rad": .0001}
    return p


class MathematicalProvider:
    offline_only = True
    signature = "f" * 64  # Clearly synthetic fixture callback signature.

    def __init__(self, fault=None):
        self.fault, self.solve_calls, self.fk_calls = fault, 0, 0

    def fk(self, q):
        self.fk_calls += 1
        result = np.repeat(np.eye(4)[None], 2, axis=0)
        result[0, :3, 3], result[1, :3, 3] = q[:3], q[7:10]
        if self.fault == "residual":
            result[0, 0, 3] += .002
        if self.fault == "rotation":
            angle = .002
            result[0, :3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        return result

    def solve(self, targets, seed, dt):
        self.solve_calls += 1
        if self.fault == "exception":
            raise RuntimeError("SYNTHETIC_PROVIDER_FAILURE")
        q = seed.copy()
        q[:3], q[7:10] = targets[0, :3, 3], targets[1, :3, 3]
        if self.fault == "joint_limit":
            q[0] = 3.
        if self.fault == "step":
            q[0] = .2
        if self.fault == "nonfinite_q":
            q[0] = np.nan
        return IKSolution(q, self.fault != "unchecked", self.fault != "collision")


def anchor():
    return {"source_frame_id": 7, "anchor_ns": 1_000_000_000,
            "joint_ns": 990_000_000, "image_ns": 995_000_000,
            "time_source": "ASSUMED_OFFLINE_CLOCK"}


def mapping():
    return {"source_frame_ids": np.arange(8, 58, dtype=np.int64),
            "timestamp_ns": np.arange(1, 51, dtype=np.int64) * 100_000_000 + 1_000_000_000,
            "anchor_ns": 1_000_000_000, "mapping_kind": "ASSUMED_OFFLINE_GRID"}


def actions():
    a = np.zeros((50, 30), np.float64)
    a[:, :6] = [.02, .01, 0, -.02, -.01, 0]
    a[:, 6:12] = a[:, 12:18] = np.eye(3)[:, :2].reshape(6)
    a[:, 18:24], a[:, 24:30] = .25, .75
    return a


def assert_not_physical(result):
    assert result["mode"] == MODE
    assert result["parameter_mode"] == "ASSUMED"
    assert result["offline_only"] is True
    assert result["device_ack_status"] == "UNKNOWN"
    assert result["robot_commands_sent"] == 0
    for key in FALSE_FLAGS:
        assert result[key] is False


@pytest.mark.parametrize("case", ["missing", "wrong"])
def test_assumed_profile_rejects_ambiguous_rotation_before_provider(case):
    p, provider = profile(), MathematicalProvider()
    if case == "missing":
        p.pop("rotation_convention")
    else:
        p["rotation_convention"] = "COLUMN_MAJOR_FIRST_TWO_COLUMNS"
    with pytest.raises(AssumedDeploymentError, match="FROZEN_ROT6_DECLARATION_REQUIRED"):
        build_assumed_ict(np.zeros(14), x_rgb=np.zeros((1, 3, 240, 320), np.float32),
            assumption_profile=p, fk_provider=provider, normalization=NORMALIZATION,
            anchor=anchor())
    assert provider.fk_calls == 0


@pytest.mark.parametrize("case", ["declared_defaults", "transform_directions", "no_provider",
    "future_joint", "future_image", "nonfinite_q", "bad_rgb", "nonidentity_normalization"])
def test_assumed_ict_preserves_training_math_and_input_unknowns(case):
    p, clock, provider = profile(), anchor(), MathematicalProvider()
    rgb, q = np.zeros((1, 3, 240, 320), np.uint8), np.zeros(14)
    q[:3], q[7:10] = [.1, .2, .3], [-.2, .15, .5]
    norm = deepcopy(NORMALIZATION)
    if case == "transform_directions":
        camera = np.eye(4)
        camera[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        camera[:3, 3] = [.1, .2, .3]
        bases, tools = np.repeat(np.eye(4)[None], 2, axis=0), np.repeat(np.eye(4)[None], 2, axis=0)
        bases[:, :3, 3] = [[.3, 0, 0], [-.3, 0, 0]]
        tools[:, :3, 3] = [[.05, 0, .01], [-.05, 0, .01]]
        p["transforms"] = {"T_world_from_camera": camera.tolist(),
            "T_world_from_arm_base": bases.tolist(),
            "T_model_tool_from_training_wrist_prior": tools.tolist()}
    if case == "no_provider":
        provider = None
    if case == "future_joint":
        clock["joint_ns"] = clock["anchor_ns"] + 1
    if case == "future_image":
        clock["image_ns"] = clock["anchor_ns"] + 1
    if case == "nonfinite_q":
        q[0] = np.nan
    if case == "bad_rgb":
        rgb = np.full((1, 3, 240, 320), 2., np.float32)
    if case == "nonidentity_normalization":
        norm["position_std"] = [2, 1, 1]
    kwargs = dict(x_rgb=rgb, assumption_profile=p, fk_provider=provider,
                  normalization=norm, anchor=clock)
    if case in {"future_joint", "future_image", "nonfinite_q", "bad_rgb", "nonidentity_normalization"}:
        with pytest.raises(ValueError):
            build_assumed_ict(q, **kwargs)
        assert provider.fk_calls == 0
        return
    before_q, before_rgb = q.copy(), rgb.copy()
    result = build_assumed_ict(q, **kwargs)
    assert_not_physical(result)
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(rgb, before_rgb)
    if case == "no_provider":
        assert result["status"] == "NO_PROVIDER"
        assert result["ict"] is result["current_input_npz"] is None
        return
    assert result["status"] == "BUILT_ASSUMED_ICT_OFFLINE_ONLY"
    assert len(result["assumption_profile_sha256"]) == 64
    ict, current = result["ict"], result["current_input_npz"]
    assert ict["x_ict"].shape == (8, 29)
    np.testing.assert_array_equal(ict["ict_mask"], [True, True, False, False, False, False, False, False])
    assert ict["grasp_status"] == "UNKNOWN" and ict["object_status"] == "UNKNOWN_MASKED"
    assert ict["coordinate_qualification"] == "ASSUMED_NOT_MEASURED"
    assert current["x_rgb"].dtype == np.float32
    for key in FALSE_FLAGS:
        assert ict[key] is False
    expected = np.repeat(np.eye(4)[None], 2, axis=0)
    expected[0, :3, 3], expected[1, :3, 3] = q[:3], q[7:10]
    if case == "transform_directions":
        expected = bases @ expected @ tools
    np.testing.assert_allclose(current["wrist_pose_world"], expected)
    relative_camera = np.linalg.inv(current["reference_camera_to_world"]) @ expected
    np.testing.assert_allclose(ict["x_ict"][:2, 1:4], relative_camera[:, :3, 3], atol=1e-7)
    np.testing.assert_allclose(ict["x_ict"][:2, 4:10], relative_camera[:, :3, :2].reshape(2, 6), atol=1e-7)


@pytest.mark.parametrize("case", ["success_50", "success_30", "transforms", "no_provider",
    "no_limits", "hand_outside_last_frame", "degenerate_rotation", "nonfinite_action",
    "joint_limit", "step", "speed", "unchecked", "collision", "residual", "rotation",
    "exception", "nonfinite_q", "initial_limit", "duplicate_ids", "backwards_time",
    "future_mapping_claim", "fractional_time", "unsupported_horizon"])
def test_bridge_retains_raw_h50_and_existing_numeric_reject_gates(case):
    p, source, raw, q = profile(), mapping(), actions(), np.zeros(14)
    provider = MathematicalProvider(case)
    horizon = 30 if case in {"success_30", "hand_outside_last_frame"} else 50
    if case == "transforms":
        camera = np.eye(4); camera[0, 3] = .01
        bases = np.repeat(np.eye(4)[None], 2, axis=0); bases[:, 0, 3] = [.005, -.005]
        tools = np.repeat(np.eye(4)[None], 2, axis=0); tools[:, 0, 3] = [.002, -.002]
        p["transforms"] = {"T_world_from_camera": camera.tolist(),
            "T_world_from_arm_base": bases.tolist(), "T_model_tool_from_training_wrist_prior": tools.tolist()}
    if case == "no_provider":
        provider = None
        del p["numerical_limits"]  # Must report NO_PROVIDER without faking success.
    if case == "no_limits":
        del p["numerical_limits"]
    if case == "hand_outside_last_frame":
        raw[-1, 29] = 1.1
    if case == "degenerate_rotation":
        raw[0, 6:12] = 0
    if case == "nonfinite_action":
        raw[0, 0] = np.nan
    if case == "speed":
        p["numerical_limits"]["joint_speed_max_rad_s"] = [.01] * 14
    if case == "initial_limit":
        q[0] = 3
    if case == "duplicate_ids":
        source["source_frame_ids"][1] = source["source_frame_ids"][0]
    if case == "backwards_time":
        source["timestamp_ns"][1] = source["timestamp_ns"][0] - 1
    if case == "future_mapping_claim":
        source["timebase_verified"] = True
    if case == "fractional_time":
        source["timestamp_ns"] = source["timestamp_ns"].astype(np.float64)
    if case == "unsupported_horizon":
        horizon = 49
    before = raw.copy()
    result = bridge_assumed_action_chunk(raw, action_mapping=source, assumption_profile=p,
        initial_q_rad=q, ik_provider=provider, normalization=NORMALIZATION, output_horizon=horizon)
    assert_not_physical(result)
    np.testing.assert_allclose(result["raw_actions"], before, equal_nan=True)
    np.testing.assert_allclose(raw, before, equal_nan=True)
    assert result["execution_mapping_status"] == "UNKNOWN_NO_CLIENT_OR_DEVICE_EXECUTION"
    if case in {"success_50", "success_30", "transforms"}:
        assert result["numerical_candidate_accepted"] is True
        assert result["status"] == "PASSED_ASSUMED_MODEL_NUMERICS_ONLY"
        assert result["candidate_actions"].shape == (horizon, 26)
        assert result["valid"].all() and provider.solve_calls == 50
        np.testing.assert_array_equal(result["output_source_frame_ids"], source["source_frame_ids"][:horizon])
        np.testing.assert_array_equal(result["output_timestamp_ns"], source["timestamp_ns"][:horizon])
        np.testing.assert_allclose(result["candidate_actions"][:, 7:13], .25)
        np.testing.assert_allclose(result["candidate_actions"][:, 20:26], .75)
        expected = raw[:, :6].reshape(50, 2, 3).copy()
        if case == "transforms":
            expected[:, :, 0] += .01 - bases[:, 0, 3] - tools[:, 0, 3]
        np.testing.assert_allclose(result["candidate_actions"][:, :3], expected[:horizon, 0])
        np.testing.assert_allclose(result["candidate_actions"][:, 13:16], expected[:horizon, 1])
    else:
        assert result["numerical_candidate_accepted"] is False
        assert result["candidate_actions"] is None and result["errors"]
        assert not result["valid"].all()
        if case == "no_provider":
            assert result["status"] == "NO_PROVIDER"
        if case in {"joint_limit", "step", "speed", "unchecked", "collision", "residual",
                    "rotation", "exception", "nonfinite_q"}:
            assert provider.solve_calls == 1  # No auto-retry after bad predecessor.
            assert result["frame_errors"][1] == ["INVALID_PREDECESSOR"]


@pytest.mark.parametrize("case", [*FALSE_FLAGS, "ack_claim", "measured_mode", "measured_assumption",
    "missing_matrix", "identity_without_named_prior", "wrong_matrix_direction",
    "nonrigid_matrix", "unsafe_provider", "missing_provider_sha", "measured_path_rejects_assumed"])
def test_assumptions_cannot_enter_measured_or_hardware_qualification(case):
    p, provider = profile(), MathematicalProvider()
    if case in FALSE_FLAGS:
        p[case] = True
    elif case == "ack_claim":
        p["device_ack_status"] = "PASSED"
    elif case == "measured_mode":
        p["mode"] = "MEASURED"
    elif case == "measured_assumption":
        p["assumptions"][0]["source_class"] = "MEASURED"
    elif case == "missing_matrix":
        del p["assumptions"][0]["default_matrix"]
    elif case == "identity_without_named_prior":
        p["assumptions"][2]["id"] = "UNDECLARED_TOOL_PRIOR_EQUALITY"
    elif case == "wrong_matrix_direction":
        p["transforms"] = {"T_camera_from_world": np.eye(4).tolist()}
    elif case == "nonrigid_matrix":
        p["assumptions"][0]["default_matrix"][0][0] = 2
    elif case == "unsafe_provider":
        provider.offline_only = False
    elif case == "missing_provider_sha":
        provider.signature = "UNKNOWN"
    elif case == "measured_path_rejects_assumed":
        with pytest.raises(DeploymentAdapterError, match="MISSING_OR_UNQUALIFIED_CALIBRATION"):
            build_measured_ict({}, x_rgb=np.zeros((1, 3, 240, 320), np.float32),
                calibration=p, fk_provider=provider, normalization=NORMALIZATION, anchor=anchor())
        assert provider.fk_calls == 0
        return
    with pytest.raises(ValueError):
        build_assumed_ict(np.zeros(14), x_rgb=np.zeros((1, 3, 240, 320), np.float32),
            assumption_profile=p, fk_provider=provider, normalization=NORMALIZATION, anchor=anchor())
    assert provider.fk_calls == 0
