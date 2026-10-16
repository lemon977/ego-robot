"""Synthetic CPU protocol/FK fixtures; no real model, IK, SDK or robot."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from chaoyang.humanego_ict.bridge import IKSolution, JOINT_NAMES
from chaoyang.humanego_ict.deployment_adapter import (
    DeploymentAdapterError, LAYOUT, SCHEDULE_SCHEMA, build_measured_ict,
    dry_run_protocol_chunk,
)
from chaoyang.humanego_ict.inference import build_hand_ict
from chaoyang.humanego_ict.retarget import CHANNEL_NAMES


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def pose(angle=0.0, translation=(0, 0, 0)):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def fixture():
    anchor_time = 1_000_000_000_000_000_000  # timestamps must never pass through float
    model_sha = sha("synthetic-robot-model")
    calibration = {"schema": "offline-pose-ik-calibration-v1", "synthetic_fixture_only": True,
        "geometry_qualified": True, "source_sha256": sha("synthetic-calibration"),
        "robot_model_sha256": model_sha, "fk_provider_signature": sha("toy-fk"),
        "ik_provider_signature": sha("toy-ik"), "joint_order": list(JOINT_NAMES),
        "position_unit": "meters", "clock_domain": "toy-native-clock",
        "camera_frame_id": "toy-camera", "world_frame_id": "toy-world", "ik_frame_id": "toy-ik",
        "predicted_tcp_frame_ids": ["toy-L-anchor", "toy-R-anchor"],
        "tool_frame_ids": ["toy-L-tool", "toy-R-tool"],
        "T_world_from_camera": pose(0.2, [1, 2, 3]), "T_ik_from_world": pose(-0.3, [-0.5, 0.8, 0.1]),
        "T_predicted_tcp_from_tool": np.stack([pose(0.1, [0.04, 0, 0.02]), pose(-0.1, [0, 0.03, 0.05])]),
        "joint_lower_rad": [-2] * 14, "joint_upper_rad": [2] * 14,
        "joint_speed_max_rad_s": [2] * 14, "joint_step_max_rad": [0.05] * 14,
        "position_tolerance_m": 1e-4, "rotation_tolerance_rad": 1e-4,
        "initial_joint_max_age_ns": 20_000_000, "camera_pose_qpc_ns": anchor_time - 2_000_000,
        "camera_pose_max_age_ns": 20_000_000, "camera_pose_source_sha256": sha("toy-camera-pose")}
    measured = {"measured": True, "synthetic_fixture_only": True, "source_sha256": sha("toy-feedback"),
        "q_valid": np.ones(14, dtype=bool), "feedback_qualification": "VALIDATED_FEEDBACK",
        "feedback_qualification_sha256": sha("toy-feedback-error-state-validation"), "timebase_verified": True,
        "timebase_sha256": sha("toy-clock-evidence"), "clock_domain": "toy-native-clock",
        "joint_unit": "radians", "joint_order": list(JOINT_NAMES), "timestamp_unit": "nanoseconds",
        "sample_origin": "SDK_JOINT_FEEDBACK", "qpc_ns": anchor_time - 1_000_000,
        "q_rad": np.linspace(-0.1, 0.1, 14)}
    anchor = {"timebase_verified": True, "synthetic_fixture_only": True,
        "source_sha256": sha("toy-causal-RGB"), "timebase_sha256": sha("toy-clock-evidence"),
        "clock_domain": "toy-native-clock", "timestamp_unit": "nanoseconds",
        "camera_frame_id": "toy-camera", "time_mode": "CAUSAL",
        "image_preprocessing": "RGB_RESIZE_BILINEAR_320_240", "source_frame_id": 71,
        "anchor_qpc_ns": anchor_time, "image_qpc_ns": anchor_time - 1_000_000}
    normalization = {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]}
    rgb = np.zeros((1, 3, 240, 320), dtype=np.uint8)
    return calibration, measured, anchor, normalization, rgb


class ToyFK:
    offline_only = True
    signature = sha("toy-fk")
    robot_model_sha256 = sha("synthetic-robot-model")
    ik_frame_id = "toy-ik"
    tool_frame_ids = ("toy-L-tool", "toy-R-tool")
    joint_order = JOINT_NAMES

    def __init__(self):
        self.calls = []

    def fk(self, q_rad):
        self.calls.append(q_rad.copy())
        return np.stack([pose(0.4, [0.2, 0.1, 0.3]), pose(-0.2, [-0.2, 0.3, 0.4])])


class ToyIK:
    """A target echo oracle only, not kinematics or collision certification."""
    offline_only = True
    signature = sha("toy-ik")
    robot_model_sha256 = sha("synthetic-robot-model")
    ik_frame_id = "toy-ik"
    tool_frame_ids = ("toy-L-tool", "toy-R-tool")
    joint_order = JOINT_NAMES

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def solve(self, target_poses, seed_q_rad, dt_s):
        self.target = target_poses.copy()
        self.calls.append(dt_s)
        q = seed_q_rad + (0.01 if self.fail == "large_safe_native_step" else 0.001)
        if self.fail == "joint_limit":
            q[0] = 5
        if self.fail == "speed":
            q[0] += 0.5
        return IKSolution(q, self.fail != "unchecked", self.fail != "collision")

    def fk(self, q_rad):
        result = self.target.copy()
        if self.fail == "residual":
            result[0, 0, 3] += 0.1
        return result


def measured_build(**changes):
    calibration, measured, anchor, normalization, rgb = fixture()
    calibration.update(changes.pop("calibration", {}))
    measured.update(changes.pop("measured", {}))
    anchor.update(changes.pop("anchor", {}))
    return build_measured_ict(measured, x_rgb=changes.pop("x_rgb", rgb), calibration=calibration,
        fk_provider=changes.pop("fk_provider", ToyFK()), normalization=changes.pop("normalization", normalization), anchor=anchor,
        allow_synthetic_fixture=changes.pop("allow_synthetic_fixture", True), **changes)


def protocol_fixture():
    calibration, measured, anchor, _, _ = fixture()
    actions = np.zeros((50, 30))
    actions[:, :6] = [0.1, 0.2, 0.3, -0.1, 0.3, 0.4]
    actions[:, 6:12] = pose(0.4)[:3, :2].reshape(6)
    actions[:, 12:18] = pose(-0.2)[:3, :2].reshape(6)
    actions[:, 18:30] = np.linspace(0.1, 0.9, 12)
    anchor_time = anchor["anchor_qpc_ns"]
    # This explicit synthetic 30Hz schedule is supplied, never generated by production adapter.
    times = anchor_time + np.arange(1, 51, dtype=np.int64) * 1_000_000_000 // 30
    contract = {"schema": "humanego-camera-revo2-actions-v1", "synthetic_fixture_only": True,
        "source_signature": sha("toy-predictions"), "source_frame_ids": np.arange(50) * 2 + 73,
        "qpc_ns": times, "anchor_qpc_ns": anchor_time, "clock_domain": anchor["clock_domain"],
        "camera_frame_id": "toy-camera", "tcp_frame_ids": calibration["predicted_tcp_frame_ids"],
        "position_mean": [0, 0, 0], "position_std": [1, 1, 1],
        "channel_order": list(CHANNEL_NAMES), "hand_command_unit": "NORMALIZED_0_1"}
    metadata = {"protocol_version": 2, "policy_id": "toy-humanego-not-pi05-policy", "task_id": "toy-task",
        "checkpoint": "synthetic-tensor-candidate", "config": "synthetic-ict-revo2", "default_prompt": "toy-task",
        "norm_sha256": sha("toy-norm"), "action_dim": 26, "action_horizon": 30,
        "layout": LAYOUT, "action_semantics_id": "absolute_joint_targets", "control_hz": 30}
    schedule = {"schema": SCHEDULE_SCHEMA, "synthetic_fixture_only": True, "status": "PASSED",
        "timebase_verified": True, "mapping_kind": "FROZEN_NATIVE_H50_PREDICTION_MAPPING",
        "source_sha256": sha("toy-native-prediction-schedule"), "timebase_sha256": sha("toy-clock-evidence"),
        "timestamp_unit": "nanoseconds", "clock_domain": anchor["clock_domain"], "anchor_qpc_ns": anchor_time,
        "source_signature": contract["source_signature"], "source_frame_ids": contract["source_frame_ids"].copy(),
        "qpc_ns": times.copy(), "policy_id": metadata["policy_id"],
        "legacy_fixed_rate_compatible": True, "control_hz": 30}
    return actions, contract, calibration, measured, schedule, metadata


def protocol_build(**changes):
    actions, contract, calibration, initial, schedule, metadata = protocol_fixture()
    for obj, key in ((contract, "contract"), (calibration, "calibration"), (initial, "initial"),
                     (schedule, "schedule"), (metadata, "metadata")):
        obj.update(changes.pop(key, {}))
    # Explicit toy client cropping proof, not an asserted default of the live client.
    rows = np.arange(8, 30)
    execution = {"schema": "humanego-client-execution-timing-v1", "synthetic_fixture_only": True,
        "status": "PASSED", "source_sha256": sha("toy-client-execution-receipt"),
        "runtime_source_sha256": sha("toy-client-runtime"), "runtime_configuration_sha256": sha("toy-client-config"),
        "runtime_mode": "SYNTHETIC_EXPLICIT_CROP", "smoothing_mode": "SYNTHETIC_NO_VALUE_CHANGE",
        "latency_k": 8, "action_processing": "INDEX_ONLY_NO_VALUE_CHANGE",
        "request_context": {"sequence": 9}, "policy_id": metadata["policy_id"],
        "clock_domain": contract["clock_domain"], "anchor_qpc_ns": contract["anchor_qpc_ns"],
        "source_signature": contract["source_signature"], "timestamp_unit": "nanoseconds", "timebase_verified": True,
        "control_hz": metadata["control_hz"], "response_row_indices": rows,
        "publish_qpc_ns": contract["qpc_ns"][rows].copy(), "first_publish_qpc_ns": int(contract["qpc_ns"][8])}
    execution.update(changes.pop("execution", {}))
    return dry_run_protocol_chunk(changes.pop("actions", actions), contract=contract, calibration=calibration,
        measured_initial=initial, ik_provider=changes.pop("ik_provider", ToyIK()),
        schedule=changes.pop("replace_schedule", schedule), policy_metadata=metadata,
        request_context=changes.pop("request_context", {"sequence": 9}),
        client_execution=changes.pop("client_execution", execution),
        allow_synthetic_fixture=changes.pop("allow_synthetic_fixture", True), **changes)


def test_measured_fk_transforms_match_existing_ict_and_training_rot6():
    calibration, measured, anchor, norm, rgb = fixture()
    provider = ToyFK()
    built = build_measured_ict(measured, x_rgb=rgb, calibration=calibration, fk_provider=provider,
        normalization=norm, anchor=anchor, allow_synthetic_fixture=True)
    tool = ToyFK().fk(measured["q_rad"])
    expected = np.linalg.inv(calibration["T_ik_from_world"]) @ tool @ np.linalg.inv(calibration["T_predicted_tcp_from_tool"])
    data = built["current_input_npz"]
    assert np.allclose(data["wrist_pose_world"], expected)
    assert len(provider.calls) == 1 and np.array_equal(provider.calls[0], measured["q_rad"])
    direct = build_hand_ict(expected, np.ones(2, dtype=bool), calibration["T_world_from_camera"],
        normalization=norm, source_qpc_ns=np.array([measured["qpc_ns"]] * 2), anchor_qpc_ns=anchor["anchor_qpc_ns"])
    assert np.array_equal(built["ict"]["x_ict"], direct["x_ict"])
    assert built["ict"]["ict_mask"].tolist() == [True, True, False, False, False, False, False, False]
    assert built["ict"]["x_ict"][:2, -1].tolist() == [-1, -1]
    assert built["receipt_fields"]["current_state_type"] == "MEASURED_JOINTS_FK_DERIVED"
    assert built["receipt_fields"]["input_npz_sha256"] is None
    assert not built["physical_deployment_authorized"] and built["robot_commands_sent"] == 0
    assert set(data) == {"x_rgb", "wrist_pose_world", "wrist_valid", "reference_camera_to_world", "source_qpc_ns", "anchor_qpc_ns", "source_frame_id"}
    assert int(data["anchor_qpc_ns"]) == anchor["anchor_qpc_ns"]


@pytest.mark.parametrize("changes,error", [
    ({"allow_synthetic_fixture": False}, "SYNTHETIC_FIXTURE_REJECTED"),
    ({"measured": {"measured": False}}, "MEASURED_JOINT_FEEDBACK_REQUIRED"),
    ({"measured": {"sample_origin": "COMMAND_TARGET"}}, "FEEDBACK_QUALIFICATION_REQUIRED"),
    ({"measured": {"q_valid": np.zeros(14, dtype=bool)}}, "FEEDBACK_QUALIFICATION_REQUIRED"),
    ({"measured": {"q_valid": np.ones(14)}}, "FEEDBACK_QUALIFICATION_REQUIRED"),
    ({"measured": {"feedback_qualification": "UNKNOWN"}}, "FEEDBACK_QUALIFICATION_REQUIRED"),
    ({"measured": {"joint_unit": "degrees"}}, "UNIT_OR_SOURCE_MISMATCH"),
    ({"measured": {"joint_order": list(reversed(JOINT_NAMES))}}, "ORDER_UNIT_OR_SOURCE_MISMATCH"),
    ({"measured": {"source_sha256": "UNKNOWN"}}, "INVALID_SHA"),
    ({"measured": {"clock_domain": "different"}}, "CLOCK_DOMAIN_MISMATCH"),
    ({"measured": {"qpc_ns": 1_000_000_000_000_000_001}}, "STALE_OR_FUTURE"),
    ({"measured": {"qpc_ns": 0}}, "STALE_OR_FUTURE"),
    ({"measured": {"qpc_ns": 1.0}}, "INVALID_NATIVE_INTEGER"),
    ({"measured": {"q_rad": [np.nan] * 14}}, "INVALID_FINITE_SHAPE"),
    ({"measured": {"q_rad": [3] * 14}}, "MEASURED_JOINT_LIMIT"),
    ({"measured": {"q_rad": [0] * 26}}, "INVALID_FINITE_SHAPE"),
    ({"anchor": {"timebase_verified": False}}, "TIMEBASE_UNVERIFIED"),
    ({"anchor": {"camera_frame_id": "wrong"}}, "PREPROCESSING_MISMATCH"),
    ({"anchor": {"image_qpc_ns": 1_000_000_000_000_000_001}}, "STALE_OR_FUTURE"),
    ({"calibration": {"geometry_qualified": False}}, "UNQUALIFIED_CALIBRATION"),
    ({"calibration": {"T_predicted_tcp_from_tool": None}}, "INVALID_FINITE_SHAPE"),
    ({"calibration": {"camera_pose_qpc_ns": 0}}, "CAMERA_POSE_TIME_STALE_OR_FUTURE"),
    ({"calibration": {"position_unit": "millimeters"}}, "POSITION_UNIT_MISMATCH"),
    ({"fk_provider": None}, "MISMATCHED_FROZEN_FK_PROVIDER"),
    ({"x_rgb": np.zeros((1, 3, 224, 224))}, "CURRENT_RGB_REQUIRES"),
    ({"x_rgb": np.full((1, 3, 240, 320), 2.0)}, "CURRENT_RGB_RANGE"),
    ({"normalization": {"method": "GUESS_STANDARD_SCALE"}}, "CURRENT_ICT_NORMALIZATION_CONTRACT_MISMATCH"),
])
def test_measured_state_fails_closed_before_provider(changes, error):
    provider = ToyFK()
    changes.setdefault("fk_provider", provider)
    with pytest.raises(DeploymentAdapterError, match=error):
        measured_build(**changes)
    assert provider.calls == []


def test_invalid_fk_output_and_frozen_model_mismatch():
    class BadFK(ToyFK):
        def fk(self, q):
            return np.zeros((2, 4, 4))
    with pytest.raises(DeploymentAdapterError, match="INVALID_SE3"):
        measured_build(fk_provider=BadFK())
    p = ToyFK()
    p.robot_model_sha256 = sha("different-model")
    with pytest.raises(DeploymentAdapterError, match="MISMATCHED_FROZEN_FK_PROVIDER"):
        measured_build(fk_provider=p)


def test_h50_to_h30_is_time_prefix_after_ik_not_channel_truncation():
    provider = ToyIK()
    result = protocol_build(ik_provider=provider)
    assert result["accepted"] and result["response"]["actions"].shape == (30, 26)
    assert len(provider.calls) == 50 and all(dt > 0 for dt in provider.calls)
    actions, contract, _, initial, _, metadata = protocol_fixture()
    output = result["response"]["actions"]
    assert np.allclose(output[:, 7:13], actions[:30, 18:24])
    assert np.allclose(output[:, 20:26], actions[:30, 24:30])
    assert np.allclose(output[0, :7], initial["q_rad"][:7] + 0.001)
    assert np.allclose(output[0, 13:20], initial["q_rad"][7:] + 0.001)
    assert np.array_equal(result["input_qpc_ns"], contract["qpc_ns"])
    assert np.array_equal(result["output_qpc_ns"], contract["qpc_ns"][:30])
    assert len(result["input_source_frame_ids"]) == 50 and result["omitted_suffix_frames"] == 20
    assert set(result["response"]) == {"actions", "policy_id", "request_context"}
    assert result["response"]["request_context"] == {"sequence": 9}
    assert result["response"]["policy_id"] == metadata["policy_id"]
    assert result["client_execution"]["first_publish_qpc_ns"] == int(contract["qpc_ns"][8])
    assert np.array_equal(result["client_selected_qpc_ns"], contract["qpc_ns"][8:30])
    assert result["selected_path_collision_qualification"] == "UNKNOWN_REQUIRES_SELECTED_EDGE_REVALIDATION"
    assert not result["physical_deployment_authorized"] and result["robot_commands_sent"] == 0


@pytest.mark.parametrize("changes,error", [
    ({"replace_schedule": None}, "UNKNOWN_REQUIRES_NATIVE_H50_MAPPING"),
    ({"schedule": {"timebase_verified": False}}, "UNKNOWN_REQUIRES_NATIVE_H50_MAPPING"),
    ({"schedule": {"source_sha256": "ABSENT"}}, "INVALID_SHA"),
    ({"schedule": {"clock_domain": "ros-clock"}}, "SCHEDULE_CLOCK_SOURCE_POLICY_OR_RATE_MISMATCH"),
    ({"schedule": {"policy_id": "pi05-other-checkpoint"}}, "SCHEDULE_CLOCK_SOURCE_POLICY_OR_RATE_MISMATCH"),
    ({"schedule": {"legacy_fixed_rate_compatible": False}}, "SCHEDULE_CLOCK_SOURCE_POLICY_OR_RATE_MISMATCH"),
    ({"schedule": {"source_frame_ids": np.arange(50)}}, "SCHEDULE_NATIVE_H50_MAPPING_MISMATCH"),
    ({"metadata": {"action_dim": 30}}, "26D_H30_ABSOLUTE_CONTRACT_MISMATCH"),
    ({"metadata": {"action_horizon": 50}}, "26D_H30_ABSOLUTE_CONTRACT_MISMATCH"),
    ({"metadata": {"layout": "right13,left13"}}, "26D_H30_ABSOLUTE_CONTRACT_MISMATCH"),
    ({"metadata": {"action_semantics_id": "delta_joint_targets"}}, "26D_H30_ABSOLUTE_CONTRACT_MISMATCH"),
    ({"metadata": {"protocol_version": 1}}, "V2_POLICY_METADATA_REQUIRED"),
    ({"metadata": {"action_horizon": 30.0}}, "26D_H30_ABSOLUTE_CONTRACT_MISMATCH"),
    ({"metadata": {"control_hz": 0}}, "EXPLICIT_PROTOCOL_CONTROL_HZ_REQUIRED"),
    ({"request_context": {"request_seq": 9}}, "EXACT_V2_REQUEST_CONTEXT_SEQUENCE_REQUIRED"),
    ({"request_context": {"sequence": True}}, "EXACT_V2_REQUEST_CONTEXT_SEQUENCE_REQUIRED"),
    ({"request_context": {"sequence": 0}}, "EXACT_V2_REQUEST_CONTEXT_SEQUENCE_REQUIRED"),
    ({"allow_synthetic_fixture": False}, "SYNTHETIC_FIXTURE_REJECTED"),
    ({"initial": {"q_valid": np.zeros(14, dtype=bool)}}, "FEEDBACK_QUALIFICATION_REQUIRED"),
    ({"client_execution": None}, "CLIENT_EXECUTION_TIMING_UNQUALIFIED"),
    ({"execution": {"runtime_source_sha256": "UNKNOWN"}}, "INVALID_SHA"),
    ({"execution": {"request_context": {"sequence": 8}}}, "CLIENT_EXECUTION_CLOCK_REQUEST_OR_POLICY_MISMATCH"),
    ({"execution": {"clock_domain": "different"}}, "CLIENT_EXECUTION_CLOCK_REQUEST_OR_POLICY_MISMATCH"),
    ({"execution": {"smoothing_mode": "UNKNOWN"}}, "MISSING_DECLARATION"),
    ({"execution": {"action_processing": "KAIDEX_MIXED_AND_CLIPPED"}}, "CLIENT_POSTPROCESSING_UNQUALIFIED"),
    ({"execution": {"response_row_indices": np.arange(8, 31)}}, "INVALID_CLIENT_EXECUTION_INDEX_TIME_MAPPING"),
    ({"execution": {"response_row_indices": np.full(22, 8)}}, "INVALID_CLIENT_EXECUTION_INDEX_TIME_MAPPING"),
    ({"execution": {"first_publish_qpc_ns": 1_000_000_000_000_000_001}}, "CLIENT_EXECUTION_NATIVE_PREDICTION_TIME_MISMATCH"),
])
def test_protocol_prechecks_reject_before_any_ik(changes, error):
    provider = ToyIK()
    result = protocol_build(ik_provider=provider, **changes)
    assert not result["accepted"] and result["response"] is None
    assert any(error in e for e in result["errors"]), result["errors"]
    assert provider.calls == []


@pytest.mark.parametrize("failure", ["collision", "unchecked", "speed", "joint_limit", "residual"])
def test_ik_errors_never_leak_partial_response(failure):
    result = protocol_build(ik_provider=ToyIK(fail=failure))
    assert not result["accepted"] and result["response"] is None
    assert result["errors"] and result["bridge"]["actions"] is None


def test_fixed_rate_protocol_cannot_silently_change_native_irregular_times():
    actions, contract, calibration, initial, schedule, metadata = protocol_fixture()
    contract["qpc_ns"] = contract["qpc_ns"].copy()
    contract["qpc_ns"][20:] += 4_000_000
    schedule["qpc_ns"] = contract["qpc_ns"].copy()
    provider = ToyIK()
    result = dry_run_protocol_chunk(actions, contract=contract, calibration=calibration,
        measured_initial=initial, ik_provider=provider, schedule=schedule, policy_metadata=metadata,
        request_context={"sequence": 9}, allow_synthetic_fixture=True)
    assert not result["accepted"] and provider.calls == []
    assert result["errors"] == ["NATIVE_SCHEDULE_NOT_COMPATIBLE_WITH_FIXED_RATE_CLIENT"]
    assert np.array_equal(schedule["qpc_ns"], contract["qpc_ns"])


def test_same_shape_wrong_ik_model_and_bad_hand_units_rejected():
    p = ToyIK()
    p.robot_model_sha256 = sha("unfrozen-other-model")
    assert protocol_build(ik_provider=p)["errors"] == ["IK_ROBOT_MODEL_OR_JOINT_ORDER_MISMATCH"]
    result = protocol_build(contract={"hand_command_unit": "SDK_RAW_0_1000"})
    assert not result["accepted"] and result["response"] is None
    assert "ACTION_FRAME_UNIT_OR_CHANNEL_MISMATCH" in result["errors"]


def test_native_ik_steps_legal_but_client_cropped_first_command_unsafe():
    result = protocol_build(ik_provider=ToyIK(fail="large_safe_native_step"))
    assert result["bridge"]["accepted"]  # fifty native 0.01rad steps are legal
    assert not result["accepted"] and result["response"] is None
    assert result["errors"] == ["CLIENT_EXECUTION_STEP_OR_SPEED_LIMIT"]  # actual first command jumps 0.09rad


def test_original_prefix_retains_only_native_collision_checks_not_deploy_authority():
    _, contract, _, _, _, _ = protocol_fixture()
    result = protocol_build(execution={"response_row_indices": np.arange(30),
        "publish_qpc_ns": contract["qpc_ns"][:30].copy(), "first_publish_qpc_ns": int(contract["qpc_ns"][0]),
        "latency_k": 0})
    assert result["accepted"]
    assert result["selected_path_collision_qualification"] == "ORIGINAL_NATIVE_EDGE_CHECKS_ONLY"
    assert not result["physical_deployment_authorized"]


def test_import_has_no_torch_sdk_ros_or_network_dependency():
    project = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(project / "src"))
    script = "import sys; import chaoyang.humanego_ict.deployment_adapter; forbidden=('torch','rclpy','bc_stark_sdk','websockets','openpi'); assert not any(n==x or n.startswith(x+'.') for n in sys.modules for x in forbidden)"
    result = subprocess.run([sys.executable, "-c", script], cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
