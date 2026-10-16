"""Explicit synthetic CPU fixtures, not a robot, SDK, physical calibration or IK test."""
from copy import deepcopy
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from chaoyang.humanego_ict import bridge, retarget
from chaoyang.humanego_ict.bridge import (
    BridgeError, IKSolution, JOINT_NAMES, bridge_action_chunk, decode_actions,
)
from chaoyang.humanego_ict.retarget import (
    CHANNEL_NAMES, FEATURE_NAMES, MOTOR_MAX, REFERENCE_RETARGET_SHA256, RetargetError,
    denormalize_motor, extract_hand_features, normalize_motor, retarget_features, retarget_hand,
)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_hand_contract(joint_count=25):
    """Source-declared toy topology and new mapping; never a real MANUS index claim."""
    if joint_count < 23:
        raise ValueError("toy fixture needs at least 23 points")
    opened = {name: 0.0 for name in FEATURE_NAMES}
    closed = {name: float(np.pi) for name in FEATURE_NAMES}
    opened.update(thumb_rotation=-float(np.pi), thumb_full=-1.0)
    closed.update(thumb_rotation=float(np.pi), thumb_full=1.0)
    return {
        "schema": "derived-revo2-hand-v1", "synthetic_fixture_only": True,
        "supervision_kind": "DERIVED_RETARGET", "hand_command_unit": "NORMALIZED_0_1",
        "channel_order": list(CHANNEL_NAMES), "reference_retarget_sha256": REFERENCE_RETARGET_SHA256,
        "feature_provider": {"kind": "explicit_joint_angles_v1", "source_sha256": _hash("synthetic-angle-provider")},
        "layout": {"joint_count": joint_count, "coordinate_frame": "wrist_local", "position_unit": "meters",
            "source_sha256": _hash("synthetic-layout-not-MANUS25"),
            "finger_triplets": {"thumb": [[1, 2, 3]], "index": [[5, 6, 7]], "middle": [[10, 11, 12]],
                                "ring": [[15, 16, 17]], "pinky": [[20, 21, 22]]},
            "thumb_palm_indices": {"palm_root": 0, "index_mcp": 5, "middle_mcp": 10, "pinky_mcp": 20,
                                   "thumb_mcp": 1, "thumb_ip": 2, "thumb_tip": 3},
            "thumb_rotation_side_sign": {"left": 1, "right": -1},
            "palm_depth_side_sign": {"left": -1, "right": 1}},
        "mapping": {"kind": "DERIVED_AFFINE_THUMB_GATED_V1", "source_sha256": _hash("synthetic-mapping"),
            "calibration": {side: {"open": deepcopy(opened), "closed": deepcopy(closed)} for side in ("left", "right")},
            "gains": dict.fromkeys(FEATURE_NAMES, 1.0), "thumb_rotation_enabled": True,
            "thumb_opposition_strong": 0.0, "thumb_full_gate": 0.0, "thumb_opposition_deadband": 0.0,
            "soft_margin": [[0, 0] for _ in CHANNEL_NAMES]},
    }


def synthetic_joints(joint_count=25, curl=0.7):
    """Toy wrist-local points; unused indices are deliberately not given meaning."""
    points = np.zeros((joint_count, 3))
    points[1:4] = [[0.06, 0.0, 0.0], [0.08, 0.02, 0.01], [0.10, 0.03, 0.02]]
    for start, x, y in ((5, 0.05, 0.06), (10, 0.0, 0.07), (15, -0.02, 0.06), (20, -0.05, 0.04)):
        points[start] = [x, y, 0]
        points[start + 1] = points[start] + [0, 0.03, 0]
        points[start + 2] = points[start + 1] + [0, 0.02 * np.cos(curl), 0.02 * np.sin(curl)]
    return points


def _rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def synthetic_bridge_fixture():
    action = np.zeros((50, 30))
    action[:, :6] = [0.1, 0.2, 0.3, -0.1, 0.4, 0.2]
    action[:, 6:12] = _rotation("z", 0.4)[:, :2].reshape(6)
    action[:, 12:18] = _rotation("y", -0.2)[:, :2].reshape(6)
    action[:, 18:30] = np.linspace(0.1, 0.9, 12)
    anchor = 1_000_000_000_000_000_000  # above IEEE754 exact-int range
    timestamps = anchor + np.cumsum(20_000_000 + np.arange(50) % 3 * 5_000_000)
    contract = {"schema": "humanego-camera-revo2-actions-v1", "synthetic_fixture_only": True,
        "source_signature": _hash("synthetic-actions"), "source_frame_ids": np.arange(50) * 2 + 11,
        "qpc_ns": timestamps, "anchor_qpc_ns": anchor, "clock_domain": "synthetic-qpc",
        "camera_frame_id": "toy-camera", "tcp_frame_ids": ["toy-L-tcp", "toy-R-tcp"],
        "position_mean": [0, 0, 0], "position_std": [1, 1, 1],
        "channel_order": list(CHANNEL_NAMES), "hand_command_unit": "NORMALIZED_0_1"}
    camera_world, world_ik = np.eye(4), np.eye(4)
    camera_world[:3, :3], camera_world[:3, 3] = _rotation("z", 0.3), [1, 2, 3]
    world_ik[:3, :3], world_ik[:3, 3] = _rotation("y", -0.25), [-0.4, 0.5, 0.6]
    offsets = np.stack((np.eye(4), np.eye(4)))
    offsets[0, :3, 3], offsets[1, :3, 3] = [0.04, 0, 0.03], [0, -0.03, 0.05]
    calibration = {"schema": "offline-pose-ik-calibration-v1", "synthetic_fixture_only": True,
        "geometry_qualified": True, "source_sha256": _hash("synthetic-not-physical-calibration"),
        "ik_provider_signature": _hash("synthetic-not-real-IK"),
        "camera_frame_id": "toy-camera", "world_frame_id": "toy-world", "ik_frame_id": "toy-ik-world",
        "predicted_tcp_frame_ids": contract["tcp_frame_ids"], "tool_frame_ids": ["toy-L-tool", "toy-R-tool"],
        "joint_order": list(JOINT_NAMES), "T_world_from_camera": camera_world,
        "T_ik_from_world": world_ik, "T_predicted_tcp_from_tool": offsets,
        "joint_lower_rad": [-2] * 14, "joint_upper_rad": [2] * 14,
        "joint_speed_max_rad_s": [2] * 14, "joint_step_max_rad": [0.05] * 14,
        "position_tolerance_m": 1e-4, "rotation_tolerance_rad": 1e-4, "initial_joint_max_age_ns": 10_000_000}
    initial = {"synthetic_fixture_only": True, "measured": True, "source_sha256": _hash("synthetic-q-measurement"),
        "clock_domain": "synthetic-qpc", "joint_unit": "radians", "joint_order": list(JOINT_NAMES),
        "qpc_ns": anchor - 1_000_000, "q_rad": np.zeros(14)}
    return action, contract, calibration, initial


class FakeIK:
    """A protocol oracle recording supplied targets, not a real kinematic solver."""
    offline_only = True
    signature = _hash("synthetic-not-real-IK")
    ik_frame_id = "toy-ik-world"
    tool_frame_ids = ("toy-L-tool", "toy-R-tool")

    def __init__(self, mode=None):
        self.mode, self.calls = mode, []

    def solve(self, target_poses, seed_q_rad, dt_s):
        self.target = target_poses.copy()
        self.calls.append((target_poses.copy(), seed_q_rad.copy(), dt_s))
        q = seed_q_rad + (0.3 if self.mode == "speed" else 0.001)
        if self.mode == "limit":
            q[0] = 3
        if self.mode == "nan":
            q[0] = np.nan
        return IKSolution(q, self.mode != "unchecked", self.mode != "collision")

    def fk(self, q_rad):
        pose = self.target.copy()
        if self.mode == "residual":
            pose[0, 0, 3] += 0.1
        if self.mode == "bad_rotation":
            pose[0, :3, :3] *= -1
        return pose


def _bridge(action=None, provider=None, output_horizon=None, **changes):
    original, contract, calibration, initial = synthetic_bridge_fixture()
    contract.update(changes.get("contract", {}))
    calibration = changes.get("calibration", calibration)
    initial.update(changes.get("measured_initial", {}))
    return bridge_action_chunk(original if action is None else action, contract=contract,
        calibration=calibration, measured_initial=initial, ik_provider=provider or FakeIK(),
        output_horizon=output_horizon, allow_synthetic_fixture=True)


def test_six_channel_conversion_roundtrip_and_round_to_even():
    raw = np.array([[0, 1, 2, 500, 999, 1000], [590, 1000, 999, 500, 2, 0]])
    assert np.array_equal(denormalize_motor(normalize_motor(raw)), raw)
    assert np.array_equal(denormalize_motor(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) / MOTOR_MAX), np.zeros(6))
    for bad in (np.zeros(5), [np.nan] * 6, [-1] * 6, [2] * 6):
        with pytest.raises(RetargetError):
            denormalize_motor(bad)
    with pytest.raises(RetargetError):
        normalize_motor([591, 0, 0, 0, 0, 0])
    assert normalize_motor([591, 0, 0, 0, 0, 0], clip=True)[0] == 1


def test_actual_human_features_drive_six_channel_derived_targets():
    contract = synthetic_hand_contract()
    more_open, more_closed = synthetic_joints(curl=0.1), synthetic_joints(curl=1.2)
    first = retarget_hand(more_open, side="left", hand_contract=contract, allow_synthetic_fixture=True)
    second = retarget_hand(more_closed, side="left", hand_contract=contract, allow_synthetic_fixture=True)
    assert first.valid.all() and second.valid.all()
    assert np.all(second.raw_motor[2:] > first.raw_motor[2:])
    assert second.raw_motor[2] == round(1.2 / np.pi * 1000)
    assert second.supervision_kind == "DERIVED_RETARGET" and not second.control_ground_truth
    assert second.to_dict()["diagnostics"]["legacy_complete_parity"] is False
    assert not second.physical_deployment_authorized


def test_explicit_side_sign_and_units_not_reordered_sdk_arrays():
    contract = synthetic_hand_contract()
    points = synthetic_joints()
    left = extract_hand_features(points, side="left", hand_contract=contract, allow_synthetic_fixture=True)
    right = extract_hand_features(points, side="right", hand_contract=contract, allow_synthetic_fixture=True)
    assert left["thumb_rotation"] == pytest.approx(-right["thumb_rotation"])
    assert left["thumb_full"] == pytest.approx(-right["thumb_full"])
    millimeters = deepcopy(contract)
    millimeters["layout"]["position_unit"] = "millimeters"
    converted = retarget_hand(points * 1000, side="left", hand_contract=millimeters, allow_synthetic_fixture=True)
    expected = retarget_hand(points, side="left", hand_contract=contract, allow_synthetic_fixture=True)
    assert np.array_equal(converted.raw_motor, expected.raw_motor)


def test_explicit_thumb_gate_and_software_soft_margins():
    contract = synthetic_hand_contract()
    contract["mapping"].update(thumb_opposition_strong=700, thumb_full_gate=700, thumb_opposition_deadband=50,
                                soft_margin=[[25, 25], [30, 30], *([[15, 15]] * 4)])
    features = dict(contract["mapping"]["calibration"]["left"]["open"])
    features["thumb_rotation"] = 0.0
    result = retarget_features(features, side="left", hand_contract=contract, allow_synthetic_fixture=True)
    assert result.raw_motor.tolist() == [25, 30, 15, 15, 15, 15]
    assert result.diagnostics["soft_clipped"] == [True] * 6


@pytest.mark.parametrize("missing", ["layout", "mapping", "feature_provider"])
def test_missing_source_contract_stays_unknown(missing):
    contract = synthetic_hand_contract()
    contract.pop(missing)
    result = retarget_hand(synthetic_joints(), side="left", hand_contract=contract, allow_synthetic_fixture=True)
    assert not result.valid.any() and np.isnan(result.normalized).all() and result.missing_items
    assert result.to_dict()["normalized"] == [None] * 6


def test_no_guessed_layout_or_calibration_and_fixture_gate():
    contract = synthetic_hand_contract()
    assert not retarget_hand(synthetic_joints(), side="left", hand_contract=contract).valid.any()
    contract["layout"]["finger_triplets"].pop("index")
    assert retarget_hand(synthetic_joints(), side="left", hand_contract=contract, allow_synthetic_fixture=True).status == "BLOCKED_SOURCE_HAND_LAYOUT"
    contract = synthetic_hand_contract()
    contract["mapping"]["calibration"]["left"]["closed"]["index"] = 0
    assert retarget_hand(synthetic_joints(), side="left", hand_contract=contract, allow_synthetic_fixture=True).status == "BLOCKED_RETARGET_CONTRACT"
    assert retarget_hand(np.zeros((25, 3)), side="left", hand_contract=synthetic_hand_contract(), allow_synthetic_fixture=True).status == "BLOCKED_HAND_GEOMETRY"


def test_source_bound_injected_feature_provider():
    contract = synthetic_hand_contract()
    class Provider:
        offline_only = True
        source_sha256 = contract["feature_provider"]["source_sha256"]
        def extract(self, joints, side, layout):
            return extract_hand_features(joints, side=side, hand_contract=contract, allow_synthetic_fixture=True)
    result = retarget_hand(synthetic_joints(), side="left", hand_contract=contract,
                           feature_provider=Provider(), allow_synthetic_fixture=True)
    assert result.valid.all()
    Provider.source_sha256 = _hash("wrong-source")
    assert not retarget_hand(synthetic_joints(), side="left", hand_contract=contract,
                             feature_provider=Provider(), allow_synthetic_fixture=True).valid.any()


def test_row_interleaved_rotation_and_position_normalization():
    action, *_ = synthetic_bridge_fixture()
    poses, hands = decode_actions(action, position_mean=[1, 2, 3], position_std=[2, 3, 4])
    assert np.allclose(poses[0, 0, :3, :3], _rotation("z", 0.4))
    assert np.allclose(poses[0, 1, :3, :3], _rotation("y", -0.2))
    assert np.allclose(poses[0, 0, :3, 3], [1.2, 2.6, 4.2])
    assert np.array_equal(hands[0].reshape(12), action[0, 18:30])


def test_camera_world_tool_composition_measured_seed_and_horizon_prefix():
    action, contract, calibration, initial = synthetic_bridge_fixture()
    provider = FakeIK()
    result = _bridge(provider=provider, output_horizon=30)
    poses, _ = decode_actions(action, position_mean=[0, 0, 0], position_std=[1, 1, 1])
    expected = calibration["T_ik_from_world"] @ calibration["T_world_from_camera"] @ poses[0] @ calibration["T_predicted_tcp_from_tool"]
    assert np.allclose(provider.calls[0][0], expected)
    assert np.array_equal(provider.calls[0][1], initial["q_rad"])
    assert np.allclose(provider.calls[1][1], initial["q_rad"] + 0.001)
    assert result.accepted and result.actions.shape == (30, 26) and result.valid.all()
    assert np.array_equal(result.actions[:, 7:13], action[:30, 18:24])
    assert np.array_equal(result.actions[:, 20:26], action[:30, 24:30])
    assert np.array_equal(result.source_frame_ids, contract["source_frame_ids"])
    assert np.array_equal(result.qpc_ns, contract["qpc_ns"])
    serialized = result.to_dict()
    assert len(serialized["input_source_frame_ids"]) == len(serialized["input_qpc_ns"]) == 50
    assert len(serialized["actions"]) == len(serialized["output_source_frame_ids"]) == len(serialized["output_qpc_ns"]) == 30
    assert serialized["output_source_frame_ids"] == serialized["input_source_frame_ids"][:30]
    assert serialized["output_qpc_ns"] == serialized["input_qpc_ns"][:30]
    assert len(set(row[2] for row in provider.calls[1:])) == 3  # actual nonuniform dt, not i/30
    assert result.to_dict()["offline_only"] and not result.physical_deployment_authorized
    assert _bridge().actions.shape == (50, 26)


@pytest.mark.parametrize("mode,reason", [("collision", "COLLISION"), ("unchecked", "COLLISION"),
    ("limit", "JOINT_LIMIT"), ("speed", "SPEED_LIMIT"), ("nan", "INVALID_ARRAY"),
    ("residual", "FK_RESIDUAL"), ("bad_rotation", "INVALID_ROTATION")])
def test_ik_failure_rejects_whole_chunk_and_retains_frame_errors(mode, reason):
    provider = FakeIK(mode)
    result = _bridge(provider=provider)
    assert not result.accepted and result.actions is None and not result.valid.any()
    assert reason in result.frame_errors[0][0] and result.frame_errors[1] == ("INVALID_PREDECESSOR",)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("kind", ["absent", "unqualified", "missing_transform", "wrong_frame"])
def test_missing_calibration_never_defaults_to_identity(kind):
    _, _, calibration, _ = synthetic_bridge_fixture()
    if kind == "absent":
        calibration = None
    elif kind == "unqualified":
        calibration["geometry_qualified"] = False
    elif kind == "missing_transform":
        calibration.pop("T_world_from_camera")
    else:
        calibration["camera_frame_id"] = "unbound-camera"
    provider = FakeIK()
    result = _bridge(calibration=calibration, provider=provider)
    assert not result.accepted and result.actions is None and not provider.calls


def test_bad_later_frame_does_not_export_a_partial_action_chunk():
    action, *_ = synthetic_bridge_fixture()
    action[7, 6:12] = 0
    result = _bridge(action=action)
    assert result.valid[:7].all() and not result.valid[7:].any()
    assert "DEGENERATE_ROT6" in result.frame_errors[7][0] and result.actions is None


@pytest.mark.parametrize("changes", [
    {"contract": {"qpc_ns": np.arange(50, dtype=float)}},
    {"contract": {"qpc_ns": np.full(50, 1_000_000_000_000_000_001, dtype=np.int64)}},
    {"contract": {"source_frame_ids": np.zeros(50, dtype=np.int64)}},
    {"contract": {"source_frame_ids": np.arange(50)[::-1]}},
    {"contract": {"hand_command_unit": "SDK_NORMALIZED_UNKNOWN"}},
    {"measured_initial": {"measured": False}},
    {"measured_initial": {"clock_domain": "other-clock"}},
    {"measured_initial": {"qpc_ns": 0}},
])
def test_explicit_time_unit_initial_measurement_gates(changes):
    provider = FakeIK()
    assert not _bridge(provider=provider, **changes).accepted and not provider.calls


def test_synthetic_bridge_requires_explicit_fixture_permission():
    action, contract, calibration, initial = synthetic_bridge_fixture()
    result = bridge_action_chunk(action, contract=contract, calibration=calibration,
                                 measured_initial=initial, ik_provider=FakeIK())
    assert not result.accepted and "SYNTHETIC" in result.frame_errors[0][0]


def test_imports_do_not_load_model_sdk_or_robot_libraries():
    check = ("import sys; import chaoyang.humanego_ict.retarget; import chaoyang.humanego_ict.bridge; "
             "assert not {'torch','bc_stark_sdk','rclpy','mujoco','mink'}.intersection(sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", check], check=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True)
