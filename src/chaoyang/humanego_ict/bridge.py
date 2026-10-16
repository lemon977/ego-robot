"""Offline H50/30D camera-TCP -> 26D absolute-joint candidate bridge.

No SDK, ROS, sockets, model imports, reset/home, or hardware publication. Caller
verifies source/calibration artifacts; missing physical facts are never inferred.
An existing IK backend is injected behind an explicitly offline provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from .retarget import CHANNEL_NAMES, _sha, denormalize_motor

JOINT_NAMES = tuple(f"Joint{index}_{side}" for side in ("L", "R") for index in range(1, 8))


class BridgeError(ValueError):
    pass


@dataclass(frozen=True)
class IKSolution:
    q_rad: np.ndarray
    collision_checked: bool
    collision_free: bool
    diagnostics: Mapping[str, Any] | None = None


class OfflineIKProvider(Protocol):
    offline_only: bool
    signature: str
    ik_frame_id: str
    tool_frame_ids: tuple[str, str]

    def solve(self, target_poses: np.ndarray, seed_q_rad: np.ndarray, dt_s: float) -> IKSolution: ...

    def fk(self, q_rad: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BridgeResult:
    accepted: bool
    actions: np.ndarray | None
    valid: np.ndarray
    frame_errors: tuple[tuple[str, ...], ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    source_frame_ids: np.ndarray | None
    qpc_ns: np.ndarray | None
    output_horizon: int
    supervision_kind: str = "DERIVED_RETARGET"
    control_ground_truth: bool = False
    physical_deployment_authorized: bool = False

    @property
    def output_source_frame_ids(self):
        return None if self.actions is None else self.source_frame_ids[:self.output_horizon].copy()

    @property
    def output_qpc_ns(self):
        return None if self.actions is None else self.qpc_ns[:self.output_horizon].copy()

    def to_dict(self):
        return {"schema": "humanego-revo2-offline-bridge-v1", "accepted": self.accepted,
                "actions": None if self.actions is None else self.actions.tolist(),
                "valid": self.valid.tolist(), "frame_errors": [list(x) for x in self.frame_errors],
                "diagnostics": list(self.diagnostics),
                "input_source_frame_ids": None if self.source_frame_ids is None else self.source_frame_ids.tolist(),
                "input_qpc_ns": None if self.qpc_ns is None else self.qpc_ns.tolist(),
                "output_source_frame_ids": None if self.output_source_frame_ids is None else self.output_source_frame_ids.tolist(),
                "output_qpc_ns": None if self.output_qpc_ns is None else self.output_qpc_ns.tolist(),
                "input_horizon": 50, "output_horizon": self.output_horizon,
                "layout": "left_arm7,left_hand6,right_arm7,right_hand6",
                "arm_unit": "radians", "hand_command_unit": "NORMALIZED_0_1",
                "supervision_kind": self.supervision_kind, "offline_only": True,
                "control_ground_truth": False, "physical_deployment_authorized": False}


def _se3(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all() or not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-8):
        raise BridgeError(f"INVALID_SE3:{label}")
    rotation = array[..., :3, :3]
    if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise BridgeError(f"INVALID_ROTATION:{label}")
    return array.copy()


def _array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise BridgeError(f"INVALID_ARRAY:{label}")
    return array.copy()


def _integers(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        raise BridgeError(f"INVALID_INTEGER_SOURCE:{label}")
    return array.astype(np.int64, copy=True)


def decode_actions(actions, *, position_mean, position_std):
    """Decode row-interleaved first-two-column rot6; positions remain in camera frame."""
    action = np.asarray(actions, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 30 or not np.isfinite(action).all():
        raise BridgeError("INVALID_30D_ACTION")
    mean, std = _array(position_mean, (3,), "position_mean"), _array(position_std, (3,), "position_std")
    if np.any(std <= 0):
        raise BridgeError("INVALID_POSITION_NORMALIZATION")
    first_two = action[:, 6:18].reshape(-1, 2, 3, 2)
    first = first_two[..., 0]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise BridgeError("DEGENERATE_ROT6")
    first = first / first_norm
    second = first_two[..., 1] - (first * first_two[..., 1]).sum(axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise BridgeError("DEGENERATE_ROT6")
    second = second / second_norm
    poses = np.broadcast_to(np.eye(4), (len(action), 2, 4, 4)).copy()
    poses[..., :3, :3] = np.stack((first, second, np.cross(first, second)), axis=-1)
    poses[..., :3, 3] = action[:, :6].reshape(-1, 2, 3) * std + mean
    hands = action[:, 18:30].reshape(-1, 2, 6).copy()
    denormalize_motor(hands)  # strict range check only, not an SDK command
    return poses, hands


def _initial_and_calibration(contract, calibration, measured_initial, provider, allow_synthetic_fixture):
    if not isinstance(calibration, Mapping) or calibration.get("geometry_qualified") is not True:
        raise BridgeError("MISSING_OR_UNQUALIFIED_CALIBRATION")
    if calibration.get("synthetic_fixture_only") and not allow_synthetic_fixture:
        raise BridgeError("SYNTHETIC_CALIBRATION_REJECTED")
    if calibration.get("schema") != "offline-pose-ik-calibration-v1":
        raise BridgeError("CALIBRATION_SCHEMA_MISMATCH")
    _sha(calibration.get("source_sha256"))
    _sha(calibration.get("ik_provider_signature"))
    if (calibration.get("camera_frame_id") != contract["camera_frame_id"]
            or tuple(calibration.get("predicted_tcp_frame_ids", ())) != tuple(contract["tcp_frame_ids"])
            or not isinstance(calibration.get("world_frame_id"), str) or not calibration["world_frame_id"]
            or tuple(calibration.get("joint_order", ())) != JOINT_NAMES
            or not isinstance(calibration.get("ik_frame_id"), str) or not calibration["ik_frame_id"]
            or len(calibration.get("tool_frame_ids", [])) != 2
            or any(not isinstance(x, str) or not x for x in calibration["tool_frame_ids"])):
        raise BridgeError("CALIBRATION_FRAME_OR_JOINT_ORDER_MISMATCH")
    if (provider is None or getattr(provider, "offline_only", False) is not True
            or getattr(provider, "signature", None) != calibration["ik_provider_signature"]
            or getattr(provider, "ik_frame_id", None) != calibration.get("ik_frame_id")
            or tuple(getattr(provider, "tool_frame_ids", ())) != tuple(calibration.get("tool_frame_ids", ()))):
        raise BridgeError("MISSING_OR_MISMATCHED_OFFLINE_IK_PROVIDER")
    world_camera = _se3(calibration["T_world_from_camera"], (4, 4), "world_from_camera")
    ik_world = _se3(calibration["T_ik_from_world"], (4, 4), "ik_from_world")
    tcp_tool = _se3(calibration["T_predicted_tcp_from_tool"], (2, 4, 4), "predicted_tcp_from_tool")
    lower = _array(calibration["joint_lower_rad"], (14,), "joint_lower_rad")
    upper = _array(calibration["joint_upper_rad"], (14,), "joint_upper_rad")
    speed = _array(calibration["joint_speed_max_rad_s"], (14,), "joint_speed_max_rad_s")
    step = _array(calibration["joint_step_max_rad"], (14,), "joint_step_max_rad")
    tolerances = [float(calibration[key]) for key in ("position_tolerance_m", "rotation_tolerance_rad")]
    if np.any(lower >= upper) or np.any(speed <= 0) or np.any(step <= 0) or not np.isfinite(tolerances).all() or min(tolerances) <= 0:
        raise BridgeError("INVALID_EXPLICIT_IK_LIMITS")
    if not isinstance(measured_initial, Mapping) or measured_initial.get("measured") is not True:
        raise BridgeError("MISSING_MEASURED_INITIAL_JOINTS")
    if measured_initial.get("synthetic_fixture_only") and not allow_synthetic_fixture:
        raise BridgeError("SYNTHETIC_INITIAL_JOINTS_REJECTED")
    _sha(measured_initial.get("source_sha256"))
    if (measured_initial.get("clock_domain") != contract["clock_domain"]
            or measured_initial.get("joint_unit") != "radians"
            or tuple(measured_initial.get("joint_order", ())) != JOINT_NAMES):
        raise BridgeError("INITIAL_JOINT_CONTRACT_MISMATCH")
    initial_time = int(_integers([measured_initial["qpc_ns"]], (1,), "initial_qpc_ns")[0])
    age_limit = calibration["initial_joint_max_age_ns"]
    if type(age_limit) is not int or age_limit <= 0 or not 0 <= contract["anchor_qpc_ns"] - initial_time <= age_limit:
        raise BridgeError("INITIAL_JOINT_TIME_STALE_OR_FUTURE")
    seed = _array(measured_initial["q_rad"], (14,), "initial_q_rad")
    if np.any(seed < lower) or np.any(seed > upper):
        raise BridgeError("INITIAL_JOINT_LIMIT")
    return ik_world @ world_camera, tcp_tool, seed, initial_time, lower, upper, speed, step, tolerances


def bridge_action_chunk(actions, *, contract, calibration, measured_initial, ik_provider,
                        output_horizon=None, allow_synthetic_fixture=False):
    """Reject the entire chunk on any bad frame; retain original source IDs/QPC exactly.

    H30 is an explicit temporal prefix of H50, never a 30->26 channel truncation.
    No uniform timestamps or presumed 30 Hz are generated. IK uses actual QPC dt.
    Every calibration matrix uses T_destination_from_source column-vector notation.
    """
    horizon = 50 if output_horizon is None else output_horizon
    source_ids, timestamps = None, None
    def rejected(error):
        return BridgeResult(False, None, np.zeros(50, dtype=bool),
                            tuple((str(error),) for _ in range(50)), tuple({} for _ in range(50)),
                            source_ids, timestamps, horizon)
    try:
        if type(horizon) is not int or horizon not in {30, 50}:
            raise BridgeError("OUTPUT_HORIZON_MUST_BE_EXPLICIT_30_OR_50")
        if not isinstance(contract, Mapping) or contract.get("schema") != "humanego-camera-revo2-actions-v1":
            raise BridgeError("ACTION_CONTRACT_REQUIRED")
        source_ids = _integers(contract["source_frame_ids"], (50,), "source_frame_ids")
        timestamps = _integers(contract["qpc_ns"], (50,), "qpc_ns")
        if np.any(np.diff(source_ids) <= 0):
            raise BridgeError("SOURCE_FRAME_IDS_MUST_BE_STRICTLY_INCREASING_AND_UNIQUE")
        if (contract.get("hand_command_unit") != "NORMALIZED_0_1"
                or tuple(contract.get("channel_order", ())) != CHANNEL_NAMES
                or not isinstance(contract.get("clock_domain"), str) or not contract["clock_domain"]
                or not isinstance(contract.get("camera_frame_id"), str) or not contract["camera_frame_id"]
                or len(contract.get("tcp_frame_ids", [])) != 2
                or any(not isinstance(x, str) or not x for x in contract["tcp_frame_ids"])):
            raise BridgeError("ACTION_FRAME_UNIT_OR_CHANNEL_MISMATCH")
        _sha(contract.get("source_signature"))
        anchor = contract["anchor_qpc_ns"]
        if type(anchor) is not int or anchor < 0 or np.any(timestamps <= anchor) or np.any(np.diff(timestamps) <= 0):
            raise BridgeError("INVALID_SOURCE_TIME_MAPPING")
        if contract.get("synthetic_fixture_only") and not allow_synthetic_fixture:
            raise BridgeError("SYNTHETIC_ACTION_CONTRACT_REJECTED")
        action = np.asarray(actions, dtype=np.float64)
        if action.shape != (50, 30):
            raise BridgeError("ACTION_SHAPE_MUST_BE_50_30")
        transform, tcp_tool, seed, last_time, lower, upper, speed, step, tolerance = _initial_and_calibration(
            contract, calibration, measured_initial, ik_provider, allow_synthetic_fixture)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return rejected(exc)
    valid, errors, diagnostics = np.zeros(50, dtype=bool), [], []
    converted = np.empty((50, 26), dtype=np.float64)
    predecessor_valid = True
    for frame in range(50):
        try:
            camera_tcp, hand = decode_actions(action[frame:frame + 1],
                position_mean=contract["position_mean"], position_std=contract["position_std"])
            if not predecessor_valid:
                raise BridgeError("INVALID_PREDECESSOR")
            target = transform @ camera_tcp[0] @ tcp_tool
            dt = (int(timestamps[frame]) - last_time) / 1e9
            solution = ik_provider.solve(target.copy(), seed.copy(), dt)
            if not isinstance(solution, IKSolution):
                raise BridgeError("INVALID_IK_SOLUTION_TYPE")
            q = _array(solution.q_rad, (14,), "ik_q_rad")
            if np.any(q < lower) or np.any(q > upper):
                raise BridgeError("IK_JOINT_LIMIT")
            if np.any(np.abs(q - seed) > step) or np.any(np.abs(q - seed) > speed * dt):
                raise BridgeError("IK_STEP_OR_SPEED_LIMIT")
            if solution.collision_checked is not True or solution.collision_free is not True:
                raise BridgeError("IK_COLLISION_OR_UNCHECKED")
            fk = _se3(ik_provider.fk(q.copy()), (2, 4, 4), "ik_fk")
            position_error = np.linalg.norm(fk[:, :3, 3] - target[:, :3, 3], axis=1)
            relative_rotation = np.swapaxes(target[:, :3, :3], -1, -2) @ fk[:, :3, :3]
            rotation_error = np.arccos(np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
            if np.any(position_error > tolerance[0]) or np.any(rotation_error > tolerance[1]):
                raise BridgeError("IK_UNREACHABLE_FK_RESIDUAL")
            converted[frame] = np.concatenate((q[:7], hand[0, 0], q[7:], hand[0, 1]))
            valid[frame] = True
            errors.append(())
            diagnostics.append({"position_error_m": position_error.tolist(), "rotation_error_rad": rotation_error.tolist(),
                                "dt_s": dt, "ik_provider_signature": ik_provider.signature})
            seed, last_time = q, int(timestamps[frame])
        except Exception as exc:
            predecessor_valid = False
            errors.append((str(exc),))
            diagnostics.append({})
    accepted = bool(valid.all())
    return BridgeResult(accepted, converted[:horizon].copy() if accepted else None, valid,
                        tuple(errors), tuple(diagnostics), source_ids, timestamps, horizon)
