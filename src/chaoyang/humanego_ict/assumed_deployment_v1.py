"""Assumption-driven, in-memory offline integration; never a measured adapter.

This candidate reuses the frozen numerical ICT/action decoders. It never calls
build_measured_ict, relaxes its qualifications, loads a policy, or imports a
device/transport library. An explicitly supplied provider is mathematical only:
FK/IK poses are per-arm-base_from_model_tool, in metres. A model-world backend
must be wrapped to that convention by its caller, not silently applied twice.

The root-owned ASSUMPTIONS.json format is accepted directly. Identity matrices
are used only when that file actually declares them, never as hidden defaults.
Provided numeric limits are required for IK and use the existing bridge's
joint/step/speed/collision/FK-residual checks. Their passage is MODEL NUMERICS,
not measured geometry, hardware ACK, safe deployment, or training labels.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping

import numpy as np

from .bridge import IKSolution, JOINT_NAMES, _array, _integers, _se3
from .inference import HAND_CHANNELS, _normalization, build_hand_ict, decode_actions

PROFILE_SCHEMA = "humanego-assumed-offline-deployment-profile-v1"
MODE = "ASSUMPTION_DRIVEN_OFFLINE"
ROT6_DECLARATION = "first_two_columns_row_interleaved_as_frozen_inference_contract"
PRIOR_IDS = ("LEFT_INSTALLATION_WRIST_PRIOR", "RIGHT_INSTALLATION_WRIST_PRIOR")
_DIRECTIONS = ("T_world_from_camera", "T_world_from_arm_base",
               "T_model_tool_from_training_wrist_prior")
_FALSE_FLAGS = ("measured", "geometry_qualified", "geometry_verified",
                "timebase_verified", "feedback_qualified", "hardware_ack_verified",
                "control_ground_truth", "physical_deployment_authorized")


class AssumedDeploymentError(ValueError):
    pass


def _int(value, name):
    if type(value) is not int or not 0 <= value <= np.iinfo(np.int64).max:
        raise AssumedDeploymentError(f"INVALID_OFFLINE_INTEGER:{name}")
    return value


def _base(profile, schema):
    return {"schema": schema, "mode": MODE, "parameter_mode": "ASSUMED",
            "offline_only": True, **{name: False for name in _FALSE_FLAGS},
            "device_ack_status": "UNKNOWN", "robot_commands_sent": 0,
            "qualification": "ASSUMPTIONS_NOT_PHYSICAL_EVIDENCE",
            "assumption_profile_sha256": hashlib.sha256(json.dumps(profile,
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "assumptions": deepcopy(profile["assumptions"])}


def _profile(profile):
    if (not isinstance(profile, Mapping) or profile.get("schema") != PROFILE_SCHEMA
            or profile.get("mode") != MODE or profile.get("position_unit") != "meters"
            or tuple(profile.get("hand_channel_order", ())) != HAND_CHANNELS):
        raise AssumedDeploymentError("EXPLICIT_ASSUMED_PROFILE_REQUIRED")
    if profile.get("rotation_convention") != ROT6_DECLARATION:
        raise AssumedDeploymentError("FROZEN_ROT6_DECLARATION_REQUIRED")
    # No claim can enter a measured mode through an optional/extra true flag.
    for key in _FALSE_FLAGS:
        if profile.get(key, False) is not False:
            raise AssumedDeploymentError(f"ASSUMED_PROFILE_CANNOT_DECLARE:{key}")
    if profile.get("device_ack_status") != "UNKNOWN":
        raise AssumedDeploymentError("ASSUMED_PROFILE_CANNOT_DECLARE_DEVICE_ACK")
    assumptions = profile.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions:
        raise AssumedDeploymentError("EXPLICIT_ASSUMPTIONS_REQUIRED")
    declarations = {}
    seen = set()
    for item in assumptions:
        if (not isinstance(item, Mapping) or not isinstance(item.get("id"), str)
                or not item["id"].strip() or item["id"] in seen
                or item.get("source_class") != "ASSUMED_NOT_MEASURED"
                or item.get("physical_qualification") is not False):
            raise AssumedDeploymentError("ASSUMPTION_QUALIFICATION_OR_ID_INVALID")
        seen.add(item["id"])
        direction = item.get("matrix_direction")
        if direction in _DIRECTIONS:
            if direction in declarations:
                raise AssumedDeploymentError("DUPLICATE_ASSUMED_MATRIX_DIRECTION")
            declarations[direction] = item
    transforms = profile.get("transforms", {})
    if not isinstance(transforms, Mapping) or set(transforms) - set(_DIRECTIONS):
        raise AssumedDeploymentError("UNKNOWN_ASSUMED_TRANSFORM_DIRECTION")
    matrices = {}
    for direction in _DIRECTIONS:
        if direction not in declarations:
            raise AssumedDeploymentError(f"MISSING_ASSUMPTION:{direction}")
        declaration = declarations[direction]
        if direction in transforms:
            value = transforms[direction]
        elif "default_matrix" in declaration:
            value = declaration["default_matrix"]
        else:
            raise AssumedDeploymentError(f"MISSING_EXPLICIT_MATRIX:{direction}")
        matrix = np.asarray(value, dtype=np.float64)
        shape = (4, 4) if direction == _DIRECTIONS[0] else (2, 4, 4)
        # One explicitly declared common transform may apply to both arms.
        if shape == (2, 4, 4) and matrix.shape == (4, 4):
            matrix = np.repeat(matrix[None], 2, axis=0)
        matrices[direction] = _se3(matrix, shape, direction)
    if (np.array_equal(matrices[_DIRECTIONS[2]], np.repeat(np.eye(4)[None], 2, axis=0))
            and declarations[_DIRECTIONS[2]]["id"] != "MODEL_TOOL_EQUALS_TRAINING_WRIST_PRIOR"):
        raise AssumedDeploymentError("IDENTITY_TOOL_WRIST_PRIOR_REQUIRES_EXPLICIT_ASSUMPTION")
    return matrices


def _provider(provider, method):
    if provider is None:
        return False
    signature = getattr(provider, "signature", None)
    if (getattr(provider, "offline_only", False) is not True
            or not callable(getattr(provider, method, None))
            or not isinstance(signature, str) or len(signature) != 64
            or any(c not in "0123456789abcdef" for c in signature)):
        raise AssumedDeploymentError("OFFLINE_MATHEMATICAL_PROVIDER_AND_SHA_REQUIRED")
    return True


def build_assumed_ict(initial_q_rad, *, x_rgb, assumption_profile, fk_provider,
                      normalization, anchor):
    """Mathematical q14/FK + explicit assumptions -> unchanged hand-only ICT29.

    anchor has source_frame_id, anchor_ns, image_ns, joint_ns and
    time_source=ASSUMED_OFFLINE_CLOCK or SOURCE_DECLARED_CAPTURE_NS_UNVERIFIED.
    These numbers are never upgraded to verified QPC/exposure/sample times.
    q14 is caller-provided numerical input, NOT admitted measured feedback.
    """
    matrices = _profile(assumption_profile)
    _normalization(normalization)
    q = _array(initial_q_rad, (14,), "assumed_initial_q_rad")
    if not isinstance(anchor, Mapping) or anchor.get("time_source") not in {
            "ASSUMED_OFFLINE_CLOCK", "SOURCE_DECLARED_CAPTURE_NS_UNVERIFIED"}:
        raise AssumedDeploymentError("EXPLICIT_UNVERIFIED_OFFLINE_ANCHOR_REQUIRED")
    for key in _FALSE_FLAGS:
        if anchor.get(key, False) is not False:
            raise AssumedDeploymentError(f"ASSUMED_ANCHOR_CANNOT_DECLARE:{key}")
    frame = _int(anchor.get("source_frame_id"), "source_frame_id")
    now = _int(anchor.get("anchor_ns"), "anchor_ns")
    image_time = _int(anchor.get("image_ns"), "image_ns")
    joint_time = _int(anchor.get("joint_ns"), "joint_ns")
    if image_time > now or joint_time > now:
        raise AssumedDeploymentError("OFFLINE_CURRENT_INPUT_MUST_NOT_BE_FUTURE")
    rgb = np.asarray(x_rgb)
    if rgb.shape != (1, 3, 240, 320) or rgb.dtype.kind not in "fu":
        raise AssumedDeploymentError("RGB_SHAPE_OR_DTYPE_INVALID")
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    elif rgb.dtype.kind != "f":
        raise AssumedDeploymentError("RGB_SHAPE_OR_DTYPE_INVALID")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise AssumedDeploymentError("RGB_RANGE_OR_NONFINITE")
    result = _base(assumption_profile, "humanego-assumed-offline-ict-v1")
    result.update(status="NO_PROVIDER", ict=None, current_input_npz=None,
                  errors=["NO_PROVIDER"], time_source=anchor["time_source"],
                  used_transforms={key: value.copy() for key, value in matrices.items()},
                  current_state_type="ASSUMED_MODEL_FK_NOT_MEASURED",
                  input_qualification="CALLER_INPUT_IDENTITY_NOT_AUDITED_BY_THIS_FUNCTION")
    if not _provider(fk_provider, "fk"):
        return result
    tool = _se3(fk_provider.fk(q.copy()), (2, 4, 4), "per_arm_base_from_model_tool")
    poses = (matrices["T_world_from_arm_base"] @ tool
             @ matrices["T_model_tool_from_training_wrist_prior"])
    camera = matrices["T_world_from_camera"]
    declaration = {"pose_semantics": "DERIVED_RETARGET_ANCHOR", "position_unit": "meters",
                   "pose_frame_id": "ASSUMED_OFFLINE_WORLD", "camera_frame_id": "ASSUMED_REFERENCE_CAMERA",
                   "tcp_frame_ids": list(PRIOR_IDS), "current_state_type": "ASSUMED_MODEL_FK_NOT_MEASURED",
                   "clock_domain": anchor["time_source"], "parameter_mode": "ASSUMED"}
    times = np.full(2, joint_time, dtype=np.int64)
    ict = build_hand_ict(poses, np.ones(2, bool), camera, normalization=normalization,
        source_qpc_ns=times, anchor_qpc_ns=now, frame_declaration=declaration)
    ict.update({name: False for name in _FALSE_FLAGS})
    ict["coordinate_qualification"] = "ASSUMED_NOT_MEASURED"
    result.update(status="BUILT_ASSUMED_ICT_OFFLINE_ONLY", errors=[], ict=ict,
        provider_sha256=fk_provider.signature,
        current_input_npz={"x_rgb": rgb.astype(np.float32, copy=True),
            "wrist_pose_world": poses.copy(), "wrist_valid": np.ones(2, bool),
            "reference_camera_to_world": camera.copy(), "source_qpc_ns": times,
            "anchor_qpc_ns": np.array(now, np.int64), "source_frame_id": np.array(frame, np.int64)})
    return result


def _limits(profile):
    limits = profile.get("numerical_limits")
    if not isinstance(limits, Mapping):
        raise AssumedDeploymentError("EXPLICIT_EXISTING_NUMERICAL_LIMITS_REQUIRED")
    lower = _array(limits.get("joint_lower_rad"), (14,), "joint_lower_rad")
    upper = _array(limits.get("joint_upper_rad"), (14,), "joint_upper_rad")
    speed = _array(limits.get("joint_speed_max_rad_s"), (14,), "joint_speed_max_rad_s")
    step = _array(limits.get("joint_step_max_rad"), (14,), "joint_step_max_rad")
    tolerances = [float(limits.get(key)) for key in ("position_tolerance_m", "rotation_tolerance_rad")]
    if (np.any(lower >= upper) or np.any(speed <= 0) or np.any(step <= 0)
            or not np.isfinite(tolerances).all() or min(tolerances) <= 0):
        raise AssumedDeploymentError("INVALID_EXPLICIT_IK_LIMITS")
    return lower, upper, speed, step, tolerances


def bridge_assumed_action_chunk(actions, *, action_mapping, assumption_profile,
                                initial_q_rad, ik_provider, normalization,
                                output_horizon=50):
    """H50/30D -> mathematical IK candidate26; never a client execution receipt.

    action_mapping declares source_frame_ids[50], timestamp_ns[50], anchor_ns
    and mapping_kind=ASSUMED_OFFLINE_GRID or SOURCE_DECLARED_UNVERIFIED_NATIVE.
    The caller supplies the entire grid. No default FPS, resampling, clipping,
    guessed feedback, auto-retry, or fictitious device ACK is introduced.
    All 50 frames must pass even when a 30-frame prefix is requested.
    """
    matrices = _profile(assumption_profile)
    _normalization(normalization)
    result = _base(assumption_profile, "humanego-assumed-offline-bridge-v1")
    result.update(status="FAILED_ASSUMED_MODEL_NUMERICS", numerical_candidate_accepted=False,
        candidate_actions=None, raw_actions=None, valid=np.zeros(50, bool),
        errors=[], frame_errors=[[] for _ in range(50)], diagnostics=[{} for _ in range(50)],
        output_horizon=output_horizon, arm_unit="radians", hand_command_unit="NORMALIZED_0_1",
        layout="left_arm7,left_hand6,right_arm7,right_hand6", joint_order=list(JOINT_NAMES),
        execution_mapping_status="UNKNOWN_NO_CLIENT_OR_DEVICE_EXECUTION",
        collision_qualification="ASSUMED_MODEL_ONLY_NOT_PHYSICAL_SAFETY")
    try:
        raw = np.asarray(actions, dtype=np.float64)
        if raw.shape != (50, 30):
            raise AssumedDeploymentError("ACTION_SHAPE_MUST_BE_50_30")
        result["raw_actions"] = raw.copy()  # Preserve genuine caller H50 BEFORE REJECT.
        if type(output_horizon) is not int or output_horizon not in {30, 50}:
            raise AssumedDeploymentError("OUTPUT_HORIZON_MUST_BE_EXPLICIT_30_OR_50")
        if (not isinstance(action_mapping, Mapping) or action_mapping.get("mapping_kind") not in {
                "ASSUMED_OFFLINE_GRID", "SOURCE_DECLARED_UNVERIFIED_NATIVE"}):
            raise AssumedDeploymentError("EXPLICIT_OFFLINE_H50_MAPPING_REQUIRED")
        for key in _FALSE_FLAGS:
            if action_mapping.get(key, False) is not False:
                raise AssumedDeploymentError(f"ASSUMED_MAPPING_CANNOT_DECLARE:{key}")
        ids = _integers(action_mapping.get("source_frame_ids"), (50,), "source_frame_ids")
        times = _integers(action_mapping.get("timestamp_ns"), (50,), "timestamp_ns")
        anchor = _int(action_mapping.get("anchor_ns"), "anchor_ns")
        if np.any(np.diff(ids) <= 0) or np.any(times <= anchor) or np.any(np.diff(times) <= 0):
            raise AssumedDeploymentError("NONMONOTONIC_OR_NONFUTURE_OFFLINE_H50_MAPPING")
        result.update(input_source_frame_ids=ids.copy(), input_timestamp_ns=times.copy(),
                      mapping_kind=action_mapping["mapping_kind"])
        seed = _array(initial_q_rad, (14,), "assumed_initial_q_rad")
        if not _provider(ik_provider, "solve"):
            result.update(status="NO_PROVIDER", errors=["NO_PROVIDER"])
            return result
        if not callable(getattr(ik_provider, "fk", None)):
            raise AssumedDeploymentError("OFFLINE_IK_FK_PROVIDER_REQUIRED")
        lower, upper, speed, step, tolerance = _limits(assumption_profile)
        if np.any(seed < lower) or np.any(seed > upper):
            raise AssumedDeploymentError("INITIAL_JOINT_LIMIT")
        # The existing inference decoder supplies its exact rot6/range/identity
        # normalization gates. CLIP_DIAGNOSTIC is intentionally never used.
        decoded = decode_actions(raw, matrices["T_world_from_camera"],
                                 normalization=normalization, command_policy="REJECT")
        camera_poses = decoded["wrist_pose_reference_camera"]
        hand = decoded["hand_command"]
        world_base_inv = np.linalg.inv(matrices["T_world_from_arm_base"])
        tool_prior_inv = np.linalg.inv(matrices["T_model_tool_from_training_wrist_prior"])
        converted = np.empty((50, 26), np.float64)
        last_time, predecessor_valid = anchor, True
        for frame in range(50):
            try:
                if not predecessor_valid:
                    raise AssumedDeploymentError("INVALID_PREDECESSOR")
                target = (world_base_inv @ matrices["T_world_from_camera"]
                          @ camera_poses[frame] @ tool_prior_inv)
                dt = (int(times[frame]) - last_time) / 1e9
                solution = ik_provider.solve(target.copy(), seed.copy(), dt)
                if not isinstance(solution, IKSolution):
                    raise AssumedDeploymentError("INVALID_IK_SOLUTION_TYPE")
                q = _array(solution.q_rad, (14,), "ik_q_rad")
                if np.any(q < lower) or np.any(q > upper):
                    raise AssumedDeploymentError("IK_JOINT_LIMIT")
                if np.any(np.abs(q - seed) > step) or np.any(np.abs(q - seed) > speed * dt):
                    raise AssumedDeploymentError("IK_STEP_OR_SPEED_LIMIT")
                if solution.collision_checked is not True or solution.collision_free is not True:
                    raise AssumedDeploymentError("IK_COLLISION_OR_UNCHECKED")
                fk = _se3(ik_provider.fk(q.copy()), (2, 4, 4), "per_arm_base_fk")
                position_error = np.linalg.norm(fk[:, :3, 3] - target[:, :3, 3], axis=1)
                relative_rotation = np.swapaxes(target[:, :3, :3], -1, -2) @ fk[:, :3, :3]
                rotation_error = np.arccos(np.clip((np.trace(relative_rotation,
                    axis1=-2, axis2=-1) - 1) / 2, -1, 1))
                if np.any(position_error > tolerance[0]) or np.any(rotation_error > tolerance[1]):
                    raise AssumedDeploymentError("IK_UNREACHABLE_FK_RESIDUAL")
                converted[frame] = np.concatenate((q[:7], hand[frame, 0], q[7:], hand[frame, 1]))
                result["valid"][frame] = True
                result["diagnostics"][frame] = {"position_error_m": position_error.tolist(),
                    "rotation_error_rad": rotation_error.tolist(), "dt_s": dt,
                    "provider_sha256": ik_provider.signature}
                seed, last_time = q, int(times[frame])
            except Exception as error:
                predecessor_valid = False
                result["frame_errors"][frame] = [str(error)]
        if result["valid"].all():
            result.update(status="PASSED_ASSUMED_MODEL_NUMERICS_ONLY",
                numerical_candidate_accepted=True, candidate_actions=converted[:output_horizon].copy(),
                output_source_frame_ids=ids[:output_horizon].copy(),
                output_timestamp_ns=times[:output_horizon].copy(),
                omitted_suffix_frames=50-output_horizon, provider_sha256=ik_provider.signature)
        else:
            result["errors"] = sorted({error for row in result["frame_errors"] for error in row})
    except (TypeError, ValueError, OverflowError, KeyError) as error:
        result["errors"] = [str(error)]
        result["frame_errors"] = [[str(error)] for _ in range(50)]
    return result
