"""Local measured-state and v2 protocol dry-run adapters, never a device client.

No SDK, ROS, socket, Torch, policy loading, home/reset, or command publisher is
imported here. Providers are supplied by an independently admitted caller. SHA
bindings below validate declarations; they do not replace upstream artifact,
clock, installation or physical calibration audits.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Protocol

import numpy as np

from .bridge import JOINT_NAMES, bridge_action_chunk
from .inference import _normalization, build_hand_ict

LAYOUT = "left_arm7,left_hand6,right_arm7,right_hand6"
STATE_SCHEMA = "humanego-measured-joints-fk-ict-v1"
SCHEDULE_SCHEMA = "humanego-native-h50-schedule-v1"


class DeploymentAdapterError(ValueError):
    pass


class OfflineFKProvider(Protocol):
    offline_only: bool
    signature: str
    robot_model_sha256: str
    ik_frame_id: str
    tool_frame_ids: tuple[str, str]
    joint_order: tuple[str, ...]

    def fk(self, q_rad: np.ndarray) -> np.ndarray: ...


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise DeploymentAdapterError(f"INVALID_SHA:{name}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value.strip() in {"UNKNOWN", "ABSENT"}:
        raise DeploymentAdapterError(f"MISSING_DECLARATION:{name}")
    return value


def _int(value, name, *, positive=False):
    if (type(value) is not int or value < (1 if positive else 0)
            or value > np.iinfo(np.int64).max):
        raise DeploymentAdapterError(f"INVALID_NATIVE_INTEGER:{name}")
    return value


def _integers(value, name):
    a = np.asarray(value)
    if (a.shape != (50,) or a.dtype.kind not in "iu" or np.any(a < 0)
            or np.any(a > np.iinfo(np.int64).max)):
        raise DeploymentAdapterError(f"INVALID_H50_MAPPING:{name}")
    a = a.astype(np.int64, copy=True)
    if np.any(np.diff(a) <= 0):
        raise DeploymentAdapterError(f"NONMONOTONIC_H50_MAPPING:{name}")
    return a


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise DeploymentAdapterError(f"INVALID_FINITE_SHAPE:{name}")
    return a.copy()


def _se3(value, shape, name):
    a = _array(value, shape, name)
    r = a[..., :3, :3]
    if (not np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-6)):
        raise DeploymentAdapterError(f"INVALID_SE3:{name}")
    return a


def _fixture(value, allowed, name):
    if value.get("synthetic_fixture_only") and not allowed:
        raise DeploymentAdapterError(f"SYNTHETIC_FIXTURE_REJECTED:{name}")


def _feedback_qualification(measured):
    """Caller-derived qualification, not a claim that the SDK supplies q_valid."""
    if not isinstance(measured, Mapping) or measured.get("measured") is not True:
        raise DeploymentAdapterError("MEASURED_JOINT_FEEDBACK_REQUIRED")
    valid = np.asarray(measured.get("q_valid"))
    if (valid.shape != (14,) or valid.dtype.kind != "b" or not valid.all()
            or measured.get("feedback_qualification") != "VALIDATED_FEEDBACK"
            or measured.get("sample_origin") != "SDK_JOINT_FEEDBACK"
            or measured.get("timebase_verified") is not True):
        raise DeploymentAdapterError("MEASURED_JOINT_FEEDBACK_QUALIFICATION_REQUIRED")
    _sha(measured.get("feedback_qualification_sha256"), "feedback_qualification_sha256")


def _frames(calibration):
    if (not isinstance(calibration, Mapping)
            or calibration.get("schema") != "offline-pose-ik-calibration-v1"
            or calibration.get("geometry_qualified") is not True):
        raise DeploymentAdapterError("MISSING_OR_UNQUALIFIED_CALIBRATION")
    for key in ("source_sha256", "robot_model_sha256"):
        _sha(calibration.get(key), key)
    for key in ("camera_frame_id", "world_frame_id", "ik_frame_id"):
        _text(calibration.get(key), key)
    if (tuple(calibration.get("joint_order", ())) != JOINT_NAMES
            or calibration.get("position_unit") != "meters"):
        raise DeploymentAdapterError("CALIBRATION_JOINT_ORDER_OR_POSITION_UNIT_MISMATCH")
    tcp, tool = calibration.get("predicted_tcp_frame_ids"), calibration.get("tool_frame_ids")
    for ids, name in ((tcp, "predicted_tcp_frame_ids"), (tool, "tool_frame_ids")):
        if not isinstance(ids, (list, tuple)) or len(ids) != 2 or len(set(ids)) != 2:
            raise DeploymentAdapterError(f"INVALID_FRAME_PAIR:{name}")
        for frame in ids:
            _text(frame, name)


def build_measured_ict(measured_joints, *, x_rgb, calibration, fk_provider,
                       normalization, anchor, allow_synthetic_fixture=False):
    """Measured q14 -> frozen FK -> derived current anchor -> unchanged ICT29.

    FK returns T_ik_from_tool[2,4,4]. Using column-vector transform notation:
      T_world_from_predicted_tcp = inv(T_ik_from_world) @ T_ik_from_tool
                                   @ inv(T_predicted_tcp_from_tool).
    Both arms are required. Current hand motor values are NOT ICT29 features.
    Returned NPZ fields are in memory; a publisher must freeze bytes and bind the
    resulting SHA before creating an admitted inference input receipt.
    """
    _frames(calibration)
    _feedback_qualification(measured_joints)
    if not isinstance(anchor, Mapping) or anchor.get("timebase_verified") is not True:
        raise DeploymentAdapterError("TIMEBASE_UNVERIFIED")
    for obj, name in ((calibration, "calibration"), (measured_joints, "measured_joints"), (anchor, "anchor")):
        _fixture(obj, allow_synthetic_fixture, name)
    if (measured_joints.get("joint_unit") != "radians"
            or tuple(measured_joints.get("joint_order", ())) != JOINT_NAMES
            or measured_joints.get("timestamp_unit") != "nanoseconds"
            or measured_joints.get("sample_origin") != "SDK_JOINT_FEEDBACK"):
        raise DeploymentAdapterError("MEASURED_JOINT_ORDER_UNIT_OR_SOURCE_MISMATCH")
    for obj, fields in ((measured_joints, ("source_sha256", "timebase_sha256")),
                        (anchor, ("source_sha256", "timebase_sha256"))):
        for field in fields:
            _sha(obj.get(field), field)
    clock = _text(anchor.get("clock_domain"), "clock_domain")
    if measured_joints.get("clock_domain") != clock or calibration.get("clock_domain") != clock:
        raise DeploymentAdapterError("CURRENT_STATE_CLOCK_DOMAIN_MISMATCH")
    if (anchor.get("timestamp_unit") != "nanoseconds"
            or anchor.get("camera_frame_id") != calibration["camera_frame_id"]
            or anchor.get("image_preprocessing") != "RGB_RESIZE_BILINEAR_320_240"
            or anchor.get("time_mode") != "CAUSAL"):
        raise DeploymentAdapterError("RGB_ANCHOR_FRAME_TIME_OR_PREPROCESSING_MISMATCH")
    now = _int(anchor.get("anchor_qpc_ns"), "anchor_qpc_ns")
    image_time = _int(anchor.get("image_qpc_ns"), "image_qpc_ns")
    frame = _int(anchor.get("source_frame_id"), "source_frame_id")
    measured_time = _int(measured_joints.get("qpc_ns"), "measured_qpc_ns")
    age = _int(calibration.get("initial_joint_max_age_ns"), "joint_max_age_ns", positive=True)
    if not 0 <= now - measured_time <= age or not 0 <= now - image_time <= age:
        raise DeploymentAdapterError("CURRENT_JOINT_OR_RGB_TIME_STALE_OR_FUTURE")
    camera_time = _int(calibration.get("camera_pose_qpc_ns"), "camera_pose_qpc_ns")
    camera_age = _int(calibration.get("camera_pose_max_age_ns"), "camera_pose_max_age_ns", positive=True)
    _sha(calibration.get("camera_pose_source_sha256"), "camera_pose_source_sha256")
    if not 0 <= now - camera_time <= camera_age:
        raise DeploymentAdapterError("REFERENCE_CAMERA_POSE_TIME_STALE_OR_FUTURE")
    q = _array(measured_joints.get("q_rad"), (14,), "measured_q_rad")
    lower = _array(calibration.get("joint_lower_rad"), (14,), "joint_lower_rad")
    upper = _array(calibration.get("joint_upper_rad"), (14,), "joint_upper_rad")
    if np.any(lower >= upper) or np.any(q < lower) or np.any(q > upper):
        raise DeploymentAdapterError("MEASURED_JOINT_LIMIT")
    camera = _se3(calibration.get("T_world_from_camera"), (4, 4), "world_from_camera")
    ik_world = _se3(calibration.get("T_ik_from_world"), (4, 4), "ik_from_world")
    tcp_tool = _se3(calibration.get("T_predicted_tcp_from_tool"), (2, 4, 4), "predicted_tcp_from_tool")
    expected_fk = _sha(calibration.get("fk_provider_signature"), "fk_provider_signature")
    if (fk_provider is None or getattr(fk_provider, "offline_only", False) is not True
            or getattr(fk_provider, "signature", None) != expected_fk
            or getattr(fk_provider, "robot_model_sha256", None) != calibration["robot_model_sha256"]
            or getattr(fk_provider, "ik_frame_id", None) != calibration["ik_frame_id"]
            or tuple(getattr(fk_provider, "joint_order", ())) != JOINT_NAMES
            or tuple(getattr(fk_provider, "tool_frame_ids", ())) != tuple(calibration["tool_frame_ids"])):
        raise DeploymentAdapterError("MISSING_OR_MISMATCHED_FROZEN_FK_PROVIDER")
    rgb = np.asarray(x_rgb)
    if rgb.shape != (1, 3, 240, 320) or rgb.dtype.kind not in "fu":
        raise DeploymentAdapterError("CURRENT_RGB_REQUIRES_1_3_240_320_FLOAT_OR_UINT8")
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    elif rgb.dtype.kind != "f":
        raise DeploymentAdapterError("CURRENT_RGB_REQUIRES_FLOAT_OR_UINT8")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise DeploymentAdapterError("CURRENT_RGB_RANGE_OR_NONFINITE")
    try:
        _normalization(normalization)
    except ValueError as exc:
        raise DeploymentAdapterError("CURRENT_ICT_NORMALIZATION_CONTRACT_MISMATCH") from exc
    # Validate everything before calling even the injected, offline FK provider.
    try:
        tool = _se3(fk_provider.fk(q.copy()), (2, 4, 4), "fk_ik_from_tool")
    except DeploymentAdapterError:
        raise
    except Exception as exc:
        raise DeploymentAdapterError("OFFLINE_FK_PROVIDER_FAILED") from exc
    poses = np.linalg.inv(ik_world) @ tool @ np.linalg.inv(tcp_tool)
    _se3(poses, (2, 4, 4), "world_from_predicted_tcp")
    declaration = {"clock_domain": clock, "pose_frame_id": calibration["world_frame_id"],
                   "camera_frame_id": calibration["camera_frame_id"], "position_unit": "meters",
                   "tcp_frame_ids": list(calibration["predicted_tcp_frame_ids"]),
                   "pose_semantics": "DERIVED_RETARGET_ANCHOR",
                   "current_state_type": "MEASURED_JOINTS_FK_DERIVED"}
    times = np.array([measured_time, measured_time], dtype=np.int64)
    ict = build_hand_ict(poses, np.ones(2, dtype=bool), camera,
        normalization=normalization, source_qpc_ns=times, anchor_qpc_ns=now,
        frame_declaration=declaration)
    return {"schema": STATE_SCHEMA, "status": "DRY_RUN_BUILT_NOT_PUBLISHED",
            "current_input_npz": {"x_rgb": rgb.astype(np.float32, copy=True),
                "wrist_pose_world": poses.copy(), "wrist_valid": np.ones(2, dtype=bool),
                "reference_camera_to_world": camera.copy(), "source_qpc_ns": times,
                "anchor_qpc_ns": np.array(now, dtype=np.int64), "source_frame_id": np.array(frame, dtype=np.int64)},
            "ict": ict, "receipt_fields": {**declaration, "source_frame_id": frame, "anchor_qpc_ns": now,
                "timestamp_unit": "nanoseconds", "time_mode": "CAUSAL", "image_qpc_ns": image_time,
                "image_preprocessing": anchor["image_preprocessing"],
                "measured_joint_source_sha256": measured_joints["source_sha256"],
                "feedback_qualification_sha256": measured_joints["feedback_qualification_sha256"],
                "image_source_sha256": anchor["source_sha256"],
                "calibration_source_sha256": calibration["source_sha256"],
                "robot_model_sha256": calibration["robot_model_sha256"], "fk_provider_signature": expected_fk,
                "input_npz_sha256": None, "publication_status": "UNSEALED_REQUIRES_SHA_AND_EVIDENCE_REFS"},
            "fixture_only": bool(allow_synthetic_fixture), "offline_only": True,
            "control_ground_truth": False, "physical_deployment_authorized": False, "robot_commands_sent": 0}


def _protocol_metadata(metadata):
    if (not isinstance(metadata, Mapping) or type(metadata.get("protocol_version")) is not int
            or metadata.get("protocol_version") != 2):
        raise DeploymentAdapterError("V2_POLICY_METADATA_REQUIRED")
    for key in ("policy_id", "task_id", "checkpoint", "config", "default_prompt"):
        _text(metadata.get(key), key)
    _sha(metadata.get("norm_sha256"), "norm_sha256")
    # B_016/B_046 publish/validate action_dim, not an invented state_dim key.
    if (any(type(metadata.get(key)) is not int for key in ("action_dim", "action_horizon"))
            or metadata.get("action_dim") != 26
            or metadata.get("action_horizon") != 30 or metadata.get("layout") != LAYOUT
            or metadata.get("action_semantics_id") != "absolute_joint_targets"):
        raise DeploymentAdapterError("POLICY_26D_H30_ABSOLUTE_CONTRACT_MISMATCH")
    hz = metadata.get("control_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not np.isfinite(hz) or hz <= 0:
        raise DeploymentAdapterError("EXPLICIT_PROTOCOL_CONTROL_HZ_REQUIRED")
    return hz


def _schedule(schedule, contract, metadata, hz, allowed):
    if (not isinstance(schedule, Mapping) or schedule.get("schema") != SCHEDULE_SCHEMA
            or schedule.get("status") != "PASSED" or schedule.get("timebase_verified") is not True
            or schedule.get("mapping_kind") != "FROZEN_NATIVE_H50_PREDICTION_MAPPING"):
        raise DeploymentAdapterError("UNKNOWN_REQUIRES_NATIVE_H50_MAPPING")
    _fixture(schedule, allowed, "schedule")
    for key in ("source_sha256", "timebase_sha256"):
        _sha(schedule.get(key), key)
    if (schedule.get("timestamp_unit") != "nanoseconds"
            or schedule.get("clock_domain") != contract.get("clock_domain")
            or schedule.get("anchor_qpc_ns") != contract.get("anchor_qpc_ns")
            or schedule.get("source_signature") != contract.get("source_signature")
            or schedule.get("policy_id") != metadata["policy_id"]
            or schedule.get("legacy_fixed_rate_compatible") is not True
            or schedule.get("control_hz") != hz):
        raise DeploymentAdapterError("SCHEDULE_CLOCK_SOURCE_POLICY_OR_RATE_MISMATCH")
    anchor = _int(schedule.get("anchor_qpc_ns"), "schedule_anchor_qpc_ns")
    ids = _integers(schedule.get("source_frame_ids"), "schedule_frame_ids")
    times = _integers(schedule.get("qpc_ns"), "schedule_qpc_ns")
    native_ids = _integers(contract.get("source_frame_ids"), "contract_frame_ids")
    native_times = _integers(contract.get("qpc_ns"), "contract_qpc_ns")
    if not np.array_equal(ids, native_ids) or not np.array_equal(times, native_times) or np.any(times <= anchor):
        raise DeploymentAdapterError("SCHEDULE_NATIVE_H50_MAPPING_MISMATCH")
    # The existing client executes a fixed-rate chunk, not per-row timestamps.
    # Allow only integer-nanosecond rounding of its explicitly declared period;
    # never resample or alter the source prediction mapping to make it fit.
    if np.any(np.abs(np.diff(times).astype(np.float64) - 1e9 / hz) > 1):
        raise DeploymentAdapterError("NATIVE_SCHEDULE_NOT_COMPATIBLE_WITH_FIXED_RATE_CLIENT")
    return ids, times


def _client_timing(execution, *, contract, metadata, context, times, allowed):
    if (not isinstance(execution, Mapping) or execution.get("schema") != "humanego-client-execution-timing-v1"
            or execution.get("status") != "PASSED"):
        raise DeploymentAdapterError("CLIENT_EXECUTION_TIMING_UNQUALIFIED")
    _fixture(execution, allowed, "client_execution")
    for key in ("source_sha256", "runtime_source_sha256", "runtime_configuration_sha256"):
        _sha(execution.get(key), key)
    for key in ("runtime_mode", "smoothing_mode"):
        _text(execution.get(key), key)
    _int(execution.get("latency_k"), "latency_k")
    if (execution.get("request_context") != context
            or execution.get("policy_id") != metadata["policy_id"]
            or execution.get("clock_domain") != contract["clock_domain"]
            or execution.get("anchor_qpc_ns") != contract["anchor_qpc_ns"]
            or execution.get("source_signature") != contract["source_signature"]
            or execution.get("timestamp_unit") != "nanoseconds"
            or execution.get("timebase_verified") is not True
            or execution.get("control_hz") != metadata["control_hz"]):
        raise DeploymentAdapterError("CLIENT_EXECUTION_CLOCK_REQUEST_OR_POLICY_MISMATCH")
    # An existing smoother can change values using previous chunks or limits.
    # This thin adapter does not recreate it from incomplete source snippets.
    if execution.get("action_processing") != "INDEX_ONLY_NO_VALUE_CHANGE":
        raise DeploymentAdapterError("CLIENT_POSTPROCESSING_UNQUALIFIED")
    rows = np.asarray(execution.get("response_row_indices"))
    publish = np.asarray(execution.get("publish_qpc_ns"))
    if (rows.ndim != 1 or not 1 <= len(rows) <= 30 or rows.dtype.kind not in "iu"
            or np.any(rows < 0) or np.any(rows >= 30) or np.any(np.diff(rows.astype(np.int64)) <= 0)
            or publish.shape != rows.shape or publish.dtype.kind not in "iu"
            or np.any(publish < 0) or np.any(publish > np.iinfo(np.int64).max)):
        raise DeploymentAdapterError("INVALID_CLIENT_EXECUTION_INDEX_TIME_MAPPING")
    rows, publish = rows.astype(np.int64, copy=True), publish.astype(np.int64, copy=True)
    first = _int(execution.get("first_publish_qpc_ns"), "first_publish_qpc_ns")
    if not np.array_equal(publish, times[rows]) or first != int(publish[0]):
        raise DeploymentAdapterError("CLIENT_EXECUTION_NATIVE_PREDICTION_TIME_MISMATCH")
    return {"response_row_indices": rows, "publish_qpc_ns": publish, "first_publish_qpc_ns": first,
            "source_sha256": execution["source_sha256"], "runtime_source_sha256": execution["runtime_source_sha256"],
            "runtime_configuration_sha256": execution["runtime_configuration_sha256"],
            "runtime_mode": execution["runtime_mode"], "smoothing_mode": execution["smoothing_mode"],
            "latency_k": execution["latency_k"], "action_processing": execution["action_processing"]}


def _selected_command_limits(actions, execution, measured_initial, calibration):
    """Cropping may turn individually legal native steps into an illegal jump."""
    previous = _array(measured_initial["q_rad"], (14,), "initial_q_rad")
    previous_time = _int(measured_initial["qpc_ns"], "initial_qpc_ns")
    speed = _array(calibration["joint_speed_max_rad_s"], (14,), "joint_speed_max_rad_s")
    step = _array(calibration["joint_step_max_rad"], (14,), "joint_step_max_rad")
    for row, timestamp in zip(execution["response_row_indices"], execution["publish_qpc_ns"]):
        q = np.concatenate((actions[row, :7], actions[row, 13:20]))
        dt = (int(timestamp) - previous_time) / 1e9
        delta = np.abs(q - previous)
        if dt <= 0 or np.any(delta > step) or np.any(delta > speed * dt):
            raise DeploymentAdapterError("CLIENT_EXECUTION_STEP_OR_SPEED_LIMIT")
        previous, previous_time = q, int(timestamp)


def dry_run_protocol_chunk(actions, *, contract, calibration, measured_initial,
                           ik_provider, schedule, policy_metadata, request_context,
                           client_execution=None, allow_synthetic_fixture=False):
    """H50/30D -> guarded offline IK -> H30/26D v2 response dictionary only.

    Verified source snapshots B_011/B_012 echo request_context and policy_id;
    B_073/B_074 use request_context={'sequence': N}, not request_seq. B_046
    requires the exact 26D absolute-joint layout and a matching control_hz.
    This envelope is not a transport or physical deployment authorization.
    Cropping/skipping creates edges absent from the native sequential IK checks;
    step/speed revalidation does not certify collision-free selected paths.
    """
    base = {"schema": "humanego-tianji-revo2-protocol-dry-run-v1", "accepted": False,
            "response": None, "errors": [], "bridge": None, "fixture_only": bool(allow_synthetic_fixture),
            "offline_only": True, "control_ground_truth": False,
            "physical_deployment_authorized": False, "robot_commands_sent": 0}
    try:
        hz = _protocol_metadata(policy_metadata)
        if (not isinstance(request_context, Mapping) or set(request_context) != {"sequence"}
                or type(request_context["sequence"]) is not int or request_context["sequence"] <= 0):
            raise DeploymentAdapterError("EXACT_V2_REQUEST_CONTEXT_SEQUENCE_REQUIRED")
        if not isinstance(contract, Mapping):
            raise DeploymentAdapterError("ACTION_CONTRACT_REQUIRED")
        _frames(calibration)
        _fixture(calibration, allow_synthetic_fixture, "calibration")
        _feedback_qualification(measured_initial)
        _fixture(measured_initial, allow_synthetic_fixture, "measured_initial")
        if (getattr(ik_provider, "robot_model_sha256", None) != calibration["robot_model_sha256"]
                or tuple(getattr(ik_provider, "joint_order", ())) != JOINT_NAMES):
            raise DeploymentAdapterError("IK_ROBOT_MODEL_OR_JOINT_ORDER_MISMATCH")
        ids, times = _schedule(schedule, contract, policy_metadata, hz, allow_synthetic_fixture)
        execution = _client_timing(client_execution, contract=contract, metadata=policy_metadata,
            context=request_context, times=times, allowed=allow_synthetic_fixture)
        result = bridge_action_chunk(actions, contract=contract, calibration=calibration,
            measured_initial=measured_initial, ik_provider=ik_provider, output_horizon=30,
            allow_synthetic_fixture=allow_synthetic_fixture)
        base["bridge"] = result.to_dict()
        if not result.accepted:
            base["errors"] = sorted({error for row in result.frame_errors for error in row})
            return base
        if result.actions.shape != (30, 26) or not np.isfinite(result.actions).all():
            raise DeploymentAdapterError("INVALID_BRIDGE_26D_RESPONSE")
        _selected_command_limits(result.actions, execution, measured_initial, calibration)
        selected = execution["response_row_indices"]
        collision_qualification = (
            "UNKNOWN_REQUIRES_SELECTED_EDGE_REVALIDATION"
            if selected[0] > 0 or np.any(np.diff(selected) != 1)
            else "ORIGINAL_NATIVE_EDGE_CHECKS_ONLY")
        base.update(accepted=True,
            response={"actions": result.actions.copy(), "policy_id": policy_metadata["policy_id"],
                      "request_context": deepcopy(dict(request_context))},
            input_source_frame_ids=ids.copy(), input_qpc_ns=times.copy(),
            output_source_frame_ids=ids[:30].copy(), output_qpc_ns=times[:30].copy(),
            output_horizon=30, omitted_suffix_frames=20,
            output_time_status="EXPLICIT_NATIVE_H50_PREFIX_FIXED_RATE_COMPATIBLE",
            client_execution=execution,
            selected_path_collision_qualification=collision_qualification,
            client_selected_source_frame_ids=ids[execution["response_row_indices"]].copy(),
            client_selected_qpc_ns=execution["publish_qpc_ns"].copy(),
            policy_metadata=deepcopy(dict(policy_metadata)),
            artifact_qualification="DECLARATIONS_VALIDATED_REQUIRES_UPSTREAM_SHA_AUDIT")
        return base
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        base["errors"] = [str(exc)]
        return base
