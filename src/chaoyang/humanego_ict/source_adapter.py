"""Explicit native-producer to DERIVED_RETARGET source adapter.

This is preparation, not calibration, RGB admission, independent-group proof or
real-control supervision. It writes nothing, loads no model and ignores digital
robot actions. A registered preparation consumer must publish returned arrays.
"""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Mapping
import zipfile

import numpy as np

from .data import TrainingDataError, _object, _pose_clock_declaration, _proper_se3, _shape, reference_bytes

SCHEMA = "pick-place-native-derived-source-adapter-v1"
SOURCE_ROLES = {
    "controller": "native__controller",
    "controller_valid": "native__controller_valid",
    "hand_joints": "native__manus_local_25",
    "hand_valid": "native__manus_valid",
    "qpc": "qpc_ns",
    "frame_id": "mapping__frame_id",
    "controller_source_qpc": "mapping__pico_source_qpc_ns",
    "left_hand_source_qpc": "mapping__manus_left_source_qpc_ns",
    "right_hand_source_qpc": "mapping__manus_right_source_qpc_ns",
    "camera": "c2w",
}
LAYOUT_ROLES = {
    "source_to_canonical": "native__manus25_to21",
    "canonical_joint_names": "native__joint_names_21",
    "raw_joint_types": "native__raw_manus_joint_types",
    "qualification": "native__manus_topology_qualification",
}
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def array_content_sha256(value: np.ndarray) -> str:
    """Bind shape/dtype/bytes of an actual small topology array, not a name guess."""
    value = np.asarray(value)
    if value.dtype.kind == "O":
        raise TrainingDataError("object topology arrays forbidden")
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + np.ascontiguousarray(value).tobytes()).hexdigest()


def _selected_npz(reference, keys):
    # Resource ceiling is an I/O bound, not a numerical quality threshold.
    if not isinstance(reference, Mapping) or reference.get("bytes", MAX_SOURCE_BYTES + 1) > MAX_SOURCE_BYTES:
        raise TrainingDataError("bounded numeric source NPZ required")
    encoded = reference_bytes(reference)
    try:
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            entries = archive.infolist()
            if sum(entry.file_size for entry in entries) > MAX_SOURCE_BYTES:
                raise TrainingDataError("source NPZ uncompressed resource ceiling exceeded")
        with np.load(io.BytesIO(encoded), allow_pickle=False) as source:
            if not set(keys).issubset(source.files):
                raise TrainingDataError(f"native producer roles missing: {sorted(set(keys) - set(source.files))}")
            # Only selected native and small topology arrays are materialized.
            # Even a stray RGB/digital-q member is not accessed.
            result = {key: source[key] for key in keys}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        if isinstance(error, TrainingDataError):
            raise
        raise TrainingDataError("invalid bounded numeric source NPZ") from error
    if any(value.dtype.kind == "O" for value in result.values()):
        raise TrainingDataError("object/pickled source arrays forbidden")
    return result


def _array_headers(reference, keys):
    encoded = reference_bytes(reference)
    if len(encoded) > MAX_SOURCE_BYTES:
        raise TrainingDataError("bounded numeric source NPZ required")
    metadata = {}
    try:
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > MAX_SOURCE_BYTES:
                raise TrainingDataError("source NPZ uncompressed resource ceiling exceeded")
            for key in keys:
                with archive.open(key + ".npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise TrainingDataError("unsupported source NPY header version")
                metadata[key] = {"shape": list(shape), "dtype": dtype.str}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        if isinstance(error, TrainingDataError):
            raise
        raise TrainingDataError("invalid native source NPZ header") from error
    return metadata


def _session(source_ref, session_input_ref):
    session = _object(reference_bytes(session_input_ref))
    declared = session.get("arrays", {})
    if (declared.get("sha256") != source_ref["sha256"]
            or declared.get("bytes") != source_ref["bytes"]):
        raise TrainingDataError("native source bytes/SHA differ from producer SESSION_INPUT")
    if session.get("schema_version") != "pick-place-raw-input-v2-canonical21":
        raise TrainingDataError("explicit canonical21 native producer schema required")
    if not isinstance(session.get("session_id"), str) or not session["session_id"].strip():
        raise TrainingDataError("native producer session identity missing")
    calibration = session.get("calibration")
    if (not isinstance(calibration, Mapping)
            or calibration.get("control_ground_truth") is not False
            or calibration.get("actual_motion_source") != "RAW_PICO_CONTROLLER_PLUS_RAW_MANUS_INSTALLATION_PRIOR"):
        raise TrainingDataError("source must be declared human-controller installation prior, not control GT")
    return session


def _native(source_ref):
    arrays = _selected_npz(source_ref, [*SOURCE_ROLES.values(), *LAYOUT_ROLES.values()])
    count = arrays[SOURCE_ROLES["qpc"]].size
    for role, shape, kind in (
        ("controller", (count, 2, 4, 4), "float"),
        ("controller_valid", (count, 2), "bool"),
        ("hand_joints", (count, 2, 25, 3), "float"),
        ("hand_valid", (count, 2), "bool"),
        ("qpc", (count,), "int"), ("frame_id", (count,), "int"),
        ("controller_source_qpc", (count,), "int"),
        ("left_hand_source_qpc", (count,), "int"),
        ("right_hand_source_qpc", (count,), "int"),
        ("camera", (count, 4, 4), "float"),
    ):
        _shape(arrays[SOURCE_ROLES[role]], shape, role, kind)
    for role in ("qpc", "frame_id", "controller_source_qpc", "left_hand_source_qpc", "right_hand_source_qpc"):
        value = arrays[SOURCE_ROLES[role]]
        if value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max):
            raise TrainingDataError("native index/QPC exceeds signed int64 range")
    qpc, frames = arrays[SOURCE_ROLES["qpc"]], arrays[SOURCE_ROLES["frame_id"]]
    if (count < 51 or np.any(qpc < 0) or np.any(frames < 0)
            or np.any(np.diff(qpc.astype(np.int64)) <= 0)
            or np.any(np.diff(frames.astype(np.int64)) <= 0)):
        raise TrainingDataError("native source requires monotone frames/QPC and H50+anchor")
    return arrays


def _layout(arrays):
    mapping = arrays[LAYOUT_ROLES["source_to_canonical"]]
    names = arrays[LAYOUT_ROLES["canonical_joint_names"]]
    types = arrays[LAYOUT_ROLES["raw_joint_types"]]
    qualification = arrays[LAYOUT_ROLES["qualification"]]
    _shape(mapping, (21,), "source_to_canonical", "int")
    if (np.any(mapping < 0) or np.any(mapping >= 25) or len(np.unique(mapping)) != 21
            or names.shape != (21,) or types.shape != (25,) or qualification.shape != ()
            or any(value.dtype.kind not in "US" for value in (names, types, qualification))
            or any(not str(value).strip() for value in [*names, *types])
            or not str(qualification.item()).strip()):
        raise TrainingDataError("native topology roles must be explicit nonempty verified small arrays")
    return {"roles": dict(LAYOUT_ROLES),
            "array_sha256": {role: array_content_sha256(arrays[key]) for role, key in LAYOUT_ROLES.items()},
            "source_to_canonical": mapping.tolist(), "canonical_joint_names": names.tolist(),
            "raw_joint_types": types.tolist(), "qualification": str(qualification.item()),
            "interpretation": "PRODUCER_DECLARATION_NOT_MANUFACTURER_OR_PHYSICAL_CERTIFICATION"}


def inspect_native_source_metadata(source_ref: Mapping[str, Any], session_input_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Small native NPZ/header/topology audit; no conversion, RGB or model call."""
    session = _session(source_ref, session_input_ref)
    headers = _array_headers(source_ref, [*SOURCE_ROLES.values(), *LAYOUT_ROLES.values()])
    # Inspect only tiny topology and timestamp arrays, never pose/joint/RGB payloads.
    small_keys = [*LAYOUT_ROLES.values(), SOURCE_ROLES["qpc"], SOURCE_ROLES["frame_id"],
                  SOURCE_ROLES["controller_source_qpc"], SOURCE_ROLES["left_hand_source_qpc"],
                  SOURCE_ROLES["right_hand_source_qpc"]]
    arrays = _selected_npz(source_ref, small_keys)
    qpc = arrays[SOURCE_ROLES["qpc"]]
    if qpc.ndim != 1 or session.get("frame_count") != len(qpc):
        raise TrainingDataError("producer frame_count differs from native source")
    for role in ("qpc", "frame_id", "controller_source_qpc", "left_hand_source_qpc", "right_hand_source_qpc"):
        _shape(arrays[SOURCE_ROLES[role]], qpc.shape, role, "int")
    calibration = session["calibration"]
    return {"schema": "pick-place-native-derived-source-inspection-v1", "session_id": session["session_id"],
            "input_refs": {"native_source": dict(source_ref), "session_input": dict(session_input_ref)},
            "frame_count": len(qpc), "source_roles": dict(SOURCE_ROLES),
            "array_metadata": headers,
            "hand_layout": _layout(arrays),
            "future_nearest_sources": {
                role: int(np.count_nonzero(arrays[SOURCE_ROLES[role]] > qpc))
                for role in ("controller_source_qpc", "left_hand_source_qpc", "right_hand_source_qpc")},
            "producer_qualification": dict(calibration),
            "independent_source_group": "NOT_GRANTED_BY_SOURCE_ADAPTER",
            "missing_items": ["EXPLICIT_ADAPTER_CONTRACT_REQUIRED", "EXPLICIT_SEGMENT_PROVENANCE_REQUIRED",
                              "INDEPENDENT_SOURCE_GROUP_NOT_ESTABLISHED"],
            "training_start_authorized": False, "control_ground_truth": False,
            "physical_deployment_authorized": False}


def _time_gate(contract, session):
    ref = contract.get("time_gate_source_ref")
    declared = session.get("source_refs", {}).get("dataset.hdf5", {})
    if (not isinstance(ref, Mapping) or ref.get("sha256") != declared.get("sha256")
            or ref.get("bytes") != declared.get("bytes")):
        raise TrainingDataError("past time gate must bind actual producer HDF bytes/SHA")
    if ref["bytes"] > MAX_SOURCE_BYTES:
        raise TrainingDataError("bounded HDF attributes source required")
    encoded = reference_bytes(ref)
    try:
        import h5py
        with h5py.File(io.BytesIO(encoded), "r") as source:
            # No HDF datasets are accessed. Keep all original time-gate values.
            attributes = {key: float(source.attrs[key]) for key in
                          ("gate_ms", "max_skew_ms", "max_p95_skew_ms")}
            clock = str(source.attrs["alignment_clock"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise TrainingDataError("original source HDF time-gate attributes missing/invalid") from error
    if clock != "host_qpc" or any(not np.isfinite(value) or value <= 0 for value in attributes.values()):
        raise TrainingDataError("original time gate must declare one host-QPC clock and positive bounds")
    maximum = int(round(min(attributes["gate_ms"], attributes["max_skew_ms"]) * 1_000_000))
    if type(contract.get("current_max_past_skew_ns")) is not int or contract["current_max_past_skew_ns"] != maximum:
        raise TrainingDataError("adapter past time gate differs from original HDF numeric gate")
    return maximum, {**attributes, "alignment_clock": clock, "source_ref": dict(ref),
                     "timebase_qualification": "NOT_VERIFIED_BY_READING_DECLARED_ATTRIBUTES"}


def _prior(prior):
    if (not isinstance(prior, Mapping) or prior.get("schema") != "controller_to_wrist_v1"
            or prior.get("convention") != "wrist_pose = compose_pose(controller_pose, transform)"
            or prior.get("teleop_wrist") is not False):
        raise TrainingDataError("explicit non-teleop installation prior required")
    transforms = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    for hand, side in enumerate(("left", "right")):
        try:
            position = np.asarray(prior[side]["pos"], dtype=np.float64)
            quaternion = np.asarray(prior[side]["quat"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingDataError("installation prior position/xyzw quaternion missing") from error
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5)):
            raise TrainingDataError("installation prior requires finite meters and unit xyzw quaternion")
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        transforms[hand, :3, :3] = [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ]
        transforms[hand, :3, 3] = position
    _proper_se3(transforms, "source installation prior")
    return transforms


def adapt_native_source(source_ref: Mapping[str, Any], session_input_ref: Mapping[str, Any],
                        adapter_contract_ref: Mapping[str, Any], *, anchor_row: int,
                        allow_synthetic_fixture: bool = False) -> dict[str, Any]:
    """Return one anchor+H50 source window, with explicit PAST_ONLY current input.

    The caller supplies SHA-bound segment labels/provenance; this adapter never
    invents session independence, segment boundaries or clock qualification.
    Derived future labels retain original nearest observations and their clocks.
    """
    session = _session(source_ref, session_input_ref)
    arrays = _native(source_ref)
    contract = _object(reference_bytes(adapter_contract_ref))
    if session.get("frame_count") != len(arrays[SOURCE_ROLES["qpc"]]):
        raise TrainingDataError("producer frame_count differs from native source")
    if contract.get("fixture_only") is True and not allow_synthetic_fixture:
        raise TrainingDataError("synthetic source adapter contract is not production evidence")
    if (contract.get("schema") != SCHEMA or contract.get("supervision_class") != "DERIVED_RETARGET"
            or contract.get("control_ground_truth") is not False
            or contract.get("physical_deployment_authorized") is not False
            or contract.get("native_source_sha256") != source_ref["sha256"]
            or contract.get("session_input_sha256") != session_input_ref["sha256"]
            or contract.get("source_roles") != SOURCE_ROLES):
        raise TrainingDataError("explicit SHA-bound native DERIVED_RETARGET contract required")
    pose_clock = contract.get("pose_clock")
    if not isinstance(pose_clock, Mapping):
        raise TrainingDataError("explicit source pose_clock object required")
    _pose_clock_declaration(pose_clock)
    calibration = session["calibration"]
    if (pose_clock["pose_frame_id"] != calibration.get("world_frame")
            or contract.get("source_timebase_status") != calibration.get("timebase_status")
            or contract.get("camera_role") != "c2w"
            or contract.get("hand_joint_position_unit") != "meters"
            or contract.get("hand_joint_coordinate_frame") != "wrist_local"
            or contract.get("reference_camera_policy") != "ANCHOR_NATIVE_C2W"
            or contract.get("current_source_selection") != "PAST_ONLY_SAME_SEGMENT_LATEST_SOURCE_QPC"
            or contract.get("target_source_selection") != "PRESERVE_NATIVE_NEAREST_WITH_TRACE"):
        raise TrainingDataError("native pose/camera/clock/selection contract differs")
    evidence = contract.get("evidence_refs")
    if not isinstance(evidence, list) or not evidence:
        raise TrainingDataError("source adapter requires actual evidence_refs")
    for ref in evidence:
        reference_bytes(ref)
    max_past_skew, time_gate = _time_gate(contract, session)
    expected_layout = _layout(arrays)
    layout_binding = contract.get("hand_layout", {})
    if (not isinstance(layout_binding, Mapping)
            or layout_binding.get("native_source_sha256") != source_ref["sha256"]
            or layout_binding.get("roles") != LAYOUT_ROLES
            or layout_binding.get("array_sha256") != expected_layout["array_sha256"]):
        raise TrainingDataError("source topology role/SHA binding differs from actual small arrays")
    prior = calibration.get("installation_prior_used")
    if contract.get("installation_prior") != prior:
        raise TrainingDataError("adapter cannot substitute or refit installation prior")
    transforms = _prior(prior)
    binding = contract.get("segment_binding", {})
    if (not isinstance(binding, Mapping)
            or binding.get("native_source_sha256") != source_ref["sha256"]
            or not isinstance(binding.get("rule"), str) or not binding["rule"].strip()
            or not isinstance(binding.get("evidence_refs"), list) or not binding["evidence_refs"]):
        raise TrainingDataError("explicit source-bound segment rule/evidence required")
    for ref in binding["evidence_refs"]:
        reference_bytes(ref)
    segments = _selected_npz(binding["source_ref"], ("source_frame_id", "segment_id"))
    frame_ids = arrays[SOURCE_ROLES["frame_id"]]
    _shape(segments["source_frame_id"], frame_ids.shape, "segment frames", "int")
    _shape(segments["segment_id"], frame_ids.shape, "segment_id", "int")
    if (not np.array_equal(segments["source_frame_id"], frame_ids)
            or np.any(segments["segment_id"] < 0)
            or np.any(segments["segment_id"] > np.iinfo(np.int64).max)):
        raise TrainingDataError("segment labels do not bind native frame identities")
    segment_runs = segments["segment_id"][np.r_[True, np.diff(segments["segment_id"]) != 0]]
    if len(np.unique(segment_runs)) != len(segment_runs):
        raise TrainingDataError("segment IDs cannot be reused across disconnected source runs")
    if type(anchor_row) is not int or anchor_row < 0 or anchor_row + 50 >= len(frame_ids):
        raise TrainingDataError("native anchor requires H50+anchor")
    rows = np.arange(anchor_row, anchor_row + 51, dtype=np.int64)
    segment_ids = segments["segment_id"]
    if (not np.array_equal(frame_ids[rows], np.arange(int(frame_ids[anchor_row]), int(frame_ids[anchor_row]) + 51))
            or np.any(segment_ids[rows] != segment_ids[anchor_row])):
        raise TrainingDataError("native H50 crosses a frame gap or declared segment")
    controllers = arrays[SOURCE_ROLES["controller"]][rows].copy()
    controller_valid = arrays[SOURCE_ROLES["controller_valid"]][rows].copy()
    joints = arrays[SOURCE_ROLES["hand_joints"]][rows].copy()
    hand_valid = arrays[SOURCE_ROLES["hand_valid"]][rows].copy()
    controller_times_all = arrays[SOURCE_ROLES["controller_source_qpc"]].astype(np.int64)
    hand_times_all = np.stack((arrays[SOURCE_ROLES["left_hand_source_qpc"]],
                              arrays[SOURCE_ROLES["right_hand_source_qpc"]]), axis=1).astype(np.int64)
    controller_times = np.repeat(controller_times_all[rows, None], 2, axis=1)
    hand_times = hand_times_all[rows].copy()
    qpc = arrays[SOURCE_ROLES["qpc"]].astype(np.int64)
    if np.any(controller_times[controller_valid] < 0) or np.any(hand_times[hand_valid] < 0):
        raise TrainingDataError("observed native source requires explicit nonnegative source QPC")
    anchor = int(qpc[anchor_row])
    selected_rows = {"controllers": [-1, -1], "hands": [-1, -1]}
    missing = {"INDEPENDENT_SOURCE_GROUP_NOT_ESTABLISHED", "RGB_CAUSAL_ADMISSION_NOT_GRANTED",
               "SELECTED_CURRENT_AGGREGATE_TIME_QUALITY_NOT_QUALIFIED"}
    controller_time_fail = controller_valid & (
        (controller_times > qpc[rows, None] + max_past_skew)
        | (controller_times < qpc[rows, None] - max_past_skew))
    hand_time_fail = hand_valid & (
        (hand_times > qpc[rows, None] + max_past_skew)
        | (hand_times < qpc[rows, None] - max_past_skew))
    if controller_time_fail[1:].any():
        missing.add("BLOCKED_NATIVE_CONTROLLER_TIME_GATE_VIOLATION")
        controller_valid[1:] &= ~controller_time_fail[1:]
    if hand_time_fail[1:].any():
        missing.add("BLOCKED_NATIVE_HAND_TIME_GATE_VIOLATION")
        hand_valid[1:] &= ~hand_time_fail[1:]
    for hand in range(2):
        for kind, all_times, validity_role, values_role, values, valid, times in (
            ("controllers", controller_times_all, "controller_valid", "controller", controllers, controller_valid, controller_times),
            ("hands", hand_times_all[:, hand], "hand_valid", "hand_joints", joints, hand_valid, hand_times),
        ):
            candidates = np.flatnonzero((np.arange(len(frame_ids)) <= anchor_row)
                & (segment_ids == segment_ids[anchor_row]) & arrays[SOURCE_ROLES[validity_role]][:, hand]
                & (all_times >= 0) & (all_times <= anchor)
                & (all_times >= anchor - max_past_skew))
            if not len(candidates):
                valid[0, hand] = False
                times[0, hand] = -1
                missing.add(f"UNKNOWN_PAST_CURRENT_{kind.upper()}_{hand}")
                values[0, hand] = np.eye(4) if kind == "controllers" else 0
                continue
            # Source timestamps, not nearest video row, determine past choice.
            latest = np.max(all_times[candidates])
            chosen = int(candidates[np.flatnonzero(all_times[candidates] == latest)[-1]])
            selected_rows[kind][hand] = chosen
            values[0, hand] = arrays[SOURCE_ROLES[values_role]][chosen, hand]
            valid[0, hand] = True
            times[0, hand] = all_times[chosen]
    _proper_se3(controllers[controller_valid], "observed native controllers")
    camera = arrays[SOURCE_ROLES["camera"]][anchor_row].copy()
    _proper_se3(camera, "declared native anchor camera")
    if not np.isfinite(joints[hand_valid]).all():
        raise TrainingDataError("valid native hand source contains nonfinite joints")
    if np.any(controller_times[controller_valid] < 0) or np.any(hand_times[hand_valid] < 0):
        raise TrainingDataError("observed native source requires explicit nonnegative source QPC")
    if calibration.get("physical_eye_identity_verified") is not True:
        missing.add("BLOCKED_CAMERA_EYE_IDENTITY_UNQUALIFIED")
    if calibration.get("training_eligible") is not True:
        missing.add("BLOCKED_PRODUCER_CALIBRATION_NOT_TRAINING_QUALIFIED")
    if calibration.get("timebase_status") != "VERIFIED_HOST_QPC_NS":
        missing.add("UNKNOWN_SOURCE_TIMEBASE_QUALIFICATION")
    source_arrays = {
        "controller_pose_world": controllers, "controller_to_wrist": transforms,
        "controller_valid": controller_valid, "hand_joints_wrist": joints,
        "hand_joints_valid": hand_valid, "qpc_ns": qpc[rows].copy(),
        "source_frame_id": frame_ids[rows].astype(np.int64), "segment_id": segment_ids[rows].astype(np.int64),
        "controller_source_qpc_ns": controller_times, "hand_source_qpc_ns": hand_times,
        "reference_camera_to_world": camera,
    }
    for value in source_arrays.values():
        value.flags.writeable = False
    trace = {"session_id": session["session_id"], "native_anchor_row": anchor_row,
        "output_anchor_row": 0, "anchor_qpc_ns": anchor, "native_window_rows": rows.tolist(),
        "native_window_frame_ids": frame_ids[rows].tolist(), "current_selected_native_rows": selected_rows,
        "original_nearest_controller_source_qpc_ns": int(controller_times_all[anchor_row]),
        "original_nearest_hand_source_qpc_ns": hand_times_all[anchor_row].tolist(),
        "original_native_time_gate_violation_counts": {
            "controllers": controller_time_fail.sum(axis=0).tolist(),
            "hands": hand_time_fail.sum(axis=0).tolist()},
        "selected_current_controller_source_qpc_ns": controller_times[0].tolist(),
        "selected_current_hand_source_qpc_ns": hand_times[0].tolist(),
        "pose_clock": dict(pose_clock), "hand_layout": expected_layout,
        "hand_joint_position_unit": contract["hand_joint_position_unit"],
        "hand_joint_coordinate_frame": contract["hand_joint_coordinate_frame"],
        "original_time_gates": time_gate,
        "selected_current_controller_age_ns": [anchor - int(value) if value >= 0 else None for value in controller_times[0]],
        "selected_current_hand_age_ns": [anchor - int(value) if value >= 0 else None for value in hand_times[0]],
        "input_refs": {"native_source": dict(source_ref), "session_input": dict(session_input_ref),
                       "adapter_contract": dict(adapter_contract_ref), "segment_labels": dict(binding["source_ref"]),
                       "time_gate_source": dict(contract["time_gate_source_ref"])},
        "installation_authority": "INSTALLATION_PRIOR_NOT_MEASURED_ANATOMICAL_WRIST",
        "producer_qualification": dict(calibration),
        "qualification_preservation": "ALL_ORIGINAL_FAIL_UNKNOWN_REMAIN; NUMERIC_ADAPTATION_IS_NOT_ADMISSION"}
    return {"schema": "pick-place-native-derived-source-window-v1", "source_arrays": source_arrays,
            "trace": trace, "missing_items": sorted(missing), "status": "DERIVED_SOURCE_PREPARED_NOT_QUALIFIED",
            "supervision_class": "DERIVED_RETARGET", "control_ground_truth": False,
            "physical_deployment_authorized": False, "training_start_authorized": False}
