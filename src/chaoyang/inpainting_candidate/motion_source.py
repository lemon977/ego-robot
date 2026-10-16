"""Candidate Controller/Head + MANUS source bridge, without task-name routing.

Reuse frozen V1 raw parsing/composition; never read old actions, Mask, Clean,
placement or a hardware-retarget field. A new manifest identity is accepted
without renaming it to pick_place. V1 fitted camera/installation contracts are
usable only within their exact adopted device/snapshot domain. Unknown encoded
camera axes block that Robot consumer, not native motion or RGB Mask/Clean.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from chaoyang.pick_place_1008 import source as parent
from chaoyang.pick_place_1008.camera import normalize_camera
from .contracts import SCHEMA, canonical_sha, file_ref, verify_ref
from .source import RAW_FILES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V1_MANIFEST = PROJECT_ROOT / "inpainting_v1/MANIFEST.json"
REQUIRED_MOTION_ROLES = ("manifest", "video", "controller_head", "manus", "manus_meta",
                         "camera_meta", "installation_container", "video_qpc", "video_wall")
PARENT_CODE = ("src/chaoyang/pick_place_1008/source.py",
               "src/chaoyang/pick_place_1008/camera.py",
               "src/chaoyang/pipeline/pico_manus_motion_v2.py",
               "src/chaoyang/exact78_baseline/pico_source.py",
               "src/chaoyang/inpainting_candidate/contracts.py",
               "src/chaoyang/inpainting_candidate/source.py")


class MotionSourceError(ValueError):
    pass


def _read(pin: Mapping) -> Any:
    verify_ref(pin)
    return json.loads(Path(pin["path"]).read_text(encoding="utf-8-sig"))


def _json(path: Path, value: Any) -> dict:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return file_ref(path)


def _npz(path: Path, arrays: Mapping) -> dict:
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise MotionSourceError("OBJECT_ARRAYS_FORBIDDEN")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return file_ref(path)


def _safe_root(path: Path) -> Path:
    path = Path(path).resolve()
    if path == PROJECT_ROOT.resolve() or not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise MotionSourceError("MOTION_OUTPUT_OUTSIDE_PROJECT")
    return path


def adopted_camera_binding(source_report: Mapping, *, prior_sha256: str,
                            v1_manifest_ref: Mapping | None = None) -> dict:
    """Read only the sole adopted V1 contract; no caller-provided task config.

    A non-overridden device consumes THIS capture's verified raw prior; no fitted
    installation is invented. An overridden device additionally requires its
    exact original-prior SHA. Camera fits remain visual estimates, not calibration.
    """
    result = {"status": "BLOCKED_CAMERA_AXES", "config": None, "actual_contracts": [],
              "adopted_domain": False, "encoded_camera_axes": "UNKNOWN",
              "projection_qualified": False, "metric_calibration_verified": False,
              "prior_binding_status": "CURRENT_CAPTURE_RAW_PRIOR_ONLY",
              "mask_clean_consumer_status": "READY_INDEPENDENT_OF_CAMERA"}
    refs = source_report["actual_inputs"]
    if "camera" not in refs or "camera_meta" not in refs:
        result["failure_code"] = "RAW_CAMERA_METADATA_ABSENT"
        return result
    meta = _read(refs["camera_meta"])
    serial = meta.get("device_serial")
    result.update(device_serial=serial, camera_sha256=refs["camera"]["sha256"],
                  original_installation_prior_sha256=prior_sha256)
    if not isinstance(serial, str) or not serial:
        result["failure_code"] = "RAW_CAMERA_DEVICE_SERIAL_ABSENT"
        return result
    if meta.get("sha256") != refs["camera"]["sha256"]:
        raise MotionSourceError("RAW_CAMERA_SNAPSHOT_METADATA_SHA_MISMATCH")
    if v1_manifest_ref is None:
        result["failure_code"] = "NO_ADOPTED_CAMERA_CONTRACT_REQUESTED"
        return result
    if Path(v1_manifest_ref["path"]).resolve() != V1_MANIFEST.resolve():
        raise MotionSourceError("ONLY_CURRENT_ADOPTED_V1_MANIFEST_ALLOWED")
    manifest = _read(v1_manifest_ref)
    if manifest.get("baseline_id") != "INPAINTING_RAW_TO_ROBOT_V1":
        raise MotionSourceError("WRONG_ADOPTED_BASELINE")
    config_ref = manifest["inputs"]["camera"]
    if not Path(config_ref["path"]).resolve(strict=True).is_relative_to(PROJECT_ROOT):
        raise MotionSourceError("ADOPTED_CAMERA_CONTRACT_OUTSIDE_PROJECT")
    contract = _read(config_ref)
    result["actual_contracts"] = [dict(v1_manifest_ref), dict(config_ref)]
    selected = contract.get("devices", {}).get(serial)
    if not isinstance(selected, dict):
        result["failure_code"] = "DEVICE_NOT_IN_ADOPTED_V1_DOMAIN"
        return result
    if selected.get("device_serial") != serial or selected.get("camera_sha256") != refs["camera"]["sha256"]:
        result["failure_code"] = "CAMERA_SNAPSHOT_NOT_IN_ADOPTED_V1_DOMAIN"
        return result
    if selected.get("status") != "FROZEN_VISUAL_ESTIMATE":
        raise MotionSourceError("ADOPTED_CAMERA_ESTIMATE_STATUS_DRIFT")
    correction = selected.get("correction_evidence")
    if ("K_override" in selected or "T_head_camera_override" in selected):
        if not isinstance(correction, dict):
            raise MotionSourceError("ADOPTED_CAMERA_CORRECTION_METADATA_REQUIRED")
        if not Path(correction["path"]).resolve(strict=True).is_relative_to(PROJECT_ROOT):
            raise MotionSourceError("ADOPTED_CAMERA_CORRECTION_OUTSIDE_PROJECT")
        verify_ref(correction)
        result["actual_contracts"].append(dict(correction))
    if "installation_prior_override" in selected:
        if selected.get("installation_original_prior_sha256") != prior_sha256:
            result["failure_code"] = "FITTED_INSTALLATION_ORIGINAL_PRIOR_SHA_MISMATCH"
            return result
        if selected.get("installation_group_key") != serial + ":" + prior_sha256:
            result["failure_code"] = "FITTED_INSTALLATION_DEVICE_GROUP_MISMATCH"
            return result
        result["prior_binding_status"] = "EXACT_DEVICE_ORIGINAL_PRIOR_SHA_FITTED_OVERRIDE"
    else:
        result["prior_binding_status"] = "NO_FITTED_OVERRIDE_RAW_PRIOR_CONSUMED"
    # Never use the 1280-domain override with a 640-domain RawSession.
    if selected.get("output_size") != [1280, 960]:
        raise MotionSourceError("ADOPTED_K_NATIVE_ESTIMATE_DOMAIN_DRIFT")
    if selected.get("upright_axes") not in ("capture_sensor_from_upright", "identity"):
        result["failure_code"] = "ADOPTED_ENCODED_CAMERA_AXES_ABSENT"
        return result
    if (selected.get("eye") != "right" or type(selected.get("sbs_index")) is not int
            or selected["sbs_index"] not in (0, 1)):
        result["failure_code"] = "ADOPTED_CAMERA_EYE_ROUTE_INVALID"
        return result
    reported = source_report.get("camera", {})
    if (reported.get("physical_eye") != "right"
            or reported.get("sbs_index") != selected["sbs_index"]):
        result["failure_code"] = "RAW_RGB_AND_ADOPTED_CAMERA_EYE_ROUTE_MISMATCH"
        return result
    result.update(status="READY_ADOPTED_V1_VISUAL_DOMAIN", config=deepcopy(selected),
                  adopted_domain=True, encoded_camera_axes=selected["upright_axes"],
                  failure_code=None,
                  qualification="USER_ADOPTED_VISUAL_ESTIMATE_NOT_NEW_DOMAIN_OR_METRIC_PROOF")
    return result


def _adopted_override_metadata(selected: Mapping, original_prior: Mapping,
                               binding: dict) -> dict:
    """Consume the already-adopted values, not old fit inputs or a new fit.

    The old validator reads historical source.npz/annotations and therefore is
    deliberately not called by this fresh producer. This verifies only frozen
    metadata linkage and exact raw-prior identity; it confers no new geometry
    qualification or cross-device permission.
    """
    if selected.get("fit_quality_status") != "PASSED":
        raise MotionSourceError("ADOPTED_OVERRIDE_METADATA_STATUS_DRIFT")
    evidence = selected.get("installation_correction_evidence")
    if not isinstance(evidence, dict):
        raise MotionSourceError("ADOPTED_OVERRIDE_METADATA_REQUIRED")
    if not Path(evidence["path"]).resolve(strict=True).is_relative_to(PROJECT_ROOT):
        raise MotionSourceError("ADOPTED_OVERRIDE_METADATA_OUTSIDE_PROJECT")
    report = _read(evidence)
    group = report.get("groups", {}).get(selected["installation_group_key"], {})
    prior = selected["installation_prior_override"]
    if (report.get("schema") not in ("pick_place_fixed_encoded_k_right_installation_visual_fit_v1",
                                    "pick_place_installation_group_visual_fit_v1")
            or report.get("status") != "PASSED" or group.get("status") != "PASSED"
            or group.get("device_serial") != binding["device_serial"]
            or group.get("original_prior_sha256") != binding["original_installation_prior_sha256"]
            or group.get("prior_before") != original_prior or group.get("prior_used") != prior
            or prior.get("left") != original_prior.get("left")):
        raise MotionSourceError("ADOPTED_OVERRIDE_METADATA_PRIOR_LINK_MISMATCH")
    parent.installation_matrices(prior)
    binding["actual_contracts"].append(dict(evidence))
    binding["prior_binding_status"] = "ADOPTED_FROZEN_OVERRIDE_METADATA_ONLY"
    binding["installation_geometry_revalidated"] = False
    binding["historical_fit_inputs_read"] = False
    return deepcopy(prior)


def _raw_inputs(source_report: Mapping) -> tuple[Path, dict, str, np.ndarray, np.ndarray]:
    if source_report.get("status") != "SOURCE_PREFLIGHT_CANDIDATE":
        raise MotionSourceError("RAW_SOURCE_PREFLIGHT_REQUIRED")
    directory = Path(source_report["session_dir"]).resolve(strict=True)
    refs = dict(source_report["actual_inputs"])
    missing = set(REQUIRED_MOTION_ROLES) - refs.keys()
    if missing:
        raise MotionSourceError("RAW_MOTION_INPUTS_ABSENT:" + ",".join(sorted(missing)))
    if set(refs) - RAW_FILES.keys():
        raise MotionSourceError("NON_RAW_INPUT_ROLE_FORBIDDEN")
    for role, pin in refs.items():
        resolved = Path(pin["path"]).resolve(strict=True)
        if not resolved.is_relative_to(directory):
            raise MotionSourceError("RAW_INPUT_ESCAPES_CURRENT_CAPTURE")
        if resolved != (directory / RAW_FILES[role]).resolve(strict=True):
            raise MotionSourceError("RAW_READER_PATH_SIGNATURE_MISMATCH")
        verify_ref(pin)
    relative = directory.relative_to(PROJECT_ROOT) if directory.is_relative_to(PROJECT_ROOT) else directory
    if {part.lower() for part in relative.parts} & {"_run", "processed", "review", "attempts"}:
        raise MotionSourceError("HISTORICAL_DERIVED_MOTION_SOURCE_FORBIDDEN")
    identity = _read(refs["manifest"]).get("session")
    if (not isinstance(identity, str) or not identity
            or identity != source_report.get("session_id")):
        raise MotionSourceError("RAW_MANIFEST_IDENTITY_MISMATCH")
    # Check real sidecars; caller-provided count/map is not sufficient evidence.
    qpc_rows = parent.json_rows(Path(refs["video_qpc"]["path"]))
    wall_rows = parent.json_rows(Path(refs["video_wall"]["path"]))
    qpc = parent.timeline([row for _, row in qpc_rows], "motion_video")
    if any(type(row) is not int for _, row in wall_rows):
        raise MotionSourceError("RAW_WALL_INTEGER_VECTOR_REQUIRED")
    wall = np.asarray([row for _, row in wall_rows], dtype=np.int64)
    declared = source_report.get("frame_mapping", [])
    if (len(qpc) != len(wall) or len(qpc) != source_report.get("native_sidecar_frames")
            or len(declared) != len(qpc)
            or any(type(row.get("native_frame_index")) is not int
                   or row["native_frame_index"] != i
                   or row.get("qpc_declared_ns") != int(qpc[i])
                   for i, row in enumerate(declared))):
        raise MotionSourceError("RAW_MOTION_VIDEO_MAPPING_MISMATCH")
    return directory, refs, identity, qpc, wall


def _native_shapes(native: Mapping, mapping: Mapping, count: int) -> None:
    expected = {"head": (count, 4, 4), "head_display": (count, 4, 4),
        "head_source": (count,), "head_valid": (count,),
        "controller": (count, 2, 4, 4), "wrist": (count, 2, 4, 4),
        "manus_local_25": (count, 2, 25, 3), "hands": (2, count, 21, 3),
        "hand_source": (2, count), "controller_valid": (count, 2),
        "manus_valid": (count, 2), "hand_observed": (2, count),
        "hand_display_valid": (2, count), "complete_valid": (count,),
        "joint_observed_local": (count, 2, 21)}
    if any(key not in native or np.asarray(native[key]).shape != shape
           for key, shape in expected.items()):
        raise MotionSourceError("RAW_NATIVE_ARRAY_FRAME_CONTRACT_MISMATCH")
    if any(np.asarray(value).shape != (count,) for value in mapping.values()):
        raise MotionSourceError("RAW_NATIVE_ROUTE_MAPPING_FRAME_CONTRACT_MISMATCH")


def _session(identity, root, video_ref, qpc, camera, native, mapping, quality, installation, binding):
    count = len(qpc)
    fps = (count - 1) * 1e9 / int(qpc[-1] - qpc[0])
    calibration = dict(camera.calibration)
    calibration.update(actual_motion_source="RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
        installation=installation, time_quality=quality,
        timebase_status="TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
        hand_joint_schema=parent.JOINT_MAPPING_SCHEMA,
        manus25_to21=parent.MAP_25_TO_21.tolist(),
        manus_joint_mapping=parent.joint_mapping_metadata(native),
        body_source="ABSENT_UNCHANGED_PARENT_SHOULDER_ELBOW_PRIOR",
        wrist_observation_authority="CONTROLLER_PLUS_INSTALLATION_PRIOR_NOT_ANATOMICAL_TRUTH",
        camera_binding_status=binding["status"], training_eligible=False, control_ground_truth=False)
    valid = {key: native[key] for key in ("head_valid", "controller_valid", "manus_valid",
              "hand_display_valid", "complete_valid", "joint_observed_local")}
    return parent.RawSession(identity, root, Path(video_ref["path"]), fps, count,
        camera.width, camera.height, camera.K,
        native["head_display"] @ camera.T_head_camera,
        native["head_display"] @ camera.T_head_camera_factory,
        native["hands"], native["hand_observed"], native["hand_source"],
        np.zeros((count, 24, 3)), np.zeros(count, bool), np.array([0., 0., -1.]),
        calibration, qpc, mapping, valid, native)


def prepare_motion_source(source_report: Mapping, output_dir: Path, *,
                          component_signature: str,
                          v1_manifest_ref: Mapping | None = None) -> dict:
    """Seal native wrist/hand evidence; return RawSession only for supported camera.

    This function does not run retarget, load Robot/model assets, or fabricate a
    new camera basis. Root separately registers that ready consumer. Unsupported
    camera leaves native arrays auditable and Mask/Clean independent.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", str(component_signature)):
        raise MotionSourceError("MOTION_COMPONENT_SIGNATURE_REQUIRED")
    root = _safe_root(output_dir)
    if root.exists():
        raise MotionSourceError("MOTION_FRESH_OUTPUT_REQUIRED")
    directory, refs, identity, qpc, wall = _raw_inputs(source_report)
    prior, installation = parent.read_installation(directory)
    # Pin the actual HDF bytes referenced by this capture; datasets are not read.
    if installation["source_hdf5"] != refs["installation_container"]:
        raise MotionSourceError("RAW_INSTALLATION_CONTAINER_IDENTITY_DRIFT")
    if canonical_sha(prior) != installation["prior_sha256"]:
        raise MotionSourceError("RAW_INSTALLATION_ATTRIBUTE_SIGNATURE_DRIFT")
    binding = adopted_camera_binding(source_report, prior_sha256=installation["prior_sha256"],
                                     v1_manifest_ref=v1_manifest_ref)
    selected = binding["config"]
    camera = None
    used_prior = prior
    if selected is not None:
        if "installation_prior_override" in selected:
            used_prior = _adopted_override_metadata(selected, prior, binding)
            installation["override_evidence"] = selected["installation_correction_evidence"]
        camera = normalize_camera(_read(refs["camera"]), _read(refs["camera_meta"]),
            device_serial=binding["device_serial"], camera_sha256=refs["camera"]["sha256"],
            config=selected, output_size=(1280, 960))
    # Native composition itself needs no encoded-camera axes. Device serial is
    # taken from current metadata only, not a task/device fallback.
    serial = _read(refs["camera_meta"]).get("device_serial") if "camera_meta" in refs else None
    if not isinstance(serial, str) or not serial:
        raise MotionSourceError("RAW_CONTROLLER_IDENTITY_SERIAL_ABSENT")
    native, mapping, quality = parent.load_raw_motion(directory, qpc,
                                                     expected_serial=serial, prior=used_prior)
    _native_shapes(native, mapping, len(qpc))
    installation["used_prior_sha256"] = canonical_sha(used_prior)
    installation["used_prior"] = used_prior
    mapping = {**mapping, "video_wall_ns": wall, "video_pts_us": parent.pts_from_qpc(qpc)}
    if len(qpc) < 2:
        camera = None
        binding.update(status="BLOCKED_NATIVE_DURATION", failure_code="PARENT_MOTION_NEEDS_TWO_FRAMES")
    raw_session = (_session(identity, root, refs["video"], qpc, camera, native,
                            mapping, quality, installation, binding)
                   if camera is not None else None)
    root.mkdir(parents=True, exist_ok=False)
    arrays = {"native__" + key: np.asarray(value) for key, value in native.items()}
    arrays.update({"mapping__" + key: np.asarray(value) for key, value in mapping.items()})
    arrays["qpc_ns"] = qpc
    if raw_session is not None:
        arrays.update(K=raw_session.K, c2w=raw_session.c2w,
                      c2w_factory=raw_session.c2w_editor)
    code = [file_ref(PROJECT_ROOT / relative) for relative in PARENT_CODE]
    code.append(file_ref(Path(__file__)))
    frame_rows = []
    for index in range(len(qpc)):
        sides = {}
        for side, name in enumerate(("left", "right")):
            sides[name] = {
                "wrist_position_valid": bool(native["controller_valid"][index, side]),
                "wrist_orientation_valid": bool(native["controller_valid"][index, side]),
                "local_handshape_valid": bool(native["manus_valid"][index, side]),
                "raw_composed_valid": bool(native["hand_observed"][side, index]),
                "motion_source": str(native["hand_source"][side, index]),
                "display_valid": bool(native["hand_display_valid"][side, index]),
                "control_ground_truth": False}
        frame_rows.append({"native_frame_index": index, "acquisition_frame_id": "UNKNOWN",
                           "qpc_declared_ns": int(qpc[index]), "hands": sides,
                           "camera_head_valid": bool(native["head_valid"][index]),
                           "projection_qualified": False, "timebase_status": source_report["timebase_status"],
                           "robot_consumer_status": binding["status"], "training_eligible": False})
    report = {"session_id": identity, "native_frames": len(qpc),
        "camera_binding": {key: value for key, value in binding.items() if key != "config"},
        "selected_camera_config": selected, "installation": installation, "time_quality": quality,
        "actual_source_inputs": refs, "actual_code_dependencies": code,
        "frozen_v1_contracts": binding["actual_contracts"],
        "actual_raw_motion_reader": "chaoyang.pick_place_1008.source.load_raw_motion",
        "hdf5_datasets_read": False, "historical_motion_arrays_read": False,
        "hardware_retarget_fields_read": False, "human_session_configuration_required": False,
        "historical_fit_inputs_read": False,
        "historical_override_validator_called": False,
        "retarget_calls": 0, "robot_asset_loads": 0, "model_calls": 0,
        "mask_clean_consumer_status": "READY_INDEPENDENT_OF_CAMERA",
        "robot_consumer_status": binding["status"], "metric_calibration_verified": False,
        "training_eligible": False, "control_ground_truth": False}
    for pin in refs.values():
        verify_ref(pin)
    for pin in binding["actual_contracts"]:
        verify_ref(pin)
    outputs = [_npz(root / "NATIVE_MOTION.npz", arrays),
               _json(root / "FRAME_STATUS.json", frame_rows),
               _json(root / "MOTION_SOURCE_REPORT.json", report)]
    receipt = {"schema": SCHEMA, "stage": "MotionSource", "status": "SEALED_COMPLETE",
        "component_signature": component_signature, "outputs": outputs,
        "native_frames": len(qpc), "session_id": identity,
        "consumer_status": {"NativeMotionAudit": "COMPLETE",
                            "Robot": binding["status"], "Mask": "INDEPENDENT", "Clean": "INDEPENDENT"},
        "actual_inputs": list(refs.values()) + binding["actual_contracts"] + code,
        "kinematic_actions_generated": False, "retarget_calls": 0, "model_calls": 0,
        "visual_quality": "NOT_EVALUATED", "training_eligible": False, "control_ground_truth": False}
    _json(root / "RECEIPT.json", receipt)
    return {"receipt": receipt, "raw_session": raw_session, "native_motion": native,
            "mapping": mapping, "quality": quality, "report": report}

