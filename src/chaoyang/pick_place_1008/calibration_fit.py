"""Device-only rigid camera fitting from explicit offline glove visual proxies.

K, raw clocks, metric scale, wrist priors, SBS route and coordinate composition
are fixed. These proxies and fitted transforms confer no physical qualification.
Production calls enter through a separately registered fenced operation.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import audit, source
from .camera import normalize_camera, rigid
from .source import MAP_25_TO_21

DEVELOPMENT_SESSIONS = ("001", "003", "008", "100")
IMAGE_SIZE = (640, 480)
ROTATION_BOUND_RAD = .35
TRANSLATION_BOUND_M = .25
FIT_MEDIAN_LIMIT_PX = 20.0
FIT_SESSION_P95_LIMIT_PX = 60.0
EYE_QUALIFICATION = "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT"
TIP_INDICES_21 = {4, 8, 12, 16, 20}


class CalibrationFitError(ValueError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _fixed_frames(count):
    return sorted({0, count - 1, *[
        int(round((count - 1) * fraction)) for fraction in (.25, .5, .75)]})


def _load_inputs(manifest_path, config_path):
    manifest = _read(manifest_path)
    config = _read(config_path)
    rows = {row["session_id"]: row for row in manifest["sessions"]}
    if any(sid not in rows for sid in DEVELOPMENT_SESSIONS):
        raise CalibrationFitError("FIXED_DEVELOPMENT_SESSION_MISSING")
    contexts = {}
    for sid in DEVELOPMENT_SESSIONS:
        row = rows[sid]
        if row.get("input_status") != "READY_RAW_AUDIT":
            raise CalibrationFitError(f"DEVELOPMENT_RAW_AUDIT_NOT_READY:{sid}")
        for reference in row["source_refs"].values():
            audit._verify_exact_ref(reference)
        serial = row["device_serial"]
        selected = config.get("devices", {}).get(serial)
        if not isinstance(selected, dict) or selected.get("sbs_index") != 1:
            raise CalibrationFitError(f"EXPLICIT_SBS_INDEX_ONE_REQUIRED:{serial}")
        params = _read(row["source_refs"]["camera_params"]["path"])
        metadata = _read(row["source_refs"]["camera_meta"]["path"])
        output_size = tuple(selected.get("output_size", [1280, 960]))
        model = normalize_camera(params, metadata, device_serial=serial,
                                 camera_sha256=row["camera_snapshot_sha256"],
                                 config=selected, output_size=output_size)
        sx, sy = IMAGE_SIZE[0] / model.width, IMAGE_SIZE[1] / model.height
        if not np.isclose(sx, sy):
            raise CalibrationFitError("EVIDENCE_NONUNIFORM_RESIZE")
        k_image = np.diag([sx, sy, 1.]) @ model.K
        source_dir = Path(row["source_refs"]["source_manifest"]["path"]).parent
        prior, prior_reference = source.read_installation(source_dir)
        if "installation_prior_override" in selected:
            if not selected.get("installation_correction_evidence"):
                raise CalibrationFitError("BASE_INSTALLATION_OVERRIDE_EVIDENCE_MISSING")
            prior = copy.deepcopy(selected["installation_prior_override"])
            prior_reference["override_evidence"] = selected["installation_correction_evidence"]
        prior_reference["used_prior_sha256"] = hashlib.sha256(json.dumps(
            prior, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        qpc = source.timeline(audit.read_integer_jsonl(
            row["source_refs"]["video_qpc"]["path"]), "camera_fit_video")
        if len(qpc) != row["raw_frame_count"]:
            raise CalibrationFitError("RAW_VIDEO_QPC_COUNT_MISMATCH")
        native, mapping, quality = source.load_raw_motion(
            source_dir, qpc, expected_serial=serial, prior=prior)
        contexts[sid] = {
            "row": row, "source_dir": source_dir, "model": model,
            "selected_config": copy.deepcopy(selected), "K_image": k_image,
            "qpc_ns": qpc, "prior": prior, "prior_reference": prior_reference,
            "native": native, "mapping": mapping, "time_quality": quality,
        }
    for serial in {context["row"]["device_serial"] for context in contexts.values()}:
        grouped = [context for context in contexts.values()
                   if context["row"]["device_serial"] == serial]
        if len(grouped) != 2:
            raise CalibrationFitError("EXACT_TWO_DEVELOPMENT_SESSIONS_PER_DEVICE_REQUIRED")
        first = grouped[0]["model"]
        if any(context["model"].camera_sha256 != first.camera_sha256
               or not np.array_equal(context["model"].K, first.K)
               or not np.array_equal(context["model"].T_head_camera, first.T_head_camera)
               for context in grouped[1:]):
            raise CalibrationFitError("DEVICE_CAMERA_CONFIGURATION_NOT_FIXED")
    return manifest, config, contexts


def _frame_metadata(context, frame_index):
    qpc = context["qpc_ns"]
    mapping = context["mapping"]
    serial = context["row"]["device_serial"]
    result = {
        "session_id": context["row"]["session_id"], "device_serial": serial,
        "raw_frame_index": frame_index, "qpc_ns": int(qpc[frame_index]),
        "relative_qpc_ns": int(qpc[frame_index] - qpc[0]), "image_size": list(IMAGE_SIZE),
        "eye": context["model"].eye, "sbs_index": context["model"].sbs_index,
        "eye_identity_qualification": EYE_QUALIFICATION,
        "frame_semantics": "NATIVE_DECODE_DISPLAY_ORDER_TO_RECORDED_VIDEO_QPC_ROW",
        "timestamp_semantics": "SOFTWARE_HOST_ARRIVAL_NOT_EXPOSURE_SYNCHRONIZATION",
        "K_image_fixed": context["K_image"].tolist(),
        "T_head_camera_factory": context["model"].T_head_camera_factory.tolist(),
        "T_head_camera_base": context["model"].T_head_camera.tolist(),
        "installation_prior": context["prior_reference"],
    }
    for key in ["pico_source_row", "pico_source_line", "pico_source_qpc_ns", "pico_skew_ns",
                "manus_left_source_line", "manus_left_source_qpc_ns", "manus_left_skew_ns",
                "manus_right_source_line", "manus_right_source_qpc_ns", "manus_right_skew_ns"]:
        result[key] = int(mapping[key][frame_index])
    return result


def prepare_calibration_evidence(manifest_path, output_root, base_camera_config_path):
    """Save 20 unoverlaid development PNGs from full native ordered raw decoding."""
    manifest, config, contexts = _load_inputs(manifest_path, base_camera_config_path)
    output = audit._output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    sessions = []
    devices = {}
    for sid in DEVELOPMENT_SESSIONS:
        context = contexts[sid]
        frames = _fixed_frames(len(context["qpc_ns"]))
        evidence_model = replace(context["model"], width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
                                 K=context["K_image"])
        images = []
        decoded = 0
        for frame_index, pixels in enumerate(source.crop_frames(context["source_dir"], evidence_model)):
            decoded = frame_index + 1
            if frame_index not in frames:
                continue
            if pixels.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
                raise CalibrationFitError("EVIDENCE_FRAME_PIXEL_DOMAIN_MISMATCH")
            path = audit._output_path(output / "images" / sid / f"raw_{frame_index:04}.png")
            if not path.is_relative_to(output):
                raise CalibrationFitError("EVIDENCE_IMAGE_ESCAPES_OUTPUT_ROOT")
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise CalibrationFitError("EVIDENCE_PNG_ENCODING_FAILED")
            with path.open("xb") as handle:
                handle.write(encoded.tobytes())
            images.append({**_frame_metadata(context, frame_index),
                           "image": source.ref(path), "pixels_overlaid": False,
                           "lens_remap": False})
        if decoded != len(context["qpc_ns"]) or len(images) != len(frames):
            raise CalibrationFitError(f"NATIVE_FULL_DECODE_QPC_FRAME_COUNT_MISMATCH:{sid}")
        model = context["model"]
        devices[model.device_serial] = {
            "camera_snapshot_sha256": model.camera_sha256, "eye": model.eye,
            "sbs_index": model.sbs_index, "eye_identity_qualification": EYE_QUALIFICATION,
            "K_base_fixed": model.K.tolist(), "K_image_fixed": context["K_image"].tolist(),
            "T_head_camera_factory": model.T_head_camera_factory.tolist(),
            "T_head_camera_base": model.T_head_camera.tolist(),
            "coordinate_semantics": "T_headREP_cameraEncoded maps camera into REP103 head",
        }
        sessions.append({
            "session_id": sid, "device_serial": model.device_serial,
            "source_refs": context["row"]["source_refs"],
            "installation_prior": context["prior_reference"], "prior_values_fixed": context["prior"],
            "raw_frame_count": len(context["qpc_ns"]), "decoded_frames": decoded,
            "required_fixed_frames": frames, "images": images,
            "raw_motion_composition": "RAW_CONTROLLER_HEAD_MANUS_WITH_FIXED_RECORDED_INSTALLATION_PRIOR",
            "manus_local_authority": "DECLARED_WRIST_LOCAL_NOT_MEASURED_ANATOMICAL_TRUTH",
        })
    evidence = {
        "schema": "pick_place_calibration_source_images_v1", "status": "PASSED",
        "input_manifest": source.ref(Path(manifest_path)),
        "base_camera_config": source.ref(Path(base_camera_config_path)),
        "development_sessions": list(DEVELOPMENT_SESSIONS), "holdout_used": False,
        "image_size": list(IMAGE_SIZE), "sessions": sessions, "devices": devices,
        "native_decoder_threads": 2, "image_operation": "SBS_CROP_UNIFORM_RESIZE_ONLY",
        "no_overlays": True, "lens_remap": False, "qpc_modified": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    audit._write_json(output / "SOURCE_IMAGES.json", evidence)
    result = {
        "schema": "pick_place_calibration_evidence_result_v1", "status": "PASSED",
        "source_images": source.ref(output / "SOURCE_IMAGES.json"),
        "development_sessions": list(DEVELOPMENT_SESSIONS), "image_count": sum(
            len(row["images"]) for row in sessions),
        "decoded_raw_frames": sum(row["decoded_frames"] for row in sessions),
        "holdout_used": False, "training_eligible": False, "control_ground_truth": False,
    }
    audit._write_json(output / "RESULT.json", result)
    return result


def _transform_from_parameters(parameters, seed):
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(np.asarray(parameters[:3], float)).as_matrix()
    delta[:3, 3] = np.asarray(parameters[3:], float)
    return delta @ seed


def _project_head_points(points_head, k, transform):
    points = np.asarray(points_head, float)
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    z = camera[:, 2]
    denominator = np.where(abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
    projected = camera @ np.asarray(k, float).T
    return projected[:, :2] / denominator[:, None], z


def _pixel_metrics(predicted, observed):
    errors = np.linalg.norm(np.asarray(predicted) - np.asarray(observed), axis=1)
    return {"count": len(errors), "p50_px": float(np.median(errors)),
            "p95_px": float(np.percentile(errors, 95)), "max_px": float(errors.max())}


def fit_rigid_proxy_points(points_head, uv, k, seed, session_ids):
    """Pure six-DOF bounded robust fit, used by CPU known-answer tests."""
    points, pixels, k = np.asarray(points_head, float), np.asarray(uv, float), np.asarray(k, float)
    seed = rigid(seed, "camera_fit_base")
    if points.shape != (len(pixels), 3) or pixels.ndim != 2 or pixels.shape[1] != 2 \
            or len(points) < 8 or not np.isfinite(points).all() or not np.isfinite(pixels).all():
        raise CalibrationFitError("VISIBLE_PROXY_CORRESPONDENCES_INVALID_OR_TOO_FEW")
    if len(session_ids) != len(points) or len(set(session_ids)) != 2:
        raise CalibrationFitError("TWO_DEVELOPMENT_SESSIONS_REQUIRED_FOR_DEVICE_FIT")
    bounds = np.array([ROTATION_BOUND_RAD] * 3 + [TRANSLATION_BOUND_M] * 3)
    def residual(parameters):
        predicted, _ = _project_head_points(points, k, _transform_from_parameters(parameters, seed))
        return (predicted - pixels).ravel()
    solved = least_squares(residual, np.zeros(6), bounds=(-bounds, bounds),
                           loss="soft_l1", f_scale=8., max_nfev=3000,
                           xtol=1e-11, ftol=1e-11, gtol=1e-11)
    fitted = _transform_from_parameters(solved.x, seed)
    before, before_z = _project_head_points(points, k, seed)
    after, after_z = _project_head_points(points, k, fitted)
    per_session = {}
    sessions = np.asarray(session_ids)
    for sid in sorted(set(session_ids)):
        mask = sessions == sid
        per_session[sid] = {"before": _pixel_metrics(before[mask], pixels[mask]),
                            "after": _pixel_metrics(after[mask], pixels[mask])}
    boundary_hit = np.flatnonzero(bounds - abs(solved.x) < 1e-4).tolist()
    singular = np.linalg.svd(solved.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(solved.jac))
    condition = float(singular[0] / max(singular[-1], 1e-12))
    global_after = _pixel_metrics(after, pixels)
    quality = (solved.success and not boundary_hit and rank == 6 and condition < 1e8
               and np.all(after_z > .01) and global_after["p50_px"] <= FIT_MEDIAN_LIMIT_PX
               and all(row["after"]["p50_px"] <= FIT_MEDIAN_LIMIT_PX
                       and row["after"]["p95_px"] <= FIT_SESSION_P95_LIMIT_PX
                       for row in per_session.values()))
    return {
        "status": "PASSED" if quality else "REVIEW_REQUIRED",
        "T_head_camera_before": seed.tolist(), "T_head_camera_fitted": fitted.tolist(),
        "delta_head_rotvec_rad": solved.x[:3].tolist(),
        "delta_head_translation_m": solved.x[3:].tolist(),
        "delta_applied_on_left_in_head_frame": True,
        "rotation_vector_component_bound_rad": ROTATION_BOUND_RAD,
        "translation_component_bound_m": TRANSLATION_BOUND_M,
        "boundary_hit_parameter_indices": boundary_hit,
        "optimizer": {"success": bool(solved.success), "message": str(solved.message),
                      "nfev": int(solved.nfev), "cost": float(solved.cost),
                      "loss": "soft_l1", "f_scale_px": 8.0},
        "jacobian_rank": rank, "jacobian_condition": condition,
        "before": _pixel_metrics(before, pixels), "after": global_after,
        "per_session": per_session,
        "before_nonpositive_depth_count": int((before_z <= 0).sum()),
        "after_nonpositive_depth_count": int((after_z <= 0).sum()),
        "acceptance": {"median_limit_px": FIT_MEDIAN_LIMIT_PX,
                       "each_session_p95_limit_px": FIT_SESSION_P95_LIMIT_PX,
                       "image_size": list(IMAGE_SIZE), "boundary_hit_allowed": False},
    }


def _validated_correspondences(annotations, evidence, contexts):
    records, excluded = [], []
    images = {(session["session_id"], image["raw_frame_index"]): image
              for session in evidence["sessions"] for image in session["images"]}
    seen = set()
    for index, observation in enumerate(annotations.get("observations", [])):
        sid, frame = observation.get("session_id"), observation.get("raw_frame_index")
        if sid not in DEVELOPMENT_SESSIONS or type(frame) is not int:
            raise CalibrationFitError("ANNOTATION_SESSION_OR_FRAME_NOT_FIXED_DEVELOPMENT")
        context = contexts[sid]
        if (sid, frame) not in images:
            raise CalibrationFitError("ANNOTATION_IMAGE_NOT_IN_FIXED_RAW_EVIDENCE")
        image = images[(sid, frame)]
        if (observation.get("device_serial") != context["model"].device_serial
                or observation.get("image") != image["image"]
                or observation.get("qpc_ns") != int(context["qpc_ns"][frame])):
            raise CalibrationFitError("ANNOTATION_IMAGE_DEVICE_QPC_BINDING_MISMATCH")
        audit._verify_exact_ref(image["image"], project_only=True)
        joint = observation.get("joint_index_21")
        side = observation.get("side")
        point_type = observation.get("point_type")
        if (type(joint) is not int or not 1 <= joint <= 20
                or observation.get("manus_node_id") != int(MAP_25_TO_21[joint])
                or side not in source.SIDES
                or point_type not in ["GLOVE_TIP", "GLOVE_KNUCKLE"]
                or (point_type == "GLOVE_TIP") != (joint in TIP_INDICES_21)
                or observation.get("semantics") != "VISUAL_PROXY"):
            raise CalibrationFitError("ANNOTATION_NONROOT_GLOVE_PROXY_MAPPING_REQUIRED")
        identity = (sid, frame, side, joint)
        if identity in seen:
            raise CalibrationFitError("DUPLICATE_PROXY_CORRESPONDENCE")
        seen.add(identity)
        visibility = observation.get("visibility")
        if visibility == "OCCLUDED":
            excluded.append({"annotation_index": index, "reason": "EXPLICIT_OCCLUDED",
                             "session_id": sid, "raw_frame_index": frame})
            continue
        if (visibility != "VISIBLE" or observation.get("side_confirmed") is not True
                or observation.get("joint_mapping_confirmed") is not True):
            raise CalibrationFitError("VISIBLE_PROXY_SIDE_AND_JOINT_CONFIRMATION_REQUIRED")
        uv = np.asarray(observation.get("uv"), float)
        if uv.shape != (2,) or not np.isfinite(uv).all() \
                or not 0 <= uv[0] < IMAGE_SIZE[0] or not 0 <= uv[1] < IMAGE_SIZE[1]:
            raise CalibrationFitError("ANNOTATION_PIXEL_OUTSIDE_RAW_EVIDENCE_DOMAIN")
        side_index = source.SIDES.index(side)
        native = context["native"]
        if not native["head_valid"][frame] or not native["hand_observed"][side_index, frame]:
            raise CalibrationFitError("ANNOTATION_RAW_HEAD_OR_HAND_NOT_OBSERVED")
        head = native["head"][frame]
        world = native["hands"][side_index, frame, joint]
        point_head = (world - head[:3, 3]) @ head[:3, :3]
        records.append({"annotation_index": index, "session_id": sid,
                        "device_serial": context["model"].device_serial,
                        "raw_frame_index": frame, "qpc_ns": int(context["qpc_ns"][frame]),
                        "image": image["image"], "side": side, "joint_index_21": joint,
                        "manus_node_id": int(MAP_25_TO_21[joint]),
                        "point_type": point_type, "semantics": "VISUAL_PROXY",
                        "point_world_m": world.tolist(), "point_head_m": point_head.tolist(),
                        "uv": uv.tolist(), "installation_prior": context["prior_reference"]})
    for serial in {context["model"].device_serial for context in contexts.values()}:
        selected = [record for record in records if record["device_serial"] == serial]
        sessions = {context["row"]["session_id"] for context in contexts.values()
                    if context["model"].device_serial == serial}
        if len(selected) < 8 or {record["session_id"] for record in selected} != sessions:
            raise CalibrationFitError("DEVICE_PROXY_COUNT_OR_TWO_SESSION_COVERAGE_INSUFFICIENT")
        if any(len({record["raw_frame_index"] for record in selected
                    if record["session_id"] == sid}) < 2 for sid in sessions):
            raise CalibrationFitError("EACH_DEVELOPMENT_SESSION_REQUIRES_MULTIPLE_RAW_FRAMES")
    return records, excluded


def run_camera_fit(manifest_path, annotations_path, base_camera_config_path, output_root):
    """Freeze a device rigid visual estimate; quality rejection remains explicit."""
    manifest, config, contexts = _load_inputs(manifest_path, base_camera_config_path)
    annotations = _read(annotations_path)
    pins = {"input_manifest": source.ref(Path(manifest_path)),
            "base_camera_config": source.ref(Path(base_camera_config_path))}
    for key, actual in pins.items():
        if annotations.get(key) != actual:
            raise CalibrationFitError(f"ANNOTATION_FROZEN_INPUT_REFERENCE_MISMATCH:{key}")
    evidence_ref = annotations.get("source_images")
    if not isinstance(evidence_ref, dict):
        raise CalibrationFitError("ANNOTATION_SOURCE_IMAGES_REFERENCE_MISSING")
    audit._verify_exact_ref(evidence_ref, project_only=True)
    evidence = _read(evidence_ref["path"])
    if (evidence.get("status") != "PASSED" or evidence.get("holdout_used") is not False
            or evidence.get("image_size") != list(IMAGE_SIZE)
            or any(evidence.get(key) != actual for key, actual in pins.items())):
        raise CalibrationFitError("ANNOTATION_SOURCE_IMAGES_INPUT_BINDING_MISMATCH")
    records, excluded = _validated_correspondences(annotations, evidence, contexts)
    devices = {}
    for serial in sorted({record["device_serial"] for record in records}):
        selected = [record for record in records if record["device_serial"] == serial]
        context = next(context for context in contexts.values()
                       if context["model"].device_serial == serial)
        devices[serial] = fit_rigid_proxy_points(
            [record["point_head_m"] for record in selected],
            [record["uv"] for record in selected], context["K_image"],
            context["model"].T_head_camera, [record["session_id"] for record in selected])
        devices[serial]["camera_snapshot_sha256"] = context["model"].camera_sha256
        devices[serial]["K_base_fixed"] = context["model"].K.tolist()
        devices[serial]["K_image_fixed"] = context["K_image"].tolist()
        devices[serial]["T_head_camera_factory"] = context["model"].T_head_camera_factory.tolist()
    status = "PASSED" if all(device["status"] == "PASSED" for device in devices.values()) \
        else "REVIEW_REQUIRED"
    report = {
        "schema": "pick_place_device_camera_rigid_visual_fit_v1", "status": status,
        **pins, "annotations": source.ref(Path(annotations_path)), "source_images": evidence_ref,
        "development_sessions": list(DEVELOPMENT_SESSIONS), "holdout_used": False,
        "source_sessions": [{
            "session_id": sid, "source_refs": context["row"]["source_refs"],
            "installation_prior": context["prior_reference"]}
            for sid, context in contexts.items()],
        "correspondences": records, "excluded_correspondences": excluded, "devices": devices,
        "fitted_variables": "ONE_DEVICE_HEAD_FRAME_SIX_DOF_RIGID_CAMERA_CORRECTION",
        "K_fitted": False, "time_fitted": False, "scale_fitted": False,
        "installation_prior_fitted": False, "lens_remap": False,
        "transform_semantics": "T_headREP_cameraEncoded maps encoded camera into REP103 head",
        "correspondence_authority": "GLOVE_VISUAL_PROXY_NOT_ANATOMICAL_OR_METRIC_GROUND_TRUTH",
        "manus_local_authority": "DECLARED_WRIST_LOCAL_ROOT_IDENTITY_NOT_EXTRA_AXIS_FLIP",
        "eye_identity_qualification": EYE_QUALIFICATION,
        "training_eligible": False, "control_ground_truth": False,
    }
    output = audit._output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    audit._write_json(output / "CALIBRATION_REPORT.json", report)
    report_ref = source.ref(output / "CALIBRATION_REPORT.json")
    fitted_config = copy.deepcopy(config)
    fitted_config["status"] = "FROZEN_VISUAL_ESTIMATE"
    fitted_config["fit_quality_status"] = status
    fitted_config["training_eligible"] = False
    fitted_config["control_ground_truth"] = False
    for serial, fit in devices.items():
        fitted_config["devices"][serial].update({
            "T_head_camera_override": fit["T_head_camera_fitted"],
            "correction_evidence": report_ref, "status": "FROZEN_VISUAL_ESTIMATE",
            "fit_quality_status": fit["status"], "eye_identity_qualification": EYE_QUALIFICATION,
            "physical_eye_verified": False, "metric_calibration_verified": False,
        })
    audit._write_json(output / "FITTED_CAMERA_CONFIG.json", fitted_config)
    result = {"schema": "pick_place_camera_fit_result_v1", "status": status,
              "calibration_report": report_ref,
              "fitted_camera_config": source.ref(output / "FITTED_CAMERA_CONFIG.json"),
              "correspondence_count": len(records), "excluded_count": len(excluded),
              "device_fit_status": {serial: fit["status"] for serial, fit in devices.items()},
              "holdout_used": False, "physical_eye_verified": False,
              "training_eligible": False, "control_ground_truth": False}
    audit._write_json(output / "RESULT.json", result)
    return result
