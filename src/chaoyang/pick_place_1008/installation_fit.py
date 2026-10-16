"""Fixed-camera installation-group fitting from visible glove proxies only.

This successor holds K, camera, raw clocks, metric scale and MANUS local points
fixed. Each original device/prior group receives independent left/right bounded
controller-local rigid corrections. Nothing here grants physical qualification.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import audit, calibration_fit as camera_fit, source
from .camera import rigid

ROTATION_BOUND_RAD = .35
TRANSLATION_BOUND_M = .15


class InstallationFitError(camera_fit.CalibrationFitError):
    pass


def _transform(parameters, seed):
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(np.asarray(parameters[:3], float)).as_matrix()
    delta[:3, 3] = np.asarray(parameters[3:], float)
    return delta @ seed


def _project(controller_to_camera, local_points, k, installation):
    controller_points = (np.asarray(local_points) @ installation[:3, :3].T
                         + installation[:3, 3])
    points = np.einsum("nij,nj->ni", controller_to_camera[:, :3, :3], controller_points)
    points += controller_to_camera[:, :3, 3]
    pixels, depth = camera_fit._project_head_points(points, k, np.eye(4))
    return pixels, depth, points


def fit_installation_proxy_points(controller_to_camera, local_points, uv, k, seed,
                                  session_ids, annotation_uncertainty_px=None):
    """Pure bounded six-DOF fit; matrices and observations are precomputed once."""
    matrices = np.asarray(controller_to_camera, float)
    points, pixels, k = np.asarray(local_points, float), np.asarray(uv, float), np.asarray(k, float)
    seed = rigid(seed, "installation_fit_base")
    n = len(points)
    if (points.shape != (n, 3) or matrices.shape != (n, 4, 4)
            or pixels.shape != (n, 2) or n < 8 or k.shape != (3, 3)
            or not all(np.isfinite(x).all() for x in (matrices, points, pixels, k))
            or np.any(np.linalg.norm(points, axis=1) > 1.)):
        raise InstallationFitError("INSTALLATION_PROXY_INPUT_DOMAIN_OR_COUNT")
    if (not np.allclose(matrices[:, 3], [0., 0., 0., 1.], atol=1e-6)
            or not np.allclose(matrices[:, :3, :3] @ matrices[:, :3, :3].transpose(0, 2, 1),
                               np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(matrices[:, :3, :3]), 1., atol=1e-5)):
        raise InstallationFitError("INSTALLATION_CONTROLLER_CAMERA_NOT_RIGID")
    if len(session_ids) != n or len(set(session_ids)) != 2:
        raise InstallationFitError("INSTALLATION_TWO_DEVELOPMENT_SESSIONS_REQUIRED")
    sigma = (np.full(n, 8.) if annotation_uncertainty_px is None
             else np.asarray(annotation_uncertainty_px, float))
    if sigma.shape != (n,) or not np.isfinite(sigma).all() or (sigma < 8.).any():
        raise InstallationFitError("INSTALLATION_PROXY_UNCERTAINTY_AT_LEAST_8_PX_REQUIRED")
    bounds = np.array([ROTATION_BOUND_RAD] * 3 + [TRANSLATION_BOUND_M] * 3)

    def residual(parameters):
        predicted, _, _ = _project(matrices, points, k, _transform(parameters, seed))
        return ((predicted - pixels) * (8. / sigma[:, None])).ravel()

    solved = least_squares(residual, np.zeros(6), bounds=(-bounds, bounds),
                           loss="soft_l1", f_scale=8., max_nfev=3000,
                           xtol=1e-11, ftol=1e-11, gtol=1e-11)
    fitted = _transform(solved.x, seed)
    before, before_depth, _ = _project(matrices, points, k, seed)
    after, after_depth, _ = _project(matrices, points, k, fitted)
    ids = np.asarray(session_ids)
    per_session = {
        sid: {"before": camera_fit._pixel_metrics(before[ids == sid], pixels[ids == sid]),
              "after": camera_fit._pixel_metrics(after[ids == sid], pixels[ids == sid])}
        for sid in sorted(set(session_ids))
    }
    boundary = np.flatnonzero(bounds - abs(solved.x) < 1e-4).tolist()
    singular = np.linalg.svd(solved.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(solved.jac))
    condition = float(singular[0] / max(singular[-1], 1e-12))
    after_metrics = camera_fit._pixel_metrics(after, pixels)
    passed = (solved.success and not boundary and rank == 6 and condition < 1e8
              and np.all(after_depth > .01)
              and after_metrics["p50_px"] <= camera_fit.FIT_MEDIAN_LIMIT_PX
              and all(value["after"]["p50_px"] <= camera_fit.FIT_MEDIAN_LIMIT_PX
                      and value["after"]["p95_px"] <= camera_fit.FIT_SESSION_P95_LIMIT_PX
                      for value in per_session.values()))
    return {
        "status": "PASSED" if passed else "REVIEW_REQUIRED",
        "T_controller_wrist_before": seed.tolist(),
        "T_controller_wrist_fitted": fitted.tolist(),
        "delta_controller_rotvec_rad": solved.x[:3].tolist(),
        "delta_controller_translation_m": solved.x[3:].tolist(),
        "delta_applied_on_left_in_controller_frame": True,
        "wrist_origin_translation_change_controller_m": (fitted[:3, 3] - seed[:3, 3]).tolist(),
        "rotation_vector_component_bound_rad": ROTATION_BOUND_RAD,
        "translation_component_bound_m": TRANSLATION_BOUND_M,
        "boundary_hit_parameter_indices": boundary,
        "optimizer": {"success": bool(solved.success), "message": str(solved.message),
                      "nfev": int(solved.nfev), "cost": float(solved.cost),
                      "loss": "soft_l1", "f_scale_px": 8.,
                      "uncertainty_residual_weight": "8 / declared_proxy_uncertainty_px"},
        "jacobian_rank": rank, "jacobian_condition": condition,
        "before": camera_fit._pixel_metrics(before, pixels), "after": after_metrics,
        "per_session": per_session,
        "before_nonpositive_depth_count": int((before_depth <= 0).sum()),
        "after_nonpositive_depth_count": int((after_depth <= 0).sum()),
        "acceptance": {"median_limit_px": camera_fit.FIT_MEDIAN_LIMIT_PX,
                       "each_session_p95_limit_px": camera_fit.FIT_SESSION_P95_LIMIT_PX,
                       "image_size": list(camera_fit.IMAGE_SIZE), "boundary_hit_allowed": False},
    }


def _failed_camera_report(path, expected_pins):
    """Bind the exact failed camera report to its terminal sealed producer."""
    reference = source.ref(Path(path))
    audit._verify_exact_ref(reference, project_only=True)
    report = camera_fit._read(path)
    if (report.get("schema") != "pick_place_device_camera_rigid_visual_fit_v1"
            or report.get("status") != "REVIEW_REQUIRED"
            or any(report.get(key) is not False for key in
                   ("holdout_used", "K_fitted", "time_fitted", "scale_fitted",
                    "installation_prior_fitted", "lens_remap"))
            or any(report.get(key) != ref for key, ref in expected_pins.items())):
        raise InstallationFitError("INSTALLATION_CAMERA_FIRST_FAILURE_CHAIN_MISMATCH")
    stage = Path(reference["path"]).parent
    stage_path, outer_path = stage / "RESULT.json", stage.parent / "RESULT.json"
    stage_ref = source.ref(stage_path)
    stage_result, outer = camera_fit._read(stage_path), camera_fit._read(outer_path)
    if (stage_result.get("status") != "REVIEW_REQUIRED"
            or stage_result.get("calibration_report") != reference
            or outer.get("status") != "FAILED_RUNTIME_FINAL"
            or outer.get("validation_errors") != []
            or outer.get("stage_result") != stage_ref
            or outer.get("required_outputs", {}).get("candidate/RESULT.json") != stage_ref
            or outer.get("required_outputs", {}).get("candidate/CALIBRATION_REPORT.json") != reference):
        raise InstallationFitError("INSTALLATION_CAMERA_PRODUCER_NOT_SEALED_OR_UNBOUND")
    packet_ref = audit._verify_exact_ref(outer["task_packet"], project_only=True)
    packet = camera_fit._read(packet_ref["path"])
    if (packet.get("task_id") != outer.get("task_id")
            or packet.get("run_signature") != outer.get("run_signature")
            or not outer.get("run_signature")):
        raise InstallationFitError("INSTALLATION_CAMERA_PRODUCER_SIGNATURE_MISMATCH")
    receipt_path = audit.PROJECT_ROOT / "tasks/receipts" / f"{outer['task_id'].upper()}.json"
    receipt = camera_fit._read(receipt_path)
    if receipt.get("result") != source.ref(outer_path) or receipt.get("status") != outer["status"]:
        raise InstallationFitError("INSTALLATION_CAMERA_PRODUCER_RECEIPT_MISMATCH")
    return {"report": reference, "stage_result": stage_ref,
            "executor_result": source.ref(outer_path), "task_packet": packet_ref,
            "terminal_receipt": source.ref(receipt_path)}


def _verify_inputs(manifest_path, annotations_path, config_path, camera_report_path):
    manifest, config, contexts = camera_fit._load_inputs(manifest_path, config_path)
    annotations = camera_fit._read(annotations_path)
    pins = {"input_manifest": source.ref(Path(manifest_path)),
            "annotations": source.ref(Path(annotations_path)),
            "base_camera_config": source.ref(Path(config_path)),
            "camera_fit_report": source.ref(Path(camera_report_path))}
    for key in ("input_manifest", "base_camera_config"):
        if annotations.get(key) != pins[key]:
            raise InstallationFitError(f"INSTALLATION_ANNOTATION_BINDING_MISMATCH:{key}")
    evidence_ref = annotations.get("source_images")
    if not isinstance(evidence_ref, dict):
        raise InstallationFitError("INSTALLATION_SOURCE_IMAGES_REFERENCE_REQUIRED")
    audit._verify_exact_ref(evidence_ref, project_only=True)
    evidence = camera_fit._read(evidence_ref["path"])
    if (evidence.get("status") != "PASSED" or evidence.get("holdout_used") is not False
            or evidence.get("image_size") != list(camera_fit.IMAGE_SIZE)
            or any(evidence.get(key) != pins[key]
                   for key in ("input_manifest", "base_camera_config"))):
        raise InstallationFitError("INSTALLATION_RAW_EVIDENCE_BINDING_MISMATCH")
    producer = _failed_camera_report(camera_report_path, {
        **{key: pins[key] for key in ("input_manifest", "annotations", "base_camera_config")},
        "source_images": evidence_ref})
    for context in contexts.values():
        selected = context["selected_config"]
        if "K_override" in selected:
            raise InstallationFitError("INSTALLATION_FACTORY_K_MUST_BE_FIXED")
        if "T_head_camera_override" in selected:
            if selected.get("fit_quality_status") != "PASSED":
                raise InstallationFitError("INSTALLATION_BASE_CAMERA_OVERRIDE_NOT_ACCEPTED")
            audit._verify_exact_ref(selected["correction_evidence"], project_only=True)
        prior_ref = context["prior_reference"]
        if prior_ref["used_prior_sha256"] != prior_ref["prior_sha256"]:
            raise InstallationFitError("INSTALLATION_ORIGINAL_RECORDED_PRIOR_REQUIRED")
    records, excluded = camera_fit._validated_correspondences(annotations, evidence, contexts)
    for record in records:
        observation = annotations["observations"][record["annotation_index"]]
        sigma = float(observation.get("annotation_uncertainty_px", 8.))
        if not np.isfinite(sigma) or sigma < 8.:
            raise InstallationFitError("INSTALLATION_PROXY_UNCERTAINTY_AT_LEAST_8_PX_REQUIRED")
        record["annotation_uncertainty_px"] = sigma
    return manifest, config, contexts, records, excluded, pins, evidence_ref, producer


def _group_contexts(contexts):
    groups = {}
    for sid, context in contexts.items():
        serial = context["model"].device_serial
        sha = context["prior_reference"]["prior_sha256"]
        key = serial + ":" + sha
        groups.setdefault(key, {})[sid] = context
    for key, selected in groups.items():
        if len(selected) != 2:
            raise InstallationFitError(f"INSTALLATION_GROUP_REQUIRES_TWO_DEVELOPMENT_SESSIONS:{key}")
        priors = [source.installation_matrices(value["prior"]) for value in selected.values()]
        if not all(np.array_equal(value, priors[0]) for value in priors[1:]):
            raise InstallationFitError("INSTALLATION_GROUP_PRIOR_VALUES_NOT_FIXED")
    return groups


def _precompute(records, contexts):
    matrices, points = [], []
    for record in records:
        context = contexts[record["session_id"]]
        frame, side = record["raw_frame_index"], source.SIDES.index(record["side"])
        native, model = context["native"], context["model"]
        head = native["head"][frame]
        controller = native["controller"][frame, side]
        matrix = np.linalg.inv(model.T_head_camera) @ np.linalg.inv(head) @ controller
        local = native["manus_local_25"][frame, side, record["manus_node_id"]]
        seed = source.installation_matrices(context["prior"])[side]
        _, _, camera_point = _project(matrix[None], local[None], context["K_image"], seed)
        point_head = (camera_point[0] @ model.T_head_camera[:3, :3].T
                      + model.T_head_camera[:3, 3])
        if not np.allclose(point_head, record["point_head_m"], atol=1e-9):
            raise InstallationFitError("INSTALLATION_PRECOMPUTED_RAW_COMPOSITION_MISMATCH")
        matrices.append(matrix)
        points.append(local)
    return np.asarray(matrices), np.asarray(points)


def run_installation_fit(manifest_path, annotations_path, base_camera_config_path,
                         camera_fit_report_path, output_root):
    """Fit only wrists after a sealed failed camera-only stage; never use holdout."""
    _, config, contexts, records, excluded, pins, evidence_ref, producer = _verify_inputs(
        manifest_path, annotations_path, base_camera_config_path, camera_fit_report_path)
    grouped = _group_contexts(contexts)
    matrices, points = _precompute(records, contexts)
    fits, devices = {}, {}
    record_sessions = np.asarray([row["session_id"] for row in records])
    record_sides = np.asarray([row["side"] for row in records])
    pixels = np.asarray([row["uv"] for row in records])
    uncertainty = np.asarray([row["annotation_uncertainty_px"] for row in records])
    for key, selected in sorted(grouped.items()):
        first = next(iter(selected.values()))
        serial = first["model"].device_serial
        seed_prior = copy.deepcopy(first["prior"])
        prior_matrices = source.installation_matrices(seed_prior)
        sides = {}
        for side_index, side in enumerate(source.SIDES):
            ids = np.flatnonzero(np.isin(record_sessions, list(selected)) & (record_sides == side))
            sid_values = record_sessions[ids].tolist()
            if (len(ids) < 8 or set(sid_values) != set(selected)
                    or any(len({records[i]["raw_frame_index"] for i in ids
                                if records[i]["session_id"] == sid}) < 2 for sid in selected)):
                raise InstallationFitError("INSTALLATION_SIDE_SESSION_OR_FRAME_COVERAGE_INSUFFICIENT")
            fit = fit_installation_proxy_points(
                matrices[ids], points[ids], pixels[ids], first["K_image"],
                prior_matrices[side_index], sid_values, uncertainty[ids])
            sides[side] = fit
            fitted = np.asarray(fit["T_controller_wrist_fitted"])
            before_uv, before_z, before_cam = _project(
                matrices[ids], points[ids], first["K_image"], prior_matrices[side_index])
            after_uv, after_z, after_cam = _project(matrices[ids], points[ids], first["K_image"], fitted)
            fixed_camera = first["model"].T_head_camera
            before_head = before_cam @ fixed_camera[:3, :3].T + fixed_camera[:3, 3]
            after_head = after_cam @ fixed_camera[:3, :3].T + fixed_camera[:3, 3]
            for j, index in enumerate(ids):
                records[index].update({
                    "group_key": key, "uv_before": before_uv[j].tolist(), "uv_after": after_uv[j].tolist(),
                    "point_head_m_before": before_head[j].tolist(), "point_head_m_after": after_head[j].tolist(),
                    "camera_depth_before_m": float(before_z[j]), "camera_depth_after_m": float(after_z[j]),
                    "installation_fit_status": fit["status"],
                })
        status = "PASSED" if all(value["status"] == "PASSED" for value in sides.values()) else "REVIEW_REQUIRED"
        fitted_prior = copy.deepcopy(seed_prior)
        for side in source.SIDES:
            matrix = np.asarray(sides[side]["T_controller_wrist_fitted"])
            fitted_prior[side]["pos"] = matrix[:3, 3].tolist()
            fitted_prior[side]["quat"] = Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()
        fits[key] = {
            "status": status, "device_serial": serial,
            "original_prior_sha256": first["prior_reference"]["prior_sha256"],
            "development_sessions": sorted(selected), "sides": sides,
            "prior_before": seed_prior, "prior_fitted_diagnostic": fitted_prior,
            "prior_used": fitted_prior if status == "PASSED" else seed_prior,
            "installation_prior_references": {sid: value["prior_reference"] for sid, value in selected.items()},
        }
        devices.setdefault(serial, {
            "T_head_camera_fixed": first["model"].T_head_camera.tolist(),
            "T_head_camera_factory": first["model"].T_head_camera_factory.tolist(),
            "K_base_fixed": first["model"].K.tolist(), "K_image_fixed": first["K_image"].tolist(),
            "camera_snapshot_sha256": first["model"].camera_sha256, "group_keys": [],
        })["group_keys"].append(key)
    status = "PASSED" if all(value["status"] == "PASSED" for value in fits.values()) else "REVIEW_REQUIRED"
    report = {
        "schema": "pick_place_installation_group_visual_fit_v1", "status": status, **pins,
        "source_images": evidence_ref, "preceding_failure_producer": producer,
        "development_sessions": list(camera_fit.DEVELOPMENT_SESSIONS),
        "source_sessions": [{"session_id": sid, "source_refs": value["row"]["source_refs"],
                             "installation_prior": value["prior_reference"]} for sid, value in contexts.items()],
        "groups": fits, "devices": devices, "correspondences": records, "excluded_correspondences": excluded,
        "camera_fixed": True, "camera_fitted": False, "K_fitted": False,
        "time_fitted": False, "scale_fitted": False, "holdout_used": False, "lens_remap": False,
        "installation_prior_fitted": True,
        "fitted_variables": "LEFT_RIGHT_CONTROLLER_LOCAL_SIX_DOF_PER_ORIGINAL_DEVICE_PRIOR_GROUP",
        "manus_local_authority": "DECLARED_WRIST_LOCAL_NOT_EXTRA_AXIS_FLIP",
        "correspondence_authority": "GLOVE_VISUAL_PROXY_NOT_ANATOMICAL_OR_METRIC_GROUND_TRUTH",
        "eye_identity_qualification": camera_fit.EYE_QUALIFICATION,
        "training_eligible": False, "control_ground_truth": False,
    }
    output = audit._output_path(output_root)
    if output.exists():
        raise InstallationFitError("INSTALLATION_FRESH_OUTPUT_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    audit._write_json(output / "INSTALLATION_REPORT.json", report)
    report_ref = source.ref(output / "INSTALLATION_REPORT.json")
    fitted_config = copy.deepcopy(config)
    fitted_config.update({"status": "FROZEN_VISUAL_ESTIMATE", "fit_quality_status": status,
                          "training_eligible": False, "control_ground_truth": False})
    for serial, device in devices.items():
        if len(device["group_keys"]) != 1:
            raise InstallationFitError("INSTALLATION_DEVICE_MULTIPLE_GROUPS_REQUIRE_EXPLICIT_ROUTING")
        key = device["group_keys"][0]
        group = fits[key]
        selected = fitted_config["devices"][serial]
        selected.update({"status": "FROZEN_VISUAL_ESTIMATE", "fit_quality_status": group["status"],
                         "installation_group_key": key,
                         "installation_original_prior_sha256": group["original_prior_sha256"]})
        if group["status"] == "PASSED":
            selected.update({"installation_prior_override": group["prior_used"],
                             "installation_correction_evidence": report_ref})
    audit._write_json(output / "FITTED_CAMERA_CONFIG.json", fitted_config)
    config_ref = source.ref(output / "FITTED_CAMERA_CONFIG.json")
    result = {
        "schema": "pick_place_installation_fit_result_v1", "status": status,
        "installation_report": report_ref, "fitted_camera_config": config_ref,
        "fitted_installation_config": config_ref,
        "camera_fit_report": pins["camera_fit_report"], "source_images": evidence_ref,
        "correspondence_count": len(records), "excluded_count": len(excluded),
        "group_fit_status": {key: value["status"] for key, value in fits.items()},
        "camera_fixed": True, "K_fitted": False, "time_fitted": False,
        "scale_fitted": False, "holdout_used": False, "lens_remap": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    audit._write_json(output / "RESULT.json", result)
    return result
