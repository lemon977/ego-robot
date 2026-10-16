"""Bounded encoded-domain K verification after two sealed visual-fit failures.

Factory camera extrinsics, recorded wrist priors, native QPC and metric scale
stay fixed. Glove visual proxies support an offline estimate, never physical
eye identity, measured camera calibration or robot-control ground truth.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import audit, calibration_fit as calibration, source
from .camera import rigid

FOCAL_RATIO_BOUNDS = (.5, 3.)
PRINCIPAL_OFFSET_BOUND_PX = 80.
INSTALLATION_SCHEMA = "pick_place_installation_group_visual_fit_v1"


class IntrinsicVerificationError(calibration.CalibrationFitError):
    pass


def _canonical_k(value):
    k = np.asarray(value, float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]],
                               [0, 0, 0, 0, 1], atol=1e-12)):
        raise IntrinsicVerificationError("CANONICAL_FACTORY_K_REQUIRED")
    return k


def _k_from_parameters(parameters, seed):
    k = seed.copy()
    k[0, 0] *= parameters[0]
    k[1, 1] *= parameters[1]
    k[0, 2] += parameters[2]
    k[1, 2] += parameters[3]
    return k


def _metrics_pass(metrics):
    return (metrics["p50_px"] <= calibration.FIT_MEDIAN_LIMIT_PX
            and metrics["p95_px"] <= calibration.FIT_SESSION_P95_LIMIT_PX)


def fit_intrinsic_proxy_points(points_head, uv, k_factory, t_factory, session_ids):
    """Fit only fx/fy/cx/cy against fixed metric points and camera extrinsics."""
    points, pixels = np.asarray(points_head, float), np.asarray(uv, float)
    k = _canonical_k(k_factory)
    transform = rigid(t_factory, "intrinsic_fixed_factory_camera")
    if (pixels.ndim != 2 or pixels.shape[1] != 2
            or points.shape != (len(pixels), 3) or len(points) < 8
            or not np.isfinite(points).all() or not np.isfinite(pixels).all()):
        raise IntrinsicVerificationError("VISIBLE_PROXY_CORRESPONDENCES_INVALID_OR_TOO_FEW")
    if len(session_ids) != len(points) or len(set(session_ids)) != 2:
        raise IntrinsicVerificationError("TWO_DEVELOPMENT_SESSIONS_REQUIRED_FOR_DEVICE_FIT")
    lower = np.array([FOCAL_RATIO_BOUNDS[0]] * 2 + [-PRINCIPAL_OFFSET_BOUND_PX] * 2)
    upper = np.array([FOCAL_RATIO_BOUNDS[1]] * 2 + [PRINCIPAL_OFFSET_BOUND_PX] * 2)
    def residual(parameters):
        projected, _ = calibration._project_head_points(
            points, _k_from_parameters(parameters, k), transform)
        return (projected - pixels).ravel()
    solved = least_squares(residual, [1., 1., 0., 0.], bounds=(lower, upper),
                           loss="soft_l1", f_scale=8., max_nfev=3000,
                           xtol=1e-11, ftol=1e-11, gtol=1e-11)
    fitted = _k_from_parameters(solved.x, k)
    before, depth = calibration._project_head_points(points, k, transform)
    after, after_depth = calibration._project_head_points(points, fitted, transform)
    sessions = np.asarray(session_ids)
    per_session = {}
    for sid in sorted(set(session_ids)):
        mask = sessions == sid
        per_session[sid] = {"before": calibration._pixel_metrics(before[mask], pixels[mask]),
                            "after": calibration._pixel_metrics(after[mask], pixels[mask])}
    boundary_hits = np.flatnonzero(np.minimum(solved.x - lower, upper - solved.x) < 1e-4).tolist()
    singular = np.linalg.svd(solved.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(solved.jac))
    condition = float(singular[0] / max(singular[-1], 1e-12))
    before_metrics = calibration._pixel_metrics(before, pixels)
    after_metrics = calibration._pixel_metrics(after, pixels)
    quality = (solved.success and not boundary_hits and rank == 4 and condition < 1e8
               and np.all(after_depth > .01)
               and after_metrics["p50_px"] <= calibration.FIT_MEDIAN_LIMIT_PX
               and all(_metrics_pass(row["after"]) for row in per_session.values()))
    baseline_pass = all(_metrics_pass(row["before"]) for row in per_session.values())
    diagnosis = ("ENCODED_DOMAIN_FACTORY_K_VISUAL_MISMATCH_SUPPORTED"
                 if quality and not baseline_pass else
                 "FACTORY_K_ALREADY_MEETS_VISUAL_PROXY_LIMITS"
                 if quality else "NO_QUALITY_QUALIFIED_INTRINSIC_RESOLUTION")
    return {
        "status": "PASSED" if quality else "REVIEW_REQUIRED",
        "diagnosis": diagnosis,
        "K_image_factory_fixed": k.tolist(), "K_image_fitted": fitted.tolist(),
        "focal_ratios": solved.x[:2].tolist(),
        "principal_offsets_px": solved.x[2:].tolist(),
        "focal_ratio_bounds": list(FOCAL_RATIO_BOUNDS),
        "principal_offset_bound_px": PRINCIPAL_OFFSET_BOUND_PX,
        "parameter_order": ["fx_ratio", "fy_ratio", "cx_offset_px", "cy_offset_px"],
        "boundary_hit_parameter_indices": boundary_hits,
        "jacobian_rank": rank, "jacobian_condition": condition,
        "optimizer": {"success": bool(solved.success), "message": str(solved.message),
                      "nfev": int(solved.nfev), "cost": float(solved.cost),
                      "loss": "soft_l1", "f_scale_px": 8.0},
        "before": before_metrics, "after": after_metrics, "per_session": per_session,
        "fixed_camera_depth_m": {"minimum": float(depth.min()), "maximum": float(depth.max())},
        "before_nonpositive_depth_count": int((depth <= 0).sum()),
        "after_nonpositive_depth_count": int((after_depth <= 0).sum()),
        "acceptance": {"median_limit_px": calibration.FIT_MEDIAN_LIMIT_PX,
                       "each_session_p95_limit_px": calibration.FIT_SESSION_P95_LIMIT_PX,
                       "image_size": list(calibration.IMAGE_SIZE), "boundary_hit_allowed": False,
                       "required_jacobian_rank": 4, "minimum_positive_depth_m": .01},
    }


def _failed_report(path, schema, report_name, report_field, expected_pins):
    """Require the exact prior failed algorithm report and sealed producer chain."""
    reference = source.ref(Path(path))
    audit._verify_exact_ref(reference, project_only=True)
    report = calibration._read(path)
    if (report.get("schema") != schema or report.get("status") != "REVIEW_REQUIRED"
            or report.get("holdout_used") is not False
            or report.get("time_fitted") is not False
            or report.get("scale_fitted") is not False
            or report.get("lens_remap") is not False
            or any(report.get(key) != ref for key, ref in expected_pins.items())):
        raise IntrinsicVerificationError("PRIOR_FAILED_REPORT_INPUT_OR_STATUS_MISMATCH")
    stage = Path(reference["path"]).parent
    stage_result_path = stage / "RESULT.json"
    stage_result_ref = source.ref(stage_result_path)
    stage_result = calibration._read(stage_result_path)
    outer_path = stage.parent / "RESULT.json"
    outer = calibration._read(outer_path)
    if (stage_result.get("status") != "REVIEW_REQUIRED"
            or stage_result.get(report_field) != reference
            or outer.get("status") != "FAILED_RUNTIME_FINAL"
            or outer.get("validation_errors") != []
            or outer.get("stage_result") != stage_result_ref
            or outer.get("required_outputs", {}).get("candidate/RESULT.json") != stage_result_ref
            or outer.get("required_outputs", {}).get(f"candidate/{report_name}") != reference):
        raise IntrinsicVerificationError("PRIOR_FAILED_REPORT_PRODUCER_NOT_SEALED_OR_UNBOUND")
    packet_ref = audit._verify_exact_ref(outer["task_packet"], project_only=True)
    packet = calibration._read(packet_ref["path"])
    if (packet.get("task_id") != outer.get("task_id")
            or packet.get("run_signature") != outer.get("run_signature")
            or not outer.get("run_signature")):
        raise IntrinsicVerificationError("PRIOR_FAILED_REPORT_PRODUCER_SIGNATURE_MISMATCH")
    receipt_path = audit.PROJECT_ROOT / "tasks/receipts" / f"{outer['task_id'].upper()}.json"
    receipt = calibration._read(receipt_path)
    if receipt.get("result") != source.ref(outer_path) or receipt.get("status") != outer["status"]:
        raise IntrinsicVerificationError("PRIOR_FAILED_REPORT_PRODUCER_RECEIPT_MISMATCH")
    return report, {"report": reference, "stage_result": stage_result_ref,
                    "executor_result": source.ref(outer_path),
                    "task_packet": packet_ref, "terminal_receipt": source.ref(receipt_path)}


def run_intrinsic_verification(manifest_path, annotations_path, base_camera_config_path,
                               camera_fit_report_path, installation_fit_report_path, output_root):
    """Verify bounded encoded-domain K only after sealed rigid/prior-fit failures."""
    manifest, config, contexts = calibration._load_inputs(manifest_path, base_camera_config_path)
    for selected in config.get("devices", {}).values():
        if any(key in selected for key in ["K_override", "T_head_camera_override",
                                           "installation_prior_override"]):
            raise IntrinsicVerificationError("FACTORY_CAMERA_AND_RAW_INSTALLATION_PRIOR_MUST_BE_FIXED")
    for context in contexts.values():
        if not np.array_equal(context["model"].T_head_camera,
                              context["model"].T_head_camera_factory):
            raise IntrinsicVerificationError("FACTORY_CAMERA_AND_RAW_INSTALLATION_PRIOR_MUST_BE_FIXED")
    annotations = calibration._read(annotations_path)
    pins = {"input_manifest": source.ref(Path(manifest_path)),
            "base_camera_config": source.ref(Path(base_camera_config_path))}
    if any(annotations.get(key) != ref for key, ref in pins.items()):
        raise IntrinsicVerificationError("ANNOTATION_FROZEN_INPUT_REFERENCE_MISMATCH")
    evidence_ref = annotations.get("source_images")
    if not isinstance(evidence_ref, dict):
        raise IntrinsicVerificationError("ANNOTATION_SOURCE_IMAGES_REFERENCE_MISSING")
    audit._verify_exact_ref(evidence_ref, project_only=True)
    evidence = calibration._read(evidence_ref["path"])
    if (evidence.get("status") != "PASSED" or evidence.get("holdout_used") is not False
            or evidence.get("image_size") != list(calibration.IMAGE_SIZE)
            or any(evidence.get(key) != ref for key, ref in pins.items())):
        raise IntrinsicVerificationError("ANNOTATION_SOURCE_IMAGES_INPUT_BINDING_MISMATCH")
    records, excluded = calibration._validated_correspondences(annotations, evidence, contexts)
    report_pins = {**pins, "annotations": source.ref(Path(annotations_path)),
                   "source_images": evidence_ref}
    camera_report, camera_producer = _failed_report(
        camera_fit_report_path, "pick_place_device_camera_rigid_visual_fit_v1",
        "CALIBRATION_REPORT.json", "calibration_report", report_pins)
    if camera_report.get("correspondences") != records:
        raise IntrinsicVerificationError("CAMERA_FAILURE_CORRESPONDENCES_DRIFT")
    installation_report, installation_producer = _failed_report(
        installation_fit_report_path, INSTALLATION_SCHEMA,
        "INSTALLATION_REPORT.json", "installation_report",
        {**report_pins, "camera_fit_report": camera_producer["report"]})
    devices = {}
    for serial in sorted({record["device_serial"] for record in records}):
        selected = [record for record in records if record["device_serial"] == serial]
        context = next(context for context in contexts.values()
                       if context["model"].device_serial == serial)
        devices[serial] = fit_intrinsic_proxy_points(
            [record["point_head_m"] for record in selected], [record["uv"] for record in selected],
            context["K_image"], context["model"].T_head_camera_factory,
            [record["session_id"] for record in selected])
        sx, sy = context["model"].width / calibration.IMAGE_SIZE[0], \
                 context["model"].height / calibration.IMAGE_SIZE[1]
        devices[serial].update({
            "K_output_fitted": (np.diag([sx, sy, 1.]) @
                                np.asarray(devices[serial]["K_image_fitted"])).tolist(),
            "K_output_factory_fixed": context["model"].K.tolist(),
            "output_size": [context["model"].width, context["model"].height],
            "T_head_camera_factory_fixed": context["model"].T_head_camera_factory.tolist(),
            "camera_snapshot_sha256": context["model"].camera_sha256,
        })
    status = "PASSED" if all(row["status"] == "PASSED" for row in devices.values()) \
             else "REVIEW_REQUIRED"
    report = {
        "schema": "pick_place_device_encoded_intrinsic_verification_v1", "status": status,
        **report_pins, "camera_fit_report": camera_producer["report"],
        "installation_fit_report": installation_producer["report"],
        "preceding_failure_producers": {"camera": camera_producer,
                                       "installation": installation_producer},
        "development_sessions": list(calibration.DEVELOPMENT_SESSIONS), "holdout_used": False,
        "source_sessions": [{"session_id": sid, "source_refs": context["row"]["source_refs"],
                             "installation_prior": context["prior_reference"]}
                            for sid, context in contexts.items()],
        "correspondences": records, "excluded_correspondences": excluded, "devices": devices,
        "fitted_variables": "ONE_DEVICE_ENCODED_DOMAIN_FX_FY_CX_CY_ONLY",
        "K_fitted": True, "camera_extrinsics_fitted": False,
        "camera_extrinsics_authority": "FIXED_ORIGINAL_FACTORY_TRANSFORM",
        "installation_prior_fitted": False, "installation_prior_authority": "FIXED_RECORDED_RAW_PRIOR",
        "time_fitted": False, "qpc_modified": False, "scale_fitted": False,
        "metric_scale_modified": False, "lens_remap": False,
        "image_operation": "SBS_CROP_UNIFORM_RESIZE_ONLY",
        "correspondence_authority": "GLOVE_VISUAL_PROXY_NOT_ANATOMICAL_OR_METRIC_GROUND_TRUTH",
        "eye_identity_qualification": calibration.EYE_QUALIFICATION,
        "encoded_domain_K_qualification": "DEVELOPMENT_GLOVE_VISUAL_PROXY_ESTIMATE",
        "factory_metadata_physical_K_verified": False, "physical_eye_verified": False,
        "metric_calibration_verified": False, "training_eligible": False,
        "control_ground_truth": False,
        "assessment_limits": [
            "Fixed raw prior and metric points may retain errors; K fit does not establish their physical truth.",
            "Only four development sessions and explicitly annotated raw frames inform this fit.",
            "No holdout, lens distortion estimation, source eye identity verification or temporal fitting.",
        ],
    }
    output = audit._output_path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    audit._write_json(output / "INTRINSIC_REPORT.json", report)
    report_ref = source.ref(output / "INTRINSIC_REPORT.json")
    fitted = copy.deepcopy(config)
    fitted.update(status="FROZEN_VISUAL_ESTIMATE", fit_quality_status=status,
                  training_eligible=False, control_ground_truth=False)
    for serial, row in devices.items():
        fitted["devices"][serial].update({
            "K_override": row["K_output_fitted"], "correction_evidence": report_ref,
            "status": "FROZEN_VISUAL_ESTIMATE", "fit_quality_status": row["status"],
            "eye_identity_qualification": calibration.EYE_QUALIFICATION,
            "physical_eye_verified": False, "metric_calibration_verified": False,
            "encoded_domain_K_qualification": "DEVELOPMENT_GLOVE_VISUAL_PROXY_ESTIMATE",
        })
    audit._write_json(output / "FITTED_CAMERA_CONFIG.json", fitted)
    result = {
        "schema": "pick_place_intrinsic_verification_result_v1", "status": status,
        "intrinsic_report": report_ref,
        "fitted_camera_config": source.ref(output / "FITTED_CAMERA_CONFIG.json"),
        "device_fit_status": {serial: row["status"] for serial, row in devices.items()},
        "correspondence_count": len(records), "excluded_count": len(excluded),
        "holdout_used": False, "physical_eye_verified": False,
        "metric_calibration_verified": False, "training_eligible": False,
        "control_ground_truth": False,
    }
    audit._write_json(output / "RESULT.json", result)
    return result
