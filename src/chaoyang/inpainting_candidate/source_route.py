"""Reuse the sole adopted V1 encoded-eye assumption without inventing calibration.

Only the manifest's exact camera-config reference is opened. Historical fit
evidence, seed references, motion, pixels, K overrides and installation overrides
are not consumed. Raw sourceIndex declarations always take precedence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT

from .contracts import file_ref

CURRENT_V1_MANIFEST = REPO_ROOT / "inpainting_v1/MANIFEST.json"
ADOPTED_ROUTE = "ADOPTED_FROZEN_VISUAL_ASSUMPTION"


class SourceRouteError(ValueError):
    pass


def _verified_json(reference, label):
    try:
        path = Path(reference["path"])
        actual = file_ref(path)
        if actual != reference:
            raise SourceRouteError(label + "_SHA_DRIFT")
        payload = path.read_bytes()
        import hashlib
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise SourceRouteError(label + "_CHANGED_DURING_READ")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise SourceRouteError(label + "_NOT_OBJECT")
        return value
    except SourceRouteError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceRouteError(label + "_INVALID") from exc


def _current_manifest(reference):
    try:
        if Path(reference["path"]).resolve(strict=True) != CURRENT_V1_MANIFEST.resolve(strict=True):
            raise SourceRouteError("V1_MANIFEST_NOT_CURRENT_POINTER")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceRouteError("V1_MANIFEST_NOT_CURRENT_POINTER") from exc
    manifest = _verified_json(reference, "V1_MANIFEST")
    if (manifest.get("schema_version") != "inpainting-raw-to-robot-adopted-baseline-v1"
            or manifest.get("baseline_id") != "INPAINTING_RAW_TO_ROBOT_V1"
            or manifest.get("selected_baseline_count") != 1
            or manifest.get("status") != "USER_ADOPTED_BATCH_OFFLINE_VISUAL_BASELINE_WITH_KNOWN_LIMITS"
            or manifest.get("actual_motion_source") != "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS"):
        raise SourceRouteError("V1_MANIFEST_CURRENT_CONTRACT_MISMATCH")
    config_ref = manifest.get("inputs", {}).get("camera")
    if not isinstance(config_ref, dict) or Path(config_ref.get("path", "")).name != "FITTED_CAMERA_CONFIG.json":
        raise SourceRouteError("V1_CAMERA_CONFIG_REFERENCE_CONTRACT_MISMATCH")
    return manifest


def bind_adopted_visual_route(source_report: dict, v1_manifest_ref: dict) -> dict:
    """Return a copy with a disclosed V1 crop only for exact supported raw inputs.

    This is a display-domain assumption, never physical-eye or projection
    qualification. A different camera snapshot, unknown device, partial/invalid
    original declaration, or missing metadata remains unresolved.
    """
    result = deepcopy(source_report)
    result["adopted_route_contracts"] = []
    camera = result.get("camera", {})
    routes = camera.get("eye_routes", {})
    # Even an incomplete/invalid original declaration must not be replaced by
    # a visual assumption. The original source validator owns that diagnostic.
    if (camera.get("status") == "SOURCE_INDEX_DECLARATION"
            or any(routes.get(eye) is not None for eye in ("left", "right"))):
        return result
    if camera.get("status") != "BLOCKED_PHYSICAL_EYE_UNRESOLVED":
        return result
    serial = camera.get("camera_device_serial")
    raw_camera = result.get("actual_inputs", {}).get("camera")
    if not serial or not isinstance(raw_camera, dict) or not raw_camera.get("sha256"):
        result["visual_route_adoption"] = {"status": "NOT_APPLIED", "reason": "RAW_SERIAL_OR_CAMERA_SHA_ABSENT"}
        return result
    manifest = _current_manifest(v1_manifest_ref)
    result["adopted_route_contracts"].append(deepcopy(v1_manifest_ref))
    config_ref = manifest.get("inputs", {}).get("camera")
    if not isinstance(config_ref, dict) or Path(config_ref.get("path", "")).name != "FITTED_CAMERA_CONFIG.json":
        raise SourceRouteError("V1_CAMERA_CONFIG_REFERENCE_CONTRACT_MISMATCH")
    config = _verified_json(config_ref, "V1_CAMERA_CONFIG")
    result["adopted_route_contracts"].append(deepcopy(config_ref))
    if (config.get("schema") != "pick_place_1008_device_camera_config_v1"
            or config.get("status") != "FROZEN_VISUAL_ESTIMATE"
            or config.get("control_ground_truth") is not False
            or config.get("training_eligible") is not False
            or not isinstance(config.get("devices"), dict)):
        raise SourceRouteError("V1_CAMERA_CONFIG_CURRENT_CONTRACT_MISMATCH")
    device = config["devices"].get(serial)
    if not isinstance(device, dict):
        result["visual_route_adoption"] = {"status": "NOT_APPLIED", "reason": "RAW_DEVICE_NOT_IN_ADOPTED_CONFIG"}
        return result
    if device.get("camera_sha256") != raw_camera["sha256"]:
        result["visual_route_adoption"] = {"status": "NOT_APPLIED", "reason": "RAW_CAMERA_SHA_NOT_IN_ADOPTED_CONFIG"}
        return result
    if (device.get("device_serial") != serial or device.get("eye") != "right"
            or type(device.get("sbs_index")) is not int or device["sbs_index"] != 1
            or device.get("status") != "FROZEN_VISUAL_ESTIMATE"
            or device.get("physical_eye_verified") is not False
            or device.get("metric_calibration_verified") is not False
            or device.get("eye_identity_qualification") != "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT"
            or device.get("image_mode") != "encoded_crop_resize_only"):
        raise SourceRouteError("V1_DEVICE_ROUTE_CURRENT_CONTRACT_MISMATCH")
    # Binding is to the actual original session camera file, not a caller's
    # copied old hash. No old geometry or per-session fitting inputs are read.
    try:
        expected_path = Path(result["session_dir"]) / "raw/camera_params.json"
        if Path(raw_camera["path"]).resolve(strict=True) != expected_path.resolve(strict=True):
            raise SourceRouteError("RAW_CAMERA_REFERENCE_OUTSIDE_SESSION")
        if file_ref(expected_path) != raw_camera:
            raise SourceRouteError("RAW_CAMERA_CURRENT_SHA_DRIFT")
    except SourceRouteError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceRouteError("RAW_CAMERA_CURRENT_REFERENCE_INVALID") from exc
    camera.update({"status": ADOPTED_ROUTE, "physical_eye": "right", "sbs_index": 1,
                   "physical_eye_verified": False, "projection_qualified": False,
                   "metric_calibration_verified": False,
                   "eye_identity_qualification": "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT",
                   "image_mode": "encoded_crop_resize_only", "lens_remap": False,
                   "adopted_visual_route_ref": deepcopy(config_ref),
                   "adopted_parent_v1_ref": deepcopy(v1_manifest_ref),
                   "next_action": "Evaluate the adopted visual crop; physical eye and metric projection remain unverified."})
    for row in result.get("frame_mapping", []):
        row.update({"physical_eye": "right", "sbs_index": 1, "physical_eye_verified": False,
                    "eye_route_source": ADOPTED_ROUTE})
    capabilities = result.setdefault("capabilities", {})
    capabilities.update({"physical_eye_crop": "ADOPTED_VISUAL_CROP_PHYSICAL_EYE_UNVERIFIED",
                         "encoded_crop_available": ADOPTED_ROUTE,
                         "robot_projection": "VISUAL_ESTIMATE_NOT_METRIC_QUALIFIED"})
    result["visual_route_adoption"] = {"status": ADOPTED_ROUTE, "device_serial": serial,
                                      "camera_sha256": raw_camera["sha256"],
                                      "config_ref": deepcopy(config_ref), "parent_v1_ref": deepcopy(v1_manifest_ref),
                                      "old_fit_evidence_read": False, "old_motion_or_pixels_read": False,
                                      "physical_eye_verified": False, "projection_qualified": False}
    return result
