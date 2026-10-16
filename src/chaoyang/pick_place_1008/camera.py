"""Per-recording camera parsing with explicit encoded-eye route and no lens remap.

T_A_B maps B into A. Factory extrinsics map OpenXR head into capture-sensor
optical coordinates. Encoded upright basis and SBS route are independent,
explicit choices. Absent sourceIndex never silently selects a physical eye.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from chaoyang.pipeline.pico_manus_motion_v2 import REP103_HEAD_TO_OPENXR

class CameraContractError(ValueError):
    pass

SENSOR_FROM_UPRIGHT = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def rigid(value, label: str) -> np.ndarray:
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=2e-5)):
        raise CameraContractError(f"CAMERA_NONRIGID:{label}")
    return t

@dataclass(frozen=True)
class CameraModel:
    device_serial: str
    camera_sha256: str
    eye: str
    sbs_index: int
    native_width: int
    native_height: int
    width: int
    height: int
    K: np.ndarray
    T_head_camera: np.ndarray
    T_head_camera_factory: np.ndarray
    calibration: dict[str, Any]
    @property
    def crop(self) -> tuple[int, int, int, int]:
        return (self.sbs_index * self.native_width, 0, self.native_width, self.native_height)

def normalize_camera(params: dict, meta: dict, *, device_serial: str,
                     camera_sha256: str, config: dict,
                     output_size: tuple[int, int] = (1280, 960)) -> CameraModel:
    if not device_serial or meta.get("device_serial") != device_serial:
        raise CameraContractError("CAMERA_DEVICE_IDENTITY_MISMATCH")
    if meta.get("sha256") != camera_sha256:
        raise CameraContractError("CAMERA_SNAPSHOT_SHA_MISMATCH")
    if config.get("device_serial", device_serial) != device_serial:
        raise CameraContractError("CAMERA_WRONG_DEVICE_CONFIG")
    if config.get("camera_sha256", camera_sha256) != camera_sha256:
        raise CameraContractError("CAMERA_WRONG_SNAPSHOT_CONFIG")
    eye, index = config.get("eye"), config.get("sbs_index")
    if eye not in ("left", "right") or type(index) is not int or index not in (0, 1):
        raise CameraContractError("CAMERA_EYE_ROUTE_MUST_BE_EXPLICIT")
    if config.get("extrinsic_convention") != "head_to_camera":
        raise CameraContractError("CAMERA_EXTRINSIC_DIRECTION_MUST_BE_EXPLICIT")
    if config.get("image_mode") != "encoded_crop_resize_only":
        raise CameraContractError("CAMERA_IMAGE_MODE_MUST_BE_CROP_RESIZE_ONLY")
    route_status = config.get("status")
    if route_status not in ("DEVELOPMENT_CANDIDATE", "FROZEN_VISUAL_ESTIMATE", "VERIFIED_SOURCE_ROUTE"):
        raise CameraContractError("CAMERA_ROUTE_STATUS_MISSING")
    data = params.get(eye, {})
    if "sourceIndex" in data and int(data["sourceIndex"]) != index:
        raise CameraContractError("CAMERA_SOURCE_INDEX_CONFLICT")
    if "width" in params:
        w, h = int(params["width"]), int(params["height"])
        intrinsic = data
        values = np.asarray(data.get("extrinsic"), dtype=np.float64)
        if values.shape != (12,):
            raise CameraContractError("CAMERA_FLAT_EXTRINSIC_SHAPE")
        extrinsic = np.eye(4)
        extrinsic[:3] = values.reshape(3, 4)
        schema = "flat_3x4"
    else:
        w = int(params.get("reference_resolution", {}).get("w", 0))
        h = int(params.get("reference_resolution", {}).get("h", 0))
        intrinsic = data.get("intrinsics", {})
        extrinsic = np.asarray(params.get("extrinsics", {}).get(eye), dtype=np.float64)
        schema = "nested_4x4"
    if w <= 0 or h <= 0:
        raise CameraContractError("CAMERA_NATIVE_SIZE_INVALID")
    extrinsic = rigid(extrinsic, "factory_headXR_to_sensor")
    ow, oh = map(int, output_size)
    sx, sy = ow / w, oh / h
    if min(ow, oh) <= 0 or not np.isclose(sx, sy, atol=1e-12):
        raise CameraContractError("CAMERA_NONUNIFORM_RESIZE")
    try:
        fx, fy, cx, cy = (float(intrinsic[k]) for k in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise CameraContractError("CAMERA_INTRINSIC_MISSING") from exc
    k_factory = np.array([[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]])
    k = np.asarray(config.get("K_override", k_factory), dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise CameraContractError("CAMERA_INTRINSIC_INVALID")
    axes = config.get("upright_axes")
    if axes == "capture_sensor_from_upright":
        u = SENSOR_FROM_UPRIGHT
    elif axes == "identity":
        u = np.eye(3)
    else:
        raise CameraContractError("CAMERA_UPRIGHT_AXES_MUST_BE_EXPLICIT")
    rep_to_xr = np.eye(4)
    rep_to_xr[:3, :3] = REP103_HEAD_TO_OPENXR
    sensor_from_encoded = np.eye(4)
    sensor_from_encoded[:3, :3] = u
    # Encoded camera -> REP103 head; retain this snapshot's factory translation.
    factory = rigid(np.linalg.inv(extrinsic @ rep_to_xr) @ sensor_from_encoded,
                    "REP103_head_from_encoded_camera")
    t = rigid(config.get("T_head_camera_override", factory), "selected_head_from_camera")
    if ("K_override" in config or "T_head_camera_override" in config) and not config.get("correction_evidence"):
        raise CameraContractError("CAMERA_VISUAL_CORRECTION_EVIDENCE_MISSING")
    calibration = {
        "device_serial": device_serial, "camera_snapshot_sha256": camera_sha256,
        "camera_schema": schema, "physical_eye": eye, "sbs_index": index,
        "factory_eye_parameter_key": eye,
        "encoded_eye_identity_qualification": config.get("eye_identity_qualification", "SOURCE_INDEX_DECLARATION" if "sourceIndex" in data else "VISUAL_ASSUMPTION_SOURCEINDEX_ABSENT"),
        "physical_eye_identity_verified": route_status == "VERIFIED_SOURCE_ROUTE" and "sourceIndex" in data,
        "source_sourceIndex": data.get("sourceIndex"),
        "camera_route_status": route_status, "lens_remap": False,
        "image_mode": config["image_mode"], "upright_axes": axes,
        "extrinsic_convention_selected": "T_sensor_headXR; T_headREP_cameraEncoded=inverse(E*A)*U",
        "factory_extrinsic": extrinsic.tolist(), "T_head_camera_factory": factory.tolist(),
        "T_head_camera_used": t.tolist(), "K_factory_scaled": k_factory.tolist(),
        "K_used": k.tolist(), "native_resolution": [w, h], "output_resolution": [ow, oh],
        "camera_K_authority": "FACTORY_K_SCALED_UNVERIFIED_FOR_ENCODED_DOMAIN" if "K_override" not in config else "DEVELOPMENT_FIXED_VISUAL_ESTIMATE",
        "correction_evidence": config.get("correction_evidence"),
        "source_metadata_extrinsics_included": meta.get("extrinsics_included"),
        "source_metadata_extrinsic_convention": meta.get("extrinsic_convention"),
        "source_optics_claim": params.get("format", params.get("video_info", {})),
        "encoded_domain_calibration_qualification": "VISUAL_UNVERIFIED_FACTORY_METADATA_MAY_NAME_RAW_FISHEYE",
        "training_eligible": False, "control_ground_truth": False,
    }
    return CameraModel(device_serial, camera_sha256, eye, index, w, h, ow, oh,
                       k, t, factory, calibration)
