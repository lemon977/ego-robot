#!/usr/bin/env python3
"""Convert one validated rgb30_v1 clean bundle into the tracker session contract.

The source is immutable.  A new target is published by atomic rename only after
strict validation.  Camera calibration is always taken from this source
session; physical eye identity follows camera_params.<eye>.sourceIndex.  When
``--clean-bundle`` is supplied it is the only timeline authority; legacy
aligned.jsonl/dataset.hdf5 are not consumed for time or frame selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import cv2
import h5py
import numpy as np


class ContractError(RuntimeError):
    pass


DIRECT_SOURCE_IMAGE_MODE = "passthrough_scaled_source_domain"
LEGACY_DIRECT_IMAGE_MODE = "passthrough_scaled_pinhole"
DIRECT_IMAGE_MODES = {DIRECT_SOURCE_IMAGE_MODE, LEGACY_DIRECT_IMAGE_MODE}


# HDF5 poses use REP-103 (X forward, Y left, Z up), while the PICO camera
# extrinsic consumes OpenXR head axes (X right, Y up, Z back).
REP103_HEAD_TO_OPENXR_HEAD = np.asarray([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":") if compact else None,
                      indent=None if compact else 2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def source_snapshot(source: Path) -> dict[str, dict[str, Any]]:
    names = [
        "manifest.json", "dataset.hdf5", "aligned.jsonl",
        "raw/vst.h264", "raw/vst.ts.jsonl", "raw/vst.qpc.ts.jsonl",
        "raw/camera_params.json", "raw/camera_params.meta.json",
        "raw/pico.jsonl", "raw/manus.jsonl", "raw/manus.meta.json",
        "raw/tactile.jsonl", "raw/tactile.meta.json",
    ]
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = source / name
        if not path.is_file():
            raise ContractError(f"required source file missing: {path}")
        stat = path.stat()
        result[name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    return result


def require_same_snapshot(before: dict[str, Any], after: dict[str, Any]) -> None:
    def content(value: dict[str, Any] | None) -> tuple[Any, Any]:
        return (None, None) if value is None else (value.get("bytes"), value.get("sha256"))
    changed = sorted(key for key in set(before) | set(after)
                     if content(before.get(key)) != content(after.get(key)))
    if changed:
        raise ContractError(f"source content changed during conversion: {changed}")


def same_content_snapshot(before: dict[str, Any] | None,
                          after: dict[str, Any] | None) -> bool:
    if before is None or after is None:
        return before is after
    keys = set(before) | set(after)
    return all((before.get(key, {}).get("bytes"), before.get(key, {}).get("sha256")) ==
               (after.get(key, {}).get("bytes"), after.get(key, {}).get("sha256"))
               for key in keys)


def clean_bundle_snapshot(clean_bundle: Path | None) -> dict[str, dict[str, Any]] | None:
    if clean_bundle is None:
        return None
    result = {}
    for name in ("BUNDLE_MANIFEST.json", "QA_POLICY.json", "CLEAN_TIMELINE.npz",
                 "clean_dataset.hdf5"):
        path = clean_bundle/name
        if not path.is_file():
            raise ContractError(f"clean bundle file missing: {path}")
        stat = path.stat()
        result[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "sha256": sha256(path)}
    return result


def quat_to_rot(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(list(q), dtype=np.float64)
    norm = float(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ContractError("invalid xyzw quaternion")
    s = 2.0 / norm
    return np.asarray([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ContractError("pose must be finite xyz+xyzw")
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_rot(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def rot_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    # Stable branch algorithm, returning xyzw.
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.asarray([(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s,
                        (r[1,0]-r[0,1])/s, 0.25*s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2
            q = np.asarray([0.25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s,
                            (r[2,1]-r[1,2])/s])
        elif i == 1:
            s = math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2
            q = np.asarray([(r[0,1]+r[1,0])/s, 0.25*s, (r[1,2]+r[2,1])/s,
                            (r[0,2]-r[2,0])/s])
        else:
            s = math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2
            q = np.asarray([(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, 0.25*s,
                            (r[1,0]-r[0,1])/s])
    q /= np.linalg.norm(q)
    return q


def matrix_pose_string(matrix: np.ndarray) -> str:
    values = np.r_[matrix[:3, 3], rot_to_quat(matrix[:3, :3])]
    return ",".join(f"{float(x):.12g}" for x in values)


def eye_extrinsic(calibration: dict[str, Any], eye: str) -> np.ndarray:
    if calibration.get("extrinsicConvention") != "head_to_camera_3x4_row_major":
        raise ContractError("unsupported camera extrinsic convention")
    values = np.asarray(calibration[eye]["extrinsic"], dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ContractError(f"invalid {eye} extrinsic")
    matrix = np.eye(4)
    matrix[:3] = values.reshape(3, 4)
    if abs(np.linalg.det(matrix[:3, :3]) - 1.0) > 1e-3:
        raise ContractError(f"{eye} extrinsic rotation is not proper")
    return matrix


def validate_calibration(calibration: dict[str, Any]) -> None:
    if calibration.get("cameraOrder") != "physical_by_extrinsic_camera_center_x":
        raise ContractError("camera order contract missing")
    indices = []
    for eye in ("left", "right"):
        data = calibration.get(eye, {})
        if not data.get("isValidCalibration"):
            raise ContractError(f"{eye} calibration is not valid")
        if data.get("distortion", {}).get("model") != "equiDis62":
            raise ContractError(f"{eye} distortion is not equiDis62")
        coeffs = np.asarray(data["distortion"]["coeffs"], dtype=float)
        if coeffs.shape != (8,) or not np.isfinite(coeffs).all():
            raise ContractError(f"{eye} distortion coefficients invalid")
        for key in ("fx", "fy", "cx", "cy"):
            if not math.isfinite(float(data[key])):
                raise ContractError(f"{eye}.{key} invalid")
        eye_extrinsic(calibration, eye)
        indices.append(int(data.get("sourceIndex", -1)))
    if sorted(indices) != [0, 1]:
        raise ContractError(f"physical eye sourceIndex must be a permutation of [0,1], got {indices}")


def select_physical_eye(frame: np.ndarray, calibration: dict[str, Any], eye: str) -> np.ndarray:
    if eye not in ("left", "right") or frame.ndim != 3 or frame.shape[1] % 2:
        raise ContractError("invalid stereo frame/eye")
    eye_width = frame.shape[1] // 2
    source_index = int(calibration[eye]["sourceIndex"])
    if source_index not in (0, 1):
        raise ContractError("eye sourceIndex is outside SBS frame")
    return frame[:, source_index*eye_width:(source_index+1)*eye_width]


def validate_video_indices(indices: np.ndarray, source_count: int) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ContractError("video_frame_idx must be a non-empty vector")
    if indices.min() < 0 or indices.max() >= source_count or np.any(np.diff(indices) < 0):
        raise ContractError("video_frame_idx must be in-range and monotonic non-decreasing")


def build_rectification_map(calibration: dict[str, Any], eye: str,
                            eye_width: int, eye_height: int,
                            output_width: int, output_height: int,
                            horizontal_fov_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    native_width, native_height = int(calibration["width"]), int(calibration["height"])
    sx, sy = eye_width/native_width, eye_height/native_height
    if abs(sx-sy) > 1e-9:
        raise ContractError(f"non-uniform calibration scaling: {sx} != {sy}")
    data = calibration[eye]
    coefficients = np.asarray(data["distortion"]["coeffs"], dtype=np.float64)
    yy, xx = np.indices((output_height, output_width), dtype=np.float64)
    if not 60.0 <= horizontal_fov_deg <= 140.0:
        raise ContractError("horizontal FOV must be in [60,140] degrees")
    focal = output_width / (2.0 * math.tan(math.radians(horizontal_fov_deg)/2.0))
    x = (xx - (output_width-1)/2.0) / focal
    y = (yy - (output_height-1)/2.0) / focal
    radius = np.hypot(x, y)
    theta = np.arctan(radius)
    phi = np.arctan2(y, x)
    theta_d = theta.copy()
    for i in range(6):
        theta_d += coefficients[i] * theta ** (2*i+3)
    xd = np.where(radius > 1e-12, theta_d*np.cos(phi), 0.0)
    yd = np.where(radius > 1e-12, theta_d*np.sin(phi), 0.0)
    r2 = xd*xd + yd*yd
    p1, p2 = coefficients[6:]
    x0, y0 = xd.copy(), yd.copy()
    xd = x0 + 2*p1*x0*y0 + p2*(r2+2*x0*x0)
    yd = y0 + p1*(r2+2*y0*y0) + 2*p2*x0*y0
    mx = (float(data["fx"])*xd + float(data["cx"])) * sx
    my = (float(data["fy"])*yd + float(data["cy"])) * sy
    return mx.astype(np.float32), my.astype(np.float32)


def output_intrinsics(calibration: dict[str, Any], eye: str, width: int, height: int,
                      horizontal_fov_deg: float, image_domain_mode: str) -> np.ndarray:
    if image_domain_mode in DIRECT_IMAGE_MODES:
        sx, sy = width/float(calibration["width"]), height/float(calibration["height"])
        data = calibration[eye]
        return np.asarray([[float(data["fx"])*sx,0,float(data["cx"])*sx],
                           [0,float(data["fy"])*sy,float(data["cy"])*sy],[0,0,1]],float)
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg)/2.0))
    return np.asarray([[focal,0,(width-1)/2.0],[0,focal,(height-1)/2.0],[0,0,1]],float)


def build_pinhole_to_pinhole_map(calibration: dict[str, Any], eye: str,
                                  width: int, height: int, target_k: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    data=calibration[eye]
    yy,xx=np.indices((height,width),dtype=np.float64)
    mx=float(data["fx"])*(xx-target_k[0,2])/target_k[0,0]+float(data["cx"])
    my=float(data["fy"])*(yy-target_k[1,2])/target_k[1,1]+float(data["cy"])
    return mx.astype(np.float32),my.astype(np.float32)


def load_source(source: Path, clean_bundle: Path | None = None) -> dict[str, Any]:
    calibration = json.loads((source/"raw/camera_params.json").read_text())
    validate_calibration(calibration)
    camera_meta = json.loads((source/"raw/camera_params.meta.json").read_text())
    wall_ts = np.asarray([int(x) for x in (source/"raw/vst.ts.jsonl").read_text().splitlines() if x.strip()], dtype=np.int64)
    qpc_ts = np.asarray([int(x) for x in (source/"raw/vst.qpc.ts.jsonl").read_text().splitlines() if x.strip()], dtype=np.int64)
    if wall_ts.shape != qpc_ts.shape or len(wall_ts) < 2 or np.any(np.diff(wall_ts) <= 0) or np.any(np.diff(qpc_ts) <= 0):
        raise ContractError("VST timestamp sidecars must have equal, strictly increasing timelines")
    bundle_manifest = None
    hdf5_path = source/"dataset.hdf5"
    if clean_bundle is not None:
        manifest_path = clean_bundle/"BUNDLE_MANIFEST.json"
        if not manifest_path.is_file():
            raise ContractError(f"clean bundle manifest missing: {manifest_path}")
        bundle_manifest = json.loads(manifest_path.read_text())
        required_contract = {
            "schema_version": "chips_cards_handle_rgb30_v1",
            "authoritative": True,
            "source_level": "RAW_SOURCE_ARCHIVE",
            "clock_mapping_count": 1,
            "resample_count": 1,
            "timeline_policy": "VST_QPC_RGB30",
            "timeline_rate_hz": 30,
        }
        mismatch = {key: (bundle_manifest.get(key), value) for key, value in required_contract.items()
                    if bundle_manifest.get(key) != value}
        if mismatch:
            raise ContractError(f"clean bundle rejects secondary alignment/resampling: {mismatch}")
        hdf5_path = clean_bundle/"clean_dataset.hdf5"
        if not hdf5_path.is_file():
            raise ContractError(f"canonical clean HDF5 missing: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        required = ["timestamp_ns", "recv_wall_ns", "recv_qpc_ns", "segment_id", "source_row_idx",
                    "video_frame_idx", "head_pose", "left_controller_pose", "right_controller_pose",
                    "left_wrist_pose", "right_wrist_pose",
                    "video_offset_ms", "left_hand_joints", "right_hand_joints",
                    "left_hand_valid", "right_hand_valid"]
        missing = [x for x in required if x not in f]
        if missing:
            raise ContractError(f"HDF5 datasets missing: {missing}")
        tactile = [
            f"{side}_{suffix}" for side in ("left", "right") for suffix in (
                "tactile_values", "tactile_wire_active_mask", "tactile_wire_valid_mask",
                "tactile_fingers", "tactile_fingers_active_mask", "tactile_fingers_valid_mask",
                "tactile_offline_source_index", "tactile_offline_source_valid",
                "tactile_offline_offset_ms", "tactile_causal_source_index",
                "tactile_causal_source_valid", "tactile_sequence", "tactile_causal_age_ms",
            )
        ]
        if clean_bundle is not None:
            missing_tactile = [name for name in tactile if name not in f]
            if missing_tactile:
                raise ContractError(f"clean HDF5 tactile contract missing: {missing_tactile}")
            required.extend(tactile)
        arrays = {name: f[name][:] for name in required}
        attrs = {key: f.attrs[key] for key in f.attrs}
    n = len(arrays["timestamp_ns"])
    time_varying = [value for name, value in arrays.items()
                    if "active_mask" not in name]
    if any(len(value) != n for value in time_varying):
        raise ContractError("HDF5 first dimension mismatch")
    idx = np.asarray(arrays["video_frame_idx"], dtype=np.int64)
    validate_video_indices(idx, len(wall_ts))
    if not bool(attrs.get("all_exported_frames_complete")):
        raise ContractError("HDF5 does not declare all exported frames complete")
    if attrs.get("schema") != "egodex_v1" or attrs.get("pose_layout") != "pos(xyz) + quat(x,y,z,w)":
        raise ContractError("unsupported HDF5 schema/pose layout")
    if clean_bundle is not None:
        if attrs.get("bundle_uuid") != bundle_manifest.get("bundle_uuid"):
            raise ContractError("clean HDF5/BUNDLE_MANIFEST bundle_uuid mismatch")
        if attrs.get("timeline_sha256") != bundle_manifest.get("timeline_sha256"):
            raise ContractError("clean HDF5/BUNDLE_MANIFEST timeline SHA mismatch")
    return {"calibration": calibration, "camera_meta": camera_meta,
            "wall_ts": wall_ts, "qpc_ts": qpc_ts, "arrays": arrays, "attrs": attrs,
            "bundle_manifest": bundle_manifest,
            "clean_bundle": str(clean_bundle) if clean_bundle is not None else None}


def camera_and_hand_geometry(bundle: dict[str, Any], selected_calibration_eye: str = "left") -> dict[str, Any]:
    arrays, calibration = bundle["arrays"], bundle["calibration"]
    extrinsic = eye_extrinsic(calibration, selected_calibration_eye)
    rep_to_openxr = np.eye(4)
    rep_to_openxr[:3, :3] = REP103_HEAD_TO_OPENXR_HEAD
    effective_extrinsic = extrinsic @ rep_to_openxr
    heads = np.stack([pose_to_matrix(x) for x in arrays["head_pose"]])
    world_cameras = np.stack([head @ np.linalg.inv(effective_extrinsic) for head in heads])
    inv_first = np.linalg.inv(world_cameras[0])
    c2w = np.stack([inv_first @ camera for camera in world_cameras])
    result: dict[str, Any] = {"c2w": c2w, "world_cameras": world_cameras,
                              "effective_extrinsic": effective_extrinsic,
                              "selected_calibration_eye": selected_calibration_eye}
    joint_names = json.loads(bundle["attrs"]["joint_names"])
    if len(joint_names) != 25:
        raise ContractError("MANUS joint_names must contain 25 entries")
    for side in ("left", "right"):
        controller = np.stack([pose_to_matrix(x) for x in arrays[f"{side}_controller_pose"]])
        wrist = np.stack([pose_to_matrix(x) for x in arrays[f"{side}_wrist_pose"]])
        local = np.asarray(arrays[f"{side}_hand_joints"], dtype=np.float64)
        world = np.einsum("tij,tkj->tki", wrist[:,:3,:3], local) + wrist[:,None,:3,3]
        first = np.einsum("ij,tkj->tki", inv_first[:3,:3], world) + inv_first[:3,3]
        inv_current = np.linalg.inv(world_cameras)
        current = np.einsum("tij,tkj->tki", inv_current[:,:3,:3], world) + inv_current[:,None,:3,3]
        wrist_first = np.stack([inv_first @ x for x in wrist])
        wrist_current = np.stack([inv_current[i] @ wrist[i] for i in range(len(wrist))])
        controller_first = np.stack([inv_first @ x for x in controller])
        controller_current = np.stack([inv_current[i] @ controller[i] for i in range(len(controller))])
        result[side] = {"local": local, "world": first, "camera": current,
                        "wrist_world": wrist_first, "wrist_camera": wrist_current,
                        "controller_source_rep103": controller,
                        "controller_world": controller_first,
                        "controller_camera": controller_current,
                        "joint_names": joint_names}
    return result


def semantic_visibility(geometry: dict[str, Any], width: int, height: int,
                        horizontal_fov_deg: float = 90.0,
                        camera_k: np.ndarray | None = None) -> dict[str, Any]:
    if camera_k is None:
        focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg)/2.0))
        fx=fy=focal;cx,cy=(width-1)/2.0,(height-1)/2.0
    else:
        fx,fy,cx,cy=(float(camera_k[0,0]),float(camera_k[1,1]),
                     float(camera_k[0,2]),float(camera_k[1,2]))
        horizontal_fov_deg = math.degrees(2.0*math.atan(width/(2.0*fx)))
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        points = geometry[side]["camera"]
        z = points[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx*points[...,0]/z + cx
            v = fy*points[...,1]/z + cy
        visible = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        wrist_points = geometry[side]["wrist_camera"][:, :3, 3]
        wrist_z = wrist_points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            wrist_u = fx*wrist_points[:,0]/wrist_z + cx
            wrist_v = fy*wrist_points[:,1]/wrist_z + cy
        wrist_visible = ((wrist_z > 0) & (wrist_u >= 0) & (wrist_u < width)
                          & (wrist_v >= 0) & (wrist_v < height))
        result[side] = {
            "positive_depth_joint_fraction": float((z > 0).mean()),
            "in_frame_joint_fraction": float(visible.mean()),
            "in_frame_wrist_frames": int(wrist_visible.sum()),
            "in_frame_wrist_fraction": float(wrist_visible.mean()),
        }
    issues = []
    if not all(result[side]["in_frame_wrist_fraction"] >= 0.5 for side in ("left", "right")):
        issues.append("VISUAL_SENSOR_PROJECTION_MISMATCH")
    right_translation = geometry["right"]["wrist_world"][:, :3, 3]
    right_range_mm = np.ptp(right_translation, axis=0) * 1000.0
    result["right_wrist_translation_range_mm"] = right_range_mm.tolist()
    result["right_wrist_unique_translation_1um"] = int(
        len(np.unique(np.round(right_translation, 6), axis=0))
    )
    if float(np.linalg.norm(right_range_mm)) < 0.1:
        issues.append("RIGHT_WRIST_FROZEN_TRACKING")
    result["fatal_issues"] = issues
    admissible = not issues
    result["status"] = "SESSION_CONTENT_ADMISSIBLE" if admissible else "SESSION_CONTENT_NOT_ADMISSIBLE"
    result["policy"] = "both source wrists must project inside the selected eye in >=50% of aligned rows"
    result["horizontal_fov_deg"] = horizontal_fov_deg
    return result


def write_projection_probe(staging: Path, geometry: dict[str, Any], width: int, height: int,
                           camera_k: np.ndarray) -> Path:
    indices = np.linspace(0, len(geometry["c2w"])-1, 8, dtype=int)
    tiles = []
    fx,fy,cx,cy=(float(camera_k[0,0]),float(camera_k[1,1]),
                 float(camera_k[0,2]),float(camera_k[1,2]))
    for frame_index in indices:
        image = cv2.imread(str(staging/"preprocess/all_data"/f"{frame_index:05d}"/"rgb.png"))
        if image is None:
            raise ContractError("projection probe cannot read RGB")
        for side, color in (("left", (255,0,0)), ("right", (0,0,255))):
            points = geometry[side]["camera"][frame_index]
            z = points[:,2]
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = np.stack([fx*points[:,0]/z+cx, fy*points[:,1]/z+cy], axis=1)
            inside = (z>0)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            for point in uv[inside]:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)
            wrist = geometry[side]["wrist_camera"][frame_index,:3,3]
            if wrist[2] > 0:
                point = np.asarray([fx*wrist[0]/wrist[2]+cx, fy*wrist[1]/wrist[2]+cy])
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.drawMarker(image, tuple(np.rint(point).astype(int)), color,
                                   cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
            label = f"{side[0].upper()} z={wrist[2]:+.3f}m"
            cv2.putText(image,label,(12,55 if side=="left" else 90),cv2.FONT_HERSHEY_SIMPLEX,
                        .8,color,2,cv2.LINE_AA)
        cv2.putText(image,f"target {frame_index:04d}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        tiles.append(cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA))
    sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,8,4)])
    path=staging/"review/manus_projection_eight_frame.jpg";path.parent.mkdir(parents=True)
    if not cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,92]):
        raise ContractError("projection probe write failed")
    return path


def ffmpeg_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.12f}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(path)
    ], stdin=subprocess.PIPE)


def encode_and_materialize(source: Path, staging: Path, session: str,
                           bundle: dict[str, Any], geometry: dict[str, Any],
                           width: int, height: int, final_target: Path,
                           horizontal_fov_deg: float,
                           image_domain_mode: str,
                           target_k: np.ndarray,
                           prebuilt_mono: Path | None = None) -> None:
    arrays = bundle["arrays"]
    indices = np.asarray(arrays["video_frame_idx"], dtype=np.int64)
    calibration = bundle["calibration"]
    selected_eye = geometry["selected_calibration_eye"]
    cap = cv2.VideoCapture(str(source/"raw/vst.h264"))
    ok, first = cap.read()
    if not ok or first is None:
        raise ContractError("cannot decode raw H264")
    sh, sw = first.shape[:2]
    if sw % 2 or sh != int(calibration["height"]) or sw//2 != int(calibration["width"]):
        raise ContractError(f"decoded VST geometry {sw}x{sh} does not match calibration")
    map_x = map_y = None
    if image_domain_mode == "equidis62_rectify":
        map_x, map_y = build_rectification_map(calibration, selected_eye, sw//2, sh, width, height,
                                               horizontal_fov_deg)
    elif image_domain_mode == "pinhole_to_pinhole_fov":
        map_x, map_y = build_pinhole_to_pinhole_map(calibration, selected_eye, width, height, target_k)
    elif image_domain_mode not in DIRECT_IMAGE_MODES:
        raise ContractError(f"unknown image domain mode: {image_domain_mode}")
    stereo_dir = staging/"source_stereo"
    stereo_dir.mkdir(parents=True)
    stereo_path = stereo_dir/f"CameraRecord_{session}_stereo.mp4"
    local_descriptor, local_stereo_name = tempfile.mkstemp(prefix="handle-stereo-", suffix=".mp4")
    os.close(local_descriptor)
    Path(local_stereo_name).unlink()
    local_stereo_path = Path(local_stereo_name)
    mono_path = staging/f"CameraRecord_{session}.mp4"
    local_mono_path: Path | None = None
    if prebuilt_mono is not None:
        if not prebuilt_mono.is_file():
            raise ContractError(f"prebuilt mono video missing: {prebuilt_mono}")
        prebuilt_probe = probe(prebuilt_mono)
        if (int(prebuilt_probe["width"]), int(prebuilt_probe["height"]),
                int(prebuilt_probe["nb_read_frames"])) != (width, height, len(indices)):
            raise ContractError(f"prebuilt mono video contract mismatch: {prebuilt_probe}")
        shutil.copyfile(prebuilt_mono, mono_path)
    stereo_encoder = ffmpeg_encoder(local_stereo_path, sw, sh, float(bundle["attrs"]["fps"]))
    if prebuilt_mono is None:
        mono_descriptor, local_mono_name = tempfile.mkstemp(prefix="handle-mono-", suffix=".mp4")
        os.close(mono_descriptor)
        Path(local_mono_name).unlink()
        local_mono_path = Path(local_mono_name)
        mono_encoder = ffmpeg_encoder(local_mono_path, width, height, float(bundle["attrs"]["fps"]))
    else:
        mono_encoder = None
    all_data = staging/"preprocess/all_data"
    all_data.mkdir(parents=True)
    current_idx, output_row = 0, 0
    decoded = 0
    frame = first
    failure: BaseException | None = None
    try:
        while True:
            while output_row < len(indices) and int(indices[output_row]) == current_idx:
                selected = select_physical_eye(frame, calibration, selected_eye)
                if image_domain_mode in DIRECT_IMAGE_MODES:
                    rectified = cv2.resize(selected,(width,height),interpolation=cv2.INTER_AREA)
                else:
                    assert map_x is not None and map_y is not None
                    rectified = cv2.remap(selected, map_x, map_y, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT)
                assert stereo_encoder.stdin is not None
                stereo_encoder.stdin.write(frame.tobytes())
                if mono_encoder is not None:
                    assert mono_encoder.stdin is not None
                    mono_encoder.stdin.write(rectified.tobytes())
                frame_dir = all_data/f"{output_row:05d}"
                frame_dir.mkdir()
                rgb = frame_dir/"rgb.png"
                if not cv2.imwrite(str(rgb), rectified, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise ContractError(f"failed to write {rgb}")
                # CPFS and some object-backed mounts reject hardlinks.  The
                # compatibility alias must therefore be an ordinary file.
                shutil.copyfile(rgb, frame_dir/"rgb_WoArm_WArmObjKpts.png")
                write_training_record(frame_dir, final_target, session, output_row, bundle, geometry,
                                      width, height, target_k, image_domain_mode)
                output_row += 1
            decoded += 1
            ok, frame = cap.read()
            if not ok:
                break
            current_idx += 1
    except BaseException as exc:
        failure = exc
    finally:
        cap.release()
        encoders = [stereo_encoder] + ([mono_encoder] if mono_encoder is not None else [])
        for encoder in encoders:
            if encoder.stdin is not None:
                encoder.stdin.close()
        return_codes = [encoder.wait() for encoder in encoders]
    if failure is not None:
        local_stereo_path.unlink(missing_ok=True)
        if local_mono_path is not None:
            local_mono_path.unlink(missing_ok=True)
        raise failure
    if any(return_codes):
        local_stereo_path.unlink(missing_ok=True)
        if local_mono_path is not None:
            local_mono_path.unlink(missing_ok=True)
        raise ContractError(f"ffmpeg failed return_codes={return_codes}")
    shutil.copyfile(local_stereo_path, stereo_path)
    local_stereo_path.unlink()
    if local_mono_path is not None:
        shutil.copyfile(local_mono_path, mono_path)
        local_mono_path.unlink()
    if output_row != len(indices):
        raise ContractError(f"decoded source did not cover HDF5 rows: {output_row}/{len(indices)}")
    if decoded != len(bundle["wall_ts"]):
        raise ContractError(f"decoded frame count {decoded} != timestamp count {len(bundle['wall_ts'])}")


def write_training_record(frame_dir: Path, target: Path, session: str, row: int,
                          bundle: dict[str, Any], geometry: dict[str, Any],
                          width: int, height: int, target_k: np.ndarray,
                          image_domain_mode: str) -> None:
    arrays = bundle["arrays"]
    fps = float(bundle["attrs"]["fps"])
    k = target_k.tolist()
    hands: dict[str, Any] = {}
    for side in ("left", "right"):
        hand = geometry[side]
        hands[side] = {
            "T_hand_to_world": hand["wrist_world"][row].tolist(),
            "T_wrist_to_world": hand["wrist_world"][row].tolist(),
            "T_wrist_to_camera": hand["wrist_camera"][row].tolist(),
            "controller6d": {
                "T_controller_to_source_world_rep103": hand["controller_source_rep103"][row].tolist(),
                "T_controller_to_world": hand["controller_world"][row].tolist(),
                "T_controller_to_camera": hand["controller_camera"][row].tolist(),
                "pose_source": "egodex_v1_hdf5_controller_pose",
                "units": "metres",
                "pico21_authority": False,
            },
            "confidence": 1.0 if bool(arrays[f"{side}_hand_valid"][row]) else 0.0,
            "pose_source": "egodex_v1_hdf5_wrist_pose",
            "landmark_contract": "MANUS25_WRIST_LOCAL_EXTENSION_NOT_PICO21",
            "manus25": {
                "joint_names": hand["joint_names"],
                "keypoints_3d_wrist_local": hand["local"][row].tolist(),
                "keypoints_3d_world": hand["world"][row].tolist(),
                "keypoints_3d_camera": hand["camera"][row].tolist(),
                "joint_valid": [bool(arrays[f"{side}_hand_valid"][row])]*25,
            },
        }
    tactile: dict[str, Any] = {}
    if "left_tactile_values" in arrays:
        for side in ("left", "right"):
            grid = np.asarray(arrays[f"{side}_tactile_fingers"][row], dtype=np.float32)
            # JSON has no NaN.  None plus the two masks preserves the difference
            # between a nonexistent taxel and a real zero-pressure reading.
            grid_json = np.where(np.isfinite(grid), grid, None).tolist()
            tactile[side] = {
                "schema_version": "tactile-rgb30-frame-v1",
                "wire_values_369": np.asarray(arrays[f"{side}_tactile_values"][row],
                                               dtype=np.int16).astype(int).tolist(),
                "wire_active_mask_369": np.asarray(
                    arrays[f"{side}_tactile_wire_active_mask"], dtype=bool).tolist(),
                "wire_valid_mask_369": np.asarray(
                    arrays[f"{side}_tactile_wire_valid_mask"][row], dtype=bool).tolist(),
                "finger_grid_5x4x8": grid_json,
                "active_mask_5x4x8": np.asarray(
                    arrays[f"{side}_tactile_fingers_active_mask"], dtype=bool).tolist(),
                "valid_mask_5x4x8": np.asarray(
                    arrays[f"{side}_tactile_fingers_valid_mask"][row], dtype=bool).tolist(),
                "causal_source_index": int(arrays[f"{side}_tactile_causal_source_index"][row]),
                "causal_source_valid": bool(arrays[f"{side}_tactile_causal_source_valid"][row]),
                "offline_source_index": int(arrays[f"{side}_tactile_offline_source_index"][row]),
                "offline_source_valid": bool(arrays[f"{side}_tactile_offline_source_valid"][row]),
                "offline_source_offset_ms": float(arrays[f"{side}_tactile_offline_offset_ms"][row]),
                "source_sequence": int(arrays[f"{side}_tactile_sequence"][row]),
                "causal_sample_age_ms": float(arrays[f"{side}_tactile_causal_age_ms"][row]),
                "materialized_view": "causal_previous",
                "offline_payload": "clean_bundle/clean_dataset.hdf5",
            }
    rgb_rel = f"preprocess/all_data/{row:05d}/rgb.png"
    alias_rel = f"preprocess/all_data/{row:05d}/rgb_WoArm_WArmObjKpts.png"
    source_domain_passthrough = image_domain_mode in DIRECT_IMAGE_MODES
    selected = bundle["calibration"][geometry["selected_calibration_eye"]]
    source_distortion = selected["distortion"]
    record = {
        "metadata": {
            "idx": row, "ts": int(arrays["timestamp_ns"][row]), "video_time_s": row/fps,
            "tracking_index": row, "tracking_sync_error_ms": float(bundle["attrs"].get("max_skew_ms", 30.0)),
            "w": width, "h": height, "fps": fps, "k": k,
            "d": source_distortion["coeffs"] if source_domain_passthrough else [0.0]*5,
            "k_authority": "SAME_SESSION_FACTORY_SOURCE_K_SCALED_UNVERIFIED_FOR_ENCODED_DOMAIN",
            "d_semantics": "DISTORTION_NOT_APPLIED_IN_PASSTHROUGH_ENCODED_DOMAIN",
            "c2w": geometry["c2w"][row].tolist(),
            "camera_eye": "left",
            "camera_eye_semantics": "legacy_tracker_left_slot",
            "camera_physical_calibration_eye": geometry["selected_calibration_eye"],
            "camera_source_index": int(bundle["calibration"][geometry["selected_calibration_eye"]]["sourceIndex"]),
            "camera_calibration_key": geometry["selected_calibration_eye"],
            "camera_model": ("scaled_source_equiDis62_visual"
                             if source_domain_passthrough else "rectified_pinhole"),
            "image_domain_mode": image_domain_mode,
            "rectified": not source_domain_passthrough,
            "world_coordinate_system": "first_frame_selected_camera_optical",
            "camera_coordinate_system": {"x":"right","y":"down","z":"forward","units":"metres"},
            "anchor_key": "camera_start", "is_finished": 0.0,
            "source_alignment": {
                "schema": (bundle.get("bundle_manifest") or {}).get("schema_version", "egodex_v1"),
                "hdf5_row": row,
                "source_row_idx": int(arrays["source_row_idx"][row]),
                "video_frame_idx": int(arrays["video_frame_idx"][row]),
                "segment_id": int(arrays["segment_id"][row]),
                "recv_qpc_ns": int(arrays["recv_qpc_ns"][row]),
                "source_level": (bundle.get("bundle_manifest") or {}).get(
                    "source_level", "ACQUISITION_ALIGNED_HDF5"
                ),
                "clock_mapping_count": (bundle.get("bundle_manifest") or {}).get(
                    "clock_mapping_count", 1
                ),
                "resample_count": (bundle.get("bundle_manifest") or {}).get("resample_count", 1),
            },
        },
        "obs": {"rgb_path": rgb_rel, "rgb_WoArm_WArmObjKpts_path": alias_rel},
        "entities": {"hands": hands, "tactile": tactile, "objects": {}},
    }
    json_dump(frame_dir/"training_data.json", record, compact=True)


def write_metadata(source: Path, staging: Path, session: str,
                   bundle: dict[str, Any], geometry: dict[str, Any],
                   visibility: dict[str, Any], width: int, height: int,
                   horizontal_fov_deg: float, image_domain_mode: str,
                   target_k: np.ndarray) -> None:
    calibration, arrays = bundle["calibration"], bundle["arrays"]
    selected_eye = geometry["selected_calibration_eye"]
    selected = calibration[selected_eye]
    source_domain_passthrough = image_domain_mode in DIRECT_IMAGE_MODES
    encoded_distortion = selected["distortion"]["coeffs"] if source_domain_passthrough else [0.0]*5
    encoded_camera_model = ("scaled_source_equiDis62_visual"
                            if source_domain_passthrough else "rectified_pinhole")
    n, fps = len(arrays["timestamp_ns"]), float(bundle["attrs"]["fps"])
    camera_sha = sha256(source/"raw/camera_params.json")
    effective_fov_deg = math.degrees(2.0*math.atan(width/(2.0*float(target_k[0,0]))))
    def eye(name: str) -> dict[str, Any]:
        d = calibration[name]
        return {"intrinsics": {k: float(d[k]) for k in ("fx","fy","cx","cy")},
                "distortion": d["distortion"], "sourceIndex": int(d["sourceIndex"]),
                "cameraId": d["cameraId"]}
    normalized_camera = {
        "version": "1.1", "width": int(calibration["width"]), "height": int(calibration["height"]),
        "left": eye("left"), "right": eye("right"),
        "extrinsics": {name: eye_extrinsic(calibration,name).tolist() for name in ("left","right")},
        "extrinsic_convention": "head_to_camera_4x4_row_major",
        "hdf5_rep103_head_to_openxr_head": REP103_HEAD_TO_OPENXR_HEAD.tolist(),
        "selected_effective_rep103_head_to_camera": geometry["effective_extrinsic"].tolist(),
        "selected_calibration_key": selected_eye,
        "video_info": {"width": width,"height": height,"fps": fps,"frame_count": n},
        "encoded_domain_intrinsics_authority": "SAME_SESSION_FACTORY_SOURCE_K_SCALED_UNVERIFIED_EXTERNALLY",
        "encoded_domain_distortion_policy": ("PASSTHROUGH_PIXELS_SOURCE_EQUIDIS62_NOT_APPLIED"
                                               if source_domain_passthrough
                                               else "RECTIFIED_PINHOLE_ZERO_DISTORTION"),
        "source_calibration": {"path": "source_archive/raw/camera_params.json","sha256":camera_sha,
                               "meta_path": "source_archive/raw/camera_params.meta.json",
                               "meta_sha256": sha256(source/"raw/camera_params.meta.json")},
    }
    json_dump(staging/"camera_params.json", normalized_camera)
    header = {
        "timeStampNs": int(arrays["timestamp_ns"][0]), "SN": bundle["camera_meta"].get("device_serial"),
        "RGBcamera_Intrinsics": ",".join(str(selected[x]) for x in ("fx","fy","cx","cy")),
        "RGBcamera_Extrinsics": ",".join(str(x) for x in selected["extrinsic"]),
        "RGBcamera_Left_Intrinsics": ",".join(str(calibration["left"][x]) for x in ("fx","fy","cx","cy")),
        "RGBcamera_Right_Intrinsics": ",".join(str(calibration["right"][x]) for x in ("fx","fy","cx","cy")),
        "RGBcamera_Params": calibration, "sourceCalibrationSha256": camera_sha,
    }
    with (staging/f"trackingData_{session}.txt").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header,separators=(",",":"))+"\n")
        for i in range(n):
            row = {"timeStampNs":int(arrays["timestamp_ns"][i]),"TrackerState":"egodex_aligned_complete",
                   "status":3,"Head":{"pose":matrix_pose_string(geometry["c2w"][i]),"status":3},
                   "Hand": {side:{"isActive":int(bool(arrays[f"{side}_hand_valid"][i])),
                                      "source":"MANUS25_WRIST_LOCAL_EXTENSION_NOT_PICO26"}
                            for side in ("left","right")},
                   "Controller": {side:{"pose":matrix_pose_string(geometry[side]["controller_world"][i]),
                                         "source":"egodex_v1_hdf5_controller_pose",
                                         "coordinateSystem":"first_frame_selected_camera_optical",
                                         "units":"metres"}
                                  for side in ("left","right")},
                   "source":{"hdf5_row":i,"source_row_idx":int(arrays["source_row_idx"][i]),
                             "video_frame_idx":int(arrays["video_frame_idx"][i]),
                             "segment_id":int(arrays["segment_id"][i])}}
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    with (staging/f"controller_poses_{session}.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(n):
            row = {
                "schema_version": "egodex-controller6d-sidecar-v1",
                "frame_index": i,
                "timeStampNs": int(arrays["timestamp_ns"][i]),
                "source": {"hdf5_row": i,
                           "source_row_idx": int(arrays["source_row_idx"][i]),
                           "segment_id": int(arrays["segment_id"][i])},
                "coordinate_contract": {
                    "source": "PICO world REP-103 X-forward/Y-left/Z-up metres",
                    "world": "first-frame selected camera optical X-right/Y-down/Z-forward metres",
                    "camera": "current-frame selected camera optical X-right/Y-down/Z-forward metres"
                },
                "hands": {side: {
                    "T_controller_to_source_world_rep103": geometry[side]["controller_source_rep103"][i].tolist(),
                    "T_controller_to_world": geometry[side]["controller_world"][i].tolist(),
                    "T_controller_to_camera": geometry[side]["controller_camera"][i].tolist(),
                    "T_wrist_to_world": geometry[side]["wrist_world"][i].tolist(),
                    "T_wrist_to_camera": geometry[side]["wrist_camera"][i].tolist(),
                    "wrist_derivation": "compose_pose(controller_pose, same-session controller_to_wrist_calibration)"
                } for side in ("left", "right")},
                "pico21_authority": False
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (staging/f"slam_trajectory_{session}.jsonl").open("w",encoding="utf-8") as stream:
        for i in range(n):
            source_pose = ",".join(f"{float(x):.12g}" for x in arrays["head_pose"][i])
            row={"timeStampNs":int(arrays["timestamp_ns"][i]),"mergedTimeS":i/fps,
                 "pose":matrix_pose_string(geometry["c2w"][i]),"TrackerState":"egodex_aligned_complete",
                 "status":3,"poseSegmentId":int(arrays["segment_id"][i]),
                 "sourceSession":session,"sessionIndex":int(session.rsplit("_",1)[-1]),
                 "runIndex":int(arrays["segment_id"][i]),
                 "sourcePose":source_pose,
                 "sourceMergedTimeS":float((int(arrays["recv_qpc_ns"][i])-int(arrays["recv_qpc_ns"][0]))/1e9),
                 "sourceRecordIndex":int(arrays["source_row_idx"][i]),
                 "sourceVideoFrameIndex":int(arrays["video_frame_idx"][i]),
                 "coordinateSystem":"first_frame_selected_camera_optical"}
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    idx=np.asarray(arrays["video_frame_idx"],dtype=np.int64); dif=np.diff(idx)
    timeline={"hdf5_rows":n,"source_video_frames":len(bundle["wall_ts"]),"first_source_index":int(idx[0]),
              "last_source_index":int(idx[-1]),"unique_source_indices":int(len(np.unique(idx))),
              "repeated_transitions":int((dif==0).sum()),"skipped_transitions":int((dif>1).sum()),
              "max_forward_step":int(dif.max()),"negative_transitions":int((dif<0).sum()),
              "segment_counts":{str(x):int((arrays["segment_id"]==x).sum()) for x in np.unique(arrays["segment_id"])},
              "mapping_authority":"dataset.hdf5 row -> video_frame_idx; repeated source frames are intentionally repeated"}
    humanego={"format_version":"1.0","frame_count":n,"fps":fps,"duration_s":n/fps,"eye":"left",
              "eye_semantics":"legacy_tracker_left_slot",
              "physical_calibration_eye":selected_eye,
              "source_eye_index":int(selected["sourceIndex"]),
              "source_calibration_key":selected_eye,
              "source_eye_resolution":[int(calibration["width"]),int(calibration["height"])],
              "output_resolution":[width,height],"horizontal_fov_deg":effective_fov_deg,
              "intrinsics":target_k.tolist(),
              "distortion":encoded_distortion,
              "camera_model":encoded_camera_model,
              "image_domain_mode":image_domain_mode,
              "world_coordinate_system":"first_frame_selected_camera_optical",
              "camera_coordinate_system":"current_frame_selected_camera_optical",
              "pose_convention":"4x4 column-vector SE(3)",
              "hand_model":{"name":"MANUS25 wrist-local source extension","source":"egodex_v1 HDF5",
                            "joint_names":geometry["left"]["joint_names"],
                            "pico21_status":"NOT_AVAILABLE_IN_SOURCE_SCHEMA_NO_FABRICATION",
                            "palm_center_status":"NOT_AVAILABLE_IN_SOURCE_SCHEMA"},
              "controller6d":{"sidecar":f"controller_poses_{session}.jsonl",
                              "source":"egodex_v1 HDF5 left/right_controller_pose",
                              "wrist_derivation":"same-session controller_to_wrist_calibration",
                              "pico21_authority":False},
              "active_frame_counts":{"left":int(arrays["left_hand_valid"].sum()),
                                     "right":int(arrays["right_hand_valid"].sum()),
                                     "both":int((arrays["left_hand_valid"]&arrays["right_hand_valid"]).sum())},
              "max_tracking_sync_error_ms":float(np.max(np.abs(bundle["arrays"].get("video_offset_ms", np.zeros(n))))),
              "finished_frames":0,
              "rectified_video_written":not source_domain_passthrough,
              "humanego_compatibility":{"all_data":"preprocess/all_data",
                                        "default_image_alias":"rgb_WoArm_WArmObjKpts.png",
                                        "objects":"not available","stock_rgb_hawor_input":True,
                                        "source_pico21_landmarks":False},
              "content_admission":visibility}
    humanego["intrinsics_authority"]="SAME_SESSION_FACTORY_SOURCE_K_SCALED_UNVERIFIED_EXTERNALLY"
    humanego["distortion_policy"]=("SOURCE_EQUIDIS62_COEFFICIENTS_PRESERVED_NO_WARP"
                                   if source_domain_passthrough
                                   else "RECTIFIED_PINHOLE_ZERO_DISTORTION")
    json_dump(staging/"preprocess/pico_humanego_manifest.json", humanego)
    clip={"format_version":"1.1","clip_name":session,"processing_status":"ready",
          "selection":{"eye":"left","eye_semantics":"legacy_tracker_left_slot",
                       "physical_calibration_eye":selected_eye,
                       "source_index":int(selected["sourceIndex"]),
                       "source_calibration_key":selected_eye,
                       "convention":"legacy_tracker_left" if selected_eye == "right" else "physical_left",
                       "authority":"same-session camera_params.json plus explicit tracker compatibility convention"},
          "files":{"video":f"CameraRecord_{session}.mp4",
                   "source_stereo_video":f"source_stereo/CameraRecord_{session}_stereo.mp4",
                   "tracking":f"trackingData_{session}.txt","slam":f"slam_trajectory_{session}.jsonl",
                   "controller_poses":f"controller_poses_{session}.jsonl",
                   "camera_params":"camera_params.json","humanego_all_data":"preprocess/all_data",
                   "humanego_manifest":"preprocess/pico_humanego_manifest.json"},
          "video":{"frame_count":n,"width":width,"height":height,"fps":fps},
          "source_stereo_video":{"frame_count":n,"width":int(calibration["outputWidth"]),
                                 "height":int(calibration["outputHeight"]),"fps":fps},
          "video_rectification":{"rectified":not source_domain_passthrough,
                                 "eye":"left","eye_semantics":"legacy_tracker_left_slot",
                                 "physical_calibration_eye":selected_eye,
                                 "source_index":int(selected["sourceIndex"]),
                                 "source_calibration_key":selected_eye,
                                 "camera_model":encoded_camera_model,"width":width,"height":height,
                                 "image_domain_mode":image_domain_mode,
                                 "horizontal_fov_deg":effective_fov_deg,
                                 "intrinsics":target_k.tolist(),
                                 "distortion":encoded_distortion},
          "tracking":{"tracking_frame_count":n,"slam_frame_count":n},
          "calibration_consistent":True,
          "calibration_consistency_scope":"same-session provenance and algebra only; encoded-domain metric applicability unverified externally",
          "intrinsics_applicability":"FACTORY_SOURCE_K_SCALED_NOT_GROUND_TRUTH",
          "source_calibration":{"path":"source_archive/raw/camera_params.json","sha256":camera_sha},
          "timeline_resampling":timeline,"format_compatibility":"FORMAT_COMPATIBLE",
          "content_admission":visibility["status"],
          "calibration_authority_policy":{
              "used":"source/raw/camera_params.json",
              "rejected":"dataset.hdf5 attr video_cam",
              "reason":"embedded video_cam declares stale 2160x810 native geometry incompatible with decoded 4096x1536 stream"
          },
          "claim_limit":"format compatibility does not override source visual-sensor semantic admission"}
    if bundle.get("bundle_manifest"):
        clip["files"]["clean_bundle_manifest"] = "clean_bundle/BUNDLE_MANIFEST.json"
        clip["files"]["canonical_tactile"] = "clean_bundle/clean_dataset.hdf5"
        clip["clean_bundle_contract"] = bundle["bundle_manifest"]
        clip["coordinate_contract"] = {
            "translation_unit":"meter","quaternion_order":"xyzw",
            "matrix_convention":"c2w","matrix_storage":"row_major",
            "camera_frame":"OpenCV X-right/Y-down/Z-forward",
            "controller_frame":"source REP-103 X-forward/Y-left/Z-up",
            "manus_frame":"wrist-local MANUS25",
        }
    else:
        clip["acquisition_aligned_hdf5_contract"] = {
            "path": str(source / "dataset.hdf5"),
            "sha256": sha256(source / "dataset.hdf5"),
            "schema": bundle["attrs"].get("schema"),
            "schema_revision": bundle["attrs"].get("schema_revision"),
            "all_exported_frames_complete": bool(
                bundle["attrs"].get("all_exported_frames_complete")
            ),
            "complete_frame_coverage": float(
                bundle["attrs"].get("complete_frame_coverage", 0.0)
            ),
            "timeline_authority": "ACQUISITION_ALIGNED_HDF5",
        }
    json_dump(staging/"clip_manifest.json",clip)


def probe(path: Path) -> dict[str, Any]:
    done=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                         "-show_entries","stream=width,height,avg_frame_rate,nb_read_frames",
                         "-of","json",str(path)],check=True,capture_output=True,text=True)
    return json.loads(done.stdout)["streams"][0]


def validate_output(target: Path, session: str, expected_n: int) -> dict[str, Any]:
    required=[target/"camera_params.json",target/"clip_manifest.json",
              target/f"CameraRecord_{session}.mp4",target/"source_stereo"/f"CameraRecord_{session}_stereo.mp4",
              target/f"trackingData_{session}.txt",target/f"slam_trajectory_{session}.jsonl",
              target/f"controller_poses_{session}.jsonl",target/"preprocess/pico_humanego_manifest.json"]
    for path in required:
        if not path.is_file() or path.stat().st_size<=0: raise ContractError(f"output missing/empty: {path}")
    mono,stereo=probe(required[2]),probe(required[3])
    if int(mono["nb_read_frames"])!=expected_n or int(stereo["nb_read_frames"])!=expected_n:
        raise ContractError("encoded frame count mismatch")
    all_data=target/"preprocess/all_data"
    dirs=sorted(x for x in all_data.iterdir() if x.is_dir())
    if [x.name for x in dirs] != [f"{i:05d}" for i in range(expected_n)]:
        raise ContractError("all_data frame directories are not exact contiguous range")
    for i in (0,expected_n-1):
        d=dirs[i]; record=json.loads((d/"training_data.json").read_text())
        if record.get("metadata",{}).get("idx")!=i: raise ContractError("training_data idx mismatch")
        for name in ("rgb.png","rgb_WoArm_WArmObjKpts.png"):
            image=cv2.imread(str(d/name))
            if image is None or image.shape[:2]!=(960,1280): raise ContractError(f"bad image {d/name}")
    tracking_lines=sum(1 for _ in required[4].open()); slam_lines=sum(1 for _ in required[5].open())
    controller_lines=sum(1 for _ in required[6].open())
    if tracking_lines!=expected_n+1 or slam_lines!=expected_n or controller_lines!=expected_n:
        raise ContractError("tracking/slam/controller line mismatch")
    return {"status":"PASS_FORMAT_COMPATIBLE","frame_count":expected_n,"mono_probe":mono,"stereo_probe":stereo,
            "all_data_count":len(dirs),"tracking_lines":tracking_lines,"slam_lines":slam_lines,
            "controller_lines":controller_lines}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--session-id",required=True)
    parser.add_argument("--horizontal-fov-deg", type=float, choices=(90.0,110.0,120.0), default=90.0)
    parser.add_argument("--selection-convention", choices=("legacy_tracker_left","physical_left"),
                        default="legacy_tracker_left")
    parser.add_argument("--image-domain-mode",
                        choices=(DIRECT_SOURCE_IMAGE_MODE, LEGACY_DIRECT_IMAGE_MODE,
                                 "pinhole_to_pinhole_fov","equidis62_rectify"),
                        default=DIRECT_SOURCE_IMAGE_MODE)
    parser.add_argument("--prebuilt-mono", type=Path,
                        help="Reuse a separately verified rectified mono video; it is byte-copied, not regenerated.")
    parser.add_argument("--clean-bundle", type=Path,
                        help="Validated rgb30_v1 clean_bundle; required for the formal 0909 publication path.")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    source=args.source.resolve(strict=True); target=args.target.absolute()
    if target.exists(): raise ContractError(f"target exists; refusing overwrite: {target}")
    if source==target or source in target.parents or target in source.parents:
        raise ContractError("source and target must be disjoint trees")
    clean_bundle = args.clean_bundle.resolve(strict=True) if args.clean_bundle else None
    before=source_snapshot(source); clean_before=clean_bundle_snapshot(clean_bundle)
    bundle=load_source(source, clean_bundle)
    selected_eye = "right" if args.selection_convention == "legacy_tracker_left" else "left"
    geometry=camera_and_hand_geometry(bundle, selected_eye)
    target_k=output_intrinsics(bundle["calibration"],selected_eye,1280,960,
                               args.horizontal_fov_deg,args.image_domain_mode)
    visibility=semantic_visibility(geometry,1280,960,args.horizontal_fov_deg,target_k)
    fov_candidates={str(int(fov)):semantic_visibility(geometry,1280,960,fov)
                    for fov in (90.0,110.0,120.0)}
    fov_mapping = {}
    for fov in (90.0, 110.0, 120.0):
        mx, my = build_rectification_map(bundle["calibration"], selected_eye, 2048, 1536,
                                         1280, 960, fov)
        valid = (mx >= 0) & (mx <= 2047) & (my >= 0) & (my <= 1535)
        fov_mapping[str(int(fov))] = {
            "calibrated_map_valid_fraction": float(valid.mean()),
            "invalid_output_pixels": int((~valid).sum()),
            "raw_fisheye_used_as_pinhole": False,
        }
    indices=np.asarray(bundle["arrays"]["video_frame_idx"]); dif=np.diff(indices)
    stale_video_cam = json.loads(bundle["attrs"].get("video_cam", "{}"))
    audit={"status":"AUDITED","source":str(source),"target":str(target),"session_id":args.session_id,
           "source_snapshot":before,"frame_count":len(indices),"source_video_frame_count":len(bundle["wall_ts"]),
           "selection_convention":args.selection_convention,
           "image_domain_mode":args.image_domain_mode,
           "target_intrinsics":target_k.tolist(),
           "selected_calibration_key":selected_eye,
           "selected_source_index":int(bundle["calibration"][selected_eye]["sourceIndex"]),
           "physical_left_source_index":int(bundle["calibration"]["left"]["sourceIndex"]),
           "prebuilt_mono":({"path":str(args.prebuilt_mono.resolve(strict=True)),
                              "sha256":sha256(args.prebuilt_mono.resolve(strict=True))}
                             if args.prebuilt_mono else None),
           "clean_bundle_snapshot":clean_before,
           "timeline":{"unique_indices":int(len(np.unique(indices))),"zero_diffs":int((dif==0).sum()),
                       "forward_gap_diffs":int((dif>1).sum()),"negative_diffs":int((dif<0).sum()),
                       "min_diff":int(dif.min()),"max_diff":int(dif.max())},
           "content_admission":visibility,
           "fov_candidate_geometry":fov_candidates,
           "fov_candidate_calibrated_mapping":fov_mapping,
           "calibration_authority":{
               "used":"raw/camera_params.json",
               "selected_key":selected_eye,
               "selected_source_index":int(bundle["calibration"][selected_eye]["sourceIndex"]),
               "used_sha256":before["raw/camera_params.json"]["sha256"],
               "hdf5_video_cam_rejected":True,
               "hdf5_embedded_native_geometry":stale_video_cam.get("intrinsics_native"),
               "reason":"embedded video_cam is stale (2160x810) versus decoded 2048x1536 per eye"
           },
           "landmark_policy":"preserve MANUS25 extension; do not fabricate unavailable PICO21/palm_center"}
    if args.audit_only:
        if args.report: json_dump(args.report,audit)
        else: print(json.dumps(audit,ensure_ascii=False,indent=2))
        return 0
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-",dir=target.parent))
    published=False
    try:
        encode_and_materialize(source,staging,args.session_id,bundle,geometry,1280,960,target,
                               args.horizontal_fov_deg,
                               args.image_domain_mode,target_k,
                               args.prebuilt_mono.resolve(strict=True) if args.prebuilt_mono else None)
        write_projection_probe(staging,geometry,1280,960,target_k)
        write_metadata(source,staging,args.session_id,bundle,geometry,visibility,1280,960,
                       args.horizontal_fov_deg,args.image_domain_mode,target_k)
        stereo=staging/"source_stereo"/f"CameraRecord_{args.session_id}_stereo.mp4"
        pts_target = stereo.with_suffix(stereo.suffix+".ptscache.npz")
        descriptor, local_name = tempfile.mkstemp(prefix="handle-pts-", suffix=".npz")
        os.close(descriptor)
        try:
            np.savez(local_name,
                     pts=np.arange(len(indices),dtype=np.float64)/float(bundle["attrs"]["fps"]),
                     source_size=np.asarray(stereo.stat().st_size,dtype=np.int64),
                     source_mtime_ns=np.asarray(stereo.stat().st_mtime_ns,dtype=np.int64))
            shutil.copyfile(local_name, pts_target)
        finally:
            Path(local_name).unlink(missing_ok=True)
        validation=validate_output(staging,args.session_id,len(indices))
        require_same_snapshot(before,source_snapshot(source))
        if not same_content_snapshot(clean_before, clean_bundle_snapshot(clean_bundle)):
            raise ContractError("clean bundle changed during conversion")
        result={**audit,"status":"PASS_FORMAT_COMPATIBLE","validation":validation,
                "content_status":visibility["status"]}
        json_dump(staging/"CONVERSION_RESULT.json",result)
        os.replace(staging,target);published=True
        if args.report: json_dump(args.report,result)
        print(json.dumps({"status":result["status"],"content_status":result["content_status"],
                          "target":str(target),"frames":len(indices)},ensure_ascii=False))
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
