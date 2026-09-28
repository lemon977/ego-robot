"""Normalize frozen 0902 HaWoR bounded motion and Robot R0 without solving.

The archived bounded source supplies orientation and joint rotation; the V3
exact-domain adapter supplies the verified timeline, intrinsics, validity, and
camera-space joints. No ROI or direct 3D observation is synthesized.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.pipeline.v5_motion import _canonical, _counts, _load_npz, _pinned, _ref, _segments


SCHEMA = "HUMAN_TO_ROBOT_BASELINE_V1_EXACT_MOTION"
IMAGE_DOMAIN = "EXACT0902_SOURCEINDEX0_EQUIDIS62_TO_PINHOLE90_V3"
SIDES = ("left", "right")
REQUIRED_REFS = (
    "bounded_source", "exact_source", "exact_result", "robot_r0",
    "robot_result", "domain_manifest", "mount", "asset_pin",
)


def _same_ref(receipt: dict[str, Any], key: str, expected: dict[str, Any]) -> None:
    row = receipt[key]
    if row["sha256"] != expected["sha256"]:
        raise ValueError(f"RECEIPT_{key.upper()}_SHA_MISMATCH")


def _shape(data: dict[str, np.ndarray], key: str, expected: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(data[key])
    if array.shape != expected:
        raise ValueError(f"{key}_SHAPE:{array.shape}!={expected}")
    return array


def recover(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("EXACT_CONFIG_SCHEMA")
    paths = {key: _pinned(config[key]) for key in REQUIRED_REFS}
    refs = {key: _ref(path) for key, path in paths.items()}
    config_ref = _ref(config_path)
    code_refs = {
        "exact_adapter": _ref(Path(__file__)),
        "shared_motion": _ref(Path(__file__).with_name("v5_motion.py")),
    }
    signature = hashlib.sha256(_canonical({
        "schema": SCHEMA, "config": config_ref, "code": code_refs, "inputs": refs,
    })).hexdigest()
    output = Path(config["output"])
    receipt_path = output / "RESULT.json"
    if output.exists():
        if not receipt_path.is_file():
            raise FileExistsError(f"PARTIAL_OUTPUT:{output}")
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != signature:
            raise FileExistsError(f"OUTPUT_SIGNATURE_CHANGED:{output}")
        for row in old["outputs"].values():
            _pinned(row)
        return {**old, "cache": "REUSED"}

    archived = _load_npz(paths["bounded_source"])
    exact = _load_npz(paths["exact_source"])
    robot = _load_npz(paths["robot_r0"])
    exact_result = json.loads(paths["exact_result"].read_text(encoding="utf-8"))
    robot_result = json.loads(paths["robot_result"].read_text(encoding="utf-8"))
    domain = json.loads(paths["domain_manifest"].read_text(encoding="utf-8"))
    mount = json.loads(paths["mount"].read_text(encoding="utf-8"))
    asset_pin = json.loads(paths["asset_pin"].read_text(encoding="utf-8"))
    session = config["session_id"]
    frames = int(config["frame_count"])
    if domain.get("session_id") != session or domain.get("image_domain") != IMAGE_DOMAIN:
        raise ValueError("EXACT_SESSION_OR_DOMAIN")
    if domain.get("source_index") != 0 or len(domain["frames"]) != frames:
        raise ValueError("EXACT_SOURCE_CAMERA_OR_FRAME_COUNT")
    if exact_result.get("session_id") != session or robot_result.get("session_id") != session:
        raise ValueError("RECEIPT_SESSION")
    _same_ref(exact_result, "source", refs["bounded_source"])
    _same_ref(exact_result, "output", refs["exact_source"])
    _same_ref(exact_result, "domain_manifest", refs["domain_manifest"])
    _same_ref(robot_result, "source", refs["exact_source"])
    _same_ref(robot_result, "output", refs["robot_r0"])
    _same_ref(robot_result, "mount_contract", refs["mount"])
    if asset_pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise ValueError("ASSET_PIN_STATUS")
    if len(asset_pin.get("urdfs", [])) != 3:
        raise ValueError("ASSET_PIN_URDF_CLOSURE")

    ids = _shape(exact, "original_frame_indices", (frames,)).astype(np.int64)
    ts = _shape(exact, "timestamp_ns", (frames,)).astype(np.int64)
    expected_ids = np.asarray([row["frame_id"] for row in domain["frames"]], dtype=np.int64)
    expected_ts = np.asarray([row["capture_time"]["original_time_fields"]["ts"] for row in domain["frames"]], dtype=np.int64)
    if not np.array_equal(ids, expected_ids) or not np.array_equal(ts, expected_ts):
        raise ValueError("EXACT_DOMAIN_FRAME_TIME")
    if np.any(np.diff(ids) != 1) or np.any(np.diff(ts) <= 0):
        raise ValueError("FRAME_TIME_ORDER")
    if not np.array_equal(archived["original_frame_indices"], ids) or not np.array_equal(robot["frame_id"], ids):
        raise ValueError("SOURCE_ROBOT_FRAME_MISMATCH")
    if not np.array_equal(robot["timestamp_ns"], ts):
        raise ValueError("ROBOT_TIMESTAMP_MISMATCH")
    if archived["anatomical_side_names"].tolist() != list(SIDES) or exact["anatomical_side_names"].tolist() != list(SIDES):
        raise ValueError("ANATOMICAL_SIDE_ORDER")
    if robot["human_to_physical"].tolist() != [0, 1]:
        raise ValueError("UNSUPPORTED_HUMAN_TO_PHYSICAL_MAP")
    if str(exact["image_domain"]) != IMAGE_DOMAIN:
        raise ValueError("EXACT_SOURCE_IMAGE_DOMAIN")
    camera_k = _shape(exact, "intrinsics", (frames, 3, 3))
    if not np.allclose(_shape(archived, "intrinsics", (frames, 3, 3)), camera_k, rtol=0, atol=1e-9):
        raise ValueError("ARCHIVED_CAMERA_K_MISMATCH")
    if not np.allclose(camera_k, np.asarray(domain["K"])[None], rtol=0, atol=1e-9):
        raise ValueError("DOMAIN_CAMERA_K_MISMATCH")
    camera_joints = _shape(exact, "joints_3d_camera", (2, frames, 21, 3))
    if not np.array_equal(_shape(archived, "joints_3d_camera", (2, frames, 21, 3)), camera_joints, equal_nan=True):
        raise ValueError("ARCHIVED_EXACT_JOINTS_DIFFER")
    valid = _shape(exact, "predicted_valid", (2, frames)).T.astype(bool)
    if not np.array_equal(_shape(archived, "observed", (2, frames)).T, valid):
        raise ValueError("ARCHIVED_EXACT_VALIDITY_DIFFER")
    roots = _shape(archived, "root_orient_camera", (2, frames, 3, 3))
    hand_rotmat = _shape(archived, "hand_pose_rotmat", (2, frames, 15, 3, 3))
    target_valid = _shape(robot, "target_valid", (frames, 2)).astype(bool)
    wrist_valid = _shape(robot, "wrist_valid", (frames, 2)).astype(bool)
    finger_valid = _shape(robot, "finger_valid", (frames, 2)).astype(bool)
    if not np.array_equal(target_valid, valid) or np.any(wrist_valid & ~target_valid) or np.any(finger_valid & ~target_valid):
        raise ValueError("ROBOT_VALIDITY_MISMATCH")
    if not np.allclose(_shape(robot, "T_flange_hand", (2, 4, 4)), np.asarray(mount["T_flange_hand"]), rtol=0, atol=1e-12):
        raise ValueError("ROBOT_MOUNT_MISMATCH")
    _shape(robot, "q_arm", (frames, 2, 7))
    _shape(robot, "q22", (frames, 2, 22))
    wrist_T = np.full((frames, 2, 4, 4), np.nan, dtype=np.float64)
    joints21_camera = camera_joints.transpose(1, 0, 2, 3).astype(np.float64)
    joints21_root = np.full_like(joints21_camera, np.nan)
    for side in range(2):
        accepted = valid[:, side]
        wrist_T[accepted, side] = np.eye(4)
        wrist_T[accepted, side, :3, :3] = roots[side, accepted]
        wrist_T[accepted, side, :3, 3] = camera_joints[side, accepted, 0]
        delta = joints21_camera[accepted, side] - wrist_T[accepted, side, None, :3, 3]
        joints21_root[accepted, side] = np.einsum("tji,tkj->tki", wrist_T[accepted, side, :3, :3], delta)
    if not np.isfinite(wrist_T[valid]).all() or not np.isfinite(joints21_root[valid]).all():
        raise ValueError("VALID_MOTION_NONFINITE")
    orientation = wrist_T[valid, :3, :3]
    if np.max(np.abs(orientation @ orientation.transpose(0, 2, 1) - np.eye(3))) > 1e-3:
        raise ValueError("ROOT_ROTATION_NONORTHOGONAL")

    output.mkdir(parents=True, exist_ok=False)
    hand_path = output / "HAND_MOTION_V1.npz"
    robot_path = output / "ROBOT_R0_V1.npz"
    np.savez_compressed(
        hand_path, frame_id=ids, timestamp_ns=ts,
        anatomical_side_names=np.asarray(SIDES), T_camera_wrist=wrist_T,
        joints21_camera=joints21_camera, joints21_root=joints21_root,
        position_valid=valid, rotation_valid=valid,
        joint_valid=np.broadcast_to(valid[:, :, None], (frames, 2, 21)).copy(),
        estimate_source=np.where(valid, "HAWOR_BOUNDED_INFERRED", "UNKNOWN"),
        T_world_camera=np.full((frames, 4, 4), np.nan, dtype=np.float64),
        world_valid=np.zeros(frames, dtype=bool), segment_id=_segments(valid),
        source_detection_observed=_shape(exact, "source_detection_observed", (2, frames)).T,
        geometry_observed=np.zeros((frames, 2), dtype=bool),
        hand_pose_rotmat=hand_rotmat.transpose(1, 0, 2, 3, 4),
        intrinsics=camera_k, image_domain=np.asarray(IMAGE_DOMAIN),
        units_position=np.asarray("m"), units_angle=np.asarray("rad"),
        control_ground_truth=np.asarray(False), training_eligible=np.asarray(False),
    )
    np.savez_compressed(
        robot_path, frame_id=ids, timestamp_ns=ts,
        anatomical_side_names=np.asarray(SIDES), target_valid=target_valid,
        wrist_valid=wrist_valid, finger_valid=finger_valid,
        q_arm=robot["q_arm"], q_hand22=robot["q22"],
        T_target_root_cam=robot["T_target_root_cam"],
        T_actual_root_cam=robot["T_actual_root_cam"],
        actual21_camera_m=robot["actual21_camera"],
        T_cam_base=robot["T_cam_base"], T_flange_hand=robot["T_flange_hand"],
        human_to_physical=robot["human_to_physical"],
        position_residual_mm=robot["position_residual_mm"],
        rotation_residual_deg=robot["rotation_residual_deg"],
        placement_policy=robot["placement_policy"],
        asset_pin_sha256=np.asarray(refs["asset_pin"]["sha256"]),
        legacy_candidate=np.asarray(True), control_ground_truth=np.asarray(False),
        training_eligible=np.asarray(False),
    )
    tolerance = _counts(_shape(robot, "tolerance_pass", (frames, 2)).astype(bool))
    receipt = {
        "schema_version": SCHEMA, "session_id": session, "frame_count": frames,
        "status": "LEGACY_EXACT_MOTION_RECOVERED_FOR_DIAGNOSTIC",
        "run_signature": signature, "cache": "FRESH",
        "source_valid_left_right": _counts(valid), "robot_target_valid_left_right": _counts(target_valid),
        "arm_tolerance_pass_left_right": tolerance,
        "quality_status": "C_ARM_TOLERANCE_0_OF_VALID" if max(tolerance) == 0 else "FAILED_NUMERIC_GATE",
        "numeric_quality_pass": False, "offline_visual": True,
        "roi_evidence": "UNAVAILABLE_IN_ARCHIVED_BOUNDED_SOURCE_NOT_FABRICATED",
        "world_geometry": "UNVERIFIED_C2W_NOT_EXPORTED",
        "control_ground_truth": False, "training_eligible": False,
        "inputs": refs, "config": config_ref, "code": code_refs,
        "outputs": {"hand_motion": _ref(hand_path), "robot_r0": _ref(robot_path)},
    }
    receipt_path.write_bytes(_canonical(receipt))
    return receipt
