"""Read-only adapter from the frozen PICO/MANUS V3 evidence to HandMotion V1.

The V3 C1 projection is the nominal M0 visual reference. C3 is the fixed
installation estimate fitted from session 097. Neither is measured anatomical
wrist calibration, and MANUS local positions are not robot joint angles.
"""

from __future__ import annotations

import numpy as np


SESSION_IDS = (
    "play_cards_0916_097",
    "play_cards_0916_098",
    "play_cards_0916_101",
)
SIDES = ("left", "right")
MAX_ABSOLUTE_INSTALLATION_M = 0.16
MANUS25_TO_21 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24],
    dtype=np.int64,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if np.issubdtype(np.asarray(a).dtype, np.inexact):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _rigid(name: str, transforms: np.ndarray) -> None:
    matrices = np.asarray(transforms, dtype=np.float64)
    _require(matrices.shape[-2:] == (4, 4), f"{name}: invalid transform shape")
    _require(np.isfinite(matrices).all(), f"{name}: nonfinite transform")
    rotation = matrices[..., :3, :3]
    identity = np.eye(3)
    _require(np.allclose(rotation.swapaxes(-1, -2) @ rotation, identity, atol=1e-5), f"{name}: rotation is not orthogonal")
    _require(np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5), f"{name}: improper rotation")
    _require(np.allclose(matrices[..., 3, :], [0, 0, 0, 1], atol=1e-7), f"{name}: invalid homogeneous row")


def validate_pair(nominal: dict[str, np.ndarray], fitted: dict[str, np.ndarray], session_id: str) -> dict:
    """Validate that C1 and C3 are the same 0916 capture and C3 is bounded."""
    _require(session_id in SESSION_IDS, f"unsupported session: {session_id}")
    common = (
        "frame_id", "timestamp_ns", "segment_id", "side_names", "manus25_joint_names",
        "manus25_to_21", "manus_local_25_m", "manus_local_21_m", "T_world_head",
        "T_world_controller", "T_controller_wrist_prior", "manus_hand_valid",
        "source_video_frame_id", "camera_source_index", "camera_image_size",
    )
    for key in common:
        _require(key in nominal and key in fitted, f"{session_id}: missing {key}")
        _require(_same(nominal[key], fitted[key]), f"{session_id}: C1/C3 source drift in {key}")
    n = len(fitted["frame_id"])
    _require(n > 0 and fitted["frame_id"].shape == (n,), f"{session_id}: empty frame axis")
    _require(str(fitted["session_id"]) == session_id and str(nominal["session_id"]) == session_id, f"{session_id}: identity drift")
    _require(tuple(fitted["side_names"].tolist()) == SIDES, f"{session_id}: side order drift")
    _require(_same(fitted["manus25_to_21"], MANUS25_TO_21), f"{session_id}: MANUS mapping drift")
    _require(len(set(fitted["manus25_joint_names"].tolist())) == 25, f"{session_id}: MANUS25 names not unique")
    _require(fitted["manus_local_25_m"].shape == (n, 2, 25, 3), f"{session_id}: MANUS25 shape drift")
    _require(fitted["manus_local_21_m"].shape == (n, 2, 21, 3), f"{session_id}: 21-point shape drift")
    _require(np.array_equal(fitted["frame_id"], np.arange(n)), f"{session_id}: noncontiguous frames")
    _require(np.all(np.diff(fitted["timestamp_ns"]) > 0), f"{session_id}: timestamps not increasing")
    _require(int(fitted["camera_source_index"]) == 1, f"{session_id}: selected camera changed")
    _require(str(fitted["source_kind"]) == "PICO_CONTROLLER_PLUS_MANUS_NO_HAWOR", f"{session_id}: source kind drift")
    _require("T_controller_wrist_estimate" in fitted, f"{session_id}: missing C3 installation")
    prior = fitted["T_controller_wrist_prior"]
    estimate = fitted["T_controller_wrist_estimate"]
    _rigid("prior installation", prior)
    _rigid("C3 installation", estimate)
    _require(np.max(np.abs(estimate[:, :3, 3])) <= MAX_ABSOLUTE_INSTALLATION_M + 1e-8, f"{session_id}: C3 exceeds absolute installation boundary")
    _require(np.allclose(prior[:, :3, :3], estimate[:, :3, :3], atol=1e-8), f"{session_id}: C3 changed installation rotation")
    _require(np.allclose(fitted["T_world_controller"] @ estimate[None], fitted["T_world_wrist"], atol=1e-7), f"{session_id}: C3 applied installation twice or inconsistently")
    _require(np.allclose(nominal["T_world_controller"] @ prior[None], nominal["T_world_wrist"], atol=1e-7), f"{session_id}: nominal M0 chain inconsistent")
    _require(np.allclose(fitted["T_camera_world"][:, None] @ fitted["T_world_wrist"], fitted["T_camera_wrist"], atol=1e-7), f"{session_id}: camera/world chain inconsistent")
    _require(fitted["wrist_camera_valid"].shape == (n, 2), f"{session_id}: wrist validity drift")
    _require(fitted["joint_camera_valid_25"].shape == (n, 2, 25), f"{session_id}: node validity drift")
    _require(np.all(fitted["manus_hand_valid"]), f"{session_id}: unexpected MANUS invalidity")
    return {
        "session_id": session_id,
        "frames": n,
        "selected_camera_source_index": 1,
        "side_order": list(SIDES),
        "C3_installation_translation_controller_m": estimate[:, :3, 3].tolist(),
        "C3_installation_absolute_bound_m_per_axis": MAX_ABSOLUTE_INSTALLATION_M,
        "C3_installation_prior_sigma_m": 0.05,
        "C3_rotation_fitted": False,
        "fitted_on": "play_cards_0916_097_frozen_visual_proxy_only",
        "physical_calibration_verified": False,
    }


def hand_motion_arrays(nominal: dict[str, np.ndarray], fitted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Preserve source MANUS25 and expose C3 wrist plus local finger positions."""
    keys = (
        "frame_id", "timestamp_ns", "timestamp_s", "segment_id", "time_transition_valid",
        "side_names", "joint_names", "manus25_joint_names", "manus25_to_21",
        "manus_parent_ids", "manus_local_25_m", "manus_local_21_m",
        "manus_local_joint_valid", "joint_observed_local", "joint_inferred",
        "T_world_head", "head_valid", "T_world_controller", "controller_pose_observed",
        "T_camera_world", "T_world_camera", "T_world_wrist", "T_camera_wrist",
        "T_camera_controller", "T_controller_wrist_prior", "T_controller_wrist_estimate",
        "wrist_world_valid", "wrist_camera_valid", "wrist_position_observed",
        "wrist_rotation_observed", "manus_hand_valid", "joint_camera_valid_25",
        "joint_observed_local_25", "camera_K", "camera_source_index", "camera_image_size",
        "source_video_frame_id", "source_row_id", "source_video_offset_ms",
        "display_time_s", "display_fps", "session_id", "source_kind",
        "wrist_authority", "camera_K_authority", "provenance_v2_source_clock",
        "provenance_v2_status", "training_eligible", "control_ground_truth",
        "physical_calibration_verified",
    )
    out = {key: np.array(fitted[key], copy=True) for key in keys}
    n = len(fitted["frame_id"])
    out["schema_version"] = np.array("chaoyang-HandMotion-v1")
    out["motion_source"] = np.array("controller_manus")
    out["joint_representation"] = np.array("MANUS_LOCAL_NODE_POSITIONS_METRES_NOT_ROBOT_JOINT_ANGLES")
    out["anatomical_side_names"] = np.array(SIDES)
    out["joints21_camera"] = np.array(fitted["joints_camera_m"], copy=True)
    out["joints21_root"] = np.array(fitted["manus_local_21_m"], copy=True)
    out["position_valid"] = np.array(fitted["wrist_camera_valid"], copy=True)
    out["rotation_valid"] = np.array(fitted["wrist_camera_valid"], copy=True)
    out["joint_valid"] = np.array(fitted["joint_camera_valid"], copy=True)
    out["estimate_source"] = np.full((n, 2), "CONTROLLER_MANUS_C3_VISUAL_ESTIMATE", dtype="U40")
    # The V3 encoded camera/head relation is a visual hypothesis, not an
    # authoritative camera-to-world calibration for Robot placement.
    out["development_T_world_camera"] = out.pop("T_world_camera")
    out["development_T_camera_world"] = out.pop("T_camera_world")
    out["T_world_camera"] = np.full((n, 4, 4), np.nan, dtype=np.float64)
    out["world_valid"] = np.zeros(n, dtype=bool)
    out["M0_T_world_wrist"] = np.array(nominal["T_world_wrist"], copy=True)
    out["M0_T_camera_wrist"] = np.array(nominal["T_camera_wrist"], copy=True)
    out["M0_source_candidate"] = np.array(str(nominal["alignment_candidate"]))
    out["C3_source_candidate"] = np.array(str(fitted["alignment_candidate"]))
    out["source_authority"] = np.array("OFFLINE_VISUAL")
    out["M1_historical_status"] = np.array("NOT_BOUND_TO_THIS_REPLAY_NO_ADDITIVE_OFFSET")
    return out


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points)
    good = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 1e-6)
    uv = np.full(xyz.shape[:-1] + (2,), np.nan, dtype=np.float64)
    uv[good] = xyz[good, :2] / xyz[good, 2, None] * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2]
    return uv, good
