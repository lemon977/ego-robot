"""Pure CPU Controller + MANUS composition; no visual fitting or pose imputation.

T_A_B maps B coordinates into A. World is the acquisition PICO REP103 world.
The anatomical wrist transform is a declared installation PRIOR, not measured
wrist ground truth. Projected pixels use unverified factory encoded-domain K.
"""
from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
SESSION_IDS = ("play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101")
FRAME_COUNTS = (165, 179, 122)
REP103_HEAD_TO_OPENXR = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
MANUS_NAMES = (
    "Hand_Invalid", "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_IP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_IP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_IP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_IP", "Pinky_DIP", "Pinky_TIP",
)
MAP_25_TO_21 = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24])
JOINT_NAMES_21 = ("virtual_wrist_root",) + tuple(MANUS_NAMES[i] for i in MAP_25_TO_21[1:])
EDGES_21 = tuple((0 if j == 0 else 1 + 4 * f + j - 1, 1 + 4 * f + j)
                 for f in range(5) for j in range(4))


class MotionContractError(ValueError):
    pass


def require_transform(value, name="transform"):
    a = np.asarray(value, dtype=np.float64)
    if a.shape[-2:] != (4, 4) or not np.isfinite(a).all():
        raise MotionContractError(f"{name}: finite (...,4,4) required")
    r = a[..., :3, :3]
    if (not np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-5, rtol=0)):
        raise MotionContractError(f"{name}: improper or nonrigid transform")
    return a


def pose_matrices(value):
    """Return finite transforms and per-pose computability; invalid stays NaN."""
    a = np.asarray(value, dtype=np.float64)
    if a.shape[-1:] != (7,):
        raise MotionContractError("pose layout must be xyz + quaternion xyzw")
    flat = a.reshape(-1, 7)
    norms = np.linalg.norm(flat[:, 3:], axis=-1)
    valid = np.isfinite(flat).all(axis=-1) & (np.abs(norms - 1.) <= 1e-3)
    out = np.full((len(flat), 4, 4), np.nan)
    if valid.any():
        q = flat[valid, 3:] / norms[valid, None]
        x, y, z, w = q.T
        out[valid] = np.eye(4)
        out[valid, :3, :3] = np.stack([
            1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
        ], axis=-1).reshape(-1, 3, 3)
        out[valid, :3, 3] = flat[valid, :3]
    return out.reshape(a.shape[:-1] + (4, 4)), valid.reshape(a.shape[:-1])


def validate_timeline(timestamp_ns, frame_id, segment_id):
    ts, fid, seg = (np.asarray(v) for v in (timestamp_ns, frame_id, segment_id))
    if ts.ndim != 1 or not len(ts) or fid.shape != ts.shape or seg.shape != ts.shape:
        raise MotionContractError("time/frame/segment must be nonempty matching vectors")
    if any(not np.issubdtype(v.dtype, np.integer) for v in (ts, fid, seg)):
        raise MotionContractError("integer nanosecond timestamps/frame/segment required")
    if np.any(np.diff(ts) <= 0) or np.any(np.diff(fid) <= 0):
        raise MotionContractError("duplicate/backward timestamps or frame ids")
    if len(ts) > 1 and np.median(np.diff(ts)) < 1_000_000:
        raise MotionContractError("timestamp unit is not plausible nanoseconds")
    return ts.astype(np.int64), fid.astype(np.int64), seg.astype(np.int64)


def camera_contract(calibration, size=(1280, 960)):
    if calibration.get("extrinsic_convention") != "head_to_camera_4x4_row_major":
        raise MotionContractError("head-to-camera convention missing")
    if calibration["left"].get("sourceIndex") != 1 or calibration["right"].get("sourceIndex") != 0:
        raise MotionContractError("physical camera route must be left=1/right=0")
    axis = np.asarray(calibration.get("hdf5_rep103_head_to_openxr_head"), dtype=float)
    if axis.shape != (3, 3) or not np.allclose(axis, REP103_HEAD_TO_OPENXR, atol=1e-12, rtol=0):
        raise MotionContractError("REP103 to OpenXR head contract mismatch")
    a = np.eye(4)
    a[:3, :3] = axis
    left = require_transform(calibration["extrinsics"]["left"], "left extrinsic") @ a
    right = require_transform(calibration["extrinsics"]["right"], "right extrinsic") @ a
    w, h = size
    sx, sy = w / float(calibration["width"]), h / float(calibration["height"])
    if not np.isclose(sx, sy, atol=1e-12):
        raise MotionContractError("nonuniform source pixel resize")
    intr = calibration["left"]["intrinsics"]
    k = np.array([[intr["fx"] * sx, 0., intr["cx"] * sx],
                  [0., intr["fy"] * sy, intr["cy"] * sy], [0., 0., 1.]])
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise MotionContractError("invalid physical left K")
    return left, right, k


def installation_matrices(prior):
    if (prior.get("schema") != "controller_to_wrist_v1"
            or prior.get("convention") != "wrist_pose = compose_pose(controller_pose, transform)"
            or prior.get("controller_pose_frame") != "pico_world_rh_x_forward_y_left_z_up"):
        raise MotionContractError("installation prior convention mismatch")
    poses = np.array([list(prior[s]["pos"]) + list(prior[s]["quat"]) for s in SIDES])
    matrices, valid = pose_matrices(poses)
    if not valid.all():
        raise MotionContractError("invalid installation prior quaternion")
    return require_transform(matrices, "installation prior")


def compose_motion(*, head_pose, controller_pose, manus_local, hand_valid,
                   timestamp_ns, frame_id, segment_id, calibration, prior,
                   joint_names=MANUS_NAMES, side_names=SIDES, units="m"):
    if tuple(side_names) != SIDES or tuple(joint_names) != MANUS_NAMES or units != "m":
        raise MotionContractError("side, joint-order or metre unit mismatch")
    ts, fid, seg = validate_timeline(timestamp_ns, frame_id, segment_id)
    n = len(ts)
    h, hv = pose_matrices(head_pose)
    c, cv = pose_matrices(controller_pose)
    local = np.asarray(manus_local, dtype=float)
    valid = np.asarray(hand_valid)
    if h.shape != (n, 4, 4) or c.shape != (n, 2, 4, 4) or local.shape != (n, 2, 25, 3):
        raise MotionContractError("head/controller/MANUS shape mismatch")
    if valid.shape != (n, 2) or valid.dtype != bool:
        raise MotionContractError("independent boolean hand mask required")
    if np.any(np.linalg.norm(local[:, :, 0][valid], axis=-1) > 1e-6):
        raise MotionContractError("MANUS local origin is not zero; double-root risk")
    if np.any(np.linalg.norm(local[valid], axis=-1) > 1.):
        raise MotionContractError("MANUS metre-scale sanity failed (possible mm)")
    e_left, e_right, k = camera_contract(calibration)
    install = installation_matrices(prior)
    iw = np.full_like(h, np.nan)
    iw[hv] = np.linalg.inv(h[hv])
    camera_from_world = e_left @ iw
    right_from_world = e_right @ iw
    world_from_camera = np.full_like(h, np.nan)
    world_from_camera[hv] = np.linalg.inv(camera_from_world[hv])
    wrist_world = c @ install[None]
    wrist_camera = camera_from_world[:, None] @ wrist_world
    controller_camera = camera_from_world[:, None] @ c
    legacy_camera = right_from_world[:, None] @ wrist_world
    wrist_world_valid = cv.copy()
    wrist_camera_valid = cv & hv[:, None]
    joint_local_valid = valid[:, :, None] & np.isfinite(local).all(axis=-1)
    joint_world_valid = joint_local_valid & cv[:, :, None]
    joint_camera_valid = joint_world_valid & hv[:, None, None]
    world25 = np.einsum("tsij,tskj->tski", wrist_world[:, :, :3, :3], local) + wrist_world[:, :, None, :3, 3]
    camera25 = np.einsum("tsij,tskj->tski", wrist_camera[:, :, :3, :3], local) + wrist_camera[:, :, None, :3, 3]
    world25[~joint_world_valid] = np.nan
    camera25[~joint_camera_valid] = np.nan
    wrist_world[~wrist_world_valid] = np.nan
    wrist_camera[~wrist_camera_valid] = np.nan
    mapped_world_valid = joint_world_valid[:, :, MAP_25_TO_21]
    mapped_camera_valid = joint_camera_valid[:, :, MAP_25_TO_21]
    direct = joint_local_valid[:, :, MAP_25_TO_21].copy()
    direct[:, :, 0] = False  # exported Hand_Invalid is virtual root, not anatomical observation
    transition_valid = np.zeros(n, dtype=bool)
    if n > 1:
        # Explicit derivative mask, no interpolation across missing frames or segment changes.
        transition_valid[1:] = ((np.diff(fid) == 1) & (np.diff(seg) == 0)
                                & (np.diff(ts) <= 2 * np.median(np.diff(ts))))
    return {
        "frame_id": fid, "timestamp_ns": ts, "timestamp_s": (ts - ts[0]) / 1e9,
        "segment_id": seg, "time_transition_valid": transition_valid,
        "side_names": np.array(SIDES), "joint_names": np.array(JOINT_NAMES_21),
        "manus25_joint_names": np.array(MANUS_NAMES), "manus25_to_21": MAP_25_TO_21,
        "manus_local_25_m": local, "manus_local_21_m": local[:, :, MAP_25_TO_21],
        "manus_local_joint_valid": joint_local_valid[:, :, MAP_25_TO_21],
        "joints_world_m": world25[:, :, MAP_25_TO_21],
        "joints_camera_m": camera25[:, :, MAP_25_TO_21],
        "joint_world_valid": mapped_world_valid, "joint_camera_valid": mapped_camera_valid,
        "joint_observed_local": direct, "joint_inferred": np.zeros_like(direct),
        "surface_valid": np.zeros((n, 2), dtype=bool),
        "T_world_wrist": wrist_world, "T_camera_wrist": wrist_camera,
        "T_world_controller": c, "T_camera_controller": controller_camera,
        "T_world_head": h, "head_valid": hv,
        "T_world_camera": world_from_camera, "T_camera_world": camera_from_world,
        "T_controller_wrist_prior": install,
        "wrist_world_valid": wrist_world_valid, "wrist_camera_valid": wrist_camera_valid,
        "wrist_position_observed": np.zeros((n, 2), dtype=bool),
        "wrist_rotation_observed": np.zeros((n, 2), dtype=bool),
        "controller_pose_observed": cv, "manus_hand_valid": valid,
        "legacy_right_T_camera_wrist": legacy_camera,
        "T_leftcamera_rightcamera": e_left @ np.linalg.inv(e_right),
        "camera_K": k, "camera_source_index": np.array(1),
        "camera_image_size": np.array([1280, 960]),
        "source_kind": np.array("PICO_CONTROLLER_PLUS_MANUS_NO_HAWOR"),
        "wrist_authority": np.array("VIRTUAL_INSTALLATION_PRIOR_NOT_MEASURED_ANATOMICAL_WRIST"),
        "camera_K_authority": np.array("FACTORY_K_SCALED_UNVERIFIED_FOR_ENCODED_DOMAIN"),
        "training_eligible": np.array(False), "control_ground_truth": np.array(False),
    }


def project_points(points, k, valid):
    p = np.asarray(points, dtype=float)
    good = np.asarray(valid, dtype=bool) & np.isfinite(p).all(axis=-1) & (p[..., 2] > 1e-6)
    uv = np.full(p.shape[:-1] + (2,), np.nan)
    uv[good] = p[good, :2] / p[good, 2, None] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
    return uv, good
