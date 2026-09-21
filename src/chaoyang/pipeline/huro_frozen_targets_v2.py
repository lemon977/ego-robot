"""Frozen RAW MANO21 target adapter shared by reference and HuRo evaluation.

Fixed per-session/hand shape normalization is OFFLINE_NONCAUSAL, not calibration.
No joint-angle solution or historical q22 FK participates in the observation.
"""
from __future__ import annotations
import numpy as np
from chaoyang.pipeline.huro_hand_frame_v2 import palm_basis
from chaoyang.pipeline.huro_hand_only_retarget_v1 import HuroHandOnlyError

MANO_JOINT_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def build_targets(r0, raw, neutral_hand_points, neutral_roots):
    ids = np.asarray(r0["frame_id"])
    clock = np.asarray(r0["timestamp_ns"])
    if not np.array_equal(ids, raw["original_frame_indices"]) or ids.ndim != 1 or len(ids) == 0:
        raise ValueError("R0/RAW frame identity mismatch")
    if clock.shape != ids.shape or clock.dtype.kind not in "iu" or np.any(np.diff(clock) <= 0) or np.any(np.diff(ids) <= 0):
        raise ValueError("timestamps_ns and frame ids must be strictly increasing")
    if not np.array_equal(r0["human_to_physical"], [1,0]) or list(raw["anatomical_side_names"]) != ["left","right"]:
        raise ValueError("side identity mismatch")
    if tuple(raw['mano_joint_names']) != MANO_JOINT_NAMES:
        raise ValueError('MANO joint identity/order mismatch')
    joints = np.asarray(raw["joints_3d_camera"],dtype=float)
    observed = np.asarray(raw["observed"], dtype=bool)
    n = len(ids)
    if joints.shape != (2,n,21,3) or observed.shape != (2,n):
        raise ValueError("RAW MANO21 shape mismatch")
    source_obs = observed[::-1].T.copy()
    valid = np.asarray(r0["valid_side_frame"],dtype=bool).copy()
    relative = np.asarray(r0["relative_wrist_T"],dtype=float)
    if valid.shape != (n,2) or relative.shape != (n,2,4,4) or np.any(valid & ~source_obs):
        raise ValueError("R0 wrist valid not supported by source observed")
    target_local = np.full((n,2,21,3),np.nan)
    target_base = np.full_like(target_local,np.nan)
    root_targets = np.full((n,2,4,4),np.nan)
    rotations = np.full((n,2,3,3),np.nan)
    scales = np.full(2,np.nan)
    reasons = np.full((n,2),"UNOBSERVED",dtype="U96")
    for physical in range(2):
        human = 1-physical
        source = joints[human]
        valid[:,physical] &= np.isfinite(source).all(axis=(1,2)) & np.isfinite(relative[:,physical]).all(axis=(1,2))
        selected = np.flatnonzero(valid[:,physical])
        if not len(selected):
            continue
        neutral = np.asarray(neutral_hand_points[physical],dtype=float)
        basis_robot = palm_basis(neutral)
        neutral_radii = np.linalg.norm(neutral[[5,9,13,17]]-neutral[0],axis=-1)
        source_radii = np.linalg.norm(source[selected][:,[5,9,13,17]]-source[selected,0,None],axis=-1)
        scale = float(np.median(neutral_radii)/np.median(source_radii))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("degenerate session hand shape scale")
        scales[physical] = scale
        for t in selected:
            try:
                basis_source = palm_basis(source[t])
            except (ValueError, HuroHandOnlyError):
                valid[t,physical] = False
                reasons[t,physical] = "INVALID_PALM_BASIS"
                continue
            rotations[t,physical] = basis_robot @ basis_source.T
            local = (source[t]-source[t,0]) @ basis_source @ basis_robot.T * scale
            root = np.asarray(neutral_roots[physical]) @ relative[t,physical]
            if not np.allclose(root[3],[0,0,0,1],atol=1e-8) or not np.allclose(root[:3,:3].T@root[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(root[:3,:3]) < .999999:
                raise ValueError("invalid frozen wrist transform")
            root_targets[t,physical] = root
            target_local[t,physical] = local
            target_base[t,physical] = local @ root[:3,:3].T + root[:3,3]
            reasons[t,physical] = "DIRECT_SOURCE_SHAPE_OFFLINE_NORMALIZED"
    return {"target21_root": target_local, "target21_base": target_base, "target_valid": valid,
            "target_root_T": root_targets, "source_observed_physical": source_obs,
            "fixed_scale_by_physical": scales, "camera_to_robot_palm_rotation": rotations,
            "source_frame_id": ids.copy(), "timestamp_ns": clock.copy(), "reason": reasons,
            "human_to_physical": np.asarray([1,0]), "training_eligible": np.asarray(False),
            "target_authority": np.asarray("OFFLINE_NONCAUSAL_SESSION_SHAPE_NORMALIZATION_NOT_CALIBRATION")}


def segmented_chunks(ids, times, valid, max_frames=32, max_gap_ns=250_000_000):
    """Partition without crossing missing-side changes, dropped frames or gaps."""
    if max_frames < 1 or max_gap_ns < 1:
        raise ValueError("positive segment budget required")
    out=[]
    current=[]
    for i in range(len(ids)):
        split = bool(current) and (i != current[-1]+1 or ids[i] != ids[current[-1]]+1
                    or times[i]-times[current[-1]] > max_gap_ns
                    or not np.array_equal(valid[i],valid[current[-1]]) or len(current) >= max_frames)
        if split or not np.any(valid[i]):
            if current: out.append(current)
            current=[]
        if np.any(valid[i]): current.append(i)
    if current: out.append(current)
    return out
