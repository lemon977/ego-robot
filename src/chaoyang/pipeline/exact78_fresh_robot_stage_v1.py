"""Fresh HaWoR v3 -> camera-frame Robot diagnostic, with no historical motion input.

This is a stage helper, not a runnable/maintained operation. The fenced T1-B
operation owns its invocation and output directory. Camera-only placement is
deliberately diagnostic until a separately verified capture/world time map is
available; a video emitted here never confers visual or training qualification.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_robot_raster_v1 as raster
from chaoyang.pipeline import robot_scene_state_cpu as scene
from chaoyang.pipeline import robot_wrist_kai_adapter as adapter
from chaoyang.pipeline.exact78_motion_state_v1 import ROI_SOURCES, TIME_MODES, short_gap_limit_frames
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets


ROOT = Path(__file__).resolve().parents[3]
SIDES = ("left", "right")
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
MOUNT = ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json"
FLANGE = ROOT / "assets/robot/V2相机固定法兰PRO版本.STL"
WIDTH, HEIGHT = 640, 480
POSITION_GATE_MM, ROTATION_GATE_DEG = 10.0, 5.0
HAND_BONE_GATE_DEG, HAND_TIP_GATE_DEG = 60.0, 15.0
ARM_VELOCITY_GATE_RAD, ARM_ACCELERATION_GATE_RAD = 0.12, 0.06
HAND_VELOCITY_GATE_RAD, HAND_ACCELERATION_GATE_RAD = 0.08, 0.06
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_GROUPS = {"thumb": np.arange(0, 6), "index": np.arange(6, 10),
               "middle": np.arange(10, 14), "ring": np.arange(14, 18),
               "pinky": np.arange(18, 22)}
HAND_TERMINALS = {"thumb": 6, "index": 4, "middle": 4, "ring": 4, "pinky": 4}
MANO_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
MANO_CHAINS = {"thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8),
               "middle": (0, 9, 10, 11, 12), "ring": (0, 13, 14, 15, 16),
               "pinky": (0, 17, 18, 19, 20)}


class FreshRobotError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreshRobotError(f"INVALID_JSON_OBJECT:{path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def normalize_roi_source(detail: str) -> str:
    """An ambiguous detector/tracker origin is never admitted as fresh detection."""
    source = "DIAGNOSTIC" if detail == "HAWOR_DETECTION_OR_TRACKED_RGB" else detail
    if source not in ROI_SOURCES:
        raise FreshRobotError(f"ROI_SOURCE_ENUM_DRIFT:{detail}")
    return source


def normalize_time_mode(detail: str) -> str:
    mode = "TRACKER_CLIP_METADATA_ONLY" if detail == "CLIP_FRAME_INDEX_ONLY" else detail
    if mode not in TIME_MODES:
        raise FreshRobotError(f"TIME_MODE_ENUM_DRIFT:{detail}")
    return mode


def load_fresh_inputs(hawor_npz: Path, frame_provenance: Path,
                      input_audit: Path, raw_video: Path,
                      session_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Check exact v3 lineage before importing any old-session data."""
    if (hawor_npz.name != "HAWOR_CAMERA_PREDICTIONS.npz"
            or frame_provenance.name != "FRAME_PROVENANCE.json"
            or input_audit.name != "INPUT_AUDIT.json"
            or len({hawor_npz.resolve().parent, frame_provenance.resolve().parent,
                    input_audit.resolve().parent}) != 1):
        raise FreshRobotError("FRESH_HAWOR_STAGE_SIBLING_IDENTITY")
    provenance, audit = _json(frame_provenance), _json(input_audit)
    if provenance.get("schema_version") != "exact78-hawor-frame-provenance-v3":
        raise FreshRobotError("FRESH_HAWOR_PROVENANCE_SCHEMA")
    if provenance.get("session_id") != session_id or audit.get("session_id") != session_id:
        raise FreshRobotError("FRESH_HAWOR_SESSION_IDENTITY")
    audit_roles = {"schema_version", "session_id", "source_kind", "source_role",
                   "source_split_claim", "raw_source_ref", "read_refs",
                   "model_input_jpeg_refs", "model_input_jpeg_sha256_aggregate",
                   "runtime_code", "model_assets", "vendor_model_code",
                   "calibration_claim", "same_session_slam_usage", "forbidden_reads",
                   "infiller_invoked", "world_pose_emitted"}
    if set(audit) - audit_roles:
        raise FreshRobotError("FORBIDDEN_AUDIT_INPUT_ROLE")
    if audit.get("raw_source_ref") != file_ref(raw_video):
        raise FreshRobotError("RAW_VIDEO_SOURCE_SHA_DRIFT")
    if provenance.get("intrinsics_image_size") != [1280, 960]:
        raise FreshRobotError("INTRINSICS_COORDINATE_SPACE_UNVERIFIED")
    with np.load(hawor_npz, allow_pickle=False) as archive:
        required = {"joints_3d_camera", "joints_2d", "root_orient_camera",
                    "hand_pose_rotmat", "betas", "prediction_valid", "intrinsics",
                    "original_frame_indices", "fps_nominal", "mano_joint_names",
                    "anatomical_side_names"}
        if set(archive.files) != required:
            raise FreshRobotError("FRESH_HAWOR_NPZ_ROLE_SET")
        values = {key: np.asarray(archive[key]).copy() for key in required}
    joints = values["joints_3d_camera"]
    if tuple(values["anatomical_side_names"].tolist()) != SIDES:
        raise FreshRobotError("HAWOR_SIDE_ORDER_DRIFT")
    if tuple(values["mano_joint_names"].tolist()) != MANO_NAMES:
        raise FreshRobotError("HAWOR_MANO_JOINT_ORDER_DRIFT")
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise FreshRobotError("FRESH_HAWOR_JOINT_SHAPE")
    count = joints.shape[1]
    if count == 0:
        raise FreshRobotError("FRESH_HAWOR_EMPTY_SESSION")
    shapes = {"joints_2d": (2, count, 21, 2),
              "root_orient_camera": (2, count, 3, 3),
              "hand_pose_rotmat": (2, count, 15, 3, 3),
              "betas": (2, count, 10), "prediction_valid": (2, count),
              "intrinsics": (count, 3, 3), "original_frame_indices": (count,)}
    if any(values[key].shape != shape for key, shape in shapes.items()):
        raise FreshRobotError("FRESH_HAWOR_NPZ_SHAPE")
    indices = values["original_frame_indices"]
    if not np.array_equal(indices, np.arange(count)):
        raise FreshRobotError("FRESH_HAWOR_FRAME_INDEX_DRIFT")
    if values["prediction_valid"].dtype != np.dtype(bool):
        raise FreshRobotError("FRESH_HAWOR_VALIDITY_DTYPE")
    valid = values["prediction_valid"]
    if np.any(valid & ~np.isfinite(joints).all(axis=(2, 3))):
        raise FreshRobotError("VALID_PREDICTION_NONFINITE")
    if np.any((~valid) & np.isfinite(joints).all(axis=(2, 3))):
        raise FreshRobotError("INVALID_PREDICTION_HAS_MOTION")
    if not np.isfinite(values["intrinsics"]).all() or np.any(values["intrinsics"][:, (0, 1), (0, 1)] <= 0):
        raise FreshRobotError("FRESH_HAWOR_INTRINSICS_INVALID")
    fps = float(values["fps_nominal"])
    if not math.isfinite(fps) or fps <= 0:
        raise FreshRobotError("FRESH_HAWOR_FPS_INVALID")
    rows = provenance.get("frame_rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise FreshRobotError("FRESH_HAWOR_FRAME_MAP_LENGTH")
    for index, row in enumerate(rows):
        if (row.get("decoded_display_index") != index or row.get("output_index") != index
                or not isinstance(row.get("hands"), dict)
                or set(row["hands"]) != set(SIDES)):
            raise FreshRobotError(f"FRESH_HAWOR_FRAME_MAP_DRIFT:{index}")
    values["prediction_valid"] = valid
    return values, provenance, audit


def display_tracks(joints: np.ndarray, prediction_valid: np.ndarray,
                   fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill only the display layer; do not manufacture HaWoR observations."""
    if joints.ndim != 4 or joints.shape[0] != 2 or joints.shape[2:] != (21, 3):
        raise FreshRobotError("DISPLAY_JOINT_SHAPE")
    if prediction_valid.shape != joints.shape[:2]:
        raise FreshRobotError("DISPLAY_VALID_SHAPE")
    count = joints.shape[1]
    display = np.asarray(joints, dtype=np.float64).copy()
    source = np.full((2, count), "DIAGNOSTIC_NEUTRAL", dtype="<U40")
    short = np.zeros((2, count), dtype=bool)
    edge_kind = np.full((2, count), "NONE", dtype="<U8")
    limit = short_gap_limit_frames(fps)
    for side in range(2):
        anchors = np.flatnonzero(prediction_valid[side])
        if not len(anchors):
            continue
        source[side, anchors] = "RGB_HAWOR_PREDICTED"
        for first, second in zip(anchors[:-1], anchors[1:]):
            gap = int(second - first - 1)
            if gap <= 0:
                continue
            label = "DETERMINISTIC_INTERPOLATED" if gap <= limit else "DIAGNOSTIC_LONG_GAP"
            for index in range(first + 1, second):
                ratio = (index - first) / (second - first)
                display[side, index] = (1 - ratio) * joints[side, first] + ratio * joints[side, second]
                source[side, index] = label
                short[side, index] = gap <= limit
        for edge, anchor, segment in (("PREFIX", anchors[0], range(0, anchors[0])),
                                      ("SUFFIX", anchors[-1], range(anchors[-1] + 1, count))):
            length = len(segment)
            for index in segment:
                display[side, index] = joints[side, anchor]
                source[side, index] = ("EDGE_HELD" if length <= limit
                                       else "DIAGNOSTIC_LONG_GAP")
                short[side, index] = length <= limit
                edge_kind[side, index] = edge
    return display, source, short, edge_kind


def _mounts() -> tuple[np.ndarray, dict[str, Any]]:
    payload = _json(MOUNT)
    if payload.get("status") != "FROZEN_VISUAL_REVIEW_PROXY_NOT_CALIBRATION":
        raise FreshRobotError("VISUAL_MOUNT_PROXY_AUTHORITY_DRIFT")
    proxy = payload["proxy_transforms"]
    if proxy.get("matrix_direction") != "p_flange = T_flange_hand_root_proxy @ p_hand_root":
        raise FreshRobotError("VISUAL_MOUNT_PROXY_DIRECTION_DRIFT")
    matrices = np.asarray((proxy["left_flange_L_to_hand_l_base_link"],
                           proxy["right_flange_R_to_hand_r_base_link"]), dtype=np.float64)
    if matrices.shape != (2, 4, 4) or not np.isfinite(matrices).all():
        raise FreshRobotError("VISUAL_MOUNT_PROXY_SHAPE")
    return matrices, file_ref(MOUNT)


def _rz(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(((math.cos(angle), -math.sin(angle), 0.0),
                       (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0)))


def _candidate_base(midpoint: np.ndarray, spec: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, yaw = spec
    rotation = np.asarray(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0),
                           (1.0, 0.0, 0.0))) @ _rz(yaw)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = midpoint - rotation @ np.asarray((x, y, z))
    return result


def _pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = position
    value[:3, :3] = rotation
    return value


def _palm_basis(joints: np.ndarray, side: int) -> np.ndarray:
    return adapter.final_v3_mano_palm_basis(joints, handedness=SIDES[side])


def _arm_limits(assets: Any) -> tuple[np.ndarray, np.ndarray]:
    limits = {joint.name: (joint.lower, joint.upper) for joint in assets.tianji.joints}
    lower = np.asarray([[limits[name][0] for name in row] for row in ARM_JOINT_NAMES], dtype=np.float64)
    upper = np.asarray([[limits[name][1] for name in row] for row in ARM_JOINT_NAMES], dtype=np.float64)
    if lower.shape != (2, 7) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise FreshRobotError("ARM_LIMITS_INVALID")
    return lower, upper


def _tool_fk(assets: Any, side: int, q: np.ndarray) -> np.ndarray:
    return scene._tool_fk(assets, side, q)


def _arm_margins(assets: Any, side: int, q: np.ndarray) -> np.ndarray:
    values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
    values.update(dict(zip(ARM_JOINT_NAMES[side], q, strict=True)))
    fk = forward_kinematics(assets.tianji, values)
    suffix, sign = (("L", 1.0), ("R", -1.0))[side]
    l3 = fk[f"Link3_{suffix}"][:3, 3]
    l5 = fk[f"Link5_{suffix}"][:3, 3]
    return np.asarray((sign * l3[1] - .20, sign * l5[1] - .04, l5[0] - .04))


def _arm_status(position_mm: float, rotation_deg: float, margin_m: float) -> str:
    if margin_m < -1e-8:
        return "HOLD_BRANCH"
    if position_mm > POSITION_GATE_MM:
        return "HOLD_REACH"
    if rotation_deg > ROTATION_GATE_DEG:
        return "HOLD_ORIENT"
    return "PASS_NUMERIC_CAMERA_DIAGNOSTIC"


def _arm_attempt(assets: Any, side: int, target_base: np.ndarray,
                 lower: np.ndarray, upper: np.ndarray,
                 seeds: list[np.ndarray], previous: np.ndarray | None = None) -> tuple[np.ndarray, float, float, float]:
    candidates = []
    for seed in seeds:
        initial = np.clip(np.asarray(seed, dtype=np.float64), lower, upper)

        def residual(q: np.ndarray) -> np.ndarray:
            pose = scene._pose_residual(_tool_fk(assets, side, q), target_base)
            branch = np.minimum(_arm_margins(assets, side, q) - .003, 0.0) / .001
            continuity = np.empty(0) if previous is None else .01 * (q - previous)
            return np.concatenate((pose, branch, continuity))

        free = upper - lower > 1e-12
        solved = initial.copy()
        if np.any(free):
            def residual_free(values: np.ndarray) -> np.ndarray:
                whole = initial.copy()
                whole[free] = values
                return residual(whole)
            solved[free] = least_squares(
                residual_free, initial[free], bounds=(lower[free], upper[free]),
                max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10).x
        actual = _tool_fk(assets, side, solved)
        position = float(np.linalg.norm(actual[:3, 3] - target_base[:3, 3]) * 1000.0)
        rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(
            target_base[:3, :3].T @ actual[:3, :3]).as_rotvec())))
        margin = float(np.min(_arm_margins(assets, side, solved)))
        failed = not (position <= POSITION_GATE_MM and rotation <= ROTATION_GATE_DEG and margin >= -1e-8)
        score = (failed, max(0.0, -margin),
                 max(0.0, position - POSITION_GATE_MM) + max(0.0, rotation - ROTATION_GATE_DEG),
                 position + rotation)
        candidates.append((score, solved, position, rotation, margin))
    _, chosen, position, rotation, margin = min(candidates, key=lambda row: row[0])
    return chosen, position, rotation, margin


def _sample_frames(valid: np.ndarray, maximum: int = 8) -> list[int]:
    indices = np.flatnonzero(valid.any(axis=0))
    if not len(indices):
        return []
    slots = np.rint(np.linspace(0, len(indices) - 1, min(maximum, len(indices)))).astype(int)
    return np.unique(indices[slots]).astype(int).tolist()


def _base_search(joints: np.ndarray, valid: np.ndarray, assets: Any,
                 mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 neutral: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    anchors = np.asarray([int(np.flatnonzero(valid[side])[0]) if valid[side].any() else -1
                          for side in range(2)], dtype=int)
    points = [joints[side, anchor, 0] for side, anchor in enumerate(anchors) if anchor >= 0]
    midpoint = np.mean(points, axis=0) if points else np.asarray((0.0, 0.0, 0.8))
    sampled = _sample_frames(valid)
    specs = [(x, y, z, yaw) for x in (0.50, 0.65) for y in (-0.05, 0.05)
             for z in (0.60, 0.75) for yaw in (-15.0, 0.0, 15.0)]
    if not points:
        specs = [(0.50, 0.0, 0.60, 0.0)]

    def evaluate(spec: tuple[float, float, float, float]) -> dict[str, Any]:
        base = _candidate_base(midpoint, spec)
        inverse = np.linalg.inv(base)
        offsets = np.zeros((2, 3))
        q_anchor = neutral.copy()
        neutral_roots = np.asarray([base @ _tool_fk(assets, side, neutral[side]) @ mounts[side]
                                    for side in range(2)])
        origin_basis: list[np.ndarray | None] = [None, None]
        for side, anchor in enumerate(anchors):
            if anchor < 0:
                continue
            try:
                origin_basis[side] = _palm_basis(joints[side, anchor], side)
                wanted = _pose(joints[side, anchor, 0], neutral_roots[side, :3, :3])
                solved, _, _, _ = _arm_attempt(assets, side, inverse @ wanted @ np.linalg.inv(mounts[side]),
                                                lower[side], upper[side], [neutral[side]])
                q_anchor[side] = solved
                actual = base @ _tool_fk(assets, side, solved) @ mounts[side]
                offsets[side] = actual[:3, 3] - joints[side, anchor, 0]
            except Exception:
                origin_basis[side] = None
        errors = []
        passed = 0
        for frame in sampled:
            for side in range(2):
                if not valid[side, frame] or origin_basis[side] is None:
                    continue
                try:
                    basis = _palm_basis(joints[side, frame], side)
                    rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                    wanted = _pose(joints[side, frame, 0] + offsets[side], rotation)
                    _, pos, rot, margin = _arm_attempt(
                        assets, side, inverse @ wanted @ np.linalg.inv(mounts[side]),
                        lower[side], upper[side], [q_anchor[side], neutral[side]])
                    passed += _arm_status(pos, rot, margin).startswith("PASS")
                    errors.append(pos + rot)
                except Exception:
                    continue
        score = (-passed, float(np.median(errors)) if errors else 1e9)
        return {"spec": spec, "score": score, "sample_rows": len(errors),
                "sample_passed": passed, "base": base, "offsets": offsets,
                "anchor_q": q_anchor, "neutral_roots": neutral_roots,
                "origin_basis": origin_basis}

    coarse = [evaluate(spec) for spec in specs]
    first = min(coarse, key=lambda row: row["score"])
    x, y, z, yaw = first["spec"]
    fine = [(x + dx, y + dy, z + dz, yaw + dyaw)
            for dx, dy, dz, dyaw in ((-.025, 0, 0, 0), (.025, 0, 0, 0),
                                     (0, -.025, 0, 0), (0, .025, 0, 0),
                                     (0, 0, -.025, 0), (0, 0, .025, 0),
                                     (0, 0, 0, -5.0), (0, 0, 0, 5.0))]
    evaluated = [*coarse, *(evaluate(spec) for spec in fine)] if points else coarse
    chosen = min(evaluated, key=lambda row: row["score"])
    report = {"coordinate_frame": "CAMERA_DIAGNOSTIC_FIXED",
              "rule": "same_frozen_grid_for_all_sources_v1", "sample_frames": sampled,
              "anchor_frames": anchors.tolist(), "variable_names": ["x_m", "y_m", "z_m", "yaw_deg"],
              "candidates": [{"spec": list(row["spec"]), "score": list(row["score"]),
                              "sample_rows": row["sample_rows"], "sample_passed": row["sample_passed"]}
                             for row in evaluated],
              "selected_index": evaluated.index(chosen), "selected_spec": list(chosen["spec"]),
              "T_camera_base": chosen["base"].tolist(), "offsets_camera": chosen["offsets"].tolist(),
              "world_pose_verified": False, "visual_qualified": False}
    return chosen["base"], report, chosen["offsets"], chosen["anchor_q"], chosen["neutral_roots"]


def _unit(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 1e-9):
        raise FreshRobotError("DEGENERATE_FINGER_FEATURE")
    return values / norm


def _hand_contracts(assets: Any) -> list[dict[str, Any]]:
    contracts = []
    for side, model in enumerate((assets.left_hand, assets.right_hand)):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        names = tuple(joint.name for joint in moving)
        lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
        upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
        if len(names) != 22 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise FreshRobotError("KAIHAND_LIMITS_INVALID")
        neutral = .5 * (lower + upper)
        fk = forward_kinematics(model, dict(zip(names, neutral, strict=True)))
        prefix = ("hand_l", "hand_r")[side]
        dummy = np.zeros((21, 3))
        for index, finger in ((2, "thumb"), (5, "index"), (9, "middle"), (13, "ring"), (17, "pinky")):
            dummy[index] = fk[f"{prefix}_{finger}_link1"][:3, 3]
        basis = _palm_basis(dummy, side)
        mcp = np.asarray([fk[f"{prefix}_{finger}_link1"][:3, 3] @ basis for finger in FINGERS])
        scale = float(np.mean(np.linalg.norm(mcp[1:], axis=1)))
        if scale <= 1e-9:
            raise FreshRobotError("KAIHAND_MCP_SCALE_INVALID")
        contracts.append({"model": model, "names": names, "lower": lower,
                          "upper": upper, "neutral": neutral,
                          "basis": basis, "prefix": prefix, "mcp_scale": scale})
    return contracts


def _human_finger_features(points: np.ndarray, side: int) -> dict[str, dict[str, np.ndarray]]:
    basis = _palm_basis(points, side)
    wrist = points[0]
    mcps = np.asarray([(points[MANO_MCP[finger]] - wrist) @ basis for finger in FINGERS])
    scale = float(np.mean(np.linalg.norm(mcps[1:], axis=1)))
    if scale <= 1e-9:
        raise FreshRobotError("HUMAN_MCP_SCALE_INVALID")
    bones = adapter.final_v3_mano_unit_bones_local(points, handedness=SIDES[side])
    return {finger: {"mcp": ((points[MANO_MCP[finger]] - wrist) @ basis) / scale,
                     "bones": bones[finger],
                     "tip": _unit((points[MANO_CHAINS[finger][-1]] - wrist) @ basis)}
            for finger in FINGERS}


def _robot_finger_features(contract: dict[str, Any], q: np.ndarray,
                           finger: str) -> dict[str, np.ndarray]:
    fk = forward_kinematics(contract["model"], dict(zip(contract["names"], q, strict=True)))
    points = np.asarray([np.zeros(3)] +
                        [fk[f"{contract['prefix']}_{finger}_link{index}"][:3, 3]
                         for index in range(1, HAND_TERMINALS[finger] + 1)], dtype=np.float64)
    local = points @ contract["basis"]
    bones = adapter.resample_polyline_unit_directions(points, bone_count=4) @ contract["basis"]
    return {"mcp": local[1] / contract["mcp_scale"],
            "bones": bones, "tip": _unit(local[-1])}


def _feature_row(actual: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    bone_cos = np.clip(np.sum(_unit(actual["bones"]) * _unit(target["bones"]), axis=-1), -1.0, 1.0)
    tip_cos = float(np.clip(np.sum(_unit(actual["tip"]) * _unit(target["tip"])), -1.0, 1.0))
    return {"bone_error_deg_max": float(np.degrees(np.arccos(bone_cos)).max()),
            "tip_direction_error_deg": float(np.degrees(np.arccos(tip_cos)))}


def _solve_hands(joints: np.ndarray, drawable: np.ndarray, assets: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    contracts = _hand_contracts(assets)
    q = np.asarray([[contract["neutral"] for contract in contracts] for _ in range(joints.shape[1])])
    rows: list[dict[str, Any]] = []
    for side in range(2):
        previous = contracts[side]["neutral"].copy()
        previous_previous: np.ndarray | None = None
        previous_frame: int | None = None
        for frame in range(joints.shape[1]):
            if not drawable[side, frame]:
                rows.append({"frame": frame, "side": SIDES[side], "status": "DIAGNOSTIC_NEUTRAL_NO_ANCHOR",
                             "pass_numeric": False})
                continue
            try:
                wanted = _human_finger_features(joints[side, frame], side)
                candidate = previous.copy()
                errors = []
                for finger in FINGERS:
                    group = HAND_GROUPS[finger]
                    contract = contracts[side]
                    lower, upper = contract["lower"][group], contract["upper"][group]
                    seed = np.clip(candidate[group], lower, upper)

                    def residual(local: np.ndarray) -> np.ndarray:
                        whole = candidate.copy()
                        whole[group] = local
                        actual = _robot_finger_features(contract, whole, finger)
                        return np.concatenate(((actual["bones"] - wanted[finger]["bones"]).ravel(),
                                               4.0 * (actual["tip"] - wanted[finger]["tip"]),
                                               0.1 * (actual["mcp"] - wanted[finger]["mcp"]),
                                               0.003 * (local - seed)))

                    free = upper - lower > 1e-12
                    solved = seed.copy()
                    if np.any(free):
                        def residual_free(values: np.ndarray) -> np.ndarray:
                            whole = seed.copy()
                            whole[free] = values
                            return residual(whole)
                        solved[free] = least_squares(
                            residual_free, seed[free], bounds=(lower[free], upper[free]),
                            max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9).x
                    candidate[group] = solved
                    metric = _feature_row(_robot_finger_features(contract, candidate, finger),
                                          wanted[finger])
                    errors.append(metric)
                bone = max(row["bone_error_deg_max"] for row in errors)
                tip = max(row["tip_direction_error_deg"] for row in errors)
                passed = bone <= HAND_BONE_GATE_DEG and tip <= HAND_TIP_GATE_DEG
                status = "PASS_NUMERIC_CAMERA_DIAGNOSTIC" if passed else "HOLD_HAND_ANATOMY"
                if previous_frame is not None and frame == previous_frame + 1:
                    if np.max(np.abs(candidate - previous)) > HAND_VELOCITY_GATE_RAD + 1e-9:
                        status, passed = "HOLD_HAND_VELOCITY", False
                    elif (previous_previous is not None and
                          np.max(np.abs(candidate - 2 * previous + previous_previous))
                          > HAND_ACCELERATION_GATE_RAD + 1e-9):
                        status, passed = "HOLD_HAND_ACCELERATION", False
                if status in {"HOLD_HAND_VELOCITY", "HOLD_HAND_ACCELERATION"}:
                    candidate = previous.copy()
                previous_previous = previous.copy() if previous_frame == frame - 1 else None
                q[frame, side] = candidate
                previous = candidate
                previous_frame = frame
                rows.append({"frame": frame, "side": SIDES[side],
                             "status": status,
                             "pass_numeric": bool(passed), "bone_error_deg_max": float(bone),
                             "tip_direction_error_deg_max": float(tip)})
            except Exception as exc:  # A single invalid prediction must not abort the review session.
                q[frame, side] = previous
                rows.append({"frame": frame, "side": SIDES[side], "status": "HOLD_HAND_SOLVER",
                             "pass_numeric": False, "error_type": type(exc).__name__})
    return q, rows


def solve_camera_diagnostic(joints: np.ndarray, prediction_valid: np.ndarray,
                            fps: float) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Fit one fixed camera-frame base, then solve/mark every frame and side."""
    display, sources, short, edge_kind = display_tracks(joints, prediction_valid, fps)
    drawable = np.isfinite(display).all(axis=(2, 3))
    assets = load_pinned_robot_assets(ROOT)
    mounts, _ = _mounts()
    lower, upper = _arm_limits(assets)
    neutral = .5 * (lower + upper)
    base, base_report, offsets, anchor_q, neutral_roots = _base_search(
        display, prediction_valid, assets, mounts, lower, upper, neutral)
    inverse = np.linalg.inv(base)
    anchors = np.asarray(base_report["anchor_frames"], dtype=int)
    origin_basis = []
    for side, anchor in enumerate(anchors):
        try:
            origin_basis.append(_palm_basis(display[side, anchor], side) if anchor >= 0 else None)
        except Exception:
            origin_basis.append(None)
    count = joints.shape[1]
    q_arm = np.tile(neutral[None], (count, 1, 1))
    arm_rows = []
    for side in range(2):
        previous = None
        previous_previous = None
        previous_frame = None
        for frame in range(count):
            source = str(sources[side, frame])
            if not drawable[side, frame]:
                arm_rows.append({"frame": frame, "side": SIDES[side], "motion_source": source,
                                 "status": "DIAGNOSTIC_NEUTRAL_NO_ANCHOR", "pass_numeric": False})
                continue
            if origin_basis[side] is None:
                arm_rows.append({"frame": frame, "side": SIDES[side], "motion_source": source,
                                 "status": "HOLD_BASIS_UNVERIFIED", "pass_numeric": False})
                continue
            try:
                basis = _palm_basis(display[side, frame], side)
                rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                target = _pose(display[side, frame, 0] + offsets[side], rotation)
                seeds = [anchor_q[side], neutral[side]] if previous is None else [previous, anchor_q[side]]
                solved, position, orientation, margin = _arm_attempt(
                    assets, side, inverse @ target @ np.linalg.inv(mounts[side]),
                    lower[side], upper[side], seeds, previous=previous)
                status = _arm_status(position, orientation, margin)
                if status.startswith("PASS") and previous is not None and previous_frame == frame - 1:
                    if np.max(np.abs(solved - previous)) > ARM_VELOCITY_GATE_RAD + 1e-9:
                        status = "HOLD_ARM_VELOCITY"
                    elif (previous_previous is not None and
                          np.max(np.abs(solved - 2 * previous + previous_previous))
                          > ARM_ACCELERATION_GATE_RAD + 1e-9):
                        status = "HOLD_ARM_ACCELERATION"
                if status.startswith("PASS"):
                    q_arm[frame, side] = solved
                    previous_previous = previous.copy() if previous is not None and previous_frame == frame - 1 else None
                    previous = solved
                else:
                    q_arm[frame, side] = previous if previous is not None else neutral[side]
                previous_frame = frame
                arm_rows.append({"frame": frame, "side": SIDES[side], "motion_source": source,
                                 "status": status, "position_mm": position,
                                 "rotation_deg": orientation, "branch_margin_m_min": margin,
                                 "pass_numeric": status.startswith("PASS")})
            except Exception as exc:  # Frame-local IK failure remains a frame status.
                q_arm[frame, side] = previous if previous is not None else neutral[side]
                arm_rows.append({"frame": frame, "side": SIDES[side], "motion_source": source,
                                 "status": "HOLD_ARM_SOLVER", "pass_numeric": False,
                                 "error_type": type(exc).__name__})
    q_hand, hand_rows = _solve_hands(display, drawable, assets)
    states = {"q_arm": q_arm, "q_hand": q_hand, "T_camera_base": base,
              "T_tool_hand_root": mounts, "display_joints_3d_camera": display,
              "prediction_valid": prediction_valid, "display_drawable": drawable,
              "short_gap": short, "motion_source": sources, "edge_gap_kind": edge_kind}
    rows = []
    arm_map = {(row["frame"], row["side"]): row for row in arm_rows}
    hand_map = {(row["frame"], row["side"]): row for row in hand_rows}
    for frame in range(count):
        for side in SIDES:
            arm_row, hand_row = arm_map[frame, side], hand_map[frame, side]
            rows.append({"frame": frame, "side": side, "motion_source": arm_row["motion_source"],
                         "edge_gap_kind": str(edge_kind[SIDES.index(side), frame]),
                         "source_coordinate_frame": "CAMERA", "world_pose_verified": False,
                         "robot_drawable": True, "motion_evidence_eligible": False,
                         "training_eligible": False, "arm": arm_row, "hand": hand_row,
                         "display_policy": "DIAGNOSTIC_CAMERA_FRAME_NO_OCCLUSION_PROOF"})
    return states, base_report, rows


def _fixed_external_camera(ego_camera: np.ndarray) -> np.ndarray:
    rotation = ego_camera[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray((0.0, 0.0, 3.2)) - rotation @ np.asarray((0.0, 0.0, .9))
    return result


def _encoder(path: Path, fps: float, width: int, height: int, temp_dir: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-framerate", f"{fps:.9f}",
         "-i", "pipe:0", "-an", "-fps_mode", "passthrough", "-c:v", "libx264",
         "-threads", "2", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, cwd=ROOT,
        env={**os.environ, "TMPDIR": str(temp_dir), "TMP": str(temp_dir), "TEMP": str(temp_dir)})


def _verify_video(path: Path, expected: int, size: tuple[int, int], fps: float) -> dict[str, Any]:
    reader = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            if frame.shape[1::-1] != size:
                raise FreshRobotError(f"OUTPUT_DECODE_SIZE_DRIFT:{path}")
            count += 1
    finally:
        reader.release()
    if count != expected:
        raise FreshRobotError(f"OUTPUT_DECODE_COUNT_DRIFT:{path}:{count}!={expected}")
    probe = subprocess.run(
        ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise FreshRobotError(f"OUTPUT_VIDEO_STREAM_DRIFT:{path}")
    stream = streams[0]
    duration = float(stream["duration"])
    if (stream.get("nb_frames") not in (None, "N/A", str(expected))
            or not math.isfinite(duration)
            or abs(duration - expected / fps) > max(0.05, 1.0 / fps)):
        raise FreshRobotError(f"OUTPUT_DURATION_DRIFT:{path}")
    return {"decoded_frames": count, "width": size[0], "height": size[1],
            "encoded_duration_seconds": duration, "video": file_ref(path)}


def render_camera_diagnostic(*, states: dict[str, np.ndarray], intrinsics: np.ndarray,
                             raw_video: Path, source_kind: str, fps: float,
                             output_root: Path) -> dict[str, Any]:
    """Render all N raw/display frames without masks, Clean, or depth claims."""
    if source_kind not in {"TRACKER_CLIP_RGB", "0915_RAW_VST", "tracker", "vst0915"}:
        raise FreshRobotError("RAW_SOURCE_KIND_UNSUPPORTED")
    count = len(states["q_arm"])
    output_root.mkdir(parents=True, exist_ok=False)
    temp_dir = output_root / "tmp"
    temp_dir.mkdir()
    videos = {"ego_robot": output_root / "ego_robot_diagnostic.mp4",
              "external_full_robot": output_root / "fixed_external_full_robot.mp4",
              "raw_motion_source": output_root / "raw_motion_source_review.mp4"}
    encoders = {name: _encoder(path, fps, WIDTH, HEIGHT, temp_dir) for name, path in videos.items()}
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise FreshRobotError("RAW_VIDEO_DECODE_OPEN")
    assets = load_pinned_robot_assets(ROOT)
    mounts = states["T_tool_hand_root"]
    flange = raster.naturalv2_local_triangles(FLANGE)
    cache: dict[Path, object] = {}
    ego_camera = np.linalg.inv(states["T_camera_base"])
    external_camera = _fixed_external_camera(ego_camera)
    external_k = np.asarray(((575.0, 0.0, 319.5), (0.0, 575.0, 239.5), (0.0, 0.0, 1.0)))
    rendered = []
    try:
        for frame in range(count):
            ok, raw = capture.read()
            if not ok:
                raise FreshRobotError(f"RAW_DECODE_EARLY_EOF:{frame}")
            if source_kind in {"0915_RAW_VST", "vst0915"}:
                if raw.shape[1] != 4096 or raw.shape[0] != 1536:
                    raise FreshRobotError(f"RAW_SBS_SHAPE_DRIFT:{frame}")
                raw = raw[:, 2048:4096]  # physical left is sourceIndex 1
            elif raw.shape[1] != 1280 or raw.shape[0] != 960:
                raise FreshRobotError(f"RAW_TRACKER_SHAPE_DRIFT:{frame}")
            raw_small = cv2.resize(raw, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            k = intrinsics[frame].astype(np.float64).copy()
            k[0] *= WIDTH / 1280.0
            k[1] *= HEIGHT / 960.0
            ego_rgb, ego_labels, ego_buffers = raster.render_robot(
                raster, assets, states["q_arm"][frame], states["q_hand"][frame],
                mounts, ego_camera, k, cache, flange, complete_robot=True,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            external_rgb, external_labels, external_buffers = raster.render_robot(
                raster, assets, states["q_arm"][frame], states["q_hand"][frame],
                mounts, external_camera, external_k, cache, flange, complete_robot=True,
                return_buffers=True, width=WIDTH, height=HEIGHT)
            ego_alpha = ego_labels >= 0
            ext_alpha = external_labels >= 0
            if (not np.isfinite(ego_buffers["depth_m"][ego_alpha]).all()
                    or not np.isfinite(external_buffers["depth_m"][ext_alpha]).all()):
                raise FreshRobotError(f"ROBOT_DEPTH_NONFINITE:{frame}")
            ego = raw_small.copy()
            ego[ego_alpha] = ego_rgb[ego_alpha]
            external = np.full_like(raw_small, 242)
            external[ext_alpha] = external_rgb[ext_alpha]
            for panel, label in ((ego, "CAMERA DIAGNOSTIC - NO OCCLUSION PROOF"),
                                 (external, "FIXED THIRD PERSON - FULL ROBOT"),
                                 (raw_small, "RAW RGB + HAWOR SOURCE AUDIT")):
                cv2.rectangle(panel, (0, 0), (WIDTH, 28), (0, 0, 0), -1)
                cv2.putText(panel, f"{label}  F{frame}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                            .42, (255, 255, 255), 1, cv2.LINE_AA)
            left_source = str(states["motion_source"][0, frame])
            right_source = str(states["motion_source"][1, frame])
            cv2.rectangle(raw_small, (0, HEIGHT - 45), (WIDTH, HEIGHT), (0, 0, 0), -1)
            cv2.putText(raw_small, f"L {left_source}", (5, HEIGHT - 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(raw_small, f"R {right_source}", (5, HEIGHT - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            for name, panel in (("ego_robot", ego), ("external_full_robot", external),
                                ("raw_motion_source", raw_small)):
                assert encoders[name].stdin is not None
                encoders[name].stdin.write(panel.tobytes())
            rendered.append({"frame": frame, "raw_decoded_display_index": frame,
                             "ego_robot_pixels": int(ego_alpha.sum()),
                             "external_robot_pixels": int(ext_alpha.sum()),
                             "robot_drawable": True,
                             "final_visible": bool(ego_alpha.any()),
                             "geometry_decision": "UNKNOWN",
                             "display_policy": "DIAGNOSTIC_ROBOT_OVER_RAW_NO_OCCLUSION_PROOF"})
        extra, _ = capture.read()
        if extra:
            raise FreshRobotError("RAW_DECODE_EXTRA_FRAMES")
        for worker in encoders.values():
            assert worker.stdin is not None
            worker.stdin.close()
            if worker.wait(timeout=120):
                raise FreshRobotError("VIDEO_ENCODER_FAILED")
    finally:
        capture.release()
        for worker in encoders.values():
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
    checks = {name: _verify_video(path, count, (WIDTH, HEIGHT), fps) for name, path in videos.items()}
    return {"videos": checks, "frame_rows": rendered,
            "frame_count": count, "nominal_duration_seconds": count / fps,
            "duration_basis": "N_OVER_NOMINAL_FPS_NOT_CAPTURE_SYNCHRONIZATION"}


def run_robot_stage(*, hawor_npz: Path, frame_provenance: Path, input_audit: Path,
                    raw_video: Path, source_kind: str, session_id: str,
                    output_root: Path) -> dict[str, Any]:
    """Produce immutable diagnostic state/videos under a caller-authorized root."""
    start = time.monotonic()
    hawor, provenance, audit = load_fresh_inputs(
        hawor_npz, frame_provenance, input_audit, raw_video, session_id)
    expected_kind = "tracker" if source_kind in {"TRACKER_CLIP_RGB", "tracker"} else "vst0915"
    if audit.get("source_kind") != expected_kind:
        raise FreshRobotError("FRESH_HAWOR_SOURCE_KIND_DRIFT")
    if (expected_kind == "vst0915" and provenance.get("physical_eye") != "left_sourceIndex_1"):
        raise FreshRobotError("PHYSICAL_EYE_IDENTITY_DRIFT")
    fps = float(hawor["fps_nominal"])
    solve_started = time.monotonic()
    states, base_report, frame_rows = solve_camera_diagnostic(
        hawor["joints_3d_camera"], hawor["prediction_valid"], fps)
    solve_seconds = time.monotonic() - solve_started
    if output_root.exists():
        raise FreshRobotError("FRESH_OUTPUT_REQUIRED")
    output_root.mkdir(parents=True)
    state_path = output_root / "ROBOT_CAMERA_DIAGNOSTIC_STATES.npz"
    np.savez_compressed(state_path, **states,
                        original_frame_indices=hawor["original_frame_indices"],
                        fps_nominal=np.asarray(fps))
    _write(output_root / "BASE_SEARCH.json", base_report)
    render_started = time.monotonic()
    render = render_camera_diagnostic(
        states=states, intrinsics=hawor["intrinsics"], raw_video=raw_video,
        source_kind=source_kind, fps=fps, output_root=output_root / "render")
    render_seconds = time.monotonic() - render_started
    hawor_ref = file_ref(hawor_npz)
    for row in frame_rows:
        index = row["frame"]
        source_row = provenance["frame_rows"][index]
        hand_row = source_row["hands"][row["side"]]
        row["hawor_motion_source"] = hand_row.get("motion_source", "UNKNOWN")
        roi_detail = hand_row.get("roi_source", "NONE")
        row["roi_source_detail"] = roi_detail
        row["roi_source"] = normalize_roi_source(roi_detail)
        row["visibility"] = hand_row.get("visibility", "UNKNOWN")
        time_detail = source_row.get("time_mode", provenance.get("time_mode", "UNKNOWN"))
        row["time_mode_detail"] = time_detail
        row["time_mode"] = normalize_time_mode(time_detail)
        for field in ("wrist_position_valid", "wrist_orientation_valid", "local_hand_shape_valid"):
            row[field] = bool(hand_row.get(field, False))
        row["raw_source_sha256"] = audit["raw_source_ref"]["sha256"]
        row["hawor_prediction_source_sha256"] = hawor_ref["sha256"]
        row["clean"] = "NOT_EVALUATED"
        row["scene_depth"] = "NOT_EVALUATED"
        row["object6d"] = "NOT_EVALUATED"
        row["final_visible"] = render["frame_rows"][index]["final_visible"]
        row["geometry_decision"] = "UNKNOWN"
    _write(output_root / "ROBOT_FRAME_STATES.json", {
        "schema_version": "exact78-fresh-robot-frame-states-v1", "session_id": session_id,
        "frame_count": len(hawor["original_frame_indices"]), "rows": frame_rows,
        "world_pose_verified": False, "timebase_verified": False,
        "visual_qualified": False, "training_eligible": False})
    _write(output_root / "RENDER_AUDIT.json", render)
    result = {"schema_version": "exact78-fresh-robot-stage-result-v1",
              "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
              "session_id": session_id,
              "inputs": {"hawor_npz": hawor_ref,
                         "frame_provenance": file_ref(frame_provenance),
                         "input_audit": file_ref(input_audit), "raw_video": file_ref(raw_video),
                         "mount_proxy": file_ref(MOUNT), "flange_mesh": file_ref(FLANGE)},
              "outputs": {"states": file_ref(state_path),
                          "base_search": file_ref(output_root / "BASE_SEARCH.json"),
                          "frame_states": file_ref(output_root / "ROBOT_FRAME_STATES.json"),
                          "render_audit": file_ref(output_root / "RENDER_AUDIT.json"),
                          "videos": render["videos"]},
              "frame_count": len(hawor["original_frame_indices"]),
              "nominal_duration_seconds": render["nominal_duration_seconds"],
              "duration_basis": render["duration_basis"],
              "source_time_mode": provenance["frame_rows"][0]["time_mode"],
              "input_audit_status": audit.get("status"),
              "counts": {"hawor_prediction_rows": int(hawor["prediction_valid"].sum()),
                         "model_input_frames": sum(row.get("model_input_index") is not None
                                                   for row in provenance["frame_rows"]),
                         "model_infilled_rows": 0,
                         "deterministic_short_fill_rows": sum(
                             row["motion_source"] == "DETERMINISTIC_INTERPOLATED"
                             for row in frame_rows),
                         "diagnostic_fill_rows": sum(
                             row["motion_source"].startswith("DIAGNOSTIC")
                             for row in frame_rows),
                         "display_drawable_rows": int(states["display_drawable"].sum()),
                         "final_visible_frames": sum(row["final_visible"] for row in render["frame_rows"]),
                         "arm_status": dict(Counter(row["arm"]["status"] for row in frame_rows)),
                         "hand_status": dict(Counter(row["hand"]["status"] for row in frame_rows))},
              "visual_qualified": False, "product_qualified": False,
              "training_eligible": False, "control_ground_truth": False,
              "wall_seconds": time.monotonic() - start,
              "stage_seconds": {"solve_camera_ik_and_hand": solve_seconds,
                                "render_and_decode_verify": render_seconds},
              "claim_limit": "Camera-frame diagnostic Robot with generic frozen base search and unchanged numeric gates; no verified world pose, occlusion, Clean, product or action-label authority."}
    _write(output_root / "RESULT.json", result)
    return result
