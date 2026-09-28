"""V5 motion handoff from SHA-pinned HaWoR and local Robot candidates.

This module preserves missing sides as invalid. It never infers a hand from the
other side and never upgrades a legacy visual candidate to quality PASS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "HUMAN_TO_ROBOT_BASELINE_V1_MOTION"
SIDES = ("left", "right")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _ref(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _pinned(ref: dict[str, Any]) -> Path:
    path = Path(ref["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _ref(path)
    if actual["sha256"] != ref["sha256"] or ("bytes" in ref and actual["bytes"] != ref["bytes"]):
        raise ValueError(f"SOURCE_REF_MISMATCH:{path}")
    return path


def _segments(valid: np.ndarray) -> np.ndarray:
    result = np.full(valid.shape, -1, dtype=np.int32)
    for side in range(2):
        segment = -1
        previous = False
        for frame, current in enumerate(valid[:, side]):
            if current and not previous:
                segment += 1
            if current:
                result[frame, side] = segment
            previous = bool(current)
    return result


def _counts(mask: np.ndarray) -> list[int]:
    return [int(mask[:, side].sum()) for side in range(2)]


def _model_input(result: dict[str, Any], frame_id: np.ndarray) -> np.ndarray:
    index = {int(frame): row for row, frame in enumerate(frame_id)}
    mask = np.zeros((len(frame_id), 2), dtype=bool)
    for chunk in result.get("chunks", []):
        side = int(chunk["side"])
        if side not in (0, 1):
            raise ValueError("INVALID_CHUNK_SIDE")
        for frame in chunk["frames"]:
            if int(frame) not in index:
                raise ValueError("CHUNK_FRAME_OUTSIDE_SOURCE")
            mask[index[int(frame)], side] = True
    return mask


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _require_shape(data: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    array = data[key]
    if array.shape != shape:
        raise ValueError(f"{key}_SHAPE:{array.shape}!={shape}")
    return array


def recover(config_path: Path) -> dict[str, Any]:
    """Export normalized motion and a transparent, legacy R0 candidate.

    Config must pin source/ROI/model receipt/Robot NPZ by SHA. Re-running an
    identical config uses the existing receipt only if its inputs and outputs
    still match. A changed config requires a fresh output directory.
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("CONFIG_SCHEMA")
    source_paths = {key: _pinned(config[key]) for key in ("hawor_source", "hawor_result", "roi", "robot_r0")}
    optional_refs = {key: _pinned(config[key]) for key in ("robot_result", "asset", "mount") if key in config}
    input_refs = {key: _ref(path) for key, path in {**source_paths, **optional_refs}.items()}
    config_ref = _ref(config_path)
    code_ref = _ref(Path(__file__))
    run_signature = hashlib.sha256(_canonical({"schema": SCHEMA, "config": config_ref, "code": code_ref, "inputs": input_refs})).hexdigest()
    output = Path(config["output"])
    receipt_path = output / "RESULT.json"
    if output.exists():
        if not receipt_path.is_file():
            raise FileExistsError(f"PARTIAL_OUTPUT:{output}")
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != run_signature:
            raise FileExistsError(f"OUTPUT_SIGNATURE_CHANGED:{output}")
        for ref in old["outputs"].values():
            _pinned(ref)
        return {**old, "cache": "REUSED"}

    hand = _load_npz(source_paths["hawor_source"])
    roi = _load_npz(source_paths["roi"])
    robot = _load_npz(source_paths["robot_r0"])
    source_result = json.loads(source_paths["hawor_result"].read_text(encoding="utf-8"))
    frame_id = np.asarray(hand["original_frame_indices"], dtype=np.int64)
    timestamp_ns = np.asarray(hand["timestamp_ns"], dtype=np.int64)
    frames = len(frame_id)
    if frames != int(config["frame_count"]) or len(set(frame_id.tolist())) != frames:
        raise ValueError("FRAME_ID_COUNT_OR_DUPLICATE")
    if np.any(np.diff(timestamp_ns) <= 0):
        raise ValueError("NON_MONOTONIC_TIMESTAMPS")
    if hand["anatomical_side_names"].tolist() != list(SIDES) or roi["anatomical_side_names"].tolist() != list(SIDES):
        raise ValueError("SIDE_ORDER")
    if not np.array_equal(frame_id, robot["frame_id"]) or not np.array_equal(timestamp_ns, robot["timestamp_ns"]):
        raise ValueError("ROBOT_FRAME_TIME_MISMATCH")
    roi_valid = _require_shape(roi, "roi_valid", (frames, 2)).astype(bool)
    model_input = _model_input(source_result, frame_id)
    prediction = _require_shape(hand, "predicted_valid", (2, frames)).T.astype(bool)
    target_valid = _require_shape(robot, "target_valid", (frames, 2)).astype(bool)
    wrist_valid = _require_shape(robot, "wrist_valid", (frames, 2)).astype(bool)
    finger_valid = _require_shape(robot, "finger_valid", (frames, 2)).astype(bool)
    if np.any(model_input & ~roi_valid) or np.any(prediction & ~model_input) or np.any(target_valid & ~prediction):
        raise ValueError("MOTION_STAGE_MONOTONICITY")
    if np.any(wrist_valid & ~target_valid) or np.any(finger_valid & ~target_valid):
        raise ValueError("ROBOT_VALID_WITHOUT_TARGET")
    _require_shape(hand, "joints_3d_camera", (2, frames, 21, 3))
    _require_shape(hand, "root_orient_camera", (2, frames, 3, 3))
    _require_shape(hand, "hand_pose_rotmat", (2, frames, 15, 3, 3))
    _require_shape(robot, "q_arm", (frames, 2, 7))
    _require_shape(robot, "q22", (frames, 2, 22))
    wrist_T = np.full((frames, 2, 4, 4), np.nan, dtype=np.float64)
    for side in range(2):
        accepted = prediction[:, side]
        wrist_T[accepted, side] = np.eye(4)
        wrist_T[accepted, side, :3, :3] = hand["root_orient_camera"][side, accepted]
        wrist_T[accepted, side, :3, 3] = hand["joints_3d_camera"][side, accepted, 0]
    if not np.all(np.isfinite(wrist_T[prediction])):
        raise ValueError("NONFINITE_VALID_WRIST")
    joints_camera = hand["joints_3d_camera"].transpose(1, 0, 2, 3).astype(np.float64)
    joints_root = np.full_like(joints_camera, np.nan)
    for side in range(2):
        accepted = prediction[:, side]
        delta = joints_camera[accepted, side] - wrist_T[accepted, side, None, :3, 3]
        joints_root[accepted, side] = np.einsum(
            "tji,tkj->tki", wrist_T[accepted, side, :3, :3], delta
        )
    if not np.all(np.isfinite(joints_root[prediction])):
        raise ValueError("NONFINITE_VALID_JOINTS")
    model_input_count = _counts(model_input)
    stages = {
        "roi": _counts(roi_valid),
        "model_input": model_input_count,
        "raw_prediction": _counts(prediction),
        "retarget_target": _counts(target_valid),
        "robot_wrist": _counts(wrist_valid),
        "robot_finger": _counts(finger_valid),
    }
    first_recorded_loss = []
    for side in range(2):
        loss = next((name for name in stages if stages[name][side] == 0), None)
        first_recorded_loss.append(loss)
    output.mkdir(parents=True, exist_ok=False)
    hand_path = output / "HAND_MOTION_V1.npz"
    robot_path = output / "ROBOT_R0_V1.npz"
    np.savez_compressed(
        hand_path, frame_id=frame_id, timestamp_ns=timestamp_ns,
        anatomical_side_names=np.asarray(SIDES), source_kind=np.asarray("HAWOR_INFERRED"),
        roi_valid=roi_valid, model_input=model_input, predicted_valid=prediction,
        observed_physical=_require_shape(hand, "observed", (2, frames)).T,
        inferred_physical=_require_shape(hand, "inferred", (2, frames)).T,
        segment_id=_segments(prediction), T_camera_wrist=wrist_T,
        joints21_camera=joints_camera, joints21_root=joints_root,
        position_valid=prediction, rotation_valid=prediction,
        joint_valid=np.broadcast_to(prediction[:, :, None], (frames, 2, 21)).copy(),
        estimate_source=np.where(prediction, "HAWOR_INFERRED", "UNKNOWN"),
        T_world_camera=np.full((frames, 4, 4), np.nan, dtype=np.float64),
        world_valid=np.zeros(frames, dtype=bool),
        hand_pose_rotmat=hand["hand_pose_rotmat"].transpose(1, 0, 2, 3, 4),
        intrinsics=hand["intrinsics"],
        units_position=np.asarray("m"), units_angle=np.asarray("rad"),
        control_ground_truth=np.asarray(False), training_eligible=np.asarray(False),
    )
    np.savez_compressed(
        robot_path, frame_id=frame_id, timestamp_ns=timestamp_ns,
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
        legacy_candidate=np.asarray(True), control_ground_truth=np.asarray(False),
        training_eligible=np.asarray(False),
    )
    receipt = {
        "schema_version": SCHEMA, "session_id": config["session_id"],
        "frame_count": frames, "status": "LEGACY_MOTION_RECOVERED_FOR_DIAGNOSTIC",
        "run_signature": run_signature, "cache": "FRESH", "stage_counts_left_right": stages,
        "first_recorded_zero_stage_left_right": first_recorded_loss,
        "visibility_proxy": "NOT_SIDE_RESOLVED_IN_THIS_SOURCE",
        "numeric_quality_pass": False, "offline_visual": True,
        "control_ground_truth": False, "training_eligible": False,
        "inputs": input_refs, "config": config_ref, "code": code_ref,
        "outputs": {"hand_motion": _ref(hand_path), "robot_r0": _ref(robot_path)},
    }
    receipt_path.write_bytes(_canonical(receipt))
    return receipt
