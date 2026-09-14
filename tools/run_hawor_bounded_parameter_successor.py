#!/usr/bin/env python3
"""CPU-only bounded temporal fit of an existing per-frame HaWoR MANO track.

This producer deliberately does not run HaWoR inference and does not claim
overlapping-window gauge fusion.  It consumes one already materialised raw
track, fits root translation, root/pose SO(3), and betas inside contiguous
observed segments, then rematerialises MANO21 on CPU.  Missing frames remain
missing.  A full-resolution 2D reprojection guard is applied by backtracking
the parameter update toward the immutable raw parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import torch  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
HAWOR_ROOT = PROJECT / "third_party/HaWoR"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
CHAINS = (
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
)
BONES = tuple((chain[index], chain[index + 1]) for chain in CHAINS for index in range(4))
SIDE_NAMES = ("left", "right")
CANDIDATE_COLORS_BGR = ((255, 255, 0), (255, 0, 255))
CANDIDATE_COLORS_RGB = ((0, 255, 255), (255, 0, 255))
RAW_COLORS_BGR = ((80, 170, 170), (170, 80, 170))
BACKTRACK_ALPHAS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03, 0.015)
MAX_ROOT_TRANSLATION_UPDATE_M = 0.03
MAX_ROOT_ROTATION_UPDATE_RAD = np.deg2rad(12.0)
MAX_POSE_ROTATION_UPDATE_RAD = np.deg2rad(15.0)
MAX_BETA_UPDATE_L2 = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index))
            start = None
    return segments


def whittaker(values: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    """Confidence-weighted second-difference fit for one observed segment."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 3 or strength <= 0:
        return values.copy()
    difference = sparse.diags((np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)), (0, 1, 2), shape=(count - 2, count), format="csc")
    weight_matrix = sparse.diags(np.maximum(np.asarray(weights, dtype=np.float64), 1e-5), format="csc")
    system = weight_matrix + strength * (difference.T @ difference)
    right = np.maximum(weights, 1e-5)[:, None] * values.reshape(count, -1)
    fitted = np.column_stack([spsolve(system, right[:, column]) for column in range(right.shape[1])])
    return fitted.reshape(values.shape)


def canonical_quaternions(matrices: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(np.asarray(matrices).reshape(-1, 3, 3)).as_quat().reshape(*matrices.shape[:-2], 4)
    flattened = quaternions.reshape(len(quaternions), -1, 4)
    for joint in range(flattened.shape[1]):
        for frame in range(1, len(flattened)):
            if np.dot(flattened[frame - 1, joint], flattened[frame, joint]) < 0:
                flattened[frame, joint] *= -1
    return flattened.reshape(quaternions.shape)


def smooth_rotations(matrices: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    shape = matrices.shape
    quaternions = canonical_quaternions(matrices)
    fitted = whittaker(quaternions, weights, strength)
    norms = np.linalg.norm(fitted, axis=-1, keepdims=True)
    fitted = fitted / np.maximum(norms, 1e-12)
    return Rotation.from_quat(fitted.reshape(-1, 4)).as_matrix().reshape(shape)


def interpolate_rotations(raw: np.ndarray, proposal: np.ndarray, alpha: float) -> np.ndarray:
    flattened_raw = np.asarray(raw).reshape(-1, 3, 3)
    flattened_proposal = np.asarray(proposal).reshape(-1, 3, 3)
    relative = np.transpose(flattened_raw, (0, 2, 1)) @ flattened_proposal
    increments = Rotation.from_matrix(relative).as_rotvec() * alpha
    output = flattened_raw @ Rotation.from_rotvec(increments).as_matrix()
    return output.reshape(raw.shape)


def fit_proposals(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    observed = raw["observed"]
    confidence = raw["detector_confidence"]
    proposals = {
        "root_translation_camera": raw["root_translation_camera"].copy(),
        "root_orient_camera": raw["root_orient_camera"].copy(),
        "hand_pose_rotmat": raw["hand_pose_rotmat"].copy(),
        "betas": raw["betas"].copy(),
    }
    for side in range(2):
        side_observed = observed[side]
        if side_observed.any():
            track_weights = np.clip(confidence[side, side_observed], 0.05, 1.0) ** 2
            track_betas = raw["betas"][side, side_observed]
            center = np.sum(track_betas * track_weights[:, None], axis=0) / np.sum(track_weights)
            proposals["betas"][side, side_observed] = center
        for start, end in contiguous_true_segments(side_observed):
            weights = np.clip(confidence[side, start:end], 0.05, 1.0) ** 2
            proposals["root_translation_camera"][side, start:end] = whittaker(
                raw["root_translation_camera"][side, start:end], weights, 3.0
            )
            proposals["root_orient_camera"][side, start:end] = smooth_rotations(
                raw["root_orient_camera"][side, start:end], weights, 1.5
            )
            proposals["hand_pose_rotmat"][side, start:end] = smooth_rotations(
                raw["hand_pose_rotmat"][side, start:end], weights, 0.35
            )
    observed = raw["observed"]
    proposals["root_translation_camera"] = bound_vectors(
        raw["root_translation_camera"], proposals["root_translation_camera"], observed, MAX_ROOT_TRANSLATION_UPDATE_M
    )
    proposals["betas"] = bound_vectors(raw["betas"], proposals["betas"], observed, MAX_BETA_UPDATE_L2)
    proposals["root_orient_camera"] = bound_rotation_updates(
        raw["root_orient_camera"], proposals["root_orient_camera"], observed, MAX_ROOT_ROTATION_UPDATE_RAD
    )
    proposals["hand_pose_rotmat"] = bound_rotation_updates(
        raw["hand_pose_rotmat"], proposals["hand_pose_rotmat"], np.repeat(observed[..., None], 15, axis=2), MAX_POSE_ROTATION_UPDATE_RAD
    )
    return proposals


def bound_vectors(raw: np.ndarray, proposal: np.ndarray, selected: np.ndarray, maximum: float) -> np.ndarray:
    output = proposal.copy()
    delta = output[selected] - raw[selected]
    flattened = delta.reshape(len(delta), -1)
    norms = np.linalg.norm(flattened, axis=1)
    scales = np.minimum(1.0, maximum / np.maximum(norms, 1e-12))
    output[selected] = raw[selected] + (flattened * scales[:, None]).reshape(delta.shape)
    return output


def bound_rotation_updates(raw: np.ndarray, proposal: np.ndarray, selected: np.ndarray, maximum_rad: float) -> np.ndarray:
    output = proposal.copy()
    raw_selected = raw[selected].reshape(-1, 3, 3)
    proposal_selected = proposal[selected].reshape(-1, 3, 3)
    relative = np.transpose(raw_selected, (0, 2, 1)) @ proposal_selected
    vectors = Rotation.from_matrix(relative).as_rotvec()
    norms = np.linalg.norm(vectors, axis=1)
    vectors *= np.minimum(1.0, maximum_rad / np.maximum(norms, 1e-12))[:, None]
    output[selected] = (raw_selected @ Rotation.from_rotvec(vectors).as_matrix()).reshape(output[selected].shape)
    return output


def derive_root_translation(raw: dict[str, np.ndarray]) -> np.ndarray:
    sys.path.insert(0, str(HAWOR_ROOT))
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    frame_count = raw["observed"].shape[1]
    translation = np.full((2, frame_count, 3), np.nan, dtype=np.float32)
    prior = Path.cwd()
    try:
        os.chdir(HAWOR_ROOT)
        for side, mano in ((0, run_mano_left), (1, run_mano)):
            selected = np.flatnonzero(raw["observed"][side])
            if selected.size == 0:
                continue
            roots = torch.from_numpy(raw["root_orient_camera"][side, selected][None].astype(np.float32))
            poses = torch.from_numpy(raw["hand_pose_rotmat"][side, selected][None].astype(np.float32))
            betas = torch.from_numpy(raw["betas"][side, selected][None].astype(np.float32))
            root_aa = rotation_matrix_to_angle_axis(roots)
            pose_aa = rotation_matrix_to_angle_axis(poses)
            zeros = torch.zeros((1, selected.size, 3), dtype=torch.float32)
            zero_joints = mano(zeros, root_aa, pose_aa, betas=betas, use_cuda=False)["joints"][0].detach().cpu().numpy()
            translation[side, selected] = raw["joints_3d_camera"][side, selected, 0] - zero_joints[:, 0]
    finally:
        os.chdir(prior)
    return translation


def materialize(raw: dict[str, np.ndarray], parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(HAWOR_ROOT))
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    frame_count = raw["observed"].shape[1]
    camera = np.full((2, frame_count, 21, 3), np.nan, dtype=np.float32)
    world = np.full_like(camera, np.nan)
    projected = np.full((2, frame_count, 21, 2), np.nan, dtype=np.float32)
    prior = Path.cwd()
    try:
        os.chdir(HAWOR_ROOT)
        for side, mano in ((0, run_mano_left), (1, run_mano)):
            selected = np.flatnonzero(raw["observed"][side])
            if selected.size == 0:
                continue
            roots = torch.from_numpy(parameters["root_orient_camera"][side, selected][None].astype(np.float32))
            poses = torch.from_numpy(parameters["hand_pose_rotmat"][side, selected][None].astype(np.float32))
            trans = torch.from_numpy(parameters["root_translation_camera"][side, selected][None].astype(np.float32))
            betas = torch.from_numpy(parameters["betas"][side, selected][None].astype(np.float32))
            root_aa = rotation_matrix_to_angle_axis(roots)
            pose_aa = rotation_matrix_to_angle_axis(poses)
            result = mano(trans, root_aa, pose_aa, betas=betas, use_cuda=False)["joints"][0].detach().cpu().numpy()
            camera[side, selected] = result
            for local, frame in enumerate(selected):
                xyz = result[local].astype(np.float64)
                world[side, frame] = ((raw["c2w"][frame, :3, :3] @ xyz.T).T + raw["c2w"][frame, :3, 3]).astype(np.float32)
                homogeneous = (raw["intrinsics"][frame] @ xyz.T).T
                projected[side, frame] = (homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32)
    finally:
        os.chdir(prior)
    return {"joints_3d_camera": camera, "joints_3d_world": world, "joints_2d": projected}


def interpolate_parameters(raw: dict[str, np.ndarray], proposal: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    observed = raw["observed"]
    for name in ("root_translation_camera", "betas"):
        output[name] = raw[name].copy()
        output[name][observed] = raw[name][observed] + alpha * (proposal[name][observed] - raw[name][observed])
    for name in ("root_orient_camera", "hand_pose_rotmat"):
        output[name] = raw[name].copy()
        for side in range(2):
            for start, end in contiguous_true_segments(observed[side]):
                output[name][side, start:end] = interpolate_rotations(
                    raw[name][side, start:end], proposal[name][side, start:end], alpha
                )
    return output


def quantile(values: np.ndarray, level: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, level)) if finite.size else None


def temporal_values(joints: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    for start, end in contiguous_true_segments(observed):
        segment = joints[start:end]
        if len(segment) >= 2:
            steps.append(np.linalg.norm(np.diff(segment[:, 0], axis=0), axis=-1) * 1000.0)
        if len(segment) >= 3:
            accelerations.append(np.linalg.norm(np.diff(segment, n=2, axis=0), axis=-1).reshape(-1) * 1000.0)
    return (
        np.concatenate(steps) if steps else np.asarray([], dtype=np.float64),
        np.concatenate(accelerations) if accelerations else np.asarray([], dtype=np.float64),
    )


def bone_cv(joints: np.ndarray, observed: np.ndarray) -> float | None:
    selected = joints[observed]
    if not len(selected):
        return None
    lengths = np.stack([np.linalg.norm(selected[:, child] - selected[:, parent], axis=-1) for parent, child in BONES], axis=1)
    means = np.mean(lengths, axis=0)
    return float(np.max(np.std(lengths, axis=0) / np.maximum(means, 1e-12)))


def rotation_domain(parameters: dict[str, np.ndarray], observed: np.ndarray) -> tuple[float, float]:
    roots = parameters["root_orient_camera"][observed].reshape(-1, 3, 3)
    poses = parameters["hand_pose_rotmat"][np.repeat(observed[..., None], 15, axis=2)].reshape(-1, 3, 3)
    rotations = np.concatenate((roots, poses), axis=0)
    orthogonality = float(np.max(np.abs(np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3))))
    determinant = float(np.min(np.linalg.det(rotations)))
    return orthogonality, determinant


def parameter_update_summary(raw: dict[str, np.ndarray], parameters: dict[str, np.ndarray]) -> dict[str, float]:
    observed = raw["observed"]
    translation_delta = np.linalg.norm(
        parameters["root_translation_camera"][observed] - raw["root_translation_camera"][observed], axis=-1
    )
    beta_delta = np.linalg.norm(parameters["betas"][observed] - raw["betas"][observed], axis=-1)
    root_raw = raw["root_orient_camera"][observed].reshape(-1, 3, 3)
    root_candidate = parameters["root_orient_camera"][observed].reshape(-1, 3, 3)
    pose_mask = np.repeat(observed[..., None], 15, axis=2)
    pose_raw = raw["hand_pose_rotmat"][pose_mask].reshape(-1, 3, 3)
    pose_candidate = parameters["hand_pose_rotmat"][pose_mask].reshape(-1, 3, 3)
    root_delta = Rotation.from_matrix(np.transpose(root_raw, (0, 2, 1)) @ root_candidate).magnitude()
    pose_delta = Rotation.from_matrix(np.transpose(pose_raw, (0, 2, 1)) @ pose_candidate).magnitude()
    return {
        "root_translation_update_max_mm": float(np.max(translation_delta) * 1000.0),
        "root_rotation_update_max_deg": float(np.rad2deg(np.max(root_delta))),
        "pose_rotation_update_max_deg": float(np.rad2deg(np.max(pose_delta))),
        "beta_update_l2_max": float(np.max(beta_delta)),
    }


def metrics(raw: dict[str, np.ndarray], candidate_joints: dict[str, np.ndarray], parameters: dict[str, np.ndarray]) -> dict[str, Any]:
    sides: dict[str, Any] = {}
    for side, name in enumerate(SIDE_NAMES):
        observed = raw["observed"][side]
        error = np.linalg.norm(candidate_joints["joints_2d"][side, observed] - raw["joints_2d"][side, observed], axis=-1)
        raw_step, raw_acceleration = temporal_values(raw["joints_3d_world"][side], observed)
        candidate_step, candidate_acceleration = temporal_values(candidate_joints["joints_3d_world"][side], observed)
        sides[name] = {
            "observed_frames_raw": int(observed.sum()),
            "observed_frames_candidate": int(np.isfinite(candidate_joints["joints_3d_camera"][side, :, 0]).all(axis=-1).sum()),
            "reprojection_p95_px": quantile(error, 0.95),
            "reprojection_max_px": float(np.max(error)) if error.size else None,
            "raw_wrist_step_p95_mm": quantile(raw_step, 0.95),
            "candidate_wrist_step_p95_mm": quantile(candidate_step, 0.95),
            "raw_all_joint_acceleration_p95_mm": quantile(raw_acceleration, 0.95),
            "candidate_all_joint_acceleration_p95_mm": quantile(candidate_acceleration, 0.95),
            "raw_bone_length_cv_max": bone_cv(raw["joints_3d_world"][side], observed),
            "candidate_bone_length_cv_max": bone_cv(candidate_joints["joints_3d_world"][side], observed),
        }
    orthogonality, determinant = rotation_domain(parameters, raw["observed"])
    return {
        "sides": sides,
        "rotation_orthogonality_max": orthogonality,
        "rotation_determinant_min": determinant,
        "identity_switch_count": 0,
        "identity_gate_basis": "axis labels and observed masks are immutable; no detection reassociation is performed",
        "parameter_update_bounds": parameter_update_summary(raw, parameters),
    }


def gate_failures(value: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for side, side_metrics in value["sides"].items():
        if side_metrics["reprojection_p95_px"] > thresholds["per_side_reprojection_p95_px_max"]:
            failures.append(f"{side}:REPROJECTION_P95")
        if side_metrics["observed_frames_candidate"] != side_metrics["observed_frames_raw"]:
            failures.append(f"{side}:COVERAGE_CHANGED")
        if not side_metrics["candidate_wrist_step_p95_mm"] < side_metrics["raw_wrist_step_p95_mm"]:
            failures.append(f"{side}:WRIST_STEP_NOT_IMPROVED")
        if not side_metrics["candidate_all_joint_acceleration_p95_mm"] < side_metrics["raw_all_joint_acceleration_p95_mm"]:
            failures.append(f"{side}:ACCELERATION_NOT_IMPROVED")
        if side_metrics["candidate_bone_length_cv_max"] > side_metrics["raw_bone_length_cv_max"] + 1e-7:
            failures.append(f"{side}:BONE_CV_REGRESSED")
    if value["identity_switch_count"] > thresholds["identity_switch_count_max"]:
        failures.append("IDENTITY_SWITCH")
    if value["rotation_orthogonality_max"] > thresholds["root_and_pose_rotation_orthogonality_max"]:
        failures.append("ROTATION_ORTHOGONALITY")
    if value["rotation_determinant_min"] <= thresholds["root_and_pose_rotation_determinant_min_exclusive"]:
        failures.append("ROTATION_DETERMINANT")
    bounds = value["parameter_update_bounds"]
    if bounds["root_translation_update_max_mm"] > MAX_ROOT_TRANSLATION_UPDATE_M * 1000.0 + 1e-4:
        failures.append("ROOT_TRANSLATION_UPDATE_BOUND")
    if bounds["root_rotation_update_max_deg"] > np.rad2deg(MAX_ROOT_ROTATION_UPDATE_RAD) + 1e-5:
        failures.append("ROOT_ROTATION_UPDATE_BOUND")
    if bounds["pose_rotation_update_max_deg"] > np.rad2deg(MAX_POSE_ROTATION_UPDATE_RAD) + 1e-5:
        failures.append("POSE_ROTATION_UPDATE_BOUND")
    if bounds["beta_update_l2_max"] > MAX_BETA_UPDATE_L2 + 1e-5:
        failures.append("BETA_UPDATE_BOUND")
    return failures


def draw_skeleton(frame: np.ndarray, joints: np.ndarray, colors: tuple[tuple[int, int, int], ...], thickness: int) -> None:
    for side in range(2):
        if not np.isfinite(joints[side]).all():
            continue
        points = np.rint(joints[side]).astype(np.int32)
        for chain in CHAINS:
            cv2.polylines(frame, [points[np.asarray(chain)]], False, colors[side], thickness, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(point), max(2, thickness), colors[side], -1, cv2.LINE_AA)


def render_review(video: Path, raw: dict[str, np.ndarray], candidate: dict[str, np.ndarray], output: Path, session: str, alpha: float, numeric_pass: bool) -> dict[str, Any]:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Chinese font missing: {FONT_PATH}")
    title_font = ImageFont.truetype(str(FONT_PATH), 20)
    body_font = ImageFont.truetype(str(FONT_PATH), 17)
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = raw["observed"].shape[1]
    if (width, height) != (1280, 960) or abs(fps - float(raw["fps"])) > 1e-3:
        raise RuntimeError("source video geometry/FPS differs from frozen HaWoR input")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-threads", "1", "-b:v", "1200k", "-maxrate", "1500k", "-bufsize", "2400k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= frame_count:
                raise RuntimeError("source video contains more frames than HaWoR input")
            cv2.rectangle(frame, (0, 0), (width, 106), (0, 0, 0), -1)
            draw_skeleton(frame, raw["joints_2d"][:, decoded], RAW_COLORS_BGR, 1)
            draw_skeleton(frame, candidate["joints_2d"][:, decoded], CANDIDATE_COLORS_BGR, 3)
            canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            painter = ImageDraw.Draw(canvas)
            painter.text((12, 6), f"{session}｜第 {decoded + 1}/{frame_count} 帧｜单轨参数时序候选", font=title_font, fill=(255, 255, 255))
            painter.text((12, 38), "细暗线＝原始 HaWoR　亮线＝参数空间候选；全片同一像素坐标/固定尺度", font=body_font, fill=(235, 235, 235))
            state = "数值门通过，待人工复核" if numeric_pass else "数值门未通过，仅供诊断"
            painter.text((12, 70), f"青＝解剖左手　紫＝解剖右手　回退系数={alpha:.3f}　{state}", font=body_font, fill=(255, 190, 80))
            rendered = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            assert encoder.stdin is not None
            encoder.stdin.write(rendered.tobytes())
            decoded += 1
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
    return_code = encoder.wait()
    verify = cv2.VideoCapture(str(output))
    output_frames = 0
    while True:
        ok, _ = verify.read()
        if not ok:
            break
        output_frames += 1
    verify.release()
    if return_code != 0 or decoded != frame_count or output_frames != frame_count:
        raise RuntimeError(f"review full-decode failed: ffmpeg={return_code}, input={decoded}, output={output_frames}, expected={frame_count}")
    return {"input_frames": decoded, "output_frames": output_frames, "resolution": [width, height], "fps": fps, "fixed_pixel_scale": True}


def load_raw(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(row["raw_hawor_npz"]).resolve(strict=True)
    if sha256(path) != row["raw_hawor_sha256"]:
        raise RuntimeError("raw HaWoR SHA mismatch")
    video = Path(row["source_video"]).resolve(strict=True)
    if sha256(video) != row["source_video_sha256"]:
        raise RuntimeError("source video SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in archive.files}
    frame_count = int(row["frame_count"])
    if raw["joints_3d_camera"].shape != (2, frame_count, 21, 3):
        raise RuntimeError("raw frame/MANO shape mismatch")
    if tuple(str(value) for value in raw["mano_joint_names"].tolist()) != MANO_NAMES:
        raise RuntimeError("MANO21 identity mismatch")
    if tuple(str(value) for value in raw["anatomical_side_names"].tolist()) != SIDE_NAMES:
        raise RuntimeError("anatomical side identity mismatch")
    raw["root_translation_camera"] = derive_root_translation(raw)
    return raw


def save_npz(path: Path, raw: dict[str, np.ndarray], parameters: dict[str, np.ndarray], joints: dict[str, np.ndarray], alpha: float) -> None:
    payload = {name: value for name, value in raw.items() if name != "root_translation_camera"}
    payload.update(joints)
    payload.update(parameters)
    payload.update(
        {
            "provenance": np.where(raw["observed"], "BOUNDED_PARAMETER_FIT", "MISSING").astype("U24"),
            "visibility": np.full(raw["observed"].shape, "UNKNOWN", dtype="U8"),
            "source_raw_hawor_sha256": np.asarray(sha256(Path(str(raw["_source_path"]))), dtype="U64"),
            "method": np.asarray("SINGLE_TRACK_CONFIDENCE_BOUNDED_PARAMETER_FIT_NO_WINDOW_GAUGE", dtype="U64"),
            "backtracking_alpha": np.asarray(alpha, dtype=np.float64),
            "window_gauge_status": np.asarray("NOT_APPLICABLE_NO_PER_WINDOW_INPUT", dtype="U40"),
        }
    )
    payload.pop("_source_path", None)
    np.savez_compressed(path, **payload)


def process_canary(row: dict[str, Any], thresholds: dict[str, Any], output_root: Path) -> dict[str, Any]:
    started = time.time()
    final = output_root / row["session_id"]
    if final.exists():
        raise RuntimeError(f"no-clobber output already exists: {final}")
    stage = output_root / f".{row['session_id']}.stage.{os.getpid()}"
    if stage.exists():
        raise RuntimeError(f"staging collision: {stage}")
    stage.mkdir(parents=True)
    try:
        raw = load_raw(row)
        raw["_source_path"] = np.asarray(row["raw_hawor_npz"])
        proposal = fit_proposals(raw)
        attempts: list[dict[str, Any]] = []
        accepted: tuple[float, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]] | None = None
        last: tuple[float, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]] | None = None
        for alpha in BACKTRACK_ALPHAS:
            parameters = interpolate_parameters(raw, proposal, alpha)
            joints = materialize(raw, parameters)
            value = metrics(raw, joints, parameters)
            failures = gate_failures(value, thresholds)
            attempts.append({"alpha": alpha, "failures": failures, "metrics": value})
            last = (alpha, parameters, joints, value)
            if not failures:
                accepted = last
                break
        chosen = accepted if accepted is not None else last
        assert chosen is not None
        alpha, parameters, joints, value = chosen
        failures = gate_failures(value, thresholds)
        numeric_pass = not failures
        npz_path = stage / "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
        save_npz(npz_path, raw, parameters, joints, alpha)
        video_path = stage / "HAWOR_RAW_VS_BOUNDED_PARAMETER_中文全片固定尺度.mp4"
        video_validation = render_review(Path(row["source_video"]), raw, joints, video_path, row["session_id"], alpha, numeric_pass)
        result = {
            "schema_version": "hawor-bounded-parameter-single-track-canary-v1",
            "status": "PASS_NUMERIC_NEEDS_HUMAN_REVIEW" if numeric_pass else "HOLD_NUMERIC_GATES",
            "task": row["task"],
            "session_id": row["session_id"],
            "claim_limit": "CPU single-track bounded MANO/root parameter fit only; not HaWoR re-inference, not window-overlap gauge fusion, and not batch/Robot authority.",
            "algorithm": {
                "domain": "root translation + root SO(3) + 15 pose SO(3) + betas",
                "data_weights": "clip(detector_confidence,0.05,1)^2",
                "temporal_prior": "confidence-weighted second-difference Whittaker fit within contiguous observed segments",
                "rotation_projection": "unit quaternion fit rematerialized on SO(3), then geodesic SLERP backtracking",
                "beta_prior": "per-side confidence-weighted track constancy",
                "hard_update_caps": {
                    "root_translation_mm": MAX_ROOT_TRANSLATION_UPDATE_M * 1000.0,
                    "root_rotation_deg": float(np.rad2deg(MAX_ROOT_ROTATION_UPDATE_RAD)),
                    "pose_rotation_deg": float(np.rad2deg(MAX_POSE_ROTATION_UPDATE_RAD)),
                    "beta_l2": MAX_BETA_UPDATE_L2,
                },
                "missing_policy": "missing stays NaN/MISSING; no interpolation and no cross-gap regularization",
                "window_gauge": "NOT_APPLICABLE_NO_PER_WINDOW_INPUT",
                "forbidden_routes_used": [],
            },
            "selected_backtracking_alpha": alpha,
            "numeric_gate_pass": numeric_pass,
            "failures": failures,
            "metrics": value,
            "backtracking_attempts": attempts,
            "validation": {
                "input_sha_verified": True,
                "source_video_sha_verified": True,
                "coverage_preserved_exactly": all(item["observed_frames_raw"] == item["observed_frames_candidate"] for item in value["sides"].values()),
                "missing_interpolation_performed": False,
                "gpu_calls": 0,
                "full_video": video_validation,
            },
            "inputs": {"raw_hawor": evidence(Path(row["raw_hawor_npz"])), "source_video": evidence(Path(row["source_video"]))},
            "outputs": {"npz": evidence(npz_path), "review_video": evidence(video_path)},
            "wall_seconds": time.time() - started,
        }
        atomic_json(stage / "RESULT.json", result)
        os.replace(stage, final)
        final_result = json.loads((final / "RESULT.json").read_text(encoding="utf-8"))
        for record in final_result["outputs"].values():
            record["path"] = str(final / Path(record["path"]).name)
        atomic_json(final / "RESULT.json", final_result)
        return {"session_id": row["session_id"], "status": final_result["status"], "result": evidence(final / "RESULT.json"), "failures": failures}
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    contract_path = args.contract.resolve(strict=True)
    if output_root.exists():
        raise RuntimeError(f"fresh output root required: {output_root}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks = []
    for row in contract["canaries"]:
        npz = Path(row["raw_hawor_npz"]).resolve(strict=True)
        video = Path(row["source_video"]).resolve(strict=True)
        checks.append(
            {
                "session_id": row["session_id"],
                "raw_sha_match": sha256(npz) == row["raw_hawor_sha256"],
                "video_sha_match": sha256(video) == row["source_video_sha256"],
            }
        )
    if not all(item["raw_sha_match"] and item["video_sha_match"] for item in checks):
        raise RuntimeError("preflight input SHA mismatch")
    if args.preflight_only:
        print(json.dumps({"status": "PASS_PREFLIGHT_CPU", "checks": checks, "gpu_calls": 0}, ensure_ascii=False))
        return 0
    output_root.mkdir(parents=True)
    rows = []
    for row in contract["canaries"]:
        try:
            rows.append(process_canary(row, contract["acceptance_thresholds"], output_root))
        except Exception as error:
            rows.append({"session_id": row["session_id"], "status": "HOLD_EXCEPTION", "error": repr(error)})
    overall_pass = all(row["status"] == "PASS_NUMERIC_NEEDS_HUMAN_REVIEW" for row in rows)
    result = {
        "schema_version": "hawor-bounded-parameter-two-canary-run-v1",
        "status": "PASS_TWO_NUMERIC_NEEDS_HUMAN_REVIEW" if overall_pass else "HOLD_ONE_OR_MORE_CANARIES",
        "claim_limit": "CPU single-track parameter temporal canary only; neither overlap-gauge fusion nor batch authority.",
        "contract": evidence(contract_path),
        "producer": evidence(Path(__file__)),
        "canaries": rows,
        "gpu_calls": 0,
    }
    atomic_json(output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
