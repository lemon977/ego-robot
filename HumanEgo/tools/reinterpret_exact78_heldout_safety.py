#!/usr/bin/env python3
"""Posthoc CPU safety interpretation of existing heldout prediction-only NPZs.

This never calls the model and never overwrites the source prediction.  It is
diagnostic evidence for the execution adapter only, not a repaired inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
import sys
sys.path.insert(0, str(HUMANEGO))

from inference.receding_horizon import CausalPoseRateLimiter, RecedingH50QController  # noqa: E402
from utils.utils_math import o6d_to_rotmat, rotmat_to_o6d  # noqa: E402


EXECUTION_HORIZON = 10
MAX_Q_STEP_RAD = float(np.deg2rad(10.0))
MAX_WRIST_STEP_M = 0.010
MAX_WRIST_ROTATION_STEP_RAD = float(np.deg2rad(8.0))
WRIST_EMA_ALPHA = 0.65
Q_ENSEMBLE_DECAY = 0.5
PHYSICAL_TO_HUMAN_SIDE = ("right", "left")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def limits(run_root: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    first = sorted(manifest["sidecars"])[0]
    sidecar = Path(manifest["bundle"]) / "sidecars/kai22" / first / "sidecar.npz"
    with np.load(sidecar, allow_pickle=False) as archive:
        return (
            np.asarray(archive["joint_lower"], dtype=np.float64),
            np.asarray(archive["joint_upper"], dtype=np.float64),
        )


def render(
    rgb: np.ndarray, executed: np.ndarray, mean: np.ndarray,
    std: np.ndarray, intrinsic: np.ndarray, frame: int,
) -> np.ndarray:
    panel = cv2.resize(rgb, (960, 540))
    source_width = max(1.0, float(intrinsic[0, 2]) * 2.0)
    source_height = max(1.0, float(intrinsic[1, 2]) * 2.0)
    positions = executed[:, :6].reshape(len(executed), 2, 3) * std + mean
    for hand, color in ((0, (255, 120, 30)), (1, (30, 60, 255))):
        xyz = positions[:, hand]
        uv = np.stack((
            intrinsic[0, 0] * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-6) + intrinsic[0, 2],
            intrinsic[1, 1] * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-6) + intrinsic[1, 2],
        ), axis=-1)
        uv[:, 0] *= 960 / source_width
        uv[:, 1] *= 540 / source_height
        visible = (
            (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 960)
            & (uv[:, 1] >= 72) & (uv[:, 1] < 540)
        )
        points = uv[visible].astype(np.int32)
        if len(points) > 1:
            cv2.polylines(panel, [points[:, None]], False, color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(panel, tuple(point), 3, color, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (960, 72), (0, 0, 0), -1)
    cv2.putText(panel, "POSTHOC SAFE PREFIX (NOT NEW INFERENCE)", (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame {frame:05d} | blue=Kai-L(human-R), red=Kai-R(human-L)",
                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def process_session(
    source: Path, output: Path, mean: np.ndarray, std: np.ndarray,
    lower: np.ndarray, upper: np.ndarray,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"no-clobber output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prediction = source / "heldout_checkpoint_predictions.npz"
    with np.load(prediction, allow_pickle=False) as archive:
        if bool(archive["contains_ground_truth"].item()) or bool(archive["contains_heldout_robot_q"].item()):
            raise RuntimeError("source is not prediction-only")
        plans = np.asarray(archive["checkpoint_full_plan_action"], dtype=np.float32)
        old_prefix = np.asarray(archive["causally_executed_prefix"], dtype=np.float32)
        frames = np.asarray(archive["observation_frame"], dtype=np.int64)
        c2ws = np.asarray(archive["observation_c2w"], dtype=np.float64)
    session = source.name
    hawor_path = next(source.glob(
        "heldout_eval_input/hawor_v3_sidecars/*/entities_hawor_v3.npz"
    ))
    with np.load(hawor_path, allow_pickle=False) as hawor:
        names = np.asarray(hawor["frame_names"]).astype(str)
        side_names = np.asarray(hawor["side_names"]).astype(str).tolist()
        wrist = np.asarray(hawor["T_hand_to_camera"], dtype=np.float64)
    if side_names != ["left", "right"]:
        raise RuntimeError("unexpected HaWoR side order")
    by_frame = {int(name): index for index, name in enumerate(names)}
    first_wrist_physical = wrist[by_frame[int(frames[0])], ::-1]

    q_controller = RecedingH50QController(
        22, execute_steps=EXECUTION_HORIZON,
        ensemble_decay=Q_ENSEMBLE_DECAY, max_q_step_rad=MAX_Q_STEP_RAD,
    )
    pose_controller = CausalPoseRateLimiter(
        max_translation_step_m=MAX_WRIST_STEP_M,
        max_rotation_step_rad=MAX_WRIST_ROTATION_STEP_RAD,
        ema_alpha=WRIST_EMA_ALPHA,
    )
    initial_world = np.asarray([
        c2ws[0] @ first_wrist_physical[hand] for hand in range(2)
    ])
    pose_controller.process(initial_world, np.ones(2, dtype=bool))
    current_q = np.clip(np.zeros((2, 22), dtype=np.float64), lower, upper)
    safe_prefixes = []
    raw_limit_ratios = []
    q_clip_ratios = []
    render_frames = []
    safe_world_sequence = []
    raw_world_sequence = []
    for replan, (plan, frame, c2w) in enumerate(zip(plans, frames, c2ws)):
        safe, diagnostic = q_controller.process(
            replan_frame=int(frame), plan_action=plan, current_q=current_q,
            joint_lower=lower, joint_upper=upper,
        )
        for offset in range(EXECUTION_HORIZON):
            target_world = []
            for hand in range(2):
                pose = np.eye(4, dtype=np.float64)
                pose[:3, 3] = safe[offset, hand * 3:hand * 3 + 3] * std + mean
                pose[:3, :3] = o6d_to_rotmat(
                    safe[offset, 6 + hand * 6:12 + hand * 6]
                )
                target_world.append(c2w @ pose)
            smooth_world, _ = pose_controller.process(
                np.asarray(target_world), np.ones(2, dtype=bool),
            )
            world_to_camera = np.linalg.inv(c2w)
            for hand in range(2):
                smooth_camera = world_to_camera @ smooth_world[hand]
                safe[offset, hand * 3:hand * 3 + 3] = (
                    smooth_camera[:3, 3] - mean
                ) / std
                safe[offset, 6 + hand * 6:12 + hand * 6] = rotmat_to_o6d(
                    smooth_camera[:3, :3]
                )
            safe_world_sequence.append(smooth_world[:, :3, 3])
            raw_camera = old_prefix[replan, offset, :6].reshape(2, 3) * std + mean
            raw_world_sequence.append(
                np.einsum("ij,hj->hi", c2w[:3, :3], raw_camera) + c2w[:3, 3]
            )
        current_q = safe[-1, 18:62].reshape(2, 22)
        safe_prefixes.append(safe)
        raw_limit_ratios.append(float(diagnostic["raw_joint_limit_ratio"]))
        q_clip_ratios.append(float(diagnostic["q_step_clip_ratio"]))
        frame_root = (
            source / "heldout_eval_input/production" / session
            / "09_humanego_adapter/preprocess/all_data" / f"{int(frame):05d}"
        )
        rgb = cv2.imread(str(frame_root / "rgb.png"), cv2.IMREAD_COLOR)
        metadata = json.loads((frame_root / "training_data.json").read_text(encoding="utf-8"))
        intrinsic = np.asarray(metadata.get("metadata", metadata)["k"], dtype=np.float64).reshape(3, 3)
        render_frames.append(render(rgb, safe, mean, std, intrinsic, int(frame)))
    safe_prefix = np.stack(safe_prefixes).astype(np.float32)
    artifact = output / "posthoc_safety_reinterpreted_predictions.npz"
    np.savez_compressed(
        artifact,
        schema_version=np.asarray("exact78-heldout-posthoc-safety-reinterpretation-v1"),
        source_prediction_path=np.asarray(str(prediction.resolve())),
        source_prediction_sha256=np.asarray(sha256(prediction)),
        checkpoint_full_plan_action=plans,
        source_raw_prefix=old_prefix,
        posthoc_safe_executed_prefix=safe_prefix,
        observation_frame=frames, observation_c2w=c2ws,
        joint_lower=lower, joint_upper=upper,
        physical_to_human_side=np.asarray(PHYSICAL_TO_HUMAN_SIDE),
        q_unit=np.asarray("radian"), model_was_not_rerun=np.asarray(True),
        original_causal_feedback_was_not_repaired=np.asarray(True),
    )
    video = output / "posthoc_safety_prefix_diagnostic.mp4"
    temporary = Path(tempfile.mkdtemp(prefix=".frames.", dir=output))
    try:
        for index, image in enumerate(render_frames):
            cv2.imwrite(str(temporary / f"{index:05d}.png"), image)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-framerate", "8", "-i",
            str(temporary / "%05d.png"), "-c:v", "libx264", "-pix_fmt",
            "yuv420p", "-crf", "20", str(video),
        ], check=True)
    finally:
        for path in temporary.glob("*.png"):
            path.unlink()
        temporary.rmdir()
    raw_world = np.asarray(raw_world_sequence)
    safe_world = np.asarray(safe_world_sequence)
    raw_steps = np.linalg.norm(np.diff(raw_world, axis=0), axis=-1).reshape(-1)
    safe_steps = np.linalg.norm(np.diff(safe_world, axis=0), axis=-1).reshape(-1)
    q = safe_prefix[..., 18:62].reshape(-1, 2, 22)
    result = {
        "session": session,
        "status": "PASS_POSTHOC_SAFETY_ADAPTER_DIAGNOSTIC_ONLY",
        "source_prediction": {"path": str(prediction.resolve()), "sha256": sha256(prediction)},
        "artifact": {"path": str(artifact.resolve()), "sha256": sha256(artifact)},
        "video": {"path": str(video.resolve()), "sha256": sha256(video)},
        "metrics": {
            "raw_prefix_joint_limit_ratio_mean": float(np.mean(raw_limit_ratios)),
            "safe_prefix_joint_limit_ratio": float(np.mean((q < lower) | (q > upper))),
            "q_step_clip_ratio_mean": float(np.mean(q_clip_ratios)),
            "raw_world_wrist_step_p95_m": float(np.percentile(raw_steps, 95)),
            "raw_world_wrist_step_max_m": float(np.max(raw_steps)),
            "safe_world_wrist_step_p95_m": float(np.percentile(safe_steps, 95)),
            "safe_world_wrist_step_max_m": float(np.max(safe_steps)),
        },
        "claim_limit": (
            "Posthoc CPU reinterpretation only. The checkpoint was not rerun, "
            "and the old inference's wrong bootstrap/causal feedback was not repaired."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"no-clobber output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((run_root / "dataset_stats.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["pos"]["mean"], dtype=np.float64)
    std = np.asarray(stats["pos"]["std"], dtype=np.float64)
    lower, upper = limits(run_root)
    sessions = [
        process_session(path, args.output / path.name, mean, std, lower, upper)
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / "heldout_checkpoint_predictions.npz").is_file()
    ]
    if len(sessions) != 5:
        raise RuntimeError(f"expected five source sessions, got {len(sessions)}")
    result = {
        "schema_version": "exact78-heldout-posthoc-safety-reinterpretation-batch-v1",
        "status": "PASS_FIVE_POSTHOC_DIAGNOSTICS_NOT_NEW_INFERENCE",
        "sessions": sessions,
        "model_was_not_rerun": True,
        "source_predictions_modified": False,
        "claim_limit": "Diagnostic adapter comparison only; not repaired heldout inference.",
    }
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
