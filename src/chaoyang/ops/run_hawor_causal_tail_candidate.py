#!/usr/bin/env python3
"""Run a bounded camera-space causal-tail HaWoR candidate on one session.

The model sees only frames up to each target and publishes the final element
of a left-padded 16-frame tail window.  Existing full-sequence c2w is retained
only for a separately labelled offline diagnostic; it does not make the
candidate causal-world or Robot-training eligible.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
HAWOR_ROOT = ROOT / "vendor/HaWoR"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HAWOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HAWOR_ROOT))

from chaoyang.pipeline.hawor_causal_tail_inference import inference_causal_tail
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops import run_hawor_bounded_parameter_successor as bounded


CHECKPOINT = HAWOR_ROOT / "weights/hawor/checkpoints/hawor.ckpt"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _load_npz(path: Path, frames: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        result = {key: np.asarray(value[key]) for key in value.files}
    for key, value in list(result.items()):
        if value.ndim >= 1 and (
            (value.ndim >= 2 and value.shape[1] == result["observed"].shape[1])
            or (value.shape[0] == result["observed"].shape[1] and key in {"c2w", "intrinsics", "original_frame_indices"})
        ):
            if value.ndim >= 2 and value.shape[1] == result["observed"].shape[1]:
                result[key] = value[:, :frames].copy()
            else:
                result[key] = value[:frames].copy()
    return result


def _extract_frames(video: Path, output: Path, count: int) -> list[Path]:
    output.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    paths = []
    for frame in range(count):
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"source video ended at {frame}/{count}")
        path = output / f"{frame:05d}.png"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise RuntimeError(f"failed to write {path}")
        paths.append(path)
    capture.release()
    return paths


def _mirror_left(rotations: np.ndarray) -> np.ndarray:
    shape = rotations.shape
    vectors = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec()
    vectors[:, 1:] *= -1
    return Rotation.from_rotvec(vectors).as_matrix().reshape(shape).astype(np.float32)


def _parameters(raw: dict[str, np.ndarray], paths: list[Path], model: Any, batch_windows: int) -> dict[str, np.ndarray]:
    frames = raw["observed"].shape[1]
    result = {
        "root_translation_camera": np.full((2, frames, 3), np.nan, np.float32),
        "root_orient_camera": np.full((2, frames, 3, 3), np.nan, np.float32),
        "hand_pose_rotmat": np.full((2, frames, 15, 3, 3), np.nan, np.float32),
        "betas": np.full((2, frames, 10), np.nan, np.float32),
    }
    focal = float(np.median(raw["intrinsics"][:, 0, 0]))
    center = [float(np.median(raw["intrinsics"][:, 0, 2])), float(np.median(raw["intrinsics"][:, 1, 2]))]
    for side in range(2):
        for start, end in bounded.contiguous_true_segments(raw["observed"][side]):
            segment_paths = np.asarray([str(path) for path in paths[start:end]])
            boxes = np.asarray(raw["detector_boxes_xyxy"][side, start:end], dtype=np.float32)
            prediction = inference_causal_tail(
                model, segment_paths, boxes, focal, center, device="cuda",
                do_flip=(side == 0), batch_windows=batch_windows,
            )
            rotations = prediction["pred_rotmat"].numpy()
            roots = rotations[:, 0]
            poses = rotations[:, 1:]
            if side == 0:
                roots = _mirror_left(roots)
                poses = _mirror_left(poses)
            translation = prediction["pred_trans"].numpy()
            if translation.ndim == 3 and translation.shape[1] == 1:
                translation = translation[:, 0]
            result["root_translation_camera"][side, start:end] = translation.astype(np.float32)
            result["root_orient_camera"][side, start:end] = roots.astype(np.float32)
            result["hand_pose_rotmat"][side, start:end] = poses.astype(np.float32)
            result["betas"][side, start:end] = prediction["pred_shape"].numpy().astype(np.float32)
    return result


def _stats(raw: dict[str, np.ndarray], candidate: dict[str, np.ndarray], joints: dict[str, np.ndarray]) -> dict[str, Any]:
    sides = {}
    for side, name in enumerate(("left", "right")):
        observed = raw["observed"][side]
        camera_error = np.linalg.norm(
            joints["joints_3d_camera"][side, observed] - raw["joints_3d_camera"][side, observed], axis=-1,
        ) * 1000.0
        reprojection = np.linalg.norm(joints["joints_2d"][side, observed] - raw["joints_2d"][side, observed], axis=-1)
        raw_step, raw_acc = bounded.temporal_values(raw["joints_3d_camera"][side], observed)
        new_step, new_acc = bounded.temporal_values(joints["joints_3d_camera"][side], observed)
        wrist = joints["joints_3d_camera"][side, observed, 0]
        raw_wrist = raw["joints_3d_camera"][side, observed, 0]
        def span(value: np.ndarray) -> float:
            return float(np.linalg.norm(np.quantile(value, 0.95, axis=0) - np.quantile(value, 0.05, axis=0)) * 1000.0)
        sides[name] = {
            "observed_frames": int(observed.sum()),
            "camera_joint_difference_vs_old_disjoint_p95_mm": float(np.quantile(camera_error, 0.95)),
            "reprojection_difference_vs_old_disjoint_p95_px": float(np.quantile(reprojection, 0.95)),
            "old_wrist_step_p95_mm": bounded.quantile(raw_step, 0.95),
            "causal_wrist_step_p95_mm": bounded.quantile(new_step, 0.95),
            "old_all_joint_acceleration_p95_mm": bounded.quantile(raw_acc, 0.95),
            "causal_all_joint_acceleration_p95_mm": bounded.quantile(new_acc, 0.95),
            "old_wrist_span_mm": span(raw_wrist),
            "causal_wrist_span_mm": span(wrist),
        }
    rotations = np.concatenate((
        candidate["root_orient_camera"][raw["observed"]].reshape(-1, 3, 3),
        candidate["hand_pose_rotmat"][np.repeat(raw["observed"][..., None], 15, axis=2)].reshape(-1, 3, 3),
    ))
    return {
        "sides": sides,
        "rotation_orthogonality_max": float(np.max(np.abs(np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3)))),
        "rotation_determinant_min": float(np.min(np.linalg.det(rotations))),
    }


def _render(video: Path, output: Path, raw: dict[str, np.ndarray], joints: dict[str, np.ndarray]) -> None:
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(3)), int(capture.get(4))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = raw["observed"].shape[1]
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 19)
    for frame_id in range(frames):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"render source ended at {frame_id}/{frames}")
        bounded.draw_skeleton(frame, raw["joints_2d"][:, frame_id], ((90, 90, 40), (90, 40, 90)), 1)
        bounded.draw_skeleton(frame, joints["joints_2d"][:, frame_id], ((255, 255, 0), (255, 0, 255)), 3)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil); draw.rectangle((0, 0, width, 76), fill=(0, 0, 0))
        draw.text((10, 8), f"frame {frame_id+1}/{frames} | 暗=旧不重叠16帧 亮=因果尾窗末帧", font=font, fill=(255,255,255))
        draw.text((10, 43), "仅camera-space因果；旧c2w未获因果证明，不授予Robot训练资格", font=font, fill=(255,220,80))
        assert process.stdin is not None
        process.stdin.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).tobytes())
    capture.release()
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("review video encode failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npz", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-windows", type=int, default=2)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    with np.load(args.raw_npz, allow_pickle=False) as value:
        total = int(value["observed"].shape[1])
    frames = min(total, args.max_frames or total)
    raw = _load_npz(args.raw_npz, frames)
    paths = _extract_frames(args.source_video, args.output_root / "input_frames", frames)

    from scripts.scripts_test_video.hawor_video import load_hawor
    model, _ = load_hawor(str(CHECKPOINT))
    model = model.cuda().eval()
    parameters = _parameters(raw, paths, model, args.batch_windows)
    joints = bounded.materialize(raw, parameters)
    metrics = _stats(raw, parameters, joints)
    npz = args.output_root / "HAWOR_CAUSAL_TAIL_CAMERA_SPACE.npz"
    payload = {key: value for key, value in raw.items() if key not in parameters and key not in joints}
    payload.update(parameters); payload.update(joints)
    payload["provenance"] = np.where(raw["observed"], "CAUSAL_TAIL_16_LAST", "MISSING").astype("U24")
    payload["input_mode"] = np.asarray("CAUSAL_CAMERA_SPACE_WORLD_POSE_UNVERIFIED", dtype="U48")
    np.savez_compressed(npz, **payload)
    video = args.output_root / f"{args.session}_HaWoR因果尾窗_vs_旧16帧全片.mp4"
    _render(args.source_video, video, raw, joints)
    result = {
        "schema_version": "chaoyang-hawor-causal-tail-candidate-v1",
        "created_at": now_iso(),
        "task_id": "research_hawor_causal_tail_candidate",
        "status": "PASSED_CAMERA_SPACE_CAUSAL_CANDIDATE",
        "task": args.task,
        "session": args.session,
        "frame_count": frames,
        "metrics": metrics,
        "causal_contract": {
            "model_window_frames": 16,
            "per_target_input": "max(segment_start,t-15)..t",
            "short_prefix": "left-pad segment first frame",
            "published_output": "window last frame only",
            "future_frame_consumed": False,
        },
        "world_coordinate_gate": {
            "status": "BLOCKED_PREREQ_CAUSAL_CAMERA_POSE",
            "reason": "Retained c2w originates from the prior offline world-consistency product and has not passed suffix-invariance proof.",
        },
        "outputs": {"candidate_npz": artifact_ref(npz), "review_video": artifact_ref(video)},
        "inputs": {
            "raw_npz": artifact_ref(args.raw_npz),
            "source_video": artifact_ref(args.source_video),
            "checkpoint": artifact_ref(CHECKPOINT),
            "causal_adapter": artifact_ref(ROOT / "src/chaoyang/pipeline/hawor_causal_tail_inference.py"),
            "producer": artifact_ref(Path(__file__).resolve()),
        },
        "training_eligible": False,
        "robot_causal_eligible": False,
        "authority_promoted": False,
        "control_ground_truth": False,
        "claim_limit": "Camera-space HaWoR causality candidate only; c2w, Robot, external hand accuracy, contact and physical deployment remain unsupported.",
        "host": socket.gethostname(),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_root / "RESULT.json", result)
    atomic_json(args.output_root / "METRICS.json", metrics)
    atomic_json(args.output_root / "RUN_RECEIPT.json", {
        "task_id": result["task_id"], "status": result["status"], "created_at": result["created_at"],
        "producer": result["inputs"]["producer"],
    })
    print(json.dumps({"status": result["status"], "result": artifact_ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
