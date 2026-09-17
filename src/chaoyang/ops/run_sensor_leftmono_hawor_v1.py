#!/usr/bin/env python3
"""Run pinned HaWoR on a true physical-left PICO mono view.

The source SBS video stays immutable.  The physical left half is selected by
camera_params.left.sourceIndex and mapped from equiDis62 into a 1280x960
90-degree pinhole view before inference.  Acquisition c2w is transferred from
the published physical-right view to the physical-left camera using the
same-session head-to-camera extrinsics.
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

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
HAWOR_WRAPPER = PROJECT / "src/chaoyang/ops/hawor_python.sh"
UPSTREAM_RUNNER = PROJECT / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"
HAWOR_CHECKPOINT = PROJECT / "assets/models/vendor/hawor/hawor/checkpoints/hawor.ckpt"
HAWOR_DETECTOR = PROJECT / "assets/models/vendor/hawor/external/detector.pt"
EXPECTED_HAWOR_SHA = "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2"
EXPECTED_DETECTOR_SHA = "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
BONES = tuple((chain[i], chain[i + 1]) for chain in CHAINS for i in range(4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def full_decode(path: Path, expected_frames: int) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr[-1000:]}")
    stream = json.loads(completed.stdout)["streams"][0]
    frames = int(stream["nb_read_frames"])
    if frames != expected_frames:
        raise RuntimeError(f"frame mismatch for {path}: {frames} != {expected_frames}")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        text=True, capture_output=True, check=False,
    )
    if decode.returncode:
        raise RuntimeError(f"full decode failed for {path}: {decode.stderr[-1000:]}")
    return {
        "status": "PASS_FULL_DECODE", "frames": frames,
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
    }


def load_stereo_module():
    script_dir = PROJECT / "vendor/FoundationStereo/scripts"
    sys.path.insert(0, str(script_dir))
    import pico_stereo_depth  # type: ignore
    return pico_stereo_depth


def physical_left_c2w(right_c2w: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    if camera.get("extrinsic_convention") != "head_to_camera_4x4_row_major":
        raise RuntimeError("unexpected normalized extrinsic convention")
    left = np.asarray(camera["extrinsics"]["left"], dtype=np.float64)
    right = np.asarray(camera["extrinsics"]["right"], dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise RuntimeError("camera extrinsics are not 4x4")
    # W<-Cl = W<-Cr @ Cr<-H @ H<-Cl.
    return right_c2w @ right @ np.linalg.inv(left)


def prepare_leftmono(session: Path, output: Path) -> tuple[Path, np.ndarray, list[np.ndarray], dict[str, Any]]:
    stereo_mod = load_stereo_module()
    camera_path = session / "camera_params.json"
    stereo_files = sorted((session / "source_stereo").glob("CameraRecord_*_stereo.mp4"))
    if not camera_path.is_file() or len(stereo_files) != 1:
        raise RuntimeError("same-session stereo/camera inputs are incomplete")
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    eyes, eye_width, eye_height, _baseline = stereo_mod.load_camera_params(camera_path)
    source_indices = stereo_mod.load_source_indices(camera_path)
    if source_indices != (1, 0):
        raise RuntimeError(f"expected physical (left,right) source indices (1,0), got {source_indices}")
    width, height, fps = 1280, 960, 30.0
    intrinsics = stereo_mod.virtual_intrinsics(width, height, 90.0)
    left_map = stereo_mod.make_map(eyes[0], np.eye(3), width, height, intrinsics)
    destination = output / "leftmono/LEFT_MONO_RECTIFIED.mp4"
    destination.parent.mkdir(parents=True)
    capture = cv2.VideoCapture(str(stereo_files[0]))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("left-mono video reader/writer failed to open")
    frame_count = 0
    try:
        while True:
            ok, stereo = capture.read()
            if not ok:
                break
            if stereo.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"unexpected SBS geometry: {stereo.shape}")
            raw_left = stereo[:, eye_width : 2 * eye_width]
            rectified = cv2.remap(
                raw_left, left_map[0], left_map[1], interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            writer.write(rectified)
            frame_count += 1
    finally:
        capture.release()
        writer.release()
    if frame_count != 379:
        raise RuntimeError(f"expected 379 left-mono frames, got {frame_count}")
    decode = full_decode(destination, frame_count)

    source_rows = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    if len(source_rows) != frame_count:
        raise RuntimeError(f"metadata frame count mismatch: {len(source_rows)} != {frame_count}")
    left_c2w: list[np.ndarray] = []
    for path in source_rows:
        row = json.loads(path.read_text(encoding="utf-8"))
        selected_c2w = np.asarray(row["metadata"]["c2w"], dtype=np.float64)
        left_c2w.append(physical_left_c2w(selected_c2w, camera))
    return destination, intrinsics, left_c2w, {
        "source_stereo": ref(stereo_files[0]),
        "camera_params": ref(camera_path),
        "decode": decode,
        "physical_eye": "left",
        "source_index": 1,
        "image_domain": "EQUIDIS62_TO_PINHOLE_1280X960_FOV90",
    }


def prepare_adapter(output: Path, video: Path, intrinsics: np.ndarray, c2w: list[np.ndarray]) -> Path:
    adapter = output / "adapter_session"
    all_data = adapter / "preprocess/all_data"
    all_data.mkdir(parents=True)
    os.symlink(os.path.relpath(video, adapter), adapter / "CameraRecord_play_cards_0910_001.mp4")
    for frame, pose in enumerate(c2w):
        frame_dir = all_data / f"{frame:05d}"
        frame_dir.mkdir()
        atomic_json(frame_dir / "training_data.json", {
            "metadata": {
                "c2w": pose.tolist(), "k": intrinsics.tolist(),
                "fps": 30.0, "frame_index": frame,
                "camera_eye": "physical_left", "rectified": True,
            }
        })
    return adapter


def render_review(video: Path, npz_path: Path, destination: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    observed = np.asarray(data["observed"], dtype=bool)
    joints = np.asarray(data["joints_2d"], dtype=np.float64)
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 960))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("HaWoR review reader/writer failed to open")
    colors = ((255, 180, 20), (20, 50, 255))
    try:
        for frame in range(observed.shape[1]):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"left-mono review source ended at {frame}")
            for side in (0, 1):
                if not observed[side, frame]:
                    continue
                uv = np.rint(joints[side, frame]).astype(np.int32)
                for chain in CHAINS:
                    cv2.polylines(image, [uv[np.asarray(chain)]], False, colors[side], 3, cv2.LINE_AA)
                cv2.circle(image, tuple(uv[0]), 7, colors[side], -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (1279, 66), (0, 0, 0), -1)
            cv2.putText(image, f"0915_001 PHYSICAL LEFT | fresh HaWoR | frame {frame:03d}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .67, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"cyan=left red=right | L={'OBS' if observed[0,frame] else 'MISS'} R={'OBS' if observed[1,frame] else 'MISS'}", (14, 55), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
    finally:
        capture.release()
        writer.release()
    return full_decode(destination, observed.shape[1])


def quality(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    observed = np.asarray(data["observed"], dtype=bool)
    joints_2d = np.asarray(data["joints_2d"], dtype=np.float64)
    joints_3d = np.asarray(data["joints_3d_camera"], dtype=np.float64)
    frame_count = observed.shape[1]
    coverage = observed.mean(axis=1)
    in_frame: list[float] = []
    bone_cv: list[float | None] = []
    for side in (0, 1):
        selected = observed[side]
        uv = joints_2d[side, selected]
        valid = np.isfinite(uv).all(axis=-1) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 960)
        in_frame.append(float(valid.mean()) if valid.size else 0.0)
        cvs = []
        xyz = joints_3d[side, selected]
        for start, end in BONES:
            length = np.linalg.norm(xyz[:, end] - xyz[:, start], axis=1)
            mean = float(np.nanmean(length))
            if mean > 1e-8:
                cvs.append(float(np.nanstd(length) / mean))
        bone_cv.append(float(np.mean(cvs)) if cvs else None)
    return {
        "frame_count": frame_count,
        "observed_frames": {
            "left": int(observed[0].sum()), "right": int(observed[1].sum()),
            "bilateral": int(np.all(observed, axis=0).sum()),
        },
        "observed_fraction": {"left": float(coverage[0]), "right": float(coverage[1])},
        "joint_in_frame_fraction": {"left": in_frame[0], "right": in_frame[1]},
        "mean_bone_length_cv": {"left": bone_cv[0], "right": bone_cv[1]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    session = args.session_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable output required: {output}")
    if sha256(HAWOR_CHECKPOINT) != EXPECTED_HAWOR_SHA or sha256(HAWOR_DETECTOR) != EXPECTED_DETECTOR_SHA:
        raise RuntimeError("HaWoR weight pin drift")
    output.mkdir(parents=True)
    started = time.monotonic()
    leftmono, intrinsics, c2w, mono_manifest = prepare_leftmono(session, output)
    adapter = prepare_adapter(output, leftmono, intrinsics, c2w)
    upstream = output / "upstream_hawor"
    command = [
        str(HAWOR_WRAPPER), str(UPSTREAM_RUNNER),
        "--session-root", str(adapter), "--output-root", str(upstream),
    ]
    completed = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"HaWoR failed rc={completed.returncode}:\n{completed.stderr[-6000:]}\n{completed.stdout[-3000:]}")
    npz_path = upstream / "HAWOR_RAW_MANO21.npz"
    if not npz_path.is_file():
        raise RuntimeError("HaWoR output NPZ is missing")
    review = output / "HAWOR_LEFTMONO_REVIEW.mp4"
    review_decode = render_review(leftmono, npz_path, review)
    metrics = quality(npz_path)
    both_observed = metrics["observed_frames"]["left"] > 0 and metrics["observed_frames"]["right"] > 0
    in_frame_pass = min(metrics["joint_in_frame_fraction"].values()) >= 0.90
    status = "PASS_DEVELOPMENT_HAWOR" if both_observed and in_frame_pass else "FAILED_QUALITY_C"
    result = {
        "schema_version": "sensor-leftmono-hawor-v1",
        "status": status,
        "validity": "VALID_FOR_PINNED_REVISION",
        "session_id": "get_potato_chips_0915_001",
        "model_ran": True,
        "rgb_primary": "PHYSICAL_LEFT_SOURCE_INDEX_1_EQUIDIS62_TO_PINHOLE",
        "tracker_modality": "ABSENT_NOT_CAPTURED",
        "mono_input": {**mono_manifest, "artifact": ref(leftmono)},
        "weights": {"hawor": ref(HAWOR_CHECKPOINT), "detector": ref(HAWOR_DETECTOR)},
        "metrics": metrics,
        "artifacts": {"mano21": ref(npz_path), "review": ref(review)},
        "review_decode": review_decode,
        "wall_seconds": time.monotonic() - started,
        "upstream_stdout_tail": completed.stdout[-3000:],
        "upstream_stderr_tail": completed.stderr[-3000:],
        "claim_limit": "Fresh pinned HaWoR development inference on the rectified physical-left mono view; not external anatomical or metric truth.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
