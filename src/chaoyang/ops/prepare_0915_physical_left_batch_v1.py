#!/usr/bin/env python3
"""Prepare all 0915 physical-left rectified mono sessions without PICO input."""

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
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SESSIONS = 220
EXPECTED_FRAMES = 58_686


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def ref_as(path: Path, published_path: Path) -> dict[str, Any]:
    """Hash a staging file while recording its post-rename immutable path."""
    path = path.resolve(strict=True)
    return {"path": str(published_path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def enumerate_sessions(root: Path) -> list[tuple[str, Path]]:
    counts = {"playing_cards": 120, "potato_chips": 100}
    rows: list[tuple[str, Path]] = []
    for task, count in counts.items():
        sessions = sorted(path for path in (root / "cleaned" / task).iterdir()
                          if path.is_dir())
        if len(sessions) != count:
            raise RuntimeError(f"{task} session count {len(sessions)} != {count}")
        rows.extend((task, session) for session in sessions)
    if len(rows) != EXPECTED_SESSIONS:
        raise RuntimeError("0915 denominator drift")
    return rows


def load_stereo_module() -> Any:
    script_root = ROOT / "vendor/FoundationStereo/scripts"
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import pico_stereo_depth  # type: ignore
    return pico_stereo_depth


def physical_left_c2w(selected_c2w: np.ndarray,
                      camera: dict[str, Any]) -> np.ndarray:
    if camera.get("extrinsic_convention") != "head_to_camera_4x4_row_major":
        raise RuntimeError("unexpected camera extrinsic convention")
    left = np.asarray(camera["extrinsics"]["left"], np.float64)
    right = np.asarray(camera["extrinsics"]["right"], np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise RuntimeError("camera extrinsics must be 4x4")
    return selected_c2w @ right @ np.linalg.inv(left)


def full_decode(path: Path, expected_frames: int) -> dict[str, Any]:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    stream = json.loads(completed.stdout)["streams"][0]
    frames = int(stream["nb_read_frames"])
    if frames != expected_frames:
        raise RuntimeError(f"decode frame count {frames} != {expected_frames}")
    decoded = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise RuntimeError(decoded.stderr[-2000:])
    return {"status": "PASS_FULL_DECODE", "frames": frames,
            "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": stream["r_frame_rate"]}


def prepare_session(task: str, session: Path, target: Path) -> dict[str, Any]:
    camera_path = session / "camera_params.json"
    conversion_path = session / "CONVERSION_RESULT.json"
    stereo_files = sorted(
        (session / "source_stereo").glob("CameraRecord_*_stereo.mp4")
    )
    rows = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    if not camera_path.is_file() or not conversion_path.is_file() or len(stereo_files) != 1:
        raise RuntimeError("same-session processed inputs are incomplete")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    expected = conversion.get("validation", {}).get("frame_count")
    if not isinstance(expected, int) or expected != len(rows):
        raise RuntimeError("processed metadata frame identity mismatch")

    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    stereo_module = load_stereo_module()
    eyes, eye_width, eye_height, _baseline = stereo_module.load_camera_params(
        camera_path
    )
    if stereo_module.load_source_indices(camera_path) != (1, 0):
        raise RuntimeError("physical eye/sourceIndex mapping drift")
    width, height, fps = 1280, 960, 30.0
    intrinsics = stereo_module.virtual_intrinsics(width, height, 90.0)
    left_map = stereo_module.make_map(
        eyes[0], np.eye(3), width, height, intrinsics
    )

    staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if staging.exists():
        raise RuntimeError(f"staging conflict: {staging}")
    staging.mkdir(parents=True)
    try:
        mono = staging / "leftmono/LEFT_MONO_RECTIFIED.mp4"
        mono.parent.mkdir()
        capture = cv2.VideoCapture(str(stereo_files[0]))
        writer = cv2.VideoWriter(
            str(mono), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("video reader/writer failed to open")
        frame_count = 0
        try:
            while True:
                ok, stereo = capture.read()
                if not ok:
                    break
                if stereo.shape[:2] != (eye_height, eye_width * 2):
                    raise RuntimeError(f"unexpected SBS geometry: {stereo.shape}")
                raw_left = stereo[:, eye_width:2 * eye_width]
                rectified = cv2.remap(
                    raw_left, left_map[0], left_map[1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                writer.write(rectified)
                frame_count += 1
        finally:
            capture.release()
            writer.release()
        if frame_count != expected:
            raise RuntimeError(f"SBS frames {frame_count} != {expected}")
        decode = full_decode(mono, expected)

        adapter = staging / "adapter_session"
        frame_root = adapter / "preprocess/all_data"
        frame_root.mkdir(parents=True)
        os.symlink(
            os.path.relpath(mono, adapter),
            adapter / "CameraRecord_play_cards_0910_001.mp4",
        )
        for frame_index, path in enumerate(rows):
            # Explicit allowlist: consume metadata.c2w only.  entities.hands,
            # controller sidecars, trackingData and PICO manifests are ignored.
            payload = json.loads(path.read_text(encoding="utf-8"))
            selected_c2w = np.asarray(payload["metadata"]["c2w"], np.float64)
            left_c2w = physical_left_c2w(selected_c2w, camera)
            frame_dir = frame_root / f"{frame_index:05d}"
            frame_dir.mkdir()
            atomic_json(frame_dir / "training_data.json", {
                "metadata": {
                    "c2w": left_c2w.tolist(),
                    "k": intrinsics.tolist(),
                    "fps": fps,
                    "frame_index": frame_index,
                    "camera_eye": "physical_left",
                    "camera_source_index": 1,
                    "rectified": True,
                    "image_domain": "EQUIDIS62_TO_PINHOLE_1280X960_FOV90",
                }
            })
        result = {
            "schema_version": "0915-physical-left-prepared-session-v1",
            "status": "PASS",
            "task": task,
            "session_id": session.name,
            "frame_count": frame_count,
            "source": {
                "session": str(session),
                "stereo": ref(stereo_files[0]),
                "camera": ref(camera_path),
                "conversion": ref(conversion_path),
            },
            "output": {
                "video_relative": "leftmono/LEFT_MONO_RECTIFIED.mp4",
                "adapter_relative": "adapter_session",
                "video": ref_as(
                    mono, target / "leftmono/LEFT_MONO_RECTIFIED.mp4"
                ),
                "decode": decode,
            },
            "input_policy": {
                "processed_only": True,
                "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
                "controller_pose": "NOT_CONSUMED",
                "trackingData_hand": "NOT_CONSUMED",
            },
        }
        atomic_json(staging / "RESULT.json", result)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return json.loads((target / "RESULT.json").read_text(encoding="utf-8"))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sessions = enumerate_sessions(dataset)
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, (task, session) in enumerate(sessions, 1):
        target = output / "sessions" / task / session.name
        prior = target / "RESULT.json"
        if prior.is_file():
            result = json.loads(prior.read_text(encoding="utf-8"))
            if result.get("status") != "PASS":
                raise RuntimeError(f"non-pass prepared target exists: {target}")
            result = {**result, "resume_status": "RESUMED"}
        elif target.exists() or target.is_symlink():
            raise RuntimeError(f"unreceipted prepared target exists: {target}")
        else:
            result = prepare_session(task, session, target)
        results.append(result)
        atomic_json(output / "STATE.json", {
            "schema_version": "0915-physical-left-preparation-state-v1",
            "state": "RUNNING", "session_count": len(sessions),
            "completed": index,
            "frames": sum(row["frame_count"] for row in results),
            "updated_unix": time.time(),
            "results": results,
        })
        print(json.dumps({"completed": index, "total": len(sessions),
                          "last": session.name}, sort_keys=True), flush=True)
    frames = sum(row["frame_count"] for row in results)
    status = "PASS" if len(results) == EXPECTED_SESSIONS and frames == EXPECTED_FRAMES else "FAILED"
    manifest = {
        "schema_version": "0915-physical-left-prepared-batch-v1",
        "status": status,
        "dataset_root": str(dataset),
        "session_count": len(results),
        "frame_count": frames,
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "started_unix": started,
        "finished_unix": time.time(),
        "results": results,
    }
    atomic_json(output / "BATCH_RESULT.json", manifest)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
