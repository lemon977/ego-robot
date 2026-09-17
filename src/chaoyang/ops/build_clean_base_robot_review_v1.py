#!/usr/bin/env python3
"""Build a camera-view review by placing existing Robot pixels on Clean RGB.

Legacy Visual-Aux bundles rendered Robot on the Raw branch, so visible human
pixels remained between Robot links.  This review-only exporter recovers the
pixels changed by that renderer and places them on the verified Clean master.
It does not solve object/Robot occlusion and grants no training authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def checked_json_reference(reference: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(reference["path"]).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"input reference drift: {path}")
    return path, json.loads(path.read_text())


def resolve_clean_master(clean_result_path: Path, clean_result: dict[str, Any]) -> Path:
    candidates = clean_result.get("artifacts", {})
    for key in ("clean_synthetic_master", "clean_master", "master_video", "clean_video"):
        value = candidates.get(key)
        if not isinstance(value, dict) or "path" not in value:
            continue
        path = Path(value["path"])
        if not path.is_absolute():
            path = clean_result_path.parent / path
        path = path.resolve(strict=True)
        if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
            raise RuntimeError(f"Clean master reference drift: {path}")
        return path
    raise RuntimeError("Clean RESULT has no recognized master video")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    manifest_path = args.bundle_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    output = args.output_mp4.resolve()
    result_path = args.result.resolve()
    if output.exists() or output.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise RuntimeError("fresh output and RESULT paths required")
    session = str(manifest["session_id"])
    frame_count = int(manifest["frame_count"])
    bundle_root = manifest_path.parent
    raw_root = bundle_root / "human_raw_rgb"
    robot_root = bundle_root / "robotized_rgb"
    raw_frames = sorted(raw_root.glob("*.png"))
    robot_frames = sorted(robot_root.glob("*.png"))
    if len(raw_frames) != frame_count or len(robot_frames) != frame_count:
        raise RuntimeError("bundle PNG frame count drift")
    clean_result_path, clean_result = checked_json_reference(manifest["inputs"]["clean_result"])
    clean_master = resolve_clean_master(clean_result_path, clean_result)

    capture = cv2.VideoCapture(str(clean_master))
    if not capture.isOpened():
        raise RuntimeError("cannot open Clean master")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if abs(fps - 30.0) > 0.05:
        raise RuntimeError(f"30 FPS contract failed: {fps}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    command = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{args.width}x{args.height}", "-r", "30", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(temporary),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    changed_pixels = 0
    total_pixels = 0
    try:
        for index, (raw_path, robot_path) in enumerate(zip(raw_frames, robot_frames)):
            ok, clean = capture.read()
            if not ok:
                raise RuntimeError(f"Clean master ended at frame {index}")
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            robot = cv2.imread(str(robot_path), cv2.IMREAD_COLOR)
            if raw is None or robot is None or raw.shape != robot.shape:
                raise RuntimeError(f"bundle frame decode/shape failed at {index}")
            clean_small = cv2.resize(clean, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
            changed = np.max(cv2.absdiff(raw, robot), axis=2) > 1
            composite = clean_small.copy()
            composite[changed] = robot[changed]
            changed_pixels += int(changed.sum())
            total_pixels += int(changed.size)
            frame = cv2.resize(composite, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
        ok, _ = capture.read()
        if ok:
            raise RuntimeError("Clean master has more frames than the bundle")
        assert encoder.stdin is not None
        encoder.stdin.close()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg encoder failed: {returncode}")
    except Exception:
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json",
        str(temporary),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if (int(stream["width"]), int(stream["height"]), stream["r_frame_rate"], int(stream["nb_read_frames"])) != (
        args.width, args.height, "30/1", frame_count
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"published video probe failed: {stream}")
    os.replace(temporary, output)
    result = {
        "schema_version": "clean-base-robot-camera-review-v1",
        "status": "PASS_DEVELOPMENT_REVIEW_EXPORT",
        "session_id": session,
        "frame_count": frame_count,
        "fps": 30,
        "geometry": [args.width, args.height],
        "method": "CLEAN_MASTER_BASE_PLUS_PIXELS_CHANGED_BY_LEGACY_ROBOT_RENDERER",
        "inputs": {
            "bundle_manifest": artifact(manifest_path),
            "clean_result": artifact(clean_result_path),
            "clean_master": artifact(clean_master),
        },
        "metrics": {
            "robot_renderer_changed_pixels": changed_pixels,
            "robot_renderer_changed_ratio": changed_pixels / total_pixels if total_pixels else 0.0,
        },
        "output": artifact(output),
        "claim_limit": "Camera-view development review with Clean as the base image. Object/Robot visibility ordering is not authorized; this is not contact truth, Robot action ground truth or deployment authority.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "session_id": session,
                      "frame_count": frame_count, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
