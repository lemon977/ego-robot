#!/usr/bin/env python3
"""Pair frozen old SAM3.1 tracking and native semantic-full diagnostic videos.

The native source video came from a cleanup-failed attempt and therefore this
comparison is explicitly visual-only, not a completed Mask result.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def video_info(path: Path) -> dict[str, int]:
    response = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    row = json.loads(response.stdout)["streams"][0]
    return {key: int(row[key]) for key in ("width", "height", "nb_frames")} | {"r_frame_rate": row["r_frame_rate"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    legacy = args.legacy.resolve(strict=True)
    native = args.native.resolve(strict=True)
    old_info, native_info = video_info(legacy), video_info(native)
    if old_info != {"width": 1920, "height": 480, "nb_frames": 16, "r_frame_rate": "4/1"}:
        raise ValueError(f"unexpected frozen legacy video contract: {old_info}")
    if native_info != {"width": 1280, "height": 480, "nb_frames": 16, "r_frame_rate": "4/1"}:
        raise ValueError(f"unexpected frozen native video contract: {native_info}")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    video = output / "Poker015_SAM31_旧StableTracking_vs_原生SemanticFull_同帧对照.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(legacy), "-i", str(native),
        "-filter_complex", "[0:v]crop=1280:480:0:0[old];[1:v]crop=640:480:640:0[new];[old][new]hstack=inputs=2[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    info = video_info(video)
    if info != {"width": 1920, "height": 480, "nb_frames": 16, "r_frame_rate": "4/1"}:
        raise RuntimeError(f"paired video closure mismatch: {info}")
    result = {
        "schema_version": "chaoyang-sam31-route-visual-comparison-v1",
        "created_at": now_iso(),
        "status": "PASSED_VISUAL_DIAGNOSTIC_ONLY",
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Native semantic video originated from cleanup-failed attempt_0002; visual presence comparison only, no Mask quality/accuracy or source-code root-cause claim.",
        "inputs": {"legacy": artifact_ref(legacy), "native_partial_attempt": artifact_ref(native)},
        "video": artifact_ref(video),
        "video_info": info,
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "status": result["status"], "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
        "result": artifact_ref(output / "RESULT.json"),
    })
    print(json.dumps({"status": result["status"], "video": result["video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
