#!/usr/bin/env python3
"""Publish a watermarked, full-length C-grade Robot development review.

Never changes the original rendered video or its attempt receipt.  The new
video is a visual diagnostic only and is not Robot/Visual Aux authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def probe(path: Path) -> dict:
    done = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=120)
    return json.loads(done.stdout)["streams"][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_result.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("session_id") != "play_cards_0901_001":
        raise RuntimeError("wrong session")
    if source.get("terminal_status") != "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO":
        raise RuntimeError("source is not a sealed C-grade full-video attempt")
    if source.get("training_eligible") is not False or source.get("authority_promoted") is not False:
        raise RuntimeError("claim boundary mismatch")
    source_video = Path(source["full_video"]["path"]).resolve(strict=True)
    if ref(source_video)["sha256"] != source["full_video"]["sha256"]:
        raise RuntimeError("source video SHA mismatch")
    source_probe = probe(source_video)
    if int(source_probe["nb_read_frames"]) != 645:
        raise RuntimeError("source video is not full 645 frames")
    arm_path = Path(source["arm_result"]["path"]).resolve(strict=True)
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    if arm.get("status") != "HOLD_NUMERIC_CANARY":
        raise RuntimeError("arm C evidence missing")
    if not FONT.is_file():
        raise RuntimeError("fixed watermark font missing")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_video = output_dir / "play_cards_0901_001_ROBOT_FAILED_QUALITY_C_OFFLINE_VISUAL_FULLSESSION.mp4"
    filter_chain = (
        "drawbox=x=0:y=0:w=iw:h=56:color=red@0.92:t=fill,"
        f"drawtext=fontfile={FONT}:text='FAILED_QUALITY_C  |  OFFLINE_VISUAL  |  NOT TRAINING':"
        "fontcolor=white:fontsize=25:x=20:y=13"
    )
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(source_video),
        "-vf", filter_chain, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", str(output_video),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise RuntimeError(f"watermark encode failed: {done.stderr[-500:]}")
    output_probe = probe(output_video)
    if int(output_probe["nb_read_frames"]) != 645 or output_probe["avg_frame_rate"] != source_probe["avg_frame_rate"]:
        raise RuntimeError("watermarked video frame/fps identity mismatch")
    decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(output_video),
        "-f", "null", "-"], capture_output=True, text=True, timeout=300)
    if decoded.returncode != 0:
        raise RuntimeError(f"watermarked video full decode failed: {decoded.stderr[-500:]}")
    result = {
        "schema_version": "rc1-pose-only-c-review-publication-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": "play_cards_0901_001", "status": "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO",
        "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "authority_promoted": False, "original_attempt": ref(source_path),
        "original_video": ref(source_video), "watermarked_video": ref(output_video),
        "frame_count": 645, "fps": output_probe["avg_frame_rate"], "decode": "PASS_FFMPEG_XERROR",
        "arm_failed_rows": arm["metrics"]["failed_rows"],
        "arm_observed_rows": arm["metrics"]["observed_rows"],
        "arm_position_mm_max": arm["metrics"]["position_mm_max"],
        "arm_rotation_deg_max": arm["metrics"]["rotation_deg_max"],
        "watermark_command": command,
        "publisher_code": ref(Path(__file__)), "font": ref(FONT),
        "claim_limit": "Full C-grade visual diagnosis only. No hard-geometry pass, contact truth, causal-training qualification, control truth or deployment authorization.",
    }
    (output_dir / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "video": result["watermarked_video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
