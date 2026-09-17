#!/usr/bin/env python3
"""Show raw frames, indexed PICO evidence and the blocked causal chain.

No HaWoR/Robot output is synthesized.  The archived c2w path is visibly
labelled OFFLINE / NOT CAUSAL-PROVEN throughout the video.
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
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso

FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def draw_text(panel: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 22)
    for index, (line, color) in enumerate(lines):
        painter.text((20, 16 + index * 38), line, fill=color, font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(raw_root: Path, audit_path: Path, output: Path, max_frames: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["camera_pose_producer_causality"] != "UNKNOWN_VERIFICATION_REQUIRED" or audit["robot_causal_eligible"]:
        raise ValueError("diagnostic is only for blocked upstream causal evidence")
    rows = audit["rows"][:max_frames]
    if not rows:
        raise ValueError("no diagnostic frames")
    manifest = json.loads((raw_root / "clip_manifest.json").read_text(encoding="utf-8"))
    if manifest["clip_name"] != audit["session"]:
        raise ValueError("audit and raw session mismatch")
    target = output / f"{audit['session']}_PICO_c2w_HaWoR_Robot_因果阻塞诊断.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "1280x480", "-r", "4", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for row in rows:
            frame = row["frame_id"]
            raw_path = raw_root / "preprocess/all_data" / f"{frame:05d}" / "rgb.png"
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"missing raw RGB: {raw_path}")
            raw = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            panel = np.full((480, 640, 3), (29, 27, 23), dtype=np.uint8)
            state = row.get("tracker_state", "UNKNOWN")
            status_color = (110, 220, 130) if state == "accurate" else (255, 120, 100)
            lines = [
                (f"{audit['session']}  帧 {frame:03d}", (255, 255, 255)),
                ("原始画面 ← 左侧；右侧是证据状态", (185, 205, 220)),
                (f"TrackerState: {state}; Head.status: {row.get('head_status')}", status_color),
                (f"metadata.ts = 索引 tracker.ts: {row['metadata_timestamp_equals_indexed_tracking']}", (210, 210, 210)),
                (f"名义30FPS网格偏移: {row['stored_sync_error_ms']:+.3f} ms", (210, 210, 210)),
                ("旧 c2w：仅离线元数据，生产者因果性未证", (255, 205, 110)),
                ("HaWoR 因果尾窗：未执行 / 无合格 c2w", (255, 205, 110)),
                ("Robot 因果解算：BLOCKED，未生成画面", (255, 120, 100)),
                ("control_ground_truth=false", (255, 120, 100)),
                ("本片不显示任何伪造的下游结果", (180, 180, 180)),
            ]
            panel = draw_text(panel, lines)
            assert encoder.stdin is not None
            encoder.stdin.write(np.concatenate([raw, panel], axis=1).tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg diagnostic encoding failed")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,width,height", "-of", "json", str(target),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_frames"]) != len(rows) or (stream["width"], stream["height"]) != (1280, 480):
        raise RuntimeError("diagnostic video frame/shape closure failed")
    result = {
        "schema_version": "chaoyang-rc1-pico-c2w-blocked-diagnostic-video-v1",
        "created_at": now_iso(),
        "session": audit["session"],
        "status": "PASSED_DIAGNOSTIC_VIDEO_BLOCKED_UPSTREAM",
        "frames": len(rows),
        "fps": 4,
        "training_eligible": False,
        "robot_generated": False,
        "claim_limit": "Original RGB and indexed tracker status only; archived c2w/HaWoR producer causality is unresolved and no causal Robot image is produced.",
        "inputs": {
            "raw_manifest": artifact_ref(raw_root / "clip_manifest.json"),
            "corrected_audit": artifact_ref(audit_path),
        },
        "video": artifact_ref(target),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "status": result["status"], "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
        "result": artifact_ref(output / "RESULT.json"),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=21)
    args = parser.parse_args()
    result = render(args.raw_root.resolve(strict=True), args.audit_result.resolve(strict=True), args.output_root.resolve(), args.max_frames)
    print(json.dumps({"status": result["status"], "session": result["session"], "frames": result["frames"], "video": result["video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
