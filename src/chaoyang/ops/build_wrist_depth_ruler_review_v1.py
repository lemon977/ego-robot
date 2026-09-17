#!/usr/bin/env python3
"""Append an explicit relative-Z ruler to an existing three-wrist review.

The RGB markers in the source review can overlap by construction because a
Stereo surface proxy is placed on the PICO/Controller or HaWoR image ray.  The
ruler makes the signed camera-Z differences visible without pretending that
any source is physical wrist ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "reference": (25, 220, 80),
    "hawor": (40, 155, 255),
    "stereo": (235, 50, 235),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def put_rows(frame: np.ndarray, rows: list[tuple[int, int, str, tuple[int, int, int], int]]) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for x, y, text, bgr, size in rows:
        cache.setdefault(size, font(size))
        draw.text((x, y), text, font=cache[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def draw_ruler(
    panel: np.ndarray,
    y: int,
    title: str,
    values: list[tuple[str, float | None, tuple[int, int, int], str]],
    limit: float,
) -> None:
    left, right = 60, panel.shape[1] - 60
    center = (left + right) // 2
    cv2.line(panel, (left, y), (right, y), (105, 110, 120), 2, cv2.LINE_AA)
    for tick in np.linspace(-limit, limit, 7):
        x = int(round(left + (tick + limit) / (2 * limit) * (right - left)))
        cv2.line(panel, (x, y - 7), (x, y + 7), (120, 125, 135), 1)
        cv2.putText(panel, f"{tick:+.0f}", (x - 22, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
    cv2.line(panel, (center, y - 34), (center, y + 34), (235, 235, 235), 2)
    label_rows = [(45, y - 80, title, (240, 240, 240), 25)]
    offsets = (-30, 0, 30)
    for (name, value, color, marker), dy in zip(values, offsets, strict=True):
        if value is None:
            label_rows.append((45, y + 50 + 27 * (dy // 30 + 1), f"{name}: INVALID", color, 20))
            continue
        clipped = float(np.clip(value, -limit, limit))
        x = int(round(left + (clipped + limit) / (2 * limit) * (right - left)))
        if marker == "circle":
            cv2.circle(panel, (x, y + dy), 9, color, -1, cv2.LINE_AA)
        elif marker == "cross":
            cv2.drawMarker(panel, (x, y + dy), color, cv2.MARKER_TILTED_CROSS, 20, 3, cv2.LINE_AA)
        else:
            points = np.asarray(((x, y + dy - 10), (x + 10, y + dy), (x, y + dy + 10), (x - 10, y + dy)), np.int32)
            cv2.fillConvexPoly(panel, points, color, cv2.LINE_AA)
        label_rows.append((45, y + 50 + 27 * (dy // 30 + 1), f"{name}: {value:+.1f} mm", color, 20))
    panel[:] = put_rows(panel, label_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--reference", choices=("pico", "controller"), required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    video = args.input_video.resolve(strict=True)
    metrics_path = args.metrics_json.resolve(strict=True)
    output = args.output_video.resolve()
    result_path = args.result.resolve()
    if output.exists() or output.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise RuntimeError("fresh/no-clobber output required")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    frames = data["per_frame_delta_xyz_mm"]
    frame_count = int(data["frame_count"])
    fps = float(data["fps"])
    if len(frames) != frame_count:
        raise RuntimeError("metrics frame count mismatch")
    if args.reference == "pico":
        reference_label = "PICO"
        hawor_key, stereo_key = "hawor_minus_pico", "stereo_minus_pico"
    else:
        reference_label = "Controller"
        hawor_key, stereo_key = "hawor_minus_controller", "stereo_minus_controller"

    z_values = []
    for row in frames:
        for side in ("left", "right"):
            for key in (hawor_key, stereo_key):
                value = row[side][key]
                if isinstance(value, list) and len(value) == 3 and finite(value[2]) is not None:
                    z_values.append(abs(float(value[2])))
    robust = float(np.quantile(z_values, 0.99)) if z_values else 150.0
    limit = max(150.0, math.ceil(robust / 50.0) * 50.0)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("cannot decode input video")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width + 640}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(temporary),
    ], stdin=subprocess.PIPE)
    decoded = 0
    try:
        for index, row in enumerate(frames):
            ok, source = capture.read()
            if not ok:
                raise RuntimeError(f"input video ended at frame {index}")
            panel = np.full((height, 640, 3), (27, 29, 34), np.uint8)
            panel = put_rows(panel, [
                (35, 24, "相机 Z 方向相对尺（不是二维像素距离）", (245, 245, 245), 27),
                (35, 62, f"参考 {reference_label}=0；左负=更靠近相机；范围 ±{limit:.0f} mm", (185, 190, 200), 19),
                (35, 91, "Stereo 是表面代理，不是解剖 wrist；三路均非外部真值", (80, 200, 255), 18),
            ])
            for side_index, (side, y) in enumerate((("left", 330), ("right", 745))):
                hawor_vector = row[side][hawor_key]
                stereo_vector = row[side][stereo_key]
                hz = finite(hawor_vector[2]) if isinstance(hawor_vector, list) and len(hawor_vector) == 3 else None
                sz = finite(stereo_vector[2]) if isinstance(stereo_vector, list) and len(stereo_vector) == 3 else None
                draw_ruler(panel, y, f"{'左手' if side_index == 0 else '右手'} / frame {index} / {index / fps:.2f}s", [
                    (reference_label, 0.0, COLORS["reference"], "circle"),
                    ("HaWoR", hz, COLORS["hawor"], "cross"),
                    ("Stereo surface", sz, COLORS["stereo"], "diamond"),
                ], limit)
            combined = np.concatenate((source, panel), axis=1)
            assert encoder.stdin is not None
            encoder.stdin.write(combined.tobytes())
            decoded += 1
        ok, _ = capture.read()
        if ok:
            raise RuntimeError("input video contains extra frames")
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg encoder failed")
    except Exception:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    if decoded != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("decoded frame count mismatch")
    os.replace(temporary, output)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(output),
    ], text=True))["streams"][0]
    result = {
        "schema_version": "wrist-depth-ruler-review-v1",
        "status": "PASS_DEVELOPMENT_VISUALIZATION",
        "session_id": data["session_id"],
        "frame_count": frame_count,
        "fps": fps,
        "reference_z_zero": reference_label,
        "z_ruler_limit_mm": limit,
        "inputs": {"comparison_video": artifact(video), "metrics": artifact(metrics_path)},
        "output": artifact(output),
        "probe": probe,
        "claim_limit": "Relative camera-Z visualization only; no source is external wrist ground truth and Stereo remains a visible-surface proxy.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "frames": frame_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
