#!/usr/bin/env python3
"""Build a full-session absolute 3D PICO/HaWoR/Stereo wrist review.

This review intentionally does not subtract the PICO position.  It shows both
camera-frame and first-camera-world trajectories so a stationary reference
marker cannot be mistaken for a stationary physical wrist.  Stereo remains a
visible-surface proxy reconstructed from the frozen comparison receipt.
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont


SIDES = ("left", "right")
SOURCES = ("PICO", "HaWoR", "Stereo surface")
COLORS_RGB = {
    "PICO": (0.10, 0.78, 0.28),
    "HaWoR": (1.00, 0.55, 0.10),
    "Stereo surface": (0.86, 0.18, 0.85),
}
COLORS_BGR = {
    key: tuple(int(round(channel * 255)) for channel in value[::-1])
    for key, value in COLORS_RGB.items()
}


def font_path() -> str | None:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = font_path()
    return ImageFont.truetype(path, size=size) if path else ImageFont.load_default()


def put_text(frame: np.ndarray, rows: list[tuple[int, int, str, tuple[int, int, int], int]]) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for x, y, text, bgr, size in rows:
        cache.setdefault(size, pil_font(size))
        draw.text((x, y), text, font=cache[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def finite_vector(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    array = np.asarray(value, np.float64)
    return array if np.isfinite(array).all() else None


def apply_transform(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform @ np.r_[point, 1.0])[:3]


def robust_limits(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = array.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    low = np.quantile(points, 0.005, axis=0)
    high = np.quantile(points, 0.995, axis=0)
    span = np.maximum(high - low, 0.05)
    return low - span * 0.12, high + span * 0.12


def load_trajectories(session: Path, metrics_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict], dict]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((session / "preprocess/all_data").glob("*/training_data.json"))]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    records = metrics["per_frame_delta_xyz_mm"]
    if len(rows) != len(records):
        raise RuntimeError("frame-count mismatch between PICO rows and comparison receipt")
    camera = {name: np.full((2, len(rows), 3), np.nan, np.float64) for name in SOURCES}
    world = {name: np.full((2, len(rows), 3), np.nan, np.float64) for name in SOURCES}
    for frame, (row, record) in enumerate(zip(rows, records, strict=True)):
        c2w = np.asarray(row["metadata"]["c2w"], np.float64)
        for side_index, side in enumerate(SIDES):
            hand = row["entities"]["hands"][side]
            if hand.get("pose_source") != "pico_wrist_joint":
                continue
            pico = np.asarray(hand["T_wrist_to_camera"], np.float64)[:3, 3]
            hawor_delta = finite_vector(record[side].get("hawor_minus_pico"))
            stereo_delta = finite_vector(record[side].get("stereo_minus_pico"))
            values = {
                "PICO": pico,
                "HaWoR": pico + hawor_delta / 1000.0 if hawor_delta is not None else None,
                "Stereo surface": pico + stereo_delta / 1000.0 if stereo_delta is not None else None,
            }
            for source, value in values.items():
                if value is None or not np.isfinite(value).all():
                    continue
                camera[source][side_index, frame] = value
                world[source][side_index, frame] = apply_transform(c2w, value)
    return camera, world, rows, metrics


def stack_sources(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([data[source] for source in SOURCES], axis=0)


def trajectory_stats(camera: dict[str, np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for side_index, side in enumerate(SIDES):
        side_result: dict[str, object] = {}
        pico = camera["PICO"][side_index]
        for source in SOURCES:
            values = camera[source][side_index]
            valid = np.isfinite(values).all(axis=1)
            z = values[valid, 2]
            steps = np.linalg.norm(np.diff(values[valid], axis=0), axis=1) * 1000.0
            distances = np.linalg.norm(values - pico, axis=1) * 1000.0
            distances = distances[np.isfinite(distances)]
            side_result[source] = {
                "valid_frames": int(valid.sum()),
                "z_min_m": float(z.min()),
                "z_max_m": float(z.max()),
                "z_range_mm": float(np.ptp(z) * 1000.0),
                "step_p95_mm": float(np.percentile(steps, 95)) if len(steps) else None,
                "distance_to_pico_mean_mm": float(distances.mean()) if len(distances) else None,
            }
        result[side] = side_result
    return result


class PlotPanel:
    def __init__(self, camera: dict[str, np.ndarray], world: dict[str, np.ndarray], fps: float, size: tuple[int, int]):
        self.camera = camera
        self.world = world
        self.fps = fps
        self.width, self.height = size
        self.figure = Figure(figsize=(self.width / 100, self.height / 100), dpi=100, facecolor="#17191d")
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes3d = [
            self.figure.add_subplot(2, 2, 1, projection="3d"),
            self.figure.add_subplot(2, 2, 2, projection="3d"),
        ]
        self.axes_z = [self.figure.add_subplot(2, 2, 3), self.figure.add_subplot(2, 2, 4)]
        self.cn = FontProperties(fname=font_path(), size=11) if font_path() else None
        self.frames = camera["PICO"].shape[1]
        self.time = np.arange(self.frames) / fps
        self.limits = [robust_limits(stack_sources(camera)), robust_limits(stack_sources(world))]
        self.current_lines = []
        self.current_markers = []
        self.connectors = []
        self.time_markers = []
        for axis, title, limits, dataset in zip(
            self.axes3d,
            ("相机坐标绝对 3D", "首帧相机锚定 world 3D"),
            self.limits,
            (camera, world),
            strict=True,
        ):
            self._style_3d(axis, title, limits)
            for source in SOURCES:
                for side_index, style in enumerate(("-", "--")):
                    values = dataset[source][side_index]
                    axis.plot(values[:, 0], values[:, 1], values[:, 2], style, color=COLORS_RGB[source], alpha=0.12, linewidth=1)
                    line, = axis.plot([], [], [], style, color=COLORS_RGB[source], alpha=0.85, linewidth=2)
                    marker, = axis.plot([], [], [], linestyle="", marker="o" if side_index == 0 else "^", color=COLORS_RGB[source], markersize=7)
                    self.current_lines.append((dataset, source, side_index, line))
                    self.current_markers.append((dataset, source, side_index, marker))
            for side_index in range(2):
                for _ in range(2):
                    connector, = axis.plot([], [], [], color="#cfd3d8", alpha=0.55, linewidth=1)
                    self.connectors.append((dataset, side_index, connector))
        for side_index, axis in enumerate(self.axes_z):
            side_cn = "左手" if side_index == 0 else "右手"
            self._style_z(axis, f"{side_cn}绝对相机 Z（绿色会随时间移动）")
            for source in SOURCES:
                z = camera[source][side_index, :, 2] * 1000.0
                axis.plot(self.time, z, color=COLORS_RGB[source], linewidth=1.8, label=source)
            marker = axis.axvline(0.0, color="white", linewidth=1.2, alpha=0.9)
            self.time_markers.append(marker)
        self.axes_z[0].legend(loc="best", fontsize=8, facecolor="#20242a", labelcolor="white")
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08, wspace=0.18, hspace=0.28)

    def _style_3d(self, axis, title: str, limits: tuple[np.ndarray, np.ndarray]) -> None:
        axis.set_facecolor("#17191d")
        axis.set_title(title, color="white", fontproperties=self.cn)
        axis.set_xlabel("X / m", color="white")
        axis.set_ylabel("Y / m", color="white")
        axis.set_zlabel("Z / m", color="white")
        axis.tick_params(colors="#c5c9cf", labelsize=8)
        low, high = limits
        axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_zlim(low[2], high[2])
        axis.view_init(elev=23, azim=-58)
        axis.set_box_aspect(np.maximum(high - low, 0.01))
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((0.12, 0.13, 0.15, 1.0))

    def _style_z(self, axis, title: str) -> None:
        axis.set_facecolor("#17191d")
        axis.set_title(title, color="white", fontproperties=self.cn)
        axis.set_xlabel("时间 / s", color="white", fontproperties=self.cn)
        axis.set_ylabel("optical-Z / mm", color="white")
        axis.tick_params(colors="#c5c9cf", labelsize=8)
        axis.grid(True, color="#454a52", alpha=0.35)
        for spine in axis.spines.values(): spine.set_color("#767c86")

    def render(self, frame: int) -> np.ndarray:
        start = max(0, frame - 59)
        for dataset, source, side_index, line in self.current_lines:
            values = dataset[source][side_index, start:frame + 1]
            line.set_data(values[:, 0], values[:, 1]); line.set_3d_properties(values[:, 2])
        for dataset, source, side_index, marker in self.current_markers:
            value = dataset[source][side_index, frame]
            marker.set_data([value[0]], [value[1]]); marker.set_3d_properties([value[2]])
        connector_index = 0
        for dataset in (self.camera, self.world):
            for side_index in range(2):
                pico = dataset["PICO"][side_index, frame]
                for target in ("HaWoR", "Stereo surface"):
                    other = dataset[target][side_index, frame]
                    _, _, connector = self.connectors[connector_index]
                    connector.set_data([pico[0], other[0]], [pico[1], other[1]])
                    connector.set_3d_properties([pico[2], other[2]])
                    connector_index += 1
        for marker in self.time_markers:
            marker.set_xdata([frame / self.fps, frame / self.fps])
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def project(point: np.ndarray, intrinsics: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    uvw = intrinsics @ point
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def draw_rgb_markers(frame: np.ndarray, camera: dict[str, np.ndarray], rows: list[dict], index: int) -> np.ndarray:
    intrinsics = np.asarray(rows[index]["metadata"]["k"], np.float64)
    source_width = float(rows[index]["metadata"]["w"])
    source_height = float(rows[index]["metadata"]["h"])
    intrinsics[0, :] *= frame.shape[1] / source_width
    intrinsics[1, :] *= frame.shape[0] / source_height
    for side_index, side in enumerate(SIDES):
        for source in SOURCES:
            uv = project(camera[source][side_index, index], intrinsics)
            if uv is None:
                continue
            color = COLORS_BGR[source]
            if source == "Stereo surface":
                cv2.circle(frame, uv, 14, color, 3, cv2.LINE_AA)
            elif source == "HaWoR":
                cv2.drawMarker(frame, uv, color, cv2.MARKER_TILTED_CROSS, 20, 3, cv2.LINE_AA)
            else:
                cv2.circle(frame, uv, 8, color, -1, cv2.LINE_AA)
            cv2.putText(frame, "L" if side == "left" else "R", (uv[0] + 10, uv[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-montage", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    session = args.session_root.resolve(strict=True)
    metrics_path = args.metrics_json.resolve(strict=True)
    raw_video = args.raw_video.resolve(strict=True)
    outputs = (args.output_video.resolve(), args.output_montage.resolve(), args.result.resolve())
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RuntimeError("fresh/no-clobber outputs required")
    camera, world, rows, metrics = load_trajectories(session, metrics_path)
    frame_count = len(rows); fps = float(metrics["fps"])
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened(): raise RuntimeError("cannot decode raw video")
    panel = PlotPanel(camera, world, fps, (1100, 960))
    output_video, output_montage, result_path = outputs
    output_video.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_video.stem}.", suffix=".mp4", dir=output_video.parent)
    os.close(descriptor); temporary = Path(temporary_name); temporary.unlink()
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", f"{fps:g}",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(temporary),
    ], stdin=subprocess.PIPE)
    snapshots: dict[int, np.ndarray] = {}
    snapshot_ids = {0, frame_count // 2, frame_count - 1}
    try:
        for index in range(frame_count):
            ok, raw = capture.read()
            if not ok: raise RuntimeError(f"raw video ended at frame {index}")
            canvas = np.full((1080, 1920, 3), (22, 24, 28), np.uint8)
            resized = cv2.resize(raw, (760, 570), interpolation=cv2.INTER_AREA)
            canvas[85:655, 20:780] = draw_rgb_markers(resized, camera, rows, index)
            plot = panel.render(index)
            canvas[65:1025, 800:1900] = plot
            rows_text = [
                (20, 15, "Chips023 三路手腕：绝对 3D 与 Z 轨迹（PICO 不归零）", (245, 245, 245), 31),
                (25, 670, f"frame {index}/{frame_count - 1}   time {index / fps:.2f}s", (230, 230, 230), 24),
                (25, 708, "绿色=PICO/OpenXR wrist；橙色=HaWoR MANO wrist；紫色=Stereo 可见表面代理", (220, 220, 220), 20),
                (25, 742, "圆点/实线=左手；三角/虚线=右手；细白线=当前帧相对 PICO 的三维差", (185, 190, 200), 19),
            ]
            y = 785
            for side_index, side in enumerate(SIDES):
                label = "左手" if side == "left" else "右手"
                parts = []
                pico = camera["PICO"][side_index, index]
                for source in SOURCES:
                    point = camera[source][side_index, index]
                    distance = np.linalg.norm(point - pico) * 1000.0
                    parts.append(f"{source}: Z={point[2] * 1000:+.1f}mm / Δ3D={distance:.1f}mm")
                rows_text.append((25, y, label + "  " + " | ".join(parts), (230, 230, 230), 17)); y += 34
            rows_text.extend([
                (25, 875, "判读边界：PICO 是工程锚点而非外部真值；Stereo 是表面，不是解剖手腕中心。", (80, 200, 255), 19),
                (25, 910, "相机坐标包含头部运动；world 面板用逐帧 c2w 稳定到首帧相机锚点。", (80, 200, 255), 19),
                (25, 945, "三路差异只能用于选基线和诊断，不能直接写成真实毫米误差。", (80, 200, 255), 19),
            ])
            canvas = put_text(canvas, rows_text)
            if index in snapshot_ids: snapshots[index] = canvas.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait() != 0: raise RuntimeError("ffmpeg encoder failed")
    except Exception:
        if encoder.stdin is not None and not encoder.stdin.closed: encoder.stdin.close()
        encoder.kill(); encoder.wait(); temporary.unlink(missing_ok=True); raise
    finally:
        capture.release()
    os.replace(temporary, output_video)
    montage = np.concatenate([cv2.resize(snapshots[index], (960, 540), interpolation=cv2.INTER_AREA) for index in sorted(snapshot_ids)], axis=1)
    if not cv2.imwrite(str(output_montage), montage): raise RuntimeError("failed to write montage")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(output_video),
    ], text=True))["streams"][0]
    if int(probe["nb_read_frames"]) != frame_count: raise RuntimeError("encoded frame count mismatch")
    result = {
        "schema_version": "three-wrist-absolute-3d-review-v1",
        "status": "PASS_DEVELOPMENT_VISUALIZATION",
        "session_id": metrics["session_id"],
        "frame_count": frame_count,
        "fps": fps,
        "coordinate_views": ["camera_optical_xyz", "first_frame_camera_anchored_world_xyz"],
        "reference_subtracted": False,
        "inputs": {"session_root": str(session), "raw_video": artifact(raw_video), "metrics": artifact(metrics_path)},
        "outputs": {"video": artifact(output_video), "montage": artifact(output_montage)},
        "statistics": trajectory_stats(camera),
        "claim_limit": "Cross-system development visualization only. PICO is an engineering wrist anchor, HaWoR is learned monocular MANO, and Stereo is a visible-surface proxy; none is external physical truth.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "video": str(output_video), "frames": frame_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
