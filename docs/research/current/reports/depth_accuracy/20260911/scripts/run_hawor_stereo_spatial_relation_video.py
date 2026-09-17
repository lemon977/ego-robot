#!/usr/bin/env python3
"""Render an interpretable HaWoR MANO vs registered stereo spatial diagnostic.

This is a cross-sensor diagnostic only.  Stereo samples are visible-surface
depth proxies, not joint ground truth.  The optional per-frame Z translation
is a visualization experiment and is never written back to HaWoR or Depth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_hawor_stereo_surface_consistency_qa import (
    load_role_mask,
    rasterize_visible_surface,
    registered_selected_depth,
)


FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
SIDE_NAMES = ("left", "right")
RIGHT = 1
SKELETON_EDGES = tuple(
    (start + offset, start + offset + 1)
    for start in (1, 5, 9, 13, 17)
    for offset in range(3)
) + ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))
KEY_JOINTS = (0, 4, 8, 12, 16, 20)
KEY_LABELS = ("wrist", "thumb", "index", "middle", "ring", "pinky")

CYAN = (255, 220, 40)  # BGR: registered stereo visible surface
ORANGE = (30, 145, 255)  # BGR: original HaWoR/MANO
GREEN = (70, 215, 80)  # BGR: median-Z shifted MANO
WHITE = (245, 245, 245)
MUTED = (150, 160, 170)
BACKGROUND = (21, 24, 30)

CLAIM_LIMIT = (
    "DIAGNOSTIC_ONLY_NO_EXTERNAL_GT. Stereo proxy points are visible-surface "
    "depth sampled at HaWoR rays, not stereo joints or physical ground truth. "
    "Median-Z alignment is a per-frame visualization experiment, not a pose fix."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    size: int,
    color: tuple[int, int, int] = WHITE,
) -> None:
    """Draw UTF-8 text on a BGR image."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(FONT_PATH), size)
    drawer.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
    thickness: int = 2,
) -> None:
    valid = np.isfinite(points).all(axis=1)
    for first, second in SKELETON_EDGES:
        if valid[first] and valid[second]:
            cv2.line(
                image,
                tuple(np.rint(points[first]).astype(int)),
                tuple(np.rint(points[second]).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if not valid[index]:
            continue
        cv2.circle(
            image,
            tuple(np.rint(point).astype(int)),
            radius + (1 if index in KEY_JOINTS else 0),
            color,
            -1,
            cv2.LINE_AA,
        )


def robust_proxy_joints(
    joints_uv: np.ndarray,
    stereo_z: np.ndarray,
    valid: np.ndarray,
    role_mask: np.ndarray,
    intrinsics: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample visible stereo depth near HaWoR 2D joints.

    Returned XYZ are surface-depth proxies on the HaWoR joint ray.  They are
    intentionally not called stereo joints.
    """
    height, width = stereo_z.shape
    output = np.full((len(joints_uv), 3), np.nan, np.float64)
    counts = np.zeros(len(joints_uv), np.int32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    allowed = valid & role_mask & np.isfinite(stereo_z)
    for index, (u, v) in enumerate(np.asarray(joints_uv, np.float64)):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x0 = max(0, int(np.floor(u)) - radius)
        x1 = min(width, int(np.floor(u)) + radius + 1)
        y0 = max(0, int(np.floor(v)) - radius)
        y1 = min(height, int(np.floor(v)) + radius + 1)
        window = allowed[y0:y1, x0:x1]
        values = stereo_z[y0:y1, x0:x1][window]
        if values.size < 5:
            continue
        z = float(np.median(values))
        output[index] = ((u - cx) * z / fx, (v - cy) * z / fy, z)
        counts[index] = int(values.size)
    return output, counts


def project_metric(
    values: np.ndarray,
    z_center: float,
    vertical_center: float,
    rect: tuple[int, int, int, int],
    vertical_axis: int,
    z_half_range_mm: float = 170.0,
    vertical_half_range_mm: float = 150.0,
) -> np.ndarray:
    left, top, right, bottom = rect
    z_mm = (values[:, 2] - z_center) * 1000.0
    vertical_mm = (values[:, vertical_axis] - vertical_center) * 1000.0
    px = left + (z_mm + z_half_range_mm) / (2.0 * z_half_range_mm) * (right - left)
    py = bottom - (vertical_mm + vertical_half_range_mm) / (2.0 * vertical_half_range_mm) * (bottom - top)
    return np.column_stack((px, py))


def scatter_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = image.shape[:2]
    finite = np.isfinite(points).all(axis=1)
    for x, y in np.rint(points[finite]).astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def side_panel(
    stereo_xyz: np.ndarray,
    mano_vertices: np.ndarray,
    mano_joints: np.ndarray,
    proxy_joints: np.ndarray,
    shift_m: float,
    aligned: bool,
    median_delta_mm: float,
) -> np.ndarray:
    panel = np.full((480, 640, 3), BACKGROUND, np.uint8)
    mano_color = GREEN if aligned else ORANGE
    title = "减去median-Z后的空间关系" if aligned else "原始空间关系（相机侧视）"
    put_text(panel, title, (18, 8), 24)
    put_text(panel, "青=Stereo可见表面代理；橙=原始MANO；绿=仅沿Z平移后的MANO", (18, 40), 15, MUTED)

    if stereo_xyz.size == 0:
        put_text(panel, "当前帧没有足够的共同有效表面", (120, 220), 22, (80, 120, 255))
        return panel

    shifted_vertices = mano_vertices.copy().astype(np.float64)
    shifted_vertices[:, 2] += shift_m
    shifted_joints = mano_joints.copy().astype(np.float64)
    shifted_joints[:, 2] += shift_m
    z_center = float(np.median(stereo_xyz[:, 2]))
    x_center = float(np.median(stereo_xyz[:, 0]))
    y_center = float(np.median(stereo_xyz[:, 1]))

    views = (
        ("X-Z：横向宽度 vs 前后深度", 0, x_center, (48, 88, 622, 244)),
        ("Y-Z：上下方向 vs 前后深度", 1, y_center, (48, 296, 622, 452)),
    )
    for label, axis, center, rect in views:
        left, top, right, bottom = rect
        cv2.rectangle(panel, (left, top), (right, bottom), (90, 95, 105), 1)
        # 50 mm metric grid along Z.
        for millimetres in (-150, -100, -50, 0, 50, 100, 150):
            x = int(round(left + (millimetres + 170) / 340 * (right - left)))
            cv2.line(panel, (x, top), (x, bottom), (48, 52, 60), 1)
            cv2.putText(panel, str(millimetres), (x - 12, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, MUTED, 1, cv2.LINE_AA)
        put_text(panel, label, (left + 4, top - 23), 15, MUTED)
        cv2.putText(panel, "Z mm ->", (right - 72, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

        stereo_plot = project_metric(stereo_xyz, z_center, center, rect, axis)
        mano_plot = project_metric(shifted_vertices, z_center, center, rect, axis)
        joint_plot = project_metric(shifted_joints, z_center, center, rect, axis)
        proxy_plot = project_metric(proxy_joints, z_center, center, rect, axis)
        scatter_points(panel, stereo_plot[:: max(1, len(stereo_plot) // 1800)], CYAN, 1)
        scatter_points(panel, mano_plot, mano_color, 1)
        draw_skeleton(panel, joint_plot, mano_color, radius=2, thickness=1)
        draw_skeleton(panel, proxy_plot, CYAN, radius=2, thickness=1)
        valid_proxy = np.isfinite(proxy_plot).all(axis=1) & np.isfinite(joint_plot).all(axis=1)
        for index in KEY_JOINTS:
            if valid_proxy[index]:
                cv2.line(
                    panel,
                    tuple(np.rint(joint_plot[index]).astype(int)),
                    tuple(np.rint(proxy_plot[index]).astype(int)),
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )

    direction = -median_delta_mm if np.isfinite(median_delta_mm) else 0.0
    if aligned:
        put_text(panel, f"本帧仅对MANO施加 Z 平移 {direction:+.1f} mm（诊断，不回写）", (90, 258), 17, GREEN)
    else:
        put_text(panel, f"两表面本帧中位差 {median_delta_mm:+.1f} mm", (170, 258), 19, ORANGE)
    return panel


def proxy_side_panel(
    mano_joints: np.ndarray,
    proxy_joints: np.ndarray,
    stereo_xyz: np.ndarray,
    proxy_delta_mm: np.ndarray,
) -> np.ndarray:
    """Draw only HaWoR joints and stereo surface-depth proxy points."""
    panel = np.full((480, 640, 3), BACKGROUND, np.uint8)
    put_text(panel, "Stereo surface-depth proxy skeleton", (18, 8), 23)
    put_text(panel, "青点不是Stereo关节真值；只是HaWoR关节射线附近的可见表面深度", (18, 40), 15, MUTED)
    if stereo_xyz.size == 0:
        put_text(panel, "当前帧没有足够的共同有效表面", (120, 220), 22, (80, 120, 255))
        return panel
    z_center = float(np.median(stereo_xyz[:, 2]))
    x_center = float(np.median(stereo_xyz[:, 0]))
    y_center = float(np.median(stereo_xyz[:, 1]))
    views = (
        ("X-Z", 0, x_center, (48, 88, 622, 244)),
        ("Y-Z", 1, y_center, (48, 296, 622, 452)),
    )
    for label, axis, center, rect in views:
        left, top, right, bottom = rect
        cv2.rectangle(panel, (left, top), (right, bottom), (90, 95, 105), 1)
        for millimetres in (-150, -100, -50, 0, 50, 100, 150):
            x = int(round(left + (millimetres + 170) / 340 * (right - left)))
            cv2.line(panel, (x, top), (x, bottom), (48, 52, 60), 1)
        put_text(panel, f"{label}：橙=HaWoR解剖关节；青=Stereo表面代理", (left + 4, top - 23), 15, MUTED)
        cv2.putText(panel, "Z mm ->", (right - 72, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)
        joint_plot = project_metric(mano_joints, z_center, center, rect, axis)
        proxy_plot = project_metric(proxy_joints, z_center, center, rect, axis)
        draw_skeleton(panel, joint_plot, ORANGE, radius=3, thickness=2)
        draw_skeleton(panel, proxy_plot, CYAN, radius=3, thickness=2)
        valid = np.isfinite(joint_plot).all(axis=1) & np.isfinite(proxy_plot).all(axis=1)
        for index in KEY_JOINTS:
            if not valid[index]:
                continue
            first = tuple(np.rint(joint_plot[index]).astype(int))
            second = tuple(np.rint(proxy_plot[index]).astype(int))
            cv2.line(panel, first, second, WHITE, 2, cv2.LINE_AA)
            midpoint = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
            value = proxy_delta_mm[index]
            cv2.putText(
                panel, f"{value:+.0f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                WHITE, 1, cv2.LINE_AA,
            )
    put_text(panel, "连线数字单位：mm；含关节中心到皮肤表面的定义差，不可当关节误差", (44, 258), 15, (100, 190, 255))
    return panel


def rgb_panel(
    rgb: np.ndarray,
    mano_visible: np.ndarray,
    retained: np.ndarray,
    joints_uv: np.ndarray,
    frame: int,
    fps: float,
) -> np.ndarray:
    panel = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    scale = np.array([0.5, 0.5], np.float64)
    mano_small = cv2.resize(mano_visible.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    retained_small = cv2.resize(retained.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mano_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, ORANGE, 2, cv2.LINE_AA)
    contours, _ = cv2.findContours(retained_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, WHITE, 1, cv2.LINE_AA)
    draw_skeleton(panel, joints_uv * scale, ORANGE, radius=3, thickness=2)
    dark = panel.copy()
    cv2.rectangle(dark, (0, 0), (640, 64), (0, 0, 0), -1)
    panel = cv2.addWeighted(dark, 0.70, panel, 0.30, 0)
    put_text(panel, "原始RGB + HaWoR右手MANO", (14, 5), 23)
    put_text(panel, f"frame {frame:03d}  time {frame / fps:5.2f}s   白线=共同有效比较区域", (14, 36), 15, WHITE)
    return panel


def stats_panel(
    frame: int,
    raw_values_mm: np.ndarray,
    aligned_values_mm: np.ndarray,
    proxy_delta_mm: np.ndarray,
    proxy_counts: np.ndarray,
    median_delta_mm: float,
    aligned: bool,
) -> np.ndarray:
    panel = np.full((480, 640, 3), (246, 247, 249), np.uint8)
    ink = (32, 35, 40)
    accent = GREEN if aligned else ORANGE
    put_text(panel, "当前帧深度关系", (22, 14), 27, ink)
    put_text(panel, "定义：ΔZ = MANO Z - Stereo Z", (22, 52), 18, ink)
    put_text(panel, "负值 = MANO更靠近相机", (22, 80), 18, ink)
    put_text(panel, f"表面中位 ΔZ       {median_delta_mm:+7.1f} mm", (22, 122), 25, accent)
    if raw_values_mm.size:
        put_text(panel, f"原始表面 MAE       {np.mean(np.abs(raw_values_mm)):7.1f} mm", (22, 164), 20, ink)
        put_text(panel, f"原始表面 abs P95   {np.quantile(np.abs(raw_values_mm), .95):7.1f} mm", (22, 195), 20, ink)
        put_text(panel, f"有效表面像素       {len(raw_values_mm):7d}", (22, 226), 20, ink)
    if aligned_values_mm.size:
        put_text(panel, f"平移后 residual MAE {np.mean(np.abs(aligned_values_mm)):6.1f} mm", (22, 268), 20, GREEN)
        put_text(panel, f"平移后 residual P95 {np.quantile(np.abs(aligned_values_mm), .95):6.1f} mm", (22, 299), 20, GREEN)

    put_text(panel, "Stereo surface-depth proxy（不是真值关节）", (22, 340), 17, ink)
    y = 371
    for index, label in zip(KEY_JOINTS, KEY_LABELS, strict=True):
        value = proxy_delta_mm[index]
        suffix = f"{value:+6.1f} mm" if np.isfinite(value) else "   N/A"
        count = int(proxy_counts[index])
        put_text(panel, f"{label:<7} {suffix}  n={count}", (22 + (y >= 435) * 310, 371 + ((y - 371) % 93)), 15, ink)
        y += 31
    put_text(panel, f"frame {frame:03d}  仅作跨传感器诊断，无外部GT", (22, 454), 14, (80, 85, 92))
    return panel


def curve_panel(
    history: list[dict[str, float]],
    total_frames: int,
    aligned: bool,
) -> np.ndarray:
    panel = np.full((240, 1920, 3), (250, 250, 250), np.uint8)
    left, right, top, bottom = 72, 1888, 46, 204
    cv2.rectangle(panel, (left, top), (right, bottom), (100, 100, 100), 1)
    if aligned:
        title = "减去每帧median-Z后：剩余表面 abs P95（形状/局部深度差）"
        values = [row["aligned_p95_mm"] for row in history]
        low, high = 0.0, 150.0
        color = GREEN
        zero_y = bottom
    else:
        title = "原始右手表面：每帧median ΔZ（MANO - Stereo）"
        values = [row["median_delta_mm"] for row in history]
        low, high = -150.0, 150.0
        color = ORANGE
        zero_y = int(round(bottom - (0.0 - low) / (high - low) * (bottom - top)))
        cv2.line(panel, (left, zero_y), (right, zero_y), (120, 120, 120), 1)
    put_text(panel, title, (72, 7), 22, (35, 38, 43))
    cv2.putText(panel, f"{high:.0f} mm", (8, top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1)
    cv2.putText(panel, f"{low:.0f} mm", (8, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1)
    points = []
    for index, value in enumerate(values):
        x = left + int(round((right - left) * index / max(1, total_frames - 1)))
        clipped = float(np.clip(value, low, high))
        y = bottom - int(round((bottom - top) * (clipped - low) / (high - low)))
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(panel, [np.asarray(points, np.int32)], False, color, 3, cv2.LINE_AA)
    if points:
        x, y = points[-1]
        cv2.line(panel, (x, top), (x, bottom), (40, 40, 40), 1)
        cv2.circle(panel, (x, y), 5, color, -1, cv2.LINE_AA)
        put_text(panel, f"当前 {values[-1]:+.1f} mm", (max(left, x - 150), 207), 16, color)
    put_text(panel, "横轴：frame 0 → 292", (1540, 210), 15, (70, 75, 80))
    return panel


def proxy_curve_panel(history: list[dict[str, Any]], total_frames: int) -> np.ndarray:
    panel = np.full((240, 1920, 3), (250, 250, 250), np.uint8)
    left, right, top, bottom = 72, 1888, 46, 204
    cv2.rectangle(panel, (left, top), (right, bottom), (100, 100, 100), 1)
    zero_y = int(round((top + bottom) / 2))
    cv2.line(panel, (left, zero_y), (right, zero_y), (120, 120, 120), 1)
    put_text(panel, "关键射线：HaWoR joint-center Z - Stereo surface-depth proxy Z", (72, 7), 22, (35, 38, 43))
    palette = ((40, 40, 220), (20, 150, 240), (30, 190, 120), (220, 150, 30), (200, 80, 160), (120, 80, 40))
    for label, color in zip(KEY_LABELS, palette, strict=True):
        points = []
        for index, row in enumerate(history):
            value = row["proxy_joint_delta_mm"].get(label)
            if value is None:
                continue
            x = left + int(round((right - left) * index / max(1, total_frames - 1)))
            y = zero_y - int(round((bottom - top) * 0.5 * np.clip(value, -150.0, 150.0) / 150.0))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)
    for legend_index, (label, color) in enumerate(zip(KEY_LABELS, palette, strict=True)):
        x = 1060 + legend_index * 130
        cv2.line(panel, (x, 23), (x + 24, 23), color, 3, cv2.LINE_AA)
        cv2.putText(panel, label, (x + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(panel, "+150", (8, top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1)
    cv2.putText(panel, "-150", (8, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1)
    if history:
        current_x = left + int(round((right - left) * (len(history) - 1) / max(1, total_frames - 1)))
        cv2.line(panel, (current_x, top), (current_x, bottom), (40, 40, 40), 1)
    put_text(panel, "仅用于观察共同Z偏移；表面代理 ≠ 解剖关节GT", (72, 210), 16, (70, 75, 80))
    return panel


def encode_h264(source: Path, target: Path, fps: float) -> None:
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-r", f"{fps:.8f}", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg H264 encode failed: {completed.stderr[-2000:]}")


def validate_video(path: Path, frames: int) -> dict[str, Any]:
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"video decode failed for {path}: {decode.stderr[-2000:]}")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    exact = int(stream["width"]) == 1920 and int(stream["height"]) == 720 and int(stream["nb_read_frames"]) == frames
    if not exact:
        raise RuntimeError(f"video geometry/frame validation failed for {path}: {stream}")
    return {**stream, "ffmpeg_xerror": True, "exact": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with np.load(args.hawor_npz, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.surface_cache, allow_pickle=False) as archive:
        vertices = [np.asarray(archive["vertices_left"]), np.asarray(archive["vertices_right"])]
        faces = [np.asarray(archive["faces_left"]), np.asarray(archive["faces_right"])]
    with np.load(args.registration, allow_pickle=False) as archive:
        registration = {key: np.asarray(archive[key]) for key in archive.files}

    frames = np.asarray(hawor["original_frame_indices"], np.int64)
    total = min(len(frames), args.max_frames if args.max_frames > 0 else len(frames))
    depth_files = sorted(args.depth_dir.glob("frames/*.npz"))
    if len(depth_files) != len(frames):
        raise RuntimeError(f"frame mismatch HaWoR={len(frames)} Depth={len(depth_files)}")
    if vertices[RIGHT].shape[:2] != (len(frames), 778):
        raise RuntimeError(f"unexpected MANO cache shape: {vertices[RIGHT].shape}")
    selected_k = np.asarray(registration["selected_rgb_intrinsics"], np.float64)
    if not np.allclose(hawor["intrinsics"], selected_k[None], atol=1e-6):
        raise RuntimeError("HaWoR K and registration selected-left K differ")
    fps = float(hawor["fps"])
    capture = cv2.VideoCapture(str(args.rgb))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open RGB video: {args.rgb}")

    raw_final = output / "Chips034_Right_3D_Depth_Comparison.mp4"
    aligned_final = output / "Chips034_Right_Median_Z_Aligned_Auxiliary.mp4"
    proxy_final = output / "Chips034_Right_Proxy_Skeleton.mp4"
    temp_root = Path(tempfile.mkdtemp(prefix="chips-spatial-relation-", dir="/tmp"))
    raw_temp = temp_root / "raw.mp4"
    aligned_temp = temp_root / "aligned.mp4"
    proxy_temp = temp_root / "proxy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(str(raw_temp), fourcc, fps, (1920, 720))
    aligned_writer = cv2.VideoWriter(str(aligned_temp), fourcc, fps, (1920, 720))
    proxy_writer = cv2.VideoWriter(str(proxy_temp), fourcc, fps, (1920, 720))
    if not raw_writer.isOpened() or not aligned_writer.isOpened() or not proxy_writer.isOpened():
        raise RuntimeError("cannot open temporary video writers")

    kernel_mano = np.ones((7, 7), np.uint8)
    kernel_role = np.ones((7, 7), np.uint8)
    history: list[dict[str, Any]] = []
    key_frames = {0, 30, 53, 90, 144, 192, 205, total - 1}
    key_images: list[tuple[int, np.ndarray]] = []
    all_raw: list[np.ndarray] = []
    all_aligned: list[np.ndarray] = []
    proxy_key_values: dict[str, list[float]] = {label: [] for label in KEY_LABELS}

    try:
        for local, (frame, depth_path) in enumerate(zip(frames[:total].tolist(), depth_files[:total], strict=True)):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decode failed at local frame {local}")
            with np.load(depth_path, allow_pickle=False) as archive:
                if int(archive["frame_id"]) != frame:
                    raise RuntimeError(f"depth frame identity mismatch at {frame}")
                stereo_z, stereo_valid, stereo_local = registered_selected_depth(
                    archive["depth_m"], archive["valid"], registration, (1280, 960), args.local_tolerance_m
                )

            triangles = []
            triangle_labels = []
            for side in range(2):
                if hawor["observed"][side, local]:
                    triangles.append(vertices[side][local][faces[side]])
                    triangle_labels.append(np.full(len(faces[side]), side, np.int16))
            mano_z, labels = rasterize_visible_surface(
                np.concatenate(triangles).astype(np.float64),
                np.concatenate(triangle_labels),
                float(selected_k[0, 0]), float(selected_k[1, 1]),
                float(selected_k[0, 2]), float(selected_k[1, 2]), 1280, 960,
            )
            mano_visible = labels == RIGHT
            role = load_role_mask(args.hand_mask_root, None, "right", frame, (960, 1280))
            eroded_mano = cv2.erode(mano_visible.astype(np.uint8), kernel_mano).astype(bool)
            eroded_role = cv2.erode(role.astype(np.uint8), kernel_role).astype(bool)
            retained = eroded_mano & eroded_role & stereo_valid & stereo_local
            values_m = (mano_z - stereo_z)[retained]
            if values_m.size < 100:
                raise RuntimeError(f"insufficient right-hand support at frame {frame}: {values_m.size}")
            median_delta_m = float(np.median(values_m))
            raw_values_mm = values_m * 1000.0
            aligned_values_mm = (values_m - median_delta_m) * 1000.0

            yy, xx = np.nonzero(retained)
            stereo_depths = stereo_z[yy, xx]
            stereo_xyz = np.column_stack(
                (
                    (xx + 0.5 - selected_k[0, 2]) * stereo_depths / selected_k[0, 0],
                    (yy + 0.5 - selected_k[1, 2]) * stereo_depths / selected_k[1, 1],
                    stereo_depths,
                )
            )
            joints_uv = np.asarray(hawor["joints_2d"][RIGHT, local], np.float64)
            joints_xyz = np.asarray(hawor["joints_3d_camera"][RIGHT, local], np.float64)
            proxy_xyz, proxy_counts = robust_proxy_joints(
                joints_uv,
                stereo_z,
                stereo_valid & stereo_local,
                role,
                selected_k,
                args.proxy_radius,
            )
            proxy_delta_mm = (joints_xyz[:, 2] - proxy_xyz[:, 2]) * 1000.0
            for index, label in zip(KEY_JOINTS, KEY_LABELS, strict=True):
                if np.isfinite(proxy_delta_mm[index]):
                    proxy_key_values[label].append(float(proxy_delta_mm[index]))

            row = {
                "frame": int(frame),
                "time_s": float(frame / fps),
                "samples": int(values_m.size),
                "median_delta_mm": median_delta_m * 1000.0,
                "raw_abs_mae_mm": float(np.mean(np.abs(raw_values_mm))),
                "raw_abs_p95_mm": float(np.quantile(np.abs(raw_values_mm), 0.95)),
                "applied_mano_z_shift_mm": -median_delta_m * 1000.0,
                "aligned_abs_mae_mm": float(np.mean(np.abs(aligned_values_mm))),
                "aligned_p95_mm": float(np.quantile(np.abs(aligned_values_mm), 0.95)),
                "proxy_joint_delta_mm": {
                    name: (float(proxy_delta_mm[index]) if np.isfinite(proxy_delta_mm[index]) else None)
                    for index, name in zip(KEY_JOINTS, KEY_LABELS, strict=True)
                },
                "proxy_joint_sample_counts": {
                    name: int(proxy_counts[index])
                    for index, name in zip(KEY_JOINTS, KEY_LABELS, strict=True)
                },
            }
            history.append(row)
            all_raw.append(raw_values_mm.astype(np.float32))
            all_aligned.append(aligned_values_mm.astype(np.float32))

            left_panel = rgb_panel(rgb, mano_visible, retained, joints_uv, int(frame), fps)
            raw_side = side_panel(
                stereo_xyz, vertices[RIGHT][local], joints_xyz, proxy_xyz, 0.0, False, row["median_delta_mm"]
            )
            aligned_side = side_panel(
                stereo_xyz, vertices[RIGHT][local], joints_xyz, proxy_xyz,
                -median_delta_m, True, row["median_delta_mm"],
            )
            raw_stats = stats_panel(
                int(frame), raw_values_mm, aligned_values_mm, proxy_delta_mm, proxy_counts,
                row["median_delta_mm"], False,
            )
            aligned_stats = stats_panel(
                int(frame), raw_values_mm, aligned_values_mm, proxy_delta_mm, proxy_counts,
                row["median_delta_mm"], True,
            )
            raw_frame = np.vstack((np.hstack((left_panel, raw_side, raw_stats)), curve_panel(history, total, False)))
            aligned_frame = np.vstack((np.hstack((left_panel, aligned_side, aligned_stats)), curve_panel(history, total, True)))
            proxy_side = proxy_side_panel(joints_xyz, proxy_xyz, stereo_xyz, proxy_delta_mm)
            proxy_frame = np.vstack((np.hstack((left_panel, proxy_side, raw_stats)), proxy_curve_panel(history, total)))
            raw_writer.write(raw_frame)
            aligned_writer.write(aligned_frame)
            proxy_writer.write(proxy_frame)
            if local in key_frames:
                key_images.append((int(frame), cv2.resize(proxy_frame, (960, 360))))
    finally:
        capture.release()
        raw_writer.release()
        aligned_writer.release()
        proxy_writer.release()

    encode_h264(raw_temp, raw_final, fps)
    encode_h264(aligned_temp, aligned_final, fps)
    encode_h264(proxy_temp, proxy_final, fps)
    raw_temp.unlink(missing_ok=True)
    aligned_temp.unlink(missing_ok=True)
    proxy_temp.unlink(missing_ok=True)
    temp_root.rmdir()

    if key_images:
        images = [image for _, image in key_images]
        if len(images) % 2:
            images.append(np.zeros_like(images[0]))
        sheet = np.vstack([np.hstack(images[index:index + 2]) for index in range(0, len(images), 2)])
        cv2.imwrite(str(output / "Chips034_Right_Proxy_Skeleton.png"), sheet)

    pooled_raw = np.concatenate(all_raw)
    pooled_aligned = np.concatenate(all_aligned)
    frame_json = output / "FRAME_METRICS.json"
    frame_json.write_text(
        json.dumps(
            {
                "schema_version": "hawor-stereo-spatial-relation-frame-metrics-v1",
                "session": args.session,
                "side": "right",
                "delta_definition": "Z_mano_visible_surface - Z_registered_stereo_visible_surface",
                "proxy_definition": "Stereo visible-surface depth sampled near HaWoR 2D joint rays; not stereo joints or ground truth",
                "rows": history,
                "claim_limit": CLAIM_LIMIT,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    validations = {
        raw_final.name: validate_video(raw_final, total),
        aligned_final.name: validate_video(aligned_final, total),
        proxy_final.name: validate_video(proxy_final, total),
    }
    result: dict[str, Any] = {
        "schema_version": "hawor-stereo-spatial-relation-result-v1",
        "status": "PASS_DIAGNOSTIC_RENDER_NO_EXTERNAL_GT",
        "session": args.session,
        "side": "right",
        "frames": total,
        "fps": fps,
        "claim_limit": CLAIM_LIMIT,
        "colors_bgr": {
            "registered_stereo_visible_surface_proxy": list(CYAN),
            "original_hawor_mano": list(ORANGE),
            "median_z_shifted_mano": list(GREEN),
        },
        "aggregate": {
            "frame_balanced_median_delta_mean_mm": float(np.mean([row["median_delta_mm"] for row in history])),
            "frame_balanced_raw_abs_mae_mm": float(np.mean([row["raw_abs_mae_mm"] for row in history])),
            "frame_balanced_aligned_abs_mae_mm": float(np.mean([row["aligned_abs_mae_mm"] for row in history])),
            "frame_balanced_aligned_abs_p95_mm": float(np.mean([row["aligned_p95_mm"] for row in history])),
            "pixel_weighted_raw_abs_mae_mm": float(np.mean(np.abs(pooled_raw))),
            "pixel_weighted_aligned_abs_mae_mm": float(np.mean(np.abs(pooled_aligned))),
            "pixel_weighted_aligned_abs_p95_mm": float(np.quantile(np.abs(pooled_aligned), 0.95)),
            "proxy_key_joint_median_delta_mm": {
                name: (float(np.median(values)) if values else None)
                for name, values in proxy_key_values.items()
            },
        },
        "inputs": {
            "rgb": evidence(args.rgb),
            "hawor_npz": evidence(args.hawor_npz),
            "surface_cache": evidence(args.surface_cache),
            "registration": evidence(args.registration),
        },
        "artifacts": {},
        "validation": validations,
        "authority": False,
        "external_ground_truth": False,
    }
    for path in (raw_final, aligned_final, proxy_final, output / "Chips034_Right_Proxy_Skeleton.png", frame_json):
        result["artifacts"][path.name] = evidence(path)
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--session", required=True)
    value.add_argument("--rgb", type=Path, required=True)
    value.add_argument("--hawor-npz", type=Path, required=True)
    value.add_argument("--surface-cache", type=Path, required=True)
    value.add_argument("--depth-dir", type=Path, required=True)
    value.add_argument("--registration", type=Path, required=True)
    value.add_argument("--hand-mask-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--local-tolerance-m", type=float, default=0.02)
    value.add_argument("--proxy-radius", type=int, default=4)
    value.add_argument("--max-frames", type=int, default=0)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
