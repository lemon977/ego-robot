#!/usr/bin/env python3
"""Build a clear PICO/HaWoR/Stereo wrist comparison video.

The three sources have different semantics.  PICO is an engineering hand-
tracking wrist, HaWoR is a learned MANO joint centre, and Stereo is visible
surface optical-Z sampled on the HaWoR wrist ray.  None is external truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIDES = ("left", "right")
COLORS = {
    "pico": (25, 220, 80),
    "hawor": (255, 155, 30),
    "stereo": (235, 50, 235),
    "stereo_hawor": (70, 180, 255),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_text_bgr(image: np.ndarray, rows: list[tuple[tuple[int, int], str, tuple[int, int, int], int]]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    painter = ImageDraw.Draw(pil)
    cache: dict[int, ImageFont.FreeTypeFont] = {}
    for xy, text, color_bgr, size in rows:
        cache.setdefault(size, font(size))
        painter.text(xy, text, fill=color_bgr[::-1], font=cache[size])
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)] * 1000.0
    if not len(valid):
        return {"count": 0, "mean_mm": None, "p50_mm": None, "p95_mm": None, "max_mm": None}
    return {
        "count": int(len(valid)),
        "mean_mm": float(valid.mean()),
        "p50_mm": float(np.quantile(valid, 0.50)),
        "p95_mm": float(np.quantile(valid, 0.95)),
        "max_mm": float(valid.max()),
    }


def draw_marker(image: np.ndarray, uv: np.ndarray, kind: str, color: tuple[int, int, int], label: str) -> None:
    if not np.isfinite(uv).all():
        return
    raw_x, raw_y = np.rint(uv).astype(int)
    x = int(np.clip(raw_x, 30, image.shape[1] - 31))
    y = int(np.clip(raw_y, 30, image.shape[0] - 31))
    outside = (x != raw_x) or (y != raw_y)
    if kind == "circle":
        cv2.circle(image, (x, y), 27, color, 5, cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
    elif kind == "cross":
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_TILTED_CROSS, 42, 6, cv2.LINE_AA)
    else:
        points = np.asarray(((x, y - 22), (x + 22, y), (x, y + 22), (x - 22, y)), np.int32)
        cv2.polylines(image, [points], True, color, 5, cv2.LINE_AA)
    label_x = min(x + 26, image.shape[1] - 190)
    label_y = max(y - 18, 24)
    cv2.putText(image, f"{label}{' [OUT]' if outside else ''}", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def plot_signed_z(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    frame: int,
    limit_mm: float,
) -> None:
    x0, y0, width, height = rect
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (58, 61, 68), 1)
    zero_y = y0 + height // 2
    cv2.line(panel, (x0, zero_y), (x0 + width, zero_y), (135, 135, 135), 1)
    total = len(series[0][1])
    for _, values, color in series:
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            if not np.isfinite(value):
                if len(points) >= 2:
                    cv2.polylines(panel, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)
                points = []
                continue
            x = x0 + int(round(index * width / max(1, total - 1)))
            normalized = np.clip(value * 1000.0 / limit_mm, -1.0, 1.0)
            y = zero_y - int(round(normalized * (height / 2 - 8)))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, color, 2, cv2.LINE_AA)
    current_x = x0 + int(round(frame * width / max(1, total - 1)))
    cv2.line(panel, (current_x, y0), (current_x, y0 + height), (255, 255, 255), 1)


def format_vector(delta: np.ndarray) -> str:
    if not np.isfinite(delta).all():
        return "INVALID"
    values = delta * 1000.0
    return f"ΔXYZ=({values[0]:+.1f},{values[1]:+.1f},{values[2]:+.1f}) mm  |Δ|={np.linalg.norm(values):.1f} mm"


def json_vector_mm(delta: np.ndarray) -> list[float | None]:
    """Encode an XYZ delta without non-standard JSON NaN literals."""
    return [float(value * 1000.0) if np.isfinite(value) else None for value in delta]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--robot-reference", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    session = args.session_root.resolve(strict=True)
    hawor_path = args.hawor_npz.resolve(strict=True)
    depth_root = args.depth_root.resolve(strict=True)
    robot_reference = args.robot_reference.resolve(strict=True) if args.robot_reference else None
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh/no-clobber output required: {output}")
    output.mkdir(parents=True)

    frame_jsons = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in frame_jsons]
    with np.load(hawor_path, allow_pickle=False) as source:
        hawor = {key: np.asarray(source[key]) for key in source.files}
    frame_count = len(rows)
    if frame_count != hawor["joints_3d_camera"].shape[1]:
        raise RuntimeError("PICO/HaWoR frame count mismatch")

    pico_xyz = np.full((2, frame_count, 3), np.nan, np.float64)
    pico_uv = np.full((2, frame_count, 2), np.nan, np.float64)
    for index, row in enumerate(rows):
        for side_index, side in enumerate(SIDES):
            hand = row["entities"]["hands"][side]
            if hand.get("pose_source") != "pico_wrist_joint":
                continue
            pico_xyz[side_index, index] = np.asarray(hand["T_wrist_to_camera"], np.float64)[:3, 3]
            pico_uv[side_index, index] = np.asarray(hand["keypoints_2d"], np.float64)[5]

    all_hawor_xyz = np.asarray(hawor["joints_3d_camera"], np.float64)
    all_hawor_uv = np.asarray(hawor["joints_2d"], np.float64)
    hawor_xyz = all_hawor_xyz[:, :, 0].copy()
    hawor_uv = all_hawor_uv[:, :, 0].copy()
    observed = np.asarray(hawor["observed"], bool)
    hawor_xyz[~observed] = np.nan
    hawor_uv[~observed] = np.nan

    registration_path = depth_root / "REGISTRATION_AUTHORITY.npz"
    with np.load(registration_path, allow_pickle=False) as registration:
        selected_to_depth = np.asarray(registration["H_selected_rgb_to_depth_pixel"], np.float64)
        stereo_to_selected = np.asarray(registration["T_stereo_rectified_camera_to_selected_camera"], np.float64)
    stereo_xyz = np.full_like(hawor_xyz, np.nan)
    stereo_uv = np.full_like(hawor_uv, np.nan)
    stereo_support_joints = np.zeros((2, frame_count), np.int16)
    for frame in range(frame_count):
        with np.load(depth_root / "frames" / f"{frame:06d}.npz", allow_pickle=False) as depth_file:
            depth = np.asarray(depth_file["depth_m"], np.float64)
            valid = np.asarray(depth_file["valid"], bool) & np.isfinite(depth)
            intrinsics = np.asarray(depth_file["scaled_intrinsics"], np.float64)
        for side in range(2):
            if not observed[side, frame]:
                continue
            z_offsets: list[float] = []
            for joint in range(all_hawor_uv.shape[2]):
                uv = all_hawor_uv[side, frame, joint]
                xyz = all_hawor_xyz[side, frame, joint]
                if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
                    continue
                homogeneous = selected_to_depth @ np.asarray((*uv, 1.0))
                uv_depth = homogeneous[:2] / homogeneous[2]
                x, y = np.rint(uv_depth).astype(int)
                if x < 4 or y < 4 or x >= depth.shape[1] - 4 or y >= depth.shape[0] - 4:
                    continue
                patch_depth = depth[y - 4:y + 5, x - 4:x + 5]
                patch_valid = valid[y - 4:y + 5, x - 4:x + 5]
                values = patch_depth[patch_valid]
                if len(values) < 8:
                    continue
                z = float(np.median(values))
                rectified = np.asarray(
                    ((uv_depth[0] - intrinsics[0, 2]) / intrinsics[0, 0] * z,
                     (uv_depth[1] - intrinsics[1, 2]) / intrinsics[1, 1] * z,
                     z, 1.0),
                    np.float64,
                )
                selected = stereo_to_selected @ rectified
                z_offsets.append(float(selected[2] - xyz[2]))
            if not z_offsets:
                continue
            # Stereo sees skin/hand surface, not the wrist joint centre.  Use the
            # robust hand-wide surface Z disagreement as a translation proxy on
            # the HaWoR wrist ray.  This remains a proxy, never joint truth.
            stereo_support_joints[side, frame] = len(z_offsets)
            proxy_z = float(hawor_xyz[side, frame, 2] + np.median(z_offsets))
            if proxy_z <= 0 or hawor_xyz[side, frame, 2] <= 0:
                continue
            stereo_xyz[side, frame] = hawor_xyz[side, frame] * (proxy_z / hawor_xyz[side, frame, 2])
            stereo_uv[side, frame] = hawor_uv[side, frame]

    deltas = {
        "hawor_minus_pico": hawor_xyz - pico_xyz,
        "stereo_minus_pico": stereo_xyz - pico_xyz,
        "stereo_minus_hawor": stereo_xyz - hawor_xyz,
    }
    norms = {key: np.linalg.norm(value, axis=2) for key, value in deltas.items()}
    finite_z = np.concatenate([
        value[..., 2][np.isfinite(value[..., 2])] * 1000.0 for value in deltas.values()
    ])
    z_limit_mm = float(max(50.0, np.ceil(np.quantile(np.abs(finite_z), 0.99) / 25.0) * 25.0))

    raw_video = next(session.glob("CameraRecord_*.mp4"))
    fps = float(rows[0]["metadata"]["fps"])
    video_path = output / "CHIPS023_三路手腕清晰对比_全片.mp4"
    temporary_video = Path(tempfile.mkstemp(prefix="chips023_wrist_", suffix=".mp4", dir="/tmp")[1])
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1920x1080", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", str(temporary_video),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(raw_video))
    keyframes: dict[int, np.ndarray] = {}
    keyframe_ids = {0, frame_count // 5, 2 * frame_count // 5, 3 * frame_count // 5, 4 * frame_count // 5, frame_count - 1}
    try:
        for frame in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"raw video ended at frame {frame}")
            canvas = np.full((1080, 1920, 3), (27, 29, 34), np.uint8)
            canvas[100:1060, :1280] = raw
            for side in range(2):
                suffix = "L" if side == 0 else "R"
                draw_marker(canvas, pico_uv[side, frame] + (0, 100), "circle", COLORS["pico"], f"PICO-{suffix}")
                draw_marker(canvas, hawor_uv[side, frame] + (0, 100), "cross", COLORS["hawor"], f"HaWoR-{suffix}")
                draw_marker(canvas, stereo_uv[side, frame] + (0, 100), "diamond", COLORS["stereo"], f"Stereo-{suffix}")
                finite_points = [
                    point + (0, 100) for point in (pico_uv[side, frame], hawor_uv[side, frame], stereo_uv[side, frame])
                    if np.isfinite(point).all()
                ]
                if len(finite_points) >= 2:
                    cv2.polylines(canvas, [np.rint(finite_points).astype(np.int32)], False, (220, 220, 220), 2, cv2.LINE_AA)

            cv2.rectangle(canvas, (1280, 100), (1919, 1060), (39, 42, 48), -1)
            text_rows = [
                ((22, 17), "Chips023 三路手腕空间对比（420帧）", (255, 255, 255), 34),
                ((720, 25), f"帧 {frame:03d}/{frame_count - 1}   时间 {frame / fps:5.2f}s", (210, 215, 225), 27),
                ((1300, 112), "颜色/符号", (255, 255, 255), 25),
                ((1300, 148), "绿色圆圈：PICO 手部追踪 wrist", COLORS["pico"], 21),
                ((1300, 180), "橙色叉号：HaWoR MANO wrist", COLORS["hawor"], 21),
                ((1300, 212), "紫色菱形：Stereo 表面深度代理", COLORS["stereo"], 21),
                ((1300, 246), "注意：三者都不是外部真值", (95, 195, 255), 22),
            ]
            y = 282
            for side, side_name in enumerate(("左手", "右手")):
                text_rows.append(((1300, y), side_name, (255, 255, 255), 24))
                y += 30
                for label, key, color in (
                    ("HaWoR-PICO", "hawor_minus_pico", COLORS["hawor"]),
                    ("Stereo-PICO", "stereo_minus_pico", COLORS["stereo"]),
                    ("Stereo-HaWoR", "stereo_minus_hawor", COLORS["stereo_hawor"]),
                ):
                    text_rows.append(((1300, y), f"{label}: {format_vector(deltas[key][side, frame])}", color, 17))
                    y += 25
                y += 8
            canvas = draw_text_bgr(canvas, text_rows)

            graph_y = (660, 855)
            for side, top in enumerate(graph_y):
                plot_signed_z(
                    canvas,
                    (1300, top, 590, 145),
                    [
                        ("H-P", deltas["hawor_minus_pico"][side, :, 2], COLORS["hawor"]),
                        ("S-P", deltas["stereo_minus_pico"][side, :, 2], COLORS["stereo"]),
                        ("S-H", deltas["stereo_minus_hawor"][side, :, 2], COLORS["stereo_hawor"]),
                    ],
                    frame,
                    z_limit_mm,
                )
                canvas = draw_text_bgr(canvas, [
                    ((1300, top - 28), f"{('左手','右手')[side]} signed ΔZ 历史曲线（±{z_limit_mm:.0f} mm，中线=0）", (230, 230, 235), 19),
                ])
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            encoder.stdin.write(canvas.tobytes())
            if frame in keyframe_ids:
                keyframes[frame] = cv2.resize(canvas, (960, 540), interpolation=cv2.INTER_AREA)
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg encoder failed with {return_code}")
    shutil.copyfile(temporary_video, video_path)
    temporary_video.unlink()

    montage = np.full((1080, 2880, 3), (27, 29, 34), np.uint8)
    for index, frame in enumerate(sorted(keyframes)):
        row, column = divmod(index, 3)
        montage[row * 540:(row + 1) * 540, column * 960:(column + 1) * 960] = keyframes[frame]
    montage_path = output / "CHIPS023_三路手腕关键帧总览.png"
    cv2.imwrite(str(montage_path), montage)

    metrics: dict[str, object] = {}
    for side, name in enumerate(SIDES):
        metrics[name] = {
            key: {
                "norm": finite_stats(norms[key][side]),
                "signed_dx_mm": finite_stats(np.abs(deltas[key][side, :, 0])),
                "signed_dy_mm": finite_stats(np.abs(deltas[key][side, :, 1])),
                "signed_dz": {
                    **finite_stats(np.abs(deltas[key][side, :, 2])),
                    "mean_mm": (
                        float(np.nanmean(deltas[key][side, :, 2]) * 1000.0)
                        if np.isfinite(deltas[key][side, :, 2]).any() else None
                    ),
                    "p50_mm": (
                        float(np.nanmedian(deltas[key][side, :, 2]) * 1000.0)
                        if np.isfinite(deltas[key][side, :, 2]).any() else None
                    ),
                },
            }
            for key in deltas
        }
    per_frame = []
    for frame in range(frame_count):
        per_frame.append({
            "frame": frame,
            "time_s": frame / fps,
            "left": {key: json_vector_mm(deltas[key][0, frame]) for key in deltas},
            "right": {key: json_vector_mm(deltas[key][1, frame]) for key in deltas},
        })
    metrics_path = output / "CHIPS023_三路手腕逐帧差值.json"
    metrics_path.write_text(json.dumps({
        "schema_version": "three-wrist-clear-comparison-v1",
        "session_id": session.name,
        "frame_count": frame_count,
        "fps": fps,
        "sources": {
            "pico": "PICO/OpenXR hand-tracking wrist joint; engineering reference, not ground truth",
            "hawor": "HaWoR temporal MANO wrist joint centre; learned monocular 3D",
            "stereo": "FoundationStereo visible hand-surface optical-Z: median Z offset over visible HaWoR joint rays, applied to the HaWoR wrist ray; not an anatomical joint",
        },
        "metrics": metrics,
        "per_frame_delta_xyz_mm": per_frame,
        "claim_limit": "Cross-system internal comparison only; no source is external physical truth.",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video_path),
    ], text=True)
    probe_data = json.loads(probe)["streams"][0]
    if int(probe_data["nb_read_frames"]) != frame_count:
        raise RuntimeError("encoded video frame count mismatch")
    result_path = output / "RESULT.json"
    result = {
        "schema_version": "three-wrist-clear-comparison-result-v1",
        "status": "PASS_DEVELOPMENT_CROSS_SYSTEM_VISUALIZATION",
        "session_id": session.name,
        "frame_count": frame_count,
        "fps": fps,
        "video_probe": probe_data,
        "inputs": {
            "raw_video": artifact(raw_video),
            "hawor_npz": artifact(hawor_path),
            "depth_result": artifact(depth_root / "RESULT.json"),
            "registration": artifact(registration_path),
            "robot_reference_video": artifact(robot_reference) if robot_reference else None,
        },
        "outputs": {
            "video": artifact(video_path),
            "montage": artifact(montage_path),
            "metrics": artifact(metrics_path),
        },
        "coverage": {
            "pico_left": int(np.isfinite(pico_xyz[0]).all(axis=1).sum()),
            "pico_right": int(np.isfinite(pico_xyz[1]).all(axis=1).sum()),
            "hawor_left": int(observed[0].sum()),
            "hawor_right": int(observed[1].sum()),
            "stereo_proxy_left": int(np.isfinite(stereo_xyz[0]).all(axis=1).sum()),
            "stereo_proxy_right": int(np.isfinite(stereo_xyz[1]).all(axis=1).sum()),
            "stereo_support_joints_left_p50": float(np.median(stereo_support_joints[0])),
            "stereo_support_joints_right_p50": float(np.median(stereo_support_joints[1])),
        },
        "claim_limit": "Development visualization of cross-system disagreement; PICO is not external truth and Stereo proxy is not an anatomical wrist.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksums = output / "SHA256SUMS.txt"
    files = (video_path, montage_path, metrics_path, result_path)
    checksums.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": result["status"], "outputs": result["outputs"], "coverage": result["coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
