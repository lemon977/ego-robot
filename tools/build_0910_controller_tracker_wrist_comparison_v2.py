#!/usr/bin/env python3
"""Build a clear Controller/HaWoR/Stereo wrist comparison for 0910_001.

Controller wrist is the recorded controller 6D pose composed with the session's
fixed controller-to-wrist calibration.  HaWoR is an independent visual tracker.
Stereo is visible-surface optical-Z sampled on the Controller wrist ray.  None
is external ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tools.build_three_wrist_clear_comparison_v1 import (
    COLORS as BASE_COLORS,
    artifact,
    draw_marker,
    draw_text_bgr,
    finite_stats,
    format_vector,
    json_vector_mm,
    plot_signed_z,
    sha256,
)


SIDES = ("left", "right")
COLORS = {**BASE_COLORS, "hawor": (30, 155, 255)}  # BGR: true orange


def project_equidis(points: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    x, y, z = points.T
    radius = np.hypot(x, y)
    theta = np.arctan2(radius, z)
    direction_x = np.divide(x, radius, out=np.zeros_like(x), where=radius > 1e-12)
    direction_y = np.divide(y, radius, out=np.zeros_like(y), where=radius > 1e-12)
    theta2 = theta * theta
    radial = np.ones_like(theta)
    power = theta2.copy()
    for coefficient in distortion[:6]:
        radial += coefficient * power
        power *= theta2
    distorted_x = theta * radial * direction_x
    distorted_y = theta * radial * direction_y
    radius2 = distorted_x * distorted_x + distorted_y * distorted_y
    p1, p2 = distortion[6:]
    source_x, source_y = distorted_x.copy(), distorted_y.copy()
    distorted_x = source_x + 2 * p1 * source_x * source_y + p2 * (radius2 + 2 * source_x * source_x)
    distorted_y = source_y + p1 * (radius2 + 2 * source_y * source_y) + 2 * p2 * source_x * source_y
    return np.column_stack((
        intrinsics[0, 0] * distorted_x + intrinsics[0, 2],
        intrinsics[1, 1] * distorted_y + intrinsics[1, 2],
    ))


def signed_component_stats(values: np.ndarray) -> dict[str, float | int | None]:
    valid = np.asarray(values, np.float64)
    valid = valid[np.isfinite(valid)] * 1000.0
    if not len(valid):
        return {"count": 0, "mean_mm": None, "p50_mm": None, "p05_mm": None, "p95_mm": None}
    return {
        "count": int(len(valid)), "mean_mm": float(valid.mean()), "p50_mm": float(np.median(valid)),
        "p05_mm": float(np.quantile(valid, 0.05)), "p95_mm": float(np.quantile(valid, 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    session = args.session_root.resolve(strict=True)
    hawor_path = args.hawor_npz.resolve(strict=True)
    depth_root = args.depth_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh/no-clobber output required: {output}")
    output.mkdir(parents=True)

    frame_jsons = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in frame_jsons]
    frame_count = len(rows)
    fps = float(rows[0]["metadata"]["fps"])
    intrinsics = np.asarray(rows[0]["metadata"]["k"], np.float64)
    distortion = np.asarray(rows[0]["metadata"]["d"], np.float64)
    controller_xyz = np.full((2, frame_count, 3), np.nan, np.float64)
    for frame, row in enumerate(rows):
        for side_index, side in enumerate(SIDES):
            hand = row["entities"]["hands"][side]
            controller_xyz[side_index, frame] = np.asarray(hand["T_wrist_to_camera"], np.float64)[:3, 3]
    controller_uv = np.stack([
        np.stack([project_equidis(controller_xyz[side, frame][None], intrinsics, distortion)[0]
                  for frame in range(frame_count)])
        for side in range(2)
    ])

    with np.load(hawor_path, allow_pickle=False) as source:
        hawor = {key: np.asarray(source[key]) for key in source.files}
    if hawor["joints_3d_camera"].shape[1] != frame_count:
        raise RuntimeError("HaWoR frame count mismatch")
    hawor_xyz = np.asarray(hawor["joints_3d_camera"][:, :, 0], np.float64)
    hawor_uv = np.asarray(hawor["joints_2d"][:, :, 0], np.float64)
    observed = np.asarray(hawor["observed"], bool)
    hawor_xyz[~observed] = np.nan
    hawor_uv[~observed] = np.nan

    with np.load(depth_root / "frames/000000.npz", allow_pickle=False) as first_depth:
        rectification = np.asarray(first_depth["rectification_rotation_right"], np.float64)
        depth_intrinsics = np.asarray(first_depth["scaled_intrinsics"], np.float64)
    stereo_xyz = np.full_like(controller_xyz, np.nan)
    stereo_uv = np.full_like(controller_uv, np.nan)
    stereo_valid_pixels = np.zeros((2, frame_count), np.int16)
    for frame in range(frame_count):
        with np.load(depth_root / "frames" / f"{frame:06d}.npz", allow_pickle=False) as source:
            depth = np.asarray(source["depth_m"], np.float64)
            valid = np.asarray(source["valid"], bool) & np.isfinite(depth)
        for side in range(2):
            rectified_ray = rectification @ controller_xyz[side, frame]
            if not np.isfinite(rectified_ray).all() or rectified_ray[2] <= 0:
                continue
            uv = np.asarray((
                depth_intrinsics[0, 0] * rectified_ray[0] / rectified_ray[2] + depth_intrinsics[0, 2],
                depth_intrinsics[1, 1] * rectified_ray[1] / rectified_ray[2] + depth_intrinsics[1, 2],
            ))
            x, y = np.rint(uv).astype(int)
            if x < 5 or y < 5 or x >= depth.shape[1] - 5 or y >= depth.shape[0] - 5:
                continue
            patch_depth = depth[y - 5:y + 6, x - 5:x + 6]
            patch_valid = valid[y - 5:y + 6, x - 5:x + 6]
            values = patch_depth[patch_valid]
            if len(values) < 12:
                continue
            stereo_valid_pixels[side, frame] = len(values)
            z = float(np.median(values))
            point_rectified = np.asarray((
                (uv[0] - depth_intrinsics[0, 2]) / depth_intrinsics[0, 0] * z,
                (uv[1] - depth_intrinsics[1, 2]) / depth_intrinsics[1, 1] * z,
                z,
            ))
            stereo_xyz[side, frame] = rectification.T @ point_rectified
            stereo_uv[side, frame] = project_equidis(stereo_xyz[side, frame][None], intrinsics, distortion)[0]

    deltas = {
        "hawor_minus_controller": hawor_xyz - controller_xyz,
        "stereo_minus_controller": stereo_xyz - controller_xyz,
        "stereo_minus_hawor": stereo_xyz - hawor_xyz,
    }
    norms = {key: np.linalg.norm(value, axis=2) for key, value in deltas.items()}
    finite_z = np.concatenate([value[..., 2][np.isfinite(value[..., 2])] * 1000 for value in deltas.values()])
    z_limit_mm = float(max(50, np.ceil(np.quantile(np.abs(finite_z), 0.99) / 25) * 25))

    raw_video = session / "CameraRecord_play_cards_0910_001.mp4"
    video_path = output / "PLAY_CARDS_0910_001_Controller_HaWoR_Stereo三路手腕.mp4"
    temporary_video = Path(tempfile.mkstemp(prefix="cards0910_wrist_", suffix=".mp4", dir="/tmp")[1])
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", str(temporary_video),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    capture = cv2.VideoCapture(str(raw_video))
    keyframes: dict[int, np.ndarray] = {}
    keyframe_ids = set(np.linspace(0, frame_count - 1, 6, dtype=int).tolist())
    try:
        for frame in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"raw video ended at frame {frame}")
            canvas = np.full((1080, 1920, 3), (27, 29, 34), np.uint8)
            canvas[100:1060, :1280] = raw
            for side in range(2):
                suffix = "L" if side == 0 else "R"
                draw_marker(canvas, controller_uv[side, frame] + (0, 100), "circle", COLORS["pico"], f"CTRL-WRIST-{suffix}")
                draw_marker(canvas, hawor_uv[side, frame] + (0, 100), "cross", COLORS["hawor"], f"HaWoR-{suffix}")
                draw_marker(canvas, stereo_uv[side, frame] + (0, 100), "diamond", COLORS["stereo"], f"StereoSurface-{suffix}")
            cv2.rectangle(canvas, (1280, 100), (1919, 1060), (39, 42, 48), -1)
            rows_text = [
                ((22, 17), "PlayCards0910_001 Controller / HaWoR / Stereo 三路手腕", (255, 255, 255), 31),
                ((940, 25), f"帧 {frame:03d}/{frame_count - 1}  {frame / fps:5.2f}s", (215, 220, 230), 25),
                ((1300, 112), "颜色/语义", (255, 255, 255), 25),
                ((1300, 148), "绿色圆：Controller×固定外参 wrist", COLORS["pico"], 19),
                ((1300, 179), "橙色叉：HaWoR 视觉 tracker wrist", COLORS["hawor"], 19),
                ((1300, 210), "紫色菱形：Controller射线 Stereo表面Z", COLORS["stereo"], 19),
                ((1300, 243), "三者均非外部真值；Stereo不是关节", (95, 195, 255), 20),
                ((1300, 270), f"HaWoR覆盖 L={observed[0].sum()}/{frame_count} R={observed[1].sum()}/{frame_count}", (95, 195, 255), 19),
            ]
            y = 310
            for side, name in enumerate(("左手", "右手")):
                rows_text.append(((1300, y), name, (255, 255, 255), 24))
                y += 30
                for label, key, color in (
                    ("HaWoR-CTRL", "hawor_minus_controller", COLORS["hawor"]),
                    ("Stereo-CTRL", "stereo_minus_controller", COLORS["stereo"]),
                    ("Stereo-HaWoR", "stereo_minus_hawor", COLORS["stereo_hawor"]),
                ):
                    rows_text.append(((1300, y), f"{label}: {format_vector(deltas[key][side, frame])}", color, 17))
                    y += 25
                y += 8
            canvas = draw_text_bgr(canvas, rows_text)
            for side, top in enumerate((660, 855)):
                plot_signed_z(
                    canvas, (1300, top, 590, 145),
                    [
                        ("H-C", deltas["hawor_minus_controller"][side, :, 2], COLORS["hawor"]),
                        ("S-C", deltas["stereo_minus_controller"][side, :, 2], COLORS["stereo"]),
                        ("S-H", deltas["stereo_minus_hawor"][side, :, 2], COLORS["stereo_hawor"]),
                    ], frame, z_limit_mm,
                )
                canvas = draw_text_bgr(canvas, [
                    ((1300, top - 28), f"{('左手','右手')[side]} signed ΔZ（±{z_limit_mm:.0f} mm，中线=0）", (230, 230, 235), 19),
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
    if return_code:
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg encoder failed: {return_code}")
    shutil.copyfile(temporary_video, video_path)
    temporary_video.unlink()

    montage = np.full((1080, 2880, 3), (27, 29, 34), np.uint8)
    for index, frame in enumerate(sorted(keyframes)):
        row, column = divmod(index, 3)
        montage[row * 540:(row + 1) * 540, column * 960:(column + 1) * 960] = keyframes[frame]
    montage_path = output / "PLAY_CARDS_0910_001_三路手腕关键帧.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError("failed to write montage")

    metrics: dict[str, object] = {}
    for side, side_name in enumerate(SIDES):
        metrics[side_name] = {
            key: {"norm": finite_stats(norms[key][side]), "signed_dz": signed_component_stats(deltas[key][side, :, 2])}
            for key in deltas
        }
    per_frame = [
        {"frame": frame, "time_s": frame / fps,
         "left": {key: json_vector_mm(value[0, frame]) for key, value in deltas.items()},
         "right": {key: json_vector_mm(value[1, frame]) for key, value in deltas.items()}}
        for frame in range(frame_count)
    ]
    metrics_path = output / "PLAY_CARDS_0910_001_三路手腕逐帧差值.json"
    metrics_payload = {
        "schema_version": "controller-hawor-stereo-wrist-comparison-v2", "session_id": session.name,
        "frame_count": frame_count, "fps": fps,
        "sources": {
            "controller_wrist": "Recorded Controller 6D composed with same-session fixed controller-to-wrist calibration",
            "hawor_wrist": "Independent HaWoR monocular visual tracker MANO wrist; missing detections remain invalid",
            "stereo_surface": "FoundationStereo visible-surface optical-Z sampled on the Controller wrist ray; not anatomical wrist",
        },
        "metrics": metrics, "per_frame_delta_xyz_mm": per_frame,
        "claim_limit": "Cross-system engineering comparison only; Controller may be temporally stable but is not external wrist ground truth.",
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video_path),
    ], text=True))["streams"][0]
    if int(probe["nb_read_frames"]) != frame_count:
        raise RuntimeError("video frame count mismatch")
    result_path = output / "RESULT.json"
    result = {
        "schema_version": "controller-hawor-stereo-wrist-comparison-result-v2",
        "status": "PASS_DEVELOPMENT_CROSS_SYSTEM_VISUALIZATION", "session_id": session.name,
        "frame_count": frame_count, "fps": fps, "video_probe": probe,
        "coverage": {
            "controller_left": frame_count, "controller_right": frame_count,
            "hawor_left": int(observed[0].sum()), "hawor_right": int(observed[1].sum()),
            "stereo_left": int(np.isfinite(stereo_xyz[0]).all(axis=1).sum()),
            "stereo_right": int(np.isfinite(stereo_xyz[1]).all(axis=1).sum()),
            "stereo_patch_valid_p50_left": float(np.median(stereo_valid_pixels[0])),
            "stereo_patch_valid_p50_right": float(np.median(stereo_valid_pixels[1])),
        },
        "inputs": {"raw_video": artifact(raw_video), "hawor_npz": artifact(hawor_path),
                   "depth_result": artifact(depth_root / "RESULT.json")},
        "outputs": {"video": artifact(video_path), "montage": artifact(montage_path), "metrics": artifact(metrics_path)},
        "claim_limit": "Development visualization only; stability is not accuracy and no source is external truth.",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path = output / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (video_path, montage_path, metrics_path, result_path)),
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "coverage": result["coverage"], "outputs": result["outputs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
