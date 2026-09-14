#!/usr/bin/env python3
"""Build four-hand and Chips034 right-hand regional difference tables.

The report is an internal HaWoR-vs-Stereo comparison.  Neither input is
external ground truth.  Region labels are MANO-surface proxies derived from
the nearest posed MANO joint group and are not anatomical segmentation GT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_hawor_stereo_surface_consistency_qa import (
    load_role_mask,
    rasterize_visible_surface,
    registered_selected_depth,
)


FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
WINDOWS = ((20, 60), (90, 120), (180, 210))
REGION_NAMES = ("palm", "thumb", "index", "middle", "ring", "pinky")
REGION_ZH = {
    "palm": "手掌",
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
}
CLAIM_LIMIT = (
    "INTERNAL_CROSS_SENSOR_COMPARISON_ONLY_NO_EXTERNAL_GT. Region labels are "
    "MANO-surface proxies, not anatomical ground-truth segmentation."
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


def summary(values_mm: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values_mm, np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"samples": 0, "signed_mean_mm": None, "signed_p50_mm": None, "abs_mae_mm": None, "abs_p50_mm": None, "abs_p95_mm": None}
    absolute = np.abs(values)
    return {
        "samples": int(values.size),
        "signed_mean_mm": float(np.mean(values)),
        "signed_p50_mm": float(np.median(values)),
        "abs_mae_mm": float(np.mean(absolute)),
        "abs_p50_mm": float(np.median(absolute)),
        "abs_p95_mm": float(np.quantile(absolute, 0.95)),
    }


def longest_run(values: list[float], threshold: float) -> dict[str, int]:
    best_start = best_end = -1
    start: int | None = None
    for index, value in enumerate(values + [float("inf")]):
        if index < len(values) and value < threshold:
            if start is None:
                start = index
        elif start is not None:
            if best_start < 0 or index - start > best_end - best_start + 1:
                best_start, best_end = start, index - 1
            start = None
    return {"start_frame": best_start, "end_frame": best_end, "length_frames": best_end - best_start + 1 if best_start >= 0 else 0}


def vertex_regions(vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    nearest = np.argmin(np.linalg.norm(vertices[:, None, :] - joints[None, :, :], axis=2), axis=1)
    regions = np.zeros(len(vertices), np.int16)
    regions[np.isin(nearest, (1, 2, 3, 4))] = 1
    regions[np.isin(nearest, (6, 7, 8))] = 2
    regions[np.isin(nearest, (10, 11, 12))] = 3
    regions[np.isin(nearest, (14, 15, 16))] = 4
    regions[np.isin(nearest, (18, 19, 20))] = 5
    return regions


def triangle_regions(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    labels = vertex_labels[faces]
    result = np.empty(len(faces), np.int16)
    for index, row in enumerate(labels):
        counts = np.bincount(row, minlength=len(REGION_NAMES))
        result[index] = int(np.argmax(counts))
    return result


def analyze_windows(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.hawor_npz, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.surface_cache, allow_pickle=False) as archive:
        left_vertices = np.asarray(archive["vertices_left"])
        right_vertices = np.asarray(archive["vertices_right"])
        left_faces = np.asarray(archive["faces_left"])
        right_faces = np.asarray(archive["faces_right"])
    with np.load(args.registration, allow_pickle=False) as archive:
        registration = {key: np.asarray(archive[key]) for key in archive.files}
    selected_k = np.asarray(registration["selected_rgb_intrinsics"], np.float64)
    frames = np.asarray(hawor["original_frame_indices"], np.int64)
    depth_files = sorted(args.depth_dir.glob("frames/*.npz"))
    if len(depth_files) != len(frames):
        raise RuntimeError("Depth and HaWoR frame count mismatch")
    target_frames = {frame for start, end in WINDOWS for frame in range(start, end + 1)}
    kernel = np.ones((7, 7), np.uint8)
    accumulators: dict[str, dict[str, Any]] = {}
    for start, end in WINDOWS:
        name = f"{start:03d}-{end:03d}"
        accumulators[name] = {
            "start_frame": start,
            "end_frame": end,
            "frame_median_offsets_mm": [],
            "whole_raw": [],
            "whole_aligned": [],
            "regions_raw": {region: [] for region in REGION_NAMES},
            "regions_aligned": {region: [] for region in REGION_NAMES},
            "support": {region: 0 for region in REGION_NAMES},
            "retained": {region: 0 for region in REGION_NAMES},
        }

    for local, (frame, depth_path) in enumerate(zip(frames.tolist(), depth_files, strict=True)):
        if frame not in target_frames:
            continue
        with np.load(depth_path, allow_pickle=False) as archive:
            if int(archive["frame_id"]) != frame:
                raise RuntimeError(f"Depth frame identity mismatch at {frame}")
            stereo_z, stereo_valid, stereo_local = registered_selected_depth(
                archive["depth_m"], archive["valid"], registration, (1280, 960), args.local_tolerance_m
            )
        right_region_per_vertex = vertex_regions(right_vertices[local], hawor["joints_3d_camera"][1, local])
        right_region_per_triangle = triangle_regions(right_region_per_vertex, right_faces)
        triangles = np.concatenate((left_vertices[local][left_faces], right_vertices[local][right_faces])).astype(np.float64)
        labels = np.concatenate((np.full(len(left_faces), -1, np.int16), right_region_per_triangle))
        mano_z, region_image = rasterize_visible_surface(
            triangles, labels,
            float(selected_k[0, 0]), float(selected_k[1, 1]),
            float(selected_k[0, 2]), float(selected_k[1, 2]), 1280, 960,
        )
        right_visible = region_image >= 0
        role = load_role_mask(args.hand_mask_root, None, "right", frame, (960, 1280))
        eroded_mano = cv2.erode(right_visible.astype(np.uint8), kernel).astype(bool)
        eroded_role = cv2.erode(role.astype(np.uint8), kernel).astype(bool)
        support_all = eroded_mano & eroded_role
        retained_all = support_all & stereo_valid & stereo_local
        raw_values = (mano_z - stereo_z) * 1000.0
        if np.count_nonzero(retained_all) < 100:
            raise RuntimeError(f"insufficient retained support at frame {frame}")
        frame_median = float(np.median(raw_values[retained_all]))
        window_name = next(f"{start:03d}-{end:03d}" for start, end in WINDOWS if start <= frame <= end)
        bucket = accumulators[window_name]
        bucket["frame_median_offsets_mm"].append(frame_median)
        bucket["whole_raw"].append(raw_values[retained_all].astype(np.float32))
        bucket["whole_aligned"].append((raw_values[retained_all] - frame_median).astype(np.float32))
        for region_index, region_name in enumerate(REGION_NAMES):
            region_support = support_all & (region_image == region_index)
            region_retained = retained_all & (region_image == region_index)
            bucket["support"][region_name] += int(np.count_nonzero(region_support))
            bucket["retained"][region_name] += int(np.count_nonzero(region_retained))
            if np.any(region_retained):
                values = raw_values[region_retained]
                bucket["regions_raw"][region_name].append(values.astype(np.float32))
                bucket["regions_aligned"][region_name].append((values - frame_median).astype(np.float32))

    windows: dict[str, Any] = {}
    for name, bucket in accumulators.items():
        raw = np.concatenate(bucket["whole_raw"])
        aligned = np.concatenate(bucket["whole_aligned"])
        offsets = np.asarray(bucket["frame_median_offsets_mm"], np.float64)
        region_rows: dict[str, Any] = {}
        for region in REGION_NAMES:
            raw_region = np.concatenate(bucket["regions_raw"][region]) if bucket["regions_raw"][region] else np.array([])
            aligned_region = np.concatenate(bucket["regions_aligned"][region]) if bucket["regions_aligned"][region] else np.array([])
            support = int(bucket["support"][region])
            retained = int(bucket["retained"][region])
            region_rows[region] = {
                "region_semantics": "MANO surface proxy grouped by nearest posed MANO joint; not anatomical GT",
                "raw": summary(raw_region),
                "after_per_frame_median_z_correction": summary(aligned_region),
                "stereo_valid_coverage": float(retained / max(1, support)),
                "support_pixels": support,
                "retained_pixels": retained,
            }
        windows[name] = {
            "start_frame": bucket["start_frame"],
            "end_frame": bucket["end_frame"],
            "frame_count": len(offsets),
            "whole_hand_frame_median_offset_mm": {
                "mean": float(np.mean(offsets)),
                "median": float(np.median(offsets)),
                "p05": float(np.quantile(offsets, 0.05)),
                "p95": float(np.quantile(offsets, 0.95)),
            },
            "whole_hand_raw": summary(raw),
            "whole_hand_after_per_frame_median_z_correction": summary(aligned),
            "regions": region_rows,
        }
    return {
        "schema_version": "hawor-stereo-three-window-regional-analysis-v1",
        "session": args.session,
        "side": "right",
        "windows": windows,
        "claim_limit": CLAIM_LIMIT,
        "inputs": {
            "hawor_npz": evidence(args.hawor_npz),
            "surface_cache": evidence(args.surface_cache),
            "registration": evidence(args.registration),
        },
    }


def four_hand_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    mapping = (
        ("get_potato_chips_0902_034", "Chips034", "left", "左"),
        ("get_potato_chips_0902_034", "Chips034", "right", "右"),
        ("play_cards_0902_042", "Poker042", "left", "左"),
        ("play_cards_0902_042", "Poker042", "right", "右"),
    )
    for session, short, side, side_zh in mapping:
        metrics = aggregate["sessions"][session][f"{side}_frame_balanced_mm"]
        rows.append({
            "section": "four_hand_summary",
            "session": short,
            "side": side_zh,
            "signed_bias_mm": metrics["signed_bias"],
            "abs_mae_mm": metrics["abs_mae"],
            "abs_p50_mm": metrics["abs_p50"],
            "abs_p95_mm": metrics["abs_p95"],
            "claim": "HaWoR-vs-Stereo internal difference, not error vs GT",
        })
    return rows


def build_rows(aggregate: dict[str, Any], frame_metrics: dict[str, Any], regional: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = four_hand_rows(aggregate)
    right = [float(row["sides"]["right"]["signed_bias_mean_mm"]) for row in frame_metrics["rows"]]
    run = longest_run(right, -50.0)
    rows.append({
        "section": "chips_right_global_counts",
        "session": "Chips034",
        "side": "右",
        "frames": len(right),
        "negative_frames": sum(value < 0 for value in right),
        "below_minus_30_frames": sum(value < -30 for value in right),
        "below_minus_50_frames": sum(value < -50 for value in right),
        "longest_below_minus_50": f"{run['start_frame']}-{run['end_frame']} ({run['length_frames']} frames)",
        "claim": "Counts use per-frame signed mean delta-Z",
    })
    for start in range(0, len(right), 30):
        chunk = np.asarray(right[start:min(start + 30, len(right))], np.float64)
        rows.append({
            "section": "chips_right_per_second",
            "session": "Chips034",
            "side": "右",
            "start_frame": start,
            "end_frame": min(start + 29, len(right) - 1),
            "time_s": f"{start / 30:.1f}-{min(start + 30, len(right)) / 30:.1f}",
            "signed_bias_mm": float(np.mean(chunk)),
            "signed_p50_mm": float(np.median(chunk)),
            "claim": "Frame-mean delta-Z averaged within one-second bin",
        })
    for window, value in regional["windows"].items():
        whole = value["whole_hand_after_per_frame_median_z_correction"]
        rows.append({
            "section": "three_window_whole_hand",
            "session": "Chips034",
            "side": "右",
            "window": window,
            "median_offset_mm": value["whole_hand_frame_median_offset_mm"]["median"],
            "aligned_abs_mae_mm": whole["abs_mae_mm"],
            "aligned_abs_p50_mm": whole["abs_p50_mm"],
            "aligned_abs_p95_mm": whole["abs_p95_mm"],
            "claim": "Per-frame median-Z removed before pooling residuals",
        })
        for region, region_value in value["regions"].items():
            aligned = region_value["after_per_frame_median_z_correction"]
            rows.append({
                "section": "three_window_region",
                "session": "Chips034",
                "side": "右",
                "window": window,
                "region": region,
                "region_zh": REGION_ZH[region],
                "aligned_signed_p50_mm": aligned["signed_p50_mm"],
                "aligned_abs_mae_mm": aligned["abs_mae_mm"],
                "aligned_abs_p50_mm": aligned["abs_p50_mm"],
                "aligned_abs_p95_mm": aligned["abs_p95_mm"],
                "stereo_valid_coverage": region_value["stereo_valid_coverage"],
                "samples": aligned["samples"],
                "claim": "MANO nearest-joint surface region proxy; not anatomical GT",
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw_table(draw: ImageDraw.ImageDraw, xy: tuple[int, int], widths: list[int], headers: list[str], rows: list[list[str]], font: ImageFont.FreeTypeFont, row_height: int = 44) -> int:
    x0, y0 = xy
    total_width = sum(widths)
    draw.rectangle((x0, y0, x0 + total_width, y0 + row_height), fill="#17324d")
    x = x0
    for header, width in zip(headers, widths, strict=True):
        draw.text((x + 8, y0 + 8), header, font=font, fill="white")
        x += width
    y = y0 + row_height
    for row_index, row in enumerate(rows):
        fill = "#f2f6f9" if row_index % 2 == 0 else "#e5edf3"
        draw.rectangle((x0, y, x0 + total_width, y + row_height), fill=fill)
        x = x0
        for value, width in zip(row, widths, strict=True):
            draw.text((x + 8, y + 8), value, font=font, fill="#18222d")
            x += width
        y += row_height
    draw.rectangle((x0, y0, x0 + total_width, y), outline="#7790a3", width=2)
    return y


def write_png(path: Path, aggregate: dict[str, Any], frame_metrics: dict[str, Any], regional: dict[str, Any]) -> None:
    image = Image.new("RGB", (2200, 2500), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(FONT), 54)
    subtitle = ImageFont.truetype(str(FONT), 28)
    body = ImageFont.truetype(str(FONT), 23)
    small = ImageFont.truetype(str(FONT), 19)
    draw.text((70, 48), "HaWoR MANO 表面 vs FoundationStereo 可见表面", font=title, fill="#102a43")
    draw.text((70, 120), "跨传感器内部差异，不是任一系统相对真实世界的误差", font=subtitle, fill="#b42318")

    draw.text((70, 185), "四手全片结果（mm）", font=subtitle, fill="#17324d")
    table_rows = []
    for row in four_hand_rows(aggregate):
        table_rows.append([
            row["session"], row["side"], f"{row['signed_bias_mm']:+.2f}",
            f"{row['abs_mae_mm']:.2f}", f"{row['abs_p50_mm']:.2f}", f"{row['abs_p95_mm']:.2f}",
        ])
    y = draw_table(draw, (70, 235), [300, 150, 300, 300, 300, 300], ["会话", "手", "signed bias", "abs MAE", "abs P50", "abs P95"], table_rows, body)

    right = [float(row["sides"]["right"]["signed_bias_mean_mm"]) for row in frame_metrics["rows"]]
    run = longest_run(right, -50.0)
    draw.rounded_rectangle((70, y + 35, 2050, y + 180), radius=20, fill="#fff1e8", outline="#ed7d31", width=3)
    draw.text((95, y + 55), "Chips034 右手：持续、同方向的整手 Z 分歧", font=subtitle, fill="#9a3412")
    draw.text(
        (95, y + 105),
        f"293帧：负偏 {sum(v < 0 for v in right)}；< -30 mm {sum(v < -30 for v in right)}；< -50 mm {sum(v < -50 for v in right)}；最长连续 < -50 mm：{run['start_frame']}–{run['end_frame']}（{run['length_frames']}帧）",
        font=body, fill="#5f2b0a",
    )
    y += 220

    draw.text((70, y), "每秒 signed bias（每帧像素均值再做1秒平均）", font=subtitle, fill="#17324d")
    second_rows = []
    for start in range(0, len(right), 30):
        chunk = np.asarray(right[start:min(start + 30, len(right))])
        second_rows.append([f"{start / 30:.1f}–{min(start + 30, len(right)) / 30:.1f}", f"{start}–{min(start + 29, len(right)-1)}", f"{np.mean(chunk):+.2f}", f"{np.median(chunk):+.2f}"])
    y = draw_table(draw, (70, y + 50), [260, 260, 360, 360], ["时间(s)", "帧", "mean bias(mm)", "median(mm)"], second_rows, small, 38)
    y += 45

    draw.text((70, y), "三个重点时间窗：每帧移除整手 median-Z 后", font=subtitle, fill="#17324d")
    window_rows = []
    for name, value in regional["windows"].items():
        aligned = value["whole_hand_after_per_frame_median_z_correction"]
        window_rows.append([
            name,
            f"{value['whole_hand_frame_median_offset_mm']['median']:+.2f}",
            f"{aligned['abs_mae_mm']:.2f}",
            f"{aligned['abs_p50_mm']:.2f}",
            f"{aligned['abs_p95_mm']:.2f}",
        ])
    y = draw_table(draw, (70, y + 50), [300, 360, 330, 330, 330], ["帧窗", "median Z offset", "residual MAE", "residual P50", "residual P95"], window_rows, body)
    y += 45

    draw.text((70, y), "手掌/五指局部 residual（MAE / P95 / Stereo valid coverage）", font=subtitle, fill="#17324d")
    region_rows = []
    for name, value in regional["windows"].items():
        for region in REGION_NAMES:
            entry = value["regions"][region]
            aligned = entry["after_per_frame_median_z_correction"]
            region_rows.append([
                name, REGION_ZH[region],
                f"{aligned['abs_mae_mm']:.1f}" if aligned["abs_mae_mm"] is not None else "N/A",
                f"{aligned['abs_p95_mm']:.1f}" if aligned["abs_p95_mm"] is not None else "N/A",
                f"{100 * entry['stereo_valid_coverage']:.1f}%",
            ])
    y = draw_table(draw, (70, y + 50), [240, 220, 300, 300, 360], ["帧窗", "区域", "residual MAE", "residual P95", "valid coverage"], region_rows, small, 34)
    draw.text((70, 2440), "区域为 MANO 最近关节分组的表面代理，不是解剖分割真值；完整定义与样本数见 CSV/JSON。", font=small, fill="#7a1f1f")
    image.save(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(args.aggregate_result.read_text(encoding="utf-8"))
    frame_metrics = json.loads(args.frame_metrics.read_text(encoding="utf-8"))
    regional = analyze_windows(args)
    regional_path = output / "CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json"
    regional_path.write_text(json.dumps(regional, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = build_rows(aggregate, frame_metrics, regional)
    csv_path = output / "HAWOR_STEREO_DIFFERENCE_TABLE.csv"
    png_path = output / "HAWOR_STEREO_DIFFERENCE_TABLE.png"
    write_csv(csv_path, rows)
    write_png(png_path, aggregate, frame_metrics, regional)
    result = {
        "schema_version": "hawor-stereo-difference-report-result-v1",
        "status": "PASS_INTERNAL_DIAGNOSTIC_NO_EXTERNAL_GT",
        "claim_limit": CLAIM_LIMIT,
        "artifacts": {
            path.name: evidence(path) for path in (regional_path, csv_path, png_path)
        },
        "inputs": {
            "aggregate_result": evidence(args.aggregate_result),
            "frame_metrics": evidence(args.frame_metrics),
        },
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--session", required=True)
    value.add_argument("--aggregate-result", type=Path, required=True)
    value.add_argument("--frame-metrics", type=Path, required=True)
    value.add_argument("--hawor-npz", type=Path, required=True)
    value.add_argument("--surface-cache", type=Path, required=True)
    value.add_argument("--depth-dir", type=Path, required=True)
    value.add_argument("--registration", type=Path, required=True)
    value.add_argument("--hand-mask-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--local-tolerance-m", type=float, default=0.02)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
