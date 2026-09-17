#!/usr/bin/env python3
"""Evaluate a Poker causal planar donor with a non-oracle runtime gate.

The runtime decision uses only source pixels, currently visible target pixels,
feature geometry, and provenance.  Pixels hidden by the synthetic holdout are
used exclusively for after-the-fact evaluation and cannot affect acceptance.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import math
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
from chaoyang.ops.run_poker_causal_plane_donor_holdout import _holdout, _load_frames


FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _hull_fraction(points: np.ndarray, mask: np.ndarray) -> float:
    bbox = _mask_bbox(mask)
    if bbox is None or len(points) < 3:
        return 0.0
    x0, y0, x1, y1 = bbox
    denominator = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    return float(cv2.contourArea(hull) / denominator)


def evaluate_pair_v2(
    source: np.ndarray,
    target: np.ndarray,
    source_visible_mask: np.ndarray,
    target_visible_mask: np.ndarray,
    evaluation_hole: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return separate runtime and oracle evaluation results.

    ``evaluation_hole`` is excluded from every runtime feature and photometric
    calculation.  It is consulted only after ``runtime_pass`` is frozen.
    """
    source_valid = np.asarray(source_visible_mask) > 0
    target_valid = np.asarray(target_visible_mask) > 0
    hole = np.asarray(evaluation_hole, dtype=bool)
    # ORB descriptors sample a 31x31 neighbourhood.  Exclude a wider ring so
    # changing hidden pixels cannot alter descriptors whose centres are in the
    # runtime-visible region.
    forbidden = cv2.dilate(hole.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    runtime_target_visible = target_valid & ~forbidden
    runtime_target = target.copy()
    runtime_target[forbidden] = 127

    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=5)
    source_keypoints, source_desc = orb.detectAndCompute(source, source_valid.astype(np.uint8) * 255)
    target_keypoints, target_desc = orb.detectAndCompute(runtime_target, runtime_target_visible.astype(np.uint8) * 255)
    matches = []
    if source_desc is not None and target_desc is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_desc, target_desc, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                matches.append(pair[0])

    homography = None
    inlier_mask = None
    source_points = np.zeros((0, 2), np.float32)
    target_points = np.zeros((0, 2), np.float32)
    if len(matches) >= 4:
        source_points = np.float32([source_keypoints[item.queryIdx].pt for item in matches])
        target_points = np.float32([target_keypoints[item.trainIdx].pt for item in matches])
        homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_fraction = inliers / max(len(matches), 1)
    symmetric_p90 = float("inf")
    target_hull_fraction = 0.0
    projected_area_ratio = 0.0
    warped = np.zeros_like(target)
    warped_source_mask = np.zeros_like(target_valid)
    visible_overlap_fraction = 0.0
    visible_abs_median = float("inf")
    visible_abs_p90 = float("inf")

    if homography is not None and inliers >= 4:
        keep = inlier_mask.reshape(-1).astype(bool)
        src_inliers = source_points[keep]
        dst_inliers = target_points[keep]
        forward = cv2.perspectiveTransform(src_inliers.reshape(-1, 1, 2), homography).reshape(-1, 2)
        inverse = np.linalg.inv(homography)
        backward = cv2.perspectiveTransform(dst_inliers.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        symmetric = 0.5 * (np.linalg.norm(forward - dst_inliers, axis=1) + np.linalg.norm(backward - src_inliers, axis=1))
        symmetric_p90 = float(np.quantile(symmetric, 0.90))
        target_hull_fraction = _hull_fraction(dst_inliers, runtime_target_visible)

        source_bbox = _mask_bbox(source_valid)
        if source_bbox is not None:
            x0, y0, x1, y1 = source_bbox
            corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography)
            projected_area = abs(float(cv2.contourArea(projected)))
            source_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
            projected_area_ratio = projected_area / source_area

        warped = cv2.warpPerspective(source, homography, (target.shape[1], target.shape[0]), flags=cv2.INTER_LINEAR)
        warped_source_mask = cv2.warpPerspective(source_valid.astype(np.uint8), homography, (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST) > 0
        overlap = warped_source_mask & runtime_target_visible
        visible_overlap_fraction = float(overlap.sum() / max(runtime_target_visible.sum(), 1))
        if overlap.any():
            absolute = np.abs(warped.astype(np.float32) - runtime_target.astype(np.float32)).mean(axis=2)
            visible_abs_median = float(np.median(absolute[overlap]))
            visible_abs_p90 = float(np.quantile(absolute[overlap], 0.90))

    runtime_checks = {
        "minimum_ratio_matches": len(matches) >= 16,
        "minimum_ransac_inliers": inliers >= 12,
        "minimum_inlier_fraction": inlier_fraction >= 0.35,
        "symmetric_reprojection_p90": symmetric_p90 <= 5.0,
        "target_match_hull_fraction": target_hull_fraction >= 0.04,
        "projected_area_ratio": 0.25 <= projected_area_ratio <= 4.0,
        "visible_overlap_fraction": visible_overlap_fraction >= 0.40,
        "visible_photometric_median": visible_abs_median <= 20.0,
        "visible_photometric_p90": visible_abs_p90 <= 60.0,
    }
    runtime_pass = all(runtime_checks.values())

    # Oracle metrics begin here.  They cannot affect runtime_pass.
    recovered = hole & warped_source_mask if homography is not None else np.zeros_like(hole)
    oracle_coverage = float(recovered.sum() / max(hole.sum(), 1))
    if recovered.any():
        difference = np.abs(warped.astype(np.float32) - target.astype(np.float32))
        oracle_mae = float(difference[recovered].mean())
        oracle_mse = float(np.square(difference[recovered]).mean())
        oracle_psnr = float(20 * math.log10(255.0 / math.sqrt(max(oracle_mse, 1e-12))))
    else:
        difference = np.zeros_like(target, dtype=np.float32)
        oracle_mae = None
        oracle_psnr = None

    metrics = {
        "runtime_decision": "RUNTIME_ACCEPT" if runtime_pass else "RUNTIME_REJECT",
        "runtime_pass": runtime_pass,
        "runtime_checks": runtime_checks,
        "runtime_evidence": {
            "source_keypoints": len(source_keypoints),
            "target_keypoints_visible_only": len(target_keypoints),
            "ratio_matches": len(matches),
            "ransac_inliers": inliers,
            "inlier_fraction": inlier_fraction,
            "symmetric_reprojection_p90_px": symmetric_p90,
            "target_match_hull_fraction": target_hull_fraction,
            "projected_area_ratio": projected_area_ratio,
            "visible_overlap_fraction": visible_overlap_fraction,
            "visible_abs_rgb_median": visible_abs_median,
            "visible_abs_rgb_p90": visible_abs_p90,
        },
        "oracle_evaluation": {
            "holdout_pixels": int(hole.sum()),
            "recovered_pixels": int(recovered.sum()),
            "coverage": oracle_coverage,
            "mae_rgb": oracle_mae,
            "psnr_db": oracle_psnr,
        },
    }
    return metrics, {"hole": hole, "recovered": recovered, "warped": warped, "difference": difference}


def _render(output: Path, rows: list[dict[str, Any]], visuals: list[dict[str, np.ndarray]], frames: list[np.ndarray]) -> Path:
    target_path = output / "Poker245_因果平面Donor_V2运行门与Oracle分离.mp4"
    width, height = 640, 480
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width * 4}x{height}", "-r", "2", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 19)
    for row, visual in zip(rows, visuals):
        target = frames[row["target_frame"]]
        hidden = target.copy(); hidden[visual["hole"]] = 127
        reconstructed = hidden.copy()
        if row["runtime_pass"]:
            reconstructed[visual["recovered"]] = visual["warped"][visual["recovered"]]
        error = np.clip(visual["difference"].mean(axis=2) * 4, 0, 255).astype(np.uint8)
        error = cv2.applyColorMap(error, cv2.COLORMAP_TURBO)
        images = (frames[row["source_frame"]], hidden, reconstructed, error)
        labels = ("历史真实源帧", "目标伪遮挡", "仅运行门通过时写回", "Oracle误差仅评价")
        cells = []
        for image, label in zip(images, labels):
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil); draw.rectangle((0, 0, width, 62), fill=(0, 0, 0))
            draw.text((8, 4), f"{label} | {row['source_frame']}→{row['target_frame']}", font=font, fill=(255,255,255))
            oracle = row["oracle_evaluation"]
            draw.text((8, 33), f"{row['runtime_decision']} | oracle覆盖{oracle['coverage']:.1%}", font=font, fill=(255,230,80))
            cells.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        assert process.stdin is not None
        process.stdin.write(np.concatenate(cells, axis=1).tobytes())
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("comparison video encode failed")
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    manifest = json.loads(args.object_manifest.read_text(encoding="utf-8"))
    if manifest.get("task") != "poker" or manifest.get("session") != "play_cards_0903_245":
        raise RuntimeError("this frozen canary requires Poker245")
    frames = _load_frames(Path(manifest["input"]["selected_rgb"]["path"]), 34)
    pairs = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (28, 33)]
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, np.ndarray]] = []
    for source_frame, target_frame in pairs:
        source_entry = manifest["frames"][source_frame]["physical_instances"]["0"]
        target_entry = manifest["frames"][target_frame]["physical_instances"]["0"]
        if not source_entry["observed"] or not target_entry["observed"]:
            raise RuntimeError("pseudo holdout pair must be directly observed")
        source_mask = cv2.imread(source_entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(target_entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
        hole = _holdout(target_mask)
        metrics, visual = evaluate_pair_v2(frames[source_frame], frames[target_frame], source_mask, target_mask, hole)
        metrics.update(source_frame=source_frame, target_frame=target_frame, causal=source_frame < target_frame)
        rows.append(metrics); visuals.append(visual)

    runtime_accepted = [row for row in rows if row["runtime_pass"]]
    result = {
        "schema_version": "chaoyang-poker-causal-plane-donor-runtime-gate-v2",
        "created_at": now_iso(),
        "task_id": "research_poker_causal_plane_donor_runtime_gate_v2",
        "status": "PASSED_DEVELOPMENT" if runtime_accepted else "FAILED_QUALITY_C",
        "session": "play_cards_0903_245",
        "pairs": rows,
        "summary": {
            "pair_count": len(rows),
            "runtime_accepted_pairs": len(runtime_accepted),
            "runtime_rejected_pairs": len(rows) - len(runtime_accepted),
            "oracle_metrics_used_for_runtime_decision": False,
        },
        "runtime_gate_inputs": [
            "past source RGB and visible-instance mask",
            "current visible target RGB and visible-instance mask outside the evaluation holdout",
            "feature match distribution and RANSAC geometry",
            "visible-overlap photometric consistency",
            "same-session, same-instance, source_frame<target_frame provenance",
        ],
        "oracle_evaluation_only": ["synthetic holdout coverage", "holdout RGB MAE", "holdout PSNR"],
        "evaluation_mode": "VISIBLE_RUNTIME_GATE_PLUS_ORACLE_PSEUDO_OCCLUSION_EVALUATION",
        "training_eligible": False,
        "authority_promoted": False,
        "input": artifact_ref(args.object_manifest),
        "claim_limit": "Runtime-gate separation on visible Poker245 pseudo holdouts only; not real hidden-surface recovery or Clean authority.",
    }
    video = _render(args.output_root, rows, visuals, frames)
    result["review_video"] = artifact_ref(video)
    atomic_json(args.output_root / "METRICS.json", {"pairs": rows, "summary": result["summary"]})
    atomic_json(args.output_root / "RESULT.json", result)
    atomic_json(args.output_root / "RUN_RECEIPT.json", {
        "task_id": result["task_id"], "status": result["status"], "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
    })
    print(json.dumps({"status": result["status"], "summary": result["summary"], "result": artifact_ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
