#!/usr/bin/env python3
"""Evaluate causal planar donors on visible Poker245 frames.

The experiment programmatically hides an interior patch that is visible in the
raw target frame, estimates a source-to-target homography without using target
pixels inside that patch, and compares the causal warp with the saved raw
pixels.  It tests whether historical donor appearance can be registered in
principle; it is not a deployable hidden-surface result or Object6D authority.
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
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _load_frames(video: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"video decoded {len(frames)}/{count}")
    return frames


def _holdout(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    ys, xs = np.where(binary)
    if not len(xs):
        return np.zeros_like(binary)
    x0, x1 = int(np.quantile(xs, 0.30)), int(np.quantile(xs, 0.70))
    y0, y1 = int(np.quantile(ys, 0.30)), int(np.quantile(ys, 0.70))
    result = np.zeros_like(binary)
    result[y0:y1 + 1, x0:x1 + 1] = True
    return result & binary


def evaluate_pair(source: np.ndarray, target: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_valid = source_mask > 0
    target_valid = target_mask > 0
    hole = _holdout(target_mask)
    match_target = target_valid & ~cv2.dilate(hole.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=5)
    source_keypoints, source_desc = orb.detectAndCompute(source, source_valid.astype(np.uint8) * 255)
    target_keypoints, target_desc = orb.detectAndCompute(target, match_target.astype(np.uint8) * 255)
    good = []
    if source_desc is not None and target_desc is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_desc, target_desc, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
    status = "OBSERVED_BUT_UNREGISTERED"
    homography = None
    inliers = 0
    if len(good) >= 4:
        source_points = np.float32([source_keypoints[item.queryIdx].pt for item in good])
        target_points = np.float32([target_keypoints[item.trainIdx].pt for item in good])
        homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        status = "REGISTERED_BUT_REJECTED"
    geometric_accepted = homography is not None and inliers >= 20 and inliers / max(len(good), 1) >= 0.35
    warped = np.zeros_like(target)
    warped_source_mask = np.zeros_like(target_valid)
    if homography is not None:
        warped = cv2.warpPerspective(source, homography, (target.shape[1], target.shape[0]), flags=cv2.INTER_LINEAR)
        warped_source_mask = cv2.warpPerspective(source_valid.astype(np.uint8), homography, (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST) > 0
    recovered = hole & warped_source_mask if geometric_accepted else np.zeros_like(hole)
    coverage = float(recovered.sum() / max(hole.sum(), 1))
    if geometric_accepted and recovered.any():
        difference = np.abs(warped.astype(np.float32) - target.astype(np.float32))
        mae = float(difference[recovered].mean())
        mse = float(np.square(difference[recovered]).mean())
        psnr = float(20 * math.log10(255.0 / math.sqrt(max(mse, 1e-12))))
        quality_accepted = coverage >= 0.80 and mae <= 20.0 and psnr >= 20.0
        status = "CAUSAL_WARP_ACCEPTED" if quality_accepted else "REGISTERED_BUT_REJECTED"
    else:
        difference = np.zeros_like(target, dtype=np.float32)
        mae, psnr = None, None
        quality_accepted = False
    return {
        "classification": status,
        "source_keypoints": len(source_keypoints),
        "target_keypoints_outside_holdout": len(target_keypoints),
        "ratio_matches": len(good),
        "ransac_inliers": inliers,
        "inlier_fraction": inliers / max(len(good), 1),
        "geometric_registration_pass": geometric_accepted,
        "holdout_quality_pass": quality_accepted,
        "holdout_pixels": int(hole.sum()),
        "recovered_pixels": int(recovered.sum()),
        "coverage": coverage,
        "mae_rgb": mae,
        "psnr_db": psnr,
    }, {"hole": hole, "recovered": recovered, "warped": warped, "difference": difference}


def _render(output: Path, rows: list[dict[str, Any]], visuals: list[dict[str, np.ndarray]], frames: list[np.ndarray]) -> Path:
    target_path = output / "Poker245_因果平面Donor_伪遮挡留出对照.mp4"
    width, height = 640, 480
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width * 4}x{height}", "-r", "2", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 20)
    for row, visual in zip(rows, visuals):
        target = frames[row["target_frame"]]
        hidden = target.copy(); hidden[visual["hole"]] = 127
        reconstructed = hidden.copy(); reconstructed[visual["recovered"]] = visual["warped"][visual["recovered"]]
        error = np.clip(visual["difference"].mean(axis=2) * 4, 0, 255).astype(np.uint8)
        error = cv2.applyColorMap(error, cv2.COLORMAP_TURBO)
        source = frames[row["source_frame"]]
        cells = []
        labels = ["历史真实源帧", "目标伪遮挡", "因果单应性恢复", "RGB误差热图"]
        for image, label in zip((source, hidden, reconstructed, error), labels):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil); draw.rectangle((0, 0, width, 58), fill=(0, 0, 0))
            draw.text((8, 5), f"{label} | {row['source_frame']}→{row['target_frame']}", font=font, fill=(255,255,255))
            draw.text((8, 31), f"{row['classification']} 覆盖{row['coverage']:.1%}", font=font, fill=(255,230,80))
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
    rows, visuals = [], []
    for source_frame, target_frame in pairs:
        source_entry = manifest["frames"][source_frame]["physical_instances"]["0"]
        target_entry = manifest["frames"][target_frame]["physical_instances"]["0"]
        if not source_entry["observed"] or not target_entry["observed"]:
            raise RuntimeError("pseudo holdout pair must be directly observed")
        source_mask = cv2.imread(source_entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(target_entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
        metrics, visual = evaluate_pair(frames[source_frame], frames[target_frame], source_mask, target_mask)
        metrics.update(source_frame=source_frame, target_frame=target_frame, causal=source_frame < target_frame)
        rows.append(metrics); visuals.append(visual)
    accepted = [row for row in rows if row["classification"] == "CAUSAL_WARP_ACCEPTED"]
    result = {
        "schema_version": "chaoyang-poker-causal-plane-donor-holdout-v1",
        "created_at": now_iso(),
        "task_id": "research_poker_causal_plane_donor_holdout",
        "status": "PASSED_DEVELOPMENT" if accepted else "FAILED_QUALITY_C",
        "session": "play_cards_0903_245",
        "pairs": rows,
        "summary": {
            "pair_count": len(rows),
            "accepted_pairs": len(accepted),
            "median_coverage": float(np.median([row["coverage"] for row in accepted])) if accepted else 0.0,
            "median_mae_rgb": float(np.median([row["mae_rgb"] for row in accepted])) if accepted else None,
            "median_psnr_db": float(np.median([row["psnr_db"] for row in accepted])) if accepted else None,
        },
        "evaluation_mode": "ORACLE_VISIBLE_MASK_PSEUDO_OCCLUSION_DIAGNOSTIC",
        "training_eligible": False,
        "authority_promoted": False,
        "input": artifact_ref(args.object_manifest),
        "claim_limit": "Visible-region causal planar donor holdout only; target mask is evaluation evidence and the result does not prove hidden-surface recovery or Object6D/contact accuracy.",
    }
    video = _render(args.output_root, rows, visuals, frames)
    result["review_video"] = artifact_ref(video)
    atomic_json(args.output_root / "METRICS.json", {"pairs": rows, "summary": result["summary"]})
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
