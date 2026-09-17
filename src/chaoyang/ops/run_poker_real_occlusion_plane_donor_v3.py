#!/usr/bin/env python3
"""Replay one real Poker245 occlusion event with a causal, non-oracle donor.

Only observations at or before the target frame participate in the runtime
decision.  Direct object masks after the occlusion are loaded solely for the
separately labelled after-the-fact evaluation panel; they never enter donor
selection, homography estimation, acceptance, or atlas updates.
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
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
SOURCE_FRAMES = tuple(range(0, 34, 2)) + (33,)
EVENT_START = 34
EVENT_END = 89
EVALUATION_START = 90
EVALUATION_END = 98


def _decode(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {count}: {path}")
    return frames


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expanded_roi(mask: np.ndarray, shape: tuple[int, int], factor: float = 3.0) -> np.ndarray:
    roi = np.zeros(shape, np.uint8)
    box = _bbox(mask)
    if box is None:
        return roi
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width = max((x1 - x0 + 1) * factor, 160)
    height = max((y1 - y0 + 1) * factor, 180)
    xa, xb = max(0, int(cx - width / 2)), min(shape[1], int(cx + width / 2))
    ya, yb = max(0, int(cy - height / 2)), min(shape[0], int(cy + height / 2))
    roi[ya:yb, xa:xb] = 255
    return roi


def _source_features(
    frames: list[np.ndarray], object_manifest: dict[str, Any], orb: cv2.ORB,
) -> dict[int, tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray | None]]:
    result = {}
    for frame_id in SOURCE_FRAMES:
        entry = object_manifest["frames"][frame_id]["physical_instances"]["0"]
        if not entry.get("observed"):
            continue
        mask = cv2.imread(entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"missing source mask for frame {frame_id}")
        keypoints, descriptors = orb.detectAndCompute(frames[frame_id], mask)
        result[frame_id] = (mask > 0, keypoints, descriptors)
    return result


def _candidate(
    *, source_frame: int, target_frame: int, source: np.ndarray, target: np.ndarray,
    source_mask: np.ndarray, source_keypoints: list[cv2.KeyPoint], source_desc: np.ndarray | None,
    target_keypoints: list[cv2.KeyPoint], target_desc: np.ndarray | None,
    target_allowed: np.ndarray, search_roi: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    matches: list[cv2.DMatch] = []
    if source_desc is not None and target_desc is not None:
        forward = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_desc, target_desc, k=2)
        backward = cv2.BFMatcher(cv2.NORM_HAMMING).match(target_desc, source_desc)
        reverse = {item.queryIdx: item.trainIdx for item in backward}
        for pair in forward:
            if len(pair) != 2 or pair[0].distance >= 0.75 * pair[1].distance:
                continue
            match = pair[0]
            if reverse.get(match.trainIdx) == match.queryIdx:
                matches.append(match)

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
    reprojection_p90 = float("inf")
    source_hull_fraction = 0.0
    projected_area_ratio = 0.0
    centroid_in_search_roi = False
    visible_fraction = 0.0
    visible_pixels = 0
    photometric_median = float("inf")
    photometric_p90 = float("inf")
    warped = np.zeros_like(target)
    projected_mask = np.zeros(source_mask.shape, bool)

    if homography is not None and inliers >= 4 and np.isfinite(homography).all():
        keep = inlier_mask.reshape(-1).astype(bool)
        src = source_points[keep]
        dst = target_points[keep]
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        reprojection_p90 = float(np.quantile(errors, 0.90))
        source_box = _bbox(source_mask)
        if source_box is not None:
            x0, y0, x1, y1 = source_box
            box_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
            if len(src) >= 3:
                source_hull_fraction = float(cv2.contourArea(cv2.convexHull(src.reshape(-1, 1, 2))) / box_area)
        warped = cv2.warpPerspective(source, homography, (target.shape[1], target.shape[0]))
        projected_mask = cv2.warpPerspective(
            source_mask.astype(np.uint8), homography, (target.shape[1], target.shape[0]),
            flags=cv2.INTER_NEAREST,
        ) > 0
        projected_area_ratio = float(projected_mask.sum() / max(source_mask.sum(), 1))
        box = _bbox(projected_mask)
        if box is not None:
            x0, y0, x1, y1 = box
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            centroid_in_search_roi = bool(search_roi[cy, cx])
        visible = projected_mask & target_allowed
        visible_pixels = int(visible.sum())
        visible_fraction = float(visible_pixels / max(projected_mask.sum(), 1))
        if visible_pixels:
            difference = np.abs(warped.astype(np.float32) - target.astype(np.float32)).mean(axis=2)
            photometric_median = float(np.median(difference[visible]))
            photometric_p90 = float(np.quantile(difference[visible], 0.90))

    checks = {
        "mutual_ratio_matches": len(matches) >= 8,
        "ransac_inliers": inliers >= 6,
        "inlier_fraction": inlier_fraction >= 0.40,
        "reprojection_p90_px": reprojection_p90 <= 4.0,
        "source_match_hull_fraction": source_hull_fraction >= 0.02,
        "projected_area_ratio": 0.30 <= projected_area_ratio <= 3.0,
        "projected_centroid_in_causal_search_roi": centroid_in_search_roi,
        "visible_pixels": visible_pixels >= 64,
        "visible_fraction": visible_fraction >= 0.08,
        "visible_photometric_median": photometric_median <= 25.0,
        "visible_photometric_p90": photometric_p90 <= 70.0,
    }
    passed = all(checks.values())
    metrics = {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "runtime_pass": passed,
        "runtime_checks": checks,
        "runtime_evidence": {
            "mutual_ratio_matches": len(matches),
            "ransac_inliers": inliers,
            "inlier_fraction": inlier_fraction,
            "reprojection_p90_px": reprojection_p90,
            "source_match_hull_fraction": source_hull_fraction,
            "projected_area_ratio": projected_area_ratio,
            "projected_centroid_in_causal_search_roi": centroid_in_search_roi,
            "visible_pixels": visible_pixels,
            "visible_fraction": visible_fraction,
            "visible_abs_rgb_median": photometric_median,
            "visible_abs_rgb_p90": photometric_p90,
        },
    }
    return metrics, {"warped": warped, "projected_mask": projected_mask}


def _score(row: dict[str, Any]) -> tuple[int, int, float, float]:
    evidence = row["runtime_evidence"]
    return (
        int(row["runtime_pass"]),
        int(evidence["ransac_inliers"]),
        -float(evidence["reprojection_p90_px"]),
        -float(evidence["visible_abs_rgb_median"]),
    )


def _render(
    path: Path, frames: list[np.ndarray], clean_frames: list[np.ndarray], rows: list[dict[str, Any]],
    projected: list[np.ndarray], human_masks: list[np.ndarray], source_maps: list[np.ndarray],
) -> None:
    height, width = 480, 640
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width * 4}x{height}", "-r", "10", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 18)
    for row, new, human, source_map in zip(rows, projected, human_masks, source_maps):
        frame_id = row["target_frame"]
        raw = frames[frame_id]
        old = clean_frames[frame_id]
        unknown = raw.copy()
        unknown[human] = (80, 80, 80)
        overlay = cv2.addWeighted(raw, 0.70, source_map, 0.30, 0)
        panels = (raw, old, new if row["runtime_pass"] else unknown, overlay)
        labels = ("Raw", "旧Clean基线", "V3因果Donor/拒绝则UNKNOWN", "像素来源/投影")
        cells = []
        for panel, label in zip(panels, labels):
            image = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.rectangle((0, 0, width, 70), fill=(0, 0, 0))
            draw.text((8, 5), f"{label} | frame {frame_id}", font=font, fill=(255, 255, 255))
            decision = "ACCEPT" if row["runtime_pass"] else "UNKNOWN"
            draw.text((8, 38), f"{decision} | donor={row.get('source_frame')} | 仅过去证据", font=font, fill=(255, 230, 80))
            cells.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        assert process.stdin is not None
        process.stdin.write(np.concatenate(cells, axis=1).tobytes())
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encoding failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--old-clean-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)

    objects = json.loads(args.object_manifest.read_text(encoding="utf-8"))
    roles = json.loads(args.role_manifest.read_text(encoding="utf-8"))
    if objects.get("session") != "play_cards_0903_245" or roles.get("session") != "play_cards_0903_245":
        raise RuntimeError("frozen canary is Poker245 only")
    raw_path = Path(objects["input"]["selected_rgb"]["path"])
    frames = _decode(raw_path, EVALUATION_END + 1)
    clean_frames = _decode(args.old_clean_video, EVALUATION_END + 1)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=5, edgeThreshold=10, patchSize=21)
    sources = _source_features(frames, objects, orb)
    last_direct = sources[33][0]
    search_roi = _expanded_roi(last_direct, last_direct.shape)

    rows: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    human_masks: list[np.ndarray] = []
    source_maps: list[np.ndarray] = []
    accepted_projected_mask: np.ndarray | None = None
    for target_frame in range(EVENT_START, EVALUATION_END + 1):
        human = cv2.imread(roles["frames"][target_frame]["hand_tracker_union"]["path"], cv2.IMREAD_GRAYSCALE) > 0
        human_expanded = cv2.dilate(human.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        runtime_allowed = (search_roi > 0) & ~human_expanded
        target_keypoints, target_desc = orb.detectAndCompute(frames[target_frame], runtime_allowed.astype(np.uint8) * 255)

        candidates: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
        for source_frame, (source_mask, source_keypoints, source_desc) in sources.items():
            if source_frame >= target_frame:
                continue
            candidates.append(_candidate(
                source_frame=source_frame, target_frame=target_frame,
                source=frames[source_frame], target=frames[target_frame],
                source_mask=source_mask, source_keypoints=source_keypoints, source_desc=source_desc,
                target_keypoints=target_keypoints, target_desc=target_desc,
                target_allowed=runtime_allowed, search_roi=search_roi > 0,
            ))
        best_row, best_visual = max(candidates, key=lambda item: _score(item[0]))
        best_row["event_phase"] = "REAL_OCCLUSION_RUNTIME" if target_frame <= EVENT_END else "FUTURE_REAPPEARANCE_EVALUATION_ONLY"
        best_row["future_direct_mask_used_for_runtime"] = False
        if target_frame > EVENT_END:
            # The future reappearance is deliberately excluded from runtime admission.
            best_row["runtime_pass_before_evaluation_override"] = best_row["runtime_pass"]
            best_row["runtime_pass"] = False
            best_row["evaluation_note"] = "Direct observation reappears with a potentially different card face; no back-atlas writeback."

        output = frames[target_frame].copy()
        source_map = np.zeros_like(output)
        projected_mask = best_visual["projected_mask"]
        write = projected_mask & human
        if best_row["runtime_pass"]:
            output[write] = best_visual["warped"][write]
            source_map[write] = (0, 220, 0)
            source_map[projected_mask & ~human] = (220, 180, 0)
            accepted_projected_mask = projected_mask
            search_roi = _expanded_roi(projected_mask, projected_mask.shape, factor=3.0)
        else:
            source_map[human & (search_roi > 0)] = (0, 0, 230)
            if accepted_projected_mask is not None:
                search_roi = _expanded_roi(accepted_projected_mask, accepted_projected_mask.shape, factor=3.5)
        best_row["write_pixels"] = int(write.sum()) if best_row["runtime_pass"] else 0
        best_row["unknown_pixels_in_causal_search_roi"] = int((human & (search_roi > 0)).sum()) if not best_row["runtime_pass"] else 0
        rows.append(best_row)
        outputs.append(output)
        human_masks.append(human)
        source_maps.append(source_map)

    accepted = [row for row in rows if row["event_phase"] == "REAL_OCCLUSION_RUNTIME" and row["runtime_pass"]]
    runtime_rows = [row for row in rows if row["event_phase"] == "REAL_OCCLUSION_RUNTIME"]
    video = args.output_root / "Poker245_真实遮挡_因果Donor_V3_全事件.mp4"
    _render(video, frames, clean_frames, rows, outputs, human_masks, source_maps)
    atomic_json(args.output_root / "FRAME_DECISIONS.json", {
        "schema_version": "chaoyang-poker-real-occlusion-plane-donor-frame-decisions-v3",
        "rows": rows,
    })
    result = {
        "schema_version": "chaoyang-poker-real-occlusion-plane-donor-v3",
        "created_at": now_iso(),
        "task_id": "research_poker_real_occlusion_plane_donor_v3",
        "status": "PASSED_DEVELOPMENT" if accepted else "FAILED_QUALITY_C",
        "session": "play_cards_0903_245",
        "event": {"source_visible": [0, 33], "real_occlusion": [EVENT_START, EVENT_END], "future_reappearance_evaluation_only": [EVALUATION_START, EVALUATION_END]},
        "summary": {
            "runtime_event_frames": len(runtime_rows),
            "runtime_accepted_frames": len(accepted),
            "runtime_unknown_frames": len(runtime_rows) - len(accepted),
            "runtime_acceptance_fraction": len(accepted) / max(len(runtime_rows), 1),
            "future_frames_used_for_runtime_or_atlas": False,
            "hidden_ground_truth_used_for_runtime_gate": False,
        },
        "runtime_gate_inputs": [
            "past directly observed same-instance back-face masks and RGB",
            "current RGB outside the human-removal mask",
            "causal search ROI propagated only from past direct/accepted geometry",
            "mutual feature matches, RANSAC geometry, and current visible photometric evidence",
        ],
        "outputs": {
            "review_video": artifact_ref(video),
            "frame_decisions": artifact_ref(args.output_root / "FRAME_DECISIONS.json"),
        },
        "inputs": {
            "object_manifest": artifact_ref(args.object_manifest),
            "role_manifest": artifact_ref(args.role_manifest),
            "raw_video": artifact_ref(raw_path),
            "old_clean_video": artifact_ref(args.old_clean_video),
        },
        "training_eligible": False,
        "authority_promoted": False,
        "claim_limit": "One Poker245 real-occlusion development replay. Runtime admission is non-oracle and causal; it is not Clean, Object6D, contact, or physical-truth authority.",
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
