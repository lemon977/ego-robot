#!/usr/bin/env python3
"""Add causal instance-continuity checks to the Poker245 real donor replay.

V3 demonstrated that geometry and photometric consistency alone can switch to
another visually identical card.  V4 therefore anchors frame 33's directly
observed instance and advances its geometry only with consecutive-frame LK
flow.  A donor homography must agree with that causal motion prediction; when
the instance cannot be followed, the frame remains UNKNOWN.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
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
from chaoyang.ops import run_poker_real_occlusion_plane_donor_v3 as v3


FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _flow_prediction(
    previous: np.ndarray,
    current: np.ndarray,
    previous_mask: np.ndarray,
    previous_human: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    allowed = previous_mask & ~cv2.dilate(previous_human.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    gray0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    points0 = cv2.goodFeaturesToTrack(
        gray0, maxCorners=200, qualityLevel=0.01, minDistance=3,
        mask=allowed.astype(np.uint8) * 255,
    )
    evidence: dict[str, Any] = {
        "seed_points": 0 if points0 is None else int(len(points0)),
        "forward_backward_valid": 0,
        "forward_backward_p90_px": float("inf"),
        "affine_inliers": 0,
        "affine_inlier_fraction": 0.0,
        "flow_runtime_pass": False,
    }
    if points0 is None or len(points0) < 6:
        return np.zeros_like(previous_mask), evidence
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points0, None)
    if points1 is None or status1 is None:
        return np.zeros_like(previous_mask), evidence
    points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, points1, None)
    if points0_back is None or status0 is None:
        return np.zeros_like(previous_mask), evidence
    valid = status1.reshape(-1).astype(bool) & status0.reshape(-1).astype(bool)
    fb = np.linalg.norm(points0_back.reshape(-1, 2) - points0.reshape(-1, 2), axis=1)
    valid &= np.isfinite(fb) & (fb <= 1.5)
    source = points0.reshape(-1, 2)[valid]
    target = points1.reshape(-1, 2)[valid]
    evidence["forward_backward_valid"] = int(len(source))
    if not len(source):
        return np.zeros_like(previous_mask), evidence
    evidence["forward_backward_p90_px"] = float(np.quantile(fb[valid], 0.90))
    if len(source) < 6:
        return np.zeros_like(previous_mask), evidence
    affine, inlier_mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if affine is None or inlier_mask is None or not np.isfinite(affine).all():
        return np.zeros_like(previous_mask), evidence
    inliers = int(inlier_mask.sum())
    fraction = inliers / max(len(source), 1)
    evidence["affine_inliers"] = inliers
    evidence["affine_inlier_fraction"] = fraction
    predicted = cv2.warpAffine(
        previous_mask.astype(np.uint8), affine, (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_NEAREST,
    ) > 0
    area_ratio = float(predicted.sum() / max(previous_mask.sum(), 1))
    evidence["predicted_area_ratio"] = area_ratio
    evidence["flow_runtime_pass"] = bool(
        len(source) >= 6 and inliers >= 5 and fraction >= 0.60 and 0.55 <= area_ratio <= 1.80
    )
    return predicted, evidence


def _identity_checks(candidate_mask: np.ndarray, predicted_mask: np.ndarray, flow: dict[str, Any]) -> dict[str, bool]:
    candidate_centroid = _centroid(candidate_mask)
    predicted_centroid = _centroid(predicted_mask)
    if candidate_centroid is None or predicted_centroid is None:
        distance = float("inf")
    else:
        distance = float(np.linalg.norm(np.asarray(candidate_centroid) - np.asarray(predicted_centroid)))
    union = candidate_mask | predicted_mask
    intersection = candidate_mask & predicted_mask
    iou = float(intersection.sum() / max(union.sum(), 1))
    scale = max(np.sqrt(max(predicted_mask.sum(), 1)), 1.0)
    checks = {
        "consecutive_flow_valid": bool(flow["flow_runtime_pass"]),
        "candidate_centroid_agrees_with_flow": distance <= max(15.0, 0.20 * scale),
        "candidate_mask_iou_with_flow": iou >= 0.25,
    }
    flow["candidate_vs_flow_centroid_distance_px"] = distance
    flow["candidate_vs_flow_iou"] = iou
    return checks


def _render(
    path: Path, frames: list[np.ndarray], old_clean: list[np.ndarray], rows: list[dict[str, Any]],
    outputs: list[np.ndarray], maps: list[np.ndarray], human_masks: list[np.ndarray],
) -> None:
    width, height = 640, 480
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width * 4}x{height}", "-r", "10", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)
    font = ImageFont.truetype(str(FONT), 18)
    for row, output, source_map, human in zip(rows, outputs, maps, human_masks):
        frame_id = row["target_frame"]
        unknown = frames[frame_id].copy()
        unknown[human] = (80, 80, 80)
        shown = output if row["runtime_pass"] else unknown
        overlay = cv2.addWeighted(frames[frame_id], 0.68, source_map, 0.32, 0)
        panels = (frames[frame_id], old_clean[frame_id], shown, overlay)
        labels = ("Raw", "旧Clean基线", "V4身份连续Donor/UNKNOWN", "绿色接受 红色拒绝")
        cells = []
        for panel, label in zip(panels, labels):
            image = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.rectangle((0, 0, width, 72), fill=(0, 0, 0))
            draw.text((8, 5), f"{label} | frame {frame_id}", font=font, fill=(255, 255, 255))
            decision = "ACCEPT" if row["runtime_pass"] else "UNKNOWN"
            flow = row["identity_evidence"]
            draw.text((8, 39), f"{decision} donor={row['source_frame']} flowIoU={flow['candidate_vs_flow_iou']:.2f}", font=font, fill=(255,230,80))
            cells.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
        assert process.stdin is not None
        process.stdin.write(np.concatenate(cells, axis=1).tobytes())
    assert process.stdin is not None
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("video encode failed")


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
    raw_path = Path(objects["input"]["selected_rgb"]["path"])
    frames = v3._decode(raw_path, v3.EVALUATION_END + 1)
    clean_frames = v3._decode(args.old_clean_video, v3.EVALUATION_END + 1)
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=5, edgeThreshold=10, patchSize=21)
    sources = v3._source_features(frames, objects, orb)
    state_mask = sources[33][0]
    previous_frame = frames[33]
    previous_human = cv2.imread(roles["frames"][33]["hand_tracker_union"]["path"], cv2.IMREAD_GRAYSCALE) > 0

    rows: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    source_maps: list[np.ndarray] = []
    human_masks: list[np.ndarray] = []
    for target_frame in range(v3.EVENT_START, v3.EVALUATION_END + 1):
        human = cv2.imread(roles["frames"][target_frame]["hand_tracker_union"]["path"], cv2.IMREAD_GRAYSCALE) > 0
        predicted_mask, flow_evidence = _flow_prediction(previous_frame, frames[target_frame], state_mask, previous_human)
        flow_pass = bool(flow_evidence["flow_runtime_pass"])
        search_seed = predicted_mask if flow_pass else state_mask
        search_roi = v3._expanded_roi(search_seed, search_seed.shape, factor=2.0 if flow_pass else 2.5)
        human_expanded = cv2.dilate(human.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        target_allowed = (search_roi > 0) & ~human_expanded
        target_keypoints, target_desc = orb.detectAndCompute(frames[target_frame], target_allowed.astype(np.uint8) * 255)
        candidates = []
        for source_frame, (source_mask, source_keypoints, source_desc) in sources.items():
            metrics, visual = v3._candidate(
                source_frame=source_frame, target_frame=target_frame,
                source=frames[source_frame], target=frames[target_frame], source_mask=source_mask,
                source_keypoints=source_keypoints, source_desc=source_desc,
                target_keypoints=target_keypoints, target_desc=target_desc,
                target_allowed=target_allowed, search_roi=search_roi > 0,
            )
            candidates.append((metrics, visual))
        row, visual = max(candidates, key=lambda item: v3._score(item[0]))
        identity = _identity_checks(visual["projected_mask"], predicted_mask, flow_evidence)
        row["homography_gate_pass"] = bool(row["runtime_pass"])
        row["identity_checks"] = identity
        row["identity_evidence"] = flow_evidence
        row["runtime_pass"] = bool(row["runtime_pass"] and all(identity.values()))
        row["event_phase"] = "REAL_OCCLUSION_RUNTIME" if target_frame <= v3.EVENT_END else "FUTURE_REAPPEARANCE_EVALUATION_ONLY"
        row["future_direct_mask_used_for_runtime"] = False
        if target_frame > v3.EVENT_END:
            row["runtime_pass_before_evaluation_override"] = row["runtime_pass"]
            row["runtime_pass"] = False
            row["evaluation_note"] = "Future reappearance is evaluation-only and shows a potentially different face."

        output = frames[target_frame].copy()
        source_map = np.zeros_like(output)
        write = visual["projected_mask"] & human
        if row["runtime_pass"]:
            output[write] = visual["warped"][write]
            source_map[visual["projected_mask"]] = (0, 220, 0)
            state_mask = visual["projected_mask"]
        else:
            source_map[search_seed] = (0, 0, 230)
            if flow_pass:
                # Geometry may advance causally, but rejected pixels are never written.
                state_mask = predicted_mask
        row["write_pixels"] = int(write.sum()) if row["runtime_pass"] else 0
        row["unknown_pixels_in_causal_search_roi"] = int((human & (search_roi > 0)).sum()) if not row["runtime_pass"] else 0
        rows.append(row)
        outputs.append(output)
        source_maps.append(source_map)
        human_masks.append(human)
        previous_frame = frames[target_frame]
        previous_human = human

    runtime_rows = [row for row in rows if row["event_phase"] == "REAL_OCCLUSION_RUNTIME"]
    accepted = [row for row in runtime_rows if row["runtime_pass"]]
    video = args.output_root / "Poker245_真实遮挡_因果Donor_V4_身份连续全事件.mp4"
    _render(video, frames, clean_frames, rows, outputs, source_maps, human_masks)
    decisions = args.output_root / "FRAME_DECISIONS.json"
    atomic_json(decisions, {"schema_version": "chaoyang-poker-real-occlusion-plane-donor-v4-frames", "rows": rows})
    result = {
        "schema_version": "chaoyang-poker-real-occlusion-plane-donor-v4",
        "created_at": now_iso(),
        "task_id": "research_poker_real_occlusion_plane_donor_v4",
        "status": "PASSED_DEVELOPMENT" if accepted else "FAILED_QUALITY_C",
        "session": "play_cards_0903_245",
        "summary": {
            "runtime_event_frames": len(runtime_rows),
            "runtime_accepted_frames": len(accepted),
            "runtime_unknown_frames": len(runtime_rows) - len(accepted),
            "runtime_acceptance_fraction": len(accepted) / max(len(runtime_rows), 1),
            "identity_continuity_required": True,
            "future_frames_used_for_runtime_or_atlas": False,
            "hidden_ground_truth_used_for_runtime_gate": False,
        },
        "v3_review_finding": "V3 accepted 56/56 but visibly switched to a sibling purple-backed card after the selected card moved/flipped; V4 adds causal consecutive-flow identity continuity.",
        "outputs": {"review_video": artifact_ref(video), "frame_decisions": artifact_ref(decisions)},
        "inputs": {
            "object_manifest": artifact_ref(args.object_manifest),
            "role_manifest": artifact_ref(args.role_manifest),
            "old_clean_video": artifact_ref(args.old_clean_video),
            "v3_implementation": artifact_ref(ROOT / "src/chaoyang/ops/run_poker_real_occlusion_plane_donor_v3.py"),
            "v4_implementation": artifact_ref(Path(__file__).resolve()),
        },
        "training_eligible": False,
        "authority_promoted": False,
        "claim_limit": "One causal identity-continuity development replay; not Clean, Object6D, contact, or external-truth authority.",
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
