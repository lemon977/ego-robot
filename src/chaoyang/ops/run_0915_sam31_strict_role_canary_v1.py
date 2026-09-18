#!/usr/bin/env python3
"""Run the single-session strict-role SAM3.1 canary for 0915.

The parent process validates the current task route and acquires the central
GPU lease.  The worker keeps one pinned model resident, initializes each role
instance independently, propagates both directions, and performs at most one
quality-triggered fallback replay per instance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import jsonschema
import numpy as np
import torch

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import run_0915_leftmono_sam31_masks_v1 as legacy
from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module
from chaoyang.pipeline.sam31_0915_strict_role_contract_v1 import (
    FRAME_COUNT,
    FROZEN_INSTANCE_PROMPTS,
    HEIGHT,
    IMAGE_DOMAIN,
    REMOVAL_ROLES,
    ROLE_NAMES,
    SESSION_ID,
    WIDTH,
    InstancePrompt,
    Seed,
    build_prompt_plan,
    should_reseed,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_sam31_strict_role_canary_v5"
PHASE = "0915_SAM31_STRICT_ROLE_SINGLE_SESSION_CANARY_V5"
VIDEO = (
    ROOT / "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001/"
    "input/PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
)
HAWOR = (
    ROOT / "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/"
    "bounded_output_guarded_identity_fixed/play_cards_0915_001/"
    "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
)
CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
CODE_ROOT = ROOT / "vendor/SAM3"
if str(CODE_ROOT) not in sys.path:
    # The checkpoint is paired with this exact repository-local official tree.
    # Keep the runtime self-contained without installing or mutating the vendor.
    sys.path.insert(0, str(CODE_ROOT))
SCHEMA = ROOT / "contracts/visual_role_mask_v2.schema.json"
COLORS = {
    "left_hand_00": (255, 180, 40),
    "right_hand_00": (30, 90, 255),
    "left_forearm_00": (255, 230, 70),
    "right_forearm_00": (30, 210, 255),
    "left_finger_sleeve_cluster_00": (255, 80, 220),
    "right_finger_sleeve_cluster_00": (190, 40, 255),
    "left_cable_00": (0, 220, 255),
    "right_cable_00": (0, 150, 255),
    "playing_card_00": (60, 255, 60),
    "playing_card_01": (255, 80, 80),
    "playing_card_02": (255, 60, 200),
}


class SeedQualityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size,
            "sha256": sha256(candidate)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from its frozen specification")
    if packet.get("weights") != ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"]:
        raise RuntimeError("strict role canary must bind exactly one SAM3.1 weight")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("strict role canary is not current next_task")
    task = next((row for row in state.get("tasks", [])
                 if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"
    }:
        raise RuntimeError("strict role canary is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize this canary")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat() -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", "RUNNING", "--phase", PHASE, "--gpu-id", "0",
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def video_gate(path: Path) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    result = {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]), "fps": stream["r_frame_rate"],
        "full_decode": decoded.returncode == 0,
    }
    if result != {
        "width": WIDTH, "height": HEIGHT, "frames": FRAME_COUNT,
        "fps": "30/1", "full_decode": True,
    }:
        raise RuntimeError(f"resize-only video identity drift: {result}")
    return result


def extract_frames(video: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
        "-vsync", "0", str(destination / "%05d.jpg"),
    ], capture_output=True, text=True, check=False)
    paths = sorted(destination.glob("*.jpg"))
    if completed.returncode or len(paths) != FRAME_COUNT:
        raise RuntimeError("strict canary frame extraction failed")
    # ffmpeg numbers image sequences from one; normalize names for direct indexing.
    for index, path in enumerate(paths):
        target = destination / f".{index:05d}.tmp.jpg"
        os.replace(path, target)
    for index in range(FRAME_COUNT):
        os.replace(destination / f".{index:05d}.tmp.jpg",
                   destination / f"{index:05d}.jpg")


def _normalised_seed(seed: Seed) -> tuple[list[list[float]], torch.Tensor, torch.Tensor]:
    x, y, w, h = seed.box_xywh
    points = [*seed.positive_points_xy, *seed.negative_points_xy]
    labels = [1] * len(seed.positive_points_xy) + [0] * len(seed.negative_points_xy)
    return (
        [[x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT]],
        torch.tensor([[px / WIDTH, py / HEIGHT] for px, py in points],
                     dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int32),
    )


def _point_distance(mask: np.ndarray, point: tuple[float, float]) -> float:
    return legacy.anchor_distance(mask, np.asarray(point, np.float64))


def choose_seed_candidate(
    masks: np.ndarray, scores: np.ndarray, ids: np.ndarray,
    prompt: InstancePrompt, seed: Seed,
) -> tuple[int, list[dict[str, Any]]]:
    x, y, w, h = seed.box_xywh
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = int(np.ceil(x + w)), int(np.ceil(y + h))
    box_area = max(1, (x1 - x0) * (y1 - y0))
    rows = []
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        intersection = int(mask[y0:y1, x0:x1].sum())
        distances = [_point_distance(mask, point)
                     for point in seed.positive_points_xy]
        negatives = [_point_distance(mask, point) == 0
                     for point in seed.negative_points_xy]
        eligible = bool(
            prompt.minimum_area <= area <= int(mask.size * prompt.maximum_area_fraction)
            and intersection > 0
            and min(distances, default=float("inf")) <= 45.0
            and (not negatives or not all(negatives))
        )
        rows.append({
            "raw_id": int(ids[index]), "score": float(scores[index]),
            "area_pixels": area, "box_intersection_pixels": intersection,
            "box_intersection_over_box": intersection / box_area,
            "positive_distance_px": float(min(distances, default=float("inf"))),
            "negative_points_covered": int(sum(negatives)),
            "eligible": eligible,
        })
    eligible_rows = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            row["positive_distance_px"], -row["box_intersection_over_box"],
            -row["score"], row["raw_id"],
        ),
    )
    if not eligible_rows:
        raise SeedQualityError(f"no eligible SAM3.1 seed candidate: {rows}")
    return int(eligible_rows[0]["raw_id"]), rows


def _collect_bidirectional(
    model: Any, state: dict[str, Any], *, anchor: int, raw_id: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros((FRAME_COUNT, HEIGHT, WIDTH), bool)
    direction_evidence: list[dict[str, Any]] = []
    routes = [(False, FRAME_COUNT - anchor)]
    if anchor:
        routes.append((True, anchor + 1))
    for reverse, maximum in routes:
        yielded = 0
        try:
            for frame_index, outputs in model.propagate_in_video(
                inference_state=state, start_frame_idx=anchor,
                max_frame_num_to_track=maximum, reverse=reverse,
                output_prob_thresh=0.5,
            ):
                yielded += 1
                masks, _scores, ids = legacy.normalize(outputs, HEIGHT, WIDTH)
                matches = np.flatnonzero(ids == raw_id)
                if len(matches) == 1:
                    result[int(frame_index)] = masks[int(matches[0])]
            direction_evidence.append({
                "direction": "backward" if reverse else "forward",
                "status": "COMPLETE", "frames_yielded": yielded,
            })
        except RuntimeError as exc:
            if str(exc) != "No points are provided; please add points first":
                raise
            direction_evidence.append({
                "direction": "backward" if reverse else "forward",
                "status": "UNKNOWN_DIRECTION_TRACKER_HAS_NO_CONFIRMED_INSTANCE",
                "frames_yielded_before_hold": yielded,
                "reason": str(exc),
            })
    return result, direction_evidence


def run_seed(
    model: Any, frames: Path, prompt: InstancePrompt, seed: Seed,
    *, label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    state = model.init_state(
        resource_path=str(frames), offload_video_to_cpu=True,
        async_loading_frames=False,
    )
    boxes, points, labels = _normalised_seed(seed)
    try:
        _, initial = model.add_prompt(
            inference_state=state, frame_idx=seed.frame_index,
            text_str=prompt.text, boxes_xywh=boxes, box_labels=[1],
            clear_old_boxes=True, output_prob_thresh=0.5,
        )
        masks, scores, ids = legacy.normalize(initial, HEIGHT, WIDTH)
        raw_id, candidates = choose_seed_candidate(
            masks, scores, ids, prompt, seed,
        )
        selected_index = int(np.flatnonzero(ids == raw_id)[0])
        route_before, action_ids_before = model.parse_action_history_for_propagation(state)
        if route_before != "propagation_full" or action_ids_before is not None:
            raise RuntimeError(
                "initial text+box did not retain the pinned full semantic propagation route"
            )
        tracked, direction_evidence = _collect_bidirectional(
            model, state, anchor=seed.frame_index, raw_id=raw_id,
        )
        if not tracked[seed.frame_index].any():
            tracked[seed.frame_index] = masks[selected_index]
        return tracked, {
            "status": "COMPLETE", "seed_label": label,
            "frame_index": seed.frame_index, "box_xywh": list(seed.box_xywh),
            "text": prompt.text, "selected_raw_id": raw_id,
            "candidate_rows": candidates,
            "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
            "propagation_route_before": route_before,
            "action_object_ids_before": action_ids_before,
            "direction_evidence": direction_evidence,
            "point_refined": False,
            "point_usage": "SEED_SELECTION_AND_QUALITY_EVIDENCE_ONLY",
            "point_refinement_hold_reason": (
                "Pinned multiplex point refinement switches to a partial SAM2 route; "
                "the v3 two-hand diagnostic did not preserve temporal continuity."
            ),
        }
    finally:
        state.clear()


def _near_mask(mask: np.ndarray, point: np.ndarray, radius: int) -> bool:
    if not np.isfinite(point).all():
        return False
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if x < -radius or y < -radius or x >= WIDTH + radius or y >= HEIGHT + radius:
        return False
    x0, x1 = max(0, x - radius), min(WIDTH, x + radius + 1)
    y0, y1 = max(0, y - radius), min(HEIGHT, y + radius + 1)
    return bool(mask[y0:y1, x0:x1].any())


def evaluate_stream(
    masks: np.ndarray, prompt: InstancePrompt, seed: Seed,
    joints: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    areas = masks.reshape(FRAME_COUNT, -1).sum(axis=1).astype(np.int64)
    anchor_area = int(areas[seed.frame_index])
    rows: list[dict[str, Any]] = []
    centroids: list[tuple[float, float] | None] = []
    for mask in masks:
        yy, xx = np.where(mask)
        centroids.append(None if not len(xx) else (float(xx.mean()), float(yy.mean())))
    valid = np.ones(FRAME_COUNT, bool)
    for frame, mask in enumerate(masks):
        reasons: list[str] = []
        area = int(areas[frame])
        if area < prompt.minimum_area:
            reasons.append("AREA_BELOW_MINIMUM")
        if area > int(mask.size * prompt.maximum_area_fraction):
            reasons.append("AREA_ABOVE_MAXIMUM")
        if anchor_area <= 0:
            reasons.append("ANCHOR_EMPTY")
        elif area / anchor_area < prompt.area_ratio_range[0]:
            reasons.append("AREA_RATIO_LOW")
        elif area / anchor_area > prompt.area_ratio_range[1]:
            reasons.append("AREA_RATIO_HIGH")
        components = 0
        if area:
            components = int(cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1)
            if components > prompt.maximum_components:
                reasons.append("TOO_MANY_COMPONENTS")
        if frame and centroids[frame] is not None and centroids[frame - 1] is not None:
            jump = float(np.linalg.norm(
                np.asarray(centroids[frame]) - np.asarray(centroids[frame - 1])
            ))
            if jump > prompt.maximum_centroid_jump_px:
                reasons.append("CENTROID_JUMP")
        else:
            jump = None
        if prompt.role.endswith("_hand"):
            side = 0 if prompt.role.startswith("left_") else 1
            if observed[side, frame]:
                finite = np.isfinite(joints[side, frame]).all(axis=1)
                points = joints[side, frame][finite]
                near = sum(_near_mask(mask, point, 22) for point in points)
                coverage = near / len(points) if len(points) else 0.0
                if len(points) >= 5 and coverage < 0.20:
                    reasons.append("HAWOR_PROJECTED_REGION_COVERAGE_LOW")
            else:
                coverage = None
        elif prompt.role.endswith("_forearm"):
            side = 0 if prompt.role.startswith("left_") else 1
            coverage = None
            wrist = joints[side, frame, 0]
            if observed[side, frame] and np.isfinite(wrist).all():
                if not _near_mask(mask, wrist, 85):
                    reasons.append("WRIST_CONNECTION_MISSING")
        else:
            coverage = None
        if prompt.role == "task_object" and centroids[frame] is not None:
            cx, cy = centroids[frame]
            if not (300 <= cx <= 920 and 430 <= cy <= 900):
                reasons.append("OBJECT_OUTSIDE_TASK_ROI")
        valid[frame] = not reasons
        rows.append({
            "frame_index": frame, "valid": bool(valid[frame]),
            "reasons": reasons, "area_pixels": area,
            "component_count": components,
            "centroid_xy": list(centroids[frame]) if centroids[frame] else None,
            "centroid_jump_px": jump,
            "hawor_region_coverage": coverage,
        })
    return valid, rows


def _empty_seed_result(label: str, seed: Seed, error: Exception) -> tuple[np.ndarray, dict[str, Any]]:
    return np.zeros((FRAME_COUNT, HEIGHT, WIDTH), bool), {
        "status": "REJECTED_SEED_QUALITY", "seed_label": label,
        "frame_index": seed.frame_index, "box_xywh": list(seed.box_xywh),
        "error": f"{type(error).__name__}: {error}",
        "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
    }


def process_instance(
    model: Any, frames: Path, prompt: InstancePrompt,
    joints: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    try:
        primary, primary_evidence = run_seed(
            model, frames, prompt, prompt.primary, label="primary",
        )
    except SeedQualityError as exc:
        primary, primary_evidence = _empty_seed_result("primary", prompt.primary, exc)
    primary_valid, primary_rows = evaluate_stream(
        primary, prompt, prompt.primary, joints, observed,
    )
    trigger, trigger_reasons = should_reseed(
        primary_valid, has_fallback=prompt.fallback is not None,
    )
    fallback = np.zeros_like(primary)
    fallback_valid = np.zeros(FRAME_COUNT, bool)
    fallback_rows: list[dict[str, Any]] = []
    fallback_evidence: dict[str, Any] | None = None
    if trigger and prompt.fallback is not None:
        try:
            fallback, fallback_evidence = run_seed(
                model, frames, prompt, prompt.fallback, label="fallback",
            )
        except SeedQualityError as exc:
            fallback, fallback_evidence = _empty_seed_result(
                "fallback", prompt.fallback, exc,
            )
        fallback_valid, fallback_rows = evaluate_stream(
            fallback, prompt, prompt.fallback, joints, observed,
        )
    merged = np.zeros_like(primary)
    ledger = []
    for frame in range(FRAME_COUNT):
        if primary_valid[frame]:
            merged[frame] = primary[frame]
            source = "primary"
            state = "seeded" if frame == prompt.primary.frame_index else "tracked"
            reasons: list[str] = []
        elif trigger and fallback_valid[frame]:
            merged[frame] = fallback[frame]
            source = "fallback"
            state = (
                "reseeded" if prompt.fallback is not None
                and frame == prompt.fallback.frame_index else "tracked"
            )
            reasons = primary_rows[frame]["reasons"]
        else:
            source = None
            state = "unknown"
            reasons = primary_rows[frame]["reasons"]
            if trigger and fallback_rows:
                reasons = sorted(set(reasons + fallback_rows[frame]["reasons"]))
        ledger.append({
            "frame_index": frame, "state": state, "source_seed": source,
            "quality_trigger_reasons": reasons,
        })
    evidence = {
        "instance_id": prompt.instance_id, "role": prompt.role,
        "primary": primary_evidence,
        "primary_valid_frames": int(primary_valid.sum()),
        "reseed_triggered": trigger, "reseed_trigger_reasons": trigger_reasons,
        "fallback": fallback_evidence,
        "fallback_valid_frames": int(fallback_valid.sum()),
        "merged_state_counts": dict(Counter(row["state"] for row in ledger)),
        "quality_gates": {
            "minimum_area": prompt.minimum_area,
            "maximum_area_fraction": prompt.maximum_area_fraction,
            "area_ratio_range": list(prompt.area_ratio_range),
            "maximum_components": prompt.maximum_components,
            "maximum_centroid_jump_px": prompt.maximum_centroid_jump_px,
        },
        "primary_frame_metrics": primary_rows,
        "fallback_frame_metrics": fallback_rows,
    }
    return merged, ledger, evidence


def save_packed(path: Path, masks: np.ndarray) -> np.ndarray:
    packed = np.packbits(masks.reshape(FRAME_COUNT, -1), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, packed=packed, frame_count=np.int32(FRAME_COUNT),
        height=np.int32(HEIGHT), width=np.int32(WIDTH), bitorder=np.asarray("big"),
    )
    return packed


def open_encoder(path: Path) -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x480",
        "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ], stdin=subprocess.PIPE)


def render_review(
    video: Path, packed_by_id: dict[str, np.ndarray],
    states_by_id: dict[str, list[dict[str, Any]]], destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    encoder = open_encoder(destination)
    assert encoder.stdin is not None
    frame = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            overlay = raw.copy()
            for instance_id, packed in packed_by_id.items():
                mask = np.unpackbits(
                    packed[frame], bitorder="big", count=HEIGHT * WIDTH,
                ).reshape(HEIGHT, WIDTH).astype(bool)
                if mask.any():
                    color = np.asarray(COLORS[instance_id], np.float32)
                    overlay[mask] = (
                        0.48 * overlay[mask].astype(np.float32) + 0.52 * color
                    ).astype(np.uint8)
            left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            right = cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA)
            canvas = np.hstack((left, right))
            cv2.rectangle(canvas, (0, 0), (1279, 72), (0, 0, 0), -1)
            cv2.putText(
                canvas, f"{SESSION_ID} | resize-only | SAM3.1 strict roles | frame {frame:03d}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            unknown = sum(
                states_by_id[name][frame]["state"] == "unknown"
                for name in states_by_id
            )
            reseeded = [
                name for name in states_by_id
                if states_by_id[name][frame]["state"] == "reseeded"
            ]
            cv2.putText(
                canvas,
                f"left=RGB | right=role instances | unknown={unknown}/11 | reseeded={','.join(reseeded) or '-'}",
                (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "hand cyan/red | forearm yellow | sleeve violet | cable orange | cards green/blue/magenta",
                (10, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                cv2.LINE_AA,
            )
            encoder.stdin.write(canvas.tobytes())
            frame += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frame != FRAME_COUNT:
        raise RuntimeError("strict role review encoding failed")
    decode = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i",
        str(destination), "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decode.returncode:
        raise RuntimeError("strict role review full-decode failed")
    return {"video": ref(destination), "full_decode": True,
            "frame_count": FRAME_COUNT, "geometry": [1280, 480]}


def run_worker(output: Path, visual: Path, video: Path, hawor: Path) -> int:
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    gate = video_gate(video)
    with np.load(hawor, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
    if joints.shape != (2, FRAME_COUNT, 21, 2) or observed.shape != (2, FRAME_COUNT):
        raise RuntimeError("accepted bounded HaWoR shape drift")
    if sides != ["left", "right"]:
        raise RuntimeError("bounded HaWoR anatomical side order drift")
    frames = output / "input_frames"
    extract_frames(video, frames)
    plan = build_prompt_plan(sha256(video), sha256(hawor))
    atomic_json(output / "PROMPT_PLAN.json", plan)
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT,
    )
    packed_by_id: dict[str, np.ndarray] = {}
    states_by_id: dict[str, list[dict[str, Any]]] = {}
    quality_instances = []
    manifest_instances = []
    human_equipment_union = np.zeros((FRAME_COUNT, HEIGHT, WIDTH), bool)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index, prompt in enumerate(FROZEN_INSTANCE_PROMPTS, start=1):
            masks, ledger, evidence = process_instance(
                adapter.model, frames, prompt, joints, observed,
            )
            archive_path = output / "masks" / f"{prompt.instance_id}.npz"
            packed_by_id[prompt.instance_id] = save_packed(archive_path, masks)
            states_by_id[prompt.instance_id] = ledger
            if prompt.role in REMOVAL_ROLES:
                human_equipment_union |= masks
            counts = Counter(row["state"] for row in ledger)
            state_counts = {name: int(counts.get(name, 0))
                            for name in ("seeded", "tracked", "reseeded", "unknown")}
            manifest_instances.append({
                "instance_id": prompt.instance_id, "role": prompt.role,
                "physical_identity_policy": prompt.physical_identity_policy,
                "mask_archive": str(archive_path.relative_to(output)),
                "state_counts": state_counts,
                "initial_box_prompt": {
                    "frame_index": prompt.primary.frame_index,
                    "box_xywh": list(prompt.primary.box_xywh),
                    "runtime_semantics": "MULTIPLEX_GEOMETRIC_BOX",
                },
                "reseed_count": int(evidence["reseed_triggered"]),
            })
            quality_instances.append(evidence)
            print(json.dumps({
                "completed_instances": index,
                "total_instances": len(FROZEN_INSTANCE_PROMPTS),
                "instance_id": prompt.instance_id,
                "states": state_counts,
                "reseed_triggered": evidence["reseed_triggered"],
            }, ensure_ascii=False), flush=True)
            del masks
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        else:
            adapter.predictor.shutdown()
    union_path = output / "masks/HUMAN_EQUIPMENT_UNION.npz"
    save_packed(union_path, human_equipment_union)
    temporal = {
        "schema_version": "sam31-role-temporal-state-ledger-v1",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "state_semantics": {
            "seeded": "primary initial box/point result is quality-admissible",
            "tracked": "propagated result from a quality-admissible primary or fallback seed",
            "reseeded": "fallback anchor result after a quality trigger",
            "unknown": "no quality-admissible direct/tracked mask; not evidence of absence",
        },
        "instances": [
            {"instance_id": prompt.instance_id,
             "frames": states_by_id[prompt.instance_id]}
            for prompt in FROZEN_INSTANCE_PROMPTS
        ],
        "empty_mask_semantics": "UNKNOWN_NOT_ABSENT",
    }
    atomic_json(output / "TEMPORAL_STATE_LEDGER.json", temporal)
    quality = {
        "schema_version": "sam31-role-quality-trigger-ledger-v1",
        "session_id": SESSION_ID,
        "policy": "QUALITY_TRIGGERED_NOT_PERIODIC_MAX_ONE_RESEED",
        "instances": quality_instances,
    }
    atomic_json(output / "QUALITY_TRIGGER_LEDGER.json", quality)
    manifest = {
        "schema_version": "visual-role-mask-v2",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "image_domain": IMAGE_DOMAIN,
        "model": {
            "identity": "SAM3.1", "weight_sha256": CHECKPOINT_SHA256,
            "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
            "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
        },
        "roles": list(ROLE_NAMES), "instances": manifest_instances,
        "derived_masks": {
            "human_equipment_union": {
                "mask_archive": str(union_path.relative_to(output)),
                "source_roles": list(REMOVAL_ROLES),
                "direction": "DERIVED_FROM_ROLE_MASKS_ONLY",
            }
        },
        "temporal_state_ledger": "TEMPORAL_STATE_LEDGER.json",
        "tracker_role_created": False, "controller_role_created": False,
        "pico26_consumed": False,
    }
    schema = load_json(SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(manifest)
    atomic_json(output / "ROLE_MANIFEST.json", manifest)
    review = render_review(
        video, packed_by_id, states_by_id,
        visual / "0915_SAM31_STRICT_ROLE_REVIEW.mp4",
    )
    unknown_by_instance = {
        row["instance_id"]: row["state_counts"]["unknown"]
        for row in manifest_instances
    }
    card_unknown = [
        row["state_counts"]["unknown"] for row in manifest_instances
        if row["role"] == "task_object"
    ]
    worker_result = {
        "schema_version": "0915-sam31-strict-role-worker-result-v1",
        "status": "COMPLETED_DEVELOPMENT_CANARY",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "image_domain": IMAGE_DOMAIN,
        "video_gate": gate,
        "inputs": {"video": ref(video), "hawor": ref(hawor),
                   "checkpoint": ref(CHECKPOINT)},
        "access_contract": {
            "allowed_algorithm_inputs": [str(video), str(hawor), str(CHECKPOINT)],
            "pico26": "NOT_CONSUMED", "controller_pose": "NOT_CONSUMED",
            "trackingData_hand": "NOT_CONSUMED", "source_tree_mutated": False,
        },
        "prompt_plan": ref(output / "PROMPT_PLAN.json"),
        "role_manifest": ref(output / "ROLE_MANIFEST.json"),
        "temporal_state_ledger": ref(output / "TEMPORAL_STATE_LEDGER.json"),
        "quality_trigger_ledger": ref(output / "QUALITY_TRIGGER_LEDGER.json"),
        "unknown_frames_by_instance": unknown_by_instance,
        "all_three_task_objects_have_observed_evidence": all(
            value < FRAME_COUNT for value in card_unknown
        ),
        "review": review, "build_evidence": build_evidence,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "claim_limit": (
            "One-session development role masks for visual review. Unknown is not absence; "
            "manual boxes are prompts, not truth; no batch, contact or deployment authority."
        ),
    }
    atomic_json(output / "SAM31_WORKER_RESULT.json", worker_result)
    shutil.rmtree(frames)
    print(json.dumps({
        "status": worker_result["status"],
        "unknown_frames_by_instance": unknown_by_instance,
        "review": review["video"]["path"],
    }, ensure_ascii=False), flush=True)
    return 0


def write_terminal(
    output: Path, visual: Path, receipt: Path, packet: dict[str, Any],
    *, status: str, first_blocker: str | None, worker: dict[str, Any] | None,
) -> None:
    result = {
        "schema_version": "0915-sam31-strict-role-canary-result-v1",
        "task_id": TASK_ID, "status": status, "session_id": SESSION_ID,
        "weights": packet["weights"], "image_domain": IMAGE_DOMAIN,
        "session_admission": (
            "AWAITING_USER_VISUAL_REVIEW" if worker is not None else "NOT_PRODUCED"
        ),
        "first_blocker": first_blocker,
        "worker_result": ref(output / "SAM31_WORKER_RESULT.json")
        if worker is not None else None,
        "gpu_command_receipt": ref(output / "GPU_COMMAND_RECEIPT.json")
        if (output / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "visual": ref(visual / "0915_SAM31_STRICT_ROLE_REVIEW.mp4")
        if (visual / "0915_SAM31_STRICT_ROLE_REVIEW.mp4").is_file() else None,
        "source_mutated": False, "batch_started": False,
        "tracker_role_created": False, "controller_role_created": False,
        "pico26_consumed": False,
        "next_authority": "USER_VISUAL_REVIEW_REQUIRED_NO_AUTOMATIC_BATCH_SUCCESSOR",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-sam31-strict-role-canary-run-receipt-v1",
        "task_id": TASK_ID, "status": status,
        "result": ref(output / "RESULT.json"), "terminal_receipt": ref(receipt),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    if not VIDEO.is_file() or not HAWOR.is_file():
        raise RuntimeError("fixed canary input is missing")
    output.mkdir(parents=True)
    heartbeat()
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output-root", str(output), "--visual-root", str(visual),
        "--video", str(VIDEO), "--hawor", str(HAWOR),
    ]
    lease_command = [
        sys.executable, str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", TASK_ID, "--attempt-id", output.name,
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds), "--wall-seconds", "7200",
        "--receipt", str(gpu_receipt), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-sam31-strict-role-command-v1",
        "task_id": TASK_ID, "weights": packet["weights"],
        "worker_command": worker_command, "lease_command": lease_command,
        "inputs": {"video": ref(VIDEO), "hawor": ref(HAWOR)},
    })
    with (output / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            time.sleep(30)
            heartbeat()
    gpu = load_json(gpu_receipt) if gpu_receipt.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        blocked = gpu.get("status") == "BLOCKED_RESOURCE"
        write_terminal(
            output, visual, receipt, packet,
            status="BLOCKED_RESOURCE" if blocked else "FAILED_RUNTIME_FINAL",
            first_blocker=str(gpu.get("reason") or gpu.get("error") or "SAM31_RUNTIME_FAILED"),
            worker=None,
        )
        return 3 if blocked else 2
    worker = load_json(output / "SAM31_WORKER_RESULT.json")
    write_terminal(
        output, visual, receipt, packet, status="PASSED",
        first_blocker=None, worker=worker,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--hawor", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.video is None or args.hawor is None:
            raise RuntimeError("worker requires --video and --hawor")
        return run_worker(
            args.output_root.resolve(), args.visual_root.resolve(),
            args.video.resolve(strict=True), args.hawor.resolve(strict=True),
        )
    if args.receipt is None:
        raise RuntimeError("orchestrator requires --receipt")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
