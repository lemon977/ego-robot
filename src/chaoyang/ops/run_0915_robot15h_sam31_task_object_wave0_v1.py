#!/usr/bin/env python3
"""Run pinned SAM3.1 task-object identity tracking on the frozen 0915 W0.

The runner uses class text only.  It never contains a session-specific box,
point, or image coordinate.  Every detector proposal is retained as raw
evidence; only distinct, shape-supported visible instances are propagated and
may become development Object6D mask inputs.  Empty/rejected frames are
UNKNOWN, never evidence that the physical object is absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch

from chaoyang.governance.robot15h_task_specs_v1 import SAM31_WEIGHT, WINDOW_RUN_ID, build_packet
from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as common
from chaoyang.ops import run_0915_leftmono_sam31_masks_v1 as legacy
from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module


ROOT = Path(__file__).resolve().parents[3]
PRIMARY_TASK_ID = "0915_robot15h_sam31_task_object_wave0_v1"
RECOVERY_TASK_ID = "0915_robot15h_sam31_task_object_wave0_recovery_v1"
TASK_ID = PRIMARY_TASK_ID
PHASE = "ROBOT15H_SAM31_TASK_OBJECT_IDENTITY_WAVE0"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
PREPARED_MANIFEST = ROOT / "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/PREPARED_MANIFEST.json"
HAND_RESULT = ROOT / "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/attempts/attempt_0001/RESULT.json"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TASK_OBJECT_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TASK_OBJECT_WAVE0_V1_RESULT.json"
CHECKPOINT = ROOT / SAM31_WEIGHT
CHECKPOINT_SHA256 = adapter_module.CHECKPOINT_SHA256
CODE_ROOT = ROOT / "vendor/SAM3"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
GPU_LEASE_WRAPPER = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
WIDTH, HEIGHT = 1280, 960

TASK_POLICIES: dict[str, dict[str, Any]] = {
    "playing_cards": {
        "text": "playing card", "instance_prefix": "playing_card", "maximum_instances": 3,
        "minimum_area": 180, "maximum_area_fraction": 0.035,
        "aspect_range": (1.05, 2.35), "minimum_rotated_fill": 0.42,
    },
    "potato_chips": {
        "text": "potato chip", "instance_prefix": "potato_chip", "maximum_instances": 3,
        "minimum_area": 90, "maximum_area_fraction": 0.020,
        "aspect_range": (1.0, 2.80), "minimum_rotated_fill": 0.28,
    },
}
COLORS = ((40, 220, 255), (255, 120, 40), (210, 60, 220))


def configure_task(task_id: str) -> None:
    """Select the governed primary or bounded empty-candidate recovery namespace."""

    global TASK_ID, PHASE, OUTPUT, VISUAL, TERMINAL_RECEIPT
    if task_id == PRIMARY_TASK_ID:
        TASK_ID = task_id
        PHASE = "ROBOT15H_SAM31_TASK_OBJECT_IDENTITY_WAVE0"
        OUTPUT = ROOT / f"_run/current/{task_id}/attempts/attempt_0001"
        VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TASK_OBJECT_V1"
        TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TASK_OBJECT_WAVE0_V1_RESULT.json"
    elif task_id == RECOVERY_TASK_ID:
        TASK_ID = task_id
        PHASE = "ROBOT15H_SAM31_TASK_OBJECT_IDENTITY_WAVE0_EMPTY_CANDIDATE_RECOVERY"
        OUTPUT = ROOT / f"_run/current/{task_id}/attempts/attempt_0001"
        VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TASK_OBJECT_RECOVERY_V1"
        TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TASK_OBJECT_WAVE0_RECOVERY_V1_RESULT.json"
    else:
        raise RuntimeError(f"unsupported task-object task id: {task_id}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return common.sha256(path)


def ref(path: Path) -> dict[str, Any]:
    return common.file_ref(path)


def published_ref(current: Path, published: Path) -> dict[str, Any]:
    return common.published_ref(current, published)


def atomic_json(path: Path, value: Any) -> None:
    common.atomic_json(path, value)


def atomic_json_new(path: Path, value: Any) -> None:
    common.atomic_json_new(path, value)


def canonical_sha(value: Any) -> str:
    return common.canonical_sha256(value)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def prompt_for_task(task: str) -> str:
    return str(TASK_POLICIES[task]["text"])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != [SAM31_WEIGHT]:
        raise RuntimeError("current task-object packet differs from frozen spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task-object node is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task-object node is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task-object packet SHA drift")
    return packet, packet_path


def heartbeat(status: str, gpu_id: int | None = None) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
        "--pid", str(os.getpid()), "--status", status, "--phase", PHASE,
    ]
    if gpu_id is not None:
        command += ["--gpu-id", str(gpu_id)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def w0_rows() -> list[dict[str, Any]]:
    rows = [row for row in load_json(INVENTORY).get("sessions", []) if row.get("wave") == "W0"]
    if len(rows) != 4 or sum(int(row["frame_count"]) for row in rows) != 1058:
        raise RuntimeError("frozen task-object W0 cohort drift")
    prepared = {row["session_id"]: row for row in load_json(PREPARED_MANIFEST)["results"]}
    result: list[dict[str, Any]] = []
    for row in rows:
        item = prepared.get(row["session_id"])
        if item is None or int(item["frame_count"]) != int(row["frame_count"]):
            raise RuntimeError(f"prepared resize-only input drift: {row['session_id']}")
        expected_domain = {
            "lens_undistortion": False, "operation": "CROP_THEN_RESIZE_ONLY",
            "output_size": [WIDTH, HEIGHT], "physical_left_source_index": 1,
            "physical_right_source_index": 0, "pico26_hand_consumed": False,
            "remap_applied": False, "trackingData_hand_consumed": False,
        }
        if item["input_domain"] != expected_domain:
            raise RuntimeError(f"prepared image-domain drift: {row['session_id']}")
        video = Path(str(item["prepared_video"]["path"])).resolve(strict=True)
        if item["prepared_video"]["sha256"] != sha256(video):
            raise RuntimeError(f"prepared video SHA drift: {row['session_id']}")
        result.append({**row, "prepared_video": str(video)})
    return result


def _extract_frames(video: Path, destination: Path, expected: int) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vsync", "0",
        "-start_number", "0", str(destination / "%06d.jpg"),
    ], capture_output=True, text=True, check=False)
    if completed.returncode or len(list(destination.glob("*.jpg"))) != expected:
        raise RuntimeError("frame extraction did not preserve the frozen denominator")


def deterministic_anchor_schedule(frame_count: int) -> list[int]:
    """Finite, session-agnostic early-frame schedule; never a spatial prompt."""

    if frame_count < 1:
        return []
    candidates = [0, int(round((frame_count - 1) * 0.04)), int(round((frame_count - 1) * 0.09)),
                  int(round((frame_count - 1) * 0.16))]
    return list(dict.fromkeys(max(0, min(frame_count - 1, value)) for value in candidates))


def rank_clear_anchors(frames: Path, frame_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in deterministic_anchor_schedule(frame_count):
        image = cv2.imread(str(frames / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"missing extracted anchor frame {frame}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        saturation = float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].mean())
        score = math.log1p(max(sharpness, 0.0)) + 0.02 * contrast + 0.005 * saturation
        rows.append({"frame_index": frame, "sharpness": sharpness, "contrast": contrast,
                     "mean_saturation": saturation, "clear_frame_score": score})
    return sorted(rows, key=lambda row: (-row["clear_frame_score"], row["frame_index"]))


def _largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    return max(contours, key=cv2.contourArea), len(contours)


def candidate_row(mask: np.ndarray, score: float, raw_id: int, task: str) -> dict[str, Any]:
    """Measure one raw text proposal without granting semantic identity."""

    policy = TASK_POLICIES[task]
    area = int(mask.sum())
    contour, components = _largest_contour(mask)
    reasons: list[str] = []
    rotated_aspect = rotated_fill = None
    centroid = None
    touches_boundary = False
    if contour is None or area == 0:
        reasons.append("EMPTY")
    else:
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
        short, long = sorted((max(float(rw), 1e-6), max(float(rh), 1e-6)))
        rotated_aspect = long / short
        rotated_fill = area / max(float(rw * rh), 1.0)
        centroid = [float(cx), float(cy)]
        yy, xx = np.where(mask)
        touches_boundary = bool(xx.min() == 0 or yy.min() == 0 or xx.max() == WIDTH - 1 or yy.max() == HEIGHT - 1)
    if area < int(policy["minimum_area"]):
        reasons.append("AREA_BELOW_TASK_MINIMUM")
    if area > int(WIDTH * HEIGHT * float(policy["maximum_area_fraction"])):
        reasons.append("SUPPORT_SCALE_AREA_REJECT")
    if components != 1:
        reasons.append("MERGED_OR_FRAGMENTED_COMPONENTS")
    aspect_min, aspect_max = policy["aspect_range"]
    if rotated_aspect is not None and not float(aspect_min) <= rotated_aspect <= float(aspect_max):
        reasons.append("TASK_SHAPE_ASPECT_REJECT")
    if rotated_fill is not None and rotated_fill < float(policy["minimum_rotated_fill"]):
        reasons.append("TASK_SHAPE_FILL_REJECT")
    if touches_boundary:
        reasons.append("BOUNDARY_TRUNCATED_SUPPORT_AMBIGUITY")
    return {
        "raw_id": int(raw_id), "score": float(score), "area_pixels": area,
        "area_fraction": area / float(WIDTH * HEIGHT), "component_count": components,
        "rotated_aspect": rotated_aspect, "rotated_fill": rotated_fill,
        "centroid_xy": centroid, "touches_image_boundary": touches_boundary,
        "eligible_shape_candidate": not reasons, "reasons": reasons,
        "support_tray_or_bowl_admitted": False,
    }


def select_separate_instances(
    masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, task: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Greedily admit distinct visible masks, never a union mask."""

    rows = [candidate_row(mask, float(scores[i]), int(ids[i]), task) for i, mask in enumerate(masks)]
    index_by_id = {int(value): index for index, value in enumerate(ids)}
    selected: list[int] = []
    maximum = int(TASK_POLICIES[task]["maximum_instances"])
    for row in sorted(rows, key=lambda value: (-value["score"], value["area_pixels"], value["raw_id"])):
        if not row["eligible_shape_candidate"] or len(selected) >= maximum:
            continue
        mask = masks[index_by_id[row["raw_id"]]]
        overlaps = []
        for chosen_id in selected:
            chosen = masks[index_by_id[chosen_id]]
            intersection = int(np.logical_and(mask, chosen).sum())
            overlaps.append(intersection / max(1, min(int(mask.sum()), int(chosen.sum()))))
        if overlaps and max(overlaps) > 0.12:
            row["reasons"].append("OVERLAPS_SELECTED_PHYSICAL_INSTANCE")
            row["eligible_shape_candidate"] = False
            continue
        selected.append(int(row["raw_id"]))
    for row in rows:
        row["selected_as_separate_instance"] = row["raw_id"] in selected
        if row["eligible_shape_candidate"] and not row["selected_as_separate_instance"]:
            row["reasons"].append("BOUNDED_INSTANCE_LIMIT_OR_LOWER_RANK")
    # Spatial naming is deterministic, but does not assert cross-session identity.
    selected.sort(key=lambda raw_id: tuple(rows[[r["raw_id"] for r in rows].index(raw_id)]["centroid_xy"] or [math.inf, math.inf]))
    return selected, rows


def save_initial_candidates(
    path: Path, masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, frame: int,
    *, published_path: Path,
) -> dict[str, Any]:
    # ``reshape(0, -1)`` is ambiguous in NumPy.  Zero text candidates are a
    # valid observation and must serialize as UNKNOWN evidence, not become a
    # runtime failure or a fabricated absence claim.
    height, width = (int(masks.shape[-2]), int(masks.shape[-1]))
    packed = np.packbits(masks.reshape((len(masks), height * width)), axis=1, bitorder="big")
    atomic_npz(path, packed=packed, candidate_count=np.int32(len(masks)), height=np.int32(height),
               width=np.int32(width), bitorder=np.asarray("big"), scores=np.asarray(scores, np.float32),
               raw_ids=np.asarray(ids, np.int64), frame_index=np.int32(frame),
               semantic_admitted=np.asarray(False), union_mask_created=np.asarray(False))
    return published_ref(path, published_path)


def save_packed(path: Path, masks: np.ndarray, *, published_path: Path) -> dict[str, Any]:
    height, width = (int(masks.shape[-2]), int(masks.shape[-1]))
    packed = np.packbits(masks.reshape((len(masks), height * width)), axis=1, bitorder="big")
    atomic_npz(path, packed=packed, frame_count=np.int32(len(masks)), height=np.int32(height),
               width=np.int32(width), bitorder=np.asarray("big"))
    return published_ref(path, published_path)


def load_packed(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        packed = archive["packed"]
        count, height, width = int(archive["frame_count"]), int(archive["height"]), int(archive["width"])
    return np.unpackbits(packed, axis=1, count=height * width, bitorder="big").reshape(count, height, width).astype(bool)


def _propagate_selected(model: Any, state: Any, anchor: int, selected: Sequence[int], frame_count: int) -> dict[int, np.ndarray]:
    tracks = {int(raw_id): np.zeros((frame_count, HEIGHT, WIDTH), bool) for raw_id in selected}
    routes = [(False, frame_count - anchor)] + ([(True, anchor + 1)] if anchor else [])
    for reverse, maximum in routes:
        for frame_index, outputs in model.propagate_in_video(
            inference_state=state, start_frame_idx=anchor, max_frame_num_to_track=maximum,
            reverse=reverse, output_prob_thresh=0.5,
        ):
            masks, _scores, ids = legacy.normalize(outputs, HEIGHT, WIDTH)
            by_id = {int(raw_id): masks[index] for index, raw_id in enumerate(ids)}
            for raw_id in selected:
                if int(raw_id) in by_id and 0 <= int(frame_index) < frame_count:
                    tracks[int(raw_id)][int(frame_index)] = by_id[int(raw_id)]
    return tracks


def evaluate_identity(track: np.ndarray, anchor: int, task: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    areas = track.reshape(len(track), -1).sum(axis=1).astype(int)
    anchor_area = int(areas[anchor])
    valid = np.zeros(len(track), bool)
    rows: list[dict[str, Any]] = []
    previous_centroid: np.ndarray | None = None
    for frame, mask in enumerate(track):
        reasons: list[str] = []
        area = int(areas[frame])
        if not area:
            reasons.append("RAW_INSTANCE_NOT_VISIBLE_OR_TRACK_LOST")
        elif anchor_area <= 0 or not 0.18 <= area / anchor_area <= 4.5:
            reasons.append("AREA_RATIO_OUT_OF_RANGE")
        contour, components = _largest_contour(mask)
        if area and components > 2:
            reasons.append("TEMPORAL_FRAGMENTATION")
        centroid = None
        if contour is not None:
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"]:
                centroid = np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        jump = None
        if centroid is not None and previous_centroid is not None:
            jump = float(np.linalg.norm(centroid - previous_centroid) / max(math.sqrt(area), 1.0))
            if jump > 2.8:
                reasons.append("CENTROID_JUMP_SCALE_NORMALISED")
        if centroid is not None:
            previous_centroid = centroid
        valid[frame] = not reasons
        rows.append({"frame_id": frame, "raw_present": bool(area), "semantic_admitted": bool(valid[frame]),
                     "tracking_state": "seeded" if frame == anchor and valid[frame] else ("tracked" if valid[frame] else "unknown"),
                     "visibility_state": "VISIBLE_CANDIDATE" if area else "UNKNOWN",
                     "area_pixels": area, "centroid_xy": centroid.tolist() if centroid is not None else None,
                     "centroid_jump_scale_normalised": jump, "reasons": reasons})
    summary = summarize_identity_quality(valid, len(track), conflict_frames=0)
    return valid, rows, summary


def summarize_identity_quality(
    valid: np.ndarray, frame_count: int, *, conflict_frames: int,
) -> dict[str, Any]:
    admitted = int(np.count_nonzero(valid))
    minimum = max(12, int(math.ceil(0.20 * frame_count)))
    maximum_conflicts = max(1, int(math.floor(0.02 * frame_count)))
    return {
        "semantic_admitted_frames": admitted,
        "unknown_frames": frame_count - admitted,
        "semantic_admission_fraction": admitted / frame_count,
        "stable_instance": admitted >= minimum and conflict_frames <= maximum_conflicts,
        "minimum_stable_frames": minimum,
        "inter_instance_identity_conflict_frames": conflict_frames,
        "maximum_identity_conflict_frames": maximum_conflicts,
        "authority": "DEVELOPMENT_VISIBLE_MASK_PROXY",
    }


def apply_inter_instance_exclusion(
    tracks: Mapping[int, np.ndarray],
    evaluations: dict[int, tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]],
    *, maximum_overlap_over_smaller: float = 0.20,
) -> None:
    """Fail closed when two persistent IDs collapse onto the same pixels."""

    identities = sorted(tracks)
    conflicts = {raw_id: set() for raw_id in identities}
    for offset, left in enumerate(identities):
        for right in identities[offset + 1:]:
            for frame, (left_mask, right_mask) in enumerate(zip(tracks[left], tracks[right], strict=True)):
                smaller = min(int(left_mask.sum()), int(right_mask.sum()))
                if smaller == 0:
                    continue
                overlap = int(np.logical_and(left_mask, right_mask).sum()) / smaller
                if overlap > maximum_overlap_over_smaller:
                    conflicts[left].add(frame)
                    conflicts[right].add(frame)
    for raw_id, frames in conflicts.items():
        valid, rows, _quality = evaluations[raw_id]
        for frame in sorted(frames):
            valid[frame] = False
            rows[frame]["semantic_admitted"] = False
            rows[frame]["tracking_state"] = "unknown"
            rows[frame]["reasons"].append("INTER_INSTANCE_OVERLAP_IDENTITY_CONFLICT")
        evaluations[raw_id] = (
            valid,
            rows,
            summarize_identity_quality(valid, len(valid), conflict_frames=len(frames)),
        )


def run_text_seed(model: Any, frames: Path, anchor: int, task: str, frame_count: int) -> tuple[dict[int, np.ndarray], dict[str, Any], dict[str, Any]]:
    state = model.init_state(resource_path=str(frames), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        _, initial = model.add_prompt(inference_state=state, frame_idx=anchor,
                                      text_str=prompt_for_task(task), output_prob_thresh=0.5)
        masks, scores, ids = legacy.normalize(initial, HEIGHT, WIDTH)
        selected, candidates = select_separate_instances(masks, scores, ids, task)
        route, action_ids = model.parse_action_history_for_propagation(state)
        if route != "propagation_full" or action_ids is not None:
            raise RuntimeError("text seed did not preserve full semantic propagation route")
        tracks = _propagate_selected(model, state, anchor, selected, frame_count) if selected else {}
        by_id = {int(raw_id): masks[index] for index, raw_id in enumerate(ids)}
        for raw_id in selected:
            if not tracks[raw_id][anchor].any():
                tracks[raw_id][anchor] = by_id[raw_id]
        raw = {"masks": masks, "scores": scores, "raw_ids": ids, "frame_index": anchor}
        evidence = {"anchor_frame": anchor, "prompt_text": prompt_for_task(task),
                    "prompt_geometry": "NONE_TEXT_ONLY", "candidate_rows": candidates,
                    "selected_raw_ids": selected, "selected_instance_count": len(selected),
                    "propagation_route": route, "action_object_ids": action_ids}
        return tracks, evidence, raw
    finally:
        state.clear()


def _published(current: Path, staging: Path, target: Path) -> dict[str, Any]:
    return published_ref(current, target / current.relative_to(staging))


def render_review(
    video: Path, destination: Path, session_id: str,
    raw_tracks: Mapping[str, np.ndarray], semantic_tracks: Mapping[str, np.ndarray], expected: int,
) -> dict[str, Any]:
    """Render raw-instance and admitted-instance evidence without unioning masks."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 420))
    if not writer.isOpened():
        raise RuntimeError("cannot open task-object review writer")
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            raw_overlay, semantic_overlay = image.copy(), image.copy()
            states: list[str] = []
            for index, instance_id in enumerate(sorted(raw_tracks)):
                color = np.asarray(COLORS[index % len(COLORS)], np.float32)
                raw_mask = raw_tracks[instance_id][frame]
                semantic_mask = semantic_tracks[instance_id][frame]
                if raw_mask.any():
                    raw_overlay[raw_mask] = (0.5 * raw_overlay[raw_mask] + 0.5 * color).astype(np.uint8)
                if semantic_mask.any():
                    semantic_overlay[semantic_mask] = (0.42 * semantic_overlay[semantic_mask] + 0.58 * color).astype(np.uint8)
                states.append(f"{instance_id}={'tracked' if semantic_mask.any() else 'unknown'}")
            panels = [cv2.resize(value, (426, 320), interpolation=cv2.INTER_AREA)
                      for value in (image, raw_overlay, semantic_overlay)]
            canvas = np.zeros((420, 1280, 3), np.uint8)
            canvas[76:396, :1278] = np.hstack(panels)
            cv2.putText(canvas, f"{session_id} | frame {frame:04d} | physical-left resize-only | no lens remap",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Raw RGB | separate raw tracked instances | admitted visible semantic instances",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1, cv2.LINE_AA)
            summary = " | ".join(states) if states else "no admitted initial instance; semantic UNKNOWN"
            cv2.putText(canvas, summary[:170], (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (200, 240, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            frame += 1
    finally:
        capture.release()
        writer.release()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(destination),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decode = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(destination),
                             "-f", "null", "-"], capture_output=True, text=True, check=False)
    gate = {"frames": int(stream["nb_read_frames"]), "width": int(stream["width"]),
            "height": int(stream["height"]), "fps": stream["r_frame_rate"],
            "full_decode": decode.returncode == 0}
    if frame != expected or gate != {"frames": expected, "width": 1280, "height": 420,
                                   "fps": "30/1", "full_decode": True}:
        raise RuntimeError(f"task-object review decode mismatch: written={frame}, gate={gate}")
    return {"video": ref(destination), **gate}


def process_session(model: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    session_id, task = str(row["session_id"]), str(row["task"])
    expected, video = int(row["frame_count"]), Path(str(row["prepared_video"]))
    source_before = ref(video)
    target = OUTPUT / "sessions" / task / session_id
    staging = target.with_name(f".{session_id}.staging-{uuid.uuid4().hex}")
    if target.exists() or staging.exists():
        raise RuntimeError(f"fresh task-object destination required: {target}")
    staging.mkdir(parents=True)
    frames = staging / "input_frames"
    _extract_frames(video, frames, expected)
    ranked_anchors = rank_clear_anchors(frames, expected)
    attempts: list[dict[str, Any]] = []
    chosen_tracks: dict[int, np.ndarray] = {}
    chosen_evidence: dict[str, Any] | None = None
    chosen_anchor = -1
    # One initial prompt plus at most one quality-triggered independent reseed.
    for attempt_index, anchor_row in enumerate(ranked_anchors[:2]):
        anchor = int(anchor_row["frame_index"])
        tracks, evidence, raw = run_text_seed(model, frames, anchor, task, expected)
        raw_path = staging / "raw_candidate" / f"seed_{attempt_index}_frame_{anchor:06d}.npz"
        evidence["raw_initial_candidates"] = save_initial_candidates(
            raw_path, raw["masks"], raw["scores"], raw["raw_ids"], anchor,
            published_path=target / raw_path.relative_to(staging),
        )
        evaluations = {raw_id: evaluate_identity(track, anchor, task)[2] for raw_id, track in tracks.items()}
        evidence["preliminary_identity_quality"] = {str(key): value for key, value in evaluations.items()}
        attempts.append(evidence)
        if any(value["stable_instance"] for value in evaluations.values()):
            chosen_tracks, chosen_evidence, chosen_anchor = tracks, evidence, anchor
            break
        # The second iteration is the single quality-triggered reseed.  It uses
        # an independently ranked frame and another text-only detection.
    if chosen_evidence is None and attempts:
        chosen_evidence = attempts[-1]
        chosen_anchor = int(chosen_evidence["anchor_frame"])
    instance_rows: list[dict[str, Any]] = []
    stable_masks: dict[str, np.ndarray] = {}
    named_raw_tracks: dict[str, np.ndarray] = {}
    named_semantic_tracks: dict[str, np.ndarray] = {}
    if chosen_tracks:
        evaluations = {
            raw_id: evaluate_identity(track, chosen_anchor, task)
            for raw_id, track in chosen_tracks.items()
        }
        apply_inter_instance_exclusion(chosen_tracks, evaluations)
        ordered = sorted(chosen_tracks, key=lambda raw_id: next(
            tuple(row_["centroid_xy"] or [math.inf, math.inf])
            for row_ in chosen_evidence["candidate_rows"] if row_["raw_id"] == raw_id
        ))
        for number, raw_id in enumerate(ordered):
            valid, states, quality = evaluations[raw_id]
            semantic = chosen_tracks[raw_id] & valid[:, None, None]
            instance_id = f"{TASK_POLICIES[task]['instance_prefix']}_{number:02d}"
            raw_path = staging / "raw_track" / f"{instance_id}.npz"
            semantic_path = staging / "semantic" / f"{instance_id}.npz"
            state_path = staging / "temporal" / f"{instance_id}_STATE.json"
            atomic_json(state_path, {"schema_version": "sam31-task-object-state-v1", "instance_id": instance_id,
                                     "empty_semantic_mask": "UNKNOWN_NOT_ABSENT", "frames": states})
            stable = bool(quality["stable_instance"])
            named_raw_tracks[instance_id] = chosen_tracks[raw_id]
            named_semantic_tracks[instance_id] = semantic
            if stable:
                stable_masks[instance_id] = semantic
            instance_rows.append({
                "instance_id": instance_id, "raw_track_id_scope": "SESSION_PROMPT_STATE_ONLY",
                "seed_raw_id": int(raw_id), "status": "PASS_VISIBLE_INSTANCE_PROXY" if stable else "REJECTED_TEMPORAL_QUALITY",
                "object_mask_consumer_allowed": stable,
                "raw_archive": save_packed(raw_path, chosen_tracks[raw_id], published_path=target / raw_path.relative_to(staging)),
                "semantic_archive": save_packed(semantic_path, semantic, published_path=target / semantic_path.relative_to(staging)),
                "state_ledger": _published(state_path, staging, target), "quality": quality,
                "visible_surface_only": True, "hidden_shape_or_extent_inferred": False,
            })
    manifest_path = staging / "OBJECT_INSTANCE_MANIFEST.json"
    atomic_json(manifest_path, {
        "schema_version": "sam31-task-object-instance-manifest-v1", "session_id": session_id,
        "task": task, "frame_count": expected, "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
        "prompt_text": prompt_for_task(task), "manual_coordinates_or_boxes": False,
        "anchor_ranking": ranked_anchors, "prompt_attempts": attempts,
        "quality_triggered_reseed_count": max(0, len(attempts) - 1), "maximum_reseed_count": 1,
        "instances": instance_rows, "union_mask_created": False,
        "support_tray_or_bowl_is_task_object": False, "empty_semantic_mask": "UNKNOWN_NOT_ABSENT",
        "model": {"identity": "SAM3.1", "weight_sha256": CHECKPOINT_SHA256},
    })
    VISUAL.mkdir(parents=True, exist_ok=True)
    review = render_review(
        video, VISUAL / f"{session_id}_SAM31_TASK_OBJECT_REVIEW.mp4", session_id,
        named_raw_tracks, named_semantic_tracks, expected,
    )
    shutil.rmtree(frames)
    if ref(video) != source_before:
        raise RuntimeError("task-object input changed during execution")
    result = {
        "schema_version": "0915-robot15h-sam31-task-object-session-result-v1",
        "session_id": session_id, "task": task, "source_group": row["source_group"],
        "status": "PASSED_VISIBLE_OBJECT_MASK_PROXY" if stable_masks else "REJECTED_QUALITY",
        "execution_completed": True, "frame_count": expected,
        "instance_counts": {"raw_selected": len(instance_rows),
                            "stable_consumer_allowed": sum(item["object_mask_consumer_allowed"] for item in instance_rows),
                            "quality_rejected": sum(not item["object_mask_consumer_allowed"] for item in instance_rows)},
        "object_mask_consumer_allowed": bool(stable_masks), "instances": instance_rows,
        "instance_manifest": _published(manifest_path, staging, target), "input_video": source_before,
        "review": review,
        "source_mutated": False, "training_eligible": False, "physical_deployment_authorized": False,
    }
    atomic_json(staging / "RESULT.json", result)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    final = load_json(target / "RESULT.json")
    final["result"] = ref(target / "RESULT.json")
    return final


def build_signature(packet_path: Path, executor_epoch: int) -> dict[str, Any]:
    sessions = [{"session_id": row["session_id"], "task": row["task"], "source_group": row["source_group"],
                 "frame_count": int(row["frame_count"]), "prepared_video": ref(Path(row["prepared_video"]))}
                for row in w0_rows()]
    payload = {
        "schema_version": "0915-robot15h-sam31-task-object-run-signature-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "executor_epoch": executor_epoch,
        "weights": [SAM31_WEIGHT], "sessions": sessions,
        "contracts": {"task_packet": ref(packet_path), "prepared_manifest": ref(PREPARED_MANIFEST),
                      "hand_predecessor_result": ref(HAND_RESULT)},
        "runtime": {"runner": ref(Path(__file__)), "adapter": ref(Path(adapter_module.__file__)),
                    "checkpoint": ref(CHECKPOINT), "lease_wrapper": ref(GPU_LEASE_WRAPPER)},
        "input_policy": {"physical_left_source_index": 1, "operation": "CROP_THEN_RESIZE_ONLY",
                         "lens_undistortion": False},
        "prompt_policy": {"playing_cards": "playing card", "potato_chips": "potato chip",
                          "manual_coordinates_or_boxes": False, "maximum_quality_reseeds": 1},
        "identity_policy": {"union_masks_forbidden": True, "support_objects_rejected": True,
                            "empty_semantic_is_unknown": True, "visible_surface_only": True},
    }
    return {**payload, "run_signature_sha256": canonical_sha(payload)}


def validate_claim(path: Path, signature_sha: str, executor_epoch: int, require_descendant: bool) -> None:
    claim = load_json(path)
    pid = claim.get("pid")
    if (claim.get("task_id") != TASK_ID or claim.get("weights") != [SAM31_WEIGHT]
            or claim.get("status") != "CLAIMED" or claim.get("executor_epoch") != executor_epoch
            or claim.get("run_signature_sha256") != signature_sha
            or claim.get("unique_write_root") != str(OUTPUT.resolve()) or not isinstance(pid, int)
            or common.process_start_ticks(pid) != claim.get("proc_start_ticks")):
        raise RuntimeError("task-object writer fence mismatch")
    if require_descendant and not common.process_has_ancestor(os.getpid(), pid):
        raise RuntimeError("task-object worker is outside writer ancestry")


def run_worker(claim_path: Path, signature_path: Path, executor_epoch: int) -> int:
    validate_route()
    signature = load_json(signature_path)
    stored = signature.pop("run_signature_sha256", None)
    if stored != canonical_sha(signature):
        raise RuntimeError("task-object run signature digest mismatch")
    signature["run_signature_sha256"] = stored
    validate_claim(claim_path, stored, executor_epoch, require_descendant=True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    adapter, build_evidence = adapter_module.build_pinned_adapter(official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT)
    results: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for row in w0_rows():
            try:
                result = process_session(adapter.model, row)
            except Exception as error:
                target = OUTPUT / "sessions" / str(row["task"]) / str(row["session_id"])
                target.mkdir(parents=True, exist_ok=True)
                failure = {"schema_version": "0915-robot15h-sam31-task-object-session-result-v1",
                           "session_id": row["session_id"], "task": row["task"],
                           "source_group": row["source_group"], "status": "FAILED_RUNTIME",
                           "execution_completed": False, "frame_count": int(row["frame_count"]),
                           "first_blocker": f"{type(error).__name__}:{error}",
                           "object_mask_consumer_allowed": False, "source_mutated": False}
                atomic_json(target / "RESULT.json", failure)
                result = {**failure, "result": ref(target / "RESULT.json")}
            results.append(result)
            print(json.dumps({"session_id": row["session_id"], "status": result["status"]}, ensure_ascii=False), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        close = getattr(adapter, "close", None)
        close() if callable(close) else adapter.predictor.shutdown()
    counts = {
        "sessions_total": 4,
        "sessions_consumer_allowed": sum(row.get("object_mask_consumer_allowed") is True for row in results),
        "sessions_quality_rejected": sum(row.get("status") == "REJECTED_QUALITY" for row in results),
        "sessions_failed_runtime": sum(row.get("status") == "FAILED_RUNTIME" for row in results),
        "stable_visible_instances": sum(row.get("instance_counts", {}).get("stable_consumer_allowed", 0) for row in results),
    }
    batch = {"schema_version": "0915-robot15h-sam31-task-object-batch-v1",
             "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED_ALL_TERMINAL" if not counts["sessions_failed_runtime"] else "COMPLETED_WITH_RUNTIME_FAILURE",
             "counts": counts, "model_load_count": 1, "model_build_evidence": build_evidence,
             "fresh_session_state_per_prompt": True, "sessions": results,
             "object_mask_consumer_allowed": counts["sessions_consumer_allowed"] > 0,
             "raw_candidate_is_semantic_truth": False, "union_mask_created": False,
             "source_mutated": False, "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
             "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    atomic_json(OUTPUT / "OBJECT_IDENTITY_BATCH.json", batch)
    # Keep the role-specific name for human review and the conventional batch
    # name for generic governed downstream joins. Both bind identical content.
    atomic_json(OUTPUT / "BATCH_RESULT.json", batch)
    return 0 if not counts["sessions_failed_runtime"] else 2


def write_terminal(packet: Mapping[str, Any], status: str, blocker: str | None) -> None:
    batch = load_json(OUTPUT / "OBJECT_IDENTITY_BATCH.json") if (OUTPUT / "OBJECT_IDENTITY_BATCH.json").is_file() else None
    result = {"schema_version": "0915-robot15h-sam31-task-object-result-v1", "task_id": TASK_ID,
              "window_run_id": WINDOW_RUN_ID, "status": status, "first_blocker": blocker,
              "weights": packet["weights"], "execution_completed": bool(batch and not batch["counts"]["sessions_failed_runtime"]),
              "counts": batch.get("counts") if batch else None,
              "object_mask_consumer_allowed": bool(batch and batch.get("object_mask_consumer_allowed")),
              "object_identity_batch": ref(OUTPUT / "OBJECT_IDENTITY_BATCH.json") if batch else None,
              "gpu_command_receipt": ref(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else None,
              "source_mutated": False, "training_eligible": False, "physical_deployment_authorized": False,
              "claim_limit": packet["claim_limit"]}
    atomic_json(OUTPUT / "RESULT.json", result)
    atomic_json(TERMINAL_RECEIPT, {**result, "result": ref(OUTPUT / "RESULT.json")})
    atomic_json(OUTPUT / "RUN_RECEIPT.json", {"schema_version": "0915-robot15h-sam31-task-object-run-receipt-v1",
                "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
                "result": ref(OUTPUT / "RESULT.json"), "terminal_receipt": ref(TERMINAL_RECEIPT)})


def run_orchestrator(args: argparse.Namespace) -> int:
    packet, packet_path = validate_route()
    if args.output_root.resolve() != OUTPUT.resolve() or args.visual_root.resolve() != VISUAL.resolve() or args.receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("task-object namespace drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (OUTPUT, VISUAL, TERMINAL_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh task-object output required: {path}")
    if not HAND_RESULT.is_file() or load_json(HAND_RESULT).get("status") not in {"PASSED", "REJECTED_QUALITY"}:
        raise RuntimeError("terminal hand-quality predecessor is unavailable")
    signature = build_signature(packet_path, args.executor_epoch)
    OUTPUT.mkdir(parents=True)
    atomic_json_new(OUTPUT / "RUN_SIGNATURE.json", signature)
    claim = {"schema_version": "0915-robot15h-sam31-task-object-writer-claim-v1", "task_id": TASK_ID,
             "window_run_id": WINDOW_RUN_ID, "status": "CLAIMED", "weights": packet["weights"],
             "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"), "pid": os.getpid(),
             "proc_start_ticks": common.process_start_ticks(os.getpid()), "executor_epoch": args.executor_epoch,
             "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
             "run_signature_sha256": signature["run_signature_sha256"], "unique_write_root": str(OUTPUT.resolve())}
    atomic_json_new(OUTPUT / "CLAIM.json", claim)
    heartbeat("WAIT_GPU_RESOURCE")
    worker = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output-root", str(OUTPUT),
              "--visual-root", str(VISUAL), "--executor-epoch", str(args.executor_epoch),
              "--task-id", TASK_ID,
              "--claim", str(OUTPUT / "CLAIM.json"), "--run-signature", str(OUTPUT / "RUN_SIGNATURE.json")]
    lease = [sys.executable, str(GPU_LEASE_WRAPPER), "--task-id", TASK_ID, "--attempt-id", OUTPUT.name,
             "--executor-epoch", str(args.executor_epoch), "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
             "--min-free-mib", str(args.min_free_mib), "--wait-seconds", str(args.gpu_wait_seconds),
             "--wall-seconds", str(args.wall_seconds), "--receipt", str(OUTPUT / "GPU_COMMAND_RECEIPT.json"),
             "--claim-limit", packet["claim_limit"], "--", *worker]
    atomic_json(OUTPUT / "COMMAND.json", {"worker_command": worker, "lease_command": lease})
    with (OUTPUT / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is not None:
                break
            lease_state = load_json(ROOT / "_run/current/GPU_LEASE.json") if (ROOT / "_run/current/GPU_LEASE.json").is_file() else {}
            acquired = lease_state.get("status") == "ACQUIRED" and lease_state.get("task_id") == TASK_ID
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE", args.gpu_id if acquired else None)
    gpu = load_json(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(packet, status, str(gpu.get("reason") or gpu.get("error") or "SAM31_TASK_OBJECT_RUNTIME_FAILED"))
        return 3 if status == "BLOCKED_RESOURCE" else 2
    batch = load_json(OUTPUT / "OBJECT_IDENTITY_BATCH.json")
    if batch["counts"]["sessions_consumer_allowed"]:
        write_terminal(packet, "PASSED", None)
    else:
        write_terminal(packet, "REJECTED_QUALITY", "NO_STABLE_SEPARATE_VISIBLE_TASK_OBJECT_INSTANCE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=(PRIMARY_TASK_ID, RECOVERY_TASK_ID), default=PRIMARY_TASK_ID)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=TERMINAL_RECEIPT)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--run-signature", type=Path)
    args = parser.parse_args()
    configure_task(args.task_id)
    if args.worker:
        if args.claim is None or args.run_signature is None:
            raise RuntimeError("worker requires claim and run signature")
        return run_worker(args.claim.resolve(strict=True), args.run_signature.resolve(strict=True), args.executor_epoch)
    if args.fencing_token is None:
        raise RuntimeError("orchestrator requires fencing token")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
