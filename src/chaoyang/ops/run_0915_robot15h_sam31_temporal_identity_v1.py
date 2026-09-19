#!/usr/bin/env python3
"""Run pinned SAM3.1 hand temporal identity over the four frozen 0915 W0 sessions.

This node deliberately does less than a generic semantic-mask stage.  It only
creates side-locked hand tracks when a direct-observed HaWoR frame supplies an
independent geometric anchor.  Raw SAM candidates are preserved before quality
admission.  Roles without such an anchor are terminalized as unknown and are
not silently manufactured from text proposals.
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

from chaoyang.governance.robot15h_task_specs_v1 import (
    SAM31_WEIGHT,
    WINDOW_RUN_ID,
    build_packet,
)
from chaoyang.ops import run_0915_leftmono_sam31_masks_v1 as legacy
from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as common
from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module


ROOT = Path(__file__).resolve().parents[3]
PRIMARY_TASK_ID = "0915_robot15h_sam31_temporal_identity_v1"
RECOVERY_TASK_ID = "0915_robot15h_sam31_temporal_identity_recovery_v1"
QUALITY_TASK_ID = "0915_robot15h_sam31_temporal_identity_quality_v2"
TASK_ID = PRIMARY_TASK_ID
PHASE = "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
HAWOR_ROOT = ROOT / "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001"
HAWOR_RESULT = HAWOR_ROOT / "RESULT.json"
PREPARED_MANIFEST = HAWOR_ROOT / "PREPARED_MANIFEST.json"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_V1_RESULT.json"
CHECKPOINT = ROOT / SAM31_WEIGHT
CHECKPOINT_SHA256 = adapter_module.CHECKPOINT_SHA256
CODE_ROOT = ROOT / "vendor/SAM3"
if str(CODE_ROOT) not in sys.path:
    # The pinned official tree is repository-local.  This is a deterministic
    # import-path binding, not an installation or dependency mutation.
    sys.path.insert(0, str(CODE_ROOT))
GPU_LEASE_WRAPPER = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
WIDTH = 1280
HEIGHT = 960
PALM_INDICES = (0, 1, 5, 9, 13, 17)
ROLE_BLOCKERS = (
    "left_forearm", "right_forearm", "left_finger_sleeve", "right_finger_sleeve",
    "left_cable", "right_cable", "task_object",
)
COLORS = {"left_hand": (255, 190, 30), "right_hand": (30, 100, 255)}


def configure_task(task_id: str) -> None:
    """Select the original, runtime-recovery, or quality-v2 namespace."""

    global TASK_ID, PHASE, OUTPUT, VISUAL, TERMINAL_RECEIPT
    if task_id == PRIMARY_TASK_ID:
        TASK_ID = PRIMARY_TASK_ID
        PHASE = "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0"
        OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
        VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_V1"
        TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_V1_RESULT.json"
    elif task_id == RECOVERY_TASK_ID:
        TASK_ID = RECOVERY_TASK_ID
        PHASE = "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0_RUNTIME_RECOVERY"
        OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
        VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1"
        TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_RECOVERY_V1_RESULT.json"
    elif task_id == QUALITY_TASK_ID:
        TASK_ID = QUALITY_TASK_ID
        PHASE = "ROBOT15H_SAM31_HAND_TEMPORAL_IDENTITY_WAVE0_QUALITY_V2"
        OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
        VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_SAM31_TEMPORAL_IDENTITY_QUALITY_V2"
        TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_SAM31_TEMPORAL_IDENTITY_QUALITY_V2_RESULT.json"
    else:
        raise RuntimeError(f"unsupported SAM3.1 task namespace: {task_id}")


class SeedQualityError(RuntimeError):
    """The model ran but no raw candidate passed the seed-identity gate."""

    def __init__(self, message: str, candidate_rows: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.candidate_rows = [dict(row) for row in candidate_rows]


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
    """Hash ``current`` while publishing its post-rename path.

    Session trees are first assembled under an atomic staging directory.  A
    normal ``file_ref`` would permanently serialize that transient path.  This
    helper deliberately binds bytes from staging to the final target path.
    """

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


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != [SAM31_WEIGHT]:
        raise RuntimeError("current SAM3.1 W0 packet differs from frozen model spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("SAM3.1 temporal identity is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("SAM3.1 temporal identity is not routable")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("SAM3.1 temporal identity packet SHA drift")
    return packet, packet_path


def heartbeat(status: str, gpu_id: int | None = None) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()), "--status", status,
        "--phase", PHASE,
    ]
    if gpu_id is not None:
        command += ["--gpu-id", str(gpu_id)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def w0_rows() -> list[dict[str, Any]]:
    inventory = load_json(INVENTORY)
    rows = [row for row in inventory.get("sessions", []) if row.get("wave") == "W0"]
    if len(rows) != 4 or sum(int(row["frame_count"]) for row in rows) != 1058:
        raise RuntimeError("frozen SAM3.1 W0 cohort drift")
    prepared = {row["session_id"]: row for row in load_json(PREPARED_MANIFEST)["results"]}
    result: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row["session_id"])
        item = prepared.get(session_id)
        if item is None or int(item["frame_count"]) != int(row["frame_count"]):
            raise RuntimeError(f"prepared resize-only input drift: {session_id}")
        if item["input_domain"] != {
            "lens_undistortion": False,
            "operation": "CROP_THEN_RESIZE_ONLY",
            "output_size": [WIDTH, HEIGHT],
            "physical_left_source_index": 1,
            "physical_right_source_index": 0,
            "pico26_hand_consumed": False,
            "remap_applied": False,
            "trackingData_hand_consumed": False,
        }:
            raise RuntimeError(f"prepared input-domain drift: {session_id}")
        task = str(row["task"])
        hawor = HAWOR_ROOT / "hawor/sessions" / task / session_id / "HAWOR_RAW_MANO21.npz"
        video = Path(str(item["prepared_video"]["path"])).resolve(strict=True)
        if item["prepared_video"]["sha256"] != sha256(video) or not hawor.is_file():
            raise RuntimeError(f"prepared/HаWoR SHA closure drift: {session_id}")
        result.append({**row, "prepared_video": str(video), "hawor_npz": str(hawor.resolve())})
    return result


def _video_gate(path: Path, expected: int) -> dict[str, Any]:
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
    expected_result = {"width": WIDTH, "height": HEIGHT, "frames": expected, "fps": "30/1", "full_decode": True}
    if result != expected_result:
        raise RuntimeError(f"resize-only video identity drift: {result} != {expected_result}")
    return result


def _extract_frames(video: Path, destination: Path, expected: int) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vsync", "0",
        "-start_number", "0", str(destination / "%06d.jpg"),
    ], capture_output=True, text=True, check=False)
    if completed.returncode or len(list(destination.glob("*.jpg"))) != expected:
        raise RuntimeError("SAM3.1 frame extraction did not preserve the session denominator")


def _bbox_from_joints(points: np.ndarray, *, width: int = WIDTH, height: int = HEIGHT) -> list[float] | None:
    finite = np.isfinite(points).all(axis=1)
    valid = points[finite]
    if len(valid) < 5:
        return None
    x0, y0 = np.min(valid, axis=0)
    x1, y1 = np.max(valid, axis=0)
    if x1 <= x0 or y1 <= y0:
        return None
    pad_x = max(8.0, 0.15 * float(x1 - x0))
    pad_y = max(8.0, 0.15 * float(y1 - y0))
    x0, y0 = max(0.0, float(x0 - pad_x)), max(0.0, float(y0 - pad_y))
    x1, y1 = min(float(width - 1), float(x1 + pad_x)), min(float(height - 1), float(y1 + pad_y))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def select_anchor_candidates(
    joints: np.ndarray,
    observed: np.ndarray,
    detector_confidence: np.ndarray,
    side: int,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[dict[str, Any]]:
    """Rank only direct-observed, finite, in-frame side-specific HaWoR frames."""

    rows: list[dict[str, Any]] = []
    for frame in np.flatnonzero(observed[side]):
        points = np.asarray(joints[side, frame], np.float64)
        finite = np.isfinite(points).all(axis=1)
        in_frame = finite & (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        count = int(in_frame.sum())
        palm_count = int(in_frame[list(PALM_INDICES)].sum())
        box = _bbox_from_joints(points[in_frame], width=width, height=height)
        if box is None or count < 5 or palm_count < 2:
            continue
        x, y, w, h = box
        margin = min(x, y, width - (x + w), height - (y + h)) / max(width, height)
        area_fraction = (w * h) / float(width * height)
        area_score = -abs(math.log(max(area_fraction, 1e-8) / 0.035))
        confidence = float(detector_confidence[side, frame])
        if not np.isfinite(confidence):
            confidence = 0.0
        rows.append({
            "frame_index": int(frame), "box_xywh": [float(value) for value in box],
            "in_frame_joint_count": count, "in_frame_palm_count": palm_count,
            "boundary_margin_fraction": float(margin), "box_area_fraction": float(area_fraction),
            "detector_confidence": confidence,
            "score_tuple": [count, palm_count, float(margin), float(area_score), confidence, -int(frame)],
        })
    return sorted(rows, key=lambda row: tuple(row["score_tuple"]), reverse=True)


def choose_primary_fallback(candidates: Sequence[Mapping[str, Any]], frame_count: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not candidates:
        return None, None
    primary = dict(candidates[0])
    minimum_separation = max(15, int(math.ceil(0.1 * frame_count)))
    fallback = next((dict(row) for row in candidates[1:]
                     if abs(int(row["frame_index"]) - int(primary["frame_index"])) >= minimum_separation), None)
    return primary, fallback


def _near_mask(mask: np.ndarray, point: np.ndarray, radius: int) -> bool:
    if not np.isfinite(point).all():
        return False
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    return x0 < x1 and y0 < y1 and bool(mask[y0:y1, x0:x1].any())


def choose_seed_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
    anchor: Mapping[str, Any],
    own_points: np.ndarray,
    opposite_points: np.ndarray | None,
) -> tuple[int, list[dict[str, Any]]]:
    x, y, w, h = [float(value) for value in anchor["box_xywh"]]
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1, y1 = min(WIDTH, int(math.ceil(x + w))), min(HEIGHT, int(math.ceil(y + h)))
    box_area = max(1, (x1 - x0) * (y1 - y0))
    # A fixed 1,200-pixel floor rejected valid distant hands even though their
    # candidates agreed with the HaWoR box and joints.  Keep a hard noise floor
    # and the former 1,200-pixel ceiling, but scale the gate with the independent
    # geometric anchor.  The overlap, own-side coverage, opposite-side and area
    # upper gates below remain unchanged.
    minimum_area = min(1_200, max(128, int(math.ceil(0.08 * box_area))))
    own = own_points[np.isfinite(own_points).all(axis=1)]
    opposite = (
        opposite_points[list(PALM_INDICES)]
        if opposite_points is not None else np.empty((0, 2), np.float64)
    )
    opposite = opposite[np.isfinite(opposite).all(axis=1)]
    rows: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        area = int(mask.sum())
        intersection = int(mask[y0:y1, x0:x1].sum())
        own_covered = int(sum(_near_mask(mask, point, 18) for point in own))
        opposite_covered = int(sum(_near_mask(mask, point, 8) for point in opposite))
        own_fraction = own_covered / len(own) if len(own) else 0.0
        eligible = bool(
            minimum_area <= area <= int(mask.size * 0.30)
            and intersection / box_area >= 0.15
            and own_fraction >= 0.20
            and opposite_covered == 0
        )
        rows.append({
            "raw_id": int(ids[index]), "score": float(scores[index]), "area_pixels": area,
            "box_intersection_over_box": intersection / box_area,
            "own_joint_coverage": own_fraction,
            "opposite_palm_points_covered": opposite_covered,
            "minimum_area_pixels": minimum_area,
            "minimum_area_policy": "CLAMP_128_1200_OF_0_08_ANCHOR_BOX_AREA",
            "eligible": eligible,
        })
    eligible = sorted((row for row in rows if row["eligible"]), key=lambda row: (
        -row["own_joint_coverage"], -row["box_intersection_over_box"], -row["score"], row["raw_id"],
    ))
    if not eligible:
        raise SeedQualityError(f"no side-safe geometric-box candidate: {rows}", rows)
    return int(eligible[0]["raw_id"]), rows


def _collect_bidirectional(
    model: Any, state: dict[str, Any], *, anchor: int, raw_id: int,
    frame_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros((frame_count, HEIGHT, WIDTH), bool)
    directions: list[dict[str, Any]] = []
    routes = [(False, frame_count - anchor)]
    if anchor:
        routes.append((True, anchor + 1))
    for reverse, maximum in routes:
        yielded = 0
        try:
            for frame_index, outputs in model.propagate_in_video(
                inference_state=state, start_frame_idx=anchor,
                max_frame_num_to_track=maximum, reverse=reverse, output_prob_thresh=0.5,
            ):
                yielded += 1
                masks, _scores, ids = legacy.normalize(outputs, HEIGHT, WIDTH)
                matches = np.flatnonzero(ids == raw_id)
                if len(matches) == 1 and 0 <= int(frame_index) < frame_count:
                    result[int(frame_index)] = masks[int(matches[0])]
            directions.append({"direction": "backward" if reverse else "forward", "status": "COMPLETE", "frames_yielded": yielded})
        except RuntimeError as error:
            if str(error) != "No points are provided; please add points first":
                raise
            directions.append({
                "direction": "backward" if reverse else "forward",
                "status": "UNKNOWN_DIRECTION_TRACKER_HAS_NO_CONFIRMED_INSTANCE",
                "frames_yielded_before_hold": yielded, "reason": str(error),
            })
    return result, directions


def run_seed(
    model: Any,
    frames: Path,
    anchor: Mapping[str, Any],
    side: int,
    joints: np.ndarray,
    observed: np.ndarray,
    frame_count: int,
    *,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    frame = int(anchor["frame_index"])
    box = [float(value) for value in anchor["box_xywh"]]
    normalised_box = [[box[0] / WIDTH, box[1] / HEIGHT, box[2] / WIDTH, box[3] / HEIGHT]]
    state = model.init_state(resource_path=str(frames), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        _, initial = model.add_prompt(
            inference_state=state, frame_idx=frame, text_str="hand",
            boxes_xywh=normalised_box, box_labels=[1], clear_old_boxes=True,
            output_prob_thresh=0.5,
        )
        masks, scores, ids = legacy.normalize(initial, HEIGHT, WIDTH)
        initial_candidates = {
            "masks": np.asarray(masks, bool),
            "scores": np.asarray(scores, np.float32),
            "raw_ids": np.asarray(ids, np.int64),
            "frame_index": frame,
        }
        opposite = 1 - side
        opposite_points = joints[opposite, frame] if observed[opposite, frame] else None
        try:
            raw_id, candidates = choose_seed_candidate(
                masks, scores, ids, anchor, joints[side, frame], opposite_points,
            )
        except SeedQualityError as error:
            return np.zeros((frame_count, HEIGHT, WIDTH), bool), {
                "status": "REJECTED_SEED_QUALITY", "seed_label": label,
                "frame_index": frame, "box_xywh": box,
                "error": f"{type(error).__name__}:{error}",
                "candidate_rows": error.candidate_rows,
                "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
                "persistent_identity_selected": False,
                "initial_raw_candidates_are_semantic_truth": False,
            }, initial_candidates
        index = int(np.flatnonzero(ids == raw_id)[0])
        route, action_ids = model.parse_action_history_for_propagation(state)
        if route != "propagation_full" or action_ids is not None:
            raise RuntimeError("pinned multiplex geometric box did not retain full propagation route")
        tracked, directions = _collect_bidirectional(
            model, state, anchor=frame, raw_id=raw_id, frame_count=frame_count,
        )
        if not tracked[frame].any():
            tracked[frame] = masks[index]
        return tracked, {
            "status": "COMPLETE", "seed_label": label, "frame_index": frame,
            "box_xywh": box, "text": "hand", "selected_raw_track_id": raw_id,
            "candidate_rows": candidates, "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
            "propagation_route_before": route, "action_object_ids_before": action_ids,
            "direction_evidence": directions, "point_refined": False,
            "points_usage": "QUALITY_EVIDENCE_ONLY_NOT_SENT_TO_MULTIPLEX_RUNTIME",
            "persistent_identity_selected": True,
            "initial_raw_candidates_are_semantic_truth": False,
        }, initial_candidates
    finally:
        state.clear()


def evaluate_stream(
    masks: np.ndarray,
    side: int,
    joints: np.ndarray,
    observed: np.ndarray,
    anchor_frame: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame_count = len(masks)
    areas = masks.reshape(frame_count, -1).sum(axis=1).astype(np.int64)
    anchor_area = int(areas[anchor_frame])
    minimum_area = min(1_200, max(128, int(math.ceil(0.12 * anchor_area))))
    valid = np.ones(frame_count, bool)
    rows: list[dict[str, Any]] = []
    previous_centroid: np.ndarray | None = None
    previous_area: int | None = None
    for frame, mask in enumerate(masks):
        reasons: list[str] = []
        area = int(areas[frame])
        if area < minimum_area:
            reasons.append("AREA_BELOW_SCALE_AWARE_MINIMUM")
        if area > int(mask.size * 0.30):
            reasons.append("AREA_ABOVE_MAXIMUM")
        if anchor_area <= 0:
            reasons.append("ANCHOR_EMPTY")
        elif not 0.12 <= area / anchor_area <= 6.0:
            reasons.append("AREA_RATIO_OUT_OF_RANGE")
        components = int(cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1) if area else 0
        if components > 10:
            reasons.append("TOO_MANY_COMPONENTS")
        yy, xx = np.where(mask)
        centroid = np.asarray([xx.mean(), yy.mean()], np.float64) if len(xx) else None
        jump_normalised = None
        if centroid is not None and previous_centroid is not None and previous_area:
            jump_normalised = float(np.linalg.norm(centroid - previous_centroid) / max(math.sqrt(previous_area), 1.0))
            if jump_normalised > 2.5:
                reasons.append("CENTROID_JUMP_SCALE_NORMALISED")
        if centroid is not None:
            previous_centroid, previous_area = centroid, area
        own_coverage = None
        opposite_coverage = None
        identity_conflict = False
        if observed[side, frame]:
            points = joints[side, frame]
            points = points[np.isfinite(points).all(axis=1)]
            own_coverage = sum(_near_mask(mask, point, 20) for point in points) / len(points) if len(points) else 0.0
            if len(points) >= 5 and own_coverage < 0.20:
                reasons.append("HAWOR_OWN_SIDE_COVERAGE_LOW")
        opposite = 1 - side
        if observed[opposite, frame]:
            points = joints[opposite, frame, list(PALM_INDICES)]
            points = points[np.isfinite(points).all(axis=1)]
            opposite_coverage = sum(_near_mask(mask, point, 8) for point in points) / len(points) if len(points) else 0.0
            if opposite_coverage > 0.0:
                identity_conflict = True
                reasons.append("OPPOSITE_PALM_IDENTITY_CONFLICT")
        valid[frame] = not reasons
        rows.append({
            "frame_id": frame, "raw_present": bool(area), "semantic_admitted": bool(valid[frame]),
            "reasons": reasons, "area_pixels": area, "component_count": components,
            "minimum_area_pixels": minimum_area,
            "centroid_xy": centroid.tolist() if centroid is not None else None,
            "centroid_jump_scale_normalised": jump_normalised,
            "own_direct_hawor_coverage": own_coverage,
            "opposite_palm_coverage": opposite_coverage,
            "identity_conflict": identity_conflict,
        })
    return valid, rows


def assess_hand_temporal_proxy(
    *, direct_count: int, direct_fraction: float, unknown_fraction: float,
    identity_conflicts: int,
) -> tuple[bool, str, list[str]]:
    """Admit hand evidence from direct observations, not inferred absence.

    Whole-video unknown coverage remains a continuity diagnostic.  In
    particular it cannot turn true absence outside direct HaWoR observations
    into a hand-identity quality failure.  Side-identity conflicts remain a
    strict rejection.
    """

    diagnostics: list[str] = []
    if unknown_fraction > 0.35:
        diagnostics.append("WHOLE_VIDEO_UNKNOWN_FRACTION_ABOVE_0_35")
    if direct_count < 10:
        return False, "REJECTED_INSUFFICIENT_DIRECT_QA", diagnostics
    if direct_fraction < 0.65:
        return False, "REJECTED_DIRECT_OBSERVED_ADMISSION", diagnostics
    if identity_conflicts:
        return False, "REJECTED_IDENTITY_CONFLICT", diagnostics
    return True, "PASS_HAND_DIRECT_OBSERVED_PROXY", diagnostics


def should_reseed(valid: np.ndarray, has_fallback: bool) -> tuple[bool, list[str]]:
    if not has_fallback:
        return False, ["NO_SEPARATED_DIRECT_FALLBACK_ANCHOR"]
    reasons: list[str] = []
    longest = 0
    current = 0
    for item in valid:
        current = 0 if item else current + 1
        longest = max(longest, current)
    unknown_fraction = float(np.mean(~valid))
    if longest >= 3:
        reasons.append("THREE_OR_MORE_CONSECUTIVE_UNKNOWN")
    if unknown_fraction >= 0.08:
        reasons.append("UNKNOWN_FRACTION_AT_LEAST_0_08")
    return bool(reasons), reasons


def merge_tracks(
    primary: np.ndarray,
    primary_valid: np.ndarray,
    fallback: np.ndarray | None,
    fallback_valid: np.ndarray | None,
    primary_anchor: int,
    fallback_anchor: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame_count = len(primary)
    raw_display = np.zeros_like(primary)
    semantic = np.zeros_like(primary)
    ledger: list[dict[str, Any]] = []
    for frame in range(frame_count):
        fallback_present = fallback is not None and bool(fallback[frame].any())
        raw_source = "primary" if primary[frame].any() else ("fallback" if fallback_present else None)
        if raw_source == "primary":
            raw_display[frame] = primary[frame]
        elif raw_source == "fallback" and fallback is not None:
            raw_display[frame] = fallback[frame]
        if primary_valid[frame]:
            semantic[frame] = primary[frame]
            source = "primary"
            state = "seeded" if frame == primary_anchor else "tracked"
        elif fallback is not None and fallback_valid is not None and fallback_valid[frame]:
            semantic[frame] = fallback[frame]
            source = "fallback"
            state = "reseeded" if frame == fallback_anchor else "tracked"
        else:
            source = None
            state = "unknown"
        ledger.append({
            "frame_id": frame, "raw_present": bool(raw_display[frame].any()),
            "raw_source_seed": raw_source, "semantic_admitted": bool(semantic[frame].any()),
            "semantic_source_seed": source, "tracking_state": state,
            "visibility_state": "VISIBLE_CANDIDATE" if raw_display[frame].any() else "UNKNOWN",
        })
    return raw_display, semantic, ledger


def save_packed(path: Path, masks: np.ndarray, *, published_path: Path | None = None) -> dict[str, Any]:
    frame_count, height, width = masks.shape
    packed = np.packbits(masks.reshape(frame_count, -1), axis=1, bitorder="big")
    atomic_npz(path, packed=packed, frame_count=np.int32(frame_count), height=np.int32(height),
               width=np.int32(width), bitorder=np.asarray("big"))
    return published_ref(path, published_path) if published_path is not None else ref(path)


def save_initial_seed_candidates(
    path: Path, payload: Mapping[str, Any], *, published_path: Path | None = None,
) -> dict[str, Any]:
    """Preserve all pre-admission seed candidates without granting identity."""

    masks = np.asarray(payload["masks"], bool)
    candidate_count, height, width = masks.shape
    packed = np.packbits(masks.reshape(candidate_count, -1), axis=1, bitorder="big")
    atomic_npz(
        path, packed=packed, candidate_count=np.int32(candidate_count),
        height=np.int32(height), width=np.int32(width), bitorder=np.asarray("big"),
        raw_ids=np.asarray(payload["raw_ids"], np.int64),
        scores=np.asarray(payload["scores"], np.float32),
        frame_index=np.int32(payload["frame_index"]),
        semantic_admitted=np.asarray(False), persistent_identity_selected=np.asarray(False),
    )
    return published_ref(path, published_path) if published_path is not None else ref(path)


def load_packed(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        packed = archive["packed"]
        n, height, width = int(archive["frame_count"]), int(archive["height"]), int(archive["width"])
    return np.unpackbits(packed, axis=1, count=height * width, bitorder="big").reshape(n, height, width).astype(bool)


def process_side(
    model: Any,
    frames: Path,
    destination: Path,
    published_destination: Path,
    side: int,
    joints: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    role = "left_hand" if side == 0 else "right_hand"
    frame_count = joints.shape[1]
    candidates = select_anchor_candidates(joints, observed, confidence, side)
    primary_anchor, fallback_anchor = choose_primary_fallback(candidates, frame_count)
    if primary_anchor is None:
        return {
            "role": role, "side": side, "status": "BLOCKED_NO_DIRECT_HAWOR_ANCHOR",
            "direct_observed_frames": int(observed[side].sum()), "anchor_candidates": candidates,
            "consumer_allowed": False,
        }
    primary, primary_evidence, primary_initial = run_seed(
        model, frames, primary_anchor, side, joints, observed, frame_count, label="primary",
    )
    primary_initial_path = destination / "raw_candidate" / f"{role}_primary_seed_candidates.npz"
    primary_evidence["initial_raw_candidate_archive"] = save_initial_seed_candidates(
        primary_initial_path, primary_initial,
        published_path=published_destination / primary_initial_path.relative_to(destination),
    )
    primary_valid, primary_rows = evaluate_stream(
        primary, side, joints, observed, int(primary_anchor["frame_index"]),
    )
    trigger, trigger_reasons = should_reseed(primary_valid, fallback_anchor is not None)
    fallback: np.ndarray | None = None
    fallback_valid: np.ndarray | None = None
    fallback_rows: list[dict[str, Any]] = []
    fallback_evidence: dict[str, Any] | None = None
    if trigger and fallback_anchor is not None:
        fallback, fallback_evidence, fallback_initial = run_seed(
            model, frames, fallback_anchor, side, joints, observed, frame_count, label="fallback",
        )
        fallback_initial_path = destination / "raw_candidate" / f"{role}_fallback_seed_candidates.npz"
        fallback_evidence["initial_raw_candidate_archive"] = save_initial_seed_candidates(
            fallback_initial_path, fallback_initial,
            published_path=published_destination / fallback_initial_path.relative_to(destination),
        )
        fallback_valid, fallback_rows = evaluate_stream(
            fallback, side, joints, observed, int(fallback_anchor["frame_index"]),
        )
    raw_display, semantic, ledger = merge_tracks(
        primary, primary_valid, fallback, fallback_valid,
        int(primary_anchor["frame_index"]),
        int(fallback_anchor["frame_index"]) if fallback_anchor is not None else None,
    )
    raw_primary_path = destination / "raw_candidate" / f"{role}_primary.npz"
    raw_display_path = destination / "raw_candidate" / f"{role}_display.npz"
    semantic_path = destination / "semantic" / f"{role}.npz"
    raw_refs = {
        "primary": save_packed(
            raw_primary_path, primary,
            published_path=published_destination / raw_primary_path.relative_to(destination),
        ),
        "display": save_packed(
            raw_display_path, raw_display,
            published_path=published_destination / raw_display_path.relative_to(destination),
        ),
    }
    if fallback is not None:
        fallback_path = destination / "raw_candidate" / f"{role}_fallback.npz"
        raw_refs["fallback"] = save_packed(
            fallback_path, fallback,
            published_path=published_destination / fallback_path.relative_to(destination),
        )
    semantic_ref = save_packed(
        semantic_path, semantic,
        published_path=published_destination / semantic_path.relative_to(destination),
    )
    observed_ids = np.flatnonzero(observed[side])
    direct_admitted = int(sum(bool(semantic[frame].any()) for frame in observed_ids))
    direct_count = int(len(observed_ids))
    direct_fraction = direct_admitted / direct_count if direct_count else 0.0
    identity_conflicts = sum(bool(row["identity_conflict"]) for row in primary_rows)
    if fallback_rows:
        identity_conflicts += sum(bool(row["identity_conflict"]) for row in fallback_rows)
    unknown = sum(row["tracking_state"] == "unknown" for row in ledger)
    proxy_pass, status, continuity_diagnostics = assess_hand_temporal_proxy(
        direct_count=direct_count, direct_fraction=direct_fraction,
        unknown_fraction=unknown / frame_count, identity_conflicts=identity_conflicts,
    )
    quality = {
        "role": role, "status": status, "direct_observed_frames": direct_count,
        "direct_observed_semantic_admitted_frames": direct_admitted,
        "direct_observed_semantic_admission_fraction": direct_fraction,
        "raw_output_frames": sum(row["raw_present"] for row in ledger),
        "semantic_admitted_frames": frame_count - unknown, "unknown_frames": unknown,
        "unknown_fraction": unknown / frame_count, "identity_conflicts": identity_conflicts,
        "direct_observed_admission_is_authoritative_gate": True,
        "whole_video_unknown_is_diagnostic_only": True,
        "continuity_diagnostics": continuity_diagnostics,
        "reseed_triggered": bool(trigger and fallback is not None),
        "reseed_trigger_reasons": trigger_reasons, "maximum_reseed_count": 1,
        "authority": "DEVELOPMENT_PROXY_NO_MANUAL_GROUND_TRUTH",
    }
    state_path = destination / "temporal" / f"{role}_STATE_LEDGER.json"
    quality_path = destination / "quality" / f"{role}_QUALITY_LEDGER.json"
    atomic_json(state_path, {
        "schema_version": "sam31-hand-temporal-state-ledger-v1", "role": role,
        "physical_role_id": role, "raw_track_id_scope": "SESSION_STATE_ONLY",
        "empty_semantic_mask": "UNKNOWN_NOT_ABSENT", "frames": ledger,
    })
    atomic_json(quality_path, {
        "schema_version": "sam31-hand-quality-trigger-ledger-v1", "role": role,
        "anchor_candidates": candidates, "primary_anchor": primary_anchor,
        "fallback_anchor": fallback_anchor, "primary_seed": primary_evidence,
        "fallback_seed": fallback_evidence, "primary_frame_metrics": primary_rows,
        "fallback_frame_metrics": fallback_rows, "summary": quality,
    })
    return {
        "role": role, "side": side, "status": status, "consumer_allowed": proxy_pass,
        "raw_candidate_archives": raw_refs, "semantic_archive": semantic_ref,
        "state_ledger": published_ref(
            state_path, published_destination / state_path.relative_to(destination),
        ),
        "quality_ledger": published_ref(
            quality_path, published_destination / quality_path.relative_to(destination),
        ),
        "quality": quality,
    }


def _review_decode(path: Path, expected: int) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    decode = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    result = {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]), "fps": stream["r_frame_rate"],
        "full_decode": decode.returncode == 0,
    }
    if result["frames"] != expected or not result["full_decode"]:
        raise RuntimeError(f"SAM review decode mismatch: {result}")
    return result


def _current_artifact_path(path_ref: Mapping[str, Any], current_root: Path, published_root: Path) -> Path:
    published = Path(str(path_ref["path"]))
    relative = published.relative_to(published_root.resolve())
    current = current_root / relative
    if not current.is_file() or current.stat().st_size != int(path_ref["bytes"]) or sha256(current) != path_ref["sha256"]:
        raise RuntimeError(f"staged artifact does not match its publication ref: {published}")
    return current


def render_review(
    video: Path, destination: Path, session_id: str,
    side_results: Sequence[Mapping[str, Any]], expected: int,
    *, current_root: Path, published_root: Path,
) -> dict[str, Any]:
    raw: dict[str, np.ndarray] = {}
    semantic: dict[str, np.ndarray] = {}
    state: dict[str, list[dict[str, Any]]] = {}
    for result in side_results:
        if "semantic_archive" not in result:
            continue
        role = str(result["role"])
        raw[role] = load_packed(_current_artifact_path(
            result["raw_candidate_archives"]["display"], current_root, published_root,
        ))
        semantic[role] = load_packed(_current_artifact_path(
            result["semantic_archive"], current_root, published_root,
        ))
        state[role] = load_json(_current_artifact_path(
            result["state_ledger"], current_root, published_root,
        ))["frames"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 420))
    if not writer.isOpened():
        raise RuntimeError("cannot open SAM review writer")
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            raw_overlay = image.copy()
            semantic_overlay = image.copy()
            state_text: list[str] = []
            for role in ("left_hand", "right_hand"):
                color = np.asarray(COLORS[role], np.float32)
                if role in raw and raw[role][frame].any():
                    mask = raw[role][frame]
                    raw_overlay[mask] = (0.50 * raw_overlay[mask] + 0.50 * color).astype(np.uint8)
                if role in semantic and semantic[role][frame].any():
                    mask = semantic[role][frame]
                    semantic_overlay[mask] = (0.45 * semantic_overlay[mask] + 0.55 * color).astype(np.uint8)
                state_text.append(f"{role[0].upper()}={state[role][frame]['tracking_state'] if role in state else 'blocked'}")
            panels = [cv2.resize(value, (426, 320), interpolation=cv2.INTER_AREA)
                      for value in (image, raw_overlay, semantic_overlay)]
            canvas = np.zeros((420, 1280, 3), np.uint8)
            canvas[76:396, :1278] = np.hstack(panels)
            cv2.putText(canvas, f"{session_id} | frame {frame:04d} | physical-left resize-only | no lens remap",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Raw RGB | raw SAM candidate (pre-gate) | admitted hand semantic mask",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
            cv2.putText(canvas, " | ".join(state_text) + " | empty semantic = UNKNOWN, not absence",
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200, 240, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            frame += 1
    finally:
        capture.release()
        writer.release()
    if frame != expected:
        raise RuntimeError(f"SAM review frame count drift: {frame}!={expected}")
    return {"video": ref(destination), **_review_decode(destination, expected)}


def process_session(model: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    session_id, task = str(row["session_id"]), str(row["task"])
    expected = int(row["frame_count"])
    video, hawor_path = Path(str(row["prepared_video"])), Path(str(row["hawor_npz"]))
    source_before = {"video": ref(video), "hawor": ref(hawor_path)}
    gate = _video_gate(video, expected)
    with np.load(hawor_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
        confidence = np.asarray(archive["detector_confidence"], np.float64)
        sides = np.asarray(archive["anatomical_side_names"]).astype(str).tolist()
    if joints.shape != (2, expected, 21, 2) or observed.shape != (2, expected) or sides != ["left", "right"]:
        raise RuntimeError("direct-observed HaWoR hand-anchor shape/order drift")
    target = OUTPUT / "sessions" / task / session_id
    staging = target.with_name(f".{session_id}.staging-{uuid.uuid4().hex}")
    if target.exists() or staging.exists():
        raise RuntimeError(f"fresh per-session SAM destination required: {target}")
    staging.mkdir(parents=True)
    frames = staging / "input_frames"
    _extract_frames(video, frames, expected)
    started = time.monotonic()
    side_results: list[dict[str, Any]] = []
    try:
        for side in (0, 1):
            side_results.append(process_side(
                model, frames, staging, target, side, joints, observed, confidence,
            ))
            gc.collect()
            torch.cuda.empty_cache()
        role_blockers = [{"role": role, "status": "BLOCKED_MISSING_ROLE_SPECIFIC_VISUAL_ANCHOR",
                          "semantic_state": "UNKNOWN", "persistent_instance_id_created": False}
                         for role in ROLE_BLOCKERS]
        role_blocker_path = staging / "ROLE_BLOCKER_LEDGER.json"
        role_manifest_path = staging / "ROLE_MANIFEST.json"
        atomic_json(role_blocker_path, {
            "schema_version": "sam31-role-blocker-ledger-v1", "session_id": session_id,
            "blocked_roles": role_blockers,
            "task_object_mask_consumer_allowed": False,
            "reason": "NO_MANUAL_OR_INDEPENDENTLY_ADMITTED_ROLE_SPECIFIC_VISUAL_BOX",
        })
        atomic_json(role_manifest_path, {
            "schema_version": "sam31-hand-temporal-identity-manifest-v2", "session_id": session_id,
            "frame_count": expected, "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
            "model": {"identity": "SAM3.1", "weight_sha256": CHECKPOINT_SHA256,
                      "runtime_class": "Sam3MultiplexTrackingWithInteractivity",
                      "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX"},
            "hand_instances": side_results, "blocked_roles": role_blockers,
            "raw_candidate_is_semantic_truth": False,
            "tracker_role_created": False, "controller_role_created": False,
            "pico26_consumed": False, "trackingData_hand_consumed": False,
        })
        VISUAL.mkdir(parents=True, exist_ok=True)
        review_path = VISUAL / f"{session_id}_SAM31_HAND_TEMPORAL_REVIEW.mp4"
        review = render_review(
            video, review_path, session_id, side_results, expected,
            current_root=staging, published_root=target,
        )
        passed_hands = sum(str(result["status"]).startswith("PASS_HAND_") for result in side_results)
        blocked_hands = sum(result["status"].startswith("BLOCKED") for result in side_results)
        session_status = "PARTIAL_HAND_ONLY_OBJECTS_BLOCKED" if passed_hands else "REJECTED_HAND_TEMPORAL_PROXY"
        result = {
            "schema_version": "0915-robot15h-sam31-temporal-session-result-v1",
            "session_id": session_id, "task": task, "source_group": row["source_group"],
            "status": session_status, "execution_completed": True, "frame_count": expected,
            "hand_temporal_artifact_exported": True,
            "hand_counts": {"total": 2, "proxy_pass": passed_hands,
                            "quality_rejected": 2 - passed_hands - blocked_hands, "blocked": blocked_hands},
            "object_semantic_admitted": 0, "object_mask_consumer_allowed": False,
            "video_gate": gate, "inputs": source_before, "side_results": side_results,
            "role_blocker_ledger": published_ref(role_blocker_path, target / role_blocker_path.relative_to(staging)),
            "role_manifest": published_ref(role_manifest_path, target / role_manifest_path.relative_to(staging)),
            "review": review, "wall_seconds": time.monotonic() - started,
            "source_mutated": False, "training_eligible": False,
            "physical_deployment_authorized": False,
        }
        atomic_json(staging / "RESULT.json", result)
        shutil.rmtree(frames)
        if {"video": ref(video), "hawor": ref(hawor_path)} != source_before:
            raise RuntimeError("SAM3.1 W0 input changed during execution")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        final = load_json(target / "RESULT.json")
        final["result"] = ref(target / "RESULT.json")
        return final
    except Exception:
        if staging.exists():
            # Preserve the staging tree as failure evidence rather than deleting it.
            failed = target.with_name(f"{session_id}.FAILED_RUNTIME_STAGING")
            if not failed.exists():
                os.replace(staging, failed)
        raise


def audit_publication_refs(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that no serialized artifact reference retained a staging path."""

    checked: list[dict[str, Any]] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            if {"path", "bytes", "sha256"}.issubset(value):
                path = Path(str(value["path"]))
                reasons: list[str] = []
                if ".staging-" in str(path) or ".FAILED_RUNTIME_STAGING" in str(path):
                    reasons.append("TRANSIENT_PATH_SERIALIZED")
                if not path.is_file():
                    reasons.append("MISSING")
                else:
                    if path.stat().st_size != int(value["bytes"]):
                        reasons.append("BYTE_COUNT_MISMATCH")
                    if sha256(path) != value["sha256"]:
                        reasons.append("SHA256_MISMATCH")
                checked.append({"location": location, "path": str(path), "reasons": reasons})
                return
            for key, item in value.items():
                visit(item, f"{location}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    visit(results, "sessions")
    # The quality ledgers carry the initial seed-candidate references while the
    # session result only references the ledger itself.  Audit every published
    # JSON sidecar as well, so nested refs cannot evade the post-rename check.
    for index, result in enumerate(results):
        result_ref = result.get("result")
        if not isinstance(result_ref, Mapping) or "path" not in result_ref:
            continue
        session_root = Path(str(result_ref["path"])).parent
        if not session_root.is_dir():
            continue
        for json_path in sorted(session_root.rglob("*.json")):
            visit(load_json(json_path), f"published_session_json[{index}].{json_path.relative_to(session_root)}")
    failures = [row for row in checked if row["reasons"]]
    audit = {
        "schema_version": "sam31-publication-ref-audit-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID,
        "status": "PASSED" if not failures else "FAILED",
        "refs_checked": len(checked), "refs_failed": len(failures),
        "transient_paths_serialized": sum(
            "TRANSIENT_PATH_SERIALIZED" in row["reasons"] for row in failures
        ),
        "failures": failures,
    }
    atomic_json(OUTPUT / "PUBLICATION_REF_AUDIT.json", audit)
    if failures:
        raise RuntimeError(f"SAM publication ref audit failed for {len(failures)} refs")
    return audit


def build_signature(packet_path: Path, executor_epoch: int) -> dict[str, Any]:
    sessions = []
    for row in w0_rows():
        sessions.append({
            "session_id": row["session_id"], "task": row["task"], "source_group": row["source_group"],
            "frame_count": int(row["frame_count"]), "prepared_video": ref(Path(row["prepared_video"])),
            "direct_hawor": ref(Path(row["hawor_npz"])),
        })
    payload = {
        "schema_version": "0915-robot15h-sam31-temporal-identity-run-signature-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "executor_epoch": executor_epoch,
        "weights": [SAM31_WEIGHT], "sessions": sessions,
        "contracts": {"task_packet": ref(packet_path), "hawor_result": ref(HAWOR_RESULT),
                      "prepared_manifest": ref(PREPARED_MANIFEST)},
        "runtime": {"runner": ref(Path(__file__)), "adapter": ref(Path(adapter_module.__file__)),
                    "checkpoint": ref(CHECKPOINT), "lease_wrapper": ref(GPU_LEASE_WRAPPER)},
        "input_policy": {"physical_left_source_index": 1, "operation": "CROP_THEN_RESIZE_ONLY",
                         "lens_undistortion": False, "direct_observed_hawor_anchor_only": True},
        "prompt_policy": {"runtime_semantics": "MULTIPLEX_GEOMETRIC_BOX",
                          "point_refinement": False, "text": "hand"},
        "quality_policy": {
            "seed_minimum_area": "CLAMP_128_1200_OF_0_08_ANCHOR_BOX_AREA",
            "temporal_minimum_area": "CLAMP_128_1200_OF_0_12_ANCHOR_MASK_AREA",
            "direct_observed_admission": True,
            "whole_video_unknown_is_diagnostic_only": True,
            "identity_conflict_is_hard_rejection": True,
            "initial_raw_seed_candidates_preserved": True,
        },
        "unanchored_role_policy": "FAIL_CLOSED_UNKNOWN_NO_PERSISTENT_ID",
    }
    return {**payload, "run_signature_sha256": canonical_sha(payload)}


def validate_claim(path: Path, signature_sha: str, executor_epoch: int, require_descendant: bool) -> None:
    claim = load_json(path)
    pid = claim.get("pid")
    if (
        claim.get("task_id") != TASK_ID or claim.get("weights") != [SAM31_WEIGHT]
        or claim.get("status") != "CLAIMED" or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("unique_write_root") != str(OUTPUT.resolve()) or not isinstance(pid, int)
        or common.process_start_ticks(pid) != claim.get("proc_start_ticks")
    ):
        raise RuntimeError("SAM3.1 W0 writer fence mismatch")
    if require_descendant and not common.process_has_ancestor(os.getpid(), pid):
        raise RuntimeError("SAM3.1 W0 GPU worker is outside writer ancestry")


def run_worker(claim_path: Path, signature_path: Path, executor_epoch: int) -> int:
    packet, _ = validate_route()
    signature = load_json(signature_path)
    stored = signature.pop("run_signature_sha256", None)
    if stored != canonical_sha(signature):
        raise RuntimeError("SAM3.1 W0 signature digest mismatch")
    signature["run_signature_sha256"] = stored
    validate_claim(claim_path, stored, executor_epoch, require_descendant=True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT,
    )
    results: list[dict[str, Any]] = []
    model_load_count = 1
    torch.cuda.reset_peak_memory_stats()
    try:
        for row in w0_rows():
            try:
                result = process_session(adapter.model, row)
            except Exception as error:
                target = OUTPUT / "sessions" / str(row["task"]) / str(row["session_id"])
                target.mkdir(parents=True, exist_ok=True)
                failure = {
                    "schema_version": "0915-robot15h-sam31-temporal-session-result-v1",
                    "session_id": row["session_id"], "task": row["task"],
                    "source_group": row["source_group"], "status": "FAILED_RUNTIME",
                    "execution_completed": False, "frame_count": int(row["frame_count"]),
                    "first_blocker": f"{type(error).__name__}:{error}",
                    "object_semantic_admitted": 0, "object_mask_consumer_allowed": False,
                    "source_mutated": False,
                }
                atomic_json(target / "RESULT.json", failure)
                result = {**failure, "result": ref(target / "RESULT.json")}
            results.append(result)
            print(json.dumps({"session_id": row["session_id"], "status": result["status"]}, ensure_ascii=False), flush=True)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        else:
            adapter.predictor.shutdown()
    publication_audit = audit_publication_refs(results)
    failed = sum(row["status"] == "FAILED_RUNTIME" for row in results)
    exported = sum(row.get("hand_temporal_artifact_exported") is True for row in results)
    side_rows = [side for row in results for side in row.get("side_results", [])]
    batch = {
        "schema_version": "0915-robot15h-sam31-temporal-identity-batch-v1",
        "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED_ALL_TERMINAL" if not failed else "COMPLETED_WITH_RUNTIME_FAILURE",
        "counts": {
            "sessions_total": 4, "sessions_hand_artifacts_exported": exported,
            "sessions_failed_runtime": failed, "hand_instances_attemptable": 7,
            "hand_instances_proxy_pass": sum(str(side.get("status", "")).startswith("PASS_HAND_") for side in side_rows),
            "hand_instances_quality_rejected": sum(str(side.get("status", "")).startswith("REJECTED") for side in side_rows),
            "hand_instances_blocked": sum(str(side.get("status", "")).startswith("BLOCKED") for side in side_rows),
            "object_semantic_admitted": 0,
        },
        "model_load_count": model_load_count, "fresh_session_state_count": len(side_rows),
        "model_build_evidence": build_evidence, "sessions": results,
        "object_mask_consumer_allowed": False, "pico26_consumed": False,
        "trackingData_hand_consumed": False, "source_mutated": False,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "quality_status": "PARTIAL_HAND_ONLY_OBJECTS_BLOCKED",
        "publication_ref_audit": ref(OUTPUT / "PUBLICATION_REF_AUDIT.json"),
        "publication_refs_valid": publication_audit["status"] == "PASSED",
    }
    atomic_json(OUTPUT / "BATCH_RESULT.json", batch)
    return 0 if failed == 0 else 2


def write_role_blocker_ledger() -> None:
    atomic_json(OUTPUT / "ROLE_BLOCKER_LEDGER.json", {
        "schema_version": "0915-robot15h-sam31-role-blocker-batch-v1",
        "window_run_id": WINDOW_RUN_ID, "sessions": [row["session_id"] for row in w0_rows()],
        "blocked_roles": list(ROLE_BLOCKERS),
        "status": "BLOCKED_MISSING_ROLE_SPECIFIC_VISUAL_ANCHOR",
        "semantic_state": "UNKNOWN_NOT_ABSENT", "persistent_instance_ids_created": False,
        "object_mask_consumer_allowed": False,
    })


def write_terminal(packet: Mapping[str, Any], status: str, blocker: str | None) -> None:
    batch = load_json(OUTPUT / "BATCH_RESULT.json") if (OUTPUT / "BATCH_RESULT.json").is_file() else None
    result = {
        "schema_version": "0915-robot15h-sam31-temporal-identity-result-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "first_blocker": blocker, "weights": packet["weights"],
        "execution_completed": bool(batch and batch.get("counts", {}).get("sessions_failed_runtime") == 0),
        "quality_status": "PARTIAL_HAND_ONLY_OBJECTS_BLOCKED" if batch else "NOT_PRODUCED",
        "counts": batch.get("counts") if batch else None,
        "object_semantic_admitted": 0, "object_mask_consumer_allowed": False,
        "raw_candidate_is_semantic_truth": False, "empty_semantic_mask": "UNKNOWN_NOT_ABSENT",
        "initial_box_semantics": "MULTIPLEX_GEOMETRIC_BOX",
        "batch_result": ref(OUTPUT / "BATCH_RESULT.json") if batch else None,
        "role_blocker_ledger": ref(OUTPUT / "ROLE_BLOCKER_LEDGER.json"),
        "publication_ref_audit": ref(OUTPUT / "PUBLICATION_REF_AUDIT.json") if (OUTPUT / "PUBLICATION_REF_AUDIT.json").is_file() else None,
        "gpu_command_receipt": ref(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "source_mutated": False, "training_eligible": False, "physical_deployment_authorized": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    atomic_json(TERMINAL_RECEIPT, {**result, "result": ref(OUTPUT / "RESULT.json")})
    atomic_json(OUTPUT / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-sam31-temporal-identity-run-receipt-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "result": ref(OUTPUT / "RESULT.json"), "terminal_receipt": ref(TERMINAL_RECEIPT),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet, packet_path = validate_route()
    if args.output_root.resolve() != OUTPUT.resolve() or args.visual_root.resolve() != VISUAL.resolve() or args.receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("SAM3.1 W0 namespace drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (OUTPUT, VISUAL, TERMINAL_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh SAM3.1 W0 output required: {path}")
    if not HAWOR_RESULT.is_file() or load_json(HAWOR_RESULT).get("status") != "PASSED":
        raise RuntimeError("frozen HaWoR recovery result is unavailable")
    signature = build_signature(packet_path, args.executor_epoch)
    OUTPUT.mkdir(parents=True)
    atomic_json_new(OUTPUT / "RUN_SIGNATURE.json", signature)
    write_role_blocker_ledger()
    claim = {
        "schema_version": "0915-robot15h-sam31-temporal-identity-writer-claim-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "CLAIMED",
        "weights": packet["weights"], "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(), "proc_start_ticks": common.process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"], "unique_write_root": str(OUTPUT.resolve()),
    }
    atomic_json_new(OUTPUT / "CLAIM.json", claim)
    heartbeat("WAIT_GPU_RESOURCE")
    worker_command = [
        sys.executable, str(Path(__file__).resolve()), "--task-id", TASK_ID,
        "--worker", "--output-root", str(OUTPUT),
        "--visual-root", str(VISUAL), "--executor-epoch", str(args.executor_epoch),
        "--claim", str(OUTPUT / "CLAIM.json"), "--run-signature", str(OUTPUT / "RUN_SIGNATURE.json"),
    ]
    lease_command = [
        sys.executable, str(GPU_LEASE_WRAPPER), "--task-id", TASK_ID, "--attempt-id", OUTPUT.name,
        "--executor-epoch", str(args.executor_epoch), "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib), "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.wall_seconds), "--receipt", str(OUTPUT / "GPU_COMMAND_RECEIPT.json"),
        "--claim-limit", packet["claim_limit"], "--", *worker_command,
    ]
    atomic_json(OUTPUT / "COMMAND.json", {"worker_command": worker_command, "lease_command": lease_command})
    with (OUTPUT / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is not None:
                break
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = load_json(lease_path) if lease_path.is_file() else {}
            acquired = lease.get("status") == "ACQUIRED" and lease.get("task_id") == TASK_ID
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE", args.gpu_id if acquired else None)
    gpu = load_json(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(packet, status, str(gpu.get("reason") or gpu.get("error") or "SAM31_W0_RUNTIME_FAILED"))
        return 3 if status == "BLOCKED_RESOURCE" else 2
    # Execution closes cleanly, but missing independent object anchors make the
    # semantic stage a quality rejection rather than an object-mask success.
    write_terminal(packet, "REJECTED_QUALITY", "TASK_OBJECT_AND_WEAK_ROLES_LACK_INDEPENDENT_VISUAL_ANCHORS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id", choices=(PRIMARY_TASK_ID, RECOVERY_TASK_ID, QUALITY_TASK_ID),
        default=PRIMARY_TASK_ID,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=TERMINAL_RECEIPT)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=5400)
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
