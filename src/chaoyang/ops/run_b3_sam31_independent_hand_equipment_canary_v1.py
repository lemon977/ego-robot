#!/usr/bin/env python3
"""Bounded SAM3.1 geometry-only Hand/equipment successor canary.

This challenger deliberately does not read HaWoR, MANO, PICO, controller or
previous Hand masks.  It runs two independent text-only SAM3.1 states per
frozen temporal chunk:

* ``hand`` -> unassigned Hand geometry candidates;
* ``wrist-worn device`` -> unassigned worn-equipment geometry candidates.

Candidates are never assigned left/right, never stitched across chunks and
never made consumer-eligible.  Any pixel overlap between the two roles is a
hard separation rejection, not something silently subtracted from either
mask.  The program is a development canary only; it cannot establish Mask
accuracy or physical tracker identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping
import uuid

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def projected_path(path: Path, staging_root: Path, final_root: Path) -> Path:
    """Project a staging artifact path into the not-yet-visible final tree."""
    return final_root / path.relative_to(staging_root)


def begin_staging(final_root: Path) -> Path:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists() or final_root.is_symlink():
        raise RuntimeError(f"immutable final output exists: {final_root}")
    staging = final_root.parent / f".{final_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging collision: {staging}")
    staging.mkdir()
    if os.stat(staging).st_dev != os.stat(final_root.parent).st_dev:
        shutil.rmtree(staging)
        raise RuntimeError("staging and final output are not on the same filesystem")
    return staging


def abort_staging(staging: Path) -> None:
    if staging.exists() and staging.is_dir() and not staging.is_symlink():
        shutil.rmtree(staging)


def commit_staging(staging: Path, final_root: Path) -> None:
    if final_root.exists() or final_root.is_symlink():
        raise RuntimeError(f"immutable final output appeared before commit: {final_root}")
    if os.stat(staging).st_dev != os.stat(final_root.parent).st_dev:
        raise RuntimeError("cross-filesystem commit forbidden")
    os.replace(staging, final_root)


def published_ref(staging_path: Path, staging_root: Path, final_root: Path) -> dict[str, Any]:
    return {
        "path": str(projected_path(staging_path, staging_root, final_root)),
        "bytes": staging_path.stat().st_size,
        "sha256": sha256(staging_path),
    }


def exact_ref(project: Path, ref: Mapping[str, Any], label: str) -> Path:
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = project / path
    path = path.resolve(strict=True)
    actual = sha256(path)
    if actual != ref["sha256"]:
        raise RuntimeError(f"{label} SHA drift: {actual} != {ref['sha256']}")
    if "bytes" in ref and path.stat().st_size != int(ref["bytes"]):
        raise RuntimeError(f"{label} byte-size drift")
    return path


def full_decode_gate(video: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {video}")
    stream = streams[0]
    if (
        int(stream["width"]) != 1280
        or int(stream["height"]) != 960
        or stream["r_frame_rate"] != "30/1"
        or int(stream["nb_read_frames"]) != expected_frames
    ):
        raise RuntimeError(f"video contract mismatch: {stream}")
    # ffprobe frame counting is not a decode-completion proof by itself.
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-map", "0:v:0", "-f", "null", "-",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {
        "path": str(video),
        "bytes": video.stat().st_size,
        "sha256": sha256(video),
        "full_decode": True,
        "frames": expected_frames,
        "width": 1280,
        "height": 960,
        "fps": "30/1",
    }


def validate_windows(frame_count: int, windows: Iterable[Mapping[str, Any]]) -> list[dict[str, int]]:
    rows = [{"start": int(row["start"]), "end_exclusive": int(row["end_exclusive"])}
            for row in windows]
    if not rows or rows[0]["start"] != 0 or rows[-1]["end_exclusive"] != frame_count:
        raise ValueError("windows must cover the whole session")
    cursor = 0
    for row in rows:
        if row["start"] != cursor or not row["start"] < row["end_exclusive"]:
            raise ValueError("windows must be ordered, contiguous and non-empty")
        cursor = row["end_exclusive"]
    return rows


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "0915-robot-recovery-v21-b3-independent-hand-equipment-canary-v1":
        raise ValueError("config schema mismatch")
    if config.get("status") != "FROZEN_NOT_EXECUTED":
        raise ValueError("only the frozen unexecuted package may start a canary")
    if set(config.get("immutable_parent_evidence", {})) != {"b1_result", "b1r_result"}:
        raise ValueError("B1/B1R immutable evidence pins changed")
    if set(config.get("runtime", {})) != {
        "runner", "shallow_renderer", "adapter", "checkpoint", "tracker_source",
        "gpu_lease_wrapper", "official_code_commit"
    }:
        raise ValueError("runtime code closure drift")
    if set(config.get("prompts", {})) != {"hand", "worn_equipment"}:
        raise ValueError("exactly two independent visual roles are required")
    if config["prompts"]["hand"].get("text") != "hand":
        raise ValueError("Hand prompt drift")
    if config["prompts"]["worn_equipment"].get("text") != "wrist-worn device":
        raise ValueError("worn-equipment prompt drift")
    policy = config.get("fixed_runtime_policy", {})
    required_policy = {
        "output_prob_thresh": 0.5,
        "chunk_frames": 64,
        "fresh_state_per_role_per_chunk": True,
        "native_full_is_last_batch": True,
        "left_right_identity": "UNKNOWN",
        "cross_chunk_identity": "NOT_STITCHED",
        "hand_equipment_overlap_hard_max_pixels": 0,
        "overlap_resolution": "REJECT_DO_NOT_SUBTRACT_OR_UNION",
        "empty_output_semantics": "UNKNOWN_NOT_ABSENT",
        "consumer_allowed": False,
    }
    if policy != required_policy:
        raise ValueError("fixed runtime policy drift")
    execution = config.get("execution_contract", {})
    expected_lease = {
        "gpu_id": 0,
        "priority": "CANARY",
        "min_free_mib": 49152,
        "wait_seconds": 1800,
        "wall_seconds": 3600,
        "concurrent_gpu_jobs": 1,
    }
    if execution.get("gpu_lease") != expected_lease:
        raise ValueError("GPU lease budget drift")
    for key in ("preflight_output", "model_output", "gpu_receipt", "shallow_visual_output"):
        value = execution.get(key)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"execution path must be a non-empty project-relative path: {key}")
    sessions = config.get("sessions", [])
    if [row.get("session_id") for row in sessions] != [
        "play_cards_0915_044", "get_potato_chips_0915_097"
    ]:
        raise ValueError("frozen two-session cohort drift")
    for row in sessions:
        if set(row) != {"session_id", "task", "frame_count", "prepared_rgb", "windows"}:
            raise ValueError("session may consume only the pinned prepared RGB")
        validate_windows(int(row["frame_count"]), row["windows"])


def normalize(outputs: Mapping[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    def array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            if not value:
                return np.asarray([])
            return np.stack([array(item) for item in value])
        return np.asarray(value)

    masks = array(outputs.get("out_binary_masks", np.zeros((0, height, width), bool)))
    scores = array(outputs.get("out_probs", outputs.get("out_scores", np.zeros(len(masks)))))
    ids = array(outputs.get("out_obj_ids", np.arange(len(masks))))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"SAM mask geometry drift: {masks.shape}")
    return masks.astype(bool), scores.reshape(-1).astype(float), ids.reshape(-1).astype(np.int64)


def component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1)


def select_geometry_candidates(
    masks: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
    *,
    minimum_area_pixels: int,
    maximum_area_fraction: float,
    maximum_components: int,
    maximum_candidates: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not (len(masks) == len(scores) == len(ids)):
        raise RuntimeError("SAM output array length mismatch")
    if len(set(int(value) for value in ids)) != len(ids):
        raise RuntimeError("duplicate SAM object ID")
    rows: list[dict[str, Any]] = []
    for index, (mask, score, raw_id) in enumerate(zip(masks, scores, ids)):
        area = int(mask.sum())
        components = component_count(mask)
        reasons = []
        if area < minimum_area_pixels:
            reasons.append("AREA_BELOW_MINIMUM")
        if area > int(mask.size * maximum_area_fraction):
            reasons.append("AREA_ABOVE_MAXIMUM")
        if components > maximum_components:
            reasons.append("TOO_MANY_COMPONENTS")
        rows.append({
            "array_index": index,
            "raw_id": int(raw_id),
            "score": float(score),
            "area_pixels": area,
            "component_count": components,
            "geometry_eligible": not reasons,
            "reasons": reasons,
        })
    eligible = sorted(
        (row for row in rows if row["geometry_eligible"]),
        key=lambda row: (-row["score"], -row["area_pixels"], row["raw_id"]),
    )
    selected = [int(row["raw_id"]) for row in eligible[:maximum_candidates]]
    for row in rows:
        row["selected"] = row["raw_id"] in selected
        if row["geometry_eligible"] and not row["selected"]:
            row["reasons"].append("ABOVE_FROZEN_CANDIDATE_CAP")
    return selected, rows


def separation_gate(hand_masks: Iterable[np.ndarray], equipment_masks: Iterable[np.ndarray]) -> dict[str, Any]:
    hand = list(hand_masks)
    equipment = list(equipment_masks)
    overlap = 0
    pair_rows = []
    for hand_index, hand_mask in enumerate(hand):
        for equipment_index, equipment_mask in enumerate(equipment):
            pixels = int(np.count_nonzero(hand_mask & equipment_mask))
            overlap += pixels
            pair_rows.append({
                "hand_candidate": hand_index,
                "equipment_candidate": equipment_index,
                "overlap_pixels": pixels,
            })
    hand_present = any(mask.any() for mask in hand)
    equipment_present = any(mask.any() for mask in equipment)
    if overlap:
        state = "SEPARATION_REJECTED_OVERLAP"
    elif not hand_present:
        state = "SEPARATION_UNKNOWN_NO_HAND_CANDIDATE"
    elif not equipment_present:
        state = "SEPARATION_UNKNOWN_NO_EQUIPMENT_CANDIDATE"
    else:
        state = "SEPARATION_PASS_GEOMETRY_ONLY"
    # Zero is intentional.  This canary must not invent an overlap threshold
    # or carve equipment pixels out of the Hand candidate.  Missing either
    # role is UNKNOWN rather than a vacuous separation pass.
    return {
        "state": state,
        "pass": state == "SEPARATION_PASS_GEOMETRY_ONLY",
        "hand_candidate_present": hand_present,
        "equipment_candidate_present": equipment_present,
        "total_overlap_pixels": overlap,
        "hard_max_overlap_pixels": 0,
        "pairs": pair_rows,
        "nonpass_policy": "REJECT_OVERLAP_OR_PRESERVE_MISSING_ROLE_AS_UNKNOWN",
    }


def aggregate_separation_state(states: Iterable[str]) -> str:
    values = list(states)
    if any(value == "SEPARATION_REJECTED_OVERLAP" for value in values):
        return "COMPLETED_SEPARATION_REJECTED"
    if any(value.startswith("SEPARATION_UNKNOWN_") for value in values) or not values:
        return "COMPLETED_SEPARATION_UNKNOWN"
    if all(value == "SEPARATION_PASS_GEOMETRY_ONLY" for value in values):
        return "COMPLETED_SEPARATION_GEOMETRY_PASS"
    raise RuntimeError(f"unknown separation states: {sorted(set(values))}")


def decode_frames(video: Path, destination: Path, expected_frames: int) -> tuple[int, int]:
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vsync", "0", str(destination / "%06d.png"),
    ], check=True)
    frames = sorted(destination.glob("*.png"))
    if len(frames) != expected_frames:
        raise RuntimeError(f"decoded frame mismatch: {len(frames)} != {expected_frames}")
    image = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("first decoded frame unreadable")
    height, width = image.shape[:2]
    if (width, height) != (1280, 960):
        raise RuntimeError(f"unexpected image geometry: {(width, height)}")
    return height, width


def save_packed(
    path: Path,
    masks: np.ndarray,
    *,
    staging_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(np.asarray(masks, bool).reshape(len(masks), -1), axis=1)
    np.savez_compressed(path, packed=packed, frame_count=len(masks), height=masks.shape[1], width=masks.shape[2])
    return published_ref(path, staging_root, final_root)


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    if mask.any():
        result[mask] = np.rint(0.52 * result[mask] + 0.48 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    return result


def write_review_frame(
    frame_root: Path,
    destination: Path,
    *,
    frame_index: int,
    hand_masks: Iterable[np.ndarray],
    equipment_masks: Iterable[np.ndarray],
) -> None:
    raw = cv2.imread(str(frame_root / f"{frame_index + 1:06d}.png"), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"review source frame unreadable: {frame_index}")
    hand_parts, equipment_parts = list(hand_masks), list(equipment_masks)
    hand = np.logical_or.reduce(hand_parts) if hand_parts else np.zeros(raw.shape[:2], bool)
    equipment = np.logical_or.reduce(equipment_parts) if equipment_parts else np.zeros(raw.shape[:2], bool)
    overlap = hand & equipment
    hand_tile = tint(raw, hand, (255, 150, 30))
    equipment_tile = tint(raw, equipment, (30, 230, 255))
    separation_tile = tint(tint(raw, hand, (255, 150, 30)), equipment, (30, 230, 255))
    separation_tile = tint(separation_tile, overlap, (20, 20, 255))
    for tile, title in zip(
        (hand_tile, equipment_tile, separation_tile),
        ("HAND CANDIDATES / ID UNKNOWN", "WORN EQUIPMENT / ID UNKNOWN", "SEPARATION / RED=REJECT"),
    ):
        cv2.rectangle(tile, (0, 0), (1280, 42), (0, 0, 0), -1)
        cv2.putText(tile, f"{title} | frame {frame_index} | OFFLINE DIAGNOSTIC",
                    (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    panels = [cv2.resize(tile, (640, 480), interpolation=cv2.INTER_AREA)
              for tile in (hand_tile, equipment_tile, separation_tile)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write review frame: {destination}")


def encode_review(
    frame_root: Path,
    destination: Path,
    expected_frames: int,
    *,
    staging_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30",
        "-i", str(frame_root / "%06d.jpg"), "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", str(destination),
    ], check=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(destination),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != expected_frames or stream["r_frame_rate"] != "30/1":
        raise RuntimeError(f"review decode mismatch: {stream}")
    return {**published_ref(destination, staging_root, final_root),
            "full_decode": True, "decode": stream}


def process_prompt_window(
    model: Any,
    frame_root: Path,
    *,
    prompt_spec: Mapping[str, Any],
    start: int,
    end_exclusive: int,
    height: int,
    width: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True,
                             async_loading_frames=False)
    try:
        _, initial = model.add_prompt(
            inference_state=state,
            frame_idx=start,
            text_str=str(prompt_spec["text"]),
            output_prob_thresh=0.5,
        )
        seed_masks, seed_scores, seed_ids = normalize(initial, height, width)
        selected, seed_ledger = select_geometry_candidates(
            seed_masks, seed_scores, seed_ids,
            minimum_area_pixels=int(prompt_spec["minimum_area_pixels"]),
            maximum_area_fraction=float(prompt_spec["maximum_area_fraction"]),
            maximum_components=int(prompt_spec["maximum_components"]),
            maximum_candidates=int(prompt_spec["maximum_candidates"]),
        )
        stream = {str(raw_id): np.zeros((end_exclusive - start, height, width), bool)
                  for raw_id in selected}
        by_id = {int(value): seed_masks[index] for index, value in enumerate(seed_ids)}
        for raw_id in selected:
            stream[str(raw_id)][0] = by_id[raw_id]
        rows: dict[int, dict[str, Any]] = {}
        for frame_index, outputs in model.propagate_in_video(
            inference_state=state,
            start_frame_idx=start,
            max_frame_num_to_track=end_exclusive - start,
            reverse=False,
            output_prob_thresh=0.5,
            is_last_batch=True,
        ):
            frame = int(frame_index)
            if not start <= frame < end_exclusive:
                raise RuntimeError("SAM emitted frame outside frozen window")
            masks, scores, ids = normalize(outputs, height, width)
            by_id = {int(value): masks[index] for index, value in enumerate(ids)}
            score_by_id = {int(value): float(scores[index]) for index, value in enumerate(ids)}
            for raw_id in selected:
                mask = by_id.get(raw_id, np.zeros((height, width), bool))
                stream[str(raw_id)][frame - start] = mask
            rows[frame] = {
                "frame_index": frame,
                "output_object_ids": [int(value) for value in ids],
                "selected_present_ids": [raw_id for raw_id in selected if by_id.get(raw_id, np.zeros((), bool)).any()],
                "scores": {str(raw_id): score_by_id.get(raw_id) for raw_id in selected},
            }
        return stream, {
            "prompt": prompt_spec["text"],
            "start": start,
            "end_exclusive": end_exclusive,
            "seed_ledger": seed_ledger,
            "selected_raw_ids": selected,
            "frames": [rows.get(frame, {"frame_index": frame, "reason": "NO_STREAM_OUTPUT"})
                       for frame in range(start, end_exclusive)],
        }
    finally:
        state.clear()


def validate_all_pins(
    config_path: Path,
    config: Mapping[str, Any],
    project: Path,
    *,
    decode_videos: bool,
) -> dict[str, Any]:
    pins: dict[str, Any] = {
        "config": {"path": str(config_path), "bytes": config_path.stat().st_size,
                   "sha256": sha256(config_path)},
        "immutable_parent_evidence": {},
        "runtime": {},
        "videos": {},
    }
    for label, ref in config["immutable_parent_evidence"].items():
        path = exact_ref(project, ref, label)
        pins["immutable_parent_evidence"][label] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        }
    for label in (
        "runner", "shallow_renderer", "checkpoint", "adapter", "tracker_source",
        "gpu_lease_wrapper",
    ):
        path = exact_ref(project, config["runtime"][label], label)
        pins["runtime"][label] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        }
    for session in config["sessions"]:
        session_id = session["session_id"]
        path = exact_ref(project, session["prepared_rgb"], f"{session_id} RGB")
        pins["videos"][session_id] = (
            full_decode_gate(path, int(session["frame_count"]))
            if decode_videos else
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
             "full_decode": False}
        )
    return pins


def run_preflight(config_path: Path, project: Path, final_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    # No torch/SAM import occurs in preflight.
    pins = validate_all_pins(config_path, config, project, decode_videos=True)
    staging = begin_staging(final_root)
    try:
        result = {
            "schema_version": "0915-robot-recovery-v21-b3-preflight-result-v1",
            "status": "PREFLIGHT_PASSED",
            "model_loaded": False,
            "model_executed": False,
            "gpu_consumed": False,
            "claim_limit": config["claim_limit"],
            "pins": pins,
        }
        atomic_json(staging / "PREFLIGHT_RECEIPT.json", result)
        atomic_json(staging / "RESULT.json", result)
        commit_staging(staging, final_root)
        return result
    except BaseException:
        abort_staging(staging)
        raise


def _run_model_into(
    config_path: Path,
    config: Mapping[str, Any],
    project: Path,
    staging_root: Path,
    final_root: Path,
    pins: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(pins["runtime"]["checkpoint"]["path"])
    sys.path[:0] = [str(project / "src"), str(project / "vendor/SAM3")]
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    adapter, build_evidence = build_pinned_adapter(
        official_code_root=project / "vendor/SAM3", checkpoint_path=checkpoint,
    )
    model = adapter.model
    session_results = []
    try:
        for session in config["sessions"]:
            session_id = session["session_id"]
            session_root = staging_root / session_id
            frame_root = session_root / "frames"
            video = Path(pins["videos"][session_id]["path"])
            frame_count = int(session["frame_count"])
            windows = validate_windows(frame_count, session["windows"])
            height, width = decode_frames(video, frame_root, frame_count)
            window_results = []
            session_frame_states: list[str] = []
            for window_index, window in enumerate(windows):
                start, end = window["start"], window["end_exclusive"]
                streams: dict[str, dict[str, np.ndarray]] = {}
                ledgers = {}
                for role in ("hand", "worn_equipment"):
                    streams[role], ledgers[role] = process_prompt_window(
                        model, frame_root, prompt_spec=config["prompts"][role],
                        start=start, end_exclusive=end, height=height, width=width,
                    )
                per_frame_separation = []
                for local in range(end - start):
                    gate = separation_gate(
                        [masks[local] for masks in streams["hand"].values()],
                        [masks[local] for masks in streams["worn_equipment"].values()],
                    )
                    gate["frame_index"] = start + local
                    per_frame_separation.append(gate)
                    session_frame_states.append(gate["state"])
                    write_review_frame(
                        frame_root,
                        session_root / "review_frames" / f"{start + local:06d}.jpg",
                        frame_index=start + local,
                        hand_masks=[masks[local] for masks in streams["hand"].values()],
                        equipment_masks=[masks[local] for masks in streams["worn_equipment"].values()],
                    )
                refs = {}
                for role, candidates in streams.items():
                    refs[role] = {}
                    for raw_id, masks in candidates.items():
                        refs[role][raw_id] = save_packed(
                            session_root / "candidates" / f"window_{window_index:03d}" / role / f"raw_{raw_id}.npz",
                            masks, staging_root=staging_root, final_root=final_root,
                        )
                window_state = aggregate_separation_state(row["state"] for row in per_frame_separation)
                ledger = {
                    "window_index": window_index,
                    "start": start,
                    "end_exclusive": end,
                    "status": window_state,
                    "left_right_identity": "UNKNOWN",
                    "cross_window_identity": "NOT_STITCHED",
                    "consumer_allowed": False,
                    "mask_accuracy_claimed": False,
                    "prompt_ledgers": ledgers,
                    "candidate_archives": refs,
                    "separation": per_frame_separation,
                }
                atomic_json(session_root / "ledgers" / f"window_{window_index:03d}.json", ledger)
                window_results.append(ledger)
            review = encode_review(
                session_root / "review_frames",
                session_root / f"{session_id}_B3_HAND_EQUIPMENT_GEOMETRY_REVIEW_ONLY.mp4",
                frame_count, staging_root=staging_root, final_root=final_root,
            )
            session_state = aggregate_separation_state(session_frame_states)
            counts = {state: session_frame_states.count(state) for state in sorted(set(session_frame_states))}
            session_result = {
                "session_id": session_id,
                "frame_count": frame_count,
                "status": session_state,
                "separation_state_counts": counts,
                "model_execution_completed": True,
                "consumer_allowed": False,
                "mask_accuracy_claimed": False,
                "physical_tracker_identity_claimed": False,
                "left_right_identity": "UNKNOWN",
                "review": review,
                "windows": window_results,
            }
            atomic_json(session_root / "RESULT.json", session_result)
            session_results.append(session_result)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    statuses = [row["status"] for row in session_results]
    if any(value == "COMPLETED_SEPARATION_REJECTED" for value in statuses):
        overall = "COMPLETED_DIAGNOSTIC_SEPARATION_REJECTED"
    elif any(value == "COMPLETED_SEPARATION_UNKNOWN" for value in statuses):
        overall = "COMPLETED_DIAGNOSTIC_SEPARATION_UNKNOWN"
    else:
        overall = "COMPLETED_DIAGNOSTIC_SEPARATION_GEOMETRY_PASS"
    result = {
        "schema_version": "0915-robot-recovery-v21-b3-independent-hand-equipment-result-v1",
        "status": overall,
        "model_execution_completed": True,
        "mask_quality_status": "NOT_EVALUATED_NO_GROUND_TRUTH",
        "claim_limit": config["claim_limit"],
        "training_eligible": False,
        "consumer_allowed": False,
        "mask_accuracy_claimed": False,
        "physical_tracker_identity_claimed": False,
        "hawor_consumed": False,
        "mano_consumed": False,
        "pico_consumed": False,
        "previous_hand_masks_consumed": False,
        "preflight": pins,
        "build_evidence": build_evidence,
        "sessions": session_results,
    }
    atomic_json(staging_root / "RESULT.json", result)
    return result


def run(config_path: Path, project: Path, final_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    # The formal run repeats all preflight checks.  It does not trust a prior
    # receipt whose sources might have changed between preflight and GPU use.
    pins = validate_all_pins(config_path, config, project, decode_videos=True)
    staging = begin_staging(final_root)
    try:
        result = _run_model_into(config_path, config, project, staging, final_root, pins)
        commit_staging(staging, final_root)
        return result
    except BaseException:
        abort_staging(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/workspace/code/chaoyang"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    project = args.project_root.resolve(strict=True)
    final_root = args.output_root.resolve()
    result = (
        run_preflight(config_path, project, final_root)
        if args.preflight_only else
        run(config_path, project, final_root)
    )
    print(json.dumps({"status": result["status"], "result": str(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
