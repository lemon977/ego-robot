#!/usr/bin/env python3
"""Run one bounded, bidirectional attachment-mask canary on 031 and 007."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref
from chaoyang.pipeline.attachment_tracker_v1 import (
    AttachmentTrackGateV1,
    admit_bidirectional_candidate,
    propagate_one_frame,
    read_binary_mask,
    seed_brackets,
)

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene"
RAW = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1"
CASES = {
    "play_cards_0915_031": {
        "short": "031", "raw": RAW / "play_cards_0915_031/raw",
        "masks": V5 / "masks_031_v1", "seeds": [66, 67, 68, 74, 79, 80, 81],
    },
    "get_potato_chips_0915_007": {
        "short": "007", "raw": RAW / "get_potato_chips_0915_007/raw",
        "masks": V5 / "masks_007_v1", "seeds": [181, 182, 183, 189, 194, 195, 196],
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame(folder: Path, frame_id: int) -> np.ndarray:
    value = cv2.imread(str(folder / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(folder / f"{frame_id:06d}.png")
    return value


def propagate_sequence(raw: Path, start: int, stop: int, mask: np.ndarray) -> dict[int, np.ndarray]:
    result = {start: mask}
    direction = 1 if stop > start else -1
    current_frame = load_frame(raw, start)
    current_mask = mask
    for target_id in range(start + direction, stop + direction, direction):
        target_frame = load_frame(raw, target_id)
        current_mask = propagate_one_frame(current_frame, target_frame, current_mask)
        result[target_id] = current_mask
        current_frame = target_frame
    return result


def overlay(frame: np.ndarray, candidate: np.ndarray, task_object: np.ndarray, state: str, frame_id: int) -> np.ndarray:
    panel = frame.copy()
    panel[candidate] = (0.45 * panel[candidate] + 0.55 * np.asarray([0, 0, 255])).astype(np.uint8)
    object_only = task_object & ~candidate
    panel[object_only] = (0.45 * panel[object_only] + 0.55 * np.asarray([0, 255, 0])).astype(np.uint8)
    overlap = task_object & candidate
    panel[overlap] = (0.35 * panel[overlap] + 0.65 * np.asarray([0, 255, 255])).astype(np.uint8)
    cv2.putText(panel, f"frame {frame_id:03d} | {state}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(panel, "red=device candidate green=object yellow=overlap", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def run_case(session: str, spec: dict, root: Path) -> dict:
    gate = AttachmentTrackGateV1()
    seeds = spec["seeds"]
    states: dict[int, str] = {frame_id: "seeded" for frame_id in seeds}
    candidates = {
        frame_id: read_binary_mask(str(spec["masks"] / "capture_device" / f"{frame_id:06d}.png"))
        for frame_id in seeds
    }
    diagnostics: list[dict] = []
    for left, right in seed_brackets(seeds, gate=gate):
        forward = propagate_sequence(spec["raw"], left, right, candidates[left])
        reverse = propagate_sequence(spec["raw"], right, left, candidates[right])
        for frame_id in range(left + 1, right):
            passed, metrics = admit_bidirectional_candidate(
                forward[frame_id], reverse[frame_id],
                left_seed_area=int(np.count_nonzero(candidates[left])),
                right_seed_area=int(np.count_nonzero(candidates[right])), gate=gate,
            )
            diagnostics.append({"frame_id": frame_id, "left_seed": left, "right_seed": right, "passed": passed, **metrics})
            if passed:
                candidates[frame_id] = forward[frame_id] & reverse[frame_id]
                states[frame_id] = "tracked"
            else:
                states.setdefault(frame_id, "unknown")

    first, last = min(seeds), max(seeds)
    case_root = root / session
    mask_root = case_root / "candidate_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    writer = None
    video_path = case_root / "ATTACHMENT_TRACK_REVIEW.mp4"
    rows = []
    try:
        for frame_id in range(first, last + 1):
            frame = load_frame(spec["raw"], frame_id)
            state = states.get(frame_id, "unknown")
            candidate = candidates.get(frame_id, np.zeros(frame.shape[:2], dtype=bool))
            task_object = read_binary_mask(str(spec["masks"] / "task_object" / f"{frame_id:06d}.png"))
            if state != "unknown":
                cv2.imwrite(str(mask_root / f"{frame_id:06d}.png"), candidate.astype(np.uint8) * 255)
            rendered = overlay(frame, candidate, task_object, state, frame_id)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open writer: {video_path}")
            writer.write(rendered)
            rows.append({
                "frame_id": frame_id, "state": state,
                "candidate_area_px": int(np.count_nonzero(candidate)),
                "object_overlap_px": int(np.count_nonzero(candidate & task_object)),
            })
    finally:
        if writer is not None:
            writer.release()
    capture = cv2.VideoCapture(str(video_path)); decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok: break
        decoded += 1
    capture.release()
    expected = last - first + 1
    if decoded != expected:
        raise RuntimeError(f"VIDEO_DECODE_MISMATCH:{decoded}!={expected}")
    tracked = sum(row["state"] == "tracked" for row in rows)
    unknown = sum(row["state"] == "unknown" for row in rows)
    result = {
        "schema_version": "LOCAL_ATTACHMENT_TRACK_CANARY_V1",
        "session_id": session,
        "created_at": now(),
        "input_authority": "EXISTING_SAM31_DEVICE_SEEDS_PLUS_RAW_RGB",
        "window": [first, last],
        "seed_frames": seeds,
        "states": rows,
        "gap_diagnostics": diagnostics,
        "counts": {"seeded": len(seeds), "tracked": tracked, "unknown": unknown},
        "gate": gate.__dict__,
        "quality": "NOT_EVALUATED_VISUAL_REVIEW_REQUIRED",
        "adoption": "NOT_ADOPTED",
        "outside_window": "UNKNOWN",
        "contact_authority": False,
        "object6d_authority": False,
        "training_eligible": False,
        "review_video": {"path": str(video_path), "sha256": sha256(video_path), "decoded_frames": decoded},
    }
    write_once(case_root / "RESULT.json", result)
    return {"result": artifact_ref(case_root / "RESULT.json"), "review_video": artifact_ref(video_path), "counts": result["counts"]}


def main() -> int:
    root = ATTEMPT / "lanes/scene_evidence/attachment_track_canary_v1"
    if root.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{root}")
    outputs = {session: run_case(session, spec, root) for session, spec in CASES.items()}
    summary = {
        "schema_version": "LOCAL_ATTACHMENT_TRACK_CANARY_SUMMARY_V1",
        "task_id": TASK, "created_at": now(), "status": "EXECUTED_PENDING_VISUAL_REVIEW",
        "sessions": outputs,
        "same_signature_scene_retry": False,
        "new_model_invocations": 0,
        "claim_limit": "Local evidence repair only; UNKNOWN outside seed brackets and no Contact/Object6D authority.",
    }
    write_once(root / "RESULT.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
