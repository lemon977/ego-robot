#!/usr/bin/env python3
"""Render fail-closed D1_CLEAN_PREP full-session offline review videos.

The renderer consumes the published PREPARED_V1 RESULT/FRAME_MANIFEST files.  It
does not infer, fill, inpaint, or modify any Clean data.  Every referenced file
is SHA checked before use and the program refuses to render unless:

* candidate decoded RGB is byte-identical to same-frame Raw;
* M_remove ⊆ M_write ⊆ M_flow and UNKNOWN == M_write;
* no Task Object source code overlaps M_write;
* Poker has no admitted Hand side and physical card/face identity is UNKNOWN;
* Chips admits the right Hand only, keeps left Hand UNKNOWN, and keeps the three
  object source codes separate.

The resulting MP4s are visual diagnostics only.  They are explicitly labelled
``CLEAN NOT MATERIALIZED / UNKNOWN / OFFLINE REVIEW``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
import uuid

import cv2
import numpy as np
from PIL import Image


SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-visual-review-v1"
PRODUCER_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-cpu-v1-result-v1"
FRAME_SCHEMA = "0915-robot-recovery-v21-d1-clean-prep-cpu-v1-frame-manifest-v1"
EXPECTED_COUNTS = {
    "play_cards_0915_044": 166,
    "get_potato_chips_0915_097": 394,
}
EXPECTED_TASKS = {
    "play_cards_0915_044": "playing_cards",
    "get_potato_chips_0915_097": "potato_chips",
}
EXPECTED_INPUT_SHAPE = (960, 1280)
FPS = 30.0
PANEL_WIDTH = 480
PANEL_HEIGHT = 360
HEADER_HEIGHT = 82
FOOTER_HEIGHT = 64
OUTPUT_SIZE = (PANEL_WIDTH * 3, HEADER_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT)

SOURCE_RAW = 0
SOURCE_POKER_ADMITTED = 1
SOURCE_POKER_UNKNOWN = 2
SOURCE_CHIP_00 = 10
SOURCE_CHIP_01 = 11
SOURCE_CHIP_02 = 12
SOURCE_UNKNOWN = 250
ALLOWED_SOURCE_CODES = {
    SOURCE_RAW,
    SOURCE_POKER_ADMITTED,
    SOURCE_POKER_UNKNOWN,
    SOURCE_CHIP_00,
    SOURCE_CHIP_01,
    SOURCE_CHIP_02,
    SOURCE_UNKNOWN,
}
OBJECT_SOURCE_CODES = {
    SOURCE_POKER_ADMITTED,
    SOURCE_POKER_UNKNOWN,
    SOURCE_CHIP_00,
    SOURCE_CHIP_01,
    SOURCE_CHIP_02,
}

# RGB colors.  OpenCV receives BGR only at the final VideoWriter boundary.
COLOR_UNKNOWN = np.array([0, 220, 255], dtype=np.uint8)
COLOR_REMOVE = np.array([255, 32, 180], dtype=np.uint8)
COLOR_FLOW = np.array([255, 230, 0], dtype=np.uint8)
SOURCE_COLORS = {
    SOURCE_POKER_ADMITTED: np.array([30, 230, 90], dtype=np.uint8),
    SOURCE_POKER_UNKNOWN: np.array([255, 128, 0], dtype=np.uint8),
    SOURCE_CHIP_00: np.array([30, 230, 90], dtype=np.uint8),
    SOURCE_CHIP_01: np.array([30, 130, 255], dtype=np.uint8),
    SOURCE_CHIP_02: np.array([190, 80, 255], dtype=np.uint8),
}


class ReviewError(RuntimeError):
    """A provenance, domain, semantic, encode, or decode gate failed."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"JSON object required: {path}")
    return value


def checked_ref(value: Mapping[str, Any], label: str) -> Path:
    if not isinstance(value, Mapping) or not {"path", "bytes", "sha256"}.issubset(value):
        raise ReviewError(f"{label}: path/bytes/sha256 reference required")
    path = Path(str(value["path"])).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ReviewError(f"{label}: regular non-symlink file required: {path}")
    if path.stat().st_size != int(value["bytes"]):
        raise ReviewError(f"{label}: byte count drift: {path}")
    if sha256_file(path) != value["sha256"]:
        raise ReviewError(f"{label}: SHA256 drift: {path}")
    return path


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def projected_ref(existing: Path, destination: Path) -> dict[str, Any]:
    """Bind staged bytes to their post-commit path without publishing a stale staging ref."""
    existing = existing.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "bytes": existing.stat().st_size,
        "sha256": sha256_file(existing),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_rgb(reference: Mapping[str, Any], label: str) -> np.ndarray:
    path = checked_ref(reference, label)
    if path.suffix.lower() != ".png":
        raise ReviewError(f"{label}: lossless PNG required")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ReviewError(f"{label}: exact RGB PNG required, got {image.mode}")
        value = np.asarray(image).copy()
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ReviewError(f"{label}: uint8 HxWx3 required")
    return value


def read_gray(reference: Mapping[str, Any], label: str) -> np.ndarray:
    path = checked_ref(reference, label)
    if path.suffix.lower() != ".png":
        raise ReviewError(f"{label}: PNG required")
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def read_mask(reference: Mapping[str, Any], label: str) -> np.ndarray:
    value = read_gray(reference, label)
    if not set(np.unique(value).tolist()).issubset({0, 255}):
        raise ReviewError(f"{label}: binary 0/255 mask required")
    return value > 0


def alpha_fill(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if np.any(mask):
        mixed = image[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
        image[mask] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def contour(image: np.ndarray, mask: np.ndarray, color: np.ndarray, thickness: int = 3) -> None:
    if not np.any(mask):
        return
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, tuple(int(x) for x in color[::-1]), thickness, cv2.LINE_AA)
    image[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    return (dilated != eroded)


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55,
             color: tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def resize_panel(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)


def render_frame(raw: np.ndarray, candidate: np.ndarray, m_remove: np.ndarray, m_write: np.ndarray,
                 m_flow: np.ndarray, unknown: np.ndarray, source: np.ndarray, session_id: str,
                 task: str, frame_id: int, frame_count: int) -> np.ndarray:
    domain = candidate.copy()
    alpha_fill(domain, m_flow, COLOR_FLOW, 0.14)
    alpha_fill(domain, unknown, COLOR_UNKNOWN, 0.40)
    domain[boundary(m_flow)] = COLOR_FLOW
    contour(domain, m_remove, COLOR_REMOVE, thickness=4)

    source_view = raw.copy()
    for code, color in SOURCE_COLORS.items():
        mask = source == code
        alpha_fill(source_view, mask, color, 0.42)
        contour(source_view, mask, color, thickness=3)
    source_view[source == SOURCE_UNKNOWN] = (
        0.55 * source_view[source == SOURCE_UNKNOWN].astype(np.float32)
        + 0.45 * COLOR_UNKNOWN.astype(np.float32)
    ).astype(np.uint8)

    panels = [resize_panel(raw), resize_panel(domain), resize_panel(source_view)]
    canvas = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:HEADER_HEIGHT + PANEL_HEIGHT] = np.concatenate(panels, axis=1)

    put_text(canvas, "D1 CLEAN PREP | CLEAN NOT MATERIALIZED / UNKNOWN / OFFLINE REVIEW",
             (18, 27), 0.72, (0, 230, 255), 2)
    put_text(canvas, f"{session_id} | frame {frame_id + 1}/{frame_count} | 30 FPS",
             (18, 56), 0.58, (255, 255, 255), 1)
    if task == "playing_cards":
        semantic = "HAND: NONE ADMITTED (L/R UNKNOWN) | CARD INSTANCE/FACE: UNKNOWN | PROTECTION ONLY"
    else:
        semantic = "HAND: RIGHT ONLY | LEFT UNKNOWN | CHIPS: 3 SEPARATE SLOTS (NO UNION)"
    put_text(canvas, semantic, (18, 76), 0.47, (80, 210, 255), 1)

    panel_y = HEADER_HEIGHT + 24
    put_text(canvas, "RAW (same-frame source)", (12, panel_y), 0.55, (255, 255, 255), 2)
    put_text(canvas, "DOMAINS (candidate == Raw)", (PANEL_WIDTH + 12, panel_y), 0.55, (255, 255, 255), 2)
    put_text(canvas, "SOURCE / TASK-OBJECT PROTECTION", (2 * PANEL_WIDTH + 12, panel_y), 0.52,
             (255, 255, 255), 2)

    footer_y = HEADER_HEIGHT + PANEL_HEIGHT + 24
    put_text(canvas, "cyan fill: M_write = UNKNOWN | magenta edge: M_remove | yellow edge: M_flow",
             (18, footer_y), 0.48, (240, 240, 240), 1)
    if task == "playing_cards":
        legend = "orange: protected Raw, identity UNKNOWN | no pixels filled; no Hand removal authority"
    else:
        legend = "green/blue/violet: chip slots 00/01/02 | no object masks admitted in this preparation"
    put_text(canvas, legend, (18, footer_y + 25), 0.48, (220, 220, 220), 1)
    return canvas


def validate_semantics(manifest: Mapping[str, Any], session_id: str, task: str) -> None:
    unknown_sides = manifest.get("unknown_hand_sides")
    if task == "playing_cards":
        if unknown_sides != ["left", "right"]:
            raise ReviewError(f"{session_id}: Poker must keep both Hand sides UNKNOWN")
        if manifest.get("poker_identity") != "UNKNOWN_PHYSICAL_CARD_OR_FACE_IDENTITY":
            raise ReviewError(f"{session_id}: frozen Poker identity-UNKNOWN contract drift")
    else:
        if unknown_sides != ["left"]:
            raise ReviewError(f"{session_id}: Chips must keep left Hand UNKNOWN only")
        if manifest.get("chips_instance_mode") != "THREE_SEPARATE_NO_UNION":
            raise ReviewError(f"{session_id}: Chips three-instance/no-union contract drift")


def validate_frame(row: Mapping[str, Any], session_id: str, task: str, frame_id: int,
                   expected_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    if row.get("frame_id") != frame_id:
        raise ReviewError(f"{session_id}: non-contiguous frame id at {frame_id}")
    raw = read_rgb(row.get("raw_rgb", {}), f"{session_id}/{frame_id}/Raw")
    candidate = read_rgb(row.get("candidate_rgb", {}), f"{session_id}/{frame_id}/candidate")
    if raw.shape[:2] != expected_shape or candidate.shape != raw.shape:
        raise ReviewError(f"{session_id}/{frame_id}: RGB shape drift")
    if row.get("raw_decoded_sha256") != decoded_sha256(raw):
        raise ReviewError(f"{session_id}/{frame_id}: Raw decoded SHA drift")
    if row.get("candidate_decoded_sha256") != decoded_sha256(candidate):
        raise ReviewError(f"{session_id}/{frame_id}: candidate decoded SHA drift")
    if not np.array_equal(raw, candidate):
        raise ReviewError(f"{session_id}/{frame_id}: candidate is not decoded-byte-identical to Raw")

    m_remove = read_mask(row.get("M_remove", {}), f"{session_id}/{frame_id}/M_remove")
    m_flow = read_mask(row.get("M_flow", {}), f"{session_id}/{frame_id}/M_flow")
    m_write = read_mask(row.get("M_write", {}), f"{session_id}/{frame_id}/M_write")
    unknown = read_mask(row.get("UNKNOWN", {}), f"{session_id}/{frame_id}/UNKNOWN")
    source = read_gray(row.get("source_map", {}), f"{session_id}/{frame_id}/source_map")
    arrays = (m_remove, m_flow, m_write, unknown, source)
    if any(array.shape != expected_shape for array in arrays):
        raise ReviewError(f"{session_id}/{frame_id}: mask/source shape drift")
    codes = set(np.unique(source).tolist())
    if not codes.issubset(ALLOWED_SOURCE_CODES):
        raise ReviewError(f"{session_id}/{frame_id}: forbidden source codes {sorted(codes)}")
    if np.any(m_remove & ~m_write) or np.any(m_write & ~m_flow):
        raise ReviewError(f"{session_id}/{frame_id}: M_remove ⊆ M_write ⊆ M_flow failed")
    if not np.array_equal(unknown, m_write):
        raise ReviewError(f"{session_id}/{frame_id}: UNKNOWN != M_write")
    if not np.array_equal(source == SOURCE_UNKNOWN, unknown):
        raise ReviewError(f"{session_id}/{frame_id}: source-map UNKNOWN mismatch")
    protected = np.isin(source, list(OBJECT_SOURCE_CODES))
    if np.any(protected & m_write):
        raise ReviewError(f"{session_id}/{frame_id}: Task Object protection overlaps M_write")

    statuses = row.get("hand_side_status")
    if not isinstance(statuses, Mapping):
        raise ReviewError(f"{session_id}/{frame_id}: hand-side status required")
    if task == "playing_cards":
        if any(str(statuses.get(side, "")).startswith("ADMITTED_") for side in ("left", "right")):
            raise ReviewError(f"{session_id}/{frame_id}: Poker unexpectedly admits a Hand side")
        if any(code in codes for code in (SOURCE_POKER_ADMITTED, SOURCE_CHIP_00, SOURCE_CHIP_01, SOURCE_CHIP_02)):
            raise ReviewError(f"{session_id}/{frame_id}: Poker source identity/instance code drift")
    else:
        if statuses.get("right") != "ADMITTED_B1_PER_SIDE_HAND_MASK":
            raise ReviewError(f"{session_id}/{frame_id}: Chips right Hand is not admitted")
        if str(statuses.get("left", "")).startswith("ADMITTED_"):
            raise ReviewError(f"{session_id}/{frame_id}: Chips left Hand must stay UNKNOWN")
        if any(code in codes for code in (SOURCE_POKER_ADMITTED, SOURCE_POKER_UNKNOWN)):
            raise ReviewError(f"{session_id}/{frame_id}: Chips contains Poker source code")

    actual_counts = {
        "m_remove_pixels": int(m_remove.sum()),
        "m_flow_pixels": int(m_flow.sum()),
        "m_write_pixels": int(m_write.sum()),
        "unknown_pixels": int(unknown.sum()),
        "protected_pixels": int(protected.sum()),
        "changed_pixels": int(np.any(raw != candidate, axis=2).sum()),
    }
    for key, value in actual_counts.items():
        if int(row.get(key, -1)) != value:
            raise ReviewError(f"{session_id}/{frame_id}: {key} drift")
    return raw, candidate, m_remove, m_write, m_flow, unknown, source


def decode_video(path: Path, expected_count: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ReviewError(f"cannot open encoded video: {path}")
    count = 0
    shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3:
                raise ReviewError(f"invalid decoded frame {count}: {path}")
            current = (int(frame.shape[1]), int(frame.shape[0]))
            if shape is None:
                shape = current
            elif current != shape:
                raise ReviewError(f"decoded shape changes in {path}")
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if count != expected_count:
        raise ReviewError(f"{path}: decoded {count}, expected {expected_count}")
    if shape != OUTPUT_SIZE:
        raise ReviewError(f"{path}: decoded size {shape}, expected {OUTPUT_SIZE}")
    if abs(fps - FPS) > 0.01:
        raise ReviewError(f"{path}: decoded FPS {fps}, expected {FPS}")
    return {"decoded_frames": count, "decoded_fps": fps, "decoded_size": list(shape)}


def render_session(summary: Mapping[str, Any], staged_root: Path, final_root: Path, expected_count: int,
                   expected_shape: tuple[int, int]) -> dict[str, Any]:
    session_id = str(summary.get("session_id"))
    task = str(summary.get("task"))
    session_result_path = checked_ref(summary.get("result", {}), f"{session_id}/RESULT")
    session_result = load_json(session_result_path)
    if session_result.get("session_id") != session_id or session_result.get("task") != task:
        raise ReviewError(f"{session_id}: session result identity drift")
    if session_result.get("clean_terminal") is not False:
        raise ReviewError(f"{session_id}: renderer accepts only non-terminal Clean preparation")
    if session_result.get("fresh_inpainting_executed") is not False:
        raise ReviewError(f"{session_id}: fresh inpainting must not have executed")
    if int(session_result.get("materialized_changed_pixels", -1)) != 0:
        raise ReviewError(f"{session_id}: materialized changed pixels must be zero")
    manifest_path = checked_ref(summary.get("frame_manifest", {}), f"{session_id}/FRAME_MANIFEST")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != FRAME_SCHEMA:
        raise ReviewError(f"{session_id}: frame manifest schema drift")
    if manifest.get("session_id") != session_id or manifest.get("task") != task:
        raise ReviewError(f"{session_id}: frame manifest identity drift")
    if int(manifest.get("frame_count", -1)) != expected_count:
        raise ReviewError(f"{session_id}: expected {expected_count} frames")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_count:
        raise ReviewError(f"{session_id}: complete ordered frame list required")
    validate_semantics(manifest, session_id, task)

    video_name = f"{session_id}_D1_CLEAN_PREP_OFFLINE_REVIEW.mp4"
    video_path = staged_root / video_name
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, OUTPUT_SIZE
    )
    if not writer.isOpened():
        raise ReviewError(f"cannot open MP4 writer: {video_path}")
    total = {"m_remove_pixels": 0, "m_write_pixels": 0, "m_flow_pixels": 0,
             "unknown_pixels": 0, "protected_pixels": 0}
    try:
        for frame_id, row in enumerate(frames):
            if not isinstance(row, Mapping):
                raise ReviewError(f"{session_id}/{frame_id}: object row required")
            arrays = validate_frame(row, session_id, task, frame_id, expected_shape)
            raw, candidate, m_remove, m_write, m_flow, unknown, source = arrays
            output = render_frame(
                raw, candidate, m_remove, m_write, m_flow, unknown, source,
                session_id, task, frame_id, expected_count,
            )
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            protected = np.isin(source, list(OBJECT_SOURCE_CODES))
            total["m_remove_pixels"] += int(m_remove.sum())
            total["m_write_pixels"] += int(m_write.sum())
            total["m_flow_pixels"] += int(m_flow.sum())
            total["unknown_pixels"] += int(unknown.sum())
            total["protected_pixels"] += int(protected.sum())
    finally:
        writer.release()
    for label, recorded in (("batch summary", summary.get("totals")),
                            ("session result", session_result.get("totals"))):
        if not isinstance(recorded, Mapping) or any(int(recorded.get(key, -1)) != value
                                                    for key, value in total.items()):
            raise ReviewError(f"{session_id}: recomputed totals drift in {label}")
    decoded = decode_video(video_path, expected_count)
    return {
        "session_id": session_id,
        "task": task,
        "frame_count": expected_count,
        "video_name": video_name,
        "video": projected_ref(video_path, final_root / video_name),
        "session_result": file_ref(session_result_path),
        "frame_manifest": file_ref(manifest_path),
        "totals_recomputed": total,
        "candidate_decoded_byte_equal_raw_all_frames": True,
        "clean_materialized": False,
        "offline_review_only": True,
        **decoded,
    }


def render_all(prepared_root: Path, output_root: Path,
               expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
               expected_shape: tuple[int, int] = EXPECTED_INPUT_SHAPE) -> dict[str, Any]:
    prepared_root = prepared_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ReviewError(f"fresh output root required: {output_root}")
    result_path = prepared_root / "RESULT.json"
    result = load_json(result_path)
    if result.get("schema_version") != PRODUCER_SCHEMA:
        raise ReviewError("D1_CLEAN_PREP producer schema drift")
    if result.get("execution_mode") != "CPU_PREPARE_ONLY_NO_MODEL_NO_GPU":
        raise ReviewError("renderer only accepts CPU_PREPARE_ONLY_NO_MODEL_NO_GPU")
    if result.get("gpu_used") is not False or result.get("model_execution_performed") is not False:
        raise ReviewError("renderer refuses materialized/model-executed input")
    if result.get("clean_terminal") is not False or result.get("training_eligible") is not False:
        raise ReviewError("renderer requires non-terminal, non-training D1 preparation")
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise ReviewError("producer session rows required")
    by_id = {row.get("session_id"): row for row in sessions if isinstance(row, Mapping)}
    if set(by_id) != set(expected_counts):
        raise ReviewError(f"fixed session set drift: {sorted(by_id)}")
    for session_id in expected_counts:
        if by_id[session_id].get("task") != EXPECTED_TASKS[session_id]:
            raise ReviewError(f"{session_id}: task drift")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staged.mkdir()
    try:
        rows = [
            render_session(
                by_id[session_id], staged, output_root,
                expected_counts[session_id], expected_shape,
            )
            for session_id in expected_counts
        ]
        receipt = {
            "schema_version": SCHEMA,
            "created_at": now(),
            "status": "PASSED_FULL_DECODE_OFFLINE_REVIEW",
            "producer_result": file_ref(result_path),
            "fixed_sessions": list(expected_counts),
            "fps": FPS,
            "sessions": rows,
            "hard_gates": {
                "all_referenced_artifact_sha_exact": "PASS",
                "all_candidate_decoded_rgb_equals_raw": "PASS",
                "M_remove_subset_M_write_subset_M_flow": "PASS",
                "UNKNOWN_equals_M_write": "PASS",
                "Task_Object_protection_disjoint_from_M_write": "PASS",
                "Poker_no_Hand_and_identity_UNKNOWN": "PASS",
                "Chips_right_Hand_only_left_UNKNOWN_three_slots": "PASS",
                "full_MP4_decode_and_frame_count": "PASS",
            },
            "clean_materialized": False,
            "training_eligible": False,
            "control_ground_truth": False,
            "claim_limit": "CPU offline visualization of PREPARED_V1 only; no Clean quality, inpainting, identity, contact, training, control, or deployment authority.",
        }
        write_json(staged / "D1_CLEAN_PREP_VISUAL_REVIEW_RECEIPT.json", receipt)
        os.replace(staged, output_root)
        return receipt
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = render_all(args.prepared_root, args.output_root)
    except (ReviewError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_FAIL_CLOSED", "error": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
