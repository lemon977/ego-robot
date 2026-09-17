#!/usr/bin/env python3
"""CPU-only authoring of temporal Mask role/union review videos.

This module consumes a completed PICO/raw-point Mask canary.  It does not run
segmentation and it does not authorize downstream consumption.  Every source
frame and binary role mask is decoded again, the published removal-union
semantics are checked, and each of the three preregistered contiguous windows
is encoded as a compact three-panel MP4 for human temporal review.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class MaskTemporalReviewError(RuntimeError):
    """Raised when review evidence is incomplete, inconsistent, or corrupt."""


ROLE_NAMES = (
    "left_human",
    "right_human",
    "left_tracker",
    "right_tracker",
    "task_object",
    "fixture",
)
REMOVAL_ROLES = ROLE_NAMES[:4]
PROTECT_ROLES = ROLE_NAMES[4:]

# Values are RGB in the manifest; conversion to BGR happens at draw time.
ROLE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "left_human": (217, 60, 245),
    "right_human": (255, 160, 35),
    "left_tracker": (35, 220, 235),
    "right_tracker": (45, 110, 250),
    "task_object": (45, 255, 70),
    "fixture": (255, 212, 59),
}
REMOVAL_COLOR_RGB = (255, 92, 45)
PROTECT_COLOR_RGB = (45, 255, 70)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaskTemporalReviewError(f"cannot decode JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise MaskTemporalReviewError(f"JSON authority root is not an object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    resolved = path.resolve(strict=True)
    value = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim != 2 or value.shape != shape:
        observed = None if value is None else value.shape
        raise MaskTemporalReviewError(
            f"mask decode/geometry failure: {resolved}; observed={observed}, expected={shape}"
        )
    unique = set(map(int, np.unique(value)))
    if not unique.issubset({0, 255}):
        raise MaskTemporalReviewError(f"mask is not exact 0/255 binary: {resolved}; values={sorted(unique)}")
    return value == 255


def _tint(image: np.ndarray, mask: np.ndarray, color_rgb: Sequence[int], alpha: float = 0.60) -> None:
    color_bgr = np.asarray(tuple(reversed(tuple(map(int, color_rgb)))), dtype=np.float32)
    image[mask] = np.rint((1.0 - alpha) * image[mask] + alpha * color_bgr).astype(np.uint8)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def render_panel(
    raw_bgr: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    task_id: str,
    session_id: str,
    event: str,
    source_frame: int,
    frame_status: str,
    panel_width: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Render RAW | role classes | removal/protect union and return pixel counts."""
    if raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3:
        raise MaskTemporalReviewError("source RGB frame did not decode as three-channel image")
    shape = raw_bgr.shape[:2]
    if set(masks) != {*ROLE_NAMES, "removal_union"}:
        raise MaskTemporalReviewError("role-mask inventory differs from the frozen seven-mask schema")
    if any(mask.shape != shape or mask.dtype != bool for mask in masks.values()):
        raise MaskTemporalReviewError("role mask geometry/dtype drift")

    removal_raw = np.logical_or.reduce([masks[name] for name in REMOVAL_ROLES])
    protect = np.logical_or.reduce([masks[name] for name in PROTECT_ROLES])
    expected_removal = removal_raw & ~protect
    if not np.array_equal(masks["removal_union"], expected_removal):
        mismatch = int(np.count_nonzero(masks["removal_union"] != expected_removal))
        raise MaskTemporalReviewError(f"published removal_union semantic mismatch: {mismatch} pixels")
    overlap = int(np.count_nonzero(masks["removal_union"] & protect))
    if overlap:
        raise MaskTemporalReviewError(f"published removal_union overlaps protected union: {overlap} pixels")

    panel_height = int(round(panel_width * shape[0] / shape[1]))
    if panel_height % 2:
        panel_height += 1
    raw = cv2.resize(raw_bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    roles_native = raw_bgr.copy()
    for name in ROLE_NAMES:
        _tint(roles_native, masks[name], ROLE_COLORS_RGB[name])
    roles = cv2.resize(roles_native, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    union_native = raw_bgr.copy()
    _tint(union_native, masks["removal_union"], REMOVAL_COLOR_RGB)
    _tint(union_native, protect, PROTECT_COLOR_RGB)
    union = cv2.resize(union_native, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

    header_height = 72
    canvas = np.zeros((header_height + panel_height, panel_width * 3, 3), dtype=np.uint8)
    canvas[header_height:, :panel_width] = raw
    canvas[header_height:, panel_width:2 * panel_width] = roles
    canvas[header_height:, 2 * panel_width:] = union
    _put_label(
        canvas,
        f"{task_id} | {session_id} | {event} | source f{source_frame:05d} | {frame_status}",
        (8, 20),
        (255, 255, 255),
    )
    _put_label(canvas, "RAW", (8, 45), (230, 230, 230))
    _put_label(canvas, "ROLE CLASSES", (panel_width + 8, 45), (230, 230, 230))
    _put_label(canvas, "UNION: REMOVE / PROTECT", (2 * panel_width + 8, 45), (230, 230, 230))
    _put_label(canvas, "LH RH LT RT OBJ FIX", (panel_width + 8, 65), (200, 200, 200))
    _put_label(canvas, "ORANGE=REMOVE  GREEN=PROTECT", (2 * panel_width + 8, 65), (200, 200, 200))
    return canvas, {
        **{f"{name}_pixels": int(np.count_nonzero(masks[name])) for name in ROLE_NAMES},
        "removal_union_pixels": int(np.count_nonzero(masks["removal_union"])),
        "protected_union_pixels": int(np.count_nonzero(protect)),
        "removal_protected_overlap_pixels": overlap,
    }


def _start_encoder(path: Path, width: int, height: int, fps: float) -> tuple[subprocess.Popen[bytes], list[str]]:
    command = [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        "1",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE), command


def _full_decode(path: Path, expected_frames: int, expected_shape: tuple[int, int]) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode:
        raise MaskTemporalReviewError(f"ffprobe failed: {probe.stderr[-1000:]}")
    try:
        stream = json.loads(probe.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        raise MaskTemporalReviewError("ffprobe returned malformed stream evidence") from error
    observed = (int(stream["nb_read_frames"]), int(stream["height"]), int(stream["width"]))
    if observed != (expected_frames, *expected_shape):
        raise MaskTemporalReviewError(
            f"encoded video geometry/count drift: observed={observed}, expected={(expected_frames, *expected_shape)}"
        )
    decode = subprocess.run(
        ["/usr/bin/ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if decode.returncode:
        raise MaskTemporalReviewError(f"full video decode failed: {decode.stderr[-1000:]}")
    return {
        "full_decode": True,
        "decoded_frames": expected_frames,
        "width": expected_shape[1],
        "height": expected_shape[0],
        "codec_name": stream["codec_name"],
        "average_frame_rate": stream["avg_frame_rate"],
    }


def author_temporal_reviews(
    *,
    run_root: Path,
    evaluation_plan_path: Path,
    output_root: Path,
    fps: float = 6.0,
    panel_width: int = 480,
) -> dict[str, Any]:
    """Author and audit exactly three 12-frame representative review videos."""
    started = time.perf_counter()
    if output_root.exists():
        raise MaskTemporalReviewError(f"fresh output root required: {output_root}")
    if panel_width < 160 or panel_width % 2 or not 0 < fps <= 60:
        raise MaskTemporalReviewError("panel width must be even and >=160; fps must be in (0,60]")
    run_root = run_root.resolve(strict=True)
    result_path = run_root / "RESULT.json"
    result_before = evidence(result_path)
    plan_before = evidence(evaluation_plan_path)
    result = load_json(result_path)
    plan = load_json(evaluation_plan_path)
    if result.get("schema_version") != "pico-raw-point-geometry-mask-canary-result-v1":
        raise MaskTemporalReviewError("unexpected Mask canary result schema")
    if result.get("automatic_gate_pass") is not True:
        raise MaskTemporalReviewError("temporal review MP4 requires automatic Mask gate PASS")
    for name in ("task_id", "session_id", "split"):
        if result.get(name) != plan.get(name):
            raise MaskTemporalReviewError(f"result/evaluation-plan identity mismatch: {name}")
    plan_windows = plan.get("temporal_windows", [])
    result_windows = result.get("temporal", {}).get("windows", [])
    if len(plan_windows) != 3 or len(result_windows) != 3:
        raise MaskTemporalReviewError("exactly three temporal windows are required")
    all_data = (Path(plan["canonical_session_path"]) / plan["all_data_relative_path"]).resolve(strict=True)

    output_root.mkdir(parents=True)
    windows_output: list[dict[str, Any]] = []
    poster_frames: list[np.ndarray] = []
    for plan_window, result_window in zip(plan_windows, result_windows, strict=True):
        event = str(plan_window["event"])
        frames = list(map(int, plan_window["source_frames"]))
        if len(frames) != 12 or frames != list(range(frames[0], frames[0] + 12)):
            raise MaskTemporalReviewError(f"window is not a 12-frame unit-stride sequence: {event}")
        if result_window.get("event") != event or list(map(int, result_window.get("source_frames", []))) != frames:
            raise MaskTemporalReviewError(f"result/evaluation-plan temporal mapping mismatch: {event}")
        if result_window.get("status") != "PASS":
            raise MaskTemporalReviewError(f"window did not pass automatic temporal gate: {event}")
        rows = result_window.get("frames", [])
        if len(rows) != 12:
            raise MaskTemporalReviewError(f"result does not contain 12 temporal frame rows: {event}")

        partial = output_root / f".{event}.partial.mp4"
        final = output_root / f"{event}_ROLE_CLASSES_AND_UNION.mp4"
        encoder: subprocess.Popen[bytes] | None = None
        command: list[str] = []
        inventory: list[dict[str, Any]] = []
        frame_shape: tuple[int, int] | None = None
        canvas_shape: tuple[int, int] | None = None
        try:
            for slot, (frame, row) in enumerate(zip(frames, rows, strict=True)):
                if int(row.get("slot", -1)) != slot or int(row.get("source_frame", -1)) != frame:
                    raise MaskTemporalReviewError(f"temporal result row mapping mismatch: {event} slot {slot}")
                source = all_data / f"{frame:05d}" / "rgb.png"
                raw = cv2.imread(str(source.resolve(strict=True)), cv2.IMREAD_COLOR)
                if raw is None:
                    raise MaskTemporalReviewError(f"source RGB full decode failed: {source}")
                if frame_shape is None:
                    frame_shape = raw.shape[:2]
                elif raw.shape[:2] != frame_shape:
                    raise MaskTemporalReviewError(f"source frame geometry drift inside window: {event}")
                mask_evidence: dict[str, Any] = {}
                masks: dict[str, np.ndarray] = {}
                for role in (*ROLE_NAMES, "removal_union"):
                    mask_path = run_root / "temporal" / event / role / f"{frame:05d}.png"
                    masks[role] = _binary_mask(mask_path, frame_shape)
                    mask_evidence[role] = evidence(mask_path)
                panel, counts = render_panel(
                    raw,
                    masks,
                    task_id=str(plan["task_id"]),
                    session_id=str(plan["session_id"]),
                    event=event,
                    source_frame=frame,
                    frame_status=str(row.get("status", "UNKNOWN")),
                    panel_width=panel_width,
                )
                if canvas_shape is None:
                    canvas_shape = panel.shape[:2]
                    encoder, command = _start_encoder(partial, panel.shape[1], panel.shape[0], fps)
                elif panel.shape[:2] != canvas_shape:
                    raise MaskTemporalReviewError(f"rendered panel geometry drift inside window: {event}")
                assert encoder is not None and encoder.stdin is not None
                encoder.stdin.write(panel.tobytes())
                if slot == 5:
                    poster_frames.append(panel.copy())
                inventory.append({
                    "slot": slot,
                    "source_frame": frame,
                    "source_rgb": evidence(source),
                    "role_masks": mask_evidence,
                    "pixel_counts": counts,
                })
            assert encoder is not None and encoder.stdin is not None and canvas_shape is not None
            encoder.stdin.close()
            stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
            return_code = encoder.wait()
            if return_code:
                raise MaskTemporalReviewError(f"ffmpeg encode failed for {event}: {stderr[-1000:]}")
            os.replace(partial, final)
        except BaseException:
            if encoder is not None and encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            raise
        decode = _full_decode(final, 12, canvas_shape)
        windows_output.append({
            "event": event,
            "source_frames": frames,
            "result_status": result_window["status"],
            "input_inventory": inventory,
            "encoder_command": command,
            "decode_audit": decode,
            "video": evidence(final),
        })

    if len(poster_frames) != 3:
        raise MaskTemporalReviewError("one poster frame per temporal window was not produced")
    poster = np.vstack(poster_frames)
    poster_path = output_root / "TEMPORAL_ROLE_CLASSES_AND_UNION_POSTER_3.png"
    if not cv2.imwrite(str(poster_path), poster):
        raise MaskTemporalReviewError("failed to write temporal review poster")
    if cv2.imread(str(poster_path), cv2.IMREAD_COLOR) is None:
        raise MaskTemporalReviewError("temporal review poster failed full decode")

    if evidence(result_path) != result_before or evidence(evaluation_plan_path) != plan_before:
        raise MaskTemporalReviewError("result or evaluation-plan authority changed during authoring")
    manifest = {
        "schema_version": "mask-temporal-role-union-review-v1",
        "status": "PASS_FULL_DECODE_REVIEW_ONLY",
        "task_id": plan["task_id"],
        "session_id": plan["session_id"],
        "split": plan["split"],
        "automatic_mask_gate_pass": True,
        "human_temporal_review_required": True,
        "consumption_authorized": False,
        "claim_limit": "THREE_PREREGISTERED_12_FRAME_WINDOWS_ROLE_AND_UNION_VISUAL_REVIEW_ONLY",
        "render_contract": {
            "panels": ["raw", "role_classes", "removal_and_protected_union"],
            "role_order": list(ROLE_NAMES),
            "role_colors_rgb": {name: list(color) for name, color in ROLE_COLORS_RGB.items()},
            "removal_color_rgb": list(REMOVAL_COLOR_RGB),
            "protected_color_rgb": list(PROTECT_COLOR_RGB),
            "fps": fps,
            "panel_width": panel_width,
            "windows": 3,
            "frames_per_window": 12,
            "gpu_calls": 0,
        },
        "authority": {
            "canary_result": result_before,
            "evaluation_plan": plan_before,
            "producer": evidence(Path(__file__)),
        },
        "windows": windows_output,
        "poster": evidence(poster_path),
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path = output_root / "RESULT.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "result": evidence(manifest_path)}

