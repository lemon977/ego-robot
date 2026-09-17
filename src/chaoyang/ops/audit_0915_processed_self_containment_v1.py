#!/usr/bin/env python3
"""Audit the published 0915 processed dataset as a self-contained input.

This operation deliberately treats the published processed tree as the only
data authority.  It never opens the source paths recorded by conversion
receipts, controller sidecars, trackingData, or the PICO preprocessing
manifest.  Per-frame JSON is consumed through an explicit field allowlist:
RGB metadata and ``entities.tactile`` only; ``entities.hands`` is ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid


SCHEMA_VERSION = "processed-session-self-containment-v1"
DATASET_ID = "chips_cards_hands__0915_v3"
EXPECTED_SESSIONS = 220
EXPECTED_FRAMES = 58_686
EXPECTED_TASK_COUNTS = {"playing_cards": 120, "potato_chips": 100}
TACTILE_SIDES = ("left", "right")
DENIED_BASENAMES = {
    "controller_poses",
    "trackingData",
    "pico_humanego_manifest.json",
}
DENIED_JSON_POINTERS = (
    "/entities/hands",
    "/Hand",
    "/Controller",
    "/pico26",
)


class SelfContainmentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def video_identity(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False)
    if completed.returncode:
        raise SelfContainmentError(
            f"ffprobe failed for {path}: {completed.stderr[-1000:]}"
        )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise SelfContainmentError(f"expected one video stream: {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["r_frame_rate"],
        "frames": int(stream["nb_read_frames"]),
    }


def _shape_5x4x8(value: Any) -> bool:
    return (
        isinstance(value, list) and len(value) == 5
        and all(isinstance(finger, list) and len(finger) == 4 for finger in value)
        and all(
            isinstance(row, list) and len(row) == 8
            for finger in value for row in finger
        )
    )


def _check_tactile(value: Any, *, session: str, frame: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(TACTILE_SIDES):
        raise SelfContainmentError(
            f"{session} frame {frame}: tactile sides are not exactly left/right"
        )
    valid_sides = 0
    max_abs_offset_ms = 0.0
    for side in TACTILE_SIDES:
        row = value[side]
        if row.get("schema_version") != "tactile-rgb30-frame-v1":
            raise SelfContainmentError(
                f"{session} frame {frame}: {side} tactile schema mismatch"
            )
        if len(row.get("wire_values_369", [])) != 369:
            raise SelfContainmentError(
                f"{session} frame {frame}: {side} tactile wire shape mismatch"
            )
        for key in ("finger_grid_5x4x8", "active_mask_5x4x8",
                    "valid_mask_5x4x8"):
            if not _shape_5x4x8(row.get(key)):
                raise SelfContainmentError(
                    f"{session} frame {frame}: {side} {key} shape mismatch"
                )
        offline_valid = row.get("offline_source_valid")
        if not isinstance(offline_valid, bool):
            raise SelfContainmentError(
                f"{session} frame {frame}: {side} offline validity is not bool"
            )
        if offline_valid:
            valid_sides += 1
            offset = abs(float(row.get("offline_source_offset_ms")))
            max_abs_offset_ms = max(max_abs_offset_ms, offset)
            if offset > 40.0:
                raise SelfContainmentError(
                    f"{session} frame {frame}: {side} tactile offset {offset}ms"
                )
    return {"valid_sides": valid_sides,
            "max_abs_offset_ms": max_abs_offset_ms}


def enumerate_sessions(root: Path) -> list[tuple[str, Path]]:
    cleaned = root / "cleaned"
    rows: list[tuple[str, Path]] = []
    for task in sorted(EXPECTED_TASK_COUNTS):
        task_root = cleaned / task
        if not task_root.is_dir():
            raise SelfContainmentError(f"missing task root: {task_root}")
        sessions = sorted(path for path in task_root.iterdir() if path.is_dir())
        if len(sessions) != EXPECTED_TASK_COUNTS[task]:
            raise SelfContainmentError(
                f"{task} count {len(sessions)} != {EXPECTED_TASK_COUNTS[task]}"
            )
        rows.extend((task, path) for path in sessions)
    if len(rows) != EXPECTED_SESSIONS:
        raise SelfContainmentError(
            f"session count {len(rows)} != {EXPECTED_SESSIONS}"
        )
    return rows


def audit_session(task: str, session: Path) -> dict[str, Any]:
    conversion_path = session / "CONVERSION_RESULT.json"
    camera_path = session / "camera_params.json"
    stereo_files = sorted(
        (session / "source_stereo").glob("CameraRecord_*_stereo.mp4")
    )
    if not conversion_path.is_file() or not camera_path.is_file() or len(stereo_files) != 1:
        raise SelfContainmentError(f"{session.name}: core processed inputs missing")

    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    if camera.get("left", {}).get("sourceIndex") != 1:
        raise SelfContainmentError(f"{session.name}: physical left sourceIndex != 1")
    if camera.get("right", {}).get("sourceIndex") != 0:
        raise SelfContainmentError(f"{session.name}: physical right sourceIndex != 0")
    for side in TACTILE_SIDES:
        if camera.get(side, {}).get("distortion", {}).get("model") != "equiDis62":
            raise SelfContainmentError(
                f"{session.name}: {side} distortion is not equiDis62"
            )

    frames = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    expected = conversion.get("validation", {}).get("frame_count")
    if not isinstance(expected, int) or expected <= 0:
        raise SelfContainmentError(f"{session.name}: missing conversion frame count")
    if len(frames) != expected:
        raise SelfContainmentError(
            f"{session.name}: training rows {len(frames)} != {expected}"
        )
    stereo = video_identity(stereo_files[0])
    if (stereo["width"], stereo["height"]) != (4096, 1536):
        raise SelfContainmentError(
            f"{session.name}: SBS geometry {(stereo['width'], stereo['height'])}"
        )
    if stereo["frames"] < expected:
        raise SelfContainmentError(
            f"{session.name}: SBS frames {stereo['frames']} < {expected}"
        )

    tactile_valid_frames = 0
    max_abs_offset_ms = 0.0
    for index, path in enumerate(frames):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Intentional allowlist: never inspect entities.hands or any controller
        # field even though legacy compatibility JSON preserves those bytes.
        metadata = payload.get("metadata")
        observation = payload.get("obs")
        tactile = payload.get("entities", {}).get("tactile")
        if not isinstance(metadata, dict) or not isinstance(observation, dict):
            raise SelfContainmentError(
                f"{session.name} frame {index}: RGB metadata missing"
            )
        if metadata.get("idx") != index:
            raise SelfContainmentError(
                f"{session.name} frame {index}: non-contiguous metadata index"
            )
        if not isinstance(metadata.get("c2w"), list) or len(metadata["c2w"]) != 4:
            raise SelfContainmentError(
                f"{session.name} frame {index}: c2w shape mismatch"
            )
        metrics = _check_tactile(tactile, session=session.name, frame=index)
        tactile_valid_frames += int(metrics["valid_sides"] > 0)
        max_abs_offset_ms = max(max_abs_offset_ms, metrics["max_abs_offset_ms"])

    return {
        "task": task,
        "session_id": session.name,
        "status": "PASS",
        "frame_count": expected,
        "stereo": {**stereo, "path": str(stereo_files[0]),
                   "sha256": sha256(stereo_files[0])},
        "camera": {"path": str(camera_path), "sha256": sha256(camera_path),
                   "physical_left_source_index": 1,
                   "physical_right_source_index": 0,
                   "distortion_model": "equiDis62"},
        "tactile": {
            "frames_with_at_least_one_offline_valid_side": tactile_valid_frames,
            "max_abs_offline_offset_ms": max_abs_offset_ms,
            "source": "processed/preprocess/all_data/*/training_data.json:entities.tactile",
            "raw_source_fallback_used": False,
        },
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SelfContainmentError(f"fresh immutable output required: {output}")

    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sessions = enumerate_sessions(root)
    for index, (task, session) in enumerate(sessions, 1):
        try:
            result = audit_session(task, session)
        except Exception as exc:  # noqa: BLE001
            result = {
                "task": task, "session_id": session.name,
                "status": "FAILED", "error": repr(exc),
            }
            failures.append(result)
        results.append(result)
        if index == 1 or index % 10 == 0 or index == len(sessions):
            print(json.dumps({
                "completed": index, "total": len(sessions),
                "passed": index - len(failures), "failed": len(failures),
            }, sort_keys=True), flush=True)

    frame_count = sum(int(row.get("frame_count", 0)) for row in results)
    status = "PASS" if not failures and frame_count == EXPECTED_FRAMES else "FAILED"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_id": DATASET_ID,
        "dataset_root": str(root),
        "session_count": len(results),
        "frame_count": frame_count,
        "task_counts": EXPECTED_TASK_COUNTS,
        "passed": sum(row["status"] == "PASS" for row in results),
        "failed": len(failures),
        "input_policy": {
            "processed_only": True,
            "raw_source_fallback": "FORBIDDEN",
            "physical_left_rgb_source": "SBS_SOURCE_INDEX_1",
            "allowed_per_frame_json_pointers": [
                "/metadata", "/obs", "/entities/tactile",
            ],
            "denied_json_pointers": list(DENIED_JSON_POINTERS),
            "denied_sidecars": sorted(DENIED_BASENAMES),
            "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "results": results,
        "claim_limit": (
            "Processed self-containment, structural RGB/SBS/calibration and raw "
            "tactile activity only; no PICO26 hand, force, contact truth, external "
            "depth accuracy, or deployment authority."
        ),
    }
    atomic_json(output, receipt)
    print(json.dumps({
        "status": status, "sessions": len(results), "frames": frame_count,
        "failed": len(failures), "output": str(output),
    }, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
