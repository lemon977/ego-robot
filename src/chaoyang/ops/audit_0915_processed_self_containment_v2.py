#!/usr/bin/env python3
"""Audit the 0915 processed publication with its released tactile-v2 schema.

Only files below the published processed root are opened.  Per-frame JSON is
read through the RGB/tactile allowlist; preserved hand/controller/PICO fields
are neither inspected nor used as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from chaoyang.ops import audit_0915_processed_self_containment_v1 as shared


SCHEMA_VERSION = "processed-session-self-containment-v2"
TACTILE_SCHEMA = "tactile-acquisition-aligned-frame-v2"


def _check_vector(row: dict[str, Any], key: str, length: int, *, context: str) -> None:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise shared.SelfContainmentError(f"{context}: {key} shape mismatch")


def _check_tactile(value: Any, *, session: str, frame: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(shared.TACTILE_SIDES):
        raise shared.SelfContainmentError(
            f"{session} frame {frame}: tactile sides are not exactly left/right"
        )
    valid_sides = 0
    max_abs_offset_ms = 0.0
    for side in shared.TACTILE_SIDES:
        context = f"{session} frame {frame}: {side}"
        row = value[side]
        if not isinstance(row, dict) or row.get("schema_version") != TACTILE_SCHEMA:
            raise shared.SelfContainmentError(f"{context} tactile schema mismatch")
        for key in ("wire_values_369", "wire_active_mask_369", "wire_valid_mask_369"):
            _check_vector(row, key, 369, context=context)
        for key in ("finger_grid_5x4x8", "active_mask_5x4x8", "valid_mask_5x4x8"):
            if not shared._shape_5x4x8(row.get(key)):
                raise shared.SelfContainmentError(f"{context} {key} shape mismatch")
        offline_valid = row.get("offline_source_valid")
        if not isinstance(offline_valid, bool):
            raise shared.SelfContainmentError(f"{context} offline validity is not bool")
        if row.get("materialized_view") != "nearest_host_qpc":
            raise shared.SelfContainmentError(f"{context} tactile materialized view mismatch")
        if row.get("offline_payload") != "source/raw/tactile.jsonl":
            raise shared.SelfContainmentError(f"{context} tactile payload provenance mismatch")
        if offline_valid:
            valid_sides += 1
            offset_value = row.get("offline_source_offset_ms")
            if not isinstance(offset_value, (int, float)):
                raise shared.SelfContainmentError(f"{context} missing offline offset")
            offset = abs(float(offset_value))
            max_abs_offset_ms = max(max_abs_offset_ms, offset)
            if offset > 40.0:
                raise shared.SelfContainmentError(
                    f"{context} tactile offset {offset}ms"
                )
    return {"valid_sides": valid_sides, "max_abs_offset_ms": max_abs_offset_ms}


def audit_session(task: str, session: Path) -> dict[str, Any]:
    conversion_path = session / "CONVERSION_RESULT.json"
    camera_path = session / "camera_params.json"
    stereo_files = sorted(
        (session / "source_stereo").glob("CameraRecord_*_stereo.mp4")
    )
    if not conversion_path.is_file() or not camera_path.is_file() or len(stereo_files) != 1:
        raise shared.SelfContainmentError(f"{session.name}: core processed inputs missing")

    access_digest = hashlib.sha256()
    opened_counts = {"conversion": 0, "camera": 0, "stereo": 0, "training_data": 0}

    def read_json(path: Path, kind: str) -> dict[str, Any]:
        payload = path.read_bytes()
        relative = str(path.relative_to(session))
        digest = hashlib.sha256(payload).hexdigest()
        access_digest.update(relative.encode("utf-8") + b"\0")
        access_digest.update(str(len(payload)).encode("ascii") + b"\0")
        access_digest.update(digest.encode("ascii") + b"\n")
        opened_counts[kind] += 1
        return json.loads(payload)

    conversion = read_json(conversion_path, "conversion")
    camera = read_json(camera_path, "camera")
    if camera.get("left", {}).get("sourceIndex") != 1:
        raise shared.SelfContainmentError(f"{session.name}: physical left sourceIndex != 1")
    if camera.get("right", {}).get("sourceIndex") != 0:
        raise shared.SelfContainmentError(f"{session.name}: physical right sourceIndex != 0")
    for side in shared.TACTILE_SIDES:
        if camera.get(side, {}).get("distortion", {}).get("model") != "equiDis62":
            raise shared.SelfContainmentError(
                f"{session.name}: {side} distortion is not equiDis62"
            )

    frames = sorted((session / "preprocess/all_data").glob("*/training_data.json"))
    expected = conversion.get("validation", {}).get("frame_count")
    if not isinstance(expected, int) or expected <= 0:
        raise shared.SelfContainmentError(f"{session.name}: missing conversion frame count")
    if len(frames) != expected:
        raise shared.SelfContainmentError(
            f"{session.name}: training rows {len(frames)} != {expected}"
        )
    stereo = shared.video_identity(stereo_files[0])
    stereo_sha = shared.sha256(stereo_files[0])
    access_digest.update(
        str(stereo_files[0].relative_to(session)).encode("utf-8") + b"\0"
        + str(stereo_files[0].stat().st_size).encode("ascii") + b"\0"
        + stereo_sha.encode("ascii") + b"\n"
    )
    opened_counts["stereo"] += 1
    if (stereo["width"], stereo["height"]) != (4096, 1536):
        raise shared.SelfContainmentError(
            f"{session.name}: SBS geometry {(stereo['width'], stereo['height'])}"
        )
    if stereo["frames"] < expected:
        raise shared.SelfContainmentError(
            f"{session.name}: SBS frames {stereo['frames']} < {expected}"
        )

    tactile_valid_frames = 0
    max_abs_offset_ms = 0.0
    for index, path in enumerate(frames):
        payload = read_json(path, "training_data")
        metadata = payload.get("metadata")
        observation = payload.get("obs")
        tactile = payload.get("entities", {}).get("tactile")
        if not isinstance(metadata, dict) or not isinstance(observation, dict):
            raise shared.SelfContainmentError(
                f"{session.name} frame {index}: RGB metadata missing"
            )
        if metadata.get("idx") != index:
            raise shared.SelfContainmentError(
                f"{session.name} frame {index}: non-contiguous metadata index"
            )
        if not isinstance(metadata.get("c2w"), list) or len(metadata["c2w"]) != 4:
            raise shared.SelfContainmentError(
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
                   "sha256": stereo_sha},
        "camera": {"path": str(camera_path), "sha256": shared.sha256(camera_path),
                   "physical_left_source_index": 1,
                   "physical_right_source_index": 0,
                   "distortion_model": "equiDis62"},
        "tactile": {
            "schema_version": TACTILE_SCHEMA,
            "frames_with_at_least_one_offline_valid_side": tactile_valid_frames,
            "max_abs_offline_offset_ms": max_abs_offset_ms,
            "source": "processed/preprocess/all_data/*/training_data.json:entities.tactile",
            "raw_source_fallback_used": False,
        },
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "access_ledger": {
            "opened_file_counts": opened_counts,
            "opened_file_count": sum(opened_counts.values()),
            "allowed_access_manifest_sha256": access_digest.hexdigest(),
            "denied_sidecar_files_opened": 0,
            "denied_hand_or_controller_fields_consumed": 0,
            "training_json_field_consumption": [
                "/metadata", "/obs", "/entities/tactile",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise shared.SelfContainmentError(f"fresh immutable output required: {output}")

    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sessions = shared.enumerate_sessions(root)
    for index, (task, session) in enumerate(sessions, 1):
        try:
            result = audit_session(task, session)
        except Exception as exc:  # noqa: BLE001
            result = {"task": task, "session_id": session.name,
                      "status": "FAILED", "error": repr(exc)}
            failures.append(result)
        results.append(result)
        if index == 1 or index % 10 == 0 or index == len(sessions):
            print(json.dumps({
                "completed": index, "total": len(sessions),
                "passed": index - len(failures), "failed": len(failures),
            }, sort_keys=True), flush=True)

    frame_count = sum(int(row.get("frame_count", 0)) for row in results)
    status = "PASS" if not failures and frame_count == shared.EXPECTED_FRAMES else "FAILED"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_id": shared.DATASET_ID,
        "dataset_root": str(root),
        "session_count": len(results),
        "frame_count": frame_count,
        "task_counts": shared.EXPECTED_TASK_COUNTS,
        "passed": sum(row["status"] == "PASS" for row in results),
        "failed": len(failures),
        "input_policy": {
            "processed_only": True,
            "raw_source_fallback": "FORBIDDEN",
            "physical_left_rgb_source": "SBS_SOURCE_INDEX_1",
            "allowed_per_frame_json_pointers": [
                "/metadata", "/obs", "/entities/tactile",
            ],
            "denied_json_pointers": list(shared.DENIED_JSON_POINTERS),
            "denied_sidecars": sorted(shared.DENIED_BASENAMES),
            "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        },
        "access_proof": {
            "session_ledgers": len(results),
            "opened_file_count": sum(
                int(row.get("access_ledger", {}).get("opened_file_count", 0))
                for row in results
            ),
            "denied_sidecar_files_opened": sum(
                int(row.get("access_ledger", {}).get("denied_sidecar_files_opened", 0))
                for row in results
            ),
            "denied_hand_or_controller_fields_consumed": sum(
                int(row.get("access_ledger", {}).get("denied_hand_or_controller_fields_consumed", 0))
                for row in results
            ),
            "instrumentation": "EXPLICIT_OPEN_AND_JSON_POINTER_CONSUMPTION_LEDGER",
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "results": results,
        "claim_limit": (
            "Processed self-containment, structural RGB/SBS/calibration and "
            "released tactile-v2 activity only; no PICO26 hand, force, contact "
            "truth, external depth accuracy, or deployment authority."
        ),
    }
    shared.atomic_json(output, receipt)
    print(json.dumps({
        "status": status, "sessions": len(results), "frames": frame_count,
        "failed": len(failures), "output": str(output),
    }, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
