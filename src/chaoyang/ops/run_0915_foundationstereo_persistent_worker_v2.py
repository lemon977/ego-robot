#!/usr/bin/env python3
"""Persistent FoundationStereo batch for 0915 same-session processed SBS.

The model is loaded once.  Every session estimates and validates its own
rectification, preserves physical-left reference pixels, and publishes only
visible-surface optical-Z with no external millimetre-accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.ops import run_0915_leftmono_foundation_depth_v1 as session_v1
from chaoyang.ops import run_exact78_foundationstereo_corrected_depth_worker as fs_worker


EXPECTED_SESSIONS = 220
EXPECTED_FRAMES = 58_686
CHECKPOINT = session_v1.CHECKPOINT
CHECKPOINT_SHA256 = session_v1.CHECKPOINT_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str((published_path or path).resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def physical_left_depth(
    disparity_flipped: np.ndarray, *, focal_px: float, baseline_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disparity = cv2.flip(np.asarray(disparity_flipped, np.float32), 1)
    x = np.arange(disparity.shape[1], dtype=np.float32)[None]
    valid = (
        np.isfinite(disparity) & (disparity > 0.25)
        & ((x + disparity) < disparity.shape[1])
    )
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[valid] = np.float32(focal_px * baseline_m) / disparity[valid]
    valid &= (depth >= 0.10) & (depth <= 3.0)
    depth[~valid] = np.nan
    return disparity, depth, valid


def quality_gate(
    rows: list[dict[str, Any]], *, frame_count: int,
    rectification_passed: bool,
) -> dict[str, Any]:
    valid = np.asarray([row["valid_pixels"] for row in rows], np.int64)
    coverage = np.asarray([row["valid_fraction"] for row in rows], np.float64)
    medians = np.asarray([
        np.nan if row["depth_p50_m"] is None else row["depth_p50_m"]
        for row in rows
    ], np.float64)
    adjacent = np.abs(np.diff(medians))
    adjacent = adjacent[np.isfinite(adjacent)]
    return {
        "rectification_passed": rectification_passed,
        "frame_count_exact": len(rows) == frame_count,
        "median_valid_pixels": int(np.median(valid)) if valid.size else 0,
        "frames_at_least_5000_valid_pixels": int(np.count_nonzero(valid >= 5_000)),
        "median_valid_fraction": float(np.median(coverage)) if coverage.size else 0.0,
        "temporal_depth_p50_step_p90_m": (
            float(np.percentile(adjacent, 90)) if adjacent.size else None
        ),
        "passed": bool(
            rectification_passed
            and len(rows) == frame_count
            and valid.size
            and np.median(valid) >= 5_000
            and np.count_nonzero(valid >= 5_000) >= int(0.90 * frame_count)
        ),
    }


def process_session(
    row: dict[str, Any], *, prepared_root: Path, output_root: Path,
    model: Any, pico: Any,
) -> dict[str, Any]:
    task, session_id = row["task"], row["session_id"]
    target = output_root / "sessions" / task / session_id
    prior = target / "RESULT.json"
    if prior.is_file():
        value = json.loads(prior.read_text(encoding="utf-8"))
        return {**value, "resume_status": "RESUMED"}
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"unreceipted depth target exists: {target}")
    staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    started = time.time()
    try:
        source = Path(row["source"]["session"]).resolve(strict=True)
        prepared = prepared_root / "sessions" / task / session_id
        leftmono = prepared / row["output"]["video_relative"]
        if not leftmono.is_file() or sha256(leftmono) != row["output"]["video"]["sha256"]:
            raise RuntimeError("prepared physical-left video closure mismatch")
        camera = source / "camera_params.json"
        stereo_candidates = sorted(
            (source / "source_stereo").glob("CameraRecord_*_stereo.mp4")
        )
        if len(stereo_candidates) != 1:
            raise RuntimeError("same-session SBS identity is not singular")
        stereo = stereo_candidates[0]
        frame_count = int(row["frame_count"])
        if frame_count < 40:
            raise RuntimeError("held-out rectification requires at least 40 frames")
        eyes, eye_width, _eye_height, baseline = pico.load_camera_params(camera)
        source_indices = pico.load_source_indices(camera)
        if source_indices != (1, 0):
            raise RuntimeError("physical left/right source indices drifted")
        maps, rectified_k, calibration = session_v1.calibrate(
            stereo, pico, eyes, eye_width, source_indices, frame_count,
        )
        atomic_json(staging / "CALIBRATION.json", calibration)

        primary_k = pico.virtual_intrinsics(1280, 960, 90.0)
        left_rotation = np.asarray(
            calibration["rectification_rotation_left"], np.float64,
        )
        full_to_primary = primary_k @ left_rotation.T @ np.linalg.inv(rectified_k)
        full_to_primary /= full_to_primary[2, 2]
        half_to_full = np.asarray([
            [2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0],
        ])
        depth_to_primary = full_to_primary @ half_to_full
        depth_to_primary /= depth_to_primary[2, 2]
        depth_k = rectified_k.copy()
        depth_k[:2] *= 0.5
        registration = {
            "schema_version": "0915-depth-registration-v2",
            "H_depth_pixel_to_primary_physical_left_pixel": depth_to_primary.tolist(),
            "H_primary_physical_left_pixel_to_depth_pixel": np.linalg.inv(depth_to_primary).tolist(),
            "T_rectified_left_camera_to_primary_left_camera": np.block([
                [left_rotation.T, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]).tolist(),
            "primary_intrinsics": primary_k.tolist(),
            "depth_intrinsics": depth_k.tolist(),
            "coordinate_domain": "PHYSICAL_LEFT_SOURCE_INDEX_1",
        }
        atomic_json(staging / "REGISTRATION.json", registration)
        frame_root = staging / "frames"
        frame_root.mkdir()
        stereo_capture = cv2.VideoCapture(str(stereo))
        frame_rows: list[dict[str, Any]] = []
        try:
            for frame_index in range(frame_count):
                ok, sbs = stereo_capture.read()
                if not ok:
                    raise RuntimeError(f"SBS decode ended at frame {frame_index}")
                left, right = pico.remap_pair(
                    sbs, eye_width, maps, source_indices=source_indices,
                )
                disparity_flipped, _unused_depth, _unused_valid = model.infer(
                    cv2.flip(left, 1), cv2.flip(right, 1), baseline,
                )
                _disparity, depth, valid = physical_left_depth(
                    disparity_flipped, focal_px=float(depth_k[0, 0]),
                    baseline_m=float(baseline),
                )
                frame_path = frame_root / f"{frame_index:06d}.npz"
                atomic_npz(
                    frame_path,
                    frame_id=np.asarray(frame_index, np.int32),
                    depth_m=depth,
                    valid=valid,
                    scaled_rectified_intrinsics=depth_k,
                    depth_reference=np.asarray("PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"),
                )
                values = depth[valid]
                frame_rows.append({
                    "frame": frame_index,
                    "artifact": ref(
                        frame_path,
                        published_path=(target / "frames" / frame_path.name),
                    ),
                    "valid_pixels": int(valid.sum()),
                    "valid_fraction": float(valid.mean()),
                    "depth_p50_m": float(np.median(values)) if values.size else None,
                })
        finally:
            stereo_capture.release()
        gate = quality_gate(
            frame_rows, frame_count=frame_count,
            rectification_passed=bool(calibration["quality"]["passed"]),
        )
        result = {
            "schema_version": "0915-foundationstereo-session-v2",
            "status": "PASS" if gate["passed"] else "REJECTED_QUALITY",
            "task": task,
            "session_id": session_id,
            "frame_count": frame_count,
            "physical_eye_source_indices": {"left": 1, "right": 0},
            "depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z",
            "stored_frame_arrays": [
                "frame_id", "depth_m", "valid", "scaled_rectified_intrinsics",
                "depth_reference",
            ],
            "disparity_storage": (
                "NOT_DUPLICATED; valid disparity is exactly focal_px*baseline_m/depth_m"
            ),
            "inference_orientation": "FLIP_BOTH_EYES_THEN_FLIP_RESULT_BACK_NO_EYE_SWAP",
            "quality": gate,
            "rectification_quality": calibration["quality"],
            "inputs": {
                "camera_params": ref(camera),
                "source_stereo": ref(stereo),
                "primary_leftmono": ref(leftmono),
                "checkpoint": ref(CHECKPOINT),
            },
            "artifacts": {
                "calibration": ref(
                    staging / "CALIBRATION.json",
                    published_path=target / "CALIBRATION.json",
                ),
                "registration": ref(
                    staging / "REGISTRATION.json",
                    published_path=target / "REGISTRATION.json",
                ),
            },
            "frames": frame_rows,
            "wall_seconds": time.time() - started,
            "claim_limit": (
                "Same-session visible-surface optical-Z internal consistency "
                "only; no external millimetre accuracy, hidden geometry, "
                "contact truth, force, or deployment authority."
            ),
        }
        atomic_json(staging / "RESULT.json", result)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return json.loads((target / "RESULT.json").read_text(encoding="utf-8"))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.prepared_manifest.resolve(strict=True)
    prepared_root = manifest_path.parent
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("FoundationStereo checkpoint SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS"
            or manifest.get("session_count") != EXPECTED_SESSIONS
            or manifest.get("frame_count") != EXPECTED_FRAMES):
        raise RuntimeError("prepared 0915 cohort identity mismatch")
    pico = session_v1.load_module(session_v1.PICO_PATH, "pico_0915_batch_v2")
    model = fs_worker.Model()
    results: list[dict[str, Any]] = []
    started = time.time()
    for row in manifest["results"]:
        try:
            result = process_session(
                row, prepared_root=prepared_root, output_root=output,
                model=model, pico=pico,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "schema_version": "0915-foundationstereo-session-v2",
                "status": "FAILED_RUNTIME", "task": row["task"],
                "session_id": row["session_id"], "error": repr(exc),
            }
        results.append(result)
        atomic_json(output / "STATE.json", {
            "schema_version": "0915-foundationstereo-batch-state-v2",
            "state": "RUNNING", "completed": len(results),
            "session_count": len(manifest["results"]),
            "passed": sum(item["status"] == "PASS" for item in results),
            "rejected_quality": sum(item["status"] == "REJECTED_QUALITY" for item in results),
            "failed_runtime": sum(item["status"] == "FAILED_RUNTIME" for item in results),
            "model_load_count": model.model_load_count,
            "model_inference_count": model.inference_count,
            "results": results,
        })
        print(json.dumps({
            "completed": len(results), "total": len(manifest["results"]),
            "last": f"{row['task']}/{row['session_id']}",
            "status": result["status"],
        }, sort_keys=True), flush=True)
    summary = {
        "schema_version": "0915-foundationstereo-batch-v2",
        "status": "COMPLETED_ALL_TERMINAL",
        "session_count": len(results),
        "frame_count": sum(int(item.get("frame_count", 0)) for item in results),
        "passed": sum(item["status"] == "PASS" for item in results),
        "rejected_quality": sum(item["status"] == "REJECTED_QUALITY" for item in results),
        "failed_runtime": sum(item["status"] == "FAILED_RUNTIME" for item in results),
        "model_load_count": model.model_load_count,
        "model_inference_count": model.inference_count,
        "wall_seconds": time.time() - started,
        "results": results,
        "claim_limit": "0915 internal stereo optical-Z only; no external metric-accuracy authority.",
    }
    atomic_json(output / "BATCH_RESULT.json", summary)
    return 0 if summary["failed_runtime"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
