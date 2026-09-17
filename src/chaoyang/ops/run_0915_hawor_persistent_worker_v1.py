#!/usr/bin/env python3
"""Run one pinned HaWoR model persistently across prepared 0915 sessions.

This file is executed through ``hawor_python.sh``.  It caches both the HaWoR
model and YOLO detector weights.  The YOLO predictor/tracker is reset before
each session, so no tracker identity or memory crosses a session boundary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HAWOR = ROOT / "vendor/HaWoR"
UPSTREAM_PATH = ROOT / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py"
HAWOR_WEIGHT = ROOT / "assets/models/vendor/hawor/hawor/checkpoints/hawor.ckpt"
DETECTOR_WEIGHT = ROOT / "assets/models/vendor/hawor/external/detector.pt"
HAWOR_SHA = "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2"
DETECTOR_SHA = "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
BONES = tuple((chain[i], chain[i + 1]) for chain in CHAINS for i in range(4))


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def finite_percentile(values: np.ndarray, quantile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, quantile)) if finite.size else None


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


def quality(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        available = set(data.files)
        observed = np.asarray(data["observed"], dtype=bool)
        joints_2d = np.asarray(data["joints_2d"], dtype=np.float64)
        joints_3d = np.asarray(data["joints_3d_camera"], dtype=np.float64)
        world = (np.asarray(data["joints_3d_world"], dtype=np.float64)
                 if "joints_3d_world" in available else None)
        rotations = (np.asarray(data["root_orient_camera"], dtype=np.float64)
                     if "root_orient_camera" in available else None)
        confidence = (np.asarray(data["detector_confidence"], dtype=np.float64)
                      if "detector_confidence" in available else None)
        provenance = (np.asarray(data["provenance"]).astype(str)
                      if "provenance" in available else None)
        fps = float(np.asarray(data["fps"]).item()) if "fps" in available else 30.0
    frame_count = observed.shape[1]
    in_frame: list[float] = []
    side_metrics: dict[str, dict[str, Any]] = {}
    for side in (0, 1):
        side_name = ("left", "right")[side]
        selected = observed[side]
        uv = joints_2d[side, selected]
        valid = (
            np.isfinite(uv).all(axis=-1)
            & (uv[..., 0] >= 0) & (uv[..., 0] < 1280)
            & (uv[..., 1] >= 0) & (uv[..., 1] < 960)
        )
        in_frame.append(float(valid.mean()) if valid.size else 0.0)
        cvs: list[float] = []
        xyz_all = world[side] if world is not None else joints_3d[side]
        xyz = xyz_all[selected]
        for start, end in BONES:
            length = np.linalg.norm(xyz[:, end] - xyz[:, start], axis=1)
            mean = float(np.nanmean(length)) if length.size else 0.0
            if mean > 1e-8:
                cvs.append(float(np.nanstd(length) / mean))
        bone_cv_max = float(max(cvs)) if cvs else None
        positive_depth = (
            float((joints_3d[side, selected, :, 2] > 0).mean())
            if selected.any() else None
        )
        if rotations is not None and selected.any():
            chosen_rotations = rotations[side, selected]
            orthogonality = float(np.max(np.abs(
                np.transpose(chosen_rotations, (0, 2, 1))
                @ chosen_rotations - np.eye(3)
            )))
            determinant_min = float(np.min(np.linalg.det(chosen_rotations)))
        else:
            orthogonality = determinant_min = None
        adjacent = selected[:-1] & selected[1:]
        steps = np.linalg.norm(np.diff(xyz_all[:, 0], axis=0), axis=-1) * 1000.0
        selected_steps = steps[adjacent]
        p99_step = finite_percentile(selected_steps, 99)
        max_step = float(np.max(selected_steps)) if selected_steps.size else None
        chosen_confidence = confidence[side, selected] if confidence is not None else np.asarray([])
        confidence_median = finite_percentile(chosen_confidence, 50)
        confidence_p05 = finite_percentile(chosen_confidence, 5)
        constant_one = bool(chosen_confidence.size and np.allclose(
            chosen_confidence, 1.0, atol=1e-7,
        ))
        provenance_pass = bool(
            provenance is not None
            and set(np.unique(provenance[side])).issubset({"OBSERVED", "MISSING"})
            and np.array_equal(selected, provenance[side] == "OBSERVED")
        )
        finite_gate_values = (
            confidence_median, confidence_p05, positive_depth, bone_cv_max,
            orthogonality, determinant_min,
        )
        finite_common = all(
            value is not None and np.isfinite(float(value))
            for value in finite_gate_values
        )
        numeric_mask_gate = bool(
            finite_common
            and float(selected.mean()) >= 0.95
            and longest_false_run(selected) <= 8
            and float(confidence_median) >= 0.65
            and float(confidence_p05) >= 0.45
            and not constant_one
            and float(positive_depth) >= 0.995
            and float(bone_cv_max) <= 0.08
            and float(orthogonality) <= 1e-4
            and float(determinant_min) > 0
            and provenance_pass
            and in_frame[-1] >= 0.90
        )
        side_metrics[side_name] = {
            "observed_frames": int(selected.sum()),
            "observed_fraction": float(selected.mean()),
            "longest_missing_gap_frames": longest_false_run(selected),
            "joint_in_frame_fraction": in_frame[-1],
            "positive_depth_fraction": positive_depth,
            "bone_length_cv_max": bone_cv_max,
            "root_rotation_orthogonality_max": orthogonality,
            "root_rotation_determinant_min": determinant_min,
            "confidence_median": confidence_median,
            "confidence_p05": confidence_p05,
            "confidence_uninformative_constant_one": constant_one,
            "wrist_step_p99_mm": p99_step,
            "wrist_step_max_mm": max_step,
            "wrist_step_p99_limit_mm_at_fps": 80.0 * 30.0 / fps,
            "wrist_step_max_limit_mm_at_fps": 150.0 * 30.0 / fps,
            "provenance_consistent": provenance_pass,
            "numeric_mask_gate_pass": numeric_mask_gate,
        }
    return {
        "frame_count": frame_count,
        "observed_frames": {
            "left": int(observed[0].sum()),
            "right": int(observed[1].sum()),
            "bilateral": int(np.all(observed, axis=0).sum()),
        },
        "observed_fraction": {
            "left": float(observed[0].mean()),
            "right": float(observed[1].mean()),
        },
        "joint_in_frame_fraction": {"left": in_frame[0], "right": in_frame[1]},
        "sides": side_metrics,
        "numeric_mask_gate_pass": all(
            row["numeric_mask_gate_pass"] for row in side_metrics.values()
        ),
        "identity_authority": "REQUIRES_INDEPENDENT_MASK_REVIEW",
    }


def load_upstream() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("chaoyang_hawor_upstream_runner", UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load HaWoR upstream runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_caches() -> dict[str, Any]:
    if str(HAWOR) not in sys.path:
        sys.path.insert(0, str(HAWOR))
    from scripts.scripts_test_video import hawor_video
    from lib.pipeline import tools

    original_hawor = hawor_video.load_hawor
    original_yolo = tools.YOLO
    cache: dict[str, Any] = {
        "hawor": None, "yolo": None,
        "hawor_load_count": 0, "detector_load_count": 0,
        "tracker_reset_count": 0,
    }

    def cached_hawor(path: str) -> tuple[Any, Any]:
        if cache["hawor"] is None:
            cache["hawor"] = original_hawor(path)
            cache["hawor_load_count"] += 1
        return cache["hawor"]

    def cached_yolo(path: str) -> Any:
        if cache["yolo"] is None:
            cache["yolo"] = original_yolo(path)
            cache["detector_load_count"] += 1
        else:
            # Reset predictor and ByteTrack state without reloading weights.
            cache["yolo"].predictor = None
        cache["tracker_reset_count"] += 1
        return cache["yolo"]

    hawor_video.load_hawor = cached_hawor
    tools.YOLO = cached_yolo
    return cache


def run_upstream(upstream: Any, adapter: Path, output: Path) -> int:
    previous = sys.argv
    try:
        sys.argv = [
            str(UPSTREAM_PATH), "--session-root", str(adapter),
            "--output-root", str(output),
        ]
        return int(upstream.main() or 0)
    finally:
        sys.argv = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.prepared_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if sha256(HAWOR_WEIGHT) != HAWOR_SHA or sha256(DETECTOR_WEIGHT) != DETECTOR_SHA:
        raise RuntimeError("HaWoR/detector weight pin drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("session_count") != 220:
        raise RuntimeError("prepared manifest is not the frozen 220-session cohort")
    upstream = load_upstream()
    cache = install_caches()
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, row in enumerate(manifest["results"], 1):
        task = row["task"]
        session_id = row["session_id"]
        prepared = manifest_path.parent / "sessions" / task / session_id
        adapter = prepared / "adapter_session"
        target = output / "sessions" / task / session_id
        prior = target / "RESULT.json"
        if prior.is_file():
            result = json.loads(prior.read_text(encoding="utf-8"))
            results.append({**result, "resume_status": "RESUMED"})
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"unreceipted HaWoR target exists: {target}")
        staging = target.with_name(
            f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        staging.parent.mkdir(parents=True, exist_ok=True)
        session_started = time.time()
        try:
            rc = run_upstream(upstream, adapter, staging)
            if rc:
                raise RuntimeError(f"upstream HaWoR returned {rc}")
            upstream_result = staging / "RESULT.json"
            if upstream_result.is_file():
                os.replace(upstream_result, staging / "UPSTREAM_RESULT.json")
            npz = staging / "HAWOR_RAW_MANO21.npz"
            if not npz.is_file():
                raise RuntimeError("HaWoR NPZ missing")
            metrics = quality(npz)
            status = (
                "PASS_DEVELOPMENT_HAWOR"
                if metrics["numeric_mask_gate_pass"]
                else "FAILED_QUALITY_C"
            )
            result = {
                "schema_version": "0915-hawor-persistent-session-v1",
                "status": status,
                "task": task,
                "session_id": session_id,
                "frame_count": metrics["frame_count"],
                "metrics": metrics,
                "weights": {
                    "model": {"path": str(HAWOR_WEIGHT), "sha256": HAWOR_SHA},
                    "detector": {"path": str(DETECTOR_WEIGHT), "sha256": DETECTOR_SHA},
                },
                "model_load_policy": "ONE_PROCESS_PERSISTENT_CACHE",
                "detector_tracker_policy": "WEIGHTS_CACHED_TRACKER_RESET_PER_SESSION",
                "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
                "wall_seconds": time.time() - session_started,
                "claim_limit": "Development HaWoR inference on processed physical-left RGB; missing hands are not filled and no anatomical ground truth is implied.",
            }
            atomic_json(staging / "RESULT.json", result)
            os.replace(staging, target)
        except BaseException as exc:  # noqa: BLE001
            failed = output / "failed" / task / session_id
            failed.parent.mkdir(parents=True, exist_ok=True)
            if failed.exists():
                raise
            if staging.exists():
                os.replace(staging, failed)
            atomic_json(failed / "FAILED_RUNTIME.json", {
                "status": "FAILED_RUNTIME", "task": task,
                "session_id": session_id, "error": repr(exc),
            })
            result = {
                "schema_version": "0915-hawor-persistent-session-v1",
                "status": "FAILED_RUNTIME", "task": task,
                "session_id": session_id, "error": repr(exc),
            }
        results.append(result)
        atomic_json(output / "STATE.json", {
            "schema_version": "0915-hawor-persistent-state-v1",
            "state": "RUNNING", "session_count": len(manifest["results"]),
            "completed": len(results),
            "passed": sum(row["status"] == "PASS_DEVELOPMENT_HAWOR" for row in results),
            "quality_c": sum(row["status"] == "FAILED_QUALITY_C" for row in results),
            "failed_runtime": sum(row["status"] == "FAILED_RUNTIME" for row in results),
            "model_load_count": cache["hawor_load_count"],
            "detector_load_count": cache["detector_load_count"],
            "tracker_reset_count": cache["tracker_reset_count"],
            "updated_unix": time.time(), "results": results,
        })
        print(json.dumps({
            "completed": len(results), "total": len(manifest["results"]),
            "last": session_id, "status": result["status"],
            "model_load_count": cache["hawor_load_count"],
            "detector_load_count": cache["detector_load_count"],
        }, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "0915-hawor-persistent-batch-v1",
        "status": "COMPLETED_ALL_TERMINAL",
        "session_count": len(results),
        "frame_count": sum(int(row.get("frame_count", 0)) for row in results),
        "passed": sum(row["status"] == "PASS_DEVELOPMENT_HAWOR" for row in results),
        "quality_c": sum(row["status"] == "FAILED_QUALITY_C" for row in results),
        "failed_runtime": sum(row["status"] == "FAILED_RUNTIME" for row in results),
        "model_load_count": cache["hawor_load_count"],
        "detector_load_count": cache["detector_load_count"],
        "tracker_reset_count": cache["tracker_reset_count"],
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "wall_seconds": time.time() - started,
        "results": results,
    }
    atomic_json(output / "BATCH_RESULT.json", summary)
    return 0 if summary["failed_runtime"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
