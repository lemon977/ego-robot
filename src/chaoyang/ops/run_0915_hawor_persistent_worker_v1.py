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
        observed = np.asarray(data["observed"], dtype=bool)
        joints_2d = np.asarray(data["joints_2d"], dtype=np.float64)
        joints_3d = np.asarray(data["joints_3d_camera"], dtype=np.float64)
    frame_count = observed.shape[1]
    in_frame: list[float] = []
    bone_cv: list[float | None] = []
    for side in (0, 1):
        selected = observed[side]
        uv = joints_2d[side, selected]
        valid = (
            np.isfinite(uv).all(axis=-1)
            & (uv[..., 0] >= 0) & (uv[..., 0] < 1280)
            & (uv[..., 1] >= 0) & (uv[..., 1] < 960)
        )
        in_frame.append(float(valid.mean()) if valid.size else 0.0)
        cvs: list[float] = []
        xyz = joints_3d[side, selected]
        for start, end in BONES:
            length = np.linalg.norm(xyz[:, end] - xyz[:, start], axis=1)
            mean = float(np.nanmean(length)) if length.size else 0.0
            if mean > 1e-8:
                cvs.append(float(np.nanstd(length) / mean))
        bone_cv.append(float(np.mean(cvs)) if cvs else None)
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
        "mean_bone_length_cv": {"left": bone_cv[0], "right": bone_cv[1]},
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
            both_observed = (
                metrics["observed_frames"]["left"] > 0
                and metrics["observed_frames"]["right"] > 0
            )
            in_frame = min(metrics["joint_in_frame_fraction"].values()) >= 0.90
            status = "PASS_DEVELOPMENT_HAWOR" if both_observed and in_frame else "FAILED_QUALITY_C"
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
