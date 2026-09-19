#!/usr/bin/env python3
"""Run the pinned HaWoR bundle persistently over a resize-only W0 manifest."""

from __future__ import annotations

# ``hawor_python.sh`` deliberately clears an ambient PYTHONPATH.  Make this
# repository script self-locating before importing project modules so the
# pinned environment stays activation-free and reproducible.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import torch

from chaoyang.ops import run_0915_hawor_persistent_worker_v1 as legacy


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.prepared_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh HaWoR worker output required: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in pinned HaWoR runtime")
    weight_bundle = legacy.validate_logical_weight_bundle()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("results")
    if (
        manifest.get("schema_version") != "0915-hawor-resize-only-wave0-prepared-manifest-v1"
        or manifest.get("status") != "PASS"
        or manifest.get("session_count") != 4
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise RuntimeError("prepared manifest is not the frozen four-session W0")
    for row in rows:
        domain = row.get("input_domain", {})
        if (
            domain.get("physical_left_source_index") != 1
            or domain.get("operation") != "CROP_THEN_RESIZE_ONLY"
            or domain.get("lens_undistortion") is not False
            or domain.get("remap_applied") is not False
        ):
            raise RuntimeError(f"wrong HaWoR input domain: {row.get('session_id')}")

    upstream = legacy.load_upstream()
    cache = legacy.install_caches()
    results: list[dict[str, Any]] = []
    started = time.time()
    for row in rows:
        task = str(row["task"])
        session_id = str(row["session_id"])
        adapter = Path(str(row["adapter_session"])).resolve(strict=True)
        target = output / "sessions" / task / session_id
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"unreceipted HaWoR target exists: {target}")
        staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        session_started = time.time()
        try:
            upstream_output = staging / "upstream_hawor"
            rc = legacy.run_upstream(upstream, adapter, upstream_output)
            if rc:
                raise RuntimeError(f"upstream HaWoR returned {rc}")
            upstream_npz = upstream_output / "HAWOR_RAW_MANO21.npz"
            if not upstream_npz.is_file():
                raise RuntimeError("HaWoR NPZ missing")
            npz = staging / "HAWOR_RAW_MANO21.npz"
            os.replace(upstream_npz, npz)
            upstream_result = upstream_output / "RESULT.json"
            if upstream_result.is_file():
                os.replace(upstream_result, staging / "UPSTREAM_RESULT.json")
            shutil.rmtree(upstream_output)
            metrics = legacy.quality(npz)
            status = "PASS_DEVELOPMENT_HAWOR" if metrics["numeric_mask_gate_pass"] else "FAILED_QUALITY_C"
            published_npz = target / "HAWOR_RAW_MANO21.npz"
            result = {
                "schema_version": "0915-hawor-resize-only-persistent-session-v2",
                "status": status,
                "task": task,
                "session_id": session_id,
                "frame_count": metrics["frame_count"],
                "metrics": metrics,
                "npz": {
                    "path": str(published_npz.resolve()),
                    "bytes": npz.stat().st_size,
                    "sha256": legacy.sha256(npz),
                },
                "input_domain": row["input_domain"],
                "source_group": row["source_group"],
                "weights": {
                    "model": {"path": str(legacy.HAWOR_WEIGHT), "sha256": legacy.HAWOR_SHA},
                    "detector": {"path": str(legacy.DETECTOR_WEIGHT), "sha256": legacy.DETECTOR_SHA},
                },
                "logical_weight_bundle": {
                    "path": str(legacy.BUNDLE),
                    "sha256": legacy.sha256(legacy.BUNDLE),
                    "bundle_id": weight_bundle["bundle_id"],
                },
                "model_load_policy": "ONE_PROCESS_PERSISTENT_CACHE",
                "detector_tracker_policy": "WEIGHTS_CACHED_TRACKER_RESET_PER_SESSION",
                "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
                "wall_seconds": time.time() - session_started,
                "claim_limit": (
                    "Development HaWoR prediction in the physical-left resize-only domain; "
                    "missing hands remain invalid and no anatomical ground truth is implied."
                ),
            }
            atomic_json(staging / "RESULT.json", result)
            os.replace(staging, target)
        except BaseException as exc:  # noqa: BLE001
            failed = output / "failed" / task / session_id
            failed.parent.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                os.replace(staging, failed)
            else:
                failed.mkdir(parents=True, exist_ok=False)
            atomic_json(failed / "FAILED_RUNTIME.json", {
                "schema_version": "0915-hawor-resize-only-worker-failure-v2",
                "status": "FAILED_RUNTIME",
                "task": task,
                "session_id": session_id,
                "error": repr(exc),
            })
            result = {
                "schema_version": "0915-hawor-resize-only-persistent-session-v2",
                "status": "FAILED_RUNTIME",
                "task": task,
                "session_id": session_id,
                "error": repr(exc),
            }
        results.append(result)
        atomic_json(output / "STATE.json", {
            "schema_version": "0915-hawor-resize-only-persistent-state-v2",
            "state": "RUNNING",
            "session_count": len(rows),
            "completed": len(results),
            "passed": sum(item["status"] == "PASS_DEVELOPMENT_HAWOR" for item in results),
            "quality_c": sum(item["status"] == "FAILED_QUALITY_C" for item in results),
            "failed_runtime": sum(item["status"] == "FAILED_RUNTIME" for item in results),
            "model_load_count": cache["hawor_load_count"],
            "detector_load_count": cache["detector_load_count"],
            "tracker_reset_count": cache["tracker_reset_count"],
            "updated_unix": time.time(),
            "results": results,
        })
        print(json.dumps({
            "completed": len(results),
            "total": len(rows),
            "last": session_id,
            "status": result["status"],
            "model_load_count": cache["hawor_load_count"],
            "detector_load_count": cache["detector_load_count"],
        }, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "0915-hawor-resize-only-persistent-batch-v2",
        "status": "COMPLETED_ALL_TERMINAL",
        "session_count": len(results),
        "frame_count": sum(int(item.get("frame_count", 0)) for item in results),
        "passed": sum(item["status"] == "PASS_DEVELOPMENT_HAWOR" for item in results),
        "quality_c": sum(item["status"] == "FAILED_QUALITY_C" for item in results),
        "failed_runtime": sum(item["status"] == "FAILED_RUNTIME" for item in results),
        "model_load_count": cache["hawor_load_count"],
        "detector_load_count": cache["detector_load_count"],
        "tracker_reset_count": cache["tracker_reset_count"],
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "wall_seconds": time.time() - started,
        "results": results,
    }
    atomic_json(output / "BATCH_RESULT.json", summary)
    return 0 if summary["failed_runtime"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
