#!/usr/bin/env python3
"""Run fixed 106/029 W1-ADOPTION with one pinned persistent HaWoR load."""

from __future__ import annotations

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

from chaoyang.ops import run_0915_hawor_resize_only_persistent_worker_v2 as w0


EXPECTED_IDS = ("play_cards_0915_106", "get_potato_chips_0915_029")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.prepared_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh HaWoR W1-ADOPTION worker output required: {output}")
    output.mkdir(parents=True)
    if not w0.torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in pinned HaWoR runtime")
    weight_bundle = w0.legacy.validate_logical_weight_bundle()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("results")
    if (
        manifest.get("schema_version") != "0915-hawor-resize-only-w1-adoption-prepared-manifest-v1"
        or manifest.get("status") != "PASS"
        or manifest.get("cohort_role") != "W1_ADOPTION"
        or manifest.get("session_count") != 2
        or manifest.get("frame_count") != 404
        or not isinstance(manifest.get("parent_candidate_signature_sha256"), str)
        or len(manifest["parent_candidate_signature_sha256"]) != 64
        or not isinstance(manifest.get("adoption_adapter_or_run_signature_sha256"), str)
        or len(manifest["adoption_adapter_or_run_signature_sha256"]) != 64
        or not isinstance(rows, list)
        or tuple(row.get("session_id") for row in rows) != EXPECTED_IDS
    ):
        raise RuntimeError("prepared manifest is not the signed fixed W1-ADOPTION pair")
    for row in rows:
        domain = row.get("input_domain", {})
        if (
            row.get("cohort_role") != "W1_ADOPTION"
            or row.get("access_state") != "OPENED_AFTER_CANDIDATE_AMENDMENT_V2"
            or domain.get("physical_left_source_index") != 1
            or domain.get("operation") != "CROP_THEN_RESIZE_ONLY"
            or domain.get("lens_undistortion") is not False
            or domain.get("remap_applied") is not False
        ):
            raise RuntimeError(f"wrong W1-ADOPTION input contract: {row.get('session_id')}")

    upstream = w0.legacy.load_upstream()
    cache = w0.legacy.install_caches()
    results: list[dict[str, Any]] = []
    started = time.time()
    for row in rows:
        task = str(row["task"])
        session_id = str(row["session_id"])
        adapter = Path(str(row["adapter_session"])).resolve(strict=True)
        target = output / "sessions" / task / session_id
        staging = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        session_started = time.time()
        try:
            upstream_output = staging / "upstream_hawor"
            rc = w0.legacy.run_upstream(upstream, adapter, upstream_output)
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
            metrics = w0.legacy.quality(npz)
            session_status = "PASS_DEVELOPMENT_HAWOR" if metrics["numeric_mask_gate_pass"] else "FAILED_QUALITY_C"
            per_side_strict = {
                side: (
                    "PASS_DEVELOPMENT_HAWOR_SIDE_STRICT"
                    if value["numeric_mask_gate_pass"]
                    else "FAILED_QUALITY_C_SIDE_STRICT"
                )
                for side, value in metrics["sides"].items()
            }
            published_npz = target / npz.name
            result = {
                "schema_version": "0915-hawor-w1-adoption-persistent-session-v1",
                "status": session_status,
                "session_strict_status": session_status,
                "per_side_strict": per_side_strict,
                "task": task,
                "session_id": session_id,
                "cohort_role": "W1_ADOPTION",
                "frame_count": metrics["frame_count"],
                "metrics": metrics,
                "npz": {
                    "path": str(published_npz.resolve()),
                    "bytes": npz.stat().st_size,
                    "sha256": w0.legacy.sha256(npz),
                },
                "input_domain": row["input_domain"],
                "source_group": row["source_group"],
                "parent_candidate_signature_sha256": manifest["parent_candidate_signature_sha256"],
                "adoption_adapter_or_run_signature_sha256": manifest["adoption_adapter_or_run_signature_sha256"],
                "weights": {
                    "model": {"path": str(w0.legacy.HAWOR_WEIGHT), "sha256": w0.legacy.HAWOR_SHA},
                    "detector": {"path": str(w0.legacy.DETECTOR_WEIGHT), "sha256": w0.legacy.DETECTOR_SHA},
                },
                "logical_weight_bundle": {
                    "path": str(w0.legacy.BUNDLE),
                    "sha256": w0.legacy.sha256(w0.legacy.BUNDLE),
                    "bundle_id": weight_bundle["bundle_id"],
                },
                "model_load_policy": "ONE_PROCESS_PERSISTENT_CACHE",
                "detector_tracker_policy": "WEIGHTS_CACHED_TRACKER_RESET_PER_SESSION",
                "wall_seconds": time.time() - session_started,
                "claim_limit": "W1-ADOPTION HaWoR evidence only; no automatic R0, adoption decision, control, training or deployment authority.",
            }
            w0.atomic_json(staging / "RESULT.json", result)
            os.replace(staging, target)
        except BaseException as exc:  # noqa: BLE001
            failed = output / "failed" / task / session_id
            failed.parent.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                os.replace(staging, failed)
            else:
                failed.mkdir(parents=True, exist_ok=False)
            w0.atomic_json(failed / "FAILED_RUNTIME.json", {
                "schema_version": "0915-hawor-w1-adoption-worker-failure-v1",
                "status": "FAILED_RUNTIME",
                "task": task,
                "session_id": session_id,
                "parent_candidate_signature_sha256": manifest["parent_candidate_signature_sha256"],
                "adoption_adapter_or_run_signature_sha256": manifest["adoption_adapter_or_run_signature_sha256"],
                "error": repr(exc),
            })
            result = {
                "schema_version": "0915-hawor-w1-adoption-persistent-session-v1",
                "status": "FAILED_RUNTIME",
                "task": task,
                "session_id": session_id,
                "parent_candidate_signature_sha256": manifest["parent_candidate_signature_sha256"],
                "adoption_adapter_or_run_signature_sha256": manifest["adoption_adapter_or_run_signature_sha256"],
                "error": repr(exc),
            }
        results.append(result)
        w0.atomic_json(output / "STATE.json", {
            "schema_version": "0915-hawor-w1-adoption-persistent-state-v1",
            "state": "RUNNING",
            "cohort_role": "W1_ADOPTION",
            "parent_candidate_signature_sha256": manifest["parent_candidate_signature_sha256"],
            "adoption_adapter_or_run_signature_sha256": manifest["adoption_adapter_or_run_signature_sha256"],
            "session_count": 2,
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
        print(json.dumps({"completed": len(results), "total": 2, "last": session_id, "status": result["status"]}, sort_keys=True), flush=True)
        gc.collect()
        w0.torch.cuda.empty_cache()

    summary = {
        "schema_version": "0915-hawor-w1-adoption-persistent-batch-v1",
        "status": "COMPLETED_ADOPTION_ALL_TERMINAL",
        "cohort_role": "W1_ADOPTION",
        "parent_candidate_signature_sha256": manifest["parent_candidate_signature_sha256"],
        "adoption_adapter_or_run_signature_sha256": manifest["adoption_adapter_or_run_signature_sha256"],
        "session_count": len(results),
        "frame_count": sum(int(item.get("frame_count", 0)) for item in results),
        "passed": sum(item["status"] == "PASS_DEVELOPMENT_HAWOR" for item in results),
        "quality_c": sum(item["status"] == "FAILED_QUALITY_C" for item in results),
        "failed_runtime": sum(item["status"] == "FAILED_RUNTIME" for item in results),
        "model_load_count": cache["hawor_load_count"],
        "detector_load_count": cache["detector_load_count"],
        "tracker_reset_count": cache["tracker_reset_count"],
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY",
        "wall_seconds": time.time() - started,
        "results": results,
    }
    w0.atomic_json(output / "BATCH_RESULT.json", summary)
    return 0 if summary["failed_runtime"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
