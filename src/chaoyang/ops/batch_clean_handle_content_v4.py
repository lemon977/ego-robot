#!/usr/bin/env python3
"""Parallel, resumable coordinator for the unchanged V3 handle converter.

V4 changes scheduling only.  Session conversion, quality gates and output
schema stay owned by ``convert_handle_egodex_v3``/V3.  At most two disjoint
session targets are processed concurrently under one dataset-level lock.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

from chaoyang.ops import batch_clean_handle_content_v3 as v3


SCHEMA_VERSION = "handle-content-gate-dataset-result-v4"
MAX_WORKERS = 2


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed": len(rows),
        "cleaned": sum(row.get("classification") == "CLEANED" for row in rows),
        "rejected": sum(row.get("classification") == "REJECTED" for row in rows),
        "failed": sum(row.get("classification") == "FAILED" for row in rows),
    }


def _run_one(index: int, item: dict[str, str], converter: Path,
             root: Path) -> tuple[int, dict[str, Any]]:
    try:
        result = v3.convert_one(item, converter, root)
    except Exception as exc:  # noqa: BLE001
        result = {
            "task": item["task"], "session_id": item["session_id"],
            "source": item["source"], "classification": "FAILED",
            "status": "FAILED", "error": repr(exc),
            "traceback": traceback.format_exc()[-12000:],
        }
    return index, result


def _ordered_completed(slots: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    return [row for row in slots if row is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-source", action="append", required=True)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")

    converter = Path(v3.__file__).resolve().parent / "convert_handle_egodex_v3.py"
    items, sources = v3.enumerate_items(args.task_source, args.date_tag)
    root = args.target_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock_descriptor, lock_receipt = v3.acquire_lock(root)
    terminal = root / "DATASET_RESULT.json"
    if terminal.is_file():
        prior = json.loads(terminal.read_text(encoding="utf-8"))
        if prior.get("state") == "COMMITTED":
            os.close(lock_descriptor)
            raise v3.BatchError(f"immutable completed target: {root}")

    (root / "cleaned").mkdir(exist_ok=True)
    (root / "rejected").mkdir(exist_ok=True)
    started = time.time()
    state: dict[str, Any] = {
        "schema_version": "handle-content-gate-batch-state-v4",
        "state": "RUNNING", "dataset_id": args.dataset_id,
        "date_tag": args.date_tag, "sources": sources,
        "session_count": len(items), "completed": 0, "cleaned": 0,
        "rejected": 0, "failed": 0, "converter": str(converter),
        "execution_policy": "PARALLEL_2_CPU_LOW_PRIORITY_DISJOINT_SESSION_TARGETS",
        "workers": args.workers, "lock": lock_receipt,
        "started_unix": started, "results": [],
    }
    v3.atomic_json(root / "QUALITY_POLICY.json", {
        "schema_version": "handle-content-quality-policy-v4",
        "conversion_and_quality_semantics": "UNCHANGED_FROM_V3",
        "tactile": "tactile-content-quality-policy-v1",
        "admitted_content_statuses": sorted(v3.ADMITTED),
        "single_inactive_tactile_side": "WARNING_NOT_REJECTION",
        "manus_absence": "DECLARED_ABSENT_NOT_FABRICATED_NOT_REJECTION",
        "source_immutability": "READ_ONLY_SNAPSHOT_VALIDATED_PER_SESSION",
        "embedded_2160x810_video_cam": "REJECTED_STALE_METADATA",
        "camera_authority": "raw/camera_params.json plus decoded 2048x1536 per eye",
        "downstream_pipeline": "NOT_AUTHORIZED_FOR_0916_CLEANING_TASK",
    })
    v3.atomic_json(root / "STATE.json", state)

    slots: list[dict[str, Any] | None] = [None] * len(items)
    futures: dict[Future[tuple[int, dict[str, Any]]], int] = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers,
                                thread_name_prefix="handle-clean-v4") as pool:
            for index, item in enumerate(items):
                future = pool.submit(_run_one, index, item, converter, root)
                futures[future] = index
            completed_count = 0
            for future in as_completed(futures):
                index, result = future.result()
                slots[index] = result
                completed_count += 1
                rows = _ordered_completed(slots)
                state.update({
                    **_counts(rows), "updated_unix": time.time(),
                    "results": rows,
                })
                v3.atomic_json(root / "STATE.json", state)
                print(json.dumps({
                    "phase": "CONVERT", "completed": completed_count,
                    "total": len(items), "cleaned": state["cleaned"],
                    "rejected": state["rejected"], "failed": state["failed"],
                    "last": f"{result['task']}/{result['session_id']}",
                }, sort_keys=True), flush=True)
    finally:
        os.close(lock_descriptor)

    results = _ordered_completed(slots)
    if len(results) != len(items):
        raise v3.BatchError("coordinator exited without one terminal per session")
    counts = _counts(results)
    final_state = "COMMITTED" if counts["failed"] == 0 else "COMPLETED_WITH_FAILURES"
    result = {
        **state, **counts, "results": results,
        "schema_version": SCHEMA_VERSION,
        "state": final_state, "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "claim_limit": (
            "0916 content cleaning only; V3 conversion/quality semantics are "
            "unchanged and no HaWoR, Mask, Depth, Contact or Robot authority is produced."
        ),
    }
    v3.atomic_json(terminal, result)
    state.update({**counts, "state": final_state,
                  "finished_unix": result["finished_unix"], "results": results})
    v3.atomic_json(root / "STATE.json", state)
    print(json.dumps({
        "state": final_state, "session_count": len(items), **counts,
        "target": str(root),
    }, sort_keys=True), flush=True)
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
