#!/usr/bin/env python3
"""Run the second and final bounded Robot arm method for round-1 HOLD rows.

Only sessions explicitly classified ``HOLD_ARM_ROUND1_ELIGIBLE_ROUND2`` by
the immutable round-1 aggregate are executed.  Round-1 PASS rows are carried
forward without recomputation.  A numeric HOLD produced by this method has
consumed the two-method budget and is a quality-C arm terminal candidate; it
must not be retried with an unregistered third solver.

This wrapper is CPU-only.  It does not run hand retargeting, rendering,
contact inference, action publication, Robot authority, or physical control.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
ROUND2_TOOL = PROJECT / "tools/run_robot_arm_segment_bidirectional_v3.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_ref(value: dict[str, Any], label: str) -> Path:
    if set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: exact ref required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = PROJECT / path
    actual = ref(path)
    if actual["bytes"] != value["bytes"] or actual["sha256"] != value["sha256"]:
        raise ValueError(f"{label}: ref mismatch")
    return Path(actual["path"])


def unique_sessions(rows: list[dict[str, Any]], label: str) -> set[str]:
    sessions = [row.get("session") for row in rows]
    if any(not isinstance(session, str) or not session for session in sessions):
        raise ValueError(f"{label}: every row requires a non-empty session")
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{label}: duplicate session rows")
    return set(sessions)


def classify_placement_status(status: str) -> str:
    if status == "PASS_ARM_METHOD1_PLACEMENT_SELECTED":
        return "CARRY_METHOD1_PASS"
    if status == "HOLD_ARM_METHOD1_ALL_PLACEMENTS_ELIGIBLE_METHOD2":
        return "RUN_METHOD2"
    raise ValueError(f"unexpected method-1 placement status {status}")


def preflight_lineage_allows(
    preflight: dict[str, Any], current_preflight_ref: dict[str, Any], placement_preflight_ref: dict[str, Any]
) -> bool:
    return (
        placement_preflight_ref == current_preflight_ref
        or preflight.get("predecessor_preflight") == placement_preflight_ref
    )


def run_session(row: dict[str, Any], preflight: dict[str, Any], output_root: Path) -> dict[str, Any]:
    task = row["task"]
    session = row["session"]
    action = classify_placement_status(row["status"])
    if action == "CARRY_METHOD1_PASS":
        verify_ref(row["selected_result"], f"{session}:method1_result")
        verify_ref(row["selected_states"], f"{session}:method1_states")
        return {
            "task": task,
            "session": session,
            "status": "PASS_ARM_METHOD1_CARRIED_NO_ROUND2",
            "arm_method_rounds_consumed": 1,
            "selected_backoff_m": row["selected_backoff_m"],
            "method1_result": row["selected_result"],
            "method1_states": row["selected_states"],
        }

    forward_result = verify_ref(row["method2_forward_seed_result"], f"{session}:forward_result")
    forward_states = verify_ref(row["method2_forward_seed_states"], f"{session}:forward_states")
    accepted_states = verify_ref(preflight["accepted_templates"][task]["states"], f"{task}:accepted_states")
    session_root = output_root / task / session / "arm_round2_bidirectional"
    session_root.mkdir(parents=True)
    result_path = session_root / "RESULT.json"
    log_path = session_root / "execution.log"
    command = [
        sys.executable,
        str(ROUND2_TOOL),
        "--task",
        task,
        "--session",
        session,
        "--forward-result",
        str(forward_result),
        "--forward-states",
        str(forward_states),
        "--accepted-states",
        str(accepted_states),
        "--output",
        str(result_path),
    ]
    started = now()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.flush()
        os.fsync(log.fileno())
    item: dict[str, Any] = {
        "task": task,
        "session": session,
        "started_at": started,
        "finished_at": now(),
        "returncode": process.returncode,
        "command": command,
        "execution_log": ref(log_path),
        "method1_best_hold_backoff_m": row["best_hold_backoff_m"],
        "method1_result": row["method2_forward_seed_result"],
        "method1_states": row["method2_forward_seed_states"],
        "arm_method_rounds_consumed": 1,
    }
    if process.returncode in {0, 2} and result_path.is_file():
        result = load(result_path)
        if result.get("task") != task or result.get("session") != session:
            raise ValueError(f"{session}: round-2 identity mismatch")
        states_path = verify_ref(result["output_states"], f"{session}:round2_states")
        result_status = result.get("status")
        if result_status == "PASS_NUMERIC_CANARY_NO_AUTHORITY" and process.returncode == 0:
            classification = "PASS_ARM_ROUND2_BIDIRECTIONAL"
        elif result_status == "HOLD_NUMERIC_CANARY" and process.returncode == 2:
            classification = "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C"
        else:
            raise ValueError(
                f"{session}: inconsistent round-2 result/returncode {result_status}/{process.returncode}"
            )
        item.update(
            status=classification,
            arm_method_rounds_consumed=2,
            result=ref(result_path),
            states=ref(states_path),
            metrics=result.get("metrics"),
            gates=result.get("gates"),
        )
    else:
        item.update(
            status="FAILED_RUNTIME_RETRYABLE",
            error="round-2 command failed without a valid bounded result",
        )
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--placement-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")
    if not 1 <= args.max_workers <= 3:
        raise SystemExit("--max-workers must be in [1,3]")

    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    expected_tool = preflight.get("programs", {}).get(ROUND2_TOOL.name)
    if expected_tool is None or ref(ROUND2_TOOL) != expected_tool:
        raise SystemExit("round-2 tool differs from preflight closure")
    expected_wrapper = preflight.get("programs", {}).get(Path(__file__).name)
    if expected_wrapper is None or ref(Path(__file__)) != expected_wrapper:
        raise SystemExit("arm round-2 wrapper differs from preflight closure")

    placement = load(args.placement_result)
    if placement.get("status") != "PASS_METHOD1_PLACEMENT_EVALUATION_PUBLISHED":
        raise SystemExit("complete method-1 placement aggregate required")
    if not preflight_lineage_allows(preflight, ref(args.preflight), placement.get("preflight")):
        raise SystemExit("placement/preflight closure mismatch")
    preflight_rows = preflight.get("sessions", [])
    unique_sessions(preflight_rows, "preflight")
    preflight_by_session = {row["session"]: row for row in preflight_rows}
    placement_rows = placement.get("sessions", [])
    if set(preflight_by_session) != unique_sessions(placement_rows, "placement"):
        raise SystemExit("placement/preflight session set mismatch")

    args.output_root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_session, row, preflight, args.output_root): row["session"]
            for row in placement_rows
        }
        by_session = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
    rows = [by_session[row["session"]] for row in placement_rows]
    counts = {
        "method1_pass_carried": sum(row["status"] == "PASS_ARM_METHOD1_CARRIED_NO_ROUND2" for row in rows),
        "round2_pass": sum(row["status"] == "PASS_ARM_ROUND2_BIDIRECTIONAL" for row in rows),
        "round2_hold_two_methods_exhausted": sum(
            row["status"] == "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C" for row in rows
        ),
        "round2_runtime_failed_retryable": sum(row["status"] == "FAILED_RUNTIME_RETRYABLE" for row in rows),
    }

    aggregate_status = (
        "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED"
        if counts["round2_runtime_failed_retryable"] == 0
        else "FAILED_RUNTIME_RETRYABLE"
    )
    aggregate = {
        "schema_version": "exact78-robot-arm-bidirectional-round2-v52-v1",
        "created_at": now(),
        "status": aggregate_status,
        "preflight": ref(args.preflight),
        "placement_result": ref(args.placement_result),
        "round2_tool": ref(ROUND2_TOOL),
        "parallel_workers": args.max_workers,
        "counts": counts,
        "sessions": rows,
        "gpu_calls": 0,
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "Final bounded arm numeric method only. Numeric HOLD rows exhausted the two-method arm budget and "
            "are quality-C candidates; PASS rows may proceed to bounded hand retarget. No render, contact, "
            "action, deployment, or Robot authority is implied."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(
        json.dumps(
            {
                "status": aggregate_status,
                "counts": counts,
                "result": ref(args.output_root / "RESULT.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if counts["round2_runtime_failed_retryable"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
