#!/usr/bin/env python3
"""Run bounded current-READY Robot30 offline-visual sessions serially.

This is an executor, not a ledger writer.  It registers one immutable task at a
time through the governance scheduler, uses the governance CLIs for all state
changes, and only proceeds after the current session has a terminal receipt.
The outputs are explicitly offline visual evidence and are never causal RC1
training inputs.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from chaoyang.governance.common import RECEIPT_PATH, REPO_ROOT, atomic_json, load_json, now_iso, process_identity


RUN = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1"
SELECTION = RUN / "robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json"
MATRIX = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"


def current_revision() -> int:
    return int(load_json(RECEIPT_PATH)["governance_revision"])


def run_checked(argv: list[str], *, stdout=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=REPO_ROOT, text=True, stdout=stdout, stderr=subprocess.STDOUT, check=False)


def governance_update(task_id: str, status: str, *, attempt: int, session: str,
                      result: Path | None = None, message: str, clear_runtime: bool = False,
                      clear_next_task: bool = False) -> None:
    identity = process_identity(os.getpid())
    for _ in range(20):
        argv = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", task_id, "--status", status,
            "--attempt", str(attempt), "--session", session,
            "--message", message, "--expected-revision", str(current_revision()),
        ]
        if clear_runtime:
            argv.append("--clear-runtime")
        else:
            argv += ["--pid", str(os.getpid()), "--proc-start-ticks", str(identity["start_ticks"])]
        if clear_next_task:
            argv.append("--clear-next-task")
        if result is not None:
            argv += ["--result", str(result)]
        proc = run_checked(argv, stdout=subprocess.DEVNULL)
        if proc.returncode == 0:
            return
        time.sleep(0.75)
    raise RuntimeError(f"governance update did not converge: {task_id} -> {status}")


def heartbeat(task_id: str, session: str, stage: str) -> None:
    proc = run_checked([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", task_id, "--pid", str(os.getpid()), "--status", "RUNNING",
        "--phase", stage, "--session", session,
    ], stdout=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"heartbeat failed for {task_id}")


def register(session: str, batch_id: int) -> tuple[str, str]:
    task_id = f"rc1_robot30_offline_visual_fresh_batch_{batch_id:03d}"
    stage = f"RC1_ROBOT30_OFFLINE_VISUAL_FRESH_BATCH_{batch_id:03d}"
    proc = run_checked([
        sys.executable, "-m", "chaoyang.governance.schedule_rc1_robot30_offline_fresh_session",
        "--session", session, "--batch-id", str(batch_id),
        "--expected-revision", str(current_revision()),
    ], stdout=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to register {task_id}")
    return task_id, stage


def run_attempt(session: str, batch_id: int, attempt: int, task_id: str, stage: str,
                heartbeat_seconds: int, phase_timeout_seconds: int) -> tuple[int, Path]:
    output = RUN / f"robot30_offline_visual/fresh_batch_{batch_id:03d}/attempt_{attempt:04d}"
    if output.exists():
        raise RuntimeError(f"no-clobber: attempt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"attempt_{attempt:04d}.executor.log"
    governance_update(task_id, "CLAIMED", attempt=attempt, session=session,
                      message=f"Queue executor claimed immutable attempt {attempt}.")
    argv = [
        sys.executable, "-m", "chaoyang.ops.run_rc1_robot30_offline_visual_batch",
        "--sessions", session, "--matrix", str(MATRIX), "--selection", str(SELECTION),
        "--output-root", str(output), "--phase-timeout-seconds", str(phase_timeout_seconds),
        "--contract-mode", "fresh",
    ]
    with log_path.open("wb") as log:
        child = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        governance_update(task_id, "RUNNING", attempt=attempt, session=session,
                          message=f"Fresh offline visual attempt {attempt} started.")
        while child.poll() is None:
            time.sleep(heartbeat_seconds)
            heartbeat(task_id, session, stage)
        return int(child.returncode), output / "RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--start-batch-id", type=int, required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--phase-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    queue_root = RUN / f"robot30_offline_visual/queue_{args.start_batch_id:03d}"
    queue_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for offset, session in enumerate(args.sessions):
        batch_id = args.start_batch_id + offset
        task_id, stage = register(session, batch_id)
        terminal_result: Path | None = None
        last_returncode: int | None = None
        for attempt in (1, 2):
            last_returncode, result = run_attempt(
                session, batch_id, attempt, task_id, stage,
                args.heartbeat_seconds, args.phase_timeout_seconds,
            )
            if last_returncode == 0 and result.is_file():
                payload = load_json(result)
                if payload.get("status") == "PASSED_TERMINAL_COVERAGE":
                    terminal_result = result
                    break
        if terminal_result is not None:
            payload = load_json(terminal_result)
            row = payload["rows"][0]
            governance_update(
                task_id, "PASSED", attempt=attempt, session=session, result=terminal_result,
                message=(f"Offline terminal coverage complete: row={row['terminal_status']}, "
                         f"hard_geometry_pass={row['hard_geometry_pass']}; never causal training input."),
                clear_runtime=True, clear_next_task=True,
            )
            rows.append({"session": session, "task_status": "PASSED", "attempt": attempt,
                         "row_terminal_status": row["terminal_status"],
                         "hard_geometry_pass": bool(row["hard_geometry_pass"]),
                         "result": str(terminal_result)})
            continue
        failure = queue_root / session / "FAILED_RUNTIME_FINAL.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(failure, {
            "schema_version": "chaoyang-rc1-robot30-queue-runtime-failure-v1",
            "task_id": task_id, "session": session, "status": "FAILED_RUNTIME_FINAL",
            "created_at": now_iso(), "runtime_attempts": 2, "last_returncode": last_returncode,
            "claim_limit": "Executor/runtime failure only; no Robot quality conclusion.",
        })
        governance_update(task_id, "FAILED_RUNTIME_FINAL", attempt=2, session=session, result=failure,
                          message="Two runtime attempts exhausted; no Robot quality conclusion.",
                          clear_runtime=True, clear_next_task=True)
        rows.append({"session": session, "task_status": "FAILED_RUNTIME_FINAL", "attempt": 2,
                     "hard_geometry_pass": False, "result": str(failure)})
    result = queue_root / "RESULT.json"
    atomic_json(result, {
        "schema_version": "chaoyang-rc1-robot30-offline-queue-v1",
        "status": "PASSED_BOUNDED_QUEUE_TERMINATED", "created_at": now_iso(),
        "rows": rows,
        "counts": {
            "sessions": len(rows),
            "hard_geometry_pass": sum(bool(row.get("hard_geometry_pass")) for row in rows),
            "runtime_failed_final": sum(row["task_status"] == "FAILED_RUNTIME_FINAL" for row in rows),
        },
        "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
        "training_eligible": False,
    })
    print(json.dumps(load_json(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
