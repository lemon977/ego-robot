#!/usr/bin/env python3
"""Keep Wave0 Clean-ready Robot batches moving until the finite queue is empty."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from chaoyang.governance.common import atomic_json, load_json, now_iso


PROJECT = Path(__file__).resolve().parents[3]
MATRIX_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices"
DEFAULT_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_wave0_supervisor_v55"
TASK_STATE = PROJECT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
LOCK = PROJECT / "_run/current/exact78_wave0_supervisor_v55.lock"
ROBOT_TASK = "exact78_v52_lane_c_contact_robot"
CLEAN_TASK = "exact78_wave0_clean_runtime_recovery_v53"
ACTIVE = {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def start_ticks(pid: int) -> int | None:
    try:
        return int((Path("/proc") / str(pid) / "stat").read_text().split()[21])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def task_row(task_id: str) -> dict:
    state = load_json(TASK_STATE)
    return next(row for row in state["tasks"] if row["task_id"] == task_id)


def task_is_live(row: dict) -> bool:
    if row.get("status") not in ACTIVE:
        return False
    pid = row.get("pid")
    ticks = row.get("proc_start_ticks")
    return isinstance(pid, int) and isinstance(ticks, int) and start_ticks(pid) == ticks


def write_state(root: Path, **values: object) -> None:
    atomic_json(
        root / "STATE.json",
        {
            "schema_version": "exact78-wave0-completion-supervisor-v55-v1",
            "updated_at": now_iso(),
            "pid": os.getpid(),
            "proc_start_ticks": start_ticks(os.getpid()),
            **values,
        },
    )


def rebuild_matrix() -> Path:
    subprocess.run(
        [
            sys.executable,
            "src/chaoyang/ops/build_exact78_v52_current_matrices.py",
            "--output-root",
            str(MATRIX_ROOT),
        ],
        cwd=PROJECT,
        check=True,
    )
    return MATRIX_ROOT / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"


def ready_sessions(matrix_path: Path) -> list[str]:
    matrix = load_json(matrix_path)
    return [
        row["session_id"]
        for row in matrix["rows"]
        if row["robot_current_state"] == "READY_FOR_ROBOT_CURRENT_DRAFT"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wall-seconds", type=int, default=172800)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with LOCK.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another exact78 Wave0 supervisor owns the lock") from error
        cycle = 0
        while time.monotonic() - started < args.max_wall_seconds:
            robot = task_row(ROBOT_TASK)
            clean = task_row(CLEAN_TASK)
            if task_is_live(robot):
                write_state(
                    root,
                    status="WAIT_EXISTING_ROBOT_EXECUTOR",
                    robot_pid=robot["pid"],
                    clean_status=clean["status"],
                )
                time.sleep(args.poll_seconds)
                continue
            matrix_path = rebuild_matrix()
            ready = ready_sessions(matrix_path)
            if ready:
                cycle += 1
                cycle_root = root / "cycles" / f"cycle_{cycle:04d}"
                write_state(
                    root,
                    status="RUNNING_ROBOT_SUCCESSOR",
                    ready_count=len(ready),
                    first_session=ready[0],
                    cycle=cycle,
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
                        "--matrix",
                        str(matrix_path),
                        "--output-root",
                        str(cycle_root),
                    ],
                    cwd=PROJECT,
                    check=False,
                )
                if completed.returncode:
                    write_state(
                        root,
                        status="FAILED_RUNTIME_RETRYABLE",
                        cycle=cycle,
                        returncode=completed.returncode,
                    )
                    return completed.returncode
                continue
            if not task_is_live(clean):
                write_state(
                    root,
                    status="PASSED_WAVE0_NO_CLEAN_READY_ROBOT_ROWS_REMAIN",
                    clean_status=clean["status"],
                    ready_count=0,
                )
                return 0
            write_state(
                root,
                status="WAIT_CLEAN_RUNTIME_SUCCESSOR",
                clean_pid=clean["pid"],
                ready_count=0,
            )
            time.sleep(args.poll_seconds)
    write_state(root, status="BLOCKED_RESOURCE_SUPERVISOR_WALL_BUDGET_EXHAUSTED")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
