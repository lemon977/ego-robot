#!/usr/bin/env python3
"""CPU-only V7.1 aggregator supervisor.

It never launches GPU work.  It observes immutable worker receipts, performs
CAS task-state updates through governance tools, and finalizes exact78 only
after the already-running Clean executor reaches zero pending rows.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
GOV = ROOT / "docs/governance"
STATE = GOV / "LONG_HORIZON_TASK_STATE.json"
MIN = GOV / "CURRENT_PROJECT_STATUS_MIN.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation"
LOCK = ROOT / "_run/current/v71_finite_convergence_supervisor.lock"

RECEIPTS = {
    "object_pose_hypothesis_v2": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/object_pose/RESULT.json",
    "sensor_h3_stereo_v1": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h3_stereo/attempts/attempt_0002/RESULT.json",
    "sensor_h4_mask_v1": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h4_mask/RESULT.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def task(task_id: str) -> dict:
    return next(row for row in load(STATE)["tasks"] if row["task_id"] == task_id)


def normalize_status(receipt: dict) -> str:
    value = str(receipt.get("formal_data_task_status", receipt.get("status", "")))
    if value.startswith("PASS"):
        return "PASSED"
    for status in ("BLOCKED_RESOURCE", "BLOCKED_PREREQ", "BLOCKED_EXTERNAL", "BLOCKED_REFERENCE_PROOF", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL"):
        if value.startswith(status):
            return status
    return "UNKNOWN_VERIFICATION_REQUIRED"


def update(task_id: str, result: Path, phase: str) -> None:
    row = task(task_id)
    if row["status"] != "PENDING":
        return
    receipt = load(result)
    status = normalize_status(receipt)
    for _ in range(5):
        revision = load(STATE)["governance_revision"]
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", task_id, "--status", status, "--phase", phase,
            "--clear-runtime", "--message", f"Automation adopted immutable receipt: {receipt.get('status')}",
            "--result", str(result), "--expected-revision", str(revision),
        ]
        if subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"CAS update failed for {task_id}")


def maybe_finalize_clean() -> bool:
    minimum = load(MIN)
    if minimum["waves"]["wave0_clean_pending"] != 0 or minimum["active_tasks"]:
        return False
    subprocess.run([sys.executable, "src/chaoyang/ops/build_exact78_v52_current_matrices.py", "--output-root", "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "src/chaoyang/ops/build_exact78_final_terminal_matrix_v71.py"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    result = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal/RESULT.json"
    if task("exact78_clean_r70_v71")["status"] == "PENDING":
        update("exact78_clean_r70_v71", result, "exact78_terminal_complete")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wall-seconds", type=int, default=172800)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another V7.1 finite convergence supervisor is live") from exc
        while time.monotonic() - started < args.max_wall_seconds:
            validation = subprocess.run([sys.executable, "-m", "chaoyang.governance.validate_governance_state"], cwd=ROOT, capture_output=True, text=True)
            if validation.returncode:
                atomic_json(OUT / "STATE.json", {"status": "STATUS_CONFLICT", "updated_at_epoch": time.time(), "stderr_tail": validation.stderr[-2000:]})
                time.sleep(args.poll_seconds)
                continue
            for task_id, path in RECEIPTS.items():
                if path.is_file():
                    update(task_id, path, task_id)
            clean_done = maybe_finalize_clean()
            minimum = load(MIN)
            atomic_json(OUT / "STATE.json", {
                "schema_version": "V71_FINITE_CONVERGENCE_SUPERVISOR_STATE",
                "status": "RUNNING" if not clean_done else "PASSED_AUTOMATION_CLOSURE",
                "pid": os.getpid(),
                "updated_at_epoch": time.time(),
                "gpu_work_launched": False,
                "clean_passed": minimum["waves"]["wave0_clean_passed"],
                "clean_pending": minimum["waves"]["wave0_clean_pending"],
                "observed_receipts": {key: value.is_file() for key, value in RECEIPTS.items()},
            })
            if clean_done and all(path.is_file() for path in RECEIPTS.values()):
                subprocess.run([sys.executable, "-m", "chaoyang.governance.create_meeting_snapshot"], cwd=ROOT, check=False, stdout=subprocess.DEVNULL)
                return 0
            time.sleep(args.poll_seconds)
    atomic_json(OUT / "STATE.json", {"status": "BLOCKED_RESOURCE_WALL_BUDGET", "pid": os.getpid(), "updated_at_epoch": time.time(), "gpu_work_launched": False})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
