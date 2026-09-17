#!/usr/bin/env python3
"""Launch the prompt-bootstrap canary with a local heartbeat and V7.1 GPU lease."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def startticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, IndexError, ValueError):
        return None


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--executor-epoch", type=int, default=1)
    args = parser.parse_args()
    attempt = args.attempt_root.resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    heartbeat = attempt / "LOCAL_HEARTBEAT.json"
    gpu_receipt = attempt / "GPU_COMMAND_RECEIPT.json"
    command = [
        sys.executable, str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", "mask_sam31_prompt_bootstrap_canary_r22",
        "--attempt-id", args.attempt_id,
        "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY",
        "--gpu-id", "0",
        "--min-free-mib", "61440",
        "--wait-seconds", "1800",
        "--wall-seconds", "1800",
        "--receipt", str(gpu_receipt),
        "--claim-limit", "Two frozen SAM3.1 prompt-bootstrap development canaries; no authority.",
        "--", sys.executable, str(ROOT / "src/chaoyang/ops/run_mask_sam31_prompt_bootstrap_canary_r22.py"),
        "--output-root", str(attempt),
    ]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop = threading.Event()

    def beat() -> None:
        while not stop.is_set():
            atomic(heartbeat, {
                "schema_version": "mask-prompt-bootstrap-local-heartbeat-r22-v1",
                "task_id": "mask_sam31_prompt_bootstrap_canary_r22",
                "attempt_id": args.attempt_id,
                "status": "RUNNING" if process.poll() is None else "TERMINATING",
                "heartbeat_at": now(),
                "launcher_pid": os.getpid(),
                "launcher_startticks": startticks(os.getpid()),
                "lease_wrapper_pid": process.pid,
                "lease_wrapper_startticks": startticks(process.pid),
            })
            stop.wait(30)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    stdout, stderr = process.communicate()
    stop.set(); thread.join(timeout=5)
    atomic(heartbeat, {
        "schema_version": "mask-prompt-bootstrap-local-heartbeat-r22-v1",
        "task_id": "mask_sam31_prompt_bootstrap_canary_r22",
        "attempt_id": args.attempt_id,
        "status": "TERMINAL",
        "heartbeat_at": now(),
        "launcher_pid": os.getpid(),
        "launcher_startticks": startticks(os.getpid()),
        "lease_wrapper_pid": process.pid,
        "lease_wrapper_startticks": startticks(process.pid),
        "returncode": process.returncode,
    })
    (attempt / "GPU_WRAPPER_STDOUT.log").write_text(stdout, encoding="utf-8")
    (attempt / "GPU_WRAPPER_STDERR.log").write_text(stderr, encoding="utf-8")
    print(stdout[-4000:])
    if stderr:
        print(stderr[-4000:], file=sys.stderr)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
