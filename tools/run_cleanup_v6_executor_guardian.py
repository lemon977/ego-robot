#!/usr/bin/env python3
"""Keep the bounded V6 cleanup executor visible to the governance ledger.

The guardian performs no deletion.  It owns the executor identity/fencing
receipt and publishes heartbeats while the primary process performs separately
audited target-local cleanup commands.  Creating ``EXECUTOR_STOP`` terminates
it cleanly after the current heartbeat interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def start_ticks(pid: int) -> int:
    return int((Path("/proc") / str(pid) / "stat").read_text().split()[21])


def atomic_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.interval_seconds < 10:
        raise SystemExit("interval must be at least 10 seconds")
    run_root = args.run_root.resolve(strict=True)
    pid = os.getpid()
    ticks = start_ticks(pid)
    token = hashlib.sha256(
        f"{socket.gethostname()}:{pid}:{ticks}:{args.executor_epoch}:{now()}".encode()
    ).hexdigest()
    claim = {
        "schema_version": "exact78-executor-claim-v1",
        "status": "CLAIMED",
        "task_id": "chaoyang_current_only_cleanup_v6",
        "executor_id": "cleanup-v6-primary",
        "executor_epoch": args.executor_epoch,
        "fencing_token": token,
        "pid": pid,
        "proc_start_ticks": ticks,
        "claimed_at": now(),
        "claim_limit": "Heartbeat/fencing identity only; deletion requires a separate target-local manifest and receipt.",
    }
    atomic_new(run_root / "task_packet" / f"EXECUTOR_CLAIM_V{args.executor_epoch}.json", claim)
    stop_path = run_root / "EXECUTOR_STOP"
    while not stop_path.exists():
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.governance.heartbeat_task",
                "--task-id",
                "chaoyang_current_only_cleanup_v6",
                "--pid",
                str(pid),
                "--status",
                "RUNNING",
                "--phase",
                "per_target_cleanup_active",
                "--session",
                "workspace",
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
