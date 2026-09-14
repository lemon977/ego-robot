#!/usr/bin/env python3
"""Run one explicit command under Chaoyang's central GPU lease.

This is a small general launcher for bounded development jobs.  It never waits
while holding the lease, records PID/start-ticks and a heartbeat, and always
publishes a terminal receipt before returning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEASE = ROOT / "_run/GPU_LEASE.json"
LOCK = ROOT / "_run/GPU_LEASE.lock"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def gpu_state() -> dict[str, Any]:
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    fields = [part.strip() for part in line.split(",")]
    return {
        "index": int(fields[0]), "name": fields[1], "total_mib": int(fields[2]),
        "used_mib": int(fields[3]), "free_mib": int(fields[4]), "utilization_percent": int(fields[5]),
    }


def live_gpu_compute_pids() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True,
    )
    values: list[int] = []
    for line in output.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if Path(f"/proc/{pid}").exists():
            values.append(pid)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holder", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--claim-limit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("command required after --")
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise RuntimeError(f"no-clobber receipt exists: {receipt}")
    physical = live_gpu_compute_pids()
    if physical:
        raise RuntimeError(f"physical GPU already has live compute PIDs: {physical}")

    acquired_at = datetime.now().astimezone()
    launcher_ticks = start_ticks(os.getpid())
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8"))
        if current.get("status") != "RELEASED":
            raise RuntimeError(f"GPU lease held by {current.get('holder')}")
        lease = {
            "schema_version": "gpu-lease-v1", "status": "ACQUIRED", "holder": args.holder,
            "holder_pid": os.getpid(), "holder_start_ticks": launcher_ticks, "worker_pid": None,
            "since": acquired_at.isoformat(timespec="seconds"), "heartbeat_at": now(),
            "scope": {"gpu_indices": [0], "purpose": args.purpose, "max_wall_seconds": args.wall_seconds,
                      "expires_at": (acquired_at + timedelta(seconds=args.wall_seconds)).isoformat(timespec="seconds")},
            "pre_start_gpu_state": gpu_state(), "claim_limit": args.claim_limit,
        }
        atomic_json(LEASE, lease)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    worker_ticks = start_ticks(process.pid)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8"))
        if current.get("holder") != args.holder or current.get("holder_pid") != os.getpid():
            process.terminate()
            raise RuntimeError("lost GPU lease before worker registration")
        current.update({"worker_pid": process.pid, "worker_start_ticks": worker_ticks, "heartbeat_at": now()})
        atomic_json(LEASE, current)

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(25):
            with LOCK.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                current = json.loads(LEASE.read_text(encoding="utf-8"))
                if current.get("holder") != args.holder or current.get("holder_pid") != os.getpid():
                    return
                current["heartbeat_at"] = now()
                atomic_json(LEASE, current)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    error: str | None = None
    try:
        stdout, stderr = process.communicate(timeout=args.wall_seconds)
    except subprocess.TimeoutExpired:
        error = "WALL_TIMEOUT"
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    finally:
        stop.set()
        thread.join(timeout=5)

    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8"))
        if current.get("holder") != args.holder or current.get("holder_pid") != os.getpid():
            raise RuntimeError("refuse to release an unowned GPU lease")
        current.update({"status": "RELEASED", "worker_pid": None, "released_at": now(),
                        "release_reason": error or f"WORKER_RC_{process.returncode}",
                        "post_run_gpu_state": gpu_state()})
        atomic_json(LEASE, current)

    payload = {
        "schema_version": "generic-gpu-command-receipt-v1", "created_at": now(),
        "status": "PASSED" if process.returncode == 0 and error is None else "FAILED_RUNTIME_FINAL",
        "holder": args.holder, "launcher_pid": os.getpid(), "launcher_start_ticks": launcher_ticks,
        "worker_pid": process.pid, "worker_start_ticks": worker_ticks, "command": command,
        "returncode": process.returncode, "error": error, "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:], "claim_limit": args.claim_limit,
    }
    atomic_json(receipt, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
