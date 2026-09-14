#!/usr/bin/env python3
"""Run one synthetic ProPainter Clean job under the shared GPU lease."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
LEASE = PROJECT / "_run/GPU_LEASE.json"
LOCK = PROJECT / "_run/GPU_LEASE.lock"
RUNNER = PROJECT / "tools/run_clean_synthetic_propainter_baseline.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def gpu_state() -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    fields = [value.strip() for value in query.split(",")]
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "total_mib": int(fields[2]),
        "used_mib": int(fields[3]),
        "free_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
    }


def acquire(holder: str, wall_seconds: int) -> dict[str, Any]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(LEASE.read_text(encoding="utf-8"))
        if current.get("status") != "RELEASED":
            raise RuntimeError(f"GPU lease held by {current.get('holder')}")
        started = datetime.now().astimezone()
        value = {
            "schema_version": "gpu-lease-v1",
            "status": "ACQUIRED",
            "holder": holder,
            "holder_pid": os.getpid(),
            "worker_pid": None,
            "since": started.isoformat(timespec="seconds"),
            "scope": {
                "gpu_indices": [0],
                "purpose": "explicit SYNTHETIC Clean baseline with commit-pinned ProPainter",
                "max_wall_seconds": wall_seconds,
                "expires_at": (started + timedelta(seconds=wall_seconds)).isoformat(timespec="seconds"),
            },
            "pre_start_gpu_state": gpu_state(),
            "claim_limit": "Generated pixels are SYNTHETIC_PROPAINTER and are not physical background truth.",
        }
        atomic_json(LEASE, value)
        return value


def set_worker(holder: str, worker_pid: int) -> None:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(LEASE.read_text(encoding="utf-8"))
        if value.get("status") != "ACQUIRED" or value.get("holder") != holder:
            raise RuntimeError("lost GPU lease before worker start")
        value["worker_pid"] = worker_pid
        atomic_json(LEASE, value)


def release(holder: str, reason: str) -> dict[str, Any]:
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(LEASE.read_text(encoding="utf-8"))
        if value.get("status") != "ACQUIRED" or value.get("holder") != holder:
            raise RuntimeError("refuse to release unowned GPU lease")
        value.update(
            {
                "status": "RELEASED",
                "worker_pid": None,
                "released_at": now(),
                "release_reason": reason,
                "post_run_gpu_state": gpu_state(),
            }
        )
        atomic_json(LEASE, value)
        return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--holder", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=3600)
    args = parser.parse_args()
    spec = args.spec.resolve(strict=True)
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise RuntimeError(f"no-clobber receipt exists: {receipt}")

    acquired = acquire(args.holder, args.wall_seconds)
    completed: subprocess.CompletedProcess[str] | None = None
    error: str | None = None
    try:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        process = subprocess.Popen(
            [sys.executable, os.fspath(RUNNER), "--spec", os.fspath(spec)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        set_worker(args.holder, process.pid)
        stdout, stderr = process.communicate(timeout=args.wall_seconds)
        completed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        reason = "WORKER_COMPLETE" if process.returncode == 0 else f"WORKER_RC_{process.returncode}"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        reason = f"WORKER_EXCEPTION_{type(exception).__name__}"
    released = release(args.holder, reason)
    payload = {
        "schema_version": "clean-synthetic-propainter-launch-receipt-v1",
        "created_at": now(),
        "status": "COMPLETE_GPU_RELEASED" if completed and completed.returncode == 0 else "TERMINAL_FAILURE_GPU_RELEASED",
        "holder": args.holder,
        "spec": {"path": str(spec), "bytes": spec.stat().st_size, "sha256": sha256(spec)},
        "runner": {"path": str(RUNNER), "bytes": RUNNER.stat().st_size, "sha256": sha256(RUNNER)},
        "lease_acquired": acquired,
        "worker_returncode": completed.returncode if completed else None,
        "worker_stdout_tail": completed.stdout[-12000:] if completed else "",
        "worker_stderr_tail": completed.stderr[-12000:] if completed else "",
        "worker_error": error,
        "lease_released": released,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "COMPLETE_GPU_RELEASED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
