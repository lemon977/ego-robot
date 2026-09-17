#!/usr/bin/env python3
"""Run one bounded command under the V7.1 TTL GPU lease.

Waiting does not consume an algorithm attempt.  A lease is reclaimed only when
heartbeat, PID/startticks and physical-GPU evidence all agree that it is stale.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.gpu_lease_v71 import (
    audit_reclaim,
    build_lease,
    heartbeat,
    query_gpu_memory_mib,
    query_gpu_pids,
    release,
)


DEFAULT_LEASE = ROOT / "_run/current/GPU_LEASE.json"
DEFAULT_LOCK = ROOT / "_run/current/GPU_LEASE.lock"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "RELEASED"}


def physical_gpu_capacity(gpu_id: int, min_free_mib: int) -> dict[str, Any] | None:
    memory = query_gpu_memory_mib(gpu_id)
    if memory is None or memory["free_mib"] < min_free_mib:
        return None
    return memory


def try_acquire(args: argparse.Namespace, token: str) -> dict[str, Any] | None:
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(args.lease)
        if current.get("status") != "RELEASED":
            if current.get("schema_version") == "chaoyang-gpu-lease-v71":
                audit = audit_reclaim(current, gpu_pids=query_gpu_pids())
                if audit["safe_to_reclaim"]:
                    current = dict(current)
                    current.update(
                        status="RELEASED", released_at=now_iso(),
                        release_reason="TTL_PID_GPU_THREE_EVIDENCE_RECOVERY",
                        recovery_evidence=audit,
                    )
                    atomic_json(args.lease, current)
                else:
                    return None
            else:
                return None
        capacity = physical_gpu_capacity(args.gpu_id, args.min_free_mib)
        if capacity is None:
            return None
        lease = build_lease(
            task_id=args.task_id,
            attempt_id=args.attempt_id,
            pid=os.getpid(), gpu_id=args.gpu_id,
            executor_epoch=args.executor_epoch,
            fencing_token=token, priority=args.priority,
        )
        atomic_json(args.lease, lease)
        return lease


def acquire_with_wait_budget(args: argparse.Namespace, token: str) -> dict[str, Any] | None:
    """Try once even when the caller requests no resource wait."""
    deadline = time.monotonic() + args.wait_seconds
    while True:
        lease = try_acquire(args, token)
        if lease is not None:
            return lease
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(30.0, remaining))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--executor-epoch", type=int, default=1)
    parser.add_argument("--priority", choices=("CANARY", "REGRESSION", "FULL_BATCH", "CHECKPOINT_TRAINING"), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61440)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lease", type=Path, default=DEFAULT_LEASE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--claim-limit", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise RuntimeError("command required after --")
    args.receipt = args.receipt.resolve()
    args.lease = args.lease.resolve()
    args.lock = args.lock.resolve()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise RuntimeError(f"immutable receipt exists: {args.receipt}")
    if args.wait_seconds < 0:
        raise ValueError("--wait-seconds must be non-negative")

    token = uuid.uuid4().hex
    wait_started = time.monotonic()
    lease = acquire_with_wait_budget(args, token)
    if lease is None:
        payload = {
            "schema_version": "v71-gpu-command-receipt-v1", "status": "BLOCKED_RESOURCE",
            "created_at": now_iso(), "task_id": args.task_id, "attempt_id": args.attempt_id,
            "reason": "GPU_WAIT_BUDGET_EXHAUSTED", "wait_seconds": args.wait_seconds,
            "claim_limit": args.claim_limit,
        }
        atomic_json(args.receipt, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 3

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    with args.lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(args.lease)
        if current.get("fencing_token") != token or current.get("pid") != os.getpid():
            os.killpg(process.pid, signal.SIGTERM)
            raise RuntimeError("lost fencing token before GPU worker registration")
        current["gpu_process_pid"] = process.pid
        atomic_json(args.lease, current)
        lease = current

    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(30):
            with args.lock.open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                current = load(args.lease)
                try:
                    updated = heartbeat(current, pid=os.getpid(), fencing_token=token)
                except RuntimeError:
                    return
                updated["gpu_process_pid"] = process.pid
                atomic_json(args.lease, updated)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    error = None
    try:
        stdout, stderr = process.communicate(timeout=args.wall_seconds)
    except subprocess.TimeoutExpired:
        error = "WALL_TIMEOUT"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    finally:
        stop.set()
        thread.join(timeout=5)

    with args.lock.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(args.lease)
        released = release(
            current, pid=os.getpid(), fencing_token=token,
            reason=error or f"WORKER_RC_{process.returncode}",
        )
        atomic_json(args.lease, released)

    payload = {
        "schema_version": "v71-gpu-command-receipt-v1",
        "status": "PASSED" if error is None and process.returncode == 0 else "FAILED_RUNTIME_FINAL",
        "created_at": now_iso(), "task_id": args.task_id, "attempt_id": args.attempt_id,
        "command": command, "returncode": process.returncode, "error": error,
        "wait_seconds_observed": round(time.monotonic() - wait_started, 3),
        "stdout_tail": stdout[-12000:], "stderr_tail": stderr[-12000:],
        "fencing_token": token, "claim_limit": args.claim_limit,
    }
    atomic_json(args.receipt, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
