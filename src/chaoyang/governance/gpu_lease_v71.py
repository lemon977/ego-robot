from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""TTL GPU lease with PID/startticks/GPU three-evidence stale recovery."""

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from chaoyang.governance.common import atomic_json, load_json, process_identity, validate_schema


DEFAULT_TTL_SECONDS = 120
PRIORITY_ORDER = {"CANARY": 0, "REGRESSION": 1, "FULL_BATCH": 2, "CHECKPOINT_TRAINING": 3}


def now() -> datetime:
    return datetime.now().astimezone()


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.astimezone()


def boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "UNKNOWN_BOOT_ID"


def query_gpu_pids() -> set[int] | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def query_gpu_memory_mib(gpu_id: int = 0) -> dict[str, int] | None:
    """Return physical memory capacity without requiring an empty GPU.

    Processes from another container can appear in ``nvidia-smi`` while being
    absent from this container's ``/proc``.  They are valid capacity users but
    must not permanently deadlock the project lease when ample memory remains.
    """
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        return None
    try:
        total, used, free = (int(value.strip()) for value in result.stdout.strip().split(","))
    except (TypeError, ValueError):
        return None
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def build_lease(
    *,
    task_id: str,
    attempt_id: str,
    pid: int,
    gpu_id: int,
    executor_epoch: int,
    fencing_token: str,
    priority: str,
    at: datetime | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    if priority not in PRIORITY_ORDER:
        raise ValueError(f"invalid priority: {priority}")
    identity = process_identity(pid)
    if not identity["alive"]:
        raise RuntimeError(f"lease owner is not alive: {pid}")
    current = at or now()
    lease = {
        "schema_version": "chaoyang-gpu-lease-v71",
        "status": "ACQUIRED",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "hostname": socket.gethostname(),
        "boot_id": boot_id(),
        "pid": pid,
        "process_startticks": identity["start_ticks"],
        "gpu_process_pid": None,
        "gpu_id": gpu_id,
        "acquired_at": iso(current),
        "heartbeat_at": iso(current),
        "expires_at": iso(current + timedelta(seconds=ttl_seconds)),
        "executor_epoch": executor_epoch,
        "fencing_token": fencing_token,
        "priority": priority,
    }
    validate_schema("gpu_lease_v71.schema.json", lease)
    return lease


def audit_reclaim(
    lease: dict[str, Any],
    *,
    at: datetime | None = None,
    identity_probe: Callable[[int], dict[str, Any]] = process_identity,
    gpu_pids: set[int] | None = None,
) -> dict[str, Any]:
    validate_schema("gpu_lease_v71.schema.json", lease)
    current = at or now()
    heartbeat_expired = current > parse_time(lease["expires_at"])
    identity = identity_probe(int(lease["pid"]))
    pid_mismatch = not identity["alive"] or identity.get("start_ticks") != lease["process_startticks"]
    gpu_evidence_available = gpu_pids is not None
    expected_gpu_pids = {int(lease["pid"])}
    if lease.get("gpu_process_pid") is not None:
        expected_gpu_pids.add(int(lease["gpu_process_pid"]))
    gpu_process_absent = gpu_pids is not None and expected_gpu_pids.isdisjoint(gpu_pids)
    safe_to_reclaim = bool(heartbeat_expired and pid_mismatch and gpu_process_absent)
    return {
        "heartbeat_expired": heartbeat_expired,
        "pid_startticks_mismatch": pid_mismatch,
        "gpu_evidence_available": gpu_evidence_available,
        "gpu_process_absent": gpu_process_absent,
        "safe_to_reclaim": safe_to_reclaim,
        "status": "STALE_RECLAIMABLE" if safe_to_reclaim else "HOLD",
    }


def heartbeat(lease: dict[str, Any], *, pid: int, fencing_token: str, at: datetime | None = None) -> dict[str, Any]:
    if lease["status"] != "ACQUIRED" or lease["pid"] != pid or lease["fencing_token"] != fencing_token:
        raise RuntimeError("lease fencing mismatch")
    identity = process_identity(pid)
    if not identity["alive"] or identity["start_ticks"] != lease["process_startticks"]:
        raise RuntimeError("lease process identity mismatch")
    current = at or now()
    updated = dict(lease)
    updated["heartbeat_at"] = iso(current)
    updated["expires_at"] = iso(current + timedelta(seconds=DEFAULT_TTL_SECONDS))
    validate_schema("gpu_lease_v71.schema.json", updated)
    return updated


def release(lease: dict[str, Any], *, pid: int, fencing_token: str, reason: str, at: datetime | None = None) -> dict[str, Any]:
    if lease["status"] != "ACQUIRED" or lease["pid"] != pid or lease["fencing_token"] != fencing_token:
        raise RuntimeError("lease fencing mismatch")
    updated = dict(lease)
    updated.update(status="RELEASED", released_at=iso(at or now()), release_reason=reason)
    validate_schema("gpu_lease_v71.schema.json", updated)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lease", type=Path)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    lease = load_json(args.lease)
    audit = audit_reclaim(lease, gpu_pids=query_gpu_pids())
    if args.recover:
        if not audit["safe_to_reclaim"]:
            raise SystemExit("lease is not safely reclaimable")
        lease = dict(lease)
        lease.update(
            status="RELEASED",
            released_at=iso(now()),
            release_reason="TTL_PID_GPU_THREE_EVIDENCE_RECOVERY",
            recovery_evidence=audit,
        )
        atomic_json(args.lease, lease)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
