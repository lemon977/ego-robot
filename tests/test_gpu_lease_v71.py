from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chaoyang.governance.gpu_lease_v71 import audit_reclaim, query_gpu_memory_mib
from chaoyang.ops.run_gpu_command_with_v71_lease import physical_gpu_capacity


NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def lease() -> dict[str, object]:
    return {
        "schema_version": "chaoyang-gpu-lease-v71",
        "status": "ACQUIRED",
        "task_id": "task",
        "attempt_id": "attempt_0001",
        "hostname": "host",
        "boot_id": "boot",
        "pid": 123,
        "process_startticks": 456,
        "gpu_process_pid": 124,
        "gpu_id": 0,
        "acquired_at": (NOW - timedelta(minutes=5)).isoformat(),
        "heartbeat_at": (NOW - timedelta(minutes=3)).isoformat(),
        "expires_at": (NOW - timedelta(minutes=1)).isoformat(),
        "executor_epoch": 1,
        "fencing_token": "0123456789abcdef",
        "priority": "CANARY",
    }


def dead(_: int) -> dict[str, object]:
    return {"alive": False, "start_ticks": None}


def live(_: int) -> dict[str, object]:
    return {"alive": True, "start_ticks": 456}


def test_reclaim_requires_all_three_evidence_classes() -> None:
    assert audit_reclaim(lease(), at=NOW, identity_probe=dead, gpu_pids=set())["safe_to_reclaim"]
    assert not audit_reclaim(lease(), at=NOW, identity_probe=live, gpu_pids=set())["safe_to_reclaim"]
    assert not audit_reclaim(lease(), at=NOW, identity_probe=dead, gpu_pids={124})["safe_to_reclaim"]
    assert not audit_reclaim(lease(), at=NOW, identity_probe=dead, gpu_pids=None)["safe_to_reclaim"]


def test_unexpired_lease_is_held() -> None:
    value = lease()
    value["expires_at"] = (NOW + timedelta(seconds=30)).isoformat()
    assert not audit_reclaim(value, at=NOW, identity_probe=dead, gpu_pids=set())["safe_to_reclaim"]


def test_schema_rejects_missing_fencing_token() -> None:
    value = lease()
    value.pop("fencing_token")
    with pytest.raises(Exception):
        audit_reclaim(value, at=NOW, identity_probe=dead, gpu_pids=set())


def test_capacity_gate_allows_external_pid_when_memory_is_sufficient(monkeypatch) -> None:
    # Process presence is still used for stale-lease evidence, but admission is
    # based on capacity once the central lease is free.
    monkeypatch.setattr(
        "chaoyang.ops.run_gpu_command_with_v71_lease.query_gpu_memory_mib",
        lambda gpu_id: {"total_mib": 97871, "used_mib": 5411, "free_mib": 91956},
    )
    assert physical_gpu_capacity(0, 61440)["free_mib"] == 91956
    assert physical_gpu_capacity(0, 95000) is None


def test_memory_query_parses_nvidia_smi(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "97871, 5411, 91956\n"

    monkeypatch.setattr("chaoyang.governance.gpu_lease_v71.subprocess.run", lambda *args, **kwargs: Result())
    assert query_gpu_memory_mib(0) == {"total_mib": 97871, "used_mib": 5411, "free_mib": 91956}
