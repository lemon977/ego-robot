import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from chaoyang.ops import run_gpu_command_with_v71_lease as gpu_command


ROOT = Path(__file__).resolve().parents[1]


def test_blocked_wait_does_not_launch_command(tmp_path):
    lease = tmp_path / "lease.json"
    lease.write_text(json.dumps({"schema_version": "legacy", "status": "ACQUIRED"}))
    marker = tmp_path / "must_not_exist"
    receipt = tmp_path / "receipt.json"
    completed = subprocess.run(
        [
            sys.executable, "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
            "--task-id", "unit", "--attempt-id", "attempt_0001",
            "--priority", "CANARY", "--wait-seconds", "0", "--wall-seconds", "10",
            "--receipt", str(receipt), "--lease", str(lease), "--lock", str(tmp_path / "lock"),
            "--claim-limit", "unit only", "--", sys.executable, "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 3
    assert not marker.exists()
    assert json.loads(receipt.read_text())["status"] == "BLOCKED_RESOURCE"


def test_source_uses_v71_three_evidence_and_ttl():
    source = (ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py").read_text()
    assert "audit_reclaim" in source
    assert "build_lease" in source
    assert "heartbeat" in source
    assert "query_gpu_pids" in source


def test_zero_wait_budget_still_attempts_lease_once(monkeypatch):
    calls = []

    def acquire(_args, token):
        calls.append(token)
        return {"status": "ACQUIRED"}

    monkeypatch.setattr(gpu_command, "try_acquire", acquire)
    lease = gpu_command.acquire_with_wait_budget(SimpleNamespace(wait_seconds=0), "token")
    assert lease == {"status": "ACQUIRED"}
    assert calls == ["token"]


def test_zero_wait_budget_returns_blocked_after_one_failed_attempt(monkeypatch):
    calls = []

    def acquire(_args, token):
        calls.append(token)
        return None

    monkeypatch.setattr(gpu_command, "try_acquire", acquire)
    lease = gpu_command.acquire_with_wait_budget(SimpleNamespace(wait_seconds=0), "token")
    assert lease is None
    assert calls == ["token"]
