from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from chaoyang.governance import execute_exact78_0915_cards001_base_temporal_replay_v1 as executor


def test_packet_draft_is_bounded_and_cpu_only() -> None:
    packet = executor.packet_value()
    assert packet["execution_mode"] == "EXACT78_CANDIDATE"
    assert packet["phase"] == executor.operation.PHASE
    assert packet["budgets"]["session_attempts_max"] == 1
    assert packet["budgets"]["wall_seconds_max"] == 3600
    assert packet["budgets"]["gpu_count"] == 0
    assert packet["config"]["model_calls"] == 0
    assert packet["config"]["top_k"] == 8
    assert len(packet["read_set"]) <= 8
    assert len(packet["commands"]) == 1
    assert executor.OPERATION in packet["commands"][0]
    assert datetime.fromisoformat(packet["deadline_at"]) <= executor.H24_HARD_CUTOFF
    actual = {Path(ref["path"]).relative_to(executor.REPO_ROOT).as_posix()
              for ref in packet["runtime_code"]}
    assert actual == set(executor.CODE)
    self_ref = next(ref for ref in packet["runtime_code"] if ref["path"] == str(executor.SCRIPT))
    assert self_ref == executor.artifact_ref(executor.SCRIPT)


def test_h20_blocks_registration_and_new_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    class AfterCutoff(datetime):
        @classmethod
        def now(cls, tz=None):
            return executor.H20_LATEST_START

    monkeypatch.setattr(executor, "datetime", AfterCutoff)
    with pytest.raises(RuntimeError, match="H20_NEW_DISPATCH_CUTOFF"):
        executor.readiness({"deadline_at": executor.H24_HARD_CUTOFF.isoformat()})
    monkeypatch.setattr(executor, "verify_fence", lambda _packet: None)
    with pytest.raises(RuntimeError, match="H20_NEW_DISPATCH_CUTOFF"):
        executor.run_worker({})


def test_execute_refuses_unready_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor, "packet_value", lambda: {"run_signature": "draft"})
    monkeypatch.setattr(executor, "readiness", lambda _packet: (_ for _ in ()).throw(
        RuntimeError("ACTIVE_TASK_OR_ROUTE_UNREADY")))
    monkeypatch.setattr(executor, "register", lambda _packet: pytest.fail("registered"))
    monkeypatch.setattr(executor, "run_worker", lambda _packet: pytest.fail("worker started"))
    with pytest.raises(RuntimeError, match="ACTIVE_TASK_OR_ROUTE_UNREADY"):
        executor.main(["--execute"])


def test_check_only_reports_blockers_without_registering(monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(executor, "packet_value", lambda: {"run_signature": "draft"})
    monkeypatch.setattr(executor, "readiness", lambda _packet: (_ for _ in ()).throw(
        RuntimeError("ACTIVE_TASK")))
    monkeypatch.setattr(executor, "route_readiness", lambda _packet: (_ for _ in ()).throw(
        RuntimeError("ROUTE_NOT_PUBLISHED")))
    monkeypatch.setattr(executor, "register", lambda _packet: pytest.fail("registered"))
    assert executor.main(["--check-only"]) == 0
    output = capsys.readouterr().out
    assert '"status": "DRY_RUN_NOT_ROUTABLE"' in output
    assert '"candidate_route": "ROUTE_NOT_PUBLISHED"' in output
