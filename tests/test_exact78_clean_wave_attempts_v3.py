from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_exact78_clean_wave_guardian_v3 as guardian


def setup_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "plan"
    session = "get_potato_chips_test"
    (root / "sessions" / session).mkdir(parents=True)
    (root / "specs").mkdir()
    (root / "specs" / f"{session}_propainter.json").write_text("{}\n", encoding="utf-8")
    row: dict[str, object] = {
        "session_id": session,
        "task": "chips",
        "frame_count": 3,
    }
    return root, row


def write_placeholder_result(root: Path, session: str) -> Path:
    path = root / "propainter_v1" / session / "RESULT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"session": session}) + "\n", encoding="utf-8")
    return path


def test_runtime_failure_gets_fresh_attempt_and_preserves_failed_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, row = setup_root(tmp_path)
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = write_placeholder_result(root, str(row["session_id"]))
        if calls == 1:
            raise guardian.RuntimeAttemptError("simulated CUDA initialization error")
        return {"returncode": 0, "result": str(result)}

    monkeypatch.setattr(guardian, "run", fake_run)
    monkeypatch.setattr(guardian, "state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guardian, "runtime_gate", lambda: {"pass": True})
    monkeypatch.setattr(guardian, "governance_heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        guardian,
        "validate_clean",
        lambda path, _row: guardian.ref(path),
    )
    outcome = guardian.run_propainter_with_retries(root, row, {})
    assert outcome["status"] == "PASSED"
    assert outcome["attempt"] == 2
    assert (root / "sessions" / str(row["session_id"]) / "attempts/attempt_0001/runtime_failed_output").is_dir()
    assert (root / "sessions" / str(row["session_id"]) / "final/RESULT_REF.json").is_file()


def test_quality_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, row = setup_root(tmp_path)
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        write_placeholder_result(root, str(row["session_id"]))
        return {"returncode": 0}

    def fail_quality(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise guardian.QualityGateError("simulated visual quality C")

    monkeypatch.setattr(guardian, "run", fake_run)
    monkeypatch.setattr(guardian, "state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guardian, "runtime_gate", lambda: {"pass": True})
    monkeypatch.setattr(guardian, "governance_heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guardian, "validate_clean", fail_quality)
    outcome = guardian.run_propainter_with_retries(root, row, {})
    assert outcome["status"] == "FAILED_QUALITY_C"
    assert outcome["attempt"] == 1
    assert calls == 1
    attempts = root / "sessions" / str(row["session_id"]) / "attempts"
    assert [path.name for path in attempts.iterdir()] == ["attempt_0001"]


def test_lease_race_waits_without_consuming_runtime_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _row = setup_root(tmp_path)
    observations = iter(
        [
            {"pass": False, "lease": {"status": "ACQUIRED"}},
            {"pass": True, "lease": {"status": "RELEASED"}},
        ]
    )
    states: list[str] = []
    monkeypatch.setattr(guardian, "runtime_gate", lambda: next(observations))
    monkeypatch.setattr(
        guardian,
        "state",
        lambda _root, status, *_args, **_kwargs: states.append(status),
    )
    monkeypatch.setattr(guardian, "governance_heartbeat", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(guardian.time, "sleep", lambda _seconds: None)

    result = guardian.wait_runtime_gate(
        root,
        {},
        "get_potato_chips_test",
        guardian.time.monotonic() + 1800,
    )

    assert result["pass"] is True
    assert states == ["WAIT_GPU_RESOURCE_BETWEEN_ATTEMPTS"]
