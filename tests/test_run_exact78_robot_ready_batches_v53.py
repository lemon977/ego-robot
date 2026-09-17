import sys
import subprocess
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops import run_exact78_robot_ready_batches_v53 as runner


def test_phase_timeout_kills_process_group_and_converges(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "PHASE_TIMEOUT_SECONDS", 0.04)
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    with pytest.raises(RuntimeError, match="two runtime attempts exhausted"):
        runner.run_phase(
            tmp_path,
            "sleep",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "never-created",
            "fixture",
        )
    logs = sorted((tmp_path / "attempts").glob("*.log"))
    assert len(logs) == 2
    assert all("PHASE_TIMEOUT_SECONDS" in path.read_text() for path in logs)


def test_phase_reuse_requires_matching_command_and_output_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "heartbeat", lambda *args, **kwargs: None)
    expected = tmp_path / "RESULT.json"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(expected)!r}).write_text('ok')",
    ]
    runner.run_phase(tmp_path, "fixture", command, expected, "fixture")
    receipt = tmp_path / "attempts/fixture_SUCCESS.json"
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["expected"]["sha256"] == runner.sha256(expected)

    # An exact second invocation is a proven local reuse and does not run.
    runner.run_phase(tmp_path, "fixture", command, expected, "fixture")

    expected.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        runner.run_phase(tmp_path, "fixture", command, expected, "fixture")


def test_phase_reuse_refuses_existing_output_without_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "heartbeat", lambda *args, **kwargs: None)
    expected = tmp_path / "RESULT.json"
    expected.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without phase success receipt"):
        runner.run_phase(tmp_path, "fixture", [sys.executable, "-c", "pass"], expected, "fixture")


def test_runner_exposes_new_v71_task_id_option():
    source = runner.Path(runner.__file__).read_text(encoding="utf-8")
    assert '"--task-id"' in source
    assert "robot_geometry_v1" in source


def _ref(label: str):
    return {"path": f"/{label}", "bytes": 1, "sha256": "0" * 64}


def test_final_hand_artifacts_uses_method1_fields_for_carried_pass():
    row = {
        "status": "PASS_HAND_METHOD1_CARRIED_NO_ROUND2",
        "method1_result": _ref("method1-result"),
        "method1_states": _ref("method1-states"),
    }
    assert runner.final_hand_artifacts(row, "session") == (
        row["method1_result"],
        row["method1_states"],
    )


def test_final_hand_artifacts_uses_round2_fields_for_round2_hold():
    row = {
        "status": "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C",
        "result": _ref("round2-result"),
        "states": _ref("round2-states"),
    }
    assert runner.final_hand_artifacts(row, "session") == (row["result"], row["states"])


def test_final_hand_artifacts_fails_closed_on_incomplete_known_status():
    row = {
        "status": "PASS_HAND_METHOD1_CARRIED_NO_ROUND2",
        "method1_result": _ref("method1-result"),
    }
    with pytest.raises(RuntimeError, match="missing required artifact fields"):
        runner.final_hand_artifacts(row, "session")


def test_final_hand_artifacts_fails_closed_on_unknown_status():
    with pytest.raises(RuntimeError, match="unsupported final hand status"):
        runner.final_hand_artifacts({"status": "RUNNING"}, "session")


@pytest.mark.parametrize(
    "script",
    [
        "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
        "src/chaoyang/ops/audit_robot_hard_soft_gate_v71.py",
        "src/chaoyang/ops/run_robot_hard_soft_audit_watcher_v71.py",
    ],
)
def test_robot_cli_help_is_independent_of_current_working_directory(script):
    completed = subprocess.run(
        [sys.executable, str(runner.PROJECT / script), "--help"],
        cwd="/tmp",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
