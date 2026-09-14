from pathlib import Path
import importlib.util

MODULE = Path(__file__).resolve().parents[1] / "tools/reconcile_two_task_9h_evidence.py"
spec = importlib.util.spec_from_file_location("reconciler", MODULE)
r = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(r)


def test_required_slots_are_explicit():
    status = r.latest_status([])
    assert len(status["slots"]) == 12
    assert all(row["state"] == "NOT_RECORDED" for row in status["slots"].values())


def test_complete_requires_evidence(tmp_path, monkeypatch):
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"mask","task":"chips","state":"COMPLETE_PASS","summary":"x","evidence":[]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert row is None
    assert "complete_without_pinned_evidence" in errors


def test_bad_pin_is_rejected(tmp_path):
    artifact = tmp_path / "RESULT.json"
    artifact.write_text("{}", encoding="utf-8")
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"clean","task":"poker","state":"HOLD","summary":"x","evidence":[{"path":"' + str(artifact) + '","sha256":"bad"}]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert row is None
    assert "evidence_0_sha256" in errors


def test_latest_record_wins_per_slot():
    base = {"component":"robot", "task":"chips", "summary":"a", "evidence":[], "source_sha256":"1"}
    older = dict(base, recorded_at="2026-09-10T20:00:00+08:00", state="RUNNING")
    newer = dict(base, recorded_at="2026-09-10T21:00:00+08:00", state="HOLD", source_sha256="2")
    status = r.latest_status([newer, older])
    assert status["slots"]["robot:chips"]["state"] == "HOLD"


def test_partial_batch_complete_is_downgraded(tmp_path):
    formal = tmp_path / "RESULT.json"
    formal.write_text('{"status":"PASS"}', encoding="utf-8")
    sha = r.digest(formal)
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"mask","task":"chips","state":"COMPLETE_PASS","summary":"31 only","completion_closure":{"required_sessions":78,"terminal_sessions":31,"gate_pass_sessions":31,"terminal_fail_sessions":0,"quality_gate_pass":true},"evidence":[{"path":"' + str(formal) + '","sha256":"' + sha + '"}]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert not errors
    assert row["asserted_state"] == "COMPLETE_PASS"
    assert row["state"] == "RUNNING"
    assert "terminal_sessions_not_78" in row["closure_warnings"]


def test_complete_both_tasks_is_downgraded(tmp_path):
    formal = tmp_path / "RESULT.json"
    formal.write_text('{"status":"COMPLETE"}', encoding="utf-8")
    sha = r.digest(formal)
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"robot","task":"both","state":"COMPLETE_PASS","summary":"bad aggregate","completion_closure":{"required_sessions":78,"terminal_sessions":78,"gate_pass_sessions":78,"terminal_fail_sessions":0,"quality_gate_pass":true},"evidence":[{"path":"' + str(formal) + '","sha256":"' + sha + '"}]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert not errors
    assert row["state"] == "RUNNING"
    assert "complete_pass_must_be_single_task" in row["closure_warnings"]


def test_valid_78_delivery_can_close(tmp_path):
    formal = tmp_path / "RESULT.json"
    formal.write_text('{"status":"COMPLETE_PASS"}', encoding="utf-8")
    sha = r.digest(formal)
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"clean","task":"poker","state":"COMPLETE_PASS","summary":"78 closed","completion_closure":{"required_sessions":78,"terminal_sessions":78,"gate_pass_sessions":78,"terminal_fail_sessions":0,"quality_gate_pass":true},"evidence":[{"path":"' + str(formal) + '","sha256":"' + sha + '"}]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert not errors
    assert row["state"] == "COMPLETE_PASS"
    assert not row.get("closure_warnings")


def test_mixed_78_terminal_is_delivery_complete(tmp_path):
    formal = tmp_path / "RESULT.json"
    formal.write_text('{"status":"TERMINAL"}', encoding="utf-8")
    sha = r.digest(formal)
    record = tmp_path / "STATE.json"
    record.write_text('{"schema_version":"two-task-78-nine-hour-evidence-v1","component":"robot","task":"chips","state":"COMPLETE_TERMINAL_MIXED","summary":"72 pass 6 fail","completion_closure":{"required_sessions":78,"terminal_sessions":78,"gate_pass_sessions":72,"terminal_fail_sessions":6,"grade_c_promoted":false},"evidence":[{"path":"' + str(formal) + '","sha256":"' + sha + '"}]}', encoding="utf-8")
    row, errors = r.validate_record(record)
    assert not errors
    assert row["state"] == "COMPLETE_TERMINAL_MIXED"


def test_partial_successor_does_not_reopen_mixed_delivery():
    complete = {"component":"mask", "task":"poker", "state":"COMPLETE_TERMINAL_MIXED", "summary":"78 terminal", "evidence":[], "source_sha256":"a", "recorded_at":"2026-09-10T20:00:00+08:00"}
    activity = {"component":"mask", "task":"poker", "state":"RUNNING", "summary":"successor 2/41", "evidence":[], "source_sha256":"b", "recorded_at":"2026-09-10T21:00:00+08:00"}
    status = r.latest_status([complete, activity])
    slot = status["slots"]["mask:poker"]
    assert slot["state"] == "COMPLETE_TERMINAL_MIXED"
    assert slot["latest_activity"]["state"] == "RUNNING"
