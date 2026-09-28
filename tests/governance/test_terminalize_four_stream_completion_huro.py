import json
from pathlib import Path

import pytest

from chaoyang.governance import terminalize_four_stream_completion_huro as terminalize
from chaoyang.governance.common import artifact_ref


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _state():
    rows = [
        {
            "task_id": terminalize.TASK,
            "phase": "COMPLETION_PHASE_AB",
            "attempt": 1,
            "status": "PENDING",
            "updated_at": "before",
        }
    ]
    for suffix in ("scene", "sensor", "motion", "huro"):
        rows.append(
            {
                "task_id": terminalize.TASK + "_" + suffix,
                "parent_task_id": terminalize.TASK,
                "phase": "COMPLETION_PHASE_AB",
                "attempt": 1,
                "status": "PENDING",
                "pid": None,
                "proc_start_ticks": None,
                "updated_at": "before",
            }
        )
    return {
        "tasks": rows,
        "next_task": {"task_id": terminalize.TASK, "session": "all_fixed_sessions"},
        "recent_events": [],
    }


def test_terminalize_changes_only_huro_child_row():
    state = _state()
    before = {row["task_id"]: dict(row) for row in state["tasks"]}
    result = {"path": "/repo/terminal.json", "bytes": 10, "sha256": "a" * 64}
    updated = terminalize.terminalize_huro_row(state, result, "after")
    rows = {row["task_id"]: row for row in updated["tasks"]}
    assert rows[terminalize.HURO_TASK]["status"] == "REJECTED_QUALITY"
    assert rows[terminalize.HURO_TASK]["adoption"] == "NOT_ADOPTED"
    assert rows[terminalize.HURO_TASK]["result"] == result
    for task_id, row in before.items():
        if task_id != terminalize.HURO_TASK:
            assert rows[task_id] == row
    assert updated["next_task"] == state["next_task"]
    assert state["tasks"] == list(before.values())


def test_terminalize_rejects_owned_or_nonpending_huro_child():
    state = _state()
    child = next(row for row in state["tasks"] if row["task_id"] == terminalize.HURO_TASK)
    child["pid"] = 123
    with pytest.raises(RuntimeError, match="HURO_CHILD_HAS_WRITER"):
        terminalize.terminalize_huro_row(state, {}, "after")
    child["pid"] = None
    child["status"] = "REJECTED_QUALITY"
    with pytest.raises(RuntimeError, match="HURO_CHILD_NOT_PENDING"):
        terminalize.terminalize_huro_row(state, {}, "after")


def test_terminalize_rejects_inactive_parent():
    state = _state()
    state["tasks"][0]["status"] = "PASSED"
    with pytest.raises(RuntimeError, match="PARENT_NOT_ACTIVE"):
        terminalize.terminalize_huro_row(state, {}, "after")


def _evidence_fixture(tmp_path: Path, monkeypatch):
    candidate_path = tmp_path / "WINDOW_RESULT.json"
    audit_path = tmp_path / "OBJECTIVE.json"
    _write(
        candidate_path,
        {
            "task_id": terminalize.HURO_TASK,
            "execution": "REAL_OFFICIAL_CORE_ADAPTED_WINDOW_EXECUTED",
            "pose_gate_pass": False,
            "pose_pass_side_frames": 0,
            "rotation_valid_side_frames": 32,
            "adoption": "NOT_ADOPTED",
            "full_session_executed": False,
        },
    )
    _write(
        audit_path,
        {
            "schema_version": "H01_OBJECTIVE_CONFLICT_AUDIT_V1",
            "solver_called": False,
            "new_candidate_created": False,
            "conclusion": {
                "rotation_penalty_increased": True,
                "local_and_wrist_rotation_gradients_opposed_at_final": True,
                "hard_wrist_constraint_present": False,
            },
        },
    )
    monkeypatch.setattr(terminalize, "CANDIDATE", candidate_path)
    monkeypatch.setattr(terminalize, "OBJECTIVE_AUDIT", audit_path)
    monkeypatch.setattr(terminalize, "REPO_ROOT", tmp_path)
    progress = {
        "schema_version": "FOUR_STREAM_COMPLETION_PROGRESS_V1",
        "scope_change": {"decision": "STOP_HURO_AFTER_ROOT_CAUSE_REVIEW"},
        "huro": {
            "quality": "TERMINAL_RESEARCH_CONCLUSION_NOT_ADOPTED",
            "automatic_candidates_exhausted": True,
            "next_review": "NONE_UNLESS_USER_REOPENS_HURO_SCOPE",
            "candidate3_result": artifact_ref(candidate_path),
            "rotation_audit": artifact_ref(audit_path),
        },
    }
    return progress, candidate_path, audit_path


def test_terminal_evidence_accepts_frozen_rejection(tmp_path, monkeypatch):
    progress, candidate, audit = _evidence_fixture(tmp_path, monkeypatch)
    result = terminalize.validate_terminal_evidence(progress)
    assert result == {"candidate": artifact_ref(candidate), "objective_audit": artifact_ref(audit)}


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("scope_change", "decision", "CONTINUE_HURO", "HURO_STOP_AUTHORITY_MISSING"),
        ("huro", "automatic_candidates_exhausted", False, "HURO_CANDIDATES_NOT_EXHAUSTED"),
        ("huro", "quality", "PASS", "HURO_PROGRESS_QUALITY_MISMATCH"),
    ],
)
def test_terminal_evidence_rejects_authority_promotion(
    tmp_path, monkeypatch, section, key, value, error
):
    progress, _, _ = _evidence_fixture(tmp_path, monkeypatch)
    progress[section][key] = value
    with pytest.raises(RuntimeError, match=error):
        terminalize.validate_terminal_evidence(progress)


def test_terminal_evidence_rejects_candidate_drift(tmp_path, monkeypatch):
    progress, candidate, _ = _evidence_fixture(tmp_path, monkeypatch)
    value = json.loads(candidate.read_text(encoding="utf-8"))
    value["pose_pass_side_frames"] = 1
    _write(candidate, value)
    with pytest.raises(RuntimeError, match="HURO_CANDIDATE_SEMANTICS_MISMATCH"):
        terminalize.validate_terminal_evidence(progress)


def test_terminal_evidence_rejects_objective_drift(tmp_path, monkeypatch):
    progress, _, audit = _evidence_fixture(tmp_path, monkeypatch)
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["conclusion"]["hard_wrist_constraint_present"] = True
    _write(audit, value)
    with pytest.raises(RuntimeError, match="HURO_OBJECTIVE_AUDIT_MISMATCH"):
        terminalize.validate_terminal_evidence(progress)
