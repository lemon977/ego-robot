import json
from pathlib import Path

import pytest

from chaoyang.governance import four_stream_completion as completion
from chaoyang.governance.common import artifact_ref


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    attempt = tmp_path / "_run/current/four_stream_completion_20260928/attempts/attempt_0001"
    checkpoint_dir = attempt / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    evidence = attempt / "lanes/motion/EVIDENCE.json"
    _write(evidence, {"status": "PASS"})
    current = checkpoint_dir / "PROGRESS_0006.json"
    _write(current, {"task_id": completion.TASK})
    parent = {"progress_checkpoint": artifact_ref(current)}
    value = {
        "schema_version": completion.PROGRESS_SCHEMA,
        "task_id": completion.TASK,
        "predecessor": "checkpoints/PROGRESS_0006.json",
        "governance_revision_before_binding": 14202,
        "project_complete": False,
        "motion": {"evidence": artifact_ref(evidence)},
        "claims": {
            "training_executed": False,
            "raw_or_processed_modified": False,
            "physical_accuracy_proven": False,
        },
    }
    successor = checkpoint_dir / "PROGRESS_0007.json"
    _write(successor, value)
    monkeypatch.setattr(completion, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(completion, "ATTEMPT", attempt)
    return parent, successor, value, evidence


def test_checkpoint_accepts_only_consecutive_bound_successor(tmp_path, monkeypatch):
    parent, successor, value, _ = _fixture(tmp_path, monkeypatch)
    assert completion.validate_progress_checkpoint(successor, parent, 14202) == value


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "UNKNOWN", "CHECKPOINT_IDENTITY_MISMATCH"),
        ("task_id", "another_task", "CHECKPOINT_IDENTITY_MISMATCH"),
        ("predecessor", "checkpoints/PROGRESS_0005.json", "CHECKPOINT_PREDECESSOR_MISMATCH"),
        ("governance_revision_before_binding", 14201, "CHECKPOINT_REVISION_MISMATCH"),
        ("project_complete", True, "CHECKPOINT_CANNOT_COMPLETE_PROJECT"),
    ],
)
def test_checkpoint_rejects_identity_and_authority_drift(
    tmp_path, monkeypatch, field, value, error
):
    parent, successor, payload, _ = _fixture(tmp_path, monkeypatch)
    payload[field] = value
    _write(successor, payload)
    with pytest.raises(RuntimeError, match=error):
        completion.validate_progress_checkpoint(successor, parent, 14202)


def test_checkpoint_rejects_skipped_sequence(tmp_path, monkeypatch):
    parent, successor, value, _ = _fixture(tmp_path, monkeypatch)
    skipped = successor.with_name("PROGRESS_0008.json")
    _write(skipped, value)
    with pytest.raises(RuntimeError, match="CHECKPOINT_NOT_CONSECUTIVE"):
        completion.validate_progress_checkpoint(skipped, parent, 14202)


def test_checkpoint_rejects_evidence_sha_drift(tmp_path, monkeypatch):
    parent, successor, _, evidence = _fixture(tmp_path, monkeypatch)
    evidence.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="CHECKPOINT_ARTIFACT_DRIFT"):
        completion.validate_progress_checkpoint(successor, parent, 14202)


def test_checkpoint_rejects_partial_artifact_identity(tmp_path, monkeypatch):
    parent, successor, value, _ = _fixture(tmp_path, monkeypatch)
    value["motion"]["evidence"].pop("bytes")
    _write(successor, value)
    with pytest.raises(RuntimeError, match="CHECKPOINT_PARTIAL_ARTIFACT_REF"):
        completion.validate_progress_checkpoint(successor, parent, 14202)


def test_checkpoint_rejects_outside_checkpoint_directory(tmp_path, monkeypatch):
    parent, _, value, _ = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "PROGRESS_0007.json"
    _write(outside, value)
    with pytest.raises(RuntimeError, match="CHECKPOINT_SCOPE_MISMATCH"):
        completion.validate_progress_checkpoint(outside, parent, 14202)


def test_checkpoint_rejects_claim_promotion(tmp_path, monkeypatch):
    parent, successor, value, _ = _fixture(tmp_path, monkeypatch)
    value["claims"]["physical_accuracy_proven"] = True
    _write(successor, value)
    with pytest.raises(RuntimeError, match="CHECKPOINT_CLAIM_PROMOTION"):
        completion.validate_progress_checkpoint(successor, parent, 14202)
