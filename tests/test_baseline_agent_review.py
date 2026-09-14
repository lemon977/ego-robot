from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import validate_baseline_agent_review as validator


def reference(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": validator.digest(path)}


def review(tmp_path: Path, grade: str, gate: str, authorized: bool) -> Path:
    artifact = tmp_path / "video.mp4"
    artifact.write_bytes(b"video")
    value = {
        "schema_version": "baseline-agent-stage-review-v1",
        "created_at": "2026-09-08T17:00:00+08:00",
        "run_id": "20260908_two_task_e2e_baseline_v1",
        "stage": "MASK",
        "task": "chips",
        "session": "get_potato_chips_0902_034",
        "grade": grade,
        "downstream_authorized": authorized,
        "hard_gates": {"identity": gate},
        "soft_defects": [],
        "inputs": {},
        "artifacts": {"review_video": reference(artifact)},
        "metrics": {},
        "claim_limit": "baseline only",
    }
    path = tmp_path / "AGENT_REVIEW.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(("grade", "gate", "authorized"), [("A", "PASS", True), ("B", "PASS", True), ("C", "FAIL", False)])
def test_valid_grade_contract(tmp_path: Path, grade: str, gate: str, authorized: bool) -> None:
    assert validator.validate(review(tmp_path, grade, gate, authorized))["grade"] == grade


def test_ab_cannot_contain_failed_hard_gate(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        validator.validate(review(tmp_path, "B", "FAIL", True))


def test_c_requires_a_failed_hard_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hard-gate/grade"):
        validator.validate(review(tmp_path, "C", "PASS", False))


def test_artifact_sha_drift_is_rejected(tmp_path: Path) -> None:
    path = review(tmp_path, "A", "PASS", True)
    (tmp_path / "video.mp4").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes/SHA"):
        validator.validate(path)
