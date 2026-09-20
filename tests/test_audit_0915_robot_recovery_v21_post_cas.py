from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from chaoyang.ops import audit_0915_robot_recovery_v21_post_cas as post


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def prepare(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    campaign = repo / "campaign"
    dump(campaign / "ADOPTION_DECISION_V1.json", {"decision": "REJECT"})
    dump(campaign / "ALGORITHM_AUDIT_LIVE_V2.json", {"classification": "REJECTED_NO_RECOVERY"})
    dump(campaign / "FINAL_AUDIT.json", {
        "status": "PASS_PRE_CAS",
        "post_cas_governance_audit_required": True,
    })
    dump(campaign / "RESULT.json", {
        "task_id": post.TASK_ID,
        "status": "REJECTED_QUALITY",
        "campaign_classification": "REJECTED_NO_RECOVERY",
        "final_audit": post.reference(campaign / "FINAL_AUDIT.json"),
    })
    dump(campaign / "RUN_RECEIPT.json", {
        "result": post.reference(campaign / "RESULT.json"),
    })
    dump(repo / "tasks/current/INDEX.json", {"status": "PASS_NO_ACTIVE_TASKS", "task_packets": []})
    dump(repo / "docs/governance/LONG_HORIZON_TASK_STATE.json", {"next_task": None})
    finalization = repo / "FINALIZATION.json"
    dump(finalization, {"task_id": post.TASK_ID, "status": "REJECTED_QUALITY"})
    governance = repo / "GOVERNANCE.json"
    references = repo / "REFERENCES.json"
    dump(governance, {"status": "PASS"})
    dump(references, {"status": "PASS"})
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "closure"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, campaign, finalization, governance, references, head


def run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    campaign: Path,
    finalization: Path,
    governance: Path,
    references: Path,
    head: str,
    output: Path,
) -> int:
    monkeypatch.setattr(sys, "argv", [
        "audit_0915_robot_recovery_v21_post_cas",
        "--repo-root", str(repo),
        "--campaign-root", str(campaign),
        "--finalization", str(finalization),
        "--governance-report", str(governance),
        "--reference-report", str(references),
        "--expected-head", head,
        "--output", str(output),
    ])
    return post.main()


def test_post_cas_audit_passes_only_after_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, campaign, finalization, governance, references, head = prepare(tmp_path)
    output = tmp_path / "POST_AUDIT.json"
    assert run(
        monkeypatch, repo, campaign, finalization, governance, references, head, output
    ) == 0
    assert json.loads(output.read_text())["status"] == "PASS"


def test_post_cas_audit_rejects_dirty_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, campaign, finalization, governance, references, head = prepare(tmp_path)
    (repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="repository is not clean"):
        run(
            monkeypatch,
            repo,
            campaign,
            finalization,
            governance,
            references,
            head,
            tmp_path / "POST_AUDIT.json",
        )
