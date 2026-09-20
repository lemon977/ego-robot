#!/usr/bin/env python3
"""Verify the post-CAS, post-commit closure of the 0915 recovery campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


TASK_ID = "0915_robot_quality_recovery_15h_v2"
REQUIRED_CAMPAIGN_FILES = (
    "ADOPTION_DECISION_V1.json",
    "ALGORITHM_AUDIT_LIVE_V2.json",
    "FINAL_AUDIT.json",
    "RESULT.json",
    "RUN_RECEIPT.json",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def reference(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def validate_ref(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("missing artifact reference")
    path = Path(str(value.get("path", "")))
    if not path.is_file():
        raise RuntimeError(f"artifact missing: {path}")
    if path.stat().st_size != value.get("bytes") or digest(path) != value.get("sha256"):
        raise RuntimeError(f"artifact reference drift: {path}")


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--finalization", required=True, type=Path)
    parser.add_argument("--governance-report", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve(strict=True)
    campaign = args.campaign_root.resolve(strict=True)
    files = {name: campaign / name for name in REQUIRED_CAMPAIGN_FILES}
    for path in files.values():
        if not path.is_file():
            raise RuntimeError(f"campaign closure missing: {path}")

    result = load(files["RESULT.json"])
    final_audit = load(files["FINAL_AUDIT.json"])
    run_receipt = load(files["RUN_RECEIPT.json"])
    if result.get("task_id") != TASK_ID or result.get("status") != "REJECTED_QUALITY":
        raise RuntimeError("unexpected campaign terminal")
    if final_audit.get("status") != "PASS_PRE_CAS" or not final_audit.get(
        "post_cas_governance_audit_required"
    ):
        raise RuntimeError("campaign did not request this post-CAS audit")
    validate_ref(result.get("final_audit"))
    validate_ref(run_receipt.get("result"))

    finalization = load(args.finalization.resolve(strict=True))
    if finalization.get("task_id") != TASK_ID or finalization.get("status") != "REJECTED_QUALITY":
        raise RuntimeError("CAS finalization identity/status mismatch")
    current = load(repo / "tasks/current/INDEX.json")
    if current.get("status") != "PASS_NO_ACTIVE_TASKS" or current.get("task_packets") != []:
        raise RuntimeError("current task index is not terminal and empty")
    long_state = load(repo / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if long_state.get("next_task") is not None:
        raise RuntimeError("long-horizon state still routes a task")

    governance = load(args.governance_report.resolve(strict=True))
    references = load(args.reference_report.resolve(strict=True))
    if governance.get("status") != "PASS":
        raise RuntimeError("post-CAS governance is not PASS")
    if references.get("status") != "PASS":
        raise RuntimeError("post-CAS reference closure is not PASS")

    head = git(repo, "rev-parse", "HEAD")
    if head != args.expected_head:
        raise RuntimeError(f"unexpected HEAD: {head}")
    porcelain = git(repo, "status", "--porcelain")
    if porcelain:
        raise RuntimeError("repository is not clean after final commit")

    payload = {
        "schema_version": "0915-robot-recovery-v21-post-cas-audit-v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "campaign_terminal": result["status"],
        "campaign_classification": result.get("campaign_classification"),
        "governance": reference(args.governance_report),
        "reference_closure": reference(args.reference_report),
        "finalization": reference(args.finalization),
        "result": reference(files["RESULT.json"]),
        "git_head": head,
        "git_status_porcelain": "",
        "claim_limit": "Post-CAS closure PASS proves bookkeeping and artifact integrity only; algorithm outcome remains REJECTED_NO_RECOVERY.",
    }
    write_new(args.output.resolve(), payload)
    print(json.dumps({"status": "PASS", "output": reference(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
