#!/usr/bin/env python3
"""Bounded ledger finalizer after Clean and S1 development automation.

The script performs no model inference and promotes no Mask/Robot/Contact
authority.  It waits for the immutable post-Clean S1 receipts, runs the
receipt-bound regression set, publishes the 156-row finite terminal matrix,
and records task outcomes through the governance CAS writer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
POST_STATE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_clean_s1/AUTOMATION_STATE.json"
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_s1_finalize"
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
REGRESSION = ROOT / "docs/governance/CURRENT_REGRESSION_MANIFEST.json"
EXACT_RESULT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/exact78_terminal/RESULT.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def development_task_status(raw: str) -> str:
    if raw in {"PASSED", "COMPARISON_COMPLETE_REVIEW_REQUIRED"}:
        return "PASSED"
    if raw in {"FAILED_QUALITY_C", "NO_GO_BACKEND_EXECUTION"}:
        return "FAILED_QUALITY_C"
    if raw in {"BLOCKED_RESOURCE", "BLOCKED_PREREQ", "BLOCKED_REFERENCE_PROOF"}:
        return raw
    return "FAILED_RUNTIME_FINAL"


def run_logged(command: list[str], log_path: Path, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            return subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                text=True, timeout=timeout, check=False,
            ).returncode
    except subprocess.TimeoutExpired:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nAUTOMATION_TIMEOUT\n")
        return 124


def update_task(task_id: str, status: str, phase: str, result: Path) -> None:
    """Retry only CAS conflicts; all semantic inputs stay fixed."""
    for _ in range(8):
        revision = int(load_json(STATUS_RECEIPT)["governance_revision"])
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", task_id, "--status", status, "--phase", phase,
            "--clear-runtime", "--result", str(result.resolve()),
            "--message", "V7.1 bounded post-S1 automation receipt aggregation",
            "--expected-revision", str(revision),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return
        if "revision conflict" not in (completed.stdout + completed.stderr).lower():
            raise RuntimeError((completed.stdout + completed.stderr)[-2000:])
        time.sleep(1)
    raise RuntimeError(f"governance CAS retry budget exhausted: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-state", type=Path, default=POST_STATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    post_state = args.post_state.resolve()
    output = args.output.resolve()
    state_path = output / "AUTOMATION_STATE.json"
    state = {
        "schema_version": "v71-post-s1-finalize-v1",
        "status": "DRY_RUN_READY" if args.dry_run else "WAITING_POST_S1",
        "pid": os.getpid(),
        "post_state": str(post_state),
        "claim_limit": "Finite accounting and development receipts only; no Mask/Robot/Contact/Gold/physical authority promotion.",
    }
    atomic_json(state_path, state)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False))
        return 0

    started = time.monotonic()
    while True:
        post = load_json(post_state) if post_state.is_file() else {}
        if post.get("status") == "TERMINAL":
            break
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="POST_S1_WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        state.update(updated_at=datetime.now().astimezone().isoformat(timespec="seconds"), post_status=post.get("status", "MISSING"))
        atomic_json(state_path, state)
        time.sleep(max(5, args.poll_seconds))

    validate_log = output / "validate_governance.log"
    validate_rc = run_logged(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        validate_log, 300,
    )
    manifest = load_json(REGRESSION)
    tests = [str(Path(row["path"]).resolve()) for row in manifest["tests"]]
    regression_log = output / "current_regression.log"
    regression_rc = run_logged(
        [sys.executable, "-m", "pytest", "-q", *tests], regression_log, 1800,
    )
    exact_log = output / "exact78_terminal_matrix.log"
    exact_rc = run_logged(
        [sys.executable, "src/chaoyang/ops/build_exact78_final_terminal_matrix_v71.py"], exact_log, 600,
    )
    post_results = {row["task"] if "task" in row else row.get("backend", "unknown"): row for row in post["results"]}
    role = post_results.get("role_mask_1_plus_2_bundle", {})
    comparison = post_results.get("s1_development_comparison", {})
    passed = validate_rc == regression_rc == exact_rc == 0 and EXACT_RESULT.is_file()
    result = {
        "schema_version": "v71-post-s1-finalize-result-v1",
        "status": "PASSED" if passed else "FAILED_RUNTIME_FINAL",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "return_codes": {"governance": validate_rc, "regression": regression_rc, "exact78_matrix": exact_rc},
        "post_s1": file_ref(post_state),
        "exact78_result": file_ref(EXACT_RESULT) if EXACT_RESULT.is_file() else None,
        "logs": [file_ref(path) for path in (validate_log, regression_log, exact_log)],
        "claim_limit": state["claim_limit"],
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    if passed:
        update_task("exact78_clean_r70_v71", "PASSED", "exact78_156_terminal_matrix_published", EXACT_RESULT)
    if role.get("result") and Path(role["result"]).is_file():
        update_task(
            "successor_role_mask_v71", development_task_status(str(role.get("status"))),
            "bounded_runtime_contract_canary_terminal", Path(role["result"]),
        )
    if comparison.get("result") and Path(comparison["result"]).is_file():
        update_task(
            "successor_object_identity_v71", development_task_status(str(comparison.get("status"))),
            "dual_backend_development_comparison_terminal_no_authority", Path(comparison["result"]),
        )
    state.update(status="TERMINAL", result=file_ref(result_path))
    atomic_json(state_path, state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
