#!/usr/bin/env python3
"""Bounded CAS finalizer for the immutable Role Mask R7_2 recovery canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
DEFAULT_RESULT = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/"
    "role_mask_runtime_contract_repair_poker042/R7_2/sessions/"
    "play_cards_0901_042/attempts/attempt_0001/RESULT.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    result = args.result.resolve()
    started = time.monotonic()
    while not result.is_file():
        if time.monotonic() - started >= args.max_wait_seconds:
            return 3
        time.sleep(max(args.poll_seconds, 5))
    raw = str(load(result).get("status"))
    if raw == "PASSED":
        status = "BLOCKED_PREREQ"
        phase = "r72_canary_passed_two_regressions_required_before_authority"
    elif raw in {"FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE"}:
        status = raw
        phase = "r72_recovery_canary_terminal"
    else:
        status = "FAILED_RUNTIME_FINAL"
        phase = "r72_unrecognized_terminal_receipt"
    for _ in range(12):
        revision = int(load(RECEIPT)["governance_revision"])
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", "successor_role_mask_v71", "--status", status,
            "--phase", phase, "--clear-runtime", "--result", str(result),
            "--message", "R7.2 canonical-tree recovery canary aggregation; no authority promotion",
            "--expected-revision", str(revision),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode == 0:
            return 0
        if "revision conflict" not in (completed.stdout + completed.stderr).lower():
            return 2
        time.sleep(1)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
