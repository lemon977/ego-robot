#!/usr/bin/env python3
"""Close the Occlusion Silver task after the incremental watcher terminates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def terminal_task_status(report: dict[str, Any]) -> str:
    return "FAILED_QUALITY_C" if report.get("status") == "FAILED_QUALITY_C" else "PASSED"


def run_cas(command_prefix: list[str]) -> None:
    for _ in range(20):
        revision = int(load(RECEIPT)["governance_revision"])
        completed = subprocess.run(
            command_prefix + ["--expected-revision", str(revision)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode == 0:
            return
        output = (completed.stdout + completed.stderr).lower()
        if "revision mismatch" not in output and "revision conflict" not in output:
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(0.5)
    raise RuntimeError("governance CAS retry exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watcher-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_out = output / "AUTOMATION_STATE.json"
    final_out = output / "RESULT.json"
    if final_out.is_file():
        return 0
    started = time.monotonic()
    while True:
        upstream = load(args.watcher_state) if args.watcher_state.is_file() else {}
        state = {
            "schema_version": "occlusion-silver-governance-finalizer-v71-v1",
            "status": "WAITING_UPSTREAM",
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "upstream_status": upstream.get("status", "MISSING"),
        }
        atomic_json(state_out, state)
        if upstream.get("status") == "TERMINAL":
            report_path = args.watcher_state.parent / "OCCLUSION_SILVER_REPORT.json"
            if not report_path.is_file():
                raise RuntimeError("terminal watcher missing OCCLUSION_SILVER_REPORT.json")
            # Let the heartbeat mirror observe worker exit before publishing the
            # final task state, preventing a late PENDING update from winning.
            worker_pid = int(upstream.get("pid", 0))
            deadline = time.monotonic() + 90
            while worker_pid > 0 and Path(f"/proc/{worker_pid}").exists() and time.monotonic() < deadline:
                time.sleep(1)
            time.sleep(35)
            report = load(report_path)
            task_status = terminal_task_status(report)
            result = {
                "schema_version": "occlusion-silver-governance-finalizer-result-v71-v1",
                "status": "PASSED_TERMINAL_ACCOUNTING",
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "occlusion_task_status": task_status,
                "silver_report": str(report_path.resolve()),
                "claim_limit": report["claim_limit"],
            }
            atomic_json(final_out, result)
            run_cas([
                sys.executable, "-m", "chaoyang.governance.update_task_state",
                "--task-id", "occlusion_silver_v1", "--status", task_status,
                "--phase", "visible_surface_terminal_full_silver_unclosed",
                "--clear-runtime", "--result", str(report_path.resolve()),
                "--message", "Incremental visible-surface diagnostics closed; full Silver contract remains unclosed.",
            ])
            run_cas([
                sys.executable, "-m", "chaoyang.governance.register_claim",
                "--claim", "Occlusion visible-surface expansion has a finite terminal report",
                "--status", "DEVELOPMENT_EVIDENCE",
                "--scope", "exact78/occlusion/visible_surface_terminal_R7_3",
                "--claim-limit", report["claim_limit"],
                "--evidence", str(report_path.resolve()),
            ])
            state.update(status="TERMINAL", result=str(final_out))
            atomic_json(state_out, state)
            return 0
        if upstream.get("status") == "BLOCKED_RESOURCE":
            state.update(status="BLOCKED_RESOURCE", reason=upstream.get("reason"))
            atomic_json(final_out, state)
            return 3
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="WATCHER_WAIT_BUDGET_EXHAUSTED")
            atomic_json(final_out, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
