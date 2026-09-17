#!/usr/bin/env python3
"""Wait for Robot expansion closure and audit a reduced placement candidate set."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from chaoyang.ops.audit_robot_placement_candidate_reduction_v71 import audit_rows, atomic_json, ref
from chaoyang.ops.audit_robot_hand_round2_value_v71 import audit as audit_hand_round2


ROOT = Path(__file__).resolve().parents[3]
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def collect_rows(robot_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = sorted(robot_root.resolve(strict=True).glob("batch_*/arm_placement/RESULT.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load(path).get("sessions", []))
    return rows, paths


def decision(audit: dict[str, Any]) -> dict[str, Any]:
    by_task = Counter(str(row.get("task")) for row in audit["sessions"])
    cross_task = by_task.get("chips", 0) > 0 and by_task.get("poker", 0) > 0
    exact = audit["exact_winner_agreement"] == 1.0
    enough = audit["counts"]["sessions"] >= 12
    ready = cross_task and exact and enough
    return {
        "status": "READY_FOR_IMMUTABLE_SUCCESSOR_CANARY" if ready else "KEEP_FULL_CANDIDATE_SET",
        "cross_task_coverage": cross_task,
        "minimum_sample_met": enough,
        "exact_winner_agreement_met": exact,
        "sessions_by_task": dict(by_task),
        "next_action": (
            "Build a new pinned revision with one Poker canary and two frozen regressions; do not mutate R7.3."
            if ready
            else "Keep the seven-candidate set; inspect replay mismatches or missing task coverage."
        ),
    }


def hand_round2_decision(audit: dict[str, Any]) -> dict[str, Any]:
    tasks = Counter(str(row.get("task")) for row in audit["sessions"])
    cross_task = tasks.get("chips", 0) > 0 and tasks.get("poker", 0) > 0
    enough = audit["counts"]["sessions"] >= 12
    no_summary_value = audit["counts"]["summary_improved"] == 0
    ready = cross_task and enough and no_summary_value
    return {
        "status": "READY_FOR_CONDITIONAL_SKIP_CANARY" if ready else "KEEP_ROUND2",
        "cross_task_coverage": cross_task,
        "minimum_sample_met": enough,
        "no_summary_value_observed": no_summary_value,
        "sessions_by_task": dict(tasks),
        "next_action": (
            "Freeze a per-row reachability/saturation predicate and test conditional skip in a new pinned canary."
            if ready
            else "Keep round-2; inspect sessions where it improved or missing task coverage."
        ),
    }


def register_claim(result: Path) -> None:
    for _ in range(12):
        revision = int(load(STATUS_RECEIPT)["governance_revision"])
        completed = subprocess.run(
            [
                sys.executable, "-m", "chaoyang.governance.register_claim",
                "--claim", "Robot placement candidate reduction has a terminal full-expansion replay decision",
                "--status", "DEVELOPMENT_EVIDENCE",
                "--scope", "exact78/robot_geometry/placement_reduction_terminal_R7_3",
                "--claim-limit", "Retrospective completed-sweep replay only; not wall-clock speedup, Robot authority, or permission to mutate R7.3.",
                "--evidence", str(result.resolve()),
                "--expected-revision", str(revision),
            ],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        if completed.returncode == 0:
            return
        if "revision" not in (completed.stdout + completed.stderr).lower():
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(1)
    raise RuntimeError("governance CAS retry exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-soft-state", type=Path, required=True)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reduced-backoff-m", type=float, action="append", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    args = parser.parse_args()

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "AUTOMATION_STATE.json"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return 0
    started = time.monotonic()
    while True:
        upstream = load(args.hard_soft_state) if args.hard_soft_state.is_file() else {}
        state = {
            "schema_version": "robot-placement-reduction-watcher-v71-v1",
            "status": "WAITING_UPSTREAM",
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "upstream_status": upstream.get("status", "MISSING"),
            "claim_limit": "Retrospective placement replay only; no Robot, Contact, control or physical authority.",
        }
        atomic_json(state_path, state) if not state_path.exists() else _replace_json(state_path, state)
        if upstream.get("status") == "TERMINAL":
            rows, paths = collect_rows(args.robot_root)
            audit = audit_rows(rows, args.reduced_backoff_m)
            outcome = decision(audit)
            hand_audit, hand_paths = audit_hand_round2(args.robot_root)
            hand_outcome = hand_round2_decision(hand_audit)
            result = {
                "schema_version": "robot-placement-candidate-reduction-terminal-v71-v1",
                "status": "PASSED_DEVELOPMENT_TERMINAL_REPLAY",
                "created_at": state["updated_at"],
                "reduced_backoff_m": sorted(set(args.reduced_backoff_m)),
                "audit": audit,
                "decision": outcome,
                "hand_round2_audit": hand_audit,
                "hand_round2_decision": hand_outcome,
                "inputs": [ref(path) for path in paths + hand_paths],
                "authority": False,
                "claim_limit": state["claim_limit"] + " Candidate-count reduction is not measured wall-clock speedup.",
            }
            atomic_json(result_path, result)
            register_claim(result_path)
            state.update(status="TERMINAL", result=ref(result_path), decision=outcome["status"])
            _replace_json(state_path, state)
            return 0
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="UPSTREAM_WAIT_BUDGET_EXHAUSTED")
            _replace_json(state_path, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


def _replace_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
