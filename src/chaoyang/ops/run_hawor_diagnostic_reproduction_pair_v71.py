#!/usr/bin/env python3
"""Run the two frozen CPU-only HaWoR temporal diagnostic contracts finitely."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
GOVERNANCE_PYTHON = Path(os.environ.get("CHA0YANG_GOVERNANCE_PYTHON", "/usr/local/bin/python"))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    canaries = [
        row
        for result in results.values()
        for row in result.get("canaries", result.get("rows", []))
    ]
    counts = Counter(str(row.get("status")) for row in canaries)
    return {
        "contracts": len(results),
        "canaries": len(canaries),
        "by_status": dict(counts),
        "numeric_review_ready": counts.get("PASS_NUMERIC_NEEDS_HUMAN_REVIEW", 0),
        "numeric_hold": sum(value for key, value in counts.items() if key.startswith("HOLD")),
        "runtime_exception": counts.get("HOLD_EXCEPTION", 0),
    }


def execution_complete(
    results: dict[str, dict[str, Any]], runtime: dict[str, dict[str, Any]], expected_canaries: int,
) -> bool:
    """Quality HOLD (exit 2) is a completed diagnostic, not a runtime failure."""
    summary = summarize(results)
    return (
        len(results) == 2
        and all(item["return_code"] in {0, 2} for item in runtime.values())
        and summary["canaries"] == expected_canaries
        and summary["runtime_exception"] == 0
    )


def run_one(contract: Path, output: Path, log: Path, timeout: int, child_python: Path) -> int:
    if (output / "RESULT.json").is_file():
        return 0
    if output.exists():
        raise RuntimeError(f"nonterminal diagnostic output exists: {output}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(child_python), "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py", "--contract", str(contract), "--output-root", str(output)],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            handle.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


def register_claim(result: Path, passed: bool) -> None:
    for _ in range(20):
        revision = int(load(RECEIPT)["governance_revision"])
        completed = subprocess.run([
            str(GOVERNANCE_PYTHON), "-m", "chaoyang.governance.register_claim",
            "--claim", "Frozen HaWoR temporal diagnostic contracts have executable CPU reproductions",
            "--status", "DEVELOPMENT_EVIDENCE" if passed else "WITHDRAWN",
            "--scope", "exact78/hawor/temporal_diagnostic_reproduction_R7_2",
            "--claim-limit", (
                "CPU post-processing diagnostics over existing HaWoR tracks only; not HaWoR re-inference, weight provenance, successor authority, external 3D truth or Wave delta."
                if passed else
                "The earlier pair summary was a false pass caused by reading rows instead of canaries; this immutable successor records the bounded runtime failure and withdraws that conclusion."
            ),
            "--evidence", str(result.resolve()), "--expected-revision", str(revision),
        ], cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return
        if "revision mismatch" not in (completed.stdout + completed.stderr).lower():
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(0.5)
    raise RuntimeError("governance CAS retry exhausted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips-contract", type=Path, required=True)
    parser.add_argument("--poker-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-timeout-seconds", type=int, default=7200)
    parser.add_argument("--child-python", type=Path, default=ROOT / "src/chaoyang/ops/hawor_python.sh")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = output / "RESULT.json"
    if final.exists():
        return 0
    results: dict[str, dict[str, Any]] = {}
    runtime = {}
    expected_canaries = 0
    for task, contract in (("poker", args.poker_contract), ("chips", args.chips_contract)):
        expected_canaries += len(load(contract.resolve(strict=True)).get("canaries", []))
        destination = output / task
        code = run_one(
            contract.resolve(strict=True), destination, output / "logs" / f"{task}.log",
            args.contract_timeout_seconds, args.child_python.resolve(strict=True),
        )
        result_path = destination / "RESULT.json"
        runtime[task] = {"return_code": code, "result": str(result_path) if result_path.is_file() else None}
        if result_path.is_file():
            results[task] = load(result_path)
    summary = summarize(results)
    passed = execution_complete(results, runtime, expected_canaries)
    payload = {
        "schema_version": "hawor-diagnostic-reproduction-pair-v71-v1",
        "status": "PASSED_DEVELOPMENT_DIAGNOSTIC_REPRODUCTION" if passed else "FAILED_RUNTIME_FINAL",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "runtime": runtime,
        "authority": False,
        "wave_delta_published": False,
        "claim_limit": "CPU temporal post-processing over existing HaWoR tracks only; not model re-inference, recovered weight provenance, successor authority, external truth or permission to change the frozen cohort.",
    }
    atomic_json(final, payload)
    register_claim(final, passed)
    print(json.dumps({"status": payload["status"], "summary": summary, "output": str(final)}, ensure_ascii=False))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
