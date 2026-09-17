#!/usr/bin/env python3
"""Incrementally build causal Visual Aux bundles from hard-feasible Robot candidates.

This worker is intentionally upstream of formal Visual Aux training.  It only
pre-builds immutable session bundles in the formal bundle directory; the
existing post-Robot trainer remains responsible for dataset-volume gates,
epoch-0, checkpoints, and governance publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def exact(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference["path"])).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_bounded(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
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
            handle.write(f"\nBUNDLE_TIMEOUT_SECONDS={timeout}\n")
            return 124


def validate_bundle(bundle: Path) -> dict[str, Any] | None:
    result_path = bundle / "RESULT.json"
    manifest_path = bundle / "VISUAL_AUX_SESSION_MANIFEST.json"
    if not result_path.is_file() or not manifest_path.is_file():
        return None
    result = load(result_path)
    manifest = load(manifest_path)
    if result.get("status") != "PASS_DEVELOPMENT_VISUAL_AUX_BUNDLE_NO_OCCLUSION_AUTHORITY":
        return None
    if manifest.get("input_mode") != "CAUSAL_TRAINING_INPUT":
        raise RuntimeError(f"non-causal bundle: {bundle}")
    if manifest.get("causal_proof", {}).get("enabled") is not True:
        raise RuntimeError(f"missing causal proof: {bundle}")
    if manifest.get("h50_eligibility_mode") != "ANY_ENDPOINT_40_OF_50":
        raise RuntimeError(f"unexpected H50 eligibility mode: {bundle}")
    return {
        "status": "READY",
        "eligible_h50_windows": int(result["eligible_h50_windows"]),
        "rgb_valid_pixel_fraction": float(result["rgb_valid_pixel_fraction"]),
        "result": ref(result_path),
        "manifest": ref(manifest_path),
    }


def matrix_rows(path: Path) -> dict[str, dict[str, Any]]:
    value = load(path)
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"matrix rows missing: {path}")
    indexed = {str(row["session_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError("duplicate session_id in matrix")
    return indexed


def build_one(
    candidate_result: Path,
    matrix_row: dict[str, Any],
    bundle_root: Path,
    log_root: Path,
    timeout: int,
) -> dict[str, Any]:
    candidate = load(candidate_result)
    session = str(candidate["session"])
    bundle = bundle_root / session
    adopted = validate_bundle(bundle)
    if adopted is not None:
        return {"session": session, "task": matrix_row["task"], "split": matrix_row["split"], **adopted}
    clean_reference = matrix_row.get("clean_result")
    if matrix_row.get("clean_state") != "PASSED_GRADE_B" or not isinstance(clean_reference, dict):
        return {
            "session": session,
            "task": matrix_row.get("task"),
            "split": matrix_row.get("split"),
            "status": "BLOCKED_PREREQ_CLEAN",
        }
    clean_result = exact(clean_reference, f"{session}.clean")
    if bundle.exists():
        return {
            "session": session,
            "task": matrix_row["task"],
            "split": matrix_row["split"],
            "status": "FAILED_RUNTIME_FINAL",
            "reason": "NONTERMINAL_BUNDLE_DIRECTORY_EXISTS",
        }
    return_code = None
    for attempt in range(1, 3):
        log = log_root / f"bundle_{session}_attempt_{attempt:04d}.log"
        if log.exists():
            continue
        return_code = run_bounded(
            [
                sys.executable,
                "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py",
                "--robot-review-result",
                str(candidate_result.resolve(strict=True)),
                "--clean-result",
                str(clean_result),
                "--split",
                str(matrix_row["split"]),
                "--output-root",
                str(bundle_root),
                "--causal-training",
                "--h50-eligibility-mode",
                "ANY_ENDPOINT_40_OF_50",
            ],
            log,
            timeout,
        )
        adopted = validate_bundle(bundle)
        if adopted is not None:
            return {
                "session": session,
                "task": matrix_row["task"],
                "split": matrix_row["split"],
                "return_code": return_code,
                **adopted,
            }
    return {
        "session": session,
        "task": matrix_row["task"],
        "split": matrix_row["split"],
        "status": "FAILED_RUNTIME_FINAL",
        "return_code": return_code,
        "reason": "TWO_BOUNDED_ATTEMPTS_EXHAUSTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-state", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    parser.add_argument("--bundle-timeout-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    candidate_state = args.candidate_state.resolve()
    matrix_path = args.matrix.resolve(strict=True)
    bundle_root = args.bundle_root.resolve()
    output = args.output_root.resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    rows_by_session = matrix_rows(matrix_path)
    state_path = output / "AUTOMATION_STATE.json"
    result_path = output / "RESULT.json"
    started = time.monotonic()
    terminals: dict[str, dict[str, Any]] = {}

    while True:
        for result in sorted(candidate_root.glob("*/RESULT.json")):
            session = result.parent.name
            if session in terminals:
                continue
            matrix_row = rows_by_session.get(session)
            if matrix_row is None:
                terminals[session] = {"session": session, "status": "BLOCKED_PREREQ_NOT_IN_MATRIX"}
                continue
            atomic_json(state_path, {
                "schema_version": "visual-aux-candidate-bundle-watcher-v71-v1",
                "status": "RUNNING",
                "pid": os.getpid(),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "phase": "BUILD_CAUSAL_BUNDLE",
                "current_session": session,
                "terminal_count": len(terminals),
                "claim_limit": "Prebuilt causal Visual Aux bundles only; no Robot, Contact, occlusion, action, policy or deployment authority.",
            })
            terminals[session] = build_one(
                result,
                matrix_row,
                bundle_root,
                output / "logs",
                args.bundle_timeout_seconds,
            )
            atomic_json(output / "INCREMENTAL_RESULT.json", {
                "schema_version": "visual-aux-candidate-bundle-incremental-v71-v1",
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rows": [terminals[key] for key in sorted(terminals)],
                "claim_limit": "Incremental causal bundle receipts only; no authority.",
            })

        upstream = load(candidate_state) if candidate_state.is_file() else {}
        upstream_terminal = upstream.get("status") == "TERMINAL"
        candidate_count = len(list(candidate_root.glob("*/RESULT.json")))
        state = {
            "schema_version": "visual-aux-candidate-bundle-watcher-v71-v1",
            "status": "TERMINAL" if upstream_terminal and len(terminals) == candidate_count else "RUNNING",
            "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "candidate_count": candidate_count,
            "terminal_count": len(terminals),
            "ready_bundle_count": sum(row["status"] == "READY" for row in terminals.values()),
            "eligible_h50_windows": sum(int(row.get("eligible_h50_windows", 0)) for row in terminals.values()),
            "upstream_status": upstream.get("status", "MISSING"),
            "claim_limit": "Prebuilt causal Visual Aux bundles only; no Robot, Contact, occlusion, action, policy or deployment authority.",
        }
        atomic_json(state_path, state)
        if state["status"] == "TERMINAL" or args.once:
            atomic_json(result_path, {
                "schema_version": "visual-aux-candidate-bundle-index-v71-v1",
                "status": "PASSED_TERMINAL_COVERAGE" if state["status"] == "TERMINAL" else "PASSED_INCREMENTAL_SNAPSHOT",
                "created_at": state["updated_at"],
                "matrix": ref(matrix_path),
                "candidate_state": ref(candidate_state) if candidate_state.is_file() else None,
                "counts": {
                    "candidate": candidate_count,
                    "terminal": len(terminals),
                    "ready": state["ready_bundle_count"],
                    "eligible_h50_windows": state["eligible_h50_windows"],
                },
                "rows": [terminals[key] for key in sorted(terminals)],
                "control_ground_truth": False,
                "claim_limit": state["claim_limit"],
            })
            return 0
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
