#!/usr/bin/env python3
"""Recover the 19 Wave0 Clean rows invalidated by a deleted runtime import.

The predecessor terminals remain immutable.  This successor proves that every
selected predecessor failed before CPU preparation for the same missing module,
then executes fresh attempts under a new root and publishes a combined 58-row
matrix.  Synthetic Clean remains visual-only evidence.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from chaoyang.ops import run_exact78_clean_wave_guardian_v52_1 as clean
from chaoyang.governance.common import artifact_ref, atomic_json, load_json, now_iso


PROJECT = Path(__file__).resolve().parents[3]
PREDECESSOR = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
DEFAULT_OUTPUT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53"
TASK_ID = "exact78_wave0_clean_runtime_recovery_v53"
SELECTION_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
PREDECESSOR_AUDIT_SHA = "a5ab80ecc95aa9a899b5996aa4d1714706f099cabda72ccced90763e19e098d9"
RESTORED_CORE_SHA = "8f4654a95bebabc313086b90edf676152c81083e4d93e5463682b2244ee826de"
IMPORT_ERROR = "cannot import name 'prepare_exact78_clean_expanded_role_v3'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heartbeat(status: str, phase: str, session: str | None, gpu_id: int | None = None) -> None:
    command = [
        sys.executable,
        "-m",
        "chaoyang.governance.heartbeat_task",
        "--task-id",
        TASK_ID,
        "--pid",
        str(os.getpid()),
        "--status",
        status,
        "--phase",
        phase,
    ]
    if session:
        command.extend(["--session", session])
    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])
    completed = subprocess.run(
        command, cwd=PROJECT, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(f"governance heartbeat failed: {completed.stdout[-2000:]}")


def terminal_governance(status: str, result: Path, message: str) -> None:
    for _ in range(20):
        revision = load_json(PROJECT / "docs/governance/LONG_HORIZON_TASK_STATE.json")[
            "governance_revision"
        ]
        command = [
            sys.executable,
            "-m",
            "chaoyang.governance.update_task_state",
            "--task-id",
            TASK_ID,
            "--status",
            status,
            "--phase",
            "runtime_import_recovery_terminal",
            "--attempt",
            "1",
            "--clear-runtime",
            "--message",
            message,
            "--result",
            str(result),
            "--expected-revision",
            str(revision),
        ]
        completed = subprocess.run(
            command, cwd=PROJECT, text=True, capture_output=True, check=False
        )
        if completed.returncode == 0:
            return
    raise RuntimeError("could not publish terminal governance state after 20 CAS attempts")


def validate_and_select() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    selection_path = PREDECESSOR / "EXACT78_WAVE0_SELECTION.json"
    audit_path = PREDECESSOR / "TERMINAL_AUDIT_V52_1.json"
    if sha256(selection_path) != SELECTION_SHA:
        raise RuntimeError("Wave0 selection SHA mismatch")
    if sha256(audit_path) != PREDECESSOR_AUDIT_SHA:
        raise RuntimeError("predecessor terminal audit SHA mismatch")
    restored = PROJECT / "src/chaoyang/ops/prepare_exact78_clean_expanded_role_v3.py"
    if sha256(restored) != RESTORED_CORE_SHA:
        raise RuntimeError("restored Clean core SHA mismatch")
    selection = load_json(selection_path)
    audit = load_json(audit_path)
    matrix = load_json(Path(audit["matrix"]["path"]))
    rows_by_id = {row["session_id"]: row for row in selection["sessions"]}
    failed: list[dict[str, Any]] = []
    for terminal_row in matrix["sessions"]:
        if terminal_row["status"] != "FAILED_RUNTIME_FINAL":
            continue
        terminal = load_json(Path(terminal_row["terminal_result"]["path"]))
        attempt_import_failures = []
        for attempt_ref in terminal.get("attempts", []):
            attempt = load_json(Path(attempt_ref["path"]))
            error = attempt.get("error", "")
            if "log=" not in error:
                attempt_import_failures.append(False)
                continue
            log_path = Path(error.split("log=", 1)[1].split("; command=", 1)[0])
            attempt_import_failures.append(
                log_path.is_file() and IMPORT_ERROR in log_path.read_text(errors="replace")
            )
        if (
            terminal.get("failed_phase") != "CPU_PREPARE"
            or len(attempt_import_failures) != 3
            or not all(attempt_import_failures)
        ):
            raise RuntimeError(f"unexpected predecessor failure: {terminal_row['session']}")
        failed.append(rows_by_id[terminal_row["session"]])
    if len(failed) != 19 or len({row["session_id"] for row in failed}) != 19:
        raise RuntimeError("expected exactly 19 unique import-invalidated rows")
    return selection, matrix, failed


def ensure_root(root: Path, selection: dict[str, Any], failed: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "EXACT78_WAVE0_SELECTION.json"
    if not selection_path.exists():
        shutil.copyfile(PREDECESSOR / selection_path.name, selection_path)
    if sha256(selection_path) != SELECTION_SHA:
        raise RuntimeError("successor selection copy differs")
    recovery_selection = root / "RECOVERY_19_SELECTION.json"
    if not recovery_selection.exists():
        atomic_json(
            recovery_selection,
            {
                "schema_version": "exact78-clean-runtime-import-recovery-selection-v53",
                "created_at": now_iso(),
                "status": "FROZEN_19_RUNTIME_IMPORT_FAILURES",
                "predecessor_audit": artifact_ref(PREDECESSOR / "TERMINAL_AUDIT_V52_1.json"),
                "wave0_selection": artifact_ref(selection_path),
                "restored_core": artifact_ref(
                    PROJECT / "src/chaoyang/ops/prepare_exact78_clean_expanded_role_v3.py"
                ),
                "sessions": [
                    {"task": row["task"], "session": row["session_id"], "frames": row["frame_count"]}
                    for row in failed
                ],
                "claim_limit": "Runtime recovery selection only; no Clean quality or downstream authority.",
            },
        )
    code_closure = root / "CODE_CLOSURE.json"
    if not code_closure.exists():
        atomic_json(
            code_closure,
            {
                path.name: artifact_ref(path)
                for path in (
                    Path(__file__),
                    PROJECT / "src/chaoyang/ops/prepare_exact78_clean_expanded_role_v3.py",
                    PROJECT / "src/chaoyang/ops/prepare_exact78_clean_wave_session_v52.py",
                    PROJECT / "src/chaoyang/ops/run_exact78_clean_wave_guardian_v52_1.py",
                    PROJECT / "src/chaoyang/ops/run_generic_same_session_real_donor_v1.py",
                    PROJECT / "src/chaoyang/ops/validate_generic_same_session_real_donor_v1.py",
                    PROJECT / "src/chaoyang/ops/launch_clean_synthetic_propainter_once.py",
                    PROJECT / "src/chaoyang/ops/run_clean_synthetic_propainter_baseline.py",
                )
            },
        )


def counts(root: Path, failed: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum((root / "propainter_v1" / row["session_id"] / "RESULT.json").is_file() for row in failed)
    terminal = sum((root / "clean_terminals" / row["session_id"] / "RESULT.json").is_file() for row in failed)
    return {"selected": 19, "passed": passed, "failed": terminal, "pending": 19 - passed - terminal}


def process_one(root: Path, row: dict[str, Any], failed: list[dict[str, Any]]) -> None:
    sid = row["session_id"]
    result = root / "propainter_v1" / sid / "RESULT.json"
    terminal = root / "clean_terminals" / sid / "RESULT.json"
    if result.is_file():
        clean.validate_clean(result, row)
        return
    if terminal.is_file():
        return
    current = counts(root, failed)
    clean.state(root, "RUNNING_CPU_PREPARE", current, sid)
    receipt = root / "preparation_receipts" / f"{sid}.json"
    if not receipt.is_file():
        outcome = clean.run_cpu_phase_with_retries(
            root,
            row,
            "running_cpu_prepare",
            [
                sys.executable,
                str(PROJECT / "src/chaoyang/ops/prepare_exact78_clean_wave_session_v52.py"),
                "--plan-root",
                str(root),
                "--selection",
                str(root / "EXACT78_WAVE0_SELECTION.json"),
                "--session",
                sid,
            ],
            [
                root / "sessions" / sid,
                root / "specs" / f"{sid}_real_donor_input.json",
                root / "specs" / f"{sid}_propainter.json",
                receipt,
            ],
        )
        if outcome["status"] != "PASSED":
            clean.publish_failure_terminal(
                root, row, "FAILED_RUNTIME_FINAL", outcome["error"], "CPU_PREPARE", outcome["attempts"]
            )
            return
    donor_result = root / "real_donor_v1" / sid / "RESULT.json"
    donor_spec = root / "specs" / f"{sid}_real_donor_input.json"
    if not donor_result.is_file():
        outcome = clean.run_cpu_phase_with_retries(
            root,
            row,
            "running_real_donor_cpu",
            [sys.executable, str(clean.DONOR), "--spec", str(donor_spec)],
            [root / "real_donor_v1" / sid],
        )
        if outcome["status"] != "PASSED":
            clean.publish_failure_terminal(
                root, row, "FAILED_RUNTIME_FINAL", outcome["error"], "REAL_DONOR", outcome["attempts"]
            )
            return
    authority = root / "real_donor_v1" / sid / "INDEPENDENT_AUTHORITY.json"
    if not authority.is_file():
        outcome = clean.run_cpu_phase_with_retries(
            root,
            row,
            "validating_real_donor",
            [sys.executable, str(clean.VALIDATOR), "--result", str(donor_result), "--authority", str(authority)],
            [authority],
        )
        if outcome["status"] != "PASSED":
            clean.publish_failure_terminal(
                root,
                row,
                "FAILED_RUNTIME_FINAL",
                outcome["error"],
                "REAL_DONOR_VALIDATION",
                outcome["attempts"],
            )
            return
    clean.state(root, "WAIT_GPU_RESOURCE", counts(root, failed), sid)
    try:
        outcome = clean.run_propainter_with_retries(root, row, counts(root, failed))
    except clean.RuntimeAttemptError as error:
        attempts = [
            artifact_ref(path)
            for path in sorted((root / "sessions" / sid / "attempts").glob("attempt_*/ATTEMPT.json"))
        ]
        clean.publish_failure_terminal(
            root, row, "FAILED_RUNTIME_FINAL", str(error), "PROPAINTER", attempts
        )
        return
    if outcome["status"] == "FAILED_QUALITY_C":
        clean.publish_failure_terminal(
            root,
            row,
            "FAILED_QUALITY_C",
            outcome["error"],
            "PROPAINTER",
            [outcome["attempt_ref"]],
        )


def publish_combined(
    root: Path,
    predecessor_matrix: dict[str, Any],
    failed: list[dict[str, Any]],
) -> Path:
    recovered_ids = {row["session_id"] for row in failed}
    combined = []
    for old in predecessor_matrix["sessions"]:
        sid = old["session"]
        if sid not in recovered_ids:
            combined.append(old)
            continue
        clean_result = root / "propainter_v1" / sid / "RESULT.json"
        failure_result = root / "clean_terminals" / sid / "RESULT.json"
        if clean_result.is_file():
            status, ready, reference = "PASSED", True, artifact_ref(clean_result)
        elif failure_result.is_file():
            status, ready, reference = load_json(failure_result)["status"], False, artifact_ref(failure_result)
        else:
            raise RuntimeError(f"successor terminal gap: {sid}")
        combined.append({**old, "status": status, "clean_join_ready": ready, "terminal_result": reference})
    if len(combined) != 58 or len({row["session"] for row in combined}) != 58:
        raise RuntimeError("combined matrix uniqueness failed")
    summary: dict[str, int] = {}
    for row in combined:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    output = root / "EXACT78_WAVE0_CLEAN_COMBINED_TERMINAL_MATRIX_V53.json"
    atomic_json(
        output,
        {
            "schema_version": "exact78-clean-wave0-combined-terminal-matrix-v53",
            "created_at": now_iso(),
            "status": "PASS_58_UNIQUE_TERMINALS",
            "counts": summary,
            "predecessor_matrix": artifact_ref(Path(PREDECESSOR_AUDIT_MATRIX)),
            "recovery_selection": artifact_ref(root / "RECOVERY_19_SELECTION.json"),
            "sessions": combined,
            "claim_limit": "Only PASSED rows are visual Clean inputs; synthetic pixels are not physical truth.",
        },
    )
    return output


PREDECESSOR_AUDIT_MATRIX = str(
    Path(load_json(PREDECESSOR / "TERMINAL_AUDIT_V52_1.json")["matrix"]["path"])
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    selection, predecessor_matrix, failed = validate_and_select()
    ensure_root(root, selection, failed)
    clean.governance_heartbeat = heartbeat
    heartbeat("CLAIMED", "validate_19_runtime_failures", None)
    try:
        for row in failed:
            process_one(root, row, failed)
        matrix = publish_combined(root, predecessor_matrix, failed)
        current = counts(root, failed)
        result_path = root / "RESULT.json"
        atomic_json(
            result_path,
            {
                "schema_version": "exact78-clean-runtime-import-recovery-result-v53",
                "created_at": now_iso(),
                "status": "PASSED" if current["pending"] == 0 else "FAILED_RUNTIME_FINAL",
                "counts": current,
                "combined_matrix": artifact_ref(matrix),
                "code_closure": artifact_ref(root / "CODE_CLOSURE.json"),
                "claim_limit": "Runtime recovery result; only recovered Grade-B rows authorize visual Clean consumption.",
            },
        )
        terminal_governance("PASSED", result_path, "19-row Clean runtime-import successor reached terminal coverage")
        return 0
    except Exception as error:
        failure = root / "FAILED_RUNTIME_RESULT.json"
        if not failure.exists():
            atomic_json(
                failure,
                {
                    "schema_version": "exact78-clean-runtime-import-recovery-failure-v53",
                    "created_at": now_iso(),
                    "status": "FAILED_RUNTIME_FINAL",
                    "error": f"{type(error).__name__}: {error}",
                    "counts": counts(root, failed),
                },
            )
        terminal_governance("FAILED_RUNTIME_FINAL", failure, str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
