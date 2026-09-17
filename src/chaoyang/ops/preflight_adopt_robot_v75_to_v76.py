#!/usr/bin/env python3
"""Prove exact adoption of the nine complete v7.5 terminals into v7.6."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops import run_exact78_robot_ready_batches_v75 as predecessor  # noqa: E402
from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    RobotTargetContractError,
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    verify_ref,
)


def _exact_adopt_rows(v75_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection_path = v75_run / "SELECTION.json"
    selection = load_json(selection_path)
    if selection.get("schema_version") != "exact78-robot-ready-batches-selection-v75":
        raise RobotTargetContractError("v75 selection schema required")
    expected = list(selection.get("adopted_sessions", []))
    if len(expected) != 9 or len(set(expected)) != 9:
        raise RobotTargetContractError("v75 selection must bind exactly nine adopted sessions")
    if set(expected) & set(selection.get("execution_sessions", [])):
        raise RobotTargetContractError("v75 adopted/execution sets overlap")
    rows: list[dict[str, Any]] = []
    for session in expected:
        adopt_path = v75_run / "adopted" / session / "RESULT.json"
        adopted = load_json(adopt_path)
        if (
            adopted.get("schema_version") != "robot-v75-exact-adopt-map-v1"
            or adopted.get("status") != "ADOPTED_EXACT_PHASE_CLOSURE_AND_REFINALIZED"
            or adopted.get("session") != session
        ):
            raise RobotTargetContractError(f"{session}: invalid v75 adopt map")
        terminal_path = verify_ref(adopted.get("terminal"), f"{session}.terminal")
        source_artifacts = adopted.get("source_artifacts")
        if not isinstance(source_artifacts, dict) or not source_artifacts:
            raise RobotTargetContractError(f"{session}: v75 source artifacts missing")
        verified = {
            name: artifact_ref(verify_ref(ref, f"{session}.{name}"))
            for name, ref in source_artifacts.items()
        }
        payload = {
            "task": adopted.get("task"),
            "session": session,
            "source_artifact_revision": "R7_ROBOT_5",
            "target_artifact_revision": "R7_ROBOT_6",
            "source_adopt_map": artifact_ref(adopt_path),
            "source_terminal": artifact_ref(terminal_path),
            "source_artifacts": verified,
        }
        rows.append(
            {
                **payload,
                "status": "ADOPTABLE_EXACT_V75_TERMINAL",
                "phase_closure_signature": predecessor.canonical_sha(payload),
                "authority": False,
            }
        )
    return selection, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v75-attempt", type=Path, required=True)
    parser.add_argument("--governance-status-min", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempt = args.v75_attempt.resolve(strict=True)
    run_root = attempt / "run"
    failure_path = attempt / "FAILED_RUNTIME_FINAL.json"
    failure = load_json(failure_path)
    failures: list[dict[str, Any]] = []
    if failure.get("terminal_status") != "FAILED_RUNTIME_FINAL":
        failures.append({"code": "V75_NOT_RUNTIME_FINAL"})
    if (run_root / "RESULT.json").exists():
        failures.append({"code": "V75_ROOT_RESULT_UNEXPECTED"})
    try:
        selection, rows = _exact_adopt_rows(run_root)
    except (OSError, KeyError, ValueError, RobotTargetContractError) as error:
        selection, rows = {}, []
        failures.append({"code": "V75_ADOPT_CLOSURE_INVALID", "error": str(error)})
    if rows:
        expected_error = str(failure.get("error", ""))
        if "build_contract" not in expected_error or "two runtime attempts exhausted" not in expected_error:
            failures.append({"code": "V75_FAILURE_NOT_BUILD_CONTRACT"})
    status_min = args.governance_status_min.resolve(strict=True)
    load_json(status_min)
    validation = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        governance = json.loads(validation.stdout)
    except json.JSONDecodeError:
        governance = {"status": "INVALID", "stderr_tail": validation.stderr[-2000:]}
    if validation.returncode or governance.get("status") != "PASS":
        failures.append({"code": "GOVERNANCE_NOT_PASS", "detail": governance})
    signatures = [
        {"session": row["session"], "signature": row["phase_closure_signature"]}
        for row in rows
    ]
    code_paths = (
        Path(__file__),
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v76.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
        ROOT / "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
    )
    payload = {
        "schema_version": "robot-v75-to-v76-adopt-preflight-v1",
        "artifact_revision": "R7_ROBOT_6_PREFLIGHT",
        "created_at": now_iso(),
        "task_id": "RECOVER-V75-V76",
        "terminal_status": "PASSED" if rows and not failures else "BLOCKED_REFERENCE_PROOF",
        "adopt_preflight_status": (
            "PASS_EXACT_SHA_ADOPTABLE" if rows and not failures else "FAILED_REFERENCE_PROOF"
        ),
        "counts": {
            "frozen_selection": len(selection.get("sessions", [])),
            "v75_exact_adoptable": len(rows),
            "remaining_to_execute": len(selection.get("execution_sessions", [])),
            "reference_failures": len(failures),
        },
        "adopt_bundle_signature": predecessor.canonical_sha(signatures),
        "selection": artifact_ref(run_root / "SELECTION.json"),
        "v75_failure": artifact_ref(failure_path),
        "governance_status_min": artifact_ref(status_min),
        "governance_validation": governance,
        "rows": rows,
        "failures": failures,
        "code_closure": [artifact_ref(path) for path in code_paths],
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Exact SHA adoption proof only; no v75 file is modified and no Robot authority is granted.",
    }
    atomic_new_json(args.output.resolve(), payload)
    return 0 if payload["terminal_status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
