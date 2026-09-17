#!/usr/bin/env python3
"""V7.6 Robot runner: fix only the builder-compatible matrix snapshot layout.

V7.5 copied the current matrix to ``run/input_snapshot/<content-name>.json``.
The unchanged v5.2 contract builder deliberately requires a byte-identical
content-addressed file at ``<matrix-parent>/snapshots/<fixed-name>_<sha>.json``.
V7.6 creates both files in a fresh immutable run and otherwise delegates the
Robot phase DAG unchanged to v5.4/v5.3.
"""

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
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops import run_exact78_robot_ready_batches_v53 as legacy  # noqa: E402
from chaoyang.ops import run_exact78_robot_ready_batches_v54 as anchor_v3  # noqa: E402
from chaoyang.ops import run_exact78_robot_ready_batches_v75 as predecessor  # noqa: E402
from chaoyang.governance.v52_contracts import build_run_signature, claim_executor, process_identity  # noqa: E402
from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    RobotTargetContractError,
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    verify_ref,
)

TASK_ID = "robot_geometry_expansion_v76"
ARTIFACT_REVISION = "R7_ROBOT_6"
MATRIX_NAME = "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"


def freeze_builder_compatible_matrix(
    matrix_path: Path, run_root: Path
) -> tuple[Path, dict[str, Any]]:
    """Freeze the live copy and the exact companion required by the v5.2 builder."""
    source = matrix_path.resolve(strict=True)
    source_ref = artifact_ref(source)
    live_copy = run_root / "input_snapshot" / MATRIX_NAME
    companion = (
        live_copy.parent
        / "snapshots"
        / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{source_ref['sha256']}.json"
    )
    predecessor.write_exact_new(live_copy, source.read_bytes())
    predecessor.write_exact_new(companion, source.read_bytes())
    live_ref = artifact_ref(live_copy)
    companion_ref = artifact_ref(companion)
    for label, reference in (("live copy", live_ref), ("companion", companion_ref)):
        if (
            reference["bytes"] != source_ref["bytes"]
            or reference["sha256"] != source_ref["sha256"]
        ):
            raise RobotTargetContractError(f"matrix {label} byte/SHA mismatch")
    return live_copy, {
        "source": source_ref,
        "builder_input": live_ref,
        "content_addressed_snapshot": companion_ref,
    }


def verify_adopt_preflight(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = load_json(path.resolve(strict=True))
    if value.get("schema_version") != "robot-v75-to-v76-adopt-preflight-v1":
        raise RobotTargetContractError("v75-to-v76 adopt preflight schema required")
    if value.get("adopt_preflight_status") != "PASS_EXACT_SHA_ADOPTABLE":
        raise RobotTargetContractError("v76 adopt preflight is not exact-adoptable")
    if value.get("failures"):
        raise RobotTargetContractError("v76 adopt preflight contains failures")
    closure = value.get("code_closure")
    if not isinstance(closure, list):
        raise RobotTargetContractError("v76 code closure is required")
    by_path = {str(item.get("path")): item for item in closure if isinstance(item, dict)}
    required = (
        ROOT / "src/chaoyang/ops/preflight_adopt_robot_v75_to_v76.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v76.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
        ROOT / "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
    )
    for code in required:
        reference = artifact_ref(code)
        if by_path.get(reference["path"]) != reference:
            raise RobotTargetContractError(f"v76 adopt code closure is stale: {code}")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise RobotTargetContractError("v76 requires exactly nine exact-adopt rows")
    sessions: set[str] = set()
    signatures: list[dict[str, str]] = []
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or not session or session in sessions:
            raise RobotTargetContractError("v76 adopt sessions must be unique")
        sessions.add(session)
        if (
            row.get("source_artifact_revision") != "R7_ROBOT_5"
            or row.get("target_artifact_revision") != ARTIFACT_REVISION
        ):
            raise RobotTargetContractError(f"{session}: v76 revision mismatch")
        source_map = verify_ref(row.get("source_adopt_map"), f"{session}.source_adopt_map")
        terminal = verify_ref(row.get("source_terminal"), f"{session}.source_terminal")
        source_artifacts = row.get("source_artifacts")
        if not isinstance(source_artifacts, dict) or not source_artifacts:
            raise RobotTargetContractError(f"{session}: source artifacts missing")
        verified = {
            name: artifact_ref(verify_ref(ref, f"{session}.{name}"))
            for name, ref in source_artifacts.items()
        }
        payload = {
            "task": row.get("task"),
            "session": session,
            "source_artifact_revision": "R7_ROBOT_5",
            "target_artifact_revision": ARTIFACT_REVISION,
            "source_adopt_map": artifact_ref(source_map),
            "source_terminal": artifact_ref(terminal),
            "source_artifacts": verified,
        }
        signature = predecessor.canonical_sha(payload)
        if signature != row.get("phase_closure_signature"):
            raise RobotTargetContractError(f"{session}: v76 phase closure signature mismatch")
        signatures.append({"session": session, "signature": signature})
    if predecessor.canonical_sha(signatures) != value.get("adopt_bundle_signature"):
        raise RobotTargetContractError("v76 adopt bundle signature mismatch")
    return value, rows


def install_fenced_heartbeat(
    task_id: str, task_packet_path: Path, claim: dict[str, Any]
) -> None:
    original = legacy.heartbeat

    def fenced(status: str, phase: str, session: str | None = None) -> None:
        identity = process_identity(os.getpid())
        if (
            not identity.get("alive")
            or identity.get("start_ticks") != claim.get("proc_start_ticks")
            or claim.get("pid") != os.getpid()
        ):
            raise RobotTargetContractError("v76 PID/startticks fencing failed")
        live_claim = load_json(task_packet_path.parent / "EXECUTOR_CLAIM.json")
        if live_claim.get("fencing_token") != claim.get("fencing_token"):
            raise RobotTargetContractError("v76 fencing token is no longer current")
        task_state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
        row = next(
            (item for item in task_state.get("tasks", []) if item.get("task_id") == task_id),
            None,
        )
        if row is None:
            raise RobotTargetContractError("v76 task row disappeared")
        predecessor.validate_task_row_registration(row, task_packet_path)
        original(status, phase, session)

    legacy.heartbeat = fenced


def publish_task_terminal(
    task_id: str,
    result_path: Path,
    phase: str,
    *,
    status: str = "PASSED",
    message: str,
) -> dict[str, Any]:
    receipt_path = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    for attempt in range(20):
        revision = int(load_json(receipt_path)["governance_revision"])
        command = [
            sys.executable,
            "-m",
            "chaoyang.governance.update_task_state",
            "--task-id",
            task_id,
            "--status",
            status,
            "--phase",
            phase,
            "--clear-runtime",
            "--result",
            str(result_path.resolve(strict=True)),
            "--expected-revision",
            str(revision),
            "--message",
            message,
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return json.loads(completed.stdout)
        combined = completed.stdout + completed.stderr
        if "CAS revision mismatch" not in combined or attempt == 19:
            raise RobotTargetContractError(f"v76 terminal publication failed: {combined[-4000:]}")
        time.sleep(min(0.1 * (attempt + 1), 1.0))
    raise RobotTargetContractError("v76 terminal publication attempts exhausted")


def finalize_adopted(run_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adopted = []
    for row in rows:
        result = {
            "schema_version": "robot-v76-exact-adopt-map-v1",
            "artifact_revision": ARTIFACT_REVISION,
            "created_at": now_iso(),
            "task": row["task"],
            "session": row["session"],
            "status": "ADOPTED_EXACT_V75_TERMINAL",
            "phase_closure_signature": row["phase_closure_signature"],
            "source_adopt_map": row["source_adopt_map"],
            "source_terminal": row["source_terminal"],
            "source_artifacts": row["source_artifacts"],
            "terminal": row["source_terminal"],
            "authority": False,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
        }
        path = run_root / "adopted" / row["session"] / "RESULT.json"
        atomic_new_json(path, result)
        adopted.append({"session": row["session"], "task": row["task"], "result": artifact_ref(path)})
    return adopted


def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-preflight", type=Path, required=True)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--phase-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.phase_timeout_seconds < 60:
        raise SystemExit("--phase-timeout-seconds must be >=60")
    governance = predecessor.validate_governance()
    preflight, adopt_rows = verify_adopt_preflight(args.adopt_preflight)
    task_packet_path = args.task_packet.resolve(strict=True)
    task_packet = load_json(task_packet_path)
    if task_packet.get("task_id") != args.task_id:
        raise RobotTargetContractError("task packet/task-id mismatch")
    packet_registration = predecessor.verify_current_packet_registration(
        task_packet_path, args.task_id
    )
    task_state_path = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
    task_state = load_json(task_state_path)
    task_row = next(
        (row for row in task_state.get("tasks", []) if row.get("task_id") == args.task_id),
        None,
    )
    if task_row is None:
        raise RobotTargetContractError("governance aggregator must register v76 before launch")
    if task_row.get("status") not in {"PENDING", "CLAIMED", "BLOCKED_PREREQ"}:
        raise RobotTargetContractError(f"v76 task is not launchable: {task_row.get('status')}")
    predecessor.validate_task_row_registration(task_row, task_packet_path)

    source_selection_path = verify_ref(preflight["selection"], "v76.selection")
    frozen = list(load_json(source_selection_path).get("sessions", []))
    adopted_sessions, remaining = predecessor.split_frozen_sessions(frozen, adopt_rows)
    matrix_path = args.matrix.resolve(strict=True)
    matrix = load_json(matrix_path)
    matrix_rows = predecessor.validate_execution_readiness(matrix, frozen, remaining)
    run_root = args.output_root.resolve()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"fresh immutable output root required: {run_root}")

    input_manifest_path = task_packet_path.parent / "RUN_INPUT_MANIFEST.json"
    code_closure_path = task_packet_path.parent / "RUN_CODE_CLOSURE.json"
    code_paths = (
        Path(__file__),
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
        ROOT / "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
        ROOT / "src/chaoyang/ops/preflight_exact78_robot_ready_batch_v52.py",
        ROOT / "src/chaoyang/ops/preflight_exact78_robot_ready_anchor_v3.py",
    )
    atomic_new_json(
        code_closure_path,
        {
            "schema_version": "robot-v76-code-closure-v1",
            "artifacts": [artifact_ref(path) for path in code_paths],
        },
    )
    atomic_new_json(
        input_manifest_path,
        {
            "schema_version": "robot-v76-run-input-manifest-v1",
            "artifact_revision": ARTIFACT_REVISION,
            "created_at": now_iso(),
            "adopt_preflight": artifact_ref(args.adopt_preflight),
            "source_selection": artifact_ref(source_selection_path),
            "current_matrix_to_freeze": artifact_ref(matrix_path),
            "code_closure": artifact_ref(code_closure_path),
            "adopted_sessions": adopted_sessions,
            "execution_sessions": remaining,
        },
    )
    run_signature = build_run_signature(
        {
            "input_manifest": artifact_ref(input_manifest_path),
            "code_closure": artifact_ref(code_closure_path),
            "config": artifact_ref(task_packet_path),
            "weights": "ABSENT",
            "calibration_or_absent": "ABSENT",
            "schema": "ABSENT",
        }
    )
    claim = claim_executor(
        task_packet_path,
        executor_id=f"robot-v76-{os.uname().nodename}",
        pid=os.getpid(),
        run_signature=run_signature["run_signature_sha256"],
    )
    install_fenced_heartbeat(args.task_id, task_packet_path, claim)
    run_root.mkdir(parents=True)
    builder_matrix, snapshot_closure = freeze_builder_compatible_matrix(matrix_path, run_root)
    selection = {
        "schema_version": "exact78-robot-ready-batches-selection-v76",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "status": "FROZEN_V75_SELECTION_SPLIT_BY_EXACT_ADOPT",
        "governance_validation": governance,
        "task_state": artifact_ref(task_state_path),
        "source_selection": artifact_ref(source_selection_path),
        "matrix_snapshot_closure": snapshot_closure,
        "adopt_preflight": artifact_ref(args.adopt_preflight),
        "adopt_bundle_signature": preflight["adopt_bundle_signature"],
        "run_signature": run_signature,
        "executor_claim": artifact_ref(task_packet_path.parent / "EXECUTOR_CLAIM.json"),
        "task_packet_registration": packet_registration,
        "fencing_token": claim["fencing_token"],
        "sessions": frozen,
        "adopted_sessions": adopted_sessions,
        "execution_sessions": remaining,
    }
    atomic_new_json(run_root / "SELECTION.json", selection)

    legacy.RUN_ROOT = run_root
    legacy.TASK_ID = args.task_id
    legacy.PHASE_TIMEOUT_SECONDS = args.phase_timeout_seconds
    legacy.POSE_ONLY_ROOT = run_root / "final/pose_only"
    legacy.TERMINAL_BASE = run_root / "final/quality_c"
    legacy.heartbeat("RUNNING", "robot_v76_exact_adopt", adopted_sessions[0])
    adopted = finalize_adopted(run_root, adopt_rows)
    for index in range(0, len(remaining), 3):
        sessions = remaining[index : index + 3]
        anchor_v3.batch_commands_v54(
            run_root / f"batch_{index // 3 + 1:03d}", sessions, builder_matrix
        )
    executed = [
        {
            "session": session,
            "task": matrix_rows[session]["task"],
            "terminal": predecessor.terminal_ref(
                run_root, matrix_rows[session]["task"], session
            ),
        }
        for session in remaining
    ]
    if {row["session"] for row in adopted + executed} != set(frozen):
        raise RobotTargetContractError("v76 merged coverage differs from frozen selection")
    result = {
        "schema_version": "exact78-robot-ready-batches-result-v76",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "status": "PASSED_BATCH_TERMINAL_COVERAGE",
        "terminal_status": "PASSED",
        "robot_tier": "NONE",
        "run_signature": run_signature,
        "selection": artifact_ref(run_root / "SELECTION.json"),
        "counts": {
            "frozen": len(frozen),
            "exact_adopted": len(adopted),
            "newly_executed": len(executed),
        },
        "adopted": adopted,
        "executed": executed,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Digital Robot terminal coverage only; no Robot/contact/control/deployment authority.",
    }
    result_path = run_root / "RESULT.json"
    atomic_new_json(result_path, result)
    publish_task_terminal(
        args.task_id,
        result_path,
        "robot_v76_terminal_coverage_complete",
        message="V7.6 bounded snapshot-layout successor completed; no Robot authority promoted.",
    )
    return 0


def _failure_context(argv: list[str]) -> tuple[str | None, Path | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--task-packet", type=Path)
    args, _ = parser.parse_known_args(argv)
    return args.task_id, args.task_packet


def main() -> int:
    try:
        return _run()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as error:
        task_id, packet_path = _failure_context(sys.argv[1:])
        if task_id and packet_path:
            packet_path = packet_path.resolve()
            claim_path = packet_path.parent / "EXECUTOR_CLAIM.json"
            if claim_path.is_file():
                failure_path = packet_path.parent / "FAILED_RUNTIME_FINAL.json"
                if not failure_path.exists():
                    atomic_new_json(
                        failure_path,
                        {
                            "schema_version": "robot-v76-runtime-failure-v1",
                            "artifact_revision": ARTIFACT_REVISION,
                            "created_at": now_iso(),
                            "task_id": task_id,
                            "terminal_status": "FAILED_RUNTIME_FINAL",
                            "robot_tier": "NONE",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback_tail": traceback.format_exc().splitlines()[-80:],
                            "executor_claim": artifact_ref(claim_path),
                            "task_packet": artifact_ref(packet_path),
                            "authority": False,
                            "control_ground_truth": False,
                            "physical_deployment_authorized": False,
                            "claim_limit": "Runtime terminal only; partial output cannot be promoted.",
                        },
                    )
                publish_task_terminal(
                    task_id,
                    failure_path,
                    "robot_v76_runtime_failure",
                    status="FAILED_RUNTIME_FINAL",
                    message=f"V7.6 runtime failure converged: {type(error).__name__}: {error}",
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
