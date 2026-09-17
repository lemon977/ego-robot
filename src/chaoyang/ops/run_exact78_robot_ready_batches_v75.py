#!/usr/bin/env python3
"""V7.5 Robot runner: exact-adopt completed v7.4 phases, execute only the remainder."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
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
from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    RobotTargetContractError,
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    verify_ref,
)
from chaoyang.governance.v52_contracts import (  # noqa: E402
    build_run_signature,
    claim_executor,
    process_identity,
)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_governance() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RobotTargetContractError(
            f"governance validator did not return JSON: {process.stderr[-2000:]}"
        ) from error
    if process.returncode != 0 or payload.get("status") != "PASS":
        raise RobotTargetContractError(f"governance is not PASS: {payload}")
    return payload


def publish_task_terminal(
    task_id: str,
    result_path: Path,
    phase: str,
    *,
    status: str = "PASSED",
    message: str | None = None,
) -> dict[str, Any]:
    """Publish a terminal task state through the sole governance writer CLI."""
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
            message
            or "V7.5 exact-adopt/execution terminal published; no Robot authority promoted.",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise RobotTargetContractError(
                    f"terminal publisher returned non-JSON: {completed.stdout[-2000:]}"
                ) from error
        combined = completed.stdout + completed.stderr
        if "CAS revision mismatch" not in combined or attempt == 19:
            raise RobotTargetContractError(
                f"terminal task publication failed: {combined[-4000:]}"
            )
        time.sleep(min(0.1 * (attempt + 1), 1.0))
    raise RobotTargetContractError("terminal task publication attempts exhausted")


def verify_current_packet_registration(
    task_packet_path: Path, task_id: str
) -> dict[str, Any]:
    pointer_path = ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json"
    pointer = load_json(pointer_path)
    index_path = Path(str(pointer.get("index_path", "")))
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    if not index_path.is_file():
        raise RobotTargetContractError("current Task Packet index is missing")
    if artifact_ref(index_path)["sha256"] != pointer.get("index_sha256"):
        raise RobotTargetContractError("current Task Packet pointer SHA mismatch")
    index = load_json(index_path)
    rows = [row for row in index.get("task_packets", []) if row.get("task_id") == task_id]
    if len(rows) != 1:
        raise RobotTargetContractError(f"current Task Packet index must contain one {task_id}")
    row = rows[0]
    registered = Path(str(row.get("packet_path", "")))
    if not registered.is_absolute():
        registered = ROOT / registered
    registered = registered.resolve(strict=True)
    expected = artifact_ref(task_packet_path)
    if registered != task_packet_path or row.get("packet_sha256") != expected["sha256"]:
        raise RobotTargetContractError("current Task Packet registration does not bind this packet")
    return {"pointer": artifact_ref(pointer_path), "index": artifact_ref(index_path), "packet": expected}


def validate_task_row_registration(
    task_row: dict[str, Any], task_packet_path: Path
) -> None:
    expected = artifact_ref(task_packet_path)
    if task_row.get("task_packet") != expected:
        raise RobotTargetContractError("task state row does not bind this exact Task Packet")


def install_fenced_heartbeat(
    task_id: str,
    task_packet_path: Path,
    claim: dict[str, Any],
) -> None:
    original_heartbeat = legacy.heartbeat

    def fenced(status: str, phase: str, session: str | None = None) -> None:
        identity = process_identity(os.getpid())
        if (
            not identity.get("alive")
            or identity.get("start_ticks") != claim.get("proc_start_ticks")
            or claim.get("pid") != os.getpid()
        ):
            raise RobotTargetContractError("executor PID/startticks fencing check failed")
        claim_path = task_packet_path.parent / "EXECUTOR_CLAIM.json"
        current_claim = load_json(claim_path)
        if (
            current_claim.get("fencing_token") != claim.get("fencing_token")
            or current_claim.get("task_id") != task_id
        ):
            raise RobotTargetContractError("executor fencing token is no longer current")
        task_state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
        row = next(
            (item for item in task_state.get("tasks", []) if item.get("task_id") == task_id),
            None,
        )
        if row is None:
            raise RobotTargetContractError("v75 task row disappeared during execution")
        validate_task_row_registration(row, task_packet_path)
        original_heartbeat(status, phase, session)

    legacy.heartbeat = fenced


def verify_adopt_preflight(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preflight = load_json(path.resolve(strict=True))
    if preflight.get("schema_version") != "robot-v74-to-v75-adopt-preflight-v1":
        raise RobotTargetContractError("v74-to-v75 adopt preflight schema required")
    if preflight.get("adopt_preflight_status") != "PASS_EXACT_SHA_ADOPTABLE" or preflight.get("failures"):
        raise RobotTargetContractError("adopt preflight did not close all phase references")
    closure = preflight.get("code_closure")
    if not isinstance(closure, list):
        raise RobotTargetContractError("adopt preflight code closure required")
    closure_by_path = {str(item.get("path")): item for item in closure if isinstance(item, dict)}
    required_code = (
        ROOT / "src/chaoyang/ops/preflight_adopt_robot_v74_to_v75.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
        ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
    )
    for code in required_code:
        expected = artifact_ref(code)
        if closure_by_path.get(expected["path"]) != expected:
            raise RobotTargetContractError(f"adopt preflight code closure is stale: {code}")
    rows = preflight.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RobotTargetContractError("non-empty adopt rows required")
    sessions: set[str] = set()
    signatures = []
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or not session or session in sessions:
            raise RobotTargetContractError("adopt sessions must be non-empty and unique")
        sessions.add(session)
        if row.get("source_artifact_revision") != "R7_ROBOT_4" or row.get("target_artifact_revision") != "R7_ROBOT_5":
            raise RobotTargetContractError(f"{session}: artifact revision mismatch")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise RobotTargetContractError(f"{session}: artifact map required")
        verified = {name: artifact_ref(verify_ref(ref, f"{session}.{name}")) for name, ref in artifacts.items()}
        signature_payload = {
            "task": row.get("task"),
            "session": session,
            "source_artifact_revision": "R7_ROBOT_4",
            "target_artifact_revision": "R7_ROBOT_5",
            "artifacts": verified,
        }
        signature = canonical_sha(signature_payload)
        if signature != row.get("phase_closure_signature"):
            raise RobotTargetContractError(f"{session}: phase closure signature mismatch")
        signatures.append({"session": session, "signature": signature})
    if canonical_sha(signatures) != preflight.get("adopt_bundle_signature"):
        raise RobotTargetContractError("adopt bundle signature mismatch")
    return preflight, rows


def terminal_ref(run_root: Path, task: str, session: str) -> dict[str, Any]:
    candidates = (
        run_root / "final/pose_only" / session / "RESULT.json",
        run_root / "final/quality_c" / task / session / "RESULT.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RobotTargetContractError(f"{session}: expected one v75 terminal, found {existing}")
    return artifact_ref(existing[0])


def write_exact_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def split_frozen_sessions(
    frozen: list[str], adopt_rows: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    if not frozen or len(frozen) != len(set(frozen)):
        raise RobotTargetContractError("v74 frozen selection is empty or duplicated")
    adopted_set = {row.get("session") for row in adopt_rows}
    if None in adopted_set or len(adopted_set) != len(adopt_rows):
        raise RobotTargetContractError("adopt sessions are missing or duplicated")
    if not adopted_set.issubset(frozen):
        raise RobotTargetContractError("adopt set is not a subset of frozen v74 selection")
    return [session for session in frozen if session in adopted_set], [
        session for session in frozen if session not in adopted_set
    ]


def validate_execution_readiness(
    matrix: dict[str, Any], frozen: list[str], remaining: list[str]
) -> dict[str, dict[str, Any]]:
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise RobotTargetContractError("Robot matrix rows are required")
    matrix_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        session = row.get("session_id") if isinstance(row, dict) else None
        if not isinstance(session, str) or not session or session in matrix_rows:
            raise RobotTargetContractError("Robot matrix session ids must be non-empty and unique")
        matrix_rows[session] = row
    missing = [session for session in frozen if session not in matrix_rows]
    if missing:
        raise RobotTargetContractError(f"frozen session missing from current matrix: {missing}")
    invalid_remaining = [
        session
        for session in remaining
        if matrix_rows[session].get("robot_current_state")
        != "READY_FOR_ROBOT_CURRENT_DRAFT"
    ]
    if invalid_remaining:
        raise RobotTargetContractError(
            f"remaining v75 sessions are no longer Robot-ready: {invalid_remaining}"
        )
    return matrix_rows


def finalize_adopted(run_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches: dict[Path, list[dict[str, Any]]] = {}
    for row in rows:
        preflight_path = verify_ref(row["artifacts"]["preflight_result"], f"{row['session']}.preflight")
        batches.setdefault(preflight_path.parent.parent, []).append(row)
    for batch in batches:
        legacy.finalize_sessions(
            batch / "preflight/RESULT.json",
            batch / "hawor_temporal",
            batch / "arm_round2",
            batch / "hand_round2",
            batch / "render",
        )
    adopted = []
    for row in rows:
        result = {
            "schema_version": "robot-v75-exact-adopt-map-v1",
            "artifact_revision": "R7_ROBOT_5",
            "created_at": now_iso(),
            "task": row["task"],
            "session": row["session"],
            "status": "ADOPTED_EXACT_PHASE_CLOSURE_AND_REFINALIZED",
            "phase_closure_signature": row["phase_closure_signature"],
            "source_artifacts": row["artifacts"],
            "terminal": terminal_ref(run_root, row["task"], row["session"]),
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
    parser.add_argument("--task-id", default="robot_geometry_expansion_v75")
    parser.add_argument("--phase-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.phase_timeout_seconds < 60:
        raise SystemExit("--phase-timeout-seconds must be >=60")
    governance = validate_governance()
    preflight, adopt_rows = verify_adopt_preflight(args.adopt_preflight)
    task_packet_path = args.task_packet.resolve(strict=True)
    task_packet = load_json(task_packet_path)
    if task_packet.get("task_id") != args.task_id:
        raise RobotTargetContractError("task packet/task-id mismatch")
    packet_registration = verify_current_packet_registration(task_packet_path, args.task_id)
    task_state_path = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
    task_state = load_json(task_state_path)
    task_row = next((row for row in task_state.get("tasks", []) if row.get("task_id") == args.task_id), None)
    if task_row is None:
        raise RobotTargetContractError(
            "governance aggregator must register robot_geometry_expansion_v75 before launch"
        )
    if task_row.get("status") not in {"PENDING", "CLAIMED", "BLOCKED_PREREQ"}:
        raise RobotTargetContractError(f"v75 task is not launchable: {task_row.get('status')}")
    validate_task_row_registration(task_row, task_packet_path)
    source_selection_path = verify_ref(preflight["selection"], "v74.selection")
    source_selection = load_json(source_selection_path)
    frozen = list(source_selection.get("sessions", []))
    adopted_sessions, remaining = split_frozen_sessions(frozen, adopt_rows)
    # The v7.4 selection stores a SHA of the matrix used at selection time, but
    # its path was historically mutable.  V7.5 freezes the current matrix into
    # its own input_snapshot and only requires the original 25 session ids to
    # stay unchanged.  Exact-adopted rows remain bound to their own per-batch
    # preflight snapshots and signatures.
    matrix_path = args.matrix.resolve(strict=True)
    matrix_input_ref = artifact_ref(matrix_path)
    matrix = load_json(matrix_path)
    matrix_rows = validate_execution_readiness(matrix, frozen, remaining)

    run_root = args.output_root.resolve()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"fresh immutable output root required: {run_root}")
    input_manifest_path = task_packet_path.parent / "RUN_INPUT_MANIFEST.json"
    atomic_new_json(
        input_manifest_path,
        {
            "schema_version": "robot-v75-run-input-manifest-v1",
            "artifact_revision": "R7_ROBOT_5",
            "created_at": now_iso(),
            "adopt_preflight": artifact_ref(args.adopt_preflight),
            "source_selection": artifact_ref(source_selection_path),
            "current_matrix_to_freeze": matrix_input_ref,
            "adopted_sessions": adopted_sessions,
            "execution_sessions": remaining,
        },
    )
    run_signature = build_run_signature(
        {
            "input_manifest": artifact_ref(input_manifest_path),
            "code_closure": artifact_ref(Path(__file__)),
            "config": artifact_ref(task_packet_path),
            "weights": "ABSENT",
            "calibration_or_absent": "ABSENT",
            "schema": "ABSENT",
        }
    )
    claim = claim_executor(
        task_packet_path,
        executor_id=f"robot-v75-{os.uname().nodename}",
        pid=os.getpid(),
        run_signature=run_signature["run_signature_sha256"],
    )
    install_fenced_heartbeat(args.task_id, task_packet_path, claim)
    run_root.mkdir(parents=True)
    matrix_snapshot_path = (
        run_root
        / "input_snapshot"
        / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{matrix_input_ref['sha256']}.json"
    )
    write_exact_new(matrix_snapshot_path, matrix_path.read_bytes())
    matrix_snapshot_ref = artifact_ref(matrix_snapshot_path)
    if matrix_snapshot_ref["bytes"] != matrix_input_ref["bytes"] or matrix_snapshot_ref["sha256"] != matrix_input_ref["sha256"]:
        raise RobotTargetContractError("matrix input snapshot byte/SHA mismatch")
    selection = {
        "schema_version": "exact78-robot-ready-batches-selection-v75",
        "artifact_revision": "R7_ROBOT_5",
        "created_at": now_iso(),
        "status": "FROZEN_V74_SELECTION_SPLIT_BY_EXACT_ADOPT",
        "governance_validation": governance,
        "task_state": artifact_ref(task_state_path),
        "source_selection": artifact_ref(source_selection_path),
        "matrix_source": matrix_input_ref,
        "matrix_snapshot": matrix_snapshot_ref,
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
    legacy.heartbeat("RUNNING", "robot_v75_exact_adopt", selection["adopted_sessions"][0])
    adopted = finalize_adopted(run_root, adopt_rows)
    for index in range(0, len(remaining), 3):
        sessions = remaining[index : index + 3]
        anchor_v3.batch_commands_v54(run_root / f"batch_{index // 3 + 1:03d}", sessions, matrix_snapshot_path)
    executed = [
        {
            "session": session,
            "task": matrix_rows[session]["task"],
            "terminal": terminal_ref(run_root, matrix_rows[session]["task"], session),
        }
        for session in remaining
    ]
    if {row["session"] for row in adopted} | {row["session"] for row in executed} != set(frozen):
        raise RobotTargetContractError("v75 merged terminal coverage does not equal frozen selection")
    result = {
        "schema_version": "exact78-robot-ready-batches-result-v75",
        "artifact_revision": "R7_ROBOT_5",
        "created_at": now_iso(),
        "status": "PASSED_BATCH_TERMINAL_COVERAGE",
        "terminal_status": "PASSED",
        "robot_tier": "NONE",
        "run_signature": run_signature,
        "executor_claim": artifact_ref(task_packet_path.parent / "EXECUTOR_CLAIM.json"),
        "fencing_token": claim["fencing_token"],
        "selection": artifact_ref(run_root / "SELECTION.json"),
        "counts": {"frozen": len(frozen), "exact_adopted": len(adopted), "newly_executed": len(executed)},
        "adopted": adopted,
        "executed": executed,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Digital visual Robot terminal coverage only; per-session tier requires separate hard geometry/compositor promotion.",
    }
    result_path = run_root / "RESULT.json"
    atomic_new_json(result_path, result)
    publish_task_terminal(args.task_id, result_path, "robot_v75_terminal_coverage_complete")
    return 0


def _failure_context(argv: list[str]) -> tuple[str | None, Path | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-id", default="robot_geometry_expansion_v75")
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
        # Pre-claim failures are intentionally read-only blockers.  Once a
        # fencing claim exists, however, the attempt must not remain a ghost
        # RUNNING task: publish an immutable failure receipt and terminate it
        # through the sole governance writer.
        if task_id and packet_path:
            packet_path = packet_path.resolve()
            claim_path = packet_path.parent / "EXECUTOR_CLAIM.json"
            if claim_path.is_file():
                failure_path = packet_path.parent / "FAILED_RUNTIME_FINAL.json"
                if not failure_path.exists():
                    atomic_new_json(
                        failure_path,
                        {
                            "schema_version": "robot-v75-runtime-failure-v1",
                            "artifact_revision": "R7_ROBOT_5",
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
                            "claim_limit": "Runtime terminal only; partial outputs are not promoted or reused without exact phase receipts.",
                        },
                    )
                publish_task_terminal(
                    task_id,
                    failure_path,
                    "robot_v75_runtime_failure",
                    status="FAILED_RUNTIME_FINAL",
                    message=f"V7.5 runtime failure converged: {type(error).__name__}: {error}",
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
