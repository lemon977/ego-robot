#!/usr/bin/env python3
"""V7.7 bounded Robot recovery using immutable session evidence and micro-batches.

This wrapper deliberately does not publish governance state.  It freezes the
current run inputs, republishes only the twelve SHA-closed single-session
hard/soft audit rows, and delegates new computation to the unchanged v7.6
Robot algorithm closure in independent 3/3/3/3/1 attempts.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


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
from chaoyang.ops.run_robot_hard_soft_audit_watcher_v71 import audit_batch  # noqa: E402


ARTIFACT_REVISION = "R7_ROBOT_7"
TASK_ID = "robot_geometry_expansion_v77"
MATRIX_NAME = "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
EXPECTED_SELECTION_COUNT = 25
EXPECTED_ADOPT_COUNT = 12
UPSTREAM_DENOMINATOR = 101
REQUIRED_ALGORITHM_PATHS = (
    "src/chaoyang/ops/run_exact78_robot_ready_batches_v54.py",
    "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
    "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
    "src/chaoyang/ops/preflight_exact78_robot_ready_batch_v52.py",
    "src/chaoyang/ops/preflight_exact78_robot_ready_anchor_v3.py",
)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _exact_ref(reference: dict[str, Any], label: str) -> Path:
    return verify_ref(reference, label).resolve(strict=True)


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_runtime_json(path: Path, value: dict[str, Any]) -> None:
    """Replace only a declared runtime heartbeat, never an immutable result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = load_json(path.resolve(strict=True))
    if not isinstance(value, dict):
        raise RobotTargetContractError(f"{label}: JSON object required")
    return value


def verify_algorithm_closure(source_path: Path) -> dict[str, Any]:
    """Pin the phase algorithm to the unchanged v7.6 closure.

    The v7.7 wrapper is a different orchestration implementation, but the
    phase builder/preflight/solver/render code must remain byte-identical.
    """
    source = _load_object(source_path, "v76 code closure")
    rows = source.get("artifacts")
    if not isinstance(rows, list):
        raise RobotTargetContractError("v76 code closure artifacts missing")
    by_path = {str(row.get("path")): row for row in rows if isinstance(row, dict)}
    verified = []
    for relative in REQUIRED_ALGORITHM_PATHS:
        path = (ROOT / relative).resolve(strict=True)
        current = artifact_ref(path)
        if by_path.get(str(path)) != current:
            raise RobotTargetContractError(f"v77 algorithm changed from v76: {relative}")
        verified.append(current)
    payload = {
        "schema_version": "robot-v77-algorithm-closure-v1",
        "algorithm_id": "WORLD_FIRST_ROBOT_V76_PHASE_DAG_UNCHANGED",
        "artifacts": verified,
        "source_v76_code_closure": artifact_ref(source_path.resolve(strict=True)),
    }
    payload["algorithm_signature_sha256"] = canonical_sha(payload["artifacts"])
    return payload


def verify_hard_soft_index(index_path: Path) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    index = _load_object(index_path, "hard/soft candidate index")
    if index.get("schema_version") != "robot-hard-soft-candidate-index-v71-v1":
        raise RobotTargetContractError("hard/soft candidate index schema mismatch")
    counts = index.get("counts") or {}
    if (
        counts.get("expected_sessions") != EXPECTED_SELECTION_COUNT
        or counts.get("sessions") != EXPECTED_ADOPT_COUNT
        or counts.get("hard_geometry_pass") != 10
    ):
        raise RobotTargetContractError(f"unexpected hard/soft coverage: {counts}")
    selection_path = _exact_ref(index.get("selection"), "hard_soft.selection")
    selection = _load_object(selection_path, "source selection")
    sessions = list(selection.get("sessions", []))
    if len(sessions) != EXPECTED_SELECTION_COUNT or len(set(sessions)) != len(sessions):
        raise RobotTargetContractError("source selection must contain 25 unique sessions")
    rows = index.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ADOPT_COUNT:
        raise RobotTargetContractError("hard/soft index must contain twelve rows")
    seen: set[str] = set()
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or session in seen or session not in sessions:
            raise RobotTargetContractError(f"invalid hard/soft session: {session!r}")
        seen.add(session)
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise RobotTargetContractError(f"{session}: evidence map missing")
        for name, reference in evidence.items():
            _exact_ref(reference, f"{session}.{name}")
        terminal = row.get("terminal_status")
        hard = row.get("hard_geometry_pass") is True
        hard_gates = row.get("hard_gates")
        if not isinstance(hard_gates, dict):
            raise RobotTargetContractError(f"{session}: hard gates missing")
        if terminal == "PASSED":
            if not hard or not all(value is True for value in hard_gates.values()):
                raise RobotTargetContractError(f"{session}: invalid hard-pass classification")
        elif terminal == "FAILED_QUALITY_C":
            if hard or all(value is True for value in hard_gates.values()):
                raise RobotTargetContractError(f"{session}: invalid quality-C classification")
        else:
            raise RobotTargetContractError(f"{session}: unsupported terminal {terminal!r}")
    by_session = {row["session"]: row for row in rows}
    for required, expected_terminal in (
        ("get_potato_chips_0902_104", "PASSED"),
        ("play_cards_0903_189", "FAILED_QUALITY_C"),
        ("play_cards_0903_202", "FAILED_QUALITY_C"),
    ):
        if by_session.get(required, {}).get("terminal_status") != expected_terminal:
            raise RobotTargetContractError(f"required v77 classification missing: {required}")
    return index, sessions, rows


def microbatch_plan(selection: list[str], adopted: set[str]) -> list[list[str]]:
    remaining = [session for session in selection if session not in adopted]
    if len(remaining) != 13:
        raise RobotTargetContractError(f"v77 requires thirteen execution sessions, got {len(remaining)}")
    batches = [remaining[index : index + 3] for index in range(0, len(remaining), 3)]
    if [len(batch) for batch in batches] != [3, 3, 3, 3, 1]:
        raise RobotTargetContractError("v77 microbatch shape must be 3/3/3/3/1")
    return batches


def verify_matrix(matrix_path: Path, sessions: list[str]) -> dict[str, dict[str, Any]]:
    matrix = _load_object(matrix_path, "Robot matrix")
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise RobotTargetContractError("Robot matrix rows missing")
    by_session = {
        row.get("session_id"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("session_id"), str)
    }
    if len(by_session) != len(rows):
        raise RobotTargetContractError("Robot matrix contains duplicate/invalid session ids")
    missing = [session for session in sessions if session not in by_session]
    if missing:
        raise RobotTargetContractError(f"selection missing from matrix: {missing}")
    return by_session


def freeze_builder_matrix(matrix_path: Path, output_root: Path) -> dict[str, Any]:
    source = matrix_path.resolve(strict=True)
    source_ref = artifact_ref(source)
    live = output_root / "input_snapshot" / MATRIX_NAME
    companion = live.parent / "snapshots" / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{source_ref['sha256']}.json"
    _write_exact(live, source.read_bytes())
    _write_exact(companion, source.read_bytes())
    for label, path in (("live", live), ("companion", companion)):
        if artifact_ref(path) != source_ref | {"path": str(path.resolve())}:
            raise RobotTargetContractError(f"frozen matrix {label} differs from source")
    return {"source": source_ref, "builder_input": artifact_ref(live), "companion": artifact_ref(companion)}


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "head": head,
        "dirty_entry_count": status.count(b"\0"),
        "porcelain_sha256": hashlib.sha256(status).hexdigest(),
    }


def _terminal_payload(index_ref: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    hard = row["hard_geometry_pass"] is True
    return {
        "schema_version": "robot-v77-single-session-terminal-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "task": row["task"],
        "session": row["session"],
        "terminal_status": "PASSED" if hard else "FAILED_QUALITY_C",
        "status": row["status"],
        "robot_tier": "POSE_ONLY_VISUAL" if hard else "NONE",
        "hard_geometry_pass": hard,
        "soft_pose_similarity_pass": row.get("soft_pose_similarity_pass") is True,
        "source_kind": "SHA_CLOSED_SINGLE_SESSION_HARD_SOFT_AUDIT",
        "source_hard_soft_index": index_ref,
        "source_row_sha256": canonical_sha(row),
        "hard_gates": row["hard_gates"],
        "soft_gates": row.get("soft_gates", {}),
        "metrics": {
            "arm_soft": row.get("arm_soft_metrics"),
            "hand_soft": row.get("hand_soft_metrics"),
        },
        "evidence": row["evidence"],
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Digital URDF hard-feasibility terminal only; no Robot/contact/control/physical authority.",
    }


def _temporal_quality_terminal(
    session: str,
    task: str,
    batch_id: int,
    log_path: Path,
) -> dict[str, Any]:
    """Fail one session closed when the pinned temporal successor has no safe alpha."""
    return {
        "schema_version": "robot-v77-single-session-terminal-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "task": task,
        "session": session,
        "terminal_status": "FAILED_QUALITY_C",
        "status": "FAILED_HAWOR_TEMPORAL_SUCCESSOR_QUALITY_C",
        "robot_tier": "NONE",
        "hard_geometry_pass": False,
        "soft_pose_similarity_pass": False,
        "source_kind": "FRESH_V77_ISOLATED_SESSION_TEMPORAL_FAILURE",
        "microbatch_id": batch_id,
        "hard_gates": {
            "hawor_temporal_successor": False,
            "finite_valid_states": False,
            "arm_temporal_and_unknown_contract": False,
            "hand_temporal_and_semantic_contract": False,
            "digital_collision_geometry_loaded": False,
            "fullsession_collision_absent": False,
            "urdf_joint_limits": False,
        },
        "soft_gates": {},
        "metrics": {"temporal_failure": "NO_SAFE_SUCCESSOR_ALPHA"},
        "evidence": {"hawor_temporal_attempt_log": artifact_ref(log_path)},
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Pinned temporal quality-C terminal; downstream Robot phases were not run and no authority is promoted.",
    }


def bootstrap(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"fresh output root required: {output_root}")
    g0_snapshot_path = args.g0_snapshot.resolve(strict=True)
    g0_snapshot = _load_object(g0_snapshot_path, "G0 run-start snapshot")
    if g0_snapshot.get("schema_version") != "chaoyang-r22-run-start-snapshot-v1":
        raise RobotTargetContractError("R2.2 G0 run-start snapshot required")
    g0_selection_path = args.g0_selection.resolve(strict=True)
    if artifact_ref(g0_selection_path) != g0_snapshot.get("integration_selection"):
        raise RobotTargetContractError("G0 integration selection reference mismatch")
    _load_object(g0_selection_path, "G0 integration selection")
    packet_path = args.task_packet.resolve(strict=True)
    packet = _load_object(packet_path, "v77 adopt task packet")
    if packet.get("task_id") != "robot_v77_adopt12_r22":
        raise RobotTargetContractError("registered R2.2 adopt task packet required")
    declared_writes = [Path(path).resolve() for path in packet.get("write_set", [])]
    if not any(output_root.is_relative_to(path) for path in declared_writes):
        raise RobotTargetContractError("output root is outside registered adopt write_set")
    if artifact_ref(args.hard_soft_index.resolve(strict=True)) != g0_snapshot.get(
        "robot_hard_soft_candidate_index"
    ):
        raise RobotTargetContractError("hard/soft index differs from G0 pin")
    index, sessions, rows = verify_hard_soft_index(args.hard_soft_index)
    verify_matrix(args.matrix, sessions)
    batches = microbatch_plan(sessions, {row["session"] for row in rows})
    algorithm = verify_algorithm_closure(args.v76_code_closure)
    output_root.mkdir(parents=True)
    matrix_closure = freeze_builder_matrix(args.matrix, output_root)
    algorithm_contract_pin = g0_snapshot["algorithm_contract"]
    algorithm_contract_path = Path(str(algorithm_contract_pin["path"])).resolve(strict=True)
    current_algorithm_contract = artifact_ref(algorithm_contract_path)
    algorithm_contract_pin_matches_current = current_algorithm_contract == algorithm_contract_pin
    run_snapshot = {
        "schema_version": "robot-v77-run-input-manifest-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "governance_revision": g0_snapshot["governance_revision"],
        "generation_id": g0_snapshot["generation_id"],
        "g0_run_start_snapshot": artifact_ref(g0_snapshot_path),
        "g0_current_status_receipt_pin": g0_snapshot["current_status_receipt"],
        "task_packet": artifact_ref(packet_path),
        "algorithm_contract_pin": algorithm_contract_pin,
        "algorithm_contract_current_at_worker_start": current_algorithm_contract,
        "algorithm_contract_pin_matches_current": algorithm_contract_pin_matches_current,
        "provenance_debt": ([] if algorithm_contract_pin_matches_current else [
            {
                "code": "G0_MUTABLE_ALGORITHM_CONTRACT_PIN_DRIFTED",
                "impact": "The G0 governance-contract bytes were not content-addressed; phase execution remains separately locked to the exact v76 code closure.",
            }
        ]),
        "hard_soft_candidate_index": artifact_ref(args.hard_soft_index.resolve(strict=True)),
        "matrix_snapshot": matrix_closure,
        "git": _git_snapshot(),
        "authority": False,
        "claim_limit": "Frozen execution inputs only; this worker does not update current governance.",
    }
    atomic_new_json(output_root / "RUN_INPUT_MANIFEST.json", run_snapshot)
    atomic_new_json(output_root / "ROBOT_ALGORITHM_CLOSURE.json", algorithm)
    selection_payload = {
        "schema_version": "exact78-robot-v77-microbatch-selection-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "sessions": sessions,
        "adopted_sessions": [row["session"] for row in rows],
        "execution_sessions": [session for batch in batches for session in batch],
        "microbatches": [
            {"batch_id": index + 1, "sessions": batch}
            for index, batch in enumerate(batches)
        ],
        "source_hard_soft_index": artifact_ref(args.hard_soft_index.resolve(strict=True)),
        "run_input_manifest": artifact_ref(output_root / "RUN_INPUT_MANIFEST.json"),
        "algorithm_closure": artifact_ref(output_root / "ROBOT_ALGORITHM_CLOSURE.json"),
        "authority": False,
    }
    atomic_new_json(output_root / "SELECTION.json", selection_payload)
    attempt = output_root
    index_ref = artifact_ref(args.hard_soft_index.resolve(strict=True))
    adopted = []
    for row in rows:
        terminal_path = attempt / "terminals" / row["session"] / "RESULT.json"
        atomic_new_json(terminal_path, _terminal_payload(index_ref, row))
        adopted.append({
            "session": row["session"],
            "task": row["task"],
            "terminal_status": row["terminal_status"],
            "result": artifact_ref(terminal_path),
        })
    result = {
        "schema_version": "robot-v77-adopt-12-result-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "terminal_status": "PASSED",
        "status": "PASSED_SHA_CLOSED_SINGLE_SESSION_REPUBLICATION",
        "counts": {
            "sessions": len(adopted),
            "hard_geometry_pass": sum(row["terminal_status"] == "PASSED" for row in adopted),
            "failed_quality_c": sum(row["terminal_status"] == "FAILED_QUALITY_C" for row in adopted),
        },
        "rows": adopted,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Adopted from the immutable hard/soft audit index, not from the v76 partial batch root.",
    }
    result_path = attempt / "RESULT.json"
    atomic_new_json(result_path, result)
    artifacts = [artifact_ref(path) for path in sorted(attempt.rglob("RESULT.json"))]
    atomic_new_json(attempt / "ARTIFACT_MANIFEST.json", {"schema_version": "artifact-manifest-v1", "artifacts": artifacts})
    atomic_new_json(attempt / "METRICS.json", {"schema_version": "robot-v77-adopt-metrics-v1", **result["counts"]})
    atomic_new_json(attempt / "RUN_RECEIPT.json", {"schema_version": "robot-v77-run-receipt-v1", "result": artifact_ref(result_path), "selection": artifact_ref(output_root / "SELECTION.json")})
    _write_exact(attempt / "DECISION.md", b"# Decision\n\nRepublished 12 SHA-closed single-session hard/soft terminals; no v76 partial root was adopted.\n")
    atomic_new_json(attempt / "NEXT_ACTION.json", {"next_task_id": "V77-BATCH-001", "sessions": batches[0], "status": "READY"})
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _verify_frozen_root(bootstrap_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = _load_object(bootstrap_root / "SELECTION.json", "v77 selection")
    snapshot_path = _exact_ref(selection["run_input_manifest"], "selection.run_input_manifest")
    snapshot = _load_object(snapshot_path, "run input manifest")
    algorithm_path = _exact_ref(selection["algorithm_closure"], "selection.algorithm_closure")
    algorithm = _load_object(algorithm_path, "algorithm closure")
    if canonical_sha(algorithm["artifacts"]) != algorithm["algorithm_signature_sha256"]:
        raise RobotTargetContractError("algorithm signature mismatch")
    for reference in algorithm["artifacts"]:
        _exact_ref(reference, "algorithm artifact")
    matrix_ref = snapshot["matrix_snapshot"]["builder_input"]
    _exact_ref(matrix_ref, "frozen builder matrix")
    return selection, snapshot


def _local_heartbeat(path: Path, phase: str, session: str | None) -> None:
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    _atomic_runtime_json(path, {
        "schema_version": "robot-v77-local-heartbeat-v1",
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "process_startticks": stat[21],
        "phase": phase,
        "session": session,
        "heartbeat_at": now_iso(),
        "authority": False,
    })


def _run_phase_dag_without_legacy_finalizer(
    work: Path,
    sessions: list[str],
    matrix: Path,
) -> None:
    """Run the byte-pinned v7.6 phase DAG but not its obsolete publisher.

    ``batch_commands_v54`` delegates its final publication step to the v5.3
    ``finalize_sessions`` global.  That publisher reads the historical Wave0
    selection by default, rather than the R2.2 frozen matrix/preflight.  It is
    therefore neither an algorithm phase nor a valid v7.7 publication path.
    V7.7 publishes fresh terminals from the hard/soft audit below, so suppress
    only that legacy side effect while keeping every pinned compute/render
    phase byte-identical.
    """
    original = legacy.finalize_sessions
    legacy.finalize_sessions = lambda *_args, **_kwargs: None
    try:
        anchor_v3.batch_commands_v54(work, sessions, matrix)
    finally:
        legacy.finalize_sessions = original


def run_batch(args: argparse.Namespace) -> int:
    bootstrap_root = args.bootstrap_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    selection, snapshot = _verify_frozen_root(bootstrap_root)
    batches = {int(row["batch_id"]): row["sessions"] for row in selection["microbatches"]}
    if args.batch_id not in batches:
        raise RobotTargetContractError(f"unknown batch id: {args.batch_id}")
    sessions = batches[args.batch_id]
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"fresh batch attempt required: {output_root}")
    packet_path = args.task_packet.resolve(strict=True)
    packet = _load_object(packet_path, "v77 batch task packet")
    if packet.get("task_id") != f"robot_v77_batch_{args.batch_id:03d}_r22":
        raise RobotTargetContractError("registered batch Task Packet id mismatch")
    declared_writes = [Path(path).resolve() for path in packet.get("write_set", [])]
    if not any(output_root.is_relative_to(path) for path in declared_writes):
        raise RobotTargetContractError("batch output root is outside registered write_set")
    attempt = output_root
    attempt.mkdir(parents=True)
    started = time.monotonic()
    heartbeat_path = attempt / "HEARTBEAT.json"

    def heartbeat(_status: str, phase: str, session: str | None = None) -> None:
        _local_heartbeat(heartbeat_path, phase, session)

    legacy.heartbeat = heartbeat
    legacy.RUN_ROOT = attempt
    legacy.TASK_ID = TASK_ID
    legacy.PHASE_TIMEOUT_SECONDS = args.phase_timeout_seconds
    legacy.POLL_SECONDS = min(25.0, float(args.heartbeat_seconds))
    legacy.POSE_ONLY_ROOT = attempt / "legacy_final/pose_only"
    legacy.TERMINAL_BASE = attempt / "legacy_final/quality_c"
    matrix = _exact_ref(snapshot["matrix_snapshot"]["builder_input"], "builder matrix")
    rows = []
    audit_rows: list[dict[str, Any]] = []
    audit_refs: list[dict[str, Any]] = []
    heartbeat("RUNNING", "batch_start", sessions[0])
    if args.isolate_sessions:
        for session in sessions:
            work = attempt / "work_sessions" / session
            heartbeat("RUNNING", "isolated_session_start", session)
            try:
                _run_phase_dag_without_legacy_finalizer(work, [session], matrix)
            except RuntimeError:
                temporal_log = work / "attempts/hawor_temporal_attempt_0001.log"
                if temporal_log.is_file() and "no safe successor alpha" in temporal_log.read_text(errors="replace"):
                    terminal_path = attempt / "terminals" / session / "RESULT.json"
                    task = "poker" if session.startswith("play_cards_") else "chips"
                    terminal = _temporal_quality_terminal(session, task, args.batch_id, temporal_log)
                    atomic_new_json(terminal_path, terminal)
                    rows.append({
                        "session": session,
                        "task": task,
                        "terminal_status": "FAILED_QUALITY_C",
                        "result": artifact_ref(terminal_path),
                    })
                    continue
                raise
            session_audit_root = attempt / "hard_soft/sessions" / session
            session_audit_result = audit_batch(
                work,
                session_audit_root,
                attempt / "visual_candidates" / session,
            )
            session_audit = _load_object(session_audit_result, f"{session} hard/soft audit")
            audit_refs.append(artifact_ref(session_audit_result))
            audit_rows.extend(session_audit.get("rows", []))
        hard_soft_root = attempt / "hard_soft"
        hard_soft_result = hard_soft_root / "RESULT.json"
        atomic_new_json(hard_soft_result, {
            "schema_version": "robot-v77-isolated-hard-soft-audit-index-v1",
            "created_at": now_iso(),
            "batch_id": args.batch_id,
            "rows": audit_rows,
            "session_audits": audit_refs,
            "authority": False,
        })
    else:
        work = attempt / "work"
        _run_phase_dag_without_legacy_finalizer(work, sessions, matrix)
        hard_soft_root = attempt / "hard_soft"
        hard_soft_result = audit_batch(work, hard_soft_root, attempt / "visual_candidates")
        audit = _load_object(hard_soft_result, "batch hard/soft audit")
        audit_rows = audit.get("rows", [])
    for row in audit_rows:
        terminal_path = attempt / "terminals" / row["session"] / "RESULT.json"
        terminal = _terminal_payload(artifact_ref(hard_soft_result), row)
        terminal["source_kind"] = "FRESH_V77_MICROBATCH_HARD_SOFT_AUDIT"
        terminal["microbatch_id"] = args.batch_id
        atomic_new_json(terminal_path, terminal)
        rows.append({
            "session": row["session"],
            "task": row["task"],
            "terminal_status": terminal["terminal_status"],
            "result": artifact_ref(terminal_path),
        })
    if {row["session"] for row in rows} != set(sessions):
        raise RobotTargetContractError("microbatch audit coverage differs from plan")
    elapsed = time.monotonic() - started
    result = {
        "schema_version": "robot-v77-microbatch-result-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "batch_id": args.batch_id,
        "terminal_status": "PASSED",
        "status": "PASSED_MICROBATCH_TERMINAL_COVERAGE",
        "sessions": sessions,
        "counts": {
            "sessions": len(rows),
            "hard_geometry_pass": sum(row["terminal_status"] == "PASSED" for row in rows),
            "failed_quality_c": sum(row["terminal_status"] == "FAILED_QUALITY_C" for row in rows),
        },
        "wall_seconds": elapsed,
        "hard_soft_audit": artifact_ref(hard_soft_result),
        "rows": rows,
        "algorithm_signature_sha256": _load_object(bootstrap_root / "ROBOT_ALGORITHM_CLOSURE.json", "algorithm")["algorithm_signature_sha256"],
        "task_packet": artifact_ref(packet_path),
        "bootstrap_result": artifact_ref(bootstrap_root / "RESULT.json"),
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Fresh digital Robot microbatch terminal coverage only.",
    }
    result_path = attempt / "RESULT.json"
    atomic_new_json(result_path, result)
    atomic_new_json(attempt / "METRICS.json", {"schema_version": "robot-v77-batch-metrics-v1", **result["counts"], "wall_seconds": elapsed})
    atomic_new_json(attempt / "RUN_RECEIPT.json", {"schema_version": "robot-v77-run-receipt-v1", "result": artifact_ref(result_path), "selection": artifact_ref(bootstrap_root / "SELECTION.json")})
    atomic_new_json(attempt / "ARTIFACT_MANIFEST.json", {"schema_version": "artifact-manifest-v1", "artifacts": [artifact_ref(result_path), artifact_ref(hard_soft_result)]})
    _write_exact(attempt / "DECISION.md", f"# Decision\n\nMicrobatch {args.batch_id:03d} reached terminal coverage for {len(rows)} sessions.\n".encode())
    next_batch = args.batch_id + 1
    atomic_new_json(attempt / "NEXT_ACTION.json", {
        "next_task_id": f"V77-BATCH-{next_batch:03d}" if next_batch in batches else "V77-TERMINAL-INDEX",
        "status": "READY",
    })
    heartbeat("PASSED", "batch_terminal", sessions[-1])
    print(json.dumps(result, ensure_ascii=False))
    return 0


def close_runtime_failure(args: argparse.Namespace) -> int:
    """Close one stopped attempt without converting it into a task terminal."""
    attempt = args.output_root.resolve(strict=True)
    result_path = attempt / "RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError(result_path)
    phase_evidence = []
    fixed = (
        "work/preflight/RESULT.json",
        "work/hawor_temporal/RESULT.json",
        "work/arm_round2/RESULT.json",
        "work/hand_round2/RESULT.json",
        "work/render/RESULT.json",
        "finalize/play_cards_0903_195/attempts/adopt_pose_only_attempt_0001.log",
        "finalize/play_cards_0903_195/attempts/adopt_pose_only_attempt_0002.log",
    )
    for relative in fixed:
        path = attempt / relative
        if path.is_file():
            phase_evidence.append(artifact_ref(path))
    for path in sorted((attempt / "work/attempts").glob("*.log")):
        reference = artifact_ref(path)
        if reference not in phase_evidence:
            phase_evidence.append(reference)
    result = {
        "schema_version": "robot-v77-attempt-runtime-failure-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "attempt_status": "FAILED_RUNTIME",
        "terminal_status": "FAILED_RUNTIME_RETRYABLE",
        "task_status": "PENDING",
        "retryable": True,
        "failure_phase": args.failure_phase,
        "error": args.error,
        "phase_evidence": phase_evidence,
        "partial_outputs_are_successor_inputs": False,
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Attempt-level runtime failure only; no session terminal or authority was published.",
    }
    atomic_new_json(result_path, result)
    atomic_new_json(attempt / "METRICS.json", {
        "schema_version": "robot-v77-attempt-failure-metrics-v1",
        "runtime_attempts_consumed": 1,
        "runtime_attempts_remaining": 1,
        "session_terminals_published": 0,
    })
    atomic_new_json(attempt / "RUN_RECEIPT.json", {
        "schema_version": "robot-v77-run-receipt-v1",
        "result": artifact_ref(result_path),
        "phase_evidence_count": len(phase_evidence),
    })
    atomic_new_json(attempt / "ARTIFACT_MANIFEST.json", {
        "schema_version": "artifact-manifest-v1",
        "artifacts": [artifact_ref(result_path), *phase_evidence],
    })
    _write_exact(
        attempt / "DECISION.md",
        (
            "# Decision\n\n"
            f"Attempt stopped during {args.failure_phase}: {args.error}. "
            "Its partial outputs are retained only as failure evidence and "
            "must not be consumed by attempt 0002.\n"
        ).encode(),
    )
    atomic_new_json(attempt / "NEXT_ACTION.json", {
        "next_task_id": args.next_task_id,
        "status": "READY",
        "fresh_attempt_required": True,
        "reuse_partial_outputs": False,
    })
    _local_heartbeat(attempt / "HEARTBEAT.json", "attempt_failed_runtime_retryable", None)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _evaluated_hard_gates(terminal: dict[str, Any]) -> dict[str, Any]:
    """Return measured hard gates, excluding fail-closed unexecuted phases."""
    gates = terminal.get("hard_gates", {})
    if terminal.get("source_kind") == "FRESH_V77_ISOLATED_SESSION_TEMPORAL_FAILURE":
        return {"hawor_temporal_successor": gates.get("hawor_temporal_successor")}
    return gates


def build_index(args: argparse.Namespace) -> int:
    bootstrap_root = args.bootstrap_root.resolve(strict=True)
    selection, _ = _verify_frozen_root(bootstrap_root)
    packet_path = args.task_packet.resolve(strict=True) if args.task_packet else None
    if packet_path is not None:
        packet = _load_object(packet_path, "v77 terminal-index Task Packet")
        if packet.get("task_id") != "robot_v77_terminal_index_r22":
            raise RobotTargetContractError("registered terminal-index Task Packet required")
        declared_writes = [Path(path).resolve() for path in packet.get("write_set", [])]
        if not any(args.output.resolve().is_relative_to(path) for path in declared_writes):
            raise RobotTargetContractError("terminal-index output is outside registered write_set")
    sources = [bootstrap_root / "RESULT.json"]
    sources.extend(path.resolve(strict=True) for path in args.batch_result)
    rows: list[dict[str, Any]] = []
    source_refs = []
    for source in sources:
        value = _load_object(source, "terminal source")
        source_refs.append(artifact_ref(source))
        rows.extend(value.get("rows", []))
    sessions = [row["session"] for row in rows]
    if len(sessions) != len(set(sessions)):
        raise RobotTargetContractError("duplicate session across v77 terminal sources")
    expected = list(selection["sessions"])
    unexpected = sorted(set(sessions) - set(expected))
    if unexpected:
        raise RobotTargetContractError(f"unexpected v77 terminals: {unexpected}")
    missing = [session for session in expected if session not in set(sessions)]
    hard_pass = sum(row["terminal_status"] == "PASSED" for row in rows)
    failed_c = sum(row["terminal_status"] == "FAILED_QUALITY_C" for row in rows)
    processed = len(rows)
    status = "PASSED_TERMINAL_COVERAGE" if not missing else "DEVELOPMENT_PARTIAL"
    hard_gate_failures: dict[str, int] = {}
    available_metric_fields: set[str] = set()
    session_metric_evidence = []
    for row in rows:
        terminal_path = _exact_ref(row["result"], f"{row['session']}.terminal_result")
        terminal = _load_object(terminal_path, f"{row['session']} terminal")
        # A temporal-successor quality terminal is emitted before arm, hand,
        # collision and limit evaluation.  Its fail-closed booleans prevent
        # downstream use, but they are not measurements of those downstream
        # gates and must not be reported as collision/limit/finite failures.
        # Count only the gate that was actually evaluated in that case.
        for gate, passed in _evaluated_hard_gates(terminal).items():
            if passed is False:
                hard_gate_failures[gate] = hard_gate_failures.get(gate, 0) + 1
        for group, values in terminal.get("metrics", {}).items():
            if isinstance(values, dict):
                available_metric_fields.update(f"{group}.{key}" for key in values)
        session_metric_evidence.append(artifact_ref(terminal_path))
    yield_metrics = {
        "schema_version": "robot-yield-20-v1",
        "created_at": now_iso(),
        "processed_sessions": processed,
        "hard_geometry_pass_sessions": hard_pass,
        "failed_quality_c_sessions": failed_c,
        "hard_pass_over_processed": hard_pass / processed if processed else None,
        "hard_pass_over_frozen_25": hard_pass / EXPECTED_SELECTION_COUNT,
        "hard_pass_over_upstream_101": hard_pass / UPSTREAM_DENOMINATOR,
        "hard_gate_failure_counts": hard_gate_failures,
        "collision_failure_sessions": hard_gate_failures.get("fullsession_collision_absent", 0),
        "joint_limit_failure_sessions": hard_gate_failures.get("urdf_joint_limits", 0),
        "finite_state_failure_sessions": hard_gate_failures.get("finite_valid_states", 0),
        "arm_temporal_or_unknown_failure_sessions": hard_gate_failures.get("arm_temporal_and_unknown_contract", 0),
        "action_amplitude": {"value": None, "status": "NOT_AVAILABLE_IN_PINNED_V77_TERMINAL_EVIDENCE"},
        "wrist_object_approach": {"value": None, "status": "NOT_AVAILABLE_IN_PINNED_V77_TERMINAL_EVIDENCE"},
        "ik_frame_valid_rate": {"value": None, "status": "REQUIRES_SEPARATE_PROFILE_PASS"},
        "collision_evidence": {
            "available": ["fullsession_collision_absent", "digital_collision_geometry_loaded"],
            "missing": ["penetration_depth", "penetration_volume", "collision_count"],
        },
        "available_terminal_metric_fields": sorted(available_metric_fields),
        "missing_profile_fields": [
            "action_amplitude",
            "wrist_object_approach",
            "ik_frame_valid_rate",
            "per_phase_p50_seconds",
            "per_phase_p95_seconds",
        ],
        "session_metric_evidence": session_metric_evidence,
        "claim_limit": "Session-level digital conversion yield; missing framewise reach metrics are not guessed.",
    }
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    atomic_new_json(output / "ROBOT_YIELD_20.json", yield_metrics)
    result = {
        "schema_version": "robot-v77-terminal-index-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "terminal_status": "PASSED" if (not missing or args.development_partial) else "BLOCKED_PREREQ",
        "status": status,
        "counts": {
            "expected": len(expected),
            "processed": processed,
            "hard_geometry_pass": hard_pass,
            "failed_quality_c": failed_c,
            "missing": len(missing),
        },
        "missing_sessions": missing,
        "pending_batches": [
            row for row in selection["microbatches"]
            if any(session in set(missing) for session in row["sessions"])
        ],
        "sources": source_refs,
        "rows": rows,
        "yield_metrics": artifact_ref(output / "ROBOT_YIELD_20.json"),
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Frozen 25-session digital Robot terminal coverage and yield only; "
            "no Robot/contact/control/physical authority."
            if not missing
            else "Development partial digital Robot yield only; no Robot/contact/control/physical authority."
        ),
    }
    result_path = output / "RESULT.json"
    atomic_new_json(result_path, result)
    atomic_new_json(output / "METRICS.json", {"schema_version": "robot-v77-index-metrics-v1", **result["counts"]})
    atomic_new_json(output / "RUN_RECEIPT.json", {"schema_version": "robot-v77-run-receipt-v1", "result": artifact_ref(result_path), "selection": artifact_ref(bootstrap_root / "SELECTION.json"), "task_packet": artifact_ref(packet_path) if packet_path else None})
    atomic_new_json(output / "ARTIFACT_MANIFEST.json", {"schema_version": "artifact-manifest-v1", "artifacts": [artifact_ref(result_path), artifact_ref(output / "ROBOT_YIELD_20.json")]})
    decision_scope = "frozen terminal coverage" if not missing else "development partial coverage"
    pending_text = (
        "All five microbatches are closed"
        if not missing
        else f"Pending batches: {[row['batch_id'] for row in result['pending_batches']]}"
    )
    _write_exact(
        output / "DECISION.md",
        (
            f"# Decision\n\nV77 {decision_scope}: {processed}/{len(expected)}; "
            f"hard pass: {hard_pass}; quality C: {failed_c}. {pending_text}. "
            "Terminal coverage is not Robot authority; no authority is promoted.\n"
        ).encode(),
    )
    next_pending = next((row for row in selection["microbatches"] if any(session in set(missing) for session in row["sessions"])), None)
    atomic_new_json(output / "NEXT_ACTION.json", {
        "next_task_id": "V77-TERMINAL-COMPLETE" if not missing else f"V77-BATCH-{int(next_pending['batch_id']):03d}",
        "missing_sessions": missing,
        "status": "COMPLETE" if not missing else "READY",
    })
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not missing else 3


def repackage_yield(args: argparse.Namespace) -> int:
    """Publish the CPU-only yield task in its own declared immutable write set."""
    packet_path = args.task_packet.resolve(strict=True)
    packet = _load_object(packet_path, "Robot yield Task Packet")
    if packet.get("task_id") != "robot_yield_20_r22":
        raise RobotTargetContractError("registered robot_yield_20_r22 Task Packet required")
    output = args.output.resolve()
    declared_writes = [Path(path).resolve() for path in packet.get("write_set", [])]
    if not any(output.is_relative_to(path) for path in declared_writes):
        raise RobotTargetContractError("Robot yield output is outside registered write_set")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source_index_path = args.source_index.resolve(strict=True)
    source_yield_path = args.source_yield.resolve(strict=True)
    source_index = _load_object(source_index_path, "source partial terminal index")
    source_yield = _load_object(source_yield_path, "source Robot yield")
    source_status = source_index.get("status")
    if source_status not in {"DEVELOPMENT_PARTIAL", "PASSED_TERMINAL_COVERAGE"}:
        raise RobotTargetContractError(
            "source index must be DEVELOPMENT_PARTIAL or PASSED_TERMINAL_COVERAGE"
        )
    if source_index.get("yield_metrics") != artifact_ref(source_yield_path):
        raise RobotTargetContractError("source index does not bind source Robot yield")
    output.mkdir(parents=True)
    result = {
        "schema_version": "robot-yield-20-task-result-v1",
        "artifact_revision": ARTIFACT_REVISION,
        "created_at": now_iso(),
        "terminal_status": "PASSED",
        "status": (
            "DEVELOPMENT_COMPLETE" if source_status == "PASSED_TERMINAL_COVERAGE"
            else "DEVELOPMENT_PARTIAL"
        ),
        "counts": source_index["counts"],
        "missing_sessions": source_index["missing_sessions"],
        "pending_batches": source_index["pending_batches"],
        "source_partial_index": artifact_ref(source_index_path),
        "robot_yield_20": artifact_ref(source_yield_path),
        "task_packet": artifact_ref(packet_path),
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "CPU-only frozen 25-session digital Robot yield; terminal coverage is not "
            "Robot/contact/control/physical authority."
            if source_status == "PASSED_TERMINAL_COVERAGE"
            else "CPU-only development partial yield republish; terminal-index task remains "
            "pending and no authority is promoted."
        ),
    }
    result_path = output / "RESULT.json"
    atomic_new_json(result_path, result)
    metrics = {
        "schema_version": "robot-yield-20-task-metrics-v1",
        "processed_sessions": source_yield["processed_sessions"],
        "frozen_sessions": EXPECTED_SELECTION_COUNT,
        "upstream_sessions": UPSTREAM_DENOMINATOR,
        "hard_geometry_pass_sessions": source_yield["hard_geometry_pass_sessions"],
        "failed_quality_c_sessions": source_yield["failed_quality_c_sessions"],
        "hard_pass_over_processed": source_yield["hard_pass_over_processed"],
        "hard_pass_over_frozen_25": source_yield["hard_pass_over_frozen_25"],
        "hard_pass_over_upstream_101": source_yield["hard_pass_over_upstream_101"],
        "hard_gate_failure_counts": source_yield["hard_gate_failure_counts"],
        "missing_profile_fields": source_yield["missing_profile_fields"],
    }
    atomic_new_json(output / "METRICS.json", metrics)
    atomic_new_json(output / "RUN_RECEIPT.json", {
        "schema_version": "robot-v77-run-receipt-v1",
        "result": artifact_ref(result_path),
        "source_partial_index": artifact_ref(source_index_path),
        "source_robot_yield_20": artifact_ref(source_yield_path),
        "task_packet": artifact_ref(packet_path),
    })
    atomic_new_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "artifact-manifest-v1",
        "artifacts": [
            artifact_ref(result_path),
            artifact_ref(output / "METRICS.json"),
            artifact_ref(source_index_path),
            artifact_ref(source_yield_path),
        ],
    })
    is_complete = source_status == "PASSED_TERMINAL_COVERAGE"
    _write_exact(output / "DECISION.md", (
        "# Decision\n\nPublished Robot Yield 20 in its declared task write set. "
        + (
            "The frozen 25-session denominator is complete. Hard-pass yield remains "
            "development evidence and no authority is promoted.\n"
            if is_complete
            else "This is development-partial evidence only; terminal-index work remains pending.\n"
        )
    ).encode())
    atomic_new_json(output / "NEXT_ACTION.json", {
        "next_task_id": "V77-TERMINAL-COMPLETE" if is_complete else "V77-NEXT-PENDING-BATCH",
        "status": "COMPLETE" if is_complete else "PENDING_CAS",
        "start_computation": False,
        "pending_batches": [] if is_complete else [row["batch_id"] for row in source_index["pending_batches"]],
    })
    print(json.dumps(result, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    start = sub.add_parser("bootstrap")
    start.add_argument("--g0-snapshot", type=Path, required=True)
    start.add_argument("--g0-selection", type=Path, required=True)
    start.add_argument("--task-packet", type=Path, required=True)
    start.add_argument("--hard-soft-index", type=Path, required=True)
    start.add_argument("--matrix", type=Path, required=True)
    start.add_argument("--v76-code-closure", type=Path, required=True)
    start.add_argument("--output-root", type=Path, required=True)
    start.set_defaults(func=bootstrap)
    batch = sub.add_parser("run-batch")
    batch.add_argument("--bootstrap-root", type=Path, required=True)
    batch.add_argument("--output-root", type=Path, required=True)
    batch.add_argument("--task-packet", type=Path, required=True)
    batch.add_argument("--batch-id", type=int, required=True)
    batch.add_argument("--phase-timeout-seconds", type=int, default=7200)
    batch.add_argument("--heartbeat-seconds", type=int, default=25)
    batch.add_argument("--isolate-sessions", action="store_true")
    batch.set_defaults(func=run_batch)
    failure = sub.add_parser("close-runtime-failure")
    failure.add_argument("--output-root", type=Path, required=True)
    failure.add_argument("--failure-phase", required=True)
    failure.add_argument("--error", required=True)
    failure.add_argument("--next-task-id", default="V77-BATCH-RETRY-ATTEMPT-0002")
    failure.set_defaults(func=close_runtime_failure)
    index = sub.add_parser("build-index")
    index.add_argument("--bootstrap-root", type=Path, required=True)
    index.add_argument("--batch-result", type=Path, action="append", default=[])
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--task-packet", type=Path)
    index.add_argument("--development-partial", action="store_true")
    index.set_defaults(func=build_index)
    package = sub.add_parser("repackage-yield")
    package.add_argument("--task-packet", type=Path, required=True)
    package.add_argument("--source-index", type=Path, required=True)
    package.add_argument("--source-yield", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.set_defaults(func=repackage_yield)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
