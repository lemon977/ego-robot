#!/usr/bin/env python3
"""Fail-forward Robot visual processing for every currently Clean-ready row."""

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
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.governance.common import artifact_ref, atomic_json, load_json, now_iso  # noqa: E402


LANE = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot"
MATRICES = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices"
RUN_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_robot_ready_batches_v53"
TASK_ID = "exact78_v52_lane_c_contact_robot"
PHASE_TIMEOUT_SECONDS = 7200
POLL_SECONDS = 25.0
POSE_ONLY_ROOT = LANE / "pose_only_visual_robot_v1"
TERMINAL_BASE = LANE / "robot_terminals_v1"
HAWOR_PYTHON = Path("/root/.venvs/wilor/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _phase_success_receipt(batch_root: Path, name: str) -> Path:
    return batch_root / "attempts" / f"{name}_SUCCESS.json"


def _validate_phase_reuse(
    batch_root: Path,
    name: str,
    command: list[str],
    expected: Path,
) -> bool:
    """Reuse only output bound to this exact phase command and local receipt."""
    if not expected.is_file():
        return False
    receipt_path = _phase_success_receipt(batch_root, name)
    if not receipt_path.is_file():
        raise RuntimeError(f"{name}: expected exists without phase success receipt: {expected}")
    receipt = load_json(receipt_path)
    expected_ref = artifact_ref(expected)
    valid = (
        receipt.get("schema_version") == "robot-phase-success-receipt-v1"
        and receipt.get("phase") == name
        and receipt.get("command_sha256") == _phase_command_sha256(command)
        and receipt.get("expected") == expected_ref
    )
    if not valid:
        raise RuntimeError(f"{name}: phase success receipt mismatch for {expected}")
    return True


def _publish_phase_success(
    batch_root: Path,
    name: str,
    command: list[str],
    expected: Path,
) -> None:
    receipt_path = _phase_success_receipt(batch_root, name)
    payload = {
        "schema_version": "robot-phase-success-receipt-v1",
        "phase": name,
        "command": command,
        "command_sha256": _phase_command_sha256(command),
        "expected": artifact_ref(expected),
        "created_at": now_iso(),
    }
    if receipt_path.exists():
        _validate_phase_reuse(batch_root, name, command, expected)
        return
    atomic_json(receipt_path, payload)


def heartbeat(status: str, phase: str, session: str | None = None) -> None:
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
    completed = subprocess.run(
        command, cwd=PROJECT, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            f"stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}"
        )


def run_phase(
    batch_root: Path,
    name: str,
    command: list[str],
    expected: Path,
    session_label: str,
) -> None:
    if _validate_phase_reuse(batch_root, name, command, expected):
        return
    attempts = batch_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 3):
        log = attempts / f"{name}_attempt_{attempt:04d}.log"
        if log.exists():
            continue
        heartbeat("RUNNING", f"robot_batch_{name}", session_label)
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            while process.poll() is None:
                time.sleep(POLL_SECONDS)
                heartbeat("RUNNING", f"robot_batch_{name}", session_label)
                if time.monotonic() - started > PHASE_TIMEOUT_SECONDS:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
                    handle.write(f"\nPHASE_TIMEOUT_SECONDS={PHASE_TIMEOUT_SECONDS}\n")
                    break
            handle.flush()
            os.fsync(handle.fileno())
        if process.returncode in {0, 2} and expected.is_file():
            _publish_phase_success(batch_root, name, command, expected)
            return
    raise RuntimeError(f"{name}: two runtime attempts exhausted; expected={expected}")


def only(rows: list[dict[str, Any]], session: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("session") == session]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {session}")
    return selected[0]


def exact_path(reference: dict[str, Any], label: str) -> str:
    path = Path(reference["path"])
    if (
        not path.is_file()
        or path.stat().st_size != reference["bytes"]
        or sha256(path) != reference["sha256"]
    ):
        raise RuntimeError(f"{label} reference mismatch")
    return str(path)


def final_hand_artifacts(row: dict[str, Any], session: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the terminal Hand artifacts from the status-specific schema.

    The bounded round-2 aggregator intentionally keeps carried round-1 results
    under ``method1_result``/``method1_states``.  Round-2 pass/hold rows use
    ``result``/``states``.  Treating both layouts as the latter caused the v7.4
    finalizer crash after otherwise complete Hand and Render phases.

    Unknown statuses and incomplete rows fail closed before a terminal can be
    published.  This function never guesses from whichever key happens to be
    present.
    """
    status = row.get("status")
    if status == "PASS_HAND_METHOD1_CARRIED_NO_ROUND2":
        keys = ("method1_result", "method1_states")
    elif status in {
        "PASS_HAND_ROUND2_BIDIRECTIONAL",
        "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C",
    }:
        keys = ("result", "states")
    else:
        raise RuntimeError(f"{session}: unsupported final hand status {status!r}")
    missing = [key for key in keys if not isinstance(row.get(key), dict)]
    if missing:
        raise RuntimeError(
            f"{session}: final hand row {status!r} is missing required artifact fields {missing}"
        )
    return row[keys[0]], row[keys[1]]


def finalize_sessions(
    preflight_path: Path,
    temporal_root: Path,
    arm_root: Path,
    hand_root: Path,
    render_root: Path,
) -> None:
    preflight = load_json(preflight_path)
    temporal = load_json(temporal_root / "RESULT.json")
    arm = load_json(arm_root / "RESULT.json")
    hand = load_json(hand_root / "RESULT.json")
    render = load_json(render_root / "RESULT.json")
    temporal_rows = temporal["sessions"]
    arm_rows = arm["sessions"]
    hand_rows = hand["sessions"]
    review_rows = render["sessions"]
    for input_row in preflight["sessions"]:
        task, session = input_row["task"], input_row["session"]
        temporal_row = only(temporal_rows, session)
        arm_row = only(arm_rows, session)
        hand_row = only(hand_rows, session)
        review_row = only(review_rows, session)
        arm_pass = arm_row["status"] in {
            "PASS_ARM_METHOD1_CARRIED_NO_ROUND2",
            "PASS_ARM_ROUND2_BIDIRECTIONAL",
        }
        hand_pass = hand_row["status"] in {
            "PASS_HAND_METHOD1_CARRIED_NO_ROUND2",
            "PASS_HAND_ROUND2_BIDIRECTIONAL",
        }
        hand_result_ref, hand_states_ref = final_hand_artifacts(hand_row, session)
        if arm_pass and hand_pass:
            final = POSE_ONLY_ROOT / session / "RESULT.json"
            if final.is_file():
                continue
            command = [
                sys.executable,
                "-m",
                "chaoyang.ops.adopt_exact78_pose_only_visual_robot_v52",
                "--session",
                session,
                "--legacy-result",
                exact_path(review_row["result"], f"{session}.review"),
                "--output-root",
                str(POSE_ONLY_ROOT),
            ]
            run_phase(
                RUN_ROOT / "finalize" / session,
                "adopt_pose_only",
                command,
                final,
                session,
            )
            continue
        final = TERMINAL_BASE / task / session / "RESULT.json"
        if final.is_file():
            continue
        rounds = max(
            int(arm_row.get("arm_method_rounds_consumed", 1)),
            int(hand_row.get("hand_method_rounds_consumed", 1)),
        )
        command = [
            sys.executable,
            "src/chaoyang/ops/publish_exact78_robot_failed_quality_terminal_v52.py",
            "--task",
            task,
            "--session",
            session,
            "--clean-result",
            exact_path(input_row["clean_result"], f"{session}.clean"),
            "--hawor-result",
            exact_path(temporal_row["result"], f"{session}.temporal"),
            "--arm-result",
            exact_path(hand_row["arm_result"], f"{session}.arm"),
            "--arm-states",
            exact_path(hand_row["arm_states"], f"{session}.arm_states"),
            "--hand-result",
            exact_path(hand_result_ref, f"{session}.hand"),
            "--hand-states",
            exact_path(hand_states_ref, f"{session}.hand_states"),
            "--review-result",
            exact_path(review_row["result"], f"{session}.review"),
            "--method-rounds-consumed",
            str(rounds),
            "--output-root",
            str(TERMINAL_BASE),
        ]
        run_phase(
            RUN_ROOT / "finalize" / session,
            "publish_quality_c",
            command,
            final,
            session,
        )


def batch_commands(
    batch_root: Path,
    sessions: list[str],
    matrix_snapshot: Path,
    *,
    prepared_inputs: bool = False,
) -> None:
    label = "__".join(sessions)
    contract = batch_root / "ROBOT_READY_INPUT.json"
    preflight = batch_root / "preflight" / "RESULT.json"
    temporal_contract = batch_root / "HAWOR_TEMPORAL_CONTRACT.json"
    temporal_root = batch_root / "hawor_temporal"
    temporal_audit = temporal_root / "AUDIT.json"
    arm1_root = batch_root / "arm_round1"
    placement_root = batch_root / "arm_placement"
    arm2_root = batch_root / "arm_round2"
    hand1_root = batch_root / "hand_round1"
    hand2_root = batch_root / "hand_round2"
    render_root = batch_root / "render"
    if not prepared_inputs:
        run_phase(
            batch_root,
            "build_contract",
            [
                sys.executable,
                "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py",
                "--matrix",
                str(matrix_snapshot),
                "--sessions",
                *sessions,
                "--output",
                str(contract),
            ],
            contract,
            label,
        )
    elif not contract.is_file() or not preflight.is_file():
        raise RuntimeError("prepared Robot inputs require existing contract and preflight")
    frozen_matrix = Path(load_json(contract)["matrix_snapshot"]["path"])
    if not prepared_inputs:
        run_phase(
            batch_root,
            "preflight",
            [
                sys.executable,
                "src/chaoyang/ops/preflight_exact78_robot_ready_batch_v52.py",
                "--contract",
                str(contract),
                "--matrix",
                str(frozen_matrix),
                "--output",
                str(preflight),
            ],
            preflight,
            label,
        )
    run_phase(
        batch_root,
        "build_temporal_contract",
        [
            sys.executable,
            "src/chaoyang/ops/build_exact78_hawor_temporal_contract_v52.py",
            "--robot-contract",
            str(contract),
            "--output",
            str(temporal_contract),
        ],
        temporal_contract,
        label,
    )
    run_phase(
        batch_root,
        "hawor_temporal",
        [
            str(HAWOR_PYTHON),
            "src/chaoyang/ops/run_hawor_temporal_jerk_successor.py",
            "--contract",
            str(temporal_contract),
            "--output-root",
            str(temporal_root),
        ],
        temporal_root / "RESULT.json",
        label,
    )
    run_phase(
        batch_root,
        "temporal_audit",
        [
            sys.executable,
            "src/chaoyang/ops/audit_exact78_hawor_temporal_batch_v52.py",
            "--preflight",
            str(preflight),
            "--contract",
            str(temporal_contract),
            "--aggregate",
            str(temporal_root / "RESULT.json"),
            "--output",
            str(temporal_audit),
        ],
        temporal_audit,
        label,
    )
    phases = [
        (
            "arm_round1",
            [sys.executable, "src/chaoyang/ops/run_exact78_robot_arm_prior_canaries_v52.py", "--preflight", str(preflight), "--temporal-root", str(temporal_root), "--output-root", str(arm1_root)],
            arm1_root / "RESULT.json",
        ),
        (
            "arm_placement",
            [sys.executable, "src/chaoyang/ops/run_exact78_robot_arm_placement_sweep_v52.py", "--preflight", str(preflight), "--round1-prior-result", str(arm1_root / "RESULT.json"), "--output-root", str(placement_root), "--max-workers", "3"],
            placement_root / "RESULT.json",
        ),
        (
            "arm_round2",
            [sys.executable, "src/chaoyang/ops/run_exact78_robot_arm_bidirectional_round2_v52.py", "--preflight", str(preflight), "--placement-result", str(placement_root / "RESULT.json"), "--output-root", str(arm2_root), "--max-workers", "3"],
            arm2_root / "RESULT.json",
        ),
        (
            "hand_round1",
            [sys.executable, "src/chaoyang/ops/run_exact78_robot_hand_round1_v52.py", "--preflight", str(preflight), "--temporal-root", str(temporal_root), "--arm-final-result", str(arm2_root / "RESULT.json"), "--output-root", str(hand1_root), "--max-workers", "3"],
            hand1_root / "RESULT.json",
        ),
        (
            "hand_round2",
            [sys.executable, "src/chaoyang/ops/run_exact78_robot_hand_bidirectional_round2_v52.py", "--preflight", str(preflight), "--temporal-root", str(temporal_root), "--hand-round1-result", str(hand1_root / "RESULT.json"), "--output-root", str(hand2_root)],
            hand2_root / "RESULT.json",
        ),
        (
            "render",
            [sys.executable, "src/chaoyang/ops/render_exact78_robot_passed_batch_v52.py", "--preflight", str(preflight), "--temporal-root", str(temporal_root), "--arm-final-result", str(arm2_root / "RESULT.json"), "--hand-final-result", str(hand2_root / "RESULT.json"), "--output-root", str(render_root)],
            render_root / "RESULT.json",
        ),
    ]
    for name, command, expected in phases:
        run_phase(batch_root, name, command, expected, label)
    finalize_sessions(preflight, temporal_root, arm2_root, hand2_root, render_root)


def main() -> int:
    global RUN_ROOT, TASK_ID, PHASE_TIMEOUT_SECONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRICES / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--task-id",
        default=TASK_ID,
        help="Governance task receiving heartbeats; V7.1 automation uses robot_geometry_v1.",
    )
    parser.add_argument("--phase-timeout-seconds", type=int, default=PHASE_TIMEOUT_SECONDS)
    args = parser.parse_args()
    RUN_ROOT = args.output_root.resolve()
    TASK_ID = args.task_id
    if args.phase_timeout_seconds < 60:
        raise ValueError("--phase-timeout-seconds must be >=60")
    PHASE_TIMEOUT_SECONDS = args.phase_timeout_seconds
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    matrix_path = args.matrix.resolve(strict=True)
    matrix = load_json(matrix_path)
    current_ready = [row["session_id"] for row in matrix["rows"] if row["robot_current_state"] == "READY_FOR_ROBOT_CURRENT_DRAFT"]
    selection_path = RUN_ROOT / "SELECTION.json"
    if not selection_path.exists():
        if not current_ready:
            raise RuntimeError("no current Clean-ready Robot rows")
        ready = current_ready
        atomic_json(
            selection_path,
            {
                "schema_version": "exact78-robot-ready-batches-selection-v53",
                "created_at": now_iso(),
                "status": "FROZEN_READY_ROWS",
                "matrix": artifact_ref(matrix_path),
                "sessions": ready,
            },
        )
    else:
        selected = load_json(selection_path)
        ready = list(selected["sessions"])
        if not ready or len(ready) != len(set(ready)):
            raise RuntimeError("existing selection is empty or contains duplicates")
        matrix_by_session = {row["session_id"]: row for row in matrix["rows"]}
        missing = [session for session in ready if session not in matrix_by_session]
        invalid = [
            session for session in ready
            if matrix_by_session[session]["robot_current_state"] not in {
                "READY_FOR_ROBOT_CURRENT_DRAFT",
                "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY",
                "FAILED_QUALITY_C",
            }
        ]
        if missing or invalid:
            raise RuntimeError(f"existing selection cannot resume: missing={missing} invalid={invalid}")
    heartbeat("RUNNING", "robot_ready_batches_start", ready[0])
    for index in range(0, len(ready), 3):
        sessions = ready[index : index + 3]
        batch_commands(RUN_ROOT / f"batch_{index // 3 + 1:03d}", sessions, matrix_path)
        subprocess.run(
            [sys.executable, "src/chaoyang/ops/build_exact78_v52_current_matrices.py", "--output-root", str(MATRICES)],
            cwd=PROJECT,
            check=True,
        )
    result = RUN_ROOT / "RESULT.json"
    atomic_json(
        result,
        {
            "schema_version": "exact78-robot-ready-batches-result-v53",
            "created_at": now_iso(),
            "status": "PASSED_BATCH_TERMINAL_COVERAGE",
            "selection": artifact_ref(selection_path),
            "sessions": ready,
            "matrix_after": artifact_ref(MATRICES / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"),
            "claim_limit": "Digital visual Robot terminal coverage only; control_ground_truth=false and physical_deployment=false.",
        },
    )
    heartbeat("CLAIMED", "wait_clean_runtime_recovery_or_next_batch", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
