#!/usr/bin/env python3
"""Add the frozen calibration-missing Poker Visual Tier rows and reserve.

This runs causal real-donor Clean and pose-only Robot only.  It never grants
Depth/Object6D/Contact/metric/physical authority.  A combined immutable matrix
is produced solely for H50 eligibility and downstream Visual Aux automation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
# The original R7_0 supervisor is terminal but predates the current immutable
# Robot expansion.  Waiting on it allowed Visual Tier pose-only Robot work to
# overlap the live R7_3 metric Robot writer.  Default to the current supervisor;
# callers can still pin another immutable state file explicitly.
UPSTREAM = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_finalize_robot_expansion_R7_3/AUTOMATION_STATE.json"
CURRENT_MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
HARD_SOFT_STATE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/AUTOMATION_STATE.json"
PLAN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/expansion_plan_R7_2/VISUAL_AUX_ROBOT_EXPANSION_PLAN.json"
CLEAN_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_clean_R7_2"
POSE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1"
TERMINAL_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_terminals_v1"
HARD_SOFT_CANDIDATE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_robot_hard_soft_watcher_v71/candidates"
DEFAULT_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2"
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any], *, no_clobber: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if no_clobber and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_bounded(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
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
            handle.write(f"\nAUTOMATION_TIMEOUT_SECONDS={timeout}\n")
            return 124


def snapshot_matrix(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload, no_clobber=True)
    snapshot = path.parent / "snapshots" / f"CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_{sha256(path)}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copyfile(path, snapshot)


def selected_sessions() -> list[str]:
    plan = load(PLAN)
    return [row["session_id"] for row in plan["selection"] if row["tier"] == "TIER_V_VISUAL"]


def update_task(task_id: str, status: str, phase: str, result: Path, message: str) -> None:
    for _ in range(12):
        revision = int(load(STATUS_RECEIPT)["governance_revision"])
        completed = subprocess.run([
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", task_id, "--status", status,
            "--phase", phase, "--clear-runtime", "--result", str(result.resolve()),
            "--message", message,
            "--expected-revision", str(revision),
        ], cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return
        output = (completed.stdout + completed.stderr).lower()
        if not any(token in output for token in ("revision conflict", "cas revision mismatch")):
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(1)
    raise RuntimeError("robot task CAS retry exhausted")


def initial_matrix(output: Path, sessions: list[str]) -> Path:
    path = output / "matrix_input" / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
    if path.is_file():
        return path
    payload = load(CURRENT_MATRIX)
    selected = set(sessions)
    for row in payload["rows"]:
        if row["session_id"] in selected:
            clean = CLEAN_ROOT / "clean_results" / row["session_id"] / "RESULT.json"
            row.update(clean_state="PASSED_GRADE_B", clean_result=ref(clean), robot_current_state="READY_FOR_ROBOT_CURRENT_DRAFT", final_robot_terminal_proven=False)
        elif row.get("robot_current_state") == "READY_FOR_ROBOT_CURRENT_DRAFT":
            row["robot_current_state"] = "BLOCKED_PREREQ_NOT_SELECTED_VISUAL_TIER_RUN"
    payload.update(
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        claim_limit="Visual Tier pose-only Robot input matrix; not current authority or metric geometry.",
    )
    snapshot_matrix(path, payload)
    return path


def combined_matrix(output: Path, sessions: list[str]) -> Path:
    path = output / "combined_eligibility" / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
    if path.is_file():
        return path
    payload = load(CURRENT_MATRIX)
    selected = set(sessions)
    for row in payload["rows"]:
        hard_soft = HARD_SOFT_CANDIDATE_ROOT / row["session_id"] / "RESULT.json"
        if hard_soft.is_file():
            row.update(
                formal_robot_terminal_state=row.get("robot_current_state"),
                robot_current_state="POSE_ONLY_VISUAL_ROBOT_REVIEW_READY",
                robot_candidate_result=ref(hard_soft),
                hard_soft_visual_aux_candidate=True,
            )
        if row["session_id"] not in selected:
            continue
        session = row["session_id"]
        clean = CLEAN_ROOT / "clean_results" / session / "RESULT.json"
        robot = POSE_ROOT / session / "RESULT.json"
        row.update(clean_state="PASSED_GRADE_B", clean_result=ref(clean), final_robot_terminal_proven=False)
        if robot.is_file():
            row.update(robot_current_state="POSE_ONLY_VISUAL_ROBOT_REVIEW_READY", robot_candidate_result=ref(robot))
        else:
            terminals = list(TERMINAL_ROOT.glob(f"*/{session}/RESULT.json"))
            row["robot_current_state"] = "FAILED_QUALITY_C" if terminals else "FAILED_RUNTIME_FINAL"
    payload.update(
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        claim_limit="Combined metric plus Visual Tier eligibility view only; not current authority.",
    )
    snapshot_matrix(path, payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upstream-state", type=Path, default=UPSTREAM)
    parser.add_argument("--hard-soft-state", type=Path, default=HARD_SOFT_STATE)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    parser.add_argument("--hard-soft-wait-seconds", type=int, default=604800)
    parser.add_argument("--clean-timeout-seconds", type=int, default=43200)
    parser.add_argument("--robot-timeout-seconds", type=int, default=172800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    state_path = output / "AUTOMATION_STATE.json"
    state = {
        "schema_version": "v71-post-robot-visual-tier-automation-v1",
        "status": "DRY_RUN_READY" if args.dry_run else "WAITING_METRIC_ROBOT_EXPANSION",
        "pid": os.getpid(), "upstream_state": str(args.upstream_state.resolve()),
        "hard_soft_state": str(args.hard_soft_state.resolve()),
        "claim_limit": "Frozen calibration-missing Visual Tier rows; causal Clean and pose-only Robot only, no metric/Contact/physical authority.",
    }
    atomic_json(state_path, state)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False)); return 0
    started = time.monotonic()
    while True:
        upstream = load(args.upstream_state) if args.upstream_state.is_file() else {}
        if upstream.get("status") == "TERMINAL": break
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="UPSTREAM_WAIT_BUDGET_EXHAUSTED"); atomic_json(state_path, state); return 3
        state.update(updated_at=datetime.now().astimezone().isoformat(timespec="seconds"), upstream_status=upstream.get("status", "MISSING")); atomic_json(state_path, state)
        time.sleep(max(5, args.poll_seconds))
    logs = output / "logs"
    clean_log = logs / "visual_tier_clean.log"
    clean_rc = run_bounded(
        [sys.executable, "src/chaoyang/ops/run_visual_tier_causal_clean_v71.py", "--plan", str(PLAN), "--matrix", str(CURRENT_MATRIX), "--output-root", str(CLEAN_ROOT)],
        clean_log, args.clean_timeout_seconds,
    )
    sessions = selected_sessions()
    robot_rc = 0
    input_matrix = None
    if clean_rc == 0:
        input_matrix = initial_matrix(output, sessions)
        robot_rc = run_bounded([
                sys.executable, "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py", "--matrix", str(input_matrix),
                "--output-root", str(output / "robot_batch"), "--task-id", "robot_geometry_v1",
                "--phase-timeout-seconds", "7200",
            ], logs / "visual_tier_robot.log", args.robot_timeout_seconds)
    eligibility_rc = 1
    combined = None
    eligibility = output / "combined_eligibility" / "ELIGIBILITY_INDEX.json"
    if clean_rc == robot_rc == 0:
        hard_soft_started = time.monotonic()
        while True:
            hard_soft = load(args.hard_soft_state) if args.hard_soft_state.is_file() else {}
            if hard_soft.get("status") == "TERMINAL":
                break
            if time.monotonic() - hard_soft_started > args.hard_soft_wait_seconds:
                eligibility_rc = 3
                break
            state.update(
                status="WAITING_HARD_SOFT_TERMINAL_FOR_COMBINED_ELIGIBILITY",
                updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                hard_soft_status=hard_soft.get("status", "MISSING"),
            )
            atomic_json(state_path, state)
            time.sleep(max(5, args.poll_seconds))
        if eligibility_rc != 3:
            combined = combined_matrix(output, sessions)
            eligibility_rc = run_bounded([
                    sys.executable, "src/chaoyang/human_ego/tools/build_visual_aux_eligibility_index_v56.py",
                    "--matrix", str(combined), "--output", str(eligibility),
                    "--h50-eligibility-mode", "ANY_ENDPOINT_40_OF_50",
                ], logs / "eligibility.log", 3600)
    result = {
        "schema_version": "v71-post-robot-visual-tier-result-v1",
        "status": "PASSED_DEVELOPMENT_VISUAL_TIER" if clean_rc == robot_rc == eligibility_rc == 0 else "FAILED_RUNTIME_FINAL",
        "sessions": sessions, "return_codes": {"clean": clean_rc, "robot": robot_rc, "hard_soft_terminal_gate": 0 if eligibility_rc != 3 else 3, "eligibility": eligibility_rc},
        "input_matrix": ref(input_matrix) if input_matrix else None,
        "combined_matrix": ref(combined) if combined else None,
        "eligibility": ref(eligibility) if eligibility.is_file() else None,
        "control_ground_truth": False, "metric_geometry": False, "claim_limit": state["claim_limit"],
    }
    result_path = output / "RESULT.json"; atomic_json(result_path, result)
    passed = result["status"].startswith("PASSED")
    update_task(
        "robot_geometry_v1",
        "PASSED" if passed else "FAILED_RUNTIME_FINAL",
        "visual_tier_pose_only_terminal_coverage_complete" if passed else "visual_tier_runtime_failed",
        result_path,
        "V7.1 metric and Visual Tier Robot Geometry terminals published; no control/physical authority",
    )
    update_task(
        "robotized_compositor_causal_v1",
        "BLOCKED_PREREQ" if passed else "FAILED_RUNTIME_FINAL",
        "wait_unified_zbuffer_causal_compositor" if passed else "visual_tier_runtime_failed",
        result_path,
        "Visual Tier Clean/pose-only geometry closed; causal unified-z-buffer Robotized RGB remains separate",
    )
    state.update(status="TERMINAL", result=ref(result_path)); atomic_json(state_path, state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"].startswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
