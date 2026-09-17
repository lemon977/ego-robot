#!/usr/bin/env python3
"""Resume exact78 pose-only Robot expansion after the bounded manual sprint.

The worker waits for the post-S1 finalizer and the frozen manual deadline.  It
then rebuilds the current matrix, runs only rows that are Clean-joined and still
READY, refreshes the projection eligibility index, and closes with an immutable
receipt.  It never promotes Robot/contact/occlusion/action authority.
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
FINALIZER_STATE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_s1_finalize/AUTOMATION_STATE.json"
MATRIX_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices"
MATRIX = MATRIX_ROOT / "CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
POSE_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1"
DEFAULT_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_finalize_robot_expansion"
ROBOT_BATCH = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v71"
ELIGIBILITY = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/revision_R7_2/ELIGIBILITY_INDEX.json"
EXPANSION_PLAN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/expansion_plan_R7_2/VISUAL_AUX_ROBOT_EXPANSION_PLAN.json"
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_logged(command: list[str], log: Path, timeout: int) -> int:
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
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            handle.write(f"\nAUTOMATION_TIMEOUT_SECONDS={timeout}\n")
            return 124


def update_task(status: str, phase: str, result: Path) -> None:
    for _ in range(12):
        revision = int(load(STATUS_RECEIPT)["governance_revision"])
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", "robot_geometry_v1", "--status", status,
            "--phase", phase, "--clear-runtime", "--result", str(result.resolve()),
            "--message", "V7.1 automatic pose-only Robot expansion receipt; no authority promotion",
            "--expected-revision", str(revision),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return
        if "revision conflict" not in (completed.stdout + completed.stderr).lower():
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(1)
    raise RuntimeError("governance CAS retry exhausted")


def ready_count() -> int:
    return sum(
        row.get("robot_current_state") == "READY_FOR_ROBOT_CURRENT_DRAFT"
        for row in load(MATRIX).get("rows", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalizer-state", type=Path, default=FINALIZER_STATE)
    parser.add_argument("--not-before", default="2026-09-15T04:16:18+08:00")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=259200)
    parser.add_argument("--robot-timeout-seconds", type=int, default=172800)
    parser.add_argument(
        "--robot-batch-root",
        type=Path,
        default=ROBOT_BATCH,
        help="Immutable output root for this Robot batch revision.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    state_path = output / "AUTOMATION_STATE.json"
    state = {
        "schema_version": "v71-post-finalize-robot-expansion-v1",
        "status": "DRY_RUN_READY" if args.dry_run else "WAITING_GATES",
        "pid": os.getpid(),
        "not_before": args.not_before,
        "finalizer_state": str(args.finalizer_state.resolve()),
        "claim_limit": "Pose-only Robot development expansion only; no Contact/Occlusion/Robot/action/physical authority.",
    }
    atomic_json(state_path, state)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False))
        return 0

    deadline = datetime.fromisoformat(args.not_before)
    started = time.monotonic()
    while True:
        finalizer = load(args.finalizer_state) if args.finalizer_state.is_file() else {}
        after_deadline = datetime.now().astimezone() >= deadline
        if finalizer.get("status") == "TERMINAL" and after_deadline:
            break
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="WAIT_GATE_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        state.update(
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            finalizer_status=finalizer.get("status", "MISSING"),
            after_deadline=after_deadline,
        )
        atomic_json(state_path, state)
        time.sleep(max(5, args.poll_seconds))

    logs = output / "logs"
    commands: list[tuple[str, list[str], int]] = [
        ("validate_before", [sys.executable, "-m", "chaoyang.governance.validate_governance_state"], 300),
        ("matrix_before", [sys.executable, "src/chaoyang/ops/build_exact78_v52_current_matrices.py", "--output-root", str(MATRIX_ROOT)], 900),
    ]
    return_codes: dict[str, int] = {}
    for name, command, timeout in commands:
        return_codes[name] = run_logged(command, logs / f"{name}.log", timeout)
        if return_codes[name] != 0:
            break

    if all(value == 0 for value in return_codes.values()):
        if not EXPANSION_PLAN.exists():
            return_codes["expansion_plan"] = run_logged(
                [sys.executable, "src/chaoyang/ops/build_visual_aux_robot_expansion_plan_v71.py", "--matrix", str(MATRIX), "--output", str(EXPANSION_PLAN)],
                logs / "expansion_plan.log", 600,
            )
        else:
            return_codes["expansion_plan"] = 0

    before = ready_count() if MATRIX.is_file() else -1
    if all(value == 0 for value in return_codes.values()) and before > 0:
        return_codes["robot_batch"] = run_logged(
            [
                sys.executable, "src/chaoyang/ops/run_exact78_robot_ready_batches_v53.py",
                "--matrix", str(MATRIX), "--output-root", str(args.robot_batch_root.resolve()),
                "--task-id", "robot_geometry_v1", "--phase-timeout-seconds", "7200",
            ],
            logs / "robot_batch.log", args.robot_timeout_seconds,
        )
    elif before == 0:
        return_codes["robot_batch"] = 0

    followups = [
        ("candidate_index", [sys.executable, "src/chaoyang/ops/audit_exact78_pose_only_visual_robot_v52.py", "--root", str(POSE_ROOT)], 1800),
        ("matrix_after", [sys.executable, "src/chaoyang/ops/build_exact78_v52_current_matrices.py", "--output-root", str(MATRIX_ROOT)], 900),
    ]
    if all(value == 0 for value in return_codes.values()):
        for name, command, timeout in followups:
            return_codes[name] = run_logged(command, logs / f"{name}.log", timeout)
            if return_codes[name] != 0:
                break
    if all(value == 0 for value in return_codes.values()) and not ELIGIBILITY.exists():
        return_codes["eligibility"] = run_logged(
            [sys.executable, "src/chaoyang/human_ego/tools/build_visual_aux_eligibility_index_v56.py", "--matrix", str(MATRIX), "--output", str(ELIGIBILITY)],
            logs / "eligibility.log", 3600,
        )

    after = ready_count() if MATRIX.is_file() else -1
    passed = bool(return_codes) and all(value == 0 for value in return_codes.values())
    result = {
        "schema_version": "v71-post-finalize-robot-expansion-result-v1",
        "status": "PASSED_DEVELOPMENT_EXPANSION" if passed else "FAILED_RUNTIME_FINAL",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ready_rows_before": before,
        "ready_rows_after": after,
        "return_codes": return_codes,
        "matrix": ref(MATRIX) if MATRIX.is_file() else None,
        "candidate_index": ref(POSE_ROOT / "CURRENT_CANDIDATE_INDEX.json") if (POSE_ROOT / "CURRENT_CANDIDATE_INDEX.json").is_file() else None,
        "eligibility": ref(ELIGIBILITY) if ELIGIBILITY.is_file() else None,
        "expansion_plan": ref(EXPANSION_PLAN) if EXPANSION_PLAN.is_file() else None,
        "control_ground_truth": False,
        "authority": False,
        "claim_limit": state["claim_limit"],
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    update_task(
        "BLOCKED_PREREQ" if passed else "FAILED_RUNTIME_FINAL",
        "pose_only_expanded_wait_unified_zbuffer_and_causal_compositor" if passed else "automatic_expansion_runtime_failed",
        result_path,
    )
    state.update(status="TERMINAL", result=ref(result_path))
    atomic_json(state_path, state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
