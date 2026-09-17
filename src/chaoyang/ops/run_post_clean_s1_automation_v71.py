#!/usr/bin/env python3
from __future__ import annotations

"""Wait for the frozen Clean wave and bounded sprint, then run S1 canaries.

This is intentionally a narrow automation handoff.  It never updates the
governance ledger and never promotes authority.  The GPU adapter owns the
V7.1 lease only during model construction/inference and publishes immutable
attempt receipts.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STATUS_MIN = ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"
SPRINT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/OPTIMIZATION_SPRINT_V71.json"
DEFAULT_INPUT_INDEX = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
    "s1_input_preflight/S1_INPUT_PREFLIGHT_INDEX.json"
)
DEFAULT_OUTPUT = ROOT / (
    "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/"
    "s1_object_identity_gpu_canary/revision_R7_3"
)
DEFAULT_AUTOMATION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/automation/post_clean_s1"
ROLE_REPAIR_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/role_mask_runtime_contract_repair_poker042/R7_1"
ROBOT_ZBUFFER_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/robot_geometry/canaries/get_potato_chips_0902_023/unified_zbuffer_fullsession_R7_2"
ROBOT_HAWOR = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/hawor_temporal_candidates_v1/batch_023_v1/chips/get_potato_chips_0902_023/HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz"
ROBOT_ARM = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_numeric_candidates_v1/batch_023_anchor_v3_method1_placement_v1/chips/get_potato_chips_0902_023/arm_method1_backoff_0p2/ARM_CANARY_STATES.npz"
ROBOT_HAND = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/robot_numeric_candidates_v1/batch_023_anchor_v3_hand_round1_singleton_patch_v2/chips/get_potato_chips_0902_023/hand_round1_forward/HAND_STATES.npz"
LEASE = ROOT / "_run/current/GPU_LEASE.json"
LEASE_LOCK = ROOT / "_run/current/GPU_LEASE.lock"
CROSS_STAGE_MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
TERMINAL = {"PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_REFERENCE_PROOF"}
ROLE_REGRESSIONS = (
    (
        "play_cards_0903_189",
        ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_bounded_v2_1_configs/play_cards_0903_189.json",
        "ab15e8cc24aa1db61fa23fd1584639d384e8fac5ba93e5cffbcf35f7c105a259",
    ),
    (
        "play_cards_0903_202",
        ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_bounded_v2_1_configs/play_cards_0903_202.json",
        "6b82aa598feaa28850ecbb87bf70b19b80cccf55df6f01106fe83f48f3a7f61d",
    ),
)


def now() -> datetime:
    return datetime.now().astimezone()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_is_closed(status: dict[str, Any]) -> bool:
    waves = status.get("waves", {})
    if int(waves.get("wave0_clean_pending", -1)) != 0:
        return False
    for task in status.get("active_tasks", []):
        if "clean" in str(task.get("task_id", "")).lower():
            return False
    return True


def validate_governance() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0, (result.stdout + result.stderr)[-4000:]


def selected_input(index_path: Path, session_id: str) -> dict[str, Any]:
    index = load_json(index_path)
    rows = [row for row in index["rows"] if row["session_id"] == session_id]
    if len(rows) != 1 or rows[0]["status"] != "PASSED":
        raise RuntimeError(f"S1 input is not uniquely PASSED: {session_id}")
    return rows[0]


def previous_attempts(output_root: Path, session_id: str, backend: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base = output_root / backend / "sessions" / session_id / "attempts"
    for path in sorted(base.glob("attempt_*/RESULT.json")):
        try:
            value = load_json(path)
            value["_path"] = str(path.resolve())
            results.append(value)
        except Exception:
            continue
    return results


def payload_status(result: dict[str, Any]) -> str:
    return str(result.get("payload", result).get("status", "UNKNOWN"))


def run_role_repair(*, automation_root: Path, executor_epoch: int, timeout_seconds: int = 3600,
                    role_root: Path | None = None, artifact_revision: str = "R7_1") -> dict[str, Any]:
    role_root = ROLE_REPAIR_ROOT if role_root is None else role_root
    attempts = role_root / "sessions/play_cards_0901_042/attempts"
    existing = sorted(attempts.glob("attempt_*/RESULT.json"))
    for path in existing:
        value = load_json(path)
        if value.get("status") in {"PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ"}:
            return {"task": "role_mask_runtime_contract_repair", "status": value["status"], "adopted": True, "result": str(path.resolve())}
    attempt_id = f"attempt_{len(existing) + 1:04d}"
    token = secrets.token_hex(24)
    log_path = automation_root / f"role_mask_poker042_{attempt_id}.log"
    command = [
        sys.executable, "src/chaoyang/ops/run_role_mask_runtime_contract_repair_v71.py",
        "--output-root", str(role_root), "--attempt-id", attempt_id,
        "--artifact-revision", artifact_revision,
        "--executor-epoch", str(executor_epoch), "--fencing-token", token,
        "--lease-path", str(LEASE), "--lease-lock-path", str(LEASE_LOCK),
    ]
    automation_root.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False, timeout=timeout_seconds,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nAUTOMATION_TIMEOUT\n")
    result_path = attempts / attempt_id / "RESULT.json"
    status = load_json(result_path).get("status") if result_path.is_file() else "FAILED_RUNTIME_FINAL"
    result = {
        "task": "role_mask_runtime_contract_repair", "attempt_id": attempt_id,
        "status": status, "return_code": return_code, "log": str(log_path.resolve()),
        "result": str(result_path.resolve()) if result_path.is_file() else None,
    }
    atomic_json(automation_root / "role_mask_poker042_LATEST.json", result)
    return result


def run_robot_zbuffer_fullsession(*, automation_root: Path, timeout_seconds: int = 3600) -> dict[str, Any]:
    result_path = ROBOT_ZBUFFER_OUTPUT / "RESULT.json"
    if result_path.is_file():
        return {
            "task": "robot_unified_zbuffer_fullsession",
            "status": load_json(result_path).get("status", "UNKNOWN"),
            "adopted": True,
            "result": str(result_path.resolve()),
        }
    log_path = automation_root / "robot_unified_zbuffer_fullsession.log"
    command = [
        sys.executable, "src/chaoyang/ops/export_robot_unified_zbuffer_canary_v71.py",
        "--session-id", "get_potato_chips_0902_023",
        "--hawor", str(ROBOT_HAWOR), "--arm-states", str(ROBOT_ARM),
        "--hand-states", str(ROBOT_HAND), "--frame-count", "420",
        "--output-dir", str(ROBOT_ZBUFFER_OUTPUT),
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False, timeout=timeout_seconds,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nAUTOMATION_TIMEOUT\n")
    status = load_json(result_path).get("status") if result_path.is_file() else "FAILED_RUNTIME_FINAL"
    return {
        "task": "robot_unified_zbuffer_fullsession", "status": status,
        "return_code": return_code, "log": str(log_path.resolve()),
        "result": str(result_path.resolve()) if result_path.is_file() else None,
    }


def run_role_regression(
    *, session_id: str, config: Path, config_sha: str,
    automation_root: Path, executor_epoch: int, timeout_seconds: int = 3600,
    role_root: Path | None = None, artifact_revision: str = "R7_1",
) -> dict[str, Any]:
    role_root = ROLE_REPAIR_ROOT if role_root is None else role_root
    output_root = role_root / "regressions"
    attempts = output_root / "sessions" / session_id / "attempts"
    existing = sorted(attempts.glob("attempt_*/RESULT.json"))
    for path in existing:
        value = load_json(path)
        if value.get("status") in {"PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ"}:
            return {"task": f"role_mask_regression_{session_id}", "status": value["status"], "adopted": True, "result": str(path.resolve())}
    attempt_id = f"attempt_{len(existing) + 1:04d}"
    token = secrets.token_hex(24)
    automation_root.mkdir(parents=True, exist_ok=True)
    log_path = automation_root / f"role_mask_regression_{session_id}_{attempt_id}.log"
    command = [
        sys.executable, "src/chaoyang/ops/run_role_mask_runtime_contract_repair_v71.py",
        "--output-root", str(output_root), "--config", str(config),
        "--confirm-config-sha", config_sha, "--attempt-id", attempt_id,
        "--artifact-revision", artifact_revision,
        "--executor-epoch", str(executor_epoch), "--fencing-token", token,
        "--lease-path", str(LEASE), "--lease-lock-path", str(LEASE_LOCK),
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False, timeout=timeout_seconds,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nAUTOMATION_TIMEOUT\n")
    result_path = attempts / attempt_id / "RESULT.json"
    status = load_json(result_path).get("status") if result_path.is_file() else "FAILED_RUNTIME_FINAL"
    return {
        "task": f"role_mask_regression_{session_id}", "status": status,
        "return_code": return_code, "log": str(log_path.resolve()),
        "result": str(result_path.resolve()) if result_path.is_file() else None,
    }


def publish_role_bundle(rows: list[dict[str, Any]], *, role_root: Path | None = None,
                        artifact_revision: str = "R7_1") -> dict[str, Any]:
    role_root = ROLE_REPAIR_ROOT if role_root is None else role_root
    path = role_root / "ROLE_MASK_1_PLUS_2_BUNDLE_RESULT.json"
    if path.is_file():
        value = load_json(path)
        return {"task": "role_mask_1_plus_2_bundle", "status": value["status"], "adopted": True, "result": str(path.resolve())}
    statuses = [str(row.get("status")) for row in rows]
    complete = len(rows) == 3
    matrix_rows = {row["session_id"]: row for row in load_json(CROSS_STAGE_MATRIX)["rows"]}
    expected_sessions = ["play_cards_0901_042", *(item[0] for item in ROLE_REGRESSIONS)]
    actual_sessions = [
        load_json(Path(str(row["result"]))).get("session_id")
        for row in rows if row.get("result") and Path(str(row["result"])).is_file()
    ]
    regression_ab = all(matrix_rows[session]["role_mask"]["grade"] == "B" for session in expected_sessions[1:])
    complete = complete and actual_sessions == expected_sessions and regression_ab
    status = "PASSED" if complete and statuses == ["PASSED", "PASSED", "PASSED"] else (
        "FAILED_QUALITY_C" if "FAILED_QUALITY_C" in statuses else (
            "BLOCKED_RESOURCE" if "BLOCKED_RESOURCE" in statuses else "FAILED_RUNTIME_FINAL"
        )
    )
    value = {
        "schema_version": "role-mask-1-plus-2-bundle-result-v71",
        "artifact_revision": artifact_revision, "status": status,
        "canary": rows[0] if rows else None, "regressions": rows[1:],
        "complete_one_failure_plus_two_ab": complete,
        "selection_evidence": {
            "matrix": {"path": str(CROSS_STAGE_MATRIX.resolve()), "sha256": sha256(CROSS_STAGE_MATRIX)},
            "failure_canary": expected_sessions[0], "frozen_ab_regressions": expected_sessions[1:],
            "regressions_were_role_mask_grade_b": regression_ab,
        },
        "authority": False,
        "claim_limit": "1 failure canary plus 2 frozen A/B regressions; task completion is not automatic Role Mask authority.",
    }
    atomic_json(path, value)
    return {"task": "role_mask_1_plus_2_bundle", "status": status, "result": str(path.resolve())}


def run_backend(
    *, backend: str, row: dict[str, Any], output_root: Path, automation_root: Path,
    executor_epoch: int, max_attempts: int, timeout_seconds: int,
    artifact_revision: str,
) -> dict[str, Any]:
    session_id = row["session_id"]
    existing = previous_attempts(output_root, session_id, backend)
    for item in existing:
        if payload_status(item) in TERMINAL:
            return {"backend": backend, "status": payload_status(item), "adopted": True, "result": item["_path"]}
    for attempt_number in range(len(existing) + 1, max_attempts + 1):
        attempt_id = f"attempt_{attempt_number:04d}"
        token = secrets.token_hex(24)
        log_path = automation_root / f"{backend}_{session_id}_{attempt_id}.log"
        command = [
            sys.executable, "src/chaoyang/ops/run_causal_modal_mask_gpu_adapter_v71.py",
            "--backend", backend,
            "--execute-development-canary",
            "--rgb-manifest", row["rgb_manifest"]["path"],
            "--prompt-manifest", row["prompt_manifest"]["path"],
            "--output-root", str((output_root / backend).resolve()),
            "--task-id", f"s1_{backend}_causal_reentry_canary_v71",
            "--attempt-id", attempt_id,
            "--artifact-revision", artifact_revision,
            "--executor-epoch", str(executor_epoch),
            "--fencing-token", token,
            "--gpu-id", "0",
            "--lease-path", str(LEASE),
            "--lease-lock-path", str(LEASE_LOCK),
        ]
        automation_root.mkdir(parents=True, exist_ok=True)
        started = now().isoformat(timespec="seconds")
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    text=True, check=False, timeout=timeout_seconds,
                )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nAUTOMATION_TIMEOUT\n")
        result_path = output_root / backend / "sessions" / session_id / "attempts" / attempt_id / "RESULT.json"
        status = payload_status(load_json(result_path)) if result_path.is_file() else "FAILED_RUNTIME_FINAL"
        attempt_summary = {
            "backend": backend, "attempt_id": attempt_id, "started_at": started,
            "ended_at": now().isoformat(timespec="seconds"), "return_code": return_code,
            "status": status, "log": str(log_path.resolve()),
            "result": str(result_path.resolve()) if result_path.is_file() else None,
        }
        atomic_json(automation_root / f"{backend}_LATEST.json", attempt_summary)
        if status != "BLOCKED_RESOURCE":
            return attempt_summary
        if attempt_number < max_attempts:
            time.sleep(60)
    return {"backend": backend, "status": "BLOCKED_RESOURCE", "attempts_exhausted": max_attempts}


def run_development_comparison(
    *, backend_results: list[dict[str, Any]], output_root: Path, automation_root: Path,
    artifact_revision: str,
) -> dict[str, Any]:
    """Evaluate available receipts without inventing Gold accuracy."""

    comparison_path = output_root / f"S1_DEVELOPMENT_COMPARISON_{artifact_revision}.json"
    if comparison_path.is_file():
        return {
            "task": "s1_development_comparison",
            "status": load_json(comparison_path).get("status", "UNKNOWN"),
            "adopted": True,
            "result": str(comparison_path.resolve()),
        }
    receipts = [
        Path(str(item["result"])) for item in backend_results
        if item.get("result") and Path(str(item["result"])).is_file()
    ]
    if not receipts:
        return {
            "task": "s1_development_comparison",
            "status": "NO_GO_BACKEND_EXECUTION",
            "reason": "NO_BACKEND_RECEIPT_AVAILABLE",
            "result": None,
        }
    command = [sys.executable, "src/chaoyang/ops/evaluate_s1_modal_mask_canaries_v71.py"]
    for receipt in receipts:
        command.extend(["--receipt", str(receipt.resolve())])
    command.extend(["--output", str(comparison_path.resolve())])
    log_path = automation_root / "s1_development_comparison.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False, timeout=1800,
        )
    if comparison_path.is_file():
        status = load_json(comparison_path).get("status", "UNKNOWN")
    else:
        status = "FAILED_RUNTIME_FINAL"
    return {
        "task": "s1_development_comparison",
        "status": status,
        "return_code": completed.returncode,
        "log": str(log_path.resolve()),
        "result": str(comparison_path.resolve()) if comparison_path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-Clean bounded S1 GPU automation")
    parser.add_argument("--session-id", default="play_cards_0901_042")
    parser.add_argument("--input-index", type=Path, default=DEFAULT_INPUT_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--automation-root", type=Path, default=DEFAULT_AUTOMATION)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--per-backend-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--artifact-revision", default="R7_3")
    parser.add_argument("--ignore-sprint-deadline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected_row = selected_input(args.input_index.resolve(), args.session_id)
    deadline = datetime.fromisoformat(load_json(SPRINT)["manual_engineering_deadline"])
    executor_epoch = int(time.time())
    initial = {
        "schema_version": "post-clean-s1-automation-v71",
        "status": "DRY_RUN_READY" if args.dry_run else "WAITING",
        "session_id": args.session_id,
        "tasks": [
            "robot_unified_zbuffer_fullsession", "role_mask_runtime_contract_repair", "cutie_object_mask",
            "sam2_1_object_mask", "s1_development_comparison",
        ],
        "backends": ["cutie", "sam2_1"],
        "sprint_deadline": deadline.isoformat(),
        "input_index": {"path": str(args.input_index.resolve()), "sha256": sha256(args.input_index.resolve())},
        "output_root": str(args.output_root.resolve()),
        "artifact_revision": args.artifact_revision,
        "executor_epoch": executor_epoch,
        "pid": os.getpid(),
        "claim_limit": "Development task-object modal-mask canary only; not a Role Mask successor and no Mask/Contact/Object6D/Gold/physical authority.",
    }
    atomic_json(args.automation_root / "AUTOMATION_STATE.json", initial)
    if args.dry_run:
        print(json.dumps(initial, ensure_ascii=False, indent=2))
        return 0
    wait_started = time.monotonic()
    while True:
        ok, detail = validate_governance()
        status = load_json(STATUS_MIN) if STATUS_MIN.is_file() else {}
        time_ready = args.ignore_sprint_deadline or now() >= deadline
        clean_ready = clean_is_closed(status)
        waiting = dict(initial)
        waiting.update(
            status="WAITING" if not (ok and time_ready and clean_ready) else "READY",
            updated_at=now().isoformat(timespec="seconds"), governance_pass=ok,
            governance_detail=detail, clean_closed=clean_ready, sprint_deadline_reached=time_ready,
            clean_counts=status.get("waves", {}),
        )
        atomic_json(args.automation_root / "AUTOMATION_STATE.json", waiting)
        if ok and time_ready and clean_ready:
            break
        if time.monotonic() - wait_started >= args.max_wait_seconds:
            waiting["status"] = "BLOCKED_RESOURCE"
            waiting["reason"] = "WAIT_BUDGET_EXHAUSTED_BEFORE_GOVERNANCE_AND_CLEAN_GATE"
            atomic_json(args.automation_root / "AUTOMATION_STATE.json", waiting)
            return 3
        time.sleep(max(args.poll_seconds, 5))
    results = [run_robot_zbuffer_fullsession(automation_root=args.automation_root.resolve())]
    role_rows = [run_role_repair(automation_root=args.automation_root.resolve(), executor_epoch=executor_epoch)]
    results.extend(role_rows)
    if role_rows[0].get("status") == "PASSED":
        for session_id, config, config_sha in ROLE_REGRESSIONS:
            regression_row = run_role_regression(
                session_id=session_id, config=config, config_sha=config_sha,
                automation_root=args.automation_root.resolve(), executor_epoch=executor_epoch,
            )
            role_rows.append(regression_row)
            results.append(regression_row)
            if regression_row.get("status") != "PASSED":
                break
    results.append(publish_role_bundle(role_rows))
    backend_results = []
    for backend in ("cutie", "sam2_1"):
        backend_result = run_backend(
            backend=backend, row=selected_row, output_root=args.output_root.resolve(),
            automation_root=args.automation_root.resolve(), executor_epoch=executor_epoch,
            max_attempts=args.max_attempts, timeout_seconds=args.per_backend_timeout_seconds,
            artifact_revision=args.artifact_revision,
        )
        backend_results.append(backend_result)
        results.append(backend_result)
    results.append(run_development_comparison(
        backend_results=backend_results, output_root=args.output_root.resolve(),
        automation_root=args.automation_root.resolve(), artifact_revision=args.artifact_revision,
    ))
    terminal = {
        **initial, "status": "TERMINAL", "ended_at": now().isoformat(timespec="seconds"),
        "results": results,
    }
    atomic_json(args.automation_root / "AUTOMATION_STATE.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
