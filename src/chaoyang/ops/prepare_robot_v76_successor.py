#!/usr/bin/env python3
"""Materialize immutable v7.6 evidence, Task Packet, and registration candidates."""

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

from chaoyang.ops import run_exact78_robot_ready_batches_v76 as runner  # noqa: E402
from chaoyang.ops.robot_target_reach_v75 import artifact_ref, atomic_new_json, load_json, now_iso  # noqa: E402


def write_text_new(path: Path, text: str) -> None:
    runner.predecessor.write_exact_new(path, text.encode("utf-8"))


def builder_real_preflight(attempt: Path, matrix: Path) -> dict[str, Any]:
    root = attempt / "builder_real_preflight"
    root.mkdir(parents=True, exist_ok=False)
    builder_input, closure = runner.freeze_builder_compatible_matrix(matrix, root)
    sessions = [
        "get_potato_chips_0902_104",
        "play_cards_0903_189",
        "play_cards_0903_202",
    ]
    command = [
        sys.executable,
        str(ROOT / "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py"),
        "--matrix",
        str(builder_input),
        "--sessions",
        *sessions,
        "--output",
        str(root / "DRY_RUN_ONLY_ROBOT_READY_INPUT.json"),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"real builder dry-run failed: {(completed.stdout + completed.stderr)[-4000:]}")
    contract = json.loads(completed.stdout)
    if [row.get("session") for row in contract.get("sessions", [])] != sessions:
        raise RuntimeError("real builder dry-run returned unexpected sessions")
    snapshot_ref = contract.get("matrix_snapshot")
    if snapshot_ref != closure["content_addressed_snapshot"]:
        raise RuntimeError("builder did not consume the v76 content-addressed companion")
    contract_path = root / "BUILDER_DRY_RUN_OUTPUT.json"
    atomic_new_json(contract_path, contract)
    result = {
        "schema_version": "robot-v76-builder-real-preflight-v1",
        "created_at": now_iso(),
        "status": "PASSED_REAL_BUILDER_DRY_RUN",
        "sessions": sessions,
        "matrix_snapshot_closure": closure,
        "builder": artifact_ref(ROOT / "src/chaoyang/ops/build_exact78_robot_ready_contract_v52.py"),
        "contract": artifact_ref(contract_path),
        "returncode": completed.returncode,
        "authority": False,
        "claim_limit": "Real input-contract construction preflight only; no Robot phase was started.",
    }
    path = root / "RESULT.json"
    atomic_new_json(path, result)
    return result


def preflight_receipts(attempt: Path, builder: dict[str, Any]) -> None:
    result_path = attempt / "RESULT.json"
    result = load_json(result_path)
    if result.get("terminal_status") != "PASSED":
        raise RuntimeError("v76 adopt RESULT is not PASSED")
    metrics_path = attempt / "METRICS.json"
    atomic_new_json(
        metrics_path,
        {
            "schema_version": "robot-v76-preflight-metrics-v1",
            "created_at": now_iso(),
            "terminal_status": "PASSED",
            "v75_exact_adoptable": result["counts"]["v75_exact_adoptable"],
            "remaining_to_execute": result["counts"]["remaining_to_execute"],
            "reference_failures": result["counts"]["reference_failures"],
            "real_builder_preflight": builder["status"],
            "real_builder_sessions": len(builder["sessions"]),
        },
    )
    decision_path = attempt / "DECISION.md"
    write_text_new(
        decision_path,
        "# Robot v76 有界 successor 决定\n\n"
        "状态：`PASSED`（只表示 adopt 与 builder 预检通过）。\n\n"
        "v75 的 9 条已闭合终态可按完整 SHA 精确采用；余下 16 条只能在治理注册后由新 v76 immutable attempt 执行。"
        "v76 只修正矩阵快照目录合同，不修改 Robot 算法、质量门或 hard/soft watcher。\n",
    )
    next_path = attempt / "NEXT_ACTION.json"
    atomic_new_json(
        next_path,
        {
            "schema_version": "robot-v76-next-action-v1",
            "created_at": now_iso(),
            "task_id": runner.TASK_ID,
            "next_action": "GOVERNANCE_REGISTER_V76_THEN_EXPLICIT_LAUNCH",
            "prerequisites": [
                "governance=PASS/FRESH",
                "v75=FAILED_RUNTIME_FINAL",
                "v76 Task Packet and task-state row are atomically registered",
            ],
            "execution_started": False,
        },
    )
    run_receipt_path = attempt / "RUN_RECEIPT.json"
    atomic_new_json(
        run_receipt_path,
        {
            "schema_version": "robot-v76-preflight-run-receipt-v1",
            "created_at": now_iso(),
            "task_id": "RECOVER-V75-V76",
            "terminal_status": "PASSED",
            "result": artifact_ref(result_path),
            "metrics": artifact_ref(metrics_path),
            "builder_real_preflight": artifact_ref(
                attempt / "builder_real_preflight/RESULT.json"
            ),
            "producer": artifact_ref(Path(__file__)),
            "governance_current_modified": False,
            "execution_started": False,
            "authority": False,
        },
    )
    manifest_path = attempt / "ARTIFACT_MANIFEST.json"
    atomic_new_json(
        manifest_path,
        {
            "schema_version": "robot-v76-preflight-artifact-manifest-v1",
            "created_at": now_iso(),
            "artifacts": [
                artifact_ref(path)
                for path in (
                    result_path,
                    metrics_path,
                    run_receipt_path,
                    decision_path,
                    next_path,
                    attempt / "builder_real_preflight/RESULT.json",
                    attempt / "builder_real_preflight/BUILDER_DRY_RUN_OUTPUT.json",
                )
            ],
            "claim_limit": "Preflight evidence only; excludes this self-referential manifest.",
        },
    )


def execution_materials(
    attempt: Path, preflight: Path, matrix: Path, v75_failure: Path
) -> dict[str, Any]:
    attempt.mkdir(parents=True, exist_ok=False)
    packet_path = attempt / "TASK_PACKET.json"
    packet = {
        "schema_version": "exact78-task-packet-v1",
        "task_id": runner.TASK_ID,
        "objective": "Exact-adopt nine SHA-closed v75 terminals and execute only the remaining sixteen sessions using the builder-compatible immutable matrix snapshot layout.",
        "plan_revision": "chaoyang-v7.1",
        "read_set": [
            str(ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"),
            str(ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"),
            str(ROOT / "docs/governance/PLAN_REVISION.json"),
            str(preflight),
            str(v75_failure),
            str(ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_exact78_robot_expansion_v75/attempts/attempt_0016_execution/run/SELECTION.json"),
            str(matrix),
            str(ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v76.py"),
        ],
        "write_set": [str(attempt)],
        "prerequisites": [
            "governance=PASS/FRESH",
            "task_state_contains_robot_geometry_expansion_v76",
            "current_task_packet_index_contains_this_exact_packet",
            "robot_geometry_expansion_v75=FAILED_RUNTIME_FINAL",
            "v75_to_v76_adopt_preflight=PASS_EXACT_SHA_ADOPTABLE",
            "real_builder_preflight=PASSED_REAL_BUILDER_DRY_RUN",
            "remaining_16_robot_current_state=READY_FOR_ROBOT_CURRENT_DRAFT",
        ],
        "gates": [
            "EXACT_V75_TERMINAL_ADOPT_9",
            "EXECUTE_16_NO_OVERLAP",
            "BUILDER_COMPATIBLE_MATRIX_LIVE_AND_SNAPSHOT_PAIR",
            "CURRENT_MATRIX_BYTE_SHA_EXACT",
            "PHASE_COMMAND_AND_OUTPUT_RECEIPT",
            "TASK_PACKET_PID_STARTTICKS_FENCING",
            "FINITE_PROPER_SE3",
            "JOINT_LIMITS",
            "FULLSESSION_REVIEW_OR_EXPLICIT_TERMINAL",
            "NO_CLOBBER",
        ],
        "budgets": {"cpu_seconds": 72000, "gpu_seconds": 0, "wall_seconds": 86400},
        "attempt_max": 1,
        "stop_condition": "All 25 sessions close as exactly nine adopts plus sixteen executions, or publish FAILED_RUNTIME_FINAL.",
        "output_contract": [
            "RUN_INPUT_MANIFEST.json",
            "RUN_CODE_CLOSURE.json",
            "EXECUTOR_CLAIM.json",
            "run/input_snapshot/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json",
            "run/input_snapshot/snapshots/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX_<sha>.json",
            "run/SELECTION.json",
            "run/adopted/<session>/RESULT.json",
            "run/final/<tier>/<session>/RESULT.json",
            "run/RESULT.json",
            "RUNNER.log",
        ],
        "claim_limit": "Digital visual Robot terminal generation only; no contact/control/physical deployment authority.",
        "executor_epoch": 4,
        "fencing": {"pid_startticks_required": True, "immutable_final": True},
        "artifact_revision_contract": {
            "required_revision_status": "VALID_FOR_PINNED_REVISION",
            "forbid_in_place_overwrite": True,
            "output_artifact_revision": "R7_6",
        },
        "initial_search_result_limit": 20,
        "initial_log_line_limit": 80,
    }
    atomic_new_json(packet_path, packet)
    write_text_new(
        attempt / "CONTEXT_CARD.md",
        "# robot_geometry_expansion_v76\n\n"
        "只修复 v75 的矩阵快照目录合同；算法、25条冻结分母、9 adopt/16 execute、质量门均不变。\n\n"
        "执行前必须由治理 aggregator 原子注册本 Task Packet。不得覆盖 v75 或启动第二 executor。\n",
    )
    command = [
        "python",
        "-m",
        "chaoyang.ops.run_exact78_robot_ready_batches_v76",
        "--adopt-preflight",
        str(preflight),
        "--task-packet",
        str(packet_path),
        "--matrix",
        str(matrix),
        "--output-root",
        str(attempt / "run"),
        "--task-id",
        runner.TASK_ID,
        "--phase-timeout-seconds",
        "7200",
    ]
    run_command = {
        "schema_version": "robot-v76-launch-command-v1",
        "task_id": runner.TASK_ID,
        "status": "READY_AFTER_GOVERNANCE_REGISTRATION",
        "working_directory": str(ROOT),
        "background_log": str(attempt / "RUNNER.log"),
        "output_root": str(attempt / "run"),
        "command": command,
        "runner": artifact_ref(ROOT / "src/chaoyang/ops/run_exact78_robot_ready_batches_v76.py"),
        "task_packet": artifact_ref(packet_path),
        "adopt_preflight": artifact_ref(preflight),
        "matrix_to_snapshot": artifact_ref(matrix),
        "execution_started": False,
        "claim_limit": "Launch material only; governance registration and explicit claim are still required.",
    }
    atomic_new_json(attempt / "RUN_COMMAND.json", run_command)
    pointer_path = ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json"
    pointer = load_json(pointer_path)
    index_path = Path(pointer["index_path"])
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    if artifact_ref(index_path)["sha256"] != pointer["index_sha256"]:
        raise RuntimeError("current Task Packet pointer SHA mismatch")
    atomic_new_json(
        attempt / "TASK_STATE_ROW_CANDIDATE.json",
        {
            "schema_version": "robot-v76-task-state-row-candidate-v1",
            "created_at": now_iso(),
            "action": "REGISTER_SUCCESSOR_AFTER_V75_TERMINAL",
            "predecessor_task_id": "robot_geometry_expansion_v75",
            "task": {
                "task_id": runner.TASK_ID,
                "phase": "registered_wait_launch",
                "attempt": 0,
                "status": "PENDING",
                "updated_at": now_iso(),
                "heartbeat_at": None,
                "session": "v75_frozen_25_adopt9_execute16",
                "pid": None,
                "proc_start_ticks": None,
                "gpu_id": None,
                "task_packet": artifact_ref(packet_path),
                "executor_epoch": 4,
                "result": None,
            },
            "claim_limit": "Aggregator candidate only; no execution or Robot authority.",
        },
    )
    atomic_new_json(
        attempt / "TASK_PACKET_INDEX_CANDIDATE.json",
        {
            "schema_version": "robot-v76-task-packet-index-candidate-v1",
            "created_at": now_iso(),
            "action": "ADD_UNIQUE_TASK_PACKET_ENTRY",
            "task_packet_entry": {
                "task_id": runner.TASK_ID,
                "packet_path": str(packet_path.relative_to(ROOT)),
                "packet_sha256": artifact_ref(packet_path)["sha256"],
            },
            "predecessor_pointer": artifact_ref(pointer_path),
            "predecessor_index": artifact_ref(index_path),
            "required_invariants": [
                "all predecessor entries preserved",
                "v75 remains FAILED_RUNTIME_FINAL",
                "exactly one robot_geometry_expansion_v76 entry",
                "receipt binds successor index and task state revision",
            ],
            "claim_limit": "Aggregator index candidate only; not the current index.",
        },
    )
    return run_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-attempt", type=Path, required=True)
    parser.add_argument("--execution-attempt", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--v75-failure", type=Path, required=True)
    args = parser.parse_args()
    preflight_attempt = args.preflight_attempt.resolve(strict=True)
    matrix = args.matrix.resolve(strict=True)
    v75_failure = args.v75_failure.resolve(strict=True)
    builder_result_path = preflight_attempt / "builder_real_preflight/RESULT.json"
    if builder_result_path.is_file():
        builder = load_json(builder_result_path)
        if builder.get("status") != "PASSED_REAL_BUILDER_DRY_RUN":
            raise RuntimeError("existing real builder preflight is not PASSED")
    else:
        builder = builder_real_preflight(preflight_attempt, matrix)
    receipt_set = (
        "ARTIFACT_MANIFEST.json",
        "METRICS.json",
        "RUN_RECEIPT.json",
        "DECISION.md",
        "NEXT_ACTION.json",
    )
    if not all((preflight_attempt / name).is_file() for name in receipt_set):
        preflight_receipts(preflight_attempt, builder)
    command = execution_materials(
        args.execution_attempt.resolve(), preflight_attempt / "RESULT.json", matrix, v75_failure
    )
    print(json.dumps({"status": "PASSED", "execution_started": False, "command": command}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
