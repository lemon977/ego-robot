#!/usr/bin/env python3
"""Run bounded, read-only Visual Aux preflights at Robot audit milestones.

This watcher never builds RGB, acquires a GPU, starts epoch-0/training, or
updates current governance.  It only observes the Robot hard/soft watcher and
materializes immutable Chips/Poker prerequisite snapshots at 12, 18 and 25
processed sessions.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = ROOT / "src/chaoyang/ops/preflight_visual_aux_pairs_r3.py"
DEFAULT_MILESTONES = (12, 18, 25)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output already exists: {path}")
    atomic_json(path, value)


def write_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot_json(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one already-read JSON input to an immutable local snapshot."""

    payload = load(source.resolve(strict=True))
    write_json(destination, payload)
    return {"source": ref(source), "snapshot": ref(destination)}


def completed_attempt(milestone_root: Path) -> Path | None:
    if not milestone_root.is_dir():
        return None
    for attempt in sorted(milestone_root.glob("attempt_*")):
        result = attempt / "RESULT.json"
        if result.is_file():
            payload = load(result)
            if payload.get("execution_status") == "PASSED_PREFLIGHT_EXECUTION":
                return attempt
    return None


def next_attempt(milestone_root: Path) -> Path:
    numbers = []
    if milestone_root.is_dir():
        for path in milestone_root.glob("attempt_*"):
            try:
                numbers.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return milestone_root / f"attempt_{max(numbers, default=0) + 1:04d}"


def run_pair(
    *,
    task: str,
    attempt: Path,
    snapshots: dict[str, Path],
    robotized_ledger: Path,
) -> Path:
    output = attempt / task
    command = [
        sys.executable,
        str(PREFLIGHT),
        "--task",
        task,
        "--output",
        str(output),
        "--status-receipt",
        str(snapshots["status_receipt"]),
        "--task-packet",
        str(snapshots[f"{task}_task_packet"]),
        "--algorithm-contract",
        str(snapshots["algorithm_contract"]),
        "--eligibility-index",
        str(snapshots["eligibility_index"]),
        "--prior-result",
        str(snapshots["prior_result"]),
        "--robotized-ledger",
        str(robotized_ledger.resolve()),
        "--capacity-preflight",
        str(snapshots["capacity_preflight"]),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    write_text(attempt / f"{task}_PREFLIGHT_STDOUT.log", completed.stdout)
    write_text(attempt / f"{task}_PREFLIGHT_STDERR.log", completed.stderr)
    result = output / "RESULT.json"
    if completed.returncode != 0 or not result.is_file():
        raise RuntimeError(
            f"{task} preflight failed rc={completed.returncode}: "
            f"{completed.stderr[-500:]}"
        )
    payload = load(result)
    if payload.get("execution_status") != "PASSED_PREFLIGHT_EXECUTION":
        raise RuntimeError(f"{task} preflight did not execute successfully")
    if payload.get("checkpoint_count") != 0 or payload.get("loss_curve_count") != 0:
        raise RuntimeError(f"{task} preflight attempted to claim training outputs")
    return result


def materialize_milestone(args: argparse.Namespace, milestone: int, upstream: dict[str, Any]) -> Path:
    milestone_root = args.output_root / "milestones" / f"processed_{milestone:04d}"
    adopted = completed_attempt(milestone_root)
    if adopted is not None:
        return adopted
    attempt = next_attempt(milestone_root)
    attempt.mkdir(parents=True, exist_ok=False)
    inputs = attempt / "input_snapshot"
    inputs.mkdir()

    upstream_snapshot = inputs / "UPSTREAM_AUTOMATION_STATE.json"
    write_json(upstream_snapshot, upstream)
    if int(upstream.get("processed_sessions", -1)) < milestone:
        raise RuntimeError("milestone materialized before processed-session threshold")

    source_paths = {
        "status_receipt": args.status_receipt,
        "algorithm_contract": args.algorithm_contract,
        "eligibility_index": args.eligibility_index,
        "prior_result": args.prior_result,
        "capacity_preflight": args.capacity_preflight,
        "chips_task_packet": args.chips_task_packet,
        "poker_task_packet": args.poker_task_packet,
    }
    bindings: dict[str, Any] = {
        "upstream_state": {
            "source_path": str(args.robot_state.resolve()),
            "snapshot": ref(upstream_snapshot),
        }
    }
    snapshots: dict[str, Path] = {}
    for key, source in source_paths.items():
        destination = inputs / f"{key.upper()}.json"
        bindings[key] = snapshot_json(source, destination)
        snapshots[key] = destination

    pair_results = {
        task: run_pair(
            task=task,
            attempt=attempt,
            snapshots=snapshots,
            robotized_ledger=args.robotized_ledger,
        )
        for task in ("chips", "poker")
    }
    pairs = {task: load(path) for task, path in pair_results.items()}
    pair_status = {task: row.get("terminal_status") for task, row in pairs.items()}
    ledger_missing = not args.robotized_ledger.is_file()
    if ledger_missing and any(status != "BLOCKED_PREREQ" for status in pair_status.values()):
        raise RuntimeError("missing formal causal ledger must close both pairs as BLOCKED_PREREQ")

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    metrics = {
        "schema_version": "visual-aux-milestone-preflight-r3-metrics-v1",
        "milestone": milestone,
        "processed_sessions_observed": int(upstream["processed_sessions"]),
        "expected_sessions": int(upstream.get("expected_sessions", 0)),
        "pair_terminal_status": pair_status,
        "pair_observed": {
            task: load(attempt / task / "METRICS.json").get("observed")
            for task in ("chips", "poker")
        },
        "formal_causal_robotized_ledger_present": not ledger_missing,
        "started_training": False,
        "checkpoint_count": 0,
        "loss_curve_count": 0,
        "claim_limit": "Milestone prerequisite snapshot only; no checkpoint or training authority.",
    }
    write_json(attempt / "METRICS.json", metrics)
    write_text(
        attempt / "DECISION.md",
        "# Visual Aux 里程碑前置检查\n\n"
        f"Robot hard/soft watcher 已处理 `{upstream['processed_sessions']}` 条，"
        f"触发 `{milestone}` 条里程碑。\n\n"
        f"Chips：`{pair_status['chips']}`；Poker：`{pair_status['poker']}`。\n\n"
        "该检查只读取已有证据，不启动 epoch-0 或训练，也不更新 current governance。"
        "缺少正式 causal Robotized ledger、Silver/compositor 绑定或冻结 hardset 时，"
        "对应 pair 必须保持 `BLOCKED_PREREQ`。\n",
    )
    write_json(
        attempt / "NEXT_ACTION.json",
        {
            "schema_version": "visual-aux-milestone-preflight-r3-next-action-v1",
            "status": "WAIT_NEXT_MILESTONE" if milestone < max(args.milestones) else "MILESTONE_SERIES_COMPLETE",
            "next_milestone": next((item for item in args.milestones if item > milestone), None),
            "do_not_start_training": True,
            "claim_limit": "Routing instruction only.",
        },
    )
    write_json(
        attempt / "RUN_RECEIPT.json",
        {
            "schema_version": "visual-aux-milestone-preflight-r3-run-receipt-v1",
            "created_at": created_at,
            "execution_mode": "CPU_READ_ONLY_PREFLIGHT_WATCHER",
            "execution_status": "PASSED_PREFLIGHT_EXECUTION",
            "milestone": milestone,
            "inputs": bindings,
            "pair_results": {task: ref(path) for task, path in pair_results.items()},
            "mutated_current_governance": False,
            "started_training": False,
            "claim_limit": "Immutable milestone observation only.",
        },
    )
    write_json(
        attempt / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "visual-aux-milestone-preflight-r3-artifact-manifest-v1",
            "created_at": created_at,
            "artifacts": {
                name: ref(attempt / name)
                for name in ("METRICS.json", "DECISION.md", "NEXT_ACTION.json", "RUN_RECEIPT.json")
            },
            "pair_artifact_manifests": {
                task: ref(attempt / task / "ARTIFACT_MANIFEST.json")
                for task in ("chips", "poker")
            },
            "claim_limit": "Immutable preflight evidence only.",
        },
    )
    write_json(
        attempt / "RESULT.json",
        {
            "schema_version": "visual-aux-milestone-preflight-r3-result-v1",
            "created_at": created_at,
            "task_id": "visual_aux_milestone_preflight_watcher_r3",
            "execution_status": "PASSED_PREFLIGHT_EXECUTION",
            "terminal_status": "BLOCKED_PREREQ" if any(status == "BLOCKED_PREREQ" for status in pair_status.values()) else "READY_FOR_EPOCH0_REVIEW",
            "milestone": milestone,
            "processed_sessions_observed": int(upstream["processed_sessions"]),
            "pair_terminal_status": pair_status,
            "metrics": ref(attempt / "METRICS.json"),
            "decision": ref(attempt / "DECISION.md"),
            "next_action": ref(attempt / "NEXT_ACTION.json"),
            "run_receipt": ref(attempt / "RUN_RECEIPT.json"),
            "artifact_manifest": ref(attempt / "ARTIFACT_MANIFEST.json"),
            "checkpoint_authority": False,
            "started_training": False,
            "mutated_current_governance": False,
            "claim_limit": "Milestone preflight only; no checkpoint, policy, action, Contact or Robot authority.",
        },
    )
    return attempt


def parse_milestones(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("milestones must be positive comma-separated integers")
    return result


def state_payload(args: argparse.Namespace, *, status: str, upstream: dict[str, Any], completed: list[int], reason: str | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": "visual-aux-milestone-preflight-watcher-r3-state-v1",
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "robot_state": str(args.robot_state.resolve()),
        "processed_sessions": int(upstream.get("processed_sessions", 0)),
        "upstream_status": upstream.get("status", "MISSING"),
        "milestones": list(args.milestones),
        "completed_milestones": completed,
        "started_training": False,
        "mutated_current_governance": False,
        "claim_limit": "Read-only Visual Aux prerequisite milestones; no training or authority.",
    }
    if reason:
        payload["reason"] = reason
    return payload


def materialize_setup(args: argparse.Namespace) -> Path:
    setup = args.output_root / "setup" / "attempt_0001"
    result_path = setup / "RESULT.json"
    if result_path.is_file():
        return setup
    if setup.exists():
        raise RuntimeError(f"incomplete immutable setup attempt requires a new output root: {setup}")
    setup.mkdir(parents=True)
    packet = load(args.task_packet.resolve(strict=True))
    launch = load(args.launch_command.resolve(strict=True))
    if packet.get("task_id") != "visual_aux_milestone_preflight_watcher_r3":
        raise RuntimeError("wrong milestone watcher task packet")
    if launch.get("task_id") != packet["task_id"]:
        raise RuntimeError("launch command and task packet task_id mismatch")
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    inputs = {
        "task_packet": ref(args.task_packet),
        "launch_command": ref(args.launch_command),
        "watcher_code": ref(Path(__file__)),
        "pair_preflight_code": ref(PREFLIGHT),
    }
    write_json(
        setup / "METRICS.json",
        {
            "schema_version": "visual-aux-milestone-watcher-r3-setup-metrics-v1",
            "milestones": list(args.milestones),
            "gpu_tasks_started": 0,
            "training_tasks_started": 0,
            "current_governance_writes": 0,
            "claim_limit": "Configuration verification only.",
        },
    )
    write_text(
        setup / "DECISION.md",
        "# Visual Aux 里程碑 watcher 配置结论\n\n"
        "采用独立只读 watcher，在 Robot hard/soft 已处理 12、18、25 条时调用现有 "
        "`preflight_visual_aux_pairs_r3.py`。该 watcher 不调用训练脚本、不申请 GPU，"
        "也不更新 current governance。\n",
    )
    write_json(
        setup / "NEXT_ACTION.json",
        {
            "schema_version": "visual-aux-milestone-watcher-r3-setup-next-action-v1",
            "status": "START_WATCHER",
            "milestones": list(args.milestones),
            "do_not_start_training": True,
            "claim_limit": "Local automation routing only.",
        },
    )
    write_json(
        setup / "RUN_RECEIPT.json",
        {
            "schema_version": "visual-aux-milestone-watcher-r3-setup-run-receipt-v1",
            "created_at": created_at,
            "status": "PASSED_SETUP",
            "inputs": inputs,
            "mutated_current_governance": False,
            "started_training": False,
            "claim_limit": "Setup receipt only.",
        },
    )
    write_json(
        setup / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "visual-aux-milestone-watcher-r3-setup-artifact-manifest-v1",
            "created_at": created_at,
            "artifacts": {
                name: ref(setup / name)
                for name in ("METRICS.json", "DECISION.md", "NEXT_ACTION.json", "RUN_RECEIPT.json")
            },
            "claim_limit": "Immutable setup evidence only.",
        },
    )
    write_json(
        result_path,
        {
            "schema_version": "visual-aux-milestone-watcher-r3-setup-result-v1",
            "created_at": created_at,
            "task_id": packet["task_id"],
            "execution_status": "PASSED_SETUP",
            "terminal_status": "WAITING_MILESTONES",
            "inputs": inputs,
            "metrics": ref(setup / "METRICS.json"),
            "decision": ref(setup / "DECISION.md"),
            "next_action": ref(setup / "NEXT_ACTION.json"),
            "run_receipt": ref(setup / "RUN_RECEIPT.json"),
            "artifact_manifest": ref(setup / "ARTIFACT_MANIFEST.json"),
            "checkpoint_authority": False,
            "started_training": False,
            "mutated_current_governance": False,
            "claim_limit": "Watcher setup only; milestone observations remain separate immutable attempts.",
        },
    )
    return setup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-receipt", type=Path, required=True)
    parser.add_argument("--algorithm-contract", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--capacity-preflight", type=Path, required=True)
    parser.add_argument("--chips-task-packet", type=Path, required=True)
    parser.add_argument("--poker-task-packet", type=Path, required=True)
    parser.add_argument("--robotized-ledger", type=Path, required=True)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--launch-command", type=Path, required=True)
    parser.add_argument("--milestones", type=parse_milestones, default=DEFAULT_MILESTONES)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    materialize_setup(args)

    lock_path = args.output_root / ".watcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another Visual Aux milestone watcher owns this output root")

        started = time.monotonic()
        while True:
            upstream = load(args.robot_state) if args.robot_state.is_file() else {}
            processed = int(upstream.get("processed_sessions", 0))
            completed = []
            for milestone in args.milestones:
                if processed >= milestone:
                    materialize_milestone(args, milestone, upstream)
                if completed_attempt(args.output_root / "milestones" / f"processed_{milestone:04d}") is not None:
                    completed.append(milestone)

            if len(completed) == len(args.milestones):
                atomic_json(args.output_root / "AUTOMATION_STATE.json", state_payload(args, status="TERMINAL", upstream=upstream, completed=completed))
                return 0
            if args.once:
                atomic_json(args.output_root / "AUTOMATION_STATE.json", state_payload(args, status="POLL_COMPLETE", upstream=upstream, completed=completed))
                return 0
            if upstream.get("status") in {"TERMINAL", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ", "BLOCKED_RESOURCE"}:
                atomic_json(args.output_root / "AUTOMATION_STATE.json", state_payload(args, status="BLOCKED_PREREQ", upstream=upstream, completed=completed, reason="UPSTREAM_TERMINAL_BEFORE_ALL_MILESTONES"))
                return 3
            if time.monotonic() - started > args.max_wait_seconds:
                atomic_json(args.output_root / "AUTOMATION_STATE.json", state_payload(args, status="BLOCKED_RESOURCE", upstream=upstream, completed=completed, reason="WAIT_BUDGET_EXHAUSTED"))
                return 3
            atomic_json(args.output_root / "AUTOMATION_STATE.json", state_payload(args, status="WAITING_UPSTREAM", upstream=upstream, completed=completed))
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
