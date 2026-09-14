#!/usr/bin/env python3
"""Run bounded KaiHand method 1 for sessions with a passed arm path.

The wrapper consumes the final arm-method aggregate and preserves the temporal
HaWoR and accepted-template closure.  Hand retarget is evaluated independently
even for arm quality-C rows so a complete watermarked failure review can be
sealed later; a hand PASS can never promote an arm HOLD.  It is CPU-only and
publishes no visual, contact, action, deployment, training, or Robot authority.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
HAND_TOOL = PROJECT / "tools/run_robot_hand_fullsession_v2.py"
HAND_FIT_IMPLEMENTATION = PROJECT / "tools/run_newtask_robot_shared_v4_hand.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_ref(value: dict[str, Any], label: str) -> Path:
    if set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: exact ref required")
    path = Path(value["path"])
    if not path.is_absolute():
        path = PROJECT / path
    actual = ref(path)
    if actual["bytes"] != value["bytes"] or actual["sha256"] != value["sha256"]:
        raise ValueError(f"{label}: ref mismatch")
    return Path(actual["path"])


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def arm_lineage(
    row: dict[str, Any], session: str
) -> tuple[dict[str, Any], dict[str, Any], int, bool]:
    status = row["status"]
    if status == "PASS_ARM_METHOD1_CARRIED_NO_ROUND2":
        return row["method1_result"], row["method1_states"], 1, True
    if status == "PASS_ARM_ROUND2_BIDIRECTIONAL":
        return row["result"], row["states"], 2, True
    if status == "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C":
        return row["result"], row["states"], 2, False
    raise ValueError(f"{session}: unexpected final arm status {status}")


def unique_sessions(rows: list[dict[str, Any]], label: str) -> set[str]:
    sessions = [row.get("session") for row in rows]
    if any(not isinstance(session, str) or not session for session in sessions):
        raise ValueError(f"{label}: every row requires a non-empty session")
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{label}: duplicate session rows")
    return set(sessions)


def run_session(
    *,
    row: dict[str, Any],
    preflight: dict[str, Any],
    temporal_by_session: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    task = row["task"]
    session = row["session"]
    arm_result_ref, arm_states_ref, arm_rounds, arm_numeric_pass = arm_lineage(row, session)
    verify_ref(arm_result_ref, f"{session}:arm_result")
    verify_ref(arm_states_ref, f"{session}:arm_states")
    temporal_result_path = verify_ref(
        temporal_by_session[session]["result"], f"{session}:temporal_result"
    )
    temporal_result = load(temporal_result_path)
    if temporal_result.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise ValueError(f"{session}: temporal status is not passed")
    hawor_path = verify_ref(temporal_result["outputs"]["npz"], f"{session}:temporal_npz")
    accepted_states = verify_ref(
        preflight["accepted_templates"][task]["states"], f"{task}:accepted_states"
    )
    session_root = output_root / task / session / "hand_round1_forward"
    session_root.mkdir(parents=True)
    result_path = session_root / "RESULT.json"
    log_path = session_root / "execution.log"
    command = [
        sys.executable,
        str(HAND_TOOL),
        "--task",
        task,
        "--session",
        session,
        "--hawor",
        str(hawor_path),
        "--hawor-result",
        str(temporal_result_path),
        "--accepted-states",
        str(accepted_states),
        "--output",
        str(result_path),
    ]
    started = now()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.flush()
        os.fsync(log.fileno())
    item: dict[str, Any] = {
        "task": task,
        "session": session,
        "started_at": started,
        "finished_at": now(),
        "returncode": process.returncode,
        "command": command,
        "execution_log": ref(log_path),
        "arm_result": arm_result_ref,
        "arm_states": arm_states_ref,
        "arm_method_rounds_consumed": arm_rounds,
        "arm_numeric_pass": arm_numeric_pass,
        "hand_method_rounds_consumed": 0,
    }
    if process.returncode in {0, 2} and result_path.is_file():
        result = load(result_path)
        if result.get("task") != task or result.get("session") != session:
            raise ValueError(f"{session}: hand result identity mismatch")
        states_path = verify_ref(result["output_states"], f"{session}:hand_states")
        if result.get("status") == "PASS_NUMERIC_CANARY_NO_AUTHORITY" and process.returncode == 0:
            status = "PASS_HAND_ROUND1_FORWARD"
        elif result.get("status") == "HOLD_NUMERIC_CANARY" and process.returncode == 2:
            status = "HOLD_HAND_ROUND1_ELIGIBLE_ROUND2"
        else:
            raise ValueError(f"{session}: hand result/returncode mismatch")
        item.update(
            status=status,
            hand_method_rounds_consumed=1,
            result=ref(result_path),
            states=ref(states_path),
            metrics=result.get("metrics"),
            gates=result.get("gates"),
        )
    else:
        item.update(
            status="FAILED_RUNTIME_RETRYABLE",
            error="hand method-1 command failed without a valid bounded result",
        )
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--arm-final-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")
    if not 1 <= args.max_workers <= 3:
        raise SystemExit("--max-workers must be in [1,3]")
    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    expected_tool = preflight.get("programs", {}).get(HAND_TOOL.name)
    if expected_tool is None or ref(HAND_TOOL) != expected_tool:
        raise SystemExit("hand tool differs from preflight closure")
    expected_wrapper = preflight.get("programs", {}).get(Path(__file__).name)
    if expected_wrapper is None or ref(Path(__file__)) != expected_wrapper:
        raise SystemExit("hand round-1 wrapper differs from preflight closure")
    expected_implementation = preflight.get("programs", {}).get(HAND_FIT_IMPLEMENTATION.name)
    if expected_implementation is None or ref(HAND_FIT_IMPLEMENTATION) != expected_implementation:
        raise SystemExit("shared hand implementation differs from preflight closure")
    temporal_result_path = args.temporal_root / "RESULT.json"
    temporal = load(temporal_result_path)
    if temporal.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise SystemExit("passed temporal root required")
    temporal_rows = temporal.get("sessions", [])
    unique_sessions(temporal_rows, "temporal")
    temporal_by_session = {row["session"]: row for row in temporal_rows}
    arm = load(args.arm_final_result)
    if arm.get("status") != "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED":
        raise SystemExit("complete final arm-method aggregate required")
    arm_rows = arm.get("sessions", [])
    session_set = unique_sessions(preflight.get("sessions", []), "preflight")
    arm_session_set = unique_sessions(arm_rows, "arm")
    if set(temporal_by_session) != session_set or arm_session_set != session_set:
        raise SystemExit("preflight/temporal/arm session set mismatch")

    args.output_root.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                run_session,
                row=row,
                preflight=preflight,
                temporal_by_session=temporal_by_session,
                output_root=args.output_root,
            ): row["session"]
            for row in arm_rows
        }
        by_session = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
    rows = [by_session[row["session"]] for row in arm_rows]
    counts = {
        "pass": sum(row["status"] == "PASS_HAND_ROUND1_FORWARD" for row in rows),
        "hold": sum(row["status"] == "HOLD_HAND_ROUND1_ELIGIBLE_ROUND2" for row in rows),
        "arm_quality_c_processed_for_review": sum(row.get("arm_numeric_pass") is False for row in rows),
        "runtime_failed": sum(row["status"] == "FAILED_RUNTIME_RETRYABLE" for row in rows),
    }
    status = (
        "PASS_BOUNDED_HAND_ROUND1_RESULTS_PUBLISHED"
        if counts["runtime_failed"] == 0
        else "FAILED_RUNTIME_RETRYABLE"
    )
    aggregate = {
        "schema_version": "exact78-robot-hand-round1-v52-v1",
        "created_at": now(),
        "status": status,
        "preflight": ref(args.preflight),
        "temporal_root_result": ref(temporal_result_path),
        "arm_final_result": ref(args.arm_final_result),
        "hand_tool": ref(HAND_TOOL),
        "shared_hand_implementation": ref(HAND_FIT_IMPLEMENTATION),
        "parallel_workers": args.max_workers,
        "counts": counts,
        "sessions": rows,
        "gpu_calls": 0,
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "KaiHand numeric method round 1 for every row with a complete arm result. Arm quality-C rows are "
            "processed only to close a watermarked failure review and can never be promoted by a hand PASS. "
            "Hand HOLD rows may consume the second/final bidirectional method. No render, contact, action, "
            "training, deployment, or Robot authority."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(json.dumps({"status": status, "counts": counts, "result": ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0 if counts["runtime_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
