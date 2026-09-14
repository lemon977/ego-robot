#!/usr/bin/env python3
"""Run the second and final bounded KaiHand method for round-1 HOLD rows.

Forward-hand PASS rows are carried without recomputation.  Only
``HOLD_HAND_ROUND1_ELIGIBLE_ROUND2`` rows run the frozen segment-level
bidirectional/lookahead solver.  Arm quality-C rows still retain an independent
hand audit solely to close a watermarked failure review; hand success never
promotes an arm failure.  A numeric hand HOLD after this step exhausts the
two-method hand budget and is a quality-C terminal candidate.

This wrapper is CPU-only and publishes no render, contact, action, deployment,
training, physical-accuracy, or Robot authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
ROUND2_TOOL = PROJECT / "tools/run_robot_hand_segment_bidirectional_v3.py"
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


def unique_sessions(rows: list[dict[str, Any]], label: str) -> set[str]:
    sessions = [row.get("session") for row in rows]
    if any(not isinstance(session, str) or not session for session in sessions):
        raise ValueError(f"{label}: every row requires a non-empty session")
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{label}: duplicate session rows")
    return set(sessions)


def classify_input(status: str) -> str:
    mapping = {
        "PASS_HAND_ROUND1_FORWARD": "CARRY_HAND_PASS",
        "HOLD_HAND_ROUND1_ELIGIBLE_ROUND2": "RUN_HAND_ROUND2",
    }
    if status not in mapping:
        raise ValueError(f"unexpected hand round-1 status {status}")
    return mapping[status]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--hand-round1-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")

    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    expected_tool = preflight.get("programs", {}).get(ROUND2_TOOL.name)
    if expected_tool is None or ref(ROUND2_TOOL) != expected_tool:
        raise SystemExit("hand round-2 tool differs from preflight closure")
    expected_wrapper = preflight.get("programs", {}).get(Path(__file__).name)
    if expected_wrapper is None or ref(Path(__file__)) != expected_wrapper:
        raise SystemExit("hand round-2 wrapper differs from preflight closure")
    expected_implementation = preflight.get("programs", {}).get(HAND_FIT_IMPLEMENTATION.name)
    if expected_implementation is None or ref(HAND_FIT_IMPLEMENTATION) != expected_implementation:
        raise SystemExit("shared hand implementation differs from preflight closure")

    temporal_result_path = args.temporal_root / "RESULT.json"
    temporal = load(temporal_result_path)
    if temporal.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise SystemExit("passed temporal root required")
    temporal_rows = temporal.get("sessions", [])
    temporal_set = unique_sessions(temporal_rows, "temporal")
    temporal_by_session = {row["session"]: row for row in temporal_rows}

    round1 = load(args.hand_round1_result)
    if round1.get("status") != "PASS_BOUNDED_HAND_ROUND1_RESULTS_PUBLISHED":
        raise SystemExit("complete hand round-1 aggregate required")
    if round1.get("preflight") != ref(args.preflight):
        raise SystemExit("hand round-1/preflight closure mismatch")
    if round1.get("temporal_root_result") != ref(temporal_result_path):
        raise SystemExit("hand round-1/temporal closure mismatch")
    round1_rows = round1.get("sessions", [])
    round1_set = unique_sessions(round1_rows, "hand round-1")
    preflight_set = unique_sessions(preflight.get("sessions", []), "preflight")
    if round1_set != preflight_set or temporal_set != preflight_set:
        raise SystemExit("preflight/temporal/hand-round1 session set mismatch")

    args.output_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    counts = {
        "method1_pass_carried": 0,
        "round2_pass": 0,
        "round2_hold_two_methods_exhausted": 0,
        "round2_runtime_failed_retryable": 0,
        "arm_quality_c_rows": 0,
    }
    for row in round1_rows:
        task = row["task"]
        session = row["session"]
        action = classify_input(row["status"])
        if row.get("arm_numeric_pass") is False:
            counts["arm_quality_c_rows"] += 1
        if action == "CARRY_HAND_PASS":
            verify_ref(row["result"], f"{session}:method1_result")
            verify_ref(row["states"], f"{session}:method1_states")
            rows.append(
                {
                    "task": task,
                    "session": session,
                    "status": "PASS_HAND_METHOD1_CARRIED_NO_ROUND2",
                    "arm_result": row["arm_result"],
                    "arm_states": row["arm_states"],
                    "arm_method_rounds_consumed": row["arm_method_rounds_consumed"],
                    "arm_numeric_pass": row["arm_numeric_pass"],
                    "hand_method_rounds_consumed": 1,
                    "method1_result": row["result"],
                    "method1_states": row["states"],
                }
            )
            counts["method1_pass_carried"] += 1
            continue

        forward_result = verify_ref(row["result"], f"{session}:forward_result")
        forward_states = verify_ref(row["states"], f"{session}:forward_states")
        temporal_session_result_path = verify_ref(
            temporal_by_session[session]["result"], f"{session}:temporal_result"
        )
        temporal_session_result = load(temporal_session_result_path)
        if temporal_session_result.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
            raise SystemExit(f"{session}: temporal status is not passed")
        hawor_path = verify_ref(
            temporal_session_result["outputs"]["npz"], f"{session}:temporal_npz"
        )
        session_root = args.output_root / task / session / "hand_round2_bidirectional"
        session_root.mkdir(parents=True)
        result_path = session_root / "RESULT.json"
        log_path = session_root / "execution.log"
        command = [
            sys.executable,
            str(ROUND2_TOOL),
            "--task",
            task,
            "--session",
            session,
            "--hawor",
            str(hawor_path),
            "--forward-result",
            str(forward_result),
            "--forward-states",
            str(forward_states),
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
            "arm_result": row["arm_result"],
            "arm_states": row["arm_states"],
            "arm_method_rounds_consumed": row["arm_method_rounds_consumed"],
            "arm_numeric_pass": row["arm_numeric_pass"],
            "hand_method_rounds_consumed": 1,
            "method1_result": row["result"],
            "method1_states": row["states"],
        }
        if process.returncode in {0, 2} and result_path.is_file():
            result = load(result_path)
            if result.get("task") != task or result.get("session") != session:
                raise SystemExit(f"{session}: hand round-2 identity mismatch")
            states_path = verify_ref(result["output_states"], f"{session}:round2_states")
            if result.get("status") == "PASS_NUMERIC_CANARY_NO_AUTHORITY" and process.returncode == 0:
                status = "PASS_HAND_ROUND2_BIDIRECTIONAL"
                counts["round2_pass"] += 1
            elif result.get("status") == "HOLD_NUMERIC_CANARY" and process.returncode == 2:
                status = "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C"
                counts["round2_hold_two_methods_exhausted"] += 1
            else:
                raise SystemExit(
                    f"{session}: inconsistent hand round-2 result/returncode "
                    f"{result.get('status')}/{process.returncode}"
                )
            item.update(
                status=status,
                hand_method_rounds_consumed=2,
                result=ref(result_path),
                states=ref(states_path),
                metrics=result.get("metrics"),
                gates=result.get("gates"),
            )
        else:
            item.update(
                status="FAILED_RUNTIME_RETRYABLE",
                error="hand round-2 command failed without a valid bounded result",
            )
            counts["round2_runtime_failed_retryable"] += 1
        rows.append(item)

    aggregate_status = (
        "PASS_BOUNDED_TWO_METHOD_HAND_RESULTS_PUBLISHED"
        if counts["round2_runtime_failed_retryable"] == 0
        else "FAILED_RUNTIME_RETRYABLE"
    )
    aggregate = {
        "schema_version": "exact78-robot-hand-bidirectional-round2-v52-v1",
        "created_at": now(),
        "status": aggregate_status,
        "preflight": ref(args.preflight),
        "temporal_root_result": ref(temporal_result_path),
        "hand_round1_result": ref(args.hand_round1_result),
        "round2_tool": ref(ROUND2_TOOL),
        "shared_hand_implementation": ref(HAND_FIT_IMPLEMENTATION),
        "counts": counts,
        "sessions": rows,
        "gpu_calls": 0,
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "Final bounded KaiHand numeric method only. Numeric HOLD rows exhausted the two-method hand "
            "budget and are quality-C candidates. Independent hand results for arm quality-C rows exist "
            "only to close watermarked failure reviews and cannot promote the arm. No render, contact, "
            "action, training, deployment, physical-accuracy, or Robot authority is implied."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(
        json.dumps(
            {"status": aggregate_status, "counts": counts, "result": ref(args.output_root / "RESULT.json")},
            ensure_ascii=False,
        )
    )
    return 0 if counts["round2_runtime_failed_retryable"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
