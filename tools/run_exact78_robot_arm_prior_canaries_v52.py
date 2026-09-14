#!/usr/bin/env python3
"""Run the first bounded Robot arm method at the shared 0.26 m prior.

The input preflight and temporal-successor closure are mandatory.  This wrapper
uses CPU only, writes per-session attempt evidence, and publishes its aggregate
RESULT.json last.  It does not run the second method, hand retarget, rendering,
contact inference, action publication or authority promotion.
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
DEFAULT_ARM_TOOL = PROJECT / "tools/run_robot_motion_transfer_arm_canary_v2.py"


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


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def select_arm_tool(preflight: dict[str, Any]) -> Path:
    override = preflight.get("arm_forward_tool")
    if override is not None:
        path = verify_ref(override, "arm_forward_tool")
        if preflight.get("programs", {}).get(path.name) != ref(path):
            raise ValueError("arm forward override is not in preflight program closure")
        return path
    expected = preflight.get("programs", {}).get(DEFAULT_ARM_TOOL.name)
    if expected is None or ref(DEFAULT_ARM_TOOL) != expected:
        raise ValueError("default arm tool differs from preflight closure")
    return DEFAULT_ARM_TOOL


def deterministic_bounds_failure(returncode: int, result_exists: bool, log_text: str) -> bool:
    """Recognize the solver's exact, deterministic continuity infeasibility."""
    return (
        returncode != 0
        and not result_exists
        and "TemporalReviewError: empty previous-accepted bounds:" in log_text
    )


def command_without_output(command: list[str]) -> tuple[str, ...]:
    """Return the immutable computation contract while ignoring only its fresh output path."""
    if command.count("--output") != 1:
        raise ValueError("exactly one --output argument required")
    index = command.index("--output")
    if index + 1 >= len(command):
        raise ValueError("--output value missing")
    return tuple(command[: index + 1] + ["<FRESH_OUTPUT>"] + command[index + 2 :])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adopt-deterministic-attempt", type=Path)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit(f"fresh --output-root required: {args.output_root}")
    preflight = load(args.preflight)
    if preflight.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed Robot input preflight required")
    try:
        arm_tool = select_arm_tool(preflight)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    temporal_result_path = args.temporal_root / "RESULT.json"
    temporal_root_result = load(temporal_result_path)
    if temporal_root_result.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise SystemExit("passed temporal root required")
    temporal_by_session = {row["session"]: row for row in temporal_root_result.get("sessions", [])}
    sessions = preflight.get("sessions", [])
    if set(temporal_by_session) != {row["session"] for row in sessions}:
        raise SystemExit("temporal/preflight session set mismatch")

    adopted_attempt_ref: dict[str, Any] | None = None
    adopted_by_session: dict[str, dict[str, Any]] = {}
    if args.adopt_deterministic_attempt is not None:
        adopted = load(args.adopt_deterministic_attempt)
        adopted_attempt_ref = ref(args.adopt_deterministic_attempt)
        if adopted.get("status") != "FAILED_RUNTIME_RETRYABLE":
            raise SystemExit("only a failed retryable aggregate can be reclassified")
        if adopted.get("preflight") != ref(args.preflight):
            raise SystemExit("adopted attempt/preflight closure mismatch")
        if adopted.get("temporal_root_result") != ref(temporal_result_path):
            raise SystemExit("adopted attempt/temporal closure mismatch")
        if adopted.get("arm_tool") != ref(arm_tool):
            raise SystemExit("adopted attempt/arm tool closure mismatch")
        adopted_by_session = {row["session"]: row for row in adopted.get("sessions", [])}
        if set(adopted_by_session) != {row["session"] for row in sessions}:
            raise SystemExit("adopted attempt/session set mismatch")

    args.output_root.mkdir(parents=True)
    rows = []
    counts = {"pass": 0, "hold": 0, "runtime_failed": 0}
    for row in sessions:
        task = row["task"]
        session = row["session"]
        temporal_ref = temporal_by_session[session]["result"]
        temporal_session_result_path = verify_ref(temporal_ref, f"{session}:temporal_result")
        temporal_session_result = load(temporal_session_result_path)
        if temporal_session_result.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
            raise SystemExit(f"{session}: temporal result is not passed")
        if temporal_session_result.get("inputs", {}).get("bounded_npz") != row["bounded_npz"]:
            raise SystemExit(f"{session}: temporal bounded lineage mismatch")
        hawor_ref = temporal_session_result.get("outputs", {}).get("npz")
        hawor_path = verify_ref(hawor_ref, f"{session}:temporal_npz")
        accepted = preflight["accepted_templates"][task]
        accepted_states = verify_ref(accepted["states"], f"{task}:accepted_states")
        accepted_hawor = verify_ref(accepted["hawor"], f"{task}:accepted_hawor")
        session_root = args.output_root / task / session / "arm_round1_prior026"
        session_root.mkdir(parents=True)
        result_path = session_root / "RESULT.json"
        log_path = session_root / "execution.log"
        command = [
            sys.executable,
            str(arm_tool),
            "--task", task,
            "--session", session,
            "--hawor", str(hawor_path),
            "--hawor-result", str(temporal_session_result_path),
            "--accepted-states", str(accepted_states),
            "--accepted-hawor", str(accepted_hawor),
            "--output", str(result_path),
            "--motion-gain", "1.0",
            "--task-base-backoff-m", "0.26",
        ]
        adopted_row = adopted_by_session.get(session)
        if adopted_row is not None:
            if adopted_row.get("status") != "FAILED_RUNTIME_RETRYABLE":
                raise SystemExit(f"{session}: adopted row is not failed retryable")
            old_command = adopted_row.get("command")
            if not isinstance(old_command, list) or command_without_output(old_command) != command_without_output(command):
                raise SystemExit(f"{session}: adopted command contract mismatch")
            old_output = Path(old_command[old_command.index("--output") + 1])
            if not old_output.is_absolute():
                old_output = PROJECT / old_output
            execution_log_ref = adopted_row.get("execution_log")
            old_log_path = verify_ref(execution_log_ref, f"{session}:adopted_execution_log")
            returncode = int(adopted_row.get("returncode", 0))
            started = str(adopted_row.get("started_at"))
            finished = str(adopted_row.get("finished_at"))
            result_exists = old_output.is_file()
            log_text = old_log_path.read_text(encoding="utf-8", errors="replace")
        else:
            started = now()
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = ""
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=PROJECT,
                    env=environment,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                log.flush()
                os.fsync(log.fileno())
            returncode = process.returncode
            finished = now()
            result_exists = result_path.is_file()
            execution_log_ref = ref(log_path)
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        item: dict[str, Any] = {
            "task": task,
            "session": session,
            "started_at": started,
            "finished_at": finished,
            "returncode": returncode,
            "command": command,
            "execution_log": execution_log_ref,
            "arm_method_rounds_consumed": 0,
        }
        if returncode in {0, 2} and result_exists:
            result = load(result_path)
            if result.get("task") != task or result.get("session") != session or result.get("motion_gain") != 1.0:
                raise SystemExit(f"{session}: arm result identity/contract mismatch")
            states_path = verify_ref(result["output_states"], f"{session}:arm_states")
            status = str(result.get("status"))
            if status == "PASS_NUMERIC_CANARY_NO_AUTHORITY" and returncode == 0:
                classification = "PASS_ARM_ROUND1_PRIOR026"
                counts["pass"] += 1
            elif status == "HOLD_NUMERIC_CANARY" and returncode == 2:
                classification = "HOLD_ARM_ROUND1_ELIGIBLE_ROUND2"
                counts["hold"] += 1
            else:
                raise SystemExit(f"{session}: inconsistent result/returncode {status}/{returncode}")
            item.update(
                status=classification,
                arm_method_rounds_consumed=1,
                result=ref(result_path),
                states=ref(states_path),
                metrics=result.get("metrics"),
                gates=result.get("gates"),
            )
        elif deterministic_bounds_failure(
            returncode,
            result_exists,
            log_text,
        ):
            item.update(
                status="HOLD_ARM_PRIOR_DETERMINISTIC_INFEASIBLE_ELIGIBLE_PLACEMENT_SWEEP",
                arm_method_rounds_consumed=1,
                selectable_for_method2_seed=False,
                adopted_from_attempt=adopted_attempt_ref,
                reason=(
                    "the frozen 0.26 m prior has an empty intersection of hard joint, velocity, and "
                    "acceleration bounds; other placements in the same method-1 family remain eligible"
                ),
            )
            counts["hold"] += 1
        else:
            item.update(status="FAILED_RUNTIME_RETRYABLE", error="arm command failed without a valid bounded result")
            counts["runtime_failed"] += 1
        rows.append(item)

    aggregate = {
        "schema_version": "exact78-robot-arm-prior-canaries-v52-v1",
        "created_at": now(),
        "status": "PASS_BOUNDED_ROUND1_RESULTS_PUBLISHED" if counts["runtime_failed"] == 0 else "FAILED_RUNTIME_RETRYABLE",
        "preflight": ref(args.preflight),
        "temporal_root_result": ref(temporal_result_path),
        "arm_tool": ref(arm_tool),
        "placement_backoff_m": 0.26,
        "motion_gain": 1.0,
        "adopted_deterministic_attempt": adopted_attempt_ref,
        "counts": counts,
        "sessions": rows,
        "gpu_calls": 0,
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "Arm numeric method round 1 at the shared placement prior only. HOLD rows may consume the bounded "
            "round-2 bidirectional method; no hand, render, contact, action, deployment or Robot authority."
        ),
    }
    atomic_new(args.output_root / "RESULT.json", aggregate)
    print(json.dumps({"status": aggregate["status"], "counts": counts, "result": ref(args.output_root / "RESULT.json")}, ensure_ascii=False))
    return 0 if counts["runtime_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
