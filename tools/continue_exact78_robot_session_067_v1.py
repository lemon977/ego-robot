#!/usr/bin/env python3
"""Finite no-clobber continuation for the interrupted exact78 Chips067 Robot session.

This is a digital visual-proxy continuation only. It waits for the already-running
method-1 placement process, then executes the frozen second arm method, two hand
methods, diagnostic render and explicit quality-C terminal publication. It never
publishes control, contact, physical-deployment or training authority.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LANE = PROJECT / "tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot"
SESSION = "get_potato_chips_0902_067"
TASK = "chips"
PREFLIGHT = LANE / "ready_batch_067_preflight_anchor_v3_current_v1/RESULT.json"
TEMPORAL = LANE / "hawor_temporal_candidates_v1/batch_067_v1/RESULT.json"
PLACEMENT = LANE / "robot_numeric_candidates_v1/batch_067_anchor_v3_method1_placement_resume_v4/RESULT.json"
ARM2_ROOT = LANE / "robot_numeric_candidates_v1/batch_067_anchor_v3_arm_round2_bidirectional_resume_v1"
HAND1_ROOT = LANE / "robot_numeric_candidates_v1/batch_067_anchor_v3_hand_round1_resume_v1"
HAND2_ROOT = LANE / "robot_numeric_candidates_v1/batch_067_anchor_v3_hand_round2_bidirectional_resume_v1"
RENDER_ROOT = LANE / "robot_visual_candidates_v1/batch_067_resume_v1"
TERMINAL_ROOT = LANE / "robot_terminals_v1/chips/get_potato_chips_0902_067"
STATE = LANE / "CHIPS067_CONTINUATION_STATE.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_state(status: str, phase: str, **extra: object) -> None:
    value = {
        "schema_version": "exact78-chips067-continuation-v1",
        "updated_at": now(),
        "status": status,
        "phase": phase,
        "session": SESSION,
        "pid": os.getpid(),
        "control_ground_truth": False,
        "physical_deployment": False,
        **extra,
    }
    tmp = STATE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, STATE)


def alive(pid: int, startticks: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[21]) == startticks
    except (FileNotFoundError, IndexError, ValueError):
        return False


def run_phase(name: str, command: list[str], output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"no-clobber output already exists: {output}")
    write_state("RUNNING", name, command=command)
    log = LANE / f"chips067_{name}.log"
    if log.exists() or log.is_symlink():
        raise RuntimeError(f"no-clobber log already exists: {log}")
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
        handle.flush()
        os.fsync(handle.fileno())
    result = output / "RESULT.json"
    if process.returncode not in {0, 2} or not result.is_file():
        raise RuntimeError(f"{name} failed rc={process.returncode}; log={log}")
    return json.loads(result.read_text(encoding="utf-8"))


def only_row(value: dict) -> dict:
    rows = value.get("sessions", [])
    if len(rows) != 1 or rows[0].get("session") != SESSION:
        raise RuntimeError("expected exactly Chips067 in aggregate")
    return rows[0]


def ref_path(value: dict, label: str) -> str:
    path = value.get("path")
    if not isinstance(path, str) or not Path(path).is_file():
        raise RuntimeError(f"missing {label}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement-pid", type=int, required=True)
    parser.add_argument("--placement-startticks", type=int, required=True)
    args = parser.parse_args()
    heartbeat = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.governance.heartbeat_task",
            "--task-id",
            "exact78_v52_lane_c_contact_robot",
            "--pid",
            str(os.getpid()),
            "--status",
            "RUNNING",
            "--phase",
            "robot_067_finite_continuation",
            "--session",
            SESSION,
        ],
        cwd=PROJECT,
        stdout=(LANE / "chips067_continuation_heartbeat.log").open("x"),
        stderr=subprocess.STDOUT,
    )
    try:
        write_state("RUNNING", "wait_method1_placement")
        while alive(args.placement_pid, args.placement_startticks):
            time.sleep(5)
        if not PLACEMENT.is_file():
            raise RuntimeError("placement process ended without aggregate RESULT")
        placement = json.loads(PLACEMENT.read_text(encoding="utf-8"))
        if placement.get("status") != "PASS_METHOD1_PLACEMENT_EVALUATION_PUBLISHED":
            raise RuntimeError(f"placement terminal is {placement.get('status')}")

        arm2 = run_phase(
            "arm_round2",
            [sys.executable, "tools/run_exact78_robot_arm_bidirectional_round2_v52.py", "--preflight", str(PREFLIGHT), "--placement-result", str(PLACEMENT), "--output-root", str(ARM2_ROOT), "--max-workers", "1"],
            ARM2_ROOT,
        )
        hand1 = run_phase(
            "hand_round1",
            [sys.executable, "tools/run_exact78_robot_hand_round1_v52.py", "--preflight", str(PREFLIGHT), "--temporal-root", str(TEMPORAL), "--arm-final-result", str(ARM2_ROOT / "RESULT.json"), "--output-root", str(HAND1_ROOT), "--max-workers", "1"],
            HAND1_ROOT,
        )
        hand2 = run_phase(
            "hand_round2",
            [sys.executable, "tools/run_exact78_robot_hand_bidirectional_round2_v52.py", "--preflight", str(PREFLIGHT), "--temporal-root", str(TEMPORAL), "--hand-round1-result", str(HAND1_ROOT / "RESULT.json"), "--output-root", str(HAND2_ROOT)],
            HAND2_ROOT,
        )
        render = run_phase(
            "render_review",
            [sys.executable, "tools/render_exact78_robot_passed_batch_v52.py", "--preflight", str(PREFLIGHT), "--temporal-root", str(TEMPORAL), "--arm-final-result", str(ARM2_ROOT / "RESULT.json"), "--hand-final-result", str(HAND2_ROOT / "RESULT.json"), "--output-root", str(RENDER_ROOT)],
            RENDER_ROOT,
        )

        arm_row = only_row(arm2)
        hand_row = only_row(hand2)
        review_row = only_row(render)
        preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        temporal = json.loads(TEMPORAL.read_text(encoding="utf-8"))
        input_row = only_row(preflight)
        temporal_row = only_row(temporal)
        rounds = max(int(arm_row.get("arm_method_rounds_consumed", 1)), int(hand_row.get("hand_method_rounds_consumed", 1)))
        terminal_command = [
            sys.executable,
            "tools/publish_exact78_robot_failed_quality_terminal_v52.py",
            "--task", TASK,
            "--session", SESSION,
            "--clean-result", ref_path(input_row["clean_result"], "clean result"),
            "--hawor-result", ref_path(temporal_row["result"], "temporal result"),
            "--arm-result", ref_path(hand_row["arm_result"], "arm result"),
            "--arm-states", ref_path(hand_row["arm_states"], "arm states"),
            "--hand-result", ref_path(hand_row["result"], "hand result"),
            "--hand-states", ref_path(hand_row["states"], "hand states"),
            "--review-result", ref_path(review_row["result"], "review result"),
            "--method-rounds-consumed", str(rounds),
            "--output-root", str(TERMINAL_ROOT),
        ]
        if TERMINAL_ROOT.exists() or TERMINAL_ROOT.is_symlink():
            raise RuntimeError(f"terminal already exists: {TERMINAL_ROOT}")
        write_state("RUNNING", "publish_quality_terminal", command=terminal_command)
        terminal_log = LANE / "chips067_publish_terminal.log"
        with terminal_log.open("x", encoding="utf-8") as handle:
            process = subprocess.run(terminal_command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
            handle.flush()
            os.fsync(handle.fileno())
        if process.returncode != 0 or not (TERMINAL_ROOT / "RESULT.json").is_file():
            raise RuntimeError(f"terminal publication failed rc={process.returncode}")
        terminal = json.loads((TERMINAL_ROOT / "RESULT.json").read_text(encoding="utf-8"))
        write_state("PASSED", "chips067_terminal_published", terminal_status=terminal.get("status"), terminal_result=str(TERMINAL_ROOT / "RESULT.json"))
        return 0
    except Exception as error:
        write_state("FAILED_RUNTIME_FINAL", "continuation_stopped", error=str(error))
        return 1
    finally:
        heartbeat.send_signal(signal.SIGTERM)
        try:
            heartbeat.wait(timeout=10)
        except subprocess.TimeoutExpired:
            heartbeat.kill()


if __name__ == "__main__":
    raise SystemExit(main())
