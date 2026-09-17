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
import time
from pathlib import Path

from chaoyang.governance.common import RECEIPT_PATH, load_json, process_identity


def publish_exit_state(task_id: str, status: str, phase: str, message: str) -> int:
    """Close a mirrored runtime identity without leaving a ghost RUNNING row."""
    for attempt in range(20):
        revision = int(load_json(RECEIPT_PATH)["governance_revision"])
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "chaoyang.governance.update_task_state",
                "--task-id",
                task_id,
                "--status",
                status,
                "--phase",
                phase,
                "--clear-runtime",
                "--message",
                message,
                "--expected-revision",
                str(revision),
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0:
            return 0
        output = (completed.stdout + completed.stderr).lower()
        if not any(token in output for token in ("revision conflict", "cas revision mismatch")):
            print(output[-2000:], file=sys.stderr)
            return completed.returncode
        time.sleep(min(0.1 * (attempt + 1), 1.0))
    return 2


def read_nested(path: Path, dotted_key: str) -> object:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"missing state key {dotted_key!r} in {path}")
        value = value[key]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror an existing worker into the governance heartbeat without restarting it."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--target-start-ticks", type=int, required=True)
    parser.add_argument("--status", choices=["CLAIMED", "RUNNING"], required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--state-key", default="status")
    parser.add_argument("--while-state-value")
    parser.add_argument("--exit-status")
    parser.add_argument("--exit-phase")
    parser.add_argument("--exit-message", default="Mirrored worker exited; runtime identity cleared.")
    args = parser.parse_args()

    if args.interval_seconds < 5:
        raise SystemExit("--interval-seconds must be >= 5")
    if bool(args.state_json) != bool(args.while_state_value):
        raise SystemExit("--state-json and --while-state-value must be supplied together")
    if bool(args.exit_status) != bool(args.exit_phase):
        raise SystemExit("--exit-status and --exit-phase must be supplied together")

    while True:
        identity = process_identity(args.target_pid)
        if not identity["alive"] or identity["start_ticks"] != args.target_start_ticks:
            break
        if args.state_json and read_nested(args.state_json, args.state_key) != args.while_state_value:
            break

        command = [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            args.task_id,
            "--pid",
            str(args.target_pid),
            "--status",
            args.status,
            "--phase",
            args.phase,
            "--session",
            args.session,
        ]
        if args.gpu_id is not None:
            command.extend(["--gpu-id", str(args.gpu_id)])
        completed = subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[3])
        if completed.returncode != 0:
            return completed.returncode
        time.sleep(args.interval_seconds)
    if args.exit_status:
        return publish_exit_state(
            args.task_id,
            args.exit_status,
            args.exit_phase,
            args.exit_message,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
