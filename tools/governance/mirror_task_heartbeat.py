from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tools.governance.common import process_identity


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
    args = parser.parse_args()

    if args.interval_seconds < 5:
        raise SystemExit("--interval-seconds must be >= 5")
    if bool(args.state_json) != bool(args.while_state_value):
        raise SystemExit("--state-json and --while-state-value must be supplied together")

    while True:
        identity = process_identity(args.target_pid)
        if not identity["alive"] or identity["start_ticks"] != args.target_start_ticks:
            return 0
        if args.state_json and read_nested(args.state_json, args.state_key) != args.while_state_value:
            return 0

        command = [
            sys.executable,
            "-m",
            "tools.governance.heartbeat_task",
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
        completed = subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[2])
        if completed.returncode != 0:
            return completed.returncode
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
