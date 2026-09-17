#!/usr/bin/env python3
"""Relay a verified immutable worker heartbeat through the governance aggregator."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import process_identity  # noqa: E402


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--startticks", required=True)
    parser.add_argument("--local-heartbeat", type=Path, required=True)
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=25)
    parser.add_argument("--max-local-age-seconds", type=int, default=60)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    started = time.monotonic()
    while time.monotonic() - started <= args.max_wall_seconds:
        if args.terminal_result.is_file():
            return 0
        identity = process_identity(args.pid)
        if not identity["alive"] or str(identity["start_ticks"]) != str(args.startticks):
            return 2
        try:
            local = json.loads(args.local_heartbeat.read_text())
            age = (datetime.now().astimezone() - parse_time(str(local["heartbeat_at"]))).total_seconds()
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return 3
        if age < -5 or age > args.max_local_age_seconds:
            return 4
        command = [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            args.task_id,
            "--pid",
            str(args.pid),
            "--status",
            "RUNNING",
            "--phase",
            str(local.get("phase", "RUNNING")),
        ]
        session = local.get("session")
        if session:
            command.extend(["--session", str(session)])
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        time.sleep(max(5, args.interval_seconds))
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
