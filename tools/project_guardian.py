#!/usr/bin/env python3
"""Read-only project progress guardian.

The guardian records heartbeats, repository state, gate state, and suspicious
producer processes.  It never edits producer outputs, restarts a producer, or
crosses a human review gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path("/mnt/workspace/code/chaoyang")
STATE_ROOT = PROJECT_ROOT / "_run" / "guardian_v1"
PID_FILE = STATE_ROOT / "guardian.pid"
LOCK_FILE = STATE_ROOT / "guardian.lock"
HEARTBEAT_FILE = STATE_ROOT / "HEARTBEAT.json"
EVENT_FILE = STATE_ROOT / "events.jsonl"

FORBIDDEN_BEFORE_MASK_CLEAN_REVIEW = (
    "train_embodiment.py",
    "autotune_training.py",
    "freeze_best_checkpoint.py",
    "retarget_producer",
    "base_ik_producer",
    "robot_renderer",
    "compositor",
    "harmonizer",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_event(payload: dict[str, Any]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(EVENT_FILE, flags, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def command(args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def numbered_g2_candidates() -> list[tuple[int, Path, dict[str, Any]]]:
    """Return explicitly versioned G2 candidates without following symlinks."""

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    pattern = re.compile(r"^g2_004_mask_clean_candidate_v([1-9][0-9]*)$")
    run_root = PROJECT_ROOT / "_run"
    try:
        children = list(run_root.iterdir())
    except OSError:
        return candidates
    for child in children:
        match = pattern.fullmatch(child.name)
        if match is None or child.is_symlink() or not child.is_dir():
            continue
        manifest = load_json_if_present(child / "run_manifest.json")
        if manifest is None:
            continue
        candidates.append((int(match.group(1)), child, manifest))
    return sorted(candidates, key=lambda item: item[0])


def snapshot() -> dict[str, Any]:
    git_rc, git_status = command(["git", "status", "--short"])
    ps_rc, ps_output = command(["ps", "-eo", "pid=,etimes=,args="])
    process_lines = [line.strip() for line in ps_output.splitlines() if str(PROJECT_ROOT) in line]
    suspicious = [
        line
        for line in process_lines
        if any(token in line for token in FORBIDDEN_BEFORE_MASK_CLEAN_REVIEW)
        and "project_guardian.py" not in line
    ]
    g0 = load_json_if_present(
        PROJECT_ROOT / "archive/legacy_runs/unclassified/g0_20260826_parallel_v1/raw/RAW_VALIDATION.json"
    )
    g1 = load_json_if_present(PROJECT_ROOT / "archive/legacy_runs/unclassified/g0_20260826_parallel_v1/qa/G1_CONTRACT_VALIDATION.json")
    g2_candidates = numbered_g2_candidates()
    latest_g2 = None if not g2_candidates else g2_candidates[-1]
    g2 = None if latest_g2 is None else latest_g2[2]
    latest_g2_root = None if latest_g2 is None else latest_g2[1]
    review = None if latest_g2_root is None else load_json_if_present(
        latest_g2_root / "sessions/grap_a_cap_004/review/REVIEW_MANIFEST.json"
    )
    review_state = None if review is None else review.get("status")
    return {
        "schema_version": "project-guardian-heartbeat-v1",
        "timestamp": now_iso(),
        "pid": os.getpid(),
        "mode": "READ_ONLY_MONITOR_NO_AUTOMATIC_RESTART",
        "git_status_returncode": git_rc,
        "git_changed_entry_count": len([line for line in git_status.splitlines() if line]),
        "project_process_count": len(process_lines),
        "suspicious_pre_review_processes": suspicious,
        "gates": {
            "g0": None if g0 is None else g0.get("verdict"),
            "g1": None if g1 is None else g1.get("overall_verdict"),
            "g2": None if g2 is None else g2.get("status"),
            "g2_candidate_count": len(g2_candidates),
            "g2_latest_run_id": None if latest_g2_root is None else latest_g2_root.name,
            "mask_clean_review": review_state,
        },
        "policy": {
            "restart_producer": False,
            "modify_outputs": False,
            "cross_human_gate": False,
        },
    }


def run(interval_seconds: int, once: bool) -> None:
    if PROJECT_ROOT.resolve() != PROJECT_ROOT:
        raise RuntimeError("project root identity mismatch")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("guardian already running") from error
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    append_event({"event": "GUARDIAN_STARTED", "timestamp": now_iso(), "pid": os.getpid()})
    previous_alert: list[str] = []
    try:
        while not stop:
            state = snapshot()
            atomic_json(HEARTBEAT_FILE, state)
            current_alert = state["suspicious_pre_review_processes"]
            if current_alert != previous_alert:
                append_event({
                    "event": "PRE_REVIEW_PROCESS_ALERT_CHANGED",
                    "timestamp": now_iso(),
                    "processes": current_alert,
                })
                previous_alert = current_alert
            if once:
                break
            deadline = time.monotonic() + interval_seconds
            while not stop and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
    finally:
        append_event({"event": "GUARDIAN_STOPPED", "timestamp": now_iso(), "pid": os.getpid()})
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.interval_seconds < 10:
        raise SystemExit("interval must be at least 10 seconds")
    run(parsed.interval_seconds, parsed.once)
