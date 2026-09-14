#!/usr/bin/env python3
"""Bounded, scheduling-only CPU prefetch for frozen exact78 Clean v5.2.1.

This helper runs the *same* pinned per-session prepare producer used by the
authoritative guardian.  It never runs the donor, ProPainter, validation, or
publication stages.  A preparation is useful only after the producer has
atomically published its receipt; partial temporary directories are ignored
by the main guardian.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PREPARE = PROJECT / "tools/prepare_exact78_clean_wave_session_v52.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_one(plan_root: Path, selection: Path, receipt_root: Path, session: str) -> dict[str, Any]:
    receipt = plan_root / "preparation_receipts" / f"{session}.json"
    if receipt.is_file():
        return {"session": session, "status": "ADOPT_EXISTING", "receipt": ref(receipt)}
    final_result = plan_root / "propainter_v1" / session / "RESULT.json"
    terminal = plan_root / "clean_terminals" / session / "RESULT.json"
    if final_result.is_file() or terminal.is_file():
        return {"session": session, "status": "SKIP_TERMINAL"}

    log_path = receipt_root / "logs" / f"{session}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    command = [
        sys.executable,
        str(PREPARE),
        "--plan-root",
        str(plan_root),
        "--selection",
        str(selection),
        "--session",
        session,
    ]
    with log_path.open("xb") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            preexec_fn=lambda: os.nice(10),
        )
        log.flush()
        os.fsync(log.fileno())
    result: dict[str, Any] = {
        "session": session,
        "started_at": started,
        "finished_at": now(),
        "returncode": process.returncode,
        "log": ref(log_path),
    }
    if process.returncode == 0 and receipt.is_file():
        result.update({"status": "PASSED_PREPARE", "receipt": ref(receipt)})
    else:
        result["status"] = "FAILED_PREPARE"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    plan_root = args.plan_root.resolve(strict=True)
    selection = (plan_root / "EXACT78_WAVE0_SELECTION.json").resolve(strict=True)
    receipt_root = args.receipt_root.resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    lock_path = receipt_root / "PREFETCH.lock"
    lock_handle = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("another prepare prefetch executor owns the lock") from error

    state_path = receipt_root / "STATE.json"
    base = {
        "schema_version": "exact78-clean-prepare-prefetch-v521",
        "claim_limit": "Scheduling-only CPU prepare; no Clean grade or authority.",
        "pid": os.getpid(),
        "proc_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
        "started_at": now(),
        "plan_root": str(plan_root),
        "selection": ref(selection),
        "prepare_producer": ref(PREPARE),
        "sessions": args.sessions,
        "workers": args.workers,
    }
    atomic_json(state_path, {**base, "status": "RUNNING"})
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(args.sessions)))) as pool:
        futures = {
            pool.submit(run_one, plan_root, selection, receipt_root, session): session
            for session in args.sessions
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:  # fail closed while allowing other unique sessions to finish
                results.append(
                    {"session": futures[future], "status": "FAILED_RUNTIME", "error": repr(error)}
                )
            atomic_json(state_path, {**base, "status": "RUNNING", "results": results, "updated_at": now()})
    passed = all(row["status"] in {"PASSED_PREPARE", "ADOPT_EXISTING", "SKIP_TERMINAL"} for row in results)
    final = {
        **base,
        "finished_at": now(),
        "status": "PASSED" if passed else "FAILED_RUNTIME_FINAL",
        "results": sorted(results, key=lambda row: args.sessions.index(row["session"])),
    }
    atomic_json(state_path, final)
    atomic_json(receipt_root / "RESULT.json", final)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
