#!/usr/bin/env python3
"""Incrementally stage non-heldout exact78 HaWoR/Object ICT inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1"
COHORT = CONTROL / "COHORT_EXACT78_FROZEN.json"
HAWOR = CONTROL / "HAWOR_BATCH_CHECKPOINT_V2.json"
ROOT = CONTROL / "training_inputs"
STATE = ROOT / "STAGING_CHECKPOINT.json"
HEARTBEAT = ROOT / "STAGING_HEARTBEAT.json"
COHORT_SHA = "d482ae4b627efa78653b69af26a357cec66381e3416dd78dd278504579e7790e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def identity() -> dict[str, object]:
    fields = Path("/proc/self/stat").read_text().split()
    return {"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(),
            "sid": os.getsid(0), "proc_start_ticks": int(fields[21]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def counts(state: dict) -> dict:
    result = {}
    for task in ("chips", "poker"):
        selected = [value for value in state["sessions"].values() if value["task"] == task]
        result[task] = {status: sum(value["status"] == status for value in selected)
                        for status in ("PASS", "FAIL", "WAITING", "HELDOUT_EXCLUDED")}
    return result


def heartbeat_loop(state: dict, stop: threading.Event) -> None:
    while not stop.wait(10):
        atomic_json(HEARTBEAT, {"schema_version": "exact78-ict-staging-heartbeat-v1",
                    "updated_at": now(), "process": identity(), "counts": counts(state)})


def main() -> int:
    if digest(COHORT) != COHORT_SHA:
        raise RuntimeError("exact78 cohort SHA drift")
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    if STATE.is_file():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("cohort_sha256") != COHORT_SHA:
            raise RuntimeError("staging state cohort drift")
    else:
        state = {"schema_version": "exact78-ict-staging-checkpoint-v1",
                 "created_at": now(), "updated_at": now(), "status": "RUNNING",
                 "cohort_sha256": COHORT_SHA, "hawor_checkpoint": str(HAWOR.resolve()),
                 "sessions": {row["session_id"]: {"task": row["task"], "split": row["split"],
                    "raw_path": row["raw_path"],
                    "status": "HELDOUT_EXCLUDED" if row["split"] == "heldout" else "WAITING"}
                    for row in cohort["sessions"]}}
        atomic_json(STATE, state)
    atomic_json(HEARTBEAT, {"schema_version": "exact78-ict-staging-heartbeat-v1",
                "updated_at": now(), "process": identity(), "counts": counts(state)})
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat_loop, args=(state, stop), daemon=True)
    thread.start()
    try:
        while True:
            hawor = json.loads(HAWOR.read_text(encoding="utf-8")) if HAWOR.is_file() else {}
            available = {row["session_id"]: row for row in hawor.get("results", [])}
            progressed = False
            for session, row in state["sessions"].items():
                if row["status"] != "WAITING" or session not in available:
                    continue
                source = available[session]
                result_ref = source.get("result", {})
                result_path = Path(str(result_ref.get("path", "")))
                if source.get("grade") not in {"A", "B"} or not source.get("consumption_authorized") or not result_path.is_file():
                    row.update({"status": "FAIL", "defect": "HAWOR_NOT_CONSUMABLE_A_OR_B"})
                else:
                    command = ["python3", "HumanEgo/tools/stage_exact78_grade_b_ict_session.py",
                        "--task", row["task"], "--session", session, "--split", row["split"],
                        "--mps-path", row["raw_path"], "--hawor-result", str(result_path),
                        "--output-root", str(ROOT / row["task"])]
                    log = ROOT / row["task"] / "logs" / f"{session}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    receipt = ROOT / row["task"] / "receipts" / f"{session}.json"
                    if receipt.is_file():
                        returncode = 0
                    else:
                        with log.open("ab") as stream:
                            returncode = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT).returncode
                    if returncode == 0 and receipt.is_file():
                        row.update({"status": "PASS", "receipt": str(receipt.resolve()),
                                    "hawor_grade": source["grade"], "training_weight": 0.25})
                    else:
                        row.update({"status": "FAIL", "defect": "ICT_STAGING_PROCESS_FAILED",
                                    "returncode": returncode, "log": str(log.resolve())})
                state["updated_at"] = now()
                state["counts"] = counts(state)
                atomic_json(STATE, state)
                progressed = True
            terminal = all(row["status"] != "WAITING" for row in state["sessions"].values())
            if terminal:
                state["status"] = "COMPLETE"
                state["updated_at"] = now()
                atomic_json(STATE, state)
                return 0
            time.sleep(5 if progressed else 15)
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
