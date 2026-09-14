#!/usr/bin/env python3
"""Detach and identity-verify the exact78 ICT staging daemon."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "tasks/control/runs/20260908_nine_hour_156_pipeline_v1/training_inputs"
HEARTBEAT = ROOT / "STAGING_HEARTBEAT.json"
RECEIPT = ROOT / "STAGING_LAUNCH_RECEIPT.json"
LOG = ROOT / "staging_daemon.log"

def main() -> int:
    if RECEIPT.exists(): raise SystemExit("launch receipt already exists")
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab", buffering=0) as stream:
        process = subprocess.Popen(["python3", "HumanEgo/tools/run_exact78_ict_staging_daemon.py"],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
            start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None: raise SystemExit(f"staging daemon exited {process.returncode}")
        if HEARTBEAT.is_file():
            heartbeat = json.loads(HEARTBEAT.read_text())
            fields = (Path("/proc") / str(process.pid) / "stat").read_text().split()
            if heartbeat.get("process", {}).get("proc_start_ticks") == int(fields[21]): break
        time.sleep(.25)
    else: raise SystemExit("no identity-bound staging heartbeat")
    payload = {"schema_version": "exact78-ict-staging-launch-v1", "status": "PASS",
               "process": heartbeat["process"], "heartbeat": str(HEARTBEAT.resolve()),
               "checkpoint": str((ROOT / "STAGING_CHECKPOINT.json").resolve()),
               "log": str(LOG.resolve())}
    temp = RECEIPT.with_name(f".{RECEIPT.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n"); os.replace(temp, RECEIPT)
    print(json.dumps({"pid": process.pid, "receipt": str(RECEIPT)}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
