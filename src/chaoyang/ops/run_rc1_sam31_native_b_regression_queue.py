#!/usr/bin/env python3
"""One bounded, sequential GPU queue for the two frozen Poker B regressions.

This is development evidence only. The central V7.1 launcher owns the lease;
the queue never reclaims, preempts, or retries a quality failure.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso, validate_artifact_ref


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_task_object_identity_v1/poker"
SESSIONS = (("play_cards_0901_001", 645, "attempt_0002"),
            ("play_cards_0901_005", 520, "attempt_0001"))
MIN_FREE_MIB = 61440
TOTAL_WAIT_BUDGET_S = 1800
PER_SESSION_WALL_S = 1200


def _startticks() -> str:
    return Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue_root.resolve()
    queue.mkdir(parents=True, exist_ok=False)
    inputs = []
    for session, frames, _ in SESSIONS:
        result_ref = artifact_ref(BASELINE / session / "RESULT.json")
        result = json.loads(Path(result_ref["path"]).read_text(encoding="utf-8"))
        if result.get("grade") != "B" or result.get("frame_count") != frames:
            raise RuntimeError(f"frozen B identity/frame mismatch: {session}")
        manifest_ref = result["artifacts"]["manifest"]
        errors = validate_artifact_ref(manifest_ref)
        if errors:
            raise RuntimeError(f"frozen B closure mismatch: {session}: {errors}")
        manifest = json.loads(Path(manifest_ref["path"]).read_text(encoding="utf-8"))
        if manifest.get("session") != session or len(manifest.get("frames", [])) != frames:
            raise RuntimeError(f"frozen B manifest frame mismatch: {session}")
        inputs.append({"session_id": session, "frame_count": frames,
                       "old_B_result": result_ref, "old_B_manifest": manifest_ref})

    started = time.monotonic()
    atomic_json(queue / "QUEUE_START.json", {
        "schema_version": "chaoyang-rc1-sam31-native-b-regression-queue-v1",
        "created_at": now_iso(), "pid": os.getpid(), "process_startticks": _startticks(),
        "session_order": [item[0] for item in SESSIONS],
        "inputs": inputs,
        "min_free_mib": MIN_FREE_MIB,
        "total_wait_budget_seconds": TOTAL_WAIT_BUDGET_S,
        "per_session_wall_seconds": PER_SESSION_WALL_S,
        "queue_code": artifact_ref(Path(__file__)),
        "worker_code": artifact_ref(ROOT / "src/chaoyang/ops/run_sam31_native_poker_b_regression.py"),
        "claim_limit": "Offline native SAM3.1 regression against old B only; old B is not Gold; no Mask authority or training eligibility.",
    })

    outcomes = []
    for session, frames, attempt_id in SESSIONS:
        remaining_wait = max(0, int(TOTAL_WAIT_BUDGET_S - (time.monotonic() - started)))
        session_root = queue / session
        receipt = session_root / f"GPU_WRAPPER_{attempt_id.upper()}.json"
        worker_root = session_root / "attempts" / attempt_id
        command = [
            sys.executable, "-m", "chaoyang.ops.run_gpu_command_with_v71_lease",
            "--task-id", f"research_sam31_native_poker_b_regression_{session}",
            "--attempt-id", attempt_id, "--priority", "REGRESSION",
            "--gpu-id", "0", "--min-free-mib", str(MIN_FREE_MIB),
            "--wait-seconds", str(remaining_wait),
            "--wall-seconds", str(PER_SESSION_WALL_S),
            "--receipt", str(receipt),
            "--claim-limit", "Frozen old B internal regression; offline full-sequence diagnostic; no Gold or authority",
            "--", sys.executable, "-m", "chaoyang.ops.run_sam31_native_poker_b_regression",
            "--session-id", session, "--output-root", str(worker_root),
        ]
        process = subprocess.Popen(command, cwd=ROOT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            atomic_json(queue / "QUEUE_HEARTBEAT.json", {
                "schema_version": "chaoyang-rc1-research-queue-heartbeat-v1",
                "updated_at": now_iso(), "queue_pid": os.getpid(),
                "queue_startticks": _startticks(), "child_pid": process.pid,
                "session_id": session, "attempt_id": attempt_id,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "remaining_wait_budget_seconds": max(0, int(TOTAL_WAIT_BUDGET_S - (time.monotonic() - started))),
            })
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        stdout, stderr = process.communicate()
        if not receipt.is_file():
            raise RuntimeError(f"GPU launcher exited without receipt: {session}: {process.returncode}: {stderr[-800:]}")
        launcher = json.loads(receipt.read_text(encoding="utf-8"))
        item = {"session_id": session, "frame_count": frames,
                "launcher_status": launcher.get("status"), "launcher_exit_code": process.returncode,
                "launcher_receipt": artifact_ref(receipt)}
        result_path = worker_root / "RESULT.json"
        if result_path.is_file():
            item["worker_result"] = artifact_ref(result_path)
        outcomes.append(item)
        if launcher.get("status") != "PASSED":
            break

    status = "PASSED_DEVELOPMENT_REGRESSION" if len(outcomes) == len(SESSIONS) and all(
        item["launcher_status"] == "PASSED" for item in outcomes) else (
        "BLOCKED_RESOURCE" if outcomes and outcomes[-1]["launcher_status"] == "BLOCKED_RESOURCE"
        else "FAILED_RUNTIME_FINAL")
    atomic_json(queue / "QUEUE_RESULT.json", {
        "schema_version": "chaoyang-rc1-sam31-native-b-regression-queue-v1",
        "finished_at": now_iso(), "status": status,
        "scheduled_sessions": len(SESSIONS), "attempted_sessions": len(outcomes),
        "outcomes": outcomes,
        "unstarted_sessions": [item[0] for item in SESSIONS[len(outcomes):]],
        "authority_promoted": False, "training_eligible": False,
        "claim_limit": "Internal old-B regression only; no pixel Gold, physical instance truth, or causal training proof.",
    })
    print(json.dumps({"status": status, "attempted_sessions": len(outcomes),
                      "result": str(queue / "QUEUE_RESULT.json")}, ensure_ascii=False))
    return 0 if status == "PASSED_DEVELOPMENT_REGRESSION" else 3


if __name__ == "__main__":
    raise SystemExit(main())
