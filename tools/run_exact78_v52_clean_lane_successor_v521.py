from __future__ import annotations

"""Run the fenced V5.2.1 Clean recovery after the 58-row preflight passes."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.governance.common import (
    RECEIPT_PATH, artifact_ref, atomic_json, load_json, now_iso, sha256_file,
    validate_artifact_ref,
)
from tools.governance.v52_contracts import build_run_signature, claim_executor


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "tasks/control/runs/20260913_exact78_v52/lane_a_clean_successor_v521"
PACKET = ROOT / "tasks/control/runs/20260913_exact78_v52/task_packets/exact78_v52_lane_a_clean_successor_v521/TASK_PACKET.json"
WAVE_ROOT = ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1"
PREFLIGHT = LANE / "preflight/EXACT78_WAVE0_CLEAN_PREFLIGHT.json"
LEGACY_PREFLIGHT = ROOT / "tools/preflight_exact78_clean_wave_v52_1.py"
GUARDIAN = ROOT / "tools/run_exact78_clean_wave_guardian_v52_1.py"
LEASE = ROOT / "_run/GPU_LEASE.json"


def gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if gpu.returncode:
        return {"pass": False, "error": gpu.stderr[-1000:]}
    parts = [item.strip() for item in gpu.stdout.splitlines()[0].split(",")]
    visible = []
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            fields = [item.strip() for item in line.split(",", 2)]
            if len(fields) == 3 and fields[1] != "[Not Found]":
                visible.append({"pid": fields[0], "name": fields[1], "used_memory_mib": fields[2]})
    lease = load_json(LEASE)
    free_disk = shutil.disk_usage(WAVE_ROOT).free
    required_disk = int(load_json(WAVE_ROOT / "DISK_BUDGET.json")["projection"]["required_free_bytes"])
    memory_used, memory_total, utilization = int(parts[1]), int(parts[2]), int(parts[3])
    passed = bool(
        lease.get("status") == "RELEASED" and not visible and utilization < 20
        and memory_total - memory_used >= 80 * 1024 and free_disk >= required_disk
    )
    return {
        "pass": passed, "observed_at": now_iso(), "gpu_index": int(parts[0]),
        "memory_used_mib": memory_used, "memory_total_mib": memory_total,
        "utilization_percent": utilization, "visible_compute_processes": visible,
        "not_found_driver_residue_ignored_as_process_ownership": True,
        "central_lease_status": lease.get("status"), "disk_free_bytes": free_disk,
        "disk_required_bytes": required_disk,
    }


def heartbeat(pid: int, status: str, phase: str, session: str | None = None, gpu: bool = False) -> None:
    command = [
        sys.executable, "-m", "tools.governance.heartbeat_task",
        "--task-id", "exact78_v52_lane_a_clean_successor_v521", "--pid", str(pid),
        "--status", status, "--phase", phase,
    ]
    if session:
        command += ["--session", session]
    if gpu:
        command += ["--gpu-id", "0"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"governance heartbeat failed: {completed.stdout[-2000:]} {completed.stderr[-1000:]}")


def prepare_signature() -> dict[str, Any]:
    LANE.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "wave0_selection": artifact_ref(WAVE_ROOT / "EXACT78_WAVE0_SELECTION.json"),
        "cpu_preflight": artifact_ref(PREFLIGHT),
        "terminal_passed_count": int(load_json(PREFLIGHT)["counts"]["TERMINAL_PASSED"]),
        "remaining_ready_count": int(load_json(PREFLIGHT)["counts"]["READY"]),
    }
    code_closure = {
        path.name: artifact_ref(path) for path in (
            GUARDIAN, Path(__file__), LEGACY_PREFLIGHT,
            ROOT / "tools/prepare_exact78_clean_wave_session_v52.py",
            ROOT / "tools/run_generic_same_session_real_donor_v1.py",
            ROOT / "tools/validate_generic_same_session_real_donor_v1.py",
            ROOT / "tools/launch_clean_synthetic_propainter_once.py",
            ROOT / "tools/run_clean_synthetic_propainter_baseline.py",
        )
    }
    config = {
        "execution_authority": artifact_ref(WAVE_ROOT / "EXECUTION_AUTHORITY_V52_1.json"),
        "gpu_gate": {"samples": 3, "interval_seconds": 10, "utilization_max_exclusive": 20, "minimum_free_memory_mib": 81920},
        "gpu_wait_timeout_seconds": 1800,
    }
    authority = load_json(WAVE_ROOT / "EXECUTION_AUTHORITY_V52_1.json")
    weights = authority["vendor"]["weights"]
    schema = {"task_packet": artifact_ref(PACKET), "plan_revision": "exact78-v5.2"}
    files = {}
    for name, value in (
        ("INPUT_MANIFEST.json", input_manifest), ("CODE_CLOSURE.json", code_closure),
        ("CONFIG.json", config), ("WEIGHTS.json", weights), ("SCHEMA.json", schema),
    ):
        path = LANE / name
        atomic_json(path, value)
        files[name] = artifact_ref(path)
    signature = build_run_signature({
        "input_manifest": files["INPUT_MANIFEST.json"],
        "code_closure": files["CODE_CLOSURE.json"], "config": files["CONFIG.json"],
        "weights": files["WEIGHTS.json"], "calibration_or_absent": "ABSENT",
        "schema": files["SCHEMA.json"],
    })
    atomic_json(LANE / "RUN_SIGNATURE.json", signature)
    return signature


def main() -> int:
    current = load_json(RECEIPT_PATH)
    for reference in current["files"].values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError(errors)
    preflight = load_json(PREFLIGHT)
    counts = preflight.get("counts", {})
    if (preflight.get("status") != "PASS" or counts.get("BLOCKED_PREREQ") != 0
            or counts.get("CONFLICT") != 0
            or counts.get("TERMINAL_PASSED", 0) + counts.get("READY", 0) != 58):
        raise RuntimeError("58-row CPU preflight does not close to terminal-passed + READY")
    signature = prepare_signature()
    claim = claim_executor(PACKET, "exact78-v52.1-clean-lane-successor", os.getpid(), signature["run_signature_sha256"])
    atomic_json(LANE / "EXECUTION_START.json", {
        "schema_version": "exact78-v52-clean-execution-start-v1", "created_at": now_iso(),
        "claim": claim, "run_signature": artifact_ref(LANE / "RUN_SIGNATURE.json"),
    })
    heartbeat(os.getpid(), "WAIT_GPU_RESOURCE", "gpu_gate_3x10s")
    samples = []
    deadline = time.monotonic() + 1800
    while len(samples) < 3:
        sample = gpu_snapshot()
        if sample.get("pass"):
            samples.append(sample)
            if len(samples) < 3:
                time.sleep(10)
        else:
            samples = []
            if time.monotonic() >= deadline:
                result = {"status": "BLOCKED_RESOURCE", "created_at": now_iso(), "last_gpu_sample": sample, "recovery_command": f"python -m tools.run_exact78_v52_clean_lane"}
                atomic_json(LANE / "RUNNER_RESULT.json", result)
                return 3
            time.sleep(30)
            heartbeat(os.getpid(), "WAIT_GPU_RESOURCE", "gpu_gate_3x10s")
    gate = {"schema_version": "exact78-v52-gpu-gate-v1", "status": "PASS", "samples": samples}
    atomic_json(LANE / "GPU_GATE_3X10S.json", gate)
    successor = WAVE_ROOT / "preflights" / f"PREFLIGHT_V52_{os.getpid()}_{int(time.time())}.json"
    preflight_run = subprocess.run(
        [sys.executable, str(LEGACY_PREFLIGHT), "--plan-root", str(WAVE_ROOT), "--output", str(successor)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    atomic_json(LANE / "LEGACY_PREFLIGHT_EXECUTION.json", {
        "returncode": preflight_run.returncode, "stdout_tail": preflight_run.stdout[-4000:],
        "stderr_tail": preflight_run.stderr[-2000:], "output": artifact_ref(successor) if successor.is_file() else None,
    })
    if preflight_run.returncode or load_json(successor).get("launch_authorized") is not True:
        raise RuntimeError("fresh pinned legacy preflight failed after V5.2 gate")
    heartbeat(os.getpid(), "RUNNING", "clean_guardian_start", gpu=True)
    log_path = LANE / "GUARDIAN.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(GUARDIAN), "--plan-root", str(WAVE_ROOT), "--preflight", str(successor)],
            cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(30)
            # The child also publishes per-phase heartbeats; this wrapper keeps
            # the executor claim alive without changing the child's session.
            identity = Path(f"/proc/{process.pid}/stat")
            if identity.is_file():
                heartbeat(process.pid, "RUNNING", "clean_guardian_active", gpu=True)
        log.flush()
        os.fsync(log.fileno())
    result = {
        "schema_version": "exact78-v52-clean-runner-result-v1", "created_at": now_iso(),
        "status": "PASSED" if process.returncode == 0 else "FAILED_RUNTIME_FINAL",
        "returncode": process.returncode, "guardian_log": artifact_ref(log_path),
        "run_signature": artifact_ref(LANE / "RUN_SIGNATURE.json"),
        "terminal_audit": artifact_ref(WAVE_ROOT / "TERMINAL_AUDIT_V52_1.json") if (WAVE_ROOT / "TERMINAL_AUDIT_V52_1.json").is_file() else None,
        "claim_limit": "Clean lane runner receipt only; final authority requires Wave0 terminal matrix audit.",
    }
    atomic_json(LANE / "RUNNER_RESULT.json", result)
    return 0 if process.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
