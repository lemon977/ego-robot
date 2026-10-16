"""Bounded CPU engineering publisher using the existing Chaoyang CAS ledger.

No model, dataset production, GPU lease or robot execution is started here.
Only a signed engineering completion marker can finish the owned new task.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso
from chaoyang.governance.register_single_task_packet import _validate_packet

INFRA_SHA = "d3fb4d710501c1b2da4c943f84094fe8e0abd03cbdb9bc9e4d0e06c5ba2980dd"


def require(value, code):
    if not value:
        raise RuntimeError(code)


def project_path(value, existing=False):
    p = Path(value)
    require(p.is_absolute() and not any(x.is_symlink() for x in (p, *p.parents)),
            "ABSOLUTE_NON_SYMLINK_PROJECT_PATH_REQUIRED")
    p = p.resolve(strict=existing)
    require(p != REPO_ROOT and p.is_relative_to(REPO_ROOT), "PROJECT_ONLY_PATH_REQUIRED")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec_path = project_path(args.spec, True)
    spec = load_json(spec_path)
    require(spec["schema"] == "humanego-joint26-cpu-engineering-task-spec-v1", "NEW_SPEC_REQUIRED")
    wall = spec["budgets"]["wall_seconds_max"]
    require(type(wall) is int and 60 <= wall <= 7200, "FINITE_CPU_BUDGET_REQUIRED")
    require(spec["budgets"]["gpu_count"] == 0 and spec["budgets"]["real_model_calls"] == 0,
            "CPU_ENGINEERING_ONLY")
    require(spec["physical_deployment_authorized"] is False and spec["control_ground_truth"] is False,
            "NO_PHYSICAL_AUTHORIZATION")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    require(artifact_ref(infra_path)["sha256"] == INFRA_SHA, "INFRASTRUCTURE_SHA_DRIFT")
    module_spec = importlib.util.spec_from_file_location("_joint26_existing_infrastructure", infra_path)
    infra = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(infra)
    task_id = spec["task_id"]
    require(task_id.replace("_", "").isalnum(), "SAFE_TASK_ID_REQUIRED")
    out = project_path(spec["output_dir"])
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    require(not packet_path.exists() and not (out / "RESULT.json").exists(), "NEW_TASK_ONLY")
    initial = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    require(initial.returncode == 0 and json.loads(initial.stdout)["status"] == "PASS", "GOVERNANCE_REQUIRED")
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    for ref in spec["inputs"].values():
        if isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}:
            infra._verify(ref)
    for ref in spec["runtime_code"]:
        infra._verify(ref)
    writer = infra._writer_identity(task_id, 1)
    started = datetime.now(timezone.utc)
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": spec["objective"], "phase": "HUMANEGO_ICT_JOINT26_CPU_ENGINEERING",
        "plan_revision": "JOINT26_20261012", "execution_revision": task_id.upper(),
        "execution_mode": "CPU_DEVELOPMENT_ONLY", "read_set": spec["read_set"],
        "read_set_expansions": ["Exact current source/cache/provider refs only; no unrelated branch or NAS writes"],
        "write_set": spec["write_set"], "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER"],
        "required_outputs": ["ENGINEERING_DONE.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": spec["budgets"], "stop_conditions": ["deadline", "SHA drift", "double writer"],
        "claim_limit": "CPU engineering and synthetic tests, not a new trained checkpoint, data admission or physical qualification",
        "commands": ["Prepare and test additive joint26 sources under the sole fenced CPU engineering writer"],
        "inputs": {**spec["inputs"], "engineering_spec": artifact_ref(spec_path)},
        "runtime_code": spec["runtime_code"], "writer": writer, "writer_file": str(out / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(out),
        "execution_started_at": started.isoformat(), "deadline_at": (started + timedelta(seconds=wall)).isoformat(),
        "control_ground_truth": False, "physical_deployment_authorized": False}
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    require(not _validate_packet(packet), "INVALID_PACKET")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    packet_ref = infra._json_once(packet_path, packet)
    infra._json_once(out / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "updated_at": now_iso(), "heartbeat_at": now_iso(),
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "task_packet": packet_ref, "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "ICT_JOINT26_ENGINEERING_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "writer": writer,
        "run_signature": packet["run_signature"], "deadline_at": packet["deadline_at"]}), flush=True)
    last_heartbeat = time.monotonic()
    marker_path = out / "ENGINEERING_DONE.json"
    errors, marker_ref = [], None
    signaled = []
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda n, f: signaled.append(n))
    try:
        while not marker_path.exists():
            require(not signaled, "OWN_PUBLISHER_INTERRUPTED")
            require(datetime.now(timezone.utc) < started + timedelta(seconds=wall), "FINITE_DEADLINE_EXCEEDED")
            if time.monotonic() - last_heartbeat >= 60:
                receipt, state, index = infra._snapshot()
                row, _ = infra._own_row(packet, state, index)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                infra._publish(state, index, "ICT_JOINT26_ENGINEERING_HEARTBEAT", expected_revision=receipt["governance_revision"])
                last_heartbeat = time.monotonic()
            time.sleep(1)
        marker = load_json(marker_path)
        require(marker["task_id"] == task_id and marker["run_signature"] == packet["run_signature"], "MARKER_BINDING_DRIFT")
        require(marker["status"] == "PASSED_CPU_ENGINEERING_NOT_MODEL_QUALITY", "CPU_PROOF_REQUIRED")
        for ref in marker["artifacts"]:
            infra._verify(ref)
        marker_ref = artifact_ref(marker_path)
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    receipt, state, index = infra._snapshot()
    row, pointer = infra._own_row(packet, state, index)
    result = {"schema": "humanego-joint26-engineering-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED" if not errors else "FAILED_RUNTIME_FINAL",
        "stage_result_ref": marker_ref, "errors": errors, "real_optimizer_updates": 0,
        "robot_commands_sent": 0, "control_ground_truth": False, "physical_deployment_authorized": False,
        "completed_at": now_iso()}
    result_ref = infra._json_once(out / "RESULT.json", result)
    row.update(status=result["status"], result=result_ref, pid=None, proc_start_ticks=None,
        updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    # Engineering may add signed Joint26 helpers while this writer is alive.
    # Refresh that route builder only; frozen infrastructure is never reloaded.
    importlib.invalidate_caches()
    importlib.reload(importlib.import_module("chaoyang.humanego_ict.joint26_contract"))
    infra._publish(state, index, "ICT_JOINT26_ENGINEERING_SEALED", expected_revision=receipt["governance_revision"])
    checked = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(out / "POST_VALIDATION.json", {"exit_code": checked.returncode,
        "stdout": checked.stdout, "stderr": checked.stderr})
    print(json.dumps({"status": result["status"], "result": result_ref,
        "governance_exit": checked.returncode, "errors": errors}), flush=True)
    return 0 if not errors and checked.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
