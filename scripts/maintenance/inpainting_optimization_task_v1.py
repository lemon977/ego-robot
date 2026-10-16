"""Finite CPU candidate engineering, using the existing fenced CAS publisher.

This bootstrap cannot run models, training, cleanup, or physical robot commands.
One Task Packet owns bounded candidate-only code, audit, tests, and documentation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso
from chaoyang.governance.register_single_task_packet import _validate_packet


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def _publish(infra, *args, **kwargs):
    """Refresh only candidate route builders before the existing CAS publish.

    The finite CPU bootstrap can stay alive while its candidate-only code is
    edited. Cached route builders must not sign the pre-edit dependency closure.
    Shared publisher/infra modules and frozen V1 implementations are not reloaded.
    A refresh error aborts publication rather than publishing a stale contract.
    """
    importlib.invalidate_caches()
    candidate = importlib.import_module("chaoyang.governance.inpainting_candidate_registration")
    importlib.reload(candidate)
    return infra._publish(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.spec.absolute()
    require(spec_path.resolve(strict=True) == spec_path and spec_path.is_relative_to(REPO_ROOT), "PROJECT_SPEC_REQUIRED")
    spec = load_json(spec_path)
    require(spec["schema"] == "inpainting-finite-cpu-engineering-spec-v1", "NEW_SPEC_REQUIRED")
    require(0 < spec["budgets"]["wall_seconds_max"] <= 7200 and spec["budgets"]["gpu_count"] == 0, "FINITE_CPU_ONLY")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    require(artifact_ref(infra_path) == spec["infrastructure"], "INFRASTRUCTURE_DRIFT")
    module = importlib.util.spec_from_file_location("_existing_inpainting_publisher", infra_path)
    infra = importlib.util.module_from_spec(module)
    module.loader.exec_module(infra)
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    require(check.returncode == 0, "GOVERNANCE_REQUIRED:" + check.stdout[-2000:] + check.stderr[-2000:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    for ref in spec["inputs"].values():
        infra._verify(ref)
    task_id = spec["task_id"]
    require(task_id.replace("_", "").isalnum(), "SAFE_TASK_ID_REQUIRED")
    output = Path(spec["output_dir"])
    require(output.is_absolute() and output.resolve() == output and output.is_relative_to(REPO_ROOT), "PROJECT_OUTPUT_REQUIRED")
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    require(not packet_path.exists() and not output.exists(), "FRESH_TASK_REQUIRED")
    writer = infra._writer_identity(task_id, 1)
    started = datetime.now(timezone.utc)
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": spec["objective"], "phase": "INPAINTING_G0_CANDIDATE_CPU_ENGINEERING",
        "plan_revision": "INPAINTING_GENERALIZATION_199_0918_20261012",
        "execution_revision": task_id.upper(), "execution_mode": "CPU_DEVELOPMENT_ONLY",
        "read_set": spec["read_set"], "read_set_expansions": spec["read_set_expansions"],
        "write_set": spec["write_set"], "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER"],
        "required_outputs": ["MAINTENANCE_DONE.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": spec["budgets"], "deletion_allowed": False, "algorithm_execution_allowed": False,
        "stop_conditions": ["deadline", "writer conflict", "protected input SHA drift", "unregistered model execution"],
        "claim_limit": "CPU candidate engineering and preflight only; no visual adoption, model execution, training or deployment claims.",
        "commands": ["Bounded candidate-only development, synthetic CPU tests, read-only source/signature/resource audit and project-local bounded storage probe"],
        "inputs": {**spec["inputs"], "spec": artifact_ref(spec_path)},
        "runtime_code": [artifact_ref(Path(__file__).resolve()), spec["infrastructure"]],
        "writer": writer, "writer_file": str(output / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(output),
        "execution_started_at": started.isoformat(),
        "deadline_at": (started + timedelta(seconds=spec["budgets"]["wall_seconds_max"])).isoformat(),
        "control_ground_truth": False, "physical_deployment_authorized": False,
    }
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    require(not _validate_packet(packet), "INVALID_TASK_PACKET")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    ref = infra._json_once(packet_path, packet)
    infra._json_once(output / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "updated_at": now_iso(), "heartbeat_at": now_iso(),
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "task_packet": ref, "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_CANDIDATE_ENGINEERING_NO_MODELS"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": ref["sha256"], "execution_allowed": True,
        "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    _publish(infra, state, index, "INPAINTING_G0_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "run_signature": packet["run_signature"], "writer": writer}), flush=True)
    marker_path = output / "MAINTENANCE_DONE.json"
    last = time.monotonic()
    errors, marker_ref = [], None
    try:
        while not marker_path.exists():
            require(datetime.now(timezone.utc).isoformat() < packet["deadline_at"], "FINITE_DEADLINE")
            if time.monotonic() - last >= 60:
                receipt, state, index = infra._snapshot()
                row, _ = infra._own_row(packet, state, index)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                _publish(infra, state, index, "INPAINTING_G0_HEARTBEAT", expected_revision=receipt["governance_revision"])
                last = time.monotonic()
            time.sleep(1)
        marker = load_json(marker_path)
        require(marker["task_id"] == task_id and marker["run_signature"] == packet["run_signature"] and marker["status"] in {"PASS", "BLOCKED"}, "MARKER_BINDING_REQUIRED")
        for ref in marker["artifacts"]:
            infra._verify(ref)
        for ref in spec["inputs"].values():
            infra._verify(ref)
        marker_ref = artifact_ref(marker_path)
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    receipt, state, index = infra._snapshot()
    row, pointer = infra._own_row(packet, state, index)
    result = {"schema": "inpainting-candidate-cpu-result-v1", "task_id": task_id, "run_signature": packet["run_signature"],
        "status": "PASSED" if not errors else "FAILED_RUNTIME_FINAL", "evidence": marker_ref,
        "scope_status": marker.get("status") if marker_ref else "FAILED",
        "errors": errors, "files_deleted": 0, "algorithm_calls": 0, "training_steps": 0,
        "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "control_ground_truth": False, "physical_deployment_authorized": False, "completed_at": now_iso()}
    row.update(status=result["status"], result=infra._json_once(output / "RESULT.json", result),
        pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    _publish(infra, state, index, "INPAINTING_G0_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(output / "POST_VALIDATION.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "scope_status": result["scope_status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if not errors and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
