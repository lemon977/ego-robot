"""One finite CPU successor after this owned publisher's cache interruption."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso, process_identity
from chaoyang.governance.register_single_task_packet import _validate_packet


def main():
    parent_id = "inpainting_generalization_g0_publication_20261013"
    old_packet = load_json(REPO_ROOT / "tasks/current" / parent_id / "TASK_PACKET.json")
    old_writer = old_packet["writer"]
    identity = process_identity(old_writer["pid"])
    if identity["alive"] and identity["start_ticks"] == old_writer["proc_start_ticks"]:
        raise RuntimeError("OLD_WRITER_STILL_LIVE")
    parent = Path(old_packet["output_dir"])
    output = parent / "recovery_0001"
    task_id = "inpainting_generalization_g0_publication_recovery_20261013"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    if output.exists() or packet_path.exists():
        raise RuntimeError("NEW_FINITE_RECOVERY_REQUIRED")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    if artifact_ref(infra_path) != old_packet["runtime_code"][1]:
        raise RuntimeError("EXISTING_INFRASTRUCTURE_DRIFT")
    module = importlib.util.spec_from_file_location("_recovery_existing_publisher", infra_path)
    infra = importlib.util.module_from_spec(module)
    module.loader.exec_module(infra)
    # A verified partial publication is the only reason to read these raw
    # projections instead of the normal receipt-verified _snapshot().
    receipt = load_json(REPO_ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json")
    state = load_json(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    active = [r for r in state["tasks"] if r.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    pointers = [r for r in index["task_packets"] if r.get("execution_allowed") is True]
    if len(active) != 1 or active[0]["task_id"] != parent_id or len(pointers) != 1 or pointers[0]["task_id"] != parent_id:
        raise RuntimeError("UNRELATED_OR_DUAL_WRITER_RECOVERY_FORBIDDEN")
    if active[0].get("task_packet") != artifact_ref(Path(old_packet["packet_path"])):
        raise RuntimeError("OLD_PACKET_REF_DRIFT")
    if (active[0].get("pid") != old_writer["pid"]
            or active[0].get("proc_start_ticks") != old_writer["proc_start_ticks"]
            or active[0].get("run_signature") != old_packet["run_signature"]):
        raise RuntimeError("OLD_EXACT_WRITER_BINDING_DRIFT")
    for name, reference in old_packet["inputs"].items():
        infra._verify(reference)
    infra._json_once(output / "RECOVERY_INPUT_AUDIT.json", {
        "reason": "Owned Python publisher cached old selection module; publish stopped mid-projection, process exited.",
        "old_writer": old_writer, "old_process_identity": identity,
        "receipt": artifact_ref(REPO_ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"),
        "actual_state": artifact_ref(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"),
        "actual_index": artifact_ref(REPO_ROOT / "tasks/current/INDEX.json"),
        "parent_packet": artifact_ref(Path(old_packet["packet_path"])), "model_calls": 0})
    failure = {"schema": "inpainting-candidate-cpu-result-v1", "task_id": parent_id,
        "run_signature": old_packet["run_signature"], "status": "FAILED_RUNTIME_FINAL",
        "scope_status": "PUBLISHER_CACHE_INTERRUPTED", "algorithm_calls": 0,
        "files_deleted": 0, "training_steps": 0, "candidate_adopted": False,
        "visual_quality": "NOT_EVALUATED", "evidence": artifact_ref(output / "RECOVERY_INPUT_AUDIT.json"),
        "completed_at": now_iso(), "sealed_source_stages_disposition": "Complete same-signature stages retained; unverified partial forbidden."}
    old_result_ref = infra._json_once(parent / "RESULT.json", failure)
    active[0].update(status="FAILED_RUNTIME_FINAL", pid=None, proc_start_ticks=None,
                     result=old_result_ref, updated_at=now_iso(), heartbeat_at=now_iso())
    pointers[0].update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    writer = infra._writer_identity(task_id, 2)
    packet = {**old_packet, "task_id": task_id,
        "objective": "One finite CPU successor to restore coherent publication with fresh module imports; retain parent failure, finish verified Source, tests and G1 spec. No models.",
        "writer": writer, "writer_file": str(output / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(output),
        "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_code": [artifact_ref(Path(__file__)), artifact_ref(infra_path)],
        "write_set": [*old_packet["write_set"], str(packet_path), str(output)],
        "inputs": {**old_packet["inputs"], "interrupted_parent": artifact_ref(Path(old_packet["packet_path"]))},
        "prerequisites": ["EXACT_PARENT_WRITER_DEAD", "NO_OTHER_WRITER", "PROTECTED_SHA_UNCHANGED"],
        "budgets": {**old_packet["budgets"], "wall_seconds_max": 1800}}
    packet.pop("run_signature")
    packet["run_signature"] = hashlib.sha256(__import__("json").dumps(packet, sort_keys=True).encode()).hexdigest()
    if _validate_packet(packet):
        raise RuntimeError("RECOVERY_PACKET_INVALID")
    ref = infra._json_once(packet_path, packet)
    infra._json_once(output / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "task_packet": ref, "run_signature": packet["run_signature"], "heartbeat_at": now_iso(), "updated_at": now_iso()})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_PUBLICATION_RECOVERY_NO_MODELS"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "INPAINTING_OWNED_PUBLICATION_RECOVERED", expected_revision=receipt["governance_revision"])
    print({"status": "RECOVERED_CPU_READY", "task_id": task_id, "writer": writer}, flush=True)
    marker_path, last = output / "MAINTENANCE_DONE.json", time.monotonic()
    while not marker_path.exists():
        if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
            raise RuntimeError("RECOVERY_FINITE_DEADLINE")
        if time.monotonic() - last > 60:
            receipt, state, index = infra._snapshot()
            row, _ = infra._own_row(packet, state, index)
            row.update(heartbeat_at=now_iso(), updated_at=now_iso())
            infra._publish(state, index, "INPAINTING_RECOVERY_HEARTBEAT", expected_revision=receipt["governance_revision"])
            last = time.monotonic()
        time.sleep(1)
    marker = load_json(marker_path)
    if marker["task_id"] != task_id or marker["run_signature"] != packet["run_signature"] or marker["status"] != "PASS":
        raise RuntimeError("RECOVERY_MARKER_BINDING")
    for reference in marker["artifacts"]:
        infra._verify(reference)
    receipt, state, index = infra._snapshot()
    row, pointer = infra._own_row(packet, state, index)
    result = {"schema": "inpainting-candidate-cpu-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED", "scope_status": "PASS",
        "evidence": artifact_ref(marker_path), "algorithm_calls": 0, "files_deleted": 0,
        "training_steps": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "completed_at": now_iso()}
    row.update(status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(),
               result=infra._json_once(output / "RESULT.json", result))
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, "INPAINTING_RECOVERY_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(output / "POST_VALIDATION.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    print({"status": "PASSED", "governance_exit": check.returncode}, flush=True)
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
