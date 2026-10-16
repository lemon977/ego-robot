"""One bounded CPU recovery of the exactly identified, dead C02 executor."""
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso, process_identity
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.inpainting_candidate.contracts import verify_sealed_stage
from chaoyang.inpainting_candidate.resources import CANDIDATE_CARD_LOCK_ROOT


def main():
    old_id = "inpainting_generalization_g1_mask_c02_20261013"
    old_path = REPO_ROOT / "tasks/current" / old_id / "TASK_PACKET.json"
    old = load_json(old_path)
    identity = process_identity(old["writer"]["pid"])
    if identity["alive"] and identity["start_ticks"] == old["writer"]["proc_start_ticks"]:
        raise RuntimeError("EXACT_OLD_WRITER_STILL_LIVE")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    if gpu.stdout.strip():
        raise RuntimeError("GPU_CHILD_STOP_NOT_PROVED")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    if artifact_ref(infra_path) != load_json(Path(old["inputs"]["spec"]["path"]))["infrastructure"]:
        raise RuntimeError("PERMANENT_PUBLISHER_SHA_DRIFT")
    module = importlib.util.spec_from_file_location("_existing_finite_model_recovery", infra_path)
    infra = importlib.util.module_from_spec(module)
    module.loader.exec_module(infra)
    # SIGTERM interrupted the original multi-file heartbeat publication after
    # INDEX but before receipt. Read only these exact projections under the
    # existing lock, then prove the single dead parent below before mutation.
    with (REPO_ROOT / "_run/current/locks/governance.lock").open("rb") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_SH)
        receipt = load_json(REPO_ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json")
        state = load_json(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
        index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
        projection_refs = {name: artifact_ref(REPO_ROOT / relative) for name, relative in {
            "receipt": "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "state": "docs/governance/LONG_HORIZON_TASK_STATE.json",
            "index": "tasks/current/INDEX.json"}.items()}
        fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    pointers = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(active) != 1 or active[0].get("task_id") != old_id
            or len(pointers) != 1 or pointers[0].get("task_id") != old_id
            or active[0].get("task_packet") != artifact_ref(old_path)
            or active[0].get("pid") != old["writer"]["pid"]
            or active[0].get("proc_start_ticks") != old["writer"]["proc_start_ticks"]
            or active[0].get("run_signature") != old["run_signature"]):
        raise RuntimeError("UNRELATED_OR_DOUBLE_WRITER_NO_RECOVERY")
    for reference in old["inputs"].values():
        infra._verify(reference)
    previous_root = Path(old["output_dir"])
    stage_path = previous_root / "sessions/001/Mask/RECEIPT.json"
    stage = load_json(stage_path)
    verify_sealed_stage(stage, old["sessions"][0]["mask_signature"], output_root=stage_path.parent)
    retained = load_json(previous_root / "sessions/001/SESSION_RESULT.json")
    if retained["execution"] != "COMPLETE_MASK_ONLY" or retained["native_frames"] != 238:
        raise RuntimeError("EXACT_COMPLETE_001_STAGE_REQUIRED")
    infra._verify(retained["review"]["video"])
    progress = load_json(previous_root / "MODEL_PROGRESS.json")
    if progress["model_call_accounting"]["attempted_api_calls"] != 30:
        raise RuntimeError("EXACT_PRIOR_ATTEMPT_ACCOUNTING_REQUIRED")
    task_id = "inpainting_generalization_g1_c02_executor_recovery_20261013"
    output = REPO_ROOT / "_run" / task_id / "attempt_0001"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    if output.exists() or packet_path.exists():
        raise RuntimeError("RECOVERY_ALREADY_EXISTS_DO_NOT_REVIVE")
    lease_path = CANDIDATE_CARD_LOCK_ROOT / (old["gpu_mapping"]["uuid"] + ".json")
    lease = load_json(lease_path)
    expected = {"task_id": old_id, "run_signature": old["run_signature"],
        "fencing_token": old["writer"]["fencing_token"], "pid": old["writer"]["pid"],
        "proc_start_ticks": old["writer"]["proc_start_ticks"], "uuid": old["gpu_mapping"]["uuid"]}
    if any(lease.get(key) != value for key, value in expected.items()):
        raise RuntimeError("STALE_LEASE_NOT_EXACT_PARENT_NO_RECOVERY")
    audit_ref = infra._json_once(output / "RECOVERY_INPUT_AUDIT.json", {
        "reason": "C02 tool session exited143 about900s after start; exact writer and GPU child now absent; external SIGTERM cause unconfirmed",
        "old_process_identity": identity, "gpu_compute_apps": gpu.stdout,
        "old_packet": artifact_ref(old_path), "old_progress": artifact_ref(previous_root / "MODEL_PROGRESS.json"),
        "interrupted_projection_refs": projection_refs,
        "retained_001_mask": artifact_ref(stage_path), "retained_001_session": artifact_ref(previous_root / "sessions/001/SESSION_RESULT.json"),
        "old_card_lease": artifact_ref(lease_path), "prior_model_api_calls": 30,
        "failed_partial_003_consumable": False, "new_model_calls": 0})
    # Preserve these generated control bytes; no data/code/result deletion.
    lease_path.rename(output / "STALE_CARD_LEASE.json")
    failure = {"schema": "inpainting-g1-mask-executor-result-v1", "task_id": old_id,
        "run_signature": old["run_signature"], "status": "FAILED_RUNTIME_FINAL",
        "errors": ["EXECUTOR_SIGTERM_143", "WRITER_PROCESS_UNAVAILABLE"],
        "completed_mask_sessions": 1, "planned_sessions": 6, "prior_model_api_calls": 30,
        "evidence": audit_ref, "model_result": None, "model_result_consumable": False,
        "retained_sealed_001": artifact_ref(stage_path), "partial_consumable": False,
        "candidate_adopted": False, "visual_quality": "001_AI_SAMPLE_FAILED_REMAINDER_NOT_REVIEWED",
        "files_deleted": 0, "training_steps": 0, "completed_at": now_iso()}
    old_result = infra._json_once(previous_root / "RESULT.json", failure)
    active[0].update(status="FAILED_RUNTIME_FINAL", result=old_result, pid=None,
                     proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointers[0].update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    writer = infra._writer_identity(task_id, 2)
    started = datetime.now(timezone.utc)
    parent_cpu = load_json(REPO_ROOT / "tasks/current/inpainting_generalization_g1_c01_runtime_repair_20261013/TASK_PACKET.json")
    packet = {**parent_cpu, "task_id": task_id,
        "objective": "Bounded CPU repair of executor lifetime, fixed remaining-five successor with original deadline and call budget, retain sealed001 evidence; no segmentation recipe/weights/threshold changes or models",
        "execution_revision": task_id.upper(), "writer": writer, "writer_file": str(output / "WRITER.json"),
        "packet_path": str(packet_path), "output_dir": str(output),
        "execution_started_at": started.isoformat(), "deadline_at": (started + timedelta(minutes=30)).isoformat(),
        "budgets": {**parent_cpu["budgets"], "wall_seconds_max": 1800},
        "runtime_code": [artifact_ref(Path(__file__)), artifact_ref(infra_path)],
        "inputs": {**parent_cpu["inputs"], "interrupted_c02": artifact_ref(old_path), "failed_c02_result": old_result},
        "read_set_expansions": [*parent_cpu["read_set_expansions"], "Exact C02 dead process/lease, complete001 Mask and reviews, no003 partial consumption"],
        "write_set": ["scripts/maintenance/inpainting_candidate_model_task_v1.py", "scripts/maintenance/inpainting_candidate_detach_v1.py",
            "scripts/maintenance/inpainting_candidate_model_recover_v1.py", "scripts/maintenance/inpainting_candidate_specs_v1.py",
            "tests/unit/test_inpainting_candidate_model_execution.py", "inpainting_optimization/",
            str(output), str(packet_path.parent), "Project .git selective owned candidate-only commit",
            "Existing CAS publisher generated projections only; exact dead generated GPU lease preserved into recovery audit"]}
    packet.pop("run_signature")
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    if _validate_packet(packet):
        raise RuntimeError("CPU_RECOVERY_PACKET_INVALID")
    packet_ref = infra._json_once(packet_path, packet)
    infra._json_once(output / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "task_packet": packet_ref,
        "run_signature": packet["run_signature"], "heartbeat_at": now_iso(), "updated_at": now_iso()})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_EXECUTOR_RECOVERY_NO_MODELS"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "INPAINTING_C02_EXACT_DEAD_WRITER_RECOVERED", expected_revision=receipt["governance_revision"])
    print({"status": "RECOVERED_CPU_READY", "task_id": task_id, "writer": writer}, flush=True)
    marker_path, last = output / "MAINTENANCE_DONE.json", time.monotonic()
    while not marker_path.exists():
        if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
            raise RuntimeError("CPU_RECOVERY_DEADLINE")
        if time.monotonic() - last >= 60:
            receipt, state, index = infra._snapshot()
            row, _ = infra._own_row(packet, state, index)
            row.update(heartbeat_at=now_iso(), updated_at=now_iso())
            infra._publish(state, index, "INPAINTING_C02_EXECUTOR_REPAIR_HEARTBEAT", expected_revision=receipt["governance_revision"])
            last = time.monotonic()
        time.sleep(1)
    marker = load_json(marker_path)
    if (marker["task_id"] != task_id or marker["run_signature"] != packet["run_signature"]
            or marker["status"] != "PASS"):
        raise RuntimeError("CPU_RECOVERY_MARKER_BINDING")
    for reference in marker["artifacts"]:
        infra._verify(reference)
    receipt, state, index = infra._snapshot()
    row, pointer = infra._own_row(packet, state, index)
    value = {"schema": "inpainting-candidate-cpu-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED", "scope_status": "PASS",
        "evidence": artifact_ref(marker_path), "algorithm_calls": 0, "files_deleted": 0,
        "candidate_adopted": False, "training_steps": 0, "visual_quality": "NOT_EVALUATED", "completed_at": now_iso()}
    row.update(status="PASSED", result=infra._json_once(output / "RESULT.json", value), pid=None,
               proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, "INPAINTING_C02_EXECUTOR_REPAIR_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(output / "POST_VALIDATION.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    print({"status": "PASSED", "governance_exit": check.returncode}, flush=True)
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
