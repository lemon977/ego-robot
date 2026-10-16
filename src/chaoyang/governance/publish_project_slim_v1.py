"""Finite CPU maintenance window; cache deletion requires a separate sealed proof."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import shlex
import shutil
import sys

from chaoyang.governance.common import AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, publish_bundle
from chaoyang.governance import publish_legacy_exact_delete_v1 as util
from chaoyang.governance.publish_project_branch_convergence_v1 import index_body, refs, validate_current
from chaoyang.governance.register_single_task_packet import _validate_packet

PLAN = "docs/plans/PROJECT_SLIM_V1_20261008"
INDEX = REPO_ROOT/"tasks/current/INDEX.json"
EDITS = ("scripts/takeover/autopilot_watchdog.sh", "scripts/takeover/progress_pulse.sh",
         "src/chaoyang/human_ego/README.md", "docs/current/PROJECT_BRANCH_MAP_ZH.md", "docs/current/AI_WORK_ENTRY_ZH.md")
ACTIVE = {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}


def paths(task):
    return (REPO_ROOT/f"_run/current/{task}/attempts/attempt_0001", REPO_ROOT/f"tasks/current/{task}/TASK_PACKET.json",
            REPO_ROOT/f"tasks/receipts/{task.upper()}.json")


def register(phase, proof, revision):
    validate_current()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]) or any(r["status"] in ACTIVE for r in state["tasks"]):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    task = f"project_slim_{phase}_{revision}_20261008"
    out, pp, receipt = paths(task)
    if any(p.exists() for p in (out, pp, receipt)): raise RuntimeError("FRESH_TASK_REQUIRED")
    keep = util.protected()
    before = refs(EDITS)
    if set(before)&set(keep): raise RuntimeError("EDIT_INTERSECTS_FROZEN_BASELINE")
    inputs = {"protected": keep, "edit_before": before, "plan": artifact_ref(REPO_ROOT/PLAN/"00_EXECUTION_ZH.md"),
              "code": [artifact_ref(Path(__file__)), artifact_ref(Path(util.__file__)), artifact_ref(REPO_ROOT/"src/chaoyang/governance/project_slim_cache_proof_v1.py")],
              "proof": artifact_ref(proof) if proof else None}
    target_paths = []
    if phase == "c1":
        if not proof: raise RuntimeError("SEALED_C0_REQUIRED")
        c0 = proof.parent
        terminal = load_json(c0/"RESULT.json")
        if terminal["status"] != "PASSED" or terminal["outputs"].get("PROOF.json") != inputs["proof"]:
            raise RuntimeError("C0_PROOF_NOT_SEALED")
        target_paths = [r["path"] for r in load_json(proof)["rows"] if r["status"] == "ELIGIBLE_C1"]
    writer = {**{k: util.identity(os.getpid())[k] for k in ("pid", "proc_start_ticks")}, "executor_epoch": 1,
              "fencing_token": secrets.token_hex(24), "scope": task}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task, "objective": "System recheck and precise regenerable-cache slimming without changing selected V1 algorithms",
        "phase": "CPU_SYSTEM_SLIM_"+phase.upper(), "plan_revision": "PROJECT_SLIM_V1_20261008", "execution_revision": task.upper(), "session_id": None,
        "deadline_at": (datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md", PLAN+"/00_EXECUTION_ZH.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
                     "docs/governance/DOC_AUTHORITY_MAP.json", "docs/governance/ALGORITHM_CONTRACT.json", "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json", "baseline_v1/MANIFEST.json"],
        "expanded_read_set": [*keep, *EDITS, "docs/governance/CURRENT_REGRESSION_MANIFEST.json", "tasks/current/INDEX.json", "/proc", *CACHE_ROOTS,
                              "docs/plans/EXACT78_TAKEOVER_20261004", "review", "docs/current", "_run/current/worktrees"],
        "read_expansion_reason": "Concrete space and stale background-loop findings require exact cache/ref/FD and navigation checks; no unbounded archive read.",
        "write_set": [str(out.relative_to(REPO_ROOT)), str(pp.relative_to(REPO_ROOT)), str(receipt.relative_to(REPO_ROOT)), "docs/governance", "tasks/current/INDEX.json", PLAN,
                      "_run/caches/project_slim_v1_20261008", *([*target_paths, *EDITS, "docs/current/PROJECT_SLIM_V1_ZH.md", "tests/unit/test_project_slim_v1.py"] if phase == "c1" else [])],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "BASELINE_PROTECTED_SHA_MATCH", *(["SEALED_C0_PROOF"] if phase == "c1" else [])],
        "required_outputs": ["RESULT.json", "PROOF.json"] if phase == "c0" else ["RESULT.json", "DELETE_RECEIPT.json", "REGRESSION.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0, "model_calls_max": 0, "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["protected_sha_drift", "double_writer", "live_holder", "nonregenerable", "deadline"], "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": [shlex.join([sys.executable, "-B", "-m", "chaoyang.governance.publish_project_slim_v1", *sys.argv[1:]])],
        "claim_limit": "Maintenance only; no model, algorithm, qualification, dataset or MANUS implementation change. Exact generated caches only; immutable evidence retained."}
    if errors := _validate_packet(packet): raise RuntimeError(str(errors))
    out.mkdir(parents=True); pp.parent.mkdir(parents=True)
    atomic_json(out/"WRITER.json", writer); atomic_json(pp, packet)
    if phase == "c1":
        for name in EDITS:
            dest = out/"before"/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO_ROOT/name, dest)
    shutil.copyfile(Path(__file__), out/"PUBLISHER_SOURCE.py")
    shutil.copyfile(REPO_ROOT/"src/chaoyang/governance/project_slim_cache_proof_v1.py", out/"PROOFER_SOURCE.py")
    state["tasks"].append({"task_id": task, "phase": packet["phase"], "attempt": 1, "status": "RUNNING", "session": "project_maintenance", "updated_at": now_iso(), "heartbeat_at": now_iso(),
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None, "task_packet": artifact_ref(pp), "run_signature": signature,
        "quality": "NOT_APPLICABLE", "adoption": "BASELINES_UNCHANGED", "review": "NOT_APPLICABLE"})
    state["next_task"] = {"task_id": task, "session": "project_maintenance", "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS", "stop_condition": "finite maintenance terminal"}
    index["task_packets"].append({"task_id": task, "packet_path": str(pp.relative_to(REPO_ROOT)), "packet_sha256": artifact_ref(pp)["sha256"], "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="PROJECT_SLIM_REGISTERED", expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index)
    return packet


def guard(packet):
    out, pp, _ = paths(packet["task_id"])
    identity = util.identity(os.getpid())
    if not identity or any(identity[k] != packet["writer"][k] for k in ("pid", "proc_start_ticks")) or load_json(out/"WRITER.json") != packet["writer"] or load_json(pp) != packet:
        raise RuntimeError("WRITER_OR_PACKET_DRIFT")
    allowed = [r for r in load_json(INDEX)["task_packets"] if r.get("execution_allowed")]
    live = [r for r in load_json(TASK_STATE_PATH)["tasks"] if r["status"] in ACTIVE]
    if len(allowed) != 1 or allowed[0]["task_id"] != packet["task_id"] or allowed[0]["packet_sha256"] != artifact_ref(pp)["sha256"] or len(live) != 1 or live[0]["task_id"] != packet["task_id"] or live[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("DOUBLE_WRITER_OR_ROUTING_DRIFT")
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]): raise RuntimeError("DEADLINE")
    if refs(tuple(packet["inputs"]["protected"])) != packet["inputs"]["protected"]: raise RuntimeError("PROTECTED_SHA_DRIFT")
    for ref in [*packet["inputs"]["code"], packet["inputs"]["plan"], *([packet["inputs"]["proof"]] if packet["inputs"]["proof"] else [])]:
        if artifact_ref(ref["path"]) != ref: raise RuntimeError("INPUT_SIGNATURE_DRIFT")


def heartbeat(packet):
    guard(packet); state = copy.deepcopy(load_json(TASK_STATE_PATH))
    next(r for r in state["tasks"] if r["task_id"] == packet["task_id"]).update(updated_at=now_iso(), heartbeat_at=now_iso())
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="PROJECT_SLIM_HEARTBEAT", expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__))


def finish(packet, failure=None):
    guard(packet); task = packet["task_id"]; out, _, receipt = paths(task)
    phase = "c1" if packet["inputs"]["proof"] else "c0"
    if failure is None:
        if phase == "c0" and not (out/"PROOF.json").is_file(): raise RuntimeError("PROOF_REQUIRED")
        if phase == "c1" and load_json(out/"REGRESSION.json")["status"] != "PASS": raise RuntimeError("REGRESSION_NOT_PASS")
    deleted = load_json(out/"DELETE_RECEIPT.json") if (out/"DELETE_RECEIPT.json").is_file() else {"files_deleted": 0, "bytes_deleted": 0, "allocated_bytes_deleted": 0}
    result = {"schema_version": "PROJECT_SLIM_RESULT_V1", "task_id": task, "status": "PASSED" if failure is None else "FAILED_RUNTIME_FINAL", "failure_code": failure,
              "files_deleted": deleted["files_deleted"], "bytes_deleted": deleted["bytes_deleted"], "allocated_bytes_deleted": deleted["allocated_bytes_deleted"],
              "protected_after": refs(tuple(packet["inputs"]["protected"])), "algorithm_changed": False, "numeric_gates_changed": False, "models_run": 0, "training_steps": 0,
              "edit_before": packet["inputs"]["edit_before"], "edit_after": refs(EDITS), "claim_limit": packet["claim_limit"], "outputs": {}}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in {"RESULT.json", "WRITER.json"}: result["outputs"][p.name] = artifact_ref(p)
    atomic_json(out/"RESULT.json", result); atomic_json(receipt, {"task_id": task, "status": result["status"], "result": artifact_ref(out/"RESULT.json")})
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    next(r for r in state["tasks"] if r["task_id"] == task).update(status=result["status"], pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(), result=artifact_ref(out/"RESULT.json"), terminal_receipt=artifact_ref(receipt))
    state["next_task"] = None; next(r for r in index["task_packets"] if r["task_id"] == task)["execution_allowed"] = False
    index.update(status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)), state, event_type="PROJECT_SLIM_TERMINAL", expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index)
    atomic_json(out/"POST_PUBLICATION_VALIDATION.json", validate_current())
    print(json.dumps({"status": result["status"], "files_deleted": result["files_deleted"], "bytes_deleted": result["bytes_deleted"], "result": artifact_ref(out/"RESULT.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("c0", "c1"), required=True); parser.add_argument("--proof", type=Path)
    parser.add_argument("--revision", choices=("v1", "v2"), default="v1")
    args = parser.parse_args(); packet = register(args.phase, args.proof, args.revision)
    print("READY_"+args.phase.upper(), flush=True)
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 30)
            if not ready: heartbeat(packet); continue
            command = sys.stdin.readline().strip()
            if command == "AUDIT" and args.phase == "c0": audit(packet)
            elif command == "DELETE" and args.phase == "c1": delete(packet)
            elif command == "STOP_LOOPS" and args.phase == "c1": stop_loops(packet)
            elif command == "HEARTBEAT": heartbeat(packet)
            elif command == "PUBLISH": finish(packet); return
            else: raise RuntimeError("ABORTED_OR_INVALID_COMMAND")
    except BaseException as exc:
        out, _, _ = paths(packet["task_id"])
        if not (out/"RESULT.json").exists(): finish(packet, repr(exc))
        raise


from chaoyang.governance.project_slim_cache_proof_v1 import CACHE_ROOTS, audit, delete, stop_loops
if __name__ == "__main__": main()
