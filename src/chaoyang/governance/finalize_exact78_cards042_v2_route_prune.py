#!/usr/bin/env python3
"""Remove the last two stale Cards/Chips packets from current routing."""

from __future__ import annotations
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from chaoyang.governance.common import (AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "exact78_cards042_v2_final_route_prune_20260929"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_V2_FINAL_ROUTE_PRUNE.json"
STALE = {"cards042_static_donor_8h_20260928", "exact78_chips103_existing_arm_completion_v1_20260929"}
CHIPS_ROOT = REPO_ROOT / "_run/current/exact78_chips103_existing_arm_completion_v1_20260929"


def index_body():
    return {k: copy.deepcopy(v) for k, v in load_json(INDEX).items()
            if k not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists():
        raise RuntimeError("FRESH_PRUNE_REQUIRED")
    targets = sorted(path.resolve() for path in CHIPS_ROOT.rglob("*.npz") if path.is_file())
    inventory = [artifact_ref(path) for path in targets]
    baseline = artifact_ref(REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json")
    other_branch = artifact_ref(REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json")
    ROOT.mkdir(parents=True)
    atomic_json(ROOT / "PRUNE_PLAN.json", {"stale_task_ids": sorted(STALE), "binary_targets": inventory,
        "removed_source_files": ["src/chaoyang/ops/run_exact78_arm_completion_v1.py",
                                 "src/chaoyang/governance/execute_exact78_arm_completion_v1.py"],
        "baseline_protected": baseline, "other_branch_protected": other_branch})
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"plan": artifact_ref(ROOT / "PRUNE_PLAN.json"), "baseline": baseline, "other_branch": other_branch}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Remove final stale Cards donor and rejected Chips arm routes from the current Exact78 index",
        "phase": "EXACT78_CARDS042_V2_FINAL_ROUTE_PRUNE", "plan_revision": "EXACT78_CARDS042_VISUAL_BASELINE_V2",
        "execution_revision": "EXACT78_CARDS042_V2_FINAL_ROUTE_PRUNE", "session_id": "Exact78_navigation",
        "deadline_at": (datetime.now().astimezone()+timedelta(minutes=30)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"],
        "write_set": [str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["NO_ACTIVE_WRITER", "V2_CLEANUP_PASS", "STALE_ROWS_TERMINAL_ONLY"],
        "required_outputs": ["stale_routes_absent", "chips_binary_removed", "governance_PASS"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0, "training_steps": 0,
                    "wall_seconds_max": 1800, "session_attempts_max": 1},
        "stop_conditions": ["active_writer", "protected_sha_drift", "one_attempt_terminal"],
        "claim_limit": "Navigation and rejected Chips binary retirement only; current V2 and 0916 Robot100 are unchanged.",
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": ["python -m chaoyang.governance.finalize_exact78_cards042_v2_route_prune"]}
    errors = _validate_packet(packet)
    if errors: raise RuntimeError("INVALID_PACKET:"+";".join(errors))
    atomic_json(ROOT/"WRITER.json", writer); PACKET.parent.mkdir(parents=True); atomic_json(PACKET, packet)
    now=now_iso(); state["tasks"].append({"task_id":TASK,"phase":packet["phase"],"attempt":1,"status":"RUNNING",
        "session":"Exact78_navigation","updated_at":now,"heartbeat_at":now,"pid":writer["pid"],
        "proc_start_ticks":writer["proc_start_ticks"],"gpu_id":None,"task_packet":artifact_ref(PACKET),
        "run_signature":signature,"quality":"PENDING","adoption":"NOT_APPLICABLE","review":"MACHINE_AUDIT"})
    state["next_task"]={"task_id":TASK,"session":"Exact78_navigation","prerequisites":packet["prerequisites"],
                        "expected_resource":"CPU_ONE_THREAD","stop_condition":"one terminal prune"}
    index["task_packets"]=[row for row in index["task_packets"] if row.get("task_id") not in STALE]
    index["task_packets"].append({"task_id":TASK,"packet_path":str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256":artifact_ref(PACKET)["sha256"],"execution_class":"CURRENT_LEDGER_ROUTABLE","execution_allowed":True})
    index.update(packet_revision="EXACT78_CARDS042_V2_FINAL_PRUNE_RUNNING",status="PASS",supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH),state,event_type="EXACT78_CARDS042_V2_FINAL_PRUNE_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),
        task_packet_index_path=INDEX,task_packet_index_value=index)
    for path in targets: path.unlink()
    if any(path.exists() for path in targets): raise RuntimeError("TARGET_REMAINS")
    if artifact_ref(baseline["path"])!=baseline or artifact_ref(other_branch["path"])!=other_branch:
        raise RuntimeError("PROTECTED_SHA_DRIFT")
    result={"status":"PASSED_FINAL_ROUTE_PRUNED","removed_task_ids":sorted(STALE),
            "removed_binary_files":len(targets),"removed_binary_bytes":sum(x["bytes"] for x in inventory),
            "baseline":baseline,"other_ai_branch_unchanged":True}
    atomic_json(ROOT/"RESULT.json",result); atomic_json(RECEIPT,{"task_id":TASK,"status":"PASSED",
        "result":artifact_ref(ROOT/"RESULT.json"),"claim_limit":packet["claim_limit"]})
    state=copy.deepcopy(load_json(TASK_STATE_PATH)); row=next(x for x in state["tasks"] if x["task_id"]==TASK)
    row.update(status="PASSED",pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),
               result=artifact_ref(ROOT/"RESULT.json"),terminal_receipt=artifact_ref(RECEIPT),
               quality="ROUTE_AND_PROTECTED_SHA_PASS",review="MACHINE_AUDITED"); state["next_task"]=None
    index=index_body()
    for item in index["task_packets"]:
        if item["task_id"]==TASK: item["execution_allowed"]=False
    index.update(packet_revision="EXACT78_CARDS042_V2_FINAL_PRUNE_TERMINAL",status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published=publish_bundle(load_json(AUTHORITY_PATH),state,event_type="EXACT78_CARDS042_V2_FINAL_PRUNE_COMPLETE",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),
        task_packet_index_path=INDEX,task_packet_index_value=index)
    check=subprocess.run([sys.executable,"-B","-m","chaoyang.cli","validate-governance"],cwd=REPO_ROOT,
        env={**os.environ,"PYTHONPATH":str(REPO_ROOT/"src")},capture_output=True,text=True)
    if check.returncode: raise RuntimeError(check.stdout[-4000:]+check.stderr[-4000:])
    print(json.dumps({"status":"PASSED","revision":published["governance_revision"],
                      "result":artifact_ref(ROOT/"RESULT.json")},ensure_ascii=False,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
