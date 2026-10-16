"""One fenced CPU source release; no execution-path or algorithm switch."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import subprocess
import sys
import tarfile
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
)
from chaoyang.governance.publish_project_branch_convergence_v1 import index_body, refs, validate_current
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance.baseline_v1_source_release import source_plan, parameters, copy_plan, verify_bundle, SELECTED_OPERATIONS

TASK = "baseline_v1_source_release_20261008"
PLAN = "docs/plans/BASELINE_V1_SOURCE_RELEASE_20261008"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PACKET = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
RECEIPT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
BUNDLE = REPO_ROOT / "baseline_v1"
EDITS = ("src/chaoyang/governance/current_r3_contracts.py", "README.md",
         "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/DUAL_BRANCH_BASELINE_V1_ZH.md")
WINDOW = 2700


def register():
    validate_current()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(p.exists() for p in (ATTEMPT, PACKET, RECEIPT, BUNDLE)):
        raise RuntimeError("FRESH_TASK_AND_BUNDLE_REQUIRED")
    parent = REPO_ROOT / "_run/current/dual_branch_phase_v1_20261008/attempts/attempt_0001"
    protected = load_json(parent / "RESULT.json")["protected_after"]
    protected.update(refs(tuple(str((parent/name).relative_to(REPO_ROOT)) for name in
                                ("RESULT.json", "CODE_SNAPSHOT.tar.gz", "CODE_SNAPSHOT_MANIFEST.json"))))
    protected.update(refs(("docs/current/DUAL_BRANCH_BASELINE_V1.json",)))
    plan = source_plan(REPO_ROOT)
    protected.update({row["relative_path"]:row["source"] for row in plan["sources"]+plan["metadata"] if row["relative_path"] not in EDITS})
    if refs(tuple(protected)) != protected:
        raise RuntimeError("PROTECTED_SOURCE_DRIFT")
    writer = {"pid":os.getpid(), "proc_start_ticks":process_identity(os.getpid())["start_ticks"],
              "executor_epoch":1, "fencing_token":secrets.token_hex(24), "scope":TASK}
    inputs = {"protected":protected, "source_plan_before":plan, "edit_before":refs(EDITS),
              "publisher":artifact_ref(Path(__file__)), "builder":artifact_ref(REPO_ROOT/"src/chaoyang/governance/baseline_v1_source_release.py"),
              "plan":artifact_ref(REPO_ROOT/PLAN/"00_EXECUTION_ZH.md")}
    signature = hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet = {"schema_version":"chaoyang-r22-task-packet-v1", "task_id":TASK,
        "objective":"Consolidate adopted visual v13 and MANUS v16 into one byte-preserving V1 source release; retain current runtime paths",
        "phase":"CPU_BASELINE_V1_SOURCE_RELEASE", "plan_revision":TASK.upper(), "execution_revision":TASK.upper(),
        "session_id":None, "deadline_at":(datetime.now().astimezone()+timedelta(seconds=WINDOW+180)).isoformat(),
        "read_set":["AGENTS.md", "docs/current/AI_WORK_ENTRY_ZH.md", PLAN+"/00_EXECUTION_ZH.md",
            "docs/governance/DOC_AUTHORITY_MAP.json", "docs/governance/ALGORITHM_CONTRACT.json",
            "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json", "docs/current/DUAL_BRANCH_BASELINE_V1.json",
            "docs/current/DUAL_BRANCH_BASELINE_V1_ZH.md"],
        "expanded_read_set":[*protected,*EDITS], "read_expansion_reason":"Pinned-path migration conflict requires the actual dependency and source closure.",
        "write_set":[str(ATTEMPT.relative_to(REPO_ROOT)),str(PACKET.relative_to(REPO_ROOT)),str(RECEIPT.relative_to(REPO_ROOT)),
            "baseline_v1", "tasks/current/INDEX.json", "docs/governance", PLAN, "_run/caches/baseline_v1_source_release", *EDITS,
            "tests/unit/test_baseline_v1_source_release.py"],
        "prerequisites":["GOVERNANCE_PASS","NO_ACTIVE_TASK","BASELINE_POINTERS_AND_SOURCE_PINNED"],
        "required_outputs":["RESULT.json","REGRESSION.json","SOURCE_RELEASE.tar.gz","baseline_v1/MANIFEST.json"],
        "budgets":{"workers_max":1,"cpu_threads_total":2,"gpu_count":0,"model_calls_max":0,"training_steps":0,
                   "wall_seconds_max":WINDOW+180,"session_attempts_max":1},
        "stop_conditions":["source_sha_drift","double_writer","bundle_conflict","deadline"],
        "claim_limit":"Frozen source consolidation only; no algorithm, numerical gate, production-loader, baseline-pointer, media, qualification or deletion change.",
        "writer":writer,"inputs":inputs,"run_signature":signature,
        "commands":[f"{sys.executable} -B -m chaoyang.governance.publish_baseline_v1_source_release"]}
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    ATTEMPT.mkdir(parents=True); PACKET.parent.mkdir(parents=True)
    atomic_json(ATTEMPT/"WRITER.json",writer); atomic_json(PACKET,packet)
    for name in EDITS:
        dst=ATTEMPT/"before"/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO_ROOT/name,dst)
    state["tasks"].append({"task_id":TASK,"phase":packet["phase"],"attempt":1,"status":"RUNNING","session":"source_release_only",
        "updated_at":now_iso(),"heartbeat_at":now_iso(),"pid":writer["pid"],"proc_start_ticks":writer["proc_start_ticks"],
        "gpu_id":None,"task_packet":artifact_ref(PACKET),"run_signature":signature,"quality":"NOT_ASSESSED",
        "adoption":"BASELINES_UNCHANGED","review":"NOT_APPLICABLE"})
    state["next_task"]={"task_id":TASK,"session":"source_release_only","prerequisites":packet["prerequisites"],
                        "expected_resource":"CPU_TWO_THREADS","stop_condition":"one terminal publication"}
    index["task_packets"].append({"task_id":TASK,"packet_path":str(PACKET.relative_to(REPO_ROOT)),"packet_sha256":artifact_ref(PACKET)["sha256"],
        "execution_class":"CURRENT_LEDGER_ROUTABLE","execution_allowed":True})
    index.update(status="PASS",supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="BASELINE_SOURCE_RELEASE_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),
        task_packet_index_path=INDEX,task_packet_index_value=index)
    return packet


def build(packet):
    if refs(tuple(packet["inputs"]["protected"])) != packet["inputs"]["protected"]:
        raise RuntimeError("PROTECTED_SOURCE_DRIFT")
    from chaoyang.exact78_baseline.stage_reuse import dependencies
    from chaoyang.governance.dual_branch_phase_v1 import validate_phase
    validate_phase(REPO_ROOT)
    plan=source_plan(REPO_ROOT)
    # Hash the already installed dependencies; no downloads, imports of models or inference.
    external={ref["path"]:ref for stage in ("motion","gpu","render") for ref in dependencies(REPO_ROOT,stage)}
    revision=load_json(REPO_ROOT/"docs/current/ROBOT100_HARDWARE_BASELINE_V16.json")["reference_revision"]
    reference=load_json(Path(revision["path"]))
    manus_manifest=load_json(Path(reference["manifest"]["path"]))
    for key in ("assets","frozen_inputs","snapshot_refs"):
        external.update({ref["path"]:ref for ref in manus_manifest[key]})
    external[manus_manifest["environment"]["path"]]=manus_manifest["environment"]
    copy_plan(plan,BUNDLE)
    for filename,alias in (("VISUAL.json","EXACT78_PHASE_V1"),("MANUS.json","MANUS_PHASE_V1")):
        rows=[r for r in plan["sources"] if alias in r["owners"]]
        atomic_json(BUNDLE/filename,{"alias":alias,"operation":SELECTED_OPERATIONS[alias],"code_paths":[r["bundle_path"] for r in rows],
            "shared_modules_have_one_copy":True,"claim_limit":"Index only; no operation or quality promotion."})
    atomic_json(BUNDLE/"SHARED.json",{"code_paths":[r["bundle_path"] for r in plan["sources"] if len(r["owners"])==2],
        "standalone_operations_authorized":False})
    atomic_json(BUNDLE/"PARAMETERS.json",parameters(plan))
    atomic_json(BUNDLE/"EXTERNAL_DEPENDENCIES.json",{"dependencies":[external[k] for k in sorted(external)],
        "visual_dependencies_full_sha_verified_now":True,"manus_source_contract":reference["manifest"],
        "manus_references":"Inherited from selected V16 reproduction manifest; not new acquisition replay or fresh verification of every dataset/weight.",
        "payloads_copied":False,"claim_limit":"Assets, weights, vendor, data and environment remain explicit prerequisites, not embedded files."})
    indexes={name:artifact_ref(BUNDLE/name) for name in ("VISUAL.json","MANUS.json","SHARED.json","PARAMETERS.json","EXTERNAL_DEPENDENCIES.json")}
    manifest={"schema_version":"DUAL_BRANCH_BASELINE_SOURCE_RELEASE_V1","status":"FROZEN_SOURCE_RELEASE_NOT_RUNTIME_MIGRATION",
        "task_id":TASK,"phase_contract":artifact_ref(REPO_ROOT/"docs/current/DUAL_BRANCH_BASELINE_V1.json"),**plan,
        "selected_operations":SELECTED_OPERATIONS,"indexes":indexes,"production_loader_changed":False,"algorithm_changed":False,
        "files_deleted":0,"files_moved":0,"claim_limit":packet["claim_limit"]}
    atomic_json(BUNDLE/"MANIFEST.json",manifest)
    atomic_json(ATTEMPT/"BUNDLE_CHECK.json",verify_bundle(REPO_ROOT,BUNDLE))
    print(json.dumps({"status":"BUNDLE_READY","counts":plan["counts"]}),flush=True)


def finish(packet,failure=None):
    protected=refs(tuple(packet["inputs"]["protected"]))
    if protected != packet["inputs"]["protected"]:
        failure="PROTECTED_SOURCE_DRIFT"
    outputs={}
    if failure is None:
        verify_bundle(REPO_ROOT,BUNDLE)
        if load_json(ATTEMPT/"REGRESSION.json")["status"]!="PASS":
            raise RuntimeError("REGRESSION_NOT_PASS")
        with tarfile.open(ATTEMPT/"SOURCE_RELEASE.tar.gz","x:gz") as tar:
            for path in sorted(BUNDLE.rglob("*")):
                if path.is_symlink():raise RuntimeError("BUNDLE_SYMLINK")
                if path.is_file():tar.add(path,arcname=str(path.relative_to(REPO_ROOT)),recursive=False)
        outputs={n:artifact_ref(ATTEMPT/n) for n in ("REGRESSION.json","BUNDLE_CHECK.json","SOURCE_RELEASE.tar.gz")}
        outputs["source_manifest"]=artifact_ref(BUNDLE/"MANIFEST.json")
    result={"schema_version":"BASELINE_V1_SOURCE_RELEASE_RESULT","task_id":TASK,"status":"PASSED" if failure is None else "FAILED_RUNTIME_FINAL",
        "failure_code":failure,"protected_before":packet["inputs"]["protected"],"protected_after":protected,
        "edit_before":packet["inputs"]["edit_before"],"edit_after":refs(EDITS),"outputs":outputs,
        "production_loader_changed":False,"algorithm_changed":False,"numeric_gates_changed":False,"baseline_pointers_changed":False,
        "files_deleted":0,"files_moved":0,"models_run":0,"claim_limit":packet["claim_limit"]}
    atomic_json(ATTEMPT/"RESULT.json",result)
    atomic_json(RECEIPT,{"schema_version":"BASELINE_V1_SOURCE_RELEASE_RECEIPT","status":result["status"],"task_id":TASK,
        "result":artifact_ref(ATTEMPT/"RESULT.json"),"source_release":artifact_ref(BUNDLE/"MANIFEST.json") if BUNDLE.joinpath("MANIFEST.json").is_file() else None})
    state,index=copy.deepcopy(load_json(TASK_STATE_PATH)),index_body()
    next(r for r in state["tasks"] if r["task_id"]==TASK).update(status=result["status"],pid=None,proc_start_ticks=None,
        updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(ATTEMPT/"RESULT.json"),terminal_receipt=artifact_ref(RECEIPT),quality="SOURCE_CONSOLIDATION_ONLY")
    state["next_task"]=None
    next(r for r in index["task_packets"] if r["task_id"]==TASK)["execution_allowed"]=False
    index.update(status="PASS_NO_ACTIVE_TASKS",supersedes_index=artifact_ref(INDEX))
    importlib.reload(importlib.import_module("chaoyang.governance.current_r3_contracts"))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="BASELINE_SOURCE_RELEASE_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),
        task_packet_index_path=INDEX,task_packet_index_value=index)
    atomic_json(ATTEMPT/"POST_PUBLICATION_VALIDATION.json",validate_current())
    print(json.dumps({"status":result["status"],"result":artifact_ref(ATTEMPT/"RESULT.json")}),flush=True)


def main():
    packet=register();print("READY",flush=True);deadline=time.monotonic()+WINDOW
    try:
        while time.monotonic()<deadline:
            ready,_,_=select.select([sys.stdin],[],[],30)
            if ready:
                command=sys.stdin.readline().strip()
                if command=="BUILD":build(packet)
                elif command=="PUBLISH":finish(packet);return
                else:raise RuntimeError("ABORTED")
            else:
                state=copy.deepcopy(load_json(TASK_STATE_PATH))
                next(r for r in state["tasks"] if r["task_id"]==TASK).update(heartbeat_at=now_iso(),updated_at=now_iso())
                publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="BASELINE_SOURCE_RELEASE_HEARTBEAT",
                    expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__))
        raise RuntimeError("DEADLINE")
    except BaseException as exc:
        if not ATTEMPT.joinpath("RESULT.json").exists():finish(packet,str(exc))
        raise


if __name__=="__main__":main()
