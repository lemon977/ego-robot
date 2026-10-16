"""One CPU-only successor fixing the stale snapshot publisher cache."""
import copy
from datetime import datetime, timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import select
import sys
import tarfile
import time

from chaoyang.governance.common import AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle
from chaoyang.governance.publish_exact78_runtime_hardening_v1 import validate_current, index_body, refs
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK="exact78_runtime_snapshot_repair_v1_20261008"
ATTEMPT=REPO_ROOT/f"_run/current/{TASK}/attempts/attempt_0001"
PACKET=REPO_ROOT/f"tasks/current/{TASK}/TASK_PACKET.json"
RECEIPT=REPO_ROOT/f"tasks/receipts/{TASK.upper()}.json"
INDEX=REPO_ROOT/"tasks/current/INDEX.json"
PLAN="docs/plans/EXACT78_RUNTIME_SNAPSHOT_REPAIR_20261008"
PARENT=REPO_ROOT/"_run/current/exact78_runtime_hardening_v1_20261008/attempts/attempt_0001"
EDITS=("src/chaoyang/governance/project_branch_navigation_v1.py","src/chaoyang/governance/current_r3_contracts.py","docs/current/EXACT78_RUNTIME_HARDENING_V1.json","docs/current/EXACT78_RUNTIME_HARDENING_ZH.md")

def register():
    validate_current()
    state,index=copy.deepcopy(load_json(TASK_STATE_PATH)),index_body()
    if state.get("next_task") or any(r.get("execution_allowed") for r in index["task_packets"]):raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(p.exists() for p in (ATTEMPT,PACKET,RECEIPT)):raise RuntimeError("FRESH_TASK_REQUIRED")
    protected=load_json(PARENT/"RESULT.json")["protected_after"]
    protected={**protected,**{str((PARENT/name).relative_to(REPO_ROOT)):artifact_ref(PARENT/name) for name in ("RESULT.json","REGRESSION.json","CODE_SNAPSHOT.tar.gz","CODE_SNAPSHOT_MANIFEST.json","RESTORE_CHECK_FINAL.json","TEST_OUTPUT.json","DEPENDENCY_CHECK.json")}}
    if refs(tuple(protected))!=protected:raise RuntimeError("PARENT_DRIFT")
    writer={"pid":os.getpid(),"proc_start_ticks":process_identity(os.getpid())["start_ticks"],"executor_epoch":1,"fencing_token":secrets.token_hex(24),"scope":TASK}
    inputs={"protected":protected,"edit_before":refs(EDITS),"publisher":artifact_ref(Path(__file__)),"plan":artifact_ref(REPO_ROOT/PLAN/"00_EXECUTION_ZH.md")}
    signature=hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet={"schema_version":"chaoyang-r22-task-packet-v1","task_id":TASK,"objective":"Repair incomplete immutable snapshot via a new reloaded publisher; retain parent evidence","phase":"CPU_SNAPSHOT_PUBLICATION_REPAIR","plan_revision":TASK.upper(),"execution_revision":TASK.upper(),"session_id":None,"deadline_at":(datetime.now().astimezone()+timedelta(seconds=960)).isoformat(),"read_set":["AGENTS.md","docs/current/AI_WORK_ENTRY_ZH.md",PLAN+"/00_EXECUTION_ZH.md",str((PARENT/"RESULT.json").relative_to(REPO_ROOT)),"docs/governance/DOC_AUTHORITY_MAP.json","docs/governance/ALGORITHM_CONTRACT.json","docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"],"write_set":[str(ATTEMPT.relative_to(REPO_ROOT)),str(PACKET.relative_to(REPO_ROOT)),str(RECEIPT.relative_to(REPO_ROOT)),"tasks/current/INDEX.json","docs/governance",PLAN,"_run/caches/exact78_runtime_snapshot_repair_20261008",*EDITS],"prerequisites":["GOVERNANCE_PASS","NO_ACTIVE_TASK","PARENT_IMMUTABLE"],"required_outputs":["RESULT.json","REGRESSION.json","CODE_SNAPSHOT.tar.gz","CODE_SNAPSHOT_MANIFEST.json"],"budgets":{"workers_max":1,"cpu_threads_total":2,"gpu_count":0,"training_steps":0,"model_calls_max":0,"wall_seconds_max":960,"session_attempts_max":1},"stop_conditions":["protected_sha_drift","double_writer","deadline","one_terminal_attempt"],"claim_limit":"Source publication repair only; no model, algorithm, baseline, numerical gate or training promotion.","writer":writer,"inputs":inputs,"run_signature":signature,"commands":[f"{sys.executable} -B -m chaoyang.governance.publish_exact78_runtime_snapshot_repair_v1"]}
    if errors:=_validate_packet(packet):raise RuntimeError(str(errors))
    ATTEMPT.mkdir(parents=True);PACKET.parent.mkdir(parents=True);atomic_json(ATTEMPT/"WRITER.json",writer);atomic_json(PACKET,packet)
    state["tasks"].append({"task_id":TASK,"phase":packet["phase"],"attempt":1,"status":"RUNNING","session":"source_snapshot_only","updated_at":now_iso(),"heartbeat_at":now_iso(),"pid":writer["pid"],"proc_start_ticks":writer["proc_start_ticks"],"gpu_id":None,"task_packet":artifact_ref(PACKET),"run_signature":signature,"quality":"NOT_ASSESSED","adoption":"BASELINES_UNCHANGED","review":"NOT_APPLICABLE"})
    state["next_task"]={"task_id":TASK,"session":"source_snapshot_only","prerequisites":packet["prerequisites"],"expected_resource":"CPU_TWO_THREADS","stop_condition":"one terminal publication"}
    index["task_packets"].append({"task_id":TASK,"packet_path":str(PACKET.relative_to(REPO_ROOT)),"packet_sha256":artifact_ref(PACKET)["sha256"],"execution_class":"CURRENT_LEDGER_ROUTABLE","execution_allowed":True})
    index.update(status="PASS",supersedes_index=artifact_ref(INDEX))
    publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="EXACT78_SNAPSHOT_REPAIR_REGISTERED",expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
    return packet

def finish(packet,failure=None):
    if refs(tuple(packet["inputs"]["protected"]))!=packet["inputs"]["protected"]:failure="PROTECTED_SHA_DRIFT"
    outputs={}
    if failure is None:
        if load_json(ATTEMPT/"REGRESSION.json")["status"]!="PASS":raise RuntimeError("REGRESSION_NOT_PASS")
        # Reload BEFORE obtaining the function; the previous publisher did so
        # only after it had already packed the old cached path selection.
        nav=importlib.reload(importlib.import_module("chaoyang.governance.project_branch_navigation_v1"))
        rows=[artifact_ref(p) for p in nav.snapshot_paths(REPO_ROOT)]
        atomic_json(ATTEMPT/"CODE_SNAPSHOT_MANIFEST.json",{"schema_version":"EXACT78_HARDENED_SOURCE_SNAPSHOT_V2","files":rows,"contains_models_videos_external_data":False,"git_head":load_json(PARENT/"CODE_SNAPSHOT_MANIFEST.json")["git_head"],"claim_limit":"Source/metadata overlay only, not environment/assets/media/full governance backup."})
        with tarfile.open(ATTEMPT/"CODE_SNAPSHOT.tar.gz","x:gz") as tar:
            for row in rows:
                path=Path(row["path"])
                if artifact_ref(path)!=row or path.is_symlink():raise RuntimeError("SNAPSHOT_DRIFT")
                tar.add(path,arcname=str(path.relative_to(REPO_ROOT)),recursive=False)
        outputs={n:artifact_ref(ATTEMPT/n) for n in ("REGRESSION.json","CODE_SNAPSHOT_MANIFEST.json","CODE_SNAPSHOT.tar.gz")}
    result={"schema_version":"EXACT78_RUNTIME_SNAPSHOT_REPAIR_RESULT_V1","task_id":TASK,"status":"PASSED" if failure is None else "FAILED_RUNTIME_FINAL","failure_code":failure,"parent_result":artifact_ref(PARENT/"RESULT.json"),"parent_snapshot_disposition":"INCOMPLETE_STALE_PUBLISHER_CACHE_RETAINED_NOT_RECOMMENDED","protected_before":packet["inputs"]["protected"],"protected_after":refs(tuple(packet["inputs"]["protected"])),"outputs":outputs,"models_run":0,"files_deleted":0,"numeric_gates_changed":False,"claim_limit":packet["claim_limit"]}
    atomic_json(ATTEMPT/"RESULT.json",result);atomic_json(RECEIPT,{"schema_version":"EXACT78_RUNTIME_SNAPSHOT_REPAIR_RECEIPT_V1","task_id":TASK,"status":result["status"],"result":artifact_ref(ATTEMPT/"RESULT.json"),"baseline_pointers_changed":False})
    state,index=copy.deepcopy(load_json(TASK_STATE_PATH)),index_body()
    row=next(r for r in state["tasks"] if r["task_id"]==TASK);row.update(status=result["status"],pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(ATTEMPT/"RESULT.json"),terminal_receipt=artifact_ref(RECEIPT),quality="SOURCE_RESTORE_ONLY")
    state["next_task"]=None;next(r for r in index["task_packets"] if r["task_id"]==TASK)["execution_allowed"]=False;index.update(status="PASS_NO_ACTIVE_TASKS",supersedes_index=artifact_ref(INDEX))
    importlib.reload(importlib.import_module("chaoyang.governance.current_r3_contracts"))
    published=publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="EXACT78_SNAPSHOT_REPAIR_TERMINAL",expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
    atomic_json(ATTEMPT/"POST_PUBLICATION_VALIDATION.json",validate_current());print(json.dumps({"status":result["status"],"revision":published["governance_revision"],"result":artifact_ref(ATTEMPT/"RESULT.json")}),flush=True)

def main():
    packet=register();print("READY",flush=True);deadline=time.monotonic()+900
    try:
        while time.monotonic()<deadline:
            ready,_,_=select.select([sys.stdin],[],[],30)
            if ready:
                if sys.stdin.readline().strip()!="PUBLISH":raise RuntimeError("ABORTED")
                finish(packet);return
            state=copy.deepcopy(load_json(TASK_STATE_PATH));next(r for r in state["tasks"] if r["task_id"]==TASK).update(heartbeat_at=now_iso(),updated_at=now_iso())
            publish_bundle(copy.deepcopy(load_json(AUTHORITY_PATH)),state,event_type="EXACT78_SNAPSHOT_REPAIR_HEARTBEAT",expected_revision=load_json(RECEIPT_PATH)["governance_revision"],generator_path=Path(__file__))
        raise RuntimeError("DEADLINE")
    except BaseException as exc:
        if not (ATTEMPT/"RESULT.json").exists():finish(packet,str(exc))
        raise

if __name__=="__main__":main()
