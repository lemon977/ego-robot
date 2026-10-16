"""Independent finite q-frozen cup-conflict Clean postprocess executor. Original files are immutable.

One spec file under ``docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/specs`` describes one
finite task.  ``--execute`` registers the packet through the governance CAS,
runs exactly one ``chaoyang run`` command as a fenced descendant, heartbeats,
and seals an immutable terminal result.  ``--recover`` seals a task whose writer
died (container restart, OOM, lost session) so that a successor can be routed.
"""

from __future__ import annotations

import argparse

import copy

import hashlib

import json

import os

from datetime import datetime, timedelta

from zoneinfo import ZoneInfo

from pathlib import Path

import secrets

import shlex

import signal

import subprocess

import time

from typing import Any

from chaoyang.governance.common_candidate_optimization_v6 import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)

from chaoyang.governance.register_single_task_packet import _validate_packet

SCRIPT=Path(__file__)
from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import TASK,OPERATION,FAMILY,BUDGETS
PLAN_REVISION='PICK_PLACE_1008_RAW_TO_ROBOT_V1'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'
PYTHON='/usr/local/bin/python'
HEARTBEAT_SECONDS=50
ACTIVE_STATUSES=('RUNNING','CLAIMED','WAIT_GPU_RESOURCE')
class WorkerNotStoppedError(RuntimeError):pass

def _paths(task_id: str) -> dict[str, Path]:
    output = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    return {
        "output": output,
        "stage": output / "candidate",
        "packet": REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json",
        "receipt": REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json",
    }

def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

def _project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise RuntimeError(f"PICK_PLACE_PATH_ESCAPES_PROJECT:{raw}")
    return resolved

def _input_ref(raw: str) -> dict[str, Any]:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return artifact_ref(path)

def _governance_pass() -> dict[str, Any]:
    result = subprocess.run(
        [PYTHON, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True, timeout=600, check=False)
    try:
        value = json.loads(result.stdout)
    except ValueError:
        value = {"status": "UNPARSEABLE", "stdout": result.stdout[-400:], "stderr": result.stderr[-400:]}
    return value

def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}

def register(spec: dict[str, Any], packet: dict[str, Any]) -> None:
    task_id = packet["task_id"]
    paths = _paths(task_id)
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": spec.get("session"),
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": spec.get("gpu_id"), "task_packet": _virtual_ref(paths["packet"], packet),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": task_id, "session": spec.get("session"),
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": f"GPU_{packet['budgets'].get('gpu_count', 0)}",
                          "stop_condition": "one attempt or hard deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("PICK_PLACE_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    index["task_packets"].append({
        "task_id": task_id, "packet_path": str(paths["packet"].relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(paths["packet"], packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=f"{task_id.upper()}_RUNNING", plan_revision=PLAN_REVISION,
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    paths["output"].mkdir(parents=True)
    atomic_json(paths["output"] / "WRITER.json", packet["writer"])
    paths["packet"].parent.mkdir(parents=True)
    atomic_json(paths["packet"], packet)
    if artifact_ref(paths["packet"])["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("PICK_PLACE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="PICK_PLACE_TASK_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)

def verify_fence(packet: dict[str, Any]) -> None:
    task_id = packet["task_id"]
    writer = packet["writer"]
    paths = _paths(task_id)
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(paths["output"] / "WRITER.json") != writer):
        raise RuntimeError("PICK_PLACE_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"] if row["status"] in ACTIVE_STATUSES]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != task_id
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != task_id
            or routed[0]["packet_sha256"] != artifact_ref(paths["packet"])["sha256"]):
        raise RuntimeError("PICK_PLACE_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")

def heartbeat(packet: dict[str, Any]) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="PICK_PLACE_TASK_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)

def _stop_fenced(worker: subprocess.Popen[Any], ticks: int) -> None:
    if worker.poll() is not None:
        return
    try:
        if process_identity(worker.pid)["start_ticks"] != ticks:
            raise WorkerNotStoppedError("PICK_PLACE_WORKER_IDENTITY_LOST")
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if process_identity(worker.pid)["start_ticks"] != ticks:
                raise WorkerNotStoppedError("PICK_PLACE_WORKER_IDENTITY_LOST")
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=10)
    except (ProcessLookupError, ChildProcessError, OSError, KeyError,
            ValueError, subprocess.TimeoutExpired):
        if worker.poll() is None:
            raise WorkerNotStoppedError("PICK_PLACE_WORKER_STOP_UNPROVEN")

def _packet_command_tokens(packet: dict[str, Any]) -> list[str]:
    tokens = shlex.split(packet["commands"][0])
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens.pop(0)
    return tokens

def _write_worker_identity(packet: dict[str, Any], worker_pid: int,
                           ticks: int, command: list[str]) -> None:
    path = _paths(packet["task_id"])["output"] / "WORKER.json"
    if path.exists():
        raise RuntimeError("PICK_PLACE_WORKER_IDENTITY_ALREADY_EXISTS")
    atomic_json(path, {
        "schema_version": "pick-place-fenced-worker-v1", "task_id": packet["task_id"],
        "pid": worker_pid, "proc_start_ticks": ticks, "command": command,
        "packet_command": packet["commands"][0], "run_signature": packet["run_signature"],
        "writer": packet["writer"], "recorded_at": now_iso(),
    })

def _verify_recovery_worker(packet: dict[str, Any], paths: dict[str, Path]
                            ) -> dict[str, Any] | None:
    path = paths["output"] / "WORKER.json"
    if not path.is_file():
        if (paths["output"] / "worker.log").exists():
            raise RuntimeError("PICK_PLACE_RECOVER_WORKER_IDENTITY_UNPROVEN")
        return None
    worker = load_json(path)
    if (worker.get("task_id") != packet["task_id"]
            or worker.get("run_signature") != packet["run_signature"]
            or worker.get("command") != _packet_command_tokens(packet)
            or worker.get("packet_command") != packet["commands"][0]
            or worker.get("writer") != packet["writer"]
            or type(worker.get("pid")) is not int or worker["pid"] <= 0
            or type(worker.get("proc_start_ticks")) is not int or worker["proc_start_ticks"] < 0):
        raise RuntimeError("PICK_PLACE_RECOVER_WORKER_IDENTITY_DRIFT")
    identity = process_identity(worker["pid"])
    if identity["alive"] and identity["start_ticks"] == worker["proc_start_ticks"]:
        raise RuntimeError("PICK_PLACE_RECOVER_WORKER_ALIVE")
    return {"worker": artifact_ref(path), "recorded_identity": worker,
            "current_identity": identity, "original_worker_stopped": True}

def _seal(packet: dict[str, Any], status: str, payload: dict[str, Any], event: str) -> dict[str, Any]:
    task_id = packet["task_id"]
    paths = _paths(task_id)
    if (paths["output"] / "RESULT.json").exists() or paths["receipt"].exists():
        raise RuntimeError("PICK_PLACE_DUPLICATE_TERMINAL")
    result = {"schema_version": "pick-place-1008-execution-result-v1", "task_id": task_id,
              "status": status, "task_packet": artifact_ref(paths["packet"]),
              "run_signature": packet["run_signature"], "visual_qualified": False,
              "training_eligible": False, "control_ground_truth": False,
              "adoption": "NOT_ADOPTED", "user_review": "PENDING",
              "files_deleted": False, **payload}
    atomic_json(paths["output"] / "RESULT.json", result)
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
    atomic_json(paths["receipt"], {"schema_version": "pick-place-1008-receipt-v1",
                                   "task_id": task_id, "status": status,
                                   "result": artifact_ref(paths["output"] / "RESULT.json"),
                                   "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == task_id)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(paths["output"] / "RESULT.json"),
               terminal_receipt=artifact_ref(paths["receipt"]),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if status == "PASSED" else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    if (state.get("next_task") or {}).get("task_id") == task_id:
        state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("PICK_PLACE_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == task_id:
            item["execution_allowed"] = False
            item["result"] = artifact_ref(paths["output"] / "RESULT.json")
            item["terminal_receipt"] = artifact_ref(paths["receipt"])
            item["required_outputs"] = copy.deepcopy(payload.get("required_outputs", {}))
    index.update(packet_revision=f"{task_id.upper()}_TERMINAL", plan_revision=PLAN_REVISION,
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=SCRIPT, task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(paths["output"] / "RESULT.json")}

def packet_from_context(ctx,preparation=False):
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import expected_runtime,plan_shape,command,signature
 spec=ctx['spec'];plan=ctx['plan'];plan_shape(plan,spec,verify=not preparation)
 if not preparation and plan['publication_status']!='ROOT_PLANNED_FINITE_OPERATION_APPROVED':raise ValueError('CUP_ROOT_APPROVAL_PENDING')
 runtime=expected_runtime(ctx,verify=not preparation);paths=_paths(TASK);now=datetime.now(ZoneInfo('Asia/Shanghai'))
 writer=dict(pid=os.getpid(),proc_start_ticks=process_identity(os.getpid())['start_ticks'],executor_epoch=1,fencing_token=secrets.token_hex(24),scope=TASK)
 inputs=dict(spec=plan['approved_spec'],files=[spec['contract']],maintenance_render_plan=ctx['plan_ref'])
 config=dict(session='ORIGINAL_DEV4_Q_FROZEN_VISUAL_CLEAN_POSTPROCESS',budgets=dict(BUDGETS),deletion_authorized=False,numeric_gates_changed=False,gpu_id=0,gpu_min_free_mib=0)
 cmd=command(plan['approved_spec'])
 readset=[plan['approved_spec']['path'],ctx['plan_ref']['path'],spec['contract']['path'],ctx['contract']['original_queue']['path']]
 for p in ctx['contract']['original_producers']:readset.extend(p[k]['path'] for k in ['index','packet'])
 if len(set(readset))>8:raise ValueError('CUP_INITIAL_READSET_MAX8')
 packet=dict(schema_version='chaoyang-r22-task-packet-v1',task_id=TASK,objective='Finite original DEV4/760 q-frozen model0 Raw restore in observed-cup semantic conflict; new standalone visual Clean postprocess stage, old inputs immutable',phase='pick_place_observed_cup_restore_clean_postprocess',plan_revision=PLAN_REVISION,execution_revision=TASK.upper(),execution_mode='EXACT78_CANDIDATE',registered_at=now.isoformat(),deadline_at=(now+timedelta(seconds=1200)).isoformat(),read_set=sorted(set(readset)),write_set=[str(paths['output'].relative_to(REPO_ROOT)),str(paths['packet'].relative_to(REPO_ROOT)),str(paths['receipt'].relative_to(REPO_ROOT)),'tasks/current/INDEX.json','docs/governance'],prerequisites=['GOVERNANCE_PASS','ROOT_EXPLICIT_KNOWN7_FAMILY_DESCRIPTOR','NO_ACTIVE_TASK_PACKET_NEXT','SIGNED_PREDECESSOR_ACTORS_DEAD','STRICT_RELEASED_LEASE','ORIGINAL_SOURCE_AND_DEV_GLOBAL_PASSED_FRESH_SOURCE_MOTION_GPU_BINDINGS','OLD93_PLUS_EIGHT_PLUS_SHARED3','NO_Q_MODEL_FK_RENDER_CALLS'],required_outputs=['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],budgets=dict(BUDGETS),stop_conditions=['ONE_ATTEMPT_NO_RETRY','900S_TOTAL_WALL','IDENTITY_OR_INPUT_SHA_DRIFT','N1_STREAMING_256MIB_RSS','NO_OLD_OUTPUT_WRITE','NO_NEW_RENDER_CONTACT_OR_AI'],claim_limit='Derived visual Clean/P/E only; no q/source/CAM/seed changes; no fresh Motion/Render/Contact or quality/control/training/physical truth.',writer=writer,inputs=inputs,runtime_code=runtime,config=config,run_signature=signature(inputs,runtime,config,cmd),commands=[cmd])
 errors=_validate_packet(packet)
 if errors:raise ValueError('CUP_ACTUAL_R22_PACKET_SCHEMA:'+str(errors))
 return packet
def actors_dead(queue):
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import pin,read,ROOT,need,path
 rows=[queue['source']]+queue['gates']+queue['records'];need(len(rows)==23,'CUP_ORIGINAL23_ACTOR_INVENTORY')
 for row in rows:
  attempt=ROOT/'_run/current'/row['task_id']/'attempts/attempt_0001'
  writer=load_json(attempt/'WRITER.json');worker=load_json(attempt/'WORKER.json')
  need(type(writer['pid']) is int and type(writer['proc_start_ticks']) is int and writer['scope']==row['task_id'],'CUP_ORIGINAL_ACTOR_TYPED_WRITER')
  need(type(worker['pid']) is int and type(worker['proc_start_ticks']) is int and worker['task_id']==row['task_id'] and worker['writer']==writer,'CUP_ORIGINAL_ACTOR_SIGNED_WORKER')
  for actor in [writer,worker]:
   actual=process_identity(actor['pid'])
   need(not(actual['alive'] and actual['start_ticks']==actor['proc_start_ticks']),'CUP_ORIGINAL_ACTOR_STILL_LIVE')
 return True
def readiness(ctx):
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import verify_baselines,read
 state=load_json(TASK_STATE_PATH);index=load_json(INDEX)
 if state.get('next_task') is not None or any(r['status'] in ACTIVE_STATUSES for r in state['tasks']):raise RuntimeError('CUP_ACTIVE_TASK_OR_NEXT')
 if any(r.get('execution_allowed') for r in index['task_packets']):raise RuntimeError('CUP_ACTIVE_PACKET')
 if load_json(REPO_ROOT/'_run/current/GPU_LEASE.json').get('status')!='RELEASED':raise RuntimeError('CUP_STRICT_RELEASED_LEASE_REQUIRED')
 if _governance_pass().get('status')!='PASS':raise RuntimeError('CUP_GOVERNANCE_NOT_PASS')
 verify_baselines(ctx['contract']);actors_dead(read(ctx['contract']['original_queue']))
 p=_paths(TASK)
 if any(p[k].exists() for k in ['output','packet','receipt']):raise RuntimeError('CUP_FRESH_TASK_ONE_ATTEMPT_ONLY')
 if (REPO_ROOT/'_run/current'/f'{TASK}.EXECUTION_INTENT.json').exists():raise RuntimeError('CUP_EXISTING_EXCLUSIVE_INTENT')
def run_worker(packet):
 verify_fence(packet);paths=_paths(TASK);cache=paths['output']/'cache';cache.mkdir()
 env=os.environ.copy();env.update(PYTHONPATH=str(REPO_ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',TMPDIR=str(cache),TMP=str(cache),TEMP=str(cache),XDG_CACHE_HOME=str(cache),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 log=paths['output']/'worker.log';start=time.monotonic()
 with log.open('x') as stream:
  worker=subprocess.Popen(_packet_command_tokens(packet),cwd=REPO_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ticks=process_identity(worker.pid)['start_ticks'];_write_worker_identity(packet,worker.pid,ticks,_packet_command_tokens(packet))
  except (OSError,KeyError,ValueError) as exc:
   if worker.poll() is None:raise WorkerNotStoppedError('CUP_WORKER_START_IDENTITY_LOST') from exc
   ticks=-1
  last=0.
  try:
   while worker.poll() is None:
    elapsed=time.monotonic()-start
    if elapsed>=900:raise RuntimeError('CUP_HARD_WALL900')
    if elapsed-last>=HEARTBEAT_SECONDS:heartbeat(packet);last=elapsed
    time.sleep(.25)
  finally:_stop_fenced(worker,ticks)
 return worker.returncode,log
def _collect_required_outputs(packet,paths):
 result={};errors=[]
 for name in packet['required_outputs']:
  if not name.startswith('candidate/'):continue
  p=paths['output']/name
  if p.is_file():result[name]=artifact_ref(p)
  else:errors.append('CUP_REQUIRED_OUTPUT_MISSING:'+name)
 return result,errors
def finish(packet,code,log,elapsed,ctx,reason=None):
 verify_fence(packet);paths=_paths(TASK);outputs,errors=_collect_required_outputs(packet,paths)
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import validate_packet_context,expected_runtime
 validate_packet_context(packet,ctx)
 for v in packet['runtime_code']+[packet['inputs']['spec'],packet['inputs']['maintenance_render_plan'],*packet['inputs']['files']]:errors.extend(validate_artifact_ref(v))
 validation=None;collector_seconds=0.
 if type(code) is not int:errors.append('CUP_WORKER_RC_INTEGER_REQUIRED')
 if code==0 and not errors and reason is None:
  start=time.monotonic()
  try:
   from chaoyang.pick_place_observed_cup_restore_clean_v1.collector import collect
   validation=collect(packet,ctx,lambda: (_ for _ in ()).throw(RuntimeError('CUP_COLLECT_TOTAL_WALL900')) if time.monotonic()-start+elapsed>900 else None)
   atomic_json(paths['output']/'INDEPENDENT_CUP_COLLECTOR_VALIDATION.json',validation)
  except Exception as exc:errors.append('CUP_COLLECTOR_FAILED:'+type(exc).__name__+':'+str(exc))
  collector_seconds=time.monotonic()-start
 total=elapsed+collector_seconds
 if total>900:errors.append('CUP_TOTAL_WALL900_EXCEEDED')
 status='PASSED' if type(code) is int and code==0 and not errors and validation is not None and reason is None else 'FAILED_RUNTIME_FINAL'
 payload=dict(worker_returncode=code,failure_code=reason,validation_errors=errors,elapsed_seconds=total,worker_guard_wall_seconds=elapsed,collector_wall_seconds=collector_seconds,required_outputs=outputs,algorithm_attempts_consumed=1,resource_wait=False,stage_result=artifact_ref(paths['stage']/'RESULT.json') if (paths['stage']/'RESULT.json').is_file() else None,worker_log=artifact_ref(log),independent_collector=artifact_ref(paths['output']/'INDEPENDENT_CUP_COLLECTOR_VALIDATION.json') if validation is not None else None,next_action='Root reviews new visual Clean stage; fresh Render and Contact require separate registered Task; no auto adoption or retry')
 return _seal(packet,status,payload,'CONTACT_CUP_EVIDENCE_'+status)
def publish_route(descriptor_path):
 from chaoyang.governance.candidate_optimization_family_v5 import descriptor
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import FAMILY,need,ref
 need(SCRIPT.resolve()==REPO_ROOT/'src/chaoyang/governance/execute_pick_place_observed_cup_restore_clean_v1.py','CUP_CANONICAL_INSTALLED_EXECUTOR')
 dp=ref(descriptor_path);d=descriptor(dp,REPO_ROOT)
 need(d['family_key']==FAMILY,'CUP_ONLY_THIS_EXPLICIT_FAMILY')
 value=publish_bundle(load_json(AUTHORITY_PATH),copy.deepcopy(load_json(TASK_STATE_PATH)),event_type='ROOT_CUP_RESTORE_CLEAN_FAMILY_PUBLISHED',expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=SCRIPT,planned_family_descriptor_ref=dp)
 after=_governance_pass();return dict(status=after['status'],errors_after=after.get('errors',[]),governance_revision=value['governance_revision'])
def preparation_check(spec_path,plan_path):
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import ref,read,plan_shape,need
 spec=load_json(spec_path);plan=load_json(plan_path)
 need(plan['publication_status']=='DRAFT_ROOT_APPROVAL_REQUIRED_NOT_AUTHORIZED','CUP_PREPARATION_DRAFT_ONLY')
 plan_shape(plan,spec,verify=False)
 ctx=dict(spec=spec,plan=plan,plan_ref=artifact_ref(plan_path),contract=read(spec['contract']))
 packet=packet_from_context(ctx,True)
 return packet,dict(status='PRIVATE_SCHEMA_PASS_NOT_FORMAL_READY',formal_READY=False,budgets=BUDGETS,runtime_code_files=len(packet['runtime_code']),actual_R22_errors=_validate_packet(packet),mask_NPZ_opened=False,model_FK_Render_called=False,register_called=False,publish_called=False)
def recover(task_id):
 if task_id!=TASK:raise ValueError('CUP_RECOVER_FIXED_TASK_ONLY')
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import approved_context,validate_packet_context
 paths=_paths(TASK);packet=load_json(paths['packet']);ctx=approved_context(packet['inputs']['spec']['path']);validate_packet_context(packet,ctx)
 state=load_json(TASK_STATE_PATH);row=next((r for r in state['tasks'] if r['task_id']==TASK),None)
 if row is None or row['status'] not in ACTIVE_STATUSES:raise RuntimeError('CUP_RECOVER_NOT_ACTIVE')
 writer=packet['writer'];identity=process_identity(writer['pid'])
 if identity['alive'] and identity['start_ticks']==writer['proc_start_ticks']:raise RuntimeError('CUP_RECOVER_WRITER_ALIVE')
 if row['run_signature']!=packet['run_signature'] or row['task_packet']!=artifact_ref(paths['packet']) or load_json(paths['output']/'WRITER.json')!=writer:raise RuntimeError('CUP_RECOVER_PACKET_FENCE_DRIFT')
 evidence=_verify_recovery_worker(packet,paths)
 return _seal(packet,'FAILED_RUNTIME_FINAL',dict(worker_returncode=None,failure_code='INTERRUPTED_WRITER_LOST',validation_errors=['INTERRUPTED_WORKER_RC_UNKNOWN'],algorithm_attempts_consumed=1,resource_wait=False,worker_evidence=evidence,partial_outputs_are_successor_inputs=False,next_action='Root inspect true interruption; never reuse partial or auto retry'),'CONTACT_CUP_INTERRUPTED')
def main(argv=None):
 p=argparse.ArgumentParser(description='Only the Root-approved finite original DEV4 Clean postprocess')
 g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check-only',type=Path);g.add_argument('--execute',type=Path);g.add_argument('--publish-route',type=Path);g.add_argument('--recover');g.add_argument('--preparation-check',type=Path);p.add_argument('--draft-plan',type=Path)
 a=p.parse_args(argv)
 if a.preparation_check:
  if not a.draft_plan:p.error('--draft-plan required with --preparation-check')
  _,v=preparation_check(a.preparation_check,a.draft_plan);print(json.dumps(v));return 0
 if a.draft_plan:p.error('--draft-plan forbidden outside preparation')
 if a.execute or a.recover or a.publish_route:
  if SCRIPT.resolve()!=REPO_ROOT/'src/chaoyang/governance/execute_pick_place_observed_cup_restore_clean_v1.py':raise ValueError('CUP_CANONICAL_MUTATING_EXECUTOR_REQUIRED')
 if a.publish_route:v=publish_route(a.publish_route);print(json.dumps(v));return 0 if v['status']=='PASS' else 1
 if a.recover:print(json.dumps(recover(a.recover)));return 0
 from chaoyang.pick_place_observed_cup_restore_clean_v1.contract import approved_context
 try:
  ctx=approved_context(a.check_only or a.execute);packet=packet_from_context(ctx);readiness(ctx)
 except (ValueError,RuntimeError,FileNotFoundError) as exc:
  if not a.check_only:raise
  print(json.dumps(dict(status='DRY_RUN_NOT_AUTHORIZED_OR_NOT_ROUTABLE',formal_READY=False,reason=str(exc),register_called=False)));return 0
 if a.check_only:
  print(json.dumps(dict(status='DRY_RUN_READY',formal_READY=True,task_id=TASK,run_signature=packet['run_signature'],runtime_code_files=len(packet['runtime_code']),budgets=BUDGETS)));return 0
 readiness(ctx)
 intent=REPO_ROOT/'_run/current'/f'{TASK}.EXECUTION_INTENT.json'
 with intent.open('x') as f:json.dump(dict(task_id=TASK,pid=os.getpid(),proc_start_ticks=process_identity(os.getpid())['start_ticks'],spec=packet['inputs']['spec'],run_signature=packet['run_signature'],single_attempt=True,automatic_retry=False),f)
 register(ctx['spec'],packet);start=time.monotonic()
 try:code,log=run_worker(packet);reason=None
 except Exception as exc:
  if isinstance(exc,WorkerNotStoppedError):raise
  code=1;reason=type(exc).__name__+':'+str(exc);log=_paths(TASK)['output']/'executor_error.log'
  with log.open('x') as f:f.write(reason+'\n')
 v=finish(packet,code,log,time.monotonic()-start,ctx,reason);print(json.dumps(v));return 0 if v['status']=='PASSED' else 1
if __name__=='__main__':raise SystemExit(main())
