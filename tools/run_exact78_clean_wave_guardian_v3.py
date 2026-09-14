#!/usr/bin/env python3
"""Recoverable, fail-closed guardian for frozen exact78 Clean Wave-0.

The first pending Chips session is a mandatory canary.  Only a Grade-B Clean
terminal with all hard gates passing permits the remaining 53 jobs.  Each GPU
job is serial and uses the existing central lease launcher.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROJECT=Path(__file__).resolve().parents[1]
LEASE=PROJECT/"_run/GPU_LEASE.json"
PREP=PROJECT/"tools/prepare_exact78_clean_wave_session_v3.py"
DONOR=PROJECT/"tools/run_generic_same_session_real_donor_v1.py"
VALIDATOR=PROJECT/"tools/validate_generic_same_session_real_donor_v1.py"
LAUNCHER=PROJECT/"tools/launch_clean_synthetic_propainter_once.py"
EXECUTION_AUTHORITY_NAME="EXECUTION_AUTHORITY_V3.json"


class RuntimeAttemptError(RuntimeError):
    pass


class QualityGateError(RuntimeError):
    pass

def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(p:Path):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def ref(p:Path):
 p=p.resolve(strict=True);return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def load(p:Path):return json.loads(p.read_text(encoding="utf-8"))
def atomic(p:Path,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".tmp",dir=p.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def write_new(p:Path,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def gpu():
 q=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
 return {"used_mib":int(q[0].strip()),"utilization_percent":int(q[1].strip())}
def conflicts():
 tokens=("train_usb_tict_ddp.py","ProPainter/inference_propainter.py","FoundationStereo","run_clean_synthetic_propainter_baseline.py")
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
  except OSError:continue
  if c and any(x in c for x in tokens):out.append({"pid":int(p.name),"cmd":c})
 return sorted(out,key=lambda x:x['pid'])
def runtime_gate():
 lease=load(LEASE);g=gpu();c=conflicts()
 return {"pass":lease.get("status")=="RELEASED" and g["used_mib"]<=8192 and g["utilization_percent"]<=10 and not c,
         "lease":lease,"gpu":g,"conflicts":c}


def wait_runtime_gate(root:Path,counts:dict,sid:str,deadline:float):
 while True:
  gate=runtime_gate()
  if gate['pass']:return gate
  state(root,'WAIT_GPU_RESOURCE_BETWEEN_ATTEMPTS',counts,sid)
  governance_heartbeat('CLAIMED','wait_gpu_resource_between_attempts',sid,None)
  if time.monotonic()>=deadline:
   raise RuntimeAttemptError(f"{sid}: GPU resource wait exceeded 1800 seconds")
  time.sleep(min(30,max(1,deadline-time.monotonic())))
def run(command:list[str],log:Path,heartbeat:tuple[str,str,str|None,int|None]|None=None):
 started=now();log.parent.mkdir(parents=True,exist_ok=True)
 with log.open("x",encoding="utf-8") as f:
  p=subprocess.Popen(command,text=True,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if heartbeat:governance_heartbeat(*heartbeat)
   time.sleep(25)
  f.flush();os.fsync(f.fileno())
 if p.returncode:raise RuntimeAttemptError(f"command rc={p.returncode}; log={log}; command={command}")
 return {"started_at":started,"finished_at":now(),"returncode":p.returncode,"log":ref(log)}
def validate_clean(path:Path,row:dict):
 d=load(path)
 if d.get("status")!="PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B" or d.get("grade")!="B" or d.get("downstream_authorized") is not True:raise QualityGateError(f"{row['session_id']}: Clean grade/authority gate")
 if d.get("session")!=row['session_id'] or d.get("task")!=row['task'] or d.get("frame_count")!=row['frame_count']:raise QualityGateError(f"{row['session_id']}: Clean identity gate")
 if any(v!='PASS' for v in d.get('hard_gates',{}).values()):raise QualityGateError(f"{row['session_id']}: Clean hard gate")
 for x in d.get('artifacts',{}).values():
  if isinstance(x,dict) and {'path','bytes','sha256'}<=set(x):
   if ref(Path(x['path']))!=x:raise QualityGateError(f"{row['session_id']}: Clean artifact SHA gate")
 return ref(path)


def quarantine_failed_output(root:Path,sid:str,attempt_root:Path,label:str):
 output=root/'propainter_v1'/sid
 if not output.exists() and not output.is_symlink():return None
 destination=attempt_root/label
 if destination.exists() or destination.is_symlink():raise RuntimeError(f"attempt quarantine collision: {destination}")
 os.rename(output,destination)
 return str(destination)


def run_propainter_with_retries(root:Path,row:dict,counts:dict):
 sid=row['session_id'];session_root=root/'sessions'/sid;attempts=session_root/'attempts';attempts.mkdir(parents=True,exist_ok=True)
 gpu_wait_deadline=time.monotonic()+1800
 final_ref_path=session_root/'final'/'RESULT_REF.json';canonical_result=root/'propainter_v1'/sid/'RESULT.json'
 if final_ref_path.exists():
  final_ref=load(final_ref_path);validated=validate_clean(Path(final_ref['result']['path']),row)
  if validated!=final_ref['result']:raise RuntimeError(f"{sid}: final result SHA conflict")
  return {'status':'PASSED','result':validated,'attempt':final_ref['attempt'],'skipped_existing_final':True}
 for attempt_index in range(1,4):
  attempt_root=attempts/f'attempt_{attempt_index:04d}'
  if attempt_root.exists():continue
  attempt_root.mkdir()
  attempt_meta={'schema_version':'exact78-clean-wave-attempt-v3','created_at':now(),'updated_at':now(),'status':'CLAIMED','attempt':attempt_index,
                'task':row['task'],'session':sid,'input_spec':ref(root/'specs'/f'{sid}_propainter.json'),'max_runtime_attempts':3,
                'claim_limit':'One runtime attempt only; no authority until final/RESULT_REF.json is atomically published.'}
  atomic(attempt_root/'ATTEMPT.json',attempt_meta)
  try:
   if canonical_result.parent.exists():raise RuntimeAttemptError(f"{sid}: unclaimed canonical output exists before attempt")
   wait_runtime_gate(root,counts,sid,gpu_wait_deadline)
   receipt=attempt_root/'GPU_LAUNCH_RECEIPT.json';state(root,'RUNNING_PROPAINTER_GPU',counts,sid)
   run([sys.executable,str(LAUNCHER),'--spec',str(root/'specs'/f'{sid}_propainter.json'),'--holder',f'exact78-clean-wave0-v3-{sid}-attempt{attempt_index}',
        '--receipt',str(receipt),'--wall-seconds','7200'],attempt_root/'gpu_launcher.log',('RUNNING','propainter_gpu',sid,0))
   validated=validate_clean(canonical_result,row)
  except QualityGateError as error:
   moved=quarantine_failed_output(root,sid,attempt_root,'quality_c_output')
   attempt_meta.update(updated_at=now(),status='FAILED_QUALITY_C',error=f'{type(error).__name__}: {error}',quarantined_output=moved)
   atomic(attempt_root/'ATTEMPT.json',attempt_meta)
   return {'status':'FAILED_QUALITY_C','attempt':attempt_index,'error':str(error),'attempt_ref':ref(attempt_root/'ATTEMPT.json')}
  except (RuntimeAttemptError,OSError,subprocess.SubprocessError) as error:
   moved=quarantine_failed_output(root,sid,attempt_root,'runtime_failed_output')
   terminal=attempt_index==3
   attempt_meta.update(updated_at=now(),status='FAILED_RUNTIME_FINAL' if terminal else 'FAILED_RUNTIME_RETRYABLE',
                       error=f'{type(error).__name__}: {error}',quarantined_output=moved)
   atomic(attempt_root/'ATTEMPT.json',attempt_meta)
   event(root,{'status':attempt_meta['status'],'session':sid,'attempt':attempt_index,'attempt_result':ref(attempt_root/'ATTEMPT.json')})
   if terminal:raise RuntimeAttemptError(f"{sid}: runtime retry budget exhausted") from error
   continue
  attempt_meta.update(updated_at=now(),status='PASSED',result=validated)
  atomic(attempt_root/'ATTEMPT.json',attempt_meta)
  final_root=session_root/'final';final_root.mkdir(exist_ok=True)
  write_new(final_ref_path,{'schema_version':'exact78-clean-wave-final-ref-v3','created_at':now(),'status':'PASSED','task':row['task'],'session':sid,
                            'attempt':attempt_index,'result':validated,'attempt_result':ref(attempt_root/'ATTEMPT.json'),
                            'claim_limit':'Pointer to immutable validated Clean terminal; authority remains bounded by the Wave terminal audit.'})
  return {'status':'PASSED','result':validated,'attempt':attempt_index,'final_ref':ref(final_ref_path)}
 raise RuntimeAttemptError(f"{sid}: all attempt directories already exist without a valid final")
def governance_heartbeat(status:str,phase:str,session:str|None,gpu_id:int|None=None):
 command=[sys.executable,'-m','tools.governance.heartbeat_task','--task-id','exact78_wave0_clean_v1','--pid',str(os.getpid()),'--status',status,'--phase',phase]
 if session:command.extend(['--session',session])
 if gpu_id is not None:command.extend(['--gpu-id',str(gpu_id)])
 p=subprocess.run(command,cwd=PROJECT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f"governance heartbeat failed rc={p.returncode}: {p.stdout[-2000:]}")
def state(root:Path,status:str,counts:dict,current=None,error=None,preflight=None):
 payload={"schema_version":"exact78-clean-wave-state-v3","updated_at":now(),"status":status,"pid":os.getpid(),"current_session":current,
          "counts":counts,"selection":ref(root/'EXACT78_WAVE0_SELECTION.json'),"preflight":ref(preflight) if preflight else None,"error":error}
 atomic(root/'STATE.json',payload);atomic(root/'HEARTBEAT.json',payload)
 if status=='RUNNING_PROPAINTER_GPU':governance_heartbeat('RUNNING','propainter_gpu',current,0)
 elif status.startswith('RUNNING_'):governance_heartbeat('CLAIMED',status.lower(),current)
def event(root:Path,v:dict):
 p=root/'EVENTS.jsonl';v={"at":now(),**v}
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())


def validate_execution_authority(root:Path,pf:dict):
 authority_path=root/EXECUTION_AUTHORITY_NAME;authority=load(authority_path);authority_ref=ref(authority_path)
 if pf.get('execution_authority')!=authority_ref or authority.get('status')!='FROZEN_RECOVERABLE_WAIT_GPU':raise RuntimeError('preflight/execution authority mismatch')
 for name,expected in authority.get('scripts',{}).items():
  if ref(PROJECT/'tools'/name)!=expected:raise RuntimeError(f'execution script changed after authority freeze: {name}')
 return authority_ref

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan-root',type=Path,required=True);ap.add_argument('--preflight',type=Path,required=True);ap.add_argument('--canary-only',action='store_true');a=ap.parse_args()
 root=a.plan_root.resolve(strict=True);preflight=a.preflight.resolve(strict=True);pf=load(preflight)
 if pf.get('status')!='PASS_READY_TO_START_GUARDIAN' or pf.get('launch_authorized') is not True:raise RuntimeError('passed fresh preflight required')
 selection=load(root/'EXACT78_WAVE0_SELECTION.json')
 if pf.get('selection')!=ref(root/'EXACT78_WAVE0_SELECTION.json'):raise RuntimeError('preflight/selection ref mismatch')
 validate_execution_authority(root,pf)
 attempt=f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
 pid_path=root/'guardian_attempts'/f'{attempt}.json'
 write_new(pid_path,{"schema_version":"exact78-clean-wave-guardian-attempt-v3","started_at":now(),"status":"RUNNING","pid":os.getpid(),"preflight":ref(preflight),"guardian":ref(Path(__file__))})
 if not (root/'GUARDIAN_PID.json').exists():write_new(root/'GUARDIAN_PID.json',load(pid_path))
 rows=[x for x in selection['sessions'] if x['existing_clean'] is None]
 by={x['session_id']:x for x in rows};canary=selection['canary'];ordered=[by[canary],*[x for x in rows if x['session_id']!=canary]]
 counts={"selected":58,"existing":4,"pending":54,"completed_new":0,"failed":0}
 # Recover completed immutable terminals, but never accept a partial directory.
 completed=[]
 for row in ordered:
  rp=root/'propainter_v1'/row['session_id']/'RESULT.json'
  if rp.exists():validate_clean(rp,row);completed.append(row['session_id'])
 counts['completed_new']=len(completed)
 try:
  for ordinal,row in enumerate(ordered):
   sid=row['session_id'];result=root/'propainter_v1'/sid/'RESULT.json'
   if sid in completed:continue
   state(root,'RUNNING_CPU_PREPARE',counts,sid,preflight=preflight);event(root,{"status":"START_SESSION","session":sid})
   prep_receipt=root/'preparation_receipts'/f'{sid}.json'
   if not prep_receipt.exists():run([sys.executable,str(PREP),'--plan-root',str(root),'--selection',str(root/'EXACT78_WAVE0_SELECTION.json'),'--session',sid],root/'logs'/f'{sid}_prepare.log',('CLAIMED','running_cpu_prepare',sid,None))
   donor_result=root/'real_donor_v1'/sid/'RESULT.json';donor_spec=root/'specs'/f'{sid}_real_donor_input.json'
   if not donor_result.exists():
    state(root,'RUNNING_REAL_DONOR_CPU',counts,sid,preflight=preflight);run([sys.executable,str(DONOR),'--spec',str(donor_spec)],root/'logs'/f'{sid}_donor.log',('CLAIMED','running_real_donor_cpu',sid,None))
   authority=root/'real_donor_v1'/sid/'INDEPENDENT_AUTHORITY.json'
   if not authority.exists():run([sys.executable,str(VALIDATOR),'--result',str(donor_result),'--authority',str(authority)],root/'logs'/f'{sid}_donor_validate.log',('CLAIMED','validating_real_donor',sid,None))
   gate=runtime_gate()
   if not gate['pass']:
    raise RuntimeError(f"runtime GPU gate became busy before {sid}: {json.dumps(gate,ensure_ascii=False)}")
   state(root,'RUNNING_PROPAINTER_GPU',counts,sid,preflight=preflight)
   outcome=run_propainter_with_retries(root,row,counts)
   if outcome['status']=='FAILED_QUALITY_C':
    counts['failed']+=1;event(root,{"status":"FAILED_QUALITY_C","session":sid,"attempt":outcome['attempt'],"attempt_result":outcome['attempt_ref']})
    if ordinal==0:raise QualityGateError(f"mandatory canary {sid} failed quality gates")
    continue
   counts['completed_new']+=1;event(root,{"status":"PASS_SESSION","session":sid,"attempt":outcome['attempt'],"result":outcome['result']})
   if ordinal==0:
    atomic(root/'CANARY_GATE.json',{"schema_version":"exact78-clean-wave-canary-gate-v3","created_at":now(),"status":"PASS_CANARY_GRADE_B_ALLOW_BULK","session":sid,"result":ref(result),"selection":ref(root/'EXACT78_WAVE0_SELECTION.json')})
    if a.canary_only:
     state(root,'PASS_CANARY_ONLY_STOP_REQUESTED',counts,None,preflight=preflight);break
  if not a.canary_only and counts['completed_new']==54:
   terminals={x['session_id']:ref(root/'propainter_v1'/x['session_id']/'RESULT.json') for x in ordered if (root/'propainter_v1'/x['session_id']/'RESULT.json').exists()}
   audit={"schema_version":"exact78-clean-wave-terminal-audit-v3","created_at":now(),"status":"PASS_WAVE0_CLEAN_TERMINAL","selection":ref(root/'EXACT78_WAVE0_SELECTION.json'),"counts":counts,"new_clean_terminals":terminals,"existing_clean_count":4,"central_lease_terminal":load(LEASE),"claim_limit":"Synthetic Clean Grade-B only; no Robot/contact truth/training authority."}
   atomic(root/'TERMINAL_AUDIT.json',audit);state(root,'PASS_WAVE0_CLEAN_TERMINAL',counts,None,preflight=preflight)
  final=load(pid_path);final.update({"finished_at":now(),"status":"EXIT_0","counts":counts});atomic(pid_path,final);return 0
 except Exception as e:
  counts['failed']+=1;state(root,'STOPPED_FAIL_CLOSED',counts,row['session_id'] if 'row' in locals() else None,f'{type(e).__name__}: {e}',preflight)
  final=load(pid_path);final.update({"finished_at":now(),"status":"STOPPED_FAIL_CLOSED","error":f'{type(e).__name__}: {e}',"counts":counts});atomic(pid_path,final);raise

if __name__=='__main__':raise SystemExit(main())
