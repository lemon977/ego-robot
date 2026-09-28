#!/usr/bin/env python3
"""Claim V4, produce a bounded evidence inventory, and keep the coordinator alive."""
from pathlib import Path
import argparse, json, os, time, hashlib
from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH,
 RECEIPT_PATH, artifact_ref, atomic_json, load_json, now_iso, process_identity, publish_bundle)

TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'; INDEX=REPO_ROOT/'tasks/current/INDEX.json'; PACKET=REPO_ROOT/f'tasks/current/{TASK}/TASK_PACKET.json'
LANES=('exact78','controller_manus','hawor_retarget','huro')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20), b''): h.update(block)
 return h.hexdigest()
def ref(p): return artifact_ref(p)
def claim(expected):
 rec=load_json(RECEIPT_PATH); rev=int(rec['governance_revision'])
 if rev!=expected: raise RuntimeError(f'CAS mismatch {expected}!={rev}')
 state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK)
 if row.get('status')!='PENDING' or state.get('next_task',{}).get('task_id')!=TASK: raise RuntimeError('V4 not pending/current')
 ident=process_identity(os.getpid()); t=now_iso()
 row.update(status='RUNNING',pid=os.getpid(),proc_start_ticks=ident['start_ticks'],heartbeat_at=t,updated_at=t,phase='V4_CPU_PREFLIGHT_AND_REUSE_AUDIT',session='four_lane')
 state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK,'attempt':1,'status':'RUNNING','created_at':t,'message':'V4 coordinator claimed; CPU evidence production started.','pid':os.getpid(),'proc_start_ticks':ident['start_ticks']}])[-100:]
 return publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_FULL_PIPELINE_V4_CLAIMED',expected_revision=rev,generator_path=Path(__file__))['governance_revision']
def inventory():
 out=[]
 def exists(label,p):
  p=Path(p); out.append({'label':label,'path':str(p),'present':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'sha256':sha(p) if p.is_file() else None})
 exists('packet',PACKET); exists('run_signature',ATT/'RUN_SIGNATURE.json'); exists('v3_handoff',REPO_ROOT/'_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/handoff_epoch_0006/HANDOFF_CHECKPOINT.json')
 for lane, paths in {
  'exact78':['tasks/current/four_stream_pretraining_baseline_v32/TASK_PACKET.json','src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py'],
  'controller_manus':['src/chaoyang/ops/run_wiyh_ai1_static_wrist_candidate_v32.py','src/chaoyang/ops/build_wiyh_wrist_dual_input_v1.py'],
  'hawor_retarget':['src/chaoyang/ops/run_0915_robot15h_kai22_r0_wave0_v1.py','src/chaoyang/ops/run_hawor_resize_only_persistent_worker_v2.py'],
  'huro':['src/chaoyang/ops/run_huro_common_review_v2.py','_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai4_huro/known_answer_core_v1/RESULT.json'],
 }.items():
  for rel in paths: exists(lane+':'+rel, REPO_ROOT/rel)
 exists('wilor_code','/mnt/workspace/code/WiLoR/video_mano.py'); exists('wilor_weight','/mnt/workspace/code/WiLoR/pretrained_models/wilor_final.ckpt'); exists('diffueraser_vendor',REPO_ROOT/'vendor/DiffuEraser')
 atomic_json(ATT/'V4_REUSE_AND_DEPENDENCY_AUDIT.json',{'schema_version':'chaoyang-v4-reuse-audit-v1','task_id':TASK,'generated_at':now_iso(),'source_data_write_authorized':False,'checks':out,'interpretation':'Presence is readiness evidence only; no quality or adoption claim.'})
 for lane in LANES:
  p=ATT/'lanes'/lane/'STATE.json'; s=load_json(p); s.update(status='RUNNING_CPU_AUDIT',current_action='REUSE_AND_DEPENDENCY_AUDIT',updated_at=now_iso(),latest_artifacts=[ref(ATT/'V4_REUSE_AND_DEPENDENCY_AUDIT.json')]); atomic_json(p,s)
 atomic_json(ATT/'PROGRESS_2H.json',{'schema_version':'chaoyang-v4-progress-receipt-v1','task_id':TASK,'generated_at':now_iso(),'elapsed_stage':'INITIAL_CPU_PREFLIGHT','completed':['V4 route claim','four lane attempt initialization','V3 evidence reuse audit','WiLoR/DiffuEraser dependency probe'],'quality_results':{},'next':['Exact78 pair production','Controller/MANUS frozen comparison','HaWoR difficult-session input audit','HuRo wrist-objective canary'],'claims':{'control_ground_truth':False,'physical_deployable':False}})
def heartbeat():
 rec=load_json(RECEIPT_PATH); rev=int(rec['governance_revision']); state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK); t=now_iso(); row.update(status='RUNNING',pid=os.getpid(),proc_start_ticks=process_identity(os.getpid())['start_ticks'],heartbeat_at=t,updated_at=t,phase='V4_CPU_PREFLIGHT_AND_REUSE_AUDIT')
 publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_FULL_PIPELINE_V4_HEARTBEAT',expected_revision=rev,generator_path=Path(__file__))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expected-revision',type=int,required=True); ap.add_argument('--hold-seconds',type=int,default=43200); a=ap.parse_args()
 claim(a.expected_revision); inventory(); deadline=time.time()+a.hold_seconds
 while time.time()<deadline:
  time.sleep(30); heartbeat()
if __name__=='__main__': main()
