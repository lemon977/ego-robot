#!/usr/bin/env python3
"""Create the immutable V4 attempt and its four isolated lane ledgers."""
from pathlib import Path
import argparse, hashlib, json, os
from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH,
 RECEIPT_PATH, artifact_ref, atomic_json, load_json, now_iso, publish_bundle)

TASK='four_stream_full_pipeline_v4'; INDEX=REPO_ROOT/'tasks/current/INDEX.json'; PACKET=REPO_ROOT/f'tasks/current/{TASK}/TASK_PACKET.json'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'; LANES=('exact78','controller_manus','hawor_retarget','huro')
def tok(lane,t): return hashlib.sha256(f'{TASK}\0{lane}\0{t}\0{os.getpid()}'.encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expected-revision',type=int,required=True); a=ap.parse_args()
 receipt=load_json(RECEIPT_PATH); rev=int(receipt['governance_revision'])
 if rev!=a.expected_revision: raise RuntimeError(f'CAS mismatch {a.expected_revision}!={rev}')
 index=load_json(INDEX); entry=index.get('task_packets',[{}])[0]
 if entry.get('task_id')!=TASK or entry.get('execution_allowed') is not True: raise RuntimeError('V4 route is not executable')
 if ATT.exists():
  existing = {p.name for p in ATT.iterdir()}
  if existing - {'PREFLIGHT_READINESS.json'} or (ATT/'RUN_SIGNATURE.json').exists():
   raise RuntimeError('attempt already exists with execution artifacts')
 state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK)
 if state.get('next_task',{}).get('task_id')!=TASK or row.get('status')!='PENDING': raise RuntimeError('V4 not pending successor')
 t=now_iso(); ATT.mkdir(parents=True, exist_ok=True)
 sig={'schema_version':'chaoyang-four-stream-v4-run-signature-v1','task_id':TASK,'attempt':1,'t0':t,'deadline_at':row.get('deadline_at'),'task_packet':artifact_ref(PACKET),'source_data_write_authorized':False,'control_ground_truth':False,'physical_deployable':False,'external_metric_authority':False,'image_domain':'physical_left_sourceIndex_1_encoded_resize_only','publisher_pid':os.getpid()}
 atomic_json(ATT/'RUN_SIGNATURE.json',sig); refs={}
 for lane in LANES:
  root=ATT/'lanes'/lane; root.mkdir(parents=True); v={'schema_version':'chaoyang-four-stream-v4-lane-state-v1','parent_task_id':TASK,'lane':lane,'status':'READY_CPU_PREFLIGHT','current_action':'CPU_PREFLIGHT_READY','latest_artifacts':[],'blocker':None,'next_step':'RUN_FROZEN_FIRST_MILESTONE','writer':{'pid':None,'proc_start_ticks':None,'executor_epoch':1,'fencing_token':tok(lane,t),'writer_root':str(root.resolve())},'authority':'DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE','updated_at':t,'claims':{'PIPELINE_COMPLETE':False,'NUMERIC_QUALITY_PASS':False,'VISUAL_REVIEW_STATUS':'NOT_REVIEWED','TRAINING_COMPLETE':False,'TRAINING_ELIGIBLE':False,'CONTROL_GROUND_TRUTH':False,'PHYSICAL_DEPLOYABLE':False}}
  atomic_json(root/'STATE.json',v); refs[lane]=artifact_ref(root/'STATE.json')
 atomic_json(ATT/'LANE_LEDGER.json',{'schema_version':'chaoyang-four-stream-v4-lane-ledger-v1','task_id':TASK,'created_at':t,'cpu_threads_soft_cap':8,'single_publisher':True,'cross_lane_writes_allowed':False,'lanes':refs})
 atomic_json(ATT/'GPU_QUEUE_LEDGER.json',{'schema_version':'chaoyang-four-stream-v4-gpu-queue-v1','task_id':TASK,'created_at':t,'concurrent_owners_max':1,'lease_required':True,'current_owner':None,'allocations_seconds':{'exact78':43200,'observation':14400,'huro':14400,'stereo':900},'consumed_seconds':{'exact78':0,'observation':0,'huro':0,'stereo':0}})
 row.update(attempt=1,attempt_root=str(ATT),updated_at=t)
 state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK,'attempt':1,'status':'PENDING','created_at':t,'message':'V4 attempt initialized; lane writers remain unclaimed.','run_signature':artifact_ref(ATT/'RUN_SIGNATURE.json')}])[-100:]
 atomic_json(ATT/'TASK_STATE_INITIALIZED.json',state)
 pub=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_FULL_PIPELINE_V4_INITIALIZED',expected_revision=rev,generator_path=Path(__file__))
 atomic_json(ATT/'INITIALIZATION_RESULT.json',{'schema_version':'chaoyang-v4-initialization-result-v1','task_id':TASK,'status':'PASSED','created_at':t,'lane_ledger':artifact_ref(ATT/'LANE_LEDGER.json'),'gpu_queue':artifact_ref(ATT/'GPU_QUEUE_LEDGER.json'),'algorithm_execution_started':False,'governance_revision':pub['governance_revision']})
 print(json.dumps({'status':'PASSED','task_id':TASK,'attempt_root':str(ATT),'governance_revision':pub['governance_revision']}))
if __name__=='__main__': main()
