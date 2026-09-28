#!/usr/bin/env python3
"""Register the bounded V4 successor after V3 terminalization."""
from pathlib import Path
import argparse, json, shutil
from datetime import datetime, timedelta
from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH,
 RECEIPT_PATH, V71_TASK_PACKET_INDEX_PATH, artifact_ref, load_json, now_iso,
 publish_bundle, atomic_json)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK_ID='four_stream_full_pipeline_v4'; PLAN='FOUR_STREAM_FULL_PIPELINE_V4'; EXEC='FOUR_STREAM_FULL_PIPELINE_V4'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'; REG=REPO_ROOT/f'_run/current/{TASK_ID}/registration_0001';
ATT=REPO_ROOT/f'_run/current/{TASK_ID}/attempts/attempt_0001'
LANES=('exact78','controller_manus','hawor_retarget','huro')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expected-revision',type=int,required=True); a=ap.parse_args()
 rec=load_json(RECEIPT_PATH); rev=int(rec['governance_revision'])
 if rev!=a.expected_revision: raise RuntimeError(f'CAS mismatch {a.expected_revision}!={rev}')
 state=load_json(TASK_STATE_PATH)
 if state.get('next_task') is not None or any(x.get('status') in {'PENDING','READY','CLAIMED','RUNNING','WAIT_GPU_RESOURCE'} for x in state.get('tasks',[])): raise RuntimeError('active task exists')
 if any(x.get('task_id')==TASK_ID for x in state.get('tasks',[])): raise RuntimeError('V4 already registered')
 current=load_json(INDEX)
 if current.get('task_packets') != [] or current.get('status')!='PASS_NO_ACTIVE_TASKS': raise RuntimeError('index not terminal empty')
 REG.mkdir(parents=True,exist_ok=False); shutil.copyfile(INDEX,REG/'PREDECESSOR_TASK_PACKET_INDEX.json')
 packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK_ID,
  'objective':'Twelve-hour bounded four-lane development delivery: preserve V3 evidence, validate DiffuEraser/WiLoR and HuRo candidates, and produce reproducible baseline artifacts without control or deployment authority.',
  'phase':TASK_ID.upper(),'plan_revision':PLAN,'execution_revision':EXEC,
  'read_set':['docs/governance/CURRENT_STATUS_RECEIPT.json','docs/governance/ALGORITHM_CONTRACT.json','docs/current/PLAN.md','docs/current/EXACT78.md','docs/current/AI1.md','docs/current/AI2.md','_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/handoff_epoch_0006/HANDOFF_CHECKPOINT.json','_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/V3_STALE_RUNTIME_TERMINAL.json'],
  'write_set':[f'_run/current/{TASK_ID}','tasks/receipts/FOUR_STREAM_FULL_PIPELINE_V4_RESULT.json','docs/current/visuals/FOUR_STREAM_FULL_PIPELINE_V4'],
  'prerequisites':['governance_PASS_FRESH','no_active_current_task','source_processed_archive_sealed_READ_ONLY','single_publisher','four_lane_writer_roots_ISOLATED','single_GPU_lease_owner','noncommercial_research_evaluation','physical_left_eye_encoded_resize_only'],
  'required_outputs':[f'attempts/attempt_0001/lanes/{x}/STATE.json' for x in LANES]+['attempts/attempt_0001/RUN_SIGNATURE.json','attempts/attempt_0001/PROGRESS_2H.json','attempts/attempt_0001/RESULT.json'],
  'budgets':{'wall_seconds_max':43200,'cpu_threads_soft_cap':8,'gpu_concurrent_owners_max':1,'progress_hours':[2,4,6,9,12],'candidate_max_per_method':1,'same_signature_retry_max':1},
  'stop_conditions':['deadline_or_budget_saves_resumable_state','quality_failure_no_same_signature_retry','writer_or_GPU_lease_violation_fails_closed','no_source_or_sealed_writes','control_ground_truth_false','physical_deployable_false'],
  'weights':'ABSENT','child_weights_policy':'EACH_CHILD_BINDS_EXACTLY_ONE_PINNED_WEIGHT_OR_ABSENT','calibration_or_absent':'PER_LANE_PINNED_EVIDENCE_OR_ABSENT','external_metric_authority':False,
  'expected_resource':'FOUR_ISOLATED_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE','executor_epoch':1,'fencing':{'pid_startticks_required':True,'immutable_final':True,'unique_primary_writer':True,'lane_writer_roots':{x:f'_run/current/{TASK_ID}/attempts/attempt_0001/lanes/{x}' for x in LANES}},'attempt_max':1,
  'claim_limit':'Development evidence only. No model result is automatically adopted; no control ground truth, physical deployment, or external metric authority.'}
 errs=_validate_packet(packet)
 if errs: raise RuntimeError('invalid packet: '+'; '.join(errs))
 ppath=REPO_ROOT/f'tasks/current/{TASK_ID}/TASK_PACKET.json'; ppath.parent.mkdir(parents=True,exist_ok=True); atomic_json(ppath,packet)
 pref=artifact_ref(ppath); frozen=artifact_ref(REG/'PREDECESSOR_TASK_PACKET_INDEX.json')
 successor={'schema_version':'chaoyang-v71-task-packet-index-v3','packet_revision':'FOUR_STREAM_FULL_PIPELINE_V4_ROUTABLE','plan_revision':PLAN,'execution_revision':EXEC,'status':'PASS','supersedes_index':frozen,'task_packets':[{'task_id':TASK_ID,'packet_path':str(ppath.relative_to(REPO_ROOT)),'packet_sha256':pref['sha256'],'execution_class':'CURRENT_LEDGER_ROUTABLE','execution_allowed':True,'weights':'ABSENT'}],'claim_limit':'One bounded V4 successor; four lanes isolated and GPU serialized.'}
 t=now_iso(); deadline=(datetime.now().astimezone()+timedelta(hours=12)).isoformat(timespec='seconds')
 row={'task_id':TASK_ID,'phase':TASK_ID.upper(),'plan_execution_revision':EXEC,'attempt':0,'status':'PENDING','updated_at':t,'heartbeat_at':None,'session':None,'pid':None,'proc_start_ticks':None,'gpu_id':None,'task_packet':pref,'deadline_at':deadline,'t0':t}
 state['tasks'].append(row); state['next_task']={'task_id':TASK_ID,'session':'four_lane','prerequisites':packet['prerequisites'],'expected_resource':packet['expected_resource'],'stop_condition':'12h bounded lanes with resumable terminal evidence; no deployment authority.'}; state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK_ID,'attempt':0,'status':'PENDING','created_at':t,'message':'Registered V4 successor after terminal V3; execution not yet claimed.','task_packet':pref}])[-100:]
 out=REG/'PROJECTED_TASK_STATE.json'; atomic_json(out,state)
 publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_FULL_PIPELINE_V4_REGISTERED',expected_revision=rev,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=successor)
 atomic_json(REG/'RESULT.json',{'schema_version':'chaoyang-v4-registration-result-v1','task_id':TASK_ID,'status':'REGISTERED','t0':t,'deadline_at':deadline,'task_packet':pref})
 print(json.dumps({'status':'PASSED','task_id':TASK_ID,'governance_revision':rev+1,'t0':t,'deadline_at':deadline}))
if __name__=='__main__': main()
