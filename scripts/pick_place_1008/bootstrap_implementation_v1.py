"""Fenced finite implementation task for the user-authorized new pick/place batch."""
from pathlib import Path
import copy,hashlib,json,os,secrets,time,importlib
from datetime import datetime,timedelta
from chaoyang.governance.common import AUTHORITY_PATH,RECEIPT_PATH,REPO_ROOT,TASK_STATE_PATH,artifact_ref,atomic_json,canonical_bytes,load_json,now_iso,process_identity,publish_bundle,validate_task_state
TASK='exact78_pick_place_implementation_v1_20261008'
ROOT=REPO_ROOT; INDEX=ROOT/'tasks/current/INDEX.json'
OUT=ROOT/'_run/current'/TASK/'attempts/attempt_0001'
PACKET=ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
RECEIPT=ROOT/'tasks/receipts'/f'{TASK.upper()}.json'
SCRIPT=Path(__file__).resolve()
ACTIVE={'RUNNING','CLAIMED','WAIT_GPU_RESOURCE'}
def publish(state,event,index=None):
    kw={} if index is None else dict(task_packet_index_path=INDEX,task_packet_index_value=index)
    return publish_bundle(load_json(AUTHORITY_PATH),state,event_type=event,expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=SCRIPT,**kw)
def main():
    state=copy.deepcopy(load_json(TASK_STATE_PATH)); index=copy.deepcopy(load_json(INDEX))
    if OUT.exists() or PACKET.exists():raise RuntimeError('FRESH_IMPLEMENTATION_TASK_REQUIRED')
    if state.get('next_task') or any(t['status'] in ACTIVE for t in state['tasks']) or any(r.get('execution_allowed') for r in index['task_packets']):raise RuntimeError('LIVE_TASK_CONFLICT')
    writer={'pid':os.getpid(),'proc_start_ticks':process_identity(os.getpid())['start_ticks'],'executor_epoch':1,'fencing_token':secrets.token_hex(24),'scope':TASK}
    inputs=[artifact_ref(ROOT/'docs/current/DUAL_BRANCH_BASELINE_V1.json'),artifact_ref(ROOT/'baseline_v1/MANIFEST.json')]
    signature=hashlib.sha256(canonical_bytes({'inputs':inputs,'bootstrap':artifact_ref(SCRIPT),'mode':'IMPLEMENTATION_CPU_ONLY','objective':'199/29322; contact visual only; storage full is not a stop'})).hexdigest()
    packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK,'objective':'Implement isolated pick_place_1008 raw Controller/Head+MANUS adapter, protected Clean, visual contact, audit, candidate runner and finite coordinator; no model calls in this task.','phase':'PICK_PLACE_1008_IMPLEMENTATION_CPU','plan_revision':'PICK_PLACE_1008_RAW_TO_ROBOT_V1','execution_revision':TASK.upper(),'execution_mode':'IMPLEMENTATION_CPU_ONLY','registered_at':now_iso(),'deadline_at':(datetime.now().astimezone()+timedelta(hours=3)).isoformat(),'read_set':['AGENTS.md','baseline_v1/MANIFEST.json','docs/current/DUAL_BRANCH_BASELINE_V1.json','src/chaoyang/exact78_baseline','src/chaoyang/pipeline/pico_manus_motion_v2.py','src/chaoyang/governance','/path/to/private_storage/datasets/ego/pick_place_1008','docs/current/PROJECT_USAGE_V1_SNAPSHOT.json'],'write_set':[str(OUT.relative_to(ROOT)),str(PACKET.relative_to(ROOT)),str(RECEIPT.relative_to(ROOT)),'src/chaoyang/pick_place_1008','tests/pick_place_1008','docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1','scripts/pick_place_1008','src/chaoyang/governance/execute_pick_place_1008_v1.py','src/chaoyang/governance/pick_place_1008_contract.py','src/chaoyang/governance/current_r3_contracts.py','src/chaoyang/governance/exact78_visual_selection_v1.py','src/chaoyang/ops/run_exact78_pick_place_raw_to_robot_candidate_v1.py','src/chaoyang/ops/run_exact78_pick_place_audit_candidate_v1.py','tasks/current/INDEX.json','docs/governance'],'prerequisites':['GOVERNANCE_PASS','NO_ACTIVE_TASK','EXTERNAL_RAW_READ_ONLY','FROZEN_PARENT_SHA','NO_GPU_IN_IMPLEMENTATION'],'required_outputs':['candidate/RESULT.json','RESULT.json','terminal_receipt'],'budgets':{'wall_seconds_max':10800,'gpu_count':0,'model_calls':0,'render_calls':0,'cpu_threads_total':4,'workers_max':2,'session_attempts_max':1,'training_steps':0},'stop_conditions':['DOUBLE_WRITER','SOURCE_DRIFT','WRITE_OUTSIDE_PROJECT','DEADLINE'],'claim_limit':'Implementation and CPU tests only; no model or production claims. Parent visual/MANUS algorithms and pointers remain unchanged.','writer':writer,'inputs':{'files':inputs},'runtime_code':[artifact_ref(SCRIPT)],'run_signature':signature,'commands':[]}
    OUT.mkdir(parents=True);PACKET.parent.mkdir(parents=True)
    atomic_json(OUT/'WRITER.json',writer);atomic_json(PACKET,packet)
    state['tasks'].append({'task_id':TASK,'phase':packet['phase'],'attempt':1,'status':'RUNNING','session':'pick_place_1008','updated_at':now_iso(),'heartbeat_at':now_iso(),'pid':writer['pid'],'proc_start_ticks':writer['proc_start_ticks'],'gpu_id':None,'task_packet':artifact_ref(PACKET),'run_signature':signature,'quality':'PENDING','adoption':'NOT_ADOPTED','review':'IMPLEMENTATION_CPU'})
    state['next_task']={'task_id':TASK,'session':'pick_place_1008','prerequisites':packet['prerequisites'],'expected_resource':'CPU','stop_condition':'implementation sealed or 3-hour deadline'}
    index['task_packets'].append({'task_id':TASK,'packet_path':str(PACKET.relative_to(ROOT)),'packet_sha256':artifact_ref(PACKET)['sha256'],'execution_class':'CURRENT_LEDGER_ROUTABLE','execution_allowed':True})
    index.update(packet_revision=TASK.upper()+'_RUNNING',plan_revision=packet['plan_revision'],execution_revision=packet['execution_revision'],status='PASS',supersedes_index=artifact_ref(INDEX))
    if errors:=validate_task_state(state):raise RuntimeError(errors)
    publish(state,'PICK_PLACE_1008_IMPLEMENTATION_REGISTERED',index)
    atomic_json(OUT/'REGISTERED.json',{'task_id':TASK,'packet':artifact_ref(PACKET),'writer':writer})
    last=time.monotonic(); deadline=time.monotonic()+10800
    marker=OUT/'IMPLEMENTATION_DONE.json'
    while not marker.is_file() and time.monotonic()<deadline:
        time.sleep(2)
        if time.monotonic()-last>=50:
            state=copy.deepcopy(load_json(TASK_STATE_PATH)); row=next(t for t in state['tasks'] if t['task_id']==TASK)
            if row.get('pid')!=writer['pid'] or load_json(OUT/'WRITER.json')!=writer:raise RuntimeError('WRITER_FENCE_DRIFT')
            row.update(heartbeat_at=now_iso(),updated_at=now_iso());publish(state,'PICK_PLACE_1008_IMPLEMENTATION_HEARTBEAT');last=time.monotonic()
    payload=load_json(marker) if marker.is_file() else {'status':'FAILED_RUNTIME_FINAL','reason':'IMPLEMENTATION_DEADLINE','next_action':'new finite successor'}
    status='PASSED' if payload.get('status')=='PASSED' else 'FAILED_RUNTIME_FINAL'
    result={'schema_version':'pick-place-implementation-result-v1','task_id':TASK,'status':status,'task_packet':artifact_ref(PACKET),'run_signature':signature,'training_eligible':False,'control_ground_truth':False,**payload}
    atomic_json(OUT/'RESULT.json',result);atomic_json(RECEIPT,{'schema_version':'pick-place-implementation-receipt-v1','task_id':TASK,'status':status,'result':artifact_ref(OUT/'RESULT.json')})
    state=copy.deepcopy(load_json(TASK_STATE_PATH));row=next(t for t in state['tasks'] if t['task_id']==TASK)
    row.update(status=status,pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(OUT/'RESULT.json'),terminal_receipt=artifact_ref(RECEIPT))
    if (state.get('next_task') or {}).get('task_id')==TASK:state['next_task']=None
    index=copy.deepcopy(load_json(INDEX))
    for r in index['task_packets']:
        if r['task_id']==TASK:r['execution_allowed']=False
    index.update(packet_revision=TASK.upper()+'_TERMINAL',status='PASS_NO_ACTIVE_TASKS',supersedes_index=artifact_ref(INDEX))
    from chaoyang.governance import current_r3_contracts
    importlib.reload(current_r3_contracts)
    publish(state,'PICK_PLACE_1008_IMPLEMENTATION_'+status,index)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
