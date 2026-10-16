"""Bounded single-writer registration, execution, and terminal publication."""
import copy
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from datetime import datetime,timedelta
from chaoyang.governance.common import REPO_ROOT,RECEIPT_PATH,TASK_STATE_PATH,AUTHORITY_PATH,artifact_ref,atomic_json,canonical_bytes,load_json,now_iso,publish_bundle
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK='robot100_retire_stale_navigation_20260929'
ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'

def main():
    # Waiting does not execute algorithms or mutate another publisher's state.
    deadline=time.monotonic()+900
    while True:
        pending=load_json(TASK_STATE_PATH)
        registry=load_json(INDEX)
        active=pending.get('next_task') or any(x.get('execution_allowed') for x in registry['task_packets']) or any(x.get('status') in ('PENDING','RUNNING','CLAIMED','WAIT_GPU_RESOURCE') for x in pending['tasks'])
        if not active: break
        if time.monotonic()>=deadline: raise RuntimeError('PUBLISHER_WAIT_TIMEOUT_NO_ALGORITHM_EXECUTED')
        print('WAIT_EXISTING_PUBLISHER',now_iso(),flush=True)
        time.sleep(15)
    revision=int(load_json(RECEIPT_PATH)['governance_revision'])
    state=copy.deepcopy(load_json(TASK_STATE_PATH)); index=copy.deepcopy(load_json(INDEX))
    if ROOT.exists() or PACKET.exists(): raise RuntimeError('FRESH_ATTEMPT_REQUIRED')
    if state.get('next_task') or any(x.get('execution_allowed') for x in index['task_packets']): raise RuntimeError('OTHER_TASK_ACTIVE')
    if any(x.get('status') in ('PENDING','RUNNING','CLAIMED','WAIT_GPU_RESOURCE') for x in state['tasks']): raise RuntimeError('OTHER_WRITER_ACTIVE')
    pid=os.getpid(); ticks=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
    writer={'pid':pid,'proc_start_ticks':ticks,'executor_epoch':1,'fencing_token':secrets.token_hex(24),'scope':TASK,'created_at':now_iso()}
    op=REPO_ROOT/'src/chaoyang/ops/run_robot100_retire_stale_navigation.py'
    selection=REPO_ROOT/'_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001/SELECTION.json'
    inputs={'operation':artifact_ref(op),'selection':artifact_ref(selection),'publisher':artifact_ref(Path(__file__)),'predecessor':artifact_ref(REPO_ROOT/'_run/current/robot100_constraint_review_v16_retry1_20260929/attempts/attempt_0001/RESULT.json')}
    signature=hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK,'objective':'Retire one untracked obsolete 0916 navigation backup with reference and active FD checks','phase':'ROBOT100_V16_BASELINE_OVERLAY','plan_revision':'ROBOT100_SIDE_FIX_V16','execution_revision':TASK.upper(),'deadline_at':(datetime.now().astimezone()+timedelta(minutes=45)).isoformat(timespec='seconds'),'read_set':['AGENTS.md','docs/current/ROBOT100_BATCH_ZH.md',str(op.relative_to(REPO_ROOT)),str(selection.relative_to(REPO_ROOT))],'write_set':[str(ROOT),str(REPO_ROOT/'docs/current/ROBOT100_BATCH_ZH.md.orig')],'prerequisites':['governance_PASS','no_other_executable_task','fresh_attempt','0916_only'],'required_outputs':['RESULT.json','RETIRED_NAVIGATION_BACKUP.bin'],'budgets':{'cpu_threads_total':2,'workers_max':1,'gpu_count':0,'training_steps':0,'session_attempts_max':1,'wall_seconds_max':2700},'stop_conditions':['deadline','writer_fence_drift','one_attempt_terminal','runtime_failure'],'claim_limit':'One 0916 visual canary; no batch adoption, control or physical accuracy authority.','writer':writer,'inputs':inputs,'run_signature':signature,'commands':['chaoyang run run_robot100_retire_stale_navigation']}
    errors=_validate_packet(packet)
    if errors: raise RuntimeError(str(errors))
    subprocess.run(['patch','--dry-run','-p1','-i','scripts/retire_nav_contract.patch'],cwd=REPO_ROOT,check=True)
    subprocess.run(['patch','-p1','-i','scripts/retire_nav_contract.patch'],cwd=REPO_ROOT,check=True)
    ROOT.mkdir(parents=True); PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET,packet); atomic_json(ROOT/'WRITER.json',writer)
    state['tasks'].append({'task_id':TASK,'phase':packet['phase'],'attempt':1,'status':'RUNNING','session':'play_cards_0916_025','updated_at':now_iso(),'heartbeat_at':now_iso(),'pid':pid,'proc_start_ticks':ticks,'gpu_id':None,'task_packet':artifact_ref(PACKET),'run_signature':signature,'quality':'PENDING','adoption':'NOT_ADOPTED','review':'PENDING'})
    state['next_task']={'task_id':TASK,'session':'play_cards_0916_025','prerequisites':packet['prerequisites'],'expected_resource':'TWO_CPU_ZERO_GPU','stop_condition':'one canary terminal'}
    index['task_packets'].append({'task_id':TASK,'packet_path':str(PACKET.relative_to(REPO_ROOT)),'packet_sha256':artifact_ref(PACKET)['sha256'],'execution_class':'CURRENT_LEDGER_ROUTABLE','execution_allowed':True})
    def publish(event):
        nonlocal revision
        index['packet_revision']=event
        index['status']='PASS' if state.get('next_task') else 'PASS_NO_ACTIVE_TASKS'
        value=publish_bundle(load_json(AUTHORITY_PATH),state,event_type=event,expected_revision=revision,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
        revision=int(value['governance_revision'])
    publish('ROBOT100_V16_REGISTERED')
    atomic_json(ROOT/'REGISTRATION.json',{'task':TASK,'packet':artifact_ref(PACKET),'revision':revision})
    env=dict(os.environ,PYTHONPATH=str(REPO_ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    (ROOT/'tmp').mkdir(); (ROOT/'cache').mkdir()
    code=1
    with (ROOT/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-B','-m','chaoyang.cli','run','run_robot100_retire_stale_navigation'],cwd=REPO_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        while process.poll() is None:
            time.sleep(15)
            if int(load_json(RECEIPT_PATH)['governance_revision'])!=revision:
                process.terminate(); process.wait(); raise RuntimeError('WRITER_FENCE_DRIFT')
            if time.monotonic()-started>2700:
                process.terminate(); process.wait(); break
            state['tasks'][-1]['heartbeat_at']=now_iso()
            publish('ROBOT100_V16_HEARTBEAT')
        code=process.returncode
    if code or not (ROOT/'RESULT.json').exists():
        atomic_json(ROOT/'FAILURE.json',{'status':'FAILED_RUNTIME_FINAL','exit_code':code,'log':artifact_ref(ROOT/'worker.log')})
    result=ROOT/('RESULT.json' if code==0 and (ROOT/'RESULT.json').exists() else 'FAILURE.json')
    row=state['tasks'][-1]
    row.update(status='PASSED' if result.name=='RESULT.json' else 'FAILED_RUNTIME_FINAL',pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(result),quality='NOT_APPLICABLE' if result.name=='RESULT.json' else 'NOT_EVALUATED',adoption='NOT_ADOPTED',review='NOT_APPLICABLE')
    state['next_task']=None
    for item in index['task_packets']:
        if item['task_id']==TASK: item['execution_allowed']=False
    publish('ROBOT100_V16_TERMINAL')
    atomic_json(ROOT/'TERMINAL.json',{'status':row['status'],'result':artifact_ref(result),'revision':revision,'quality':row['quality']})
    print(row['status'],str(ROOT),flush=True)
    return code or 0

if __name__=='__main__': raise SystemExit(main())
