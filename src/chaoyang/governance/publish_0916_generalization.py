"""Finite zero-GPU publisher, fresh attempt and immutable input manifest."""
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

TASK='sensor0916_generalization_visuals_retry1_20260930'
ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'

def main():
    state=copy.deepcopy(load_json(TASK_STATE_PATH));index=copy.deepcopy(load_json(INDEX))
    revision=int(load_json(RECEIPT_PATH)['governance_revision'])
    if state.get('next_task') or any(x.get('execution_allowed') for x in index['task_packets']):raise RuntimeError('OTHER_TASK_ACTIVE')
    if any(x.get('status') in ('PENDING','RUNNING','CLAIMED','WAIT_GPU_RESOURCE') for x in state['tasks']):raise RuntimeError('OTHER_WRITER_ACTIVE')
    if ROOT.exists() or PACKET.exists():raise RuntimeError('FRESH_ATTEMPT_REQUIRED')
    data=Path('/path/to/private_data/egodata/datasets/ego');files=[]
    for sid in ('025','097','098','101'):
        folder=data/'processed/chips_cards_handle_highview_0916/cleaned/playing_cards'/('play_cards_0916_'+sid)
        files.append(artifact_ref(data/'chips_cards_handle_highview_0916/cards_130_0916'/sid/'dataset.hdf5'))
        for name in ('clip_manifest.json','camera_params.json','CameraRecord_play_cards_0916_'+sid+'.mp4'):files.append(artifact_ref(folder/name))
    op=REPO_ROOT/'src/chaoyang/ops/run_0916_generalization_visuals.py'
    for path in (op,REPO_ROOT/'src/chaoyang/pipeline/pico_manus_motion_v2.py'):files.append(artifact_ref(path))
    inputs={'files':files,'baseline':artifact_ref(REPO_ROOT/'docs/current/ROBOT100_HARDWARE_BASELINE_V16.json'),'fixed_sessions':['025','097','098','101'],'no_parameter_fitting':True}
    pid=os.getpid();ticks=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
    writer={'pid':pid,'proc_start_ticks':ticks,'executor_epoch':1,'fencing_token':secrets.token_hex(24),'scope':TASK,'created_at':now_iso()}
    signature=hashlib.sha256(canonical_bytes(inputs)).hexdigest()
    packet={'schema_version':'chaoyang-r22-task-packet-v1','task_id':TASK,'objective':'Fixed-parameter sensor regression on025/097/098/101 with 2D projection and 3D skeleton wrist trajectories','phase':'SENSOR_2D_3D_REGRESSION','plan_revision':'SENSOR_VISUALS_V1','execution_revision':TASK.upper(),'deadline_at':(datetime.now().astimezone()+timedelta(minutes=30)).isoformat(timespec='seconds'),'read_set':['AGENTS.md','docs/current/ROBOT100_BATCH_ZH.md',str(op.relative_to(REPO_ROOT))],'write_set':[str(ROOT)],'prerequisites':['governance_PASS','no_other_executable_task','fresh_attempt','0916_only'],'required_outputs':['RESULT.json','eight decoded videos','motion and projection arrays','per-session metrics'],'budgets':{'cpu_threads_total':2,'workers_max':1,'gpu_count':0,'training_steps':0,'session_attempts_max':1,'wall_seconds_max':1800},'stop_conditions':['deadline','writer_fence_drift','runtime_failure'],'claim_limit':'Sensor regression only, no robot IK or accuracy adoption.','writer':writer,'inputs':{'operation':artifact_ref(op)},'run_signature':signature,'commands':['chaoyang run run_0916_generalization_visuals']}
    errors=_validate_packet(packet)
    if errors:raise RuntimeError(str(errors))
    subprocess.run(['git','apply','--check','scripts/generalization_contract.patch'],cwd=REPO_ROOT,check=True)
    subprocess.run(['git','apply','scripts/generalization_contract.patch'],cwd=REPO_ROOT,check=True)
    import ast
    ast.parse((REPO_ROOT/'src/chaoyang/governance/current_r3_contracts.py').read_text())
    ROOT.mkdir(parents=True);PACKET.parent.mkdir(parents=True)
    atomic_json(ROOT/'INPUTS.json',inputs);packet['inputs']['manifest']=artifact_ref(ROOT/'INPUTS.json')
    atomic_json(PACKET,packet);atomic_json(ROOT/'WRITER.json',writer)
    row={'task_id':TASK,'phase':packet['phase'],'attempt':1,'status':'RUNNING','session':'Cards025_097_098_101','updated_at':now_iso(),'heartbeat_at':now_iso(),'pid':pid,'proc_start_ticks':ticks,'gpu_id':None,'task_packet':artifact_ref(PACKET),'run_signature':signature,'quality':'NOT_EVALUATED','adoption':'NOT_ADOPTED','review':'PENDING'}
    state['tasks'].append(row);state['next_task']={'task_id':TASK,'session':row['session'],'prerequisites':packet['prerequisites'],'expected_resource':'TWO_CPU_ZERO_GPU','stop_condition':'four session diagnostics or runtime failure'}
    index['task_packets'].append({'task_id':TASK,'packet_path':str(PACKET.relative_to(REPO_ROOT)),'packet_sha256':artifact_ref(PACKET)['sha256'],'execution_class':'CURRENT_LEDGER_ROUTABLE','execution_allowed':True})
    def publish(event):
        nonlocal revision
        index['packet_revision']=event;index['status']='PASS' if state.get('next_task') else 'PASS_NO_ACTIVE_TASKS'
        v=publish_bundle(load_json(AUTHORITY_PATH),state,event_type=event,expected_revision=revision,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
        revision=int(v['governance_revision'])
    publish('SENSOR_2D3D_REGISTERED')
    (ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    env=dict(os.environ,PYTHONPATH=str(REPO_ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    failure=None;process=None
    try:
        with (ROOT/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B','-m','chaoyang.cli','run','run_0916_generalization_visuals'],cwd=REPO_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            while process.poll() is None:
                time.sleep(10)
                if int(load_json(RECEIPT_PATH)['governance_revision'])!=revision:raise RuntimeError('WRITER_FENCE_DRIFT')
                if time.monotonic()-started>1800:raise RuntimeError('DEADLINE')
                row['heartbeat_at']=now_iso();publish('SENSOR_2D3D_HEARTBEAT')
            if process.returncode or not (ROOT/'RESULT.json').is_file():raise RuntimeError('WORKER_FAILED:'+str(process.returncode))
    except Exception as e:
        failure=str(e)
        if process and process.poll() is None:
            import signal
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
        atomic_json(ROOT/'FAILURE.json',{'status':'FAILED_RUNTIME_FINAL','error':failure,'log':artifact_ref(ROOT/'worker.log')})
    result=ROOT/('FAILURE.json' if failure else 'RESULT.json')
    if int(load_json(RECEIPT_PATH)['governance_revision'])!=revision:raise RuntimeError('PUBLICATION_CONFLICT_SELF_STOPPED_RECEIPT_PRESERVED')
    row.update(status='FAILED_RUNTIME_FINAL' if failure else 'PASSED',pid=None,proc_start_ticks=None,updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(result),quality='NOT_ACCURACY_CERTIFIED',adoption='NOT_ADOPTED',review='PENDING')
    state['next_task']=None
    for item in index['task_packets']:
        if item['task_id']==TASK:item['execution_allowed']=False
    publish('SENSOR_2D3D_TERMINAL');atomic_json(ROOT/'TERMINAL.json',{'status':row['status'],'result':artifact_ref(result),'revision':revision})
    print(row['status'],str(ROOT),flush=True)
    return 1 if failure else 0

if __name__=='__main__':raise SystemExit(main())
