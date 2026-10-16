from pathlib import Path
from chaoyang.governance.common import *
TASK='sensor0916_generalization_visuals_20260930'
root=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
state=load_json(TASK_STATE_PATH);index=load_json(REPO_ROOT/'tasks/current/INDEX.json')
assert state['next_task']['task_id']==TASK
active=[x for x in state['tasks'] if x['status'] in ('RUNNING','PENDING','CLAIMED','WAIT_GPU_RESOURCE')]
assert len(active)==1 and active[0]['task_id']==TASK
row=active[0]
assert not Path('/proc/'+str(row['pid'])).exists(),'OWN_WRITER_STILL_LIVE'
atomic_json(root/'FAILURE.json',{'status':'FAILED_RUNTIME_FINAL','stage':'REGISTRATION_CONTRACT_IMPORT','algorithm_executed':False,'reason':'Context anchor removed in current generator; fuzz incorrectly applied insertion. Exact insertion reverted before this closure.'})
row.update(status='FAILED_RUNTIME_FINAL',pid=None,proc_start_ticks=None,heartbeat_at=now_iso(),updated_at=now_iso(),result=artifact_ref(root/'FAILURE.json'))
state['next_task']=None
for x in index['task_packets']:
    if x['task_id']==TASK:x['execution_allowed']=False
index['status']='PASS_NO_ACTIVE_TASKS';index['packet_revision']='SENSOR_REGISTRATION_FAILED_TERMINAL'
v=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='SENSOR_REGISTRATION_FAILED_TERMINAL',expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=Path(__file__),task_packet_index_path=REPO_ROOT/'tasks/current/INDEX.json',task_packet_index_value=index)
atomic_json(root/'TERMINAL.json',{'status':'FAILED_RUNTIME_FINAL','revision':v['governance_revision']})
print('OWN_FAILED_REGISTRATION_CLOSED',v['governance_revision'])
