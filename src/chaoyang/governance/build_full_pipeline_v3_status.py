"""Current V3 snapshot: actual lane states, never an inferred quality PASS."""
from pathlib import Path
from chaoyang.governance.build_algorithm_baseline_v2_status import snapshot,alive

TASK='four_stream_full_pipeline_v3'

def build_status(root,state,at):
    root=Path(root);out=root/'_run/current'/TASK/'attempts/attempt_0001'
    parent=next(t for t in state['tasks'] if t['task_id']==TASK)
    lanes={}
    for lane in ('exact78','ai1','ai2','ai4_huro'):
        s=snapshot(out/'lanes'/lane/'STATE.json');v=s.pop('value',{})
        lanes[lane]=dict(status=v.get('status','UNKNOWN'),current_action=v.get('current_action',v.get('phase')),
          owner_pid=v.get('pid'),owner_live=alive(v.get('pid'),v.get('proc_start_ticks')),
          heartbeat_at=v.get('heartbeat_at',v.get('last_heartbeat_at')),source_snapshot=s,
          pipeline_complete=v.get('PIPELINE_COMPLETE',False),algorithm_quality_pass=v.get('NUMERIC_QUALITY_PASS',False),
          human_review=v.get('human_review','NOT_CONFIRMED'))
    return dict(schema_version='FOUR_STREAM_FULL_PIPELINE_V3_STATUS',generated_at=at,parent_task_id=TASK,
      parent_status=parent['status'],run_root=str(out),lanes=lanes,
      training_eligible=False,physical_deployment_authorized=False,
      claim_limit='Live snapshot only; actual stage results and adopted release are distinct; unknown never means PASS.')

def publish_navigation_if_v3(root,state,at):
    root=Path(root);plan=root/'docs/current/PLAN.md'
    if not plan.exists() or 'FOUR_STREAM_FULL_PIPELINE_V3' not in plan.read_text():return
    from chaoyang.governance.common import atomic_json
    atomic_json(root/'docs/current/STATUS.json',build_status(root,state,at))
