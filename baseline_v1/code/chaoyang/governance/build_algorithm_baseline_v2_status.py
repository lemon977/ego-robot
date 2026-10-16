"""Bounded V2 navigation snapshots; heartbeat, execution and quality stay separate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

TASK_ID='four_stream_algorithm_baseline_v2'
LANES=('exact78','ai1','ai2','ai4_huro')


def snapshot(path):
    path=Path(path)
    if not path.is_file():return {'read_status':'ABSENT','live_source_path':str(path)}
    before=path.stat();raw=path.read_bytes();after=path.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):
        return {'read_status':'UNSTABLE','live_source_path':str(path)}
    try:value=json.loads(raw)
    except (ValueError,UnicodeDecodeError):return {'read_status':'INVALID_JSON','live_source_path':str(path)}
    if not isinstance(value,dict):return {'read_status':'INVALID_SCHEMA','live_source_path':str(path)}
    return {'read_status':'SNAPSHOT','live_source_path':str(path),
            'observed_bytes':len(raw),'observed_sha256':hashlib.sha256(raw).hexdigest(),'value':value}


def alive(pid,ticks,proc_root=Path('/proc')):
    if not isinstance(pid,int) or pid<=0 or ticks is None:return False
    try:
        fields=(proc_root/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
        return fields[0]!='Z' and int(fields[19])==int(ticks)
    except (OSError,ValueError,IndexError):return False


def build_status(repo_root,task_state,generated_at):
    root=Path(repo_root)
    run=root/f'_run/current/{TASK_ID}/attempts/attempt_0001'
    parents=[x for x in task_state.get('tasks',[]) if x.get('task_id')==TASK_ID]
    if len(parents)!=1:raise ValueError('V2_PARENT_NOT_UNIQUE')
    parent=parents[0];lanes={}
    for lane in LANES:
        s=snapshot(run/'lanes'/lane/'STATE.json');value=s.pop('value',{})
        entry=dict(status=value.get('status','UNKNOWN_NOT_PASS'),source_snapshot=s,
                   owner_pid=value.get('pid'),owner_live=alive(value.get('pid'),value.get('proc_start_ticks')),
                   heartbeat_at=value.get('heartbeat_at',value.get('last_heartbeat_at')),
                   current_action=value.get('current_action',value.get('phase')),
                   source_snapshot_is_not_frozen_result=True,
                   pipeline_complete=value.get('PIPELINE_COMPLETE',False),
                   algorithm_quality_pass=value.get('NUMERIC_QUALITY_PASS',False),
                   training_eligible=False,control_ground_truth=False)
        if lane=='ai2':
            job=snapshot(run/'lanes/ai2/ROBOT_RUN_STATE.json');j=job.pop('value',{})
            entry['job_snapshot']=job
            entry['job_status']=j.get('status','NOT_STARTED')
            entry['job_pid']=j.get('pid')
            entry['job_live']=alive(j.get('pid'),j.get('proc_start_ticks'))
            # A live registered child must not be hidden by its parent's old READY state.
            if entry['job_live'] and j.get('status')=='RUNNING':entry['status']='RUNNING_FULL_ROBOT_V2'
            elif j.get('status')=='EXECUTED':entry['status']='ROBOT_EXECUTED_PENDING_EVALUATION'
        lanes[lane]=entry
    route=snapshot(root/'tasks/current/INDEX.json').get('value',{})
    rows=[r for r in route.get('task_packets',[]) if r.get('task_id')==TASK_ID]
    return dict(schema_version='FOUR_STREAM_ALGORITHM_BASELINE_V2_STATUS',generated_at=generated_at,
                parent_task_id=TASK_ID,run_root=str(run),
                parent=dict(status=parent.get('status'),routable=len(rows)==1 and rows[0].get('execution_allowed') is True),
                lanes=lanes,training_eligible=False,control_ground_truth=False,
                freshness_scope='Parent receipt FRESH is process heartbeat only, not document freshness or algorithm quality',
                claim_limit='Point-in-time navigation. Only immutable RESULT and baseline release decide adoption; no unknown becomes PASS.')


def publish_navigation_if_v2(repo_root,task_state,generated_at):
    root=Path(repo_root);plan=root/'docs/current/PLAN.md'
    if not plan.is_file() or 'FOUR_STREAM_ALGORITHM_BASELINE_V2' not in plan.read_text():return
    from chaoyang.governance.common import atomic_json
    atomic_json(root/'docs/current/STATUS.json',build_status(root,task_state,generated_at))
