#!/usr/bin/env python3
"""Bind V4's twelve-hour budget to the actual governed claim time."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from chaoyang.governance.common import REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH, RECEIPT_PATH, atomic_json, artifact_ref, load_json, now_iso, publish_bundle, process_identity

TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--expected-revision',type=int,required=True); a=p.parse_args()
 rec=load_json(RECEIPT_PATH); rev=int(rec['governance_revision'])
 if rev!=a.expected_revision: raise RuntimeError(f'CAS mismatch {a.expected_revision}!={rev}')
 state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK)
 if row.get('status')!='RUNNING' or not row.get('pid') or not process_identity(int(row['pid']))['alive']: raise RuntimeError('V4 coordinator is not live')
 t=now_iso(); deadline=(datetime.now().astimezone()+timedelta(hours=12)).isoformat(timespec='seconds')
 path=ATT/'CLAIM_WINDOW.json'
 atomic_json(path,{'schema_version':'chaoyang-v4-claim-window-v1','task_id':TASK,'attempt':1,'claimed_at':t,'deadline_at':deadline,'wall_seconds_max':43200,'claim_pid':row['pid'],'claim_proc_start_ticks':row['proc_start_ticks'],'basis':'actual_governed_claim_time'})
 row.update(t0=t,deadline_at=deadline,updated_at=t,heartbeat_at=t)
 state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK,'attempt':1,'status':'RUNNING','created_at':t,'message':'Twelve-hour V4 window bound to actual claim time.','claim_window':artifact_ref(path)}])[-100:]
 pub=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_V4_CLAIM_WINDOW_BOUND',expected_revision=rev,generator_path=Path(__file__))
 print({'status':'PASSED','claim_window':str(path),'governance_revision':pub['governance_revision'],'deadline_at':deadline})
if __name__=='__main__': main()
