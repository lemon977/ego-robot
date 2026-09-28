#!/usr/bin/env python3
"""Requeue one bounded V4 attempt after a corrected wrapper-only crash."""
from pathlib import Path
import argparse
from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH, RECEIPT_PATH, load_json, publish_bundle, now_iso)
TASK='four_stream_full_pipeline_v4'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expected-revision',type=int,required=True); ap.add_argument('--reason',default='UNBOUND_SENTINEL_WRAPPER_BUG'); a=ap.parse_args()
 rec=load_json(RECEIPT_PATH); rev=int(rec['governance_revision'])
 if rev!=a.expected_revision: raise RuntimeError('CAS mismatch')
 state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK)
 if row.get('status')!='FAILED_RUNTIME_RETRYABLE': raise RuntimeError('V4 is not retryable')
 t=now_iso(); row.update(status='PENDING',pid=None,proc_start_ticks=None,heartbeat_at=None,updated_at=t,phase='V4_CPU_PREFLIGHT_AND_REUSE_AUDIT')
 state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK,'attempt':1,'status':'PENDING','created_at':t,'message':'Requeued after bounded coordinator restart; immutable lane artifacts retained.','reason':a.reason}])[-100:]
 result=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_V4_WRAPPER_REQUEUED',expected_revision=rev,generator_path=Path(__file__))
 print(result)
if __name__=='__main__': main()
