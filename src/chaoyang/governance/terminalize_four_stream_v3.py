#!/usr/bin/env python3
"""Close the expired V3 publisher without deleting its partial evidence."""
from pathlib import Path
import json, os, argparse
from chaoyang.governance.common import (REPO_ROOT, AUTHORITY_PATH, TASK_STATE_PATH,
    RECEIPT_PATH, V71_TASK_PACKET_INDEX_PATH, artifact_ref, load_json, now_iso,
    publish_bundle, atomic_json)

TASK_ID = "four_stream_full_pipeline_v3"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--expected-revision', type=int, required=True); a=ap.parse_args()
    receipt=load_json(RECEIPT_PATH); rev=int(receipt['governance_revision'])
    if rev != a.expected_revision: raise RuntimeError(f"CAS mismatch {a.expected_revision}!={rev}")
    state=load_json(TASK_STATE_PATH); row=next(x for x in state['tasks'] if x['task_id']==TASK_ID)
    if row.get('status') != 'FAILED_RUNTIME_RETRYABLE': raise RuntimeError('V3 is not retryable stale')
    pid=row.get('pid')
    if pid and Path(f'/proc/{pid}').exists(): raise RuntimeError('publisher pid still alive')
    terminal=ATTEMPT/'V3_STALE_RUNTIME_TERMINAL.json'
    if not terminal.exists():
        terminal.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal, {'schema_version':'chaoyang-v3-terminal-result-v1','task_id':TASK_ID,
          'status':'FAILED_RUNTIME_FINAL','partial_delivery':True,'pipeline_complete':False,
          'quality_pass':False,'reason':'DEAD_PUBLISHER_AFTER_CHECKPOINT',
          'evidence':[str(ATTEMPT/'handoff_epoch_0006/HANDOFF_CHECKPOINT.json'),
                      str(ATTEMPT/'handoff_epoch_0006/CONTINUATION_EPOCH6_ZH.md')]})
    row.update(status='FAILED_RUNTIME_FINAL', phase='FOUR_STREAM_FULL_PIPELINE_V3_TERMINAL',
               pid=None, proc_start_ticks=None, gpu_id=None, heartbeat_at=None,
               updated_at=now_iso(), result=artifact_ref(terminal),
               last_attempt_terminal='FAILED_RUNTIME_FINAL',
               last_attempt_reason='DEAD_PUBLISHER_AFTER_CHECKPOINT')
    state['next_task']=None
    state['recent_events']=(state.get('recent_events',[])+[{'task_id':TASK_ID,'attempt':row.get('attempt',1),
      'status':'FAILED_RUNTIME_FINAL','created_at':now_iso(),'message':'Expired publisher closed; partial Epoch6 evidence preserved.',
      'result':artifact_ref(terminal)}])[-100:]
    idx={'schema_version':'chaoyang-v71-task-packet-index-v3','packet_revision':'V3_TERMINAL',
      'plan_revision':'FULL_PIPELINE_V3','execution_revision':'FULL_PIPELINE_V3','status':'PASS_NO_ACTIVE_TASKS',
      'supersedes_index':artifact_ref(INDEX),'task_packets':[],
      'claim_limit':'V3 is terminal; preserved partial evidence is not a quality pass.'}
    publish_bundle(load_json(AUTHORITY_PATH),state,event_type='FOUR_STREAM_V3_TERMINALIZED',
      expected_revision=rev,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=idx)
    print(json.dumps({'status':'PASSED','task_id':TASK_ID,'terminal':str(terminal)}))
if __name__=='__main__': main()
