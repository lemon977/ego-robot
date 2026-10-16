"""Publish only this 0916 review navigation after its terminal receipt exists."""
from pathlib import Path
import subprocess
from chaoyang.governance.common import REPO_ROOT,RECEIPT_PATH,TASK_STATE_PATH,AUTHORITY_PATH,load_json,publish_bundle

state=load_json(TASK_STATE_PATH)
index=load_json(REPO_ROOT/'tasks/current/INDEX.json')
if state.get('next_task') or any(x.get('execution_allowed') for x in index['task_packets']):
    raise RuntimeError('OTHER_PUBLISHER_ACTIVE')
revision=int(load_json(RECEIPT_PATH)['governance_revision'])
root=REPO_ROOT/'_run/current/robot100_constraint_review_v16_retry1_20260929/attempts/attempt_0001'
assert load_json(root/'RESULT.json')['status']=='COMPLETE_PENDING_HUMAN_REVIEW'
assert load_json(root/'TERMINAL.json')['status']=='PASSED'
assert load_json(root/'DELIVERY_VERIFICATION.json')['local_full_decode']=='PASS'
subprocess.run(['patch','--dry-run','-p1','-i','scripts/v16_docs_fixed.patch'],cwd=REPO_ROOT,check=True)
subprocess.run(['patch','-p1','-i','scripts/v16_docs_fixed.patch'],cwd=REPO_ROOT,check=True)
value=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='ROBOT100_V16_REVIEW_DOCS_BOUND',expected_revision=revision,generator_path=Path(__file__))
print(value['governance_revision'])
