"""Bind only the 0916 review documentation after terminal publication."""
from pathlib import Path
import subprocess
from chaoyang.governance.common import REPO_ROOT,RECEIPT_PATH,TASK_STATE_PATH,AUTHORITY_PATH,load_json,publish_bundle

state=load_json(TASK_STATE_PATH)
if state.get('next_task'): raise RuntimeError('OTHER_PUBLISHER_ACTIVE')
revision=int(load_json(RECEIPT_PATH)['governance_revision'])
result=REPO_ROOT/'_run/current/robot100_side_review_v15_20260929/attempts/attempt_0001/RESULT.json'
if load_json(result).get('status')!='COMPLETE_PENDING_HUMAN_REVIEW': raise RuntimeError('RESULT_NOT_READY')
subprocess.run(['patch','--dry-run','-p1','-i','scripts/v15_docs.patch'],cwd=REPO_ROOT,check=True)
subprocess.run(['patch','-p1','-i','scripts/v15_docs.patch'],cwd=REPO_ROOT,check=True)
value=publish_bundle(load_json(AUTHORITY_PATH),state,event_type='ROBOT100_V15_REVIEW_DOCS_BOUND',expected_revision=revision,generator_path=Path(__file__))
print(value['governance_revision'])
