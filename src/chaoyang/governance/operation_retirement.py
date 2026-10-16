"""Deny retired direct entrypoints without breaking their reusable functions."""
import json
from pathlib import Path


def require_not_retired(repo_root,operation):
    path=Path(repo_root)/'docs/governance/RETIRED_OPERATION_ENTRYPOINTS.json'
    if not path.is_file(): return
    value=json.loads(path.read_text(encoding='utf8'))
    if value.get('schema_version')!='CHAOYANG_RETIRED_ENTRYPOINTS_V1': raise ValueError('RETIREMENT_SCHEMA')
    row=value['operations'].get(operation)
    if row is not None:
        raise SystemExit('RETIRED_OPERATION: '+operation+'; use '+row['replacement']+'; '+row['reason'])
