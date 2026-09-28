#!/usr/bin/env python3
"""Bounded read-only readiness probe for the V4 successor."""
from pathlib import Path
import json, hashlib
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, now_iso

ROOT=REPO_ROOT/'_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 checks=[]
 for label,rel in [('V3 terminal','_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/V3_STALE_RUNTIME_TERMINAL.json'),('V3 handoff','_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/handoff_epoch_0006/HANDOFF_CHECKPOINT.json'),('PLAN','docs/current/PLAN.md'),('AI2','docs/current/AI2.md'),('AI1','docs/current/AI1.md')]:
  p=REPO_ROOT/rel; checks.append({'name':label,'path':str(p),'present':p.is_file(),'sha256':sha(p) if p.is_file() else None,'bytes':p.stat().st_size if p.is_file() else None})
 for label, candidates in [('DiffuEraser',['src/chaoyang','vendor/DiffuEraser','/mnt/data']),('WiLoR',['src/chaoyang','/root/.venvs/wilor/bin/python'])]:
  checks.append({'name':label+'_asset_probe','candidates':candidates,'present':[str((REPO_ROOT/x).exists()) if not x.startswith('/') else str(Path(x).exists()) for x in candidates]})
 out={'schema_version':'chaoyang-v4-readiness-preflight-v1','task_id':'four_stream_full_pipeline_v4','generated_at':now_iso(),'read_only':True,'source_writes':False,'checks':checks,'next':'claim task only through governed writer; no algorithm result inferred'}
 atomic_json(ROOT/'PREFLIGHT_READINESS.json',out); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
