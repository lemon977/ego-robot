#!/usr/bin/env python3
"""Publish the first V4 CPU milestone into the attempt-local evidence ledger."""
from pathlib import Path
import json
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, load_json, now_iso
TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'
def main():
 reuse=ATT/'V3_REUSE_MANIFEST.json'; audit=ATT/'V4_REUSE_AND_DEPENDENCY_AUDIT.json'; t=now_iso()
 lanes=load_json(reuse)['lanes']; blocker_map={'exact78':None,'controller_manus':None,'hawor_retarget':'MISSING_REUSABLE_PERSISTENT_WORKER_CODE','huro':None}
 for lane in lanes:
  p=ATT/'lanes'/lane/'STATE.json'; s=load_json(p); blocker=blocker_map[lane]; s.update(status='READY_CPU' if blocker is None else 'BLOCKED_PREREQ',current_action='V3_REUSE_FROZEN' if blocker is None else 'INPUT_CODE_ASSET_AUDIT',blocker=blocker,updated_at=t,latest_artifacts=[artifact_ref(reuse),artifact_ref(audit)]); atomic_json(p,s)
 progress={'schema_version':'chaoyang-v4-progress-receipt-v1','task_id':TASK,'generated_at':t,'stage':'CPU_REUSE_AND_DEPENDENCY_AUDIT','completed':['V4 route claimed','four isolated lanes initialized','V3 result references frozen by SHA','WiLoR and DiffuEraser availability checked','lane-local blockers recorded'],'reuse_manifest':artifact_ref(reuse),'dependency_audit':artifact_ref(audit),'lane_status':{lane:('BLOCKED_PREREQ' if blocker_map[lane] else 'READY_CPU') for lane in lanes},'quality_claims':{'numeric':False,'visual':False,'adoption':False},'next':['Exact78 pair producer readiness','AI1 same-session comparison inputs','HaWoR difficult-session producer audit','HuRo explicit wrist-objective canary'],'control_ground_truth':False,'physical_deployable':False}
 atomic_json(ATT/'PROGRESS_2H.json',progress); print(json.dumps(progress,ensure_ascii=False))
if __name__=='__main__': main()
