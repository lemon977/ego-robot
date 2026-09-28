#!/usr/bin/env python3
"""Audit the frozen Exact78 cohort for V4 without producing training data."""
from pathlib import Path
import json
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, now_iso
from chaoyang.human_ego.exact78_v32 import validate_cohort_split
TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001/lanes/exact78'
HIST=REPO_ROOT/'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json'; COHORT=REPO_ROOT/'manifests/human_ego/exact78_cohort_split_v32.json'
def main():
 payload=json.loads(COHORT.read_text()); current=validate_cohort_split(payload); historical=json.loads(HIST.read_text())
 old={str(x['session_id']) for x in historical.get('sessions',[])}; new=set(current); audit={'schema_version':'chaoyang-exact78-v4-cohort-audit-v1','task_id':TASK,'generated_at':now_iso(),'cohort_ref':artifact_ref(COHORT),'historical_identity_ref':artifact_ref(HIST),'frozen_count':len(new),'historical_count':len(old),'intersection_count':len(new&old),'new_not_historical':sorted(new-old),'historical_not_current':sorted(old-new),'split_counts':payload['counts']['splits'],'source_group_count':len(payload['source_groups']),'replacement_policy':payload['replacement_policy'],'development_final_exposed':payload['development_final_policy']=='EXPOSED_NOT_FIRST_BLIND_TEST','training_ready':False,'pair_production_started':False}
 atomic_json(ATT/'EXACT78_COHORT_AUDIT.json',audit); print(json.dumps({'status':'PASSED','frozen_count':len(new),'historical_intersection':len(new&old),'unresolved_members':len(new-old)}))
if __name__=='__main__': main()
