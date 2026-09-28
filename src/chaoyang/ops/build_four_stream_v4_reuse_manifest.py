#!/usr/bin/env python3
"""Freeze reusable V3 evidence for V4 without copying large payloads."""
from pathlib import Path
import argparse, json
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, now_iso
TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'; V3=REPO_ROOT/'_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes'
ITEMS={'exact78':['exact78/clean_007_full_v2/RESULT.json','exact78/depth_encoded007_diagnostic_v1/RESULT.json'],'controller_manus':['ai1/candidate1_nominal_run1/RESULT.json','ai1/candidate2_focal_run1/RESULT.json','ai1/candidate3_installation_run1/RESULT.json'],'hawor_retarget':['ai2/hawor_full007_roi_c2/RESULT.json','ai2/hawor_full031_epoch7_c3/RESULT.json'],'huro':['ai4_huro/known_answer_core_v1/RESULT.json','ai4_huro/nonneutral_core_c2/RESULT.json']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ATT/'V3_REUSE_MANIFEST.json'); a=ap.parse_args(); lanes={}
 for lane,rels in ITEMS.items():
  refs=[]
  for rel in rels:
   p=V3/rel
   if p.is_file(): refs.append(artifact_ref(p))
  lanes[lane]={'reusable_result_refs':refs,'reuse_only':True,'quality_adopted':False,'external_metric_authority':False}
 atomic_json(a.output,{'schema_version':'chaoyang-v4-reuse-manifest-v1','task_id':TASK,'generated_at':now_iso(),'source_task':'four_stream_full_pipeline_v3','source_task_terminal':True,'payloads_copied':False,'lanes':lanes,'claim_limit':'References preserve V3 evidence; V4 must re-evaluate fixed inputs before adoption.'})
 print(json.dumps({'status':'PASSED','output':str(a.output),'lanes':{k:len(v['reusable_result_refs']) for k,v in lanes.items()}}))
if __name__=='__main__': main()
