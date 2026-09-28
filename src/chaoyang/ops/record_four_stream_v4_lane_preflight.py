#!/usr/bin/env python3
"""Record lane-local readiness and blockers from the V4 immutable audit."""
from pathlib import Path
import argparse, json
from chaoyang.governance.common import REPO_ROOT, atomic_json, artifact_ref, load_json, now_iso
TASK='four_stream_full_pipeline_v4'; ATT=REPO_ROOT/f'_run/current/{TASK}/attempts/attempt_0001'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); a=ap.parse_args(); audit=load_json(a.audit); by={x['label']:x for x in audit['checks']}
 specs={
  'exact78':(['exact78:tasks/current/four_stream_pretraining_baseline_v32/TASK_PACKET.json','exact78:src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py'],'PAIR_PRODUCTION_NOT_STARTED'),
  'controller_manus':(['controller_manus:src/chaoyang/ops/run_wiyh_ai1_static_wrist_candidate_v32.py','controller_manus:src/chaoyang/ops/build_wiyh_wrist_dual_input_v1.py'],'STATIC_WRIST_COMPARISON_READY'),
  'hawor_retarget':(['hawor_retarget:src/chaoyang/ops/run_0915_robot15h_kai22_r0_wave0_v1.py','hawor_retarget:src/chaoyang/ops/run_hawor_resize_only_persistent_worker_v2.py'],'HAWOR_R0_INPUT_AUDIT_READY'),
  'huro':(['huro:src/chaoyang/ops/run_huro_common_review_v2.py','huro:_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai4_huro/known_answer_core_v1/RESULT.json'],'V3_RESULT_REUSE_ONLY'),
 }
 for lane,(keys,action) in specs.items():
  rows=[by[k] for k in keys]; ok=all(r.get('present') for r in rows); status='READY_CPU' if ok else 'BLOCKED_PREREQ'; blocker=None if ok else 'MISSING_OR_UNVERIFIED_INPUT'
  if lane=='huro' and ok: status='READY_REUSE_AUDIT'
  result={'schema_version':'chaoyang-v4-lane-preflight-v1','task_id':TASK,'lane':lane,'generated_at':now_iso(),'status':status,'current_action':action,'checks':rows,'blocker':blocker,'quality_claim':False,'adoption':False,'source_data_write_authorized':False}
  root=ATT/'lanes'/lane; atomic_json(root/'PREFLIGHT_RESULT.json',result); s=load_json(root/'STATE.json'); s.update(status=status,current_action=action,blocker=blocker,updated_at=now_iso(),latest_artifacts=[artifact_ref(root/'PREFLIGHT_RESULT.json')]); atomic_json(root/'STATE.json',s)
 print(json.dumps({'status':'PASSED','lanes':list(specs)},ensure_ascii=False))
if __name__=='__main__': main()
