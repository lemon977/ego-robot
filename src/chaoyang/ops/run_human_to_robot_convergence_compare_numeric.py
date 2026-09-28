#!/usr/bin/env python3
"""Freeze a bounded Local/HuRo comparison from existing common-contract arrays."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from chaoyang.governance.common import artifact_ref, load_json

REPO=Path("/mnt/workspace/code/chaoyang");TASK="human_to_robot_baseline_v1_convergence_20260923"
OUT=REPO/f"_run/current/{TASK}/attempts/attempt_0001/lanes/compare/numeric_v1"
INDEX=REPO/"tasks/current/INDEX.json"
CASES={
"get_potato_chips_0915_007":(REPO/"_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_007_wave0/ROBOT_R0_V1.npz",REPO/"_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/full_0001/get_potato_chips_0915_007/HURO_CORE_V1.npz"),
"play_cards_0915_031":(REPO/"_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz",REPO/"_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/full_0001/play_cards_0915_031/HURO_CORE_V1.npz")}

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def stats(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)];return None if not len(v) else {"count":int(len(v)),"p50":float(np.percentile(v,50)),"p95":float(np.percentile(v,95)),"max":float(v.max())}
def write(p,v):
 t=p.with_name(f".{p.name}.tmp-{os.getpid()}");t.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def main():
 idx=load_json(INDEX)
 if len(idx.get("task_packets",[]))!=1 or idx["task_packets"][0].get("task_id")!=TASK:raise RuntimeError("TASK_NOT_ROUTABLE")
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);rows=[]
 for session,(lp,hp) in CASES.items():
  local,huro=load(lp),load(hp)
  for key in ("frame_id","timestamp_ns","target_valid","T_target_root_cam","T_cam_base","T_flange_hand","human_to_physical"):
   if not np.array_equal(local[key],huro[key],equal_nan=True):raise ValueError(f"COMMON_CONTRACT_DRIFT:{session}:{key}")
  common=local["wrist_valid"]&huro["wrist_valid"]
  fk_delta=np.linalg.norm(local["actual21_camera_m"]-huro["actual21_camera_m"],axis=-1)*1000
  row={"session_id":session,"frame_count":len(local["frame_id"]),"common_valid_side_frames":int(common.sum()),
       "local_position_residual_mm":stats(local["position_residual_mm"][common]),"huro_position_residual_mm":stats(huro["position_residual_mm"][common]),
       "local_full_rotation_deg":stats(local["rotation_residual_deg"][common]),"huro_full_rotation_deg":stats(huro["rotation_residual_deg"][common]),
       "method_fk21_difference_mm":stats(fk_delta[common]),"local_q_finite":bool(np.isfinite(local["q_arm"][common]).all() and np.isfinite(local["q_hand22"][common]).all()),
       "huro_q_finite":bool(np.isfinite(huro["q_arm"][common]).all() and np.isfinite(huro["q_hand22"][common]).all()),
       "common_target_exact":True,"common_mount_exact":True,"winner":None,"inputs":{"local":artifact_ref(lp),"huro":artifact_ref(hp)}}
  rows.append(row)
 result={"schema_version":"HUMAN_TO_ROBOT_LOCAL_HURO_NUMERIC_COMPARISON_V1","task_id":TASK,"created_at":datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
         "execution":"EXECUTED_REUSED_ARRAYS","structure":"PASS","quality":"INCONCLUSIVE_NO_INDEPENDENT_TRUTH","improvement":"FINITE_NUMERIC_CONCLUSION_BOUND_TO_COMMON_OLD_CONTRACT","adoption":"NOT_ADOPTED",
         "sessions":rows,"new_solver_invocations":0,"winner":None,"claim_limit":"Common old-contract residual and difference comparison only; no independent truth, partial-direction comparison, winner, control or deployment authority.",
         "training_eligible":False,"control_ground_truth":False,"physical_deployable":False,"external_metric_authority":False}
 write(OUT/"RESULT.json",result);print(json.dumps({"status":result["quality"],"sessions":len(rows),"output":str(OUT)},ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
