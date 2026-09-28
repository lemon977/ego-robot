#!/usr/bin/env python3
"""Render the rejected 031 hard-position/partial-direction window as honest evidence."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np
from chaoyang.governance.common import artifact_ref, load_json

REPO=Path("/mnt/workspace/code/chaoyang");TASK="human_to_robot_baseline_v1_convergence_20260923"
ROOT=REPO/f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/partial_direction_031/wave0"
OUT=REPO/f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product/partial_direction_031/review_v1"
OLD=REPO/"_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
HAWOR=REPO/"_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
RAW=REPO/"_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/play_cards_0915_031/raw"
INDEX=REPO/"tasks/current/INDEX.json";P=np.asarray([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def proj(x,k):
 z=x[:,2];u=np.full((len(x),2),np.nan);g=np.isfinite(x).all(1)&(z>1e-6);u[g]=x[g,:2]/z[g,None]*[k[0,0],k[1,1]]+k[:2,2];return u,g
def draw(im,u,g,color):
 h,w=im.shape[:2]
 for n in range(1,21):
  p=int(P[n]);
  if not(g[p] and g[n]):continue
  a=tuple(np.rint(u[p]).astype(int));b=tuple(np.rint(u[n]).astype(int));ok,a,b=cv2.clipLine((0,0,w,h),a,b)
  if ok:cv2.line(im,a,b,color,3,cv2.LINE_AA)
 for pt,good in zip(u,g,strict=True):
  if good and 0<=pt[0]<w and 0<=pt[1]<h:cv2.circle(im,tuple(np.rint(pt).astype(int)),4,color,-1,cv2.LINE_AA)
def write(p,v):
 t=p.with_name(f".{p.name}.tmp-{os.getpid()}");t.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def main():
 idx=load_json(INDEX)
 if len(idx.get('task_packets',[]))!=1 or idx['task_packets'][0].get('task_id')!=TASK:raise RuntimeError('TASK_NOT_ROUTABLE')
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);old,new,h=load(OLD),load(ROOT/"IK_WINDOW_PARTIAL_DIRECTION_V1.npz"),load(HAWOR)
 video=OUT/"031_IK_WINDOW_OLD_NEW_REVIEW.mp4";writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),6,(1920,600))
 if not writer.isOpened():raise RuntimeError('WRITER')
 try:
  for f in range(66,82):
   raw=cv2.imread(str(RAW/f'{f:06d}.png'));k=h['intrinsics'][f];oldpts=old['actual21_camera_m'][f,1]
   delta=new['T_actual_root_cam_eval'][f,1]@np.linalg.inv(old['T_actual_root_cam'][f,1])
   newpts=(delta@np.c_[oldpts,np.ones(21)].T).T[:,:3]
   target=old['T_target_root_cam'][f,1,:3,3]
   panels=[]
   for title,points,color in [('RAW + target',None,(0,255,0)),('OLD R0 FK',oldpts,(255,100,20)),('NEW hard-position IK',newpts,(20,90,255))]:
    panel=cv2.resize(raw,(640,480),interpolation=cv2.INTER_AREA)
    target_uv,target_ok=proj(target[None],k);target_uv*=.5
    if target_ok[0]:cv2.drawMarker(panel,tuple(np.rint(target_uv[0]).astype(int)),(0,255,0),cv2.MARKER_CROSS,24,3)
    if points is not None:
     u,g=proj(points,k);draw(panel,u*.5,g,color)
    cv2.putText(panel,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),2,cv2.LINE_AA);panels.append(panel)
   canvas=np.zeros((600,1920,3),np.uint8);canvas[:480]=np.hstack(panels)
   pos=new['position_residual_mm'][f,1];d0,d1=new['direction_angle_deg'][f,1];rot=new['full_rotation_angle_deg'][f,1]
   text=f"frame {f} | position {pos:.2f} mm <=20 PASS | longitudinal {d0:.2f} deg | normal {d1:.2f} deg | full SO(3) {rot:.2f} deg"
   cv2.putText(canvas,text,(20,525),cv2.FONT_HERSHEY_SIMPLEX,.68,(235,235,235),2,cv2.LINE_AA)
   cv2.putText(canvas,"RESULT: REJECTED_QUALITY (0/16 supported-direction gate); green cross is diagnostic target and never drives drawing",(20,565),cv2.FONT_HERSHEY_SIMPLEX,.62,(40,120,255),2,cv2.LINE_AA)
   writer.write(canvas)
 finally:writer.release()
 decoded=0;cap=cv2.VideoCapture(str(video))
 while True:
  ok,_=cap.read()
  if not ok:break
  decoded+=1
 cap.release()
 result={"schema_version":"HUMAN_TO_ROBOT_031_IK_WINDOW_REVIEW_V1","task_id":TASK,"created_at":datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
         "execution":"EXECUTED","structure":"PASS","quality":"REJECTED_QUALITY","improvement":"POSITION_GATE_16_OF_16_BUT_SUPPORTED_DIRECTION_0_OF_16","adoption":"NOT_ADOPTED",
         "frame_ids":list(range(66,82)),"decoded_frames":decoded,"video":artifact_ref(video),"numeric_candidate":artifact_ref(ROOT/"IK_WINDOW_PARTIAL_DIRECTION_V1.npz"),
         "claim_limit":"Skeleton/FK evidence for rejected development candidate; no mesh product, full-pose, contact, control or deployment authority."}
 write(OUT/"RESULT.json",result);print(json.dumps({"status":result['quality'],"decoded":decoded,"video":str(video)},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
