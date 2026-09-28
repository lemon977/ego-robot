#!/usr/bin/env python3
"""Render complete Sensor reviews from saved HandMotion and common-backend FK."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.pipeline.v5_sensor import project

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/sensor/review_v1"
INDEX = REPO / "tasks/current/INDEX.json"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane3_sensor"
CASES = {
    "play_cards_0916_097": R2 / "run_097_wave0",
    "play_cards_0916_098": R2 / "run_098_wave1",
    "play_cards_0916_101": R2 / "run_101_wave1",
}
PARENTS21 = np.asarray([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])


def _load(path: Path) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}


def _write(path: Path,value:dict)->None:
    tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)


def _draw_clipped_skeleton(image:np.ndarray,uv:np.ndarray,valid:np.ndarray,color:tuple[int,int,int])->tuple[int,int]:
    height,width=image.shape[:2]; drawn=0; crossing=0
    for node in range(1,21):
        parent=int(PARENTS21[node])
        if not(valid[parent] and valid[node]): continue
        a=tuple(np.rint(uv[parent]).astype(int));b=tuple(np.rint(uv[node]).astype(int))
        inside_a=0<=a[0]<width and 0<=a[1]<height;inside_b=0<=b[0]<width and 0<=b[1]<height
        okay,p0,p1=cv2.clipLine((0,0,width,height),a,b)
        if okay:
            cv2.line(image,p0,p1,color,2,cv2.LINE_AA);drawn+=1
            if not(inside_a and inside_b): crossing+=1
    for point,good in zip(uv,valid,strict=True):
        if good and 0<=point[0]<width and 0<=point[1]<height:
            cv2.circle(image,tuple(np.rint(point).astype(int)),3,color,-1,cv2.LINE_AA)
    return drawn,crossing


def _map_xz(point:np.ndarray,center:np.ndarray,span:float,box:tuple[int,int,int,int])->tuple[int,int]:
    x0,y0,w,h=box; normalized=(point[[0,2]]-center)/span
    return int(x0+w/2+normalized[0]*w*.44),int(y0+h/2-normalized[1]*h*.44)


def _world(panel:np.ndarray,motion:dict,frame:int,center:np.ndarray,span:float)->None:
    box=(0,0,640,480); cv2.rectangle(panel,(0,0),(639,479),(38,38,38),-1)
    transforms=[motion["T_world_head"][frame],motion["T_world_wrist"][frame,0],motion["T_world_wrist"][frame,1]]
    names=["HEAD","L WRIST","R WRIST"];colors=[(190,190,190),(255,120,20),(30,70,255)]
    histories=[motion["T_world_head"][:frame+1,:3,3],motion["T_world_wrist"][:frame+1,0,:3,3],motion["T_world_wrist"][:frame+1,1,:3,3]]
    for transform,name,color,history in zip(transforms,names,colors,histories,strict=True):
        pts=np.asarray([_map_xz(p,center,span,box) for p in history],np.int32)
        if len(pts)>1: cv2.polylines(panel,[pts],False,color,1,cv2.LINE_AA)
        origin=_map_xz(transform[:3,3],center,span,box);cv2.circle(panel,origin,5,color,-1,cv2.LINE_AA)
        for axis,axis_color in enumerate(((80,80,255),(80,255,80),(255,160,60))):
            endpoint=transform[:3,3]+.08*transform[:3,axis]
            cv2.arrowedLine(panel,origin,_map_xz(endpoint,center,span,box),axis_color,2,cv2.LINE_AA,tipLength=.25)
        cv2.putText(panel,name,(origin[0]+6,origin[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    cv2.putText(panel,f"WORLD X/Z fixed origin | equal scale | span {span:.2f} m",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)


def _root_panel(panel:np.ndarray,points:np.ndarray,valid:np.ndarray,title:str)->None:
    panel[:]=(32,32,32); colors=((255,120,20),(30,70,255));
    finite=points[np.isfinite(points).all(axis=-1)]
    scale=max(float(np.max(np.abs(finite[...,:2]))) if finite.size else .1,.12)
    for side in range(2):
        uv=np.empty((21,2),float);uv[:,0]=320+points[side,:,0]/scale*250;uv[:,1]=260-points[side,:,1]/scale*250
        _draw_clipped_skeleton(panel,uv,valid[side],colors[side])
    cv2.putText(panel,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1,cv2.LINE_AA)


def _run_case(session:str,root:Path)->dict:
    result=load_json(root/"RESULT.json");motion=_load(root/"HAND_MOTION_V1.npz");robot=_load(root/"KAI22_COMMON_BACKEND_V1.npz")
    n=int(result["frames"]);source=Path(result["source_video"]["path"])
    folder=OUT/session;folder.mkdir(parents=True)
    output=folder/"SENSOR_COMPLETE_REVIEW.mp4";writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),30,(1920,1080))
    if not writer.isOpened():raise RuntimeError("VIDEO_WRITER")
    cap=cv2.VideoCapture(str(source));uv,positive=project(motion["joints21_camera"],motion["camera_K"])
    joint_valid=positive & motion["joint_valid"]
    world=np.concatenate([motion["T_world_head"][:,:3,3],motion["T_world_wrist"].reshape(-1,4,4)[:,:3,3]],axis=0)
    center=np.mean(np.stack([np.min(world[:,[0,2]],axis=0),np.max(world[:,[0,2]],axis=0)]),axis=0)
    span=max(float(np.max(np.ptp(world[:,[0,2]],axis=0)))*1.25,.5)
    crossing_total=0;line_total=0
    try:
        for frame in range(n):
            ok,stereo=cap.read()
            if not ok or stereo.shape[:2]!=(1536,4096):raise ValueError(f"SOURCE_DECODE:{frame}")
            source_frame=cv2.resize(stereo[:,2048:],(960,540),interpolation=cv2.INTER_AREA)
            overlay=source_frame.copy()
            scale=np.asarray([960/1280,540/960]);uv_scaled=uv[frame]*scale
            for side,color in enumerate(((255,120,20),(30,70,255))):
                lines,cross=_draw_clipped_skeleton(overlay,uv_scaled[side],joint_valid[frame,side],color);line_total+=lines;crossing_total+=cross
            canvas=np.zeros((1080,1920,3),np.uint8);canvas[40:580,:960]=source_frame;canvas[40:580,960:]=overlay
            _world(canvas[600:1080,:640],motion,frame,center,span)
            _root_panel(canvas[600:1080,640:1280],motion["manus_local_21_m"][frame],motion["manus_local_joint_valid"][frame],"MANUS wrist-local 21 | metres")
            _root_panel(canvas[600:1080,1280:1920],robot["fk21_root_relative"][frame],robot["valid"][frame,:,None].repeat(21,axis=1),"Kai22 saved FK | root-relative")
            cv2.putText(canvas,"physical-left RGB",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),2)
            cv2.putText(canvas,"MANUS projection (clipped segments retained)",(975,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),2)
            cv2.putText(canvas,f"{session} frame {frame}/{n-1} | motion_source=controller_manus | OFFLINE_VISUAL",(650,596),cv2.FONT_HERSHEY_SIMPLEX,.52,(220,220,220),1)
            writer.write(canvas)
        if cap.read()[0]:raise ValueError("EXTRA_SOURCE_FRAMES")
    finally:
        cap.release();writer.release()
    decoded=0;check=cv2.VideoCapture(str(output))
    while True:
        ok,_=check.read()
        if not ok:break
        decoded+=1
    check.release()
    if decoded!=n:raise ValueError(f"DECODE:{decoded}/{n}")
    row={"session_id":session,"execution":"EXECUTED_REUSED_ARRAYS_NEW_REVIEW","structure":"PASS",
         "quality":"PENDING_USER_VISUAL_REVIEW","adoption":"CANDIDATE_ONLY","frames":n,
         "clipped_crossing_segments_drawn":crossing_total,"total_segments_drawn":line_total,
         "source_motion":artifact_ref(root/"HAND_MOTION_V1.npz"),"source_robot_fk":artifact_ref(root/"KAI22_COMMON_BACKEND_V1.npz"),
         "video":{**artifact_ref(output),"decoded_frames":decoded},
         "claim_limit":"Readable review of saved arrays; no new solve, external GT, calibration, control or deployment authority."}
    _write(folder/"RESULT.json",row);return row


def main()->int:
    index=load_json(INDEX)
    if len(index.get("task_packets",[]))!=1 or index["task_packets"][0].get("task_id")!=TASK:raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True);rows=[_run_case(session,root) for session,root in CASES.items()]
    result={"schema_version":"HUMAN_TO_ROBOT_SENSOR_COMPLETE_REVIEW_V1","task_id":TASK,
            "created_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "execution":"EXECUTED","structure":"PASS","quality":"PENDING_USER_VISUAL_REVIEW",
            "improvement":"CLIPPED_SKELETON_WORLD_ORIENTATION_LOCAL_AND_SAVED_FK_PANELS",
            "adoption":"CANDIDATE_ONLY","sessions":rows,"new_solver_invocations":0,
            "training_eligible":False,"control_ground_truth":False,"physical_deployable":False}
    _write(OUT/"RESULT.json",result);print(json.dumps({"status":"PASS","sessions":len(rows),"frames":sum(r["frames"] for r in rows),"output":str(OUT)},ensure_ascii=False));return 0


if __name__=="__main__":raise SystemExit(main())
