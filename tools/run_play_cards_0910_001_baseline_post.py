#!/usr/bin/env python3
"""Build Mask/Object6D/Clean/Robot-input and three-way wrist QA for 0910_001.

This deliberately publishes a no-quality-gate development baseline.  The role
mask is Controller+MANUS geometry, the card mask is a deterministic colour/flow
track, Clean is per-frame Telea inpaint, and Object6D is observed-only stereo
depth geometry.  None of these outputs is physical or control ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

PROJECT=Path("/mnt/workspace/code/chaoyang")
MANO_NAMES=("wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip","index_mcp","index_pip","index_dip","index_tip","middle_mcp","middle_pip","middle_dip","middle_tip","ring_mcp","ring_pip","ring_dip","ring_tip","pinky_mcp","pinky_pip","pinky_dip","pinky_tip")
CHAINS=((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))
MANUS_TO_MANO=np.asarray((0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,18,19,20,21,23,24),np.int32)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def load_module(path:Path,name:str):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def project_equidis(points:np.ndarray,k:np.ndarray,d:np.ndarray)->np.ndarray:
    p=np.asarray(points,np.float64);x,y,z=p.T;r=np.hypot(x,y);theta=np.arctan2(r,z);dx=np.divide(x,r,out=np.zeros_like(x),where=r>1e-12);dy=np.divide(y,r,out=np.zeros_like(y),where=r>1e-12)
    theta2=theta*theta;rad=np.ones_like(theta);power=theta2.copy()
    for c in d[:6]:rad+=c*power;power*=theta2
    xd=theta*rad*dx;yd=theta*rad*dy;r2=xd*xd+yd*yd;p1,p2=d[6:]
    xt=xd+2*p1*xd*yd+p2*(r2+2*xd*xd);yt=yd+p1*(r2+2*yd*yd)+2*p2*xd*yd
    return np.column_stack((k[0,0]*xt+k[0,2],k[1,1]*yt+k[1,2]))

def stats_mm(values:np.ndarray)->dict:
    v=np.asarray(values,np.float64);v=v[np.isfinite(v)]
    if not len(v):return {"count":0,"mean_mm":None,"p50_mm":None,"p95_mm":None,"max_mm":None}
    return {"count":int(len(v)),"mean_mm":float(v.mean()*1000),"p50_mm":float(np.quantile(v,.5)*1000),"p95_mm":float(np.quantile(v,.95)*1000),"max_mm":float(v.max()*1000)}

def seed_card_mask(frame:np.ndarray)->np.ndarray:
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);purple=cv2.inRange(hsv,np.asarray((105,45,35),np.uint8),np.asarray((170,255,255),np.uint8))
    purple[:100]=0;purple[650:]=0;purple=cv2.morphologyEx(purple,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,lab,st,cent=cv2.connectedComponentsWithStats(purple)
    choices=[]
    for i in range(1,n):
        area=int(st[i,cv2.CC_STAT_AREA]);w=int(st[i,cv2.CC_STAT_WIDTH]);h=int(st[i,cv2.CC_STAT_HEIGHT]);cx,cy=cent[i]
        if 300<=area<=30000 and 15<=w<=220 and 15<=h<=240:choices.append((cx,i))
    if not choices:
        mask=np.zeros(frame.shape[:2],np.uint8);cv2.rectangle(mask,(850,150),(1080,400),255,-1);return mask
    i=max(choices)[1];mask=(lab==i).astype(np.uint8)*255
    x,y,w,h,_=st[i];cv2.rectangle(mask,(max(0,x-8),max(0,y-8)),(min(mask.shape[1]-1,x+w+8),min(mask.shape[0]-1,y+h+8)),255,-1)
    return mask

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--session-root",type=Path,required=True);ap.add_argument("--hawor-root",type=Path,required=True);ap.add_argument("--depth-root",type=Path,required=True);ap.add_argument("--output-root",type=Path,required=True);args=ap.parse_args()
    session=args.session_root.resolve(strict=True);hawor_root=args.hawor_root.resolve(strict=True);depth_root=args.depth_root.resolve(strict=True);out=args.output_root.resolve()
    if out.exists():raise RuntimeError(f"fresh output required: {out}")
    for name in ("mask/role","mask/object","mask/review","clean/frames","object6d/review","wrist/review"):(out/name).mkdir(parents=True,exist_ok=True)
    with np.load(hawor_root/"HAWOR_RAW_MANO21.npz",allow_pickle=False) as z:hawor={k:np.asarray(z[k]) for k in z.files}
    frame_jsons=sorted((session/"preprocess/all_data").glob("*/training_data.json"));rows=[json.loads(p.read_text()) for p in frame_jsons];n=len(rows);fps=float(rows[0]["metadata"]["fps"])
    c2w=np.asarray([r["metadata"]["c2w"] for r in rows],np.float64);k=np.asarray(rows[0]["metadata"]["k"],np.float64);d=np.asarray(rows[0]["metadata"]["d"],np.float64)
    sensor_world=np.empty((2,n,21,3),np.float32);controller_cam=np.empty((2,n,3),np.float32)
    for f,row in enumerate(rows):
        for s,name in enumerate(("left","right")):
            hand=row["entities"]["hands"][name];m=np.asarray(hand["manus25"]["keypoints_3d_world"],np.float64);sensor_world[s,f]=m[MANUS_TO_MANO]
            controller_cam[s,f]=np.asarray(hand["T_wrist_to_camera"],np.float64)[:3,3]
    w2c=np.linalg.inv(c2w);sensor_cam=np.einsum("fij,sfkj->sfki",w2c[:,:3,:3],sensor_world)+w2c[:,:3,3][None,:,None,:]
    sensor_uv=np.empty((2,n,21,2),np.float32)
    for s in range(2):
        for f in range(n):sensor_uv[s,f]=project_equidis(sensor_cam[s,f],k,d)
    hybrid_world=sensor_world.copy();hybrid_cam=sensor_cam.copy();hybrid_uv=sensor_uv.copy();prov=np.full((2,n),"CONTROLLER_MANUS_FALLBACK",dtype="<U32")
    direct=np.asarray(hawor["observed"],bool)
    for s in range(2):
        idx=np.flatnonzero(direct[s]);hybrid_world[s,idx]=hawor["joints_3d_world"][s,idx];hybrid_cam[s,idx]=hawor["joints_3d_camera"][s,idx];hybrid_uv[s,idx]=hawor["joints_2d"][s,idx];prov[s,idx]="OBSERVED"
    robot_input=out/"ROBOT_INPUT_HYBRID_MANO21.npz"
    np.savez_compressed(robot_input,joints_3d_world=hybrid_world,joints_3d_camera=hybrid_cam,joints_2d=hybrid_uv,observed=np.ones((2,n),bool),provenance=prov,c2w=c2w,intrinsics=np.repeat(k[None],n,axis=0),original_frame_indices=np.arange(n,dtype=np.int32),fps=np.asarray(fps),mano_joint_names=np.asarray(MANO_NAMES),mano_wrist_index=np.asarray(0,np.int32),mano_tip_indices=np.asarray((4,8,12,16,20),np.int32),anatomical_side_names=np.asarray(("left","right")))

    params=json.loads((session/"camera_params.json").read_text());cal=json.loads((PROJECT/"tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json").read_text());rr=np.asarray(cal["rectification_rotation_right"],np.float64)
    pico=load_module(PROJECT/"third_party/FoundationStereo/scripts/pico_stereo_depth.py","pico_post_0910")
    eye_d=params["right"];ii=eye_d["intrinsics"];eye=pico.Eye(np.asarray([ii["fx"],ii["fy"],ii["cx"],ii["cy"]],np.float64),np.asarray(eye_d["distortion"]["coeffs"],np.float64));kvirt=pico.virtual_intrinsics(1280,960,90);mx,my=pico.make_map(eye,rr,1280,960,kvirt);mx*=1280/params["width"];my*=960/params["height"]
    kd=np.asarray([[320.,0,319.5],[0,320.,239.5],[0,0,1.]])
    stereo_wrist=np.full((2,n,3),np.nan,np.float32);sample_uv=np.full((2,n,2),np.nan,np.float32)
    for f in range(n):
        with np.load(depth_root/"frames"/f"{f:06d}.npz") as z:depth=np.asarray(z["depth_m"])
        for s in range(2):
            if not direct[s,f]:continue
            pr=rr@hawor["joints_3d_camera"][s,f,0].astype(np.float64);uv=np.asarray((kd[0,0]*pr[0]/pr[2]+kd[0,2],kd[1,1]*pr[1]/pr[2]+kd[1,2]));sample_uv[s,f]=uv
            x,y=np.rint(uv).astype(int)
            if x<3 or y<3 or x>=637 or y>=477:continue
            patch=depth[y-3:y+4,x-3:x+4];vals=patch[np.isfinite(patch)]
            if len(vals)<5:continue
            z=float(np.median(vals));rect=np.asarray(((uv[0]-319.5)/320*z,(uv[1]-239.5)/320*z,z));stereo_wrist[s,f]=(rr.T@rect).astype(np.float32)
    hawor_wrist=np.asarray(hawor["joints_3d_camera"][:,:,0],np.float32)
    e_hc=np.linalg.norm(hawor_wrist-controller_cam,axis=2);e_sc=np.linalg.norm(stereo_wrist-controller_cam,axis=2);e_sh=np.linalg.norm(stereo_wrist-hawor_wrist,axis=2)

    video=session/"CameraRecord_play_cards_0910_001.mp4";cap=cv2.VideoCapture(str(video));frames=[]
    prev_gray=None;object_mask=None
    mask_video=out/"mask/MASK_BASELINE_REVIEW.mp4";clean_video=out/"clean/CLEAN_BASELINE.mp4";obj_video=out/"object6d/OBJECT6D_BASELINE_REVIEW.mp4";wrist_video=out/"wrist/WRIST_THREE_WAY_COMPARISON.mp4"
    wm=cv2.VideoWriter(str(mask_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1280,960));wc=cv2.VideoWriter(str(clean_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1280,960));wo=cv2.VideoWriter(str(obj_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1280,960));ww=cv2.VideoWriter(str(wrist_video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1280,960))
    obj_cam=np.full((n,4,4),np.nan,np.float64);obj_world=np.full_like(obj_cam,np.nan);obj_valid=np.zeros(n,bool);nearfar=np.full((n,2),np.nan,np.float32);visibility=np.zeros(n,np.float32);obj_centroid=np.full((n,2),np.nan,np.float32)
    colors=((255,100,20),(20,30,255))
    started=time.time()
    try:
        for f in range(n):
            ok,img=cap.read()
            if not ok:raise RuntimeError(f"raw ended {f}")
            frames.append(img.copy());gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if object_mask is None:object_mask=seed_card_mask(img)
            else:
                flow=cv2.calcOpticalFlowFarneback(gray,prev_gray,None,.5,3,25,3,5,1.2,0);yy,xx=np.indices(gray.shape,np.float32);object_mask=cv2.remap(object_mask,xx+flow[...,0],yy+flow[...,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);object_mask=cv2.morphologyEx(object_mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            prev_gray=gray
            role=np.zeros(img.shape[:2],np.uint8)
            for s in range(2):
                uv=np.rint(sensor_uv[s,f]).astype(np.int32)
                for chain in CHAINS:cv2.polylines(role,[uv[np.asarray(chain)]],False,255,34,cv2.LINE_AA)
                hull=cv2.convexHull(uv);cv2.fillConvexPoly(role,hull,255);cv2.circle(role,tuple(uv[0]),72,255,-1)
            role=cv2.dilate(role,np.ones((17,17),np.uint8));cv2.imwrite(str(out/"mask/role"/f"{f:06d}.png"),role);cv2.imwrite(str(out/"mask/object"/f"{f:06d}.png"),object_mask)
            overlay=img.copy();overlay[role>0]=(0.35*overlay[role>0]+0.65*np.asarray((255,200,0))).astype(np.uint8);overlay[object_mask>0]=(0.35*overlay[object_mask>0]+0.65*np.asarray((180,0,255))).astype(np.uint8);cv2.putText(overlay,f"Mask baseline f{f:03d} | cyan=Controller+MANUS roles magenta=card flow",(12,35),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA);wm.write(overlay)
            protect=cv2.dilate(object_mask,np.ones((15,15),np.uint8));remove=role.copy();remove[protect>0]=0;clean=cv2.inpaint(img,remove,7,cv2.INPAINT_TELEA);cv2.putText(clean,f"Clean baseline f{f:03d} | Telea | object protected",(12,35),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA);wc.write(clean);cv2.imwrite(str(out/"clean/frames"/f"{f:06d}.png"),clean)
            rect_mask=cv2.remap(object_mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT);half=cv2.resize(rect_mask,(640,480),interpolation=cv2.INTER_NEAREST)>0
            with np.load(depth_root/"frames"/f"{f:06d}.npz") as z:dep=np.asarray(z["depth_m"])
            yy2,xx2=np.where(half & np.isfinite(dep))
            panel=overlay.copy()
            if len(xx2)>=20:
                zs=dep[yy2,xx2];z=float(np.median(zs));u=float(np.median(xx2));v=float(np.median(yy2));rect=np.asarray(((u-319.5)/320*z,(v-239.5)/320*z,z));p_cam=rr.T@rect;t=np.eye(4);t[:3,:3]=rr.T;t[:3,3]=p_cam;obj_cam[f]=t;obj_world[f]=c2w[f]@t;obj_valid[f]=True;nearfar[f]=np.quantile(zs,(.05,.95));visibility[f]=float(len(xx2)/max(1,int(half.sum())));uvraw=project_equidis(p_cam[None],k,d)[0];obj_centroid[f]=uvraw;cv2.drawMarker(panel,tuple(np.rint(uvraw).astype(int)),(0,255,255),cv2.MARKER_CROSS,30,3);cv2.putText(panel,f"Object6D observed Z={z:.3f}m",(12,70),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,255),2,cv2.LINE_AA)
            else:cv2.putText(panel,"Object6D UNKNOWN: insufficient mask depth",(12,70),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,255),2,cv2.LINE_AA)
            wo.write(panel)
            wrist_panel=img.copy();labels=(("CTRL",(0,255,0)),("HAWOR",(0,180,255)),("FS-Z",(255,0,255)))
            for s in range(2):
                pts=(controller_cam[s,f],hawor_wrist[s,f],stereo_wrist[s,f])
                for (label,col),point in zip(labels,pts):
                    if np.isfinite(point).all():
                        uv=project_equidis(point[None],k,d)[0];cv2.drawMarker(wrist_panel,tuple(np.rint(uv).astype(int)),col,cv2.MARKER_CROSS,22,3);cv2.putText(wrist_panel,f"{label}-{('L','R')[s]}",(int(uv[0])+8,int(uv[1])-8),cv2.FONT_HERSHEY_SIMPLEX,.43,col,1,cv2.LINE_AA)
            cv2.rectangle(wrist_panel,(0,0),(1279,82),(0,0,0),-1);cv2.putText(wrist_panel,f"Wrist comparison f{f:03d}: green Controller ref | orange HaWoR | magenta HaWoR-ray+FS surface-Z",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
            text=[]
            for s,nm in enumerate(("L","R")):
                hc=e_hc[s,f]*1000 if np.isfinite(e_hc[s,f]) else math.nan;sc=e_sc[s,f]*1000 if np.isfinite(e_sc[s,f]) else math.nan;text.append(f"{nm}: H-C={hc:.1f}mm FS-C={sc:.1f}mm")
            cv2.putText(wrist_panel," | ".join(text),(12,60),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA);ww.write(wrist_panel)
            if f%30==0:print(json.dumps({"stage":"post","frame":f,"total":n}),flush=True)
    finally:
        cap.release();wm.release();wc.release();wo.release();ww.release()

    obj_npz=out/"object6d/OBJECT6D_BASELINE.npz";np.savez_compressed(obj_npz,frame_indices=np.arange(n,dtype=np.int32),valid=obj_valid,observed=obj_valid,visibility=visibility,physical_instance_id=np.where(obj_valid,0,-1).astype(np.int32),T_object_to_camera=obj_cam,T_object_to_world=obj_world,observed_near_far_optical_z_m=nearfar,analytic_near_far_optical_z_m=nearfar,object_size_m=np.asarray((.063,.088,.001),np.float32),object_centroid_raw_xy=obj_centroid)
    wrist_npz=out/"wrist/WRIST_COMPARISON.npz";np.savez_compressed(wrist_npz,controller_wrist_camera_m=controller_cam,hawor_wrist_camera_m=hawor_wrist,foundationstereo_surface_proxy_camera_m=stereo_wrist,hawor_minus_controller_norm_m=e_hc,foundationstereo_minus_controller_norm_m=e_sc,foundationstereo_minus_hawor_norm_m=e_sh,foundationstereo_sample_uv_rectified_right=sample_uv,hawor_observed=direct)
    metrics={}
    for s,name in enumerate(("left","right")):metrics[name]={"hawor_vs_controller":stats_mm(e_hc[s]),"foundationstereo_vs_controller":stats_mm(e_sc[s]),"foundationstereo_vs_hawor":stats_mm(e_sh[s]),"signed_z_hawor_minus_controller_mean_mm":float(np.nanmean((hawor_wrist[s,:,2]-controller_cam[s,:,2])*1000)),"signed_z_foundationstereo_minus_controller_mean_mm":float(np.nanmean((stereo_wrist[s,:,2]-controller_cam[s,:,2])*1000))}
    (out/"wrist/METRICS.json").write_text(json.dumps({"schema_version":"wrist-three-way-baseline-v1","reference":"CONTROLLER_TO_WRIST_ENGINEERING_CALIBRATION_NOT_GROUND_TRUTH","foundationstereo_semantics":"HAWOR_WRIST_RAY_PLUS_STEREO_VISIBLE_SURFACE_Z_NOT_ANATOMICAL_WRIST_CENTER","metrics":metrics,"control_ground_truth":False},indent=2,ensure_ascii=False)+"\n")
    result={"schema_version":"play-cards-0910-baseline-post-v1","status":"COMPLETE_BASELINE_NO_QUALITY_GATE","session_id":"play_cards_0910_001","frame_count":n,"hawor_direct_frames":{"left":int(direct[0].sum()),"right":int(direct[1].sum())},"robot_input":{"path":str(robot_input),"sha256":sha(robot_input)},"mask_review":{"path":str(mask_video),"sha256":sha(mask_video)},"clean_video":{"path":str(clean_video),"sha256":sha(clean_video)},"object6d":{"npz":str(obj_npz),"observed_frames":int(obj_valid.sum()),"review":str(obj_video)},"wrist":{"npz":str(wrist_npz),"metrics":str(out/"wrist/METRICS.json"),"review":str(wrist_video)},"methods":{"mask":"CONTROLLER_MANUS_GEOMETRY_PLUS_PURPLE_SEED_DENSE_FLOW_CARD","clean":"OPENCV_TELEA_PER_FRAME_BASELINE","robot_input":"HAWOR_DIRECT_ELSE_CONTROLLER_MANUS_FALLBACK"},"control_ground_truth":False,"physical_deployment":False,"claim_limit":"No-quality-gate development baseline. Controller wrist is an engineering reference, not external truth.","wall_seconds":time.time()-started}
    (out/"RESULT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    hawor_result={"schema_version":"baseline-hawor-result-adapter-v1","status":"COMPLETE_BASELINE","session_id":"play_cards_0910_001","gates":{"robot_numeric_candidate":"DEVELOPMENT_HYBRID_SENSOR_FALLBACK"},"control_ground_truth":False};(out/"HAWOR_RESULT_FOR_ROBOT.json").write_text(json.dumps(hawor_result,indent=2)+"\n")
    admission={"schema_version":"baseline-source-admission-v1","combined":{"upstream_terminal_status":"DEVELOPMENT_BASELINE_NO_QUALITY_GATE"},"control_ground_truth":False};(out/"SOURCE_ADMISSION_FOR_ROBOT.json").write_text(json.dumps(admission,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"metrics":metrics},ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
