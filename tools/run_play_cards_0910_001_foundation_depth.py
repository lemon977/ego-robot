#!/usr/bin/env python3
"""FoundationStereo right-reference metric depth for play_cards_0910_001.

The selected RGB is physical camera `right`/sourceIndex0.  The calibrated pair
is rectified first; swap+horizontal-flip makes FoundationStereo predict positive
disparity in the right-camera reference.  Flipping the result back yields a
right-rectified optical-Z map.  This is engineering depth, not external truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT=Path("/mnt/workspace/code/chaoyang")
PICO_PATH=PROJECT/"third_party/FoundationStereo/scripts/pico_stereo_depth.py"
CALIBRATION=PROJECT/"tasks/control/runs/20260907_stereo_object6d_formal_prepared_v1/calibration.json"

def load_module(path:Path,name:str):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--session-root",type=Path,required=True);ap.add_argument("--output-root",type=Path,required=True);args=ap.parse_args()
    session=args.session_root.resolve(strict=True);out=args.output_root.resolve()
    if out.exists():raise RuntimeError(f"fresh output required: {out}")
    (out/"frames").mkdir(parents=True)
    pico=load_module(PICO_PATH,"pico_0910_baseline")
    worker=load_module(PROJECT/"tools/run_exact78_foundationstereo_corrected_depth_worker.py","depth_worker_0910_baseline")
    params=json.loads((session/"camera_params.json").read_text())
    ew,eh=int(params["width"]),int(params["height"])
    eyes=[]
    for name in ("left","right"):
        d=params[name];i=d["intrinsics"]
        eyes.append(pico.Eye(np.asarray([i["fx"],i["fy"],i["cx"],i["cy"]],np.float64),np.asarray(d["distortion"]["coeffs"],np.float64)))
    el=np.asarray(params["extrinsics"]["left"],np.float64);er=np.asarray(params["extrinsics"]["right"],np.float64)
    cl=-el[:3,:3].T@el[:3,3];cr=-er[:3,:3].T@er[:3,3];baseline=float(np.linalg.norm(cr-cl))
    calibration=json.loads(CALIBRATION.read_text())
    k0=pico.virtual_intrinsics(1280,960,90.0)
    rl=np.asarray(calibration["rectification_rotation_left"],np.float64);rr=np.asarray(calibration["rectification_rotation_right"],np.float64)
    maps=[pico.make_map(eyes[0],rl,1280,960,k0),pico.make_map(eyes[1],rr,1280,960,k0)]
    stereo=session/"source_stereo/CameraRecord_play_cards_0910_001_stereo.mp4"
    selected=session/"CameraRecord_play_cards_0910_001.mp4"
    cap=cv2.VideoCapture(str(stereo));rgb=cv2.VideoCapture(str(selected))
    frame_count=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)));fps=float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count!=191:raise RuntimeError(f"unexpected frame count {frame_count}")
    model=worker.Model();review=out/"FOUNDATIONSTEREO_RIGHT_DEPTH_REVIEW.mp4";vw=cv2.VideoWriter(str(review),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1280,480))
    rows=[];started=time.time()
    try:
        for frame in range(frame_count):
            ok,sbs=cap.read();ok2,raw=rgb.read()
            if not ok or not ok2:raise RuntimeError(f"decode ended at {frame}")
            left_rect,right_rect=pico.remap_pair(sbs,ew,maps)
            disparity_f,depth_f,valid_f=model.infer(cv2.flip(right_rect,1),cv2.flip(left_rect,1),baseline)
            disparity=cv2.flip(disparity_f,1);depth=cv2.flip(depth_f,1);valid=cv2.flip(valid_f.astype(np.uint8),1).astype(bool)
            target=out/"frames"/f"{frame:06d}.npz"
            np.savez_compressed(target,frame_id=np.asarray(frame,np.int32),disparity_px=disparity.astype(np.float32),depth_m=depth.astype(np.float32),valid=valid,scaled_intrinsics=worker.K_DEPTH,rectification_rotation_right=rr)
            finite=depth[valid]
            rows.append({"frame":frame,"valid_pixels":int(valid.sum()),"depth_p50_m":float(np.median(finite)) if finite.size else None})
            scalar=np.zeros((480,640),np.uint8)
            scalar[valid]=np.rint(255*(1-np.clip((depth[valid]-.1)/2.9,0,1))).astype(np.uint8)
            colour=cv2.applyColorMap(scalar,cv2.COLORMAP_TURBO);colour[~valid]=0
            panel=np.concatenate((cv2.resize(raw,(640,480)),colour),axis=1)
            cv2.rectangle(panel,(0,0),(1279,52),(0,0,0),-1)
            cv2.putText(panel,f"frame {frame:03d} | Raw selected right",(12,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(panel,"FoundationStereo right-reference optical-Z",(650,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
            vw.write(panel)
            if frame%20==0:print(json.dumps({"stage":"depth","frame":frame,"total":frame_count}),flush=True)
    finally:
        cap.release();rgb.release();vw.release()
    meta={"schema_version":"play-cards-0910-foundationstereo-right-depth-v1","status":"COMPLETE_BASELINE_NO_QUALITY_GATE","session_id":"play_cards_0910_001","frame_count":frame_count,"fps":fps,"baseline_m":baseline,"depth_formula":"Z=320*B/disparity","depth_reference":"PHYSICAL_RIGHT_RECTIFIED_OPTICAL","selected_rgb_reference":"PHYSICAL_RIGHT_RAW_PASSTHROUGH","swap_flip_method":True,"rectification_rotation_right":rr.tolist(),"frames":rows,"review":{"path":str(review),"bytes":review.stat().st_size,"sha256":sha(review)},"control_ground_truth":False,"physical_deployment":False,"claim_limit":"FoundationStereo metric optical-Z engineering estimate; no external accuracy truth.","wall_seconds":time.time()-started}
    (out/"RESULT.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":meta["status"],"frames":frame_count,"wall_seconds":meta["wall_seconds"]}),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
