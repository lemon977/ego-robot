#!/usr/bin/env python3
"""Build a causal Controller/MANUS wrist + Stereo relative-Z correction canary."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from chaoyang.ops.build_wrist_fusion_pico_hawor_stereo_canary_v1 import COLORS, artifact, project, put_text


SIDES=("left","right")


def finite_vec(value):
    if not isinstance(value,list) or len(value)!=3: return None
    v=np.asarray(value,np.float64); return v if np.isfinite(v).all() else None


def stats(values):
    values=np.asarray(values,np.float64); values=values[np.isfinite(values)]
    steps=np.abs(np.diff(values))
    return {'valid_frames':int(len(values)),'range_mm':float(np.ptp(values)) if len(values) else None,'step_p95_mm':float(np.percentile(steps,95)) if len(steps) else None}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--session-root',type=Path,required=True); p.add_argument('--h1-sidecar',type=Path,required=True); p.add_argument('--comparison-json',type=Path,required=True); p.add_argument('--raw-video',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--prefix-valid',type=int,default=30); args=p.parse_args()
    session=args.session_root.resolve(strict=True); h1_path=args.h1_sidecar.resolve(strict=True); comparison_path=args.comparison_json.resolve(strict=True); raw_path=args.raw_video.resolve(strict=True); out=args.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    h1=np.load(h1_path,allow_pickle=True); comparison=json.loads(comparison_path.read_text(encoding='utf-8')); records=comparison['per_frame_delta_xyz_mm']; T=len(records)
    if h1['T_wrist_to_camera'].shape[:2]!=(T,2): raise RuntimeError('frame count mismatch')
    controller=np.transpose(h1['T_wrist_to_camera'][:,:,:3,3],(1,0,2)).astype(np.float64)
    stereo=np.full_like(controller,np.nan)
    for frame,record in enumerate(records):
        for side_idx,side in enumerate(SIDES):
            delta=finite_vec(record[side].get('stereo_minus_controller'))
            if delta is not None: stereo[side_idx,frame]=controller[side_idx,frame]+delta/1000.0
    corrected=controller.copy(); correction_mm=np.full((2,T),np.nan,np.float64); calibration={}
    for side_idx,side in enumerate(SIDES):
        gap=(stereo[side_idx,:,2]-controller[side_idx,:,2])*1000.0; valid=np.flatnonzero(np.isfinite(gap))[:args.prefix_valid]
        if len(valid)<5: raise RuntimeError(f'not enough stereo frames for {side}')
        gap0=float(np.median(gap[valid])); prev=0.0; history=[]
        for frame in range(T):
            if not np.isfinite(gap[frame]):
                correction_mm[side_idx,frame]=prev; corrected[side_idx,frame,2]+=prev/1000.0; continue
            history.append(float(gap[frame]))
            robust_gap=float(np.median(history[-15:]))
            innovation=float(np.clip(robust_gap-gap0,-30.0,30.0))
            measurement=.15*innovation
            current=.15*measurement+.85*prev
            correction_mm[side_idx,frame]=current; corrected[side_idx,frame,2]+=current/1000.0; prev=current
        calibration[side]={'prefix_valid_frames':valid.tolist(),'median_stereo_surface_minus_controller_z_mm':gap0,'causal_rolling_median_frames':15,'stereo_relative_innovation_clip_mm':30.0,'stereo_weight':0.15,'causal_correction_alpha':0.15,'correction_abs_max_mm':float(np.max(np.abs(correction_mm[side_idx])))}
    joints=np.asarray(h1['joint_xyz_camera_m'],np.float32).copy(); fused=joints.copy()
    for side_idx in range(2): fused[:,side_idx,:,2]+=correction_mm[side_idx,:,None].astype(np.float32)/1000.0
    npz=out/'HAND21_CONTROLLER_STEREO_RELATIVE_Z_CANARY.npz'; np.savez_compressed(npz,joint_xyz_camera_m_original=joints,joint_xyz_camera_m_corrected=fused,T_wrist_to_camera_original=h1['T_wrist_to_camera'],wrist_xyz_camera_corrected=np.transpose(corrected,(1,0,2)),stereo_surface_xyz_camera=np.transpose(stereo,(1,0,2)),stereo_relative_z_correction_mm=np.transpose(correction_mm,(1,0)),control_ground_truth=np.bool_(False),external_truth=np.bool_(False))
    rows=[json.loads(x.read_text(encoding='utf-8')) for x in sorted((session/'preprocess/all_data').glob('*/training_data.json'))]
    cap=cv2.VideoCapture(str(raw_path)); fps=float(comparison['fps']); video=out/'PLAY_CARDS_0910_001_Controller_MANUS_Stereo相对Z修正Canary_全片.mp4'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    colors={'Controller':(42,208,70),'Stereo surface':(220,50,220),'Corrected':(255,240,60)}
    gap_all=(stereo[:,:,2]-controller[:,:,2])*1000.0
    for frame in range(T):
        ok,raw=cap.read()
        if not ok: raise RuntimeError(f'video ended at {frame}')
        canvas=np.full((1080,1920,3),24,np.uint8); rgb=cv2.resize(raw,(900,675),interpolation=cv2.INTER_AREA)
        K=np.asarray(rows[frame]['metadata']['k'],np.float64).copy(); K[0]*=rgb.shape[1]/float(rows[frame]['metadata']['w']); K[1]*=rgb.shape[0]/float(rows[frame]['metadata']['h'])
        for side_idx in range(2):
            for name,point in [('Controller',controller[side_idx,frame]),('Stereo surface',stereo[side_idx,frame]),('Corrected',corrected[side_idx,frame])]:
                uv=project(point,K)
                if uv is None: continue
                if name=='Controller': cv2.circle(rgb,uv,12,colors[name],-1,cv2.LINE_AA)
                elif name=='Stereo surface': cv2.circle(rgb,uv,19,colors[name],3,cv2.LINE_AA)
                else: cv2.drawMarker(rgb,uv,colors[name],cv2.MARKER_TILTED_CROSS,26,3,cv2.LINE_AA)
        canvas[90:765,30:930]=rgb
        graph=(990,135,1880,720); cv2.rectangle(canvas,graph[:2],graph[2:],(80,80,80),1)
        max_abs=max(40,float(np.nanpercentile(np.abs(gap_all),95))); cr=(-max_abs,max_abs)
        for side_idx in range(2):
            for name,series,color in [('Stereo-Controller gap',gap_all[side_idx],(220,50,220)),('applied correction',correction_mm[side_idx],(255,240,60))]:
                ids=np.arange(frame+1); valid=np.isfinite(series[:frame+1]); ids=ids[valid]
                if len(ids)>1:
                    xx=graph[0]+ids/max(T-1,1)*(graph[2]-graph[0]); yy=graph[3]-(series[ids]-cr[0])/(cr[1]-cr[0])*(graph[3]-graph[1]); cv2.polylines(canvas,[np.c_[xx,yy].astype(np.int32)],False,color,2 if side_idx==0 else 1,cv2.LINE_AA)
        zero=int(graph[3]-(0-cr[0])/(cr[1]-cr[0])*(graph[3]-graph[1])); cv2.line(canvas,(graph[0],zero),(graph[2],zero),(100,100,100),1)
        cx=int(graph[0]+frame/max(T-1,1)*(graph[2]-graph[0])); cv2.line(canvas,(cx,graph[1]),(cx,graph[3]),(255,255,255),1)
        text_rows=[(30,22,'新数据 wrist-depth canary：Controller/MANUS主锚 + Stereo相对Z限幅修正',(255,255,255),31),(30,770,'RGB：绿色实心=Controller；紫圈=Stereo表面；黄叉=修正腕',(230,230,230),22),(1010,92,'曲线：紫=表面与腕的原始Z间隔；黄=实际施加的小幅修正',(230,230,230),22),(1500,25,f'帧 {frame:03d}/{T-1:03d}  {frame/fps:5.2f}s',(255,255,255),24)]
        y=750
        for side_idx,cn in enumerate(('左','右')):
            gapv=gap_all[side_idx,frame]; corv=correction_mm[side_idx,frame]
            text_rows.append((1010,y,f'{cn}手：surface-wrist gap={gapv:+.1f} mm，施加 correction={corv:+.1f} mm',(235,235,235),21)); y+=36
        text_rows.append((30,820,'注意：三点在RGB上可重合，因为Stereo点位于Controller wrist同一射线；差异必须看右侧毫米曲线。',(100,220,255),22))
        canvas=put_text(canvas,text_rows); writer.write(canvas)
    writer.release(); cap.release(); subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    result={'schema_version':'controller-manus-stereo-relative-z-canary-v2','status':'DEVELOPMENT_CANARY_GENERATED','session':'play_cards_0910_001','frame_count':T,'method':'Controller/MANUS wrist anchor plus prefix-calibrated causal 15-frame Stereo gap median, clipped slow-drift correction, weight=0.15 and EMA alpha=0.15','calibration':calibration,'metrics':{side:{'controller_z':stats(controller[i,:,2]*1000),'corrected_z':stats(corrected[i,:,2]*1000),'stereo_surface_z':stats(stereo[i,:,2]*1000),'applied_correction':stats(correction_mm[i])} for i,side in enumerate(SIDES)},'inputs':{'h1_sidecar':artifact(h1_path),'comparison':artifact(comparison_path),'raw_video':artifact(raw_path)},'outputs':{'npz':artifact(npz),'video':artifact(video)},'claim_limit':'Development relative-depth correction only. Controller remains the primary wrist anchor; Stereo is a visible surface and does not establish anatomical wrist or external metric truth.'}
    result_path=out/'RESULT.json'; result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'result':str(result_path),'video':str(video),'calibration':calibration,'metrics':result['metrics']},ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
