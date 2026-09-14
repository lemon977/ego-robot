#!/usr/bin/env python3
"""Generic CPU temporal successor for an existing bounded MANO21 track."""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

import run_hawor_bounded_parameter_successor as base_tool

PROJECT = Path(__file__).resolve().parents[1]
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
ROOT_TRANS_CAP_M = 0.006
ROOT_ROT_CAP = np.deg2rad(4.0)
POSE_ROT_CAP = np.deg2rad(6.0)
ALPHAS = (1.0, .75, .5, .35, .25, .15, .1, .06, .03)
TIP_IDX = np.asarray([4, 8, 12, 16, 20])


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha(path)}


def atomic_json(path: Path, value: Any) -> None:
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)


def smooth_order3(values: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    values=np.asarray(values,dtype=np.float64); n=len(values)
    if n<4:return values.copy()
    d=sparse.diags((-np.ones(n-3),3*np.ones(n-3),-3*np.ones(n-3),np.ones(n-3)),(0,1,2,3),shape=(n-3,n),format="csc")
    w=sparse.diags(np.maximum(weights,1e-5),format="csc")
    rhs=np.maximum(weights,1e-5)[:,None]*values.reshape(n,-1)
    system=w+strength*(d.T@d)
    out=np.column_stack([spsolve(system,rhs[:,i]) for i in range(rhs.shape[1])])
    return out.reshape(values.shape)


def smooth_so3(mats: np.ndarray, weights: np.ndarray, strength: float) -> np.ndarray:
    q=base_tool.canonical_quaternions(mats)
    q=smooth_order3(q,weights,strength)
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    return Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(mats.shape)


def proposal(track: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    out={k:track[k].copy() for k in ("root_translation_camera","root_orient_camera","hand_pose_rotmat","betas")}
    for side in range(2):
        for a,b in base_tool.contiguous_true_segments(track["observed"][side]):
            w=np.clip(track["detector_confidence"][side,a:b],.05,1.)**2
            out["root_translation_camera"][side,a:b]=smooth_order3(track["root_translation_camera"][side,a:b],w,1.25)
            out["root_orient_camera"][side,a:b]=smooth_so3(track["root_orient_camera"][side,a:b],w,.65)
            out["hand_pose_rotmat"][side,a:b]=smooth_so3(track["hand_pose_rotmat"][side,a:b],w,.22)
    obs=track["observed"]
    out["root_translation_camera"]=base_tool.bound_vectors(track["root_translation_camera"],out["root_translation_camera"],obs,ROOT_TRANS_CAP_M)
    out["root_orient_camera"]=base_tool.bound_rotation_updates(track["root_orient_camera"],out["root_orient_camera"],obs,ROOT_ROT_CAP)
    out["hand_pose_rotmat"]=base_tool.bound_rotation_updates(track["hand_pose_rotmat"],out["hand_pose_rotmat"],np.repeat(obs[...,None],15,axis=2),POSE_ROT_CAP)
    return out


def q(v: np.ndarray, p: float) -> float|None:
    v=np.asarray(v);v=v[np.isfinite(v)]
    return float(np.quantile(v,p)) if len(v) else None


def xyz_diffs(x:np.ndarray,m:np.ndarray)->dict[str,Any]:
    vals={1:[],2:[],3:[]}
    for a,b in base_tool.contiguous_true_segments(m):
        for n in vals:
            if b-a>n: vals[n].append(np.linalg.norm(np.diff(x[a:b],n=n,axis=0),axis=-1).reshape(-1)*1000)
    return {str(n):{"p95":q(np.concatenate(v) if v else np.array([]),.95),"max":q(np.concatenate(v) if v else np.array([]),1.)} for n,v in vals.items()}


def rot_diffs(r:np.ndarray,m:np.ndarray)->dict[str,Any]:
    vals={1:[],2:[],3:[]}
    r=r.reshape(len(m),-1,3,3)
    for a,b in base_tool.contiguous_true_segments(m):
        if b-a<2:continue
        rv=Rotation.from_matrix((np.transpose(r[a:b-1],(0,1,3,2))@r[a+1:b]).reshape(-1,3,3)).as_rotvec().reshape(b-a-1,-1,3)
        vals[1].append(np.linalg.norm(rv,axis=-1).reshape(-1)*180/np.pi)
        if len(rv)>1: vals[2].append(np.linalg.norm(np.diff(rv,axis=0),axis=-1).reshape(-1)*180/np.pi)
        if len(rv)>2: vals[3].append(np.linalg.norm(np.diff(rv,n=2,axis=0),axis=-1).reshape(-1)*180/np.pi)
    return {str(n):{"p95":q(np.concatenate(v) if v else np.array([]),.95),"max":q(np.concatenate(v) if v else np.array([]),1.)} for n,v in vals.items()}


def spatial_span(x:np.ndarray,m:np.ndarray)->float:
    v=x[m].reshape(-1,3)
    return float(np.linalg.norm(np.quantile(v,.95,axis=0)-np.quantile(v,.05,axis=0))*1000) if len(v) else 0.


def evaluate(src:dict[str,np.ndarray],j:dict[str,np.ndarray],params:dict[str,np.ndarray])->tuple[dict[str,Any],list[str]]:
    out={"sides":{}};fail=[]
    for si,name in enumerate(("left","right")):
        m=src["observed"][si]
        err=np.linalg.norm(j["joints_2d"][si,m]-src["joints_2d"][si,m],axis=-1)
        s={"observed":int(m.sum()),"segments":len(base_tool.contiguous_true_segments(m)),
           "reprojection_px":{"p95":q(err,.95),"max":q(err,1.)},
           "wrist":{"input":xyz_diffs(src["joints_3d_world"][si,:,0],m),"candidate":xyz_diffs(j["joints_3d_world"][si,:,0],m)},
           "tips":{"input":xyz_diffs(src["joints_3d_world"][si][:,TIP_IDX],m),"candidate":xyz_diffs(j["joints_3d_world"][si][:,TIP_IDX],m)},
           "root_so3_deg":{"input":rot_diffs(src["root_orient_camera"][si],m),"candidate":rot_diffs(params["root_orient_camera"][si],m)},
           "pose_so3_deg":{"input":rot_diffs(src["hand_pose_rotmat"][si],m),"candidate":rot_diffs(params["hand_pose_rotmat"][si],m)}}
        for part,idx in (("wrist",[0]),("tips",TIP_IDX)):
            a=spatial_span(src["joints_3d_world"][si][:,idx],m);b=spatial_span(j["joints_3d_world"][si][:,idx],m)
            s[part]["spatial_span_input_mm"]=a;s[part]["spatial_span_candidate_mm"]=b;s[part]["span_ratio"]=b/max(a,1e-9)
            if a>10 and b/a<.90:fail.append(f"{name}:{part}:ACTION_SPAN_LT_90PCT")
            if s[part]["candidate"]["3"]["p95"]>s[part]["input"]["3"]["p95"]*1.001:fail.append(f"{name}:{part}:JERK_REGRESSION")
        if q(err,.95)>2.0 or q(err,1.)>10.:fail.append(f"{name}:REPROJECTION")
        out["sides"][name]=s
    updates=base_tool.parameter_update_summary(src,params);out["parameter_updates"]=updates
    if updates["root_translation_update_max_mm"]>6.001:fail.append("ROOT_TRANSLATION_CAP")
    if updates["root_rotation_update_max_deg"]>4.001:fail.append("ROOT_ROTATION_CAP")
    if updates["pose_rotation_update_max_deg"]>6.001:fail.append("POSE_ROTATION_CAP")
    return out,fail


def save_npz(path:Path,src:dict[str,np.ndarray],params:dict[str,np.ndarray],j:dict[str,np.ndarray],alpha:float,input_sha:str)->None:
    payload={k:v for k,v in src.items() if not k.startswith("_")}
    payload.update(params);payload.update(j)
    payload["provenance"]=np.where(src["observed"],"TEMPORAL_SO3_JERK_V3","MISSING").astype("U24")
    payload["method"]=np.asarray("GENERIC_SEGMENT_LOCAL_SO3_THIRD_DIFFERENCE_V3",dtype="U64")
    payload["source_bounded_npz_sha256"]=np.asarray(input_sha,dtype="U64")
    payload["temporal_backtracking_alpha"]=np.asarray(alpha,dtype=np.float64)
    np.savez_compressed(path,**payload)


def render(video:Path,j:dict[str,np.ndarray],out:Path,session:str,ab_input:dict[str,np.ndarray]|None=None,limit:int|None=None)->dict[str,Any]:
    cap=cv2.VideoCapture(str(video));w=int(cap.get(3));h=int(cap.get(4));fps=float(cap.get(5));n=j["joints_2d"].shape[1];n=min(n,limit or n)
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}","-r",f"{fps:.8f}","-i","-","-an","-c:v","libx264","-preset","veryfast","-threads","1","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
    enc=subprocess.Popen(cmd,stdin=subprocess.PIPE);font=ImageFont.truetype(str(FONT),18);count=0
    while count<n:
        ok,frame=cap.read()
        if not ok:break
        cv2.rectangle(frame,(0,0),(w,72),(0,0,0),-1)
        if ab_input is not None:base_tool.draw_skeleton(frame,ab_input["joints_2d"][:,count],((110,110,40),(110,40,110)),1)
        base_tool.draw_skeleton(frame,j["joints_2d"][:,count],((255,255,0),(255,0,255)),3)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
        label="V2细暗 + V3亮色 A/B" if ab_input is not None else "V3 bounded-only（无RAW抖动叠加）"
        d.text((10,8),f"{session} | {count+1}/{n} | {label}",font=font,fill=(255,255,255));d.text((10,40),"青=左手  紫=右手  保持observed gap、2D守卫与动作幅度",font=font,fill=(240,220,120))
        enc.stdin.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR).tobytes());count+=1
    cap.release();enc.stdin.close();rc=enc.wait()
    if rc or count!=n:raise RuntimeError(f"render failed {rc} {count}/{n}")
    # Hash actual decoded RGB frames for downstream frame identity.
    check=cv2.VideoCapture(str(out));hashes=[]
    while True:
        ok,f=check.read()
        if not ok:break
        hashes.append(hashlib.sha256(cv2.cvtColor(f,cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
    check.release()
    if len(hashes)!=n:raise RuntimeError("decoded output count mismatch")
    return {"frames":n,"fps":fps,"resolution":[w,h],"decoded_rgb_sha256":hashes}


def process(row:dict[str,Any],root:Path)->dict[str,Any]:
    task,session=row["task"],row["session"]
    final=root/task/session;stage=root/task/f".{session}.stage.{os.getpid()}";stage.mkdir(parents=True)
    try:
        inp=Path(row["bounded_npz"]);video=Path(row["source_video"])
        if sha(inp)!=row["bounded_sha256"] or sha(video)!=row["source_video_sha256"]:raise RuntimeError("input SHA mismatch")
        with np.load(inp,allow_pickle=False) as z:src={k:np.asarray(z[k]) for k in z.files}
        prop=proposal(src);chosen=None;attempts=[]
        for alpha in ALPHAS:
            params=base_tool.interpolate_parameters(src,prop,alpha);j=base_tool.materialize(src,params);metrics,fail=evaluate(src,j,params);attempts.append({"alpha":alpha,"failures":fail})
            if not fail:chosen=(alpha,params,j,metrics);break
        if chosen is None:raise RuntimeError(f"no safe successor alpha: {attempts}")
        alpha,params,j,metrics=chosen
        npz=stage/"HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz";save_npz(npz,src,params,j,alpha,row["bounded_sha256"])
        full=stage/"HAWOR_TEMPORAL_SO3_BOUNDED_ONLY_中文全片复核.mp4";vr=render(video,j,full,session)
        ab=stage/"HAWOR_BOUNDED_V2_VS_TEMPORAL_V3_48F.mp4";abr=render(video,j,ab,session,src,48)
        manifest=stage/"FRAME_MANIFEST.jsonl"
        with manifest.open("w") as f:
            for i,h in enumerate(vr["decoded_rgb_sha256"]):f.write(json.dumps({"frame":i,"source_frame":i,"bounded_video_decoded_rgb_sha256":h},separators=(",",":"))+"\n")
        result={"schema_version":"hawor-temporal-so3-jerk-successor-v1","status":"PASS_NUMERIC_NEEDS_HUMAN_REVIEW","task":task,"session":session,"frame_count":int(src["observed"].shape[1]),"selected_alpha":alpha,
                "algorithm":{"domain":"root translation + root/15 pose SO(3)","prior":"confidence-weighted third difference within contiguous observed segments","cross_gap_smoothing":False,"caps":{"root_translation_mm":6,"root_rotation_deg":4,"pose_rotation_deg":6},"reprojection_p95_px_max":2,"action_span_ratio_min":.9,"session_specific_rules":False},
                "metrics":metrics,"attempts":attempts,"inputs":{"bounded_npz":artifact(inp),"source_video":artifact(video)},"outputs":{"npz":artifact(npz),"bounded_only_video":artifact(full),"ab48_video":artifact(ab),"frame_manifest":artifact(manifest)},"frame_manifest_contract":{"frames":len(vr["decoded_rgb_sha256"]),"strict_range":[0,len(vr["decoded_rgb_sha256"])-1],"no_loop_or_hold":True,"hash_domain":"decoded RGB bytes after MP4 encoding"},"claim_limit":"Temporal successor and review-only video; human visual approval still required."}
        atomic_json(stage/"RESULT.json",result);final.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,final)
        d=json.loads((final/"RESULT.json").read_text());
        for v in d["outputs"].values():v["path"]=str(final/Path(v["path"]).name)
        atomic_json(final/"RESULT.json",d)
        return {"task":task,"session":session,"status":d["status"],"result":artifact(final/"RESULT.json")}
    except Exception:
        shutil.rmtree(stage,ignore_errors=True);raise


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--contract",type=Path,required=True);ap.add_argument("--output-root",type=Path,required=True);a=ap.parse_args()
    if a.output_root.exists():raise SystemExit("fresh output root required")
    c=json.loads(a.contract.read_text());rows=[]
    for r in c["sessions"]:rows.append(process(r,a.output_root))
    atomic_json(a.output_root/"RESULT.json",{"schema_version":"six-session-hawor-temporal-successor-v1","status":"PASS_NUMERIC_NEEDS_HUMAN_REVIEW","sessions":rows})
    print(json.dumps(rows,ensure_ascii=False,indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
