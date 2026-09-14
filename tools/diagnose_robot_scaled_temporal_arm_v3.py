#!/usr/bin/env python3
"""CPU-only full-session temporal arm feasibility for morphology-scaled motion."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from tools import render_poker_symmetric_chirality_flange_successor as old
from tools import render_poker_static_closure_task_translation_successor as taskfit
from tools import render_same_side_world_temporal_review as temporal

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def margins(assets,side,q):
 values={n:0. for row in old.official.ARM_JOINT_NAMES for n in row}; values.update(dict(zip(old.official.ARM_JOINT_NAMES[side],q,strict=True)))
 fk=old.official.forward_kinematics(assets.tianji,values); suf=('L','R')[side]; sign=(1.,-1.)[side]
 l3=fk[f'Link3_{suf}'][:3,3]; l5=fk[f'Link5_{suf}'][:3,3]
 return np.array((sign*l3[1]-.20,sign*l5[1]-.04,l5[0]-.04))

def error(assets,side,q,target):
 actual=old.official._tool_fk(assets,side,q)
 p=float(np.linalg.norm(actual[:3,3]-target[:3,3])*1000)
 r=float(np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec())))
 return p,r

def scaled(root0,root,scale):
 out=root.copy(); out[:3,3]=root0[:3,3]+scale*(root[:3,3]-root0[:3,3])
 rel=Rotation.from_matrix(root0[:3,:3].T@root[:3,:3])
 out[:3,:3]=root0[:3,:3]@Rotation.from_rotvec(scale*rel.as_rotvec()).as_matrix(); return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--trajectory',type=Path,required=True); ap.add_argument('--motion-scale',type=float,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 p=a.trajectory.resolve(strict=True)
 with np.load(p,allow_pickle=False) as z:
  oldq=np.asarray(z['q_arm']); roots=np.asarray(z['T_target_hand_root_world']); wb=np.asarray(z['T_world_base']); mounts=np.asarray(z['T_tool_hand_root'])
 assets=old.load_pinned_robot_assets(PROJECT); lower,upper=taskfit.arm_limits(assets); count=len(oldq)
 q=np.empty_like(oldq); q[0]=oldq[0]; rows=[]
 for frame in range(count):
  for side in range(2):
   target=np.linalg.inv(wb)@scaled(roots[0,side],roots[frame,side],a.motion_scale)@np.linalg.inv(mounts[side])
   if frame==0: chosen=q[0,side]; policy='FRAME0_LOCK'
   else:
    lo,hi=temporal.bounded_limits(lower[side],upper[side],q[frame-1,side],None if frame==1 else q[frame-2,side],temporal.ARM_SOLVER_STEP_LIMIT)
    seeds=[np.clip(q[frame-1,side],lo,hi),np.clip(oldq[frame,side],lo,hi)]
    attempts=[]
    for seed_index,seed in enumerate(seeds):
     def residual(x):
      pose=old.official._pose_residual(old.official._tool_fk(assets,side,x),target)
      branch=np.minimum(margins(assets,side,x)-.003,0.)/.001
      return np.concatenate((pose,branch,.01*(x-q[frame-1,side])))
     free=hi-lo>1e-12; x=seed.copy(); n=0
     if np.any(free):
      sol=least_squares(lambda y:residual(np.where(free,y,x)),x[free],bounds=(lo[free],hi[free]),max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10)
      x[free]=sol.x;n=sol.nfev
     pos,rot=error(assets,side,x,target); m=margins(assets,side,x)
     attempts.append((not(pos<=10 and rot<=5 and min(m)>=-1e-8),max(0.,-min(m)),max(0.,pos-10)+max(0.,rot-5),pos+rot,x,n,seed_index))
    *_,chosen,n,seed_index=min(attempts,key=lambda x:x[:4]); policy=f'BOUNDED_BRANCH_MULTISEED_{seed_index}'
    q[frame,side]=chosen; pos,rot=error(assets,side,chosen,target); m=margins(assets,side,chosen)
    rows.append({'frame':frame,'side':('left','right')[side],'position_mm':pos,'rotation_deg':rot,'branch_margins_m':m.tolist(),'pass':bool(pos<=10 and rot<=5 and min(m)>=-1e-8),'policy':policy})
  if frame%20==0: print(frame,flush=True)
 vel=float(np.max(np.abs(np.diff(q,axis=0)))); acc=float(np.max(np.abs(np.diff(q,n=2,axis=0))))
 payload={'schema_version':'robot-scaled-temporal-arm-feasibility-v1','status':'DIAGNOSTIC_ONLY_NO_AUTHORITY','trajectory':{'path':str(p),'sha256':sha(p)},'motion_scale':a.motion_scale,'frame_count':count,'q_arm':q.tolist(),'rows':rows,'metrics':{'position_mm_max':max(x['position_mm'] for x in rows),'rotation_deg_max':max(x['rotation_deg'] for x in rows),'branch_margin_m_min':min(min(x['branch_margins_m']) for x in rows),'velocity_max':vel,'acceleration_max':acc,'failed_rows':sum(not x['pass'] for x in rows)},'all_gates_pass':all(x['pass'] for x in rows) and vel<=.12+1e-9 and acc<=.06+1e-9,'authority':False,'action_sidecar_published':False}
 a.output.parent.mkdir(parents=True,exist_ok=True); t=a.output.with_suffix(a.output.suffix+f'.tmp-{os.getpid()}');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(t,a.output)
 print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'metrics':payload['metrics'],'all_gates_pass':payload['all_gates_pass']})); return 0 if payload['all_gates_pass'] else 2
if __name__=='__main__': raise SystemExit(main())
