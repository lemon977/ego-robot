#!/usr/bin/env python3
"""CPU-only KaiHand anatomical-landmark temporal feasibility diagnostic.

Non-thumb semantics are wrist->MCP, MCP->PIP, PIP->DIP, DIP->mesh-tip.
Thumb is solved independently as wrist->CMC, CMC->MCP, MCP->IP, IP->mesh-tip.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES',''); os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np, trimesh
from scipy.optimize import least_squares
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from tools import render_poker_symmetric_chirality_flange_successor as old
from tools import render_poker_same_side_outward_frame0 as shared
from tools import render_same_side_world_temporal_review as temporal
from tools import run_newtask_robot_shared_v4_hand as handfit

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def transform(T,p): return T[:3,:3]@p+T[:3,3]

def terminal_tip(model,finger,prefix):
 link_name=f'{prefix}_{finger}_link{handfit.TERMINALS[finger]}'
 matches=[visual for visual in model.visuals if visual.link==link_name]
 if len(matches)!=1: raise RuntimeError(f'cannot uniquely resolve terminal visual {link_name}')
 visual=matches[0]; mesh_path=Path(visual.mesh_path)
 mesh=trimesh.load(mesh_path,force='mesh',process=False); vertices=np.asarray(mesh.vertices)
 if finger=='thumb':
  extreme=np.max(vertices[:,0]); selected=vertices[vertices[:,0]>=extreme-0.0005]
 else:
  extreme=np.min(vertices[:,2]); selected=vertices[vertices[:,2]<=extreme+0.0005]
 local=np.asarray(visual.origin)@np.r_[selected.mean(0),1.]
 return local[:3],{'path':str(mesh_path.resolve()),'sha256':sha(mesh_path.resolve()),'selection':('max_x','min_z')[finger!='thumb'],'extreme_m':float(extreme)}

def features(official,contract,q,finger,tip_local,thumb_rotation=None):
 fk=official.forward_kinematics(contract['model'],dict(zip(contract['names'],q,strict=True))); p=contract['prefix']
 if finger=='thumb': names=[f'{p}_thumb_link3',f'{p}_thumb_link5',f'{p}_thumb_link6']; terminal=f'{p}_thumb_link6';mcp_index=1
 else: names=[f'{p}_{finger}_link2',f'{p}_{finger}_link3',f'{p}_{finger}_link4']; terminal=f'{p}_{finger}_link4';mcp_index=0
 points=np.asarray([fk[x][:3,3] for x in names]+[transform(fk[terminal],tip_local)])
 local=points@contract['basis']; bones=unit(np.diff(points,axis=0))@contract['basis']; tip=unit(local[-1])
 if finger=='thumb' and thumb_rotation is not None:
  bones=bones@thumb_rotation.T;tip=tip@thumb_rotation.T
 return {'mcp':local[mcp_index]/contract['mcp_scale'],'bones':bones,'tip':tip}

def semantic_target(target):
 return {'mcp':target['mcp'],'bones':np.asarray(target['bones'])[1:],'tip':target['tip']}

def row(actual,target):
 angles=handfit.angle_deg(actual['bones'],target['bones']);return {'mcp_normalized_l2':float(np.linalg.norm(actual['mcp']-target['mcp'])),'bone_error_deg_per_bone':angles.tolist(),'bone_error_deg_max':float(max(angles)),'tip_direction_error_deg':float(handfit.angle_deg(actual['tip'],target['tip']))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--thumb-adapter',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rp=a.result.resolve(strict=True);result=json.load(rp.open());tp=Path(result['outputs']['trajectory']['path']);hp=Path(result['lineage']['preflight']['inputs']['hawor']['path'])
 with np.load(tp,allow_pickle=False) as z: oldq=np.asarray(z['q_hand'])
 with np.load(hp,allow_pickle=False) as z: hawor={k:np.asarray(z[k]) for k in z.files}
 assets=old.load_pinned_robot_assets(PROJECT);contracts=handfit.model_contract(old.official,shared.wrist_adapter,assets);count=len(oldq);q=np.empty_like(oldq);q[:48]=oldq[:48];tips={};tip_evidence={};thumb_rotations=np.repeat(np.eye(3)[None],2,axis=0)
 if a.thumb_adapter:
  with np.load(a.thumb_adapter.resolve(strict=True),allow_pickle=False) as z: thumb_rotations=np.asarray(z['rotations'],dtype=np.float64)
  if thumb_rotations.shape!=(2,3,3):raise RuntimeError('thumb adapter shape')
 for side,c in enumerate(contracts):
  for finger in handfit.FINGERS: tips[side,finger],tip_evidence[f'{side}:{finger}']=terminal_tip(c['model'],finger,c['prefix'])
 audits=[]
 for frame in range(count):
  for side,side_name in enumerate(shared.SIDES):
   target=handfit.human_features(shared.wrist_adapter,np.asarray(hawor['joints_3d_world'][side,frame]),side_name)
   if frame<48: candidate=q[frame,side].copy();policy='ACCEPTED_V3_0_47_LOCK'
   else:
    prev=q[frame-1,side];prev2=q[frame-2,side];candidate=prev.copy()
    for finger in handfit.FINGERS:
     group=handfit.GROUPS[finger];lo,hi=temporal.bounded_limits(contracts[side]['lower'][group],contracts[side]['upper'][group],prev[group],prev2[group],temporal.HAND_SOLVER_STEP_LIMIT);free=hi-lo>1e-12
     attempts=[]
     for weights in handfit.FIT_GRID:
      base=np.clip(prev[group],lo,hi)
      def residual_free(x):
       local=base.copy();local[free]=x;whole=candidate.copy();whole[group]=local;actual=features(old.official,contracts[side],whole,finger,tips[side,finger],thumb_rotations[side]);wanted=semantic_target(target[finger]);return np.concatenate((weights['mcp']*(actual['mcp']-wanted['mcp']),weights['bone']*(actual['bones']-wanted['bones']).ravel(),weights['tip']*(actual['tip']-wanted['tip']),weights['prior']*(local-prev[group])))
      if np.any(free): sol=least_squares(residual_free,base[free],bounds=(lo[free],hi[free]),max_nfev=220,ftol=1e-10,xtol=1e-10,gtol=1e-10);base[free]=sol.x
      base[~free]=lo[~free];whole=candidate.copy();whole[group]=base;metrics=row(features(old.official,contracts[side],whole,finger,tips[side,finger],thumb_rotations[side]),semantic_target(target[finger]));score=max(0,metrics['bone_error_deg_max']-60)/60+max(0,metrics['tip_direction_error_deg']-15)/15
      attempts.append((score,metrics['bone_error_deg_max']/60+metrics['tip_direction_error_deg']/15,base))
     candidate[group]=min(attempts,key=lambda x:x[:2])[2]
    q[frame,side]=candidate;policy='BOUNDED_ANATOMICAL_FIVE_CHAIN_THUMB_INDEPENDENT'
   fingers=[]
   for finger in handfit.FINGERS: fingers.append({'finger':finger,**row(features(old.official,contracts[side],candidate,finger,tips[side,finger],thumb_rotations[side]),semantic_target(target[finger]))})
   audits.append({'frame':frame,'side':side_name,'policy':policy,'fingers':fingers,'bone_error_deg_max':max(x['bone_error_deg_max'] for x in fingers),'tip_direction_error_deg_max':max(x['tip_direction_error_deg'] for x in fingers)})
  if frame%20==0:print(frame,flush=True)
 vel=float(np.max(np.abs(np.diff(q,axis=0))));acc=float(np.max(np.abs(np.diff(q,n=2,axis=0))));bm=max(x['bone_error_deg_max'] for x in audits);tm=max(x['tip_direction_error_deg_max'] for x in audits)
 payload={'schema_version':'kaihand-anatomical-temporal-feasibility-v1','status':'DIAGNOSTIC_ONLY_NO_AUTHORITY','result_input':{'path':str(rp),'sha256':sha(rp)},'thumb_adapter':None if not a.thumb_adapter else {'path':str(a.thumb_adapter.resolve()),'sha256':sha(a.thumb_adapter.resolve())},'semantic_contract':{'nonthumb':['MCP_flexion_link2','PIP_link3','DIP_link4','terminal_mesh_tip'],'thumb':['CMC_link3','MCP_link5','IP_link6','terminal_mesh_tip'],'fixed_palm_segment_excluded_from_finger_bone_gate':True,'mano_target_bones':'indices_1_to_3_CMC_or_MCP_onward','thumb_solved_independently':True},'tip_evidence':tip_evidence,'q_hand':q.tolist(),'audits':audits,'metrics':{'bone_error_deg_max':bm,'tip_direction_error_deg_max':tm,'velocity_max':vel,'acceleration_max':acc},'all_gates_pass':bm<=60 and tm<=15 and vel<=.08+1e-9 and acc<=.06+1e-9,'authority':False,'action_sidecar_published':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);t=a.output.with_suffix(a.output.suffix+f'.tmp-{os.getpid()}');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(t,a.output);print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'metrics':payload['metrics'],'all_gates_pass':payload['all_gates_pass']}));return 0 if payload['all_gates_pass'] else 2
if __name__=='__main__':raise SystemExit(main())
