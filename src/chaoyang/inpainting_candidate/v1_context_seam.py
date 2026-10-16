"""Candidate-only gradient reconstruction inside an unchanged authorized write mask.

The existing context-exclusion output is the gradient donor, not scene truth.
Raw pixels supply Dirichlet boundary conditions only outside the write domain.
No Raw human/device pixels inside the domain are blended back into the image.
"""
from __future__ import annotations
import numpy as np


def reconstruct_fixed_boundary(raw, working, effective, protected, *, raw_boundary_allowed=None):
    """Return uint8 RGB and diagnostics; solve the exact four-neighbour problem.

    This is an unadopted offline Clean candidate, not a new quality threshold.
    No dependency on session ID, seeds, model inference, or hidden scene geometry.
    """
    if (raw.dtype != np.uint8 or working.dtype != np.uint8 or raw.shape != working.shape
            or raw.ndim != 3 or raw.shape[-1] != 3 or effective.dtype != bool
            or protected.dtype != bool or effective.shape != raw.shape[:2]
            or protected.shape != effective.shape):
        raise ValueError('CONTEXT_SEAM_NATIVE_RGB_BOOL_DOMAIN')
    if np.any(effective & protected):
        raise ValueError('CONTEXT_SEAM_PROTECTED_WRITE_CONFLICT')
    if raw_boundary_allowed is not None and (raw_boundary_allowed.dtype!=bool or raw_boundary_allowed.shape!=effective.shape):
        raise ValueError('CONTEXT_SEAM_BOUNDARY_SOURCE_BOOL_DOMAIN')
    out=raw.copy();n=int(effective.sum())
    if n==0:
        return out,{'write_pixels':0,'protected_changed_pixels':0,'outside_E_changed_pixels':0,'status':'EMPTY_WRITE_DOMAIN'}
    if n==effective.size:
        raise ValueError('CONTEXT_SEAM_DIRICHLET_BOUNDARY_ABSENT')
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    h,w=effective.shape;ys,xs=np.nonzero(effective)
    index=np.full((h,w),-1,dtype=np.int32);index[ys,xs]=np.arange(n,dtype=np.int32)
    base=working.astype(np.float64);delta=raw.astype(np.float64)-base
    if raw_boundary_allowed is not None:
        # Unknown equipment support is not a legitimate Raw colour anchor.
        # Retain the same working-output boundary there; do not infer identity
        # or change the final write domain, protection, or hidden geometry.
        delta[~raw_boundary_allowed]=0
    diagonal=np.zeros(n,np.float64);rhs=np.zeros((n,3),np.float64)
    rr=[];cc=[];vv=[];rejected_boundary_pairs=0
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny,nx=ys+dy,xs+dx;valid=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        row=np.flatnonzero(valid);y,x=ny[valid],nx[valid]
        diagonal[row]+=1
        inside=effective[y,x]
        rr.append(row[inside]);cc.append(index[y[inside],x[inside]])
        vv.append(-np.ones(int(inside.sum()),np.float64))
        rhs[row[~inside]]+=delta[y[~inside],x[~inside]]
        if raw_boundary_allowed is not None:
            rejected_boundary_pairs+=int((~raw_boundary_allowed[y[~inside],x[~inside]]).sum())
    rr.append(np.arange(n));cc.append(np.arange(n));vv.append(diagonal)
    matrix=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n,n)).tocsc()
    # One factorization, three colour RHS. No tunable blur/radius/colour clipping.
    factor=splu(matrix);correction=factor.solve(rhs)
    residual=matrix@correction-rhs
    values=base[ys,xs]+correction
    clipped=(values<0)|(values>255)
    out[ys,xs]=np.rint(np.clip(values,0,255)).astype(np.uint8)
    changed=np.any(out!=raw,axis=-1)
    diagnostics={'status':'SOLVED','write_pixels':n,
        'linear_residual_max_abs':float(np.max(np.abs(residual))),
        'clipped_channels':int(clipped.sum()),'correction_max_abs':float(np.max(np.abs(correction))),
        'protected_changed_pixels':int((changed&protected).sum()),
        'outside_E_changed_pixels':int((changed&~effective).sum()),
        'raw_boundary_veto_pairs':rejected_boundary_pairs,
        'raw_inside_E_used':False,'hidden_geometry':'UNKNOWN','candidate_adopted':False}
    if diagnostics['protected_changed_pixels']or diagnostics['outside_E_changed_pixels']:
        raise ValueError('CONTEXT_SEAM_FIXED_WRITE_INVARIANT')
    return out,diagnostics


def require_candidate(packet):
    b=packet.get('budgets',{});c=packet.get('config',{})
    if (packet.get('mode')!='Clean' or c.get('cpu_context_seam') is not True
            or c.get('algorithm')not in ['POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY','POISSON_RGB_UNCERTAIN_BOUNDARY_VETO']
            or any(b.get(k)!=0 for k in ['gpu_count','model_calls_max','render_calls_max','training_steps_max'])
            or packet.get('model_weights') or packet.get('gpu_mapping')
            or len(packet.get('sessions',[]))!=1 or 'output_frames' in packet['sessions'][0]
            or any(c.get(k) for k in ['cpu_raw_donor','cpu_spatial_clean','cpu_equipment_rgb','paired_clean','context_exclusion_pair'])):
        raise ValueError('CONTEXT_SEAM_FINITE_CPU_FULL_NATIVE_REQUIRED')


def execute_candidate(packet, *, authority):
    """Registered CPU algorithm consumer, never called by engineering diagnostics."""
    import json,time
    from pathlib import Path
    from PIL import Image,ImageDraw,ImageFont
    from .contracts import verify_ref,file_ref
    from chaoyang.governance.common import REPO_ROOT
    from .automation import verify_component,read_reference,write_once_json
    from .automation_worker import _outputs
    from .source_frames import write_review_video
    require_candidate(packet);authority();started=time.monotonic()
    for ref in [*packet['runtime_code'],*packet['inputs'].values()]:verify_ref(ref)
    pins=packet['inputs'];parent=read_reference(pins['producer_packet']);terminal=read_reference(pins['producer_result'])
    model=read_reference(pins['producer_model']);sid=packet['sessions'][0]['session_key']
    if (terminal.get('status')!='PASSED' or terminal.get('model_result')!=pins['producer_model']
            or terminal.get('task_id')!=parent['task_id'] or terminal.get('run_signature')!=parent['run_signature']
            or model.get('task_id')!=parent['task_id'] or model.get('run_signature')!=parent['run_signature']
            or model.get('status')!='SEALED_COMPLETE'):
        raise ValueError('CONTEXT_SEAM_SEALED_PARENT_BINDING_REQUIRED')
    request=read_reference(parent['inputs']['request']);rows=[r for r in request['sessions']if r['session_id']==sid]
    results=[r for r in model['sessions']if r['session_key']==sid]
    if len(rows)!=1 or len(results)!=1:raise ValueError('CONTEXT_SEAM_ONE_DECLARED_SESSION_REQUIRED')
    row=rows[0];result=results[0];refs=result['stage_receipts']
    if row['native_frames']!=packet['sessions'][0]['native_frames'] or not result['execution_complete']:
        raise ValueError('CONTEXT_SEAM_PARENT_PARTIAL_OR_FRAME_DOMAIN')
    files={}
    for stage in ['Source','Mask','Clean']:
        verify_component(refs[stage],parent['component_signatures'][sid][stage],row,stage=stage,repo_root=REPO_ROOT)
        files[stage]=_outputs(refs[stage])
    def arrays(ref):
        verify_ref(ref)
        with np.load(ref['path'],allow_pickle=False)as z:return {k:z[k]for k in z.files}
    raw=arrays(files['Source']['RAW_RGB.npz'])['rgb'];mask=arrays(files['Mask']['MASK_ARRAYS.npz'])
    old=arrays(files['Clean']['CLEAN_RGB.npz'])['rgb'];work=arrays(files['Clean']['WORKING_CLEAN_RGB.npz'])['rgb']
    mapping=read_reference(files['Source']['FRAME_MAP.json']);E=mask['effective_removal'];P=mask['protected_visible']
    allowed=None
    if packet['config']['algorithm']=='POISSON_RGB_UNCERTAIN_BOUNDARY_VETO':
        working_masks=arrays(files['Clean']['CONTEXT_WORKING_MASKS.npz']);U=working_masks['rgb_uncertain_support']
        if (U.dtype!=bool or U.shape!=E.shape or np.any(U&P)
                or not np.array_equal(working_masks['write_effective_removal'],E)):
            raise ValueError('CONTEXT_SEAM_FROZEN_UNCERTAINTY_DOMAIN')
        allowed=~U
    if (old.shape!=raw.shape or work.shape!=raw.shape or len(raw)!=row['native_frames'] or len(mapping)!=len(raw)
            or np.any(np.any(old!=raw,axis=-1)&(~E|P))):
        raise ValueError('CONTEXT_SEAM_PARENT_PIXEL_OR_TIME_DOMAIN')
    rgb=raw.copy();stats=[]
    for i in range(len(raw)):
        authority();rgb[i],d=reconstruct_fixed_boundary(raw[i],work[i],E[i],P[i],
            raw_boundary_allowed=None if allowed is None else allowed[i]);stats.append({'native_frame_index':i,**d})
        if i%16==0:print(json.dumps({'stage':'CLEAN_BOUNDARY','session':sid,'completed':i+1,'total':len(raw)}),flush=True)
    root=Path(packet['output_dir']);out=root/'sessions'/sid/'Clean';out.mkdir(parents=True,exist_ok=False)
    review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True);artifacts=[]
    def js(path,value):write_once_json(path,value);artifacts.append(file_ref(path))
    with (out/'CLEAN_RGB.npz').open('xb')as f:np.savez_compressed(f,rgb=rgb,effective_removal=E)
    artifacts.append(file_ref(out/'CLEAN_RGB.npz'))
    js(out/'FRAME_MAP.json',mapping)
    js(out/'CLEAN_EVIDENCE.json',{'method':packet['config']['algorithm'],'producer_model':pins['producer_model'],
        'stage_receipts':refs,'actual_clean_inputs':files['Clean'],'native_frames':len(raw),'statistics':stats,
        'protected_changed_pixels':int((np.any(rgb!=raw,axis=-1)&P).sum()),
        'outside_E_changed_pixels':int((np.any(rgb!=raw,axis=-1)&~E).sum()),
        'raw_inside_E_used':False,'hidden_geometry':'UNKNOWN','causal_training_input':False,'candidate_adopted':False})
    js(out/'FRAME_STATUS.json',[{**m,'clean_source':'OFFLINE_WORKING_GRADIENT_BOUNDARY_RECONSTRUCTION',
        'geometry_decision':'UNKNOWN','human_review':'PENDING','training_eligible':False}for m in mapping])
    video=write_review_video(out/'clean.mp4',rgb,mapping);artifacts.append(video['video'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',18)
    def titled(image,text):
        canvas=Image.new('RGB',(image.shape[1],image.shape[0]+32));canvas.paste(Image.fromarray(image),(0,32))
        ImageDraw.Draw(canvas).text((4,4),text,font=font,fill='white');return np.asarray(canvas)
    pair=np.stack([np.concatenate((titled(old[i],f'{sid}第{i}帧 原参考排除／固定写回'),
        titled(rgb[i],f'{sid}第{i}帧 候选／相同写域边界重建')),axis=1)for i in range(len(raw))])
    comparison=write_review_video(review/f'{sid}_Boundary_Comparison.mp4',pair,mapping);artifacts.append(comparison['video'])
    for f in packet['sessions'][0].get('review_anchors',[]):
        path=review/f'{sid}_Boundary_Compare_F{f:03d}.png';Image.fromarray(pair[f]).save(path);artifacts.append(file_ref(path))
    authority();size=sum(r['bytes']for r in artifacts)
    if size>packet['budgets']['output_bytes_estimate']:raise ValueError('CONTEXT_SEAM_OUTPUT_BUDGET')
    js(root/'MODEL_RESULT.json',{'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE',
        'task_id':packet['task_id'],'run_signature':packet['run_signature'],'mode':'Clean','model_calls':0,'render_calls':0,
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'control_ground_truth':False,
        'training_eligible':False,'human_review':'PENDING','visual_quality_pass':False,'sessions':[{'session_key':sid,
        'native_frames':len(raw),'native_output_frames':len(raw),'completed_frame_indices':list(range(len(raw))),
        'execution_complete':True,'wall_seconds':time.monotonic()-started,'video':video,'comparison':comparison,
        'visual_quality_pass':False,'candidate_adopted':False,'human_review':'PENDING'}]})
    return 0
