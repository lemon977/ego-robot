"""Fixed non-neural spatial Clean contrast; never a hidden-background truth."""
from __future__ import annotations
import json,time
from pathlib import Path
import cv2
import numpy as np
from .contracts import verify_ref


def spatial_clean(raw,effective,protected,authority=lambda:None):
    if (raw.dtype!=np.uint8 or raw.ndim!=4 or raw.shape[-1]!=3
        or effective.shape!=raw.shape[:3] or protected.shape!=effective.shape
        or effective.dtype!=bool or protected.dtype!=bool or np.any(effective&protected)):
        raise ValueError('SPATIAL_NATIVE_FIXED_MASK_DOMAIN')
    work=effective|protected
    if any(m.all() for m in work):raise ValueError('SPATIAL_NO_VISIBLE_BOUNDARY_SUPPORT')
    out=raw.copy();cv2.setNumThreads(2)
    for f,image in enumerate(raw):
        authority()
        filled=cv2.inpaint(np.ascontiguousarray(image),work[f].astype(np.uint8)*255,3.,cv2.INPAINT_TELEA)
        out[f,effective[f]]=filled[effective[f]]
    changed=np.any(out!=raw,axis=-1)
    if np.any(changed&(~effective|protected)):raise ValueError('SPATIAL_PIXEL_PRESERVATION')
    return out,work


def require_spatial(packet):
    b=packet.get('budgets',{});c=packet.get('config',{})
    if (packet.get('mode')!='Clean' or c.get('cpu_spatial_clean') is not True
        or c.get('spatial_algorithm') not in ('TELEA_FIXED3_PROTECTED_EXCLUDED','NORMALIZED_BACKGROUND_TONE_PP_DETAIL','SCALE_SEPARATED_BACKGROUND_TONE_PP_DETAIL','S3_NONINCREASING_BRIGHTNESS_GUARD')
        or any(b.get(k)!=0 for k in ('gpu_count','model_calls_max','render_calls_max','training_steps_max'))
        or packet.get('model_weights') or packet.get('gpu_mapping')
        or len(packet.get('sessions',[]))!=1 or 'output_frames' in packet['sessions'][0]
        or any(c.get(k) for k in ('cpu_raw_donor','paired_clean','mask_causal_pair','context_clean_pair'))):
        raise ValueError('SPATIAL_STRICT_ZERO_NEURAL_FULL_NATIVE')


def background_tone_clean(raw,parent,effective,protected,authority=lambda:None,*,detail_sigma=None):
    if (raw.dtype!=np.uint8 or parent.dtype!=np.uint8 or parent.shape!=raw.shape
        or raw.ndim!=4 or raw.shape[-1]!=3 or effective.dtype!=bool or protected.dtype!=bool
        or effective.shape!=raw.shape[:3] or protected.shape!=effective.shape or np.any(effective&protected)):
        raise ValueError('TONE_FIXED_NATIVE_DOMAIN')
    if np.any(np.any(parent!=raw,axis=-1)&(~effective|protected)):
        raise ValueError('TONE_PARENT_PIXEL_PRESERVATION_FAILED')
    out=parent.copy();work=effective|protected;stats=[];cv2.setNumThreads(2)
    if any(m.all() for m in work):raise ValueError('TONE_NO_VISIBLE_SUPPORT')
    for i,image in enumerate(raw):
        authority()
        valid=(~work[i]).astype(np.float32)
        distance=cv2.distanceTransform(work[i].astype(np.uint8),cv2.DIST_L2,5)
        sigma=max(1.,float(distance.max()))
        blur=lambda a:cv2.GaussianBlur(a,(0,0),sigmaX=sigma,sigmaY=sigma)
        denominator=blur(valid)
        tone=blur(image.astype(np.float32)*valid[:,:,None])/np.maximum(denominator[:,:,None],np.finfo(np.float32).tiny)
        low_parent=(blur(parent[i].astype(np.float32)) if detail_sigma is None else
                    cv2.GaussianBlur(parent[i].astype(np.float32),(0,0),detail_sigma))
        boundary=cv2.dilate(work[i].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)&~work[i]
        residual=image.astype(np.float32)-tone
        samples=residual[boundary]
        mad=np.median(np.abs(samples-np.median(samples,axis=0)),axis=0) if len(samples) else np.zeros(3)
        detail=np.clip(parent[i].astype(np.float32)-low_parent,-mad,mad)
        alpha=np.minimum(cv2.distanceTransform(effective[i].astype(np.uint8),cv2.DIST_L2,5)/4.,1.)[:,:,None]
        proposed=np.rint(np.clip((tone+detail)*alpha+parent[i]*(1-alpha),0,255)).astype(np.uint8)
        out[i,effective[i]]=proposed[effective[i]]
        stats.append({'native_frame_index':i,'sigma_from_mask_max_distance':sigma,
                      'detail_sigma':sigma if detail_sigma is None else detail_sigma,
                      'raw_boundary_samples':len(samples),'detail_MAD_RGB':mad.tolist()})
    if np.any(np.any(out!=raw,axis=-1)&(~effective|protected)):
        raise ValueError('TONE_OUTPUT_PIXEL_PRESERVATION_FAILED')
    return out,work,stats


def nonincreasing_brightness_guard(parent,proposal,effective,protected):
    if (parent.dtype!=np.uint8 or proposal.dtype!=np.uint8 or parent.ndim!=4 or parent.shape[-1]!=3 or parent.shape!=proposal.shape
        or effective.dtype!=bool or protected.dtype!=bool or effective.shape!=parent.shape[:3]
        or protected.shape!=effective.shape or np.any(effective&protected)):
        raise ValueError('GUARD_FIXED_NATIVE_DOMAIN')
    base=parent.astype(np.float32);target=proposal.astype(np.float32).mean(axis=-1)
    level=base.mean(axis=-1)
    gain=np.minimum(1.,target/np.maximum(level,np.finfo(np.float32).tiny))
    out=parent.copy();corrected=np.rint(base*gain[:,:,:,None]).clip(0,255).astype(np.uint8)
    out[effective]=corrected[effective]
    if np.any(out>parent):raise ValueError('GUARD_BRIGHTNESS_INCREASE')
    return out


def sealed_guard_proposal(packet,shape):
    pins=packet['inputs']
    from .v1_scene_donor_worker import _json
    result=_json(pins['guard_proposal_result']);model=_json(pins['guard_proposal_model']);producer=_json(pins['guard_proposal_packet'])
    if (result.get('status')!='PASSED' or result.get('model_result_consumable') is not True
        or result.get('model_result')!=pins['guard_proposal_model']
        or model.get('status')!='SEALED_COMPLETE' or pins['guard_proposal_rgb'] not in model.get('artifacts',[])
        or producer.get('config',{}).get('spatial_algorithm')!='SCALE_SEPARATED_BACKGROUND_TONE_PP_DETAIL'):
        raise ValueError('GUARD_SEALED_S3_REQUIRED')
    for key in ['frozen_source_raw','frozen_source_receipt','frozen_frame_map','frozen_repaired_masks',
                'frozen_mask_receipt','spatial_parent_clean','spatial_clean_receipt']:
        if producer['inputs'].get(key)!=pins[key]:raise ValueError('GUARD_PROPOSAL_INPUT_DRIFT:'+key)
    rows=model.get('sessions',[])
    if (len(rows)!=1 or rows[0].get('session_key')!=packet['sessions'][0]['session_key']
        or rows[0].get('completed_frame_indices')!=list(range(shape[0])) or not rows[0].get('execution_complete')):
        raise ValueError('GUARD_PARTIAL_PROPOSAL_FORBIDDEN')
    with np.load(pins['guard_proposal_rgb']['path'],allow_pickle=False) as z:rgb=z['rgb']
    if rgb.shape!=shape or rgb.dtype!=np.uint8:raise ValueError('GUARD_PROPOSAL_NATIVE_DOMAIN')
    return rgb


def execute_spatial_clean(packet,*,authority):
    require_spatial(packet);authority();started=time.monotonic()
    for ref in packet['runtime_code']:verify_ref(ref)
    from .v1_scene_donor_worker import sealed_inputs
    raw,parent,masks,mapping=sealed_inputs(packet,prefix='spatial')
    algorithm=packet['config']['spatial_algorithm'];tone_stats=[]
    if algorithm=='S3_NONINCREASING_BRIGHTNESS_GUARD':
        proposal=sealed_guard_proposal(packet,raw.shape)
        forbidden=~masks['effective_removal']|masks['protected_visible']
        if (np.any(np.any(parent!=raw,axis=-1)&forbidden)
            or np.any(np.any(proposal!=raw,axis=-1)&forbidden)):
            raise ValueError('GUARD_FROZEN_INPUT_PIXEL_PRESERVATION')
        rgb=nonincreasing_brightness_guard(parent,proposal,masks['effective_removal'],masks['protected_visible'])
        work=masks['effective_removal']|masks['protected_visible']
    elif algorithm in ('NORMALIZED_BACKGROUND_TONE_PP_DETAIL','SCALE_SEPARATED_BACKGROUND_TONE_PP_DETAIL'):
        rgb,work,tone_stats=background_tone_clean(raw,parent,masks['effective_removal'],masks['protected_visible'],authority,
            detail_sigma=3. if algorithm=='SCALE_SEPARATED_BACKGROUND_TONE_PP_DETAIL' else None)
    else:
        rgb,work=spatial_clean(raw,masks['effective_removal'],masks['protected_visible'],authority)
    from chaoyang.governance.common import atomic_json,artifact_ref
    from .source_frames import write_review_video
    from .v1_mco_worker import _comparison
    sid=packet['sessions'][0]['session_key'];root=Path(packet['output_dir']);out=root/'sessions'/sid/'Clean'
    out.mkdir(parents=True,exist_ok=False);review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    def write_json(path,value):
        if path.exists():raise ValueError('SPATIAL_FRESH_OUTPUT_REQUIRED')
        atomic_json(path,value);artifacts.append(artifact_ref(path))
    def write_npz(path,**values):
        with path.open('xb') as f:np.savez_compressed(f,**values)
        artifacts.append(artifact_ref(path))
    write_npz(out/'CLEAN_RGB.npz',rgb=rgb)
    write_npz(out/'WORKING_MASK.npz',working_mask=work,effective_write_mask=masks['effective_removal'])
    changed=np.any(rgb!=raw,axis=-1)
    evidence={'native_frames':len(raw),'method':algorithm,
        'opencv_version':cv2.__version__,'inpaint_radius':3.,'working_domain':'E_UNION_P','actual_write_domain':'E',
        'protected_changed_pixels':int((changed&masks['protected_visible']).sum()),
        'outside_effective_changed_pixels':int((changed&~masks['effective_removal']).sum()),
        'frame_local_fill':True,'upstream_mask_time_mode':'OFFLINE_NONCAUSAL',
        'causal_training_qualified':False,'training_eligible':False,'hidden':'UNKNOWN',
        'content_source':'SPATIAL_SYNTHESIS_NOT_TRUE_RECOVERY','human_points_added':0,'session_branches':False}
    if tone_stats:
        evidence.update(inpaint_radius=None,tone_statistics=tone_stats,boundary_blend_width=4,
                        source_semantics='RAW_OUTSIDE_MASK_NOT_SEMANTIC_BG_VERIFIED')
    if algorithm=='S3_NONINCREASING_BRIGHTNESS_GUARD':
        evidence.update(inpaint_radius=None,proposal_stage_reused=True,new_clean_generation_calls=0,
            brighter_than_original_parent_channels=int(np.count_nonzero(rgb>parent)),
            changes_vs_original_parent_pixels=int(np.any(rgb!=parent,axis=-1).sum()),
            guard='NONINCREASING_MEAN_RGB_BRIGHTNESS_SHARED_CHANNEL_GAIN')
    write_json(out/'CLEAN_EVIDENCE.json',evidence)
    write_json(out/'FRAME_MAP.json',mapping)
    write_json(out/'FRAME_STATUS.json',[{'native_frame_index':i,'geometry_decision':'UNKNOWN',
        'content_source':'SPATIAL_SYNTHESIS_NOT_TRUE_RECOVERY','visual_quality_pass':False,
        'human_review':'PENDING','protected_changed_pixels':0,'outside_effective_changed_pixels':0} for i in range(len(raw))])
    completed=write_review_video(out/'clean.mp4',rgb,mapping);artifacts.append(completed['video'])
    title=('候选：单向压亮保护' if algorithm=='S3_NONINCREASING_BRIGHTNESS_GUARD' else
           '底色＋原补图纹理：候选' if tone_stats else '空间补图候选：同Mask／非真实恢复')
    pair=_comparison(parent,rgb,list(range(len(raw))),sid,'父候选：三轮Mask＋原补图',title)
    compare=write_review_video(review/(sid+'_Spatial_comparison.mp4'),pair,mapping);artifacts.append(compare['video'])
    from PIL import Image
    for frame in packet['sessions'][0].get('review_anchors',[]):
        p=review/f'{sid}_Spatial_F{frame:03d}.png';Image.fromarray(pair[frame]).save(p);artifacts.append(artifact_ref(p))
    authority();size=sum(r['bytes'] for r in artifacts)
    if size>packet['budgets']['output_bytes_estimate']:raise ValueError('SPATIAL_OUTPUT_BUDGET')
    result={'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE','task_id':packet['task_id'],
        'run_signature':packet['run_signature'],'mode':'Clean','model_calls':0,'render_calls':0,
        'sessions':[{'session_key':sid,'native_frames':len(raw),'native_output_frames':len(raw),
        'completed_frame_indices':list(range(len(raw))),'execution_complete':True,'wall_seconds':time.monotonic()-started,
        'video':completed,'comparison':compare,'visual_quality_pass':False,'candidate_adopted':False,'human_review':'PENDING'}],
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'control_ground_truth':False,
        'visual_quality_pass':False,'training_eligible':False,'human_review':'PENDING'}
    write_json(root/'MODEL_RESULT.json',result)
    return 0
