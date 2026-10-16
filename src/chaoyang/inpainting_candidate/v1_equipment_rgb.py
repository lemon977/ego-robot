"""Candidate-only local RGB appearance extension of frozen V1 equipment.

This is not semantic ground truth. Existing visible seed pixels choose an RGB
Voronoi class; connectivity and the frozen 36px colour reach constrain support.
No query, session switch, hidden pixel or hardware projection supplies pixels.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import cv2
import numpy as np
from .contracts import verify_ref

DEVICE_ROLES={'controller','mount','attachment','cable'}

def palettes(raw,seed):
    n,h,w=raw.shape[:3];anchor=seed['raw_frame_index']
    if type(anchor) is not int or not 0<=anchor<n:raise ValueError('RGB_EQUIPMENT_SEED_FRAME')
    positive=set();device=set();negative=set()
    for row in seed['objects']:
        if row.get('visible_confirmed') is not True:continue
        if row.get('frame_idx',anchor)!=anchor:raise ValueError('RGB_EQUIPMENT_MIXED_SEED_ANCHORS')
        if len(row['points_px'])!=len(row['point_labels']):raise ValueError('RGB_EQUIPMENT_LABEL_LENGTH')
        for xy,label in zip(row['points_px'],row['point_labels']):
            if len(xy)!=2 or any(not isinstance(v,(float,int)) or int(v)!=v for v in xy):
                raise ValueError('RGB_EQUIPMENT_INTEGER_VISIBLE_PIXELS')
            x,y=map(int,xy)
            if not(0<=x<w and 0<=y<h):raise ValueError('RGB_EQUIPMENT_SEED_BOUNDS')
            if label==1:
                positive.add((x,y))
                if row['role'] in DEVICE_ROLES:device.add((x,y))
            elif label==0:negative.add((x,y))
            else:raise ValueError('RGB_EQUIPMENT_BINARY_LABEL')
    negative-=positive
    if not device or not negative:raise ValueError('RGB_EQUIPMENT_BOTH_VISIBLE_CLASSES_REQUIRED')
    get=lambda pts:np.asarray([raw[anchor,y,x] for x,y in sorted(pts)],np.float32)
    return get(device),get(negative),{'device_pixels':[list(v) for v in sorted(device)],
        'background_pixels':[list(v) for v in sorted(negative)],'anchor':anchor}

def motion_qualify(raw,masks,support,*,authority=lambda:None,flow_provider=None):
    from chaoyang.exact78_baseline.mask import BAND_COLOUR_REACH_PX
    n,h,w=raw.shape[:3]
    if n<2:raise ValueError('RGB_MOTION_TWO_NATIVE_FRAMES_REQUIRED')
    yy,xx=np.indices((h,w));checker=(xx+yy)%2
    qualified=np.zeros_like(support);diagnosis=[]
    for frame in range(n):
        authority();neighbor=frame+1 if frame+1<n else frame-1
        if flow_provider is None:
            a=cv2.cvtColor(raw[frame],cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(raw[neighbor],cv2.COLOR_RGB2GRAY)
            dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward=dis.calc(a,b,None);backward=dis.calc(b,a,None)
        else:forward,backward=flow_provider(frame,neighbor)
        if (forward.shape!=(h,w,2)or backward.shape!=(h,w,2)or not np.isfinite(forward).all()or not np.isfinite(backward).all()):
            raise ValueError('RGB_MOTION_FLOW_DOMAIN')
        mx=(xx+forward[...,0]).astype(np.float32);my=(yy+forward[...,1]).astype(np.float32)
        valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
        returned=cv2.remap(backward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        uncertainty=2*np.linalg.norm(forward+returned,axis=-1)
        count,labels=cv2.connectedComponents(masks['equipment'][frame].astype(np.uint8),connectivity=8)
        best=np.full((h,w),np.inf,np.float32);decision=np.zeros((h,w),bool);ties=np.zeros((h,w),bool);components=[]
        for obj in range(1,count):
            component=labels==obj
            distance=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)
            near=distance<=BAND_COLOUR_REACH_PX
            background=near&~(masks['effective_removal'][frame]|masks['core'][frame]|masks['protected_visible'][frame]|masks['cup_observed'][frame])
            accepted=valid&near&support[frame];models=[]
            for half in [0,1]:
                fg=forward[component&(checker==half)];bg=forward[background&(checker==half)]
                if not len(fg)or not len(bg):accepted[:]=False;models.append({'half':half,'status':'ABSENT_CLASS'});continue
                f=np.median(fg,axis=0);b=np.median(bg,axis=0)
                advantage=np.linalg.norm(forward-b,axis=-1)-np.linalg.norm(forward-f,axis=-1)
                accepted &=advantage>uncertainty
                models.append({'half':half,'equipment_flow_median':f.tolist(),'background_flow_median':b.tolist()})
            closer=near&(distance<best);equal=near&(distance==best)
            decision[closer]=accepted[closer];ties[closer]=False;ties[equal]=True;best[closer]=distance[closer]
            components.append({'component':obj,'equipment_pixels':int(component.sum()),'models':models})
        qualified[frame]=decision&~ties
        diagnosis.append({'frame':frame,'neighbor':neighbor,'rgb_support':int(support[frame].sum()),
            'motion_qualified_support':int(qualified[frame].sum()),'components':components})
    return qualified,diagnosis

def extend(raw,masks,seed,*,authority=lambda:None,motion_veto=False,flow_provider=None):
    from chaoyang.exact78_baseline.mask import BAND_COLOUR_REACH_PX,bracket_fill,dilate
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    if raw.dtype!=np.uint8 or raw.ndim!=4 or raw.shape[-1]!=3:raise ValueError('RGB_EQUIPMENT_NATIVE_RGB')
    shape=raw.shape[:3]
    for k in ['core','sam_core','equipment','removal','effective_removal','protected_visible','cup_observed']:
        if masks[k].dtype!=bool or masks[k].shape!=shape:raise ValueError('RGB_EQUIPMENT_MASK_DOMAIN')
    fg,bg,meta=palettes(raw,seed);support=np.zeros(shape,bool)
    for frame in range(len(raw)):
        authority();eq=masks['equipment'][frame]
        if not eq.any():continue
        near=cv2.distanceTransform((~eq).astype(np.uint8),cv2.DIST_L2,5)<=BAND_COLOUR_REACH_PX
        pixels=raw[frame][near].astype(np.float32)
        distance=lambda proto:np.min(np.sum((pixels[:,None,:]-proto[None,:,:])**2,axis=-1),axis=1)
        region=np.zeros(shape[1:],bool);region[near]=distance(fg)<distance(bg)
        region &=~(masks['protected_visible'][frame]|masks['cup_observed'][frame])
        # Only components touching existing RGB equipment are proposals; colour
        # alone and a geometric search hit never establish a new instance.
        _,labels=cv2.connectedComponents(region.astype(np.uint8),connectivity=8)
        ids=np.unique(labels[eq & region]);ids=ids[ids>0]
        support[frame]=np.isin(labels,ids) if len(ids) else False
    motion=None
    if motion_veto:support,motion=motion_qualify(raw,masks,support,authority=authority,flow_provider=flow_provider)
    addition=support &~masks['core'] &~masks['protected_visible'] &~masks['cup_observed']
    result={k:v.copy() for k,v in masks.items()}
    result['core']|=addition;result['equipment']|=addition
    # Do not mislabel model-free RGB support as SAM output.
    rule=masks.get('metadata',{}).get('removal',{}) if isinstance(masks.get('metadata'),dict) else {}
    if not rule:raise ValueError('RGB_EQUIPMENT_FROZEN_REMOVAL_RULE_REQUIRED')
    radius=rule['observed_segmentation_dilation_px'];temporal=rule['temporal_bracket_frames']
    if type(radius)is not int or type(temporal)is not int:raise ValueError('RGB_EQUIPMENT_FROZEN_RULE_TYPE')
    expansion=bracket_fill(np.stack([dilate(m,radius) for m in result['core']]),temporal)
    result['removal']|=expansion &~result['protected_visible']
    result['effective_removal'],gate=effective_clean_masks(result['removal'],result['protected_visible'])
    return result,support,{**meta,'recipe':'VISIBLE_DEVICE_RGB_NEAREST_CLASS_CONNECTED_WITHIN_FROZEN_COLOUR_REACH',
        'reach_px':BAND_COLOUR_REACH_PX,'new_direct_support_per_frame':addition.sum(axis=(1,2)).astype(int).tolist(),
        'effective_added_per_frame':(result['effective_removal']&~masks['effective_removal']).sum(axis=(1,2)).astype(int).tolist(),
        'protected_changed_pixels':int(np.count_nonzero(result['protected_visible']!=masks['protected_visible'])),
        'sam_core_changed_pixels':int(np.count_nonzero(result['sam_core']!=masks['sam_core'])),
        'manual_points_added':0,'pixel_evidence':'RAW_RGB_APPEARANCE_CANDIDATE_NOT_SEMANTIC_TRUTH',
        'generic_object_promoted':False,'hardware_pixels_as_evidence':False,'gate':gate,
        'motion_veto':motion_veto,'motion_diagnostics':motion}

def execute(packet,*,authority):
    c=packet.get('config',{});b=packet['budgets']
    if (packet['mode']!='Mask' or c.get('cpu_equipment_rgb') is not True
        or any(b.get(k)!=0 for k in ['gpu_count','model_calls_max','render_calls_max','training_steps_max'])
        or packet.get('model_weights') or len(packet['sessions'])!=1):raise ValueError('RGB_EQUIPMENT_STRICT_CPU')
    authority();start=time.monotonic()
    for ref in packet['runtime_code']:verify_ref(ref)
    from .v1_scene_donor_worker import sealed_inputs
    raw,_,masks,mapping=sealed_inputs(packet,'equipment')
    pins=packet['inputs'];verify_ref(pins['visible_seed']);seed=json.loads(Path(pins['visible_seed']['path']).read_text())
    if seed['session_id']!=packet['sessions'][0]['session_key']:raise ValueError('RGB_EQUIPMENT_SEED_SESSION')
    verify_ref(seed['image'])
    from PIL import Image
    if not np.array_equal(np.asarray(Image.open(seed['image']['path']).convert('RGB')),raw[seed['raw_frame_index']]):
        raise ValueError('RGB_EQUIPMENT_SEED_IMAGE_NOT_SOURCE_EXACT')
    report=json.loads(Path(pins['mask_report']['path']).read_text());masks['metadata']=report['metadata']
    if type(c.get('motion_veto',False))is not bool:raise ValueError('RGB_EQUIPMENT_MOTION_BOOL')
    candidate,support,diagnosis=extend(raw,masks,seed,authority=authority,motion_veto=c.get('motion_veto',False))
    from chaoyang.governance.common import atomic_json,artifact_ref
    sid=packet['sessions'][0]['session_key'];root=Path(packet['output_dir']);out=root/'sessions'/sid/'Mask'
    out.mkdir(parents=True,exist_ok=False);review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    def js(p,d):atomic_json(p,d);artifacts.append(artifact_ref(p))
    with (out/'MASK_ARRAYS.npz').open('xb') as f:np.savez_compressed(f,**{k:v for k,v in candidate.items() if isinstance(v,np.ndarray)})
    artifacts.append(artifact_ref(out/'MASK_ARRAYS.npz'))
    with (out/'RGB_SUPPORT.npz').open('xb') as f:np.savez_compressed(f,support=support)
    artifacts.append(artifact_ref(out/'RGB_SUPPORT.npz'))
    js(out/'MASK_MERGE.json',diagnosis);js(out/'FRAME_MAP.json',mapping)
    from .v1_mco_worker import _comparison
    from .source_frames import write_review_video
    def overlay(m):
        v=raw.copy();e=m['effective_removal'];v[e]=np.rint(.4*v[e]+.6*np.array([255,0,255])).astype(np.uint8)
        return v
    frames=_comparison(overlay(masks),overlay(candidate),list(range(len(raw))),sid,'父候选：三轮Mask','本轮RGB设备补漏：待审')
    video=write_review_video(review/(sid+'_Equipment_Mask.mp4'),frames,mapping);artifacts.append(video['video'])
    for i in packet['sessions'][0].get('review_anchors',[]):
        p=review/f'{sid}_Equipment_F{i:03d}.png';Image.fromarray(frames[i]).save(p);artifacts.append(artifact_ref(p))
    js(out/'FRAME_STATUS.json',[{**m,'mask_source':'V1_PLUS_VISIBLE_RGB_APPEARANCE_CANDIDATE','hidden':'UNKNOWN',
        'human_review':'PENDING','training_eligible':False}for m in mapping])
    authority();size=sum(r['bytes']for r in artifacts)
    if size>b['output_bytes_estimate']:raise ValueError('RGB_EQUIPMENT_OUTPUT_BUDGET')
    js(root/'MODEL_RESULT.json',{'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE',
        'task_id':packet['task_id'],'run_signature':packet['run_signature'],'mode':'Mask','model_calls':0,'render_calls':0,
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'training_eligible':False,
        'control_ground_truth':False,'visual_quality_pass':False,'human_review':'PENDING','sessions':[{
        'session_key':sid,'native_frames':len(raw),'native_output_frames':len(raw),'completed_frame_indices':list(range(len(raw))),
        'execution_complete':True,'wall_seconds':time.monotonic()-start,'video':video,'visual_quality_pass':False,
        'candidate_adopted':False,'human_review':'PENDING'}]})
    return 0
