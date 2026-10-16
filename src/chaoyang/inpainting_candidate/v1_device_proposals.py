"""Raw-only generic proposals and conservative V1 visible-seed identity audit.

Proposal masks are never deletion masks. Binding alone never grants deletion.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from .contracts import verify_ref

WEARER_ROLES={'human','hand','glove','controller','mount','attachment','cable','forearm','arm'}


def bind_seeded_roles(masks,seed,cup_support):
    shape=cup_support.shape
    positives=[];negatives=set();all_wearer=set();other_positive=set()
    anchor=seed['raw_frame_index']
    for obj in seed['objects']:
        if obj.get('frame_idx',anchor)!=anchor or obj.get('visible_confirmed') is not True:continue
        role=obj.get('role');side=obj.get('side')
        for xy,label in zip(obj['points_px'],obj['point_labels']):
            if (len(xy)!=2 or any(not isinstance(v,(int,float)) or int(v)!=v for v in xy)):
                raise ValueError('DEVICE_BIND_NONINTEGER_VISIBLE_SEED')
            x,y=map(int,xy)
            if not (0<=x<shape[1] and 0<=y<shape[0]):raise ValueError('DEVICE_BIND_SEED_OUTSIDE_NATIVE')
            point=(x,y)
            if label==1:
                if role in WEARER_ROLES and side in ('left','right'):
                    positives.append((point,side,role,int(obj['obj_id'])));all_wearer.add(point)
                else:other_positive.add(point)
            elif label==0:negatives.add(point)
            else:raise ValueError('DEVICE_BIND_BINARY_LABEL_REQUIRED')
    background=(negatives-all_wearer)|other_positive
    rows=[]
    for key,mask in masks.items():
        if mask.dtype!=bool or mask.shape!=shape:raise ValueError('DEVICE_BIND_RGB_NATIVE_BOOL')
        hits=[{'point':list(point),'side':side,'seed_role':role,'seed_obj_id':obj}
              for point,side,role,obj in positives if mask[point[1],point[0]]]
        bg=[list(point) for point in sorted(background) if mask[point[1],point[0]]]
        sides=sorted({r['side'] for r in hits});cup=int(np.count_nonzero(mask&cup_support))
        reasons=[]
        if not mask.any():reasons.append('EMPTY_AT_VISIBLE_SEED_FRAME')
        if not hits:reasons.append('NO_EXISTING_WEARER_POSITIVE_PIXEL')
        if len(sides)!=1:reasons.append('SIDE_NOT_UNIQUE')
        if bg:reasons.append('BACKGROUND_OR_OBJECT_POINT_CONFLICT')
        if cup:reasons.append('VISIBLE_CUP_SUPPORT_CONFLICT')
        rows.append({'array_key':key,'status':'UNKNOWN' if reasons else 'CANDIDATE_V1_SEED_BOUND_WEARER',
            'side':sides[0] if len(sides)==1 else 'UNKNOWN','positive_hits':hits,
            'background_conflicts':bg,'cup_overlap_pixels':cup,'reasons':reasons,
            'deletion_authorized':False,'semantic_ground_truth':False})
    return rows


def execute_generic_object_probe(packet,*,authority):
    c=packet.get('config',{});b=packet.get('budgets',{})
    if (packet.get('mode')!='Mask' or c.get('generic_object_probe') is not True or c.get('query')!='object'
        or c.get('anchor_frame')!=0 or b.get('gpu_count')!=1 or b.get('model_calls_max')!=1
        or b.get('render_calls_max')!=0 or b.get('training_steps_max')!=0
        or any(c.get(k) for k in ('cpu_raw_donor','cpu_spatial_clean','paired_clean','mask_causal_pair','context_clean_pair'))
        or len(packet.get('sessions',[]))!=1 or 'output_frames' in packet['sessions'][0]):
        raise ValueError('DEVICE_PROPOSAL_STRICT_ONE_GENERIC_QUERY_CONTRACT')
    authority();start=time.monotonic()
    for ref in [*packet['inputs'].values(),*packet['runtime_code'],*packet['model_weights']]:verify_ref(ref)
    pins=packet['inputs'];sid=packet['sessions'][0]['session_key'];n=packet['sessions'][0]['native_frames']
    receipt=json.loads(Path(pins['frozen_source_receipt']['path']).read_text())
    if (receipt.get('stage')!='Source' or receipt.get('status')!='SEALED_COMPLETE' or receipt.get('session_id')!=sid
        or receipt.get('native_frames')!=n or pins['frozen_source_raw'] not in receipt.get('outputs',[])
        or pins['frozen_frame_map'] not in receipt.get('outputs',[])):
        raise ValueError('DEVICE_PROPOSAL_FROZEN_SOURCE_BINDING')
    with np.load(pins['frozen_source_raw']['path'],allow_pickle=False) as z:raw=z['rgb']
    mapping=json.loads(Path(pins['frozen_frame_map']['path']).read_text())
    if (raw.dtype!=np.uint8 or raw.shape!=(n,480,640,3) or len(mapping)!=n
        or any(r.get('native_frame_index')!=i or r.get('output_frame')!=i for i,r in enumerate(mapping))):
        raise ValueError('DEVICE_PROPOSAL_FULL_NATIVE_SOURCE_DOMAIN')
    from .model_stages import configure_project_caches,load_frozen_models
    from chaoyang.governance.common import REPO_ROOT,artifact_ref,atomic_json
    configure_project_caches(packet['output_dir'])
    models=load_frozen_models(REPO_ROOT,packet['output_dir'],authorization=authority,
        frozen_inputs=[*packet['runtime_code'],*packet['model_weights']],need_sam=True,need_clean=False)
    authority();tracks=models['sam'].track_text(list(raw),'object',0);authority()
    from chaoyang.exact78_baseline.mask import unpack
    arrays={};rows=[];overlay=raw.copy();frame_counts=np.zeros(n,int)
    colors=np.asarray([[255,80,80],[80,255,80],[80,160,255],[255,200,50],[200,80,255],[80,255,220]],np.float32)
    for index,(obj,frames) in enumerate(sorted(tracks.items(),key=lambda v:int(v[0]))):
        masks=np.zeros((n,480,640),bool)
        for frame,packed in frames.items():
            frame=int(frame)
            if not 0<=frame<n:raise ValueError('DEVICE_PROPOSAL_FRAME_OUTSIDE_SOURCE')
            masks[frame]=unpack(packed,(480,640))
        key=f'object_{int(obj)}';arrays[key]=np.packbits(masks.reshape(n,-1),axis=1)
        presence=np.flatnonzero(masks.any(axis=(1,2))).tolist();frame_counts+=masks.any(axis=(1,2))
        rows.append({'array_key':key,'track_id':int(obj),'role':'object','identity_role':'UNKNOWN',
            'source':'SAM31_RGB_TEXT','native_shape':[n,480,640],'nonempty_native_frames':presence,
            'area_per_native_frame':masks.sum(axis=(1,2)).astype(int).tolist(),'deletion_authorized':False})
        overlay[masks]=np.rint(.5*overlay[masks]+.5*colors[index%len(colors)]).astype(np.uint8)
    root=Path(packet['output_dir']);out=root/'sessions'/sid/'Proposals';out.mkdir(parents=True,exist_ok=False)
    review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True);artifacts=[]
    def save_json(path,value):
        if path.exists():raise ValueError('DEVICE_PROPOSAL_FRESH_OUTPUT_REQUIRED')
        atomic_json(path,value);artifacts.append(artifact_ref(path))
    with (out/'GENERIC_OBJECT_PROPOSALS.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    artifacts.append(artifact_ref(out/'GENERIC_OBJECT_PROPOSALS.npz'))
    aborted=list(models['sam'].aborted)
    save_json(out/'GENERIC_OBJECT_PROPOSALS.json',{'schema':'v1-raw-generic-object-proposals-v1','source':pins['frozen_source_raw'],
        'frame_map':pins['frozen_frame_map'],'arrays':artifact_ref(out/'GENERIC_OBJECT_PROPOSALS.npz'),
        'query':'object','anchor_frame':0,'proposals':rows,'sam_aborted':aborted,'proposal_consumable':not aborted,
        'native_frames_submitted':n,'native_frames_with_nonempty_proposals':np.flatnonzero(frame_counts).tolist(),
        'model_actual_processed_frame_count':'NOT_INDEPENDENTLY_INSTRUMENTED','identity_role':'UNKNOWN',
        'input_policy':'FROZEN_RAW_ONLY_NO_SEEDS_OR_HISTORICAL_MASKS','new_manual_points':0,
        'deletion_mask_produced':False,'human_review':'PENDING','training_eligible':False,'control_ground_truth':False})
    save_json(out/'FRAME_MAP.json',mapping)
    save_json(out/'FRAME_STATUS.json',[{'native_frame_index':i,'rgb_proposal_count':int(frame_counts[i]),
        'proposal_role':'UNKNOWN','deletion_authorized':False,'mask_coverage_pass':False} for i in range(n)])
    from .source_frames import write_review_video
    from .v1_mco_worker import _comparison
    frames=_comparison(raw,overlay,list(range(n)),sid,'原始RGB','通用object提案：不是删除Mask')
    video=write_review_video(review/(sid+'_Generic_proposals.mp4'),frames,mapping);artifacts.append(video['video'])
    from PIL import Image
    for frame in packet['sessions'][0].get('review_anchors',[]):
        p=review/f'{sid}_Generic_F{frame:03d}.png';Image.fromarray(frames[frame]).save(p);artifacts.append(artifact_ref(p))
    authority();size=sum(r['bytes'] for r in artifacts)
    if size>packet['budgets']['output_bytes_estimate']:raise ValueError('DEVICE_PROPOSAL_OUTPUT_BUDGET')
    result={'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE','task_id':packet['task_id'],
        'run_signature':packet['run_signature'],'mode':'Mask','model_calls':1,'render_calls':0,
        'deletion_mask_produced':False,'proposal_consumable':not aborted,
        'model_calls_unit':'Sam3Video.track_text API','model_loads':{'SAM3.1':1,'ProPainter':0},'sessions':[{
        'session_key':sid,'native_frames':n,'native_output_frames':n,'completed_frame_indices':list(range(n)),
        'execution_complete':True,'frame_report_complete':True,'mask_quality_pass':False,'deletion_mask_produced':False,
        'wall_seconds':time.monotonic()-start,'video':video,'visual_quality_pass':False,'candidate_adopted':False,'human_review':'PENDING'}],
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'control_ground_truth':False,
        'visual_quality_pass':False,'training_eligible':False,'human_review':'PENDING'}
    save_json(root/'MODEL_RESULT.json',result)
    return 0
