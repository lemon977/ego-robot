"""Strict zero-neural CPU consumer of sealed Source/Mask/Clean components."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from chaoyang.governance.common import atomic_json,artifact_ref
from .contracts import verify_ref
from .v1_scene_donor import registrations,donor_fill


def require_cpu_donor(packet):
    b=packet.get('budgets',{});c=packet.get('config',{})
    if (packet.get('mode')!='Clean' or c.get('cpu_raw_donor') is not True
        or any(b.get(k)!=0 for k in ('gpu_count','model_calls_max','render_calls_max','training_steps_max'))
        or packet.get('model_weights') or packet.get('gpu_mapping')
        or len(packet.get('sessions',[]))!=1 or 'output_frames' in packet['sessions'][0]
        or c.get('donor_count')!=12
        or c.get('donor_registration','GLOBAL_ANCHOR_HOMOGRAPHY') not in ('GLOBAL_ANCHOR_HOMOGRAPHY','DIRECT_LOCAL_AFFINE')
        or any(c.get(k) for k in ('paired_clean','mask_causal_pair','context_clean_pair'))):
        raise ValueError('DONOR_STRICT_CPU_FULL_NATIVE_CONTRACT')


def _json(ref):
    verify_ref(ref);return json.loads(Path(ref['path']).read_text())


def sealed_inputs(packet, prefix='donor'):
    pins=packet['inputs'];sid=packet['sessions'][0]['session_key']
    for ref in pins.values():verify_ref(ref)
    result=_json(pins[prefix+'_parent_result']);model=_json(pins[prefix+'_parent_model'])
    if (result.get('status')!='PASSED' or result.get('model_result_consumable') is not True
        or result.get('model_result')!=pins[prefix+'_parent_model']
        or model.get('status')!='SEALED_COMPLETE'):
        raise ValueError('DONOR_TERMINAL_SEALED_PARENT_REQUIRED')
    stages={}
    bindings=[('Source','frozen_source_receipt','frozen_source_raw'),
              ('Mask','frozen_mask_receipt','frozen_repaired_masks'),
              ('Clean',prefix+'_clean_receipt',prefix+'_parent_clean')]
    for stage,receipt_key,array_key in bindings:
        receipt=_json(pins[receipt_key])
        if (receipt.get('schema')!='inpainting-candidate-component-v1'
            or receipt.get('status')!='SEALED_COMPLETE' or receipt.get('session_id')!=sid
            or receipt.get('native_frames')!=packet['sessions'][0]['native_frames']
            or receipt.get('stage')!=stage or pins[array_key] not in receipt.get('outputs',[])):
            raise ValueError('DONOR_COMPONENT_BINDING:'+stage)
        stages[stage]=receipt
    if (pins['frozen_frame_map'] not in stages['Source']['outputs']
        or stages['Clean'].get('upstream',{}).get('Mask')!=pins['frozen_mask_receipt']
        or stages['Clean'].get('upstream',{}).get('Source')!=pins['frozen_source_receipt']):
        raise ValueError('DONOR_UPSTREAM_SOURCE_MASK_MISMATCH')
    with np.load(pins['frozen_source_raw']['path'],allow_pickle=False) as z:raw=z['rgb']
    with np.load(pins[prefix+'_parent_clean']['path'],allow_pickle=False) as z:clean=z['rgb']
    with np.load(pins['frozen_repaired_masks']['path'],allow_pickle=False) as z:masks={k:z[k] for k in z.files}
    mapping=_json(pins['frozen_frame_map'])
    if (len(raw)!=packet['sessions'][0]['native_frames'] or len(mapping)!=len(raw)
        or any(row.get('native_frame_index')!=i or row.get('output_frame')!=i for i,row in enumerate(mapping))
        or any(v.dtype!=bool or v.shape!=raw.shape[:3] for v in masks.values())):
        raise ValueError('DONOR_NATIVE_MAPPING_DOMAIN')
    return raw,clean,masks,mapping


def execute_raw_donor(packet,*,authority):
    require_cpu_donor(packet);authority();started=time.monotonic()
    for ref in packet['runtime_code']:verify_ref(ref)
    raw,clean,masks,mapping=sealed_inputs(packet)
    excluded=masks['effective_removal']|masks['protected_visible']|masks['unknown']|masks['core']|masks['equipment']
    local=packet['config'].get('donor_registration')=='DIRECT_LOCAL_AFFINE'
    provider=None
    if local:
        from .v1_scene_donor_local import LocalAffineMappings
        provider=LocalAffineMappings(raw,excluded,masks['effective_removal'],authority)
        H=np.empty((0,2,3,3));diagnostics=provider.diagnostics;anchor=None
    else:
        H,diagnostics,anchor=registrations(raw,excluded,authority)
    candidate,provenance,counts,evidence=donor_fill(raw,clean,masks['effective_removal'],
        masks['protected_visible'],excluded,H,donor_count=12,authority=authority,mapping_provider=provider)
    sid=packet['sessions'][0]['session_key'];root=Path(packet['output_dir']);out=root/'sessions'/sid/'Clean'
    out.mkdir(parents=True,exist_ok=False);review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    def save_json(path,value):
        if path.exists():raise ValueError('DONOR_FRESH_OUTPUT_REQUIRED')
        atomic_json(path,value);artifacts.append(artifact_ref(path))
    def save_npz(path,**arrays):
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        artifacts.append(artifact_ref(path))
    save_npz(out/'CLEAN_RGB.npz',rgb=candidate)
    save_npz(out/'DONOR_PROVENANCE.npz',source_frame_xy=provenance,support_count=counts)
    save_npz(out/'REGISTRATION_MATRICES.npz',H=H)
    save_json(out/'REGISTRATION_DIAGNOSTICS.json',{'anchor':anchor,'frames':diagnostics,
        'registration_recipe':'DIRECT_LOCAL_AFFINE' if local else 'GLOBAL_ANCHOR_HOMOGRAPHY',
        'pixel_agreement':'TWO_DISJOINT_FITS_SAME_NEAREST_INTEGER_DONOR_PIXEL',
        'SIFT_nfeatures':2000,'homography_ransac_px':3.0,'homography_qualified':False})
    changed=np.any(candidate!=raw,axis=-1)
    evidence.update(native_frames=len(raw),anchor_frame=anchor,
        protected_changed_pixels=int((changed&masks['protected_visible']).sum()),
        outside_effective_changed_pixels=int((changed&~masks['effective_removal']).sum()),
        total_effective_pixels=int(masks['effective_removal'].sum()),
        total_supported_pixels=int((provenance[...,0]>=0).sum()),
        registration_status='PLANAR_ASSUMPTION_NOT_METRIC_OR_PIXEL_TRUTH',
        source_mask_completeness='UNVERIFIED_POSSIBLE_MISSED_DEVICE',
        production_task_specific_branches=False,human_points_added=0)
    save_json(out/'DONOR_EVIDENCE.json',evidence)
    save_json(out/'FRAME_MAP.json',mapping)
    save_json(out/'FRAME_STATUS.json',[{'native_frame_index':i,'geometry_decision':'UNKNOWN',
        'display_policy':'RAW_VISIBLE_DONOR_PROPOSAL_ELSE_FROZEN_CLEAN','time_mode':'OFFLINE_NONCAUSAL',
        'visual_quality_pass':False,'human_review':'PENDING','supported_pixels':r['supported_pixels']}
        for i,r in enumerate(evidence['per_frame'])])
    from .source_frames import write_review_video
    from .v1_mco_worker import _comparison
    completed=write_review_video(out/'clean.mp4',candidate,mapping);artifacts.append(completed['video'])
    compare=_comparison(clean,candidate,list(range(len(raw))),sid,'父候选：三轮Mask＋原补图','本轮候选：Raw供体／配准未证实')
    comparison=write_review_video(review/(sid+'_Donor_comparison.mp4'),compare,mapping);artifacts.append(comparison['video'])
    from PIL import Image
    for frame in packet['sessions'][0].get('review_anchors',[]):
        p=review/f'{sid}_Donor_F{frame:03d}.png';Image.fromarray(compare[frame]).save(p);artifacts.append(artifact_ref(p))
    authority()
    size=sum(r['bytes'] for r in artifacts)
    if size>packet['budgets']['output_bytes_estimate']:raise ValueError('DONOR_OUTPUT_BUDGET')
    result={'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE',
        'task_id':packet['task_id'],'run_signature':packet['run_signature'],'mode':'Clean',
        'model_calls':0,'render_calls':0,'cpu_raw_donor':True,'sessions':[{
            'session_key':sid,'native_frames':len(raw),'native_output_frames':len(raw),
            'completed_frame_indices':list(range(len(raw))),'execution_complete':True,
            'wall_seconds':time.monotonic()-started,'video':completed,'comparison':comparison,
            'visual_quality_pass':False,'human_review':'PENDING','candidate_adopted':False}],
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'control_ground_truth':False,
        'training_eligible':False,'visual_quality_pass':False,'human_review':'PENDING'}
    save_json(root/'MODEL_RESULT.json',result)
    return 0
