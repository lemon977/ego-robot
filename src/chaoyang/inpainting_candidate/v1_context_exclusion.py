"""Uncertain reference exclusion is not enlarged final deletion authority."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from .contracts import verify_ref

def restore_fixed_write(raw,working,effective,protected):
    if raw.dtype!=np.uint8 or working.dtype!=np.uint8 or working.shape!=raw.shape:
        raise ValueError('CONTEXT_WRITE_NATIVE_RGB')
    if any(v.dtype!=bool or v.shape!=raw.shape[:3]for v in [effective,protected])or np.any(effective&protected):
        raise ValueError('CONTEXT_WRITE_FIXED_MASK_DOMAIN')
    result=raw.copy();result[effective]=working[effective]
    changed=np.any(result!=raw,axis=-1)
    validation={'status':'PASS','protected_changed_pixels_per_frame':np.count_nonzero(changed&protected,axis=(1,2)).tolist(),
        'outside_delete_changed_pixels_per_frame':np.count_nonzero(changed&~effective,axis=(1,2)).tolist(),
        'write_domain':'FROZEN_PARENT_EFFECTIVE_MASK_NOT_WORKING_MASK'}
    return result,validation

def execute(packet,*,authority):
    c=packet.get('config',{});b=packet['budgets']
    if (packet['mode']!='Clean' or c.get('context_exclusion_pair')is not True
        or b['gpu_count']!=1 or b['model_calls_max']!=3 or b['render_calls_max']!=0 or b['training_steps_max']!=0
        or len(packet['sessions'])!=1 or any(c.get(k)for k in ['cpu_equipment_rgb','cpu_raw_donor','cpu_spatial_clean',
        'generic_object_probe','paired_clean','mask_causal_pair','context_clean_pair'])):
        raise ValueError('CONTEXT_EXCLUSION_STRICT_THREE_CALL_PAIR')
    authority();start=time.monotonic()
    for ref in packet['runtime_code']:verify_ref(ref)
    from .v1_scene_donor_worker import sealed_inputs
    raw,_,masks,mapping=sealed_inputs(packet,'exclusion');pins=packet['inputs'];sid=packet['sessions'][0]['session_key']
    from .v1_mco_worker import _require_sealed_input,_label
    _require_sealed_input(pins['uncertain_working_masks'],pins['uncertain_working_model'],sid,'Mask')
    terminal=json.loads(Path(pins['uncertain_working_result']['path']).read_text())
    if terminal.get('status')!='PASSED'or terminal.get('model_result')!=pins['uncertain_working_model']or terminal.get('model_result_consumable')is not True:
        raise ValueError('CONTEXT_EXCLUSION_UNSEALED_WORKING_PRODUCER')
    producer=json.loads(Path(pins['uncertain_working_packet']['path']).read_text())
    if (producer.get('run_signature')!=terminal.get('run_signature')
        or any(producer['inputs'].get(k)!=pins[k]for k in ['frozen_source_raw','frozen_frame_map','frozen_repaired_masks'])):
        raise ValueError('CONTEXT_EXCLUSION_WORKING_SOURCE_MASK_IDENTITY')
    with np.load(pins['uncertain_working_masks']['path'],allow_pickle=False)as z:working={k:z[k]for k in z.files}
    if (any(v.dtype!=bool or v.shape!=raw.shape[:3]for v in working.values())
        or not np.array_equal(working['protected_visible'],masks['protected_visible'])
        or np.any(masks['removal']&~working['removal'])):
        raise ValueError('CONTEXT_EXCLUSION_SOURCE_OR_PROTECTION_DRIFT')
    from .model_stages import configure_project_caches,load_frozen_models
    configure_project_caches(packet['output_dir'])
    import torch
    from .v1_clean_pair import runtime_backend,require_same_backend,array_signature,comparison_outside_titles
    before=runtime_backend(torch,'cuda')
    models=load_frozen_models(Path(__file__).resolve().parents[3],packet['output_dir'],authorization=authority,
        frozen_inputs=[*packet['runtime_code'],*packet['model_weights']],need_sam=False,need_clean=True)
    require_same_backend(before,runtime_backend(models['propainter'].torch,models['propainter'].device))
    calls=[]
    def called(v):
        authority()
        if len(calls)>=3:raise ValueError('CONTEXT_EXCLUSION_MODEL_CALL_BUDGET')
        calls.append(v)
    from .v1_mask_clean_causal import run_mask_clean_causal
    pair=run_mask_clean_causal(models['propainter'],raw,masks['removal'],working['removal'],masks['protected_visible'],
        authority=authority,on_call=called,backend_before_load=before)
    working_b=pair['B'];final,validation=restore_fixed_write(raw,working_b,masks['effective_removal'],masks['protected_visible'])
    from chaoyang.governance.common import atomic_json,artifact_ref
    out=Path(packet['output_dir'])/'sessions'/sid/'Clean';out.mkdir(parents=True,exist_ok=False)
    review=Path(packet['review_dir']);review.mkdir(parents=True,exist_ok=True);artifacts=[]
    def js(p,d):atomic_json(p,d);artifacts.append(artifact_ref(p))
    def ar(p,**d):
        with p.open('xb')as f:np.savez_compressed(f,**d)
        artifacts.append(artifact_ref(p))
    ar(out/'WORKING_B_RGB.npz',rgb=working_b)
    output={'A':pair['A'],'B':final,'A_repeat':pair['A_repeat']}
    for v,name in [('A','CLEAN_A_RGB.npz'),('B','CLEAN_RGB.npz'),('A_repeat','CLEAN_A_REPEAT_RGB.npz')]:ar(out/name,rgb=output[v])
    ar(out/'FROZEN_MASKS.npz',write_effective=masks['effective_removal'],write_removal=masks['removal'],
        working_removal=working['removal'],working_effective=working['effective_removal'],protected=masks['protected_visible'])
    ar(out/'RANDOM_STATE_ARRAYS.npz',**pair['rng_arrays'])
    js(out/'WORKING_PAIR_RECEIPT.json',pair['receipt'])
    js(out/'WORKING_B_VALIDATION.json',pair['validations']['B']);js(out/'CLEAN_B_VALIDATION.json',validation)
    js(out/'CLEAN_A_VALIDATION.json',pair['validations']['A']);js(out/'CLEAN_A_REPEAT_VALIDATION.json',pair['validations']['A_repeat'])
    js(out/'CONTEXT_EXCLUSION_RECEIPT.json',{'source':pins['frozen_source_raw'],'write_mask':pins['frozen_repaired_masks'],
        'uncertain_working_masks':pins['uncertain_working_masks'],'working_producer_quality':'REJECTED_FOR_DELETION_NO_ADOPTION',
        'role':'UNCERTAIN_CONTEXT_EXCLUSION_ONLY','working_pair_receipt':artifact_ref(out/'WORKING_PAIR_RECEIPT.json'),
        'final_output_signatures':{k:array_signature(v)for k,v in output.items()},'B_write_validation':validation,
        'A_repeat_reproducibility':pair['receipt']['A_repeat_reproducibility'],'single_change':'INTERNAL_CONTEXT_EXCLUSION_FINAL_WRITE_MASK_FIXED',
        'calls':calls,'hidden':'UNKNOWN','training_eligible':False,'candidate_adopted':False})
    js(out/'FRAME_MAP.json',mapping);js(out/'FRAME_STATUS.json',[{**m,'mask_source':'FROZEN_PARENT_WRITE_MASK',
        'context_support':'UNCERTAIN_NOT_DELETION_EVIDENCE','geometry_decision':'UNKNOWN','time_mode':'OFFLINE_NONCAUSAL',
        'human_review':'PENDING','training_eligible':False}for m in mapping])
    from .source_frames import write_review_video
    video=write_review_video(out/'clean.mp4',final,mapping);artifacts.append(video['video'])
    frames=comparison_outside_titles([pair['A'],final,pair['A_repeat']],
        ['A：三轮Mask＋原补图','B：可疑参考排除／原写域','A复测：原补图'],list(range(len(raw))),sid,_label)
    comparison=write_review_video(review/(sid+'_Context_Exclusion.mp4'),frames,mapping);artifacts.append(comparison['video'])
    from PIL import Image
    for i in packet['sessions'][0].get('review_anchors',[]):
        p=review/f'{sid}_Context_Exclusion_F{i:03d}.png';Image.fromarray(frames[i]).save(p);artifacts.append(artifact_ref(p))
    authority();size=sum(r['bytes']for r in artifacts)
    if size>b['output_bytes_estimate']:raise ValueError('CONTEXT_EXCLUSION_OUTPUT_BUDGET')
    js(Path(packet['output_dir'])/'MODEL_RESULT.json',{'schema':'inpainting-v1-mco-stage-result-v1','status':'SEALED_COMPLETE',
        'task_id':packet['task_id'],'run_signature':packet['run_signature'],'mode':'Clean','model_calls':len(calls),'render_calls':0,
        'artifacts':artifacts,'artifact_bytes':size,'candidate_adopted':False,'control_ground_truth':False,
        'training_eligible':False,'visual_quality_pass':False,'human_review':'PENDING','sessions':[{
        'session_key':sid,'native_frames':len(raw),'native_output_frames':len(raw),'completed_frame_indices':list(range(len(raw))),
        'execution_complete':True,'wall_seconds':time.monotonic()-start,'video':video,'comparison':comparison,
        'visual_quality_pass':False,'candidate_adopted':False,'human_review':'PENDING'}]})
    return 0
