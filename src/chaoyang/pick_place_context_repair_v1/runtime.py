from __future__ import annotations
from pathlib import Path
import copy,json,os,shutil,time
from datetime import datetime
import numpy as np
from chaoyang.pick_place_1008 import pipeline,source
from chaoyang.pick_place_1008.stage_reuse import _validate
from chaoyang.pick_place_1008.mask_seeds import seed_reference
from chaoyang.exact78_baseline.stage_reuse import current_producer
from .approval import ROOT,OPERATION,RECIPE,read,ref,exact,verify_root_planned_allowlist,verify_all_recipe_parents
from .bindings import repair_binding
from .context import PrivateContextHolePad4ProtectedProPainter,model_context_hole_masks
REQUIRED={'source':('source/SESSION_INPUT.json','source/physical_eye.mp4','source/source.npz'),'motion':('actions.npz','frame_mapping.jsonl','motion.json'),'gpu':('clean_rgb.npz','masks.json','masks.npz','raw_rgb.npz')}
def global_passed(receipt_pin):
    p=exact(receipt_pin);receipt=read(p);root=p.parents[2];outerp=root.parent/'RESULT.json';outer=read(outerp);idx=read(root/'INDEX.json')
    if outer.get('status')!='PASSED' or outer.get('worker_returncode')!=0 or outer.get('validation_errors')!=[] or idx.get('status')!='COMPLETE' or idx['sessions_complete']!=idx['sessions_requested']:
        raise ValueError('A_PARENT_GLOBAL_NOT_PASSED')
    if outer['required_outputs']['candidate/INDEX.json']!=ref(root/'INDEX.json'):raise ValueError('A_PARENT_INDEX_NOT_STRONGLY_BOUND')
    state=read(ROOT/'docs/governance/LONG_HORIZON_TASK_STATE.json');rows=[x for x in state['tasks'] if x['task_id']==outer['task_id']]
    if len(rows)!=1 or rows[0]['status']!='PASSED' or rows[0]['pid'] is not None or rows[0]['proc_start_ticks'] is not None or rows[0]['result']!=ref(outerp):
        raise ValueError('A_PARENT_NOT_CLOSED_GLOBAL')
    return receipt,root,idx
def prepare_frozen_context_inputs(contract):
    verify_all_recipe_parents(contract);parents=contract['parents'];manifest=read(exact(parents['manifest']));rows=[x for x in manifest['sessions'] if x['session_id']=='042']
    if len(rows)!=1 or rows[0]['raw_frame_count']!=168:raise ValueError('A_NATIVE_SOURCE_IDENTITY')
    row=rows[0];camera=parents['camera'];receipts={'source':parents['source_receipt'],'motion':parents['motion_receipt']};folders={}
    for stage in ('source','motion'):
        rc,root,_=global_passed(receipts[stage])
        expected=pipeline.binding(ROOT,stage,row,camera,{} if stage=='source' else {'source':receipts['source']})
        folder,pin=_validate(ROOT,root,'042',stage,expected,REQUIRED[stage])
        if folder is None or pin!=receipts[stage]:raise ValueError('A_CURRENT_SOURCE_MOTION_REUSE_MISS:'+stage)
        folders[stage]=folder
    gr,groot,gindex=global_passed(parents['baseline_GPU_receipt'])
    seed=seed_reference(exact(parents['seed_index']),'042')
    if seed!=parents['seed'] or gr['binding']['visible_mask_seed']!=seed:raise ValueError('A_UNCHANGED_SEED_DRIFT')
    expected=pipeline.binding(ROOT,'gpu',row,camera,{'source':receipts['source']},seed)
    gf,pin=_validate(ROOT,groot,'042','gpu',expected,REQUIRED['gpu'])
    if gf is None or pin!=parents['baseline_GPU_receipt']:raise ValueError('A_BASELINE_GPU_TYPED_INPUT_NOT_VALID_CURRENT')
    for name,key in [('masks.npz','baseline_mask_npz'),('masks.json','baseline_masks_metadata'),('raw_rgb.npz','canonical_RGB_raw_npz'),('clean_rgb.npz','baseline_Clean_npz')]:
        if gr['files'][name]!=parents[key]:raise ValueError('A_BASELINE_TYPED_ARTIFACT_NOT_GPU_BOUND')
    phase=time.perf_counter();session=source.load_prepared(folders['source']/'source');source_load_wall=time.perf_counter()-phase
    phase=time.perf_counter();masks=pipeline.load_masks(gf);masks_load_wall=time.perf_counter()-phase
    phase=time.perf_counter()
    with np.load(exact(parents['canonical_RGB_raw_npz']),allow_pickle=False) as z:raw=z['rgb']
    raw_read_wall=time.perf_counter()-phase
    if raw.shape!=(168,480,640,3) or raw.dtype!=np.uint8 or session.frame_count!=168:raise ValueError('A_CANONICAL_RGB_NATIVE_SHAPE')
    qpc=np.asarray([int(json.loads(v)) for v in exact(parents['source_QPC'],project_only=False).read_text().splitlines()],np.int64)
    if not np.array_equal(qpc,session.qpc_ns):raise ValueError('A_NATIVE_QPC_SOURCE_DRIFT')
    oldinfo=read(exact(parents['baseline_masks_metadata']))
    if oldinfo.get('source_pixel_domain')!='DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8':raise ValueError('A_CANONICAL_RGB_DOMAIN')
    return {'source_row':row,'camera':camera,'receipts':receipts,'folders':folders,'session':session,'raw':raw,'masks':masks,'oldinfo':oldinfo,'seed_ref':seed,'CPU_preflight_read_wall_seconds':{'source_load_prepared':source_load_wall,'masks_load':masks_load_wall,'canonical_RGB_npz_read':raw_read_wall}}
def _live_context_owner(approval_path,argv):
    from chaoyang.cli import _maintained_operations
    if OPERATION not in _maintained_operations():raise ValueError('A_OPERATION_NOT_FORMALLY_MAINTAINED')
    producer=current_producer(ROOT);packet=read(exact(producer['task_packet']))
    approval,contract=verify_root_planned_allowlist(approval_path,argv=argv,packet=packet,require_approved=True)
    if producer['task_id']!=approval['expected_task_id']:raise ValueError('A_LIVE_OWNER_NOT_APPROVED_TASK')
    lease=read(ROOT/'_run/current/GPU_LEASE.json')
    if lease.get('status')!='ACQUIRED' or lease.get('task_id')!=producer['task_id'] or lease.get('pid')!=os.getpid() or lease.get('gpu_process_pid')!=os.getpid():
        raise ValueError('A_FENCED_CURRENT_WORKER_GPU_LEASE_REQUIRED')
    ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    if lease.get('process_startticks')!=ticks or lease.get('executor_epoch')!=packet['writer']['executor_epoch'] or datetime.fromisoformat(lease['expires_at'])<=datetime.now().astimezone():
        raise ValueError('A_CURRENT_GPU_LEASE_IDENTITY_EPOCH_OR_EXPIRY')
    return producer,approval,contract
def run_registered_context_full(output,approval_path,argv):
    """No models/writes before maintained fence, live lease and Root exact planned scope."""
    started=time.perf_counter();producer,approval,contract=_live_context_owner(approval_path,argv)
    output=Path(output).resolve();expected=ROOT/'_run/current'/producer['task_id']/'attempts/attempt_0001/candidate'
    if output!=expected or output.exists():raise ValueError('A_EXECUTOR_FRESH_OUTPUT_PATH_REQUIRED')
    if contract.get('frozen_control_replay_required') is not True or contract.get('PP_inference_invocations_max')!=2:raise ValueError('A_EXPLICIT_FROZEN_CONTROL_REPLAY_CONTRACT_REQUIRED')
    phase=time.perf_counter();context=prepare_frozen_context_inputs(contract);preflight_wall=time.perf_counter()-phase;raw=context['raw'];masks=context['masks'];receipts=dict(context['receipts'])
    baseline_E=masks['effective_removal'].copy();P=masks['protected_visible'].copy()
    # Exact unchanged frozen parent constructor. Only the new context consumer differs.
    from chaoyang.exact78_baseline.clean import ProPainter
    from chaoyang.exact78_baseline.mask import Sam3Video
    import torch
    timing={}
    def completed_cuda_call(name,call):
        before=time.perf_counter();torch.cuda.synchronize();ready=time.perf_counter()
        value=call();torch.cuda.synchronize();done=time.perf_counter()
        timing[name]={'before_sync_wall_seconds':ready-before,'call_and_completed_GPU_wall_seconds':done-ready,'including_before_sync_wall_seconds':done-before,'cuda_completion_synchronized_before_and_after':True,'not_pure_GPU_kernel_time':True}
        return value
    sam=parent=wrapped=None
    try:
        sam=completed_cuda_call('SAM_constructor',lambda:Sam3Video(ROOT));sam_init_wall=timing['SAM_constructor']['call_and_completed_GPU_wall_seconds']
        parent=completed_cuda_call('ProPainter_constructor',lambda:ProPainter(ROOT));pp_init_wall=timing['ProPainter_constructor']['call_and_completed_GPU_wall_seconds']
        wrapped=PrivateContextHolePad4ProtectedProPainter(parent)
        trace={'schema':'actual_context_pad4_model_constructor_and_runtime_wall_trace_v1','SAM_constructor_invocations':1,'ProPainter_constructor_invocations':1,'checkpoint_file_load_count':'UNKNOWN_NOT_INSTRUMENTED','SAM_segmentation_inference_invocations':0,'sam_constructor_wall_seconds':sam_init_wall,'propainter_constructor_wall_seconds':pp_init_wall,'actual_cuda_allow_tf32_matmul':bool(parent.torch.backends.cuda.matmul.allow_tf32),'actual_cuda_allow_tf32_cudnn':bool(parent.torch.backends.cudnn.allow_tf32),'actual_cuda_autocast_enabled_before_PP':bool(parent.torch.is_autocast_enabled()),'timing_semantics':'wall time includes CPU/IO/sync; not GPU kernel profiling','actual_improvement':'UNKNOWN','CPU_preflight_total_wall_seconds':preflight_wall,'CPU_preflight_read_wall_seconds':context['CPU_preflight_read_wall_seconds'],'synchronized_model_phase_wall_ranges':timing}
        folder=output/'sessions/042';folder.mkdir(parents=True,exist_ok=False)
        from chaoyang.pick_place_1008.visual import ProtectedProPainter
        current_producer(ROOT)
        control=completed_cuda_call('frozen_control_PP',lambda:ProtectedProPainter(parent).inpaint(raw,masks['removal'],P))
        trace['frozen_control_PP_wall_seconds']=timing['frozen_control_PP']['call_and_completed_GPU_wall_seconds']
        validation_started=time.perf_counter()
        with np.load(exact(contract['parents']['baseline_Clean_npz']),allow_pickle=False) as z:oldclean=z['rgb']
        if oldclean.shape!=raw.shape or oldclean.dtype!=np.uint8:raise RuntimeError('A_CONTROL_BASELINE_CLEAN_SHAPE')
        same=np.array_equal(control,oldclean)
        delta=np.abs(control.astype(np.int16)-oldclean.astype(np.int16))
        control_record={'schema':'actual_full_native_frozen_ProtectedPP_replay_vs_sealed_Clean_v1','status':'PASS' if same else 'FAILED','native_frames':168,'exact_baseline_Clean_ref':contract['parents']['baseline_Clean_npz'],'array_equal':bool(same),'max_abs_channel_difference':int(delta.max()),'changed_pixels':int(np.any(delta,axis=-1).sum()),'per_native_changed_pixels':np.any(delta,axis=-1).sum((1,2)).tolist(),'same_parent_construction_order':'SAM_THEN_PROPainter','candidate_not_started_at_control_validation':True,'not_a_speed_AB_test':True,'physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False}
        pipeline.write(folder/'FROZEN_CONTROL_REPLAY_VALIDATION.json',control_record)
        trace['actual_frozen_control_replay_validation']=ref(folder/'FROZEN_CONTROL_REPLAY_VALIDATION.json')
        trace['actual_PP_inference_invocations']=1
        pipeline.write(folder/'MODEL_CONTEXT_REPAIR_TRACE.json',trace)
        del oldclean,delta,control
        trace['control_CPU_RGB_read_compare_validation_wall_seconds']=time.perf_counter()-validation_started
        if not same:raise RuntimeError('A_FROZEN_CONTROL_REPLAY_NOT_BITEXACT_SEALED_CLEAN')
        current_producer(ROOT);clean=completed_cuda_call('context4_candidate_PP',lambda:wrapped.inpaint(raw,masks['removal'],P));trace['context_PP_wall_seconds']=timing['context4_candidate_PP']['call_and_completed_GPU_wall_seconds'];trace['actual_PP_inference_invocations']=2;current_producer(ROOT)
        validation_started=time.perf_counter()
        if not np.array_equal(wrapped.effective_masks,baseline_E) or np.any(clean[P]!=raw[P]) or np.any(clean[~baseline_E]!=raw[~baseline_E]):
            raise RuntimeError('A_FIXED_FINAL_E_P_RGB_PROTECTION_FAILED')
        trace['candidate_fixed_E_P_CPU_validation_wall_seconds']=time.perf_counter()-validation_started
        masks['model_context_hole'],_=model_context_hole_masks(baseline_E,P,padding_px=4)
        masks['metadata']=copy.deepcopy(masks['metadata']);masks['metadata']['context_hole_repair']={'recipe_id':RECIPE,'contract':approval['contract_ref'],'original_failure':contract['parents']['original_actual_FAILED'],'original_quality_flags_retained':True,'final_write_E_and_P_fixed':True,'actual_source_pixel_domain':'DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8','observed_masks_provider':'BASELINE_GLOBAL_PASSED_GPU_MASKS_IMMUTABLE_TYPED_INPUT','observed_masks_producer_receipt':contract['parents']['baseline_GPU_receipt'],'new_SAM_segmentation_performed':False}
        disk_started=time.perf_counter()
        shutil.copyfile(exact(contract['parents']['canonical_RGB_raw_npz']),folder/'raw_rgb.npz')
        np.savez_compressed(folder/'clean_rgb.npz',rgb=clean);np.savez_compressed(folder/'masks.npz',**pipeline.pack_masks(masks))
        info=copy.deepcopy(context['oldinfo']);info.update(schema='pick_place_1008_protected_clean_context_pad4_repair_v1',status='COMPLETE',metadata=masks['metadata'],clean_validation=wrapped.last_validation,protected_pixel_channel_difference_count=0,outside_effective_difference_count=0,source_pixel_domain='DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8',repair_recipe_id=RECIPE,settings=parent.settings,training_eligible=False,control_ground_truth=False)
        trace['RGB_mask_NPZ_disk_write_wall_seconds']=time.perf_counter()-disk_started
        pipeline.write(folder/'MODEL_CONTEXT_REPAIR_TRACE.json',trace)
        info['actual_model_constructor_trace']=ref(folder/'MODEL_CONTEXT_REPAIR_TRACE.json')
        info['mask_role_frames']={name:int(value.any((1,2)).sum()) for name,value in masks.items() if isinstance(value,np.ndarray) and value.ndim==3 and value.dtype==bool}
        pipeline.write(folder/'masks.json',info)
        gb=repair_binding(ROOT,'gpu',context['source_row'],context['camera'],receipts,context['seed_ref'],approval)
        phase=time.perf_counter();receipts['gpu']=pipeline.seal(ROOT,folder,'gpu',gb);gpu_seal_wall=time.perf_counter()-phase
        current_producer(ROOT)
        phase=time.perf_counter();pipeline.render_worker(str(ROOT),str(folder),str(context['folders']['source']/'source'),str(context['folders']['motion']),str(folder));render_wall=time.perf_counter()-phase
        rb=repair_binding(ROOT,'render',context['source_row'],context['camera'],receipts,context['seed_ref'],approval)
        phase=time.perf_counter();receipts['render']=pipeline.seal(ROOT,folder,'render',rb);render_seal_wall=time.perf_counter()-phase
        validation_started=time.perf_counter()
        sealed=read(folder/'SESSION.json')
        if sealed['session_id']!='042' or sealed['frame_count']!=168 or any(sealed.get(k) is not False for k in ('control_ground_truth','physical_contact_ground_truth','training_eligible','contact_changes_q')):
            raise RuntimeError('A_FRESH_RENDER_NATIVE_OR_TRUTH_GATE')
        from chaoyang.pick_place_1008.audit import validate_action_arrays
        motion_pin=read(exact(receipts['motion']))['files']['actions.npz']
        if validate_action_arrays(motion_pin['path'],168):raise RuntimeError('A_UNCHANGED_MOTION_ARRAY_VALIDATION_FAILED')
        validation_wall=time.perf_counter()-validation_started
        fulltrace={'schema':'actual_completed_context_repair_phase_wall_trace_v1','model_phase_trace':ref(folder/'MODEL_CONTEXT_REPAIR_TRACE.json'),'frozen_control_replay_validation':ref(folder/'FROZEN_CONTROL_REPLAY_VALIDATION.json'),'CPU_preflight_wall_seconds':preflight_wall,'RGB_and_mask_disk_write_wall_seconds':trace['RGB_mask_NPZ_disk_write_wall_seconds'],'GPU_receipt_hash_verify_and_seal_wall_seconds':gpu_seal_wall,'CPU_render_worker_wall_seconds':render_wall,'render_receipt_hash_verify_and_seal_wall_seconds':render_seal_wall,'postrender_native_truth_action_validation_wall_seconds':validation_wall,'time_clock':'perf_counter_monotonic','total_before_index_wall_seconds':time.perf_counter()-started,'not_a_speed_AB_test':True,'actual_improvement':'UNKNOWN'}
        pipeline.write(folder/'FULL_CONTEXT_REPAIR_WALL_TRACE.json',fulltrace)
        mreceipt=read(exact(receipts['motion']));files={name:ref(folder/fname) for name,fname in {'session_result':'SESSION.json','final':'final.mp4','review':'review.mp4','frame_states':'frame_states.jsonl'}.items()}
        files['actions']=mreceipt['files']['actions.npz'];files['frame_mapping']=mreceipt['files']['frame_mapping.jsonl']
        row={'session_id':'042','raw_frame_count':168,'status':'COMPLETE','stage_receipts':receipts,'stages':{stage:{'folder':str(context['folders'][stage] if stage in context['folders'] else folder),'receipt':pin,**({'reused':True} if stage in ('source','motion') else {'reused':False} if stage=='gpu' else {})} for stage,pin in receipts.items()},'files':files,'visual_status':'PENDING_ACTUAL_LOCAL_AI_REVIEW','severe_issues_open':[],'accepted_defects':[],'human_review_status':'PENDING'}
        index={'schema':'pick_place_1008_pipeline_v1','status':'COMPLETE','stage':'full','sessions_requested':1,'sessions_complete':1,'frames_complete':168,'sessions':[row],'manifest':contract['parents']['manifest'],'camera_config':contract['parents']['camera'],'mask_seed_index':contract['parents']['seed_index'],'resource_recipe':{'cpu_source_workers':4,'cpu_render_workers':4,'thread_env_per_worker':2,'cpu_threads_total':10,'gpu_count':1},'repair_recipe_id':RECIPE,'actual_full_wall_trace':ref(folder/'FULL_CONTEXT_REPAIR_WALL_TRACE.json'),'actual_frozen_control_replay_validation':ref(folder/'FROZEN_CONTROL_REPLAY_VALIDATION.json'),'elapsed_seconds':time.monotonic()-started,'training_eligible':False,'control_ground_truth':False,'human_review_status':'PENDING'}
        current_producer(ROOT);pipeline.write(output/'INDEX.json',index);pipeline.write(output/'RESULT.json',index)
        return index
    finally:
        # Preserve original paired precision lifecycle through rendering.
        if sam is not None:sam.close()
        del wrapped,parent,sam
