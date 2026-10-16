"""One fixed-cohort registered CPU task, original S/G, newM, bounded four renderers."""
from pathlib import Path
import os,time,json,copy,multiprocessing,numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from .approval import ROOT,ref,exact,read,FIXED,verify_runtime_approval,MOTION_RENDER
from .collector import evaluation_selection
from .bindings import family_binding
from .runtime import metadata_context,write
def init_render_cpu_worker():
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 import cv2
 cv2.setNumThreads(1)
def registered_motion_render(output,approval_path,argv):
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 from chaoyang.pick_place_1008 import pipeline
 from chaoyang.pick_place_1008.stage_reuse import _validate
 from chaoyang.exact78_baseline.robot_model import Robot
 producer=current_producer(ROOT);packet=read(exact(producer['task_packet']))
 approval,contract=verify_runtime_approval(approval_path,argv=argv,packet=packet)
 if approval['operation']!=MOTION_RENDER:raise ValueError('QP_WRONG_REGISTERED_OPERATION')
 output=Path(output).resolve()
 if output!=ROOT/'_run/current'/producer['task_id']/'attempts/attempt_0001/candidate':raise ValueError('QP_EXECUTOR_OWNED_FRESH_OUTPUT_REQUIRED')
 evaluation,evalrows=evaluation_selection(approval)
 h,q,c,cohort,infos=metadata_context();robot=Robot(ROOT)
 started=time.monotonic();deadline=started+14400;output.mkdir(parents=True,exist_ok=False)
 results={};render_jobs=[]
 # No render pool exists during source/GPU validation and new Motion construction.
 for info,ev in zip(infos,evalrows):
  sid=info['session_id'];n=info['native_frames'];folder=output/'sessions'/sid;folder.mkdir(parents=True,exist_ok=False);rc={}
  try:
   if time.monotonic()>deadline:raise TimeoutError('QP_MOTION_RENDER_FINITE_WALL_EXHAUSTED')
   old=info['current_row'];sr=c.source[sid]
   source_ref=old['stage_receipts']['source'];gpu_ref=old['stage_receipts']['gpu'];oldmr=read(exact(old['stage_receipts']['motion']))
   expected_source=pipeline.binding(ROOT,'source',sr,c.camera,{})
   source_folder,sp=_validate(ROOT,Path(source_ref['path']).parents[2],sid,'source',expected_source,tuple(pipeline.STAGE_FILES['source']))
   if source_folder is None or sp!=source_ref:raise ValueError('QP_SOURCE_STRICT_OLD_REUSE_MISS')
   baseline_producer=c.producer(info['producer_task_id']);oldpacket=read(exact(baseline_producer['packet_ref']));oldspec=read(exact(oldpacket['inputs']['spec']))
   from chaoyang.pick_place_1008.mask_seeds import seed_reference
   expected_gpu=pipeline.binding(ROOT,'gpu',sr,c.camera,{'source':source_ref},seed_reference(Path(h.arg(oldspec['args'],'--mask-seeds')),sid))
   gpu_folder,gp=_validate(ROOT,Path(gpu_ref['path']).parents[2],sid,'gpu',expected_gpu,tuple(pipeline.STAGE_FILES['gpu']))
   if gpu_folder is None or gp!=gpu_ref:raise ValueError('QP_GPU_STRICT_OLD_REUSE_MISS')
   data=read(exact(ev['report']))
   with np.load(exact(oldmr['files']['actions.npz']),allow_pickle=False) as z:actions={k:z[k] for k in z.files}
   with np.load(exact(data['private_candidate_arrays']),allow_pickle=False) as z:candidate={k:z[k] for k in z.files}
   if not np.array_equal(actions['q_arm'],candidate['original_q_arm']) or not np.array_equal(actions['q_hand'],candidate['unchanged_original_q_hand']) or not np.array_equal(actions['qpc_ns'],candidate['unchanged_original_qpc_ns']) or not np.array_equal(actions['T_world_base'],candidate['unchanged_original_T_world_base']):raise ValueError('QP_EVALUATION_NOT_CURRENT_ORIGINAL_ACTIONS')
   original_actions={k:x.copy() for k,x in actions.items()}
   changed=['q_arm','arm_velocity_rad_s','arm_acceleration_rad_s2','qpc_velocity_violation','qpc_acceleration_violation','arm_position_mm','arm_rotation_deg']
   for dest,key in zip(changed,['private_candidate_q_arm','actual_candidate_QPC_velocity_rad_s','actual_candidate_QPC_acceleration_rad_s2','candidate_rate_velocity_fail','candidate_rate_acceleration_fail','diagnostic_only_position_mm_all_native','diagnostic_only_rotation_deg_all_native']):actions[dest]=candidate[key]
   for k in original_actions:
    if k not in changed and not np.array_equal(actions[k],original_actions[k]):raise ValueError('QP_SINGLE_FACTOR_CHANGED_OTHER_FIELD:'+k)
   for frame in range(n):
    if time.monotonic()>deadline:raise TimeoutError('QP_MOTION_RENDER_FINITE_WALL_EXHAUSTED')
    for side in range(2):
     actual=actions['T_world_base']@robot.arm_frames(side,actions['q_arm'][frame,side])['root']
     if not np.array_equal(actual,candidate['candidate_all_native_fk_roots'][side,frame]):raise ValueError('QP_ALL_NATIVE_NEW_Q_FK_DRIFT')
   np.savez_compressed(folder/'actions.npz',**actions)
   (folder/'frame_mapping.jsonl').write_bytes(exact(oldmr['files']['frame_mapping.jsonl']).read_bytes())
   motion=copy.deepcopy(read(exact(oldmr['files']['motion.json'])))
   motion['baseline_original_retarget_metrics_immutable']=copy.deepcopy(motion['metrics'])
   motion['qpc_projection']={'recipe_id':'FIXED_REAL_QPC_MINIMAL_L2_ARM_TRAJECTORY_PROJECTION_V1','actual_evaluation':ev['report'],'candidate_arrays':data['private_candidate_arrays'],'old_NUM_FAIL_retained':True,'source_invalid_pose_points':data['source_invalid_pose_point_indices'],'derived_target_precision_not_physical_GT':True}
   v=actions['arm_velocity_rad_s'];a=actions['arm_acceleration_rad_s2'];vf=actions['qpc_velocity_violation'];af=actions['qpc_acceleration_violation']
   motion['qpc_diagnostics']={'max_arm_velocity_rad_s':float(abs(v).max()),'max_arm_acceleration_rad_s2':float(abs(a).max()),'limits_rad_s':3.6,'limits_rad_s2':54.,'velocity_violation_raw_frame_indices':np.flatnonzero(vf.any(axis=1)).tolist(),'acceleration_violation_raw_frame_indices':np.flatnonzero(af.any(axis=1)).tolist(),'invalid_intervals_remain_distinct':True,'qualification':'QPC_LIMITS_PASSED','velocity_violation_count_per_side':vf.sum(axis=0).tolist(),'acceleration_violation_count_per_side':af.sum(axis=0).tolist()}
   motion['metrics']={'frame_count':n,'time_derivative':'ACTUAL_RAW_QPC_DT_AND_ADJACENT_DT_MEAN','gates':{'arm_accel_rad_s2':54.,'arm_speed_rad_s':3.6,'track_position_mm':20.,'track_rotation_deg':15.},'per_side':{}}
   for side,name in enumerate(['left','right']):
    known=candidate['source_pose_applicable'][side]
    motion['metrics']['per_side'][name]={'arm_accel_rad_s2_max':float(abs(a[:,side]).max()),'arm_speed_rad_s_max':float(abs(v[:,side]).max()),'source_qualified_pose_frames':int(known.sum()),'source_invalid_pose_UNKNOWN_frames':int((~known).sum()),'position_mm_max_known':float(actions['arm_position_mm'][side,known].max()) if known.any() else None,'rotation_deg_max_known':float(actions['arm_rotation_deg'][side,known].max()) if known.any() else None}
   motion['physical_contact_ground_truth']=False;motion['control_ground_truth']=False;motion['training_eligible']=False
   write(folder/'motion.json',motion)
   rc['source']=source_ref
   mb=family_binding(ROOT,'motion',sr,c.camera,{'source':source_ref},approval,approval['exact_extra_runtime_code'])
   rc['motion']=pipeline.seal(ROOT,folder,'motion',mb);rc['gpu']=gpu_ref
   rb=family_binding(ROOT,'render',sr,c.camera,dict(rc),approval,approval['exact_extra_runtime_code'])
   render_jobs.append((sid,n,folder,source_folder,gpu_folder,rc,rb,ev,data))
   print(json.dumps({'progress':'FIXED_QPC_MOTION_PREPARED_FRESH','session_id':sid,'native_frames':n}),flush=True)
  except Exception as e:results[sid]={'session_id':sid,'status':'FAILED','raw_frame_count':n,'stage_receipts':dict(rc),'error_type':type(e).__name__,'error':str(e),'partial_never_reused':True}
 # Four CPU render workers are fixed resource concurrency only, same frozen renderer body.
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init_render_cpu_worker) as pool:
  futures={}
  for job in render_jobs:
   sid,n,folder,source_folder,gpu_folder,rc,rb,ev,data=job
   future=pool.submit(pipeline.render_worker,str(ROOT),str(folder),str(source_folder/'source'),str(folder),str(gpu_folder));futures[future]=job
  for future in as_completed(futures):
   sid,n,folder,source_folder,gpu_folder,rc,rb,ev,data=futures[future]
   try:
    if time.monotonic()>deadline:raise TimeoutError('QP_MOTION_RENDER_FINITE_WALL_EXHAUSTED')
    future.result()
    from .output_audit import audit_new_session
    qa_ref=audit_new_session(folder,source_folder,gpu_folder,ev['report'])
    session_info=read(folder/'SESSION.json');session_info['outputs']['new_qpc_motion_render_CPU_QA']=qa_ref
    # Only this new producer metadata is extended; the frozen renderer body and images are unchanged.
    with (folder/'SESSION.json').open('w') as f:json.dump(session_info,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    rc['render']=pipeline.seal(ROOT,folder,'render',rb)
    results[sid]={'session_id':sid,'status':'COMPLETE','raw_frame_count':n,'stage_receipts':dict(rc),'new_q_evaluation':ev['report'],'source_invalid_qualified_pose_UNKNOWN_retained':data['source_invalid_pose_point_indices'],'AI_review':'PENDING_NEW_SERVER_LOCAL_AI','human_review':'PENDING','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False}
   except Exception as e:results[sid]={'session_id':sid,'status':'FAILED','raw_frame_count':n,'stage_receipts':dict(rc),'error_type':type(e).__name__,'error':str(e),'partial_never_reused':True}
 rows=[results[s] for s in FIXED]
 ix={'schema':'fixed16_QPC_fresh_motion_render_candidate_v1','status':'COMPLETE' if all(x['status']=='COMPLETE' for x in rows) else 'INCOMPLETE','sessions_requested':16,'sessions_complete':sum(x['status']=='COMPLETE' for x in rows),'frames_requested':2578,'frames_complete':sum(x['raw_frame_count'] for x in rows if x['status']=='COMPLETE'),'sessions':rows,'manifest':c.mref,'camera_config':c.camera,'evaluation':evaluation,'resource_recipe':{'cpu_threads_total':10,'render_workers':4,'render_child_threads':2,'Motion_and_Render_pools_overlap':False,'GPU_count':0,'wall_seconds_max':14400,'render_calls_max':5156},'raw_Clean_depth_metric_contact_GT_not_changed':True,'actual_GPU_SAM_ProPainter_calls':0,'actual_visual_inspection':False,'human_review':'PENDING','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False,'full_goal_achieved':False}
 write(output/'INDEX.json',ix);write(output/'RESULT.json',ix);return ix
