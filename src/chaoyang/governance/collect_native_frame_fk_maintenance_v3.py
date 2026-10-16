"""Independent bounded collector. Revalidates upstream metadata/arrays/decode; never calls renderer."""
from pathlib import Path
import json,time,math
from chaoyang.governance.native_frame_fk_maintenance_contract_v3 import *
def collect(packet,root=ROOT):
 from chaoyang.native_frame_fk_render_candidate_v3 import driver
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 from chaoyang.pick_place_1008 import pipeline,source
 from chaoyang.exact78_baseline import render as rd
 import numpy as np
 started=time.perf_counter();c=approved_context(packet['inputs']['spec']['path'],root);i=validate_packet_context(packet,c,root)
 folder=root/'_run/current'/TASKS[i]/'attempts/attempt_0001/candidate';index=read(ref(folder/'INDEX.json'),root)
 need(index==read(ref(folder/'RESULT.json'),root),'FK_COLLECT_INDEX_RESULT')
 need(index['schema']=='native_frame_fk_dual_view_complete_candidate_index_v1' and index['status']=='COMPLETE' and index['operation']==OPS[i] and index['task_id']==TASKS[i] and index['stage']==c['spec']['stage'],'FK_COLLECT_SCHEMA')
 need(index['paired_native_frames']==(12 if i==0 else 295) and index['actual_control_views']==index['actual_candidate_views']==(24 if i==0 else 590),'FK_COLLECT_COMPLETE_VIEWS')
 need(index['sessions_complete']==index['sessions_requested']==len(SCOPES[i]) and [r['session_id'] for r in index['sessions']]==SCOPES[i],'FK_COLLECT_SCOPE')
 need(index['producer']==current_producer(root) and index['contract']==c['spec']['contract'] and index['plan']==c['plan_ref'],'FK_CURRENT_PRODUCER_PLAN')
 need(index['control_runtime_code']==c['contract']['control_runtime_code'] and index['candidate_runtime_code']==c['plan']['candidate_runtime_code'],'FK_EFFECTIVE_CODE')
 need(index['all_paired_BGR_alpha_roots_exact'] is True and index['same_q_Raw_Clean_masks_camera'] is True and index['contact_changes_q'] is False and type(index['parameters_fitted']) is int and index['parameters_fitted']==0,'FK_NO_INPUT_OR_Q_CHANGE')
 for k in ['physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible','Root_adoption','full_goal_achieved']:need(index[k] is False,'FK_NO_QUALIFICATION_UPGRADE')
 need(index['human_review_status']=='PENDING' and index['full_pipeline_speedup']=='UNKNOWN_NOT_MEASURED','FK_NO_VISUAL_OR_FULL_SPEED_CLAIM')
 for value in index['timing_seconds'].values():need(type(value) in (int,float) and math.isfinite(value) and value>=0,'FK_TIMING_FINITE')
 control,cs,cr,ns,nr=driver.load_pair_modules(c,root);control._PIN_VERIFIED_SHARED.clear()
 evidence=[]
 if i==1:
  need(index['qualification']==driver.qualification_closed(root,c),'FK_QUALIFICATION_CHAIN');driver.control_existing_closed(root,c)
 for row in index['sessions']:
  sid=row['session_id'];folders,case=driver.official_upstream(root,c['contract'],sid,control)
  need(row['paired_stage_receipt']['path']==str(folder/(sid+'_SEALED_PAIRED_STAGE.json')),'FK_STAGE_PATH')
  sealed=read(row['paired_stage_receipt'],root)
  need(sealed['schema']=='native_frame_fk_sealed_paired_stage_v1' and sealed['status']=='COMPLETE' and sealed['binding']==driver.paired_stage_binding(c,sid) and sealed['producer']==current_producer(root) and sealed['evidence']=={k:v for k,v in row.items() if k!='paired_stage_receipt'},'FK_FRESH_PAIRED_STAGE_BINDING')
  session=source.load_prepared(folders['source']/'source');motion=pipeline.load_arrays(folders['motion']/'actions.npz')
  need(np.array_equal(motion['qpc_ns'],session.qpc_ns),'FK_COLLECT_SOURCE_ACTION_QPC_ALL_NATIVE')
  currentK=rd.scaled_intrinsics(session.K,640,480,(session.width,session.height));external=rd.external_camera(np.linalg.inv(session.c2w[0])@motion['T_world_base'])
  def camera_check(view,frame):
   camera=np.linalg.inv(session.c2w[frame])@motion['T_world_base'] if view['view']=='ego' else external
   k=currentK if view['view']=='ego' else rd.EXTERNAL_K
   need(view['camera']==driver.array_pin(camera) and view['K']==driver.array_pin(k),'FK_COLLECT_CURRENT_ORIGINAL_CAMERA_K')
  need(driver.freeze_arrays(motion)==row['frozen_action_array_identity'],'FK_COLLECT_CURRENT_Q_ARRAY_EXACT')
  need(row['upstream']==case['reused_stage_receipts'] and row['raw_frame_count']==FULL[sid] and row['all_paired_BGR_alpha_roots_exact'] is True,'FK_COLLECT_UPSTREAM_SCOPE')
  if i==0:
   need(row['sampled_native_frames']==SAMPLES[sid] and len(row['rows'])==3,'FK_COLLECT_ALL12_NATIVE')
   for pair,frame in zip(row['rows'],SAMPLES[sid]):
    need(type(pair['original_QPC_ns']) is int and pair['original_QPC_ns']==int(session.qpc_ns[frame]),'FK_COLLECT_CURRENT_SOURCE_INTEGER_QPC')
    need(pair['native_frame']==frame and pair['control']==pair['candidate'] and pair['all_paired_BGR_alpha_roots_exact'] is True,'FK_COLLECT_LITERAL_PAIR')
    need(len(pair['control'])==2 and [v['view'] for v in pair['control']]==['ego','external'],'FK_COLLECT_TWO_VIEWS')
    need(pair['timing']['candidate']['trace']=={'render_FK_evaluations':3,'render_FK_hits':3},'FK_COLLECT_CACHE_TRACE')
    for v in pair['control']:
     camera_check(v,frame)
     need(v['native_frame']==frame and v['q_arm']==driver.array_pin(motion['q_arm'][frame]) and v['q_hand']==driver.array_pin(motion['q_hand'][frame]),'FK_COLLECT_VIEW_SAME_Q')
  else:
   for which in ['control','candidate']:
    need(set(row['files'][which])==set(pipeline.STAGE_FILES['render']),'FK_COLLECT_COMPLETE_RENDER_FILES')
    for name,p in row['files'][which].items():need(p['path']==str(folder/which/sid/name),'FK_COLLECT_FRESH_FILE_PATH');exact(p,root)
   need(row['paired_view_trace']['path']==str(folder/(sid+'_PAIRED_ALL_NATIVE_VIEW_TRACE.json')),'FK_COLLECT_TRACE_PATH')
   traces=read(row['paired_view_trace'],root)
   need(set(traces)=={'control','candidate'} and len(traces['control'])==len(traces['candidate'])==2*FULL[sid],'FK_COLLECT_ALL590_VIEWS')
   for j,(a,b) in enumerate(zip(traces['control'],traces['candidate'])):
    f=j//2;camera_check(a,f);camera_check(b,f);need(a['native_frame']==b['native_frame']==f and a['view']==b['view']==('ego' if j%2==0 else 'external'),'FK_COLLECT_NATIVE_VIEW_ORDER')
    clean=lambda v:{k:x for k,x in v.items() if k not in ['wall_seconds','CPU_process_seconds','FK_render_evaluations_delta','FK_render_hits_delta']}
    need(clean(a)==clean(b) and a['q_arm']==driver.array_pin(motion['q_arm'][f]) and a['q_hand']==driver.array_pin(motion['q_hand'][f]),'FK_COLLECT_ALL_PREENCODE_PIXELS_ROOTS_SAME_Q')
    if j%2==1:need(b['FK_render_evaluations_delta']==b['FK_render_hits_delta']==3,'FK_COLLECT_ALL_RENDER_CACHE3')
   session=source.load_prepared(folders['source']/'source')
   comparison=control.compare_fresh_render_outputs(folder/'control'/sid,folder/'candidate'/sid,session.qpc_ns,FULL[sid])
   existing=Path(c['contract']['control_existing_sessions'][sid]['files']['SESSION.json']['path']).parent
   replay=control.compare_fresh_render_outputs(existing,folder/'control'/sid,session.qpc_ns,FULL[sid])
   need(comparison==row['comparison'] and replay==row['control_replay_against_official_T2'],'FK_COLLECT_DECODE_PTS_CONTACT_FLAGS_FK_REPLAY')
  evidence.append({'session_id':sid,'current_global_Source_Motion_GPU_fresh_bindings_exact':True,'paired_exact_views':6 if i==0 else FULL[sid]*2})
 return {'schema':'native_frame_fk_dual_view_independent_collector_validation_v1','status':'PASSED','operation':OPS[i],'task_packet':ref(root/'tasks/current'/TASKS[i]/'TASK_PACKET.json'),'run_signature':packet['run_signature'],'candidate_index':ref(folder/'INDEX.json'),'paired_native_frames':12 if i==0 else 295,'sessions':evidence,'independent_collector_wall_seconds':time.perf_counter()-started,'real_render_calls_by_collector':0,'model_calls':0,'gain_scope':'SAVED_PAIRED_RENDER_CALLER_ONLY_NOT_FULL_PIPELINE','full_pipeline_speedup':'UNKNOWN_NOT_MEASURED','human_review_status':'PENDING','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False,'Root_adoption':False}
