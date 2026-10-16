"""Only the registered single-worker DEV4 fresh Render/Contact task."""
from pathlib import Path
import json,time,resource,os,gc
from .contract import *
FILES=('final.mp4','review.mp4','frame_states.jsonl','contact_events.json','render_validation.npz','SESSION.json')
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def binding(ctx,c):
 return dict(schema='q_frozen_observed_cup_fresh_render_contact_binding_v1',stage='render_contact',session_id=c['session_id'],native_frames=c['native_frames'],views=2,spec=ctx['plan']['approved_spec'],plan=ctx['plan_ref'],contract=ctx['spec']['contract'],runtime=ctx['packet']['runtime_code'],original_source_motion_gpu_inputs=c['old_stage_receipts'],actual_visual_clean_postprocess=c['actual_clean_stage'],robot_asset_input_refs=ctx['contract']['robot_asset_input_refs'],fixed_existing_runtime_code=ctx['contract']['fixed_existing_runtime_code'],fresh_Render=True,fresh_Contact=True,old_GPU_is_input_only=True,same_q_source_camera_seed_refs=c['frozen_identity_refs'],model_calls=0)
def run(ctx):
 import cv2
 cv2.setNumThreads(1);resource_shape(ctx['contract']['resource_bounds']);resource.setrlimit(resource.RLIMIT_AS,(RESOURCE_BOUNDS['address_space_bytes_max'],RESOURCE_BOUNDS['address_space_bytes_max']))
 affinity=sorted(os.sched_getaffinity(0));need(len(affinity)>=2,'FR_CPU2_AFFINITY_AVAILABLE');os.sched_setaffinity(0,set(affinity[:2]))
 start=time.perf_counter();times={};peak=0
 def tick():
  nonlocal peak
  ctx['guard']();need(time.perf_counter()-start<1800,'FR_FINITE_WALL1800')
  vals={}
  try:
   for line in Path('/proc/self/status').read_text().splitlines():
    key,sep,text=line.partition(':')
    if key in ['VmRSS','VmHWM']:
     p=text.split();need(sep and len(p)==2 and p[0].isdigit() and p[1]=='kB' and key not in vals,'FR_PROCESS_MEMORY_SCHEMA');vals[key]=int(p[0])*1024
  except OSError as e:raise ValueError('FR_PROCESS_MEMORY_READ_FAILED:'+str(e)) from e
  need(set(vals)=={'VmRSS','VmHWM'} and vals['VmHWM']>=vals['VmRSS']>0,'FR_PROCESS_MEMORY_MISSING');peak=max(peak,vals['VmHWM']);need(peak<=RESOURCE_BOUNDS['resident_peak_bytes_max'],'FR_PROCESS_MEMORY_CAP')
 tick();t=time.perf_counter();verify_baselines(ctx['contract'],tick=tick);verify_clean_parent(ctx['contract'],tick=tick);verify_robot_assets(ctx['contract'],tick=tick);times['strong_old_S_M_G_and_actual_Clean_reuse_validate_seconds']=time.perf_counter()-t
 from .render_adapter import load_case
 from .private_render import render_session
 out=ROOT/'_run/current'/TASK/'attempts/attempt_0001/candidate';need(not out.exists(),'FR_FRESH_CANDIDATE');out.mkdir();rows=[]
 for c in ctx['contract']['cases']:
  tick();t=time.perf_counter();session,motion,clean,masks=load_case(c,tick);loaded=time.perf_counter()-t
  folder=out/c['session_id'];t=time.perf_counter();value=render_session(ROOT,folder,session,motion,clean,masks,tick=tick,asset_guard=lambda robot:validate_live_robot_assets(robot,ctx['packet'],ctx['contract'],tick=tick));renderwall=time.perf_counter()-t;need(value['actual_render_view_count']==2*c['native_frames'],'FR_ACTUAL_RENDER_VIEW_COUNT')
  tick();t=time.perf_counter();files={k:ref(folder/k,tick=tick) for k in FILES};rc=dict(schema='pick_place_q_frozen_observed_cup_fresh_render_contact_sealed_stage_v1',status='COMPLETE',producer=ctx['producer'],binding=binding(ctx,c),files=files,upstream_original_gpu_input=c['old_stage_receipts']['gpu'],upstream_new_visual_clean=c['actual_clean_stage'],physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False)
  write(folder/'STAGE_render_contact.json',rc);sealwall=time.perf_counter()-t
  rows.append(dict(session_id=c['session_id'],status='COMPLETE',raw_frame_count=c['native_frames'],native_views=2*c['native_frames'],stage_receipt=ref(folder/'STAGE_render_contact.json',tick=tick),files=files,old_source_motion_gpu_receipts=c['old_stage_receipts'],actual_visual_clean=c['actual_clean_stage'],frozen_identity_refs=c['frozen_identity_refs'],baseline_quality_and_NUM_history=c['old_quality_and_NUM_preserved'],timing=dict(load_arrays_seconds=loaded,fresh_render_session_including_original_encode_decode_contact_seconds=renderwall,stage_seal_hash_seconds=sealwall)))
  del session,motion,clean,masks;gc.collect()
 tick();ctx['full_exit_guard']();times['driver_wall_seconds']=time.perf_counter()-start
 index=dict(schema='q_frozen_observed_cup_fresh_render_contact_candidate_index_v1',status='COMPLETE',task_id=TASK,operation=OPERATION,producer=ctx['producer'],spec=ctx['plan']['approved_spec'],plan=ctx['plan_ref'],contract=ctx['spec']['contract'],sessions=rows,sessions_requested=4,sessions_complete=4,frames_requested=760,frames_complete=760,native_view_count=1520,render_calls=1520,model_calls=0,fresh_FK_executed=True,low_level_FK_function_call_count='UNKNOWN_NOT_EQUIVALENT_TO_VIEW_COUNT',new_Render_and_Contact=True,old_GPU_is_input_only=True,q_changed=False,source_flags_preserved=True,NUM_flags_preserved=True,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING',quality_accepted=False,automatic_AI=False,Root_adoption=False,full_goal_achieved=False,timing=times,worker_peak_process_VmHWM_bytes=peak,raster_only_gain='UNKNOWN_UNMEASURED',matched_control_full_render_timing='UNKNOWN')
 for k in ['INDEX.json','RESULT.json']:write(out/k,index)
 return 0
