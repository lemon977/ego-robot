"""Independent all-native byte/policy collector. No model/FK/Render calls."""
from pathlib import Path
import numpy as np
from .contract import *
from .helper import FIELDS,OLD_SCHEMA,NEW_SCHEMA
from .stream import Frames
def audit_frame(r,o,p,x,y,m,f):
 """Independent formulas; never invokes the candidate pure helper."""
 need(r.dtype==o.dtype==x.dtype==np.uint8 and r.shape==o.shape==x.shape==(1,480,640,3),'CUP_COLLECT_UINT8_NATIVE_FRAME')
 restore=m['semantic_conflict']&m['effective_removal'];P=m['cup_observed'];E=m['effective_removal']&~P
 expected=o.copy();expected[restore]=r[restore];need(np.array_equal(x,expected),'CUP_COLLECT_EXACT_RAW_RESTORE_REGION')
 need(not np.any(np.any(x!=r,axis=-1)&(~E|P)),'CUP_COLLECT_NEW_E_P_PIXEL_PROTECTION')
 for k in FIELDS:
  wanted=np.packbits((P if k=='protected_visible' else E).reshape(1,-1),axis=1,bitorder='big') if k in ['protected_visible','effective_removal'] else p[k]
  need(np.array_equal(y[k],wanted),'CUP_COLLECT_TEN_FIELD_POLICY:'+k)
 return dict(native_frame=f,restored_mask_pixels=int(np.count_nonzero(restore)),actually_restored_changed_pixels=int(np.count_nonzero(restore&np.any(r!=o,axis=-1))),new_P_pixels=int(np.count_nonzero(P)),new_E_pixels=int(np.count_nonzero(E)),protected_core_overlap_pixels=int(np.count_nonzero(P&m['core'])),outside_newE_changed_pixels=0,protected_newP_changed_pixels=0,eight_old_masks_exact=True,RGB_domain='ORIGINAL_FROZEN_GPU_CANONICAL_VST_RGB_UINT8')

def collect(packet,ctx,tick=lambda:None):
 import cv2
 cv2.setNumThreads(1)
 from .driver import stage_binding,old_effective,FILES
 validate_packet_context(packet,ctx);verify_baselines(ctx['contract'],tick=tick)
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 owner=current_producer(ROOT);need(owner['task_id']==TASK,'CUP_COLLECT_CURRENT_OWNER')
 out=ROOT/'_run/current'/TASK/'attempts/attempt_0001/candidate';ix=read(ref(out/'INDEX.json',tick=tick))
 need(ix==read(ref(out/'RESULT.json',tick=tick)) and ix['schema']=='q_frozen_cup_conflict_postprocess_candidate_index_v1' and ix['status']=='COMPLETE','CUP_COLLECT_COMPLETE_INDEX_RESULT')
 need(ix['producer']==owner and ix['task_id']==TASK and ix['operation']==OPERATION and ix['spec']==ctx['plan']['approved_spec'] and ix['contract']==ctx['spec']['contract'] and ix['plan']==ctx['plan_ref'],'CUP_COLLECT_OWNERSHIP')
 need(ix['sessions_requested']==ix['sessions_complete']==4 and ix['frames_requested']==ix['frames_complete']==760 and [r['session_id'] for r in ix['sessions']]==SCOPE,'CUP_COLLECT_SCOPE760')
 for k in ['q_changed','Contact_Render_reused_as_new_candidate','automatic_AI','automatic_successor','quality_accepted','physical_contact_ground_truth','control_ground_truth','training_eligible','Root_adoption','full_goal_achieved']:need(ix[k] is False,'CUP_COLLECT_NO_UPGRADE')
 need(type(ix['model_FK_Render_calls']) is int and ix['model_FK_Render_calls']==0 and ix['human_review_status']=='PENDING','CUP_COLLECT_DOMAIN')
 evidence=[]
 for c,row in zip(ctx['contract']['cases'],ix['sessions']):
  sid=c['session_id'];n=c['native_frames'];folder=out/sid;tick()
  need(row['status']=='COMPLETE' and row['raw_frame_count']==row['fully_audited_native_frames']==n and row['old_GPU_inputs']==c['old_GPU_inputs'] and row['frozen_identity_refs']==c['frozen_identity_refs'] and row['old_quality_and_NUM_preserved']==c['old_quality_and_NUM_preserved'],'CUP_COLLECT_CASE_INPUT_N_FLAGS')
  need(set(row['files'])==set(FILES),'CUP_COLLECT_FILES')
  for k,p in row['files'].items():need(p['path']==str(folder/k),'CUP_COLLECT_OUTPUT_PATH');pin(p,tick=tick)
  receipt=read(row['stage_receipt']);need(row['stage_receipt']['path']==str(folder/'STAGE_visual_clean_postprocess.json') and receipt['schema']=='pick_place_q_frozen_visual_clean_postprocess_sealed_stage_v1' and receipt['status']=='COMPLETE' and receipt['binding']==stage_binding(dict(ctx,packet=packet),c) and receipt['producer']==owner and receipt['files']==row['files'],'CUP_COLLECT_NEW_STAGE_NOT_OLD_GPU')
  meta=read(row['files']['candidate_masks.json']);need(meta['schema_version']==NEW_SCHEMA and meta['bitorder']=='big' and meta['native_frames']==n and meta['old_masks_metadata']==c['old_GPU_inputs']['masks.json'] and meta['Render_Contact_status']=='NOT_PRODUCED_REQUIRE_SEPARATE_REGISTERED_FRESH_TASK','CUP_COLLECT_NEW_POLICY')
  trace=read(row['files']['actual_all_native_pixel_policy_trace.json']);need(len(trace)==n,'CUP_COLLECT_ALL_NATIVE_TRACE')
  oldmeta=read(c['old_GPU_inputs']['masks.json'])
  for k in ['q_source_camera_seed_changed','physical_contact_ground_truth','control_ground_truth','training_eligible','quality_accepted']:need(meta[k] is False,'CUP_COLLECT_METADATA_GT_FLAG')
  need(meta['human_review_status']=='PENDING' and meta['fields']==list(FIELDS) and meta['shape']==[n,480,640] and meta['status']=='COMPLETE' and meta['policy']==ctx['contract']['policy'] and meta['settings']==oldmeta['settings'],'CUP_COLLECT_METADATA_CONTRACT')
  need(meta['metadata']==dict(oldmeta['metadata'],schema_version=NEW_SCHEMA,new_unknown_identity='UNCHANGED_OLD_CUP_MINUS_CORE',physical_contact_ground_truth=False),'CUP_COLLECT_NESTED_FLAGS_UNCHANGED')
  need(meta['mask_role_frames']==dict(oldmeta['mask_role_frames'],protected_visible=sum(x['new_P_pixels']>0 for x in trace),effective_removal=sum(x['new_E_pixels']>0 for x in trace)),'CUP_COLLECT_METADATA_AREA_COUNTS')
  need(meta['clean_validation']==dict(protected_pixels_per_frame=[x['new_P_pixels'] for x in trace],effective_delete_pixels_per_frame=[x['new_E_pixels'] for x in trace],outside_delete_changed_pixels_per_frame=[0]*n,protected_changed_pixels_per_frame=[0]*n,status='PASSED_DERIVED_VISUAL_POSTPROCESS'),'CUP_COLLECT_METADATA_PIXEL_VALIDATION')
  rs={'rgb':(n,480,640,3)};ms={k:(n,38400) for k in FIELDS};counts=0
  with Frames(c['old_GPU_inputs']['raw_rgb.npz'],rs,tick) as raw,Frames(c['old_GPU_inputs']['clean_rgb.npz'],rs,tick) as old,Frames(c['old_GPU_inputs']['masks.npz'],ms,tick) as masks,Frames(row['files']['candidate_clean_rgb.npz'],rs,tick) as new,Frames(row['files']['candidate_masks.npz'],ms,tick) as newm:
   for f in range(n):
    tick();r=raw.one()['rgb'];o=old.one()['rgb'];p=masks.one();x=new.one()['rgb'];y=newm.one();m=old_effective(p)
    actual=audit_frame(r,o,p,x,y,m,f)
    need(actual==trace[f],'CUP_COLLECT_ACTUAL_TRACE_COUNTS');counts+=actual['actually_restored_changed_pixels']
   for s in [raw,old,masks,new,newm]:s.finish()
  evidence.append(dict(session_id=sid,actual_native_frames=n,actual_changed_pixels_restored=counts,all_eight_old_fields_exact=True,new_P_and_outsideE_changed_pixels=0))
 return dict(schema='independent_q_frozen_clean_postprocess_collector_v1',status='PASSED',task_id=TASK,task_packet=owner['task_packet'],run_signature=packet['run_signature'],candidate_index=ref(out/'INDEX.json'),sessions=evidence,native_frames=760,real_model_FK_Render_calls=0,new_mask_policy=NEW_SCHEMA,visual_quality='PENDING_ACTUAL_LOCAL_AI',physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,Root_adoption=False)
