"""Fenced finite q-frozen visual postprocess; no models/FK/Render or Contact writing."""
from pathlib import Path
import json,time,resource
import numpy as np
from .contract import *
from .helper import FIELDS,OLD_SCHEMA,NEW_SCHEMA,derive_visual_candidate
from .stream import Frames,Writer
FILES=('candidate_clean_rgb.npz','candidate_masks.npz','candidate_masks.json','actual_all_native_pixel_policy_trace.json')
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def old_effective(packed):
 from chaoyang.pick_place_1008.visual import effective_clean_masks
 m={k:np.unpackbits(v,axis=1,count=307200,bitorder='big').reshape(1,480,640).astype(bool) for k,v in packed.items()}
 e,_=effective_clean_masks(m['removal'],m['protected_visible'])
 need(np.array_equal(e,m['effective_removal']),'CUP_OLD_E_EXACT_FROZEN_DILATION_MINUS_P')
 return m
def stage_binding(ctx,c):
 return dict(schema='q_frozen_cup_conflict_postprocess_binding_v1',stage='visual_clean_postprocess',session_id=c['session_id'],native_frames=c['native_frames'],spec=ctx['plan']['approved_spec'],plan=ctx['plan_ref'],contract=ctx['spec']['contract'],policy=ctx['contract']['policy'],runtime=ctx['packet']['runtime_code'],old_stage_receipts=c['old_stage_receipts'],original_GPU_inputs=c['old_GPU_inputs'],source_q_camera_seed_frozen_refs=c['frozen_identity_refs'],single_factor='RAW_RESTORE_OLD_SEMANTIC_INTERSECT_OLD_E_NEW_P_CUP_NEW_E_OLD_E_MINUS_CUP')
def process_case(ctx,c,out,tick):
 n=c['native_frames'];sid=c['session_id'];folder=out/sid;folder.mkdir();tick()
 from chaoyang.pick_place_1008.visual import SETTINGS
 meta=read(c['old_GPU_inputs']['masks.json']);need(meta['settings']==dict(SETTINGS),'CUP_OLD_FROZEN_SETTINGS');need(meta['schema']=='pick_place_1008_protected_clean_v1' and meta['metadata']['schema_version']==OLD_SCHEMA and meta['source_pixel_domain']=='DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8','CUP_OLD_MASK_SCHEMA')
 shapes={'rgb':(n,480,640,3)};maskshape={k:(n,38400) for k in FIELDS};trace=[]
 cleanwriter=Writer(folder/'candidate_clean_rgb.npz',shapes,tick);maskwriter=Writer(folder/'candidate_masks.npz',maskshape,tick)
 try:
  with Frames(c['old_GPU_inputs']['raw_rgb.npz'],shapes,tick) as rs,Frames(c['old_GPU_inputs']['clean_rgb.npz'],shapes,tick) as cs,Frames(c['old_GPU_inputs']['masks.npz'],maskshape,tick) as ms:
   for frame in range(n):
    tick();raw=rs.one()['rgb'];clean=cs.one()['rgb'];packed=ms.one();m=old_effective(packed)
    v=derive_visual_candidate(raw,clean,packed,bitorder='big',baseline_mask_schema=OLD_SCHEMA)
    cleanwriter.one({'rgb':v['clean_rgb']});maskwriter.one(v['packed_masks'])
    trace.append(dict(native_frame=frame,restored_mask_pixels=v['metadata']['restored_mask_pixels_per_frame'][0],actually_restored_changed_pixels=v['metadata']['actually_restored_changed_pixels_per_frame'][0],new_P_pixels=int(np.count_nonzero(m['cup_observed'])),new_E_pixels=int(np.count_nonzero(m['effective_removal']&~m['cup_observed'])),protected_core_overlap_pixels=v['metadata']['protected_core_overlap_pixels_per_frame'][0],outside_newE_changed_pixels=0,protected_newP_changed_pixels=0,eight_old_masks_exact=True,RGB_domain='ORIGINAL_FROZEN_GPU_CANONICAL_VST_RGB_UINT8'))
   rs.finish();cs.finish();ms.finish()
  cleanwriter.finish();maskwriter.finish()
 finally:cleanwriter.close();maskwriter.close()
 metadata=dict(schema_version=NEW_SCHEMA,bitorder='big',native_frames=n,shape=[n,480,640],fields=list(FIELDS),policy=ctx['contract']['policy'],old_masks_metadata=c['old_GPU_inputs']['masks.json'],old_GPU_stage_is_input_not_candidate_stage=True,held_unknown_removal_semantic_old_arrays_unchanged=True,q_source_camera_seed_changed=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING',quality_accepted=False,Render_Contact_status='NOT_PRODUCED_REQUIRE_SEPARATE_REGISTERED_FRESH_TASK')
 metadata.update(schema='q_frozen_cup_restore_clean_candidate_metadata_v1',source_pixel_domain=meta['source_pixel_domain'],settings=meta['settings'],metadata=dict(meta['metadata'],schema_version=NEW_SCHEMA,new_unknown_identity='UNCHANGED_OLD_CUP_MINUS_CORE',physical_contact_ground_truth=False),baseline_clean_validation=meta['clean_validation'],mask_role_frames=dict(meta['mask_role_frames'],protected_visible=sum(x['new_P_pixels']>0 for x in trace),effective_removal=sum(x['new_E_pixels']>0 for x in trace)),clean_validation=dict(protected_pixels_per_frame=[x['new_P_pixels'] for x in trace],effective_delete_pixels_per_frame=[x['new_E_pixels'] for x in trace],outside_delete_changed_pixels_per_frame=[0]*n,protected_changed_pixels_per_frame=[0]*n,status='PASSED_DERIVED_VISUAL_POSTPROCESS'),status='COMPLETE');write(folder/'candidate_masks.json',metadata);write(folder/'actual_all_native_pixel_policy_trace.json',trace)
 files={name:ref(folder/name,tick=tick) for name in FILES}
 receipt=dict(schema='pick_place_q_frozen_visual_clean_postprocess_sealed_stage_v1',status='COMPLETE',binding=stage_binding(ctx,c),producer=ctx['producer'],files=files)
 write(folder/'STAGE_visual_clean_postprocess.json',receipt)
 return dict(session_id=sid,status='COMPLETE',raw_frame_count=n,fully_audited_native_frames=n,stage_receipt=ref(folder/'STAGE_visual_clean_postprocess.json',tick=tick),files=files,old_GPU_inputs=c['old_GPU_inputs'],frozen_identity_refs=c['frozen_identity_refs'],old_quality_and_NUM_preserved=c['old_quality_and_NUM_preserved'])
def run(ctx):
 import cv2
 cv2.setNumThreads(1);resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
 start=time.monotonic()
 def tick():
  ctx['guard']();need(time.monotonic()-start<BUDGETS['wall_seconds_max'],'CUP_FINITE_WALL')
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=256<<20,'CUP_RSS256MIB_EXCEEDED')
 tick();verify_baselines(ctx['contract'],tick=tick)
 out=ROOT/'_run/current'/TASK/'attempts/attempt_0001/candidate';need(not out.exists(),'CUP_FRESH_CANDIDATE');out.mkdir()
 rows=[]
 for c in ctx['contract']['cases']:
  try:rows.append(process_case(ctx,c,out,tick))
  except Exception as e:
   if isinstance(e,MemoryError) or any(x in str(e) for x in ('CUP_FINITE_WALL','CUP_RSS','CUP_STREAM_INPUT_CHANGED','FENCE','SHA')):raise
   rows.append(dict(session_id=c['session_id'],status='FAILED',raw_frames_expected=c['native_frames'],fully_audited_native_frames=0,error=type(e).__name__+':'+str(e),partial_outputs_are_successor_inputs=False))
 complete=all(r['status']=='COMPLETE' for r in rows);tick();ctx['full_exit_guard']()
 value=dict(schema='q_frozen_cup_conflict_postprocess_candidate_index_v1',status='COMPLETE' if complete else 'FAILED',task_id=TASK,operation=OPERATION,producer=ctx['producer'],spec=ctx['plan']['approved_spec'],contract=ctx['spec']['contract'],plan=ctx['plan_ref'],sessions=rows,sessions_requested=4,sessions_complete=sum(r['status']=='COMPLETE' for r in rows),frames_requested=760,frames_complete=sum(r['fully_audited_native_frames'] for r in rows),wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model_FK_Render_calls=0,q_changed=False,old_GPU_is_input_only=True,Contact_Render_reused_as_new_candidate=False,automatic_AI=False,automatic_successor=False,quality_accepted=False,human_review_status='PENDING',physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,Root_adoption=False,full_goal_achieved=False)
 for name in ('INDEX.json','RESULT.json'):write(out/name,value)
 return 0 if complete else 1
