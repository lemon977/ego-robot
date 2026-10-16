"""Independent new Render/Contact consumer, full native/QPC/flags/actual PTS."""
from pathlib import Path
from fractions import Fraction
import json,time
import numpy as np
from .contract import *
def actual_pts_error(pts,time_base,qpc_ns,qpc0_ns):
 need(type(pts) is int and time_base is not None,'FR_ACTUAL_INTEGER_PTS')
 err=abs(Fraction(pts)*time_base-Fraction(int(qpc_ns)-int(qpc0_ns),1000000000))
 need(err<=Fraction(500,1000000000),'FR_ACTUAL_DECODE_RAW_QPC_PTS500NS')
 return err
def validate_render_arrays(validation,info,n):
 keys={'robot_pixels','object_front_pixels','feather_restore_pixels','final_protected_changed_pixels','removal_uncovered_pixels'}
 need(set(validation)==keys|{'robot_projected_fingertips_px'} and set(info['render_validation'])==keys,'FR_RENDER_VALIDATION_EXACT_KEYS')
 for key in keys:
  v=validation[key];saved=info['render_validation'][key]
  need(isinstance(v,np.ndarray) and v.shape==(n,) and v.dtype.kind in 'iu' and np.all(v>=0),'FR_RENDER_VALIDATION_FULL_N_INTEGER_NONNEGATIVE:'+key)
  need(isinstance(saved,list) and len(saved)==n and all(type(x) is int and x>=0 for x in saved) and saved==v.tolist(),'FR_SESSION_RENDER_VALIDATION_FULL_N_EXACT:'+key)
 need(validation['robot_projected_fingertips_px'].shape==(2,n,5,2) and validation['robot_projected_fingertips_px'].dtype==np.float32,'FR_FULL_N_FLOAT32_TIPS')
 need(np.all(validation['final_protected_changed_pixels']==0),'FR_PREENCODE_FINAL_NEW_P_PROTECTED')
 return True
def decode_exact(pinref,qpc,size,tick):
 import av
 pin(pinref,tick=tick);need(qpc.dtype.kind in 'iu' and qpc.ndim==1 and len(qpc)>0,'FR_RAW_QPC_INTEGER_PTS_SOURCE');errors=[];count=0
 with av.open(pinref['path']) as con:
  need(len(con.streams.video)==1,'FR_ONE_VIDEO_STREAM');con.streams.video[0].codec_context.thread_count=1
  for f in con.decode(video=0):
   tick();need(count<len(qpc) and (f.width,f.height)==size and f.pts is not None and f.time_base is not None,'FR_ACTUAL_DECODE_NATIVE_DOMAIN')
   err=actual_pts_error(int(f.pts),f.time_base,qpc[count],qpc[0]);errors.append(float(err));count+=1
 need(count==len(qpc),'FR_ACTUAL_DECODE_FULL_N_NO_HEADER');return dict(actual_decoded_native_frames=count,width=size[0],height=size[1],PTS_error_max_ns=max(errors,default=0)*1e9,header_used_as_count=False,PTS_comparison_domain='DIRECT_ORIGINAL_INTEGER_NS_QPC_NO_PRE_ROUNDING')
def collect(packet,ctx,tick=lambda:None):
 from .render_adapter import load_case
 from .driver import binding,FILES
 from . import private_render as rd
 from chaoyang.pick_place_1008 import visual
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 validate_packet_context(packet,ctx);owner=current_producer(ROOT);need(owner['task_id']==TASK,'FR_CURRENT_OWNER');verify_baselines(ctx['contract'],tick=tick);verify_clean_parent(ctx['contract'],tick=tick)
 out=ROOT/'_run/current'/TASK/'attempts/attempt_0001/candidate';ix=read(ref(out/'INDEX.json'))
 need(ix==read(ref(out/'RESULT.json')) and ix['schema']=='q_frozen_observed_cup_fresh_render_contact_candidate_index_v1' and ix['status']=='COMPLETE' and ix['producer']==owner and ix['task_id']==TASK,'FR_COMPLETE_INDEX_RESULT_OWNER')
 need(ix['spec']==ctx['plan']['approved_spec'] and ix['plan']==ctx['plan_ref'] and ix['contract']==ctx['spec']['contract'],'FR_INDEX_ACTUAL_SIGNED_INPUTS')
 need(all(type(ix[k]) is int for k in ['sessions_requested','sessions_complete','frames_requested','frames_complete','native_view_count','render_calls']),'FR_INDEX_TYPED_COUNTS')
 need(ix['sessions_requested']==ix['sessions_complete']==4 and ix['frames_requested']==ix['frames_complete']==760 and ix['native_view_count']==ix['render_calls']==1520 and [r['session_id'] for r in ix['sessions']]==SCOPE,'FR_EXACT_760_1520')
 need(ix['source_flags_preserved'] is True and ix['NUM_flags_preserved'] is True and ix['old_GPU_is_input_only'] is True and ix['human_review_status']=='PENDING','FR_FLAGS_DOMAIN_PRESERVED')
 need(ix['fresh_FK_executed'] is True and ix['new_Render_and_Contact'] is True and type(ix['model_calls']) is int and ix['model_calls']==0,'FR_FRESH_DOMAIN')
 for k in ['q_changed','quality_accepted','physical_contact_ground_truth','control_ground_truth','training_eligible','automatic_AI','Root_adoption','full_goal_achieved']:need(ix[k] is False,'FR_NO_TRUTH_OR_AI_UPGRADE:'+k)
 evidence=[];started=time.perf_counter()
 for c,row in zip(ctx['contract']['cases'],ix['sessions']):
  tick();n=c['native_frames'];sid=c['session_id'];folder=out/sid
  need(row['status']=='COMPLETE' and row['raw_frame_count']==n and row['native_views']==2*n and row['actual_visual_clean']==c['actual_clean_stage'] and row['frozen_identity_refs']==c['frozen_identity_refs'],'FR_CASE_INPUT_N')
  need(set(row['files'])==set(FILES),'FR_FILES')
  for k,v in row['files'].items():need(v['path']==str(folder/k),'FR_OUTPUT_CANONICAL');pin(v,tick=tick)
  rc=read(row['stage_receipt']);need(rc['schema']=='pick_place_q_frozen_observed_cup_fresh_render_contact_sealed_stage_v1' and rc['status']=='COMPLETE' and rc['producer']==owner and rc['binding']==binding(dict(ctx,packet=packet),c) and rc['files']==row['files'],'FR_NEW_STAGE_NOT_OLD_RENDER')
  session,motion,clean,masks=load_case(c,tick);info=read(row['files']['SESSION.json']);need(info['schema_version']=='pick-place-cup-restore-fresh-render-contact-session-candidate-v1' and info['execution_complete'] is True and info['frame_count']==n and info['human_review']=='PENDING' and info['training_eligible'] is False,'FR_SESSION_SCHEMA')
  need(info['baseline_quality_problems_history_not_recomputed_under_newP']==c['quality_problems'],'FR_OLD_RISK_HISTORY_PRESERVED')
  with np.load(row['files']['render_validation.npz']['path'],allow_pickle=False) as z:validation={k:z[k] for k in z.files}
  validate_render_arrays(validation,info,n)
  saved=validation['robot_projected_fingertips_px'];need(saved.shape==(2,n,5,2),'FR_FULL_N_TIPS');tips=np.full((2,n,5,2),np.nan);robot=rd._robot(ROOT);validate_live_robot_assets(robot,packet,ctx['contract'],tick=tick)
  K=rd.rd.scaled_intrinsics(session.K,640,480,(session.width,session.height));valid=rd._display_valid(session);hint=visual.head_hint_valid(session)
  for f in range(n):
   tick();camera=np.linalg.inv(session.c2w[f])@motion['T_world_base'];tips[:,f]=rd._project_robot_tips(robot,motion['q_arm'][f],motion['q_hand'][f],camera,K,valid[:,f]&hint[f])
  need(np.array_equal(saved,tips.astype(np.float32),equal_nan=True),'FR_ACTUAL_FRESH_FK_TIPS_SAME_SAVED_Q')
  contact=read(row['files']['contact_events.json']);expected=visual.build_contact_candidates(session.qpc_ns,1e9,masks['sam_core']|masks['equipment'],masks['protected_visible'],hand_points_px=tips);expected['fingertip_source']='ROBOT_FK_SAME_SAVED_Q_CAMERA_AND_RENDER';expected['action_modified_by_contact']=False
  legacy=dict(contact);legacy.pop('legacy_alias_semantics',None);legacy.pop('evidence_role_schema',None);legacy['frames']=[]
  for f,contact_row in enumerate(contact['frames']):
   rr=dict(contact_row);role=rr.pop('evidence_role');note=rr.pop('legacy_alias_semantics');domain=rr.pop('evidence_role_domain');cup=masks['protected_visible'][f]
   if not cup.any():wanted='UNKNOWN'
   else:
    near=visual._near(cup,12.0);human=bool((masks['sam_core'][f]&near).any());eq=bool((masks['equipment'][f]&near).any());wanted='NEAR_BOTH' if human and eq else 'NEAR_HUMAN_ROLE' if human else 'NEAR_EQUIPMENT_ROLE' if eq else 'NO_SUPPORT'
   need(role==wanted and 'wearer union' in note and domain=='2D_MODEL_MASK_TO_NEW_MODEL_CUP_PROTECTION_NOT_PHYSICAL_CONTACT','FR_ROLE_SIDECAR_NOT_LEGACY_HUMAN_TRUTH');legacy['frames'].append(rr)
  need(legacy==expected,'FR_FRESH_CONTACT_QPC_MASK_Q_RECOMPUTE_LEGACY_EXACT')
  states=[json.loads(x) for x in Path(row['files']['frame_states.jsonl']['path']).read_text().splitlines()];need(len(states)==n,'FR_ALL_NATIVE_FRAME_STATES')
  for f,state in enumerate(states):
   need(state['raw_frame_index']==f and state['raw_qpc_ns']==int(session.qpc_ns[f]) and state['physical_contact']=='UNKNOWN' and state['physical_contact_ground_truth'] is False,'FR_INTEGER_QPC_GTFALSE')
   for side,name in enumerate(['left','right']):
    h=state['hands'][name];need(h['raw_observed']==bool(session.hand_observed[side,f]) and h['display_valid']==bool(valid[side,f]) and h['diagnostic_display']==(not bool(valid[side,f])) and h['qpc_velocity_violation']==bool(motion['qpc_velocity_violation'][f,side]) and h['qpc_acceleration_violation']==bool(motion['qpc_acceleration_violation'][f,side]),'FR_SOURCE_DISPLAY_NUM_FLAGS_UNCHANGED')
   need(state['camera_raw_valid']==bool(session.motion_valid.get('head_valid',np.zeros(n,bool))[f]) and state['camera_source']==str(session.native_motion.get('head_source',np.full(n,'UNAVAILABLE'))[f]),'FR_HEAD_RAW_SOURCE_FLAGS')
   hand_sources=np.asarray(motion.get('hand_source',session.hand_source));need(hand_sources.shape==(2,n),'FR_HAND_SOURCE_FULL_N')
   for side,name in enumerate(['left','right']):
    h=state['hands'][name];position=float(motion['arm_position_mm'][side,f]);rotation=float(motion['arm_rotation_deg'][side,f]);pv=bool(np.isfinite(position));rv=bool(np.isfinite(rotation));controllers=session.motion_valid.get('controller_valid');manus=session.motion_valid.get('manus_valid')
    expected=dict(source=str(hand_sources[side,f]),wrist_position_valid=bool(controllers[f,side]) if controllers is not None else False,wrist_orientation_valid=bool(controllers[f,side]) if controllers is not None else False,handshape_valid=bool(manus[f,side]) if manus is not None else False,wrist_authority='CONTROLLER_PLUS_INSTALLATION_PRIOR',tracking_error_mm=position if pv else None,rotation_deg=rotation if rv else None,tracking_error_valid=pv and rv,tracking_gate_applicable=bool(session.hand_observed[side,f]),position_violation=bool(position>20.0) if pv else None,rotation_violation=bool(rotation>15.0) if rv else None,control_ground_truth=False)
    need(all(h[k]==v for k,v in expected.items()),'FR_EXACT_SOURCE_TRACKING_VALIDITY_NO_UPGRADE')
   need(state['model_cup_protected_pixels']==int(masks['protected_visible'][f].sum()) and state['legacy_unknown_pixels']==int(masks['unknown'][f].sum()),'FR_MODEL_NOT_TRUE_VISIBLE_CUP')
  need(np.all(validation['final_protected_changed_pixels']==0),'FR_PREENCODE_FINAL_NEW_P_PROTECTED')
  decoded={name:decode_exact(row['files'][name],session.qpc_ns,size,tick) for name,size in [('final.mp4',(640,480)),('review.mp4',(1920,960))]}
  evidence.append(dict(session_id=sid,native_frames=n,actual_FK_tips_recomputed=True,fresh_Contact_exact=True,raw_QPC_and_saved_NUM_source_display_preserved=True,actual_video_decode=decoded,physical_contact_ground_truth=False,control_ground_truth=False,visual_quality='PENDING_ACTUAL_REVIEW'))
  del session,motion,clean,masks,tips,robot;__import__('gc').collect()
 return dict(schema='independent_cup_restore_fresh_render_contact_collector_v1',status='PASSED',task_id=TASK,task_packet=owner['task_packet'],run_signature=packet['run_signature'],candidate_index=ref(out/'INDEX.json'),sessions=evidence,native_frames=760,native_views=1520,collector_actual_wall_seconds=time.perf_counter()-started,model_calls=0,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,Root_adoption=False,visual_quality='PENDING_ACTUAL_REVIEW')
