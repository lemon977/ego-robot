"""Finite same-task allN CPU q/FK/contact/actual-video-PTS evidence; no visual verdict."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.spatial.transform import Rotation
from .approval import ROOT,exact,ref,read
def audit_new_session(folder,source_folder,gpu_folder,evaluation_report):
 from chaoyang.pick_place_1008 import source,visual,render_stage as rs
 from chaoyang.pick_place_1008.pipeline import load_arrays,load_masks
 from chaoyang.exact78_baseline.robot_model import Robot
 rd=rs.rd
 folder=Path(folder);session=source.load_prepared(Path(source_folder)/'source');n=session.frame_count
 ev=read(exact(evaluation_report));cand=load_arrays(exact(ev['private_candidate_arrays']))
 actions=load_arrays(folder/'actions.npz');q=actions['q_arm'];hand=actions['q_hand'];qpc=actions['qpc_ns'];base=actions['T_world_base']
 if q.shape!=(n,2,7) or hand.shape!=(n,2,22) or not np.isfinite(q).all() or not np.isfinite(hand).all():raise ValueError('QP_QA_FINITE_ALL_N_ACTIONS')
 if not np.array_equal(q,cand['private_candidate_q_arm']) or not np.array_equal(hand,cand['unchanged_original_q_hand']) or not np.array_equal(qpc,session.qpc_ns) or not np.array_equal(qpc,cand['unchanged_original_qpc_ns']) or not np.array_equal(base,cand['unchanged_original_T_world_base']):raise ValueError('QP_QA_NEW_Q_OR_SOURCE_DRIFT')
 dt=np.diff(qpc).astype(float)/1e9
 if not np.isfinite(dt).all() or not (dt>0).all():raise ValueError('QP_QA_STRICT_QPC_DT')
 vel=np.diff(q,axis=0)/dt[:,None,None];acc=np.diff(vel,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None]
 if not np.array_equal(vel,actions['arm_velocity_rad_s']) or not np.array_equal(acc,actions['arm_acceleration_rad_s2']) or abs(vel).max()>3.6 or abs(acc).max()>54.:raise ValueError('QP_QA_ORIGINAL_RATE_GATES')
 robot=Robot(ROOT);K=rd.scaled_intrinsics(session.K,rs.WIDTH,rs.HEIGHT,(session.width,session.height))
 display=rs._display_valid(session);camera_valid=visual.head_hint_valid(session);roots=np.empty((2,n,4,4));tips=np.full((2,n,5,2),np.nan)
 for f in range(n):
  camera=np.linalg.inv(session.c2w[f])@base
  for s in range(2):roots[s,f]=base@robot.arm_frames(s,q[f,s])['root']
  tips[:,f]=rs._project_robot_tips(robot,q[f],hand[f],camera,K,display[:,f]&camera_valid[f])
 if not np.isfinite(roots).all() or not np.array_equal(roots,cand['candidate_all_native_fk_roots']):raise ValueError('QP_QA_ALL_N_NEW_Q_FK')
 target=cand['unchanged_original_raw_targets'];known=cand['source_pose_applicable']
 pos=np.linalg.norm(roots[...,:3,3]-target[...,:3,3],axis=-1)*1000
 rot=np.rad2deg(Rotation.from_matrix((np.swapaxes(target[...,:3,:3],-1,-2)@roots[...,:3,:3]).reshape(-1,3,3)).magnitude()).reshape(2,n)
 if np.any(known&(pos>20.)) or np.any(known&(rot>15.)):raise ValueError('QP_QA_KNOWN_DERIVED_TARGET_POSE_GATES')
 masks=load_masks(Path(gpu_folder));contact=visual.build_contact_candidates(session.qpc_ns,1e9,masks['sam_core']|masks['equipment'],masks['protected_visible'],hand_points_px=tips)
 contact['fingertip_source']='ROBOT_FK_SAME_SAVED_Q_CAMERA_AND_RENDER';contact['action_modified_by_contact']=False
 if contact!=read(folder/'contact_events.json'):raise ValueError('QP_QA_CONTACT_NOT_NEW_SAME_SAVED_Q_REBUILT')
 rv=load_arrays(folder/'render_validation.npz')
 if not np.array_equal(tips.astype(np.float32),rv['robot_projected_fingertips_px'],equal_nan=True):raise ValueError('QP_QA_RENDER_TIPS_NOT_SAME_Q')
 states=[json.loads(x) for x in (folder/'frame_states.jsonl').read_text().splitlines()]
 if len(states)!=n or any(x['raw_qpc_ns']!=int(qpc[i]) for i,x in enumerate(states)):raise ValueError('QP_QA_FRAME_STATES_RAW_QPC_N')
 for i,x in enumerate(states):
  if x['physical_contact']!='UNKNOWN' or any(x[k] is not False for k in ['physical_contact_ground_truth','control_ground_truth','training_eligible']):raise ValueError('QP_QA_TRUTH_FLAGS')
  for s,name in enumerate(['left','right']):
   h=x['hands'][name]
   if h['raw_observed']!=bool(session.hand_observed[s,i]) or h['display_valid']!=bool(display[s,i]) or h['source']!=str(session.hand_source[s,i]):raise ValueError('QP_QA_SOURCE_DISPLAY_FLAGS_DRIFT')
 if session.session_id=='192':
  right=states[145]['hands']['right']
  if known[1,145] or right['raw_observed'] is not False or right['display_valid'] is not True or right['handshape_valid'] is not False or right['source']!='EDGE_HOLD_SHORT' or right['tracking_gate_applicable'] is not False:raise ValueError('QP_QA_192_RAW_INVALID_UNKNOWN_NOT_PRESERVED')
 import av
 video={}
 for name,size in [('final.mp4',(rs.WIDTH,rs.HEIGHT)),('review.mp4',(rs.WIDTH*3,rs.HEIGHT*2))]:
  count=0;maxerr=0
  with av.open(str(folder/name)) as container:
   streams=list(container.streams.video)
   if len(streams)!=1:raise ValueError('QP_QA_EXACT_SINGLE_VIDEO_STREAM_REQUIRED')
   stream=streams[0];stream.codec_context.thread_count=1;stream.thread_count=1
   for frame in container.decode(stream):
    if count>=n or frame.pts is None or frame.time_base is None or (frame.width,frame.height)!=size:raise ValueError('QP_QA_ACTUAL_VIDEO_FRAME_DOMAIN')
    rgb=frame.to_ndarray(format='rgb24')
    if rgb.shape!=(size[1],size[0],3):raise ValueError('QP_QA_ACTUAL_VIDEO_RGB_DECODE')
    actual_ns=int(frame.pts*frame.time_base*1000000000);expected_ns=int(qpc[count]-qpc[0]);maxerr=max(maxerr,abs(actual_ns-expected_ns));count+=1
  if count!=n or maxerr>500:raise ValueError('QP_QA_NATIVE_VIDEO_PTS_500NS_OR_COUNT')
  video[name]={'file':ref(folder/name),'actual_all_native_RGB_frames_decoded':count,'actual_pts_qpc_max_error_ns':maxerr,'fixed_pts_gate_max_ns':500,'decoder':'PyAV container.decode/video0 + frame.to_ndarray(rgb24)'}
 ap=folder/'QPC_NEW_MOTION_RENDER_ALL_NATIVE_QA_ARRAYS_V1.npz'
 np.savez_compressed(ap,actual_qpc_velocity_rad_s=vel,actual_qpc_acceleration_rad_s2=acc,actual_all_native_fk_roots=roots,actual_new_same_q_projected_tips_px=tips,qualified_source_pose_applicable=known,qualified_position_mm=np.where(known,pos,np.nan),qualified_rotation_deg=np.where(known,rot,np.nan),raw_qpc_ns=qpc)
 report={'schema':'QP_new_Motion_Render_same_task_all_native_CPU_QA_v1','status':'PASS_STRUCTURAL_AND_RATE_AND_KNOWN_DERIVED_TARGET_INVALID_UNKNOWN','session_id':session.session_id,'native_frames':n,'new_actions':ref(folder/'actions.npz'),'old_evaluation_report':evaluation_report,'original_source_input':ref(Path(source_folder)/'source/SESSION_INPUT.json'),'arrays':ref(ap),'source_invalid_pose_count':int((~known).sum()),'source_invalid_pose_GATE':'UNKNOWN_NO_MEASUREMENT_PASS','actual_allN_FK_recomputed':True,'actual_Contact_rebuilt_from_saved_q_FK_QPC_original_union_and_P':True,'Clean_is_not_Contact_or_depth_or_metric_input':True,'actual_video_decode':video,'errors':[],'visual_quality':'PENDING_NEW_ACTUAL_SERVER_LOCAL_AI','human_review':'PENDING','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False}
 rp=folder/'QPC_NEW_MOTION_RENDER_ALL_NATIVE_CPU_QA_V1.json'
 with rp.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return ref(rp)
