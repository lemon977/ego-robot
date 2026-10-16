"""Current finite evaluation/Motion+Render closure; never adopts visual quality."""
from pathlib import Path
import numpy as np
from .approval import ROOT,read,ref,exact,FIXED
from chaoyang.pick_place_1008.stage_reuse import _validate
def sealed_task(tid,expected_operation):
 from chaoyang.governance.exact78_operation_gate import _command_tail
 state=read(ROOT/'docs/governance/LONG_HORIZON_TASK_STATE.json');registry=read(ROOT/'tasks/current/INDEX.json')
 rows=[x for x in state['tasks'] if x['task_id']==tid];routes=[x for x in registry['task_packets'] if x['task_id']==tid]
 if len(rows)!=1 or len(routes)!=1:raise ValueError('QP_UNIQUE_SEALED_TASK_REQUIRED')
 row,route=rows[0],routes[0]
 if row['status']!='PASSED' or row.get('pid') is not None or row.get('proc_start_ticks') is not None or route.get('execution_allowed') is not False:raise ValueError('QP_EVALUATION_GLOBAL_PASSED_REQUIRED')
 attempt=ROOT/'_run/current'/tid/'attempts/attempt_0001'
 if exact(row['result'])!=attempt/'RESULT.json' or exact(row['terminal_receipt'])!=ROOT/'tasks/receipts'/(tid.upper()+'.json'):raise ValueError('QP_TERMINAL_CANONICAL_ENVELOPE_PATH')
 outer=read(exact(row['result']));receipt=read(exact(row['terminal_receipt']))
 if exact(outer['task_packet'])!=ROOT/'tasks/current'/tid/'TASK_PACKET.json':raise ValueError('QP_CANONICAL_PACKET_PATH')
 packet=read(exact(outer['task_packet']))
 if outer.get('schema_version')!='pick-place-1008-execution-result-v1' or receipt.get('schema_version')!='pick-place-1008-receipt-v1' or receipt.get('task_id')!=tid or outer['task_id']!=tid or outer['status']!='PASSED' or type(outer['worker_returncode']) is not int or outer['worker_returncode']!=0 or outer['validation_errors']!=[] or outer.get('failure_code') is not None or outer['resource_wait'] is not False or type(outer['algorithm_attempts_consumed']) is not int or outer['algorithm_attempts_consumed']!=1:raise ValueError('QP_OUTER_NOT_PASSED_ONE_ATTEMPT')
 if receipt['status']!='PASSED' or receipt['result']!=row['result'] or route['result']!=row['result'] or route['terminal_receipt']!=row['terminal_receipt']:raise ValueError('QP_REGISTRY_OUTER_RECEIPT_DRIFT')
 if packet['task_id']!=tid or packet['run_signature']!=outer['run_signature'] or row['run_signature']!=outer['run_signature'] or route['packet_sha256']!=outer['task_packet']['sha256'] or _command_tail(packet['commands'][0])[0]!=expected_operation:raise ValueError('QP_PACKET_COMMAND_SIGNATURE_DRIFT')
 from chaoyang.governance.common import process_identity
 def require_dead(pid,ticks,label):
  if type(pid) is not int or type(ticks) is not int or ticks<=0:raise ValueError('QP_SIGNED_ACTOR_IDENTITY_REQUIRED:'+label)
  try:actual=process_identity(pid)
  except (FileNotFoundError,ProcessLookupError):return
  if actual['start_ticks']==ticks:raise ValueError('QP_SEALED_ACTOR_STILL_LIVE:'+label)
 writer=packet['writer'];require_dead(writer['pid'],writer['proc_start_ticks'],'writer')
 worker=read(ROOT/'_run/current'/tid/'attempts/attempt_0001/WORKER.json')
 if worker['task_id']!=tid or worker['writer']!=writer or worker['run_signature']!=packet['run_signature'] or worker['packet_command']!=packet['commands'][0]:raise ValueError('QP_SIGNED_WORKER_CLOSURE')
 require_dead(worker['pid'],worker['proc_start_ticks'],'worker')
 for x in packet['runtime_code']:exact(x)
 for x in packet['inputs']['files']:exact(x)
 exact(packet['inputs']['spec'])
 ixref=outer['required_outputs']['candidate/INDEX.json']
 if exact(ixref)!=attempt/'candidate/INDEX.json' or exact(outer['required_outputs']['candidate/RESULT.json'])!=attempt/'candidate/RESULT.json' or outer['stage_result']!=outer['required_outputs']['candidate/RESULT.json']:raise ValueError('QP_CANDIDATE_ENVELOPE_SEPARATION')
 ix=read(exact(ixref))
 if ix['status']!='COMPLETE':raise ValueError('QP_INDEX_COMPLETE_REQUIRED')
 return dict(task_id=tid,index=ixref,outer_result=row['result'],terminal_receipt=row['terminal_receipt'],task_packet=outer['task_packet'],run_signature=outer['run_signature']),ix,packet
def evaluation_selection(approval):
 from .approval import EVALUATION
 a=approval['evaluation_selection'];p,ix,packet=sealed_task(a['evaluation_task_id'],EVALUATION)
 if any(a[k]!=p[k] for k in ['index','outer_result','terminal_receipt']):raise ValueError('QP_EVALUATION_SELECTION_DRIFT')
 rows=ix['fixed_cases']
 if [r['session_id'] for r in rows]!=FIXED or len(rows)!=16 or sum(r['native_frames'] for r in rows)!=2578 or a['case_reports']!=[r['report'] for r in rows]:raise ValueError('QP_FIXED16_SELECTION')
 for row in rows:
  result=read(exact(row['report']))
  if result.get('candidate_emitted') is not True or result.get('all_native_rate_gate_pass') is not True or result.get('known_source_points_derived_pose_gate_pass') is not True or result.get('actual_candidate_allN_FK_evaluated') is not True:raise ValueError('QP_FAILED_CANDIDATE_CANNOT_BE_RENDER_PROMOTED:'+row['session_id'])
  exact(result['private_candidate_arrays'])
  if result.get('human_review_status')!='PENDING' or any(result.get(k) is not False for k in ['physical_contact_ground_truth','control_ground_truth','training_eligible']):raise ValueError('QP_NO_TRUTH_UPGRADE')
 return p,rows
def motion_render_session(sid,row,baseline,expected,camera,source_row):
 from .bindings import family_binding
 from chaoyang.pick_place_1008.pipeline import STAGE_FILES
 root=Path(row['stage_receipts']['motion']['path']).parents[2]
 closures={}
 for stage in ['source','gpu']:
  if row['stage_receipts'][stage]!=baseline['stage_receipts'][stage]:raise ValueError('QP_S_G_REUSE_NOT_ORIGINAL_EXACT')
 for stage in ['motion','render']:
  folder,pin=_validate(ROOT,root,sid,stage,expected[stage],tuple(STAGE_FILES[stage]))
  if folder is None or pin!=row['stage_receipts'][stage]:raise ValueError('QP_FRESH_TYPED_EXPECTED_BINDING_MISS')
  closures[stage]=pin
 mr=read(exact(row['stage_receipts']['motion']));rr=read(exact(row['stage_receipts']['render']))
 session=read(exact(rr['files']['SESSION.json']))
 if session['outputs']['actions']!=mr['files']['actions.npz'] or session['outputs']['frame_mapping']!=mr['files']['frame_mapping.jsonl']:raise ValueError('QP_RENDER_NOT_CONSUMING_NEW_MOTION')
 n=source_row['raw_frame_count']
 qa=read(exact(session['outputs']['new_qpc_motion_render_CPU_QA']))
 if qa['session_id']!=sid or qa['native_frames']!=n or qa['errors']!=[] or qa['actual_allN_FK_recomputed'] is not True or qa['actual_Contact_rebuilt_from_saved_q_FK_QPC_original_union_and_P'] is not True or qa['new_actions']!=mr['files']['actions.npz']:raise ValueError('QP_SEALED_ALL_NATIVE_QA_BINDING')
 exact(qa['arrays'])
 for name in ['final.mp4','review.mp4']:
  v=qa['actual_video_decode'][name]
  if v['file']!=rr['files'][name] or v['actual_all_native_RGB_frames_decoded']!=n or v['actual_pts_qpc_max_error_ns']>500 or v['fixed_pts_gate_max_ns']!=500:raise ValueError('QP_ACTUAL_VIDEO_ALL_N_PTS_500NS')
 with np.load(exact(mr['files']['actions.npz']),allow_pickle=False) as z:
  if z['q_arm'].shape!=(n,2,7) or z['q_hand'].shape!=(n,2,22) or not np.isfinite(z['q_arm']).all():raise ValueError('QP_NEW_ACTION_ALL_N_CONTRACT')
 contact=read(exact(rr['files']['contact_events.json']))
 if len(contact['frames'])!=n or contact.get('action_modified_by_contact') is not False or contact.get('fingertip_source')!='ROBOT_FK_SAME_SAVED_Q_CAMERA_AND_RENDER':raise ValueError('QP_CONTACT_SAME_NEW_SAVED_Q_REQUIRED')
 if session['frame_count']!=n or session['human_review']!='PENDING' or any(session.get(k) is not False for k in ['physical_contact_ground_truth','control_ground_truth','training_eligible']):raise ValueError('QP_RENDER_NO_QUALIFICATION_UPGRADE')
 return dict(session_id=sid,native_frames=n,stage_receipts=row['stage_receipts'],fresh_binding_validation=closures,actual_visual_inspection=False,human_review='PENDING',physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False)

def current_motion_render_task(tid,approval_path):
 from .approval import MOTION_RENDER,verify_runtime_approval
 from .runtime import metadata_context
 from .bindings import family_binding
 from chaoyang.governance.exact78_operation_gate import _command_tail
 producer,ix,packet=sealed_task(tid,MOTION_RENDER)
 approval,contract=verify_runtime_approval(approval_path,argv=_command_tail(packet['commands'][0])[1:],packet=packet)
 evaluation,evalrows=evaluation_selection(approval)
 h,q,c,cohort,infos=metadata_context()
 if [x['session_id'] for x in ix['sessions']]!=FIXED or not (ix['sessions_requested']==ix['sessions_complete']==16) or ix['frames_complete']!=2578 or ix['manifest']!=c.mref or ix['camera_config']!=c.camera:raise ValueError('QP_NEW_MR_FIXED_SCOPE_OR_INPUT_BINDING')
 checks=[]
 for info,row in zip(infos,ix['sessions']):
  sid=info['session_id'];source_row=c.source[sid]
  expected_motion=family_binding(ROOT,'motion',source_row,c.camera,{'source':row['stage_receipts']['source']},approval,approval['exact_extra_runtime_code'])
  up={k:row['stage_receipts'][k] for k in ['source','motion','gpu']}
  expected_render=family_binding(ROOT,'render',source_row,c.camera,up,approval,approval['exact_extra_runtime_code'])
  checks.append(motion_render_session(sid,row,info['current_row'],{'motion':expected_motion,'render':expected_render},c.camera,source_row))
 return dict(status='CURRENT_ALL16_NEW_MOTION_RENDER_STAGE_CLOSURE_PASS_VISUAL_UNREVIEWED',producer=producer,checks=checks,native_frames=2578,actual_new_AI_review=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review='PENDING')

def main(argv=None):
 import argparse,json,os
 from .approval import EVALUATION,MOTION_RENDER,verify_runtime_approval
 from chaoyang.governance.exact78_operation_gate import _command_tail
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true',required=True);p.add_argument('--task',required=True);p.add_argument('--mode',choices=['evaluation','motion-render'],required=True);p.add_argument('--root-runtime-approval',required=True);a=p.parse_args(argv)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('QP_COLLECTOR_GPU0_REQUIRED')
 if a.mode=='motion-render':result=current_motion_render_task(a.task,a.root_runtime_approval)
 else:
  producer,ix,packet=sealed_task(a.task,EVALUATION)
  approval,contract=verify_runtime_approval(a.root_runtime_approval,argv=_command_tail(packet['commands'][0])[1:],packet=packet)
  if len(ix['fixed_cases'])!=16 or [r['session_id'] for r in ix['fixed_cases']]!=FIXED or ix['native_frames_requested']!=2578:raise ValueError('QP_EVALUATION_FIXED_16_SCOPE')
  records=[]
  for row in ix['fixed_cases']:
   v=read(exact(row['report']))
   if v['session_id']!=row['session_id'] or v['native_frames']!=row['native_frames']:raise ValueError('QP_EVALUATION_REPORT_BINDING')
   if v.get('candidate_emitted'):
    exact(v['private_candidate_arrays'])
   records.append({'session_id':row['session_id'],'native_frames':row['native_frames'],'candidate_status':v['status'],'source_invalid_pose_points':v.get('source_invalid_pose_points'),'rate_gate':v.get('all_native_rate_gate_pass'),'known_pose_gate':v.get('known_source_points_derived_pose_gate_pass')})
  result={'status':'EVALUATION_TERMINAL_CURRENT_SCOPE_BOUND_CANDIDATE_FAIL_UNKNOWN_RETAINED','producer':producer,'records':records,'actual_projection_or_FK_reexecuted':False,'visual_or_physical_generalization_qualified':False}
 print(json.dumps(result,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
