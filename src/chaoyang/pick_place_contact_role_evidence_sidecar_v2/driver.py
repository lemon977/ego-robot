from __future__ import annotations
import copy,datetime,json,os,signal,time
from collections import Counter
from pathlib import Path
import cv2
from .contract import ROOT,TASK,SCOPE,NATIVE,BUDGETS,need,path,pin,read,ref,baseline_guard,validate_packet_context
from .packed_reader import PackedFrames
from .roles import classify,windows,validate_qpc,preserve_frame
def write(root,p,value):
 p=path(p,root);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
 return ref(p,root)
def load_small_case(c,root=ROOT):
 n=c['native_frames']
 q=[json.loads(x) for x in pin(c['original_QPC'],root,True).read_text().splitlines()]
 validate_qpc(q,n)
 mapping=[json.loads(x) for x in pin(c['frame_mapping'],root).read_text().splitlines()]
 frame=[json.loads(x) for x in pin(c['frame_states'],root).read_text().splitlines()]
 contact=read(c['old_contact'],root);rows=contact['frames']
 need(len(mapping)==len(frame)==len(rows)==n,'ROLE_NATIVESOURCE_CONTACT_COUNTS')
 need(contact['qpc_frequency']==1e9 and contact['near_px']==12 and contact['window_pad_s']==.5 and contact['window_merge_gap_s']==.1 and contact['physical_contact_ground_truth'] is False and contact['action_modified_by_contact'] is False,'ROLE_FROZEN_CONTACT_RULES_NO_ACTION_MODIFICATION')
 for i,t in enumerate(q):
  need(type(mapping[i]['raw_frame_index']) is int and mapping[i]['raw_frame_index']==i and type(mapping[i]['video_qpc_ns']) is int and mapping[i]['video_qpc_ns']==t,'ROLE_MAPPING_ORIGINAL_INTEGER_QPC')
  preserve_frame(frame[i],t,i)
  need(type(rows[i]['frame_index']) is int and rows[i]['frame_index']==i and type(rows[i]['qpc']) is int and rows[i]['qpc']==t and rows[i]['physical_contact']=='UNKNOWN','ROLE_OLD_CONTACT_NATIVE_QPC_TRUTH')
 return q,mapping,frame,contact
def compute_case(c,root=ROOT,budget=lambda:None):
 q,mapping,frames,contact=load_small_case(c,root)
 rows=[];triggers=[]
 path(c['masks_NPZ']['path'],root)
 with PackedFrames(c['masks_NPZ'],c['native_frames'],budget) as packed:
  for i,m in packed:
   value=classify(m['sam_core'],m['equipment'],m['protected_visible'],m['cup_observed'],True)
   original=preserve_frame(frames[i],q[i],i)
   row=dict(native_frame=i,raw_QPC_ns=q[i],**value,
            image_evidence_source='COMPLETE_OFFICIAL_CANONICAL_NATIVE_MASK_DOMAIN_NO_NEW_RGB_DECODE',
            original_frame_state=original,original_contact_frame=copy.deepcopy(contact['frames'][i]),
            original_mapping_row=copy.deepcopy(mapping[i]),source_metric_validity_inferred_from_mask=False,
            original_source_NUM_RAW_DISPLAY_flags_unchanged=True)
   rows.append(row)
   if value['near_human_role'] or value['near_equipment_role']:triggers.append(i)
 pin(c['old_contact'],root);pin(c['frame_states'],root);pin(c['original_QPC'],root,True)
 return dict(schema='pick_place_mask_role_2D_sidecar_v1',session_id=c['session_id'],
             native_frames=c['native_frames'],raw_QPC_ns=q,near_px=12.,
             frames=rows,role_only_review_windows=windows(q,triggers),
             tag_counts=dict(Counter(x['role_2D_mask_evidence'] for x in rows)),
             original_contact_events=c['old_contact'],original_contact_frames_reasons_windows_preserved=True,
             original_q_source_motion_masks_or_Render_modified=False,projected_fingertips_used=False,
             occlusion_direction='UNKNOWN',physical_contact='UNKNOWN',
             equipment_only_false_contact_claim=False,human_review_status='PENDING',
             physical_ground_truth=False,physical_contact_ground_truth=False,control_ground_truth=False,
             training_eligible=False,visual_quality_qualification=False)
def run(ctx,root=ROOT):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','ROLE_GPU_ENV_EMPTY')
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:need(os.environ.get(k)=='1','ROLE_CPU1_ENV')
 cv2.setNumThreads(1);start=time.perf_counter();deadline=start+BUDGETS['wall_seconds_max']
 def budget():need(time.perf_counter()<deadline,'ROLE_FINITE_WALL_1800')
 def expired(*_):raise TimeoutError('ROLE_FINITE_WALL_1800')
 signal.signal(signal.SIGALRM,expired);signal.alarm(BUDGETS['wall_seconds_max'])
 validate_packet_context(ctx['packet'],ctx,root)
 baseline_guard(ctx['contract'],root)
 out=path(root/'_run/current'/TASK/'attempts/attempt_0001/candidate',root)
 need(not out.exists(),'ROLE_FRESH_CANDIDATE_ONLY');out.mkdir()
 manifest=[];results=[]
 for c in ctx['contract']['cases']:
  budget();value=compute_case(c,root,budget)
  v=write(root,out/'sessions'/c['session_id']/'ROLE_EVIDENCE_SIDECAR.json',value)
  manifest.append(dict(session_id=c['session_id'],official_GPU_receipt=c['stages']['gpu']['receipt'],
                       official_masks_NPZ=c['masks_NPZ'],original_QPC=c['original_QPC'],
                       original_frame_mapping=c['frame_mapping'],original_frame_states=c['frame_states'],
                       original_contact_events=c['old_contact'],streamed_native_count=c['native_frames'],
                       packed_fields=['sam_core','equipment','protected_visible','cup_observed'],
                       packed_bitorder='big',one_native_at_a_time=True,
                       full_SHA_before_sameFD_ZIP_CRC_EOF_after=True,RGB_FK_models_Render_read_or_called=False))
  results.append(dict(session_id=c['session_id'],status='COMPLETE',raw_frame_count=c['native_frames'],
                      role_evidence=v,tag_counts=value['tag_counts']))
 baseline_guard(ctx['contract'],root)
 rm=write(root,out/'ACTUAL_READ_MANIFEST.json',dict(schema='contact_role_sidecar_actual_input_read_manifest_v1',
          producer=ctx['producer'],task_packet=ctx['producer']['task_packet'],contract=ctx['spec']['contract'],
          rows=manifest,actual_native_masks_streamed=NATIVE,original_scope=SCOPE,
          raw_RGB_arrays_models_FK_or_Render_used=False,old_inputs_unchanged=True,
          independent_source_pose_or_physical_GT_recomputed=False))
 index=dict(schema='pick_place_contact_role_sidecar_execution_index_v1',status='COMPLETE',
            stage='contact_role_evidence_only',task_id=TASK,run_signature=ctx['packet']['run_signature'],
            task_packet=ctx['producer']['task_packet'],spec=ctx['plan']['approved_spec'],plan=ctx['plan_ref'],
            contract=ctx['spec']['contract'],sessions_requested=len(SCOPE),sessions_complete=len(SCOPE),
            frames_complete=NATIVE,sessions=results,read_manifest=rm,
            model_calls=0,FK_calls=0,render_calls=0,GPU_calls=0,CPU_workers=1,
            original_contact_source_q_or_mask_outputs_modified=False,
            algorithm_recipe='MASK_ROLE_2D_SIDECAR_FIXED12PX_QPC_HALFSECOND_MERGE_POINT1',
            numeric_gates_or_source_RAW_DISPLAY_flags_changed=False,
            actual_wall_seconds=time.perf_counter()-start,
            visual_quality_qualified=False,human_review_status='PENDING',
            physical_ground_truth=False,physical_contact_ground_truth=False,control_ground_truth=False,
            training_eligible=False,adoption='NOT_ADOPTED',full_goal_achieved=False)
 write(root,out/'INDEX.json',index);write(root,out/'RESULT.json',index);signal.alarm(0)
 return 0
