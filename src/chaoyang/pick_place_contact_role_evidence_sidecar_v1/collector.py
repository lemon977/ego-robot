"""Independent post-worker recipe replay from packed N1 masks, no models/FK/Render."""
from __future__ import annotations
import json,time
from pathlib import Path
from .contract import ROOT,TASK,SCOPE,NATIVE,need,path,ref,read,pin,validate_packet_context,baseline_guard
from .driver import compute_case
def collect(packet,ctx,root=ROOT):
 start=time.perf_counter();budget_seconds=ctx.get('collector_wall_budget_seconds',900)
 need(type(budget_seconds) in [int,float] and 0<budget_seconds<=900,'ROLE_COLLECTOR_REMAINING_BUDGET')
 validate_packet_context(packet,ctx,root);baseline_guard(ctx['contract'],root)
 out=path(root/'_run/current'/TASK/'attempts/attempt_0001/candidate',root)
 index=read(ref(out/'INDEX.json',root),root);result=read(ref(out/'RESULT.json',root),root)
 need(index==result and index['schema']=='pick_place_contact_role_sidecar_execution_index_v1' and index['status']=='COMPLETE','ROLE_CANDIDATE_INDEX_RESULT')
 need(index['task_id']==TASK and index['run_signature']==packet['run_signature'] and index['task_packet']==ref(root/'tasks/current'/TASK/'TASK_PACKET.json',root),'ROLE_CANDIDATE_PACKET_RUNSIG')
 need(type(index['sessions_requested']) is int and type(index['sessions_complete']) is int and type(index['frames_complete']) is int and index['sessions_requested']==index['sessions_complete']==4 and index['frames_complete']==760,'ROLE_EXACT_COMPLETE4_760')
 need(index['spec']==ctx['plan']['approved_spec'] and index['plan']==ctx['plan_ref'] and index['contract']==ctx['spec']['contract'],'ROLE_CANDIDATE_INPUT_PIN')
 for key in ['visual_quality_qualified','physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible','full_goal_achieved','original_contact_source_q_or_mask_outputs_modified','numeric_gates_or_source_RAW_DISPLAY_flags_changed']:
  need(index[key] is False,'ROLE_TRUTH_OR_OLD_INPUT_MODIFICATION:'+key)
 for key in ['model_calls','FK_calls','render_calls','GPU_calls']:need(type(index[key]) is int and index[key]==0,'ROLE_FORBIDDEN_CALL_COUNTER')
 need(index['CPU_workers']==1 and type(index['CPU_workers']) is int,'ROLE_SINGLE_WORKER')
 manifest=read(index['read_manifest'],root)
 need(manifest['actual_native_masks_streamed']==760 and manifest['original_scope']==SCOPE and len(manifest['rows'])==4,'ROLE_ACTUAL_READ_SCOPE')
 need([x['session_id'] for x in index['sessions']]==SCOPE,'ROLE_UNIQUE_ORDERED_SCOPE')
 checked=[]
 def budget():need(time.perf_counter()-start<budget_seconds,'ROLE_COLLECTOR_FINITE_BUDGET')
 for c,row,m in zip(ctx['contract']['cases'],index['sessions'],manifest['rows']):
  need(row['session_id']==m['session_id']==c['session_id'] and row['status']=='COMPLETE' and type(row['raw_frame_count']) is int and row['raw_frame_count']==c['native_frames'],'ROLE_SESSION_COMPLETE_COUNT')
  expected=compute_case(c,root,budget);actual=read(row['role_evidence'],root)
  need(actual==expected,'ROLE_ALL_NATIVE_MASK_ROLE_SOURCE_NUM_CONTACT_REPLAY_DIFFERENCE:'+c['session_id'])
  need(m['official_masks_NPZ']==c['masks_NPZ'] and m['original_contact_events']==c['old_contact'] and m['original_frame_states']==c['frame_states'] and m['original_QPC']==c['original_QPC'] and m['streamed_native_count']==c['native_frames'] and m['full_SHA_before_sameFD_ZIP_CRC_EOF_after'] is True and m['one_native_at_a_time'] is True,'ROLE_READ_MANIFEST_BINDING')
  checked.append(dict(session_id=c['session_id'],native_frames=c['native_frames'],role_evidence=row['role_evidence'],actual_N1_mask_recipe_replay_exact=True,original_contact_BYTE_SHA_and_per_native_QPC_flags_preserved=True))
 baseline_guard(ctx['contract'],root)
 return dict(schema='contact_role_sidecar_independent_collector_v1',status='PASS',sessions_requested=4,sessions_complete=4,frames_complete=760,checked=checked,actual_wall_seconds=time.perf_counter()-start,model_FK_Render_calls=0,visual_quality_qualified=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,Root_adoption=False,full_goal_achieved=False)
