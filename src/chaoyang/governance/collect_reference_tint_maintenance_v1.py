"""Independent new maintenance collector; no model/render call or visual adoption."""
from pathlib import Path
import json,math,time
from chaoyang.governance.reference_tint_maintenance_contract_v1 import *
def collect(packet,root=ROOT):
 started=time.perf_counter()
 context=approved_context(packet['inputs']['spec']['path'],root);validate_packet_context(packet,context,root)
 plan,spec,contract=context['plan'],context['spec'],context['contract']
 module=import_consumer(plan,root);module._PIN_VERIFIED_SHARED.clear()
 folder=root/'_run/current'/TASK/'attempts/attempt_0001/candidate'
 index=read(ref(folder/'INDEX.json'),root);result=read(ref(folder/'RESULT.json'),root)
 need(index==result,'TINT_CANDIDATE_INDEX_RESULT_DIFFERENCE')
 need(index['schema']==plan['collector_schema'] and index['status']=='COMPLETE' and index['stage']=='render' and index['operation']==OPERATION,'TINT_COMPLETE_CUSTOM_COLLECTOR_SCHEMA')
 need(index['sessions_requested']==index['sessions_complete']==2 and index['frames_complete']==295,'TINT_FULL295_COMPLETE')
 need(index['manifest']==contract['manifest'] and index['camera_config']==contract['camera_config'] and index['contract']==spec['contract'],'TINT_INDEX_SOURCE_CAMERA_CONTRACT_BINDING')
 need(index['same_raw_q_camera_Clean_masks'] is True and index['HOLD_evaluation_only'] is True and type(index['parameters_fitted']) is int and index['parameters_fitted']==0,'TINT_NO_FIT_DATA_OR_MODEL_CHANGE')
 for k in ('physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible','full_goal_achieved'):need(index[k] is False,'TINT_NO_TRUTH_OR_FULL_GOAL_PROMOTION')
 need(index['human_review_status']=='PENDING' and index['full_pipeline_speedup']=='UNKNOWN_NOT_MEASURED','TINT_AI_AND_SPEEDUP_PENDING')
 rows=index['sessions'];need([r['session_id'] for r in rows]==list(SCOPE),'TINT_EXACT_NO_DUPLICATE_SESSION_SET')
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 from chaoyang.pick_place_1008 import source,pipeline
 from chaoyang.pick_place_1008.stage_reuse import _validate
 expected_producer=current_producer(root)
 evidence=[]
 for row in rows:
  sid=row['session_id'];n=SCOPE[sid];out=folder/'sessions'/sid
  need(row['status']=='COMPLETE' and row['raw_frame_count']==n,'TINT_FULL_NATIVE_SESSION_COMPLETE')
  upstream,case=module._upstream_case(root,contract,sid) # Current independent source/HOLD global producer + seed + binding + files.
  need(row['source_motion_gpu_stage_receipts']==case['reused_stage_receipts'],'TINT_INDEX_OLD_UPSTREAM_MATCH')
  prepared=source.load_prepared(upstream['source']/'source')
  oldr=read(case['original_render_receipt'],root)
  rawrow=next(r for r in read(contract['manifest'],root)['sessions'] if r['session_id']==sid)
  old_expected=pipeline.binding(root,'render',rawrow,contract['camera_config'],case['reused_stage_receipts'])
  need(oldr['binding']==old_expected,'TINT_INDEPENDENT_ORIGINAL_RENDER_BINDING')
  validated,rpin=_validate(root,Path(case['original_render_receipt']['path']).parents[2],sid,'render',old_expected,pipeline.STAGE_FILES['render'])
  need(validated is not None and rpin==case['original_render_receipt'],'TINT_ORIGINAL_CONTROL_RENDER_CURRENT')
  need(set(row['files'])==set(pipeline.STAGE_FILES['render']),'TINT_SIX_COMPLETE_RENDER_FILES')
  for name,pin in row['files'].items():need(pin['path']==str(out/name) and ref(exact(pin,root))==pin,'TINT_FRESH_RENDER_FILE_PATH_OR_SHA')
  need(row['maintenance_render_receipt']['path']==str(out/'STAGE_render_maintenance.json'),'TINT_CUSTOM_RECEIPT_FRESH_PATH')
  rr=read(row['maintenance_render_receipt'],root)
  expected_binding={'stage':'render','session_id':sid,'raw_frame_count':n,'manifest':contract['manifest'],'camera_config':contract['camera_config'],'upstream':case['reused_stage_receipts'],'original_frozen_production_code':contract['original_production_code'],'effective_render_candidate_code':plan['candidate_runtime_code'],'single_factor':'REFERENCE_TINT_EXACT_LOOKUP'}
  need(rr['schema']=='pick_place_1008_maintenance_sealed_render_v1' and rr['status']=='COMPLETE' and rr['binding']==expected_binding,'TINT_FRESH_EXPECTED_RENDER_BINDING_EXACT')
  need(rr['producer']==expected_producer and rr['files']==row['files'],'TINT_REGISTERED_CURRENT_PRODUCER_AND_FILES')
  comparison=module.compare_fresh_render_outputs(Path(case['original_session_result']['path']).parent,out,prepared.qpc_ns,n)
  need(comparison==row['comparison']==rr['decoded_pixel_and_QPC_comparison'],'TINT_INDEPENDENT_ALL_NATIVE_DECODE_PTS_CONTACT_FLAGS_FK_EXACT')
  need(row['new_visual_quality']=='PENDING_SERVER_LOCAL_ACTUAL_REVIEW','TINT_NO_AUTOMATIC_NEW_VISUAL_PASS')
  timings=row['timing_segments_seconds']
  need(set(timings)=={'strong_source_motion_GPU_reuse_validate','load_arrays','candidate_render_session_wall','fresh_output_compare_decode_PTS_arrays_contact','seal_and_hash'} and all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in timings.values()),'TINT_ACTUAL_NONNEGATIVE_TIMING_SEGMENTS')
  need(row['original_control_full_render_stage_matching_timing']=='UNKNOWN_NO_MATCHED_SEGMENT_CONTROL' and row['full_pipeline_speedup']=='UNKNOWN_NOT_MEASURED','TINT_NO_UNMATCHED_FULL_SPEEDUP')
  evidence.append({'session_id':sid,'raw_frame_count':n,'current_old_Source_Motion_GPU_and_Render_valid':True,'fresh_render_binding_exact':True,'decoded_pixels_all_native_PTS_contact_flags_FK_exact':True,'new_render_receipt':row['maintenance_render_receipt']})
 return {'schema':'reference_tint_maintenance_independent_collector_validation_v1','status':'PASSED','sessions':evidence,'frames_complete':295,'independent_collector_wall_seconds':time.perf_counter()-started,'model_calls':0,'render_calls_by_collector':0,'visual_improvement':'UNKNOWN','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False,'human_review_status':'PENDING','Root_adoption':False}
