"""Explicit paired renderer consumer; real algorithms are called only after the official fence."""
from pathlib import Path
import json,hashlib,time,types,importlib,sys
import numpy as np
from chaoyang.governance.native_frame_fk_maintenance_contract_v3 import *
def array_pin(a):
 a=np.asarray(a);need(not a.dtype.hasobject,'FK_NO_OBJECT_ARRAY')
 return {'dtype':a.dtype.str,'shape':list(a.shape),'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
def need_equal_outputs(a,b):
 need(len(a)==len(b)==3,'FK_RENDER_TRIPLE')
 for x,y in zip(a,b):need(np.asarray(x).dtype==np.asarray(y).dtype and np.array_equal(x,y,equal_nan=True),'FK_PIXEL_ALPHA_ROOT_EXACT')
def view_record(out,qa,qh,cam,K,frame,view):
 return {'native_frame':frame,'view':view,'q_arm':array_pin(qa),'q_hand':array_pin(qh),'camera':array_pin(cam),'K':array_pin(K),'BGR':array_pin(out[0]),'alpha':array_pin(out[1]),'roots':array_pin(out[2])}
def official_upstream(root,contract,sid,control):
 """Rebuild current frozen19 binding and validate original Source/Motion/GPU global owners."""
 from chaoyang.pick_place_1008 import pipeline,mask_seeds
 from chaoyang.pick_place_1008.stage_reuse import _validate
 case=contract['cases'][sid];producer=case['global_producer'];rowq=case['queue_entry']
 pref,idx,pkt=control.official_producer(root,rowq['task_id'],rowq['sessions'],rowq['native_frames'],'full',rowq['spec'],producer['index'])
 need(pref==producer,'FK_ORIGINAL_GLOBAL_PRODUCER_DRIFT')
 q=control.read_ref(root,contract['original_queue']);sr=q['source']
 sref,six,sp=control.official_producer(root,sr['task_id'],sr['sessions'],sr['native_frames'],'prepare',sr['spec'],contract['original_SOURCE_producer']['index'])
 need(sref==contract['original_SOURCE_producer'],'FK_SOURCE_GLOBAL_PRODUCER_DRIFT')
 manifest=control.read_ref(root,contract['manifest']);raw=next(r for r in manifest['sessions'] if r['session_id']==sid);prod=next(r for r in idx['sessions'] if r['session_id']==sid);srow=next(r for r in six['sessions'] if r['session_id']==sid)
 need(idx['manifest']==six['manifest']==contract['manifest'] and idx['camera_config']==six['camera_config']==contract['camera_config'],'FK_MANIFEST_CAMERA')
 seed_index=idx['mask_seed_index'];control.verify_pinned_ref(root,seed_index)
 args=control.read_ref(root,rowq['spec'])['args'];need(args[args.index('--mask-seeds')+1]==seed_index['path'],'FK_SEED_INDEX_SPEC')
 seed=mask_seeds.seed_reference(Path(seed_index['path']),sid)
 need(prod['raw_frame_count']==srow['raw_frame_count']==raw['raw_frame_count']==FULL[sid],'FK_NATIVE_COUNT')
 out={}
 for st in ['source','motion','gpu']:
  pin=case['reused_stage_receipts'][st];need(prod['stage_receipts'][st]==pin,'FK_UPSTREAM_PIN')
  rr=control.read_ref(root,pin);owner=sref if st=='source' else pref
  need(rr['producer']=={'task_id':owner['task_id'],'task_packet':owner['packet'],'run_signature':(sp if st=='source' else pkt)['run_signature'],'result_path':owner['outer']['path']},'FK_UPSTREAM_PRODUCER')
  upstream={'source':{},'motion':{'source':case['reused_stage_receipts']['source']},'gpu':{'source':case['reused_stage_receipts']['source']}}[st]
  expected=pipeline.binding(root,st,raw,contract['camera_config'],upstream,seed if st=='gpu' else None)
  need(rr['binding']==expected,'FK_FRESH_BINDING:'+st)
  folder,refpin=_validate(root,Path(pin['path']).parents[2],sid,st,expected,pipeline.STAGE_FILES[st])
  need(folder is not None and refpin==pin,'FK_STRONG_REUSE:'+st);out[st]=folder
 return out,case
def freeze_arrays(motion):
 return {k:array_pin(v) for k,v in motion.items() if isinstance(v,np.ndarray)}
def audit_renderer(base,records,n,candidate,allowed_assets=None):
 """Explicit new callable namespace injection. Original module globals/body remain immutable."""
 class Audited(base):
  def begin_native(self,**kw):
   need(allowed_assets is not None and all(r in allowed_assets for r in kw['asset_pins']),'FK2_ACTUAL_IMMUTABLE_ASSETS_SIGNED_INPUT_BOUND')
   return super().begin_native(**kw)
  def render(self,qa,qh,cam,K,**kw):
   i=len(records)//2;view='ego' if len(records)%2==0 else 'external'
   need(i<n,'FK_VIEW_OVERFLOW')
   if candidate and view=='ego':
    self._frame_counts=(getattr(getattr(self,'_native_fk_cache',None),'evaluations',0),getattr(getattr(self,'_native_fk_cache',None),'hits',0))
   t=time.perf_counter();cpu=time.process_time();out=super().render(qa,qh,cam,K,**kw)
   record=view_record(out,qa,qh,cam,K,i,view)
   record['wall_seconds']=time.perf_counter()-t;record['CPU_process_seconds']=time.process_time()-cpu
   if candidate and view=='external':
    c=self._native_fk_cache
    record['FK_render_evaluations_delta']=c.evaluations-self._frame_counts[0];record['FK_render_hits_delta']=c.hits-self._frame_counts[1]
    need(record['FK_render_evaluations_delta']==record['FK_render_hits_delta']==3,'FK_ACTUAL_RENDER_CACHE_NOT_ACTIVE')
   records.append(record);return out
 return Audited
def session_function(module,renderer):
 fn=module.render_session
 g=dict(fn.__globals__);g['LookupRobotRenderer']=renderer
 return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
def qualification_evidence(root,context):
 """Fixed prior task, exact Root plan/spec/code/packet/global collector, no partial or filenames."""
 c=context['contract'];tid=QUALIFICATION_TASK
 from chaoyang.governance import native_frame_fk_maintenance_contract_v2 as prior
 prior_source=c['qualification_source']
 need(prior_source['task_id']==QUALIFICATION_TASK and prior_source['operation']==QUALIFICATION_OPERATION,'FK3_FIXED_PRIOR_FK2_TASK_OPERATION')
 state=json.loads((root/'docs/governance/LONG_HORIZON_TASK_STATE.json').read_text());reg=json.loads((root/'tasks/current/INDEX.json').read_text())
 rows=[r for r in state['tasks'] if r['task_id']==tid];regs=[r for r in reg['task_packets'] if r['task_id']==tid]
 need(len(rows)==len(regs)==1,'FK_24_QUALIFICATION_NOT_EXECUTED');r,g=rows[0],regs[0]
 need(r['result']==prior_source['outer'] and r['terminal_receipt']==prior_source['terminal_receipt'] and r['task_packet']==prior_source['packet'],'FK3_PRIOR_CURRENT_OFFICIAL_PINS')
 outer=read(r['result'],root);terminal=read(r['terminal_receipt'],root);packet=read(outer['task_packet'],root)
 qcontext=prior.approved_context(packet['inputs']['spec']['path'],root);need(prior.mode(qcontext['spec'])==0,'FK_QUAL_TASK_MODE');prior.validate_packet_context(packet,qcontext,root)
 need(qcontext['contract']['robot_asset_input_refs']==c['robot_asset_input_refs'] and qcontext['contract']['fixed_existing_runtime_code']==c['fixed_existing_runtime_code'],'FK3_SAME_QUALIFIED_ASSET69_AND_IO1')
 need(qcontext['plan_ref']==prior_source['plan'] and packet['inputs']['spec']==prior_source['spec'] and qcontext['spec']['contract']==prior_source['contract'],'FK3_PRIOR_APPROVED_SPEC_PLAN_CONTRACT')
 need(r['status']==outer['status']==terminal['status']=='PASSED' and type(outer['worker_returncode']) is int and outer['worker_returncode']==0 and outer['validation_errors']==[] and outer['resource_wait'] is False,'FK_24_GLOBAL_PASS')
 need(r['pid'] is None and r['proc_start_ticks'] is None and g['execution_allowed'] is False,'FK_24_CLOSED')
 need(outer['task_packet']==r['task_packet'] and outer['run_signature']==packet['run_signature']==r['run_signature'] and g['packet_sha256']==outer['task_packet']['sha256'],'FK_24_FENCE_BINDING')
 need(terminal['result']==r['result']==g['result'] and r['terminal_receipt']==g['terminal_receipt'],'FK_24_TERMINAL')
 index=read(outer['required_outputs']['candidate/INDEX.json'],root);need(index==read(outer['stage_result'],root),'FK_24_INDEX_RESULT')
 need(index['schema']=='native_frame_fk_dual_view_complete_candidate_index_v1' and index['status']=='COMPLETE' and index['operation']==QUALIFICATION_OPERATION and index['paired_native_frames']==12 and index['actual_control_views']==index['actual_candidate_views']==24 and index['all_paired_BGR_alpha_roots_exact'] is True,'FK_24_FULL_SCOPE')
 need(index['sessions_complete']==index['sessions_requested']==4 and [x['session_id'] for x in index['sessions']]==SCOPES[0] and sum(len(x['rows']) for x in index['sessions'])==12,'FK_24_EXACT_FOUR_SESSIONS_AND12')
 need(index['candidate_runtime_code']==c['candidate_runtime_code'] and index['control_runtime_code']==c['control_runtime_code'],'FK_24_SAME_CANDIDATE_CONTROL')
 for row in index['sessions']:
  need(row['sampled_native_frames']==SAMPLES[row['session_id']] and row['all_paired_BGR_alpha_roots_exact'] is True and len(row['rows'])==3 and [p['native_frame'] for p in row['rows']]==SAMPLES[row['session_id']] and all(len(p['control'])==len(p['candidate'])==2 for p in row['rows']),'FK_24_SCOPE_ROWS')
 proof=read(outer['maintenance_collector'],root)
 need(proof['status']=='PASSED' and proof['operation']==QUALIFICATION_OPERATION and proof['paired_native_frames']==12 and proof['task_packet']==outer['task_packet'] and proof['run_signature']==packet['run_signature'] and proof['candidate_index']==outer['required_outputs']['candidate/INDEX.json'],'FK_24_COLLECTOR_GLOBAL')
 need(outer['required_outputs']['candidate/INDEX.json']==prior_source['index'] and outer['maintenance_collector']==prior_source['collector'] and packet['run_signature']==prior_source['run_signature'],'FK3_PRIOR_EXACT_CURRENT_GLOBAL_RESULT')
 return {'outer':r['result'],'packet':outer['task_packet'],'index':outer['required_outputs']['candidate/INDEX.json'],'collector':outer['maintenance_collector'],'run_signature':packet['run_signature']}
def qualification_closed(root,context):
 actual=qualification_evidence(root,context)
 selected=read(selected_qualification_ref(context,root,True),root)
 need(all(selected[k]==actual[k] for k in ('outer','packet','index','collector','run_signature')),'FK_ROOT_SELECTED_ACTUAL_FK2_24_CLOSURE')
 return actual
def load_pair_modules(context,root=ROOT):
 c=context['contract'];import_package(c['control_runtime_code'],'chaoyang_fk_frozen_T2_control_v1',root);import_package(c['candidate_runtime_code'],'chaoyang_fk_dualview_candidate_v1',root)
 control=importlib.import_module('chaoyang_fk_frozen_T2_control_v1.operation')
 cs=importlib.import_module('chaoyang_fk_frozen_T2_control_v1.render_session');cr=importlib.import_module('chaoyang_fk_frozen_T2_control_v1.renderer')
 ns=importlib.import_module('chaoyang_fk_dualview_candidate_v1.render_session');nr=importlib.import_module('chaoyang_fk_dualview_candidate_v1.renderer')
 return control,cs,cr,ns,nr
def render_24(*args,**kwargs):
 raise ValueError('FK3_NO_NEW_24VIEW_OPERATION_OR_QUALIFICATION_EXECUTION')
def control_existing_closed(root,context):
 c=context['contract'];v=c['actual_T2_control_producer']
 state=json.loads((root/'docs/governance/LONG_HORIZON_TASK_STATE.json').read_text());reg=json.loads((root/'tasks/current/INDEX.json').read_text())
 lr=[x for x in state['tasks'] if x['task_id']==v['task_id']];rr=[x for x in reg['task_packets'] if x['task_id']==v['task_id']]
 need(len(lr)==len(rr)==1,'FK_T2_UNIQUE');led,row=lr[0],rr[0]
 need(led['result']==row['result']==v['outer'] and led['terminal_receipt']==row['terminal_receipt']==v['terminal_receipt'],'FK_T2_OFFICIAL_CLOSURE')
 o=read(v['outer'],root);p=read(v['packet'],root);terminal=read(v['terminal_receipt'],root);index=read(v['index'],root)
 need(led['status']==o['status']==terminal['status']=='PASSED' and type(o['worker_returncode']) is int and o['worker_returncode']==0 and o['validation_errors']==[] and 'resource_wait' in o and o['resource_wait'] is None,'FK_EXACT_T2_PASS')
 need(led['pid'] is None and led['proc_start_ticks'] is None and row['execution_allowed'] is False,'FK_T2_DEAD_CLOSED')
 need(o['task_packet']==led['task_packet']==v['packet'] and p['run_signature']==o['run_signature']==led['run_signature']==v['run_signature'] and row['packet_sha256']==v['packet']['sha256'],'FK_T2_PACKET_SIGNATURE')
 need(terminal['result']==v['outer'] and o['required_outputs']['candidate/INDEX.json']==v['index'] and index==read(o['stage_result'],root),'FK_T2_GLOBAL_INDEX_RESULT')
 need(index['status']=='COMPLETE' and index['sessions_complete']==index['sessions_requested']==2 and index['frames_complete']==295 and [x['session_id'] for x in index['sessions']]==['007','199'],'FK_T2_FULL295')
 need(index['operation']=='run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v2','FK_T2_EXACT_CONTROL_OPERATION')
 need(len(p['runtime_code'])==106 and all(x in p['runtime_code'] for x in c['control_runtime_code']),'FK_T2_RUNTIME_CONTROL_SIX')
 for pin in p['runtime_code']:exact(pin,root)
 for sid in ['007','199']:
  row=next(x for x in index['sessions'] if x['session_id']==sid)
  need(row==c['control_existing_sessions'][sid],'FK_T2_SESSION_CONTROL_FREEZE')
  for pin in row['files'].values():exact(pin,root)
 return v
def paired_stage_binding(context,sid):
 return {'schema':'native_frame_fk_paired_stage_binding_v1','stage':context['spec']['stage'],'session_id':sid,'original_native_count':FULL[sid],'sampled_native_frames':SAMPLES[sid] if mode(context['spec'])==0 else list(range(FULL[sid])),'manifest':context['contract']['manifest'],'camera_config':context['contract']['camera_config'],'upstream':context['contract']['cases'][sid]['reused_stage_receipts'],'original_frozen_production_code':context['contract']['original_production_code'],'control_runtime_code':context['contract']['control_runtime_code'],'candidate_runtime_code':context['plan']['candidate_runtime_code'],'new_governance_runtime_code':context['plan']['maintenance_governance_runtime_code'],'single_factor':'SAME_NATIVE_TWO_VIEWS_CAMERA_INDEPENDENT_FK_REUSE','same_q_Raw_Clean_masks_camera':True}

def render_full(root,context,out,control,cs,cr,ns,nr,guard):
 from chaoyang.pick_place_1008 import source,pipeline
 results=[]
 for sid in SCOPES[1]:
  guard();t=time.perf_counter();folders,case=official_upstream(root,context['contract'],sid,control);reuse=time.perf_counter()-t
  t=time.perf_counter();session=source.load_prepared(folders['source']/'source');motion=pipeline.load_arrays(folders['motion']/'actions.npz');masks=pipeline.load_masks(folders['gpu']);clean=pipeline.load_arrays(folders['gpu']/'clean_rgb.npz')['rgb'];masks['raw_rgb']=pipeline.load_arrays(folders['gpu']/'raw_rgb.npz')['rgb'];masks['motion_info']=pipeline.read(folders['motion']/'motion.json')
  masks['upstream_refs']={k:source.ref(p) for k,p in {'actions':folders['motion']/'actions.npz','frame_mapping':folders['motion']/'frame_mapping.jsonl','motion_info':folders['motion']/'motion.json','raw_rgb':folders['gpu']/'raw_rgb.npz','clean_rgb':folders['gpu']/'clean_rgb.npz','masks':folders['gpu']/'masks.npz','mask_info':folders['gpu']/'masks.json'}.items()}
  load=time.perf_counter()-t;before=freeze_arrays(motion);rgb_before={'raw':array_pin(masks['raw_rgb']),'clean':array_pin(clean),'masks':freeze_arrays(masks)}
  order=['control','candidate'] if sid=='007' else ['candidate','control'];traces={};times={};filepins={}
  for which in order:
   guard();traces[which]=[];module,base=(cs,cr.LookupRobotRenderer) if which=='control' else (ns,nr.LookupRobotRenderer)
   audited=audit_renderer(base,traces[which],FULL[sid],which=='candidate',context['packet']['inputs']['files'])
   fn=session_function(module,audited);t=time.perf_counter();cpu=time.process_time()
   fn(root,out/which/sid,session,motion,clean,masks)
   times[which]={'render_session_wall_seconds':time.perf_counter()-t,'render_session_CPU_process_seconds':time.process_time()-cpu}
   need(len(traces[which])==FULL[sid]*2,'FK_ALL_NATIVE_VIEW_TRACE')
   filepins[which]={name:ref(out/which/sid/name) for name in pipeline.STAGE_FILES['render']}
  for a,b in zip(traces['control'],traces['candidate']):
   need({k:v for k,v in a.items() if k not in ['wall_seconds','CPU_process_seconds']}=={k:v for k,v in b.items() if k not in ['wall_seconds','CPU_process_seconds','FK_render_evaluations_delta','FK_render_hits_delta']},'FK_ALL_NATIVE_PREENCODE_BGR_ALPHA_ROOT_Q_CAMERA_EXACT')
  guard();t=time.perf_counter();comparison=control.compare_fresh_render_outputs(out/'control'/sid,out/'candidate'/sid,session.qpc_ns,FULL[sid])
  existing=Path(context['contract']['control_existing_sessions'][sid]['files']['SESSION.json']['path']).parent
  replay=control.compare_fresh_render_outputs(existing,out/'control'/sid,session.qpc_ns,FULL[sid]);compare=time.perf_counter()-t
  need(before==freeze_arrays(motion) and rgb_before=={'raw':array_pin(masks['raw_rgb']),'clean':array_pin(clean),'masks':freeze_arrays(masks)},'FK_Q_RAW_CLEAN_MASK_MUTATION')
  tracefile=out/(sid+'_PAIRED_ALL_NATIVE_VIEW_TRACE.json')
  with tracefile.open('x') as f:json.dump(traces,f,indent=2);f.write('\n')
  results.append({'session_id':sid,'raw_frame_count':FULL[sid],'pair_order':order,'files':filepins,'paired_view_trace':ref(tracefile),'upstream':case['reused_stage_receipts'],'frozen_action_array_identity':before,'frozen_raw_clean_mask_identity':rgb_before,'comparison':comparison,'control_replay_against_official_T2':replay,'all_paired_BGR_alpha_roots_exact':True,'timing_segments_seconds':{'reuse_validate':reuse,'load_arrays':load,'compare_and_official_T2_replay':compare},'paired_render_seconds':times})
 return results
def run(context):
 """Only the Root-approved wrapper calls this, after real CLI and packet fencing."""
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 root=ROOT;spec=context['spec'];i=mode(spec);args=['--maintenance-spec',context['plan']['approved_spec']['path']];started=time.perf_counter();cpu=time.process_time()
 def guard():
  authorize_operation_entry(OPS[i],args)
  need(time.perf_counter()-started<BUDGET_TABLE[i]['wall_seconds_max'],'FK_FINITE_WALL')
 guard();state=json.loads((root/'docs/governance/LONG_HORIZON_TASK_STATE.json').read_text());need(state['next_task']['task_id']==TASKS[i],'FK_CURRENT_TASK')
 context=dict(context,packet=json.loads((root/'tasks/current'/TASKS[i]/'TASK_PACKET.json').read_text()));validate_packet_context(context['packet'],context,root)
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_packet_asset_inputs
 validate_packet_asset_inputs(context['packet'],context['contract'],root,True)
 control,cs,cr,ns,nr=load_pair_modules(context,root);control._PIN_VERIFIED_SHARED.clear()
 qualification=None
 if i==1:
  qualification=qualification_closed(root,context);control_existing_closed(root,context)
 out=root/'_run/current'/TASKS[i]/'attempts/attempt_0001/candidate';need(not out.exists(),'FK_FRESH_CANDIDATE');out.mkdir()
 t=time.perf_counter();sessions=render_full(root,context,out,control,cs,cr,ns,nr,guard)
 for row in sessions:
  rp=out/(row['session_id']+'_SEALED_PAIRED_STAGE.json')
  receipt={'schema':'native_frame_fk_sealed_paired_stage_v1','status':'COMPLETE','binding':paired_stage_binding(context,row['session_id']),'producer':current_producer(root),'evidence':dict(row)}
  with rp.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
  row['paired_stage_receipt']=ref(rp)
 inner=time.perf_counter()-started
 guard();index={'schema':'native_frame_fk_dual_view_complete_candidate_index_v1','status':'COMPLETE','stage':spec['stage'],'task_id':TASKS[i],'operation':OPS[i],'sessions':sessions,'sessions_complete':len(SCOPES[i]),'sessions_requested':len(SCOPES[i]),'paired_native_frames':12 if i==0 else 295,'actual_control_views':24 if i==0 else 590,'actual_candidate_views':24 if i==0 else 590,'all_paired_BGR_alpha_roots_exact':True,'control_runtime_code':context['contract']['control_runtime_code'],'candidate_runtime_code':context['plan']['candidate_runtime_code'],'contract':spec['contract'],'plan':context['plan_ref'],'manifest':context['contract']['manifest'],'camera_config':context['contract']['camera_config'],'producer':current_producer(root),'qualification':qualification,'timing_seconds':{'inner_entry_guard_load_and_pair_wall_before_INDEX_seal':inner,'inner_CPU_process_seconds':time.process_time()-cpu},'outer_Task_wall':'RECORDED_SEPARATELY_BY_FENCED_RUNNER_EXCLUDES_REGISTRATION_AND_FINAL_PUBLICATION','paired_timing_scope':'24_robot_views_only' if i==0 else 'same_caller_single_control_T2_plus_candidate_full_Render295','full_pipeline_speedup':'UNKNOWN_NOT_MEASURED','same_q_Raw_Clean_masks_camera':True,'contact_changes_q':False,'parameters_fitted':0,'physical_ground_truth':False,'physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False,'human_review_status':'PENDING','Root_adoption':False,'full_goal_achieved':False}
 for name in ['INDEX.json','RESULT.json']:
  with (out/name).open('x') as f:json.dump(index,f,indent=2);f.write('\n')
 return 0
