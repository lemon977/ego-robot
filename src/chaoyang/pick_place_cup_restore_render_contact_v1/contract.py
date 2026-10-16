from __future__ import annotations
import copy,hashlib,json,os,shlex,time
from pathlib import Path
from chaoyang.governance.candidate_optimization_family_v10 import descriptor,published_current,exact
ROOT=Path('/path/to/chaoyang')
FAMILY='pick_place_observed_cup_restore_clean_dev4_render_contact_candidate_v1'
OPERATION='run_pick_place_observed_cup_restore_clean_dev4_render_contact_candidate_v1'
TASK='exact78_pick_place_observed_cup_restored_clean_dev4_render_contact_candidate_revision1_20261010'
SCOPE=['001','003','008','100'];NATIVE=760
BUDGETS=dict(cpu_threads_total=2,gpu_count=0,model_calls=0,render_calls=1520,session_attempts_max=1,training_steps=0,wall_seconds_max=1800,workers_max=1)
RESOURCE_BOUNDS=dict(resident_peak_bytes_max=3<<30,address_space_bytes_max=8<<30)
SPEC_KEYS={'schema_version','task_id','operation','sessions','native_frames','contract','budgets','resource_bounds'}
CANONICAL='src/chaoyang/pick_place_cup_restore_render_contact_v1/'
OWN_FILES=[CANONICAL+x for x in ['__init__.py','render_adapter.py','private_render.py','contract.py','driver.py','collector.py']]+['src/chaoyang/ops/'+OPERATION+'.py','src/chaoyang/governance/execute_pick_place_cup_restore_render_contact_v1.py']
SHARED_FILES=['src/chaoyang/governance/candidate_optimization_family_v10.py','src/chaoyang/governance/current_candidate_optimization_contracts_v10.py','src/chaoyang/governance/common_candidate_optimization_v11.py']
def need(v,s):
 if not v:raise ValueError(s)
def path(p,root=ROOT,nas=False):
 p=Path(p);base=Path('/path/to/private_storage/datasets/ego/pick_place_1008') if nas else root
 need(p.is_absolute() and '..' not in p.parts and p.is_relative_to(base),'CUP_PROJECT_OR_FROZEN_NAS_ONLY')
 for x in (p,*p.parents):
  need(not x.is_symlink(),'CUP_SYMLINK_REJECTED')
  if x==base:break
 return p
def pin(v,root=ROOT,nas=False,tick=lambda:None):
 need(isinstance(v,dict) and set(v)=={'path','bytes','sha256'} and type(v['bytes']) is int and v['bytes']>=0,'CUP_EXACTREF_SCHEMA')
 p=path(v['path'],root,nas);s=p.stat();before=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);h=hashlib.sha256();count=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):tick();h.update(b);count+=len(b)
 z=p.stat();after=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 need(count==v['bytes'] and h.hexdigest()==v['sha256'] and before==after,'CUP_SHA_OR_SIZE_DRIFT:'+str(p))
 return p
def read(v,root=ROOT,nas=False):
 p=path(v['path'],root,nas);need(set(v)=={'path','bytes','sha256'} and type(v['bytes']) is int and 0<=v['bytes']<=16<<20,'CUP_BOUNDED_JSON_REF')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  before=os.fstat(f.fileno());b=f.read((16<<20)+1);after=os.fstat(f.fileno())
 need(len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'] and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'CUP_JSON_SAME_FD_DRIFT')
 return json.loads(b)
def ref(p,root=ROOT,tick=lambda:None):
 p=path(p,root);h=hashlib.sha256();count=0;before=p.stat()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):tick();tick();h.update(b);count+=len(b)
 after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'CUP_OUTPUT_CHANGED_DURING_SHA')
 return dict(path=str(p),bytes=count,sha256=h.hexdigest())
def runtime_shape(rows,root=ROOT):
 need(isinstance(rows,list),'CUP_RUNTIME_LIST')
 out={}
 for r in rows:
  need(set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int,'CUP_RUNTIME_REF_SCHEMA')
  p=path(r['path'],root);need(str(p) not in out,'CUP_DUPLICATE_RUNTIME_PATH');out[str(p)]=r
 return out
def resource_shape(value):
 need(isinstance(value,dict) and set(value)==set(RESOURCE_BOUNDS) and all(type(v) is int for v in value.values()) and value==RESOURCE_BOUNDS,'CUP_TYPED_FIXED_RESOURCE_BOUNDS')
 return True
def spec_shape(spec):
 resource_shape(spec['resource_bounds'])
 need(set(spec)==SPEC_KEYS and spec['schema_version']=='pick_place_cup_restore_render_contact_spec_v1','CUP_SPEC_UNKNOWN_KEYS')
 need(spec['task_id']==TASK and spec['operation']==OPERATION and spec['sessions']==SCOPE and type(spec['native_frames']) is int and spec['native_frames']==NATIVE,'CUP_FIXED_DEV_SCOPE')
 need(spec['budgets']==BUDGETS and all(type(v) is int for v in spec['budgets'].values()),'CUP_TYPED_FINITE_BUDGETS')
def plan_shape(plan,spec,root=ROOT,verify=True):
 spec_shape(spec);resource_shape(plan['resource_bounds'])
 need(plan['schema']=='root_planned_finite_cup_restore_render_contact_operation_v1' and plan['family_key']==FAMILY and plan['operations']==[OPERATION],'CUP_PLAN_FAMILY_OP')
 need(plan['task_id']==TASK and plan['sessions']==SCOPE and type(plan['native_frames']) is int and plan['native_frames']==NATIVE and plan['budgets']==BUDGETS and all(type(v) is int for v in plan['budgets'].values()),'CUP_PLAN_SCOPE_BUDGET')
 rows=runtime_shape(plan['candidate_runtime_code'],root)
 need(set(rows)=={str(root/f) for f in OWN_FILES},'CUP_EXACT_UNIQUE_EIGHT_CANONICAL_FILES')
 shared=runtime_shape(plan['maintenance_governance_runtime_code'],root)
 need(set(shared)=={str(root/f) for f in SHARED_FILES},'CUP_EXACT_SHARED_SUCCESSOR_THREE')
 need(plan['no_original_contact_source_q_write'] is True and plan['no_HOLD_fit'] is True and type(plan['model_calls']) is int and plan['model_calls']==0 and type(plan['render_calls']) is int and plan['render_calls']==1520 and plan['fresh_FK_required'] is True,'CUP_TRUTH_OR_EXECUTION_DOMAIN')
 if verify:
  for v in list(rows.values())+list(shared.values()):pin(v,root)
def expected_runtime(ctx,root=ROOT,verify=False):
 rows=ctx['contract']['original_runtime_93'];need(len(runtime_shape(rows,root))==93,'CUP_OLD93_EXACT')
 out=runtime_shape(rows+ctx['plan']['candidate_runtime_code']+ctx['plan']['maintenance_governance_runtime_code']+ctx['contract']['fixed_existing_runtime_code'],root)
 need(len(out)==105,'FR_OLD93_OWN8_SHARED3_FIXED_IO1_RUNTIME105')
 if verify:
  for v in out.values():pin(v,root)
 return list(out.values())
def approved_context(spec_path,root=ROOT):
 authority=published_current(root);route=authority['special_status_contracts'].get(FAMILY)
 need(isinstance(route,dict),'CUP_ROOT_FAMILY_NOT_PUBLISHED')
 d=descriptor(route.get('root_registered_descriptor'),root)
 need(d['family_key']==FAMILY and d['operations']==[OPERATION] and {k:v for k,v in route.items() if k!='root_registered_descriptor'}==d['route_value'],'CUP_PUBLISHED_DESCRIPTOR_DRIFT')
 allowed=route.get('maintenance_render_operation_allowlist',{}).get(OPERATION)
 need(isinstance(allowed,dict) and allowed==d['maintenance_operation_allowlist'].get(OPERATION) and allowed.get('execution_allowed') is True,'CUP_ROOT_EXPLICIT_ALLOWLIST')
 plan=read(allowed['plan'],root);need(plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED','CUP_ROOT_PLAN_PENDING')
 need(str(path(spec_path,root))==plan['approved_spec']['path'],'CUP_UNAPPROVED_SPEC_PATH')
 spec=read(plan['approved_spec'],root);plan_shape(plan,spec,root,verify=False)
 contract=read(spec['contract'],root)
 need(contract['schema']=='pick_place_DEV4_cup_restore_render_contact_contract_v1' and contract['operation']==OPERATION and contract['sessions']==SCOPE and contract['native_frames']==NATIVE,'CUP_CONTRACT_SCOPE')
 resource_shape(contract['resource_bounds'])
 need(contract['resource_bounds']==spec['resource_bounds']==plan['resource_bounds'],'CUP_RESOURCE_BOUNDS_EXACT')
 need(contract['candidate_runtime_code']==plan['candidate_runtime_code'],'CUP_CONTRACT_CODE_PIN')
 need(plan['Clean_input_selection_approved'] is True,'FR_ROOT_CLEAN_SELECTION_PENDING')
 need(contract['robot_asset_input_refs']==plan['robot_asset_input_refs'] and contract['fixed_existing_runtime_code']==plan['fixed_existing_runtime_code'],'FR_PLAN_ASSET_IO_DIRECT_EXACT')
 verify_robot_assets(contract,root,verify=False)
 verify_clean_parent(contract,root,verify_large=False)
 return dict(plan=plan,spec=spec,contract=contract,plan_ref=allowed['plan'],descriptor=d)
def command(spec_ref):return 'PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run '+shlex.join([OPERATION,'--render-contact-spec',spec_ref['path']])
def signature(inputs,runtime,config,cmd):
 from chaoyang.governance.common import canonical_bytes
 return hashlib.sha256(canonical_bytes(dict(inputs=inputs,runtime_code=runtime,config=config,command=cmd,schema='pick-place-cup-restore-fresh-render-contact-task-v1'))).hexdigest()
def validate_packet_context(packet,ctx,root=ROOT):
 resource_shape(packet['config']['resource_bounds'])
 need(packet['task_id']==TASK and packet['execution_mode']=='EXACT78_CANDIDATE','CUP_PACKET_FIXED_MODE')
 need(packet['budgets']==packet['config']['budgets']==BUDGETS and all(type(v) is int for v in packet['budgets'].values()) and all(type(v) is int for v in packet['config']['budgets'].values()),'CUP_PACKET_TYPED_BUDGETS')
 need(packet['inputs']==dict(spec=ctx['plan']['approved_spec'],files=[ctx['spec']['contract'],*ctx['contract']['robot_asset_input_refs']],maintenance_render_plan=ctx['plan_ref']),'CUP_PACKET_INPUTS_EXACT')
 need(runtime_shape(packet['runtime_code'],root)==runtime_shape(expected_runtime(ctx,root),root),'CUP_PACKET_RUNTIME_EXACT')
 need(packet['commands']==[command(ctx['plan']['approved_spec'])],'CUP_PACKET_ONE_EXACT_COMMAND_NO_EXTRA')
 need(packet['run_signature']==signature(packet['inputs'],packet['runtime_code'],packet['config'],packet['commands'][0]),'CUP_PACKET_SIGNATURE')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'CUP_PACKET_REQUIRED_OUTPUTS')
 return True
def require_registered_context(spec_path,argv,root=ROOT):
 """Full closure at entry/exit; per-task immutable identity cache only between them."""
 need(argv==['--render-contact-spec',str(path(spec_path,root))],'CUP_ONLY_ONE_SPEC_NO_EXTRA')
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry,_caller_is_writer_descendant
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 authorize_operation_entry(OPERATION,argv)
 producer=current_producer(root);need(producer['task_id']==TASK,'CUP_EXPLICIT_CURRENT_TASK')
 packet=read(producer['task_packet'],root);ctx=approved_context(spec_path,root);validate_packet_context(packet,ctx,root);validate_fence_schema(packet['writer'])
 writer=packet['writer'];pp=Path(producer['task_packet']['path']);wp=root/'_run/current'/TASK/'attempts/attempt_0001/WRITER.json'
 def identity(p):
  p=path(p,root);s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 code_ids={r['path']:identity(r['path']) for r in packet['runtime_code']}
 nav=(root/'tasks/current/INDEX.json',root/'docs/governance/LONG_HORIZON_TASK_STATE.json',pp,wp)
 nav_ids=None
 def full_guard():
  authorize_operation_entry(OPERATION,argv)
  need(current_producer(root)==producer,'CUP_CURRENT_PRODUCER_DRIFT')
 def guard():
  nonlocal nav_ids
  for code,pinned in code_ids.items():
   if identity(code)!=pinned:
    # Preserve full-SHA verification on drift, then stop even for same-byte replacement.
    full_guard();raise ValueError('CUP_RUNTIME_CODE_FILE_IDENTITY_CHANGED_STOP')
  need(_caller_is_writer_descendant(writer['pid'],writer['proc_start_ticks'],os.getpid()),'CUP_LIVE_WRITER_DESCENDANT_FENCE')
  observed=tuple(identity(p) for p in nav)
  if nav_ids!=observed:
   # Ledger/registry heartbeats may replace navigation; recheck its signed content.
   idx=json.loads(nav[0].read_text());ledger=json.loads(nav[1].read_text())
   active=[r for r in ledger['tasks'] if r['status'] in ('RUNNING','CLAIMED','WAIT_GPU_RESOURCE')]
   routes=[r for r in idx['task_packets'] if r.get('execution_allowed') is True]
   need(len(active)==len(routes)==1 and (ledger.get('next_task') or {}).get('task_id')==TASK,'CUP_SIGNED_FENCE_UNIQUE_ACTIVE')
   a,g=active[0],routes[0];gp=Path(g['packet_path']);gp=gp if gp.is_absolute() else root/gp
   need(a['task_id']==g['task_id']==TASK and a['status']=='RUNNING' and a['run_signature']==packet['run_signature'] and a['task_packet']==producer['task_packet'] and gp==pp and g['packet_sha256']==producer['task_packet']['sha256'],'CUP_SIGNED_LEDGER_PACKET_FENCE')
   actual_writer=json.loads(wp.read_text());validate_fence_schema(actual_writer)
   need(a['pid']==writer['pid'] and a['proc_start_ticks']==writer['proc_start_ticks'] and read(producer['task_packet'],root)==packet and actual_writer==writer,'CUP_SIGNED_WRITER_IDENTITY_DRIFT')
   nav_ids=observed
 ctx.update(packet=packet,producer=producer,guard=guard,full_exit_guard=full_guard)
 guard()
 return ctx
def original_producer_shape(p):
 need(isinstance(p,dict) and set(p)=={'task_id','index','outer','packet','terminal_receipt'} and type(p['task_id']) is str and p['task_id'],'CUP_ORIGINAL_PRODUCER_PACKET_FIELD_SCHEMA')
 for key in ['index','outer','packet','terminal_receipt']:
  v=p[key]
  need(isinstance(v,dict) and set(v)=={'path','bytes','sha256'} and type(v['path']) is str and type(v['bytes']) is int and v['bytes']>=0 and type(v['sha256']) is str and len(v['sha256'])==64 and all(c in '0123456789abcdef' for c in v['sha256']),'CUP_ORIGINAL_PRODUCER_TYPED_EXACTREF')
 return True
def verify_baselines(contract,root=ROOT,tick=lambda:None):
 """Metadata + original frozen binding validation. Never changes original files."""
 from chaoyang.pick_place_1008 import pipeline,mask_seeds
 from chaoyang.pick_place_1008.stage_reuse import _validate
 queue=read(contract['original_queue'],root)
 state=json.loads(path(root/'docs/governance/LONG_HORIZON_TASK_STATE.json',root).read_text())
 reg=json.loads(path(root/'tasks/current/INDEX.json',root).read_text())
 ls={x['task_id']:x for x in state['tasks']};gs={x['task_id']:x for x in reg['task_packets']};indices={};packets={}
 for p in contract['original_producers']:
  original_producer_shape(p);tick();tid=p['task_id'];ix=read(p['index'],root);o=read(p['outer'],root);pkt=read(p['packet'],root);term=read(p['terminal_receipt'],root);l=ls[tid];g=gs[tid]
  need(o['schema_version']=='pick-place-1008-execution-result-v1' and term['schema_version']=='pick-place-1008-receipt-v1','CUP_UPSTREAM_OFFICIAL_SCHEMA')
  need(o['task_id']==pkt['task_id']==term['task_id']==tid and l['status']==o['status']==term['status']=='PASSED','CUP_UPSTREAM_PASSED')
  need(type(o['worker_returncode']) is int and o['worker_returncode']==0 and o['validation_errors']==[] and o['resource_wait'] is False and type(o['algorithm_attempts_consumed']) is int and o['algorithm_attempts_consumed']==1,'CUP_UPSTREAM_OFFICIAL_COMPLETE_ATTEMPT')
  need(ix['status']=='COMPLETE' and ix['sessions_requested']==ix['sessions_complete'] and all(x['status']=='COMPLETE' for x in ix['sessions']),'CUP_UPSTREAM_GLOBAL_COMPLETE')
  need(o['required_outputs']['candidate/INDEX.json']==p['index'] and read(o['stage_result'],root)==ix,'CUP_UPSTREAM_INDEX_RESULT')
  need(l['result']==g['result']==term['result']==p['outer'] and l['terminal_receipt']==g['terminal_receipt']==p['terminal_receipt'],'CUP_UPSTREAM_TERMINAL')
  need(l['pid'] is None and l['proc_start_ticks'] is None and g['execution_allowed'] is False,'CUP_UPSTREAM_CLOSED')
  packetpath=Path(g['packet_path']);packetpath=packetpath if packetpath.is_absolute() else root/packetpath
  need(str(packetpath)==p['packet']['path'] and g['packet_sha256']==p['packet']['sha256'] and o['task_packet']==l['task_packet']==p['packet'],'CUP_UPSTREAM_PACKET')
  need(o['run_signature']==pkt['run_signature']==l['run_signature'],'CUP_UPSTREAM_SIGNATURE')
  row=next(x for x in [queue['source']]+queue['gates']+queue['records'] if x['task_id']==tid)
  need(pkt['inputs']['spec']==row['spec'],'CUP_ORIGINAL_QUEUE_SPEC');pin(row['spec'],root)
  need(runtime_shape(pkt['runtime_code'],root)==runtime_shape(contract['original_runtime_93'],root),'CUP_ORIGINAL93')
  indices[tid]=ix;packets[tid]=pkt
 need([c['session_id'] for c in contract['cases']]==SCOPE and sum(c['native_frames'] for c in contract['cases'])==760,'CUP_CONTRACT_DEV760')
 need(indices[queue['source']['task_id']]['sessions_requested']==indices[queue['source']['task_id']]['sessions_complete']==199 and indices[queue['source']['task_id']]['frames_complete']==29322,'CUP_ORIGINAL_SOURCE199_29322')
 manifest=read(contract['manifest'],root);config=contract['camera_config'];pin(config,root)
 for c in contract['cases']:
  tick();sid=c['session_id'];n=c['native_frames'];ix=indices[c['original_task_id']];raw=next(x for x in manifest['sessions'] if x['session_id']==sid)
  row=next(x for x in ix['sessions'] if x['session_id']==sid)
  need(row['status']=='COMPLETE' and row['raw_frame_count']==raw['raw_frame_count']==n and ix['manifest']==contract['manifest'] and ix['camera_config']==config,'CUP_SOURCE_NATIVE_MANIFEST_CAMERA')
  seed_index=ix['mask_seed_index'];pin(seed_index,root);seed=mask_seeds.seed_reference(Path(seed_index['path']),sid)
  need(seed==c['visible_seed_ref'],'CUP_SEED_EXACT')
  for stage in ['source','motion','gpu']:
   tick();pr=c['old_stage_receipts'][stage];rc=read(pr,root);owner=next(p for p in contract['original_producers'] if p['task_id']==c['stage_owner_task_ids'][stage])
   expected_owner=dict(task_id=owner['task_id'],task_packet=owner['packet'],run_signature=packets[owner['task_id']]['run_signature'],result_path=owner['outer']['path'])
   need(rc['producer']==expected_owner and row['stage_receipts'][stage]==pr,'CUP_STAGE_OWNER')
   upstream={} if stage=='source' else {'source':c['old_stage_receipts']['source']}
   expected=pipeline.binding(root,stage,raw,config,upstream,seed if stage=='gpu' else None)
   need(expected==rc['binding'],'CUP_FRESH_ORIGINAL_BINDING:'+stage)
   folder,pref=_validate(root,Path(pr['path']).parents[2],sid,stage,expected,pipeline.STAGE_FILES[stage])
   need(folder is not None and pref==pr,'CUP_STRICT_OLD_STAGE_VALIDATE:'+stage)
   if stage=='gpu':
    need(all(rc['files'][k]==v for k,v in c['old_GPU_inputs'].items()),'CUP_GPU_ACTUAL_INPUT_FILES')
   if stage=='motion':need(rc['files']['actions.npz']==c['frozen_identity_refs']['actions'],'CUP_OLD_SAME_ACTIONS')
  for v in c['frozen_identity_refs'].values():pin(v,root)
 return True

def validate_fence_schema(writer):
 need(set(writer)=={'pid','proc_start_ticks','executor_epoch','fencing_token','scope'},'CUP_WRITER_EXACT_FIELDS')
 need(writer['scope']==TASK and all(type(writer[k]) is int for k in ['pid','proc_start_ticks','executor_epoch']) and writer['pid']>0 and writer['proc_start_ticks']>=0 and writer['executor_epoch']>=1,'CUP_FENCE_TYPED_IDENTITY')
 need(type(writer['fencing_token']) is str and len(writer['fencing_token'])==48 and all(c in '0123456789abcdef' for c in writer['fencing_token']),'CUP_EXPLICIT_FENCING_TOKEN')


CLEAN_SCHEMA='pick-place-q-frozen-cup-conflict-raw-restore-masks-candidate-v1'
def verify_clean_parent(contract,root=ROOT,tick=lambda:None,verify_large=True):
 """Independent official current closure; no incomplete/FAILED Clean input."""
 s=contract.get('actual_clean_parent')
 need(isinstance(s,dict) and all(s.get(k) is not None for k in ['outer','packet','terminal_receipt','index','independent_collector','Root_strong_proof']),'FR_MANDATORY_ACTUAL_CLEAN_REFS_MISSING')
 o,p,term,ix,col=[read(s[k],root) for k in ['outer','packet','terminal_receipt','index','independent_collector']]
 proof=read(s['Root_strong_proof'],root)
 need(proof['schema']=='Root_actual_cleanV3_current_closed_strong_parent_CPU_proof_v1' and proof['status']=='GLOBAL_PASSED_DEV760_CURRENT_FRESH_STAGE_BINDINGS_AND_SAVED_COLLECTOR_PIXEL_POLICY_PASS','FR_ROOT_PARENT_PROOF_SCHEMA')
 for k in ['outer','packet','terminal_receipt','index','independent_collector']:need(proof[k]==s[k],'FR_PARENT_PROOF_PIN:'+k)
 tid=p['task_id'];need(tid==contract['actual_clean_task_id'],'FR_UNIQUE_CLEAN_TASK')
 need(o['task_id']==term['task_id']==ix['task_id']==col['task_id']==tid and o['status']==term['status']==col['status']=='PASSED' and ix['status']=='COMPLETE','FR_ACTUAL_CLEAN_GLOBAL_PASSED')
 need(type(o['worker_returncode']) is int and o['worker_returncode']==0 and o['validation_errors']==[] and o['resource_wait'] is False,'FR_CLEAN_RC0')
 need(term['schema_version']=='pick-place-1008-receipt-v1' and term['result']==s['outer'] and o['task_packet']==col['task_packet']==s['packet'] and o['run_signature']==p['run_signature']==col['run_signature'],'FR_CLEAN_OFFICIAL_IDENTITY')
 need(o['required_outputs']['candidate/INDEX.json']==s['index'] and read(o['stage_result'],root)==ix and o['independent_collector']==s['independent_collector'] and col['candidate_index']==s['index'],'FR_CLEAN_RESULT_COLLECTOR')
 need(ix['sessions_requested']==ix['sessions_complete']==4 and ix['frames_requested']==ix['frames_complete']==760 and [r['session_id'] for r in ix['sessions']]==SCOPE,'FR_CLEAN760')
 need(ix['q_changed'] is False and ix['Contact_Render_reused_as_new_candidate'] is False and type(ix['model_FK_Render_calls']) is int and ix['model_FK_Render_calls']==0 and ix['physical_contact_ground_truth'] is False,'FR_CLEAN_TRUTH_DOMAIN')
 state=json.loads(path(root/'docs/governance/LONG_HORIZON_TASK_STATE.json',root).read_text());reg=json.loads(path(root/'tasks/current/INDEX.json',root).read_text())
 l=next(x for x in state['tasks'] if x['task_id']==tid);g=next(x for x in reg['task_packets'] if x['task_id']==tid)
 need(l['status']=='PASSED' and l['pid'] is None and l['proc_start_ticks'] is None and g['execution_allowed'] is False and l['result']==g['result']==s['outer'] and l['terminal_receipt']==g['terminal_receipt']==s['terminal_receipt'],'FR_CLEAN_CURRENT_CLOSED')
 gp=Path(g['packet_path']);gp=gp if gp.is_absolute() else root/gp
 need(str(gp)==s['packet']['path'] and g['packet_sha256']==s['packet']['sha256'] and l['run_signature']==p['run_signature'],'FR_CLEAN_CURRENT_PACKET')
 for c,r in zip(contract['cases'],ix['sessions']):
  tick();n=c['native_frames'];need(r['status']=='COMPLETE' and r['raw_frame_count']==r['fully_audited_native_frames']==n and r['files']==c['actual_clean_files'] and r['stage_receipt']==c['actual_clean_stage'],'FR_CLEAN_CASE_N_FILES')
  rc=read(c['actual_clean_stage'],root)
  need(rc['schema']=='pick_place_q_frozen_visual_clean_postprocess_sealed_stage_v1' and rc['status']=='COMPLETE' and rc['files']==c['actual_clean_files'] and rc['producer']==ix['producer'],'FR_CLEAN_NEW_STAGE')
  need(rc['binding']['old_stage_receipts']==c['old_stage_receipts'] and rc['binding']['original_GPU_inputs']==c['old_GPU_inputs'] and rc['binding']['source_q_camera_seed_frozen_refs']==c['frozen_identity_refs'],'FR_CLEAN_FROZEN_SOURCE_MOTION_GPU')
  meta=read(c['actual_clean_files']['candidate_masks.json'],root)
  need(meta['schema_version']==CLEAN_SCHEMA and meta['bitorder']=='big' and meta['shape']==[n,480,640] and meta['status']=='COMPLETE' and meta['Render_Contact_status']=='NOT_PRODUCED_REQUIRE_SEPARATE_REGISTERED_FRESH_TASK','FR_NEW_MASK_POLICY')
  if verify_large:
   for v in c['actual_clean_files'].values():pin(v,root,tick=tick)
 return True


def asset_shape(contract,root=ROOT):
 data=contract.get('robot_asset_input_refs');io=contract.get('fixed_existing_runtime_code')
 need(isinstance(data,list) and len(data)==69 and len(runtime_shape(data,root))==69 and data==sorted(data,key=lambda r:r['path']),'FR_SIGNED_UNIQUE_SORTED_69_STATIC_ASSETS')
 need(all(type(r['bytes']) is int and r['bytes']>0 and type(r['sha256']) is str and len(r['sha256'])==64 and all(x in '0123456789abcdef' for x in r['sha256']) and path(r['path'],root).is_relative_to(root/'assets/robot') for r in data),'FR_ASSET_TYPED_REF_DOMAIN')
 need(isinstance(io,list) and len(io)==1 and io[0]=={'path':str(root/'src/chaoyang/ops/immutable_artifact_io.py'),'bytes':2725,'sha256':'c0cdcea8b541083de65b2ccd7d1011305e2000bbf49694eef601599d7fbcb5de'},'FR_FIXED_IO_CODE1_EXACT')
 return data,io
def verify_robot_assets(contract,root=ROOT,tick=lambda:None,verify=True):
 data,io=asset_shape(contract,root)
 if not verify:return True
 import xml.etree.ElementTree as ET
 for r in data+io:pin(r,root,tick=tick)
 robot_pin=root/'assets/robot/ROBOT_ASSET_PIN.json';pin_ref=next(r for r in data if r['path']==str(robot_pin));identity=read(pin_ref,root)
 need(identity['status']=='T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION','FR_STATIC_ROBOT_IDENTITY_PIN_STATUS')
 entries={r['path']:r for r in identity['files']};urdfs=identity['urdfs'];need(len(urdfs)==3,'FR_STATIC_THREE_URDF')
 selected={str(robot_pin)};roots=set();meshes=set();visuals=0
 for row in urdfs:
  tick();urdf=path(root/row['path'],root);selected.add(str(urdf));r=next(v for v in data if v['path']==str(urdf));entry=entries[row['path']]
  need(r['bytes']==entry['bytes'] and r['sha256']==entry['sha256'],'FR_URDF_IDENTITY_TABLE')
  xml=ET.fromstring(urdf.read_bytes());links={v.get('name') for v in xml.findall('link')};children={j.find('child').get('link') for j in xml.findall('joint')};one=links-children;need(len(one)==1,'FR_URDF_ONE_ROOT');roots|=one
  for visual in xml.findall('link/visual'):
   mesh=visual.find('geometry/mesh');need(mesh is not None and mesh.get('filename'),'FR_STATIC_VISUAL_MESH');name=mesh.get('filename')
   if name.startswith('package://marvin_description/'):target=root/'assets/robot/tianji/marvin_description'/name.removeprefix('package://marvin_description/')
   elif name.startswith('package://'):raise ValueError('FR_UNKNOWN_MESH_PACKAGE_URI')
   else:target=urdf.parent/name
   need(not target.is_symlink(),'FR_STATIC_MESH_SYMLINK');target=path(target.resolve(strict=True),root);need(target.is_relative_to(root/'assets/robot'),'FR_MESH_ASSET_DOMAIN');rel=target.relative_to(root).as_posix();v=next(v for v in data if v['path']==str(target));entry=entries[rel]
   need(v['bytes']==entry['bytes'] and v['sha256']==entry['sha256'],'FR_MESH_IDENTITY_TABLE');meshes.add(str(target));selected.add(str(target));visuals+=1
 selected|={str(root/'assets/robot/V2相机固定法兰PRO版本.STL'),str(root/'assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json')}
 need(roots=={'base_link','hand_l_base_link','hand_r_base_link'} and len(meshes)==visuals==63 and len(selected)==69 and selected=={r['path'] for r in data},'FR_ACTUAL_STATIC_XML_69_ASSET_CLOSURE')
 return True
def validate_live_robot_assets(robot,packet,contract,root=ROOT,tick=lambda:None):
 data,io=asset_shape(contract,root);models=(robot.assets.tianji,robot.assets.left_hand,robot.assets.right_hand)
 from chaoyang.exact78_baseline.robot_model import FLANGE_MESH,MOUNT_PROXY
 selected={Path(m.path) for m in models};selected.update(Path(v.mesh_path) for m in models for v in m.visuals);selected.update((root/FLANGE_MESH,root/MOUNT_PROXY))
 actual=[ref(p,root,tick=tick) for p in sorted(selected)];expected=[r for r in data if r['path']!=str(root/'assets/robot/ROBOT_ASSET_PIN.json')]
 need(len(actual)==68 and actual==expected and robot.assets.asset_pin_record==next(r for r in data if r['path']==str(root/'assets/robot/ROBOT_ASSET_PIN.json')),'FR_ACTUAL_ROBOT68_AND_PIN_IDENTITY')
 need(all(r in packet['inputs']['files'] for r in data) and all(r in packet['runtime_code'] for r in io),'FR_ROBOT_INPUT69_IO1_NOT_PACKET_BOUND')
 return True
