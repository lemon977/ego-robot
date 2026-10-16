from __future__ import annotations
import copy,hashlib,json,os,shlex,time
from pathlib import Path
from chaoyang.governance.candidate_optimization_family_v6 import descriptor,published_current,exact
ROOT=Path('/path/to/chaoyang')
FAMILY='pick_place_observed_cup_conflict_restore_clean_candidate_v2'
OPERATION='run_pick_place_observed_cup_conflict_restore_clean_candidate_v2'
TASK='exact78_pick_place_observed_cup_conflict_restore_clean_dev4_cpu_candidate_revision2_20261010'
SCOPE=['001','003','008','100'];NATIVE=760
BUDGETS=dict(cpu_threads_total=1,gpu_count=0,model_calls=0,render_calls=0,session_attempts_max=1,training_steps=0,wall_seconds_max=900,workers_max=1)
SPEC_KEYS={'schema_version','task_id','operation','sessions','native_frames','contract','budgets'}
CANONICAL='src/chaoyang/pick_place_observed_cup_restore_clean_v2/'
OWN_FILES=[CANONICAL+x for x in ['__init__.py','helper.py','stream.py','contract.py','driver.py','collector.py']]+['src/chaoyang/ops/'+OPERATION+'.py','src/chaoyang/governance/execute_pick_place_observed_cup_restore_clean_v2.py']
SHARED_FILES=['src/chaoyang/governance/candidate_optimization_family_v6.py','src/chaoyang/governance/current_candidate_optimization_contracts_v6.py','src/chaoyang/governance/common_candidate_optimization_v7.py']
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
def spec_shape(spec):
 need(set(spec)==SPEC_KEYS and spec['schema_version']=='pick_place_observed_cup_restore_clean_spec_v2','CUP_SPEC_UNKNOWN_KEYS')
 need(spec['task_id']==TASK and spec['operation']==OPERATION and spec['sessions']==SCOPE and type(spec['native_frames']) is int and spec['native_frames']==NATIVE,'CUP_FIXED_DEV_SCOPE')
 need(spec['budgets']==BUDGETS and all(type(v) is int for v in spec['budgets'].values()),'CUP_TYPED_FINITE_BUDGETS')
def plan_shape(plan,spec,root=ROOT,verify=True):
 spec_shape(spec)
 need(plan['schema']=='root_planned_finite_observed_cup_restore_clean_operation_v2' and plan['family_key']==FAMILY and plan['operations']==[OPERATION],'CUP_PLAN_FAMILY_OP')
 need(plan['task_id']==TASK and plan['sessions']==SCOPE and type(plan['native_frames']) is int and plan['native_frames']==NATIVE and plan['budgets']==BUDGETS and all(type(v) is int for v in plan['budgets'].values()),'CUP_PLAN_SCOPE_BUDGET')
 rows=runtime_shape(plan['candidate_runtime_code'],root)
 need(set(rows)=={str(root/f) for f in OWN_FILES},'CUP_EXACT_UNIQUE_EIGHT_CANONICAL_FILES')
 shared=runtime_shape(plan['maintenance_governance_runtime_code'],root)
 need(set(shared)=={str(root/f) for f in SHARED_FILES},'CUP_EXACT_SHARED_SUCCESSOR_THREE')
 need(plan['no_original_contact_source_q_write'] is True and plan['no_HOLD_fit'] is True and all(type(plan[k]) is int and plan[k]==0 for k in ['model_calls','FK_calls','render_calls']),'CUP_TRUTH_OR_EXECUTION_DOMAIN')
 if verify:
  for v in list(rows.values())+list(shared.values()):pin(v,root)
def expected_runtime(ctx,root=ROOT,verify=False):
 rows=ctx['contract']['original_runtime_93'];need(len(runtime_shape(rows,root))==93,'CUP_OLD93_EXACT')
 out=runtime_shape(rows+ctx['plan']['candidate_runtime_code']+ctx['plan']['maintenance_governance_runtime_code'],root)
 need(len(out)==104,'CUP_OLD93_PLUS_EIGHT_PLUS_SHARED3')
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
 need(contract['schema']=='pick_place_original_DEV4_cup_restore_clean_contract_v2' and contract['operation']==OPERATION and contract['sessions']==SCOPE and contract['native_frames']==NATIVE,'CUP_CONTRACT_SCOPE')
 need(contract['candidate_runtime_code']==plan['candidate_runtime_code'],'CUP_CONTRACT_CODE_PIN')
 return dict(plan=plan,spec=spec,contract=contract,plan_ref=allowed['plan'],descriptor=d)
def command(spec_ref):return 'PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run '+shlex.join([OPERATION,'--clean-spec',spec_ref['path']])
def signature(inputs,runtime,config,cmd):
 from chaoyang.governance.common import canonical_bytes
 return hashlib.sha256(canonical_bytes(dict(inputs=inputs,runtime_code=runtime,config=config,command=cmd,schema='pick-place-observed-cup-restore-clean-task-v1'))).hexdigest()
def validate_packet_context(packet,ctx,root=ROOT):
 need(packet['task_id']==TASK and packet['execution_mode']=='EXACT78_CANDIDATE','CUP_PACKET_FIXED_MODE')
 need(packet['budgets']==packet['config']['budgets']==BUDGETS and all(type(v) is int for v in packet['budgets'].values()) and all(type(v) is int for v in packet['config']['budgets'].values()),'CUP_PACKET_TYPED_BUDGETS')
 need(packet['inputs']==dict(spec=ctx['plan']['approved_spec'],files=[ctx['spec']['contract']],maintenance_render_plan=ctx['plan_ref']),'CUP_PACKET_INPUTS_EXACT')
 need(runtime_shape(packet['runtime_code'],root)==runtime_shape(expected_runtime(ctx,root),root),'CUP_PACKET_RUNTIME_EXACT')
 need(packet['commands']==[command(ctx['plan']['approved_spec'])],'CUP_PACKET_ONE_EXACT_COMMAND_NO_EXTRA')
 need(packet['run_signature']==signature(packet['inputs'],packet['runtime_code'],packet['config'],packet['commands'][0]),'CUP_PACKET_SIGNATURE')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'CUP_PACKET_REQUIRED_OUTPUTS')
 return True
def require_registered_context(spec_path,argv,root=ROOT):
 """Full closure at entry/exit; per-task immutable identity cache only between them."""
 need(argv==['--clean-spec',str(path(spec_path,root))],'CUP_ONLY_ONE_SPEC_NO_EXTRA')
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
