from __future__ import annotations
import copy,hashlib,json,os,shlex,time
from pathlib import Path
from chaoyang.governance.candidate_optimization_family_v4 import descriptor,published_current,exact
ROOT=Path('/path/to/chaoyang')
FAMILY='pick_place_contact_role_evidence_sidecar_candidate_v1'
OPERATION='run_pick_place_contact_role_evidence_sidecar_candidate_v1'
TASK='exact78_pick_place_contact_role_evidence_dev4_candidate_revision1_20261010'
SCOPE=['001','003','008','100'];NATIVE=760
BUDGETS=dict(cpu_threads_total=1,gpu_count=0,model_calls=0,render_calls=0,session_attempts_max=1,training_steps=0,wall_seconds_max=900,workers_max=1)
SPEC_KEYS={'schema_version','task_id','operation','sessions','native_frames','contract','budgets'}
CANONICAL='src/chaoyang/pick_place_contact_role_evidence_sidecar_v1/'
OWN_FILES=[CANONICAL+x for x in ['__init__.py','roles.py','packed_reader.py','contract.py','driver.py','collector.py']]+['src/chaoyang/ops/'+OPERATION+'.py','src/chaoyang/governance/execute_pick_place_contact_role_evidence_sidecar_v1.py']
SHARED_FILES=['src/chaoyang/governance/candidate_optimization_family_v4.py','src/chaoyang/governance/current_candidate_optimization_contracts_v4.py','src/chaoyang/governance/common_candidate_optimization_v5.py']
def need(v,s):
 if not v:raise ValueError(s)
def path(p,root=ROOT,nas=False):
 p=Path(p);base=Path('/path/to/private_storage/datasets/ego/pick_place_1008') if nas else root
 need(p.is_absolute() and '..' not in p.parts and p.is_relative_to(base),'ROLE_PROJECT_OR_FROZEN_NAS_ONLY')
 for x in (p,*p.parents):
  need(not x.is_symlink(),'ROLE_SYMLINK_REJECTED')
  if x==base:break
 return p
def pin(v,root=ROOT,nas=False):
 need(isinstance(v,dict) and set(v)=={'path','bytes','sha256'} and type(v['bytes']) is int and v['bytes']>=0,'ROLE_EXACTREF_SCHEMA')
 p=path(v['path'],root,nas);s=p.stat();before=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);h=hashlib.sha256();count=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b);count+=len(b)
 z=p.stat();after=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 need(count==v['bytes'] and h.hexdigest()==v['sha256'] and before==after,'ROLE_SHA_OR_SIZE_DRIFT:'+str(p))
 return p
def read(v,root=ROOT,nas=False):return json.loads(pin(v,root,nas).read_text())
def ref(p,root=ROOT):
 p=path(p,root);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def runtime_shape(rows,root=ROOT):
 need(isinstance(rows,list),'ROLE_RUNTIME_LIST')
 out={}
 for r in rows:
  need(set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int,'ROLE_RUNTIME_REF_SCHEMA')
  p=path(r['path'],root);need(str(p) not in out,'ROLE_DUPLICATE_RUNTIME_PATH');out[str(p)]=r
 return out
def spec_shape(spec):
 need(set(spec)==SPEC_KEYS and spec['schema_version']=='pick_place_contact_role_sidecar_spec_v1','ROLE_SPEC_UNKNOWN_KEYS')
 need(spec['task_id']==TASK and spec['operation']==OPERATION and spec['sessions']==SCOPE and type(spec['native_frames']) is int and spec['native_frames']==NATIVE,'ROLE_FIXED_DEV_SCOPE')
 need(spec['budgets']==BUDGETS and all(type(v) is int for v in spec['budgets'].values()),'ROLE_TYPED_FINITE_BUDGETS')
def plan_shape(plan,spec,root=ROOT,verify=True):
 spec_shape(spec)
 need(plan['schema']=='root_planned_finite_contact_role_sidecar_operation_v1' and plan['family_key']==FAMILY and plan['operations']==[OPERATION] and plan['operation']==OPERATION,'ROLE_PLAN_FAMILY_OP')
 need(plan['task_id']==TASK and plan['sessions']==SCOPE and plan['native_frames']==NATIVE and plan['budgets']==BUDGETS,'ROLE_PLAN_SCOPE_BUDGET')
 rows=runtime_shape(plan['candidate_runtime_code'],root)
 need(set(rows)=={str(root/f) for f in OWN_FILES},'ROLE_EXACT_UNIQUE_EIGHT_CANONICAL_FILES')
 shared=runtime_shape(plan['maintenance_governance_runtime_code'],root)
 need(set(shared)=={str(root/f) for f in SHARED_FILES},'ROLE_EXACT_SHARED_SUCCESSOR_THREE')
 if plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED':
  need(isinstance(plan.get('root_fresh_actor_inventory'),dict),'ROLE_ROOT_COMPLETE_CURRENT_OWNER_INVENTORY_REQUIRED')
 need(plan['no_original_contact_source_q_write'] is True and plan['no_HOLD_fit'] is True and plan['model_calls']==plan['FK_calls']==plan['render_calls']==0,'ROLE_TRUTH_OR_EXECUTION_DOMAIN')
 if verify:
  for v in list(rows.values())+list(shared.values()):pin(v,root)
def expected_runtime(ctx,root=ROOT,verify=False):
 rows=ctx['contract']['original_runtime_93'];need(len(runtime_shape(rows,root))==93,'ROLE_OLD93_EXACT')
 out=runtime_shape(rows+ctx['plan']['candidate_runtime_code']+ctx['plan']['maintenance_governance_runtime_code'],root)
 need(len(out)==104,'ROLE_OLD93_PLUS_EIGHT_PLUS_SHARED3')
 if verify:
  for v in out.values():pin(v,root)
 return list(out.values())
def approved_context(spec_path,root=ROOT):
 authority=published_current(root);route=authority['special_status_contracts'].get(FAMILY)
 need(isinstance(route,dict),'ROLE_ROOT_FAMILY_NOT_PUBLISHED')
 d=descriptor(route.get('root_registered_descriptor'),root)
 need(d['family_key']==FAMILY and d['operations']==[OPERATION] and {k:v for k,v in route.items() if k!='root_registered_descriptor'}==d['route_value'],'ROLE_PUBLISHED_DESCRIPTOR_DRIFT')
 allowed=route.get('maintenance_render_operation_allowlist',{}).get(OPERATION)
 need(isinstance(allowed,dict) and allowed==d['maintenance_operation_allowlist'].get(OPERATION) and allowed.get('execution_allowed') is True,'ROLE_ROOT_EXPLICIT_ALLOWLIST')
 plan=read(allowed['plan'],root);need(plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED','ROLE_ROOT_PLAN_PENDING')
 need(str(path(spec_path,root))==plan['approved_spec']['path'],'ROLE_UNAPPROVED_SPEC_PATH')
 publication=read(d['root_preapproval'],root)
 need(publication['release_actor_inventory']==plan['root_fresh_actor_inventory'] and publication['publication_required_actor_labels']==plan['publication_required_actor_labels'],'ROLE_ROOT_RELEASE_OWNER_INVENTORY_TIE')
 spec=read(plan['approved_spec'],root);plan_shape(plan,spec,root,verify=False)
 contract=read(spec['contract'],root)
 need(contract['schema']=='pick_place_original_DEV4_contact_role_sidecar_contract_v1' and contract['operation']==OPERATION and contract['sessions']==SCOPE and contract['native_frames']==NATIVE,'ROLE_CONTRACT_SCOPE')
 need(contract['candidate_runtime_code']==plan['candidate_runtime_code'],'ROLE_CONTRACT_CODE_PIN')
 return dict(plan=plan,spec=spec,contract=contract,plan_ref=allowed['plan'],descriptor=d)
def command(spec_ref):return 'PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run '+shlex.join([OPERATION,'--role-spec',spec_ref['path']])
def signature(inputs,runtime,config,cmd):
 from chaoyang.governance.common import canonical_bytes
 return hashlib.sha256(canonical_bytes(dict(inputs=inputs,runtime_code=runtime,config=config,command=cmd,schema='pick-place-contact-role-sidecar-task-v1'))).hexdigest()
def validate_packet_context(packet,ctx,root=ROOT):
 need(packet['task_id']==TASK and packet['execution_mode']=='EXACT78_CANDIDATE','ROLE_PACKET_FIXED_MODE')
 need(packet['budgets']==packet['config']['budgets']==BUDGETS and all(type(v) is int for v in packet['budgets'].values()),'ROLE_PACKET_TYPED_BUDGETS')
 need(packet['inputs']==dict(spec=ctx['plan']['approved_spec'],files=[ctx['spec']['contract']],maintenance_render_plan=ctx['plan_ref']),'ROLE_PACKET_INPUTS_EXACT')
 need(runtime_shape(packet['runtime_code'],root)==runtime_shape(expected_runtime(ctx,root),root),'ROLE_PACKET_RUNTIME_EXACT')
 need(packet['commands']==[command(ctx['plan']['approved_spec'])],'ROLE_PACKET_ONE_EXACT_COMMAND_NO_EXTRA')
 need(packet['run_signature']==signature(packet['inputs'],packet['runtime_code'],packet['config'],packet['commands'][0]),'ROLE_PACKET_SIGNATURE')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'ROLE_PACKET_REQUIRED_OUTPUTS')
 return True
def require_registered_context(spec_path,argv,root=ROOT):
 need(argv==['--role-spec',str(path(spec_path,root))],'ROLE_ONLY_ONE_SPEC_NO_EXTRA')
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
 authorize_operation_entry(OPERATION,argv)
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 producer=current_producer(root);need(producer['task_id']==TASK,'ROLE_EXPLICIT_CURRENT_TASK')
 packet=read(producer['task_packet'],root);ctx=approved_context(spec_path,root);validate_packet_context(packet,ctx,root)
 validate_fence_schema(packet['writer'])
 ctx.update(packet=packet,producer=producer)
 return ctx
def baseline_guard(contract,root=ROOT):
 queue=read(contract['original_queue'],root)
 led=json.loads(path(root/'docs/governance/LONG_HORIZON_TASK_STATE.json',root).read_text());reg=json.loads(path(root/'tasks/current/INDEX.json',root).read_text())
 ls={x['task_id']:x for x in led['tasks']};rs={x['task_id']:x for x in reg['task_packets']};out={}
 for p in contract['original_producers']:
  ix=read(p['index'],root);outer=read(p['outer'],root);packet=read(p['packet'],root);term=read(p['terminal_receipt'],root)
  tid=p['task_id'];l=ls[tid];g=rs[tid]
  need(outer['task_id']==packet['task_id']==term['task_id']==tid and outer['status']==l['status']==term['status']=='PASSED','ROLE_UPSTREAM_OFFICIAL_PASSED')
  need(type(outer['worker_returncode']) is int and outer['worker_returncode']==0 and outer['validation_errors']==[],'ROLE_UPSTREAM_WORKER_ZERO')
  need(ix['status']=='COMPLETE' and type(ix['sessions_requested']) is int and type(ix['sessions_complete']) is int and type(ix['frames_complete']) is int and ix['sessions_requested']==ix['sessions_complete']==len(p['expected_sessions']) and ix['frames_complete']==p['expected_native_frames'] and ix['stage']==p['expected_stage'] and [r['session_id'] for r in ix['sessions']]==p['expected_sessions'] and len(set(p['expected_sessions']))==len(p['expected_sessions']) and all(r['status']=='COMPLETE' and type(r['raw_frame_count']) is int for r in ix['sessions']) and sum(r['raw_frame_count'] for r in ix['sessions'])==p['expected_native_frames'] and outer['required_outputs']['candidate/INDEX.json']==p['index'] and read(outer['stage_result'],root)==ix,'ROLE_UPSTREAM_GLOBAL_COMPLETE')
  need(term['schema_version']=='pick-place-1008-receipt-v1' and term['result']==p['outer'] and l['result']==g['result']==p['outer'] and l['terminal_receipt']==g['terminal_receipt']==p['terminal_receipt'],'ROLE_UPSTREAM_TERMINAL_PIN')
  need(l['pid'] is None and l['proc_start_ticks'] is None and g['execution_allowed'] is False and outer['run_signature']==packet['run_signature']==l['run_signature'],'ROLE_UPSTREAM_CLOSED_SIGNATURE')
  pp=Path(g['packet_path']);pp=pp if pp.is_absolute() else root/pp
  need(str(pp)==p['packet']['path'] and g['packet_sha256']==p['packet']['sha256'] and outer['task_packet']==p['packet'],'ROLE_UPSTREAM_REGISTRY_PACKET')
  row=next(x for x in [queue['source']]+queue['gates']+queue['records'] if x['task_id']==tid);need(packet['inputs']['spec']==row['spec'],'ROLE_UPSTREAM_QUEUE_SPEC');pin(row['spec'],root)
  need(runtime_shape(packet['runtime_code'],root)==runtime_shape(contract['original_runtime_93'],root),'ROLE_UPSTREAM_OLD93')
  out[tid]=ix
 need(len({x['session_id'] for x in contract['cases']})==4 and [x['session_id'] for x in contract['cases']]==SCOPE and sum(x['native_frames'] for x in contract['cases'])==NATIVE,'ROLE_BASELINE_UNIQUE_SCOPE')
 for c in contract['cases']:
  sid=c['session_id'];n=c['native_frames'];need(type(n) is int and n>0,'ROLE_NATIVE_INT')
  for stage,v in c['stages'].items():
   rc=read(v['receipt'],root);need(rc['binding']==v['expected_original_binding'],'ROLE_ORIGINAL_STAGE_BINDING_SNAPSHOT');ix=out[v['producer_task_id']];row=next(x for x in ix['sessions'] if x['session_id']==sid)
   need(row['status']=='COMPLETE' and row['raw_frame_count']==n and row['stage_receipts'][stage]==v['receipt'] and rc['status']=='COMPLETE' and rc['binding']['stage']==stage,'ROLE_ALL_FOUR_STAGE_SESSION')
   if stage=='gpu':need(rc['files']['masks.npz']==c['masks_NPZ'],'ROLE_OFFICIAL_MASK_PIN')
   if stage=='render':
    need(rc['files']['contact_events.json']==c['old_contact'] and rc['files']['frame_states.jsonl']==c['frame_states'],'ROLE_OLD_CONTACT_FLAGS_BINDING')
   if stage=='source':
    need(rc['files']['source/SESSION_INPUT.json']==c['source_session_input'],'ROLE_SOURCE_INPUT_BINDING')
    si=read(c['source_session_input'],root);need(si['source_refs']['raw/vst.qpc.ts.jsonl']==c['original_QPC'],'ROLE_ORIGINAL_QPC_SOURCE')
 return True

def validate_fence_schema(writer):
 need(set(writer)=={'pid','proc_start_ticks','executor_epoch','fencing_token','scope'},'ROLE_WRITER_EXACT_FIELDS')
 need(writer['scope']==TASK and all(type(writer[k]) is int for k in ['pid','proc_start_ticks','executor_epoch']) and writer['pid']>0 and writer['proc_start_ticks']>=0 and writer['executor_epoch']>=1,'ROLE_FENCE_TYPED_IDENTITY')
 need(type(writer['fencing_token']) is str and len(writer['fencing_token'])==48 and all(c in '0123456789abcdef' for c in writer['fencing_token']),'ROLE_EXPLICIT_FENCING_TOKEN')

def full_owner_inventory(plan,root=ROOT):
 need(isinstance(plan.get('root_fresh_actor_inventory'),dict),'ROLE_ROOT_OWNER_INVENTORY_PENDING')
 inventory=read(plan['root_fresh_actor_inventory'],root)
 need(inventory['schema']=='root_actual_finite_optimization_family_publish_idle_actor_inventory_v1' and inventory['status']=='ROOT_ACTUAL_READONLY_INVENTORY_RECORDED' and inventory['family_key']==FAMILY and inventory['operations']==[OPERATION],'ROLE_ROOT_CURRENT_OWNER_INVENTORY_SCHEMA')
 rows=inventory['actors'];labels=plan['publication_required_actor_labels']
 need(isinstance(labels,list) and len(labels)>=65 and len(labels)==len(set(labels)) and len(rows)==len(labels) and {r['actor_label'] for r in rows}==set(labels),'ROLE_CURRENT_COMPLETE_OWNER_LABELS_NOT_ORIGINAL23_ONLY')
 from chaoyang.governance.common import process_identity
 for row in rows:
  need(set(row)=={'actor_label','pid','proc_start_ticks'} and type(row['pid']) is int and type(row['proc_start_ticks']) is int and row['pid']>0 and row['proc_start_ticks']>0,'ROLE_OWNER_TYPED_SIGNED_IDENTITY')
  actual=process_identity(row['pid']);need(not(actual['alive'] and actual['start_ticks']==row['proc_start_ticks']),'ROLE_OWNER_ACTOR_LIVE')
 return dict(verified_root_current_owner_actors=len(rows),inventory=plan['root_fresh_actor_inventory'],original23_actor_subset_separate=True)
