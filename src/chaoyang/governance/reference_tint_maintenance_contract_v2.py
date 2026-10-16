"""Strict fixed HOLD maintenance plan/runtime support; no render or model work."""
from pathlib import Path
import importlib.util,sys,json,copy
from chaoyang.governance.candidate_optimization_family_v2 import ROOT,exact,ref,published_current,descriptor,FAMILIES
OPERATION='run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v2'
TASK='exact78_pick_place_reference_tint_render_only_hold007199_maintenance_revision2_20261010'
FAMILY='pick_place_reference_tint_render_only_maintenance_v2'
SCOPE={'007':158,'199':137}
BUDGETS={'cpu_threads_total':1,'gpu_count':0,'model_calls':0,'render_calls':590,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':2400,'workers_max':1}
PACKAGE_BASENAMES={'__init__.py','operation.py','reference_tint_lookup.py','render_robot.py','render_session.py','renderer.py'}
GOVERNANCE_FILES=(
 'src/chaoyang/governance/candidate_optimization_family_v2.py',
 'src/chaoyang/governance/current_candidate_optimization_contracts_v2.py',
 'src/chaoyang/governance/common_candidate_optimization_v3.py',
 'src/chaoyang/governance/reference_tint_maintenance_contract_v2.py',
 'src/chaoyang/governance/execute_reference_tint_maintenance_v2.py',
 'src/chaoyang/governance/collect_reference_tint_maintenance_v2.py',
 'src/chaoyang/ops/'+OPERATION+'.py',
)
def need(x,msg):
 if not x:raise ValueError(msg)
def read(pin,root=ROOT):return json.loads(exact(pin,root).read_text())
def runtime_shape(refs):
 need(isinstance(refs,list) and refs and len({r['path'] for r in refs})==len(refs),'TINT_RUNTIME_UNIQUE_REQUIRED')
 need(all(isinstance(r,dict) and set(r)=={'path','bytes','sha256'} for r in refs),'TINT_RUNTIME_REF_SCHEMA')
 return {r['path']:r for r in refs}
def package_shape(refs,root=ROOT,verify=True):
 runtime_shape(refs);need(len(refs)==6,'TINT_EXACT_SIX_ALGORITHM_FILES')
 paths=[Path(r['path']) for r in refs]
 need(len({p.parent for p in paths})==1 and {p.name for p in paths}==PACKAGE_BASENAMES,'TINT_ACTUAL_ADJACENT_PACKAGE_SIX')
 for p in paths:need(p.is_absolute() and p.is_relative_to(root) and '..' not in p.parts,'TINT_PACKAGE_PROJECT_ONLY')
 if verify:
  for r in refs:exact(r,root)
 return paths[0].parent
def import_consumer(plan,root=ROOT):
 folder=package_shape(plan['candidate_runtime_code'],root)
 name='chaoyang_root_approved_reference_tint_private_v6'
 existing=sys.modules.get(name)
 if existing is not None:
  need(Path(existing.__file__).parent==folder,'TINT_CONSUMER_MODULE_IDENTITY_DRIFT')
 else:
  sp=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
  module=importlib.util.module_from_spec(sp);sys.modules[name]=module;sp.loader.exec_module(module)
 return __import__(name+'.operation',fromlist=['run_operation'])
def plan_shape(plan,spec,root=ROOT,verify=True):
 need(set(spec)=={'schema','operation','task_id','stage','sessions','native_frames','budgets','upstream_policy','contract','fit_parameters','automatic_retry','automatic_AI','Root_adoption','candidate_runtime_code'},'TINT_SPEC_EXACT_SCHEMA')
 need(set(plan)=={'schema','publication_status','operation','task_ids','approved_spec','candidate_runtime_code','fixed_sessions','fixed_native_frames','budgets','zero_fit_parameters','holdout_evaluation_only','automatic_retry','automatic_AI','Root_adoption','collector_schema','maintenance_governance_runtime_code'},'TINT_PLAN_EXACT_SCHEMA')
 need(plan['schema']=='root_planned_finite_reference_tint_render_only_v1' and spec['schema']=='reference_tint_render_only_finite_spec_proposal_v1','TINT_PLAN_SPEC_SCHEMA')
 need(plan['operation']==spec['operation']==OPERATION and plan['task_ids']==[TASK] and spec['task_id']==TASK,'TINT_FIXED_TASK_OPERATION')
 need(spec['sessions']==plan['fixed_sessions']==list(SCOPE) and spec['native_frames']==plan['fixed_native_frames']==295,'TINT_FIXED_TWO_HOLD295_SCOPE')
 need(spec['stage']=='render' and spec['upstream_policy']=='EXACT_GLOBAL_PASSED_SOURCE_MOTION_GPU','TINT_RENDER_ONLY_UPSTREAM')
 for budgets in (spec['budgets'],plan['budgets']):need(budgets==BUDGETS and all(type(v) is int for v in budgets.values()),'TINT_BUDGET_EXACT_INTS')
 need(spec['fit_parameters']=={} and plan['zero_fit_parameters'] is True and plan['holdout_evaluation_only'] is True,'TINT_NO_HOLDOUT_FIT')
 for k in ('automatic_retry','automatic_AI','Root_adoption'):need(plan[k] is False and spec[k] is False,'TINT_NO_AUTO_RETRY_AI_ADOPT')
 need(plan['collector_schema']=='pick_place_1008_reference_tint_maintenance_candidate_index_v1','TINT_COLLECTOR_SCHEMA')
 need(plan['candidate_runtime_code']==spec['candidate_runtime_code'],'TINT_SPEC_PLAN_PACKAGE_BINDING')
 package_shape(plan['candidate_runtime_code'],root,verify)
 gov=runtime_shape(plan['maintenance_governance_runtime_code'])
 need(set(gov)=={str(root/p) for p in GOVERNANCE_FILES},'TINT_EXACT_SEVEN_MAINTENANCE_GOVERNANCE_MODULES')
 need(not(set(gov)&set(runtime_shape(plan['candidate_runtime_code']))),'TINT_NO_GOV_ALGORITHM_DUPLICATION')
 if verify:
  for r in gov.values():exact(r,root)
 return True
def approved_context(spec_path,root=ROOT):
 current=published_current(root);route=current['special_status_contracts'].get(FAMILY)
 need(isinstance(route,dict),'TINT_ROOT_FAMILY_NOT_PUBLISHED')
 d=descriptor(route.get('root_registered_descriptor'),root)
 need(d['family_key']==FAMILY and {k:v for k,v in route.items() if k!='root_registered_descriptor'}==d['route_value'],'TINT_PUBLISHED_FAMILY_DESCRIPTOR_DRIFT')
 allowed=route.get('maintenance_render_operation_allowlist',{}).get(OPERATION)
 need(allowed==d['maintenance_operation_allowlist'].get(OPERATION) and allowed and allowed['execution_allowed'] is True,'TINT_ROOT_ALLOWLIST_REQUIRED')
 plan=read(allowed['plan'],root);need(plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED','TINT_ROOT_APPROVAL_PENDING')
 need(str(Path(spec_path))==plan['approved_spec']['path'],'TINT_EXACT_ROOT_SPEC_PATH')
 spec=read(plan['approved_spec'],root);plan_shape(plan,spec,root)
 contract=read(spec['contract'],root)
 need(contract['operation']==OPERATION and contract['candidate_runtime_code']==plan['candidate_runtime_code'],'TINT_CONTRACT_NEW_OPERATION_AND_PACKAGE')
 return {'plan_ref':allowed['plan'],'plan':plan,'spec':spec,'contract':contract,'descriptor':d}
def original_runtime(context,root=ROOT,verify=True):
 p=read(context['contract']['original_HOLD_producer']['packet'],root)
 runtime_shape(p['runtime_code']);need(len(p['runtime_code'])==93,'TINT_FROZEN_ORIGINAL93_RUNTIME_REQUIRED')
 if verify:
  for r in p['runtime_code']:exact(r,root)
 return p['runtime_code']
def expected_runtime(context,root=ROOT,verify=True):
 rows=original_runtime(context,root,verify)+context['plan']['candidate_runtime_code']+context['plan']['maintenance_governance_runtime_code']
 out=runtime_shape(rows)
 need(len(out)==106,'TINT_OLD93_PLUS_SIX_ALGORITHM_PLUS_SEVEN_GOVERNANCE')
 if verify:
  for r in rows:exact(r,root)
 return list(out.values())
def validate_packet_context(packet,context,root=ROOT):
 plan=context['plan'];spec=context['spec']
 need(packet['task_id']==TASK and packet['execution_mode']=='EXACT78_CANDIDATE','TINT_PACKET_FIXED_TASK_MODE')
 need(packet['budgets']==packet['config']['budgets']==BUDGETS and all(type(v) is int for v in packet['budgets'].values()),'TINT_PACKET_PLAN_BUDGET')
 need(set(packet['inputs'])=={'spec','files','maintenance_render_plan'} and packet['inputs']['spec']==plan['approved_spec'] and packet['inputs']['maintenance_render_plan']==context['plan_ref'] and packet['inputs']['files']==[spec['contract']],'TINT_PACKET_EXACT_INPUTS')
 need(runtime_shape(packet['runtime_code'])==runtime_shape(expected_runtime(context,root)),'TINT_PACKET_FULL_RUNTIME_EXACT')
 from chaoyang.governance.exact78_operation_gate import _command_tail
 need(len(packet['commands'])==1 and _command_tail(packet['commands'][0])==[OPERATION,'--maintenance-spec',plan['approved_spec']['path']],'TINT_PACKET_EXACT_COMMAND_NO_OUTPUT_ROOT')
 from chaoyang.governance.common import canonical_bytes
 import hashlib
 actual=hashlib.sha256(canonical_bytes({'inputs':packet['inputs'],'runtime_code':packet['runtime_code'],'config':packet['config'],'command':packet['commands'][0],'schema':'reference-tint-render-only-maintenance-task-v1'})).hexdigest()
 need(packet['run_signature']==actual,'TINT_RECOMPUTED_RUN_SIGNATURE_DRIFT')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'TINT_STAGE_AND_ENVELOPE_REQUIRED_OUTPUTS')
 return True
