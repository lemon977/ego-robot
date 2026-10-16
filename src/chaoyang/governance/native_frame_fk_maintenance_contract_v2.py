"""Two bounded q-bitfrozen renderer tasks. No algorithm is called by metadata checks."""
from pathlib import Path
import importlib.util,sys,json,hashlib
from chaoyang.governance.candidate_optimization_family_v4 import ROOT,exact,ref,published_current,descriptor
FAMILY='pick_place_native_frame_fk_dual_view_render_candidate_v2'
OPS=('run_pick_place_native_frame_fk_dual_view_24views_qualification_candidate_v2','run_pick_place_native_frame_fk_dual_view_hold007199_render_candidate_v2')
TASKS=('exact78_pick_place_native_frame_fk_dual_view_24views_qualification_candidate_revision2_20261010','exact78_pick_place_native_frame_fk_dual_view_hold007199_render_candidate_revision2_20261010')
FULL={'003':183,'007':158,'178':162,'199':137}
SAMPLES={s:[0,n//2,n-1] for s,n in FULL.items()}
SCOPES=(list(FULL),['007','199'])
BUDGET_TABLE=(
 {'cpu_threads_total':2,'gpu_count':0,'model_calls':0,'render_calls':48,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':600,'workers_max':1},
 {'cpu_threads_total':2,'gpu_count':0,'model_calls':0,'render_calls':1180,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':3600,'workers_max':1})
PKG_NAMES={'__init__.py','operation.py','reference_tint_lookup.py','render_robot.py','render_session.py','renderer.py'}
GOVERNANCE_FILES=(
 'src/chaoyang/governance/candidate_optimization_family_v4.py',
 'src/chaoyang/governance/current_candidate_optimization_contracts_v4.py',
 'src/chaoyang/governance/common_candidate_optimization_v5.py',
 'src/chaoyang/governance/native_frame_fk_maintenance_contract_v2.py',
 'src/chaoyang/governance/execute_native_frame_fk_maintenance_v2.py',
 'src/chaoyang/governance/collect_native_frame_fk_maintenance_v2.py',
 'src/chaoyang/native_frame_fk_render_candidate_v2/driver.py',
 'src/chaoyang/native_frame_fk_render_candidate_v2/__init__.py',
 'src/chaoyang/native_frame_fk_render_candidate_v2/asset_inputs.py',
 *('src/chaoyang/ops/'+op+'.py' for op in OPS))
def need(x,msg):
 if not x:raise ValueError(msg)
def read(pin,root=ROOT):return json.loads(exact(pin,root).read_text())
def runtime_shape(rows):
 need(isinstance(rows,list) and rows and all(isinstance(r,dict) and set(r)=={'path','bytes','sha256'} for r in rows),'FK_TYPED_RUNTIME')
 need(len({r['path'] for r in rows})==len(rows),'FK_RUNTIME_DUPLICATE')
 return {r['path']:r for r in rows}
def package_shape(rows,root=ROOT,verify=True):
 runtime_shape(rows);ps=[Path(r['path']) for r in rows]
 need(len(ps)==6 and len({p.parent for p in ps})==1 and {p.name for p in ps}==PKG_NAMES,'FK_EXACT_ADJACENT_SIX')
 if verify:
  for r in rows:exact(r,root)
 return ps[0].parent
def mode(spec):
 need(spec['operation'] in OPS,'FK_EXPLICIT_OPERATION');i=OPS.index(spec['operation'])
 need(spec['task_id']==TASKS[i],'FK_EXPLICIT_TASK');return i
def plan_shape(plan,spec,root=ROOT,verify=True):
 need(set(spec)=={'schema','operation','task_id','stage','sessions','native_frames','budgets','upstream_policy','contract','fit_parameters','automatic_retry','automatic_AI','Root_adoption','candidate_runtime_code'},'FK_SPEC_EXACT14')
 need(set(plan)=={'schema','publication_status','operation','task_ids','approved_spec','candidate_runtime_code','fixed_sessions','fixed_native_frames','budgets','zero_fit_parameters','holdout_evaluation_only','automatic_retry','automatic_AI','Root_adoption','collector_schema','maintenance_governance_runtime_code'},'FK_PLAN_EXACT16')
 i=mode(spec);need(plan['schema']=='root_planned_finite_native_frame_fk_dual_view_v1' and spec['schema']=='native_frame_fk_dual_view_finite_spec_v1','FK_SCHEMA')
 need(plan['operation']==OPS[i] and plan['task_ids']==[TASKS[i]],'FK_PLAN_TASK')
 need(spec['sessions']==plan['fixed_sessions']==SCOPES[i] and spec['native_frames']==plan['fixed_native_frames']==(12 if i==0 else 295),'FK_SCOPE')
 need(spec['stage']==('render_layer_qualification' if i==0 else 'render') and spec['upstream_policy']=='EXACT_GLOBAL_PASSED_SOURCE_MOTION_GPU','FK_SCOPE_STAGE')
 for b in [spec['budgets'],plan['budgets']]:need(b==BUDGET_TABLE[i] and all(type(v) is int for v in b.values()),'FK_EXACT_TYPED_BUDGET')
 need(spec['fit_parameters']=={} and plan['zero_fit_parameters'] is True and plan['holdout_evaluation_only'] is True,'FK_NO_FIT')
 for k in ['automatic_retry','automatic_AI','Root_adoption']:need(plan[k] is False and spec[k] is False,'FK_NO_AUTO')
 need(plan['collector_schema']=='native_frame_fk_dual_view_complete_candidate_index_v1','FK_COLLECTOR_SCHEMA')
 need(plan['candidate_runtime_code']==spec['candidate_runtime_code'],'FK_SPEC_PLAN_CODE')
 package_shape(plan['candidate_runtime_code'],root,verify)
 gov=runtime_shape(plan['maintenance_governance_runtime_code'])
 need(set(gov)=={str(root/p) for p in GOVERNANCE_FILES},'FK2_EXACT_GOVERNANCE_ELEVEN')
 if verify:
  for r in gov.values():exact(r,root)
 return i
def approved_context(spec_path,root=ROOT):
 cur=published_current(root);route=cur['special_status_contracts'].get(FAMILY);need(isinstance(route,dict),'FK_FAMILY_NOT_PUBLISHED')
 d=descriptor(route.get('root_registered_descriptor'),root)
 need(d['family_key']==FAMILY and {k:v for k,v in route.items() if k!='root_registered_descriptor'}==d['route_value'],'FK_PUBLISHED_ROUTE_DRIFT')
 allowed=[v for op,v in d['maintenance_operation_allowlist'].items() if v['plan']['path'] and read(v['plan'],root)['approved_spec']['path']==str(Path(spec_path))]
 need(len(allowed)==1 and allowed[0]['execution_allowed'] is True,'FK_ROOT_EXACT_SPEC_NOT_ALLOWED')
 pin=allowed[0]['plan'];plan=read(pin,root);spec=read(plan['approved_spec'],root)
 need(plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED','FK_ROOT_APPROVAL_PENDING')
 plan_shape(plan,spec,root);contract=read(spec['contract'],root)
 need(contract['family_key']==FAMILY and contract['candidate_runtime_code']==spec['candidate_runtime_code'],'FK_CONTRACT_IDENTITY')
 need(contract['control_operation_executed_by_new_driver'] is False,'FK_OLD_OPERATION_CANNOT_BE_ENTRY')
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_contract_assets
 validate_contract_assets(contract,root,True) # Before any registration or Robot construction.
 approval=read(d['root_preapproval'],root)
 need(approval.get('robot_asset_input_refs')==contract['robot_asset_input_refs'] and approval.get('fixed_existing_runtime_code')==contract['fixed_existing_runtime_code'],'FK2_ROOT_EXACT_ASSET_DATA_AND_EXISTING_IO_WHITELIST')
 package_shape(contract['control_runtime_code'],root)
 return dict(plan_ref=pin,plan=plan,spec=spec,contract=contract,descriptor=d)
def expected_runtime(context,root=ROOT,verify=True):
 packet=read(context['contract']['original_HOLD_producer']['packet'],root)
 need(len(packet['runtime_code'])==93,'FK_ORIGINAL93')
 rows=packet['runtime_code']+context['contract']['control_runtime_code']+context['plan']['candidate_runtime_code']+context['plan']['maintenance_governance_runtime_code']+context['contract']['fixed_existing_runtime_code']
 need(len(runtime_shape(rows))==117,'FK2_RUNTIME_93_PLUS_6_PLUS_6_PLUS_11_PLUS_IO1')
 if verify:
  for p in rows:exact(p,root)
 return rows
def evidence_inputs(context,root=ROOT,verify=True):
 files=[context['spec']['contract'],*context['contract']['robot_asset_input_refs']]
 if mode(context['spec'])==1:
  path=Path(context['contract']['Root_full295_qualification_selection']['path'])
  if verify:
   pin=ref(path);selection=read(pin,root)
   need(set(selection)=={'schema','status','reviewer','qualification_task_id','outer','packet','index','collector','run_signature','contract','candidate_runtime_code','control_runtime_code','one_HOLD295_pair_only','q_or_Motion_changed','automatic_retry','automatic_AI','Root_adoption','physical_contact_ground_truth','control_ground_truth','human_review_status'},'FK_ROOT_SELECTION_EXACT_FIELDS')
   need(selection['schema']=='root_actual_FK24_qualification_selection_for_one_HOLD295_pair_v1' and selection['status']=='ROOT_ACTUAL_Q_BITFROZEN_FK24_QUALIFICATION_SELECTED_FOR_ONE_HOLD295_PAIR' and selection['reviewer']=='/root' and selection['qualification_task_id']==TASKS[0],'FK_ROOT_ACTUAL_24_SELECTION_REQUIRED')
   need(selection['contract']==context['spec']['contract'] and selection['candidate_runtime_code']==context['contract']['candidate_runtime_code'] and selection['control_runtime_code']==context['contract']['control_runtime_code'] and selection['one_HOLD295_pair_only'] is True,'FK_ROOT_SAME_EXACT_RECIPE_ONE_PAIR')
   for key in ['q_or_Motion_changed','automatic_retry','automatic_AI','Root_adoption','physical_contact_ground_truth','control_ground_truth']:need(selection[key] is False,'FK_ROOT_NO_Q_OR_TRUTH_CHANGE')
   need(selection['human_review_status']=='PENDING','FK_ROOT_SELECTION_NOT_VISUAL_ACCEPTANCE')
   for key in ['outer','packet','index','collector']:exact(selection[key],root)
  else:pin={'path':str(path),'bytes':0,'sha256':'0'*64} # Draft only, never official READY.
  files.append(pin)
 return files

def validate_packet_context(packet,c,root=ROOT,verify=True):
 i=plan_shape(c['plan'],c['spec'],root,verify);p=c['plan'];s=c['spec']
 need(packet['task_id']==TASKS[i] and packet['execution_mode']=='EXACT78_CANDIDATE','FK_PACKET_ID')
 need(packet['budgets']==packet['config']['budgets']==BUDGET_TABLE[i] and all(type(v) is int for v in packet['budgets'].values()) and all(type(v) is int for v in packet['config']['budgets'].values()),'FK_PACKET_BUDGET')
 need(packet['inputs']=={'spec':p['approved_spec'],'files':evidence_inputs(c,root,verify),'maintenance_render_plan':c['plan_ref']},'FK_PACKET_INPUTS')
 need(runtime_shape(packet['runtime_code'])==runtime_shape(expected_runtime(c,root,verify)),'FK2_PACKET_EXACT117')
 from chaoyang.governance.exact78_operation_gate import _command_tail
 need(len(packet['commands'])==1 and _command_tail(packet['commands'][0])==[OPS[i],'--maintenance-spec',p['approved_spec']['path']],'FK_EXACT_TWO_ARGS')
 from chaoyang.governance.common import canonical_bytes
 sig=hashlib.sha256(canonical_bytes({'inputs':packet['inputs'],'runtime_code':packet['runtime_code'],'config':packet['config'],'command':packet['commands'][0],'schema':'native-frame-fk-dual-view-maintenance-task-v2'})).hexdigest()
 need(packet['run_signature']==sig,'FK_SIGNATURE')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'FK_OUTPUT_ENVELOPE')
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_packet_asset_inputs
 validate_packet_asset_inputs(packet,c['contract'],root,verify)
 return i
def import_package(rows,name,root=ROOT):
 folder=package_shape(rows,root);old=sys.modules.get(name)
 if old is None:
  sp=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
  m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
 else:need(Path(old.__file__).parent==folder,'FK_IMPORT_IDENTITY')
 return __import__(name,fromlist=['render_session'])
