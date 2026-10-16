from pathlib import Path
import json,hashlib
ROOT=Path('/path/to/chaoyang')
FAMILY='pick_place_whole_mesh_screen_bounds_render_candidate_v1'
OP='run_pick_place_whole_mesh_screen_bounds_24views_qualification_candidate_v1'
TASK='exact78_pick_place_whole_mesh_screen_bounds_24views_qualification_candidate_revision1_20261010'
SCOPE=['003','007','178','199']
BUDGETS=dict(cpu_threads_total=2,gpu_count=0,model_calls=0,render_calls=48,session_attempts_max=1,training_steps=0,wall_seconds_max=600,workers_max=1)
def need(v,m):
 if not v:raise RuntimeError(m)
def pin(row,root=ROOT):
 p=Path(row['path']);need(p.is_absolute() and p.is_relative_to(root) and p.resolve()==p,'SCREEN_EXACT_PROJECT_NO_SYMLINK')
 b=p.read_bytes();need(type(row['bytes']) is int and len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],'SCREEN_REF')
 return p
def read(row,root=ROOT):return json.loads(pin(row,root).read_text())
def code_map(rows):
 need(isinstance(rows,list) and len({v['path'] for v in rows})==len(rows),'SCREEN_RUNTIME_UNIQUE');return {v['path']:v for v in rows}
def signature(inputs,rows,config,command):
 from chaoyang.governance.common import canonical_bytes
 return hashlib.sha256(canonical_bytes(dict(inputs=inputs,runtime_code=rows,config=config,command=command,schema='whole-mesh-screenbounds-fixed24-maintenance-v1'))).hexdigest()
def evidence_inputs(ctx,root=ROOT,verify=True):
 c=ctx['contract'];selection_path=Path(c['Root_one_finite24_selection']['path'])
 if not verify:return [ctx['spec']['contract'],*c['robot_asset_input_refs'],{'path':str(selection_path),'bytes':0,'sha256':'0'*64}]
 b=selection_path.read_bytes();selref=dict(path=str(selection_path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 selected=read(selref,root)
 need(selected['schema']=='root_actual_screenbounds_code_recipe_selected_for_one_finite24_trial_v1' and selected['status']=='ROOT_ACTUAL_SCREENBOUNDS_ONE_FINITE24_TRIAL_APPROVED' and selected['reviewer']=='/root','SCREEN_ROOT_SELECTION_PENDING')
 need(selected['contract']==ctx['spec']['contract'] and selected['candidate_runtime_code']==c['candidate_runtime_code'] and selected['control_runtime_code']==c['control_runtime_code'],'SCREEN_ROOT_EXACT_CODE_RECIPE')
 for k in ['q_or_Motion_modified','FK_cache_changed','codec_or_precision_changed','Root_adoption','automatic_retry','physical_contact_ground_truth']:need(selected[k] is False,'SCREEN_ROOT_SINGLE_FACTOR')
 return [ctx['spec']['contract'],*c['robot_asset_input_refs'],selref]
def expected_runtime(ctx,root=ROOT,verify=True):
 c=ctx['contract'];base=read(c['original_HOLD_producer']['packet'],root)['runtime_code']
 need(len(base)==93,'SCREEN_ORIGINAL93')
 rows=base+c['control_runtime_code']+c['candidate_runtime_code']+c['entry_runtime_code']+c['fixed_existing_runtime_code']+c['qualification_metadata_runtime_code']+ctx['plan']['maintenance_governance_runtime_code']
 code_map(rows)
 if verify:
  for v in rows:pin(v,root)
 return rows
def context_from_spec(specpath,root=ROOT):
 actual=json.loads((root/'docs/governance/ALGORITHM_CONTRACT.json').read_text());route=actual['special_status_contracts'].get(FAMILY)
 need(isinstance(route,dict),'SCREEN_FAMILY_NOT_ROOT_REGISTERED')
 from chaoyang.governance.candidate_optimization_family_v9 import descriptor
 d=descriptor(route['root_registered_descriptor'],root)
 need(d['family_key']==FAMILY and d['operations']==[OP] and {k:v for k,v in route.items() if k!='root_registered_descriptor'}==d['route_value'],'SCREEN_EXACT_REGISTERED_ROUTE')
 allowed=d['maintenance_operation_allowlist'][OP];need(allowed['execution_allowed'] is True,'SCREEN_EXPLICIT_EXECUTION_ALLOWED')
 pp=allowed['plan'];plan=read(pp,root);need(plan['publication_status']=='ROOT_PLANNED_FINITE_OPERATION_APPROVED' and plan['task_id']==TASK and plan['operation']==OP,'SCREEN_ROOT_PLAN_PENDING')
 need(plan['approved_spec']['path']==str(Path(specpath)),'SCREEN_SPEC_EXACT_PATH')
 spec=read(plan['approved_spec'],root);c=read(spec['contract'],root)
 need(spec['schema']=='screenbounds_fixed24_qualification_spec_v1' and spec['task_id']==TASK and spec['operation']==OP and spec['sessions']==SCOPE and spec['paired_native_frames']==12 and spec['rendered_views_control']==spec['rendered_views_candidate']==24,'SCREEN_FIXED24')
 need(spec['budgets']==plan['budgets']==BUDGETS and all(type(v) is int for b in [spec['budgets'],plan['budgets']] for v in b.values()),'SCREEN_TYPED_FINITE_BUDGET')
 need(c['family_key']==FAMILY and c['operation']==OP and c['candidate_runtime_code']==plan['candidate_runtime_code'] and c['entry_runtime_code']==plan['entry_runtime_code'],'SCREEN_CODE_CONTEXT')
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_contract_assets
 validate_contract_assets(c,root,True) # XML + all69 signed assets before any registration/Robot.
 approval=read(d['root_preapproval'],root)
 need(approval['robot_asset_input_refs']==c['robot_asset_input_refs'] and approval['fixed_existing_runtime_code']==c['fixed_existing_runtime_code'],'SCREEN_ASSET_WHITELIST')
 need(approval['release_actor_inventory']==plan['root_fresh_actor_inventory'],'SCREEN_ROOT_FRESH_OWNER_INVENTORY')
 ctx=dict(plan=plan,plan_ref=pp,spec=spec,contract=c,descriptor=d)
 evidence_inputs(ctx,root);expected_runtime(ctx,root)
 return ctx
def validate_packet(packet,ctx,root=ROOT):
 need(packet['task_id']==TASK and packet['execution_mode']=='EXACT78_CANDIDATE','SCREEN_FIXED_REGISTERED_TASK')
 need(packet['budgets']==packet['config']['budgets']==BUDGETS and all(type(v) is int for b in [packet['budgets'],packet['config']['budgets']] for v in b.values()),'SCREEN_PACKET_TYPED_BUDGET')
 inputs=dict(spec=ctx['plan']['approved_spec'],files=evidence_inputs(ctx,root),maintenance_render_plan=ctx['plan_ref'])
 need(packet['inputs']==inputs and code_map(packet['runtime_code'])==code_map(expected_runtime(ctx,root)),'SCREEN_PACKET_BINDING')
 from chaoyang.governance.exact78_operation_gate import _command_tail
 need(len(packet['commands'])==1 and _command_tail(packet['commands'][0])==[OP,'--maintenance-spec',ctx['plan']['approved_spec']['path']],'SCREEN_EXACT_CLI_ARGS')
 need(packet['run_signature']==signature(inputs,packet['runtime_code'],packet['config'],packet['commands'][0]),'SCREEN_FRESH_RUN_SIGNATURE')
 need(packet['required_outputs']==['candidate/INDEX.json','candidate/RESULT.json','RESULT.json','terminal_receipt'],'SCREEN_OUTPUT_ENVELOPE')
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_packet_asset_inputs
 validate_packet_asset_inputs(packet,ctx['contract'],root,True)
 return True

OPS=(OP,);TASKS=(TASK,);SCOPES=(SCOPE,);BUDGET_TABLE=(BUDGETS,)
def mode(spec):
 need(spec.get('task_id')==TASK and spec.get('operation')==OP,'SCREEN_ONLY_EXPLICIT_ONE_TASK')
 return 0
def plan_shape(plan,spec,root=ROOT,verify=True):
 mode(spec)
 need(spec['sessions']==SCOPE and spec['paired_native_frames']==12 and spec['rendered_views_control']==spec['rendered_views_candidate']==24,'SCREEN_FIXED12_NATIVE48_CALLS')
 need(plan['family_key']==FAMILY and plan['operation']==OP and plan['task_id']==TASK and spec['budgets']==plan['budgets']==BUDGETS and all(type(v) is int for b in [spec['budgets'],plan['budgets']] for v in b.values()),'SCREEN_PLAN_TYPED_BUDGET')
 need(len(plan['candidate_runtime_code'])==7 and len(plan['entry_runtime_code'])==5 and len(plan['maintenance_governance_runtime_code'])==4,'SCREEN_EXACT7_CANDIDATE5_ENTRY4_GOV')
 for rows in [plan['candidate_runtime_code'],plan['entry_runtime_code'],plan['maintenance_governance_runtime_code']]:
  code_map(rows)
  if verify:
   for v in rows:pin(v,root)
 return 0
approved_context=context_from_spec
validate_packet_context=validate_packet
