from __future__ import annotations
from pathlib import Path
import hashlib,json
ROOT=Path('/path/to/chaoyang').resolve()
OPERATION='run_pick_place_1008_context_pad4_repair_candidate_v1'
RECIPE='A_MODEL_CONTEXT_PAD4_WRITE_E_FIXED_V1'
APPROVED_STATUS='ROOT_EXPLICIT_ONE_FINITE_REGISTERED_REPAIR_RUNTIME_APPROVED'
def read(p):return json.loads(Path(p).read_text())
def ref(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def exact(pin,*,project_only=True):
    if not isinstance(pin,dict) or set(pin)!=set(('path','bytes','sha256')):raise ValueError('A_EXACT_REF_REQUIRED')
    p=Path(pin['path']).resolve(strict=True)
    if project_only and not p.is_relative_to(ROOT):raise ValueError('A_INPUT_OUTSIDE_PROJECT')
    if ref(p)!=pin:raise ValueError('A_EXACT_REF_DRIFT:'+str(p))
    return p
def without_output_root(argv):
    out=[];i=0
    while i<len(argv):
        if argv[i]=='--output-root':
            if i+1>=len(argv):raise ValueError('OUTPUT_ROOT_ARG_MISSING')
            i+=2
        elif str(argv[i]).startswith('--output-root='):i+=1
        else:out.append(str(argv[i]));i+=1
    return out
def verify_root_planned_allowlist(path,*,argv=None,packet=None,require_approved=True):
    p=Path(path).resolve(strict=True)
    if not p.is_relative_to(ROOT):raise ValueError('A_ROOT_ALLOWLIST_OUTSIDE_PROJECT')
    approval=read(p)
    if approval.get('schema')!='root_planned_context_pad4_runtime_allowlist_v2':raise ValueError('A_ROOT_PLAN_SCHEMA')
    if require_approved and approval.get('status')!=APPROVED_STATUS:raise ValueError('A_ROOT_RUNTIME_APPROVAL_PENDING')
    if approval.get('expected_operation')!=OPERATION or approval.get('recipe_id')!=RECIPE or approval.get('session_ids')!=['042'] or approval.get('native_frames')!=168:
        raise ValueError('A_ROOT_PLAN_FINITE_IDENTITY')
    if any(k in approval for k in ('producer_task_packet','packet_sha256','task_packet_ref','registered_packet_sha256')):
        raise ValueError('A_ROOT_APPROVAL_MUST_NOT_REFERENCE_FUTURE_PACKET_SHA')
    if approval.get('allowed_affected_stages')!=['gpu','render']:raise ValueError('A_AFFECTED_STAGES')
    if approval.get('human_review_status')!='PENDING' or any(approval.get(k) is not False for k in ('physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible')):
        raise ValueError('A_ROOT_PLAN_TRUTH_FLAGS')
    if argv is not None and without_output_root(argv)!=approval.get('expected_args'):raise ValueError('A_ROOT_PLAN_ARGS_NOT_EXACT')
    contract=read(exact(approval['contract_ref']))
    if contract.get('recipe_id')!=RECIPE or contract.get('session_id')!='042' or contract.get('native_frames')!=168:raise ValueError('A_CONTRACT_IDENTITY')
    if contract.get('frozen_control_replay_required') is not True or contract.get('PP_inference_invocations_max')!=2:raise ValueError('A_ROOT_PLAN_CONTROL_PLUS_CANDIDATE_NOT_EXPLICIT')
    if contract.get('model_context_extra_padding_px')!=4 or contract.get('final_write_E_and_P_fixed') is not True:raise ValueError('A_CONTEXT_FACTOR_NOT_FIXED4')
    if approval['expected_budgets']!={'cpu_threads_total':10,'gpu_count':1,'model_calls':10000,'render_calls':500,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':2400,'workers_max':4}:
        raise ValueError('A_FINITE_BUDGET_DRIFT')
    extras=approval.get('exact_extra_runtime_code')
    if not isinstance(extras,list) or not extras or len(extras)>12 or len({x['path'] for x in extras})!=len(extras):raise ValueError('A_EXACT_EXTRA_CODE_LIST')
    for x in extras:exact(x)
    fields=approval.get('exact_extra_binding_fields',{})
    for stage in ('gpu','render'):
        value=fields.get(stage)
        if not isinstance(value,dict) or set(value)!={'repair_recipe'}:raise ValueError('A_ONLY_EXACT_TYPED_REPAIR_BINDING_ALLOWED')
        recipe=value['repair_recipe']
        if recipe.get('recipe_id')!=RECIPE or recipe.get('contract')!=approval['contract_ref'] or recipe.get('session_id')!='042' or recipe.get('original_failure')!=contract['parents']['original_actual_FAILED']:
            raise ValueError('A_TYPED_RECIPE_PARENT_IDENTITY')
        if recipe.get('original_GPU_mask_receipt')!=contract['parents']['baseline_GPU_receipt'] or recipe.get('observed_mask_NPZ')!=contract['parents']['baseline_mask_npz'] or recipe.get('final_write_E_and_P_fixed') is not True or recipe.get('q_unchanged') is not True or recipe.get('frozen_control_replay_required') is not True or recipe.get('PP_inference_invocations_max')!=2:
            raise ValueError('A_TYPED_RECIPE_SINGLE_FACTOR_OR_CONTROL_DRIFT')
        if any(recipe.get(k) is not False for k in ('physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible')):raise ValueError('A_TYPED_RECIPE_GT_FLAGS')
    if packet is not None:
        if packet.get('task_id')!=approval['expected_task_id'] or packet.get('execution_mode')!='EXACT78_CANDIDATE' or packet.get('budgets')!=approval['expected_budgets']:
            raise ValueError('A_REGISTERED_PACKET_SCOPE_OR_BUDGET_DRIFT')
        inputs=packet.get('inputs',{}).get('files',[])
        if ref(p) not in inputs or approval['contract_ref'] not in inputs:raise ValueError('A_APPROVAL_AND_CONTRACT_NOT_PACKET_INPUTS')
        for x in extras:
            if x not in packet.get('runtime_code',[]):raise ValueError('A_EXTRA_CODE_NOT_REGISTERED')
        spec=read(exact(packet['inputs']['spec']))
        if spec.get('operation')!=OPERATION or spec.get('task_id')!=approval['expected_task_id'] or spec.get('args')!=approval['expected_args'] or spec.get('budgets')!=approval['expected_budgets']:
            raise ValueError('A_REGISTERED_SPEC_NOT_PREAPPROVED')
    return approval,contract
def verify_all_recipe_parents(contract):
    for k,x in contract['parents'].items():
        if isinstance(x,dict) and 'path' in x:exact(x,project_only=k!='source_QPC')
    if len(contract['original_93_runtime_code'])!=93 or len({x['path'] for x in contract['original_93_runtime_code']})!=93:raise ValueError('A_ORIGINAL93_COMPLETE_PIN_REQUIRED')
    for x in contract['original_93_runtime_code']:exact(x)
    for x in contract['parents'].get('original_model_dependency_refs',[]):exact(x)
    failure=read(exact(contract['parents']['original_actual_FAILED']))
    if failure.get('visual_status')!='FAILED':raise ValueError('A_ORIGINAL_ACTUAL_FAILED_REQUIRED')
