"""Typed Root approval and canonical CPU-only finite packet gate."""
from pathlib import Path
import json,hashlib,os
ROOT=Path('/path/to/chaoyang').resolve()
FAMILY='pick_place_qpc_arm_cohort16_cpu_candidate_v1'
EVALUATION='run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1'
MOTION_RENDER='run_pick_place_qpc_arm_cohort16_motion_render_candidate_v1'
FIXED=['001','003','008','100','007','199','035','039','042','050','051','111','114','146','147','192']
STATUS='ROOT_EXPLICIT_FIXED16_QPC_FINITE_RUNTIME_APPROVED'
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p).resolve(strict=True);h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8388608):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def exact(x):
 if not isinstance(x,dict) or set(x)!= {'path','bytes','sha256'}:raise ValueError('QP_EXACT_REF_REQUIRED')
 p=Path(x['path'])
 if not p.is_absolute() or '..' in p.parts or any(z.is_symlink() for z in [p,*p.parents] if z.is_relative_to(ROOT)):raise ValueError('QP_CANONICAL_PLAIN_PROJECT_REF_REQUIRED')
 p=p.resolve(strict=True)
 if not p.is_relative_to(ROOT) or ref(p)!=x:raise ValueError('QP_REF_DRIFT_OR_OUTSIDE_PROJECT')
 return p
FIXED_EXISTING_EXECUTED_CODE=[{'path': '/path/to/chaoyang/review/pick_place_1008/QPC_CONSTRAINED_ARM_TRAJECTORY_CPU_TRIAL_V1/project_original_arm_q_to_actual_QPC_limits_cpu_v2.py', 'bytes': 18317, 'sha256': '02ce7d82ded2d24d9a9d98400a021f0c0dd698a42e38e20f6cf82150d326c6f3'}, {'path': '/path/to/chaoyang/review/pick_place_1008/V9_REV2_FINAL_ASSEMBLY_PREPARATION/prepare_final_assembly_v9_rev2_revision5.py', 'bytes': 25856, 'sha256': '9b25ddff1fb5ccac65c99a97c9624d8ff521feaa6f53ab7c8ef9a97f286968e2'}]
def fixed_existing_code():
 for pin in FIXED_EXISTING_EXECUTED_CODE:exact(pin)
 return [dict(x) for x in FIXED_EXISTING_EXECUTED_CODE]
def argv_without_output(a):
 a=list(map(str,a));out=[];i=0
 while i<len(a):
  if a[i]=='--output-root':
   if i+1>=len(a):raise ValueError('QP_MISSING_OUTPUT_ARG')
   i+=2
  else:out.append(a[i]);i+=1
 return out
def verify_runtime_approval(path,*,argv=None,packet=None,require_approved=True):
 p=Path(path).resolve(strict=True)
 if not p.is_relative_to(ROOT):raise ValueError('QP_ROOT_APPROVAL_OUTSIDE_PROJECT')
 a=read(p)
 if a.get('schema')!='root_fixed16_QPC_runtime_allowlist_v1' or a.get('family')!=FAMILY:raise ValueError('QP_ROOT_PLAN_SCHEMA')
 if require_approved and a.get('status')!=STATUS:raise ValueError('QP_ROOT_APPROVAL_PENDING')
 op=a.get('operation')
 if op not in [EVALUATION,MOTION_RENDER] or a.get('session_ids')!=FIXED or a.get('native_frames')!=2578:raise ValueError('QP_FIXED_SCOPE')
 if any(k in a for k in ['producer_task_packet','packet_sha256','registered_packet_sha256']):raise ValueError('QP_APPROVAL_NO_FUTURE_PACKET_SELF_REFERENCE')
 if a.get('allowed_affected_stages')!= ([] if op==EVALUATION else ['motion','render']):raise ValueError('QP_SINGLE_FACTOR_STAGE_SCOPE')
 if a.get('human_review_status')!='PENDING' or any(a.get(k) is not False for k in ['physical_ground_truth','physical_contact_ground_truth','control_ground_truth','training_eligible']):raise ValueError('QP_TRUTH_FLAGS')
 budget={'cpu_threads_total':1 if op==EVALUATION else 10,'gpu_count':0,'model_calls':0,'render_calls':0 if op==EVALUATION else 5156,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':900 if op==EVALUATION else 14400,'workers_max':1 if op==EVALUATION else 4}
 if not isinstance(a.get('expected_budgets'),dict) or any(type(v) is not int for v in a['expected_budgets'].values()) or a.get('expected_budgets')!=budget:raise ValueError('QP_CPU_FINITE_BUDGET')
 if argv is not None and argv_without_output(argv)!=a.get('expected_args'):raise ValueError('QP_EXACT_APPROVED_COMMAND')
 c=read(exact(a['contract_ref']))
 if c.get('cohort',{}).get('sha256')!='7d5767139e8add74745b52a4b6ae57c2b7ef3b82f490f609be7b9518fadd6024' or c.get('max_sessions')!=16 or c.get('max_native_frames')!=2578:raise ValueError('QP_CONTRACT_SCOPE')
 if not c.get('no_parameter_grid_or_HOLD_fit_or_seed_adaptation'):raise ValueError('QP_FIXED_NO_HOLD_FIT_REQUIRED')
 for pin in [c['cohort'],c['reused_math'],c['policy']]:exact(pin)
 existing=fixed_existing_code()
 if c.get('fixed_existing_executed_code')!=existing or a.get('exact_existing_runtime_code')!=existing:raise ValueError('QP_FIXED_EXISTING_EXECUTED_CODE_NOT_ROOT_BOUND')
 plan=read(exact(a['operation_plan']))
 if plan.get('operation')!=op or plan.get('publication_status')!='ROOT_PLANNED_FINITE_OPERATION_APPROVED' or plan.get('fixed_existing_executed_code')!=existing or plan.get('contract')!=a['contract_ref']:raise ValueError('QP_EXISTING_CODE_NOT_ROOT_APPROVED_OPERATION_PLAN')
 extra=a.get('exact_extra_runtime_code')
 if not isinstance(extra,list) or not extra or len(extra)>16 or len({x['path'] for x in extra})!=len(extra):raise ValueError('QP_RUNTIME_EXACT_ALLOWLIST')
 for x in extra:exact(x)
 if op==MOTION_RENDER:
  selection=a.get('evaluation_selection')
  if not isinstance(selection,dict) or selection.get('session_ids')!=FIXED or selection.get('evaluation_task_id')!='exact78_pick_place_qpc_arm_cohort16_cpu_candidate_revision1_20261010':raise ValueError('QP_ACTUAL_EVALUATION_SELECTION_PENDING')
  for k in ['index','outer_result','terminal_receipt']:exact(selection[k])
  if not isinstance(selection.get('case_reports'),list) or len(selection['case_reports'])!=16:raise ValueError('QP_EXPLICIT_16_ACTUAL_CASE_REFS_REQUIRED')
  for x in selection['case_reports']:exact(x)
 if packet is not None:
  if not isinstance(packet.get('budgets'),dict) or any(type(v) is not int for v in packet['budgets'].values()) or packet.get('task_id')!=a['expected_task_id'] or packet.get('budgets')!=budget or packet.get('execution_mode')!='EXACT78_CANDIDATE':raise ValueError('QP_PACKET_APPROVAL_DRIFT')
  pins=packet['inputs']['files']
  if ref(path) not in pins or a['contract_ref'] not in pins:raise ValueError('QP_APPROVAL_CONTRACT_NOT_PACKET_INPUTS')
  if any(x not in packet.get('runtime_code',[]) for x in extra):raise ValueError('QP_NEW_RUNTIME_NOT_PACKET_BOUND')
  if any(x not in packet.get('runtime_code',[]) or x not in pins for x in existing) or a['operation_plan'] not in pins:raise ValueError('QP_FIXED_EXECUTED_CODE_OR_PLAN_NOT_DIRECT_PACKET_BOUND')
  spec=read(exact(packet['inputs']['spec']))
  if not isinstance(spec.get('budgets'),dict) or any(type(v) is not int for v in spec['budgets'].values()) or spec['operation']!=op or spec['args']!=a['expected_args'] or spec['task_id']!=a['expected_task_id'] or spec['budgets']!=budget:raise ValueError('QP_PACKET_SPEC_NOT_ROOT_APPROVED')
 return a,c
