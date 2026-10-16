"""Explicit Root-approved A/T/QP coexistence under the standard governance CAS."""
from pathlib import Path
import copy,hashlib,json,os,subprocess
ROOT=Path('/path/to/chaoyang')
FAMILIES={
 'pick_place_context_pad4_candidate_v1':('run_pick_place_1008_context_pad4_repair_candidate_v1',),
 'pick_place_reference_tint_render_only_maintenance_v1':('run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v1',),
 'pick_place_reference_tint_render_only_maintenance_v2':('run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v2',),
 'pick_place_qpc_arm_cohort16_cpu_candidate_v1':('run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1','run_pick_place_qpc_arm_cohort16_motion_render_candidate_v1'),
}
APPROVED='ROOT_EXPLICIT_FINITE_OPTIMIZATION_FAMILY_PUBLICATION_APPROVED'
ACTIVE={'RUNNING','CLAIMED','WAIT_GPU_RESOURCE'}
def read(p):return json.loads(Path(p).read_text())
def exact(pin,root=ROOT):
 if not isinstance(pin,dict) or set(pin)!={'path','bytes','sha256'}:raise ValueError('OPT_EXACT_PROJECT_REF_REQUIRED')
 p=Path(pin['path'])
 if not p.is_absolute() or '..' in p.parts or not p.is_relative_to(root):raise ValueError('OPT_EXACT_PROJECT_REF_REQUIRED')
 if type(pin['bytes']) is not int or pin['bytes']<0 or not isinstance(pin['sha256'],str) or len(pin['sha256'])!=64:raise ValueError('OPT_REF_SHAPE')
 for a in (p,*p.parents):
  if a.is_symlink():raise ValueError('OPT_SYMLINK_FORBIDDEN')
  if a==root:break
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8388608):h.update(b)
 if p.stat().st_size!=pin['bytes'] or h.hexdigest()!=pin['sha256']:raise ValueError('OPT_REF_DRIFT:'+str(p))
 return p
def ref(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def verify_route(family,route,root=ROOT):
 if family not in FAMILIES:raise ValueError('OPT_UNAPPROVED_FAMILY')
 if tuple(sorted(route.get('candidate_operations',[])))!=tuple(sorted(FAMILIES[family])):raise ValueError('OPT_OPERATION_SET_DRIFT')
 code=route.get('code_closure')
 if not isinstance(code,list) or not code or len({x['path'] for x in code})!=len(code):raise ValueError('OPT_EXACT_UNIQUE_CLOSURE_REQUIRED')
 for r in code:exact(r,root)
 for op in FAMILIES[family]:
  if not any(r['path']==str(root/'src/chaoyang/ops'/f'{op}.py') for r in code):raise ValueError('OPT_CANONICAL_OPERATION_NOT_ROUTE_BOUND')
 if route.get('task_packet_required') is not True or type(route.get('training_steps_authorized')) is not int or route['training_steps_authorized']!=0 or route.get('current_runnable_operation') is not None:raise ValueError('OPT_ROUTE_FENCE_OR_TRUTH_SCOPE')
 return route
def descriptor(pin,root=ROOT):
 d=read(exact(pin,root))
 if set(d)!={'schema','family_key','operations','root_preapproval','route_value','maintenance_operation_allowlist','exact_runtime_code'} or d['schema']!='root_registered_pick_place_optimization_family_descriptor_v1':raise ValueError('OPT_DESCRIPTOR_SCHEMA')
 f=d['family_key']
 if f not in FAMILIES or tuple(d['operations'])!=FAMILIES[f]:raise ValueError('OPT_EXPLICIT_FAMILY_OPERATION_SET')
 a=read(exact(d['root_preapproval'],root))
 if a.get('status')!=APPROVED or a.get('schema')!='root_planned_finite_optimization_family_publication_v1':raise ValueError('OPT_ROOT_PREAPPROVAL_PENDING')
 if any(k in a for k in ('descriptor_ref','descriptor_sha256','packet_ref','packet_sha256','registry_sha256')):raise ValueError('OPT_PREAPPROVAL_SHA_CYCLE_FORBIDDEN')
 for key in ('family_key','operations','route_value','maintenance_operation_allowlist','exact_runtime_code'):
  if a.get(key)!=d.get(key):raise ValueError('OPT_NOT_ROOT_PLANNED:'+key)
 verify_route(f,d['route_value'],root)
 extras=d['exact_runtime_code']
 if not isinstance(extras,list) or not extras or len({x['path'] for x in extras})!=len(extras):raise ValueError('OPT_EXPLICIT_EXTRA_CLOSURE_REQUIRED')
 for x in extras:exact(x,root)
 if any(x not in d['route_value']['code_closure'] for x in extras):raise ValueError('OPT_RUNTIME_NOT_ROUTE_BOUND')
 allowed=d['maintenance_operation_allowlist']
 if d['route_value'].get('maintenance_render_operation_allowlist',{})!=allowed:raise ValueError('OPT_ROUTE_NESTED_ALLOWLIST_NOT_ROOT_PLANNED')
 if not isinstance(allowed,dict) or set(allowed)-set(FAMILIES[f]):raise ValueError('OPT_ALLOWLIST_UNKNOWN_OPERATION')
 for op,v in allowed.items():
  if set(v)!={'plan','execution_allowed'} or v['execution_allowed'] is not True:raise ValueError('OPT_ALLOWLIST_EXPLICIT_PLAN_REQUIRED')
  plan=read(exact(v['plan'],root))
  # The explicit Root preapproval binds these exact bytes, not arbitrary ROOT_* labels.
  if plan.get('operation')!=op or plan.get('publication_status')!='ROOT_PLANNED_FINITE_OPERATION_APPROVED':raise ValueError('OPT_PLAN_IDENTITY_OR_APPROVAL')
 return d
def published_current(root=ROOT):
 p=root/'docs/governance/ALGORITHM_CONTRACT.json'
 receipt=read(root/'docs/governance/CURRENT_STATUS_RECEIPT.json')
 if receipt.get('files',{}).get('algorithm_contract')!=ref(p):raise ValueError('OPT_CURRENT_ALGORITHM_NOT_RECEIPT_BOUND')
 return read(p)
def preserve_and_add(value,current,registered=(),root=ROOT):
 result=copy.deepcopy(value);routes=result['special_status_contracts'];existing=current.get('special_status_contracts',{})
 keep={}
 for f in FAMILIES:
  if f not in existing:continue
  route=verify_route(f,existing[f],root)
  pin=route.get('root_registered_descriptor')
  if pin is not None:
   d=descriptor(pin,root);plain={k:v for k,v in route.items() if k!='root_registered_descriptor'}
   if d['family_key']!=f or plain!=d['route_value']:raise ValueError('OPT_PUBLISHED_DESCRIPTOR_ROUTE_DRIFT')
   if route.get('maintenance_render_operation_allowlist',{})!=d['maintenance_operation_allowlist']:raise ValueError('OPT_NESTED_PLAN_ALLOWLIST_DRIFT')
  elif f!='pick_place_context_pad4_candidate_v1':raise ValueError('OPT_NEW_FAMILY_NOT_ROOT_DESCRIPTOR_BOUND')
  routes[f]=copy.deepcopy(route)
 if 'maintenance_render_operation_allowlist' in current:raise ValueError('OPT_SCHEMA_FORBIDS_TOP_LEVEL_ALLOWLIST')
 seen=set()
 for pin in registered:
  d=descriptor(pin,root);f=d['family_key']
  if f in seen:raise ValueError('OPT_DUPLICATE_REGISTERED_FAMILY')
  seen.add(f)
  if f in existing and existing[f].get('root_registered_descriptor')!=pin:raise ValueError('OPT_ALREADY_PUBLISHED_FAMILY_IMMUTABLE_USE_NEW_VERSION')
  routes[f]=copy.deepcopy(d['route_value']);routes[f]['root_registered_descriptor']=copy.deepcopy(pin)
  if routes[f].get('maintenance_render_operation_allowlist',{})!=d['maintenance_operation_allowlist']:raise ValueError('OPT_NEW_NESTED_PLAN_ALLOWLIST_DRIFT')
 result.pop('maintenance_render_operation_allowlist',None)
 return result
def build_candidate_contract(value,root=ROOT,planned_family_descriptor_ref=None):
 current=published_current(root)
 refs=[] if planned_family_descriptor_ref is None else [planned_family_descriptor_ref]
 return preserve_and_add(value,current,refs,root)
def assert_registration_boundary(root=ROOT,root_preapproval=None):
 if not isinstance(root_preapproval,dict):raise ValueError('OPT_EXPLICIT_ROOT_RELEASE_INVENTORY_REQUIRED')
 inventory=read(exact(root_preapproval['release_actor_inventory'],root))
 if inventory.get('schema')!='root_actual_finite_optimization_family_publish_idle_actor_inventory_v1' or inventory.get('status')!='ROOT_ACTUAL_READONLY_INVENTORY_RECORDED' or inventory.get('family_key')!=root_preapproval['family_key'] or inventory.get('operations')!=root_preapproval['operations']:raise ValueError('OPT_ROOT_RELEASE_INVENTORY_SCOPE')
 actors=inventory.get('actors');required=root_preapproval.get('publication_required_actor_labels')
 if not isinstance(actors,list) or not actors or not isinstance(required,list) or not required or len(set(required))!=len(required):raise ValueError('OPT_EXPLICIT_NONEMPTY_ACTOR_SET')
 if sorted(r['actor_label'] for r in actors)!=sorted(required):raise ValueError('OPT_ACTOR_INVENTORY_NOT_EXACT_ROOT_SET')
 for actor in actors:
  if set(actor)!={'actor_label','pid','proc_start_ticks'} or type(actor['pid']) is not int or type(actor['proc_start_ticks']) is not int or actor['pid']<=0 or actor['proc_start_ticks']<=0:raise ValueError('OPT_ACTOR_IDENTITY_SCHEMA')
  p=Path('/proc')/str(actor['pid'])/'stat'
  try:line=p.read_text()
  except FileNotFoundError:continue
  fields=line[line.rfind(')')+2:].split()
  if int(fields[19])==actor['proc_start_ticks']:raise ValueError('OPT_ROOT_ACTOR_STILL_PRESENT:'+actor['actor_label'])
 state=read(root/'docs/governance/LONG_HORIZON_TASK_STATE.json')
 index=read(root/'tasks/current/INDEX.json')
 if state.get('next_task') or any(r.get('status') in ACTIVE for r in state.get('tasks',[])):raise ValueError('OPT_ACTIVE_LEDGER_CONFLICT')
 if any(r.get('execution_allowed') is True for r in index.get('task_packets',[])):raise ValueError('OPT_ACTIVE_PACKET_CONFLICT')
 lease=read(root/'_run/current/GPU_LEASE.json')
 if lease.get('status')!='RELEASED':raise ValueError('OPT_STRICT_RELEASED_LEASE_REQUIRED')
 # Standard validator checks the current receipt/registry/generation/code closure before any publish.
 cp=subprocess.run(['/usr/local/bin/python','-B','-m','chaoyang.cli','validate-governance'],cwd=root,env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
 try:g=json.loads(cp.stdout)
 except ValueError:raise ValueError('OPT_GOVERNANCE_UNPARSEABLE')
 if cp.returncode!=0 or g.get('status')!='PASS':raise ValueError('OPT_GOVERNANCE_NOT_PASS')
 return {'ledger_idle':True,'registry_idle':True,'lease_status':'RELEASED','governance_status':'PASS'}
