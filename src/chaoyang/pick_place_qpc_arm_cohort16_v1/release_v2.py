"""Readonly release barrier: Root inventory + original23 + realdead + strictidle."""
from pathlib import Path
from .approval import ROOT,read,ref,exact
from chaoyang.governance.common import process_identity
ACTIVE={'RUNNING','CLAIMED','WAIT_GPU_RESOURCE'}
STATUS='ROOT_EXPLICIT_QPC_COHORT16_NEW_FAMILY_RELEASE_APPROVED'
KNOWN_ROOT_MAINTENANCE_NULL = {'exact78_pick_place_reference_tint_render_only_hold007199_maintenance_v1_20261010': {'operation': 'run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v1',
                                                                                      'family': 'pick_place_reference_tint_render_only_maintenance_v1',
                                                                                      'run_signature': 'REDACTED_PUBLIC_EXPORT',
                                                                                      'packet_ref': {'path': '/path/to/chaoyang/tasks/current/exact78_pick_place_reference_tint_render_only_hold007199_maintenance_v1_20261010/TASK_PACKET.json',
                                                                                                     'bytes': 28799,
                                                                                                     'sha256': 'c63a89248ae43f26a2fae29f7a27d146f14a5b24cc293e5b4897840fefa12d46'},
                                                                                      'outer_ref': {'path': '/path/to/chaoyang/_run/current/exact78_pick_place_reference_tint_render_only_hold007199_maintenance_v1_20261010/attempts/attempt_0001/RESULT.json',
                                                                                                    'bytes': 1724,
                                                                                                    'sha256': '8bb72508df9dc46c627e15f62a73a07c675834f6abf5b0d6bb13e22056214ef9'},
                                                                                      'executor_ref': {'path': '/path/to/chaoyang/src/chaoyang/governance/execute_reference_tint_maintenance_v1.py',
                                                                                                       'bytes': 33740,
                                                                                                       'sha256': '1ba988a567ba6564ea3e4806809920bcb363206fa9d8fc1eaa1658afce446cc6'},
                                                                                      'expected_status': 'FAILED_RUNTIME_FINAL',
                                                                                      'expected_worker_returncode': 1},
 'exact78_pick_place_reference_tint_render_only_hold007199_maintenance_revision2_20261010': {'operation': 'run_pick_place_1008_reference_tint_render_only_maintenance_candidate_v2',
                                                                                             'family': 'pick_place_reference_tint_render_only_maintenance_v2',
                                                                                             'run_signature': 'REDACTED_PUBLIC_EXPORT',
                                                                                             'packet_ref': {'path': '/path/to/chaoyang/tasks/current/exact78_pick_place_reference_tint_render_only_hold007199_maintenance_revision2_20261010/TASK_PACKET.json',
                                                                                                            'bytes': 29400,
                                                                                                            'sha256': 'f7c57f8390833c958e9d31109d1ae05920a05aab1b701c0276fc176b7a8be474'},
                                                                                             'outer_ref': {'path': '/path/to/chaoyang/_run/current/exact78_pick_place_reference_tint_render_only_hold007199_maintenance_revision2_20261010/attempts/attempt_0001/RESULT.json',
                                                                                                           'bytes': 2888,
                                                                                                           'sha256': '257753882cd84228fb8c5c39f225b2aea6682fee1bc001d6dd89835d21d2f057'},
                                                                                             'executor_ref': {'path': '/path/to/chaoyang/src/chaoyang/governance/execute_reference_tint_maintenance_v2.py',
                                                                                                              'bytes': 33740,
                                                                                                              'sha256': 'b150f1dd52e3a5c2e6d3b952b2e7e4d08ed3acdc9da25ab2d9b19730dbad1dc1'},
                                                                                             'expected_status': 'PASSED',
                                                                                             'expected_worker_returncode': 0}}

def resource_wait_compatible(tid,outer,packet):
 """Only the two immutable Root-executed GPU0 maintenance envelopes permit null."""
 if 'resource_wait' not in outer:raise ValueError('QP_RESOURCE_WAIT_FIELD_REQUIRED')
 value=outer['resource_wait']
 if value is False:return True
 if value is not None:return False
 allowed=KNOWN_ROOT_MAINTENANCE_NULL.get(tid)
 if allowed is None:raise ValueError('QP_NULL_RESOURCE_WAIT_ONLY_TWO_FROZEN_MAINTENANCE_TASKS')
 if outer['task_id']!=tid or packet['task_id']!=tid or outer['run_signature']!=allowed['run_signature'] or packet['run_signature']!=allowed['run_signature']:
  raise ValueError('QP_MAINTENANCE_NULL_EXACT_TASK_SIGNATURE')
 if outer['task_packet']!=allowed['packet_ref'] or read(exact(allowed['packet_ref']))!=packet or ref(exact(allowed['outer_ref']))!=allowed['outer_ref'] or read(exact(allowed['outer_ref']))!=outer:
  raise ValueError('QP_MAINTENANCE_NULL_FROZEN_ENVELOPE_PACKET')
 if outer['schema_version']!='pick-place-1008-execution-result-v1' or packet['schema_version']!='chaoyang-r22-task-packet-v1' or packet['execution_mode']!='EXACT78_CANDIDATE':
  raise ValueError('QP_MAINTENANCE_NULL_OFFICIAL_SCHEMA_MODE')
 budgets={'cpu_threads_total':1,'gpu_count':0,'model_calls':0,'render_calls':590,'session_attempts_max':1,'training_steps':0,'wall_seconds_max':2400,'workers_max':1}
 if packet['budgets']!=budgets or any(type(packet['budgets'][k]) is not int for k in budgets) or packet['config']['budgets']!=budgets:
  raise ValueError('QP_MAINTENANCE_NULL_TYPED_GPU0_MODEL0_ONE_ATTEMPT')
 if type(outer['algorithm_attempts_consumed']) is not int or outer['algorithm_attempts_consumed']!=1 or type(outer['worker_returncode']) is not int or outer['worker_returncode']!=allowed['expected_worker_returncode'] or outer['status']!=allowed['expected_status']:
  raise ValueError('QP_MAINTENANCE_NULL_EXACT_ACTUAL_TERMINAL')
 if any(outer.get(k) is not False for k in ['visual_qualified','training_eligible','control_ground_truth','files_deleted']) or outer.get('adoption')!='NOT_ADOPTED' or outer.get('user_review')!='PENDING' or packet['config'].get('deletion_authorized') is not False or packet['config'].get('numeric_gates_changed') is not False:
  raise ValueError('QP_MAINTENANCE_NULL_NO_GROUND_TRUTH_OR_ADOPTION')
 expected_command='PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run '+allowed['operation']+' --maintenance-spec '+str(ROOT/'docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/specs'/(tid+'.json'))
 if packet['commands']!=[expected_command] or len(packet['runtime_code'])!=106 or len({x['path'] for x in packet['runtime_code']})!=106 or allowed['executor_ref'] not in packet['runtime_code']:
  raise ValueError('QP_MAINTENANCE_NULL_EXACT_OPERATION_RUNTIME')
 exact(allowed['executor_ref'])
 spec=read(exact(packet['inputs']['spec']))
 if spec['task_id']!=tid or spec['operation']!=allowed['operation'] or spec['stage']!='render' or spec['sessions']!=['007','199'] or spec['native_frames']!=295 or spec['budgets']!=budgets or spec['fit_parameters']!={} or any(spec[k] is not False for k in ['Root_adoption','automatic_AI','automatic_retry']):
  raise ValueError('QP_MAINTENANCE_NULL_FIXED_HOLD_RENDER_SPEC')
 route=read(ROOT/'docs/governance/ALGORITHM_CONTRACT.json')['special_status_contracts'].get(allowed['family'],{})
 if route.get('fixed_task_ids')!=[tid] or route.get('candidate_operations')!=[allowed['operation']] or sorted(route.get('code_closure',[]),key=lambda x:x['path'])!=sorted(packet['runtime_code'],key=lambda x:x['path']):
  raise ValueError('QP_MAINTENANCE_NULL_EXACT_PUBLISHED_FAMILY')
 registered=read(exact(route['root_registered_descriptor']))
 approved=read(exact(registered['root_preapproval']))
 if registered['family_key']!=allowed['family'] or registered['operations']!=[allowed['operation']] or approved['schema']!='root_planned_finite_optimization_family_publication_v1' or approved['status']!='ROOT_EXPLICIT_FINITE_OPTIMIZATION_FAMILY_PUBLICATION_APPROVED' or approved['family_key']!=allowed['family']:
  raise ValueError('QP_MAINTENANCE_NULL_EXPLICIT_ROOT_FAMILY_APPROVAL')
 return True

def require_dead(pid,ticks,label):
 if type(pid) is not int or type(ticks) is not int or ticks<=0:raise ValueError('QP_EXACT_ACTOR_IDENTITY_REQUIRED:'+label)
 try:actual=process_identity(pid)
 except (FileNotFoundError,ProcessLookupError):return
 if actual['start_ticks']==ticks:raise ValueError('QP_PINNED_ACTOR_STILL_LIVE:'+label)
def check_task_actor(tid,outer):
 packet=read(exact(outer['task_packet']))
 if packet['task_id']!=tid or packet['run_signature']!=outer['run_signature']:raise ValueError('QP_RELEASE_PACKET_SIGNATURE')
 writer=packet['writer'];require_dead(writer['pid'],writer['proc_start_ticks'],tid+':writer')
 worker=read(ROOT/'_run/current'/tid/'attempts/attempt_0001/WORKER.json')
 if worker['task_id']!=tid or worker['writer']!=writer or worker['run_signature']!=packet['run_signature'] or worker['packet_command']!=packet['commands'][0]:raise ValueError('QP_RELEASE_WORKER_IDENTITY')
 require_dead(worker['pid'],worker['proc_start_ticks'],tid+':worker')
def require_original_source_scope(ix,q):
 sids=[x['session_id'] for x in ix['sessions']]
 if len(sids)!=199 or len(set(sids))!=199 or set(sids)!=set(q['source']['sessions']) or not (ix['sessions_requested']==ix['sessions_complete']==199) or ix['frames_complete']!=29322 or any(x['status']!='COMPLETE' for x in ix['sessions']):raise ValueError('QP_ORIGINAL_SOURCE_EXACT199_29322_SCOPE')
 return True
def approved_idle(path):
 a=read(path)
 if a.get('schema')!='root_qpc_cohort16_release_approval_v1' or a.get('status')!=STATUS or a.get('family')!='pick_place_qpc_arm_cohort16_cpu_candidate_v1':raise ValueError('QP_RELEASE_ROOT_APPROVAL_PENDING')
 from chaoyang.governance.candidate_optimization_family_v2 import descriptor
 descriptor(a['family_descriptor'])
 state=read(ROOT/'docs/governance/LONG_HORIZON_TASK_STATE.json');registry=read(ROOT/'tasks/current/INDEX.json')
 if state.get('next_task') or any(x['status'] in ACTIVE for x in state['tasks']) or any(x.get('execution_allowed') for x in registry['task_packets']):raise ValueError('QP_RELEASE_NOT_IDLE')
 ledger={x['task_id']:x for x in state['tasks']};routes={x['task_id']:x for x in registry['task_packets']}
 q=read(exact(a['original_queue']));parts=q['gates']+q['records']
 sids=[s for r in parts for s in r['sessions']]
 if len(parts)!=22 or len(sids)!=199 or len(set(sids))!=199 or set(sids)!=set(q['source']['sessions']) or sum(r['native_frames'] for r in parts)!=29322 or q['source']['native_frames']!=29322:raise ValueError('QP_ORIGINAL_199_29322_STRICT_PARTITION')
 inventory=read(exact(a['actor_inventory']))
 if inventory.get('schema')!='root_QPC_release_actor_inventory_v1' or inventory.get('original_queue')!=a['original_queue'] or inventory.get('all_administrative_dispatchers_trusted_terminal') is not True:raise ValueError('QP_TRUSTED_ACTOR_INVENTORY_PENDING')
 required_roles={'original_dispatcher','p5_024_028_waiter','p5_029_039_coordinator'}
 if not required_roles<=set(x['role'] for x in inventory['actors']):raise ValueError('QP_ALL_REQUIRED_OBSERVERS_INVENTORY')
 for actor in inventory['actors']:
  start=read(exact(actor['immutable_started_ref']))
  def get(keys):
   x=start
   for key in keys:x=x[key]
   return x
  if get(actor['pid_json_keys'])!=actor['pid'] or get(actor['start_ticks_json_keys'])!=actor['proc_start_ticks']:raise ValueError('QP_STARTED_REF_OBSERVER_IDENTITY')
  terminal=read(exact(actor['trusted_terminal_ref']))
  x=terminal
  for key in actor['terminal_status_json_keys']:x=x[key]
  if x!=actor['expected_terminal_status'] or not isinstance(x,str) or x in ['WAIT','RUNNING','ACTIVE','PENDING','STOP_WAIT_ROOT','REQUIRES_ROOT_REVIEW']:raise ValueError('QP_OBSERVER_NOT_ACTUAL_TRUSTED_FINISHED')
  require_dead(actor['pid'],actor['proc_start_ticks'],actor['role'])
 def closed(tid,*,original,expected_spec=None):
  r=ledger.get(tid);rr=routes.get(tid)
  if not r or not rr or r.get('pid') is not None or r.get('proc_start_ticks') is not None or rr.get('execution_allowed') is not False:raise ValueError('QP_SEALED_TERMINAL_REQUIRED:'+tid)
  if exact(r['result'])!=ROOT/'_run/current'/tid/'attempts/attempt_0001/RESULT.json':raise ValueError('QP_RELEASE_CANONICAL_OUTER_ENVELOPE_REQUIRED')
  o=read(exact(r['result']));t=read(exact(r['terminal_receipt']))
  if o.get('schema_version')!='pick-place-1008-execution-result-v1' or t.get('schema_version')!='pick-place-1008-receipt-v1':raise ValueError('QP_RELEASE_OFFICIAL_RECEIPT_SCHEMA')
  packet_path=exact(o['task_packet'])
  rp=Path(rr['packet_path']);rp=rp if rp.is_absolute() else ROOT/rp
  if packet_path!=ROOT/'tasks/current'/tid/'TASK_PACKET.json' or rp!=packet_path or rr['packet_sha256']!=o['task_packet']['sha256']:raise ValueError('QP_RELEASE_REGISTRY_CANONICAL_PACKET_SHA')
  packet=read(packet_path)
  if packet['task_id']!=tid or packet['run_signature']!=o['run_signature'] or r['run_signature']!=o['run_signature']:raise ValueError('QP_RELEASE_LEDGER_PACKET_SIGNATURE')
  if original and packet['inputs']['spec']!=expected_spec:raise ValueError('QP_RELEASE_ORIGINAL23_QUEUE_SPEC_EXACT_PIN')
  exact(packet['inputs']['spec'])
  if o['task_id']!=tid or t['task_id']!=tid or o['status']!=r['status'] or t['status']!=r['status'] or t['result']!=r['result'] or rr['result']!=r['result'] or rr['terminal_receipt']!=r['terminal_receipt'] or type(o['worker_returncode']) is not int or type(o['algorithm_attempts_consumed']) is not int or o['algorithm_attempts_consumed']!=1 or not resource_wait_compatible(tid,o,packet):raise ValueError('QP_OFFICIAL_TERMINAL_IDENTITY_OR_TYPED_RC')
  if r['status']=='PASSED':
   if o['worker_returncode']!=0 or o['validation_errors']!=[] or o['failure_code'] is not None:raise ValueError('QP_PASS_NOT_TRUE_RC0')
  elif r['status']=='FAILED_RUNTIME_FINAL' and not original:
   if o['worker_returncode']==0 and not o['failure_code'] and not o['validation_errors']:raise ValueError('QP_FAILED_WITHOUT_KNOWN_FAILURE')
  else:raise ValueError('QP_UNKNOWN_OR_ORIGINAL_NOT_PASSED')
  check_task_actor(tid,o)
  if original:
   ix=read(exact(o['required_outputs']['candidate/INDEX.json']))
   if ix['status']!='COMPLETE' or ix['sessions_complete']!=ix['sessions_requested'] or any(x['status']!='COMPLETE' for x in ix['sessions']):raise ValueError('QP_ORIGINAL_NOT_GLOBALLY_COMPLETE')
   return ix
  return None
 originals=[q['source']['task_id']]+[r['task_id'] for r in parts]
 specs={q['source']['task_id']:q['source']['spec'],**{x['task_id']:x['spec'] for x in parts}}
 for tid in originals:closed(tid,original=True,expected_spec=specs[tid])
 source_outer=read(exact(ledger[q['source']['task_id']]['result']))
 source_index=read(exact(source_outer['required_outputs']['candidate/INDEX.json']))
 require_original_source_scope(source_index,q)
 for r in parts:
  ix=read(exact(ledger[r['task_id']]['result']))['required_outputs']['candidate/INDEX.json'];ix=read(exact(ix))
  if set(x['session_id'] for x in ix['sessions'])!=set(r['sessions']) or ix['frames_complete']!=r['native_frames']:raise ValueError('QP_ORIGINAL_PARTITION_EXACT_NATIVE')
 additional=inventory['additional_terminal_task_ids']
 if not isinstance(additional,list) or len(set(additional))!=len(additional):raise ValueError('QP_ADDITIONAL_TERMINAL_LIST')
 for tid in additional:closed(tid,original=False)
 lease=read(ROOT/'_run/current/GPU_LEASE.json')
 if lease.get('status')!='RELEASED':raise ValueError('QP_STRICT_RELEASED_LEASE_REQUIRED')
 return dict(status='ROOT_APPROVED_ACTUAL_RELEASE_BOUNDARY_IDLE',original_tasks_closed=len(originals),original_sessions=199,original_native_frames=29322,additional_terminal_tasks=len(additional),actual_actor_identities_checked=len(inventory['actors']),lease_status='RELEASED',GPU_calls=0)
