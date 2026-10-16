#!/usr/bin/env python3
"""Governed preparation and kernel for fixed16 CPU arm projection. No implicit --run."""
from pathlib import Path
import sys,os,json,hashlib,time,datetime,argparse,importlib.util,copy
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/path/to/chaoyang')
P=R/'review/pick_place_1008/QPC_ARM_COHORT16_GOVERNED_PREPARATION_V1'
COHORT=R/'review/pick_place_1008/STRATIFIED_GENERALIZATION_PRECISION_CLEAN_CONTACT_CPU_REGRESSION_V1/REGRESSION_COHORT_V1.json'
COHORT_SHA='7d5767139e8add74745b52a4b6ae57c2b7ef3b82f490f609be7b9518fadd6024'
OLD=R/'review/pick_place_1008/QPC_CONSTRAINED_ARM_TRAJECTORY_CPU_TRIAL_V1/project_original_arm_q_to_actual_QPC_limits_cpu_v2.py'
OLD_SHA='02ce7d82ded2d24d9a9d98400a021f0c0dd698a42e38e20f6cf82150d326c6f3'
OPERATION='run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1'
TASK_ID='exact78_pick_place_qpc_arm_cohort16_cpu_candidate_revision1_20261010'
FIXED=['001','003','008','100','007','199','035','039','042','050','051','111','114','146','147','192']
def ref(p):
 b=Path(p).read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
class FatalInputDrift(RuntimeError):pass
def verify(r):
 p=Path(r['path'])
 if not p.resolve().is_relative_to(R) or ref(p)!=r:raise FatalInputDrift('EXACT_INPUT_OR_CODE_REF_DRIFT')
 return p
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return ref(p)
def original_math():
 assert ref(OLD)['sha256']==OLD_SHA
 ms=importlib.util.spec_from_file_location('immutable_original_fixed_QPC_math',OLD);m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
 return m
def cohort_rows():
 assert ref(COHORT)['sha256']==COHORT_SHA
 c=read(COHORT);rows=c['representative_cases']
 assert [r['session_id'] for r in rows]==FIXED==c['candidate_required_scope'] and len(rows)==16 and sum(r['native_frames'] for r in rows)==2578
 assert not set(FIXED)&{'178','179','180'}
 assert c['original_HOLD_evaluation_only']==['007','199'] and c['fixed_numeric_limits']==dict(position_mm=20.,rotation_deg=15.,velocity_rad_s=3.6,acceleration_rad_s2=54.)
 assert all(r['independent_acquisition_group']=='UNKNOWN' for r in rows)
 for r in rows:
  if r['session_id'] in ['007','199']:verify(r['frozen_holdout_seed'])
 return c,rows
def metadata_context():
 sys.path.insert(0,str(R/'src'))
 H=R/'review/pick_place_1008/V9_REV2_FINAL_ASSEMBLY_PREPARATION/prepare_final_assembly_v9_rev2_revision5.py'
 assert ref(H)['sha256']=='9b25ddff1fb5ccac65c99a97c9624d8ff521feaa6f53ab7c8ef9a97f286968e2'
 ms=importlib.util.spec_from_file_location('cohort16_metadata_checks',H);h=importlib.util.module_from_spec(ms);ms.loader.exec_module(h)
 q=h.read(h.QUEUE);assert h.digest(h.QUEUE)==h.QUEUE_SHA
 for pin in q['production_code']:h.verify(pin)
 s=h.read(h.verify(q['source']['spec']));mp=h.path(h.arg(s['args'],'--manifest'));assert h.digest(mp)==h.MANIFEST_SHA
 c=h.Checks(q,h.read(mp),h.ref(mp));cohort,rows=cohort_rows();checked=[]
 for r in rows:
  producer=c.producer(r['producer_task_id']);assert producer['index_ref']==r['producer_index']
  row=producer['rows'][r['session_id']];assert row['status']=='COMPLETE' and row['raw_frame_count']==r['native_frames']
  qa=read(verify(r['QA']));assert qa['structural_audit_pass'] is True and qa['errors']==[] and qa['native_frame_count']==r['native_frames'] and qa['session_id']==r['session_id'] and qa['task_id']==r['producer_task_id']
  assert qa['producer']['producer_index']==producer['index_ref']
  source_rc=read(verify(row['stage_receipts']['source']));assert source_rc['files']['source/SESSION_INPUT.json']==qa['source_input']==r['source_stat']['session_input']
  assert row['stage_receipts']=={s:v['receipt'] for s,v in qa['checks']['stage_producer_closure'].items()}
  for stage in ['source','motion','gpu','render']:verify(row['stage_receipts'][stage])
  native=qa['checks']['numeric_diagnostics']['all_native_frame_numeric_and_source_validity']
  assert len(native)==r['native_frames']
  if r['session_id']=='192':
   x=native[145]['sides']['right'];assert x['source_hand_observed'] is False and x['source_complete_valid'] is False and x['display_valid'] is True and x['original_frame_state']['source']=='EDGE_HOLD_SHORT' and x['original_frame_state']['tracking_gate_applicable'] is False
  checked.append(dict(producer_task_id=r['producer_task_id'],session_id=r['session_id'],native_frames=r['native_frames'],producer_index=producer['index_ref'],current_row=row,QA=r['QA'],source_stat=r['source_stat'],role=r['role'],device_serial=r['device_serial'],calibration_group_key=r['calibration_group_key'],frozen_holdout_seed=r['frozen_holdout_seed']))
 return h,q,c,cohort,checked
def require_registered_runtime(output):
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 from .approval import verify_runtime_approval
 from chaoyang.governance.exact78_operation_gate import _command_tail
 producer=current_producer(R);packet=read(verify(producer['task_packet']))
 tail=_command_tail(packet['commands'][0]);assert tail[0]==OPERATION and packet['task_id']==TASK_ID
 args=tail[1:];assert args.count('--root-runtime-approval')==1
 approval,contract=verify_runtime_approval(args[args.index('--root-runtime-approval')+1],argv=args,packet=packet,require_approved=True)
 assert packet['budgets']['wall_seconds_max']==900 and packet['budgets']['gpu_count']==0
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1' and os.environ.get('OPENBLAS_NUM_THREADS')=='1'
 assert Path(output).resolve()==R/'_run/current'/TASK_ID/'attempts/attempt_0001/candidate'
 assert contract['reused_math']==ref(OLD) and contract['cohort']==ref(COHORT)
 return producer
def known_pose_gate(raw_flags,QA_native):
 n=len(QA_native);known=np.zeros((2,n),bool)
 for i,row in enumerate(QA_native):
  for side,name in enumerate(['left','right']):
   state=row['sides'][name]['original_frame_state']
   known[side,i]=bool(raw_flags['native__head_valid'][i] and raw_flags['native__controller_valid'][i,side] and raw_flags['native__manus_valid'][i,side] and raw_flags['native__hand_observed'][side,i] and state['tracking_gate_applicable'])
 return known
def synthetic_checks():
 m=original_math();base=m.self_tests()
 n=3;flags={'native__head_valid':np.ones(n,bool),'native__controller_valid':np.ones((n,2),bool),'native__manus_valid':np.ones((n,2),bool),'native__hand_observed':np.ones((2,n),bool)}
 native=[dict(sides={s:dict(original_frame_state=dict(tracking_gate_applicable=True)) for s in ['left','right']}) for _ in range(n)]
 flags['native__manus_valid'][-1,1]=False;flags['native__hand_observed'][1,-1]=False;native[-1]['sides']['right']['original_frame_state']['tracking_gate_applicable']=False
 known=known_pose_gate(flags,native);assert known[0].all() and known[1,:-1].all() and not known[1,-1]
 # Invalid source qualification does not remove any sample from rate computation.
 dt=np.array([.04,.025]);vel,acc,_,_=m.operators(dt);assert vel.shape==(2,3) and acc.shape==(1,3)
 return dict(schema='cohort16_unchanged_QPC_math_and_invalid_source_UNKNOWN_synthetic_v1',script=ref(Path(__file__)),original_math=ref(OLD),original_fixed_math_synthetic_tests=base,invalid_source_display_hold_never_qualifies_pose_PASS=True,all_native_rate_samples_preserved=True,actual16_run=False,model_or_GPU_called=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING')
def registered_compute(output):
 # There is deliberately no ungated actual --run path.
 producer=require_registered_runtime(output)
 started=time.monotonic();deadline=started+900
 h,q,c,cohort,rows=metadata_context();m=original_math()
 from chaoyang.exact78_baseline.robot_model import Robot
 robot=Robot(R);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 reports=[]
 for info in rows:
  sid=info['session_id'];n=info['native_frames'];out=output/'sessions'/sid
  out.mkdir(parents=True,exist_ok=False)
  try:
   if time.monotonic()>deadline:raise TimeoutError('COHORT16_CPU900_BUDGET_EXCEEDED')
   row=info['current_row'];qa=read(verify(info['QA']))
   motion=read(verify(row['stage_receipts']['motion']));gpu=read(verify(row['stage_receipts']['gpu']))
   source_meta=read(verify(qa['source_input']))
   with np.load(verify(motion['files']['actions.npz']),allow_pickle=False) as z:orig={k:z[k] for k in z.files}
   with np.load(verify(qa['checks']['independent_fk']['derived_arrays']),allow_pickle=False) as z:baseline_qa={k:z[k] for k in z.files}
   with np.load(verify(source_meta['arrays']),allow_pickle=False) as z:raw={k:z[k] for k in ['qpc_ns','native__head_valid','native__controller_valid','native__manus_valid','native__hand_observed','native__hand_display_valid','native__hand_source','native__complete_valid','mapping__manus_right_skew_ns']}
   assert orig['q_arm'].shape==(n,2,7) and orig['q_hand'].shape==(n,2,22) and np.isfinite(orig['q_arm']).all() and np.isfinite(orig['q_hand']).all()
   assert np.array_equal(orig['qpc_ns'],raw['qpc_ns']) and np.array_equal(orig['q_arm_joint_names'],np.asarray([x.joint_names for x in robot.arms]))
   dt=np.diff(raw['qpc_ns']).astype(float)/1e9;assert np.isfinite(dt).all() and np.all(dt>0)
   original_v=np.diff(orig['q_arm'],axis=0)/dt[:,None,None];original_a=np.diff(original_v,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None]
   assert np.array_equal(original_v,orig['arm_velocity_rad_s']) and np.array_equal(original_a,orig['arm_acceleration_rad_s2'])
   assert np.array_equal(original_v,baseline_qa['actual_qpc_velocity_rad_s']) and np.array_equal(original_a,baseline_qa['actual_qpc_acceleration_rad_s2'])
   known=known_pose_gate(raw,qa['checks']['numeric_diagnostics']['all_native_frame_numeric_and_source_validity'])
   if sid=='192':
    assert not known[1,145] and raw['native__manus_valid'][145,1]==False and raw['native__hand_observed'][1,145]==False and raw['native__hand_display_valid'][1,145]==True and raw['native__hand_source'][1,145]=='EDGE_HOLD_SHORT' and abs(int(raw['mapping__manus_right_skew_ns'][145]))/1e6==48.9893
   target=baseline_qa['recomputed_target_root_world'];assert np.isfinite(target).all()
   # All original-q FK must reproduce the existing independent QA before any candidate is emitted.
   baseline=np.empty((2,n,4,4))
   for i in range(n):
    if time.monotonic()>deadline:raise TimeoutError('COHORT16_CPU900_BUDGET_EXCEEDED')
    for side in range(2):baseline[side,i]=orig['T_world_base']@robot.arm_frames(side,orig['q_arm'][i,side])['root']
   if not np.array_equal(baseline,baseline_qa['recomputed_actual_fk_root_world']):raise FatalInputDrift('SOURCE_MODEL_BASELINE_FK_DRIFT')
   candidate=np.empty_like(orig['q_arm']);solver=[];complete=True
   for side in range(2):
    for joint in range(7):
     x,check=m.solve_joint(orig['q_arm'][:,side,joint],dt,float(robot.arm_lower[side,joint]),float(robot.arm_upper[side,joint]),deadline)
     solver.append(dict(side=['left','right'][side],joint_name=str(orig['q_arm_joint_names'][side,joint]),**check))
     if x is None:complete=False
     else:candidate[:,side,joint]=x
   common=dict(session_id=sid,native_frames=n,producer=producer,current_original_producer=info['producer_index'],original_stage_receipts=row['stage_receipts'],source_arrays=source_meta['arrays'],original_actions=motion['files']['actions.npz'],original_independent_QA=info['QA'],role=info['role'],device_serial=info['device_serial'],calibration_group_key=info['calibration_group_key'],solver_results=solver,original_full_native_baseline_FK_replay_exact=True,unchanged_original_source_QPC_hand_base_targets_GPU_P=True,original_NUM_FAIL_flags_kept=True,no_HOLD_parameter_or_seed_adaptation=True)
   if not complete:
    result=dict(common,status='FAILED_CONVEX_PROJECTION_NO_QUALIFIED_CANDIDATE',candidate_emitted=False,actual_candidate_allN_FK_evaluated=False,algorithmic_failure=True,source_invalid_pose_points=int((~known).sum()),physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING')
   else:
    assert np.array_equal(candidate[[0,-1]],orig['q_arm'][[0,-1]])
    assert not ((np.ptp(orig['q_arm'],axis=0)>1e-8)&(np.ptp(candidate,axis=0)<=1e-12)).any(),'SOURCE_MOVEMENT_FROZEN'
    cv=np.diff(candidate,axis=0)/dt[:,None,None];ca=np.diff(cv,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None]
    assert np.isfinite(candidate).all() and np.isfinite(cv).all() and np.isfinite(ca).all()
    vfail=np.zeros((n,2),bool);afail=np.zeros((n,2),bool);vfail[1:]=np.max(abs(cv),axis=-1)>3.6;afail[2:]=np.max(abs(ca),axis=-1)>54.
    roots=np.empty_like(baseline)
    for i in range(n):
     if time.monotonic()>deadline:raise TimeoutError('COHORT16_CPU900_BUDGET_EXCEEDED')
     for side in range(2):roots[side,i]=orig['T_world_base']@robot.arm_frames(side,candidate[i,side])['root']
    assert np.isfinite(roots).all()
    position_diag=np.linalg.norm(roots[...,:3,3]-target[...,:3,3],axis=-1)*1000
    relative=np.swapaxes(target[...,:3,:3],-1,-2)@roots[...,:3,:3]
    rotation_diag=np.rad2deg(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).reshape(2,n)
    position_qualified=np.where(known,position_diag,np.nan);rotation_qualified=np.where(known,rotation_diag,np.nan)
    pfail=known&(position_diag>20.);rfail=known&(rotation_diag>15.)
    ratepass=not(vfail.any() or afail.any());knownpass=not(pfail.any() or rfail.any());unknown=~known
    labels=np.where(unknown,'UNKNOWN_SOURCE_INVALID',np.where(pfail|rfail,'FAIL_DERIVED_TARGET','PASS_DERIVED_TARGET'))
    status=('FAILED_CANDIDATE_ORIGINAL_RATE_OR_KNOWN_POSE_GATES' if not ratepass or not knownpass else ('RATE_AND_KNOWN_POSE_PASS_SOURCE_INVALID_POINTS_UNKNOWN' if unknown.any() else 'RATE_AND_ALL_KNOWN_DERIVED_POSE_PASS_UNADOPTED'))
    ap=out/'PRIVATE_COHORT_QPC_ARM_CANDIDATE_ARRAYS_V1.npz'
    np.savez_compressed(ap,private_candidate_q_arm=candidate,unchanged_original_q_hand=orig['q_hand'],unchanged_original_qpc_ns=raw['qpc_ns'],unchanged_original_T_world_base=orig['T_world_base'],unchanged_original_raw_targets=target,candidate_all_native_fk_roots=roots,diagnostic_only_position_mm_all_native=position_diag,diagnostic_only_rotation_deg_all_native=rotation_diag,source_qualified_position_mm=position_qualified,source_qualified_rotation_deg=rotation_qualified,source_pose_applicable=known,source_pose_gate_labels=labels,actual_candidate_QPC_velocity_rad_s=cv,actual_candidate_QPC_acceleration_rad_s2=ca,candidate_rate_velocity_fail=vfail,candidate_rate_acceleration_fail=afail,original_q_arm=orig['q_arm'],original_NUM_velocity_fail=orig['qpc_velocity_violation'],original_NUM_acceleration_fail=orig['qpc_acceleration_violation'],original_raw_manus_valid=raw['native__manus_valid'],original_raw_hand_observed=raw['native__hand_observed'],original_display_valid=raw['native__hand_display_valid'],original_display_source=raw['native__hand_source'])
    # NaN exists only in the explicitly invalid/UNKNOWN qualified-error representation.
    with np.load(ap,allow_pickle=False) as z:
     assert np.array_equal(z['unchanged_original_q_hand'],orig['q_hand']) and np.array_equal(z['unchanged_original_qpc_ns'],raw['qpc_ns']) and np.array_equal(z['unchanged_original_T_world_base'],orig['T_world_base']) and np.array_equal(z['unchanged_original_raw_targets'],target)
     assert np.array_equal(np.isnan(z['source_qualified_position_mm']),unknown) and np.array_equal(np.isnan(z['source_qualified_rotation_deg']),unknown)
    result=dict(common,status=status,candidate_emitted=True,private_candidate_arrays=ref(ap),actual_candidate_allN_FK_evaluated=True,all_native_rate_gate_pass=ratepass,known_source_points_derived_pose_gate_pass=knownpass,source_invalid_pose_points=int(unknown.sum()),source_invalid_pose_point_indices=[dict(side=['left','right'][int(s)],native_frame_index=int(i),qualified_pose_gate='UNKNOWN',display_source=str(raw['native__hand_source'][s,i])) for s,i in np.argwhere(unknown)],all_source_pose_qualification=bool(not unknown.any() and knownpass),candidate_pose_FAIL_indices=[dict(side=['left','right'][int(s)],native_frame_index=int(i)) for s,i in np.argwhere(pfail|rfail)],candidate_velocity_FAIL_native=np.flatnonzero(vfail.any(axis=1)).tolist(),candidate_acceleration_FAIL_native=np.flatnonzero(afail.any(axis=1)).tolist(),max_velocity_rad_s=float(abs(cv).max()),max_acceleration_rad_s2=float(abs(ca).max()),known_source_max_position_mm=float(position_diag[known].max()) if known.any() else None,known_source_max_rotation_deg=float(rotation_diag[known].max()) if known.any() else None,max_original_q_offset_rad=float(abs(candidate-orig['q_arm']).max()),first_last_original_q_bit_exact=True,no_original_moving_joint_frozen=True,new_Contact_and_render_and_actualAI='PENDING_UNCHANGED_OLD_CONTACT_CANNOT_BIND_CANDIDATE_Q',physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING')
   for pin in [motion['files']['actions.npz'],qa['source_input'],source_meta['arrays'],row['stage_receipts']['gpu'],gpu['files']['masks.npz']]:verify(pin)
   if info['frozen_holdout_seed']:verify(info['frozen_holdout_seed'])
  except FatalInputDrift:
   raise
  except Exception as e:
   result=dict(session_id=sid,native_frames=n,status='FAILED_CASE_BOUNDED_EVALUATION_SOURCE_OR_SOLVER_OR_MODEL_CHECK',error_type=type(e).__name__,error=str(e),candidate_qualification=False,other_independent_cases_continue=True,source_or_q_or_original_outputs_modified=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING')
  rp=write(out/'CASE_CPU_CANDIDATE_EVALUATION_V1.json',result);reports.append(dict(session_id=sid,native_frames=n,status=result['status'],report=rp))
  print(json.dumps(dict(progress='GOVERNED_COHORT_CASE_EVALUATED',**reports[-1])),flush=True)
 index=write(output/'INDEX.json',dict(schema='governed_CPU_cohort16_candidate_evaluation_index_v1',status='COMPLETE',execution_semantics='ALL_FIXED16_EVALUATIONS_COMPLETE_FAILURE_AND_UNKNOWN_RETAINED',sessions_requested=16,sessions_complete=16,frames_complete=2578,cohort=ref(COHORT),runtime_producer=producer,unchanged_algorithm=ref(OLD),fixed_cases=reports,sessions_evaluated=len(reports),native_frames_requested=sum(x['native_frames'] for x in reports),native_frames_actual_candidate_FK_evaluated=sum(x['native_frames'] for x in reports if read(x['report']['path']).get('actual_candidate_allN_FK_evaluated') is True),actual_motion_contact_render_outputs_written=False,actual_GPU_SAM_ProPainter_calls=0,source_q_hand_QPC_targets_base_calibration_P_GPU_preserved=True,fit_on_HOLD=False,independent_source_group_count='UNKNOWN',physical_precision_and_visual_generalization='UNQUALIFIED_PENDING',physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review_status='PENDING',full_goal_achieved=False))
 return index
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--output');a=p.parse_args();assert sum([a.check_only,a.self_test,a.execute])==1
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 if a.check_only:
  h,q,c,cohort,rows=metadata_context();m=original_math()
  print(json.dumps(dict(status='PREPARATION_METADATA_BOUND_NOT_REGISTERED_NOT_EXECUTABLE',cohort=ref(COHORT),scope=dict(sessions=len(rows),native_frames=sum(x['native_frames'] for x in rows)),scope_no_overlap_old450=True,unchanged_algorithm=ref(OLD),same_algorithm_parameters=m.PARAMS,actual_projection_or_candidate_FK_run=False,actual_RuntimePacket_for_new_operation_required=True)))
 elif a.self_test:
  v=synthetic_checks();r=write(P/'UNCHANGED_MATH_AND_SOURCE_INVALID_UNKNOWN_SYNTHETIC_V1.json',v);print(json.dumps(dict(status='SYNTHETIC_ONLY_PASS_NEW16_NOT_EXECUTED',report=r)))
 else:
  assert a.output is not None
  print(json.dumps(dict(index=registered_compute(a.output))))
if __name__=='__main__':main()
