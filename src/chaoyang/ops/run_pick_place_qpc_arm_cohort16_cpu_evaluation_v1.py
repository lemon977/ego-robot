"""Canonical maintained finite CPU QPC operation; never implicit execution."""
import argparse,json
from pathlib import Path
OPERATION='run_pick_place_qpc_arm_cohort16_cpu_evaluation_v1'
def main(argv=None):
 from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
 authorize_operation_entry(OPERATION,argv)
 from chaoyang.cli import _maintained_operations
 if OPERATION not in _maintained_operations():raise RuntimeError('QP_OPERATION_NOT_EXPLICITLY_PUBLISHED')
 p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--root-runtime-approval',type=Path,required=True);p.add_argument('--max-sessions',type=int,choices=[16],required=True);p.add_argument('--max-native-frames',type=int,choices=[2578],required=True);p.add_argument('--cpu-workers',type=int,choices=[1],required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args(argv)
 import os,cv2
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OMP_NUM_THREADS')!='1' or os.environ.get('NUMBA_NUM_THREADS')!='1':raise RuntimeError('QP_CPU1_GPU0_ENV_REQUIRED')
 cv2.setNumThreads(1)
 from chaoyang.pick_place_qpc_arm_cohort16_v1.approval import verify_runtime_approval,ref
 real=list(argv) if argv is not None else __import__('sys').argv[1:]
 approval,contract=verify_runtime_approval(a.root_runtime_approval,argv=real)
 if ref(a.cohort)!=contract['cohort'] or ref(a.contract)!=approval['contract_ref']:raise RuntimeError('QP_ARG_INPUT_PIN_DRIFT')
 from chaoyang.pick_place_qpc_arm_cohort16_v1.runtime import registered_compute,write,read
 index_ref=registered_compute(a.output_root);index=read(index_ref['path'])
 write(a.output_root/'RESULT.json',dict(status='COMPLETE',evaluation_complete=True,index=index_ref,candidate_failure_and_UNKNOWN_retained=True,visual_qualified=False,physical_contact_ground_truth=False,control_ground_truth=False,training_eligible=False,human_review='PENDING'))
 print(json.dumps({'status':'COMPLETE','evaluation_complete':True,'candidate_failure_and_UNKNOWN_retained':True}));return 0
