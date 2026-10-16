"""One finite explicitly approved pick_place contextpad4 repair candidate operation."""
from __future__ import annotations
import argparse,json,secrets,sys
from pathlib import Path
OPERATION='run_pick_place_1008_context_pad4_repair_candidate_v1'
def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['full'],required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--camera-config',type=Path,required=True)
    p.add_argument('--mask-seeds',type=Path,required=True)
    p.add_argument('--sessions',choices=['042'],required=True)
    p.add_argument('--repair-contract',type=Path,required=True)
    p.add_argument('--root-runtime-approval',type=Path,required=True)
    p.add_argument('--context-padding-px',type=int,choices=[4],required=True)
    p.add_argument('--cpu-source-workers',type=int,choices=[4],default=4)
    p.add_argument('--cpu-render-workers',type=int,choices=[4],default=4)
    p.add_argument('--reuse-from',action='append',type=Path,default=[])
    p.add_argument('--output-root',type=Path,required=True)
    return p
def main(argv=None):
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry(OPERATION,argv)
    from chaoyang.cli import _maintained_operations
    if OPERATION not in _maintained_operations():raise RuntimeError('A_EXPLICIT_MAINTAINED_OPERATION_NOT_PUBLISHED')
    from chaoyang.pick_place_context_repair_v1.approval import ROOT,ref,exact,verify_root_planned_allowlist
    from chaoyang.exact78_baseline.stage_reuse import current_producer
    from chaoyang.pick_place_context_repair_v1.runtime import run_registered_context_full
    a=parser().parse_args(argv);producer=current_producer(ROOT);packet=json.loads(exact(producer['task_packet']).read_text())
    real_argv=list(argv) if argv is not None else __import__('sys').argv[1:]
    approval,contract=verify_root_planned_allowlist(a.root_runtime_approval,argv=real_argv,packet=packet,require_approved=True)
    parents=contract['parents']
    if ref(a.repair_contract)!=approval['contract_ref'] or ref(a.manifest)!=parents['manifest'] or ref(a.camera_config)!=parents['camera'] or ref(a.mask_seeds)!=parents['seed_index']:
        raise RuntimeError('A_REGISTERED_ARG_INPUT_SHA_NOT_PLAN_BOUND')
    expected_reuse=[Path(parents['source_receipt']['path']).parents[2],Path(parents['motion_receipt']['path']).parents[2]]
    if [p.resolve() for p in a.reuse_from]!=expected_reuse:raise RuntimeError('A_SOURCE_MOTION_REUSE_ROOTS_NOT_EXACT')
    output=a.output_root.resolve()
    if output!=ROOT/'_run/current'/producer['task_id']/'attempts/attempt_0001/candidate':raise RuntimeError('A_OUTPUT_NOT_EXECUTOR_OWNED')
    from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease
    lease=FileGpuLease(lease_path=ROOT/'_run/current/GPU_LEASE.json',lock_path=ROOT/'_run/current/GPU_LEASE.lock',
        task_id=producer['task_id'],attempt_id='attempt_0001',gpu_id=0,executor_epoch=packet['writer']['executor_epoch'],
        fencing_token=secrets.token_hex(16),min_free_mib=40960)
    try:lease.__enter__()
    except Exception as exc:
        from chaoyang.pick_place_1008.pipeline import write
        output.mkdir(parents=True,exist_ok=False)
        write(output/'RESOURCE_WAIT.json',{'status':'WAIT_GPU_RESOURCE','algorithm_started':False,'reason':str(exc)})
        return 75
    try:result=run_registered_context_full(output,a.root_runtime_approval,real_argv)
    finally:lease.__exit__(*sys.exc_info())
    print(json.dumps({'status':result['status'],'sessions_complete':result['sessions_complete'],'frames_complete':result['frames_complete'],'visual_acceptance':'PENDING_NEW_ACTUAL_LOCAL_AI_REVIEW'},ensure_ascii=False))
    return 0 if result['status']=='COMPLETE' else 1
