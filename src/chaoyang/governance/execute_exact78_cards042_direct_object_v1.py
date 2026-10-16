"""Register and close one fresh Cards042 A-diamond identity reconstruction."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from chaoyang.governance.common import (AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK='exact78_cards042_direct_object_rebuild_v1_20260929'
ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
INDEX=REPO_ROOT/'tasks/current/INDEX.json'
SESSION='play_cards_0902_042'


def idx():
    return {k:copy.deepcopy(v) for k,v in load_json(INDEX).items()
            if k not in {'governance_revision','generation_id','generated_at','supersedes_index'}}


def main():
    global TASK, ROOT, PACKET
    parser=argparse.ArgumentParser();parser.add_argument('--retry',type=int,choices=(0,1,2,3,4),default=0);parser.add_argument('--hybrid-v2',action='store_true');args=parser.parse_args()
    operation='run_exact78_cards042_direct_object_v1'
    execution_revision='EXACT78_CARDS042_DIRECT_OBJECT_V1'
    receipt_name='EXACT78_CARDS042_DIRECT_OBJECT_V1.json'
    event_prefix='EXACT78_CARDS042_DIRECT_OBJECT'
    if args.retry:
        TASK=f'exact78_cards042_direct_object_rebuild_v1_retry{args.retry}_20260929'
        ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
        PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
    if args.hybrid_v2:
        if args.retry: raise RuntimeError('HYBRID_V2_HAS_NO_RETRY_FLAG')
        TASK='exact78_cards042_direct_object_rebuild_v2_20260929'
        ROOT=REPO_ROOT/'_run/current'/TASK/'attempts/attempt_0001'
        PACKET=REPO_ROOT/'tasks/current'/TASK/'TASK_PACKET.json'
        operation='run_exact78_cards042_direct_object_v2'
        execution_revision='EXACT78_CARDS042_DIRECT_OBJECT_V2'
        receipt_name='EXACT78_CARDS042_DIRECT_OBJECT_V2.json'
        event_prefix='EXACT78_CARDS042_DIRECT_OBJECT_V2'
    state=copy.deepcopy(load_json(TASK_STATE_PATH)); index=idx()
    if state.get('next_task') or any(x.get('execution_allowed') for x in index['task_packets']):
        raise RuntimeError('ACTIVE_TASK_CONFLICT')
    if ROOT.exists() or PACKET.exists(): raise RuntimeError('FRESH_TASK_REQUIRED')
    role_root=REPO_ROOT/'_run/current/exact78_cards042_role_mask_rebuild_v1_20260929/attempts/attempt_0001'
    role_receipt=load_json(REPO_ROOT/'tasks/receipts/EXACT78_CARDS042_ROLE_MASK_REBUILD_V1.json')
    if role_receipt['status']!='PASSED': raise RuntimeError('ROLE_MASK_GRADE_B_REQUIRED')
    role_result=role_root/'runner/sessions'/SESSION/'attempts/attempt_0001/role_mask_output/RESULT.json'
    role=load_json(role_result)
    if role['grade']!='B' or not role['downstream_authorized']: raise RuntimeError('ROLE_MASK_NOT_ADMITTED')
    raw=Path('/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards')/SESSION/f'CameraRecord_{SESSION}.mp4'
    canary=(REPO_ROOT/'archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs'
        /'20260908_nine_hour_156_pipeline_v1/object6d_156_readiness_v1/rgb_object_mask_canary_v3'
        /SESSION/'masks/000125.png')
    code_paths=[
        f'src/chaoyang/ops/{operation}.py',
        'src/chaoyang/pipeline/sam31_compat_adapter_v1.py',
        'src/chaoyang/pipeline/causal_modal_mask_gpu_adapter_v71.py',
        'src/chaoyang/pipeline/exact78_task_runtime_v1.py']
    if args.hybrid_v2: code_paths.append('src/chaoyang/ops/run_exact78_cards042_direct_object_v1.py')
    code=[artifact_ref(REPO_ROOT/p) for p in code_paths]
    inputs=dict(raw_video=artifact_ref(raw),role_result=artifact_ref(role_result),
        role_manifest=role['artifacts']['frame_manifest'],accepted_canary_125=artifact_ref(canary),
        sam_checkpoint=artifact_ref(REPO_ROOT/'assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt'),
        selected_parent=artifact_ref(REPO_ROOT/'_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json'))
    writer=dict(pid=os.getpid(),proc_start_ticks=process_identity(os.getpid())['start_ticks'],executor_epoch=1,
                fencing_token=secrets.token_hex(24),scope=TASK)
    signature=hashlib.sha256(canonical_bytes(dict(inputs=inputs,runtime_code=code,
        config=dict(identity_onset=96,object_id=301,per_frame_state=True,propagation=False)))).hexdigest()
    packet=dict(schema_version='chaoyang-r22-task-packet-v1',task_id=TASK,
        objective='Fresh per-frame SAM3.1 reconstruction of the Cards042 single A-diamond identity mask',
        phase='MASK_OBJECT_IDENTITY_REBUILD',plan_revision='EXACT78_INPAINT_GENERAL_V1',
        execution_revision=execution_revision,session_id=SESSION,
        deadline_at=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
        read_set=['AGENTS.md','docs/current/EXACT78_BASELINE_CLOSURE_REVIEW_ZH.md'],write_set=[str(ROOT)],
        prerequisites=['NO_ACTIVE_WRITER','ROLE_MASK_GRADE_B','PINNED_SAM31','EXTERNAL_RGB_READ_ONLY'],
        required_outputs=['171_frame_manifest','75_direct_sam_masks','full_review_video','quality_result'],
        budgets=dict(workers_max=1,cpu_threads_total=2,gpu_count=1,training_steps=0,wall_seconds_max=3600,session_attempts_max=1),
        stop_conditions=['source_sha_drift','writer_conflict','gpu_lease_unavailable','deadline','quality_C_no_auto_retry'],
        claim_limit='One fresh task-object identity Mask. Old mask is validation only; no Clean, baseline adoption, deletion or training authority.',
        writer=writer,inputs=inputs,runtime_code=code,run_signature=signature,
        commands=[f'chaoyang run {operation} --packet {PACKET}'])
    errors=_validate_packet(packet)
    if errors: raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True); atomic_json(ROOT/'WRITER.json',writer); atomic_json(PACKET,packet)
    now=now_iso(); state['tasks'].append(dict(task_id=TASK,phase=packet['phase'],attempt=1,status='RUNNING',
        session=SESSION,updated_at=now,heartbeat_at=now,pid=writer['pid'],proc_start_ticks=writer['proc_start_ticks'],
        gpu_id=0,task_packet=artifact_ref(PACKET),run_signature=signature,quality='PENDING',adoption='NOT_ADOPTED',review='PENDING'))
    state['next_task']=dict(task_id=TASK,session=SESSION,prerequisites=packet['prerequisites'],
        expected_resource='ONE_GPU_TWO_CPU_THREADS',stop_condition='one terminal attempt or deadline')
    index['task_packets'].append(dict(task_id=TASK,packet_path=str(PACKET.relative_to(REPO_ROOT)),
        packet_sha256=artifact_ref(PACKET)['sha256'],execution_class='CURRENT_LEDGER_ROUTABLE',execution_allowed=True))
    index.update(packet_revision=f'{execution_revision}_RUNNING',status='PASS',supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH),state,event_type=f'{event_prefix}_REGISTERED',
        expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=Path(__file__),
        task_packet_index_path=INDEX,task_packet_index_value=index)
    cache=ROOT/'cache';cache.mkdir()
    env={**os.environ,'PYTHONPATH':str(REPO_ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1',
        **{x:str(cache) for x in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR')},
        **{x:'2' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    command=[sys.executable,'-B','-m','chaoyang.cli','run',operation,'--packet',str(PACKET)]
    with (ROOT/'worker.log').open('x') as log:
        worker=subprocess.Popen(command,cwd=REPO_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic()
        while True:
            try: code_value=worker.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                state=copy.deepcopy(load_json(TASK_STATE_PATH))
                if (state.get('next_task') or {}).get('task_id')!=TASK or time.monotonic()-started>3600:
                    os.killpg(worker.pid,signal.SIGTERM);code_value=worker.wait(timeout=30);break
                row=next(x for x in state['tasks'] if x['task_id']==TASK);row.update(heartbeat_at=now_iso(),updated_at=now_iso())
                publish_bundle(load_json(AUTHORITY_PATH),state,event_type=f'{event_prefix}_HEARTBEAT',
                    expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=Path(__file__),
                    task_packet_index_path=INDEX,task_packet_index_value=idx())
    if not (ROOT/'RESULT.json').is_file(): atomic_json(ROOT/'RESULT.json',dict(status='FAILED_RUNTIME_FINAL',returncode=code_value))
    value=load_json(ROOT/'RESULT.json'); status='PASSED' if code_value==0 else 'FAILED_QUALITY_C' if code_value==2 else 'FAILED_RUNTIME_FINAL'
    receipt=REPO_ROOT/'tasks/receipts'/receipt_name
    atomic_json(receipt,dict(task_id=TASK,status=status,result=artifact_ref(ROOT/'RESULT.json'),deletion_authorized=False,
        baseline_adopted=False,claim_limit=packet['claim_limit']))
    state=copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get('next_task') or {}).get('task_id')!=TASK: raise RuntimeError('FINALIZE_ROUTE_CONFLICT')
    row=next(x for x in state['tasks'] if x['task_id']==TASK);row.update(status=status,pid=None,proc_start_ticks=None,
        updated_at=now_iso(),heartbeat_at=now_iso(),result=artifact_ref(ROOT/'RESULT.json'),terminal_receipt=artifact_ref(receipt),
        quality='GRADE_B_CANDIDATE' if status=='PASSED' else status)
    state['next_task']=None;index=idx()
    for item in index['task_packets']:
        if item['task_id']==TASK:item['execution_allowed']=False
    index.update(packet_revision=f'{execution_revision}_TERMINAL',status='PASS_NO_ACTIVE_TASKS',supersedes_index=artifact_ref(INDEX))
    published=publish_bundle(load_json(AUTHORITY_PATH),state,event_type=f'{event_prefix}_TERMINAL',
        expected_revision=load_json(RECEIPT_PATH)['governance_revision'],generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=index)
    print(json.dumps(dict(status=status,result=artifact_ref(ROOT/'RESULT.json'),revision=published['governance_revision'])))
    return 0 if status=='PASSED' else 1


if __name__=='__main__': raise SystemExit(main())
