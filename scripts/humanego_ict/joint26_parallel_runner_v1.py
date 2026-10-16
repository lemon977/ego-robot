#!/usr/bin/env python3
"""One finite fenced coordinator, two simultaneous disjoint-GPU ICT lanes."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from chaoyang.governance.common import artifact_ref, atomic_json, canonical_bytes, load_json, now_iso, process_identity
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.humanego_ict.joint26_console import ConsoleReporter
from chaoyang.humanego_ict.joint26_limits import normalize_wall_seconds
from chaoyang.humanego_ict.joint26_contract import PARALLEL_MODE, PARALLEL_ROUTE, TRAIN, candidate_contract
from chaoyang.humanego_ict.project_mount_paths import reference_equal, resolve_input_path

BRANCHES = ('raw', 'robotized')
CLI = 'chaoyang.humanego_ict.joint26_cli'


def require(value, message):
    if not value:
        raise ValueError(message)


def supervise_lanes(launches, *, deadline, on_heartbeat, on_finish, stop_child,
                    signal_state, heartbeat_interval=30.0, on_progress=None):
    """Launch both before waiting; an ordinary lane failure does not stop peer."""
    require(set(launches) == set(BRANCHES), 'EXACT_TWO_LANES_REQUIRED')
    children, logs, rows = {}, {}, {}

    def finish(branch, extra_error=None):
        child = children.get(branch)
        if child is not None:
            stop_child(child)  # Also quiesce any orphaned rank process group.
        rc = child.poll() if child is not None else 127
        rows[branch] = {'returncode': rc if rc is not None else 143,
                        'errors': [] if extra_error is None else [extra_error]}
        try:
            on_finish(branch, rows[branch]['returncode'])
        except Exception as error:
            rows[branch]['errors'].append('FINISH_CALLBACK:' + repr(error))

    try:
        for branch in BRANCHES:
            lane = launches[branch]
            try:
                logs[branch] = Path(lane['log_path']).open('xb')
                children[branch] = subprocess.Popen(lane['command'], cwd=lane['cwd'], env=lane['env'],
                    stdout=logs[branch], stderr=subprocess.STDOUT, start_new_session=True)
            except Exception as error:
                finish(branch, 'SPAWN:' + repr(error))
                for peer in BRANCHES:
                    if peer not in rows:
                        finish(peer, 'PEER_SPAWN_FAILED')
                return rows
        last, last_progress = 0.0, 0.0
        while len(rows) < len(BRANCHES):
            if signal_state or (deadline is not None and datetime.now(timezone.utc) >= deadline):
                reason = 'COORDINATOR_SIGNAL' if signal_state else 'FINITE_DEADLINE'
                for branch in BRANCHES:
                    if branch not in rows:
                        finish(branch, reason)
                break
            for branch in BRANCHES:
                if branch not in rows and children[branch].poll() is not None:
                    finish(branch)
            if any(any(e.startswith('FINISH_CALLBACK:') for e in row['errors']) for row in rows.values()):
                for branch in BRANCHES:
                    if branch not in rows:
                        finish(branch, 'PEER_FINISH_CALLBACK_FAILED')
                break
            if len(rows) == len(BRANCHES):
                break
            if on_progress is not None and time.monotonic() - last_progress >= 1.0:
                try:
                    on_progress({branch: child.pid for branch, child in children.items() if branch not in rows})
                except Exception:
                    # Display errors cannot stop either lane or affect leases.
                    pass
                last_progress = time.monotonic()
            if time.monotonic() - last >= heartbeat_interval:
                try:
                    on_heartbeat({branch: child.pid for branch, child in children.items() if branch not in rows})
                except Exception as error:
                    for branch in BRANCHES:
                        if branch not in rows:
                            finish(branch, 'HEARTBEAT:' + repr(error))
                    break
                last = time.monotonic()
            time.sleep(0.05)
    finally:
        for branch, child in children.items():
            if child.poll() is None:
                stop_child(child)
        for log in logs.values():
            log.close()
    return rows


def stop_owned_group(child):
    """Only the process group created by our Popen(start_new_session=True)."""
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
    # Parent exit does not prove all descendants exited. The owned PGID never
    # aliases a fresh process while any old rank still belongs to that group.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parallel', action='store_true', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--raw-gpu-ids', required=True)
    p.add_argument('--robotized-gpu-ids', required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--raw-steps', type=int)
    p.add_argument('--robotized-steps', type=int)
    p.add_argument('--raw-config', type=Path)
    p.add_argument('--robotized-config', type=Path)
    p.add_argument('--global-batch', type=int, help='每分支全局batch；当前生产配置须为64')
    p.add_argument('--micro-batch', '--batch-size', dest='micro_batch', type=int,
                   help='两个分支每卡每次前向batch；显式覆盖各分支config')
    p.add_argument('--wall-seconds', type=int, default=None,
                   help='可选墙钟上限；默认或 0 仅按有限训练步数结束')
    p.add_argument('--output', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--json', action='store_true', help='输出原机器 JSON，关闭中文阶段与进度提示')
    return p.parse_args(argv)


def read_lane_result(directory, branch, expected_updates, infra):
    output = directory / 'lanes' / branch / 'output'
    stage = load_json(output / 'TRAINING_RESULT.json')
    require(stage.get('branch') == branch and stage.get('optimizer_updates') == expected_updates
            and stage.get('requested_optimizer_updates') == expected_updates
            and stage.get('completed_update_budget') is True, 'EXACT_LANE_UPDATES_NOT_COMPLETE')
    require(Path(stage['last_checkpoint']) == output / f'checkpoint_{expected_updates:06d}.pt',
            'LANE_CHECKPOINT_OUTPUT_BINDING_DRIFT')
    exported = load_json(output / 'inference_tensors/EXPORT_MANIFEST.json')
    require(exported.get('optimizer_updates') == expected_updates
            and exported.get('requested_optimizer_updates') == expected_updates, 'LANE_EXPORT_UPDATES_DRIFT')
    checkpoint = artifact_ref(Path(stage['last_checkpoint']))
    infra._verify(checkpoint)
    refs = [artifact_ref(output / 'TRAINING_RESULT.json'), checkpoint,
            artifact_ref(output / 'inference_tensors/EXPORT_MANIFEST.json')]
    for ref in exported['files'].values():
        infra._verify(ref)
        refs.append(ref)
    return refs


def execute(args):
    # Import-only bootstrap is shared with the unchanged single-branch entry.
    import joint26_runner_v1 as single
    from chaoyang.humanego_ict.joint26_parallel_resources import ParallelGpuLeases, validate_gpu_groups
    from chaoyang.humanego_ict.joint26_parallel_admission import lane_signature
    from chaoyang.humanego_ict.joint26_training import preflight
    from chaoyang.humanego_ict.joint26_runtime import _ordinary

    json_mode = getattr(args, 'json', False)
    reporter = ConsoleReporter(json_mode=json_mode)
    groups = validate_gpu_groups({'raw': args.raw_gpu_ids, 'robotized': args.robotized_gpu_ids})
    wall_seconds = normalize_wall_seconds(args.wall_seconds)
    termination_policy = 'OPTIMIZER_UPDATES_ONLY' if wall_seconds is None else 'OPTIMIZER_UPDATES_OR_DEADLINE'
    configs, cfgs = {}, {}
    for branch in BRANCHES:
        steps = getattr(args, branch + '_steps')
        cfgs[branch], configs[branch] = single.resolve_training_config(SimpleNamespace(
            steps=args.steps if steps is None else steps, config=getattr(args, branch + '_config'),
            nproc=len(groups[branch]), wall_seconds=wall_seconds,
            global_batch=getattr(args, 'global_batch', None),
            micro_batch=getattr(args, 'micro_batch', None)))
    manifest = _ordinary(args.manifest.absolute())
    qualifications, validation_counts = {}, {}
    for branch in BRANCHES:
        reporter.stage(branch, '预检训练与验证数据，核对输入依赖。')
        started = time.monotonic()
        train, validation, q = preflight(manifest, branch, world_size=len(groups[branch]))
        qualifications[branch] = dict(q, elapsed_seconds=round(time.monotonic() - started, 3),
                                      gpu_ids=groups[branch], requested_updates=cfgs[branch].max_updates)
        qualifications[branch]['training_topology'] = {
            'world_size': len(groups[branch]), 'global_batch': cfgs[branch].global_batch,
            'micro_batch': cfgs[branch].micro_batch,
            'gradient_accumulation': cfgs[branch].global_batch // (len(groups[branch]) * cfgs[branch].micro_batch)}
        qualifications[branch]['python'] = sys.executable
        validation_counts[branch] = len(validation)
        if not json_mode:
            reporter.preflight(branch, qualifications[branch])
        del train, validation
    precheck = {'status': 'PASS_BOTH_BRANCHES_CPU_ONLY', 'lanes': qualifications}
    if args.preflight_only:
        ConsoleReporter(json_mode=True).preflight('parallel', precheck)
        return 0
    require(args.output is not None, 'NEW_PROJECT_OUTPUT_DIRECTORY_REQUIRED')
    directory = _ordinary(args.output.absolute(), existing=False)
    require(not directory.exists(), 'OUTPUT_ALREADY_EXISTS_DUPLICATE_SUBMISSION_REJECTED')
    reporter.stage('parallel', '校验治理状态与登记条件。')
    single._preflight(['flock', '-s', str(ROOT / '_run/current/locks/governance.lock'),
                      sys.executable, '-B', '-m', CLI, 'validate-governance'])
    infra = single._infra()
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    closure = candidate_contract(ROOT).get(PARALLEL_ROUTE, {}).get('code_closure', [])
    admitted = load_json(resolve_input_path(receipt['files']['algorithm_contract']['path'], ROOT))
    current = admitted.get('special_status_contracts', {}).get(PARALLEL_ROUTE, {}).get('code_closure', [])
    require(bool(closure) and len(current) == len(closure)
            and all(reference_equal(a,b,ROOT) for a,b in zip(closure,current)), 'PARALLEL_ROUTE_CLOSURE_NOT_CURRENT')
    task_id = 'pick_place_humanego_ict_joint26_parallel_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ').lower() + '_' + secrets.token_hex(3)
    packet_path = ROOT / 'tasks/current' / task_id / 'TASK_PACKET.json'
    writer = infra._writer_identity(task_id, 1)
    identity = process_identity(writer['pid'])
    require(identity['alive'] and identity['start_ticks'] == writer['proc_start_ticks'], 'PROCFS_IDENTITY_VIEW_CONFLICT')
    start = datetime.now(timezone.utc)
    deadline = None if wall_seconds is None else start + timedelta(seconds=wall_seconds)
    directory.mkdir(parents=True, exist_ok=False)
    infra._json_once(directory / 'PRECHECK.json', precheck)
    reporter.stage('parallel', '检查输出盘容量，准备两分支训练。')
    measure_storage, _ = infra._existing_helpers()
    storage = measure_storage(directory, task_id)
    lanes = {}
    lease_options = ({'step_only_guard': True} if deadline is None else
                     {'guard_expires_at': deadline + timedelta(seconds=120)})
    pool = ParallelGpuLeases(ROOT, writer, task_id, groups, **lease_options)
    for branch in BRANCHES:
        require('warmstart_ref' not in configs[branch], 'PARALLEL_WARMSTART_NOT_SUPPORTED_USE_SINGLE_BRANCH')
        lane_dir = directory / 'lanes' / branch
        lane_dir.mkdir(parents=True, exist_ok=False)
        config_ref = infra._json_once(lane_dir / 'TRAINING_CONFIG.json', configs[branch])
        inputs = {'paired_manifest': artifact_ref(manifest), 'training_config': config_ref}
        output = lane_dir / 'output'
        cmdargs = ['train', '--task-packet', str(packet_path), '--manifest', str(manifest),
                   '--config', config_ref['path'], '--output', str(output), '--branch', branch]
        command = [sys.executable, '-B', '-m', CLI, 'run', TRAIN, *cmdargs]
        cfg = cfgs[branch]
        lane = {'branch': branch, 'inputs': inputs, 'output_dir': str(output),
                'gpu_ids': groups[branch], 'gpu_lease_paths': [str(pool.lease_paths[i]) for i in groups[branch]],
                'budgets': {'wall_seconds_max': wall_seconds, 'gpu_count': len(groups[branch]),
                            'optimizer_updates_max': cfg.max_updates, 'cpu_threads_per_worker': 2,
                            'model_calls': cfg.max_updates * (cfg.global_batch // cfg.micro_batch)
                            + ((cfg.max_updates+cfg.validation_every-1)//cfg.validation_every) * (validation_counts[branch]+len(groups[branch]))},
                'commands': [shlex.join(command)]}
        lane['lane_signature'] = lane_signature(lane)
        lanes[branch] = lane
    packet = {'schema_version': 'chaoyang-r22-task-packet-v1', 'task_id': task_id,
        'objective': 'Simultaneous independent Raw and Robotized ICT training with one fenced publisher',
        'phase': PARALLEL_MODE, 'plan_revision': 'HUMANEGO_ICT_JOINT26_V1', 'execution_revision': task_id.upper(),
        'execution_mode': PARALLEL_MODE, 'route_name': PARALLEL_ROUTE, 'lanes': lanes,
        'read_set': ['AGENTS.md','docs/governance/CURRENT_STATUS_RECEIPT.json','docs/governance/ALGORITHM_CONTRACT.json','tasks/current/INDEX.json'],
        'read_set_expansions': ['Exact signed lane inputs/configuration/runtime closure only'],
        'write_set': [str(directory.relative_to(ROOT)), str(packet_path.relative_to(ROOT)),
                      'tasks/current/INDEX.json','docs/governance','_run/current/GPU_LEASE.json','_run/current/GPU_LEASE.lock',
                      '_run/current/ICT_GPU_LEASES','_run/current/ICT_SUBMIT.lock',
                      'docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json'],
        'prerequisites': ['GOVERNANCE_PASS','NO_ACTIVE_WRITER','EXACT_PARALLEL_ROUTE','DISJOINT_GPU_GROUPS'],
        'required_outputs': ['RESULT.json','POST_VALIDATION.json','lanes/raw/RESULT.json','lanes/robotized/RESULT.json'],
        'budgets': {'wall_seconds_max': wall_seconds,'gpu_count': sum(map(len,groups.values())),
                    'optimizer_updates_max': sum(c.max_updates for c in cfgs.values()),
                    'model_calls': sum(l['budgets']['model_calls'] for l in lanes.values()),'cpu_threads_per_worker':2,'robot_commands_max':0},
        'termination_policy': termination_policy,
        'stop_conditions': (['deadline'] if deadline is not None else [])
                           + ['optimizer update budget','signal','writer conflict','source/closure drift','lease drift'],
        'claim_limit': 'Derived research ICT training, not physical control approval; CPU tests are not multi-H100 CUDA proof',
        'inputs': {'paired_manifest':artifact_ref(manifest),'storage_capacity_observation':storage},
        'runtime_code':closure,'writer':writer,'writer_file':str(directory/'WRITER.json'),
        'packet_path':str(packet_path),'output_dir':str(directory),
        'commands':[lane['commands'][0] for lane in lanes.values()],
        'execution_started_at':start.isoformat(),'deadline_at':None if deadline is None else deadline.isoformat(),
        'allocated_gpu_ids':groups['raw']+groups['robotized'], 'gpu_lease_path':str(pool.central_path),
        'gpu_lease_verified':True, 'control_ground_truth':False,'physical_deployment_authorized':False}
    packet['run_signature'] = hashlib.sha256(canonical_bytes(packet)).hexdigest()
    require(not _validate_packet(packet), 'INVALID_FINITE_PARALLEL_PACKET')
    env_by_branch = {}
    for branch in BRANCHES:
        env = infra._environment(directory / 'lanes' / branch)
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,groups[branch])),MUJOCO_GL='disable',PYTHONUNBUFFERED='1')
        env_by_branch[branch] = env
    for key in ('TMPDIR','TMP','TEMP','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','XDG_CACHE_HOME',
                'NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','HF_HOME','HUGGINGFACE_HUB_CACHE','MPLCONFIGDIR'):
        os.environ[key] = env_by_branch['raw'][key]
    registered = False
    submit_lock = ROOT / '_run/current/ICT_SUBMIT.lock'
    errors, result_refs = [], {}
    try:
        reporter.stage('parallel', '申请两分支 GPU 租约并登记任务。')
        with submit_lock.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            receipt,state,index = infra._snapshot()
            infra._no_active_writer(state,index)
            pool.acquire()
            packet_path.parent.mkdir(parents=True,exist_ok=False)
            packet_ref = infra._json_once(packet_path,packet)
            infra._json_once(directory/'WRITER.json',writer)
            infra._json_once(directory/'FROZEN_INPUTS.json',packet['inputs'])
            state['tasks'].append({'task_id':task_id,'phase':PARALLEL_MODE,'attempt':1,'status':'RUNNING',
                'updated_at':now_iso(),'heartbeat_at':now_iso(),'pid':writer['pid'],'proc_start_ticks':writer['proc_start_ticks'],
                'task_packet':packet_ref,'run_signature':packet['run_signature']})
            state['next_task']={'task_id':task_id,'expected_resource':'DISJOINT_GPU_TWO_TRAINING_LANES'}
            index['task_packets'].append({'task_id':task_id,'packet_path':str(packet_path.relative_to(ROOT)),
                'packet_sha256':packet_ref['sha256'],'execution_allowed':True,'execution_class':'CURRENT_LEDGER_ROUTABLE'})
            infra._publish(state,index,'ICT_JOINT26_PARALLEL_REGISTERED',expected_revision=receipt['governance_revision'])
            registered = True
        reporter.registered('parallel', groups, {b: cfgs[b].max_updates for b in BRANCHES}, directory,
            metadata={'status':'REGISTERED_PARALLEL','task_id':task_id,'deadline_at':packet['deadline_at'],
                      'lanes':{b:{'gpu_ids':groups[b],'output':lanes[b]['output_dir']} for b in BRANCHES}})
        launches = {}
        for branch in BRANCHES:
            cfgref = lanes[branch]['inputs']['training_config']['path']
            args_train = ['train','--task-packet',str(packet_path),'--manifest',str(manifest),
                          '--config',cfgref,'--output',lanes[branch]['output_dir'],'--branch',branch]
            launches[branch] = {'command':[sys.executable,'-B','-m','torch.distributed.run','--standalone',
                '--nnodes=1','--nproc_per_node='+str(len(groups[branch])),'--module',CLI,'run',TRAIN,*args_train],
                'env':env_by_branch[branch],'cwd':ROOT,'log_path':directory/'lanes'/branch/'EXECUTION.log'}
        lane_states = {b:{'status':'STARTING','gpu_ids':groups[b],'output':lanes[b]['output_dir']} for b in BRANCHES}

        def progress(worker_pids):
            for branch in worker_pids:
                reporter.progress(branch, Path(lanes[branch]['output_dir']), cfgs[branch].max_updates)

        def beat(worker_pids):
            pool.beat(worker_pids)
            for branch,pid in worker_pids.items():
                lane_states[branch].update(status='RUNNING',pid=pid,heartbeat_at=now_iso(),
                    proc_start_ticks=process_identity(pid)['start_ticks'])
            atomic_json(directory/'LANE_STATE.json',lane_states)
            receipt,state,index = infra._snapshot()
            row,_ = infra._own_row(packet,state,index)
            row.update(heartbeat_at=now_iso(),updated_at=now_iso(),lane_statuses=lane_states)
            infra._publish(state,index,'ICT_JOINT26_PARALLEL_HEARTBEAT',expected_revision=receipt['governance_revision'])

        def finish(branch,returncode):
            lane_errors = [] if returncode == 0 else ['CHILD_EXIT_'+str(returncode)]
            refs = []
            if returncode == 0:
                try:
                    refs = read_lane_result(directory,branch,cfgs[branch].max_updates,infra)
                except Exception as error:
                    lane_errors.append('RESULT_READBACK:'+repr(error))
            result = {'schema':'humanego-joint26-parallel-lane-result-v1','task_id':task_id,'branch':branch,
                      'run_signature':packet['run_signature'],'lane_signature':lanes[branch]['lane_signature'],
                      'status':'PASSED' if not lane_errors else 'FAILED_RUNTIME_FINAL','returncode':returncode,
                      'errors':lane_errors,'artifacts':refs,'requested_optimizer_updates':cfgs[branch].max_updates,
                      'control_ground_truth':False,'physical_deployment_authorized':False,'completed_at':now_iso()}
            result_refs[branch] = infra._json_once(directory/'lanes'/branch/'RESULT.json',result)
            lane_states[branch].update(status=result['status'],result=result_refs[branch],pid=None,heartbeat_at=now_iso())
            atomic_json(directory/'LANE_STATE.json',lane_states)
            if not json_mode:
                try:
                    stage = None
                    try:
                        stage = load_json(Path(lanes[branch]['output_dir']) / 'TRAINING_RESULT.json')
                    except (OSError, ValueError):
                        pass
                    reporter.progress(branch, Path(lanes[branch]['output_dir']), cfgs[branch].max_updates, final=True)
                    reporter.finished(branch, result['status'], result, Path(lanes[branch]['output_dir']),
                                      cfgs[branch].max_updates, stage=stage)
                except Exception:
                    # A terminal failure cannot turn a sealed lane into a
                    # finish callback failure or stop its running peer.
                    pass

        stopped = []
        handlers = {n:signal.signal(n,lambda n,f:stopped.append(n)) for n in (signal.SIGINT,signal.SIGTERM)}
        try:
            for branch in BRANCHES:
                reporter.stage(branch, '启动训练进程，等待首条训练指标。')
            rows = supervise_lanes(launches,deadline=deadline,on_heartbeat=beat,on_finish=finish,
                                   stop_child=stop_owned_group,signal_state=stopped,on_progress=progress)
            for branch,row in rows.items():
                errors.extend(branch+':'+e for e in row['errors'])
            require(set(result_refs)==set(BRANCHES),'BOTH_LANE_RESULTS_REQUIRED')
        finally:
            for number,handler in handlers.items():
                signal.signal(number,handler)
    except BaseException as error:
        errors.append(type(error).__name__+':'+str(error))
    finally:
        try:
            pool.release_all('JOINT26_PARALLEL_FINITE_EXIT')
        except Exception as error:
            errors.append('LEASE_RELEASE:'+repr(error))
    if not registered:
        infra._json_once(directory/'REGISTRATION_FAILURE.json',{'task_id':task_id,'errors':errors,'algorithm_started':False})
        reporter.stage('parallel', '任务登记失败，详见 REGISTRATION_FAILURE.json。')
        return 2
    statuses = {b:load_json(Path(ref['path']))['status'] for b,ref in result_refs.items()}
    result = {'schema':'humanego-joint26-parallel-finite-result-v1','task_id':task_id,
        'run_signature':packet['run_signature'],'status':'PASSED' if not errors and all(statuses.get(b)=='PASSED' for b in BRANCHES) else 'FAILED_RUNTIME_FINAL',
        'lanes':result_refs,'lane_statuses':statuses,'errors':errors,'completed_at':now_iso(),
        'control_ground_truth':False,'physical_deployment_authorized':False}
    receipt,state,index=infra._snapshot()
    row,pointer=infra._own_row(packet,state,index)
    result_ref=infra._json_once(directory/'RESULT.json',result)
    row.update(status=result['status'],result=result_ref,pid=None,proc_start_ticks=None,heartbeat_at=now_iso(),updated_at=now_iso())
    pointer.update(execution_allowed=False,execution_class='TERMINAL_LEDGER_EVIDENCE')
    state['next_task']=None
    handoff_path=ROOT/'docs/plans/PICK_PLACE_HUMANEGO_ICT_JOINT26_V1/STATUS.json'
    handoff=load_json(handoff_path)
    handoff.update(last_task_id=task_id,last_task_result=result_ref,last_task_status=result['status'])
    handoff['last_parallel_lane_results']=result_refs
    for branch in BRANCHES:
        if statuses.get(branch)=='PASSED':
            stage=load_json(directory/'lanes'/branch/'output/TRAINING_RESULT.json')
            handoff[branch+'_joint26_checkpoint']=artifact_ref(Path(stage['last_checkpoint']))
            handoff.setdefault('optimizer_updates',{})[branch]=stage['optimizer_updates']
            handoff.setdefault('requested_optimizer_updates',{})[branch]=cfgs[branch].max_updates
    atomic_json(handoff_path,handoff)
    infra._publish(state,index,'ICT_JOINT26_PARALLEL_SEALED',expected_revision=receipt['governance_revision'])
    check=subprocess.run([sys.executable,'-B','-m',CLI,'validate-governance'],cwd=ROOT,
                         capture_output=True,text=True,timeout=300)
    infra._json_once(directory/'POST_VALIDATION.json',{'exit_code':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
    reporter.finished('parallel', result['status'], result, directory, None, governance_exit=check.returncode)
    return 0 if result['status']=='PASSED' and check.returncode==0 else 2


def main(argv=None):
    return execute(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
