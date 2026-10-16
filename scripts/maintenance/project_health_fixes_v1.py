"""Bounded maintenance only: storage evidence and future-spec read-set repair.

No model calls, production dispatch, source-algorithm edits, or shared governance
publication. Uses the user-approved isolated maintenance channel. Old packets,
queues and results remain immutable. Generated specs still need normal execution
registration; this program cannot execute them.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import time

ROOT = Path('/path/to/chaoyang')
OUT = ROOT / '_run/maintenance/system_health_20261009'
PLAN = ROOT / 'docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1'
OLD_QUEUE = PLAN / 'reverse_native_boundary_v9/QUEUE_V9_REVISION2.json'
OLD_QUEUE_SHA = 'e99a26a219509afa6d854bb92ca2f93053db9135aa5d51d59237e9675c6cc570'
ACTIVE = {'RUNNING', 'CLAIMED', 'WAIT_GPU_RESOURCE'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def project_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_relative_to(ROOT) or path.resolve() != path:
        raise RuntimeError('OUTSIDE_OR_SYMLINK_PATH:' + str(path))
    return path


def save_new(path, data):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def check_ref(value):
    path = project_path(value['path'])
    if ref(path) != value:
        raise RuntimeError('INPUT_REF_DRIFT:' + str(path))
    return path


def option(args, name):
    if name not in args:
        return None
    if args.count(name) != 1:
        raise RuntimeError('DUPLICATE_OPTION:' + name)
    return args[args.index(name) + 1]


def covered(path, read_set):
    path = project_path(path)
    return any(path == (p := project_path(value)) or p in path.parents for value in read_set)


def normalized_spec(spec):
    """Fix declarations only; retain all algorithms, parameters and budgets."""
    result = copy.deepcopy(spec)
    required = list(spec.get('inputs', []))
    required += [value for name in ['--manifest', '--camera-config', '--mask-seeds', '--reuse-from']
                 if (value := option(spec['args'], name)) is not None]
    current = list(spec['read_set'])
    # Remove the obsolete V8 camera directory only when no actual declared input
    # or command argument requires it. This removes a declaration, not a file.
    current = [value for value in current
               if not ('shared_b_right_fixed_k_installation_v8_' in value
                       and not any(covered(item, [value]) for item in required))]
    for value in required:
        if not covered(value, current):
            current.append(value)
    # Existing parent directory declarations already include their child files.
    current = [value for value in dict.fromkeys(current)
               if not any(project_path(parent) in project_path(value).parents
                          for parent in current if parent != value)]
    if len(current) > 8:
        raise RuntimeError('INITIAL_READ_SET_OVER_EIGHT')
    result['read_set'] = current
    if any(not covered(item, current) for item in required):
        raise RuntimeError('READ_SET_STILL_INCOMPLETE')
    if {k: v for k, v in result.items() if k != 'read_set'} != {
            k: v for k, v in spec.items() if k != 'read_set'}:
        raise RuntimeError('NON_DECLARATION_CHANGE')
    return result


def storage(stage):
    path = stage / 'WRITE_FSYNC_READBACK_PROBE.bin'
    payload = bytes([17, 23, 42, 59]) * 262144
    before = os.statvfs(ROOT)
    started = time.monotonic()
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if digest(path) != hashlib.sha256(payload).hexdigest():
        raise RuntimeError('STORAGE_READBACK_MISMATCH')
    report = {
        'status': 'WRITE_FSYNC_READBACK_PASS', 'probe': ref(path),
        'bytes_tested': len(payload), 'elapsed_seconds': time.monotonic() - started,
        'statvfs_available_before': before.f_bavail * before.f_frsize,
        'statvfs_available_after': (after := os.statvfs(ROOT)).f_bavail * after.f_frsize,
        'df': subprocess.run(['df', '-B1', '/', '/tmp', str(ROOT)], capture_output=True,
                             text=True, check=True).stdout,
        'executor_temp_routing': 'Existing run_worker assigns TMPDIR/TMP/TEMP/XDG_CACHE_HOME to per-attempt project cache; unchanged.',
        'decision': 'DF_ZERO_IS_WARNING_NOT_PROOF_OF_WRITE_FAILURE; actual open/write/fsync/read failure remains a stop condition.',
        'claim_limit': 'Only this 1MiB probe proved writable; not proof of capacity for the full production batch. No disk cleanup or external writes.'}
    save_new(stage / 'STORAGE_REPORT.json', report)


def readsets(stage):
    if digest(OLD_QUEUE) != OLD_QUEUE_SHA:
        raise RuntimeError('OLD_QUEUE_DRIFT')
    queue = read(OLD_QUEUE)
    for value in queue['production_code']:
        check_ref(value)
    notes, changed, specs, rows = [], [], [], []
    for row in [queue['source'], *queue['gates'], *queue['records']]:
        path = check_ref(row['spec']); old = read(path); fixed = normalized_spec(old)
        tid = row['task_id']; terminal = ROOT / f'_run/current/{tid}/attempts/attempt_0001/RESULT.json'
        registered = (ROOT / f'tasks/current/{tid}/TASK_PACKET.json').exists()
        state = read(terminal) if terminal.exists() else None
        passed = bool(state and state['status'] == 'PASSED')
        if registered and not state:
            raise RuntimeError('REGISTERED_WITHOUT_TERMINAL:' + tid)
        if passed:
            check_ref(state['task_packet'])
            if state['validation_errors'] or state['worker_returncode'] != 0:
                raise RuntimeError('PASSED_TERMINAL_NOT_VALID:' + tid)
            notes.append({'task_id': tid, 'original_spec': row['spec'],
                          'terminal': ref(terminal), 'historical_read_set': old['read_set'],
                          'correct_actual_read_set_for_reference': fixed['read_set'],
                          'claim_limit': 'Erratum only, not retroactive change of packet or execution permission.'})
            if row in queue['records']:
                rows.append(copy.deepcopy(row))
            continue
        if row not in queue['records']:
            raise RuntimeError('UPSTREAM_NOT_SEALED_PASSED:' + tid)
        if state and state['status'] != 'FAILED_RUNTIME_FINAL':
            raise RuntimeError('UNKNOWN_TERMINAL:' + tid)
        fresh = tid.removesuffix('_20261009') + '_readset_v1_20261009'
        new_path = PLAN / 'specs' / (fresh + '.json')
        if (ROOT / f'tasks/current/{fresh}').exists() or (ROOT / f'_run/current/{fresh}').exists():
            raise RuntimeError('NEW_TASK_ALREADY_REGISTERED:' + fresh)
        fixed['task_id'] = fresh
        save_new(new_path, fixed)
        new_row = copy.deepcopy(row); new_row.update(task_id=fresh, spec=ref(new_path))
        new_row['supersedes_planned_spec'] = row['spec']
        if state:
            new_row['failed_predecessor_not_reused'] = ref(terminal)
        rows.append(new_row); specs.append(new_path)
        changed.append({'old_task_id': tid, 'new_task_id': fresh, 'spec': ref(new_path),
                        'read_set_count': len(fixed['read_set']), 'all_literal_inputs_covered': True,
                        'old_failed_partial_consumed': False})
    new_queue = copy.deepcopy(queue)
    new_queue.update(records=rows, supersedes_queue=ref(OLD_QUEUE),
                     status='READ_SET_REPAIRED_FUTURE_SPECS_NOT_REGISTERED_OR_EXECUTING')
    new_queue['read_set_repair_only'] = True
    save_new(stage / 'READ_SET_ERRATA.json', {'historical_tasks': notes,
        'claim_limit': 'Signed input paths were correct; old declaration mismatch preserved, not rewritten as historical PASS.'})
    new_queue['historical_packet_errata'] = ref(stage / 'READ_SET_ERRATA.json')
    qpath = PLAN / 'reverse_native_boundary_v9/QUEUE_V9_REVISION2_READSET_V1.json'
    save_new(qpath, new_queue)
    checks = []
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1',
           'TMPDIR': str(stage), 'XDG_CACHE_HOME': str(stage), 'CUDA_VISIBLE_DEVICES': ''}
    for path in specs:
        remaining = (datetime.fromisoformat(read(stage / 'TASK_PACKET.json')['deadline_at'])
                     - datetime.now(ZoneInfo('Asia/Shanghai'))).total_seconds()
        if remaining <= 0:
            raise RuntimeError('DEADLINE')
        check = subprocess.run(['/usr/local/bin/python', '-B', '-m',
            'chaoyang.governance.execute_pick_place_1008_v1', '--check-only', str(path)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=min(180, remaining))
        report = json.loads(check.stdout) if check.returncode == 0 else {'stderr': check.stderr[-1500:]}
        checks.append({'spec': ref(path), 'returncode': check.returncode, 'report': report})
        print(json.dumps({'stage': 'OFFICIAL_CHECK_ONLY', 'task': path.stem,
                          'returncode': check.returncode, 'status': report.get('status')}), flush=True)
        if check.returncode or report.get('status') != 'DRY_RUN_READY':
            save_new(stage / 'OFFICIAL_CHECK_ONLY.json', checks)
            raise RuntimeError('OFFICIAL_CHECK_NOT_READY:' + str(path))
    save_new(stage / 'OFFICIAL_CHECK_ONLY.json', checks)
    save_new(stage / 'READ_SET_REPAIR_REPORT.json', {
        'queue': ref(qpath), 'new_specs': changed, 'historical_packets_unchanged': True,
        'algorithm_code_unchanged': True, 'inputs_parameters_budgets_unchanged': True,
        'new_tasks_registered_or_started': 0, 'official_checks_passed': len(checks),
        'remaining_sessions': sum(len(row['sessions']) for row in rows if row['task_id'] in {x['new_task_id'] for x in changed}),
        'next_action': 'Production owner must explicitly adopt this successor queue after verifying completed predecessors; do not restart the old pinned dispatcher unchanged.'})


def git_window(stage):
    """Keep one finite maintenance identity while the coordinator edits/stages."""
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()
    save_new(stage / 'GIT_BEFORE.json', {'head': head,
        'ignore': ref(ROOT / '.gitignore'), 'no_reset_no_clean_no_push': True})
    print('READY_GIT_MAINTENANCE', flush=True)
    deadline = datetime.fromisoformat(read(stage / 'TASK_PACKET.json')['deadline_at'])
    while datetime.now(ZoneInfo('Asia/Shanghai')) < deadline:
        ready, _, _ = select.select([__import__('sys').stdin], [], [], 30)
        if not ready:
            print('GIT_MAINTENANCE_ALIVE', flush=True)
            continue
        command = __import__('sys').stdin.readline().strip()
        if command != 'FINISH':
            raise RuntimeError('ABORTED_OR_INVALID_MAINTENANCE_COMMAND')
        new_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True).stdout.strip()
        if new_head != head:
            parent = subprocess.run(['git', 'rev-parse', new_head + '^'], cwd=ROOT,
                env=env, capture_output=True, text=True, check=True).stdout.strip()
            if parent != head:
                raise RuntimeError('UNEXPECTED_GIT_HEAD_CHAIN')
        save_new(stage / 'GIT_AFTER.json', {'old_head': head, 'new_head': new_head,
            'ignore': ref(ROOT / '.gitignore'), 'pushed': False,
            'claim_limit': 'Local checkpoint and tracking hygiene only; not baseline/quality promotion.'})
        return
    raise RuntimeError('DEADLINE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['storage', 'readsets', 'git'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
    with lock_path.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = OUT / args.stage; stage.mkdir()
        raw = Path(f'/proc/{os.getpid()}/stat').read_text(); fields = raw[raw.rfind(')') + 2:].split()
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        packet = {'schema': 'isolated-bounded-maintenance-task-v1',
            'task_id': 'system_health_' + args.stage + '_v1_20261009',
            'writer': {'pid': os.getpid(), 'startticks': int(fields[19]), 'executor_epoch': 1,
                       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       'fencing_token': secrets.token_hex(24)},
            'registered_at': now.isoformat(), 'deadline_at': (now + timedelta(minutes=30)).isoformat(),
            'authorization': 'User asked to fix items1/2/4 and allowed independent maintenance channel; production and deletion proof unchanged.',
            'production_governance_write_allowed': False,
            'read_set': ['AGENTS.md', 'docs/current/AI_WORK_ENTRY_ZH.md', 'docs/current/PROJECT_USAGE_V1_ZH.md',
                         'docs/governance/CURRENT_STATUS_RECEIPT.json', 'baseline_v1/MANIFEST.json',
                         str(OLD_QUEUE), str(Path(__file__).resolve()), '/proc'],
            'expanded_read_set': ['spec refs/terminal refs bound by the frozen queue', 'current Git metadata'],
            'write_set': ([str(stage), '.gitignore', '.git/index and one local checkpoint',
                           'docs/current/GIT_DELIVERY_ZH.md'] if args.stage == 'git' else
                          [str(stage), 'new unregistered specs ending _readset_v1_20261009.json',
                           str(PLAN / 'reverse_native_boundary_v9/QUEUE_V9_REVISION2_READSET_V1.json')]),
            'inputs': [ref(Path(__file__)), ref(OLD_QUEUE), ref(ROOT / 'docs/current/DUAL_BRANCH_BASELINE_V1.json')],
            'budgets': {'cpu_threads': 1, 'gpu': 0, 'model_calls': 0, 'training_steps': 0,
                        'probe_bytes': 1048576, 'wall_seconds': 1800},
            'required_outputs': ['RESULT.json'],
            'stop_conditions': ['signature drift', 'existing output', 'maintenance writer conflict', 'I/O failure', 'deadline']}
        packet['run_signature'] = hashlib.sha256(json.dumps(packet['inputs'], sort_keys=True).encode()).hexdigest()
        save_new(stage / 'TASK_PACKET.json', packet)
        failure = None
        try:
            {'storage': storage, 'readsets': readsets, 'git': git_window}[args.stage](stage)
            if datetime.now(ZoneInfo('Asia/Shanghai')) >= datetime.fromisoformat(packet['deadline_at']):
                raise RuntimeError('DEADLINE')
            for value in packet['inputs']:
                check_ref(value)
        except BaseException as exc:
            failure = repr(exc)
        save_new(stage / 'RESULT.json', {'status': 'PASSED' if failure is None else 'STOPPED',
            'failure': failure, 'packet': ref(stage / 'TASK_PACKET.json'),
            'outputs': [ref(path) for path in stage.iterdir() if path.is_file() and path.name != 'RESULT.json'],
            'models_run': 0, 'production_started': False, 'files_deleted': 0,
            'completed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
        print(json.dumps({'stage': args.stage, 'status': 'PASSED' if failure is None else 'STOPPED',
                          'failure': failure, 'result': ref(stage / 'RESULT.json')}), flush=True)
        if failure:
            raise RuntimeError(failure)


if __name__ == '__main__':
    main()
