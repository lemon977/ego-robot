"""Bounded engineering maintenance. Never publishes production or runs models."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import sys

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/project_convergence_20261009'
LOCK = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
OLD_PROTECTED = ROOT / '_run/maintenance/project_navigation_cleanup_20261009/successor_v2/edits/PROTECTED_INPUTS.json'
DOC_PAGES = ('README.md', 'docs/README.md', 'docs/current/AI_WORK_ENTRY_ZH.md',
             'review/README.md', 'docs/current/README_ZH.md',
             'docs/current/GIT_DELIVERY_ZH.md', 'docs/current/PROJECT_LAYOUT_ZH.md')
DOC_NEW = ('docs/maintenance/PROJECT_CONVENTIONS_V1_ZH.md',
           'docs/maintenance/ENVIRONMENT_HANDOFF_V1_ZH.md')
TOOLS_NEW = ('scripts/maintenance/project_handoff_check_v1.py',
             'scripts/maintenance/run_project_checks_v1.py',
             'tests/helpers/__init__.py', 'tests/helpers/handoff_fixtures_v1.py',
             'tests/maintenance/test_project_handoff_check_v1.py',
             'tests/maintenance/test_project_cpu_profiles_v1.py',
             'Makefile')
CLEANUP_NEW = ('scripts/maintenance/project_cleanup_proof_v2.py',
               'tests/maintenance/test_project_cleanup_proof_v2.py')

def ref(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file():
        raise RuntimeError('REFERENCE_ABSENT_OR_ALIAS:' + str(path))
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    if not path.is_relative_to(BASE) or path.parent.resolve() != path.parent:
        raise RuntimeError('OUTPUT_SCOPE')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and 'sha256' in value and 'bytes' in value:
            yield value
        else:
            for item in value.values(): yield from references(item)
    elif isinstance(value, list):
        for item in value: yield from references(item)

def actual_production():
    index = read(ROOT / 'tasks/current/INDEX.json')
    rows = [row for row in index['task_packets'] if row.get('execution_allowed')]
    if len(rows) > 1: raise RuntimeError('MULTIPLE_ACTIVE_TASKS')
    packets = []
    for row in rows:
        path = ROOT / row['packet_path']
        if ref(path)['sha256'] != row['packet_sha256']: raise RuntimeError('PRODUCTION_PACKET_DRIFT')
        packets.append((path, read(path)))
    return packets

def check_conflicts(names, packets):
    for _, packet in packets:
        bound = {pin['path'] for pin in references(packet)}
        for name in names:
            path = ROOT / name
            if str(path) in bound: raise RuntimeError('PRODUCTION_INPUT_BOUND:' + name)
            for entry in packet.get('write_set', []):
                target = Path(entry) if Path(entry).is_absolute() else ROOT / entry
                if path == target or path.is_relative_to(target):
                    raise RuntimeError('PRODUCTION_WRITE_CONFLICT:' + name)

def main():
    start = datetime.now(timezone(timedelta(hours=8)))
    writer = dict(pid=os.getpid(), startticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                  executor_epoch=4, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  fencing_token=secrets.token_hex(24))
    deadline = start + timedelta(minutes=60)
    out = BASE / ('development_' + start.strftime('%H%M%S') + '_' + writer['fencing_token'][:8])
    out.mkdir(parents=True)
    lock = LOCK.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); json.dump(writer, lock); lock.flush(); os.fsync(lock.fileno())
    packets = actual_production()
    all_names = DOC_PAGES + DOC_NEW + TOOLS_NEW + CLEANUP_NEW
    check_conflicts(all_names, packets)
    for name in DOC_NEW + TOOLS_NEW + CLEANUP_NEW:
        if (ROOT / name).exists(): raise RuntimeError('NEW_OUTPUT_ALREADY_EXISTS:' + name)
    protected = [pin for pin in read(OLD_PROTECTED) if pin['path'] not in {str(ROOT / name) for name in DOC_PAGES}]
    for path, packet in packets:
        protected.append(ref(path))
        for pin in references(packet):
            path = Path(pin['path'])
            if path.is_file() and path.stat().st_size < 10_000_000:
                protected.append(ref(path))
    protected = {pin['path']: pin for pin in protected
                 if not pin['path'].startswith(str(ROOT / 'docs/governance') + '/')
                 and pin['path'] not in {str(ROOT / 'docs/current/STATUS.json'), str(ROOT / 'tasks/current/INDEX.json')}}
    save(out / 'PROTECTED_INPUTS.json', list(protected.values()))
    before = []
    backup = out / 'BEFORE'; backup.mkdir()
    for number, name in enumerate(DOC_PAGES):
        path = ROOT / name; pin = ref(path)
        saved = backup / (str(number) + '.bin')
        with saved.open('xb') as stream: stream.write(path.read_bytes())
        before.append(dict(original=pin, recovery=ref(saved)))
    save(out / 'NAVIGATION_BEFORE.json', before)
    inputs = [ref(Path(__file__)), ref(OLD_PROTECTED),
              ref(ROOT / 'baseline_v1/MANIFEST.json'), ref(ROOT / 'docs/current/DUAL_BRANCH_BASELINE_V1.json')]
    for phase, names in [('N0', DOC_PAGES + DOC_NEW), ('N1', TOOLS_NEW), ('C0_PREPARE', CLEANUP_NEW)]:
        stage = out / phase; stage.mkdir()
        packet = dict(schema='project-convergence-finite-maintenance-v1', task_id=out.name + '_' + phase,
                      phase=phase, authorization='USER_AUTHORIZED_COMPLETE_ENGINEERING_CONVERGENCE_20261009',
                      registered_at=start.isoformat(), deadline_at=deadline.isoformat(), writer=writer,
                      execution_allowed=True, algorithm_execution_allowed=False, deletion_allowed=False,
                      production_governance_write_allowed=False, independent_publisher_allowed=False,
                      read_set=['AGENTS.md','docs/current/AI_WORK_ENTRY_ZH.md',
                                'docs/governance/CURRENT_STATUS_RECEIPT.json','baseline_v1/MANIFEST.json',
                                'docs/current/DUAL_BRANCH_BASELINE_V1.json',str(OLD_PROTECTED),
                                str(out / 'PROTECTED_INPUTS.json'),str(Path(__file__).absolute())],
                      expanded_read_set=['exact before/output files for this phase',
                                         'receipt-bound current maps and packets',
                                         'read-only producer publication code for N0 adoption',
                                         'pytest profiles and synthetic fixtures for N1',
                                         'old cleanup executor and exact 16 roots for C0_PREPARE'],
                      write_set=[str(ROOT / name) for name in names] + [str(stage)],
                      inputs=inputs, budgets=dict(wall_seconds=3600, cpu_threads=1, gpu=0, model_calls=0, training_steps=0),
                      required_outputs=['RESULT.json'],
                      stop_conditions=['input drift','writer conflict','I/O failure','deadline','scope conflict'])
        packet['run_signature'] = hashlib.sha256(json.dumps(packet['inputs'], sort_keys=True).encode()).hexdigest()
        save(stage / 'TASK_PACKET.json', packet)
    print(json.dumps(dict(status='REGISTERED', root=str(out), writer=writer, deadline_at=deadline.isoformat())), flush=True)
    failure = None
    try:
        while datetime.now(timezone(timedelta(hours=8))) < deadline:
            ready, _, _ = select.select([sys.stdin], [], [], 20)
            if ready:
                command = sys.stdin.readline().strip()
                if command != 'FINISH': raise RuntimeError('NO_FINISH')
                break
        else: raise RuntimeError('DEADLINE')
        if read(LOCK) != writer: raise RuntimeError('FENCING_DRIFT')
        for pin in protected.values():
            if ref(pin['path']) != pin: raise RuntimeError('PROTECTED_DRIFT:' + pin['path'])
        for phase in ('N0', 'N1', 'C0_PREPARE'):
            report = read(out / phase / 'PHASE_REPORT.json')
            if report['status'] != 'PASSED': raise RuntimeError('PHASE_NOT_PASSED:' + phase)
            for pin in report['outputs']:
                if ref(pin['path']) != pin: raise RuntimeError('OUTPUT_DRIFT:' + pin['path'])
    except BaseException as exc:
        failure = type(exc).__name__ + ':' + str(exc)
    for phase, names in [('N0', DOC_PAGES + DOC_NEW), ('N1', TOOLS_NEW), ('C0_PREPARE', CLEANUP_NEW)]:
        stage = out / phase
        save(stage / 'RESULT.json', dict(status='PASSED' if failure is None else 'STOPPED', failure=failure,
             task_packet=ref(stage / 'TASK_PACKET.json'),
             outputs=[ref(ROOT / name) for name in names if (ROOT / name).is_file()] +
                     ([ref(stage / 'PHASE_REPORT.json')] if (stage / 'PHASE_REPORT.json').is_file() else []),
             deleted_files=0, production_governance_writes=0, production_signals=0))
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    print(json.dumps(dict(status='PASSED' if failure is None else 'STOPPED', failure=failure, root=str(out))), flush=True)
    return 0 if failure is None else 2

if __name__ == '__main__':
    raise SystemExit(main())
