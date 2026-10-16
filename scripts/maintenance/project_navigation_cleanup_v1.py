"""Finite independent documentation/test maintenance; never publishes production."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/path/to/chaoyang')
OUT = ROOT / '_run/maintenance/project_navigation_cleanup_20261009'
LOCK = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
DOC_WRITES = (
    'docs/maintenance/README.md', 'review/NAVIGATION_ZH.md',
    'docs/maintenance/NAVIGATION_PUBLISH_HANDOFF_20261009.json',
)
TEST_WRITES = (
    'tests/unit/test_baseline_v1_source_release.py',
    'tests/unit/test_project_slim_v1.py',
    'tests/unit/test_frozen_bundle_runtime_separation_v1.py',
    'tests/conftest.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path)
    content = path.read_bytes()
    return {'path': str(path), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}


def save(path, value):
    path = Path(path)
    if not path.is_relative_to(OUT) or path.parent.resolve() != path.parent:
        raise RuntimeError('OUTPUT_SCOPE_ESCAPE')
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


def stable_refs():
    receipt_path = ROOT / 'docs/governance/CURRENT_STATUS_RECEIPT.json'
    for _ in range(3):
        raw = receipt_path.read_bytes(); receipt = json.loads(raw)
        authority_ref = receipt['files']['doc_authority_map']
        path = Path(authority_ref['path'])
        if ref(path) != authority_ref:
            continue
        authority = read(path)
        if raw == receipt_path.read_bytes():
            return receipt, authority
    raise RuntimeError('GOVERNANCE_SNAPSHOT_DRIFT')


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and 'sha256' in value and 'bytes' in value:
            yield value
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['edits'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with LOCK.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = OUT / args.stage; stage.mkdir()
        receipt, authority = stable_refs()
        index = read(ROOT / 'tasks/current/INDEX.json')
        rows = [row for row in index['task_packets'] if row.get('execution_allowed')]
        if len(rows) > 1:
            raise RuntimeError('MULTIPLE_PRODUCTION_WRITERS')
        packets = [ROOT / row['packet_path'] for row in rows]
        for row, path in zip(rows, packets):
            if ref(path)['sha256'] != row['packet_sha256']:
                raise RuntimeError('PRODUCTION_PACKET_DRIFT')
        bound = {str(Path(d['path'])) for d in authority['documents']}
        edit_paths = [ROOT / name for name in DOC_WRITES + TEST_WRITES]
        protected = {}
        for item in references(authority):
            path = Path(item['path'])
            if path.is_file() and path.stat().st_size < 10_000_000:
                protected[str(path)] = ref(path)
        manifest = read(ROOT / 'baseline_v1/MANIFEST.json')
        for row in manifest['sources'] + manifest['metadata']:
            path = ROOT / 'baseline_v1' / row['bundle_path']
            if ref(path)['sha256'] != row['source']['sha256']:
                raise RuntimeError('FROZEN_BUNDLE_DRIFT')
            protected[str(path)] = ref(path)
        for path in packets:
            protected[str(path)] = ref(path)
            packet = read(path)
            for item in references(packet):
                p = Path(item['path'])
                if p.is_file() and p.stat().st_size < 10_000_000:
                    protected[str(p)] = ref(p)
            for p in edit_paths:
                if str(p) in bound or str(p) in protected:
                    raise RuntimeError('EDIT_IS_SIGNATURE_BOUND:' + str(p))
                for name in packet['write_set']:
                    claimed = Path(name) if Path(name).is_absolute() else ROOT / name
                    if p == claimed or p.is_relative_to(claimed):
                        raise RuntimeError('PRODUCTION_WRITE_CONFLICT:' + str(p))
        # Dynamic governance files are intentionally not immutable protected inputs.
        protected = {p: r for p, r in protected.items()
                     if not p.startswith(str(ROOT / 'docs/governance') + '/')
                     and p != str(ROOT / 'tasks/current/INDEX.json')}
        raw = Path(f'/proc/{os.getpid()}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        writer = {'pid': os.getpid(), 'startticks': int(fields[19]), 'executor_epoch': 1,
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'fencing_token': secrets.token_hex(24)}
        deadline = now() + timedelta(minutes=30)
        baseline = {str(p): ref(p) if p.exists() else {'path': str(p), 'status': 'ABSENT'}
                    for p in edit_paths}
        save(stage / 'PROTECTED_INPUTS.json', list(protected.values()))
        for label, names in [('docs', DOC_WRITES), ('tests', TEST_WRITES)]:
            destination = stage / label; destination.mkdir()
            packet = {
                'schema': 'isolated-bounded-maintenance-task-v1',
                'task_id': 'project_navigation_' + label + '_v1_20261009',
                'authorization': '用户授权按系统审计建议实施；独立维护通道，不暂停或写生产治理。',
                'registered_at': now().isoformat(), 'deadline_at': deadline.isoformat(),
                'writer': writer, 'execution_allowed': True,
                'production_governance_write_allowed': False,
                'read_set': ['AGENTS.md', 'docs/current/AI_WORK_ENTRY_ZH.md',
                             'docs/governance/CURRENT_STATUS_RECEIPT.json',
                             'baseline_v1/MANIFEST.json', 'docs/current/DUAL_BRANCH_BASELINE_V1.json',
                             '_run/maintenance/concurrent_cleanup_20261009/PROTOCOL.json',
                             str(stage / 'PROTECTED_INPUTS.json'), str(Path(__file__).resolve())],
                'expanded_read_set': ['actual files bound by PROTECTED_INPUTS',
                                      'scoped docs/test sources from previous audit'],
                'write_set': [str(ROOT / name) for name in names] + [str(destination)],
                'inputs': [ref(Path(__file__)), ref(ROOT / 'baseline_v1/MANIFEST.json'),
                           ref(ROOT / 'docs/current/DUAL_BRANCH_BASELINE_V1.json')]
                          + [ref(p) for p in packets],
                'before': [baseline[str(ROOT / name)] for name in names],
                'budgets': {'gpu': 0, 'model_calls': 0, 'training_steps': 0,
                            'wall_seconds': 1800, 'cpu_threads_per_worker': 1},
                'stop_conditions': ['signature drift', 'writer conflict', 'I/O failure', 'deadline'],
                'required_outputs': ['RESULT.json'],
            }
            packet['run_signature'] = hashlib.sha256(json.dumps(packet['inputs'], sort_keys=True).encode()).hexdigest()
            save(destination / 'TASK_PACKET.json', packet)
        print(json.dumps({'status': 'READY_EDITS', 'root': str(stage),
                          'runtime_packets': [ref(p) for p in packets], 'writer': writer,
                          'deadline_at': deadline.isoformat()}, ensure_ascii=False), flush=True)
        failure = None
        try:
            while now() < deadline:
                ready, _, _ = select.select([sys.stdin], [], [], 20)
                if ready:
                    if sys.stdin.readline().strip() != 'FINISH':
                        raise RuntimeError('INVALID_FINISH_COMMAND')
                    break
                print('EDITS_MAINTENANCE_ALIVE', flush=True)
            else:
                raise RuntimeError('DEADLINE')
            for expected in protected.values():
                if ref(expected['path']) != expected:
                    raise RuntimeError('PROTECTED_INPUT_DRIFT:' + expected['path'])
            for name in DOC_WRITES + TEST_WRITES:
                p = ROOT / name
                if not p.is_file() or p.is_symlink():
                    raise RuntimeError('MISSING_FINAL_OUTPUT:' + name)
            report = read(stage / 'tests/TEST_REPORT.json')
            if report['returncode'] != 0:
                raise RuntimeError('TESTS_NOT_PASSED')
        except BaseException as exc:
            failure = repr(exc)
        for label, names in [('docs', DOC_WRITES), ('tests', TEST_WRITES)]:
            destination = stage / label
            outputs = [ref(ROOT / name) for name in names if (ROOT / name).is_file()]
            outputs += [ref(p) for p in destination.iterdir()
                        if p.is_file() and p.name not in {'RESULT.json', 'TASK_PACKET.json'}]
            save(destination / 'RESULT.json', {
                'status': 'PASSED' if failure is None else 'STOPPED', 'failure': failure,
                'packet': ref(destination / 'TASK_PACKET.json'), 'outputs': outputs,
                'completed_at': now().isoformat(), 'files_deleted': 0,
                'production_started': False, 'claim_limit': 'Navigation/test maintenance only.'})
        print(json.dumps({'status': 'PASSED' if failure is None else 'STOPPED',
                          'failure': failure}, ensure_ascii=False), flush=True)
        if failure:
            raise RuntimeError(failure)


if __name__ == '__main__':
    main()
