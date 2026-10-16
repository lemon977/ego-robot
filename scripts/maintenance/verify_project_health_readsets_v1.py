"""Verify future declarations offline, without bypassing an active production fence."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import sys

ROOT = Path('/path/to/chaoyang')
HELPER = ROOT / 'scripts/maintenance/project_health_fixes_v1.py'
H = runpy.run_path(str(HELPER), run_name='bounded_maintenance_helpers')


def main():
    stage = H['OUT'] / 'readsets_offline_verification'
    with (ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage.mkdir()
        qpath = H['PLAN'] / 'reverse_native_boundary_v9/QUEUE_V9_REVISION2_READSET_V1.json'
        refs = [H['ref'](Path(__file__)), H['ref'](HELPER), H['ref'](H['OLD_QUEUE']), H['ref'](qpath)]
        stat = Path(f'/proc/{os.getpid()}/stat').read_text(); fields = stat[stat.rfind(')')+2:].split()
        now = datetime.now(ZoneInfo('Asia/Shanghai'))
        H['save_new'](stage / 'TASK_PACKET.json', {
            'schema': 'isolated-maintenance-verification-task-v1',
            'task_id': 'system_health_readsets_offline_v1_20261009',
            'writer': {'pid':os.getpid(),'startticks':int(fields[19]),'executor_epoch':1,
                       'fencing_token':secrets.token_hex(24),
                       'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
            'deadline_at':(now+timedelta(minutes=20)).isoformat(),
            'inputs':refs, 'read_set':[str(HELPER),str(H['OLD_QUEUE']),str(qpath),
                'AGENTS.md','baseline_v1/MANIFEST.json',
                'src/chaoyang/governance/execute_pick_place_1008_v1.py',
                'docs/governance/CURRENT_STATUS_RECEIPT.json','/proc'],
            'expanded_read_set':['exact old/new spec and input references; no failed algorithm partials'],
            'write_set':[str(stage),str(H['PLAN']/'reverse_native_boundary_v9/dispatch_finite20_v9_readset_v1.py')],
            'budgets':{'gpu':0,'models':0,'training':0,'wall_seconds':1200},
            'production_governance_write_allowed':False,
            'run_signature':hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest(),
            'claim_limit':'Offline correctness, not current execution permission. Active fence remains mandatory.'})
        sys.path.insert(0,str(ROOT/'src'))
        from chaoyang.governance import execute_pick_place_1008_v1 as executor
        from chaoyang.governance.register_single_task_packet import _validate_packet
        failure = None; checked = []; dispatcher = None
        try:
            old = H['read'](H['OLD_QUEUE']); new = H['read'](qpath)
            if len(old['records']) != len(new['records']) != 20:
                raise RuntimeError('QUEUE_PARTITION_DRIFT')
            for previous, current in zip(old['records'], new['records']):
                if (previous['sessions'],previous['native_frames']) != (current['sessions'],current['native_frames']):
                    raise RuntimeError('PARTITION_CHANGED')
                if previous['task_id'] == current['task_id']:
                    terminal = ROOT / f"_run/current/{previous['task_id']}/attempts/attempt_0001/RESULT.json"
                    if H['read'](terminal)['status'] != 'PASSED':
                        raise RuntimeError('UNCHANGED_PREDECESSOR_NOT_PASSED')
                    continue
                original = H['read'](H['check_ref'](previous['spec']))
                actual = executor.load_spec(H['check_ref'](current['spec']))
                expected = H['normalized_spec'](original); expected['task_id'] = current['task_id']
                if actual != expected:
                    raise RuntimeError('NON_DECLARATION_CHANGE')
                packet = executor.packet_value(Path(current['spec']['path']), actual)
                if _validate_packet(packet):
                    raise RuntimeError('OFFICIAL_PACKET_SCHEMA_ERROR')
                for value in packet['runtime_code'] + packet['inputs']['files']:
                    H['check_ref'](value)
                checked.append({'task_id':current['task_id'],'spec':current['spec'],
                    'read_set_count':len(actual['read_set']),'runtime_refs_checked':len(packet['runtime_code']),
                    'status':'OFFLINE_STRUCTURE_PASS','execution_permission':'DEFERRED_TO_NORMAL_REGISTRATION',
                    'new_packet_registered':False,'failed_partial_consumed':False})
            original_path = H['PLAN']/'reverse_native_boundary_v9/dispatch_finite20_v9_revision2.py'
            source = original_path.read_text()
            changes = {
                "WORK/'QUEUE_V9_REVISION2.json'":"WORK/'QUEUE_V9_REVISION2_READSET_V1.json'",
                H['OLD_QUEUE_SHA']:H['digest'](qpath),
                'V9_REV2_DISPATCH_STATUS.json':'V9_READSET_V1_DISPATCH_STATUS.json',
                'V9_REV2_DISPATCH.lock':'V9_READSET_V1_DISPATCH.lock',
                'checks_rev2':'checks_readset_v1',
                'single_attempt_intents_rev2':'single_attempt_intents_readset_v1',
                'execute_logs_rev2':'execute_logs_readset_v1',
                'started_identities_rev2':'started_identities_readset_v1'}
            for before, after in changes.items():
                if source.count(before) != 1:
                    raise RuntimeError('DISPATCHER_TRANSFORMATION_CONFLICT:'+before)
                source = source.replace(before,after)
            before = '    return s\n\ndef argv'
            after = '''    if '_readset_v1_' in s['task_id']:
        allowed=[path(value) for value in s['read_set']]
        required=[path(value) for value in s.get('inputs',[])]
        required += [path(option(a,flag)) for flag in ['--manifest','--camera-config','--mask-seeds','--reuse-from'] if flag in a]
        need(len(allowed)<=8 and all(any(value==parent or parent in value.parents for parent in allowed) for value in required),'READ_SET_DECLARATION_INCOMPLETE')
    return s

def argv'''
            if source.count(before)!=1: raise RuntimeError('SPEC_CHECK_INSERTION_CONFLICT')
            source=source.replace(before,after)
            ast.parse(source)
            dispatcher=H['PLAN']/'reverse_native_boundary_v9/dispatch_finite20_v9_readset_v1.py'
            with dispatcher.open('x') as f:
                f.write(source);f.flush();os.fsync(f.fileno())
            H['save_new'](stage/'OFFLINE_CHECKS.json',checked)
            H['save_new'](stage/'SUCCESSOR_DISPATCHER.json',{
                'original':H['ref'](original_path),'successor':H['ref'](dispatcher),
                'queue':H['ref'](qpath),'transformations':changes,
                'production_started':False,'active_fence_bypassed':False,
                'claim_limit':'Frozen successor ready for production owner adoption; do not launch concurrently with existing dispatcher.'})
            for value in refs: H['check_ref'](value)
        except BaseException as exc:
            failure=repr(exc)
        H['save_new'](stage/'RESULT.json',{'status':'PASSED_OFFLINE_ROUTING_DEFERRED' if failure is None else 'STOPPED',
            'failure':failure,'specs_verified':len(checked),
            'outputs':[H['ref'](p) for p in stage.iterdir() if p.is_file()],
            'dispatcher':H['ref'](dispatcher) if dispatcher and dispatcher.exists() else None,
            'algorithm_or_threshold_change':False,'production_started':False,
            'next_action':'Production owner explicitly switches to successor after current batch/dispatcher has ended; never hot-edit a packet.'})
        print(json.dumps({'status':'PASS' if failure is None else 'STOPPED','specs_verified':len(checked),
                          'failure':failure,'result':H['ref'](stage/'RESULT.json')},ensure_ascii=False),flush=True)
        if failure: raise RuntimeError(failure)


if __name__=='__main__': main()
