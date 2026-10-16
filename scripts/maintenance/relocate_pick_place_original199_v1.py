"""Finite storage-only validation/proof/deletion; no production publication.

Each invocation owns the existing maintenance lock and a distinct immutable
packet. Original algorithm/runtime bytes and all JSON evidence are preserved.
Deletion is intentionally conservative: only large full-session RGB archives
and review/final videos, never inputs of the known unadapted current consumers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import threading
import time

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/pick_place_1008_nas_export_20261010_v1'
DEST = Path('/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1')
E0 = BASE / 'E0_FREEZE_141026_b1c2b4e8/RESULT.json'
E1 = BASE / 'E1_EXPORT_141118_086c9e29/RESULT.json'
DEV = BASE / 'STORAGE_RELOCATION_DEVELOPMENT_SPEC_V1.json'
HELPER = ROOT / 'scripts/maintenance/project_cleanup_proof_v2.py'
LOCK = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
AUTH = BASE / 'STORAGE_RELOCATION_AUTHORIZATION_V1.json'
NAVIGATION = ROOT / 'review/pick_place_1008/ORIGINAL199_NAS_STORAGE_V1.json'
SCAN = ROOT / 'scripts/maintenance/original199_reference_scan_v1.py'
READER_FILES = [ROOT / 'src/chaoyang/storage_relocation' / n for n in
                ('__init__.py', 'pick_place_original199_v1.py', 'stage_reader_v1.py')]
PROTECTED_CONTRACTS = [
 ROOT / 'review/pick_place_1008/CUP_PROTECTED_CLEAN_DEV4_FRESH_RENDER_CONTACT_GOVERNED_PRIVATE_PREPARATION_V1/FINITE_DEV4_C3_ACTUAL_INPUT_FRESH_RENDER_CONTACT_CONTRACT_DRAFT_REVISION7_V1.json',
 ROOT / 'review/pick_place_1008/QPC_ARM_COHORT16_GOVERNED_PREPARATION_V1/USER_AUTHORIZED_CURRENT_KNOWN16_NEW_ONLY_EXECUTOR_RELEASE_SUCCESSOR_V14_PRIVATE_PREPARATION_V1/FINITE_COHORT16_USER_AUTHORIZED_KNOWN16_CONTRACT_V14.json',
 ROOT / 'review/pick_place_1008/CONSERVATIVE_WHOLE_MESH_SCREEN_BOUNDS_PRIVATE_RENDERER_CANDIDATE_V1_20261010/FULL_HOLD295_EGO_ARM_SCREENBOUNDS_MATCHED_NEW_FAMILY_V3_PRIVATE_PREPARATION_V1/PRIVATE_REVISION2_REMAINING_DEADLINE_AND_SAVED_PRECISION_CLOSURE_V1/FINITE_HOLD007199295_FROZEN_E2_SINGLE_FACTOR_CONTRACT_REV2_V1.json',
]
spec = importlib.util.spec_from_file_location('relocation_maintenance_helpers', HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def once(path, value):
    path = Path(path)
    need(path.is_relative_to(BASE) and path.parent.resolve() == path.parent, 'EXACT_MAINTENANCE_OUTPUT_ONLY')
    with path.open('xb') as stream:
        stream.write(H.canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def refs(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'bytes', 'sha256'}:
            yield value
        else:
            for child in value.values():
                yield from refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs(child)


def identity_metadata(path):
    # CPFS may finalize allocated blocks after a same-byte write. st_blocks is
    # accounting, not file identity; keep it separately for space statistics.
    return {k:v for k,v in H.metadata(path).items() if k != 'allocated_bytes'}


class Runtime:
    def __init__(self, phase, inputs, seconds):
        self.phase, self.inputs, self.seconds = phase, inputs, seconds
        self.started = time.monotonic()
        self.last = 0
        pid = int(Path('/proc/self/stat').read_text().split(' ', 1)[0])
        identity = H.identity(pid)
        need(identity is not None, 'HOST_WRITER_REQUIRED')
        self.writer = dict(**identity, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            executor_epoch=0, fencing_token=secrets.token_hex(24),
            deadline_at=(H.now() + timedelta(seconds=seconds)).isoformat())
        need(LOCK.resolve() == LOCK and not LOCK.is_symlink(), 'LOCK_ALIAS')
        self.lock = LOCK.open('a+')
        self.lock_id = (os.fstat(self.lock.fileno()).st_dev, os.fstat(self.lock.fileno()).st_ino)
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.seek(0)
        previous = self.lock.read()
        self.writer['executor_epoch'] = int(json.loads(previous).get('executor_epoch', 0)) + 1 if previous else 1
        self.writer.update(lock_device=self.lock_id[0], lock_inode=self.lock_id[1])
        self.lock.seek(0)
        self.lock.truncate()
        json.dump(self.writer, self.lock)
        H.read_back_private_lock(self.lock, LOCK, self.writer)
        self.out = BASE / (phase + '_' + H.now().strftime('%H%M%S') + '_' + self.writer['fencing_token'][:8])
        self.out.mkdir()
        archive = self.out/'runtime_snapshot'
        archive.mkdir()
        snapshots = []
        runtime_sources = [Path(__file__).absolute(), *READER_FILES, HELPER]
        if phase != 'R0_READONLY_VALIDATION':
            runtime_sources.append(SCAN)
        for original in runtime_sources:
            payload, _ = H.stable_bytes(original)
            # Preserve helper revisions as data, not a second executable entry.
            suffix = '.source.bin' if original in (HELPER, SCAN) else ''
            copy = archive/(original.parent.name + '__' + original.name + suffix)
            with copy.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            snapshots.append(dict(original=H.pin(original), snapshot=H.pin(copy)))
        delete_targets = []
        if phase == 'C1_EXACT_UNLINK':
            parent = H.load(Path(inputs[-1]['path']))
            need(parent['status'] == 'PASSED_CONSERVATIVE_RELOCATION_PROOF', 'C1_PARENT_PROOF_STATUS')
            H.verify(parent['proof'])
            delete_targets = H.load(Path(parent['proof']['path']))['eligible']
            need(0 < len(delete_targets) <= 796 and sum(r['bytes'] for r in delete_targets) <= 33083676127, 'EXACT_DELETE_BUDGET_EXCEEDED')
        self.packet = dict(schema='original199-finite-storage-maintenance-v1', task_id=self.out.name,
            phase=phase, registered_at=H.now().isoformat(), writer=self.writer,
            execution_allowed=True, algorithm_execution_allowed=False,
            deletion_allowed=(phase == 'C1_EXACT_UNLINK'), production_governance_write_allowed=False,
            inputs=inputs, read_set=[r['path'] for r in inputs],
            frozen_runtime_snapshots=snapshots,
            expanded_read_set='Exact E0 files, original receipt/producer closure, host process identities and current reference proof',
            write_set=[str(self.out), str(LOCK)] + [r['source_path'] for r in delete_targets]
                      + ([str(NAVIGATION)] if phase == 'E2_STORAGE_NAVIGATION' else []),
            exact_deletion_manifest=parent['proof'] if delete_targets else None,
            budgets=dict(cpu_threads=2, gpu=0, model_calls=0, training_steps=0, wall_seconds=seconds,
                         max_delete_files=796, max_delete_bytes=33083676127),
            run_signature=hashlib.sha256(H.canonical(dict(phase=phase, inputs=inputs, destination=str(DEST)))).hexdigest(),
            required_outputs=['RESULT.json'], historical_result_mutation_allowed=False)
        once(self.out / 'TASK_PACKET.json', self.packet)
        for r in inputs:
            H.verify(r)
        self.input_ids = {r['path']: identity_metadata(r['path']) for r in inputs}
        once(self.out/'INPUT_IDENTITIES.json',self.input_ids)
        self.progress = {}
        print(json.dumps(dict(status='REGISTERED', phase=phase, task=str(self.out))), flush=True)

    def guard(self):
        need(time.monotonic() - self.started < self.seconds, 'FINITE_DEADLINE')
        need(H.identity(self.writer['pid'])['startticks'] == self.writer['startticks'], 'WRITER_IDENTITY_CHANGED')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == self.writer['boot_id'], 'BOOT_CHANGED')
        s = LOCK.lstat()
        need((s.st_dev, s.st_ino) == self.lock_id and json.loads(LOCK.read_bytes()) == self.writer, 'LOCK_OR_FENCE_CHANGED')
        for p, identity in self.input_ids.items():
            need(identity_metadata(p) == identity, 'PINNED_INPUT_IDENTITY_CHANGED:' + p)
        if time.monotonic() - self.last > 60:
            self.last = time.monotonic()
            value = dict(observed_at=H.now().isoformat(), phase=self.phase,
                         elapsed_seconds=round(time.monotonic()-self.started, 2), **self.progress)
            (self.out / 'STATE.json').write_bytes(H.canonical(value) + b'\n')
            print(json.dumps(value), flush=True)

    def finish(self, value):
        # A second verification must never swallow a stopped deletion journal.
        # Success still requires every frozen input; drift instead seals an
        # explicit terminal failure with expected/actual identity evidence.
        drift = []
        for r in self.inputs:
            try:
                H.verify(r)
            except Exception as exc:
                evidence = dict(expected=r, failure=type(exc).__name__+':'+str(exc))
                try:
                    evidence['actual_metadata'] = identity_metadata(r['path'])
                except Exception as unavailable:
                    evidence['metadata_failure'] = type(unavailable).__name__+':'+str(unavailable)
                drift.append(evidence)
        if drift:
            value['input_verification_failures'] = drift
            value['pre_finish_status'] = value['status']
            value['status'] = 'PARTIAL_STOPPED' if self.progress.get('deleted_files',0) else 'STOPPED'
            value.setdefault('failure', 'PINNED_INPUT_DRIFT_AT_TERMINAL_SEAL')
            value.update(self.progress)
            journal = self.out/'DELETED_FILES_V1.jsonl'
            if journal.exists():
                value['deletion_journal'] = H.pin(journal)
        value.update(task_id=self.out.name, task_packet=H.pin(self.out / 'TASK_PACKET.json'), writer=self.writer,
                     finished_at=H.now().isoformat(), production_governance_writes=0, model_calls=0,
                     visual_quality_promoted=False, training_eligible=False)
        once(self.out / 'RESULT.json', value)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        print(json.dumps(dict(status=value['status'], result=str(self.out / 'RESULT.json'))), flush=True)
        return 0 if value['status'].startswith('PASSED') else 2


def governance():
    result = subprocess.run([sys.executable, '-B', '-m', 'chaoyang.cli', 'validate-governance'],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), CUDA_VISIBLE_DEVICES=''),
        capture_output=True, text=True, timeout=90)
    value = json.loads(result.stdout)
    need(result.returncode == 0 and value['status'] == 'PASS', 'GOVERNANCE_FAILED')
    ledger = H.load(ROOT / 'docs/governance/LONG_HORIZON_TASK_STATE.json')
    index = H.load(ROOT / 'tasks/current/INDEX.json')
    need(not [r for r in ledger['tasks'] if r['status'] in {'RUNNING','CLAIMED','WAIT_GPU_RESOURCE'}], 'PRODUCTION_ACTIVE')
    need(not [r for r in index['task_packets'] if r.get('execution_allowed') is True], 'ROUTABLE_PRODUCTION_TASK')
    return value


def inventory():
    freeze = H.load(E0)
    H.verify(freeze['inventory'])
    return H.load(Path(freeze['inventory']['path']))


def require_validated_reader_version(validation):
    """An old R0 pass cannot qualify a subsequently changed reader."""
    need(validation['status'] == 'PASSED_READONLY_STORAGE_VALIDATION', 'READERS_NOT_VALIDATED')
    H.verify(validation['task_packet'])
    packet = H.load(Path(validation['task_packet']['path']))
    need(packet.get('phase') == 'R0_READONLY_VALIDATION' and
         packet.get('deletion_allowed') is False, 'EXACT_R0_VALIDATION_PACKET_REQUIRED')
    for path in READER_FILES:
        expected = H.pin(path)
        matches = [r for r in packet['inputs'] if r['path'] == str(path)]
        need(matches == [expected], 'CURRENT_STORAGE_READER_VERSION_NOT_VALIDATED:' + str(path))


def validate(runtime):
    from chaoyang.storage_relocation.pick_place_original199_v1 import Original199Reader
    from chaoyang.storage_relocation.stage_reader_v1 import validate_sealed_stage, validate_session_actions
    from chaoyang.pick_place_1008 import source
    from chaoyang.pick_place_1008.pipeline import STAGE_FILES
    manifest = H.load(E1)
    need(manifest['status'] == 'PASSED_EXPORT', 'E1_INCOMPLETE')
    old = inventory()
    export_packet = H.load(Path(manifest['task_packet']['path']))
    H.verify(manifest['task_packet'])
    for pin in export_packet['inputs']:
        H.verify(pin)
    H.verify(manifest['export_manifest'])
    seal = H.load(DEST/'EXPORT_COMPLETE_V1.json')
    need(seal['status'] == 'PASSED_EXPORT' and seal['export_manifest'] == manifest['export_manifest'], 'EXPORT_SEAL_CONFLICT')
    binding = dict(schema='pick-place-original199-storage-binding-v1', dataset_root=str(DEST),
        export_manifest=H.pin(DEST/'EXPORT_MANIFEST_V1.json'), export_complete=H.pin(DEST/'EXPORT_COMPLETE_V1.json'),
        source_inventory=H.pin(Path(H.load(E0)['inventory']['path'])), source_manifest=old['parent_manifest'])
    once(runtime.out / 'STORAGE_BINDING_V1.json', binding)
    reader = Original199Reader(H.pin(runtime.out / 'STORAGE_BINDING_V1.json'))
    source_manifest = H.load(Path(old['parent_manifest']['path']))
    # Exercise the actual receipt/producer/raw-source contract before the full
    # copy sweep. This is a storage preflight, not algorithm execution.
    first = source_manifest['sessions'][0]
    preflight = {}
    for stage in ('source','motion','gpu','render'):
        runtime.guard()
        sr = first['stage_receipts'][stage]
        rc = reader.read_json(sr)
        folder, sealed = validate_sealed_stage(reader, ROOT, Path(sr['path']).parents[2],
            first['session_id'], stage, rc['binding'], STAGE_FILES[stage])
        need(folder is not None and sealed == sr, 'FIRST_SESSION_STAGE_PREFLIGHT:' + stage)
        preflight[stage] = sealed
    once(runtime.out/'FIRST_SESSION_STAGE_PREFLIGHT_V1.json',
        dict(session_id=first['session_id'], unchanged_four_stage_closure=preflight))
    completed = []
    stopped = threading.Event()
    def verify_item(item):
        need(not stopped.is_set() and time.monotonic()-runtime.started < runtime.seconds, 'READONLY_VALIDATION_STOPPED_OR_DEADLINE')
        p = Path(item['source_path'])
        need(H.metadata(p) == item['source_metadata'], 'SOURCE_DRIFT_AFTER_EXPORT:' + str(p))
        target = DEST / item['destination_relative']
        pin = dict(path=str(p), bytes=item['bytes'], sha256=item['sha256'])
        if item['kind'] == 'session_payload' and p.suffix in ('.mp4','.npz'):
            physical = reader.physical_path(pin)
            need(physical == target, 'MAPPING_TARGET_MISMATCH')
        else:
            if item['kind'] == 'session_payload':
                reader.verify_ref(pin)
            H.verify(dict(path=str(target), bytes=item['bytes'], sha256=item['sha256']))
        return dict(**item, destination_metadata=H.metadata(target),
                    source_metadata_still_same=True, destination_SHA_reverified=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(verify_item,item) for item in old['files']]
        try:
            for future in as_completed(futures):
                item = future.result()
                completed.append(item)
                runtime.progress = dict(verified_files=len(completed), verified_bytes=sum(r['bytes'] for r in completed))
                runtime.guard()
        except Exception:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
    once(runtime.out/'VERIFIED_BYTES_CHECKPOINT_V1.json',
        dict(status='PASSED_FILE_BYTES_ONLY_NOT_READER_ACCEPTANCE', files=completed,
             verified_files=len(completed), verified_bytes=sum(r['bytes'] for r in completed)))
    physical_files = set()
    for directory, dirs, files in os.walk(DEST, followlinks=False):
        for name in dirs + files:
            need(not (Path(directory)/name).is_symlink(), 'NAS_SYMLINK')
        physical_files.update(str((Path(directory)/name).relative_to(DEST)) for name in files)
    expected = {r['destination_relative'] for r in old['files']} | {
        'EXPORT_IN_PROGRESS.json','EXPORT_MANIFEST_V1.json','README_ZH.md','EXPORT_COMPLETE_V1.json'}
    need(physical_files == expected, 'NAS_UNEXPECTED_FILE_SET')
    parity = []
    for row in source_manifest['sessions']:
        runtime.guard()
        sid = row['session_id']
        logical_source = Path(row['source_session']) / 'source'
        before = source.load_prepared(logical_source)
        after = reader.load_prepared(logical_source)
        import numpy as np
        for name in ('session_id','frame_count','fps','width','height','raw_dir','calibration'):
            need(getattr(before,name) == getattr(after,name), 'PREPARED_PARITY:' + sid + ':' + name)
        for name in ('K','c2w','hands','hand_observed','hand_source','body','body_observed','qpc_ns'):
            a,b = getattr(before,name),getattr(after,name)
            need(np.array_equal(a,b, equal_nan=True) if np.asarray(a).dtype.kind in 'fc' else np.array_equal(a,b), 'ARRAY_PARITY:' + sid + ':' + name)
        closure = {}
        for stage in ('source','motion','gpu','render'):
            sr = row['stage_receipts'][stage]
            rc = reader.read_json(sr)
            root = Path(sr['path']).parents[2]
            folder, sealed = validate_sealed_stage(reader, ROOT, root, sid, stage, rc['binding'], STAGE_FILES[stage])
            need(folder is not None and sealed == sr, 'STAGE_RELOCATION_NOT_EQUIVALENT:' + sid + ':' + stage)
            closure[stage] = sealed
        actions = next(r for r in old['files'] if r.get('session_id') == sid and r['destination_relative'].endswith('/actions.npz'))
        errors = validate_session_actions(reader, dict(path=actions['source_path'],bytes=actions['bytes'],sha256=actions['sha256']), before.frame_count)
        need(not errors, 'UNCHANGED_ACTION_ARRAY_VALIDATION_FAILED:' + sid + ':' + str(errors))
        parity.append(dict(session_id=sid, prepared_parity=True, four_stage_closure=closure, action_errors=errors))
        runtime.progress.update(sessions_validated=len(parity))
    once(runtime.out/'VERIFIED_STORAGE_FILES_V1.json', dict(files=completed, sessions=parity))
    return dict(status='PASSED_READONLY_STORAGE_VALIDATION', storage_binding=H.pin(runtime.out/'STORAGE_BINDING_V1.json'),
        verified_inventory=H.pin(runtime.out/'VERIFIED_STORAGE_FILES_V1.json'), verified_files=len(completed),
        verified_bytes=sum(r['bytes'] for r in completed), prepared_parity_sessions=len(parity),
        preserved_four_stage_closures=4*len(parity), forbidden_algorithm_code_changes=0,
        old_frozen_commands_transparently_reenabled=False, governance=governance(), deleted_files=0)


def proof(runtime, parent):
    validation = H.load(parent)
    require_validated_reader_version(validation)
    H.verify(validation['verified_inventory'])
    rows = H.load(Path(validation['verified_inventory']['path']))['files']
    by_path = {r['source_path']:r for r in rows}
    protected = {}
    for contract in PROTECTED_CONTRACTS:
        pin = H.pin(contract)
        for r in refs(H.load(contract)):
            if r['path'] in by_path:
                need(r['bytes'] == by_path[r['path']]['bytes'] and r['sha256'] == by_path[r['path']]['sha256'], 'PROTECTED_REFERENCE_DRIFT')
                protected.setdefault(r['path'], []).append(pin)
    fixed = H.load(PROTECTED_CONTRACTS[1])['fixed_cases']
    protected_sids = {case['session_id'] for case in fixed}
    need(protected_sids == {'001','003','007','008','035','039','042','050','051','100','111','114','146','147','192','199'}, 'KNOWN_CURRENT_COHORT_DRIFT')
    # Preserve small arrays and all prepared inputs so unadapted optimization
    # consumers remain intact; only new explicit readers consume migrated data.
    allowed_names = {'raw_rgb.npz','clean_rgb.npz','final.mp4','review.mp4'}
    eligible, retained = [], []
    for r in rows:
        runtime.guard()
        p = Path(r['source_path'])
        if r['kind'] != 'session_payload' or p.name not in allowed_names or str(p) in protected or r.get('session_id') in protected_sids:
            unadapted = str(p) in protected or (r.get('session_id') in protected_sids and p.name in allowed_names)
            retained.append(dict(path=str(p), bytes=r['bytes'], reason='KNOWN_UNADAPTED_CONSUMER' if unadapted else 'PRESERVE_SOURCE_SMALL_ARRAY_OR_EVIDENCE', reference_context=protected.get(str(p),[])))
        else:
            eligible.append(r)
    need(eligible and len(eligible)<=796, 'EXACT_DELETE_SCOPE_EMPTY_OR_EXCEEDED')
    spec = importlib.util.spec_from_file_location('original199_reference_scan', SCAN)
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    reference_proof = scan.scan_reference_closure(runtime, rows, eligible, retained)
    blocked_paths = set(reference_proof['blocked_paths'])
    retained.extend(dict(path=r['source_path'],bytes=r['bytes'],reason='BLOCKED_REFERENCE_PROOF') for r in eligible if r['source_path'] in blocked_paths)
    eligible = [r for r in eligible if r['source_path'] not in blocked_paths]
    need(eligible, 'BLOCKED_REFERENCE_PROOF_NO_ELIGIBLE_FILES')
    current = governance()
    process = H.process_check([r['source_path'] for r in eligible], ROOT, runtime.guard)
    H.assert_no_process_holders(process)
    tracked = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,check=True).stdout
    candidates = {str(Path(r['source_path']).relative_to(ROOT)) for r in eligible}
    need(not candidates.intersection(os.fsdecode(tracked).split('\0')), 'TRACKED_OR_UNTRACKED_UNIQUE_FILE')
    once(runtime.out/'RELOCATION_DELETE_PROOF_V1.json', dict(schema='original199-conservative-relocation-delete-proof-v1',
        eligible=eligible, retained=retained, protected_contracts=[H.pin(p) for p in PROTECTED_CONTRACTS],
        reader_validation=H.pin(parent), storage_binding=validation['storage_binding'],
        reference_status='REFERENCED_SATISFIED_BY_EXPLICIT_VALIDATED_STORAGE_READERS',
        original_frozen_unadapted_entries_remain_unsupported_for_migrated_files=True,
        known_current_unadapted_inputs_retained=True, process_check=process,
        production_governance=current, reference_scan=reference_proof,
        no_source_code_or_license_in_delete_scope=True,
        no_raw_or_0916_targets=True, deletion_allowed=False))
    return dict(status='PASSED_CONSERVATIVE_RELOCATION_PROOF', proof=H.pin(runtime.out/'RELOCATION_DELETE_PROOF_V1.json'),
        eligible_files=len(eligible), eligible_bytes=sum(r['bytes'] for r in eligible),
        retained_files=len(retained), retained_bytes=sum(r['bytes'] for r in retained), deleted_files=0)


def unlink(runtime, parent):
    closure = H.load(parent)
    need(closure['status'] == 'PASSED_CONSERVATIVE_RELOCATION_PROOF', 'C0_NOT_COMPLETE')
    H.verify(closure['proof'])
    p = H.load(Path(closure['proof']['path']))
    H.verify(p['reader_validation'])
    require_validated_reader_version(H.load(Path(p['reader_validation']['path'])))
    spec = importlib.util.spec_from_file_location('original199_reference_scan', SCAN)
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    scan.recheck_reference_closure(p['reference_scan'], runtime)
    for r in p['protected_contracts']:
        H.verify(r)
    H.verify(p['storage_binding'])
    rows = p['eligible']
    gov = governance()
    static_nav = {str(q):identity_metadata(q) for q in (ROOT/'docs/governance/LONG_HORIZON_TASK_STATE.json',ROOT/'tasks/current/INDEX.json')}
    process = H.process_check([r['source_path'] for r in rows],ROOT,runtime.guard)
    H.assert_no_process_holders(process)
    deleted = []
    journal = runtime.out/'DELETED_FILES_V1.jsonl'
    for r in rows:
        runtime.guard()
        for q, frozen in static_nav.items():
            need(identity_metadata(q) == frozen, 'PRODUCTION_STATE_CHANGED_DURING_DELETE')
        # No new reference writer may enter unnoticed between file deletions.
        scan.recheck_reference_closure(p['reference_scan'], runtime, metadata_only=True)
        H.assert_no_process_holders(H.process_check([r['source_path']], ROOT, runtime.guard))
        tracked_now = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z','--',str(Path(r['source_path']).relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        need(not tracked_now, 'GIT_TARGET_CHANGED_DURING_DELETE')
        old = Path(r['source_path'])
        nas = DEST / r['destination_relative']
        need(old.is_relative_to(ROOT/'_run/current') and old.resolve() == old and old.name in {'raw_rgb.npz','clean_rgb.npz','final.mp4','review.mp4'}, 'DELETE_SCOPE_OR_ALIAS')
        before = H.metadata(old)
        need(before == r['source_metadata'] and stat.S_ISREG(before['mode']) and before['nlink']==1, 'SOURCE_IDENTITY_CHANGED')
        need(nas.resolve() == nas and H.metadata(nas) == r['destination_metadata'], 'VERIFIED_RECOVERY_COPY_CHANGED')
        # Validation read every NAS byte by SHA on this stable identity. Reopen
        # the same inode before deletion; no size-only or stale receipt shortcut.
        recover_fd = os.open(nas,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            opened = os.fstat(recover_fd)
            need(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1, 'RECOVERY_TYPE_OR_LINK_CHANGED')
            need((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==tuple(r['destination_metadata'][k] for k in ('device','inode','size','mtime_ns','ctime_ns')), 'RECOVERY_OPEN_RACE')
            parent_fd = os.open(old.parent, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                s=os.stat(old.name,dir_fd=parent_fd,follow_symlinks=False)
                need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(before[k] for k in ('device','inode','size','mtime_ns','ctime_ns')), 'DELETE_OPEN_RACE')
                need(stat.S_ISREG(s.st_mode) and s.st_nlink==1, 'DELETE_TYPE_OR_LINK_CHANGED')
                intent=dict(status='UNLINK_INTENT',old_path=str(old),bytes=r['bytes'],sha256=r['sha256'],recoverable_from=str(nas),source_before=before,NAS_identity=r['destination_metadata'],at=H.now().isoformat())
                with journal.open('ab') as stream:
                    stream.write(H.canonical(intent)+b'\n');stream.flush();os.fsync(stream.fileno())
                os.unlink(old.name,dir_fd=parent_fd)
                runtime.progress=dict(deleted_files=len(deleted)+1,deleted_bytes=sum(v['bytes'] for v in deleted)+r['bytes'])
                os.fsync(parent_fd)
                need(H.metadata(nas)==r['destination_metadata'], 'RECOVERY_CHANGED_AFTER_UNLINK')
            finally:
                os.close(parent_fd)
        finally:
            os.close(recover_fd)
        row=dict(status='UNLINK_CONFIRMED',old_path=str(old),bytes=r['bytes'],sha256=r['sha256'],allocated_bytes=before['allocated_bytes'],
                 recoverable_from=str(nas),source_before=before,NAS_identity=r['destination_metadata'],deleted_at=H.now().isoformat())
        with journal.open('ab') as stream:
            stream.write(H.canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
        deleted.append(row)
        runtime.progress=dict(deleted_files=len(deleted),deleted_bytes=sum(r['bytes'] for r in deleted))
    from chaoyang.storage_relocation.pick_place_original199_v1 import Original199Reader
    reader=Original199Reader(p['storage_binding'])
    stopped = threading.Event()
    def verify_after_unlink(r):
        need(not stopped.is_set() and time.monotonic()-runtime.started < runtime.seconds,
             'POST_UNLINK_VALIDATION_STOPPED_OR_DEADLINE')
        need(not Path(r['source_path']).exists(), 'LOCAL_DUPLICATE_STILL_PRESENT')
        reader.physical_path(dict(path=r['source_path'],bytes=r['bytes'],sha256=r['sha256']))
        return r['source_path']
    verified = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(verify_after_unlink, r) for r in rows]
        try:
            for future in as_completed(futures):
                future.result()
                verified += 1
                runtime.progress.update(post_unlink_SHA_verified_files=verified)
                runtime.guard()
        except Exception:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
    return dict(status='PASSED_EXACT_DUPLICATE_UNLINK',deleted_files=len(deleted),
        deleted_bytes=sum(r['bytes'] for r in deleted),recorded_allocated_bytes=sum(r['allocated_bytes'] for r in deleted),
        deletion_journal=H.pin(journal),remaining_exported_files=len(p['retained']),
        remaining_exported_bytes=sum(r['bytes'] for r in p['retained']),
        post_delete_readable_by_explicit_storage_adapter=True,governance=governance(),
        recovery='NAS exact relative paths and original SHA in deletion journal; no raw data deleted')


def cleanup_status_receipt(runtime, path, validation_ref):
    """Record a pinned failed proof without treating it as delete authority."""
    refs = [r for r in runtime.inputs if r['path'] == str(path)]
    need(len(refs) == 1, 'CLEANUP_STATUS_MUST_BE_REGISTERED_INPUT')
    H.verify(refs[0])
    value = H.load(path)
    need(value.get('status') == 'STOPPED' and value.get('deleted_files', 0) == 0,
         'CLEANUP_STATUS_REQUIRES_STOPPED_ZERO_DELETE_PROOF')
    H.verify(value['task_packet'])
    packet = H.load(Path(value['task_packet']['path']))
    need(packet.get('phase') == 'C0_RELOCATION_PROOF' and
         packet.get('deletion_allowed') is False and
         validation_ref in packet.get('inputs', []),
         'CLEANUP_STATUS_PROOF_VALIDATION_BINDING_MISMATCH')
    return dict(status='BLOCKED_REFERENCE_PROOF', result=refs[0],
        failure=value.get('failure'), locally_deleted_files=0,
        next_action='Register a new bounded C0 proof; preserve scope and safety gates. No C1 until C0 passes.')


def publish(runtime, parent, cleanup_status=None):
    """Publish a new shallow navigation receipt; never rewrite old evidence."""
    value = H.load(parent)
    deletion = None
    cleanup = None
    if value['status'] == 'PASSED_EXACT_DUPLICATE_UNLINK':
        deletion = H.pin(parent)
        packet = H.load(Path(value['task_packet']['path']))
        H.verify(value['task_packet'])
        c0 = packet['inputs'][-1]
        H.verify(c0)
        cleanup = H.load(Path(c0['path']))
    elif value['status'] == 'PASSED_CONSERVATIVE_RELOCATION_PROOF':
        cleanup = value
    else:
        need(value['status'] == 'PASSED_READONLY_STORAGE_VALIDATION', 'NAVIGATION_REQUIRES_VALIDATED_READERS')
    if cleanup is not None:
        H.verify(cleanup['proof'])
        proof_value = H.load(Path(cleanup['proof']['path']))
        validation_ref = proof_value['reader_validation']
    else:
        validation_ref = H.pin(parent)
    H.verify(validation_ref)
    validation = H.load(Path(validation_ref['path']))
    require_validated_reader_version(validation)
    need(validation['status'] == 'PASSED_READONLY_STORAGE_VALIDATION' and
         validation['prepared_parity_sessions'] == 199 and validation['preserved_four_stage_closures'] == 796,
         'NAVIGATION_READERS_NOT_ALL199_VALIDATED')
    H.verify(validation['storage_binding'])
    H.verify(H.load(E1)['export_manifest'])
    blocked_cleanup = (cleanup_status_receipt(runtime, cleanup_status, validation_ref)
                       if cleanup_status is not None else None)
    need(blocked_cleanup is None or deletion is None,
         'STOPPED_CLEANUP_STATUS_CANNOT_REPLACE_DELETE_SUCCESS')
    runtime.guard()
    need(NAVIGATION.parent.resolve() == NAVIGATION.parent and not NAVIGATION.exists() and
         not NAVIGATION.is_symlink(), 'NEW_NAVIGATION_PATH_ONLY')
    navigation = dict(schema='original199-nas-storage-navigation-v1', status='SEALED_STORAGE_RELOCATION',
        dataset_root=str(DEST), README=str(DEST/'README_ZH.md'),
        export_result=H.pin(E1), storage_validation=validation_ref,
        storage_binding=validation['storage_binding'], parent_receipt=H.pin(parent),
        deletion_result=deletion, locally_deleted_files=value.get('deleted_files',0),
        locally_deleted_bytes=value.get('deleted_bytes',0),
        cleanup_status=blocked_cleanup,
        sessions=199, native_frames=29322,
        explicit_reader='chaoyang.storage_relocation.pick_place_original199_v1.Original199Reader',
        original_frozen_commands_transparently_reenabled=False,
        preserved_scope='All prepared Source, small arrays, JSON/JSONL, known unadapted cohort16 inputs; additional uncertain references retained',
        consumer_requirement='New finite successor must explicitly bind storage reader SHA and binding; no NAS symlink workaround or revival of old tasks',
        algorithm_changes=0, threshold_changes=0, old_result_mutations=0,
        visual_quality_promoted=False, all199_visual_accepted=False, training_eligible=False,
        hidden_regions='UNKNOWN', user_review='Preserve original actual review scope, never prefill acceptance')
    with NAVIGATION.open('xb') as stream:
        stream.write(H.canonical(navigation)+b'\n');stream.flush();os.fsync(stream.fileno())
    once(runtime.out/'NAVIGATION_RECEIPT_V1.json', navigation)
    handoff = runtime.out/'CONTINUE_HERE_ZH.txt'
    text = ('原版199会话、29322原生帧的完整存储副本位于：\n'+str(DEST)+'\n\n'
        '只读适配验收：199个Source输入等价、796个旧阶段引用闭包保持；没有改算法、门限或旧结果。\n'
        '本地实际删除：'+str(navigation['locally_deleted_files'])+'文件，'+str(navigation['locally_deleted_bytes'])+'字节。\n'
        '保留全部Source、小数组、JSON/JSONL、现用16会话输入和引用证明不足的文件。\n'
        '这只是存储迁移，不是重新渲染或新增视觉验收；数值FAIL、隐藏UNKNOWN和原人工审阅范围不变。\n'
        '新任务必须显式冻结 Original199Reader、stage_reader 和 STORAGE_BINDING_V1.json 的SHA；\n'
        '原逻辑ref保持，只把已映射二进制的只读I/O指向NAS。旧冻结命令未透明开通，不得用symlink绕过。\n'
        '后续Root优化需要独立登记消费者I/O适配任务，不得直接复活旧Task。\n'
        + ('本地清理未完成：'+str(blocked_cleanup['failure'])+'。\n'
           '失败证明收据：'+blocked_cleanup['result']['path']+'\n'
           '下一步：另开有限C0，仅优化扫描效率，不减少检查范围；C0通过前不得启动C1。\n'
           if blocked_cleanup is not None else '')
        +
        '浅层导航：'+str(NAVIGATION)+'\n')
    with handoff.open('xb') as stream:
        stream.write(text.encode());stream.flush();os.fsync(stream.fileno())
    return dict(status='PASSED_STORAGE_NAVIGATION',navigation=H.pin(NAVIGATION),
        handoff=H.pin(handoff),locally_deleted_files=navigation['locally_deleted_files'],
        locally_deleted_bytes=navigation['locally_deleted_bytes'],governance=governance())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('validate','proof','unlink','publish'),required=True)
    parser.add_argument('--parent-result',type=Path)
    parser.add_argument('--cleanup-status-result', type=Path)
    args=parser.parse_args()
    need(args.cleanup_status_result is None or args.phase == 'publish',
         'CLEANUP_STATUS_ONLY_ALLOWED_FOR_NAVIGATION')
    phase={'validate':'R0_READONLY_VALIDATION','proof':'C0_RELOCATION_PROOF','unlink':'C1_EXACT_UNLINK',
           'publish':'E2_STORAGE_NAVIGATION'}[args.phase]
    auth=H.load(AUTH)
    need(auth['user_authorization']=='允许，只读适配验证后删除' and auth['destination']==str(DEST), 'NEW_EXACT_STORAGE_AUTHORIZATION_REQUIRED')
    need(H.load(E1)['status']=='PASSED_EXPORT','WAIT_COMPLETE_E1')
    inputs=[H.pin(p) for p in [AUTH,DEV,E0,E1,Path(__file__).absolute(),HELPER,*READER_FILES,*PROTECTED_CONTRACTS]]
    if args.phase != 'validate':
        inputs.extend([H.pin(SCAN),H.pin(BASE/'STORAGE_REFERENCE_SCAN_DEVELOPMENT_ADDENDUM_V1.json')])
    if args.parent_result:
        need(args.parent_result.is_relative_to(BASE) and args.parent_result.resolve()==args.parent_result,'PARENT_SCOPE')
        inputs.append(H.pin(args.parent_result))
    if args.cleanup_status_result:
        need(args.cleanup_status_result.is_relative_to(BASE) and
             args.cleanup_status_result.resolve() == args.cleanup_status_result,
             'CLEANUP_STATUS_RESULT_SCOPE')
        inputs.append(H.pin(args.cleanup_status_result))
    need(args.phase=='validate' or args.parent_result is not None,'PARENT_REQUIRED')
    runtime=Runtime(phase,inputs,5400 if args.phase=='validate' else 1800)
    try:
        value=(validate(runtime) if args.phase=='validate' else proof(runtime,args.parent_result)
            if args.phase=='proof' else unlink(runtime,args.parent_result)
            if args.phase=='unlink' else publish(runtime,args.parent_result,args.cleanup_status_result))
    except Exception as exc:
        value=dict(status='PARTIAL_STOPPED' if runtime.progress.get('deleted_files',0) else 'STOPPED',failure=type(exc).__name__+':'+str(exc),**runtime.progress)
        journal=runtime.out/'DELETED_FILES_V1.jsonl'
        if journal.exists():
            value['deletion_journal']=H.pin(journal)
    return runtime.finish(value)


if __name__=='__main__':
    raise SystemExit(main())
