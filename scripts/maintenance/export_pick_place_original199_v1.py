"""Exact, finite, byte-verified NAS export. No algorithms or deletion.

Uses the existing maintenance lock and writer identity helpers; never edits
production packets, receipts, result bytes or the existing cleanup protocol.
Only the user's explicitly authorized NAS dataset is an external write root.
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
import threading
import time

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/pick_place_1008_nas_export_20261010_v1'
NAV = ROOT / 'review/pick_place_1008/ALL199_ORIGINAL_EXECUTION_OUTPUTS_UNIFIED_DELIVERY_V1'
DEST = Path('/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1')
AUTH = BASE / 'AUTHORIZATION_V1.json'
PROTOCOL = ROOT / '_run/maintenance/concurrent_cleanup_20261009/PROTOCOL.json'
LOCK = PROTOCOL.parent / 'CHANNEL_LOCK'
HELPER = ROOT / 'scripts/maintenance/project_cleanup_proof_v2.py'
MANIFEST = NAV / 'MANIFEST_ALL199_ORIGINAL_EXECUTION_OUTPUTS_V1.json'
MANIFEST_SHA = '393f69b8b5c2da8e93ff7ec38a6fecfd0f0bcd1b4edb65691fa0dcd0b9d93cf3'

spec = importlib.util.spec_from_file_location('existing_cleanup_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def canonical(value):
    return helpers.canonical(value)


def fail(code):
    raise RuntimeError(code)


def write_once(path, value):
    path = Path(path)
    if path.parent.resolve() != path.parent:
        fail('OUTPUT_PARENT_ALIAS')
    if not (path.is_relative_to(BASE) or path.is_relative_to(DEST)):
        fail('EXACT_OUTPUT_SCOPE_REQUIRED')
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def collect_refs(value):
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'bytes', 'sha256')):
            yield value
        else:
            for item in value.values():
                yield from collect_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_refs(item)


def exact_inventory():
    helpers.verify({'path': str(MANIFEST), 'bytes': 3403430, 'sha256': MANIFEST_SHA})
    manifest = json.loads(MANIFEST.read_bytes())
    if (manifest['session_count'], manifest['original_native_frame_count']) != (199, 29322):
        fail('SESSION_SCOPE_DRIFT')
    entries = {}
    destinations = set()
    for row in manifest['sessions']:
        sid = row['session_id']
        full = Path(row['original_full_session'])
        source = Path(row['source_session'])
        if (NAV / 'sessions' / sid).resolve() != full:
            fail('NAVIGATION_LINK_DRIFT:' + sid)
        roots = (full, source)
        if any(not p.is_relative_to(ROOT / '_run/current') or p.resolve() != p for p in roots):
            fail('SOURCE_ROOT_SCOPE')
        for group in ('files', 'stage_files', 'source_files', 'stage_receipts'):
            for ref in collect_refs(row.get(group, {})):
                p = Path(ref['path'])
                if p.is_relative_to(full):
                    suffix = p.relative_to(full)
                elif p.is_relative_to(source):
                    suffix = p.relative_to(source)
                else:
                    fail('REF_OUTSIDE_SESSION:' + str(p))
                relative = str(Path('sessions') / sid / suffix)
                current = helpers.metadata(p)
                if p.resolve() != p or not stat.S_ISREG(current['mode']) or current['size'] != ref['bytes']:
                    fail('SOURCE_FILE_TYPE_OR_SIZE_DRIFT:' + str(p))
                item = dict(source_path=str(p), destination_relative=relative,
                            bytes=ref['bytes'], sha256=ref['sha256'],
                            source_metadata=current, session_id=sid, kind='session_payload')
                if str(p) in entries:
                    old = entries[str(p)]
                    if (old['destination_relative'], old['bytes'], old['sha256']) != (relative, ref['bytes'], ref['sha256']):
                        fail('DUPLICATE_REF_DISAGREEMENT')
                    continue
                if relative in destinations:
                    fail('DESTINATION_COLLISION:' + relative)
                destinations.add(relative)
                entries[str(p)] = item
        actual = set()
        for session_root in roots:
            for directory, dirs, files in os.walk(session_root, followlinks=False):
                for name in dirs + files:
                    p = Path(directory) / name
                    if p.is_symlink():
                        fail('SOURCE_TREE_ALIAS')
                actual.update(str(Path(directory) / name) for name in files)
        expected = {k for k, v in entries.items() if v['session_id'] == sid}
        if actual != expected:
            fail('UNBOUND_SESSION_FILE:' + sid)
    payload = list(entries.values())
    if len(payload) != 3980 or sum(v['bytes'] for v in payload) != 35838424392:
        fail('EXACT_PAYLOAD_BUDGET_DRIFT')
    for p in sorted(NAV.iterdir()):
        if p.name == 'sessions':
            continue
        ref = helpers.pin(p)
        entries[str(p)] = dict(source_path=str(p), destination_relative='provenance/original_unified_delivery/' + p.name,
                              bytes=ref['bytes'], sha256=ref['sha256'],
                              source_metadata=helpers.metadata(p), kind='original_navigation_metadata')
    if len(entries) != 3990:
        fail('NAVIGATION_METADATA_SCOPE_DRIFT')
    return sorted(entries.values(), key=lambda v: v['destination_relative'])


class Runtime:
    def __init__(self, phase, inputs):
        self.started = time.monotonic()
        self.wall = 300 if phase == 'E0_FREEZE' else 5400
        self.phase = phase
        self.owned = False
        self.pins = inputs
        self.last_heartbeat = 0
        self.progress = dict(completed_files=0, completed_bytes=0)
        self.stop_event = threading.Event()
        host_pid = int(Path('/proc/self/stat').read_text().split(' ', 1)[0])
        current = helpers.identity(host_pid)
        if current is None:
            fail('REAL_WRITER_IDENTITY_REQUIRED')
        self.writer = dict(**current, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                           executor_epoch=0, fencing_token=secrets.token_hex(24),
                           deadline_at=(helpers.now() + timedelta(seconds=self.wall)).isoformat())
        if LOCK.is_symlink() or LOCK.parent.resolve() != LOCK.parent:
            fail('MAINTENANCE_LOCK_ALIAS')
        self.lock = LOCK.open('a+', encoding='utf-8')
        opened = os.fstat(self.lock.fileno())
        self.writer.update(lock_device=opened.st_dev, lock_inode=opened.st_ino)
        self.out = BASE / (phase + '_' + helpers.now().strftime('%H%M%S') + '_' + self.writer['fencing_token'][:8])
        self.out.mkdir()

    def claim(self):
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owned = True
        self.lock.seek(0)
        text = self.lock.read()
        previous = json.loads(text) if text else {}
        self.writer['executor_epoch'] = int(previous.get('executor_epoch', 0)) + 1
        self.lock.seek(0)
        self.lock.truncate()
        json.dump(self.writer, self.lock)
        helpers.read_back_private_lock(self.lock, LOCK, self.writer)
        self.packet = dict(schema='exact-independent-maintenance-export-v1', task_id=self.out.name,
                           phase=self.phase, registered_at=helpers.now().isoformat(),
                           deadline_at=self.writer['deadline_at'], writer=self.writer,
                           authorization_ref=helpers.pin(AUTH), inputs=self.pins,
                           run_signature=hashlib.sha256(canonical(dict(phase=self.phase, inputs=self.pins, destination=str(DEST)))).hexdigest(),
                           read_set=[str(ROOT / 'AGENTS.md'), str(PROTOCOL), str(HELPER), str(MANIFEST),
                                     str(AUTH), *[p['path'] for p in self.pins]],
                           expanded_read_set=['EXACT_E0_INVENTORY_ONLY'],
                           write_set=[str(self.out), str(LOCK)] + ([str(DEST)] if self.phase == 'E1_EXPORT' else []),
                           execution_allowed=True, deletion_allowed=False,
                           production_governance_write_allowed=False, algorithm_execution_allowed=False,
                           budgets=dict(wall_seconds=self.wall, cpu_threads=2, io_workers=2, gpu=0,
                                        model_calls=0, training_steps=0, payload_files=3980,
                                        payload_bytes=35838424392, metadata_files=10),
                           required_outputs=['RESULT.json'])
        write_once(self.out / 'TASK_PACKET.json', self.packet)
        write_once(self.out / 'WRITER.json', self.writer)
        self.packet_pin = helpers.pin(self.out / 'TASK_PACKET.json')
        for ref in self.pins:
            helpers.verify(ref)
        print(json.dumps(dict(status='REGISTERED', phase=self.phase, task=str(self.out))), flush=True)
        self.guard()

    def guard(self):
        if time.monotonic() - self.started >= self.wall:
            fail('FINITE_DEADLINE_REACHED')
        current = helpers.identity(self.writer['pid'])
        if current is None or current['startticks'] != self.writer['startticks']:
            fail('WRITER_IDENTITY_DRIFT')
        if json.loads(LOCK.read_bytes()) != self.writer:
            fail('FENCING_TOKEN_DRIFT')
        if time.monotonic() - self.last_heartbeat >= 60:
            self.last_heartbeat = time.monotonic()
            value = dict(observed_at=helpers.now().isoformat(), phase=self.phase, writer=self.writer,
                         elapsed_seconds=round(time.monotonic() - self.started, 2), **self.progress)
            # Mutable runtime state only; no immutable receipt is overwritten.
            (self.out / 'STATE.json').write_bytes(canonical(value) + b'\n')
            print(json.dumps(value, ensure_ascii=False), flush=True)

    def finish(self, result):
        result.update(task_id=self.out.name, finished_at=helpers.now().isoformat(),
                      task_packet=self.packet_pin if hasattr(self, 'packet_pin') else None,
                      writer=self.writer, elapsed_seconds=round(time.monotonic() - self.started, 2),
                      deleted_files=0, production_governance_writes=0, model_calls=0)
        write_once(self.out / 'RESULT.json', result)
        if self.owned:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        print(json.dumps(dict(status=result['status'], result=str(self.out / 'RESULT.json')), ensure_ascii=False), flush=True)
        return result


def copy_verified(item, runtime):
    if runtime.stop_event.is_set():
        fail('EXPORT_STOPPED_BY_OTHER_FILE')
    if time.monotonic() - runtime.started >= runtime.wall:
        fail('FINITE_DEADLINE_REACHED')
    source = Path(item['source_path'])
    if helpers.metadata(source) != item['source_metadata'] or source.resolve() != source:
        fail('SOURCE_METADATA_DRIFT:' + str(source))
    target = DEST / item['destination_relative']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve() != target.parent or target.exists() or target.is_symlink():
        fail('TARGET_ALIAS_OR_ALREADY_EXISTS:' + str(target))
    temporary = target.with_name(target.name + '.partial.' + runtime.writer['fencing_token'][:8])
    digest = hashlib.sha256()
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as incoming, temporary.open('xb') as outgoing:
        opened = os.fstat(incoming.fileno())
        if (opened.st_dev, opened.st_ino) != (item['source_metadata']['device'], item['source_metadata']['inode']):
            fail('SOURCE_OPEN_RACE')
        while True:
            if runtime.stop_event.is_set():
                fail('EXPORT_STOPPED_BY_OTHER_FILE')
            block = incoming.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            outgoing.write(block)
            if time.monotonic() - runtime.started >= runtime.wall:
                fail('FINITE_DEADLINE_REACHED')
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if digest.hexdigest() != item['sha256'] or helpers.metadata(source) != item['source_metadata']:
        fail('SOURCE_SHA_OR_METADATA_DRIFT:' + str(source))
    destination_digest = hashlib.sha256()
    with temporary.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            if runtime.stop_event.is_set() or time.monotonic() - runtime.started >= runtime.wall:
                fail('EXPORT_STOPPED_OR_DEADLINE_REACHED')
            destination_digest.update(block)
    if destination_digest.hexdigest() != item['sha256'] or temporary.stat().st_size != item['bytes']:
        fail('NAS_READBACK_SHA_OR_SIZE_FAILED:' + str(target))
    # Hard-link publication is atomic and never overwrites an existing target.
    os.link(temporary, target, follow_symlinks=False)
    temporary.unlink()
    return dict(source_path=item['source_path'], destination_relative=item['destination_relative'],
                bytes=item['bytes'], sha256=item['sha256'], session_id=item.get('session_id'),
                source_metadata=item['source_metadata'], kind=item['kind'],
                source_SHA_verified=True, destination_readback_SHA_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('freeze', 'export'), required=True)
    parser.add_argument('--freeze-result')
    args = parser.parse_args()
    phase = 'E0_FREEZE' if args.phase == 'freeze' else 'E1_EXPORT'
    auth = json.loads(AUTH.read_bytes())
    if auth['destination'] != str(DEST) or auth['source_manifest_sha256'] != MANIFEST_SHA:
        fail('EXACT_AUTHORIZATION_REQUIRED')
    inputs = [helpers.pin(AUTH), helpers.pin(Path(__file__).absolute()), helpers.pin(HELPER),
              helpers.pin(PROTOCOL), helpers.pin(MANIFEST)]
    freeze = None
    if phase == 'E1_EXPORT':
        if not args.freeze_result:
            fail('FROZEN_RESULT_REQUIRED')
        p = Path(args.freeze_result)
        if not p.is_relative_to(BASE) or p.resolve() != p:
            fail('FREEZE_RESULT_SCOPE')
        inputs.append(helpers.pin(p))
        freeze = json.loads(p.read_bytes())
        if freeze['status'] != 'PASSED_FREEZE':
            fail('FREEZE_NOT_COMPLETE')
        inputs.append(freeze['inventory'])
        helpers.verify(freeze['inventory'])
    runtime = Runtime(phase, inputs)
    try:
        runtime.claim()
        if phase == 'E0_FREEZE':
            entries = exact_inventory()
            write_once(runtime.out / 'INVENTORY.json', dict(schema='exact-original199-export-inventory-v1',
                destination=str(DEST), parent_manifest=helpers.pin(MANIFEST), files=entries,
                payload_files=3980, payload_bytes=35838424392, metadata_files=10,
                deletion_allowed=False))
            result = dict(status='PASSED_FREEZE', inventory=helpers.pin(runtime.out / 'INVENTORY.json'))
        else:
            inventory = json.loads(Path(freeze['inventory']['path']).read_bytes())
            if DEST.parent.resolve() != DEST.parent or DEST.exists() or DEST.is_symlink():
                fail('DESTINATION_MUST_BE_NEW_EXACT_DIRECTORY')
            DEST.mkdir()
            write_once(DEST / 'EXPORT_IN_PROGRESS.json', dict(task_packet=runtime.packet_pin,
                status='INCOMPLETE_NOT_A_DATASET_DELIVERY', destination=str(DEST)))
            completed = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(copy_verified, item, runtime) for item in inventory['files']]
                try:
                    for future in as_completed(futures):
                        row = future.result()
                        completed.append(row)
                        runtime.progress = dict(completed_files=len(completed), completed_bytes=sum(v['bytes'] for v in completed))
                        with (runtime.out / 'COPIED_AND_SHA_VERIFIED.jsonl').open('ab') as journal:
                            journal.write(canonical(row) + b'\n')
                        runtime.guard()
                except Exception:
                    runtime.stop_event.set()
                    for future in futures:
                        future.cancel()
                    raise
            completed.sort(key=lambda v: v['destination_relative'])
            dataset = dict(schema='pick-place-original199-relocatable-export-v1', status='EXPORTED_BYTES_VERIFIED',
                sessions=199, original_native_frames=29322, payload_files=3980, payload_bytes=35838424392,
                metadata_files=10, files=completed, parent_manifest=helpers.pin(MANIFEST),
                original_execution_complete=True, all199_visual_accepted=False, human_review_status='PENDING',
                control_ground_truth=False, physical_contact_ground_truth=False, training_eligible=False,
                producer_code_or_weights_exported=False, contains_external_symlinks=False,
                original_reference_strings_are_provenance_not_current_locations=True)
            write_once(DEST / 'EXPORT_MANIFEST_V1.json', dataset)
            readme = ('# pick_place_1008 原版 Raw→Robot V9 实体数据\n\n'
                '199会话，29322原始帧。sessions/001 至 sessions/199 均为实体目录，包含原版视频、动作、帧映射、Raw/Clean/Mask及source。\n\n'
                'EXPORT_MANIFEST_V1.json 记录可移植相对路径及逐文件SHA；provenance/original_unified_delivery保留原清单原字节。原JSON中的项目绝对路径只作历史来源，不是此数据集的使用入口。\n\n'
                '此处不是后续优化版，不代表全量视觉合格、真实机器人控制或可训练标签；NUM失败、UNKNOWN和人工待审状态保留。Clean仅为离线视觉。\n\n'
                '只有EXPORT_COMPLETE_V1.json存在且status为PASSED_EXPORT才算完整导出。EXPORT_IN_PROGRESS.json是保留的开始记录，不表示完成。\n')
            with (DEST / 'README_ZH.md').open('x', encoding='utf-8') as stream:
                stream.write(readme)
                stream.flush()
                os.fsync(stream.fileno())
            exported = dict(path=str(DEST / 'EXPORT_MANIFEST_V1.json'),
                bytes=(DEST / 'EXPORT_MANIFEST_V1.json').stat().st_size,
                sha256=hashlib.sha256((DEST / 'EXPORT_MANIFEST_V1.json').read_bytes()).hexdigest())
            completion = dict(status='PASSED_EXPORT', sessions=199, original_native_frames=29322,
                verified_files=len(completed), verified_bytes=sum(v['bytes'] for v in completed),
                export_manifest=exported, source_and_NAS_full_SHA_verified=True,
                no_original_data_or_result_deleted=True, completed_at=helpers.now().isoformat(),
                visual_quality_pass=False, training_eligible=False)
            write_once(DEST / 'EXPORT_COMPLETE_V1.json', completion)
            result = dict(**completion, destination=str(DEST), deletion_status='SEPARATE_REFERENCE_PROOF_REQUIRED')
    except Exception as exc:
        result = dict(status='STOPPED', failure=type(exc).__name__ + ':' + str(exc),
                      verified_files=runtime.progress['completed_files'], verified_bytes=runtime.progress['completed_bytes'],
                      no_original_data_or_result_deleted=True)
    result = runtime.finish(result)
    return 0 if result['status'].startswith('PASSED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
