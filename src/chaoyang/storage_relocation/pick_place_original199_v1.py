"""Read original199 evidence through one sealed, exact NAS relocation.

Logical references never change. This is an explicitly selected successor IO
adapter, not a monkeypatch or permission for historical operations to run.
No algorithm, output-path resolver, writes, model calls or execution CLI exist.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/pick_place_1008_nas_export_20261010_v1'
NAV = ROOT / 'review/pick_place_1008/ALL199_ORIGINAL_EXECUTION_OUTPUTS_UNIFIED_DELIVERY_V1'
DEST = Path('/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1')
RAW_ROOT = Path('/path/to/private_storage/datasets/ego/pick_place_1008')
BINDING_SCHEMA = 'pick-place-original199-storage-binding-v1'
SOURCE_MANIFEST_SHA = '393f69b8b5c2da8e93ff7ec38a6fecfd0f0bcd1b4edb65691fa0dcd0b9d93cf3'
SOURCE_INVENTORY_SHA = '1e82ad90bf9efa4212c1783b960cb9dbfe78d54f9a6fe61e5a796a3be7a18a8a'
EXPECTED_SESSIONS = 199
EXPECTED_NATIVE_FRAMES = 29322
EXPECTED_PAYLOAD_FILES = 3980
EXPECTED_PAYLOAD_BYTES = 35838424392
EXPECTED_METADATA_FILES = 10
JSON_MAX_BYTES = 16 << 20
ORIGINAL_STAGE_SCHEMA = 'pick_place_1008_sealed_stage_v1'
ORIGINAL_STAGES = ('source', 'motion', 'gpu', 'render')


class StorageRelocationError(ValueError):
    """Fail-closed storage identity, boundary or seal rejection."""


def _need(value, code):
    if not value:
        raise StorageRelocationError(code)


def _shape(value):
    _need(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}, 'EXACTREF_SCHEMA')
    _need(type(value['path']) is str and type(value['bytes']) is int and value['bytes'] >= 0,
          'EXACTREF_TYPED_PATH_BYTES')
    digest = value['sha256']
    _need(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
          'EXACTREF_TYPED_SHA')
    return dict(value)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _plain(value, root):
    p, root = Path(value), Path(root)
    _need(p.is_absolute() and str(p) == str(value) and '..' not in p.parts and p.is_relative_to(root)
          and p != root, 'LEXICAL_READ_SCOPE')
    for item in (p, *p.parents):
        _need(not item.is_symlink(), 'SYMLINK_READ_REJECTED')
        if item == root:
            break
    _need(root.resolve(strict=True) == root, 'READ_ROOT_ALIAS')
    return p


@contextmanager
def _opened(path, root):
    """Walk beneath a checked root using no-follow read-only descriptors."""
    p = _plain(str(path), root)
    descriptors = []
    stream = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)
        directory = os.open(root, flags)
        descriptors.append(directory)
        parts = p.relative_to(root).parts
        for name in parts[:-1]:
            directory = os.open(name, flags, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0),
                     dir_fd=directory)
        stream = os.fdopen(fd, 'rb')
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode), 'REGULAR_READ_REQUIRED')
        _need(_identity(p.lstat()) == _identity(before), 'READ_OPEN_RACE')
        yield stream, before
        _plain(str(p), root)
        _need(_identity(os.fstat(stream.fileno())) == _identity(before) == _identity(p.lstat()),
              'READ_FILE_IDENTITY_DRIFT')
    except (OSError, RuntimeError) as exc:
        raise StorageRelocationError('READ_UNAVAILABLE_OR_ALIAS:' + str(path)) from exc
    finally:
        if stream is not None:
            stream.close()
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _hash_stream(stream, reference):
    digest, count = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(4 << 20), b''):
        count += len(block)
        digest.update(block)
    _need(count == reference['bytes'] and digest.hexdigest() == reference['sha256'],
          'READ_SHA_OR_BYTES_DRIFT:' + reference['path'])


def _json_exact(reference, root):
    reference = _shape(reference)
    _need(reference['bytes'] <= JSON_MAX_BYTES, 'BOUNDED_JSON_REFERENCE_REQUIRED')
    with _opened(reference['path'], root) as (stream, before):
        data = stream.read(JSON_MAX_BYTES + 1)
        _need(len(data) == reference['bytes'] and hashlib.sha256(data).hexdigest() == reference['sha256'],
              'JSON_SHA_OR_BYTES_DRIFT')
        try:
            result = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageRelocationError('JSON_INVALID') from exc
    return result, _identity(before)


class Original199Reader:
    """Only a project-maintenance binding can authorize exact relocated reads."""

    def __init__(self, binding_ref):
        self._cache = {}
        self._controls = []
        self._mapped = {}
        self._raw_refs = {}
        self._original_candidate_roots = set()
        self.binding_ref = _shape(binding_ref)
        _plain(self.binding_ref['path'], BASE)
        binding = self._control_json(self.binding_ref, BASE)
        _need(isinstance(binding, dict) and set(binding) == {
            'schema', 'dataset_root', 'export_manifest', 'export_complete',
            'source_inventory', 'source_manifest'}, 'BINDING_SCHEMA_KEYS')
        _need(binding['schema'] == BINDING_SCHEMA and binding['dataset_root'] == str(DEST),
              'EXACT_FROZEN_DATASET_ROOT')
        self.binding = binding
        complete_ref, export_ref = _shape(binding['export_complete']), _shape(binding['export_manifest'])
        _need(complete_ref['path'] == str(DEST / 'EXPORT_COMPLETE_V1.json') and
              export_ref['path'] == str(DEST / 'EXPORT_MANIFEST_V1.json'), 'EXACT_EXPORT_SEAL_PATHS')
        inventory_ref, original_ref = _shape(binding['source_inventory']), _shape(binding['source_manifest'])
        _plain(inventory_ref['path'], BASE)
        _need(inventory_ref['sha256'] == SOURCE_INVENTORY_SHA, 'FROZEN_INVENTORY_SHA')
        _need(original_ref['path'] == str(NAV / 'MANIFEST_ALL199_ORIGINAL_EXECUTION_OUTPUTS_V1.json') and
              original_ref['sha256'] == SOURCE_MANIFEST_SHA, 'FROZEN_ORIGINAL_MANIFEST')
        complete = self._control_json(complete_ref, DEST)
        exported = self._control_json(export_ref, DEST)
        inventory = self._control_json(inventory_ref, BASE)
        original = self._control_json(original_ref, ROOT)
        _need(complete.get('status') == 'PASSED_EXPORT' and complete.get('export_manifest') == export_ref and
              complete.get('source_and_NAS_full_SHA_verified') is True, 'EXPORT_NOT_FULLY_SEALED')
        _need(exported.get('schema') == 'pick-place-original199-relocatable-export-v1' and
              exported.get('status') == 'EXPORTED_BYTES_VERIFIED' and
              exported.get('parent_manifest') == inventory.get('parent_manifest') == original_ref and
              inventory.get('destination') == str(DEST), 'EXPORT_INVENTORY_PARENT_IDENTITY')
        _need(exported.get('contains_external_symlinks') is False and exported.get('training_eligible') is False
              and exported.get('control_ground_truth') is False and exported.get('all199_visual_accepted') is False
              and complete.get('visual_quality_pass') is False and complete.get('training_eligible') is False,
              'EXPORT_QUALITY_AND_TRUTH_DOMAIN')
        for value in (exported, complete):
            _need(type(value.get('sessions')) is int and value['sessions'] == EXPECTED_SESSIONS and
                  type(value.get('original_native_frames')) is int and
                  value['original_native_frames'] == EXPECTED_NATIVE_FRAMES, 'EXPORT_SESSION_FRAME_SCOPE')
        _need(original.get('session_count') == EXPECTED_SESSIONS and
              original.get('original_native_frame_count') == EXPECTED_NATIVE_FRAMES,
              'ORIGINAL_SESSION_FRAME_SCOPE')
        sessions = {row['session_id']: row for row in original['sessions']}
        _need(len(sessions) == len(original['sessions']) == EXPECTED_SESSIONS and
              set(sessions) == {f'{i:03d}' for i in range(1, EXPECTED_SESSIONS + 1)}, 'ORIGINAL_SESSION_IDS')
        for row in sessions.values():
            for name in ('original_full_session', 'source_session'):
                folder = _plain(row[name], ROOT)
                _need(folder.name == row['session_id'] and folder.parent.name == 'sessions',
                      'ORIGINAL_LOGICAL_SESSION_FOLDER')
                self._original_candidate_roots.add(folder.parent.parent)
        frozen_files, copied_files = inventory.get('files'), exported.get('files')
        expected_count = EXPECTED_PAYLOAD_FILES + EXPECTED_METADATA_FILES
        _need(isinstance(frozen_files, list) and isinstance(copied_files, list) and
              len(frozen_files) == len(copied_files) == expected_count, 'FULL_EXPORT_MAPPING_REQUIRED')
        copied = {row['source_path']: row for row in copied_files}
        _need(len(copied) == expected_count, 'DUPLICATE_EXPORTED_SOURCE')
        destinations, payload_count, payload_bytes, metadata_count = set(), 0, 0, 0
        for row in frozen_files:
            logical = _shape(dict(path=row['source_path'], bytes=row['bytes'], sha256=row['sha256']))
            p = _plain(logical['path'], ROOT)
            _need(logical['path'] not in self._mapped, 'DUPLICATE_ORIGINAL_SOURCE')
            rel = Path(row['destination_relative'])
            _need(not rel.is_absolute() and '..' not in rel.parts and str(rel) == row['destination_relative']
                  and str(rel) not in destinations, 'DESTINATION_COLLISION_OR_TRAVERSAL')
            if row['kind'] == 'session_payload':
                sid = row['session_id']
                _need(sid in sessions, 'MAPPED_SESSION_NOT_FROZEN')
                roots = [Path(sessions[sid][name]) for name in ('original_full_session', 'source_session')]
                _need(all(root.is_relative_to(ROOT / '_run/current') for root in roots), 'ORIGINAL_RUN_ROOT_SCOPE')
                matched = next((root for root in roots if p.is_relative_to(root)), None)
                _need(matched is not None and rel == Path('sessions') / sid / p.relative_to(matched),
                      'MAPPED_WRONG_SESSION_OR_SUFFIX')
                payload_count += 1
                payload_bytes += logical['bytes']
            else:
                _need(row['kind'] == 'original_navigation_metadata' and p.parent == NAV and
                      rel == Path('provenance/original_unified_delivery') / p.name,
                      'MAPPED_METADATA_SCOPE')
                metadata_count += 1
            other = copied.get(logical['path'])
            _need(isinstance(other, dict) and all(other.get(k) == row.get(k) for k in
                  ('source_path', 'destination_relative', 'bytes', 'sha256', 'source_metadata', 'kind', 'session_id'))
                  and other.get('source_SHA_verified') is True and
                  other.get('destination_readback_SHA_verified') is True, 'EXPORT_ROW_NOT_FROZEN_AND_VERIFIED')
            destinations.add(str(rel))
            self._mapped[logical['path']] = (logical, DEST / rel)
        _need((payload_count, payload_bytes, metadata_count) ==
              (EXPECTED_PAYLOAD_FILES, EXPECTED_PAYLOAD_BYTES, EXPECTED_METADATA_FILES), 'EXACT_EXPORT_TOTALS')
        _need(complete.get('verified_files') == expected_count and complete.get('verified_bytes') ==
              sum(row['bytes'] for row in frozen_files), 'COMPLETE_EXPORT_COUNTS')
        for value in (inventory, exported):
            _need(value.get('payload_files') == payload_count and value.get('payload_bytes') == payload_bytes
                  and value.get('metadata_files') == metadata_count, 'MANIFEST_EXPORT_TOTALS')
        self._source_inputs = {}
        self._source_identities = {}
        # Small retained JSONs establish the precise original RAW NAS allowlist.
        # No media or large NPZ is loaded during construction.
        for sid, row in sessions.items():
            input_ref = _shape(row['source_files']['source/SESSION_INPUT.json'])
            _need(input_ref['path'] in self._mapped and self._mapped[input_ref['path']][0] == input_ref,
                  'PREPARED_INPUT_NOT_IN_FROZEN_MAP')
            meta, _ = _json_exact(input_ref, ROOT)
            # This is the original producer's exact ID rule, not the short
            # delivery index ID: prepare_session uses "pick_place_1008_" + sid.
            expected_id = 'pick_place_1008_' + sid
            expected_raw_dir = str(RAW_ROOT / sid)
            _need(meta.get('session_id') == expected_id and meta.get('source_dir') == expected_raw_dir,
                  'PREPARED_INPUT_SESSION_IDENTITY')
            logical_folder = str(Path(input_ref['path']).parent)
            self._source_inputs[logical_folder] = input_ref
            self._source_identities[logical_folder] = (expected_id, expected_raw_dir)
            refs = meta.get('source_refs')
            _need(isinstance(refs, dict), 'ORIGINAL_RAW_REFS_REQUIRED')
            self._register_original_raw_closure(sid, row, meta, refs)
        self._guard()

    def _register_original_raw_closure(self, sid, row, meta, source_refs):
        """Admit only the RAW pins shared by the four frozen original stages.

        SESSION_INPUT keys are filenames, while stage keys are semantic roles.
        Their overlap is therefore compared by exact logical path/bytes/SHA.
        The two collection sidecars require the same stage proof as all other
        inputs; their filenames alone never grant an external read.
        """
        def by_path(refs):
            _need(isinstance(refs, dict) and bool(refs), 'ORIGINAL_RAW_CLOSURE_REQUIRED')
            normalized = {}
            for role, value in refs.items():
                _need(type(role) is str and bool(role), 'ORIGINAL_RAW_ROLE_REQUIRED')
                pin = _shape(value)
                _plain(pin['path'], RAW_ROOT / sid)
                _need(pin['path'] not in normalized, 'ORIGINAL_RAW_DUPLICATE_PATH')
                normalized[pin['path']] = pin
            return normalized

        source_by_path = by_path(source_refs)
        receipts = row.get('stage_receipts')
        _need(isinstance(receipts, dict) and set(receipts) == set(ORIGINAL_STAGES),
              'ORIGINAL_FOUR_STAGE_REFERENCES_REQUIRED')
        native_frames = row.get('original_native_frames')
        _need(type(native_frames) is int and native_frames > 0 and
              type(meta.get('frame_count')) is int and meta['frame_count'] == native_frames,
              'ORIGINAL_RAW_NATIVE_FRAME_IDENTITY')
        common_refs = None
        for stage in ORIGINAL_STAGES:
            pin = _shape(receipts[stage])
            folder = Path(row['source_session'] if stage == 'source' else row['original_full_session'])
            expected_path = folder / ('STAGE_' + stage + '.json')
            _need(pin['path'] == str(expected_path) and
                  self._mapped.get(pin['path'], (None,))[0] == pin,
                  'ORIGINAL_STAGE_NOT_EXACT_FROZEN_MAP')
            # The immutable parent manifest and E0 map pin these exact bytes.
            # Snapshot the allowlist once instead of adding 796 local files to
            # every per-reference guard. Later stage reads still verify this
            # original pin; mutable JSON can never extend the copied allowlist.
            receipt, _ = _json_exact(pin, ROOT)
            _need(isinstance(receipt, dict) and receipt.get('schema') == ORIGINAL_STAGE_SCHEMA and
                  receipt.get('status') == 'COMPLETE', 'ORIGINAL_RAW_STAGE_SCHEMA_OR_STATUS')
            binding = receipt.get('binding')
            _need(isinstance(binding, dict) and binding.get('stage') == stage and
                  binding.get('session_id') == sid and
                  type(binding.get('raw_frame_count')) is int and
                  binding['raw_frame_count'] == native_frames, 'ORIGINAL_RAW_STAGE_IDENTITY')
            digest = hashlib.sha256(json.dumps(binding, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
            _need(receipt.get('binding_sha256') == digest, 'ORIGINAL_RAW_STAGE_BINDING_DIGEST')
            raw = binding.get('raw_source')
            normalized = by_path(raw)
            if common_refs is None:
                common_refs = dict(raw)
            else:
                _need(raw == common_refs, 'ORIGINAL_RAW_FOUR_STAGE_CLOSURE_CONFLICT')
            _need(all(normalized.get(path) == value for path, value in source_by_path.items()),
                  'ORIGINAL_RAW_STAGE_SOURCE_REFERENCE_CONFLICT')

        common_by_path = by_path(common_refs)
        extras = {role: pin for role, pin in common_refs.items()
                  if pin['path'] not in source_by_path}
        _need(set(extras) == {'selection', 'workflow'}, 'ORIGINAL_RAW_UNSUPPORTED_EXTRA_ROLES')
        for role, pin in extras.items():
            _need(pin['path'] == str(RAW_ROOT / sid / 'raw' / ('capture.' + role + '.json')) and
                  pin['bytes'] <= JSON_MAX_BYTES, 'ORIGINAL_RAW_COLLECTION_SIDECAR_IDENTITY')
        for path, pin in common_by_path.items():
            old = self._raw_refs.get(path)
            _need(old is None or old == pin, 'ORIGINAL_RAW_REFERENCE_CONFLICT')
            self._raw_refs[path] = dict(pin)

    def _control_json(self, reference, root):
        value, identity = _json_exact(reference, root)
        self._controls.append((reference, root, identity))
        return value

    def _guard(self):
        for reference, root, expected in self._controls:
            p = _plain(reference['path'], root)
            _need(_identity(p.lstat()) == expected, 'BINDING_OR_MANIFEST_IDENTITY_DRIFT')

    def _location(self, reference):
        reference = _shape(reference)
        logical = reference['path']
        if logical in self._mapped:
            expected, physical = self._mapped[logical]
            _need(reference == expected, 'MAPPED_ORIGINAL_REFERENCE_DRIFT')
            # Retained metadata remains local; only frozen MP4/NPZ redirect.
            if Path(logical).suffix not in ('.mp4', '.npz'):
                return reference, _plain(logical, ROOT), ROOT
            return reference, _plain(str(physical), DEST), DEST
        p = Path(logical)
        if p.is_relative_to(ROOT):
            _need(not (p.suffix in ('.npz', '.mp4') and any(p.is_relative_to(folder)
                       for folder in self._original_candidate_roots)), 'UNBOUND_ORIGINAL_BINARY')
            return reference, _plain(logical, ROOT), ROOT
        _need(self._raw_refs.get(logical) == reference, 'UNREGISTERED_EXTERNAL_READ')
        return reference, _plain(logical, RAW_ROOT), RAW_ROOT

    def verify_ref(self, reference):
        self._guard()
        original, physical, root = self._location(reference)
        identity = _identity(physical.lstat())
        key = (str(physical), original['bytes'], original['sha256'])
        if self._cache.get(key) != identity:
            with _opened(physical, root) as (stream, before):
                _hash_stream(stream, original)
            self._cache[key] = _identity(before)
        self._guard()
        return original

    def physical_path(self, reference):
        original = self.verify_ref(reference)
        return self._location(original)[1]

    def read_json(self, reference):
        self._guard()
        original, physical, root = self._location(reference)
        actual = dict(original, path=str(physical))
        value, identity = _json_exact(actual, root)
        self._cache[(str(physical), original['bytes'], original['sha256'])] = identity
        self._guard()
        return value

    def arrays(self, reference):
        import numpy as np
        self._guard()
        original, physical, root = self._location(reference)
        _need(physical.suffix == '.npz', 'NPZ_FILE_REQUIRED')
        with _opened(physical, root) as (stream, before):
            _hash_stream(stream, original)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as values:
                result = {name: values[name] for name in values.files}
        self._cache[(str(physical), original['bytes'], original['sha256'])] = _identity(before)
        self._guard()
        return result

    def reference_for(self, logical_path):
        """Frozen original reference, or a strict retained-project metadata pin."""
        self._guard()
        p = Path(logical_path)
        if str(p) in self._mapped:
            reference = dict(self._mapped[str(p)][0])
            return self.verify_ref(reference)
        _plain(str(p), ROOT)
        digest, count = hashlib.sha256(), 0
        with _opened(p, ROOT) as (stream, _):
            for block in iter(lambda: stream.read(4 << 20), b''):
                digest.update(block)
                count += len(block)
        self._guard()
        return dict(path=str(p), bytes=count, sha256=digest.hexdigest())

    def has_ref(self, logical_path):
        self._guard()
        p = _plain(str(logical_path), ROOT)
        if str(p) in self._mapped:
            return True
        return p.is_file()

    def load_prepared(self, logical_session_folder):
        import numpy as np
        from chaoyang.pick_place_1008 import source
        folder = _plain(str(logical_session_folder), ROOT)
        reference = self._source_inputs.get(str(folder))
        _need(reference is not None, 'UNREGISTERED_PREPARED_SESSION_FOLDER')
        meta = self.read_json(reference)
        expected_id, expected_raw_dir = self._source_identities[str(folder)]
        _need(meta.get('session_id') == expected_id and meta.get('source_dir') == expected_raw_dir,
              'RAW_PREPARED_ORIGINAL_SESSION_IDENTITY')
        _need(meta.get('status') == 'COMPLETE_RAW_INPUT' and meta.get('schema_version') == source.RAW_INPUT_SCHEMA
              and meta.get('calibration', {}).get('hand_joint_schema') == source.JOINT_MAPPING_SCHEMA
              and meta['calibration'].get('manus25_to21') == source.MAP_25_TO_21.tolist(),
              'RAW_PREPARED_CANONICAL21_SCHEMA_REQUIRED')
        for name in ('video', 'arrays'):
            component = _shape(meta[name])
            _need(Path(component['path']).is_relative_to(folder), 'RAW_PREPARED_COMPONENT_OUTSIDE_SESSION')
            _need(self._mapped.get(component['path'], (None,))[0] == component,
                  'RAW_PREPARED_COMPONENT_NOT_FROZEN')
        arrays = self.arrays(meta['arrays'])
        group = lambda prefix: {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
        session = source.RawSession(meta['session_id'], folder, self.physical_path(meta['video']), meta['fps'],
            meta['frame_count'], meta['width'], meta['height'], arrays['K'], arrays['c2w'], arrays['c2w_editor'],
            arrays['hands'], arrays['hand_observed'], arrays['hand_source'], arrays['body'], arrays['body_observed'],
            arrays['gravity'], meta['calibration'], arrays['qpc_ns'], group('mapping__'), group('valid__'), group('native__'))
        _need(isinstance(session.qpc_ns, np.ndarray), 'RAW_PREPARED_ARRAY_PARITY')
        return session

    def upstream_refs(self, masks):
        refs = masks.get('upstream_refs')
        roles = {'actions', 'frame_mapping', 'raw_rgb', 'clean_rgb', 'masks', 'motion_info'}
        _need(isinstance(refs, dict) and roles <= refs.keys(), 'RENDER_UPSTREAM_REFERENCES_REQUIRED')
        result = {}
        for role in sorted(roles):
            reference = _shape(refs[role])
            _plain(reference['path'], ROOT)
            result[role] = self.verify_ref(reference)
        return result
