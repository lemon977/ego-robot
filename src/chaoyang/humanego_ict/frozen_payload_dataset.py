"""Frozen project-payload RAM reader, pending separate formal route adoption.

This is NOT the old per-sample original-source-stat contract. The owner gate
must bind historical admission and the new explicit research snapshot policy.
Construction/boundary verification read project bytes; __getitem__ never does.
No Torch, model, GPU, optimizer, publisher or automatic execution is included.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import re

ROOT = Path('/path/to/chaoyang')
SOURCE_POLICY = 'FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY'
DEFAULT_READER_REF = {
    'path': '/path/to/chaoyang/_run/maintenance/humanego_ict_io_recovery_engineering_20261012/attempt_0001/snapshot_reader_candidate.py',
    'bytes': 15309,
    'sha256': 'ccee814114c01f68f5a76c3b3329406bc24dc3dbe5405f73272da90668dde3fc',
}
FIELDS = ('x_rgb', 'x_ict', 'ict_mask', 'y_action', 'action_valid_mask', 'sample_weight')
SHAPES = ((3, 240, 320), (8, 29), (8,), (50, 30), (50, 30), ())
DTYPES = ('float32', 'float32', 'bool', 'float32', 'bool', 'float32')


class FrozenPayloadError(ValueError):
    pass


def _fail(code):
    raise FrozenPayloadError(code)


def _root(value):
    path = Path(value)
    if (not path.is_absolute() or not path.is_relative_to(ROOT) or '..' in path.parts
            or str(PurePosixPath(str(path))) != str(path)):
        _fail('CANONICAL_PROJECT_ROOT_REQUIRED')
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            _fail('PROJECT_ROOT_SYMLINK_FORBIDDEN')
    if not path.is_dir():
        _fail('PROJECT_ROOT_DIRECTORY_REQUIRED')
    return path


def _bytes(reference, root):
    if not isinstance(reference, dict) or set(reference) != {'path', 'bytes', 'sha256'}:
        _fail('EXACT_REF_REQUIRED')
    text = reference['path']
    if (not isinstance(text, str) or '\0' in text or not text.startswith('/')
            or str(PurePosixPath(text)) != text or '..' in PurePosixPath(text).parts
            or type(reference['bytes']) is not int or reference['bytes'] < 0
            or not isinstance(reference['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])):
        _fail('INVALID_REF')
    path = Path(text)
    if not path.is_relative_to(root):
        _fail('PROJECT_ONLY_REFERENCE_REQUIRED')
    # Lexical containment before ALL metadata, then parents before children.
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            _fail('PROJECT_INPUT_SYMLINK_FORBIDDEN')
    try:
        if not path.is_file():
            _fail('PROJECT_ORDINARY_FILE_REQUIRED')
        before = path.stat(); blob = path.read_bytes(); after = path.stat()
    except OSError as error:
        raise FrozenPayloadError('PROJECT_REFERENCE_READ_FAILED') from error
    fingerprint = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if fingerprint(before) != fingerprint(after):
        _fail('PROJECT_REFERENCE_CHANGED_DURING_READ')
    if len(blob) != reference['bytes'] or hashlib.sha256(blob).hexdigest() != reference['sha256']:
        _fail('PROJECT_REFERENCE_SHA_OR_BYTES_DRIFT')
    return blob


def _load_reader(reference):
    # Alternate path may only contain the EXACT same previously sealed code.
    if (reference.get('bytes') != DEFAULT_READER_REF['bytes']
            or reference.get('sha256') != DEFAULT_READER_REF['sha256']):
        _fail('EXACT_SEALED_PROJECT_READER_REQUIRED')
    _bytes(reference, _root(ROOT))
    spec = importlib.util.spec_from_file_location('_frozen_signed_project_payload_reader', reference['path'])
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    return reader


class FrozenPayloadDataset:
    """Predecoded exact-index CPU arrays with defensive public reads.

    max_ram_bytes bounds numeric ARRAY bytes, not process RSS. Source qualification
    is deliberately not granted here; the formally adopted owner gate must prove
    historical source/producer/parent/adoption bindings before model imports.
    """
    source_policy = SOURCE_POLICY

    def __init__(self, manifest_ref, map_ref, *, project_root=ROOT, expected_branch,
                 reader_ref=None, max_ram_bytes=1 << 30):
        if expected_branch not in {'raw', 'robotized'}:
            _fail('EXPLICIT_BRANCH_REQUIRED')
        if type(max_ram_bytes) is not int or max_ram_bytes <= 0:
            _fail('FINITE_POSITIVE_RAM_ARRAY_BUDGET_REQUIRED')
        self._root = _root(project_root)
        self._manifest_ref, self._map_ref = copy.deepcopy(manifest_ref), copy.deepcopy(map_ref)
        self._reader_ref = copy.deepcopy(DEFAULT_READER_REF if reader_ref is None else reader_ref)
        reader = _load_reader(self._reader_ref)
        mapping = reader._json(_bytes(self._map_ref, self._root))
        if mapping.get('branch') != expected_branch or mapping.get('split') != 'train':
            _fail('EXACT_EXPECTED_TRAIN_BRANCH_REQUIRED')
        snapshot = reader.SnapshotICTDatasetCandidate(self._manifest_ref, self._map_ref, project_root=self._root)
        self.branch, self.split = expected_branch, 'train'
        self.dataset_sha = self._manifest_ref['sha256']
        self._manifest = copy.deepcopy(snapshot.manifest)
        self._samples = copy.deepcopy(snapshot.samples)
        self._metadata = {k: copy.deepcopy(v) for k, v in snapshot.manifest.items() if k != 'samples'}
        self._causal_provenance = copy.deepcopy(snapshot.provenance())
        refs = [self._manifest_ref, self._map_ref, *[item['snapshot_ref'] for item in mapping['consumed_snapshots']]]
        unique = {}
        for ref in refs:
            if ref['path'] in unique and unique[ref['path']] != ref:
                _fail('PROJECT_REFERENCE_PATH_CONFLICT')
            unique[ref['path']] = copy.deepcopy(ref)
        self._boundary_refs = tuple(unique.values())
        import numpy as np
        rows, used = [], 0
        for index, sample in enumerate(self._samples):
            item = snapshot[index]
            if any(item.get(k) != sample[k] for k in ('sample_id', 'session_id', 'source_frame_id', 'source_group_id')) or item.get('sample_index') != index:
                _fail('FROZEN_ITEM_IDENTITY_OR_ORDER_DRIFT')
            row = {}
            for key, shape, dtype in zip(FIELDS, SHAPES, DTYPES):
                array = item[key]
                if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.dtype(dtype):
                    _fail('EXACT_FROZEN_BATCH_SHAPE_OR_DTYPE_REQUIRED:' + key)
                array = array.copy()
                used += array.nbytes
                if used > max_ram_bytes:
                    _fail('FROZEN_ARRAY_RAM_BUDGET_EXCEEDED')
                array.setflags(write=False); row[key] = array
            for key, value in item.items():
                if key not in FIELDS:
                    row[key] = copy.deepcopy(value)
            rows.append(row)
        self._rows = tuple(rows)
        self.preloaded_array_bytes = used
        self._reader_items_predecoded = len(rows)
        # Recheck all consumed PROJECT refs after the complete preload; no source
        # version files or NAS paths are included in this explicit boundary.
        self.verify_all_dependencies()

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self._rows):
            _fail('EXACT_INTEGER_SAMPLE_INDEX_REQUIRED')
        # NO stat/open/read/resolve/import/revalidation here. Training _batch
        # consumes fresh writable arrays; callers cannot poison cached samples.
        return {key: value.copy() if key in FIELDS else copy.deepcopy(value)
                for key, value in self._rows[index].items()}

    @property
    def manifest_ref(self):
        return copy.deepcopy(self._manifest_ref)

    @property
    def snapshot_map_ref(self):
        return copy.deepcopy(self._map_ref)

    @property
    def samples(self):
        return copy.deepcopy(self._samples)

    @property
    def qualification(self):
        return {'schema': 'frozen-project-payload-content-qualification-v1',
            'status': 'PROJECT_PAYLOAD_CONTENT_ONLY_NOT_SOURCE_QUALIFICATION',
            'dataset_sha': self.dataset_sha, 'prepared_pair_ref': copy.deepcopy(self._manifest_ref),
            'prepared_pair': copy.deepcopy(self._manifest),
            'production_manifest_ref': copy.deepcopy(self._manifest_ref),
            'snapshot_map_ref': copy.deepcopy(self._map_ref), 'source_policy': SOURCE_POLICY,
            'branch': self.branch, 'split': self.split, 'sample_counts': {'train': len(self)},
            'source_group_splits': {'train': sorted({s['source_group_id'] for s in self._samples})},
            'group_independence_reverified': False,
            'admitted_sample_ids': [s['sample_id'] for s in self._samples],
            'normalization': copy.deepcopy(self._metadata.get('normalization')),
            'pose_clock': copy.deepcopy(self._metadata.get('pose_clock')),
            'frozen_manifest_metadata': copy.deepcopy(self._metadata),
            'original_causal_provenance': copy.deepcopy(self._causal_provenance),
            'predecoded_train_items': self._reader_items_predecoded,
            'preloaded_numeric_array_bytes': self.preloaded_array_bytes,
            'project_payload_bytes_verified': True, 'historical_source_binding_requires_owner_gate': True,
            'source_qualification_granted': False, 'current_nas_content_verified': False,
            'original_per_sample_source_stat_equivalence_proven': False,
            'adopted': False, 'production_authorized': False,
            'control_ground_truth': False, 'physical_deployment_authorized': False}

    def verify_all_dependencies(self):
        """Owner calls at explicit admission/sealing boundaries, NEVER a batch.

        Only project map/manifest/consumed snapshot refs and frozen reader code
        are checked. Original source-version paths are deliberately not probed.
        """
        for ref in self._boundary_refs:
            _bytes(ref, self._root)
        _bytes(self._reader_ref, _root(ROOT))

    def assert_production_ready(self):
        _fail('FORMAL_FROZEN_PAYLOAD_ROUTE_ADOPTION_REQUIRED')
