"""Small project-local fixtures; no NAS, models, rendering or production writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chaoyang.storage_relocation import pick_place_original199_v1 as io


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


@pytest.fixture
def fixture(tmp_path, monkeypatch, request):
    from chaoyang.pick_place_1008 import source
    root = tmp_path / 'project'
    base = root / '_run/maintenance/export'
    nav = root / 'review/original199'
    dest = tmp_path / 'nas/exported'
    raw_root = tmp_path / 'nas/original'
    for name, value in {'ROOT': root, 'BASE': base, 'NAV': nav, 'DEST': dest,
                        'RAW_ROOT': raw_root, 'EXPECTED_SESSIONS': 1,
                        'EXPECTED_NATIVE_FRAMES': 3, 'EXPECTED_METADATA_FILES': 1}.items():
        monkeypatch.setattr(io, name, value)
    for directory in (root, base, nav, dest, raw_root / '001/raw'):
        directory.mkdir(parents=True, exist_ok=True)
    prepared = root / '_run/current/source/attempts/attempt_0001/candidate/sessions/001'
    full = root / '_run/current/full/attempts/attempt_0001/candidate/sessions/001'
    (prepared / 'source').mkdir(parents=True)
    full.mkdir(parents=True)
    raw = raw_root / '001/raw/vst.h264'
    raw.write_bytes(b'original raw source; not a decoder fixture')
    arrays = prepared / 'source/source.npz'
    fields = dict(K=np.eye(3), c2w=np.repeat(np.eye(4)[None], 3, axis=0),
                  c2w_editor=np.repeat(np.eye(4)[None], 3, axis=0),
                  hands=np.zeros((2, 3, 21, 3)), hand_observed=np.zeros((2, 3), bool),
                  hand_source=np.full((2, 3), 'DIAGNOSTIC'), body=np.zeros((3, 24, 3)),
                  body_observed=np.zeros(3, bool), gravity=np.array([0., 0., -1.]),
                  qpc_ns=np.array([0, 33333333, 66666666], np.int64),
                  mapping__raw_frame_id=np.arange(3), valid__head_valid=np.zeros(3, bool),
                  native__hand_source=np.full((2, 3), 'DIAGNOSTIC'))
    np.savez_compressed(arrays, **fields)
    video = prepared / 'source/physical_eye.mp4'
    video.write_bytes(b'pinned bytes only, not decoded by a prepared reader')
    actions = full / 'actions.npz'
    np.savez_compressed(actions, q=np.arange(6).reshape(3, 2))
    motion = full / 'motion.json'
    save(motion, dict(status='COMPLETE', limits_rad_s=3.6, fail_frames=[2]))
    meta = dict(schema_version=source.RAW_INPUT_SCHEMA, status='COMPLETE_RAW_INPUT',
                session_id='pick_place_1008_001', source_dir=str(raw_root / '001'),
                source_refs={'raw/vst.h264': reference(raw)}, frame_count=3,
                fps=30, width=640, height=480, video=reference(video), arrays=reference(arrays),
                calibration=dict(hand_joint_schema=source.JOINT_MAPPING_SCHEMA,
                                 manus25_to21=source.MAP_25_TO_21.tolist()))
    meta.update(getattr(request, 'param', {}))
    input_json = prepared / 'source/SESSION_INPUT.json'
    save(input_json, meta)
    sidecars = {}
    for role in ('selection', 'workflow'):
        path = raw_root / '001/raw' / ('capture.' + role + '.json')
        save(path, {'role': role, 'collection_fixture': True})
        sidecars[role] = path
    stage_raw = {'raw_vst': reference(raw),
                 **{role: reference(path) for role, path in sidecars.items()}}
    stage_paths, stage_receipts, stages = {}, {}, {}
    for stage in io.ORIGINAL_STAGES:
        folder = prepared if stage == 'source' else full
        path = folder / ('STAGE_' + stage + '.json')
        stage_binding = dict(stage=stage, session_id='001', raw_frame_count=3,
                             raw_source={key: dict(value) for key, value in stage_raw.items()})
        stage_value = dict(schema=io.ORIGINAL_STAGE_SCHEMA, status='COMPLETE',
                           binding=stage_binding, binding_sha256=hashlib.sha256(
                               json.dumps(stage_binding, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest())
        save(path, stage_value)
        stage_paths[stage], stage_receipts[stage], stages[stage] = path, reference(path), stage_value
    original_path = nav / 'MANIFEST_ALL199_ORIGINAL_EXECUTION_OUTPUTS_V1.json'
    original = dict(session_count=1, original_native_frame_count=3, sessions=[dict(
        session_id='001', original_native_frames=3, original_full_session=str(full),
        source_session=str(prepared), stage_receipts=stage_receipts,
        source_files={'source/SESSION_INPUT.json': reference(input_json)})])
    save(original_path, original)
    original_ref = reference(original_path)
    monkeypatch.setattr(io, 'SOURCE_MANIFEST_SHA', original_ref['sha256'])
    rows = []
    payload_rows = [
        (input_json, 'sessions/001/source/SESSION_INPUT.json', 'session_payload'),
        (arrays, 'sessions/001/source/source.npz', 'session_payload'),
        (video, 'sessions/001/source/physical_eye.mp4', 'session_payload'),
        (actions, 'sessions/001/actions.npz', 'session_payload'),
        (motion, 'sessions/001/motion.json', 'session_payload'),
        *[(path, 'sessions/001/' + path.name, 'session_payload')
          for path in stage_paths.values()],
        (original_path, 'provenance/original_unified_delivery/' + original_path.name,
         'original_navigation_metadata'),
    ]
    for p, relative, kind in payload_rows:
        item = dict(source_path=str(p), destination_relative=relative, bytes=p.stat().st_size,
                    sha256=digest(p), source_metadata={'frozen': True}, kind=kind)
        if kind == 'session_payload':
            item['session_id'] = '001'
        rows.append(item)
        copied = dest / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(p.read_bytes())
    payload_bytes = sum(row['bytes'] for row in rows if row['kind'] == 'session_payload')
    monkeypatch.setattr(io, 'EXPECTED_PAYLOAD_FILES', 9)
    monkeypatch.setattr(io, 'EXPECTED_PAYLOAD_BYTES', payload_bytes)
    inventory = dict(schema='exact-original199-export-inventory-v1', destination=str(dest),
                     parent_manifest=original_ref, files=rows, payload_files=9,
                     payload_bytes=payload_bytes, metadata_files=1, deletion_allowed=False)
    inventory_path = base / 'E0/INVENTORY.json'
    save(inventory_path, inventory)
    monkeypatch.setattr(io, 'SOURCE_INVENTORY_SHA', digest(inventory_path))
    exported = dict(schema='pick-place-original199-relocatable-export-v1', status='EXPORTED_BYTES_VERIFIED',
                    parent_manifest=original_ref, files=[dict(row, source_SHA_verified=True,
                       destination_readback_SHA_verified=True) for row in rows], sessions=1,
                    original_native_frames=3, payload_files=9, payload_bytes=payload_bytes,
                    metadata_files=1, contains_external_symlinks=False, training_eligible=False,
                    control_ground_truth=False, all199_visual_accepted=False)
    export_path = dest / 'EXPORT_MANIFEST_V1.json'
    save(export_path, exported)
    complete = dict(status='PASSED_EXPORT', sessions=1, original_native_frames=3,
                    verified_files=10, verified_bytes=sum(row['bytes'] for row in rows),
                    export_manifest=reference(export_path), source_and_NAS_full_SHA_verified=True,
                    visual_quality_pass=False, training_eligible=False)
    complete_path = dest / 'EXPORT_COMPLETE_V1.json'
    save(complete_path, complete)
    binding = dict(schema=io.BINDING_SCHEMA, dataset_root=str(dest),
                   export_manifest=reference(export_path), export_complete=reference(complete_path),
                   source_inventory=reference(inventory_path), source_manifest=original_ref)
    binding_path = base / 'BINDING.json'
    save(binding_path, binding)
    def reseal():
        save(export_path, exported)
        complete['export_manifest'] = reference(export_path)
        save(complete_path, complete)
        binding['export_manifest'] = reference(export_path)
        binding['export_complete'] = reference(complete_path)
        save(binding_path, binding)
        return reference(binding_path)
    def reseal_original_controls(stage_ref_edits=None):
        """Seal a deliberately bad tiny producer fixture, not real evidence."""
        for stage, path in stage_paths.items():
            save(path, stages[stage])
            original['sessions'][0]['stage_receipts'][stage] = reference(path)
        if stage_ref_edits:
            stage_ref_edits(original['sessions'][0]['stage_receipts'])
        original['sessions'][0]['source_files']['source/SESSION_INPUT.json'] = reference(input_json)
        save(original_path, original)
        updated_original = reference(original_path)
        monkeypatch.setattr(io, 'SOURCE_MANIFEST_SHA', updated_original['sha256'])
        for row in rows:
            path = Path(row['source_path'])
            row.update(bytes=path.stat().st_size, sha256=digest(path))
            (dest / row['destination_relative']).write_bytes(path.read_bytes())
        updated_bytes = sum(row['bytes'] for row in rows if row['kind'] == 'session_payload')
        monkeypatch.setattr(io, 'EXPECTED_PAYLOAD_BYTES', updated_bytes)
        inventory.update(parent_manifest=updated_original, payload_bytes=updated_bytes)
        save(inventory_path, inventory)
        monkeypatch.setattr(io, 'SOURCE_INVENTORY_SHA', digest(inventory_path))
        exported.update(parent_manifest=updated_original, payload_bytes=updated_bytes,
                        files=[dict(row, source_SHA_verified=True,
                                    destination_readback_SHA_verified=True) for row in rows])
        complete['verified_bytes'] = sum(row['bytes'] for row in rows)
        binding.update(source_manifest=updated_original, source_inventory=reference(inventory_path))
        return reseal()
    return dict(root=root, base=base, nav=nav, dest=dest, raw=raw, full=full,
                prepared=prepared, arrays=arrays, video=video, actions=actions, motion=motion,
                input_json=input_json, binding_path=binding_path, binding=binding,
                exported=exported, complete=complete, inventory=inventory,
                binding_ref=reference(binding_path), reseal=reseal, sidecars=sidecars,
                stages=stages, stage_paths=stage_paths, original=original,
                reseal_original_controls=reseal_original_controls)


def test_same_refs_and_verified_physical_path(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['actions'])
    assert reader.verify_ref(ref) == ref
    assert reader.physical_path(ref) == fixture['dest'] / 'sessions/001/actions.npz'
    assert reader.reference_for(fixture['actions']) == ref
    assert np.array_equal(reader.arrays(ref)['q'], np.arange(6).reshape(3, 2))
    assert reader.physical_path(reference(fixture['motion'])) == fixture['motion']


@pytest.mark.parametrize('fixture', [
    {'session_id': '001'},
    {'session_id': 'pick_place_1008_002'},
    {'source_dir': '/path/to/private_storage/001'},
], indirect=True)
def test_prepared_short_mismatched_id_or_source_dir_rejected(fixture):
    with pytest.raises(io.StorageRelocationError, match='PREPARED_INPUT_SESSION_IDENTITY'):
        io.Original199Reader(fixture['binding_ref'])


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('bytes', 1), ('bytes', True)])
def test_original_ref_drift_is_not_hidden_by_good_nas(fixture, field, value):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['actions'])
    ref[field] = value
    with pytest.raises(io.StorageRelocationError):
        reader.verify_ref(ref)


def test_unknown_external_file_not_allowed(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    unrelated = fixture['dest'].parent / 'other.npz'
    np.savez(unrelated, x=np.array([1]))
    with pytest.raises(io.StorageRelocationError, match='UNREGISTERED_EXTERNAL'):
        reader.arrays(reference(unrelated))


def test_raw_only_exact_original_source_ref(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['raw'])
    assert reader.verify_ref(ref) == ref
    unrelated = fixture['raw'].parent / 'unregistered.h264'
    unrelated.write_bytes(fixture['raw'].read_bytes())
    with pytest.raises(io.StorageRelocationError):
        reader.verify_ref(reference(unrelated))


@pytest.mark.parametrize('mutation', ['missing', 'status', 'count', 'mapped_session', 'mapped_sha', 'duplicate'])
def test_unsealed_or_incomplete_mapping_fails(fixture, mutation):
    if mutation == 'missing':
        (fixture['dest'] / 'EXPORT_COMPLETE_V1.json').unlink()
        ref = fixture['binding_ref']
    else:
        if mutation == 'status':
            fixture['complete']['status'] = 'INCOMPLETE'
        elif mutation == 'count':
            fixture['complete']['verified_files'] -= 1
        elif mutation == 'mapped_session':
            fixture['exported']['files'][0]['session_id'] = '002'
        elif mutation == 'mapped_sha':
            fixture['exported']['files'][1]['sha256'] = '0' * 64
        else:
            fixture['exported']['files'][-1] = fixture['exported']['files'][0].copy()
        ref = fixture['reseal']()
    with pytest.raises((io.StorageRelocationError, OSError)):
        io.Original199Reader(ref)


def test_mapped_data_sha_drift_stops_even_after_cache(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['actions'])
    assert reader.verify_ref(ref) == ref
    path = fixture['dest'] / 'sessions/001/actions.npz'
    path.write_bytes(b'not the old data')
    with pytest.raises(io.StorageRelocationError, match='SHA_OR_BYTES'):
        reader.verify_ref(ref)


def test_manifest_identity_change_stops_existing_reader(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    fixture['binding_path'].write_bytes(fixture['binding_path'].read_bytes() + b' ')
    with pytest.raises(io.StorageRelocationError, match='IDENTITY_DRIFT'):
        reader.verify_ref(reference(fixture['actions']))


@pytest.mark.parametrize('ancestor', [False, True])
def test_symlink_data_or_parent_fails(fixture, ancestor):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['actions'])
    target = fixture['dest'] / 'sessions/001/actions.npz'
    if ancestor:
        folder = target.parent
        retained = folder.with_name('001_original')
        folder.rename(retained)
        folder.symlink_to(retained, target_is_directory=True)
    else:
        target.unlink()
        target.symlink_to(fixture['actions'])
    with pytest.raises(io.StorageRelocationError, match='SYMLINK'):
        reader.verify_ref(ref)


def test_traversal_reference_and_external_binding_rejected(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    ref = reference(fixture['actions'])
    ref['path'] = str(fixture['full'] / '..' / '001/actions.npz')
    with pytest.raises(io.StorageRelocationError):
        reader.verify_ref(ref)
    other = fixture['root'] / 'wrong_binding.json'
    other.write_bytes(fixture['binding_path'].read_bytes())
    with pytest.raises(io.StorageRelocationError, match='SCOPE'):
        io.Original199Reader(reference(other))


def test_retained_metadata_reference_is_logical(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    evidence = fixture['root'] / 'tasks/receipt.json'
    save(evidence, dict(status='FAILED', hidden='UNKNOWN'))
    ref = reader.reference_for(evidence)
    assert ref['path'] == str(evidence)
    assert reader.read_json(ref) == dict(status='FAILED', hidden='UNKNOWN')
    assert reader.has_ref(evidence)
    assert not reader.has_ref(evidence.with_name('absent.json'))


def test_prepared_parity_and_post_removal_without_symlink(fixture):
    from chaoyang.pick_place_1008 import source
    old = source.load_prepared(fixture['prepared'] / 'source')
    reader = io.Original199Reader(fixture['binding_ref'])
    fixture['arrays'].unlink()
    fixture['video'].unlink()
    current = reader.load_prepared(fixture['prepared'] / 'source')
    assert current.raw_dir == old.raw_dir
    assert current.video == fixture['dest'] / 'sessions/001/source/physical_eye.mp4'
    for field in ('K', 'c2w', 'hands', 'hand_observed', 'hand_source', 'body', 'body_observed', 'gravity', 'qpc_ns'):
        first, second = getattr(current, field), getattr(old, field)
        assert np.array_equal(first, second, equal_nan=first.dtype.kind in 'fc')
    assert current.calibration == old.calibration
    assert current.mapping.keys() == old.mapping.keys()
    assert not fixture['arrays'].exists() and not fixture['arrays'].is_symlink()


def test_prepared_unknown_folder_rejected(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    with pytest.raises(io.StorageRelocationError, match='UNREGISTERED_PREPARED'):
        reader.load_prepared(fixture['full'])


def test_wrong_original_session_path_is_not_project_fallback(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    wrong = fixture['full'].with_name('002') / 'actions.npz'
    wrong.parent.mkdir()
    wrong.write_bytes(fixture['actions'].read_bytes())
    with pytest.raises(io.StorageRelocationError, match='UNBOUND_ORIGINAL_BINARY'):
        reader.verify_ref(reference(wrong))


def test_upstream_logical_refs_and_no_write_api(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    refs = {role: reference(fixture['actions']) for role in
            ('actions', 'frame_mapping', 'raw_rgb', 'clean_rgb', 'masks', 'motion_info')}
    assert reader.upstream_refs({'upstream_refs': refs}) == refs
    with pytest.raises(io.StorageRelocationError):
        reader.upstream_refs({'upstream_refs': {'actions': refs['actions']}})
    for name in ('write', 'save', 'delete', 'run', 'submit', 'output_path', 'relocate_output', 'open'):
        assert not hasattr(reader, name)


def test_stage_sidecars_are_exact_original_refs_not_directory_permission(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    for role, path in fixture['sidecars'].items():
        pin = reference(path)
        assert reader.verify_ref(pin) == pin
        assert reader.read_json(pin) == {'role': role, 'collection_fixture': True}
    unrelated = fixture['raw'].parent / 'capture.unregistered.json'
    save(unrelated, {'not': 'an original input'})
    with pytest.raises(io.StorageRelocationError, match='UNREGISTERED_EXTERNAL_READ'):
        reader.verify_ref(reference(unrelated))
    # Only the binding/export/inventory/original manifest are per-read controls.
    assert len(reader._controls) == 5


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('bytes', 1)])
def test_stage_sidecar_wrong_sha_or_bytes_rejected(fixture, field, value):
    reader = io.Original199Reader(fixture['binding_ref'])
    pin = reference(fixture['sidecars']['selection'])
    pin[field] = value
    with pytest.raises(io.StorageRelocationError, match='UNREGISTERED_EXTERNAL_READ'):
        reader.verify_ref(pin)


@pytest.mark.parametrize('case,reason', [
    ('schema', 'ORIGINAL_RAW_STAGE_SCHEMA_OR_STATUS'),
    ('partial', 'ORIGINAL_RAW_STAGE_SCHEMA_OR_STATUS'),
    ('stage', 'ORIGINAL_RAW_STAGE_IDENTITY'),
    ('sid', 'ORIGINAL_RAW_STAGE_IDENTITY'),
    ('frames', 'ORIGINAL_RAW_STAGE_IDENTITY'),
    ('boolean_frames', 'ORIGINAL_RAW_STAGE_IDENTITY'),
    ('digest', 'ORIGINAL_RAW_STAGE_BINDING_DIGEST'),
    ('cross_session', 'LEXICAL_READ_SCOPE'),
    ('closure_conflict', 'ORIGINAL_RAW_FOUR_STAGE_CLOSURE_CONFLICT'),
    ('source_conflict', 'ORIGINAL_RAW_STAGE_SOURCE_REFERENCE_CONFLICT'),
    ('unregistered_extra', 'ORIGINAL_RAW_UNSUPPORTED_EXTRA_ROLES'),
    ('sidecar_wrong_filename', 'ORIGINAL_RAW_COLLECTION_SIDECAR_IDENTITY'),
    ('duplicate_raw_path', 'ORIGINAL_RAW_DUPLICATE_PATH'),
])
def test_frozen_stage_raw_closure_rejects_invalid_proof(fixture, case, reason):
    stage = fixture['stages']['motion']
    if case == 'schema': stage['schema'] = 'wrong'
    elif case == 'partial': stage['status'] = 'PARTIAL'
    elif case == 'stage': stage['binding']['stage'] = 'gpu'
    elif case == 'sid': stage['binding']['session_id'] = '002'
    elif case == 'frames': stage['binding']['raw_frame_count'] = 4
    elif case == 'boolean_frames': stage['binding']['raw_frame_count'] = True
    elif case == 'cross_session':
        stage['binding']['raw_source']['selection']['path'] = str(
            fixture['sidecars']['selection'].parent.parent.with_name('002') /
            'raw/capture.selection.json')
    elif case == 'closure_conflict':
        stage['binding']['raw_source']['selection']['sha256'] = '0' * 64
    elif case == 'source_conflict':
        for value in fixture['stages'].values():
            value['binding']['raw_source']['raw_vst']['sha256'] = '0' * 64
    elif case == 'unregistered_extra':
        extra = fixture['raw'].parent / 'capture.extra.json'
        save(extra, {'not': 'a selected original collection sidecar'})
        for value in fixture['stages'].values():
            value['binding']['raw_source']['extra'] = reference(extra)
    elif case == 'sidecar_wrong_filename':
        wrong = fixture['raw'].parent / 'capture.other.json'
        save(wrong, {'not': 'selection'})
        for value in fixture['stages'].values():
            value['binding']['raw_source']['selection'] = reference(wrong)
    elif case == 'duplicate_raw_path':
        for value in fixture['stages'].values():
            value['binding']['raw_source']['workflow'] = dict(
                value['binding']['raw_source']['selection'])
    for value in fixture['stages'].values():
        value['binding_sha256'] = hashlib.sha256(json.dumps(
            value['binding'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if case == 'digest':
        stage['binding_sha256'] = '0' * 64
    pin = fixture['reseal_original_controls']()
    with pytest.raises(io.StorageRelocationError, match=reason):
        io.Original199Reader(pin)


@pytest.mark.parametrize('case,reason', [
    ('missing', 'ORIGINAL_FOUR_STAGE_REFERENCES_REQUIRED'),
    ('sha', 'ORIGINAL_STAGE_NOT_EXACT_FROZEN_MAP'),
    ('bytes', 'ORIGINAL_STAGE_NOT_EXACT_FROZEN_MAP'),
    ('path', 'ORIGINAL_STAGE_NOT_EXACT_FROZEN_MAP'),
])
def test_original_stage_pin_must_match_frozen_e0_map(fixture, case, reason):
    def edit(pins):
        if case == 'missing': pins.pop('render')
        elif case == 'sha': pins['render']['sha256'] = '0' * 64
        elif case == 'bytes': pins['render']['bytes'] += 1
        else: pins['render']['path'] = str(fixture['full'].with_name('002') / 'STAGE_render.json')
    pin = fixture['reseal_original_controls'](edit)
    with pytest.raises(io.StorageRelocationError, match=reason):
        io.Original199Reader(pin)


def test_unresealed_stage_byte_tamper_rejected(fixture):
    stage = fixture['stage_paths']['gpu']
    stage.write_bytes(stage.read_bytes() + b' ')
    with pytest.raises(io.StorageRelocationError, match='JSON_SHA_OR_BYTES_DRIFT'):
        io.Original199Reader(fixture['binding_ref'])


def test_later_mutable_stage_json_cannot_expand_snapshot_allowlist(fixture):
    reader = io.Original199Reader(fixture['binding_ref'])
    extra = fixture['raw'].parent / 'capture.later.json'
    save(extra, {'not': 'an immutable original reference'})
    value = fixture['stages']['gpu']
    value['binding']['raw_source']['later'] = reference(extra)
    value['binding_sha256'] = hashlib.sha256(json.dumps(
        value['binding'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(fixture['stage_paths']['gpu'], value)
    with pytest.raises(io.StorageRelocationError, match='UNREGISTERED_EXTERNAL_READ'):
        reader.verify_ref(reference(extra))
    original_pin = fixture['original']['sessions'][0]['stage_receipts']['gpu']
    with pytest.raises(io.StorageRelocationError, match='JSON_SHA_OR_BYTES_DRIFT'):
        reader.read_json(original_pin)
