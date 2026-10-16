"""Portable proof-negative tests; never execute production C0 or C1."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/maintenance/project_cleanup_proof_v2.py'
SPEC = importlib.util.spec_from_file_location('project_cleanup_proof_v2_tests', MODULE)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / 'project'; root.mkdir()
    roots = ('tests/fixture',)
    candidate = write(root / roots[0] / 'generated.json', {'generated': True})
    source = write(root / 'docs/source.json', {'nonconsumer': True})
    return root, roots, candidate, source


def seed_for(root, roots, producer=None):
    return cleanup.seed_sources(root, roots, producer)


def unit_for(path, root, roots, names):
    return cleanup.source_unit(cleanup.source_record(path, root), names, root, roots)


def graph_for(root, roots, names):
    seed = seed_for(root, roots)
    pending = list(seed['sources'].values()); units = []; visited = set()
    while pending:
        row = pending.pop()
        if row['path'] in visited: continue
        visited.add(row['path'])
        unit = cleanup.source_unit(row, names, root, roots)
        units.append(unit); pending += list(unit['discovered'].values())
    graph = cleanup.assemble(seed, [dict(units=units)])
    graph['seed_paths'] = sorted(seed['sources'])
    return seed, graph


def test_missing_partition_is_not_closed(sample):
    root, roots, candidate, source = sample
    other = write(root / 'docs/other.json', {'exists': True})
    seed = seed_for(root, roots)
    unit = unit_for(source, root, roots, [str(candidate.relative_to(root))])
    graph = cleanup.assemble(seed, [dict(units=[unit])])
    assert not graph['closed']
    assert str(other.relative_to(root)) in graph['pending']


def test_recursive_closure_requires_its_own_partition(sample):
    root, roots, candidate, source = sample
    dependency = write(root / '_run/frozen/evidence.json', {'value': 1})
    write(source, {'input': '_run/frozen/evidence.json'})
    unit = unit_for(source, root, roots, [str(candidate.relative_to(root))])
    seed = seed_for(root, roots)
    incomplete = cleanup.assemble(seed, [dict(units=[unit])])
    assert '_run/frozen/evidence.json' in incomplete['pending']
    seed_units = [cleanup.source_unit(row, [], root, roots) for label, row in seed['sources'].items()
                  if label != unit['path']]
    complete = cleanup.assemble(seed, [dict(units=[unit, *seed_units]), dict(units=[unit_for(dependency, root, roots, [])])])
    assert complete['closed']


def test_duplicate_partition_rejected(sample):
    root, roots, _, source = sample
    unit = unit_for(source, root, roots, [])
    with pytest.raises(cleanup.ProofError, match='DUPLICATE_REFERENCE_UNIT'):
        cleanup.assemble(seed_for(root, roots), [dict(units=[unit]), dict(units=[unit])])


def test_unreachable_invented_partition_rejected(sample):
    root, roots, _, _ = sample
    unrelated = write(root / '_run/unreferenced.json', {'hidden': True})
    with pytest.raises(cleanup.ProofError, match='UNREACHABLE_REFERENCE_UNIT'):
        cleanup.assemble(seed_for(root, roots), [dict(units=[unit_for(unrelated, root, roots, [])])])


def test_partition_count_and_bytes_are_bounded():
    rows = [dict(path=str(i), metadata=dict(size=4)) for i in range(10)]
    assert len(cleanup.partition(rows, max_bytes=8, max_files=3)) == 2
    assert len(cleanup.partition(rows, max_bytes=100, max_files=3)) == 3
    with pytest.raises(cleanup.ProofError, match='SINGLE_SOURCE_EXCEEDS_PARTITION'):
        cleanup.partition(rows, max_bytes=3)


def test_current_reverse_reference_is_recorded(sample):
    root, roots, candidate, source = sample
    name = str(candidate.relative_to(root)); write(source, {'input': name})
    unit = unit_for(source, root, roots, [name])
    graph = cleanup.assemble(seed_for(root, roots), [dict(units=[unit])])
    assert cleanup.related_references(graph, [name], root=root)[name] == ['docs/source.json']


def test_new_reference_after_proof_is_rejected(sample):
    root, roots, candidate, source = sample
    source = write(root / 'docs/current/STATUS.json', {'heartbeat': 1})
    name = str(candidate.relative_to(root)); seed, graph = graph_for(root, roots, [name])
    before = cleanup.related_references(graph, [name], root=root)
    write(source, {'input': name})
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    with pytest.raises(cleanup.ProofError, match='RELEVANT_REFERENCE_SET_CHANGED'):
        cleanup.assert_references(cleanup.related_references(fresh, [name], root=root), before)


def test_new_source_with_reference_after_proof_rejected(sample):
    root, roots, candidate, _ = sample
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    old = cleanup.related_references(graph, [name], root=root)
    write(root / 'docs/new.json', {'input': name})
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    with pytest.raises(cleanup.ProofError, match='RELEVANT_REFERENCE_SET_CHANGED'):
        cleanup.assert_references(cleanup.related_references(fresh, [name], root=root), old)


def test_source_code_drift_rejected(sample):
    root, roots, candidate, _ = sample
    source = write(root / 'src/consumer.json', {'version': 1})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    write(source, {'version': 2})
    with pytest.raises(cleanup.ProofError, match='IMMUTABLE_SOURCE_DRIFT'):
        cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)


def test_metadata_cache_uses_ctime_not_just_mtime(sample):
    root, roots, candidate, source = sample
    source = write(root / 'docs/current/STATUS.json', {'heartbeat': 1})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    stamp = source.stat()
    write(source, {'input': name})
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    assert fresh['changed_files'] == 1
    assert cleanup.related_references(fresh, [name], root=root)[name]


def test_new_alias_preserves_target_and_rejects_old_snapshot(sample):
    root, roots, candidate, _ = sample
    _, before = cleanup.fixture_inventory(root, roots)
    (candidate.parent / 'pytest-current').symlink_to(candidate.parent, target_is_directory=True)
    _, after = cleanup.fixture_inventory(root, roots)
    assert cleanup.alias_blocks(candidate, after)
    with pytest.raises(cleanup.ProofError, match='ALIAS_SET_DRIFT'):
        cleanup.assert_aliases(after, before)


def test_recursive_alias_target_drift_rejected(sample):
    root, roots, candidate, source = sample
    first = write(root / '_run/first.json', {'v': 1})
    second = write(root / '_run/second.json', {'v': 2})
    link = root / '_run/link.json'; link.symlink_to(first)
    write(source, {'input': '_run/link.json'})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    link.unlink(); link.symlink_to(second)
    with pytest.raises(cleanup.ProofError, match='REFERENCE_ALIAS_DRIFT'):
        cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)


def test_previously_absent_closure_source_is_rechecked(sample):
    root, roots, candidate, source = sample
    name = str(candidate.relative_to(root))
    write(source, {'future': '_run/later.json'})
    _, graph = graph_for(root, roots, [name]); before = cleanup.related_references(graph, [name], root=root)
    write(root / '_run/later.json', {'input': name})
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    assert fresh['closed'] and fresh['changed_files'] == 1
    with pytest.raises(cleanup.ProofError, match='RELEVANT_REFERENCE_SET_CHANGED'):
        cleanup.assert_references(cleanup.related_references(fresh, [name], root=root), before)


def test_active_partial_is_not_read_as_closure_source(sample):
    root, roots, _, source = sample
    live = write(root / '_run/live/result.json', {'looks': 'PASSED'})
    write(source, {'input': '_run/live/result.json'})
    unit = cleanup.source_unit(cleanup.source_record(source, root), [], root, roots,
                               active_outputs=[str(live.parent)])
    assert unit['errors'] and 'ACTIVE_PARTIAL_CLOSURE_UNREAD' in unit['errors'][0]
    graph = cleanup.assemble(seed_for(root, roots), [dict(units=[unit])])
    assert not graph['closed']


@pytest.mark.parametrize('value', [dict(hits={'target': [{'pid': 1}]}, errors=[]),
                                  dict(hits={}, errors=[{'pid': 1}])])
def test_proc_holder_or_unreadable_proc_rejected(value):
    with pytest.raises(cleanup.ProofError, match='PROCESS_PROOF_CHANGED'):
        cleanup.assert_no_process_holders(value)


@pytest.mark.parametrize('name,code', [('video.mp4', 'PROTECTED_MEDIA_FILE'),
                                      ('audio.flac', 'PROTECTED_MEDIA_FILE'),
                                      ('plot.svg', 'PROTECTED_MEDIA_FILE'),
                                      ('unique.py', 'UNIQUE_SOURCE_NOT_DISPOSABLE'),
                                      ('LICENSE.txt', 'PROTECTED_LICENSE_OR_BRANCH')])
def test_media_licenses_unique_source_never_eligible(sample, name, code):
    root, roots, candidate, _ = sample
    path = write(candidate.parent / name, {'fixture': True})
    row = cleanup.source_record(path, root)
    assert code in cleanup.blockers(row, [], {}, dict(hits={}, errors=[]), set(), root)


def test_hardlink_not_unique_is_blocked(sample):
    root, _, candidate, _ = sample
    os.link(candidate, candidate.parent / 'other.json')
    row = cleanup.source_record(candidate, root)
    assert 'HARDLINK_NOT_UNIQUE' in cleanup.blockers(row, [], {}, dict(hits={}, errors=[]), set(), root)


def test_exact_producer_exclusion_does_not_exclude_other_maintenance(sample):
    root, roots, _, _ = sample
    producer = write(root / 'scripts/maintenance/producer.py', {'producer': True})
    other = write(root / 'scripts/maintenance/consumer.py', {'consumer': True})
    seed = seed_for(root, roots, producer)
    assert str(producer.relative_to(root)) not in seed['sources']
    assert str(other.relative_to(root)) in seed['sources']


def test_recovery_member_size_and_sha(sample):
    root, _, candidate, _ = sample
    data, metadata = cleanup.stable_bytes(candidate)
    row = dict(path=str(candidate.relative_to(root)), metadata=metadata, sha256=cleanup.sha(data))
    archive = root / 'recovery.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo(row['path']); member.size = len(data)
        stream.addfile(member, io.BytesIO(data))
    result = cleanup.archive_verify(archive, [row])
    assert result['members'] == 1 and result['uncompressed_bytes'] == len(data)


def test_recovery_wrong_bytes_and_unlisted_member_rejected(sample):
    root, _, candidate, _ = sample
    data, metadata = cleanup.stable_bytes(candidate)
    row = dict(path=str(candidate.relative_to(root)), metadata=metadata, sha256=cleanup.sha(data))
    archive = root / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        member = tarfile.TarInfo(row['path']); member.size = len(data)
        stream.addfile(member, io.BytesIO(b'x' * len(data)))
    with pytest.raises(cleanup.ProofError, match='RECOVERY_MEMBER_SHA_DRIFT'):
        cleanup.archive_verify(archive, [row])
    with pytest.raises(cleanup.ProofError, match='RECOVERY_MEMBER_SCOPE'):
        cleanup.archive_verify(archive, [])


def test_dynamic_record_actual_sha_is_preserved_and_rechecked(sample):
    root, roots, candidate, _ = sample
    dynamic = write(root / 'docs/current/STATUS.json', {'heartbeat': 1})
    old = cleanup.source_record(dynamic, root)
    write(dynamic, {'heartbeat': 2})
    unit = cleanup.source_unit(old, [str(candidate.relative_to(root))], root, roots)
    assert unit['sha256'] == cleanup.pin(dynamic)['sha256']
    assert unit['metadata'] != old['metadata']
    assert unit['time_semantics'] == 'MUTABLE_OBSERVATION_RECHECK_REQUIRED'


def test_invalid_long_multibyte_component_rejected():
    assert not cleanup.valid_path('tests/' + '\u6d4b' * 90)
    assert not cleanup.valid_path('tests/name\x00bad')


def test_changed_source_before_partition_is_rejected(sample):
    root, roots, _, source = sample
    row = cleanup.source_record(source, root)
    write(source, {'changed': True})
    with pytest.raises(cleanup.ProofError, match='REFERENCE_SOURCE_DRIFT'):
        cleanup.source_unit(row, [], root, roots)


@pytest.mark.parametrize('consumer', ['review/current/note.json', 'Makefile'])
def test_review_and_root_makefile_references_are_not_missed(sample, consumer):
    root, roots, candidate, _ = sample
    name = str(candidate.relative_to(root))
    path = root / consumer; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('check: ' + name, encoding='utf-8')
    seed, graph = graph_for(root, roots, [name])
    assert consumer in seed['sources']
    assert consumer in cleanup.related_references(graph, [name], root=root)[name]


def test_unclassified_source_alias_blocks_coverage(sample):
    root, roots, _, _ = sample
    outside = write(root.parent / 'outside.py', {'could_consume': True})
    alias = root / 'src/consumer.py'; alias.parent.mkdir(); alias.symlink_to(outside)
    seed = seed_for(root, roots)
    assert any('SEED_ALIAS_ROLE_UNCLASSIFIED' in error for error in seed['errors'])
    units = [cleanup.source_unit(row, [], root, roots) for row in seed['sources'].values()]
    assert not cleanup.assemble(seed, [dict(units=units)])['closed']


def test_root_unknown_file_role_is_not_silently_excluded(sample):
    root, roots, _, _ = sample
    (root / 'unknown.consumer').write_text('may reference a generated input', encoding='utf-8')
    assert any('ROOT_FILE_ROLE_UNCLASSIFIED' in error for error in seed_for(root, roots)['errors'])


def test_dynamic_record_new_reference_still_rejected(sample):
    root, roots, candidate, _ = sample
    dynamic = write(root / 'docs/current/STATUS.json', {'heartbeat': 1})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    before = cleanup.related_references(graph, [name], root=root)
    write(dynamic, {'heartbeat': 2, 'consumer': name})
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    with pytest.raises(cleanup.ProofError, match='RELEVANT_REFERENCE_SET_CHANGED'):
        cleanup.assert_references(cleanup.related_references(fresh, [name], root=root), before)


def test_nondynamic_document_drift_stops_even_without_new_reference(sample):
    root, roots, candidate, source = sample
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    write(source, {'different_static_text': True})
    with pytest.raises(cleanup.ProofError, match='STATIC_REFERENCE_SOURCE_DRIFT'):
        cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)


def test_second_target_gate_stops_new_ordinary_document_reference(sample):
    root, roots, first, _ = sample
    second = write(first.parent / 'second.json', {'second': True})
    names = [str(p.relative_to(root)) for p in (first, second)]
    seed, graph = graph_for(root, roots, names)
    expected = cleanup.related_references(graph, names, root=root)
    graph = cleanup.reference_gate(graph, seed, expected, names, root, roots)
    first_check_completed = True  # portable gate test; no real cleanup invoked
    write(root / 'docs/new_consumer.json', {'input': names[1]})
    with pytest.raises(cleanup.ProofError, match='RELEVANT_REFERENCE_SET_CHANGED'):
        cleanup.reference_gate(graph, seed_for(root, roots), expected, names, root, roots)
    assert second.exists(), 'the second target must not be unlinked'
    assert first_check_completed


def test_escaped_json_path_key_is_a_reference(sample):
    root, roots, candidate, source = sample
    name = str(candidate.relative_to(root))
    escaped = name.replace('/', '\\u002f')
    source.write_text('{"' + escaped + '":true}', encoding='utf-8')
    assert name not in source.read_text()
    unit = unit_for(source, root, roots, [name])
    assert name in unit['incoming']


def test_directory_reference_is_not_silently_called_closed(sample):
    root, roots, _, source = sample
    write(root / '_run/consumers/input.json', {'input': 'tests/fixture/generated.json'})
    write(source, {'directory_input': '_run/consumers'})
    unit = unit_for(source, root, roots, [])
    assert any('BLOCKED_DIRECTORY_REFERENCE_UNPROVED' in error for error in unit['errors'])
    assert not cleanup.assemble(seed_for(root, roots), [dict(units=[unit])])['closed']


def test_cached_source_entering_active_output_is_not_reused(sample):
    root, roots, candidate, source = sample
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    with pytest.raises(cleanup.ProofError, match='ACTIVE_PARTIAL_SOURCE_UNREAD'):
        cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots,
                              active_outputs=[str(source)])


def test_parent_directory_alias_retarget_is_rejected(sample):
    root, roots, candidate, source = sample
    first = write(root / '_run/first/input.json', {'v': 1})
    second = write(root / '_run/second/input.json', {'v': 2})
    alias = root / '_run/link_dir'; alias.symlink_to(first.parent, target_is_directory=True)
    write(source, {'input': '_run/link_dir/input.json'})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    alias.unlink(); alias.symlink_to(second.parent, target_is_directory=True)
    with pytest.raises(cleanup.ProofError, match='REFERENCE_EDGE_REALPATH_DRIFT'):
        cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)


@pytest.mark.parametrize('name', ['consumer.js', 'consumer.ts', 'consumer.c', 'consumer.bash'])
def test_other_text_source_consumers_are_scanned(sample, name):
    root, roots, candidate, _ = sample
    label = str(candidate.relative_to(root))
    path = root / 'scripts' / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('input = "' + label + '"', encoding='utf-8')
    seed, graph = graph_for(root, roots, [label])
    assert str(path.relative_to(root)) in seed['sources']
    assert cleanup.related_references(graph, [label], root=root)[label]


@pytest.mark.parametrize('name', ['weights.pt', 'weights.pth', 'weights.ckpt',
                                  'weights.safetensors', 'weights.onnx',
                                  'THIRD_PARTY_NOTICES.md', 'foo_LICENSE.txt'])
def test_all_models_and_embedded_license_names_preserved(sample, name):
    root, _, candidate, _ = sample
    path = write(candidate.parent / name, {'do_not_read_as_weights': True})
    reasons = cleanup.blockers(cleanup.source_record(path, root), [], {}, dict(hits={}, errors=[]), set(), root)
    assert any(reason in reasons for reason in ('PROTECTED_MODEL_OR_BINARY_FILE', 'PROTECTED_LICENSE_OR_BRANCH'))


def test_retained_fixture_python_can_consume_sibling_json(sample):
    root, roots, candidate, _ = sample
    consumer = candidate.parent / 'retained.py'
    consumer.write_text("from pathlib import Path\ndata = Path(__file__).parent / 'generated.json'\n", encoding='utf-8')
    name = str(candidate.relative_to(root)); seed, graph = graph_for(root, roots, [name])
    assert str(consumer.relative_to(root)) in seed['sources']
    assert str(consumer.relative_to(root)) in cleanup.related_references(graph, [name], root=root)[name]


def test_relative_sibling_reference_in_retained_json_is_seen(sample):
    root, roots, candidate, _ = sample
    consumer = write(candidate.parent / 'retained.json', {'input': './generated.json'})
    name = str(candidate.relative_to(root)); _, graph = graph_for(root, roots, [name])
    assert str(consumer.relative_to(root)) in cleanup.related_references(graph, [name], root=root)[name]
    fresh = cleanup.refresh_graph(graph, seed_for(root, roots), [name], root, roots)
    assert fresh['closed'] and fresh['changed_files'] == 0


def test_compound_frozen_source_suffix_still_scans_sibling(sample):
    root, roots, candidate, _ = sample
    consumer = candidate.parent / 'consumer.py.V7_FINAL'
    consumer.write_text("from pathlib import Path\npath = Path(__file__).parent / 'generated.json'\n", encoding='utf-8')
    name = str(candidate.relative_to(root)); seed, graph = graph_for(root, roots, [name])
    assert str(consumer.relative_to(root)) in seed['sources']
    assert str(consumer.relative_to(root)) in cleanup.related_references(graph, [name], root=root)[name]


def test_private_lock_readback_precedes_strict_stable_read(sample):
    root, _, _, _ = sample
    path = root / 'private_lock.json'
    writer = {'pid': 123, 'startticks': 456, 'fencing_token': 'fixture-token'}
    with path.open('w+') as stream:
        json.dump(writer, stream)
        cleanup.read_back_private_lock(stream, path, writer)
    assert json.loads(cleanup.stable_bytes(path)[0]) == writer


def test_private_lock_readback_rejects_wrong_fencing_token(sample):
    root, _, _, _ = sample
    path = root / 'private_lock.json'
    with path.open('w+') as stream:
        json.dump({'fencing_token': 'other'}, stream)
        with pytest.raises(cleanup.ProofError, match='FENCING_TOKEN_DRIFT'):
            cleanup.read_back_private_lock(stream, path, {'fencing_token': 'expected'})


def test_input_metadata_still_rejects_ctime_drift(sample, monkeypatch):
    _, _, candidate, _ = sample
    original = cleanup.metadata
    calls = 0
    def changed(path):
        nonlocal calls
        value = original(path)
        calls += 1
        if calls > 1:
            value['ctime_ns'] += 1
        return value
    monkeypatch.setattr(cleanup, 'metadata', changed)
    with pytest.raises(cleanup.ProofError, match='INPUT_CHANGED_DURING_READ'):
        cleanup.stable_bytes(candidate)
