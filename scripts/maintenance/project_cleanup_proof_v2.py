"""Finite, source-partitioned cleanup proofs; never publishes production state.

CLI proof steps are inventory -> reference (repeat) -> merge; delete is separate.
Each call obtains the existing isolated-maintenance lock and seals its own packet.
Only the sixteen frozen fixture roots are deletion candidates. This module is a
proof producer, not a consumer of those fixtures: its own *exact signed source*
is excluded from reverse-reference parsing, never an entire maintenance folder.
Unresolved coverage is a blocker, not permission to increase a scan budget.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import fnmatch
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tarfile
import time

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/project_convergence_20261009/cleanup_v2'
PROTOCOL = ROOT / '_run/maintenance/concurrent_cleanup_20261009/PROTOCOL.json'
LOCK = PROTOCOL.parent / 'CHANNEL_LOCK'
OLD_EXECUTOR = ROOT / '_run/maintenance/project_navigation_cleanup_20261009/CLEANUP_TEST_FIXTURES_V1.py'
FIXTURES = tuple('tests/pick_place_1008/' + name for name in (
    '_tmp_multi_frame_corrections_v7_001', '_tmp_implementation_v6',
    '_tmp_v2_integration', '_tmp_source_parallel_v6', '_tmp_pipeline_v2',
    '_tmp_override_binding_v4', '_tmp_projection_v4', '_tmp_v3_final',
    '_tmp_implementation_v5', '_tmp_multi_frame_corrections_v7_002',
    '_tmp_seed_binding_v5', '_tmp_integration_v2', '_tmp_v2_final',
    '_tmp_implementation_v4', '_tmp_root_check_v1', '_tmp_closure_v5'))
SCOPES = ('src', 'tests', 'scripts', 'docs', 'tasks', 'configs', 'contracts',
          'manifests', 'baseline_v1', 'review')
TEXT = {'.py', '.md', '.json', '.jsonl', '.toml', '.yaml', '.yml', '.sh',
        '.txt', '.csv', '.html', '.ini', '.cfg', '.rst', '.bash', '.c', '.h',
        '.cpp', '.hpp', '.cu', '.rs', '.js', '.ts', '.tsx', '.jsx', '.log',
        '.xml', '.urdf', '.xacro', '.ipynb', '.conf'}
TEXT |= {'.patch', '.diff', '.tsv', '.ps1', '.lua'}
ROOT_TEXT_NAMES = {'Makefile', 'Dockerfile', 'Containerfile', 'CMakeLists.txt',
                   '.gitignore', '.gitattributes', '.dockerignore', '.editorconfig',
                   '.env', 'LICENSE', 'COPYING', 'NOTICE'}
MODEL_BINARY = {'.pth', '.pt', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz',
                '.pkl', '.pickle', '.bin', '.zip', '.gz', '.xz', '.tar', '.7z',
                '.so', '.a', '.o', '.pyc', '.stl', '.obj', '.ply', '.glb', '.gltf',
                '.woff', '.woff2', '.ttf', '.ico', '.pdf'}
MEDIA = {'.mp4', '.mov', '.avi', '.mkv', '.h264', '.h265', '.hevc', '.webm',
         '.mpeg', '.mpg', '.mjpeg', '.png', '.jpg', '.jpeg', '.gif', '.webp',
         '.bmp', '.tif', '.tiff', '.exr', '.svg', '.wav', '.mp3', '.flac',
         '.ogg', '.aac', '.m4a'}
SOURCE = {'.py', '.sh', '.bash', '.c', '.h', '.cpp', '.hpp', '.cu', '.rs',
          '.js', '.ts', '.tsx', '.jsx', '.so', '.a', '.o'}
IMMUTABLE_PREFIXES = ('src/', 'scripts/', 'tests/', 'configs/', 'contracts/',
                      'manifests/', 'baseline_v1/')
ACTIVE = {'RUNNING', 'CLAIMED', 'WAIT_GPU_RESOURCE'}
DYNAMIC_RECORDS = {
    'docs/governance/LONG_HORIZON_TASK_STATE.json',
    'docs/governance/CURRENT_STATUS_RECEIPT.json',
    'docs/governance/DOC_AUTHORITY_MAP.json',
    'docs/governance/ALGORITHM_CONTRACT.json',
    'docs/governance/CURRENT_BASELINE_REGISTRY_V2.json',
    'docs/governance/CURRENT_AUTHORITY_INDEX.json',
    'docs/governance/CURRENT_FILE_LAYOUT.json',
    'docs/governance/CURRENT_REGRESSION_MANIFEST.json',
    'docs/governance/CURRENT_STAGE_BASELINES_ZH.md',
    'docs/governance/TASK_QUEUE.json',
    'docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json',
    'docs/governance/CURRENT_PROJECT_STATUS_MIN.json',
    'docs/governance/CURRENT_PROJECT_STATUS_ZH.md',
    'docs/governance/CURRENT_RC1_STATUS_MIN.json',
    'docs/governance/STATE_CHANGELOG.jsonl',
    'docs/current/STATUS.json', 'docs/current/CURRENT_PROJECT_STATUS_MIN.json',
    'docs/current/CURRENT_PROJECT_STATUS_ZH.md', 'tasks/current/INDEX.json',
}
MAX_SOURCE_BYTES = 32_000_000  # unchanged individual-source admission limit
SHARD_BYTES, SHARD_FILES = 64 * 1024 * 1024, 2000
MAX_DELETE_BYTES, MAX_DELETE_FILES = 100_000_000, 10000
EXPECTED_CANDIDATES, EXPECTED_ALIASES = 6878, 537
PATH_RE = re.compile(r'(?:/path/to/chaoyang/|(?:tests|src|scripts|docs|tasks|configs|contracts|manifests|baseline_v1|review|_run)/)[^\s\"\'<>`),;]*')


class ProofError(RuntimeError):
    pass


def fail(code):
    raise ProofError(code)


def now():
    return datetime.now(timezone(timedelta(hours=8)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def metadata(path):
    value = Path(path).lstat()
    return dict(device=value.st_dev, inode=value.st_ino, mode=value.st_mode,
                nlink=value.st_nlink, size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, allocated_bytes=value.st_blocks * 512)


def stable_bytes(path):
    path = Path(path).absolute()
    before = metadata(path)
    if not stat.S_ISREG(before['mode']) or path.resolve() != path:
        fail('INPUT_NONREGULAR_OR_ALIAS:' + str(path))
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before['device'], before['inode']):
            fail('INPUT_OPEN_RACE:' + str(path))
        data = stream.read()
    if metadata(path) != before or len(data) != before['size']:
        fail('INPUT_CHANGED_DURING_READ:' + str(path))
    return data, before


def pin(path):
    data, _ = stable_bytes(path)
    return dict(path=str(Path(path).absolute()), bytes=len(data), sha256=sha(data))


def verify(value):
    if pin(value['path']) != value:
        fail('SIGNED_INPUT_DRIFT:' + value['path'])


def load(path):
    return json.loads(stable_bytes(path)[0])


def save(path, value, base=BASE):
    path = Path(path).absolute()
    if not path.is_relative_to(base) or path.parent.resolve() != path.parent:
        fail('OUTPUT_SCOPE')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def beneath(path, parent):
    return str(path) == str(parent) or str(path).startswith(str(parent).rstrip('/') + '/')


def is_fixture(path, root=ROOT, roots=FIXTURES):
    path = Path(path).absolute()
    return any(beneath(path, root / name) for name in roots)


def walk(directory, check=lambda: None):
    for folder, dirs, files in os.walk(directory, followlinks=False):
        check()
        for name in list(dirs):
            path = Path(folder) / name
            if path.is_symlink():
                dirs.remove(name); yield path
        for name in files:
            yield Path(folder) / name


def fixture_inventory(root=ROOT, roots=FIXTURES, check=lambda: None):
    files, aliases = [], []
    for name in roots:
        directory = root / name
        if directory.resolve() != directory or not directory.is_dir():
            fail('FIXTURE_ROOT_MISSING_OR_ALIAS:' + name)
        for path in walk(directory, check):
            value = metadata(path)
            label = path.relative_to(root).as_posix()
            if stat.S_ISLNK(value['mode']):
                aliases.append(dict(path=label, target=os.readlink(path),
                                    resolved=str(path.resolve()), metadata=value,
                                    exists=path.exists()))
            elif stat.S_ISREG(value['mode']):
                files.append(dict(path=label, metadata=value))
            else:
                fail('UNCLASSIFIED_FIXTURE_ENTRY:' + label)
    if len(files) > MAX_DELETE_FILES:
        fail('CANDIDATE_FILE_BUDGET')
    if any(not a['exists'] or not Path(a['resolved']).is_relative_to(root) for a in aliases):
        fail('ALIAS_TARGET_UNVERIFIED')
    return sorted(files, key=lambda r: r['path']), sorted(aliases, key=lambda r: r['path'])


def alias_blocks(path, aliases):
    return any(beneath(path, row['resolved']) for row in aliases)


def candidate_lookup(names, root=ROOT):
    result = defaultdict(set)
    for name in names:
        path = root / name
        for parent in (path, *path.parents):
            if parent == root or not parent.is_relative_to(root):
                break
            result[str(parent)].add(name)
    return result


def valid_path(value):
    return isinstance(value, str) and bool(value) and len(value) <= 4096 and not any(
        c in value for c in '\0\n\r') and all(len(p.encode('utf-8')) < 256 for p in value.split('/'))


def string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value: yield from string_values(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from string_values(key)
            yield from string_values(item)


def source_values(text, path, root=ROOT):
    values = list(PATH_RE.findall(text))
    suffixes = set(path.suffixes)
    if suffixes.intersection({'.json', '.ipynb'}):
        try: values += list(string_values(json.loads(text)))
        except json.JSONDecodeError: pass  # raw path extraction still applies
    elif '.jsonl' in suffixes:
        for line in text.splitlines():
            try: values += list(string_values(json.loads(line)))
            except json.JSONDecodeError: pass
    elif '.py' in suffixes:
        try:
            values += [node.value for node in ast.walk(ast.parse(text))
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        except SyntaxError:
            values += re.findall(r'''["']([^"'\n]+)["']''', text)
    elif suffixes.intersection(TEXT):
        values += re.findall(r'''["']([^"'\n]+)["']''', text)
    if path.suffix == '.md':
        for value in re.findall(r'\]\(([^)]+)\)', text):
            if value.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            value = value.strip('<>')
            if valid_path(value):
                target = Path(value) if value.startswith('/') else path.parent / value
                values.append(os.path.abspath(str(target).split('#', 1)[0]))
    # A generator may assemble a path from a base and a literal fixture name.
    # Its exact fixture-root literal is a reference, not silently missed.
    for name in FIXTURES:
        if Path(name).name in text:
            values.append(str(root / name))
    return values


def source_record(path, root=ROOT):
    return dict(path=Path(path).relative_to(root).as_posix(), metadata=metadata(path))


def text_path(path):
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return (not suffixes.intersection(MODEL_BINARY | MEDIA) and bool(suffixes.intersection(TEXT))) or path.name in ROOT_TEXT_NAMES


def seed_sources(root=ROOT, roots=FIXTURES, producer=None, extras=(), check=lambda: None):
    sources, aliases, errors = {}, {}, []
    todo = []
    for scope in SCOPES:
        todo.extend(walk(root / scope, check))
    # Root consumers have no common extension (Makefile, Dockerfile, etc.).
    # Unknown regular-file roles block coverage instead of silently disappearing.
    for path in root.iterdir():
        check()
        if path.is_symlink():
            todo.append(path)
        elif path.is_file():
            if text_path(path) or path.stat().st_size == 0:
                todo.append(path)
            elif path.suffix.lower() not in MEDIA | MODEL_BINARY:
                errors.append('ROOT_FILE_ROLE_UNCLASSIFIED:' + path.name)
    todo += [Path(path) for path in extras]
    for path in todo:
        check()
        path = path.absolute()
        if '__pycache__' in path.parts or path == producer:
            continue
        if not path.is_relative_to(root):
            errors.append('SEED_OUTSIDE_PROJECT:' + str(path)); continue
        if not path.exists() and not path.is_symlink():
            if path in [Path(item) for item in extras]:
                errors.append('REQUIRED_SEED_ABSENT:' + str(path))
            continue
        if path.is_symlink():
            row = dict(path=path.relative_to(root).as_posix(), target=os.readlink(path),
                       resolved=str(path.resolve()), metadata=metadata(path), exists=path.exists())
            aliases[row['path']] = row
            target = path.resolve()
            if not row['exists']:
                errors.append('SEED_ALIAS_UNRESOLVED:' + row['path'])
            elif is_fixture(target, root, roots):
                pass  # target is explicitly protected by related-reference lookup
            elif target.is_file() and target.is_relative_to(root) and text_path(target):
                sources[str(target.relative_to(root))] = source_record(target, root)
            elif target.is_file() and target.suffix.lower() in MEDIA | MODEL_BINARY:
                pass  # classified non-text target; no media/model bytes read
            elif target.is_dir() and target.is_relative_to(root) and any(beneath(target, root / scope) for scope in SCOPES):
                pass  # canonical subtree is already enumerated through SCOPES
            else:
                errors.append('SEED_ALIAS_ROLE_UNCLASSIFIED:' + row['path'])
            continue
        if path.is_file() and (text_path(path) or path.stat().st_size == 0):
            sources[str(path.relative_to(root))] = source_record(path, root)
        elif path.is_file() and path.suffix.lower() not in MEDIA | MODEL_BINARY:
            errors.append('SOURCE_FILE_ROLE_UNCLASSIFIED:' + str(path.relative_to(root)))
    return dict(sources=sources, aliases=sorted(aliases.values(), key=lambda r: r['path']), errors=sorted(set(errors)))


def source_unit(row, names, root=ROOT, roots=FIXTURES, producer=None,
                active_outputs=(), check=lambda: None):
    check(); path = root / row['path']
    if path == producer:
        fail('SOURCE_ROLE_COLLISION')
    if row['metadata']['size'] > MAX_SOURCE_BYTES:
        fail('REFERENCE_FILE_TOO_LARGE:' + row['path'])
    if any(beneath(path, scope) for scope in active_outputs):
        fail('ACTIVE_PARTIAL_SOURCE_UNREAD:' + row['path'])
    data, current = stable_bytes(path)
    if current != row['metadata'] and row['path'] not in DYNAMIC_RECORDS:
        fail('REFERENCE_SOURCE_DRIFT:' + row['path'])
    observed_inventory_metadata = row['metadata']
    row = dict(path=row['path'], metadata=current)
    try: text = data.decode('utf-8-sig')
    except UnicodeError: fail('REFERENCE_TEXT_UNREADABLE:' + row['path'])
    mapping = candidate_lookup(names, root)
    incoming, discovered, edges, errors, aliases = set(), {}, [], [], []
    for value in dict.fromkeys(source_values(text, path, root)):
        check()
        if not valid_path(value): continue
        value = re.sub(r':\d+$', '', value.split('#', 1)[0])
        if value.startswith(str(root) + '/'):
            target = Path(value)
        elif value.startswith(tuple(scope + '/' for scope in (*SCOPES, '_run'))):
            target = root / value
        elif value.startswith(('./', '../')) or (not value.startswith(('/', 'http:', 'https:')) and
              ':' not in value and (Path(value).suffix.lower() in TEXT | MEDIA | MODEL_BINARY or
                                    (path.parent / value).exists())):
            target = path.parent / value
            value = os.path.abspath(target)
        else:
            continue
        if any(char in value for char in '*?['):
            for key, selected in mapping.items():
                if fnmatch.fnmatchcase(key, str(target)):
                    incoming.update(selected)
            edges.append(dict(value=value, decision='CONSERVATIVE_GLOB_REFERENCE'))
            errors.append('BLOCKED_GLOB_REFERENCE_UNPROVED:' + value)
            continue
        try: resolved = target.resolve()
        except (OSError, RuntimeError):
            errors.append('REFERENCE_REALPATH_UNVERIFIED:' + value); continue
        incoming.update(mapping.get(str(resolved), ()))
        if target.is_symlink():
            aliases.append(dict(path=str(target.relative_to(root)), target=os.readlink(target),
                                resolved=str(resolved), metadata=metadata(target), exists=target.exists()))
        if not resolved.is_relative_to(root):
            edges.append(dict(value=value, target=str(resolved), decision='OUTSIDE_PROJECT_READ_NOT_FOLLOWED')); continue
        if is_fixture(resolved, root, roots):
            edges.append(dict(value=value, target=str(resolved), decision='CANDIDATE_REFERENCE')); continue
        if resolved == producer:
            edges.append(dict(value=value, target=str(resolved), decision='EXACT_SIGNED_PROOF_PRODUCER')); continue
        if any(beneath(resolved, scope) for scope in active_outputs):
            errors.append('ACTIVE_PARTIAL_CLOSURE_UNREAD:' + str(resolved)); continue
        if resolved.is_dir():
            errors.append('BLOCKED_DIRECTORY_REFERENCE_UNPROVED:' + str(resolved))
            edges.append(dict(value=value, target=str(resolved), decision='UNPROVEN_DIRECTORY_REFERENCE')); continue
        if not text_path(resolved):
            edges.append(dict(value=value, target=str(resolved), decision='NON_TEXT_NOT_PARSED')); continue
        if not resolved.is_file():
            # Absence cannot hide a currently readable consumer. Preserve it
            # explicitly; a newly appearing source is caught by re-enumeration.
            edges.append(dict(value=value, target=str(resolved), decision='ABSENT_TEXT_REFERENCE')); continue
        label = resolved.relative_to(root).as_posix()
        discovered[label] = source_record(resolved, root)
        edges.append(dict(value=value, target=label, decision='TEXT_CLOSURE'))
    return dict(**row, sha256=sha(data), observed_inventory_metadata=observed_inventory_metadata,
                time_semantics='MUTABLE_OBSERVATION_RECHECK_REQUIRED' if row['path'] in DYNAMIC_RECORDS else 'STATIC_SIGNED_SOURCE',
                incoming=sorted(incoming),
                discovered=discovered, edges=edges, aliases=aliases, errors=sorted(set(errors)))


def partition(rows, max_bytes=SHARD_BYTES, max_files=SHARD_FILES):
    chosen, total = [], 0
    for row in sorted(rows, key=lambda r: r['path']):
        size = row['metadata']['size']
        if size > max_bytes:
            fail('SINGLE_SOURCE_EXCEEDS_PARTITION')
        if chosen and (len(chosen) >= max_files or total + size > max_bytes):
            break
        chosen.append(row); total += size
    return chosen


def assemble(seed, fragments):
    """Deduplicate sources; no missing partition or unresolved edge is PASS."""
    sources = dict(seed['sources']); parsed = {}; errors = list(seed.get('errors', []))
    for fragment in fragments:
        units = fragment['units']
        if len(units) > SHARD_FILES or sum(u['metadata']['size'] for u in units) > SHARD_BYTES:
            fail('REFERENCE_PARTITION_BUDGET')
        for unit in units:
            label = unit['path']
            if label in parsed:
                fail('DUPLICATE_REFERENCE_UNIT:' + label)
            if label in sources and sources[label]['metadata'] != unit['metadata'] and label not in DYNAMIC_RECORDS:
                fail('REFERENCE_UNIT_METADATA_CONFLICT:' + label)
            if label in DYNAMIC_RECORDS: sources[label] = dict(path=label, metadata=unit['metadata'])
            parsed[label] = unit
            for target, row in unit['discovered'].items():
                if target in sources and sources[target]['metadata'] != row['metadata'] and target not in DYNAMIC_RECORDS:
                    fail('CLOSURE_METADATA_CONFLICT:' + target)
                sources[target] = row
            errors += unit.get('errors', [])
    # Reject invented units not reached from a seed. Traverse only signed edges.
    reached = set(seed['sources']); todo = list(reached)
    while todo:
        label = todo.pop()
        for target in parsed.get(label, {}).get('discovered', {}):
            if target not in reached:
                reached.add(target); todo.append(target)
    if set(parsed) - reached:
        fail('UNREACHABLE_REFERENCE_UNIT')
    pending = sorted(set(sources) - set(parsed))
    for label, unit in parsed.items():
        if label not in sources or (sources[label]['metadata'] != unit['metadata'] and label not in DYNAMIC_RECORDS):
            fail('LATE_REFERENCE_UNIT_METADATA_CONFLICT:' + label)
    return dict(sources=sources, parsed=parsed, pending=pending,
                errors=sorted(set(errors)), closed=not pending and not errors)


def related_references(graph, names, seed_aliases=(), root=ROOT):
    incoming = {name: [] for name in names}
    mapping = candidate_lookup(names, root)
    for label, unit in graph['parsed'].items():
        for name in unit['incoming']:
            incoming[name].append(label)
        for alias in unit.get('aliases', []):
            for name in mapping.get(alias['resolved'], ()):
                incoming[name].append('ALIAS:' + alias['path'])
    for alias in seed_aliases:
        for name in mapping.get(alias['resolved'], ()):
            incoming[name].append('ALIAS:' + alias['path'])
    return {name: sorted(set(refs)) for name, refs in incoming.items()}


def refresh_graph(graph, seed, names, root=ROOT, roots=FIXTURES,
                  producer=None, active_outputs=(), check=lambda: None):
    """Full metadata coverage, but read/parse only changed and new sources."""
    old_seed = graph.get('seed_paths', set(graph['sources']))
    if set(old_seed) - set(seed['sources']):
        fail('REFERENCE_SEED_REMOVED')
    parsed = dict(graph['parsed']); sources = dict(seed['sources'])
    todo = list(sorted(sources)); seen = set(); changed_bytes = changed_files = 0; changed = []
    while todo:
        check(); label = todo.pop()
        if label in seen: continue
        seen.add(label); row = sources[label]; path = root / label
        if any(beneath(path, scope) for scope in active_outputs):
            fail('ACTIVE_PARTIAL_SOURCE_UNREAD:' + label)
        current = metadata(path)
        previous = parsed.get(label)
        if previous is None or current != previous['metadata']:
            changed_bytes += current['size']; changed_files += 1
            if changed_bytes > SHARD_BYTES or changed_files > SHARD_FILES:
                fail('REFERENCE_DELTA_BUDGET')
            row = dict(path=label, metadata=current)
            unit = source_unit(row, names, root, roots, producer, active_outputs, check)
            if previous is not None and label.startswith(IMMUTABLE_PREFIXES) and unit['sha256'] != previous['sha256']:
                fail('IMMUTABLE_SOURCE_DRIFT:' + label)
            if previous is not None and label not in DYNAMIC_RECORDS and unit['sha256'] != previous['sha256']:
                fail('STATIC_REFERENCE_SOURCE_DRIFT:' + label)
            if previous is None and label.startswith(IMMUTABLE_PREFIXES):
                fail('NEW_IMMUTABLE_SOURCE:' + label)
            parsed[label] = unit
            changed.append(dict(input=dict(path=str(root / label), bytes=current['size'], sha256=unit['sha256']),
                                metadata=current, time_semantics=unit['time_semantics']))
        unit = dict(parsed[label]); unit['discovered'] = dict(unit['discovered'])
        for alias in unit.get('aliases', []):
            alias_path = root / alias['path']
            if (not alias_path.is_symlink() or metadata(alias_path) != alias['metadata'] or
                    str(alias_path.resolve()) != alias['resolved'] or alias_path.exists() != alias['exists']):
                fail('REFERENCE_ALIAS_DRIFT:' + alias['path'])
        for edge in unit.get('edges', []):
            if 'target' in edge:
                value = edge['value']
                original = Path(value) if value.startswith('/') else root / value
                expected = Path(edge['target']) if edge['target'].startswith('/') else root / edge['target']
                if original.resolve() != expected:
                    fail('REFERENCE_EDGE_REALPATH_DRIFT:' + value)
            if edge['decision'] == 'ABSENT_TEXT_REFERENCE':
                target = Path(edge['target'])
                if target.is_file():
                    if not target.is_relative_to(root) or is_fixture(target, root, roots):
                        fail('ABSENT_REFERENCE_APPEARED_IN_UNCLASSIFIED_SCOPE')
                    unit['discovered'][target.relative_to(root).as_posix()] = source_record(target, root)
        for target in list(unit['discovered']):
            unit['discovered'][target] = source_record(root / target, root)
            if target not in sources:
                sources[target] = unit['discovered'][target]
            todo.append(target)
        parsed[label] = unit
    fresh = assemble(seed, [dict(units=[parsed[name] for name in seen])]) if len(seen) <= SHARD_FILES and sum(parsed[name]['metadata']['size'] for name in seen) <= SHARD_BYTES else None
    # Large closures were already partitioned; assemble them again in bounded
    # virtual shards without rereading file contents.
    if fresh is None:
        remaining = [parsed[name] for name in seen]; fragments = []
        while remaining:
            chosen = partition(remaining); labels = {u['path'] for u in chosen}
            fragments.append(dict(units=chosen)); remaining = [u for u in remaining if u['path'] not in labels]
        fresh = assemble(seed, fragments)
    if not fresh['closed']:
        fail('REFERENCE_CLOSURE_NOT_CLOSED')
    fresh['seed_paths'] = sorted(seed['sources'])
    fresh['changed_files'], fresh['changed_bytes'] = changed_files, changed_bytes
    fresh['changed_sources'] = changed
    return fresh


def assert_aliases(current, expected):
    if current != expected: fail('ALIAS_SET_DRIFT')


def assert_no_process_holders(value):
    if value['errors'] or value['hits']: fail('PROCESS_PROOF_CHANGED')


def assert_references(current, expected):
    if current != expected: fail('RELEVANT_REFERENCE_SET_CHANGED')


def reference_gate(graph, seed, expected, names, root=ROOT, roots=FIXTURES,
                   producer=None, active_outputs=(), active_refs=None, check=lambda: None):
    fresh = refresh_graph(graph, seed, names, root, roots, producer, active_outputs, check)
    refs = related_references(fresh, names, seed['aliases'], root)
    for name, incoming in (active_refs or {}).items():
        refs[name] = sorted(set(refs[name]) | set(incoming))
    assert_references(refs, expected)
    return fresh


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, startticks=int(fields[19]), state=fields[0], cwd=os.readlink(proc / 'cwd'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def packet_path(value, root=ROOT):
    if not isinstance(value, str) or not valid_path(value):
        fail('UNCLASSIFIED_ACTIVE_SCOPE')
    path = Path(value) if value.startswith('/') else root / value
    return path.resolve()


def active_context(names, root=ROOT):
    state = load(root / 'docs/governance/LONG_HORIZON_TASK_STATE.json')
    mapping = candidate_lookup(names, root); incoming = defaultdict(list)
    packets, writers, outputs = [], [], []
    for row in state['tasks']:
        if row.get('status') not in ACTIVE: continue
        live = identity(row.get('pid')) if row.get('pid') else None
        if not live or live['startticks'] != row.get('proc_start_ticks'):
            fail('ACTIVE_WRITER_IDENTITY_UNVERIFIED:' + row['task_id'])
        packet_pin = row['task_packet']; verify(packet_pin); packet = load(packet_pin['path'])
        writer = packet.get('writer', {})
        if writer.get('pid') != live['pid'] or writer.get('proc_start_ticks', writer.get('startticks')) != live['startticks']:
            fail('ACTIVE_PACKET_WRITER_MISMATCH')
        packets.append(packet_pin); writers.append(dict(task_id=row['task_id'], **live))
        for key in ('read_set', 'expanded_read_set', 'write_set'):
            for value in packet.get(key, []):
                target = packet_path(value, root)
                for name in mapping.get(str(target), ()):
                    incoming[name].append('ACTIVE_' + key.upper() + ':' + row['task_id'])
                if key == 'write_set': outputs.append(str(target))
    return dict(incoming=dict(incoming), packets=packets, writers=writers, outputs=outputs,
                state_observed=pin(root / 'docs/governance/LONG_HORIZON_TASK_STATE.json'))


def process_check(names, root=ROOT, check=lambda: None):
    mapping = candidate_lookup(names, root); hits = defaultdict(list); errors = []; count = 0
    for proc in Path('/proc').iterdir():
        check()
        if not proc.name.isdigit(): continue
        pid = int(proc.name)
        try:
            before = identity(pid)
            if before is None: continue
            count += 1; found = set(mapping.get(before['cwd'], ()))
            for fd in (proc / 'fd').iterdir():
                try: target = os.readlink(fd).removesuffix(' (deleted)')
                except (FileNotFoundError, ProcessLookupError): continue
                found.update(mapping.get(target, ()))
            for line in (proc / 'maps').read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6: found.update(mapping.get(parts[5].removesuffix(' (deleted)'), ()))
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            for arg in argv:
                for value in (arg, arg.split('=', 1)[-1], *PATH_RE.findall(arg)):
                    if not valid_path(value) or '/' not in value or any(c.isspace() for c in value): continue
                    path = Path(value) if value.startswith('/') else Path(before['cwd']) / value
                    found.update(mapping.get(str(path.resolve()), ()))
            after = identity(pid)
            if after is None or before['startticks'] != after['startticks']: continue
            for name in found: hits[name].append(dict(pid=pid, startticks=before['startticks'], cwd=before['cwd']))
        except (FileNotFoundError, ProcessLookupError): pass
        except (OSError, RuntimeError) as exc: errors.append(dict(pid=pid, failure=type(exc).__name__))
    return dict(hits=dict(hits), errors=errors, processes_checked=count,
                checked=['PID', 'startticks', 'FD', 'CWD', 'argv', 'mmap'])


def git_context(root=ROOT):
    def run(args):
        return subprocess.run(['git', *args], cwd=root, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
                              check=True, capture_output=True).stdout
    tracked = set(run(['ls-files', '-z']).decode().split('\0'))
    dirty = run(['status', '--porcelain=v1', '-z', '--untracked-files=normal', '--', 'tests/pick_place_1008'])
    return tracked, dict(head=run(['rev-parse', 'HEAD']).decode().strip(),
                         scoped_dirty_sha256=sha(dirty), scoped_dirty_records=dirty.count(b'\0'))


def archive_verify(path, rows, check=lambda: None):
    expected = {row['path']: row for row in rows}; seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            check()
            if not member.isfile() or member.name not in expected or member.name in seen:
                fail('RECOVERY_MEMBER_SCOPE')
            if member.size != expected[member.name]['metadata']['size']:
                fail('RECOVERY_MEMBER_SIZE')
            stream = archive.extractfile(member)
            if stream is None: fail('RECOVERY_MEMBER_MISSING')
            payload = stream.read(); row = expected[member.name]
            if len(payload) != row['metadata']['size'] or sha(payload) != row['sha256']:
                fail('RECOVERY_MEMBER_SHA_DRIFT:' + member.name)
            seen.add(member.name)
    if seen != set(expected): fail('RECOVERY_MEMBER_SET')
    return dict(archive=pin(path), members=len(seen),
                uncompressed_bytes=sum(row['metadata']['size'] for row in rows),
                verification='EVERY_MEMBER_SIZE_AND_SHA')


def blockers(row, aliases, refs, proc, tracked, root=ROOT):
    path = root / row['path']; reasons = []
    if alias_blocks(path, aliases): reasons.append('BLOCKED_ALIAS_TARGET')
    if row['path'] in tracked: reasons.append('GIT_TRACKED')
    if row['metadata']['nlink'] != 1: reasons.append('HARDLINK_NOT_UNIQUE')
    if path.resolve() != path: reasons.append('REALPATH_ALIAS')
    if set(s.lower() for s in path.suffixes).intersection(MEDIA): reasons.append('PROTECTED_MEDIA_FILE')
    if set(s.lower() for s in path.suffixes).intersection(MODEL_BINARY): reasons.append('PROTECTED_MODEL_OR_BINARY_FILE')
    if set(s.lower() for s in path.suffixes).intersection(SOURCE): reasons.append('UNIQUE_SOURCE_NOT_DISPOSABLE')
    if '0916' in row['path'] or re.search(r'(?i)(licen[cs]e|notices?|copying|copyright|third[_-]?party)', path.name):
        reasons.append('PROTECTED_LICENSE_OR_BRANCH')
    if refs.get(row['path']): reasons.append('BLOCKED_REFERENCE_PROOF')
    if proc['errors']: reasons.append('PROCESS_SCAN_INCOMPLETE')
    if proc['hits'].get(row['path']): reasons.append('LIVE_PROCESS_HOLDER')
    return reasons


def read_back_private_lock(handle, path, expected_writer):
    """Settle our own write before the existing strict metadata guard.

    CPFS may materialize metadata on the first reopen after a write. This
    readback is only for the lock we hold, never for candidate/input files.
    The following guard still verifies the lock inode and all stable reads.
    """
    handle.flush()
    os.fsync(handle.fileno())
    if json.loads(Path(path).read_bytes()) != expected_writer:
        fail('FENCING_TOKEN_DRIFT')


class Runtime:
    """One real writer and one existing lock; no production-state mutation."""
    def __init__(self, phase, protected, inputs, expanded, targets=()):
        self.started = time.monotonic()
        self.wall = 1800 if phase == 'C1_DELETE' else 900
        self.phase = phase; self.cache = {}; self.owned = False
        current = identity(os.getpid())
        if current is None: fail('REAL_PROCESS_IDENTITY_REQUIRED')
        if LOCK.is_symlink() or LOCK.parent.resolve() != LOCK.parent:
            fail('LOCK_ALIAS')
        self.lock = LOCK.open('a+', encoding='utf-8')
        self.writer = dict(**current, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                           executor_epoch=0, fencing_token=secrets.token_hex(24),
                           deadline_at=(now() + timedelta(seconds=self.wall)).isoformat())
        opened = os.fstat(self.lock.fileno())
        self.writer.update(lock_device=opened.st_dev, lock_inode=opened.st_ino)
        self.out = BASE / (phase + '_' + now().strftime('%H%M%S') + '_' + self.writer['fencing_token'][:8])
        BASE.mkdir(parents=True, exist_ok=True)
        if BASE.resolve() != BASE: fail('OUTPUT_BASE_ALIAS')
        self.out.mkdir()
        self.protected = protected; self.inputs = inputs
        self.packet = dict(schema='source-partitioned-maintenance-v2', task_id=self.out.name,
                           phase=phase, authorization='USER_AUTHORIZED_INDEPENDENT_CLEANUP_PRODUCTION_PROOF_UNCHANGED',
                           registered_at=now().isoformat(), deadline_at=self.writer['deadline_at'], writer=self.writer,
                           inputs=inputs, run_signature=sha(canonical(dict(inputs=inputs, phase=phase, targets=list(targets)))),
                           read_set=['AGENTS.md', str(Path(__file__).absolute()), str(PROTOCOL),
                                     'baseline_v1/MANIFEST.json', 'docs/current/DUAL_BRANCH_BASELINE_V1.json',
                                     'tasks/current/INDEX.json', 'docs/governance/LONG_HORIZON_TASK_STATE.json', '/proc'],
                           expanded_read_set=sorted(set(map(str, expanded))),
                           write_set=[str(self.out), str(LOCK), *map(str, targets)],
                           execution_allowed=True, deletion_allowed=phase == 'C1_DELETE',
                           production_governance_write_allowed=False, algorithm_execution_allowed=False,
                           budgets=dict(wall_seconds=self.wall, cpu_threads=1, gpu=0, model_calls=0,
                                        training_steps=0, max_reference_partition_bytes=SHARD_BYTES,
                                        max_reference_partition_files=SHARD_FILES,
                                        max_delete_files=MAX_DELETE_FILES, max_delete_bytes=MAX_DELETE_BYTES),
                           required_outputs=['RESULT.json'], producer_role=dict(
                               exact_source=pin(Path(__file__).absolute()),
                               fixture_access='PROOF_AND_RECOVERY_ONLY_NOT_EXTERNAL_CONSUMER',
                               reverse_reference_exclusion='EXACT_SIGNED_SOURCE_ONLY'))

    def check(self):
        if time.monotonic() - self.started >= self.wall: fail('DEADLINE')

    def cached_verify(self, value):
        self.check(); path = Path(value['path']); current = metadata(path)
        if self.cache.get(value['path']) != current:
            verify(value); self.cache[value['path']] = current

    def claim(self):
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB); self.owned = True
        self.lock.seek(0)
        previous_text = self.lock.read()
        try: previous = json.loads(previous_text) if previous_text else {}
        except json.JSONDecodeError: fail('UNCLASSIFIED_LOCK_IDENTITY')
        self.writer['executor_epoch'] = int(previous.get('executor_epoch', 0)) + 1
        self.lock.seek(0); self.lock.truncate(); json.dump(self.writer, self.lock)
        read_back_private_lock(self.lock, LOCK, self.writer)
        save(self.out / 'TASK_PACKET.json', self.packet); save(self.out / 'WRITER.json', self.writer)
        self.packet_pin = pin(self.out / 'TASK_PACKET.json'); self.writer_pin = pin(self.out / 'WRITER.json')
        for value in [*self.inputs, *self.protected]: self.cached_verify(value)
        self.guard()

    def guard(self):
        self.check(); current = identity(os.getpid())
        if current is None or current['startticks'] != self.writer['startticks']: fail('WRITER_IDENTITY_DRIFT')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != self.writer['boot_id']: fail('BOOT_DRIFT')
        if load(LOCK) != self.writer: fail('FENCING_TOKEN_DRIFT')
        value = metadata(LOCK)
        if (value['device'], value['inode']) != (self.writer['lock_device'], self.writer['lock_inode']):
            fail('LOCK_INODE_DRIFT')
        for value in [self.packet_pin, self.writer_pin, *self.inputs]: self.cached_verify(value)

    def finish(self, value):
        value.update(task_id=self.out.name, finished_at=now().isoformat(), writer=self.writer,
                     task_packet=pin(self.out / 'TASK_PACKET.json') if (self.out / 'TASK_PACKET.json').exists() else None,
                     production_governance_writes=0, production_signals=0,
                     physical_directories_deleted=0, symlinks_deleted=0)
        save(self.out / 'RESULT.json', value)
        if self.owned: fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        return value


def read_sealed(directory, expected):
    directory = Path(directory).absolute()
    if directory.parent != BASE or directory.resolve() != directory:
        fail('SEALED_DIRECTORY_SCOPE')
    terminal = load(directory / 'RESULT.json')
    if terminal['status'] != expected: fail('UNSEALED_OR_WRONG_STAGE:' + str(directory))
    verify(terminal['task_packet']); producer = load(terminal['task_packet']['path'])
    if producer.get('production_governance_write_allowed') is not False or producer.get('algorithm_execution_allowed') is not False:
        fail('PRODUCER_CONTRACT')
    if pin(Path(__file__).absolute()) not in producer['inputs'] or pin(PROTOCOL) not in producer['inputs']:
        fail('PRODUCER_EXECUTOR_OR_PROTOCOL_DRIFT')
    return terminal, producer


def read_inventory(directory):
    terminal, producer = read_sealed(directory, 'PASSED_INVENTORY')
    if producer['phase'] != 'C0_INVENTORY': fail('WRONG_INVENTORY_PHASE')
    verify(terminal['inventory']); value = load(terminal['inventory']['path'])
    return value, terminal['inventory'], [pin(Path(directory) / 'RESULT.json'), terminal['inventory']]


def read_fragments(directories, inventory_pin):
    fragments, inputs = [], []
    for directory in directories:
        terminal, producer = read_sealed(directory, 'PASSED_REFERENCE_PART')
        if producer['phase'] != 'C0_REFERENCE': fail('WRONG_REFERENCE_PHASE')
        verify(terminal['fragment']); fragment = load(terminal['fragment']['path'])
        if fragment['inventory'] != inventory_pin: fail('PARTITION_INVENTORY_DRIFT')
        fragments.append(fragment)
        inputs += [pin(Path(directory) / 'RESULT.json'), terminal['fragment']]
    return fragments, inputs


def inventory_phase(runtime):
    runtime.guard(); files, aliases = fixture_inventory(check=runtime.check)
    if len(files) != EXPECTED_CANDIDATES or len(aliases) != EXPECTED_ALIASES:
        fail('FROZEN_FIXTURE_COUNT_DRIFT')
    names = [row['path'] for row in files]; active = active_context(names)
    extras = [pin_value['path'] for pin_value in active['packets']]
    seed = seed_sources(producer=Path(__file__).absolute(), extras=extras, check=runtime.check)
    value = dict(schema='cleanup-source-inventory-v2', fixtures=list(FIXTURES), rows=files,
                 aliases=aliases, seed=seed, active=active, protected_inputs=runtime.protected,
                 producer_exclusion=runtime.packet['producer_role'])
    runtime.guard(); save(runtime.out / 'INVENTORY.json', value)
    remaining = list(seed['sources'].values()); parts = 0
    while remaining:
        selected = partition(remaining); names_set = {row['path'] for row in selected}
        remaining = [row for row in remaining if row['path'] not in names_set]; parts += 1
    return dict(status='PASSED_INVENTORY', inventory=pin(runtime.out / 'INVENTORY.json'),
                candidate_files=len(files), candidate_bytes=sum(row['metadata']['size'] for row in files),
                aliases_preserved=len(aliases), seed_sources=len(seed['sources']),
                seed_bytes=sum(row['metadata']['size'] for row in seed['sources'].values()),
                minimum_seed_partitions=parts, reference_closed=False, deleted_files=0, deleted_bytes=0)


def reference_phase(runtime, value, inventory_pin, fragments):
    graph = assemble(value['seed'], fragments)
    rows = partition([graph['sources'][name] for name in graph['pending']])
    if not rows: fail('NO_PENDING_REFERENCE_PARTITION')
    active = active_context([row['path'] for row in value['rows']])
    # Public mutable governance views are reference sources, not successful
    # algorithm products. Do not blanket-classify their whole writer directory
    # as partial; actual output sources under _run remain unread.
    outputs = [scope for scope in active['outputs'] if beneath(scope, ROOT / '_run')]
    units = [source_unit(row, [r['path'] for r in value['rows']], producer=Path(__file__).absolute(),
                         active_outputs=outputs, check=runtime.check) for row in rows]
    fragment = dict(schema='cleanup-reference-fragment-v2', inventory=inventory_pin, units=units,
                    source_bytes=sum(row['metadata']['size'] for row in rows),
                    parsing_role='CURRENT_AND_HISTORICAL_REFERENCE_SOURCE_NOT_MEDIA_QUALITY_EVIDENCE')
    fresh = assemble(value['seed'], [*fragments, fragment])
    runtime.guard(); save(runtime.out / 'REFERENCE_PART.json', fragment)
    return dict(status='PASSED_REFERENCE_PART', fragment=pin(runtime.out / 'REFERENCE_PART.json'),
                parsed_sources=len(units), parsed_bytes=fragment['source_bytes'],
                pending_sources=len(fresh['pending']), coverage_errors=fresh['errors'],
                reference_closed=fresh['closed'], deleted_files=0, deleted_bytes=0)


def merge_phase(runtime, value, inventory_pin, fragments, shard_pins):
    graph = assemble(value['seed'], fragments); names = [row['path'] for row in value['rows']]
    if not graph['closed']:
        return dict(status='BLOCKED_REFERENCE_PROOF', pending_sources=graph['pending'],
                    coverage_errors=graph['errors'], reference_closed=False, deleted_files=0, deleted_bytes=0)
    active = active_context(names); extras = [v['path'] for v in active['packets']]
    seed = seed_sources(producer=Path(__file__).absolute(), extras=extras, check=runtime.check)
    def static_seed(current):
        return dict(sources={label: (dict(path=label, time_semantics='MUTABLE_RECHECK') if label in DYNAMIC_RECORDS else row)
                             for label, row in current['sources'].items()}, aliases=current['aliases'], errors=current['errors'])
    if static_seed(seed) != static_seed(value['seed']): fail('REFERENCE_SEED_SET_OR_METADATA_DRIFT')
    for label, row in graph['sources'].items():
        runtime.check()
        if metadata(ROOT / label) != row['metadata'] and label not in DYNAMIC_RECORDS: fail('REFERENCE_SOURCE_DRIFT:' + label)
    graph['seed_paths'] = sorted(value['seed']['sources'])
    graph = refresh_graph(graph, seed, names, producer=Path(__file__).absolute(),
                          active_outputs=[s for s in active['outputs'] if beneath(s, ROOT / '_run')], check=runtime.check)
    graph['seed_paths'] = sorted(seed['sources'])
    files, aliases = fixture_inventory(check=runtime.check)
    if files != value['rows'] or aliases != value['aliases']: fail('CANDIDATE_OR_ALIAS_DRIFT')
    tracked, git = git_context(); refs = related_references(graph, names, seed['aliases'])
    for name, incoming in active['incoming'].items(): refs[name] = sorted(set(refs[name]) | set(incoming))
    proc = process_check(names, check=runtime.check)
    combined_aliases = aliases + seed['aliases'] + [a for unit in graph['parsed'].values() for a in unit['aliases']]
    for row in files:
        row['blockers'] = blockers(row, combined_aliases, refs, proc, tracked)
        if row['path'] in graph['sources']:
            row['blockers'].append('RETAINED_REFERENCE_SOURCE')
        row['status'] = 'BLOCKED_REFERENCE_PROOF' if row['blockers'] else 'ELIGIBLE'
        if row['status'] == 'ELIGIBLE':
            data, current = stable_bytes(ROOT / row['path'])
            if current != row['metadata']: fail('CANDIDATE_SHA_METADATA_DRIFT')
            row['sha256'] = sha(data)
    selected = [row for row in files if row['status'] == 'ELIGIBLE']
    if len(selected) > MAX_DELETE_FILES or sum(row['metadata']['size'] for row in selected) > MAX_DELETE_BYTES:
        fail('DELETE_BUDGET')
    runtime.guard(); archive_path = runtime.out / 'RECOVERY.tar.gz'
    with archive_path.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w|gz', compresslevel=6) as archive:
            for row in selected:
                runtime.check(); data, current = stable_bytes(ROOT / row['path'])
                if current != row['metadata'] or sha(data) != row['sha256']: fail('ARCHIVE_SOURCE_DRIFT')
                member = tarfile.TarInfo(row['path']); member.size=len(data); member.mode=0o600
                archive.addfile(member, io.BytesIO(data))
        stream.flush(); os.fsync(stream.fileno())
    recovery = archive_verify(archive_path, selected, runtime.check)
    if fixture_inventory(check=runtime.check) != (value['rows'], aliases):
        fail('POST_RECOVERY_INVENTORY_DRIFT')
    for input_pin in runtime.protected: runtime.cached_verify(input_pin)
    final_active = active_context(names)
    final_seed = seed_sources(producer=Path(__file__).absolute(), extras=[p['path'] for p in final_active['packets']], check=runtime.check)
    if final_seed['aliases'] != seed['aliases']: fail('FINAL_REFERENCE_ALIAS_DRIFT')
    graph = reference_gate(graph, final_seed, refs, names, producer=Path(__file__).absolute(),
                           active_outputs=[s for s in final_active['outputs'] if beneath(s, ROOT / '_run')],
                           active_refs=final_active['incoming'], check=runtime.check)
    seed, active = final_seed, final_active
    runtime.guard()
    proof = dict(schema='closed-source-partitioned-cleanup-proof-v2', reference_closed=True,
                 inventory=inventory_pin, shard_pins=shard_pins, rows=files, aliases=aliases,
                 seed=seed, graph=graph, references=refs, active=active, process_check=proc, git=git,
                 recovery=recovery, protected_inputs=runtime.protected,
                 eligible_files=len(selected), eligible_bytes=sum(row['metadata']['size'] for row in selected),
                 producer_exclusion=runtime.packet['producer_role'])
    save(runtime.out / 'PROOF.json', proof)
    return dict(status='PASSED_PROOF' if not proc['errors'] else 'BLOCKED_REFERENCE_PROOF',
                proof=pin(runtime.out / 'PROOF.json'), recovery=recovery, reference_closed=True,
                eligible_files=len(selected), eligible_bytes=proof['eligible_bytes'],
                blocked_files=len(files) - len(selected), source_files=len(graph['parsed']),
                source_bytes=sum(row['metadata']['size'] for row in graph['parsed'].values()),
                deleted_files=0, deleted_bytes=0)


def deletion_phase(runtime, value):
    if value.get('reference_closed') is not True: fail('INCOMPLETE_REFERENCE_PROOF')
    selected = [row for row in value['rows'] if row['status'] == 'ELIGIBLE']
    if any(row['path'] in value['graph']['sources'] for row in selected):
        fail('PROOF_ELIGIBLE_REFERENCE_SOURCE')
    if len(selected) > MAX_DELETE_FILES or sum(row['metadata']['size'] for row in selected) > MAX_DELETE_BYTES:
        fail('DELETE_BUDGET')
    verify(value['recovery']['archive']); archive_verify(value['recovery']['archive']['path'], selected, runtime.check)
    for protected in value['protected_inputs']: runtime.cached_verify(protected)
    for shard in value['shard_pins']: verify(shard)
    names = [row['path'] for row in value['rows']]
    graph = value['graph']; aliases = value['aliases']; tracked = set(); events = []; context_pin = None
    try:
        for index, row in enumerate(selected):
            runtime.guard()
            _, current_aliases = fixture_inventory(check=runtime.check)
            assert_aliases(current_aliases, aliases)
            tracked, _ = git_context()
            for protected in value['protected_inputs']: runtime.cached_verify(protected)
            runtime.cached_verify(value['recovery']['archive'])
            path = ROOT / row['path']
            if not is_fixture(path) or path.resolve() != path or row['path'] in tracked:
                fail('TARGET_SCOPE_OR_GIT_CHANGED')
            if alias_blocks(path, aliases): fail('BLOCKED_ALIAS_TARGET')
            active = active_context([row['path']])
            if active['incoming']: fail('NEW_ACTIVE_CONSUMER')
            proc = process_check([row['path']], check=runtime.check)
            assert_no_process_holders(proc)
            data, current = stable_bytes(path)
            if current != row['metadata'] or sha(data) != row['sha256']: fail('TARGET_SHA_OR_INODE_DRIFT')
            if current['nlink'] != 1 or not stat.S_ISREG(current['mode']): fail('TARGET_TYPE_OR_HARDLINK_CHANGED')
            if blockers(row, aliases, {}, dict(errors=[], hits={}), tracked): fail('TARGET_PROTECTION_CHANGED')
            runtime.guard()
            # Every target, not every sixteenth: a normal docs/review reference
            # created after the preceding unlink must stop this unlink.
            active = active_context(names)
            seed = seed_sources(producer=Path(__file__).absolute(), extras=[v['path'] for v in active['packets']], check=runtime.check)
            if seed['aliases'] != value['seed']['aliases']: fail('REFERENCE_ALIAS_SET_DRIFT')
            graph = reference_gate(graph, seed, value['references'], names,
                                   producer=Path(__file__).absolute(),
                                   active_outputs=[s for s in active['outputs'] if beneath(s, ROOT / '_run')],
                                   active_refs=active['incoming'], check=runtime.check)
            if context_pin is None or graph['changed_sources']:
                context_path = runtime.out / ('CONTEXT_CHECK_%05d.json' % index)
                save(context_path, dict(previous=context_pin, proof_inventory=value['inventory'],
                     observed_at=now().isoformat(), sources_rechecked=len(graph['sources']),
                     seed_path_set_sha256=sha(canonical(sorted(seed['sources']))),
                     related_references_sha256=sha(canonical(value['references'])),
                     changed_source_inputs=graph['changed_sources'], active=active))
                context_pin = pin(context_path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                final_stat = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
                if (final_stat.st_dev, final_stat.st_ino, final_stat.st_ctime_ns, final_stat.st_mtime_ns,
                    final_stat.st_size, final_stat.st_mode, final_stat.st_nlink) != (
                        current['device'], current['inode'], current['ctime_ns'], current['mtime_ns'],
                        current['size'], current['mode'], current['nlink']): fail('LAST_MOMENT_TARGET_DRIFT')
                os.unlink(path.name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            finally: os.close(directory_fd)
            event = dict(path=row['path'], sha256=row['sha256'], bytes=current['size'],
                         allocated_bytes=current['allocated_bytes'], deleted_at=now().isoformat(),
                         recovery=value['recovery']['archive'], reference_context=context_pin)
            events.append(event)
            with (runtime.out / 'DELETED.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')
                stream.flush(); os.fsync(stream.fileno())
        return dict(status='PASSED_DELETE', deleted_files=len(events), deleted_bytes=sum(e['bytes'] for e in events))
    finally:
        save(runtime.out / 'DELETE_RECEIPT.json', dict(deleted=events, deleted_files=len(events),
             deleted_bytes=sum(e['bytes'] for e in events), archive=value['recovery']['archive'],
             recoverable=True, physical_directories_deleted=0, symlinks_deleted=0,
             production_paused=False, last_reference_context=context_pin, last_reference_delta=dict(
                 changed_files=graph.get('changed_files', 0), changed_bytes=graph.get('changed_bytes', 0))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('proof', 'delete'), required=True)
    parser.add_argument('--step', choices=('inventory', 'reference', 'merge'))
    parser.add_argument('--protected-inputs', required=True)
    parser.add_argument('--inventory-dir')
    parser.add_argument('--shard-dir', action='append', default=[])
    parser.add_argument('--proof-dir')
    args = parser.parse_args(argv)
    if ROOT.resolve() != ROOT or Path(__file__).absolute().resolve() != Path(__file__).absolute(): fail('ROOT_OR_EXECUTOR_ALIAS')
    if args.stage == 'proof' and (args.step is None or args.proof_dir): parser.error('proof requires --step, not --proof-dir')
    if args.stage == 'delete' and (not args.proof_dir or args.step or args.inventory_dir or args.shard_dir): parser.error('delete requires only --proof-dir')
    if args.stage == 'proof' and args.step != 'inventory' and not args.inventory_dir: parser.error('--inventory-dir required')
    if args.step == 'inventory' and (args.inventory_dir or args.shard_dir): parser.error('inventory does not consume earlier partial results')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    protected_file = Path(args.protected_inputs).absolute()
    if not protected_file.is_relative_to(ROOT) or protected_file.resolve() != protected_file: fail('PROTECTED_MANIFEST_SCOPE')
    protected = load(protected_file)
    inputs = [pin(Path(__file__).absolute()), pin(PROTOCOL), pin(OLD_EXECUTOR), pin(protected_file),
              pin(ROOT / 'THIRD_PARTY_NOTICES.md'), pin(ROOT / '.gitignore'), pin(ROOT / 'pyproject.toml')]
    value = inventory_pin = fragments = None; shard_pins = []; targets = []
    expanded = [str(protected_file), str(OLD_EXECUTOR), *FIXTURES, *SCOPES, '.git',
                'THIRD_PARTY_NOTICES.md', '.gitignore', 'pyproject.toml']
    if args.stage == 'proof' and args.step != 'inventory':
        value, inventory_pin, extra = read_inventory(args.inventory_dir); inputs += extra
        fragments, shard_pins = read_fragments(args.shard_dir, inventory_pin); inputs += shard_pins
        expanded += [str(Path(args.inventory_dir)), *map(str, args.shard_dir)]
        graph = assemble(value['seed'], fragments)
        selected_sources = partition([graph['sources'][name] for name in graph['pending']]) if args.step == 'reference' else list(graph['sources'].values())
        expanded += [str(ROOT / row['path']) for row in selected_sources]
    if args.stage == 'delete':
        terminal, producer = read_sealed(args.proof_dir, 'PASSED_PROOF')
        if producer['phase'] != 'C0_MERGE': fail('WRONG_PROOF_PHASE')
        verify(terminal['proof']); value = load(terminal['proof']['path'])
        if value['protected_inputs'] != protected: fail('C1_PROTECTED_INPUTS_MISMATCH')
        inputs += [pin(Path(args.proof_dir) / 'RESULT.json'), terminal['proof'], value['recovery']['archive']]
        targets = [str(ROOT / row['path']) for row in value['rows'] if row['status'] == 'ELIGIBLE']
        expanded += [str(Path(args.proof_dir)), *[p['path'] for p in value['shard_pins']]]
    phase = 'C1_DELETE' if args.stage == 'delete' else 'C0_' + args.step.upper()
    runtime = Runtime(phase, protected, inputs, expanded, targets)
    result = None
    try:
        runtime.claim()
        print(json.dumps(dict(stage='REGISTERED', phase=phase, path=str(runtime.out))), flush=True)
        if phase == 'C0_INVENTORY': result = inventory_phase(runtime)
        elif phase == 'C0_REFERENCE': result = reference_phase(runtime, value, inventory_pin, fragments)
        elif phase == 'C0_MERGE': result = merge_phase(runtime, value, inventory_pin, fragments, shard_pins)
        else: result = deletion_phase(runtime, value)
    except Exception as exc:
        receipt = load(runtime.out / 'DELETE_RECEIPT.json') if (runtime.out / 'DELETE_RECEIPT.json').exists() else {}
        result = dict(status='PARTIAL_STOPPED' if receipt.get('deleted_files') else 'STOPPED',
                      failure=type(exc).__name__ + ':' + str(exc),
                      deleted_files=receipt.get('deleted_files', 0), deleted_bytes=receipt.get('deleted_bytes', 0))
    result = runtime.finish(result)
    print(json.dumps(dict(path=str(runtime.out), result=pin(runtime.out / 'RESULT.json'), status=result['status'])), flush=True)
    return 0 if result['status'].startswith('PASSED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
