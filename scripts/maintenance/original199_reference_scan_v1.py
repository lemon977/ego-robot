"""Finite, conservative reverse-reference proof for one exact NAS relocation.

This is not an algorithm entry or a deletion tool.  Only a separately registered
maintenance writer may call it.  Old receipts are never rewritten, and old
algorithm commands are not made transparently NAS-aware.  An edge is satisfied
only by the explicit reader validated in R0, or its target remains protected.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import unquote

ROOT = Path('/path/to/chaoyang')
BASE = ROOT / '_run/maintenance/pick_place_1008_nas_export_20261010_v1'
DEST = Path('/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1')
LOCK = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
SELF = Path(__file__).absolute()
PARTITION_BYTES = 64 * 1024 * 1024
PARTITION_FILES = 2000
WALL_SECONDS = 1800
NAMES = {'raw_rgb.npz', 'clean_rgb.npz', 'final.mp4', 'review.mp4'}
TEXT = {'.py', '.md', '.json', '.jsonl', '.toml', '.yaml', '.yml', '.sh',
        '.txt', '.csv', '.html', '.ini', '.cfg', '.rst', '.bash', '.c', '.h',
        '.cpp', '.hpp', '.cu', '.rs', '.js', '.ts', '.tsx', '.jsx', '.log',
        '.xml', '.urdf', '.xacro', '.ipynb', '.conf', '.patch', '.diff',
        '.tsv', '.ps1', '.lua', '.sql', '.tex', '.cmake', '.bat', '.sample',
        '.out', '.err', '.stdout', '.stderr', '.pid', '.sha256', '.envrc',
        '.svg', '.mtl', '.mjcf', '.vert', '.frag', '.glsl', '.ptx'}
CODE = {'.py', '.sh', '.bash', '.c', '.h', '.cpp', '.hpp', '.cu', '.rs',
        '.js', '.ts', '.tsx', '.jsx', '.ps1', '.lua', '.bat'}
BINARY = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz',
          '.pkl', '.pickle', '.bin', '.zip', '.gz', '.xz', '.tar', '.7z',
          '.so', '.a', '.o', '.pyc', '.stl', '.obj', '.ply', '.glb', '.gltf',
          '.woff', '.woff2', '.ttf', '.ico', '.pdf', '.mp4', '.mov', '.avi',
          '.mkv', '.h264', '.h265', '.hevc', '.webm', '.mpeg', '.mpg',
          '.mjpeg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif',
          '.tiff', '.exr', '.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a',
          '.db', '.sqlite', '.sqlite3', '.zst', '.lz4', '.ninja_deps',
          '.cubin', '.fatbin', '.h5', '.hdf5', '.pb', '.tflite', '.dll',
          '.exe', '.dylib', '.pyd'}
ROOT_TEXT_NAMES = {'Makefile', 'Dockerfile', 'Containerfile', 'CMakeLists.txt',
                   'LICENSE', 'COPYING', 'NOTICE', 'HEAD', 'config',
                   'packed-refs', 'FETCH_HEAD', 'ORIG_HEAD', 'COMMIT_EDITMSG',
                   '.gitignore', '.gitattributes', '.dockerignore',
                   '.editorconfig', '.env', '.gitmodules', '.gitkeep',
                   'description', 'exclude', '.git', 'shallow', 'commondir',
                   'gitdir', 'README', 'INSTALL', 'AUTHORS', 'VERSION',
                   'CHANGELOG', '.ninja_log'}
ACTIVE = {'RUNNING', 'CLAIMED', 'WAIT_GPU_RESOURCE'}
TERMINAL = {'PASSED', 'COMPLETED', 'STOPPED', 'CANCELLED', 'CANCELED',
            'ABORTED', 'FAILED', 'REJECTED', 'ARCHIVED', 'ADOPTED',
            'FAILED_RUNTIME_FINAL', 'FAILED_QUALITY_FINAL', 'PARTIAL_STOPPED',
            'USER_STOPPED', 'FAILED_DATA_FINAL'}
PATH_RE = re.compile(r'(?:/path/to/chaoyang/|(?:_run|review|src|scripts|docs|tasks|tests|configs|contracts|manifests|baseline_v1)/|\.{1,2}/)[^\s\"\'<>`),;]*')
NAME_RE = re.compile(r'(?<![A-Za-z0-9_])(?:raw_rgb\.npz|clean_rgb\.npz|final\.mp4|review\.mp4)(?![A-Za-z0-9_])')
READ_RE = re.compile(r'\b(?:np\.load|numpy\.load|VideoCapture|ffprobe|read_bytes|read_text|open|load_gpu|reuse_gpu|reuse_stage|load_prepared)\s*\(')
OLD_LOCATION_RE = re.compile(r'original_full_session|original_producer_root|producer_root|reuse_(?:root|stage|gpu)|old_GPU_inputs')


def _need(value, code):
    if not value:
        raise RuntimeError(code)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stat_metadata(s):
    return dict(device=s.st_dev, inode=s.st_ino, mode=s.st_mode, nlink=s.st_nlink,
                size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def _metadata(path):
    return _stat_metadata(Path(path).lstat())


def _bytes(path, expected=None):
    path = Path(path)
    before = _metadata(path)
    _need(stat.S_ISREG(before['mode']) and path.resolve() == path,
          'REFERENCE_SOURCE_NONREGULAR_OR_ALIAS:' + str(path))
    if expected is not None:
        _need(before == expected, 'REFERENCE_SOURCE_METADATA_DRIFT:' + str(path))
    with path.open('rb') as stream:
        s = os.fstat(stream.fileno())
        _need((s.st_dev, s.st_ino) == (before['device'], before['inode']),
              'REFERENCE_SOURCE_OPEN_RACE:' + str(path))
        data = stream.read()
    _need(_metadata(path) == before and len(data) == before['size'],
          'REFERENCE_SOURCE_CHANGED_WHILE_READING:' + str(path))
    return data, before


def _pin(path):
    data, _ = _bytes(path)
    return dict(path=str(Path(path)), bytes=len(data), sha256=_sha(data))


def _verify(pin):
    _need(_pin(pin['path']) == pin, 'REFERENCE_PIN_DRIFT:' + pin['path'])


def _json(path):
    return json.loads(_bytes(path)[0])


def _save(runtime, name, value):
    runtime.guard()
    path = Path(runtime.out) / name
    _need(path.parent.resolve() == path.parent and path.is_relative_to(BASE),
          'REFERENCE_PROOF_OUTPUT_SCOPE')
    with path.open('xb') as stream:
        stream.write(_canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return _pin(path)


def _path(value, source):
    if not isinstance(value, str) or not value or len(value) > 4096 or any(
            c in value for c in '\0\n\r'):
        return None
    if Path(source).suffix == '.md':
        value = re.sub(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~])', r'\1', value)
        value = unquote(value)
    value = re.sub(r':\d+$', '', value.strip('<>').split('#', 1)[0])
    if value.startswith(('http:', 'https:', 'mailto:', 'data:')):
        return None
    if value.startswith('/'):
        p = Path(value)
    elif value.startswith(('_run/', 'review/', 'src/', 'scripts/', 'docs/',
                           'tasks/', 'tests/', 'configs/', 'contracts/',
                           'manifests/', 'baseline_v1/')):
        p = ROOT / value
    elif value.startswith(('./', '../')) or Path(value).name in NAMES:
        p = Path(source).parent / value
    else:
        return None
    return os.path.normpath(str(p))


def _rows_by_path(rows):
    result = {}
    for row in rows:
        p = row['source_path']
        _need(p not in result, 'DUPLICATE_STORAGE_INVENTORY_PATH')
        allowed_payload = (row.get('kind') == 'session_payload' and
                           Path(p).is_relative_to(ROOT / '_run/current'))
        allowed_navigation = (row.get('kind') == 'original_navigation_metadata' and
            Path(p).parent == ROOT / 'review/pick_place_1008/ALL199_ORIGINAL_EXECUTION_OUTPUTS_UNIFIED_DELIVERY_V1')
        _need(allowed_payload or allowed_navigation,
              'REFERENCE_INVENTORY_OUTSIDE_EXACT_EXPORT_SCOPE')
        result[p] = row
    return result


def _terminal(status):
    return status in TERMINAL or str(status).startswith(('PASSED_', 'FAILED_'))


def _indexed_packet_path(value):
    """Normalize only the frozen INDEX's root-relative packet spelling."""
    _need(isinstance(value, str) and bool(value) and
          not any(c in value for c in '\0\n\r'),
          'INDEX_PACKET_PATH_INVALID')
    original = Path(value)
    _need('..' not in original.parts, 'INDEX_PACKET_PATH_PARENT_ESCAPE')
    path = original if original.is_absolute() else ROOT / original
    _need(path.is_relative_to(ROOT) and path != ROOT,
          'INDEX_PACKET_PATH_OUTSIDE_PROJECT')
    _need(path.resolve() == path, 'INDEX_PACKET_PATH_ALIAS')
    return path


def _task_context(guard):
    ledger_path = ROOT / 'docs/governance/LONG_HORIZON_TASK_STATE.json'
    index_path = ROOT / 'tasks/current/INDEX.json'
    ledger = _json(ledger_path)
    index = _json(index_path)
    _need(not [r for r in ledger['tasks'] if r.get('status') in ACTIVE],
          'REFERENCE_PROOF_ACTIVE_PRODUCTION')
    _need(not [r for r in index['task_packets'] if r.get('execution_allowed')],
          'REFERENCE_PROOF_ROUTABLE_PRODUCTION')
    tasks = {row['task_id']: row for row in ledger['tasks']}
    closed = {}
    code = defaultdict(list)
    code_by_sha = defaultdict(list)
    metadata_pins = defaultdict(list)
    packets = {}
    for row in index['task_packets']:
        guard()
        tid = row['task_id']
        status = tasks.get(tid, {}).get('status')
        if not _terminal(status):
            continue
        p = _indexed_packet_path(row['packet_path'])
        data, _ = _bytes(p)
        _need(_sha(data) == row['packet_sha256'], 'NONROUTABLE_PACKET_SHA_DRIFT')
        packet = json.loads(data)
        pin = dict(path=str(p), bytes=len(data), sha256=_sha(data))
        provenance = dict(task_id=tid, ledger_status=status,
                          execution_allowed=False, packet=pin)
        closed[tid] = provenance
        packets[str(p)] = provenance
        reference_documents = [packet]
        result_pin = tasks[tid].get('result')
        if (isinstance(result_pin, dict) and
                set(('path', 'bytes', 'sha256')).issubset(result_pin) and
                Path(result_pin['path']).is_relative_to(ROOT) and
                Path(result_pin['path']).suffix == '.json'):
            _need(result_pin['bytes'] <= PARTITION_BYTES,
                  'SEALED_TASK_RESULT_EXCEEDS_REFERENCE_BUDGET')
            result_bytes, _ = _bytes(result_pin['path'])
            _need(len(result_bytes) == result_pin['bytes'] and
                  _sha(result_bytes) == result_pin['sha256'],
                  'SEALED_TASK_RESULT_PIN_DRIFT')
            reference_documents.append(json.loads(result_bytes))
            metadata_pins[result_pin['path']].append(dict(
                **provenance, metadata_pin=result_pin))
        for ref, _ in _typed_refs(reference_documents):
            path = ref.get('path')
            if isinstance(path, str) and set(Path(path).suffixes).intersection(CODE):
                record = dict(**provenance, code_pin={k: ref[k] for k in
                    ('path', 'bytes', 'sha256')})
                code[path].append(record)
                code_by_sha[ref['sha256']].append(record)
            elif (isinstance(path, str) and Path(path).is_relative_to(ROOT) and
                  (set(Path(path).suffixes).intersection(TEXT) or
                   Path(path).name in ROOT_TEXT_NAMES)):
                metadata_pins[path].append(dict(**provenance,
                    metadata_pin={k: ref[k] for k in ('path', 'bytes', 'sha256')}))
    pattern = re.compile(r'(?<![A-Za-z0-9_])(?:' + '|'.join(
        re.escape(tid) for tid in sorted(closed, key=len, reverse=True)) + r')(?![A-Za-z0-9_])') if closed else None
    return dict(ledger_pin=_pin(ledger_path), index_pin=_pin(index_path),
                closed=closed, code=dict(code), packets=packets,
                code_by_sha=dict(code_by_sha), metadata_pins=dict(metadata_pins),
                task_id_pattern=pattern)


def _typed_refs(value, pointer=''):
    if isinstance(value, dict):
        for key in ('path', 'source_path', 'old_path'):
            if isinstance(value.get(key), str) and 'bytes' in value and 'sha256' in value:
                yield dict(path=value[key], bytes=value['bytes'],
                           sha256=value['sha256']), pointer
        for key, child in value.items():
            yield from _typed_refs(child, pointer + '/' + str(key).replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for number, child in enumerate(value):
            yield from _typed_refs(child, pointer + '/' + str(number))


def _check_typed_pin(ref, expected, source, pointer):
    _need(ref['bytes'] == expected['bytes'] and
          ref['sha256'] == expected['sha256'],
          'TYPED_REFERENCE_SHA_OR_SIZE_DISAGREEMENT:' + str(source) + ':' + pointer)


def _strings(value, pointer=''):
    if isinstance(value, str):
        yield value, pointer
    elif isinstance(value, dict):
        for key, child in value.items():
            label = pointer + '/' + str(key).replace('~', '~0').replace('/', '~1')
            yield str(key), label + '/@key'
            yield from _strings(child, label)
    elif isinstance(value, list):
        for number, child in enumerate(value):
            yield from _strings(child, pointer + '/' + str(number))


def _owned(runtime, extra=()):
    roots = {str(Path(runtime.out)), *map(str, extra)}
    _need(all(Path(p).is_relative_to(BASE) and Path(p).resolve() == Path(p)
              for p in roots), 'REGISTERED_PROOF_OUTPUT_EXCLUSION_SCOPE')
    return sorted(roots)


def _excluded(path, output_roots):
    return any(Path(path).is_relative_to(Path(root)) for root in output_roots)


def _entry_kind(mode):
    if stat.S_ISDIR(mode):
        return 'DIRECTORY'
    if stat.S_ISLNK(mode):
        return 'SYMLINK'
    if stat.S_ISREG(mode):
        return 'FILE'
    return 'SPECIAL'


def _directory_entries(directory, eligible_paths, output_roots):
    result = []
    with os.scandir(directory) as entries:
        for entry in entries:
            p = Path(entry.path)
            if str(p) in eligible_paths or _excluded(p, output_roots):
                continue
            s = entry.stat(follow_symlinks=False)
            result.append([entry.name, _entry_kind(s.st_mode)])
    return sorted(result)


def _census(runtime, eligible_paths, output_roots, guard):
    sources, aliases, directories, unknown = [], [], [], []
    classes = Counter()
    pending = [ROOT]
    visited = set()
    regular_files = 0
    runtime.progress.update(reference_census_directories=0,
        reference_census_files=0, reference_census_text_sources=0,
        reference_census_unknown=0, reference_census_aliases=0)
    while pending:
        guard()
        directory = pending.pop()
        _need(directory.resolve() == directory, 'REFERENCE_DIRECTORY_ALIAS')
        if str(directory) in visited or _excluded(directory, output_roots):
            continue
        visited.add(str(directory))
        before = _metadata(directory)
        children = []
        runtime.progress['reference_census_directories'] = len(visited)
        with os.scandir(directory) as items:
            for entry in items:
                guard()
                p = Path(entry.path)
                if _excluded(p, output_roots):
                    classes['EXACT_REGISTERED_PROOF_OUTPUT_SCOPE'] += 1
                    continue
                s = entry.stat(follow_symlinks=False)
                # Preserve the original summary separately from classification:
                # candidate binaries are omitted only from the summary, while
                # the exact lock is summarized but not a reference source.
                if str(p) not in eligible_paths:
                    children.append([entry.name, _entry_kind(s.st_mode)])
                if p == LOCK:
                    classes['EXACT_OWNER_FENCED_MAINTENANCE_LOCK'] += 1
                    continue
                if stat.S_ISLNK(s.st_mode):
                    target = os.readlink(p)
                    try:
                        resolved = str(p.resolve())
                    except (OSError, RuntimeError):
                        resolved = None
                    aliases.append(dict(path=str(p), target=target, resolved=resolved,
                                        metadata=_metadata(p)))
                    classes['SYMLINK_NOT_FOLLOWED'] += 1
                elif stat.S_ISDIR(s.st_mode):
                    if p == ROOT / '.git/objects':
                        classes['GIT_OBJECT_STORE_EXPLICIT_BINARY_METADATA_ONLY'] += 1
                    else:
                        pending.append(p)
                elif stat.S_ISREG(s.st_mode):
                    regular_files += 1
                    suffixes = {v.lower() for v in p.suffixes}
                    if p.is_relative_to(ROOT / '.git') and p.name in {'index', 'index.lock'}:
                        classes['GIT_INDEX_EXPLICIT_BINARY_METADATA_ONLY'] += 1
                    elif p.is_relative_to(ROOT / '.git/refs') or p.is_relative_to(ROOT / '.git/logs'):
                        sources.append(dict(path=str(p), metadata=_stat_metadata(s)))
                        classes['GIT_REF_OR_REFLOG_TEXT_SOURCE'] += 1
                    elif p.suffix.lower() in BINARY or (suffixes.intersection(BINARY) and not suffixes.intersection(TEXT)):
                        classes['MEDIA_MODEL_CACHE_BINARY_METADATA_ONLY'] += 1
                    elif suffixes.intersection(TEXT) or p.name in ROOT_TEXT_NAMES:
                        sources.append(dict(path=str(p), metadata=_stat_metadata(s)))
                        classes['TEXT_REFERENCE_SOURCE'] += 1
                    elif s.st_size == 0:
                        classes['EMPTY_NONCONSUMING_REGULAR_FILE'] += 1
                    else:
                        unknown.append(dict(path=str(p), metadata=_stat_metadata(s),
                                            reason='REGULAR_FILE_ROLE_UNCLASSIFIED'))
                        classes['UNCLASSIFIED_REGULAR_FILE'] += 1
                else:
                    unknown.append(dict(path=str(p), metadata=_stat_metadata(s),
                                        reason='SPECIAL_FILE_ROLE_UNCLASSIFIED'))
                    classes['UNCLASSIFIED_SPECIAL_FILE'] += 1
                runtime.progress.update(reference_census_files=regular_files,
                    reference_census_text_sources=len(sources),
                    reference_census_unknown=len(unknown),
                    reference_census_aliases=len(aliases))
        directories.append(dict(path=str(directory), metadata=before,
            stable_entries_sha256=_sha(_canonical(sorted(children)))))
        _need(_metadata(directory) == before, 'REFERENCE_DIRECTORY_CHANGED_DURING_ENUMERATION:' + str(directory))
    return dict(sources=sorted(sources, key=lambda r: r['path']),
                aliases=sorted(aliases, key=lambda r: r['path']),
                directories=sorted(directories, key=lambda r: r['path']),
                unknown=sorted(unknown, key=lambda r: r['path']), classes=dict(classes))


def _validation(runtime):
    values = []
    for pin in runtime.inputs:
        if Path(pin['path']).suffix != '.json':
            continue
        _verify(pin)
        d = _json(pin['path'])
        if d.get('status') == 'PASSED_READONLY_STORAGE_VALIDATION':
            values.append((pin, d))
    _need(len(values) == 1, 'EXACT_SEALED_R0_REFERENCE_VALIDATION_REQUIRED')
    pin, d = values[0]
    _need(d.get('prepared_parity_sessions') == 199 and
          d.get('preserved_four_stage_closures') == 796,
          'R0_ALL199_FOUR_STAGE_CLOSURE_REQUIRED')
    _verify(d['storage_binding'])
    _verify(d['verified_inventory'])
    return dict(result=pin, storage_binding=d['storage_binding'],
                verified_inventory=d['verified_inventory'],
                preserved_sessions=199, preserved_stage_closures=796,
                transparent_old_algorithm_read_support=False)


def _associated(source, text, task_context):
    result = {}
    for match in re.finditer(r'_run/current/([^/\s\"\'<>]+)', str(source) + '\n' + text):
        tid = match.group(1)
        if tid in task_context['closed']:
            result[tid] = task_context['closed'][tid]
    # A metadata producer can carry its task ID as a JSON field, not a path.
    pattern = task_context['task_id_pattern']
    for match in pattern.finditer(text) if pattern else ():
        tid = match.group(0)
        if tid in task_context['closed']:
            result[tid] = task_context['closed'][tid]
    return [result[k] for k in sorted(result)]


def _maintenance_package(path, cache):
    for parent in (Path(path).parent, *Path(path).parents):
        if parent == BASE.parent or not parent.is_relative_to(ROOT / '_run/maintenance'):
            break
        key = str(parent)
        if key in cache:
            value = cache[key]
            if value:
                return value
            continue
        packet = parent / 'TASK_PACKET.json'
        result = parent / 'RESULT.json'
        value = None
        if packet.is_file() and result.is_file() and not packet.is_symlink() and not result.is_symlink():
            p, r = _json(packet), _json(result)
            if (p.get('algorithm_execution_allowed') is False and
                    p.get('production_governance_write_allowed') is False and
                    (_terminal(r.get('status')) or str(r.get('status')).startswith('PASSED'))):
                value = dict(packet=_pin(packet), result=_pin(result),
                             algorithm_execution_allowed=False,
                             terminal_status=r.get('status'),
                             member_pins={ref['path']: ref for ref, _ in
                                          _typed_refs([p, r])})
                value['member_pins'][str(packet)] = value['packet']
                value['member_pins'][str(result)] = value['result']
        cache[key] = value
        if value:
            return value
    return None


def _consumer_class(source, source_sha, text, associated, task_context,
                    maintenance, signed_maintenance_sources):
    path = str(source)
    if path in signed_maintenance_sources:
        _need(signed_maintenance_sources[path]['sha256'] == source_sha,
              'SIGNED_MAINTENANCE_SOURCE_DRIFT')
        return 'SIGNED_STORAGE_PROOF_OR_EXPLICIT_READER', []
    if Path(source).suffix in CODE:
        freezes = [r for r in task_context['code'].get(path, [])
                   if r['code_pin']['sha256'] == source_sha]
        if not freezes:
            # Frozen copies retain their original bytes but may have a new
            # filename.  The originating signed code pin and terminal task are
            # evidence, not directory-name-based permission to run the copy.
            freezes = task_context['code_by_sha'].get(source_sha, [])
        if freezes:
            return 'FROZEN_NONROUTABLE_CODE_NOT_TRANSPARENTLY_REENABLED', freezes
        return 'UNCLASSIFIED_CODE_CONSUMER', []
    if maintenance:
        member = maintenance['member_pins'].get(path)
        if member is not None and member['sha256'] == source_sha:
            return 'SEALED_NONALGORITHM_MAINTENANCE_EVIDENCE', [maintenance]
    if path in task_context['packets']:
        return 'SEALED_NONROUTABLE_TASK_PACKET', [task_context['packets'][path]]
    freezes = [r for r in task_context.get('metadata_pins', {}).get(path, [])
               if r['metadata_pin']['sha256'] == source_sha]
    if freezes:
        return 'HISTORICAL_METADATA_WITH_NONROUTABLE_PRODUCER', freezes
    # Mentioning a retired task or residing beside its receipts is not a pin:
    # a newly written contract can consume those old results.  Such a source
    # remains protected until its actual reader or immutable status is proved.
    return 'UNCLASSIFIED_METADATA_CONSUMER', []


def _values(text, source, parsed):
    found = []
    seen = set()
    for match in PATH_RE.finditer(text):
        value = match.group(0)
        key = (value, match.start(), 'TEXT_PATH', None)
        if key not in seen:
            seen.add(key)
            found.append(dict(value=value, offset=match.start(),
                              kind='TEXT_PATH', json_pointer=None))
    if parsed is not None:
        for value, pointer in _strings(parsed):
            if '/' not in value and value not in NAMES:
                continue
            offset = text.find(value)
            found.append(dict(value=value, offset=offset if offset >= 0 else None,
                              kind='JSON_STRING', json_pointer=pointer))
    elif Path(source).suffix == '.py':
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if '/' in value or value in NAMES:
                        found.append(dict(value=value, offset=None, kind='PYTHON_LITERAL',
                                          json_pointer=None, line=getattr(node, 'lineno', None)))
        except SyntaxError:
            pass
    if Path(source).suffix == '.md':
        for match in re.finditer(r'\]\(([^)]+)\)', text):
            found.append(dict(value=match.group(1), offset=match.start(1),
                              kind='MARKDOWN_LINK', json_pointer=None))
    return found


class _Targets:
    def __init__(self, eligible):
        self.rows = {r['source_path']: r for r in eligible}
        self.ancestors = defaultdict(set)
        for p in self.rows:
            current = Path(p)
            for parent in (current, *current.parents):
                if not parent.is_relative_to(ROOT / '_run/current'):
                    break
                # Root scope literals are not references to particular data.
                if len(parent.relative_to(ROOT).parts) < 3:
                    continue
                self.ancestors[str(parent)].add(p)

    def affected(self, value, source):
        p = _path(value, source)
        if p is None:
            return [], None, None
        if any(char in p for char in '*?['):
            hits = [n for n in self.rows if fnmatch.fnmatchcase(n, p) or
                    any(fnmatch.fnmatchcase(str(parent), p) for parent in Path(n).parents)]
            return sorted(hits), p, 'GLOB_OR_DIRECTORY_PATTERN'
        hits = self.ancestors.get(p, set())
        kind = 'EXACT_PATH' if p in self.rows else 'ANCESTOR_DIRECTORY'
        # A relative alias may resolve to a candidate without the lexical path
        # ever spelling the original target.  Never follow its contents.
        try:
            resolved = str(Path(p).resolve())
        except (OSError, RuntimeError):
            return sorted(hits), p, 'UNRESOLVED_PATH'
        if resolved != p:
            hits = set(hits) | self.ancestors.get(resolved, set())
            kind = 'RESOLVED_ALIAS_REFERENCE'
        return sorted(hits), p, kind


def scan_reference_closure(runtime, rows, eligible, retained):
    """Create bounded reverse edges; may only remove proposed eligibility."""
    started = time.monotonic()
    last_guard = [0.0]
    def guard():
        instant = time.monotonic()
        if instant - last_guard[0] >= 1:
            runtime.guard()
            last_guard[0] = instant
        _need(time.monotonic() - started < WALL_SECONDS, 'REFERENCE_SCAN_DEADLINE')
    guard()
    all_rows = _rows_by_path(rows)
    _need(0 < len(eligible) <= 732, 'REFERENCE_SCAN_ELIGIBLE_SCOPE_NOT_NARROWED')
    _need(all(r['source_path'] in all_rows and Path(r['source_path']).name in NAMES
              for r in eligible), 'REFERENCE_SCAN_WRONG_TARGET_SET')
    targets = _Targets(eligible)
    eligible_paths = set(targets.rows)
    validation = _validation(runtime)
    task_context = _task_context(guard)
    output_roots = _owned(runtime)
    census = _census(runtime, eligible_paths, output_roots, guard)
    retained_paths = {r['path'] for r in retained}
    signed_maintenance_sources = {
        p['path']: p for p in runtime.inputs
        if Path(p['path']).suffix in CODE and (
            Path(p['path']).is_relative_to(ROOT / 'scripts/maintenance') or
            Path(p['path']).is_relative_to(ROOT / 'src/chaoyang/storage_relocation'))}
    signed_maintenance_sources[str(SELF)] = _pin(SELF)
    blocked = set()
    unresolved = []
    if census['unknown']:
        blocked.update(eligible_paths)
        unresolved.extend(census['unknown'])
    partitions = []
    batch, batch_bytes = [], 0
    edges_total = 0
    class_counts = Counter()
    package_cache = {}
    source_manifest = []

    def flush():
        nonlocal batch, batch_bytes
        if not batch:
            return
        name = 'REFERENCE_PARTITION_%05d_V1.json' % len(partitions)
        value = dict(schema='original199-storage-reference-partition-v1',
                     units=batch, source_files=len(batch), source_bytes=batch_bytes,
                     max_source_files=PARTITION_FILES, max_source_bytes=PARTITION_BYTES)
        partitions.append(_save(runtime, name, value))
        batch, batch_bytes = [], 0

    for row in census['sources']:
        guard()
        p = Path(row['path'])
        if row['metadata']['size'] > PARTITION_BYTES:
            blocked.update(eligible_paths)
            unresolved.append(dict(path=str(p), reason='TEXT_SOURCE_EXCEEDS_PARTITION_BUDGET',
                                   metadata=row['metadata']))
            source_manifest.append(dict(**row, sha256=None, source_read=False))
            continue
        if len(batch) >= PARTITION_FILES or batch_bytes + row['metadata']['size'] > PARTITION_BYTES:
            flush()
        data, current = _bytes(p, row['metadata'])
        digest = _sha(data)
        source_pin = dict(path=str(p), bytes=len(data), sha256=digest)
        try:
            text = data.decode('utf-8-sig')
        except UnicodeError:
            blocked.update(eligible_paths)
            unresolved.append(dict(path=str(p), reason='TEXT_SOURCE_UNREADABLE', source=source_pin))
            source_manifest.append(dict(**row, sha256=digest, source_read=False))
            continue
        parsed = None
        if p.suffix in {'.json', '.ipynb'}:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
        elif p.suffix == '.jsonl':
            parsed = []
            for line in text.splitlines():
                guard()
                if not line.strip():
                    continue
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError:
                    # An invalid partial line cannot hide references: the
                    # original text is still scanned, and its role is explicit.
                    parsed = None
                    break
        associated = _associated(p, text, task_context)
        maintenance = _maintenance_package(p, package_cache)
        consumer, provenance = _consumer_class(p, digest, text, associated,
            task_context, maintenance, signed_maintenance_sources)
        edges = []
        typed = {}
        if parsed is not None:
            for ref, pointer in _typed_refs(parsed):
                if ref['path'] not in all_rows:
                    continue
                expected = all_rows[ref['path']]
                _check_typed_pin(ref, expected, p, pointer)
                if ref['path'] in targets.rows:
                    typed[ref['path']] = dict(pin=ref, pointer=pointer)
        seen_edges = set()
        for value in _values(text, p, parsed):
            hits, lexical, kind = targets.affected(value['value'], p)
            for target in hits:
                key = (target, lexical, kind, value.get('json_pointer'), value.get('offset'))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                item = targets.rows[target]
                pin = dict(path=target, bytes=item['bytes'], sha256=item['sha256'])
                has_exact_pin = target in typed
                allowed = consumer in {
                    'SIGNED_STORAGE_PROOF_OR_EXPLICIT_READER',
                    'SEALED_NONALGORITHM_MAINTENANCE_EVIDENCE',
                    'SEALED_NONROUTABLE_TASK_PACKET',
                    'HISTORICAL_METADATA_WITH_NONROUTABLE_PRODUCER',
                    'FROZEN_NONROUTABLE_CODE_NOT_TRANSPARENTLY_REENABLED'}
                if kind in {'GLOB_OR_DIRECTORY_PATTERN', 'UNRESOLVED_PATH', 'RESOLVED_ALIAS_REFERENCE'}:
                    allowed = False
                # Code paths are only historical frozen templates here.  No
                # running/selected unadapted consumer is silently repaired.
                if consumer.startswith('UNCLASSIFIED') or not allowed:
                    blocked.add(target)
                    decision = 'BLOCKED_REFERENCE_PROOF'
                else:
                    decision = 'REFERENCED_SATISFIED_BY_EXPLICIT_R0_READER_ONLY'
                edge = dict(source=source_pin, offset=value.get('offset'),
                    offset_unit='UNICODE_CODEPOINT',
                    json_pointer=value.get('json_pointer'), line=value.get('line'),
                    raw_reference=value['value'], lexical_path=lexical, ref_kind=kind,
                    extraction_kind=value['kind'], original_pin=pin,
                    typed_pin_verified=has_exact_pin,
                    typed_pin_pointer=typed.get(target, {}).get('pointer'),
                    NAS_mapping=dict(destination_root=str(DEST),
                                     destination_relative=item['destination_relative'],
                                     bytes=item['bytes'], sha256=item['sha256']),
                    consumer_class=consumer, producer_status_evidence=provenance,
                    reader_validation=validation['result'], decision=decision,
                    old_algorithm_or_direct_open_transparently_supported=False)
                edges.append(edge)
                class_counts[decision] += 1
        # Generic new-raw output names are not references to old sessions.
        # Only an actual read paired with an old/reuse location is unresolved.
        if (p.suffix in CODE and NAME_RE.search(text) and READ_RE.search(text) and
                OLD_LOCATION_RE.search(text) and
                consumer == 'UNCLASSIFIED_CODE_CONSUMER'):
            blocked.update(eligible_paths)
            unresolved.append(dict(source=source_pin,
                reason='UNBOUND_OLD_RUN_DYNAMIC_READ_CONSUMER',
                affected_paths='ALL_PROPOSED_CANDIDATES',
                producer_status_evidence=associated,
                actual_read_binding_verified=False))
        unit = dict(source=source_pin, metadata=current, consumer_class=consumer,
                    producer_status_evidence=provenance, edges=edges)
        batch.append(unit)
        batch_bytes += len(data)
        source_manifest.append(dict(**row, sha256=digest, source_read=True))
        edges_total += len(edges)
        runtime.progress.update(reference_sources=len(source_manifest),
            reference_edges=edges_total, reference_blocked_targets=len(blocked))
    flush()

    # A direct file alias remains a direct legacy consumer.  Directory aliases
    # are only preserved provenance navigation if they point at an exact
    # completed producer session which retains its JSON/array evidence.
    alias_edges = []
    for alias in census['aliases']:
        guard()
        resolved = alias['resolved']
        hits = targets.ancestors.get(resolved, set()) if resolved else eligible_paths
        for target in sorted(hits):
            row = targets.rows[target]
            target_parent = Path(target).parent
            exact_session_dir = resolved == str(target_parent)
            retained_evidence = any(Path(p).parent == target_parent for p in retained_paths)
            tid = target_parent.relative_to(ROOT / '_run/current').parts[0]
            closed = task_context['closed'].get(tid)
            allowed = exact_session_dir and retained_evidence and closed is not None
            if not allowed:
                blocked.add(target)
            alias_edges.append(dict(alias=alias,
                original_pin=dict(path=target, bytes=row['bytes'], sha256=row['sha256']),
                NAS_mapping=dict(destination_root=str(DEST),
                                 destination_relative=row['destination_relative']),
                ref_kind='DIRECTORY_ALIAS' if exact_session_dir else 'DIRECT_OR_UNCLASSIFIED_ALIAS',
                consumer_class='RETAINED_PROVENANCE_DIRECTORY_ALIAS_EXPLICIT_READER_ONLY' if allowed else 'UNADAPTED_ALIAS_CONSUMER',
                producer_status_evidence=closed, reader_validation=validation['result'],
                decision='REFERENCED_SATISFIED_BY_EXPLICIT_R0_READER_ONLY' if allowed else 'BLOCKED_REFERENCE_PROOF',
                legacy_binary_open_through_alias_supported=False))
    alias_partitions = []
    for start in range(0, len(alias_edges), PARTITION_FILES):
        alias_partitions.append(_save(runtime, 'REFERENCE_ALIAS_PARTITION_%05d_V1.json' % len(alias_partitions),
            dict(schema='original199-storage-alias-reference-partition-v1',
                 edges=alias_edges[start:start + PARTITION_FILES])))
    census_value = dict(schema='original199-complete-project-reference-census-v1',
        sources=source_manifest, aliases=census['aliases'], directories=census['directories'],
        unknown=census['unknown'], classes=census['classes'],
        exact_excluded_output_roots=output_roots, exact_excluded_lock=str(LOCK),
        all_root_directories_including_hidden_enumerated=True,
        external_symlinks_followed=False, binary_content_read=False,
        eligible_paths=sorted(eligible_paths),
        producer_ledger=task_context['ledger_pin'], task_index=task_context['index_pin'])
    census_pin = _save(runtime, 'REFERENCE_SOURCE_CENSUS_V1.json', census_value)
    unresolved_pin = _save(runtime, 'REFERENCE_UNRESOLVED_V1.json', dict(items=unresolved))
    proof = dict(schema='original199-storage-reference-closure-v1',
        status='CLOSED_WITH_EXPLICIT_BLOCKED_TARGETS' if blocked else 'CLOSED_FOR_PROPOSED_TARGETS',
        blocked_paths=sorted(blocked), proposed_paths=sorted(eligible_paths),
        reference_partitions=partitions, alias_partitions=alias_partitions,
        source_census=census_pin, unresolved=unresolved_pin,
        reader_validation=validation, production_context=dict(
            ledger=task_context['ledger_pin'], index=task_context['index_pin']),
        exact_excluded_output_roots=output_roots,
        coverage=dict(source_files=len(source_manifest),
            source_bytes=sum(r['metadata']['size'] for r in source_manifest),
            source_SHA_verified=sum(r['source_read'] for r in source_manifest),
            directories=len(census['directories']), aliases=len(census['aliases']),
            unknown_regular_or_special_files=len(census['unknown']),
            edges=edges_total, alias_edges=len(alias_edges),
            source_partitions=len(partitions), alias_partitions=len(alias_partitions),
            unresolved_items=len(unresolved), proposed_candidates=len(eligible_paths),
            blocked_candidates=len(blocked), remaining_candidates=len(eligible_paths - blocked),
            max_partition_bytes=PARTITION_BYTES, max_partition_files=PARTITION_FILES,
            elapsed_seconds=round(time.monotonic() - started, 3),
            root_hidden_and_run_receipts_included=True,
            numerical_or_quality_gates_changed=False,
            all_reference_edges_claimed_absent=False,
            old_frozen_algorithms_transparently_repaired=False),
        decision_counts=dict(class_counts))
    guard()
    return proof


def recheck_reference_closure(proof, runtime, metadata_only=False):
    """Re-enumerate before C1; each unlink also checks source/directory identity.

    Metadata-only calls still detect newly created consumer texts through every
    directory identity; modified directory entries are rechecked with only the
    exact planned binary unlinks and this writer's own output ignored.
    """
    started = time.monotonic()
    last_guard = [0.0]
    def guard():
        instant = time.monotonic()
        if instant - last_guard[0] >= 1:
            runtime.guard()
            last_guard[0] = instant
        _need(time.monotonic() - started < WALL_SECONDS, 'REFERENCE_RECHECK_DEADLINE')
    guard()
    cache = getattr(runtime, '_original199_reference_recheck_cache', None)
    if cache is None:
        cache = {}
        runtime._original199_reference_recheck_cache = cache
    def verify_cached(pin, decode=False):
        path = Path(pin['path'])
        identity = _metadata(path)
        _need(path.resolve() == path, 'REFERENCE_RECHECK_PIN_ALIAS')
        key = (pin['path'], pin['bytes'], pin['sha256'])
        prior = cache.get(key)
        if prior is None or prior['metadata'] != identity:
            data, actual = _bytes(path, identity)
            _need(len(data) == pin['bytes'] and _sha(data) == pin['sha256'],
                  'REFERENCE_RECHECK_PIN_DRIFT:' + str(path))
            prior = dict(metadata=actual, decoded=json.loads(data) if decode else None)
            cache[key] = prior
        elif decode and prior['decoded'] is None:
            prior['decoded'] = _json(path)
        return prior['decoded']
    census = verify_cached(proof['source_census'], decode=True)
    for pin in proof['production_context'].values():
        verify_cached(pin)
    for pin in [proof['reader_validation']['result'],
                proof['reader_validation']['storage_binding'],
                proof['reader_validation']['verified_inventory']]:
        verify_cached(pin)
    eligible_paths = set(census['eligible_paths'])
    outputs = _owned(runtime, proof['exact_excluded_output_roots'])
    for row in census['sources']:
        guard()
        _need(_metadata(row['path']) == row['metadata'],
              'REFERENCE_SOURCE_METADATA_CHANGED:' + row['path'])
        if not metadata_only:
            _need(row['source_read'] and row['sha256'] is not None,
                  'UNSCANNED_REFERENCE_SOURCE_CANNOT_AUTHORIZE_DELETE')
            data, _ = _bytes(row['path'], row['metadata'])
            _need(_sha(data) == row['sha256'], 'REFERENCE_SOURCE_SHA_CHANGED:' + row['path'])
    for alias in census['aliases']:
        guard()
        p = Path(alias['path'])
        _need(p.is_symlink() and _metadata(p) == alias['metadata'] and
              os.readlink(p) == alias['target'] and str(p.resolve()) == alias['resolved'],
              'REFERENCE_ALIAS_CHANGED:' + str(p))
    for row in census['directories']:
        guard()
        p = Path(row['path'])
        current = _metadata(p)
        _need(current['device'] == row['metadata']['device'] and
              current['inode'] == row['metadata']['inode'] and stat.S_ISDIR(current['mode']),
              'REFERENCE_DIRECTORY_IDENTITY_CHANGED:' + str(p))
        if current != row['metadata'] or not metadata_only:
            children = _directory_entries(p, eligible_paths, outputs)
            _need(_sha(_canonical(children)) == row['stable_entries_sha256'],
                  'NEW_OR_REMOVED_REFERENCE_DIRECTORY_ENTRY:' + str(p))
    for row in census['unknown']:
        guard()
        _need(_metadata(row['path']) == row['metadata'], 'UNCLASSIFIED_SOURCE_CHANGED')
    if not metadata_only:
        fresh = _census(runtime, eligible_paths, outputs, guard)
        _need([r['path'] for r in fresh['sources']] == [r['path'] for r in census['sources']],
              'NEW_OR_REMOVED_TEXT_REFERENCE_SOURCE')
        _need(fresh['aliases'] == census['aliases'], 'REFERENCE_ALIAS_SET_CHANGED')
        _need([r['path'] for r in fresh['directories']] == [r['path'] for r in census['directories']],
              'REFERENCE_DIRECTORY_SET_CHANGED')
        _need(fresh['unknown'] == census['unknown'], 'UNCLASSIFIED_REFERENCE_SCOPE_CHANGED')
        for pin in proof['reference_partitions'] + proof['alias_partitions']:
            guard()
            _verify(pin)
    return dict(status='PASSED_REFERENCE_METADATA_RECHECK' if metadata_only else 'PASSED_FULL_REFERENCE_RECHECK',
                sources=len(census['sources']), directories=len(census['directories']),
                aliases=len(census['aliases']), metadata_only=metadata_only,
                new_consumer_detection='ALL_DIRECTORY_IDENTITIES_AND_EXISTING_TEXT_METADATA',
                elapsed_seconds=round(time.monotonic() - started, 3))
