"""Finite cleanup of ten explicitly audited pytest roots; never an algorithm entry."""
from __future__ import annotations

import argparse
import ast
from datetime import timedelta
import fcntl
import hashlib
import importlib.util
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
BASE = ROOT / '_run/maintenance/original199_fixture_cleanup_20261010_v1'
DEV = BASE / 'DEVELOPMENT_TASK_PACKET.json'
HELPER = ROOT / 'scripts/maintenance/project_cleanup_proof_v2.py'
LOCK = ROOT / '_run/maintenance/concurrent_cleanup_20261009/CHANNEL_LOCK'
EMPTY_CONTROL_LOCK = '_run/current/GPU_LEASE.lock'
ROOTS = tuple('.tmp/' + n for n in (
    'original199_combined_rootcheck_20261010', 'original199_raw_closure_independent_20261010_cf461d',
    'original199_reader_dev_20261010_7df819', 'original199_reader_dev_20261010_f8b442',
    'original199_reader_dev_20261010_longid_803f51', 'original199_reader_rootcheck_20261010',
    'original199_stage_rootcheck_20261010', 'original199_storage_relocation_regression_final_v1',
    'storage_navigation_terminal_tests_20261010_v2', 'storage_relocation_combined_fixture_20261010_1451'))
spec = importlib.util.spec_from_file_location('fixture_cleanup_helpers', HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def metadata(s):
    return dict(device=s.st_dev, inode=s.st_ino, mode=s.st_mode, nlink=s.st_nlink,
                size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def once(path, value):
    need(path.is_relative_to(BASE) and path.parent.resolve() == path.parent, 'OUTPUT_SCOPE')
    with path.open('xb') as stream:
        stream.write(H.canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    need(path.read_bytes() == H.canonical(value)+b'\n', 'OUTPUT_READBACK')


def parent_fd(path, directories=None, root=ROOT):
    """Open every ancestor without following symlinks, anchored at the project."""
    need(path.is_relative_to(root), 'OUTSIDE_ROOT')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current = root
    try:
        for part in path.relative_to(root).parts[:-1]:
            current /= part
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
            if directories and str(current.relative_to(root)) in directories:
                expected = directories[str(current.relative_to(root))]['metadata']
                actual = metadata(os.fstat(fd))
                need(all(actual[k] == expected[k] for k in ('device', 'inode', 'mode')), 'PARENT_CHANGED')
        return fd
    except BaseException:
        os.close(fd); raise


def payload(row, directories=None, root=ROOT):
    p = root / row['path']; fd = parent_fd(p, directories, root)
    try:
        before = metadata(os.stat(p.name, dir_fd=fd, follow_symlinks=False))
        need(before == row['metadata'], 'TARGET_DRIFT')
        if row['kind'] == 'symlink':
            data = os.readlink(p.name, dir_fd=fd)
            need(data == row['target'], 'SYMLINK_DRIFT')
        else:
            file_fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(file_fd, 'rb') as stream:
                opened=os.fstat(stream.fileno()); need(stat.S_ISREG(opened.st_mode),'NONREGULAR_OPEN')
                need(metadata(opened) == before, 'OPEN_RACE')
                data = stream.read()
                need(metadata(os.fstat(stream.fileno())) == before, 'READ_DRIFT')
            need(len(data) == before['size'] and H.sha(data) == row['sha256'], 'TARGET_SHA_DRIFT')
        need(metadata(os.stat(p.name, dir_fd=fd, follow_symlinks=False)) == before, 'READ_DRIFT')
        return data
    finally:
        os.close(fd)


def inventory(check=lambda: None, root=ROOT, roots=ROOTS):
    need(tuple(roots) == ROOTS and root.resolve() == root, 'FROZEN_ROOT_SCOPE')
    rows = []; allowed = [root / name for name in roots]; files = size = 0
    def visit(path):
        nonlocal files,size
        check(); fd = parent_fd(path, root=root)
        try:
            s = os.stat(path.name, dir_fd=fd, follow_symlinks=False); m = metadata(s)
            need(s.st_dev == root.stat().st_dev, 'EXTERNAL_DEVICE')
            name = str(path.relative_to(root))
            need(not re.search(r'(?i)^(licen[cs]e|notices?|copying|copyright|third[_-]?party)(\.|$)', path.name), 'LICENSE_ROLE')
            if stat.S_ISDIR(s.st_mode):
                child = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    need(metadata(os.fstat(child)) == m, 'DIRECTORY_RACE')
                    children = sorted(entry.name for entry in os.scandir(child))
                finally: os.close(child)
                rows.append(dict(path=name, kind='directory', metadata=m))
                for entry in children: visit(path / entry)
            elif stat.S_ISREG(s.st_mode):
                need(s.st_nlink == 1, 'HARDLINK')
                need(files+1<=10000 and size+s.st_size<=100000000,'INVENTORY_BUDGET')
                row = dict(path=name, kind='file', metadata=m)
                file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(file_fd, 'rb') as stream:
                    opened=os.fstat(stream.fileno()); need(stat.S_ISREG(opened.st_mode),'NONREGULAR_OPEN')
                    need(metadata(opened) == m, 'OPEN_RACE')
                    data = stream.read()
                need(metadata(os.stat(path.name, dir_fd=fd, follow_symlinks=False)) == m, 'READ_DRIFT')
                row['sha256'] = H.sha(data); rows.append(row)
                files += 1; size += s.st_size
            elif stat.S_ISLNK(s.st_mode):
                target = os.readlink(path.name, dir_fd=fd)
                need(path.resolve().is_relative_to(root) and any(H.beneath(path.resolve(), a) for a in allowed), 'EXTERNAL_SYMLINK')
                need(s.st_nlink == 1, 'HARDLINK')
                rows.append(dict(path=name, kind='symlink', metadata=m, target=target))
                files += 1
            else: raise RuntimeError('SPECIAL_FILE')
        finally: os.close(fd)
        need(files <= 10000 and size <= 100000000, 'INVENTORY_BUDGET')
    for path in allowed:
        need(path.resolve() == path and path.is_dir() and not path.is_symlink(), 'ROOT_ALIAS_OR_MISSING')
        visit(path)
    return dict(roots=list(roots), rows=rows, regular_files=sum(r['kind'] == 'file' for r in rows),
                symlinks=sum(r['kind'] == 'symlink' for r in rows),
                bytes=sum(r['metadata']['size'] for r in rows if r['kind'] == 'file'))


def source_values(text, path, root):
    values=list(H.source_values(text,path,root))
    if path.suffix!='.py': return values
    try: tree=ast.parse(text)
    except SyntaxError: return values
    fragments={id(node.values[-1]) for node in ast.walk(tree) if isinstance(node,ast.JoinedStr) and node.values
               and isinstance(node.values[-1],ast.Constant) and node.values[-1].value=='.tmp'
               and any(isinstance(part,ast.FormattedValue) for part in node.values[:-1])}
    fragments |= {id(keyword.value) for node in ast.walk(tree) if isinstance(node,ast.Call)
                  and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                  and node.func.value.id=='tempfile' and node.func.attr=='mkstemp'
                  for keyword in node.keywords if keyword.arg=='suffix'
                  and isinstance(keyword.value,ast.Constant) and keyword.value.value=='.tmp'}
    def suffix_literals(node):
        if isinstance(node,ast.Constant) and node.value=='.tmp': fragments.add(id(node))
        elif isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
            suffix_literals(node.left); suffix_literals(node.right)
    for function in ast.walk(tree):
        if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        arguments=[*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs]
        path_args={arg.arg for arg in arguments if isinstance(arg.annotation,ast.Name) and arg.annotation.id=='Path'}
        todo=list(function.body)
        while todo:
            node=todo.pop()
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)): continue
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='with_suffix' and node.args:
                receiver=node.func.value
                if isinstance(receiver,ast.Name) and receiver.id in path_args: suffix_literals(node.args[0])
            todo.extend(ast.iter_child_nodes(node))
    strings=[node for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
    fragment_values={node.value for node in strings if id(node) in fragments}
    actual_values={node.value for node in strings if id(node) not in fragments}|set(H.PATH_RE.findall(text))
    return [value for value in values if value not in fragment_values or value in actual_values]


def authority(check=lambda: None, root=ROOT, roots=ROOTS):
    """Verify direct current pins; freeze actual text, not historical SHA claims."""
    receipt = root / 'docs/governance/CURRENT_STATUS_RECEIPT.json'
    todo = [receipt]; pins = {}; total = 0; direct_expected = {}
    def reference(value, source):
        if not H.valid_path(value): return
        relative_xml = source.suffix.lower() in {'.launch','.xml'} and (source.parent/value).is_file()
        p = Path(value) if value.startswith('/') else (source.parent/value if value.startswith(('./','../')) or relative_xml else root/value)
        p = Path(os.path.abspath(p))
        if not p.is_relative_to(root): return
        need(not any(H.beneath(p, root/n) or (p != root and H.beneath(root/n, p)) for n in roots), 'CURRENT_REFERENCE')
        if p == root/EMPTY_CONTROL_LOCK:
            actual=p.lstat()
            need(stat.S_ISREG(actual.st_mode) and p.resolve()==p and actual.st_size==0,'EMPTY_CONTROL_LOCK_ROLE')
            todo.append(p)
        elif p.is_file():
            if H.text_path(p) or p.suffix.lower()=='.launch': todo.append(p)
            else: need(p.suffix.lower() in H.MEDIA|H.MODEL_BINARY,'UNKNOWN_CONSUMER_ROLE:'+str(p))
    while todo:
        check(); p = todo.pop()
        if str(p) in pins: continue
        data, _ = H.stable_bytes(p); total += len(data)
        if p == root/EMPTY_CONTROL_LOCK: need(data==b'','EMPTY_CONTROL_LOCK_ROLE')
        need(len(pins) < 10000 and total <= 100000000, 'AUTHORITY_BUDGET')
        pins[str(p)] = dict(path=str(p), bytes=len(data), sha256=H.sha(data))
        if str(p) in direct_expected: need(pins[str(p)]==direct_expected[str(p)],'CURRENT_DIRECT_PIN_DRIFT')
        if p.suffix != '.json': need(not any(name in data.decode() or Path(name).name in data.decode() for name in roots),'CURRENT_REFERENCE')
        value = json.loads(data) if p.suffix == '.json' else source_values(data.decode(),p,root)
        if p.suffix.lower()=='.launch': value += re.findall(r'''["']([^"'\n]+)["']''',data.decode())
        if p == receipt:
            for pin in value['files'].values():
                direct_expected[pin['path']]=pin; H.verify(pin)
                if pin['path'] in pins: need(pins[pin['path']]==pin,'CURRENT_DIRECT_PIN_DRIFT')
                reference(pin['path'], p)
        for value in H.string_values(value): reference(value, p)
    return list(pins.values())


def gates(rows, check=lambda: None, root=ROOT):
    check(); names = [r['path'] for r in rows]
    active = H.active_context(names, root=root)
    need(not active['incoming'], 'ACTIVE_REFERENCE')
    H.assert_no_process_holders(H.process_check(names, root=root, check=check))
    tracked, git = H.git_context(root=root)
    need(not (set(names) & tracked), 'GIT_TRACKED')
    ignored = subprocess.run(['git', 'check-ignore', '--stdin', '-z'], input=b'\0'.join(n.encode() for n in ROOTS)+b'\0',
                             cwd=root, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), capture_output=True, check=False)
    need(ignored.returncode == 0 and set(ignored.stdout.decode().split('\0'))-{''} == set(ROOTS), 'GIT_NOT_IGNORED')
    dirty = subprocess.run(['git','status','--porcelain=v1','-z','--untracked-files=all','--',*ROOTS],
        cwd=root,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),capture_output=True,check=False)
    need(dirty.returncode==0 and not dirty.stdout,'GIT_SCOPED_DIRTY')
    git.update(fixture_scoped_dirty_sha256=H.sha(dirty.stdout),fixture_scoped_dirty_records=dirty.stdout.count(b'\0'))
    return dict(active=active, git=git, license=H.pin(root/'THIRD_PARTY_NOTICES.md'))


def verify_archive(path, rows, check=lambda: None):
    expected = {r['path']: r for r in rows}; seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            check(); row = expected.get(member.name)
            need(row is not None and member.name not in seen, 'ARCHIVE_MEMBER_SET')
            need(member.pax_headers.get('cleanup.identity') == H.canonical(row['metadata']).decode(), 'ARCHIVE_IDENTITY')
            if row['kind'] == 'file':
                need(member.isfile() and member.size == row['metadata']['size'], 'ARCHIVE_SIZE_OR_KIND')
                need(H.sha(archive.extractfile(member).read()) == row['sha256'], 'ARCHIVE_SHA')
            elif row['kind'] == 'symlink': need(member.issym() and member.linkname == row['target'], 'ARCHIVE_LINK')
            else: need(member.isdir(), 'ARCHIVE_DIRECTORY')
            seen.add(member.name)
    need(seen == set(expected), 'ARCHIVE_MEMBER_SET')
    return dict(archive=H.pin(path), members=len(seen), verification='EVERY_MEMBER_SHA_TARGET_AND_IDENTITY')


def backup(path, rows, check=lambda: None, root=ROOT):
    directories = {r['path']:r for r in rows if r['kind'] == 'directory'}
    with path.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w|gz', format=tarfile.PAX_FORMAT) as archive:
            for row in rows:
                check(); member = tarfile.TarInfo(row['path']); member.mode = stat.S_IMODE(row['metadata']['mode'])
                member.pax_headers = {'cleanup.identity': H.canonical(row['metadata']).decode()}
                if row['kind'] == 'file':
                    data = payload(row, directories, root); member.size = len(data); archive.addfile(member, io.BytesIO(data))
                else:
                    member.type = tarfile.SYMTYPE if row['kind'] == 'symlink' else tarfile.DIRTYPE
                    if row['kind'] == 'symlink': member.linkname = payload(row, directories, root)
                    archive.addfile(member)
        stream.flush(); os.fsync(stream.fileno())
    return verify_archive(path, rows, check)


def exact_delete(rows, journal, progress, check, root=ROOT, final_check=None):
    directories = {r['path']:r for r in rows if r['kind'] == 'directory'}
    ordered = [r for r in rows if r['kind'] != 'directory'] + sorted(directories.values(), key=lambda r: len(Path(r['path']).parts), reverse=True)
    with journal.open('xb') as stream:
        for row in ordered:
            check(); p = root / row['path']; fd = parent_fd(p, directories, root)
            try:
                if row['kind'] != 'directory': payload(row, directories, root)
                actual = metadata(os.stat(p.name, dir_fd=fd, follow_symlinks=False))
                keys = ('device','inode','mode') if row['kind'] == 'directory' else tuple(row['metadata'])
                need(all(actual[k] == row['metadata'][k] for k in keys), 'LAST_MOMENT_DRIFT')
                def event(state):
                    stream.write(H.canonical(dict(event=state, path=row['path'], kind=row['kind'], metadata=row['metadata']))+b'\n')
                    stream.flush(); os.fsync(stream.fileno())
                event('INTENT'); (final_check or check)()
                final = metadata(os.stat(p.name, dir_fd=fd, follow_symlinks=False))
                need(all(final[k] == row['metadata'][k] for k in keys), 'LAST_MOMENT_DRIFT')
                if row['kind'] == 'directory': os.rmdir(p.name, dir_fd=fd)
                else: os.unlink(p.name, dir_fd=fd)
                progress[row['kind']] += 1
                if row['kind'] == 'file': progress['bytes'] += row['metadata']['size']
                os.fsync(fd); event('CONFIRMED')
            finally: os.close(fd)


class Runtime:
    def __init__(self, delete, authorization=None):
        self.started = time.monotonic(); self.owned = False
        self.inputs = [H.pin(DEV), H.pin(HELPER), H.pin(Path(__file__).absolute()), H.pin(H.PROTOCOL)]
        need(H.load(DEV)['candidate_roots'] == list(ROOTS), 'DEVELOPMENT_ROOT_DRIFT')
        self.progress = dict(file=0, symlink=0, directory=0, bytes=0); self.recovery = None
        need(BASE.resolve() == BASE and LOCK.resolve() == LOCK, 'LOCK_ALIAS'); self.lock = LOCK.open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB); self.owned = True
        self.lock.seek(0); old = self.lock.read(); pid = int(Path('/proc/self/stat').read_text().split(' ',1)[0])
        live = H.identity(pid); need(live is not None, 'HOST_IDENTITY_REQUIRED')
        self.writer = dict(**live, executor_epoch=int(json.loads(old).get('executor_epoch',0))+1 if old else 1,
            fencing_token=secrets.token_hex(24), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            deadline_at=(H.now()+timedelta(seconds=1200)).isoformat())
        s = os.fstat(self.lock.fileno()); self.writer.update(lock_device=s.st_dev, lock_inode=s.st_ino)
        self.lock.seek(0); self.lock.truncate(); json.dump(self.writer,self.lock); H.read_back_private_lock(self.lock,LOCK,self.writer)
        self.out = BASE / (('C1_' if delete else 'C0_')+H.now().strftime('%H%M%S')+'_'+self.writer['fencing_token'][:8])
        need(self.out.parent.resolve() == BASE, 'OUTPUT_ALIAS'); self.out.mkdir()
        self.data = inventory(self.guard)
        if delete:
            need(authorization is not None and authorization.is_relative_to(BASE) and authorization.resolve()==authorization, 'AUTHORIZATION_REQUIRED')
            self.inputs.append(H.pin(authorization)); auth = H.load(authorization)
            need(auth.get('execution_allowed') is True and auth.get('deletion_allowed') is True, 'DELETE_NOT_AUTHORIZED')
            H.verify(auth['inventory']); need(H.load(auth['inventory']['path']) == self.data, 'FROZEN_INVENTORY_DRIFT')
            need(auth.get('development_packet') == self.inputs[0], 'DEVELOPMENT_PIN_DRIFT')
            need(set(auth['fixture_origins']) == set(ROOTS), 'FIXTURE_ORIGIN_SET')
            for origin in auth['fixture_origins'].values():
                need(origin.get('kind') == 'COMPLETED_PYTEST_TMP' and origin.get('source_tests'), 'FIXTURE_ORIGIN_MISSING')
                for pin in origin['source_tests']:
                    need(Path(pin['path']).is_relative_to(ROOT/'tests/storage_relocation') and Path(pin['path']).suffix == '.py', 'SOURCE_TEST_SCOPE')
                    H.verify(pin); self.inputs.append(pin)
            self.inputs.append(auth['inventory'])
        once(self.out/'INVENTORY.json',self.data)
        packet = dict(schema='original199-fixture-finite-cleanup-v1', writer=self.writer, inputs=self.inputs,
            execution_allowed=delete, deletion_allowed=delete, algorithm_execution_allowed=False, production_governance_write_allowed=False,
            write_set=[str(self.out),str(LOCK)]+([str(ROOT/r['path']) for r in self.data['rows']] if delete else []),
            inventory=H.pin(self.out/'INVENTORY.json'), budgets=dict(wall_seconds=1200,cpu_threads=2,gpu=0,max_files=10000,max_bytes=100000000),
            run_signature=H.sha(H.canonical(dict(inputs=self.inputs,inventory=self.data,delete=delete))))
        once(self.out/'TASK_PACKET.json',packet); self.inputs.append(H.pin(self.out/'TASK_PACKET.json'))

    def guard(self):
        need(time.monotonic()-self.started < 1200, 'FINITE_DEADLINE')
        live = H.identity(self.writer['pid']); need(live and live['startticks'] == self.writer['startticks'], 'WRITER_DRIFT')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == self.writer['boot_id'], 'BOOT_DRIFT')
        s = LOCK.lstat(); need((s.st_dev,s.st_ino)==(self.writer['lock_device'],self.writer['lock_inode']) and H.load(LOCK)==self.writer, 'LOCK_OR_FENCE_DRIFT')

    def execute(self):
        pins = authority(self.guard); context = gates(self.data['rows'], self.guard)
        self.inputs.extend(pins+[context['license']]); once(self.out/'PROOF.json',dict(authority=pins,context=context,
            authority_time_semantics='DIRECT_CURRENT_RECEIPT_PINS_VERIFIED_ACTUAL_TEXT_FROZEN_HISTORICAL_NESTED_PIN_CLAIMS_NOT_VERIFIED',
            retained_roles=[dict(role='EMPTY_CONTROL_LOCK',input=pin) for pin in pins if pin['path']==str(ROOT/EMPTY_CONTROL_LOCK)],
            inventory=H.pin(self.out/'INVENTORY.json')))
        self.recovery = backup(self.out/'RECOVERY.tar.gz',self.data['rows'],self.guard)
        once(self.out/'RECOVERY_VERIFIED.json',self.recovery); self.inputs.append(self.recovery['archive'])
        need(inventory(self.guard)==self.data,'POST_BACKUP_INVENTORY_DRIFT')
        unlink_packet = dict(schema='original199-fixture-unlink-packet-v1', parent=H.pin(self.out/'TASK_PACKET.json'),
            writer=self.writer, inputs=self.inputs, proof=H.pin(self.out/'PROOF.json'), recovery=self.recovery,
            execution_allowed=True,deletion_allowed=True,algorithm_execution_allowed=False,production_governance_write_allowed=False,
            write_set=[str(ROOT/r['path']) for r in self.data['rows']],
            budgets=dict(wall_seconds=1200,cpu_threads=2,gpu=0,max_files=10000,max_bytes=100000000),
            run_signature=H.sha(H.canonical(dict(inputs=self.inputs,inventory=self.data,recovery=self.recovery))))
        once(self.out/'UNLINK_TASK_PACKET.json',unlink_packet); self.inputs.append(H.pin(self.out/'UNLINK_TASK_PACKET.json'))
        observations = {}; last_inputs = last_gates = 0
        def recheck(full=True):
            nonlocal last_inputs,last_gates
            self.guard()
            current_time = time.monotonic()
            if current_time-last_inputs >= 5:
                for pin in self.inputs:
                    current = metadata(Path(pin['path']).lstat())
                    if observations.get(pin['path']) != current:
                        H.verify(pin); observations[pin['path']] = current
                last_inputs = current_time
            if full and current_time-last_gates >= 60:
                gates(self.data['rows'], self.guard)
                last_gates = current_time
        exact_delete(self.data['rows'],self.out/'DELETED.jsonl',self.progress,recheck,final_check=lambda:recheck(False))
        self.guard()
        for pin in self.inputs: H.verify(pin)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--delete',action='store_true'); parser.add_argument('--authorization',type=Path)
    args=parser.parse_args(); runtime=Runtime.__new__(Runtime)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='2'
    os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['TMPDIR']=str(ROOT/'.tmp')
    try:
        runtime.__init__(args.delete,args.authorization)
        if args.delete: runtime.execute()
        result=dict(status='PASSED_DELETE' if args.delete else 'PASSED_INVENTORY')
    except Exception as exc:
        result=dict(status='PARTIAL_STOPPED' if sum(getattr(runtime,'progress',{}).values()) else 'STOPPED',failure=repr(exc))
    try:
        if hasattr(runtime,'out'):
            result.update(progress=runtime.progress,recovery=runtime.recovery,writer=runtime.writer,
                task_packet=H.pin(runtime.out/'TASK_PACKET.json') if (runtime.out/'TASK_PACKET.json').exists() else None)
            if (runtime.out/'UNLINK_TASK_PACKET.json').exists(): result['unlink_task_packet']=H.pin(runtime.out/'UNLINK_TASK_PACKET.json')
            if (runtime.out/'DELETED.jsonl').exists(): result['journal']=H.pin(runtime.out/'DELETED.jsonl')
            once(runtime.out/'RESULT.json',result); result['output']=str(runtime.out)
    finally:
        if hasattr(runtime,'lock'):
            if runtime.owned: fcntl.flock(runtime.lock,fcntl.LOCK_UN)
            runtime.lock.close()
    print(json.dumps(result)); return 0 if result['status'].startswith('PASSED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
