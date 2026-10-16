"""Standard-library-only guards for immutable 0916 baseline replay."""
from pathlib import Path
import hashlib
import json
import os
import stat


def read_verified(ref):
    path=Path(ref['path'])
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode): raise ValueError('NOT_REGULAR')
        payload=stream.read()
        after=os.fstat(stream.fileno())
    current=path.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or identity(after)!=identity(current):
        raise ValueError('UNSTABLE_INPUT')
    if len(payload)!=ref['bytes'] or hashlib.sha256(payload).hexdigest()!=ref['sha256']:
        raise ValueError('REFERENCE_BYTES_SHA_MISMATCH:'+str(path))
    return payload


def validate_output(output, allowed_root):
    path=Path(output); root=Path(allowed_root)
    if not path.is_absolute() or '..' in path.parts or not root.is_absolute(): raise ValueError('ABSOLUTE_NORMALIZED_PATH_REQUIRED')
    if root.is_symlink() or root.resolve()!=root: raise ValueError('ALLOW_ROOT_SYMLINK')
    if path==root or not path.is_relative_to(root): raise ValueError('OUTPUT_SCOPE_ESCAPE')
    for component in (path,*path.parents):
        if component.is_symlink(): raise ValueError('OUTPUT_SYMLINK')
    if path.exists(): raise FileExistsError('FRESH_OUTPUT_REQUIRED:'+str(path))
    if not path.parent.is_dir(): raise ValueError('REGISTER_OUTPUT_PARENT_FIRST')
    return path


def claim_output(output,allowed_root):
    path=validate_output(output,allowed_root)
    # Atomic leaf reservation, never exist_ok; loser cannot touch winner's receipts.
    path.mkdir(mode=0o700)
    return path


def require_task(repo, task_id, output, operation, command_args=()):
    repo=Path(repo).resolve()
    index=json.loads((repo/'tasks/current/INDEX.json').read_text(encoding='utf8'))
    entries=[x for x in index['task_packets'] if x.get('execution_allowed') is True]
    if len(entries)!=1 or entries[0].get('task_id')!=task_id: raise ValueError('TASK_NOT_EXECUTABLE')
    entry=entries[0]; path=repo/entry['packet_path']
    if path.resolve()!=repo/'tasks/current'/task_id/'TASK_PACKET.json': raise ValueError('PACKET_PATH_SCOPE')
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=entry['packet_sha256']: raise ValueError('PACKET_SHA')
    packet=json.loads(payload)
    if packet['task_id']!=task_id: raise ValueError('PACKET_ID')
    import shlex
    expected=['chaoyang','run',operation,*command_args]
    if not any(shlex.split(s)==expected for s in packet.get('commands',[])): raise ValueError('OPERATION_NOT_DELEGATED')
    allowed=[Path(p) if Path(p).is_absolute() else repo/p for p in packet['write_set']]
    if not any(Path(output)==p or Path(output).is_relative_to(p) for p in allowed): raise ValueError('OUTPUT_NOT_IN_TASK')
    from datetime import datetime
    deadline=datetime.fromisoformat(packet['deadline_at'])
    if deadline.tzinfo is None or datetime.now(deadline.tzinfo)>=deadline: raise ValueError('TASK_EXPIRED')
    writer=packet['writer'];pid=int(writer['pid'])
    proc=Path('/proc')/str(pid)/'stat'
    if not proc.is_file(): raise ValueError('WRITER_NOT_LIVE')
    ticks=int(proc.read_text().split(') ',1)[1].split()[19])
    if ticks!=int(writer['proc_start_ticks']) or not writer.get('fencing_token'): raise ValueError('WRITER_IDENTITY')
    if writer.get('scope')!=task_id: raise ValueError('WRITER_SCOPE')
    lease_path=Path(packet['writer_lease_path'])
    if not lease_path.is_absolute() or not lease_path.resolve().is_relative_to(repo/'_run/current'):
        raise ValueError('WRITER_LEASE_SCOPE')
    lease=json.loads(lease_path.read_text(encoding='utf8'))
    for key in ('pid','proc_start_ticks','executor_epoch','fencing_token','scope'):
        if lease.get(key)!=writer.get(key): raise ValueError('WRITER_FENCE_MISMATCH:'+key)
    state=json.loads((repo/'docs/governance/LONG_HORIZON_TASK_STATE.json').read_text(encoding='utf8'))
    if (state.get('next_task') or {}).get('task_id')!=task_id: raise ValueError('TASK_NOT_CURRENT')
    rows=[r for r in state['tasks'] if r.get('task_id')==task_id]
    if len(rows)!=1 or rows[0].get('status')!='RUNNING': raise ValueError('TASK_STATE_NOT_RUNNING')
    row=rows[0]
    if any(row.get(k)!=packet.get(k) for k in ('run_signature',)) or any(row.get(k)!=writer.get(k) for k in ('pid','proc_start_ticks')):
        raise ValueError('TASK_STATE_IDENTITY')
    if row.get('task_packet',{}).get('sha256')!=entry['packet_sha256']: raise ValueError('TASK_STATE_PACKET')
    caller=os.getpid();seen=set()
    while caller>1 and caller not in seen and caller!=pid:
        seen.add(caller)
        value=(Path('/proc')/str(caller)/'stat').read_text().split(') ',1)[1].split()
        caller=int(value[1])
    if caller!=pid: raise ValueError('CALLER_NOT_WRITER_DESCENDANT')
    return packet


def require_manifest(packet, path):
    ref=packet.get('inputs',{}).get('manifest')
    if not ref or Path(ref['path'])!=Path(path): raise ValueError('MANIFEST_NOT_PINNED_BY_TASK')
    return json.loads(read_verified(ref))


def require_environment(environment):
    import importlib.metadata
    import sys
    if sys.version!=environment['python']: raise ValueError('PYTHON_VERSION_DRIFT')
    for name,version in environment['packages'].items():
        if version=='NOT_DISTRIBUTION_REGISTERED': raise ValueError('DEPENDENCY_VERSION_UNVERIFIED:'+name)
        if importlib.metadata.version(name)!=version: raise ValueError('DEPENDENCY_VERSION_DRIFT:'+name)
