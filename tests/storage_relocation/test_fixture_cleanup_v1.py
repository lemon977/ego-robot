"""Bounded cleanup gates exercised only on tiny project-local pytest fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest


@pytest.fixture
def cleanup(monkeypatch):
    path = Path(__file__).parents[2]/'scripts/maintenance/cleanup_original199_test_fixtures_v1.py'
    spec = importlib.util.spec_from_file_location('fixture_cleanup_test',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOTS',('.tmp/one','.tmp/two'))
    return module


@pytest.fixture
def project(tmp_path,cleanup):
    root=tmp_path/'project'; root.mkdir()
    for name in cleanup.ROOTS: (root/name).mkdir(parents=True)
    (root/'.tmp/one/data').write_bytes(b'fixture bytes')
    (root/'.tmp/two/link').symlink_to(root/'.tmp/one/data')
    return root


def frozen(cleanup,project):
    return cleanup.inventory(root=project,roots=cleanup.ROOTS)


def test_exact_backup_and_delete_only_fixture_entries(cleanup,project,tmp_path):
    data=frozen(cleanup,project); archive=tmp_path/'recovery.tar.gz'
    proof=cleanup.backup(archive,data['rows'],root=project)
    assert proof['members']==4
    assert cleanup.verify_archive(archive,data['rows'])==proof
    retained=project/'retained.py'; retained.write_text('# preserve source')
    progress=dict(file=0,symlink=0,directory=0,bytes=0)
    cleanup.exact_delete(data['rows'],tmp_path/'journal.jsonl',progress,lambda:None,root=project)
    assert progress==dict(file=1,symlink=1,directory=2,bytes=13)
    assert retained.exists() and archive.exists()
    events=[json.loads(line) for line in (tmp_path/'journal.jsonl').read_text().splitlines()]
    assert [row['event'] for row in events]==['INTENT','CONFIRMED']*4


def test_root_alias_stops(cleanup,project):
    (project/'.tmp/two/link').unlink(); (project/'.tmp/two').rmdir()
    (project/'.tmp/two').symlink_to(project/'.tmp/one',target_is_directory=True)
    with pytest.raises(RuntimeError,match='ROOT_ALIAS'): frozen(cleanup,project)


def test_outside_candidate_scope_stops(cleanup,project):
    with pytest.raises(RuntimeError,match='FROZEN_ROOT_SCOPE'):
        cleanup.inventory(root=project,roots=('../outside',))


def test_external_symlink_stops(cleanup,project,tmp_path):
    outside=tmp_path/'outside'; outside.write_bytes(b'preserved')
    (project/'.tmp/two/link').unlink(); (project/'.tmp/two/link').symlink_to(outside)
    with pytest.raises(RuntimeError,match='EXTERNAL_SYMLINK'): frozen(cleanup,project)
    assert outside.read_bytes()==b'preserved'


def test_hardlink_stops(cleanup,project):
    os.link(project/'.tmp/one/data',project/'.tmp/one/hardlink')
    with pytest.raises(RuntimeError,match='HARDLINK'): frozen(cleanup,project)


def test_special_file_stops(cleanup,project):
    os.mkfifo(project/'.tmp/one/fifo')
    with pytest.raises(RuntimeError,match='SPECIAL_FILE'): frozen(cleanup,project)


def test_license_is_preserved(cleanup,project):
    (project/'.tmp/one/LICENSE').write_text('retain')
    with pytest.raises(RuntimeError,match='LICENSE_ROLE'): frozen(cleanup,project)


@pytest.mark.parametrize('mutation',['sha','link','identity','missing'])
def test_archive_mismatch_blocks_delete(cleanup,project,tmp_path,mutation):
    data=frozen(cleanup,project); archive=tmp_path/'backup.tar.gz'
    cleanup.backup(archive,data['rows'],root=project)
    if mutation=='sha': next(r for r in data['rows'] if r['kind']=='file')['sha256']='0'*64
    elif mutation=='link': next(r for r in data['rows'] if r['kind']=='symlink')['target']='wrong'
    elif mutation=='identity': data['rows'][0]['metadata']['inode']+=1
    else: data['rows'].pop()
    with pytest.raises(RuntimeError,match='ARCHIVE_'): cleanup.verify_archive(archive,data['rows'])
    assert (project/'.tmp/one/data').exists()


def test_current_authority_reference_blocks(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    contract=folder/'contract.json'; contract.write_text(json.dumps({'inputs':[str(project/'.tmp/one/data')]}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


def test_authority_follows_pinned_json_closure_and_ignores_external_bytes(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    second=folder/'second.json'; second.write_text(json.dumps({'external':'/path/to/private_storage'}))
    contract=folder/'contract.json'; contract.write_text(json.dumps({'child':cleanup.H.pin(second)}))
    receipt=folder/'CURRENT_STATUS_RECEIPT.json'; receipt.write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    pins=cleanup.authority(root=project,roots=cleanup.ROOTS)
    assert {p['path'] for p in pins}=={str(receipt),str(contract),str(second)}


@pytest.mark.parametrize('blocked',['git','holders','active','not_ignored','dirty'])
def test_runtime_positive_gates_reject(cleanup,project,monkeypatch,blocked):
    rows=frozen(cleanup,project)['rows']; names=[r['path'] for r in rows]
    monkeypatch.setattr(cleanup.H,'active_context',lambda *a,**k:dict(incoming={names[0]:['active']} if blocked=='active' else {}))
    monkeypatch.setattr(cleanup.H,'process_check',lambda *a,**k:dict(errors=[],hits={names[0]:[1]} if blocked=='holders' else {}))
    monkeypatch.setattr(cleanup.H,'git_context',lambda **k:(set(names) if blocked=='git' else set(),{}))
    result=type('Ignored',(),dict(returncode=1 if blocked=='not_ignored' else 0,
        stdout=('\0'.join(cleanup.ROOTS)+'\0').encode()))()
    def run(args,**kwargs):
        if args[1]=='status': return type('Dirty',(),dict(returncode=0,stdout=b'?? fixture\0' if blocked=='dirty' else b''))()
        return result
    monkeypatch.setattr(cleanup.subprocess,'run',run)
    with pytest.raises((RuntimeError,cleanup.H.ProofError)):
        cleanup.gates(rows,root=project)
    assert (project/'.tmp/one/data').exists()


def test_deadline_stops_before_any_unlink(cleanup,project,tmp_path):
    rows=frozen(cleanup,project)['rows']; runtime=cleanup.Runtime.__new__(cleanup.Runtime)
    runtime.started=time.monotonic()-1201; progress=dict(file=0,symlink=0,directory=0,bytes=0)
    with pytest.raises(RuntimeError,match='FINITE_DEADLINE'):
        cleanup.exact_delete(rows,tmp_path/'journal.jsonl',progress,runtime.guard,root=project)
    assert progress['file']==0 and (project/'.tmp/one/data').exists()


def test_last_moment_sha_drift_keeps_target(cleanup,project,tmp_path):
    rows=frozen(cleanup,project)['rows']; (project/'.tmp/one/data').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='TARGET_DRIFT'):
        cleanup.exact_delete(rows,tmp_path/'journal.jsonl',dict(file=0,symlink=0,directory=0,bytes=0),lambda:None,root=project)
    assert (project/'.tmp/one/data').read_bytes()==b'changed'


def test_parent_replaced_with_symlink_is_not_followed(cleanup,project,tmp_path):
    rows=frozen(cleanup,project)['rows']; old=project/'.tmp/one'; retained=project/'.tmp/retained'
    old.rename(retained); old.symlink_to(retained,target_is_directory=True)
    with pytest.raises(OSError):
        cleanup.exact_delete(rows,tmp_path/'journal.jsonl',dict(file=0,symlink=0,directory=0,bytes=0),lambda:None,root=project)
    assert (retained/'data').exists()


def test_backup_failure_never_reaches_unlink(cleanup,project,tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'BASE',tmp_path)
    runtime=cleanup.Runtime.__new__(cleanup.Runtime); runtime.out=tmp_path/'attempt'; runtime.out.mkdir()
    runtime.data=frozen(cleanup,project); runtime.inputs=[]; runtime.guard=lambda:None
    (runtime.out/'INVENTORY.json').write_text(json.dumps(runtime.data))
    license_file=tmp_path/'LICENSE'; license_file.write_text('preserved')
    monkeypatch.setattr(cleanup,'authority',lambda *a:[])
    monkeypatch.setattr(cleanup,'gates',lambda *a:dict(license=cleanup.H.pin(license_file)))
    def reject(*a): raise RuntimeError('ARCHIVE_SHA')
    monkeypatch.setattr(cleanup,'backup',reject)
    reached=[]; monkeypatch.setattr(cleanup,'exact_delete',lambda *a:reached.append(True))
    with pytest.raises(RuntimeError,match='ARCHIVE_SHA'): runtime.execute()
    assert not reached and (project/'.tmp/one/data').exists()


@pytest.mark.parametrize('partial',[False,True])
def test_terminal_failure_receipt_preserves_journal(cleanup,tmp_path,monkeypatch,partial):
    monkeypatch.setattr(cleanup,'BASE',tmp_path)
    class FakeRuntime:
        def __init__(self,*a):
            self.out=tmp_path/'attempt'; self.out.mkdir(); self.writer=dict(pid=1,startticks=2)
            self.progress=dict(file=int(partial),symlink=0,directory=0,bytes=int(partial))
            self.recovery=dict(archive={'retained':True}); self.lock=(tmp_path/'lock').open('a+'); self.owned=False
            (self.out/'TASK_PACKET.json').write_text('{}')
            (self.out/'DELETED.jsonl').write_text('{"event":"INTENT"}\n')
            if not partial: raise RuntimeError('INITIALIZATION_STOP')
        def execute(self): raise RuntimeError('AFTER_UNLINK_STOP')
    monkeypatch.setattr(cleanup,'Runtime',FakeRuntime)
    monkeypatch.setattr('sys.argv',['cleanup','--delete','--authorization',str(tmp_path/'auth.json')])
    assert cleanup.main()==2
    result=json.loads((tmp_path/'attempt/RESULT.json').read_bytes())
    assert result['status']==('PARTIAL_STOPPED' if partial else 'STOPPED')
    assert result['journal']['path']==str(tmp_path/'attempt/DELETED.jsonl')
    assert result['recovery']['archive']['retained']


@pytest.mark.parametrize('extension',['py','yaml','toml'])
def test_registered_text_consumer_reference_is_checked(cleanup,project,extension):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/('consumer.'+extension)
    consumer.write_text("input = '.tmp/one/data'\n")
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


def test_full_holder_check_runs_before_parent_fd_opens(cleanup,project,tmp_path):
    rows=frozen(cleanup,project)['rows']; checked=[]; final=[]
    def holder_gate():
        for fd in Path('/proc/self/fd').iterdir():
            try: target=os.readlink(fd)
            except FileNotFoundError: continue
            assert target!=str(project/'.tmp')
            assert not any(cleanup.H.beneath(target,project/name) for name in cleanup.ROOTS)
        checked.append(True)
    cleanup.exact_delete(rows,tmp_path/'journal.jsonl',dict(file=0,symlink=0,directory=0,bytes=0),
        holder_gate,root=project,final_check=lambda:final.append(True))
    assert len(checked)==len(final)==len(rows)


@pytest.mark.parametrize('operation',['inventory','payload'])
def test_fifo_swap_at_open_fails_without_blocking(cleanup,project,monkeypatch,operation):
    row=next(r for r in frozen(cleanup,project)['rows'] if r['kind']=='file')
    original_open=cleanup.os.open; swapped=[]
    def racing_open(path,flags,*args,**kwargs):
        if path=='data' and flags & os.O_NONBLOCK and not swapped:
            (project/'.tmp/one/data').unlink(); os.mkfifo(project/'.tmp/one/data'); swapped.append(True)
        return original_open(path,flags,*args,**kwargs)
    monkeypatch.setattr(cleanup.os,'open',racing_open)
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='NONREGULAR_OPEN'):
        if operation=='inventory': frozen(cleanup,project)
        else: cleanup.payload(row,root=project)
    assert swapped and time.monotonic()-started<2


def test_split_fixture_basename_in_registered_code_stops(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'; consumer.write_text("fixture_root = '.tmp/' + 'one'\n")
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('referenced',[False,True])
def test_historical_index_multiple_sha_claims_check_actual_text(cleanup,project,referenced):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    index=project/'tasks/current/INDEX.json'; index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'input':str(project/'.tmp/one/data') if referenced else 'no fixture input'}))
    claims=[dict(path=str(index),bytes=1,sha256='0'*64),dict(path=str(index),bytes=2,sha256='1'*64)]
    contract=folder/'contract.json'; contract.write_text(json.dumps({'historical_snapshots':claims}))
    receipt=folder/'CURRENT_STATUS_RECEIPT.json'
    receipt.write_text(json.dumps({'files':{'index':cleanup.H.pin(index),'contract':cleanup.H.pin(contract)}}))
    if referenced:
        with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'): cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        pins=cleanup.authority(root=project,roots=cleanup.ROOTS)
        assert next(pin for pin in pins if pin['path']==str(index))==cleanup.H.pin(index)
        assert all(pin not in pins for pin in claims)


def test_direct_current_receipt_sha_remains_strict(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    contract=folder/'contract.json'; contract.write_text('{}')
    receipt=folder/'CURRENT_STATUS_RECEIPT.json'; receipt.write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    contract.write_text('{"changed":true}')
    with pytest.raises(cleanup.H.ProofError,match='SIGNED_INPUT_DRIFT'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('referenced',[False,True])
def test_launch_is_local_registered_text_consumer(cleanup,project,referenced):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    launch=project/'robot.launch'
    launch.write_text('<launch><param value="'+('.tmp/one/data' if referenced else '/robot/status')+'"/></launch>')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(launch)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    if referenced:
        with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'): cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        assert cleanup.H.pin(launch) in cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('referenced',[False,True])
def test_launch_relative_xml_attribute_follows_text_config(cleanup,project,referenced):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    runtime=project/'src/runtime'; runtime.mkdir(parents=True)
    config=runtime/'robot.yaml'; config.write_text('input: '+('.tmp/one/data' if referenced else '/robot/status'))
    launch=runtime/'robot.launch'; launch.write_text('<launch><include file="robot.yaml"/></launch>')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(launch)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    if referenced:
        with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'): cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        assert cleanup.H.pin(config) in cleanup.authority(root=project,roots=cleanup.ROOTS)


def test_unknown_registered_consumer_role_still_stops(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'unknown.opaque'; consumer.write_bytes(b'small unknown role')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='UNKNOWN_CONSUMER_ROLE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


def test_direct_pin_verify_then_read_race_stops(cleanup,project,monkeypatch):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    contract=folder/'contract.json'; contract.write_text('{}')
    expected=cleanup.H.pin(contract)
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':expected}}))
    original_verify=cleanup.H.verify; changed=[]
    def verify_then_change(pin):
        original_verify(pin)
        if pin['path']==str(contract) and not changed:
            contract.write_text('{"changed_after_verification":true}'); changed.append(True)
    monkeypatch.setattr(cleanup.H,'verify',verify_then_change)
    with pytest.raises(RuntimeError,match='CURRENT_DIRECT_PIN_DRIFT'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)
    assert changed


def test_fstring_suffix_fragment_is_not_tmp_directory_reference(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'
    consumer.write_text('from pathlib import Path\nimport os\npath=Path("report.json")\n'
                        'temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")\n')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    assert cleanup.H.pin(consumer) in cleanup.authority(root=project,roots=cleanup.ROOTS)


def test_fstring_expression_path_tmp_still_blocks(cleanup,project):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'
    consumer.write_text('from pathlib import Path\ntemporary=f"{Path(\'.tmp\')}/payload"\n')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('expression',["f'.tmp/{session}/actions.npz'","f'.tmp'"])
def test_fstring_tmp_prefix_or_complete_directory_still_blocks(cleanup,project,expression):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'; consumer.write_text('temporary='+expression+'\n')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('content',[b'',b'written'])
def test_exact_gpu_control_lock_role(cleanup,project,content):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    lock=project/'_run/current/GPU_LEASE.lock'; lock.parent.mkdir(parents=True); lock.write_bytes(content)
    consumer=project/'consumer.py'; consumer.write_text('gpu_lock = '+repr(str(lock))+'\n')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    if content:
        with pytest.raises(RuntimeError,match='EMPTY_CONTROL_LOCK_ROLE'):
            cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        assert cleanup.H.pin(lock) in cleanup.authority(root=project,roots=cleanup.ROOTS)
    assert lock.read_bytes()==content


@pytest.mark.parametrize('kind',['directory','fifo'])
def test_exact_gpu_control_lock_requires_regular_file(cleanup,project,kind):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    lock=project/'_run/current/GPU_LEASE.lock'; lock.parent.mkdir(parents=True)
    if kind=='directory': lock.mkdir()
    else: os.mkfifo(lock)
    consumer=project/'consumer.py'; consumer.write_text('gpu_lock = '+repr(str(lock))+'\n')
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    with pytest.raises(RuntimeError,match='EMPTY_CONTROL_LOCK_ROLE'):
        cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('real_path',[False,True])
def test_exact_mkstemp_suffix_fragment_keeps_real_tmp_path(cleanup,project,real_path):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'
    code='import tempfile\nfrom pathlib import Path\npath=Path("report.json")\n'
    code+='fd, temporary=tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)\n'
    if real_path: code+='actual_input=Path(".tmp")\n'
    consumer.write_text(code)
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    if real_path:
        with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'): cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        assert cleanup.H.pin(consumer) in cleanup.authority(root=project,roots=cleanup.ROOTS)


@pytest.mark.parametrize('real_path',[False,True])
def test_annotated_path_with_suffix_concat_keeps_actual_tmp_path(cleanup,project,real_path):
    folder=project/'docs/governance'; folder.mkdir(parents=True)
    consumer=project/'consumer.py'
    code='from pathlib import Path\ndef atomic_json(path: Path):\n'
    code+='    temporary=path.with_suffix(path.suffix+".tmp")\n'
    if real_path: code+='    actual_input=Path(".tmp")\n'
    consumer.write_text(code)
    contract=folder/'contract.json'; contract.write_text(json.dumps({'consumer':cleanup.H.pin(consumer)}))
    (folder/'CURRENT_STATUS_RECEIPT.json').write_text(json.dumps({'files':{'contract':cleanup.H.pin(contract)}}))
    if real_path:
        with pytest.raises(RuntimeError,match='CURRENT_REFERENCE'): cleanup.authority(root=project,roots=cleanup.ROOTS)
    else:
        assert cleanup.H.pin(consumer) in cleanup.authority(root=project,roots=cleanup.ROOTS)
