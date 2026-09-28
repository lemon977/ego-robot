import copy
import hashlib
import json
from pathlib import Path
import pytest
from chaoyang.ops import run_s01_semantic_overlay_audit as op


def fixture(tmp_path,monkeypatch):
    root=tmp_path;lane=root/'sensor';lane.mkdir()
    closure=[]
    for rel in op.CODE_PATHS:
        p=root/'src/chaoyang'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture code\n')
        closure.append({'path':str(p),'bytes':0,'sha256':'0'*64})
    config={'code_closure':closure,'references':{'unchanged':'value'},'mode':'REJECTED_PROVENANCE_DIAGNOSTIC','frame_indices':[14,15,16]}
    p=lane/'S01_OVERLAY_CONFIG_V3.json';p.write_text(json.dumps(config))
    monkeypatch.setattr(op,'TEMPLATE_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(op,'authority',lambda c:(root,lane,[], '123'))
    monkeypatch.setattr(op,'require_canonical_repo',lambda r:r)
    return root,lane,p,config


def test_only_pinned_code_digests_change(tmp_path,monkeypatch):
    root,lane,p,before=fixture(tmp_path,monkeypatch)
    result=op.freeze_config(p);after=json.loads((lane/'S01_OVERLAY_CONFIG_V4.json').read_text())
    assert result['execution']=='CONFIG_FROZEN'
    normalized=copy.deepcopy(after)
    for old,new in zip(before['code_closure'],normalized['code_closure']):
        assert new['path']==old['path'];new['bytes']=old['bytes'];new['sha256']=old['sha256']
    assert normalized==before
    with pytest.raises(FileExistsError):op.freeze_config(p)


@pytest.mark.parametrize('fault',['added','duplicate','escape','template_sha','template_path','symlink'])
def test_scope_drift_and_symlink_rejected(tmp_path,monkeypatch,fault):
    root,lane,p,c=fixture(tmp_path,monkeypatch)
    if fault=='added':c['code_closure'].append(c['code_closure'][0].copy())
    if fault=='duplicate':c['code_closure'][-1]=c['code_closure'][0].copy()
    if fault=='escape':c['code_closure'][0]['path']=str(root/'elsewhere.py')
    if fault=='template_path':p=lane/'WRONG_TEMPLATE.json'
    if fault=='symlink':
        target=Path(c['code_closure'][0]['path']);target.unlink();target.symlink_to(root/'missing.py')
    p.write_text(json.dumps(c))
    if fault=='template_sha':p.write_text(p.read_text()+' ')
    else:monkeypatch.setattr(op,'TEMPLATE_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises((ValueError,FileNotFoundError)):op.freeze_config(p)
    assert not (lane/'S01_OVERLAY_CONFIG_V4.json').exists()


def test_canonical_guard_precedes_publication(tmp_path,monkeypatch):
    root,lane,p,c=fixture(tmp_path,monkeypatch)
    def deny(r):raise ValueError('repo root is not canonical')
    monkeypatch.setattr(op,'require_canonical_repo',deny)
    with pytest.raises(ValueError,match='canonical'):op.freeze_config(p)
    assert not (lane/'S01_OVERLAY_CONFIG_V4.json').exists()


def test_output_symlink_not_followed(tmp_path,monkeypatch):
    root,lane,p,c=fixture(tmp_path,monkeypatch)
    target=root/'elsewhere.json';(lane/'S01_OVERLAY_CONFIG_V4.json').symlink_to(target)
    with pytest.raises(FileExistsError):op.freeze_config(p)
    assert not target.exists()
