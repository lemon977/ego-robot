import hashlib,json,os
import numpy as np
import pytest
from chaoyang.ops import run_four_stream_completion_motion_ledger as m

def put(p,d):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d)); return ref(p)
def ref(p): return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CANONICAL',tmp_path); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    lane=tmp_path/'_run/current'/m.PARENT/'attempts/attempt_0001/lanes/motion'; lane.mkdir(parents=True)
    writer={'pid':os.getpid(),'proc_start_ticks':m.ticks(os.getpid()),'fencing_token':'fixture'}
    packets={}; idx=[]
    for task in (m.PARENT,m.CHILD):
        p=tmp_path/'tasks/current'/task/'TASK_PACKET.json'
        packets[task]=put(p,dict(writer=writer,output_root=str(lane),write_set=[str(lane.relative_to(tmp_path))]))
        idx.append(dict(task_id=task,execution_allowed=True,packet_path=str(p.relative_to(tmp_path)),packet_sha256=packets[task]['sha256']))
    put(tmp_path/'tasks/current/INDEX.json',dict(task_packets=idx))
    source=tmp_path/'source.npz'; np.savez(source,original_frame_indices=np.arange(149),predicted_valid=np.zeros((2,149),bool),anatomical_side_names=np.array(['left','right']))
    sr=ref(source); receipt=put(tmp_path/'receipt.json',dict(session_id='play_cards_0915_031',inputs={'hawor_source':sr}))
    cfg=dict(schema_version='MOTION_LEDGER_FROZEN_CONFIG_V1',session_id='play_cards_0915_031',attempt='attempt_0001',writer=writer,packets=packets,source=sr,source_receipt=receipt,certificates={},evidence=[],output_name='ledger.json')
    return tmp_path,lane,cfg

def execute(f):
    root,lane,cfg=f; r=put(root/'config.json',cfg); return m.run(r['path'],r['sha256'])

def test_normal_fallback_and_repeat(fixture):
    r=execute(fixture); d=json.load(open(r['path']))
    assert d['frame_count']==149 and d['disposition_counts']['ORIGINAL_FRAME_FALLBACK']==149 and not d['product_completed']
    with pytest.raises(FileExistsError): execute(fixture)

def test_escape(fixture):
    fixture[2]['source']['path']='/outside/source.npz'
    with pytest.raises(ValueError,match='ESCAPE'): execute(fixture)

def test_sha_drift(fixture):
    fixture[2]['source']['sha256']='0'*64
    with pytest.raises(ValueError,match='DRIFT'): execute(fixture)

def test_wrong_session(fixture):
    fixture[2]['session_id']='play_cards_0916_102'
    with pytest.raises(ValueError,match='SESSION'): execute(fixture)

def test_writer_fence(fixture):
    fixture[2]['writer']=dict(fixture[2]['writer'],fencing_token='wrong')
    with pytest.raises(ValueError,match='FENCE'): execute(fixture)

def test_output_escape(fixture):
    fixture[2]['output_name']='../ledger.json'
    with pytest.raises(ValueError,match='OUTPUT_NAME'): execute(fixture)
