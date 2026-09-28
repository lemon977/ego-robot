import hashlib

import numpy as np
import pytest

from chaoyang.ops.run_four_stream_completion_sensor import read_pinned, source_summary
from chaoyang.pipeline.pico_manus_motion_v2 import MANUS_NAMES


def fixture_source():
    return dict(manus_hand_valid=np.ones((3,2),bool),
                manus_local_25_m=np.zeros((3,2,25,3)),
                manus25_joint_names=np.array(MANUS_NAMES))


def test_missing_observation_does_not_inherit_validity():
    r = source_summary(fixture_source())
    assert r['source_valid_side_frames'] == 6
    assert r['source_observed_side_frames'] == 0
    assert r['source_valid_not_directly_observed'] == 6
    assert len(r['bone_lengths']) == 38


def test_virtual_root_is_not_required_as_anatomical_observation():
    s = fixture_source()
    s['joint_observed_local_25'] = np.ones((3,2,25), bool)
    s['joint_observed_local_25'][:,:,0] = False
    s['joint_observed_local_25'][1,0,2] = False
    assert source_summary(s)['source_observed_side_frames'] == 5


def test_bad_observation_mask_fails():
    s = fixture_source()
    s['joint_observed_local_25'] = np.ones((3,2,25), int)
    with pytest.raises(ValueError,match='boolean'):
        source_summary(s)


def test_sha_pin_and_escape_rejection(tmp_path):
    file = tmp_path/'input.json'
    file.write_bytes(b'{}')
    ref = dict(path=str(file),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())
    assert read_pinned(ref,tmp_path) == b'{}'
    ref['sha256'] = '0'*64
    with pytest.raises(ValueError,match='SHA'):
        read_pinned(ref,tmp_path)
    with pytest.raises(ValueError,match='outside'):
        read_pinned(dict(path=str(tmp_path.parent),bytes=0,sha256='0'*64),tmp_path)


def test_symlink_input_rejected(tmp_path):
    file = tmp_path/'actual'
    file.write_bytes(b'{}')
    link = tmp_path/'link'
    link.symlink_to(file)
    with pytest.raises(ValueError,match='symlink'):
        read_pinned(dict(path=str(link),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest()),tmp_path)


def test_ancestor_symlink_rejected_even_if_inside_root(tmp_path):
    folder = tmp_path/'actual'
    folder.mkdir()
    file = folder/'input'
    file.write_bytes(b'{}')
    (tmp_path/'alias').symlink_to(folder,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        read_pinned(dict(path=str(tmp_path/'alias/input'),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest()),tmp_path)


@pytest.mark.parametrize('fault,message', [
    ('parent_writer','writer mismatch'), ('parent_sha','parent packet SHA'),
    ('packet_path','packet path'), ('output','exact sensor lane'),
    ('write_set','write-set'), ('parent_not_routable','parent routing'),
])
def test_delegation_guards_before_algorithm(tmp_path, monkeypatch, fault, message):
    import json
    import os
    from pathlib import Path
    from chaoyang.ops.run_four_stream_completion_sensor import PARENT, TASK, run
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    writer = dict(pid=os.getpid(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19],fencing_token='fixture')
    lane = tmp_path/'_run/current'/PARENT/'attempts/attempt_0001/lanes/sensor'
    lane.mkdir(parents=True)
    parent = dict(task_id=PARENT,writer=writer.copy())
    child = dict(task_id=TASK,writer=writer.copy(),output_root=str(lane),write_set=[str(lane)])
    if fault == 'parent_writer': parent['writer']['fencing_token']='different'
    if fault == 'output': child['output_root']=str(lane.parent)
    if fault == 'write_set': child['write_set']=[]
    def publish(relative, value):
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        raw=json.dumps(value).encode()
        path.write_bytes(raw)
        return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    p=publish(f'tasks/current/{PARENT}/TASK_PACKET.json',parent)
    c=publish(f'tasks/current/{TASK}/TASK_PACKET.json',child)
    rows=[dict(task_id=PARENT,packet_path=p['path'],packet_sha256=p['sha256'],execution_allowed=True),
          dict(task_id=TASK,packet_path=c['path'],packet_sha256=c['sha256'],execution_allowed=False)]
    if fault == 'parent_sha': rows[0]['packet_sha256']='0'*64
    if fault == 'packet_path': rows[1]['packet_path']=p['path']
    if fault == 'parent_not_routable': rows[0]['execution_allowed']=False
    publish('tasks/current/INDEX.json',dict(task_packets=rows))
    with pytest.raises(ValueError,match=message):
        run(dict(repo_root=str(tmp_path),packet=c,parent_packet=p))
