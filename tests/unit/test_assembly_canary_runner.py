import numpy as np
import pytest
from chaoyang.ops.run_four_stream_completion_assembly_canary import validate_motion, SESSION


def fixture():
    m=dict(q_arm=np.zeros((378,2,7)),q_hand22=np.zeros((378,2,22)),
      wrist_valid=np.ones((378,2),bool),finger_valid=np.ones((378,2),bool),
      legacy_candidate=np.array(True),control_ground_truth=np.array(False),training_eligible=np.array(False),
      frame_id=np.arange(378),timestamp_ns=np.arange(378,dtype=np.int64)*1000000,
      anatomical_side_names=np.array(['left','right']),human_to_physical=np.array([0,1]))
    d=dict(session_id=SESSION,width=1280,height=960,source_index=1,
      frames=[dict(frame_id=i,capture_time={'original_time_fields':{'ts':int(m['timestamp_ns'][i])}}) for i in range(378)])
    p=dict(session_id=SESSION,status='LEGACY_MOTION_RECOVERED_FOR_DIAGNOSTIC')
    return m,d,p


def test_exact_legacy_timeline():
    validate_motion(*fixture())


@pytest.mark.parametrize('field', ['training_eligible','control_ground_truth'])
def test_no_authority_upgrade(field):
    m,d,p=fixture();m[field]=True
    with pytest.raises(ValueError):validate_motion(m,d,p)


@pytest.mark.parametrize('field,value', [('q_arm',np.zeros((3,2,7))),
      ('wrist_valid',np.ones((378,2),float)),('human_to_physical',np.array([1,0])),
      ('timestamp_ns',np.arange(378)),('anatomical_side_names',np.array(['right','left']))])
def test_shape_mask_axis_and_time_negative(field,value):
    m,d,p=fixture();m[field]=value
    with pytest.raises((ValueError,AssertionError)):validate_motion(m,d,p)


def test_wrong_domain_and_short_window_rejected():
    m,d,p=fixture();d['source_index']=0
    with pytest.raises(ValueError):validate_motion(m,d,p)
    m,d,p=fixture();d['frames']=d['frames'][:3]
    with pytest.raises(ValueError):validate_motion(m,d,p)


@pytest.mark.parametrize('fault', ['none','sha','writer','route','lane','gpu'])
def test_authority_positive_and_fail_closed(tmp_path,monkeypatch,fault):
    import json,os,hashlib
    from pathlib import Path
    from chaoyang.ops.run_four_stream_completion_assembly_canary import authority,PARENT,TASK
    root=tmp_path
    current=root/'tasks/current';current.mkdir(parents=True)
    lane=root/'_run/current'/PARENT/'attempts/attempt_0001/lanes/sensor';lane.mkdir(parents=True)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19]
    writer={'pid':os.getpid(),'proc_start_ticks':ticks}
    child={'task_id':TASK,'writer':writer,'output_root':str(lane),'write_set':[str(lane)]}
    parent={'task_id':PARENT,'writer':writer}
    if fault=='writer':parent['writer']={'pid':-1}
    if fault=='lane':child['output_root']=str(root)
    config={'repo_root':str(root)};entries=[]
    for key,packet in [('packet',child),('parent_packet',parent)]:
        path=current/(key+'.json');raw=json.dumps(packet).encode();path.write_bytes(raw)
        config[key]={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        entries.append({'task_id':packet['task_id'],'packet_path':str(path),
                        'packet_sha256':config[key]['sha256'],'execution_allowed':packet['task_id']==PARENT})
    if fault=='sha':entries[0]['packet_sha256']='0'*64
    if fault=='route':entries[1]['execution_allowed']=False
    (current/'INDEX.json').write_text(json.dumps({'task_packets':entries}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0' if fault=='gpu' else '')
    if fault=='none':assert authority(config)[1]==lane
    else:
        with pytest.raises(ValueError):authority(config)
