import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_equipment_rgb import extend,palettes,execute

def fixture():
    raw=np.full((2,80,100,3),30,np.uint8);raw[:,35:45,40:65]=200
    masks={k:np.zeros(raw.shape[:3],bool)for k in ['core','sam_core','equipment','removal','effective_removal','protected_visible','cup_observed']}
    for k in ['core','sam_core','equipment','removal','effective_removal']:masks[k][:,35:45,40:46]=True
    masks['metadata']={'removal':{'observed_segmentation_dilation_px':0,'temporal_bracket_frames':0}}
    seed={'raw_frame_index':0,'objects':[{'role':'controller','visible_confirmed':True,
        'points_px':[[42,40],[10,10]],'point_labels':[1,0]}]}
    return raw,masks,seed

def test_rgb_connected_extension_not_remote_same_colour():
    raw,m,s=fixture();raw[:,0:8,80:88]=200
    r,support,report=extend(raw,m,s)
    assert r['core'][0,40,60] and not r['core'][0,3,83]
    assert not np.any(m['core']&~r['core'])
    assert report['sam_core_changed_pixels']==0 and report['protected_changed_pixels']==0

def test_protected_and_visible_cup_are_not_new_direct_support():
    raw,m,s=fixture();m['protected_visible'][:,36:39,55:58]=True;m['cup_observed'][:,40:43,60:63]=True
    r,support,report=extend(raw,m,s)
    assert not np.any(support&(m['protected_visible']|m['cup_observed']))
    assert np.array_equal(r['protected_visible'],m['protected_visible'])

def test_conflicting_positive_is_not_background():
    raw,m,s=fixture();s['objects'].append({'role':'glove','visible_confirmed':True,'points_px':[[42,40],[10,10]],'point_labels':[0,1]})
    with pytest.raises(ValueError,match='BOTH_VISIBLE_CLASSES'):palettes(raw,s)

def test_noninteger_not_rounded():
    raw,m,s=fixture();s['objects'][0]['points_px'][0]=[42.3,40]
    with pytest.raises(ValueError,match='INTEGER'):palettes(raw,s)

def test_gpu_or_model_not_allowed():
    with pytest.raises(ValueError,match='STRICT_CPU'):
        execute({'mode':'Mask','config':{'cpu_equipment_rgb':True},'budgets':{'gpu_count':1}},authority=lambda:None)

def test_equal_equipment_background_motion_is_unknown_not_support():
    raw,m,s=fixture()
    def flows(*args):return np.zeros((*raw.shape[1:3],2),np.float32),np.zeros((*raw.shape[1:3],2),np.float32)
    r,support,d=extend(raw,m,s,motion_veto=True,flow_provider=flows)
    assert not support.any() and np.array_equal(r['core'],m['core'])

def test_moving_device_static_same_colour_leg_is_excluded():
    raw,m,s=fixture();raw[:,28:35,45:50]=200
    def flows(*args):
        f=np.zeros((*raw.shape[1:3],2),np.float32);back=np.zeros_like(f)
        f[35:45,40:65,0]=3;back[35:45,43:68,0]=-3
        return f,back
    r,support,d=extend(raw,m,s,motion_veto=True,flow_provider=flows)
    assert support[0,40,60] and not support[0,30,47]

def test_closed_loop_error_rejects_ambiguous_flow_support():
    raw,m,s=fixture()
    def flows(*args):
        f=np.zeros((*raw.shape[1:3],2),np.float32);back=np.zeros_like(f)
        f[35:45,40:65,0]=3
        return f,back
    r,support,d=extend(raw,m,s,motion_veto=True,flow_provider=flows)
    assert not support.any()
