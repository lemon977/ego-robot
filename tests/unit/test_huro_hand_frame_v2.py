import numpy as np
import pytest
from chaoyang.pipeline.huro_hand_frame_v2 import palm_basis, canonical_target, temporal_metrics
from chaoyang.pipeline.huro_hand_only_retarget_v1 import HuroHandOnlyError, _root_relative_scaled_targets


def fixture_hand():
    p = np.zeros((21,3))
    for finger in range(5):
        for joint in range(4):
            p[1+4*finger+joint] = [(2-finger)*.02, .05+joint*.02, .002*joint*joint]
    return p


@pytest.mark.parametrize('angle', [0., .3, 1.57, 2.9])
@pytest.mark.parametrize('scale', [.7, 1., 1000.])
def test_camera_similarity_invariance(angle, scale):
    p = fixture_hand()
    R = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    moved = scale * p @ R.T + [1.,2.,3.]
    result, rotation = canonical_target(moved, p)
    np.testing.assert_allclose(result, p, atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)


def test_v1_coordinate_mismatch_negative_control():
    p = fixture_hand()
    rotated = p @ np.array([[0,-1,0],[1,0,0],[0,0,1]]).T
    assert np.max(np.abs(_root_relative_scaled_targets(rotated,p)-p)) > .05
    np.testing.assert_allclose(canonical_target(rotated,p)[0],p,atol=1e-12)


@pytest.mark.parametrize('kind', ['nan','zero','collinear'])
def test_bad_palm_rejected(kind):
    p = fixture_hand()
    if kind == 'nan': p[5,0] = np.nan
    if kind == 'zero': p[:] = 0
    if kind == 'collinear': p[:,1:] = 0
    with pytest.raises(HuroHandOnlyError): palm_basis(p)


def test_actual_dt_linear_and_gap():
    t = np.array([0.,.1,.25,.45,.55,.65])
    q = np.broadcast_to((3*t)[:,None,None],(6,2,22)).copy()
    valid = np.ones((6,2),bool); valid[2,1]=False
    m = temporal_metrics(q,valid,t,np.array([0,1,2,4,5,6]))
    assert np.isnan(m['velocity_rad_s'][3]).all()
    assert np.isnan(m['velocity_rad_s'][2,1]).all()
    np.testing.assert_allclose(m['velocity_rad_s'][1],3.)
    np.testing.assert_allclose(m['acceleration_rad_s2'][5],0.,atol=1e-12)


def test_bad_time_rejected():
    with pytest.raises(HuroHandOnlyError):
        temporal_metrics(np.zeros((3,2,22)),np.ones((3,2),bool),[0,0,1],[0,1,2])


def test_quadratic_nonuniform_acceleration():
    t = np.array([0.,.1,.25,.45,.65])
    q = np.broadcast_to((t*t)[:,None,None],(5,2,22)).copy()
    m = temporal_metrics(q,np.ones((5,2),bool),t,np.arange(5))
    np.testing.assert_allclose(m['acceleration_rad_s2'][2:],2.,atol=1e-12)
    np.testing.assert_allclose(m['jerk_rad_s3'][3:],0.,atol=1e-10)


def test_sequence_resets_missing_and_frame_gap(monkeypatch):
    from types import SimpleNamespace
    import chaoyang.pipeline.huro_hand_frame_v2 as module
    hand = SimpleNamespace(lower=np.zeros(22),upper=np.ones(22),joint_names=tuple(map(str,range(22))))
    p = fixture_hand()
    calls=[]
    def solve(hand,target,previous_q,max_evaluations):
        calls.append(previous_q)
        return np.full(22,.5), p, SimpleNamespace(success=True,evaluations=1,cost=0.)
    monkeypatch.setattr(module,'keypoints_from_q',lambda h,q:p)
    monkeypatch.setattr(module,'solve_frame',solve)
    joints=np.broadcast_to(p,(2,5,21,3))
    observed=np.zeros((2,5),bool); observed[0]=[True,True,False,True,True]
    out=module.solve_aligned_sequence([hand,hand],joints,observed,np.array([0,1,2,3,5]))
    assert calls[0] is None and calls[1] is not None
    assert calls[2] is None and calls[3] is None
    assert not out['valid'][:,0].any()
    assert np.isnan(out['q22'][2]).all()
