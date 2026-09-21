import numpy as np
import pytest
from chaoyang.ops.run_motion_temporal_audit_v2 import edges,derivatives,angular_metrics,wrist_metrics

def times(seconds):
    ns=np.rint(np.array(seconds)*1e9).astype(np.int64)
    edge,t=edges(ns,np.arange(len(ns),dtype=np.int64));return edge,t

def test_linear_nonuniform_known_oracle():
    edge,t=times([0,.1,.25,.4,.55,.8]);q=np.tile((3*t+2)[:,None,None],(1,2,2));v=np.ones((len(t),2),bool)
    d=derivatives(q,v,t,edge)
    assert np.allclose(d['velocity'][1:],3)
    assert np.allclose(d['acceleration'][2:],0,atol=1e-12)
    assert np.allclose(d['jerk'][3:],0,atol=1e-10)

def test_quadratic_nonuniform_known_oracle():
    edge,t=times([0,.1,.25,.4,.55,.8]);q=np.tile((2*t*t+3*t+5)[:,None,None],(1,2,1));v=np.ones((len(t),2),bool)
    d=derivatives(q,v,t,edge)
    assert np.allclose(d['velocity'][1:,0,0],2*(t[1:]+t[:-1])+3)
    assert np.allclose(d['acceleration'][2:],4,atol=1e-10)
    assert np.allclose(d['jerk'][3:],0,atol=1e-8)

def test_gap_and_independent_missing_side():
    edge,t=times([0,.1,.2,2,2.1,2.2]);q=np.tile(t[:,None,None],(1,2,2));v=np.ones((6,2),bool);v[1,0]=False
    d=derivatives(q,v,t,edge)
    assert not edge[3] and np.isnan(d['velocity'][3]).all()
    assert np.isnan(d['velocity'][1:3,0]).all()
    assert np.allclose(d['velocity'][1:3,1],1)
    assert np.isnan(d['acceleration'][3:5]).all()
    m,_=angular_metrics(q,v,t,edge)
    assert m['sides'][0]['full_timeline_frames']==6 and m['sides'][0]['missing_frames']==1

def test_gap_wrong_producer_detected():
    edge,t=times([0,.1,.2,2,2.1,2.2]);q=np.tile(t[:,None,None],(1,2,1));v=np.ones((6,2),bool)
    correct=derivatives(q,v,t,edge)['velocity']
    wrong=np.full_like(q,np.nan);wrong[1:]=np.diff(q,axis=0)/np.diff(t)[:,None,None]
    assert not np.allclose(correct,wrong,equal_nan=True)

def test_frame_hole_and_reset_disconnect():
    ns=np.arange(6,dtype=np.int64)*100000000;ids=np.array([0,1,2,4,5,6],dtype=np.int64)
    edge,_=edges(ns,ids,np.array([0,0,0,0,1,1]))
    assert edge.tolist()==[False,True,True,False,False,True]

@pytest.mark.parametrize('ns,ids',[
    ([0,1,1],[0,1,2]),([0,2,1],[0,1,2]),([0,1,2],[0,0,1])])
def test_bad_time_or_ids_rejected(ns,ids):
    with pytest.raises(ValueError):edges(np.array(ns,dtype=np.int64),np.array(ids,dtype=np.int64))

def test_frozen_trajectory_low_jerk_not_improvement():
    edge,t=times([0,.1,.2,.3,.4,.5]);valid=np.ones((6,2),bool);q=np.zeros((6,2,7))
    d=derivatives(q,valid,t,edge);assert np.allclose(d['jerk'][3:],0)
    target=np.tile(np.eye(4),(6,2,1,1));actual=target.copy();target[:,:,0,3]=t[:,None]
    result=wrist_metrics(target,actual,valid,edge)
    assert all(r['motion_state']=='FROZEN_OUTPUT_WITH_MOVING_TARGET' for r in result)
    assert result[0]['amplitude_norm_ratio']==0 and result[0]['zero_lag_following_error_mm']['max_abs']==500

def test_constant_offset_same_amplitude_still_following_error():
    edge,t=times([0,.1,.2,.3]);valid=np.ones((4,2),bool)
    target=np.tile(np.eye(4),(4,2,1,1));target[:,:,0,3]=t[:,None];actual=target.copy();actual[:,:,1,3]=.2
    r=wrist_metrics(target,actual,valid,edge)
    assert r[0]['amplitude_norm_ratio']==1 and np.isclose(r[0]['zero_lag_following_error_mm']['max_abs'],200)

def test_global_amplitude_does_not_hide_segment_scope():
    edge,t=times([0,.1,.2,2,2.1,2.2]);q=np.zeros((6,2,1));q[3:]=100;v=np.ones((6,2),bool)
    m,_=angular_metrics(q,v,t,edge)
    assert m['sides'][0]['global_joint_range_rad']==[100]
    assert m['sides'][0]['max_within_segment_joint_range_rad']==[0]
