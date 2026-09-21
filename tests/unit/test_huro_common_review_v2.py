import numpy as np
import pytest
from chaoyang.ops.run_huro_common_review_v2 import validate_inputs,compare_points,stats

def fixture():
    ids=np.arange(3,dtype=np.int64);ts=ids*30_000_000
    valid=np.ones((3,2),bool);trans=np.tile(np.eye(4),(3,2,1,1))
    core=dict(source_frame_id=ids,timestamp_ns=ts,human_to_physical=np.array([1,0]),
              source_valid_side_frame=valid,q_arm=np.zeros((3,2,7)),q22=np.zeros((3,2,22)),
              valid_side_frame=np.zeros_like(valid),limit_violation_rad=np.ones((3,2)))
    target=dict(source_frame_id=ids.copy(),timestamp_ns=ts.copy(),target_valid=valid.copy(),
                target_root_T=trans.copy(),target21_base=np.zeros((3,2,21,3)))
    base=dict(frame_id=ids.copy(),timestamp_ns=ts.copy(),T_target_root=trans.copy())
    return core,target,base

def test_limits_cannot_hide_difficult_frames():
    c,t,b=fixture();assert validate_inputs(c,t,b).sum()==6

@pytest.mark.parametrize('fault',['time','frame','placement','mask','side','nan','target_nan','rotation_drift'])
def test_mismatches_rejected(fault):
    c,t,b=fixture()
    if fault=='time':b['timestamp_ns'][1]+=1
    if fault=='frame':b['frame_id'][1]+=1
    if fault=='placement':b['T_target_root'][0,0,0,3]=.001
    if fault=='mask':t['target_valid'][0,0]=False
    if fault=='side':c['human_to_physical']=np.array([0,1])
    if fault=='nan':c['q22'][0,0,1]=np.nan
    if fault=='target_nan':t['target21_base'][0,0,1,0]=np.nan
    if fault=='rotation_drift':t['target_root_T'][0,0,0,0]+=9e-6
    with pytest.raises(ValueError):validate_inputs(c,t,b)

def test_known_error_and_independent_coverage():
    c,t,b=fixture();v=c['source_valid_side_frame'];refv=v.copy();refv[0,0]=False
    ref=np.zeros((3,2,21,3));candidate=ref.copy();candidate[...,0]=.01
    r=compare_points(ref,candidate,t,refv,v)
    assert r['total_timeline_frames']==3
    assert r['common_side_frames']==[2,3]
    assert r['reference_coverage']==[2,3] and r['candidate_coverage']==[3,3]
    assert r['official_huro']['all_joint_error_mm_common']['median']==10
    assert r['reference']['all_joint_error_mm_common']['maximum']==0

def test_empty_is_unknown_not_zero():
    assert stats([np.nan])==dict(count=0,median=None,p95=None,maximum=None)

def test_missing_joint_cannot_disappear_from_denominator():
    c,t,b=fixture();x=t['target21_base'].copy();x[0,0,1,0]=np.nan
    with pytest.raises(ValueError,match='NONFINITE_COMMON'):
        compare_points(x,t['target21_base'],t,t['target_valid'],t['target_valid'])

def test_raw_fk_retains_violation_without_mutating_model():
    from pathlib import Path
    from chaoyang.pipeline.robot_renderer_cycles import JointSpec,UrdfModel,forward_kinematics,RendererError
    from chaoyang.ops.run_huro_common_review_v2 import diagnostic_fk_unbounded
    joint=JointSpec('j','revolute','base','tip',np.eye(4),np.array([0.,0.,1.]),-.1,.1)
    model=UrdfModel(Path('synthetic'),'base',('base','tip'),(joint,),())
    with pytest.raises(RendererError):forward_kinematics(model,{'j':np.pi/2})
    fk=diagnostic_fk_unbounded(model,{'j':np.pi/2})
    np.testing.assert_allclose(fk['tip'][:3,:3]@np.array([1.,0,0]),[0,1,0],atol=1e-14)
    assert model.joints[0].upper==.1
    with pytest.raises(RendererError):diagnostic_fk_unbounded(model,{})
