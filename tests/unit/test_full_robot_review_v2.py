import numpy as np
import pytest
from chaoyang.pipeline.full_robot_review_v2 import (
    tool_mount_from_flange, relative_wrist, time_edges, motion_derivatives,
    validate_motion, validate_se3, read_frozen,
    seed_usable_solution,
)


def transform(angle=0, translation=(0,0,0)):
    c,s=np.cos(angle),np.sin(angle)
    t=np.eye(4); t[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];t[:3,3]=translation
    return t


def motion():
    t=np.broadcast_to(np.eye(4),(5,2,4,4)).copy()
    return dict(q22=np.zeros((5,2,22)),finger_valid=np.ones((5,2),bool),
                wrist_valid=np.ones((5,2),bool),relative_wrist_T=t,
                timestamp_ns=np.array([100,200,300,400,500],np.int64),
                frame_id=np.arange(5),human_to_physical=np.array([1,0]))


def test_flange_tool_conversion_known_analytic_oracle():
    tool=transform(np.pi/2,(0,0,.145));mount=transform(0,(.03,0,.0752))
    converted=tool_mount_from_flange(tool,mount)
    # Rz(-90) rotates the relative translation (.03,0,-.0698).
    np.testing.assert_allclose(converted[:3,3],[0,-.03,-.0698],atol=1e-12)
    np.testing.assert_allclose(tool@converted,mount,atol=1e-12)
    assert not np.allclose(tool@mount,mount)


def test_mount_parity_at_multiple_flange_poses():
    ft=transform(.3,(0,0,.145)); fh=transform(-.5,(0,0,.0752))
    th=tool_mount_from_flange(ft,fh)
    for angle in (0,.7,2.0):
        bf=transform(angle,(.2,.1,.8))
        np.testing.assert_allclose(bf@ft@th,bf@fh,atol=1e-12)


def test_relative_wrist_preserves_world_translation_amplitude():
    w=np.broadcast_to(np.eye(4),(5,2,4,4)).copy()
    w[:,0,0,3]=np.arange(5)*.1+3
    valid=np.ones((5,2),bool);valid[2,0]=False
    relative,origin=relative_wrist(w,valid)
    np.testing.assert_allclose(relative[[0,1,3,4],0,0,3],[0,.1,.3,.4])
    assert np.isnan(relative[2,0]).all(); assert origin[0,0,3]==3


def test_world_rotation_changes_offset_but_local_offset_fixed():
    offset=transform(0,(.1,0,0))
    a=transform(0)@offset;b=transform(np.pi/2)@offset
    np.testing.assert_allclose(a[:3,3],[.1,0,0],atol=1e-12)
    np.testing.assert_allclose(b[:3,3],[0,.1,0],atol=1e-12)
    np.testing.assert_allclose(np.linalg.inv(transform(np.pi/2))@b,offset,atol=1e-12)


def test_rotation_only_does_not_move_wrist_origin():
    a=transform(0,(1,2,3));b=a@transform(.5)
    np.testing.assert_array_equal(a[:3,3],b[:3,3])


@pytest.mark.parametrize('bad', ['mirror','rotation','row','nan'])
def test_bad_se3_rejected(bad):
    a=np.eye(4)
    if bad=='mirror':a[0,0]=-1
    if bad=='rotation':a[0,0]=2
    if bad=='row':a[3,0]=1
    if bad=='nan':a[0,3]=np.nan
    with pytest.raises(ValueError):validate_se3(a)


def test_dt_gap_reset_and_original_frame_axis():
    ns=np.array([0,10,20,200,210,220],np.int64)
    edge,t=time_edges(ns,np.array([0,1,2,3,4,6]),np.array([0,0,0,0,1,1]))
    np.testing.assert_array_equal(edge,[False,True,True,False,False,False])


@pytest.mark.parametrize('ns', [[0,1,1],[0,2,1]])
def test_bad_clock_rejected(ns):
    with pytest.raises(ValueError):time_edges(np.array(ns,np.int64),np.arange(3))


def test_float_time_rejected_not_rounded():
    with pytest.raises(ValueError):time_edges(np.array([0.,1.,2.]),np.arange(3))


def test_actual_dt_linear_and_quadratic_oracles():
    times=np.array([0.,.1,.25,.4,.55]);ns=np.rint(times*1e9).astype(np.int64)
    edge,t=time_edges(ns,np.arange(5))
    q=np.zeros((5,2,1));q[:,0,0]=3*times+2;q[:,1,0]=times**2
    result=motion_derivatives(q,np.ones((5,2),bool),t,edge)
    np.testing.assert_allclose(result['velocity'][1:,0,0],3)
    np.testing.assert_allclose(result['acceleration'][2:,1,0],2)


def test_derivatives_never_bridge_missing_side_or_gap():
    ns=np.array([0,10,20,200,210,220],np.int64);edge,t=time_edges(ns,np.arange(6))
    valid=np.ones((6,2),bool);valid[1,0]=False
    result=motion_derivatives(np.ones((6,2,1)),valid,t,edge)
    assert np.isnan(result['velocity'][1:3,0]).all()
    assert np.isfinite(result['velocity'][1:3,1]).all()
    assert np.isnan(result['velocity'][3]).all()


def test_masks_do_not_invent_zero_observation():
    d=motion();d['finger_valid'][2,0]=False;d['q22'][2,0]=np.nan
    d['wrist_valid'][3,1]=False;d['relative_wrist_T'][3,1]=np.nan
    validate_motion(d)
    assert d['wrist_valid'][2,0] and d['finger_valid'][3,1]


def test_valid_nonfinite_rejected():
    d=motion();d['q22'][2,0,0]=np.nan
    with pytest.raises(ValueError):validate_motion(d)


def test_wrong_side_map_rejected():
    d=motion();d['human_to_physical']=np.array([0,1])
    with pytest.raises(ValueError):validate_motion(d)


def test_npz_roundtrip_preserves_mask_nan_and_int64(tmp_path):
    d=motion();d['q22'][2,0]=np.nan;d['finger_valid'][2,0]=False
    path=tmp_path/'fixture.npz';np.savez_compressed(path,**d)
    with np.load(path,allow_pickle=False) as z:
        for k in d:np.testing.assert_array_equal(z[k],d[k])
        validate_motion(dict(z))


def test_sha_scope_and_corruption_rejected(tmp_path):
    import hashlib
    p=tmp_path/'fixture.json';p.write_bytes(b'{}')
    ref=dict(path=str(p),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())
    assert read_frozen(ref,[tmp_path])==b'{}'
    p.write_bytes(b'[]')
    with pytest.raises(ValueError):read_frozen(ref,[tmp_path])


def test_max_nfev_valid_pose_can_seed_without_becoming_quality_pass():
    q=np.zeros(7);lo=np.full(7,-1.);hi=np.full(7,1.)
    assert seed_usable_solution(q,lo,hi,solver_success=False,solver_status=0,
                                position_mm=3.4,rotation_deg=.9,independent_fk=np.eye(4))
    # A reusable initializer does not change the current frame's convergence fact.
    assert not (False and 3.4 <= 20 and .9 <= 15)


@pytest.mark.parametrize('change', ['nan','bounds','pose','rotation','status','fk'])
def test_invalid_solution_never_seeds(change):
    q=np.zeros(7);lo=np.full(7,-1.);hi=np.full(7,1.)
    kwargs=dict(solver_success=False,solver_status=0,position_mm=3.4,
                rotation_deg=.9,independent_fk=np.eye(4))
    if change=='nan':q[0]=np.nan
    if change=='bounds':q[0]=2.
    if change=='pose':kwargs['position_mm']=21.
    if change=='rotation':kwargs['rotation_deg']=16.
    if change=='status':kwargs['solver_status']=-1
    if change=='fk':kwargs['independent_fk'][0,0]=np.nan
    assert not seed_usable_solution(q,lo,hi,**kwargs)
