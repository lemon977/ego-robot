import numpy as np
import pytest
from chaoyang.pipeline.huro_frozen_targets_v2 import build_targets, segmented_chunks, MANO_JOINT_NAMES


def fixture():
    n=4
    p=np.zeros((21,3))
    for f in range(5):
        for k in range(4):
            p[1+f*4+k]=[(2-f)*.015, .02+k*.02, .002*f]
    points=np.tile(p,(2,n,1,1))
    points[0] += np.arange(n)[:,None,None]*np.array([.01,0,0])
    raw={"original_frame_indices":np.arange(n),"anatomical_side_names":np.array(["left","right"]),
         "joints_3d_camera":points,"observed":np.ones((2,n),bool),"mano_joint_names":np.asarray(MANO_JOINT_NAMES)}
    raw['observed'][1]=False
    relative=np.tile(np.eye(4),(n,2,1,1))
    relative[:,1,0,3]=np.arange(n)*.01
    relative[:,0]=np.nan
    r0={"frame_id":np.arange(n),"timestamp_ns":np.arange(n,dtype=np.int64)*30_000_000,
        "human_to_physical":np.array([1,0]),"relative_wrist_T":relative,
        "valid_side_frame":raw['observed'][::-1].T.copy()}
    return r0,raw,[p,p],np.tile(np.eye(4),(2,1,1))


def test_translation_preserved_missing_hand_not_invented():
    r0,raw,neutral,roots=fixture()
    result=build_targets(r0,raw,neutral,roots)
    assert np.all(result['target_valid'][:,1]) and not result['target_valid'][:,0].any()
    assert np.isnan(result['target21_base'][:,0]).all()
    np.testing.assert_allclose(result['target21_base'][:,1,0,0],[0,.01,.02,.03],atol=1e-10)
    assert result['fixed_scale_by_physical'][1] == pytest.approx(1)


def test_shape_scale_is_one_value_not_per_frame():
    r0,raw,neutral,roots=fixture()
    raw['joints_3d_camera'][0,1] *= 2
    result=build_targets(r0,raw,neutral,roots)
    assert result['fixed_scale_by_physical'].shape == (2,)
    assert np.linalg.norm(result['target21_root'][1,1,8]) > 1.9*np.linalg.norm(result['target21_root'][0,1,8])


def test_no_q22_consumed():
    r0,raw,neutral,roots=fixture()
    a=build_targets(r0,raw,neutral,roots)
    r0['q22_init']=np.full((4,2,22),987.)
    b=build_targets(r0,raw,neutral,roots)
    np.testing.assert_array_equal(a['target21_base'],b['target21_base'])


def test_wrong_side_or_time_rejected():
    r0,raw,neutral,roots=fixture()
    r0['human_to_physical']=np.array([0,1])
    with pytest.raises(ValueError): build_targets(r0,raw,neutral,roots)
    r0['human_to_physical']=np.array([1,0]);r0['timestamp_ns'][2]=0
    with pytest.raises(ValueError): build_targets(r0,raw,neutral,roots)


def test_gap_and_side_changes_split_without_omission():
    ids=np.array([0,1,4,5,6,7])
    valid=np.array([[1,0],[1,0],[1,0],[0,1],[0,0],[0,1]],bool)
    chunks=segmented_chunks(ids,np.arange(6)*30_000_000,valid)
    assert chunks == [[0,1],[2],[3],[5]]


def test_full_length_chunks_preserve_all_378_frames():
    result=segmented_chunks(np.arange(378),np.arange(378)*30_000_000,np.ones((378,2),bool))
    assert max(map(len,result))==32
    assert [x for chunk in result for x in chunk] == list(range(378))


def test_permuted_joint_names_rejected():
    r0,raw,neutral,roots=fixture()
    raw['mano_joint_names']=raw['mano_joint_names'][::-1]
    with pytest.raises(ValueError): build_targets(r0,raw,neutral,roots)
