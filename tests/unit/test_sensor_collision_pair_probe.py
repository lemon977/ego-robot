import numpy as np
import pytest
import trimesh

from chaoyang.pipeline.sensor_collision_pair_probe_v1 import (
    rigid,triangle_backend_controls,query_mesh_pair,_summary)


def test_triangle_and_convex_backend_controls_execute_real_queries():
    rows=triangle_backend_controls()
    assert len(rows)==4
    assert all(x['query_returned_points']>0 and x['pass_check'] for x in rows)
    assert [np.sign(x['minimum_signed_distance_m']) for x in rows]==[-1,1,-1,1]


@pytest.mark.parametrize('concave',[True,False])
@pytest.mark.parametrize('distance,sign',[(.015,-1),(.04,1)])
@pytest.mark.parametrize('margin',[None,.001])
def test_file_backed_cross_body_query_is_live(tmp_path,concave,distance,sign,margin):
    path=tmp_path/'box.stl'
    trimesh.creation.box(extents=[.02]*3).export(path)
    a=np.eye(4);b=np.eye(4);b[0,3]=distance
    row=query_mesh_pair(path,a,path,b,concave_a=concave,margin_m=margin)
    if margin==.001:
        assert row['collision_margins_m']==[.001,.001]
        assert row['minimum_signed_distance_m']==pytest.approx(distance-.02-.002,abs=1e-7)
    assert np.isfinite(row['closest_witness_world']['point_on_a']).all()
    assert row['cross_body'] and row['query_returned_points']>0
    assert np.sign(row['minimum_signed_distance_m'])==sign
    assert row['concave_static_a']==concave


def test_empty_query_is_not_claimed_collision_free():
    assert _summary([])['status']=='NO_CLOSE_POINT_NOT_PROOF_OF_SEPARATION'
    assert _summary([])['minimum_signed_distance_m'] is None


def test_transform_validation():
    p,q=rigid(np.eye(4))
    assert p==[0.,0.,0.] and q==[0.,0.,0.,1.]
    bad=np.eye(4);bad[0,0]=-1
    with pytest.raises(ValueError,match='proper'):rigid(bad)
    bad=np.eye(4);bad[3,3]=2
    with pytest.raises(ValueError,match='homogeneous'):rigid(bad)


def test_same_margin_controls():
    rows=triangle_backend_controls(margin_m=.001)
    assert all(row['pass_check'] for row in rows)
    assert all(row['query_returned_points']>0 for row in rows)


@pytest.mark.xfail(strict=True,reason='Known Bullet zero-margin convex box penetration depth regression; not a supported diagnostic mode')
def test_zero_margin_depth_regression_keeps_original_oracle(tmp_path):
    path=tmp_path/'zero_margin_box.stl'
    trimesh.creation.box(extents=[.02]*3).export(path)
    a=np.eye(4);b=np.eye(4);b[0,3]=.015
    row=query_mesh_pair(path,a,path,b,concave_a=False,margin_m=0.)
    assert row['minimum_signed_distance_m']==pytest.approx(-.005,abs=1e-7)


def test_runtime_file_controls_same_margin(tmp_path):
    from chaoyang.pipeline.sensor_collision_pair_probe_v1 import file_mesh_controls
    result=file_mesh_controls(tmp_path/'controls',.001)
    assert len(result['rows'])==4
    assert all(r['pass_check'] for r in result['rows'])
    assert all(r['collision_margins_m']==[.001,.001] for r in result['rows'])
    assert result['fixture']['bytes']>0
    assert all(row['query_returned_points']>0 for row in result['rows'])
