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
def test_file_backed_cross_body_query_is_live(tmp_path,concave,distance,sign):
    path=tmp_path/'box.stl'
    trimesh.creation.box(extents=[.02]*3).export(path)
    a=np.eye(4);b=np.eye(4);b[0,3]=distance
    row=query_mesh_pair(path,a,path,b,concave_a=concave)
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
