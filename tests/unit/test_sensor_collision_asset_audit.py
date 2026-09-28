import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from chaoyang.pipeline.sensor_collision_asset_audit_v1 import (
    joint_path,mesh_summary,primitive_self_collision_fixture)


def test_joint_distance_does_not_turn_grandparent_into_parent():
    tree=ET.fromstring('''<robot>
    <joint name="j1" type="revolute"><parent link="base"/><child link="one"/></joint>
    <joint name="j2" type="revolute"><parent link="one"/><child link="two"/></joint>
    </robot>''')
    chain=joint_path(tree,'base','two')
    assert [r['joint'] for r in chain]==['j1','j2']
    assert [r['type'] for r in chain]==['revolute','revolute']
    assert len(joint_path(tree,'two','base'))==2
    with pytest.raises(ValueError,match='disconnected'):
        joint_path(tree,'base','missing')


def test_convex_volume_diagnostic_known_box():
    result=mesh_summary(trimesh.creation.box(extents=[.02,.03,.04]))
    assert result['watertight']
    assert result['volume_ratio_mesh_to_hull']==pytest.approx(1.)
    assert result['convex_hull_volume_m3']==pytest.approx(.02*.03*.04)


def test_actual_nonadjacent_collision_positive_and_negative():
    rows=primitive_self_collision_fixture()
    assert [r['actual_penetration'] for r in rows]==[True,False]
    assert all(r['pass_check'] for r in rows)
    assert min(rows[0]['contact_distances_m'])==pytest.approx(-.01,abs=1e-7)
