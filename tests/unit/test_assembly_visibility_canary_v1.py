import numpy as np
import pytest

from chaoyang.pipeline.assembly_visibility_canary_v1 import (
    mount_residual, outside_frustum, visibility, rigid, FRAMES, PANEL_DESIGN)


def test_mount_moves_with_flange_and_rejects_double_offset():
    flange = np.eye(4)
    flange[:3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]
    flange[:3,3] = [.2,.3,.4]
    adapter, hand = np.eye(4), np.eye(4)
    adapter[2,3], hand[2,3] = .03,.06
    good = mount_residual(flange, flange @ adapter, flange @ hand, adapter, hand)
    assert good['adapter_to_hand_max_abs'] < 1e-12
    assert good['adapter_from_flange_max_abs'] < 1e-12
    bad = mount_residual(flange, flange @ adapter @ adapter, flange @ hand, adapter, hand)
    assert bad['adapter_from_flange_max_abs'] == pytest.approx(.03)
    assert bad['adapter_to_hand_max_abs'] == pytest.approx(.03)


@pytest.mark.parametrize('scale', [-1, 2])
def test_nonrigid_transform_rejected(scale):
    T = np.eye(4)
    T[0,0] = scale
    with pytest.raises(ValueError):
        rigid(T)


def test_visibility_positive_negative_unknown_and_invalid():
    full = np.ones((2,2), bool)
    empty = ~full
    z = np.ones((2,2))
    assert visibility(True, full, full, z, z)['status'] == 'VISIBLE'
    assert visibility(True, empty, full, z, z*.5)['status'] == 'ROBOT_OCCLUDED'
    assert visibility(True, empty, full, z, z*2)['status'] == 'UNRESOLVED_NOT_PROVEN_OCCLUDED'
    assert visibility(True, empty, empty, z, z, True)['status'] == 'OUTSIDE_CAMERA_FRUSTUM'
    assert visibility(False, empty, empty, z, z)['status'] == 'INVALID_SOURCE'
    with pytest.raises(ValueError):
        visibility(False, full, full, z, z)


def test_frustum_requires_all_vertices_outside_same_plane():
    K = np.array([[100,0,50],[0,100,50],[0,0,1.]])
    assert outside_frustum([[2,0,1],[3,0,1]], K,100,100,.02,20)
    assert not outside_frustum([[-2,0,1],[2,0,1]], K,100,100,.02,20)
    assert outside_frustum([[0,0,-1],[.1,0,-1]], K,100,100,.02,20)


def test_domain_mismatch_rejected():
    with pytest.raises(ValueError):
        visibility(True, np.zeros((2,2)), np.zeros((3,2)), np.zeros((2,2)), np.zeros((2,2)))


def test_fixed_frames_and_chinese_design():
    assert FRAMES == (0,189,377)
    assert PANEL_DESIGN['canvas'] == [1920,1080]
    assert '未经物理验证' in PANEL_DESIGN['footer']


def test_frame_report_uses_renderer_layers_and_independent_mask(monkeypatch):
    from types import SimpleNamespace
    import chaoyang.pipeline.assembly_visibility_canary_v1 as module
    ids = np.array([[2,3],[4,5]], np.uint8)
    transforms = np.stack([np.eye(4),np.eye(4)])
    transforms[:,2,3] = 1
    local = np.stack([np.eye(4),np.eye(4)])
    layers = SimpleNamespace(component_id=ids, alpha=ids!=0, depth_valid=ids!=0,
              optical_depth_m=np.ones((2,2)), T_world_flange=transforms,
              T_world_adapter=transforms, T_world_hand=transforms)
    renderer = SimpleNamespace(frame_layers=lambda frame:layers,
          motion={'wrist_valid':np.ones((378,2),bool),'finger_valid':np.ones((378,2),bool),
                  'T_flange_hand':local}, T_flange_adapter=local, T_camera_base=np.eye(4),
          k=np.array([[1,0,1],[0,1,1],[0,0,1]]),width=2,height=2,near=.02,far=20)
    monkeypatch.setattr(module,'isolated_adapter',lambda *args:(np.ones((2,2),bool),np.ones((2,2))))
    result, returned = module.frame_report(renderer,189,'unused',1,np.array([[0,0,0],[.1,0,0],[0,.1,0]]))
    assert returned is layers
    assert result['component_pixels']['2'] == 1
    assert result['sides'][0]['mount']['adapter_to_hand_max_abs'] == 0
    assert result['sides'][0]['visibility']['status'] == 'VISIBLE'
    with pytest.raises(ValueError,match='outside fixed'):
        module.frame_report(renderer,188,'unused',1,np.zeros((3,3)))
    layers.alpha = np.zeros((2,2),bool)
    with pytest.raises(ValueError,match='alpha/component'):
        module.frame_report(renderer,189,'unused',1,np.zeros((3,3)))
