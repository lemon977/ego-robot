"""Bounded renderer canary; caller must pin sources and hold publisher authority."""
from __future__ import annotations

import numpy as np

FRAMES = (0, 189, 377)
COMPONENTS = {0: '背景', 1: '机械臂', 2: '左连接件', 3: '右连接件',
              4: '左机械手', 5: '右机械手'}
PANEL_DESIGN = {
    'canvas': [1920, 1080], 'layout': '2x2_equal_letterbox',
    'panels': ['机器人装配诊断（非原场景替换）', '部件编号与像素数量',
               '相机光轴深度（米）', '左右连接件可见性与固定装配残差'],
    'footer': '真实 CAD 几何；安装关系未经物理验证；不评价环境遮挡',
    'depth_range_m': [0.02, 20.0], 'source_frame_ids': list(FRAMES),
    'no_nested_video': True, 'preserve_aspect_ratio': True,
}


def rigid(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('nonfinite or malformed rigid transform')
    r = value[:3, :3]
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('invalid homogeneous row')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('transform contains scale or reflection')
    return value


def mount_residual(flange, adapter, hand, flange_adapter, flange_hand):
    flange, adapter, hand, flange_adapter, flange_hand = map(
        rigid, (flange, adapter, hand, flange_adapter, flange_hand))
    expected = np.linalg.inv(flange_adapter) @ flange_hand
    actual = np.linalg.inv(adapter) @ hand
    return dict(adapter_from_flange_max_abs=float(np.max(np.abs(adapter - flange @ flange_adapter))),
                hand_from_flange_max_abs=float(np.max(np.abs(hand - flange @ flange_hand))),
                adapter_to_hand_max_abs=float(np.max(np.abs(actual - expected))),
                adapter_to_hand=actual.tolist(),
                interpretation='NUMERICAL_MOUNT_CONSISTENCY_NOT_PHYSICAL_CALIBRATION')


def outside_frustum(points_camera, K, width, height, near, far):
    """Sufficient rejection only; false does not establish actual visibility."""
    p = np.asarray(points_camera, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('invalid mesh vertices')
    k = np.asarray(K, dtype=float)
    z = p[:, 2]
    u = k[0, 0] * p[:, 0] + k[0, 2] * z
    v = k[1, 1] * p[:, 1] + k[1, 2] * z
    return bool(any(np.all(x) for x in (z < near, z > far, u < 0,
                u > width * z, v < 0, v > height * z)))


def visibility(valid, visible_mask, isolated_mask, isolated_depth, scene_depth, outside=False):
    visible_mask, isolated_mask = map(lambda x: np.asarray(x, dtype=bool), (visible_mask, isolated_mask))
    isolated_depth, scene_depth = map(np.asarray, (isolated_depth, scene_depth))
    if any(x.shape != visible_mask.shape for x in (isolated_mask, isolated_depth, scene_depth)):
        raise ValueError('visibility domain mismatch')
    visible, isolated = int(visible_mask.sum()), int(isolated_mask.sum())
    occluded = isolated_mask & ~visible_mask & np.isfinite(scene_depth) & np.isfinite(isolated_depth) & (scene_depth < isolated_depth - 1e-5)
    hidden = int(occluded.sum())
    if not valid:
        if visible:
            raise ValueError('invalid source rendered as valid component')
        status = 'INVALID_SOURCE'
    elif visible:
        status = 'VISIBLE_PARTIALLY_OCCLUDED' if hidden else 'VISIBLE'
    elif isolated and hidden == isolated:
        status = 'ROBOT_OCCLUDED'
    elif not isolated and outside:
        status = 'OUTSIDE_CAMERA_FRUSTUM'
    else:
        status = 'UNRESOLVED_NOT_PROVEN_OCCLUDED'
    return dict(status=status, visible_pixels=visible, isolated_pixels=isolated,
                robot_occluded_pixels=hidden, environment_occlusion='NOT_EVALUATED')


def isolated_adapter(renderer, mesh_path, mesh_scale, transform):
    """Separate static visual body, same view/projection; no shared scene mutation."""
    b = renderer.b
    client = b.connect(b.DIRECT)
    if client < 0:
        raise RuntimeError('isolated CPU renderer unavailable')
    try:
        shape = b.createVisualShape(b.GEOM_MESH, fileName=str(mesh_path),
                    meshScale=[float(mesh_scale)] * 3, rgbaColor=[.4,.43,.47,1], physicsClientId=client)
        body = b.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, physicsClientId=client)
        renderer.place_hand(client, body, rigid(transform))
        result = b.getCameraImage(renderer.width, renderer.height, renderer.view,
                    renderer.projection, renderer=b.ER_TINY_RENDERER,
                    flags=b.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, shadow=0, physicsClientId=client)
        seg = np.asarray(result[4]).reshape(renderer.height, renderer.width)
        mask = (seg >= 0) & ((seg.astype(np.int64) & ((1 << 24) - 1)) == body)
        from chaoyang.pipeline.v5_product import optical_depth_from_buffer
        depth = optical_depth_from_buffer(np.asarray(result[3]).reshape(mask.shape), renderer.near, renderer.far)
        depth[~mask] = np.nan
        return mask, depth
    finally:
        b.disconnect(client)


def frame_report(renderer, frame, mesh_path, mesh_scale, mesh_vertices):
    """Only three fixed source frames; no optimization or q mutation."""
    if frame not in FRAMES:
        raise ValueError('frame outside fixed assembly canary')
    layers = renderer.frame_layers(frame)
    ids = np.asarray(layers.component_id)
    if not set(np.unique(ids)).issubset(COMPONENTS):
        raise ValueError('unknown component ID')
    if not np.array_equal(np.asarray(layers.alpha, dtype=bool), ids != 0):
        raise ValueError('alpha/component mismatch')
    if not np.array_equal(layers.depth_valid, ids != 0) or not np.isfinite(layers.optical_depth_m[ids != 0]).all():
        raise ValueError('depth/component mismatch')
    sides = []
    for side in range(2):
        valid = bool(renderer.motion['wrist_valid'][frame, side] and renderer.motion['finger_valid'][frame, side])
        isolated = np.zeros(ids.shape, bool)
        depth = np.full(ids.shape, np.nan)
        outside, residual = False, None
        if valid:
            transform = rigid(layers.T_world_adapter[side])
            isolated, depth = isolated_adapter(renderer, mesh_path, mesh_scale, transform)
            xyz = np.asarray(mesh_vertices) * mesh_scale
            T = rigid(renderer.T_camera_base) @ transform
            camera = xyz @ T[:3,:3].T + T[:3,3]
            outside = outside_frustum(camera, renderer.k, renderer.width, renderer.height, renderer.near, renderer.far)
            residual = mount_residual(layers.T_world_flange[side], transform,
                        layers.T_world_hand[side], renderer.T_flange_adapter[side], renderer.motion['T_flange_hand'][side])
        sides.append(dict(side=('left','right')[side], mount=residual,
                      visibility=visibility(valid, ids == side + 2, isolated, depth, layers.optical_depth_m, outside)))
    report = dict(source_frame_id=frame, component_pixels={str(i): int((ids == i).sum()) for i in COMPONENTS}, sides=sides,
        conventions=dict(T_A_B='B_TO_A', units='metre', camera='x_right_y_down_z_forward',
                         depth='OPTICAL_CAMERA_Z_METRES', alpha='BOOLEAN_SEGMENTATION_NOT_MATTING',
                         scene='ROBOT_ONLY_NO_ENVIRONMENT_OCCLUSION', physical_mount='UNVERIFIED'),
        panel_design=PANEL_DESIGN)
    return report, layers
