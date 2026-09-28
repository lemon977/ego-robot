import cv2
import numpy as np
import pytest

from chaoyang.pipeline.occlusion_compositor_v1 import exclude_removed_foreground_depth
from chaoyang.pipeline.r2_dependency_signature import affected_stages, build_product_binding
from chaoyang.pipeline.v5_scene import (
    FrameRoles,
    build_conservative_repair_window,
    build_object_protected_repair_window,
)
from chaoyang.pipeline.v5_product import (
    arm_state_for_render,
    hand_root_from_flange,
    optical_depth_from_buffer,
    visible_robot_mask,
)


def roles(human=(), device=(), obj=(), shape=(32, 32)):
    arrays = []
    for points in (human, device, obj):
        value = np.zeros(shape, dtype=bool)
        for y, x in points:
            value[y, x] = True
        arrays.append(value)
    return FrameRoles(*arrays)


def test_conservative_repair_cannot_import_distant_neighbour_foreground():
    previous = roles(human=[(2, 2)])
    current = roles(human=[(20, 20)])
    result = build_conservative_repair_window([previous, current], 1, 10, support_margin=2)
    assert result["write"][20, 20]
    assert not result["write"][2, 2]


def test_device_pixels_reach_actual_reference_tensor():
    current = roles(device=[(12, 12)])
    result = build_conservative_repair_window([current], 0, 0, support_margin=1)
    assert result["context_exclude"][12, 12]
    assert result["write"][12, 12]


def test_object_is_unknown_not_silently_promoted_to_protection():
    current = roles(human=[(10, 10)], obj=[(10, 10), (25, 25)])
    result = build_conservative_repair_window([current], 0, 0, support_margin=1)
    assert not result["protect"].any()
    assert result["unknown"][10, 10]
    assert result["unknown"][25, 25]


def test_current_foreground_always_covered_without_convex_hull_bridge():
    current = roles(human=[(8, 5), (8, 25)])
    result = build_conservative_repair_window([current], 0, 0, support_margin=1)
    assert result["write"][8, 5] and result["write"][8, 25]
    assert not result["write"][8, 15]


def test_direct_visible_object_interior_can_only_reduce_nearby_write():
    current = roles(human=[(16, 10)], obj=[(y, x) for y in range(13, 20) for x in range(13, 20)])
    baseline = build_conservative_repair_window([current], 0, 0, support_margin=4)
    protected = build_object_protected_repair_window([current], 0, 0, support_margin=4)
    assert protected["protect"][16, 14]
    assert baseline["write"][16, 14]
    assert not protected["write"][16, 14]
    assert not np.any(protected["write"] & protected["protect"])


def test_object_human_overlap_stays_unknown_and_writable():
    current = roles(human=[(10, 10)], obj=[(y, x) for y in range(8, 13) for x in range(8, 13)])
    result = build_object_protected_repair_window([current], 0, 0, support_margin=4)
    assert result["unknown"][10, 10]
    assert result["write"][10, 10]
    assert not result["protect"][10, 10]


def test_invalid_side_cannot_render_default_hand_or_arm_links():
    def packed(body: int, link: int) -> int:
        return body + ((link + 1) << 24)

    segmentation = np.asarray([
        [packed(1, 2), packed(1, 7), packed(2, 0), packed(3, 0)],
        [packed(1, -1), -1, packed(1, 8), packed(1, 3)],
    ], dtype=np.int64)
    mask = visible_robot_mask(segmentation, robot=1, hands=[2, 3],
                              arm_links=[{2, 3}, {7, 8}],
                              valid_sides=[False, True])
    assert not mask[0, 0]  # invalid left-arm link
    assert not mask[0, 2]  # invalid left hand body
    assert mask[0, 1] and mask[0, 3] and mask[1, 2]  # valid right side
    assert mask[1, 0]  # robot base is not falsely removed


def test_rendered_hand_root_uses_actual_flange_fk_not_target_wrist():
    flange = np.eye(4)
    flange[:3, 3] = [0.10, -0.20, 0.30]
    mount = np.eye(4)
    mount[2, 3] = 0.0684
    target = np.eye(4)
    target[:3, 3] = [0.80, 0.90, 1.00]
    rendered = hand_root_from_flange(flange, mount)
    assert np.allclose(rendered[:3, 3], [0.10, -0.20, 0.3684])
    assert not np.allclose(rendered, target)


def test_nonzero_inertial_and_visual_origins_are_applied_once(tmp_path):
    """PyBullet base poses are COM poses; the renderer must expose the link frame."""
    import pybullet as bullet

    from chaoyang.ops.render_tianji_kai_mount_proxy_audit import (
        link_frames,
        place_hand,
    )

    urdf = tmp_path / "offsets.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="offsets">
  <link name="base_link">
    <inertial><origin xyz="0.1 0 0" rpy="0 0 0"/><mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <visual><origin xyz="0 0.2 0" rpy="0 0 0"/>
      <geometry><box size="0.02 0.02 0.02"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    client = bullet.connect(bullet.DIRECT)
    try:
        body = bullet.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        desired = np.eye(4, dtype=np.float64)
        desired[:3, 3] = [0.3, -0.4, 0.5]
        place_hand(client, body, desired)
        observed = link_frames(client, body)["base_link"]
        assert np.allclose(observed, desired, atol=1e-12, rtol=0.0)

        # The visual offset remains a local visual-frame property.  It is not
        # folded into the link pose and then applied a second time.
        visual = bullet.getVisualShapeData(body, physicsClientId=client)[0]
        assert np.allclose(visual[5], [0.0, 0.2, 0.0], atol=1e-12, rtol=0.0)
    finally:
        bullet.disconnect(client)


def test_robot_depth_buffer_is_converted_to_metric_optical_z():
    near, far = 0.02, 20.0
    metric = optical_depth_from_buffer(np.asarray([[0.0, 1.0, 0.5]]), near, far)
    assert np.isclose(metric[0, 0], near)
    assert np.isclose(metric[0, 1], far)
    assert near < metric[0, 2] < far


def test_missing_arm_nan_uses_neutral_but_valid_nan_fails():
    neutral = np.linspace(-0.3, 0.3, 7)
    missing = np.full(7, np.nan)
    assert np.array_equal(arm_state_for_render(missing, False, neutral), neutral)
    with pytest.raises(ValueError, match="VALID_ARM_RENDER_STATE_NONFINITE"):
        arm_state_for_render(missing, True, neutral)


def test_removed_human_equipment_depth_cannot_occlude_robot():
    depth = np.asarray([[0.5, 0.8], [1.0, 1.2]], dtype=np.float64)
    valid = np.ones((2, 2), dtype=np.bool_)
    removed = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    kept_depth, kept_valid = exclude_removed_foreground_depth(
        scene_depth_m=depth,
        scene_depth_valid=valid,
        human_equipment_mask=removed,
    )
    assert not np.any(kept_valid[removed])
    assert np.isnan(kept_depth[removed]).all()
    assert np.array_equal(kept_depth[~removed], depth[~removed])


def test_r2_dependency_invalidation_is_stage_local():
    assert affected_stages({"scene_masks"}) == {"scene_clean", "product_render"}
    assert affected_stages({"roi"}) == {"motion_r0", "product_render"}
    assert affected_stages({"render_config"}) == {"product_render"}
    assert affected_stages({"scene_depth"}) == {"occlusion", "product_render"}


def test_product_cache_binding_is_content_addressed(tmp_path):
    files = {}
    for name in ("scene_clean", "motion_r0", "camera_domain", "robot_asset", "renderer_code"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    first = build_product_binding(
        files=files, render_config={"color": "gray"}, occlusion_status="UNKNOWN")
    second = build_product_binding(
        files=files, render_config={"color": "gray"}, occlusion_status="UNKNOWN")
    assert first["signature_sha256"] == second["signature_sha256"]
    files["scene_clean"].write_bytes(b"changed-clean")
    changed = build_product_binding(
        files=files, render_config={"color": "gray"}, occlusion_status="UNKNOWN")
    assert changed["signature_sha256"] != first["signature_sha256"]


def test_s2_side_mapping_uses_bound_semantics_not_legacy_constant():
    from chaoyang.ops.run_human_to_robot_baseline_v1 import validate_s2_side_mapping

    motion = {"human_to_physical": np.asarray([0, 1], dtype=np.int64)}
    binding = {"expected_human_to_physical": [0, 1]}
    evidence = {"human_to_physical": [0, 1], "non_symmetric_test_pass": True}
    validate_s2_side_mapping(motion, binding, evidence)
    with pytest.raises(ValueError, match="S2_SIDE_MAPPING_BINDING_MISMATCH"):
        validate_s2_side_mapping(motion, {"expected_human_to_physical": [1, 0]}, evidence)
    with pytest.raises(ValueError, match="S2_SIDE_MAPPING_ASYMMETRIC_TEST_REQUIRED"):
        validate_s2_side_mapping(
            motion, binding, {**evidence, "non_symmetric_test_pass": False})


def test_formal_s2_entry_rejects_legacy_renderer_or_rgb_overlay_fallback():
    from chaoyang.ops.run_human_to_robot_baseline_v1 import validate_s2_interfaces

    valid = {
        "renderer_interface": "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1",
        "compositor_interface": "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK",
    }
    validate_s2_interfaces(valid)
    with pytest.raises(ValueError, match="S2_RENDERER_INTERFACE_REQUIRED"):
        validate_s2_interfaces({**valid, "renderer_interface": "LEGACY_RGB_MASK"})
    with pytest.raises(ValueError, match="S2_COMPOSITOR_INTERFACE_REQUIRED"):
        validate_s2_interfaces({**valid, "compositor_interface": "RGB_OVERLAY_FALLBACK"})


def test_s2_role_mask_preserves_low_uint16_instance_ids(tmp_path):
    from chaoyang.ops.run_human_to_robot_baseline_v1 import _read_mask

    source = np.zeros((4, 5), dtype=np.uint16)
    source[2, 3] = 1
    path = tmp_path / "mask.png"
    assert cv2.imwrite(str(path), source)
    observed = _read_mask(path, source.shape)
    assert observed.sum() == 1
    assert observed[2, 3]
