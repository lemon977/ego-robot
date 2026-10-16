"""V5 offline robot-to-clean-image compositor; no geometry is read from Clean.

The optical projection for unrectified 0915 RGB is an explicit factory-K
approximation, not an externally validated registration or contact authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ROBOT_COMPONENT_BACKGROUND = 0
ROBOT_COMPONENT_ARM = 1
ROBOT_COMPONENT_ADAPTER_LEFT = 2
ROBOT_COMPONENT_ADAPTER_RIGHT = 3
ROBOT_COMPONENT_HAND_LEFT = 4
ROBOT_COMPONENT_HAND_RIGHT = 5


@dataclass(frozen=True)
class RobotRenderLayers:
    """Renderer output used by S2 products; depth is physical camera optical-Z."""

    rgb: np.ndarray
    alpha: np.ndarray
    optical_depth_m: np.ndarray
    depth_valid: np.ndarray
    component_id: np.ndarray
    T_world_flange: np.ndarray
    T_world_adapter: np.ndarray
    T_world_hand: np.ndarray
    body_id: np.ndarray
    link_index: np.ndarray
    link_frames_camera: dict[tuple[int, int], np.ndarray]
    link_names: dict[tuple[int, int], str]


def visible_robot_mask(segmentation: np.ndarray, robot: int, hands: list[int],
                       arm_links: list[set[int]], valid_sides: list[bool],
                       adapters: list[int] | None = None) -> np.ndarray:
    """Return only robot pixels backed by valid per-side motion.

    PyBullet packs ``objectUniqueId + ((linkIndex + 1) << 24)`` in its
    segmentation image. Invalid hands and all descendant links of an invalid
    arm are removed instead of visualizing default/zero poses as observations.
    """
    value = np.asarray(segmentation, dtype=np.int64)
    body = value & ((1 << 24) - 1)
    link = (value >> 24) - 1
    adapters = adapters or []
    mask = (value >= 0) & np.isin(body, [robot, *hands, *adapters])
    for side, valid in enumerate(valid_sides):
        if valid:
            continue
        mask &= body != hands[side]
        if adapters:
            mask &= body != adapters[side]
        mask &= ~((body == robot) & np.isin(link, tuple(arm_links[side])))
    return mask


def hand_root_from_flange(T_world_flange: np.ndarray,
                          T_flange_hand: np.ndarray) -> np.ndarray:
    """Return the rendered hand root from actual arm FK, never from a target wrist."""
    flange = np.asarray(T_world_flange, dtype=np.float64)
    mount = np.asarray(T_flange_hand, dtype=np.float64)
    if flange.shape != (4, 4) or mount.shape != (4, 4):
        raise ValueError("HAND_ROOT_TRANSFORM_SHAPE")
    if not np.isfinite(flange).all() or not np.isfinite(mount).all():
        raise ValueError("HAND_ROOT_TRANSFORM_NONFINITE")
    return flange @ mount


def arm_state_for_render(q: np.ndarray, valid: bool,
                         neutral: np.ndarray) -> np.ndarray:
    """Return finite arm state while keeping a missing side non-observed."""
    value = np.asarray(q, dtype=np.float64)
    fallback = np.asarray(neutral, dtype=np.float64)
    if value.shape != (7,) or fallback.shape != (7,) or not np.isfinite(fallback).all():
        raise ValueError("ARM_RENDER_STATE_SHAPE_OR_NEUTRAL")
    if not valid:
        return fallback.copy()
    if not np.isfinite(value).all():
        raise ValueError("VALID_ARM_RENDER_STATE_NONFINITE")
    return value.copy()


def optical_depth_from_buffer(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert OpenGL depth buffer to camera optical-Z in metres."""
    value = np.asarray(depth_buffer, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("ROBOT_DEPTH_BUFFER_INVALID")
    if not (0.0 < near < far):
        raise ValueError("ROBOT_DEPTH_CLIP_INVALID")
    return far * near / (far - (far - near) * value)


def camera_intrinsics(domain: dict) -> tuple[np.ndarray, str]:
    if "K" in domain:
        k = np.asarray(domain["K"], dtype=np.float64)
        method = "EXPLICIT_DOMAIN_K"
    else:
        import json
        from pathlib import Path

        source = domain.get("source_camera") or {}
        path = Path(source.get("path", ""))
        if not path.is_file():
            raise ValueError("CAMERA_INTRINSICS_UNAVAILABLE")
        camera = json.loads(path.read_text(encoding="utf-8"))
        side = "left" if domain.get("source_index") == 1 else "right"
        native = camera[side]["intrinsics"]
        sx = domain["width"] / camera["width"]
        sy = domain["height"] / camera["height"]
        k = np.asarray([[native["fx"] * sx, 0, native["cx"] * sx],
                        [0, native["fy"] * sy, native["cy"] * sy],
                        [0, 0, 1]], dtype=np.float64)
        method = "SAME_SESSION_FACTORY_K_SCALED_UNRECTIFIED_VISUAL_APPROXIMATION"
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("CAMERA_INTRINSICS_INVALID")
    return k, method


def camera_matrices(bullet, t_camera_base: np.ndarray, k: np.ndarray,
                    width: int, height: int) -> tuple[list[float], list[float]]:
    t = np.asarray(t_camera_base, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("T_CAMERA_BASE_INVALID")
    t_base_camera = np.linalg.inv(t)
    eye = t_base_camera[:3, 3]
    rotation = t_base_camera[:3, :3]
    front = eye + rotation @ np.asarray([0.0, 0.0, 1.0])
    up = rotation @ np.asarray([0.0, -1.0, 0.0])
    view = bullet.computeViewMatrix(eye.tolist(), front.tolist(), up.tolist())
    near, far = 0.02, 20.0
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    projection = bullet.computeProjectionMatrix(
        -cx * near / fx, (width - cx) * near / fx,
        -(height - cy) * near / fy, cy * near / fy, near, far)
    return view, projection


class ProductRobotRenderer:
    """PyBullet TinyRenderer with per-pixel robot segmentation, no arm IK."""

    def __init__(self, project_root, motion: dict, domain: dict, *, include_adapter: bool = True):
        import pybullet as bullet
        from chaoyang.pipeline.robot_renderer_eevee_fullchain import (
            ARM_JOINT_NAMES, load_pinned_robot_assets,
        )
        from chaoyang.ops.render_tianji_kai_mount_proxy_audit import place_hand

        self.b = bullet
        self.project_root = Path(project_root).resolve()
        self.client = bullet.connect(bullet.DIRECT)
        self.assets = load_pinned_robot_assets(project_root)
        self.motion = motion
        self.arm_names = ARM_JOINT_NAMES
        self.place_hand = place_hand
        self.width, self.height = int(domain["width"]), int(domain["height"])
        self.k, self.intrinsics_method = camera_intrinsics(domain)
        self.T_camera_base = np.asarray(motion["T_cam_base"], dtype=np.float64)
        self.near, self.far = 0.02, 20.0
        self.view, self.projection = camera_matrices(
            bullet, motion["T_cam_base"], self.k, self.width, self.height)
        flags = bullet.URDF_USE_SELF_COLLISION | bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.robot = bullet.loadURDF(str(self.assets.tianji.path), useFixedBase=True,
                                     flags=flags, physicsClientId=self.client)
        self.hands = [bullet.loadURDF(str(model.path), useFixedBase=True,
                                      flags=flags, physicsClientId=self.client)
                      for model in (self.assets.left_hand, self.assets.right_hand)]
        mount_path = self.project_root / (
            "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
            "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
        )
        mount = json.loads(mount_path.read_text(encoding="utf-8"))
        if mount.get("classification") != "REAL_CAD_GEOMETRY_WITH_UNMEASURED_VIRTUAL_INSTALLATION":
            raise ValueError("ADAPTER_MOUNT_CONTRACT_CLASSIFICATION")
        mesh = Path(mount["decoded_review_mesh"]["path"]).resolve(strict=True)
        scale = float(mount["decoded_review_mesh"]["mesh_scale_to_metre"])
        self.T_flange_adapter = np.asarray([
            mount["transforms"]["left_flange_to_adapter"],
            mount["transforms"]["right_flange_to_adapter"],
        ], dtype=np.float64)
        contract_hand = np.asarray([
            mount["transforms"]["left_flange_to_hand_root"],
            mount["transforms"]["right_flange_to_hand_root"],
        ], dtype=np.float64)
        motion_hand = np.asarray(motion["T_flange_hand"], dtype=np.float64)
        if contract_hand.shape != (2, 4, 4) or not np.allclose(
                contract_hand, motion_hand, rtol=0.0, atol=1e-10):
            raise ValueError("MOTION_MOUNT_CONTRACT_MISMATCH")
        self.include_adapter = bool(include_adapter)
        if self.include_adapter:
            visual = bullet.createVisualShape(
                bullet.GEOM_MESH, fileName=str(mesh), meshScale=[scale, scale, scale],
                rgbaColor=(0.40, 0.43, 0.47, 1.0), physicsClientId=self.client)
            self.adapters = [
                bullet.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual,
                                       physicsClientId=self.client)
                for _ in range(2)
            ]
        else:
            self.adapters = []
        self.adapter_collision_status = "UNVERIFIED_VISUAL_GEOMETRY_ONLY"
        self.mount_contract_path = mount_path
        self.maps = {}
        for body in (self.robot, *self.hands):
            self.maps[body] = {
                bullet.getJointInfo(body, i, physicsClientId=self.client)[1].decode(): i
                for i in range(bullet.getNumJoints(body, physicsClientId=self.client))
            }
        self.hand_names = [tuple(j.name for j in model.joints if j.joint_type != "fixed")
                           for model in (self.assets.left_hand, self.assets.right_hand)]
        arm_by_name = {joint.name: joint for joint in self.assets.tianji.joints}
        self.arm_neutral = np.asarray([
            [0.5 * (arm_by_name[name].lower + arm_by_name[name].upper) for name in names]
            for names in self.arm_names
        ], dtype=np.float64)
        parent = {}
        for index in range(bullet.getNumJoints(self.robot, physicsClientId=self.client)):
            info = bullet.getJointInfo(self.robot, index, physicsClientId=self.client)
            parent[index] = int(info[16])
        self.arm_links = []
        for side in range(2):
            roots = {self.maps[self.robot][name] for name in self.arm_names[side]}
            descendants = set(roots)
            changed = True
            while changed:
                changed = False
                for index, parent_index in parent.items():
                    if parent_index in descendants and index not in descendants:
                        descendants.add(index)
                        changed = True
            self.arm_links.append(descendants)
        for link in range(-1, bullet.getNumJoints(self.robot, physicsClientId=self.client)):
            bullet.changeVisualShape(self.robot, link, rgbaColor=(0.72, 0.76, 0.81, 1),
                                     physicsClientId=self.client)
        for hand in self.hands:
            for link in range(-1, bullet.getNumJoints(hand, physicsClientId=self.client)):
                bullet.changeVisualShape(hand, link, rgbaColor=(0.94, 0.94, 0.92, 1),
                                         physicsClientId=self.client)

    def _set_joints(self, body: int, names, values) -> None:
        for name, value in zip(names, values, strict=True):
            self.b.resetJointState(body, self.maps[body][name], float(value),
                                   physicsClientId=self.client)

    def frame_layers(self, index: int) -> RobotRenderLayers:
        from chaoyang.ops.render_tianji_kai_mount_proxy_audit import link_frames

        m, b = self.motion, self.b
        q_arm = m["q_arm"][index]
        q_hand = m["q_hand22"][index]
        valid_sides = [bool(m["wrist_valid"][index, side] and m["finger_valid"][index, side])
                       for side in range(2)]
        for side in range(2):
            state = arm_state_for_render(q_arm[side], valid_sides[side], self.arm_neutral[side])
            self._set_joints(self.robot, self.arm_names[side], state)
        fk = link_frames(self.client, self.robot)
        T_world_flange = np.full((2, 4, 4), np.nan, dtype=np.float64)
        T_world_adapter = np.full((2, 4, 4), np.nan, dtype=np.float64)
        T_world_hand = np.full((2, 4, 4), np.nan, dtype=np.float64)
        for side, hand in enumerate(self.hands):
            valid = valid_sides[side]
            if valid:
                flange = fk[("flange_L", "flange_R")[side]]
                root = hand_root_from_flange(flange, m["T_flange_hand"][side])
                adapter_root = flange @ self.T_flange_adapter[side]
                T_world_flange[side] = flange
                T_world_adapter[side] = adapter_root
                T_world_hand[side] = root
            else:
                # Invalid input must not be visualized as an observed hand.
                root = np.eye(4, dtype=np.float64)
                root[0, 3] = 100.0 + side
                adapter_root = root.copy()
            self.place_hand(self.client, hand, root)
            if self.include_adapter:
                self.place_hand(self.client, self.adapters[side], adapter_root)
            if valid:
                self._set_joints(hand, self.hand_names[side], q_hand[side])
        result = b.getCameraImage(
            self.width, self.height, self.view, self.projection,
            renderer=b.ER_TINY_RENDERER,
            flags=b.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            shadow=0, physicsClientId=self.client)
        rgba = np.asarray(result[2], dtype=np.uint8).reshape(self.height, self.width, 4)
        depth_buffer = np.asarray(result[3], dtype=np.float64).reshape(self.height, self.width)
        segmentation = np.asarray(result[4], dtype=np.int32).reshape(self.height, self.width)
        robot_pixel = visible_robot_mask(segmentation, self.robot, self.hands,
                                         self.arm_links, valid_sides, self.adapters)
        body = np.asarray(segmentation, dtype=np.int64) & ((1 << 24) - 1)
        component = np.zeros(robot_pixel.shape, dtype=np.uint8)
        component[robot_pixel & (body == self.robot)] = ROBOT_COMPONENT_ARM
        if self.include_adapter:
            component[robot_pixel & (body == self.adapters[0])] = ROBOT_COMPONENT_ADAPTER_LEFT
            component[robot_pixel & (body == self.adapters[1])] = ROBOT_COMPONENT_ADAPTER_RIGHT
        component[robot_pixel & (body == self.hands[0])] = ROBOT_COMPONENT_HAND_LEFT
        component[robot_pixel & (body == self.hands[1])] = ROBOT_COMPONENT_HAND_RIGHT
        depth = optical_depth_from_buffer(depth_buffer, self.near, self.far)
        depth[~robot_pixel] = np.nan
        visible_body = np.where(robot_pixel, body, -1).astype(np.int32)
        visible_link = np.where(robot_pixel, (np.asarray(segmentation, dtype=np.int64) >> 24) - 1, -2).astype(np.int16)
        link_frames_camera: dict[tuple[int, int], np.ndarray] = {}
        link_names: dict[tuple[int, int], str] = {}
        for body_id in (self.robot, *self.hands, *self.adapters):
            visible_indices = np.unique(visible_link[visible_body == body_id])
            if not len(visible_indices):
                continue
            names_to_frames = link_frames(self.client, body_id)
            for link_id in visible_indices:
                link_id = int(link_id)
                name = ("base_link" if link_id == -1 else
                        b.getJointInfo(body_id, link_id, physicsClientId=self.client)[12].decode())
                if name not in names_to_frames:
                    raise ValueError(f"MISSING_LINK_FRAME:{body_id}:{link_id}:{name}")
                key = (int(body_id), link_id)
                link_names[key] = name
                link_frames_camera[key] = self.T_camera_base @ names_to_frames[name]
        return RobotRenderLayers(
            rgb=rgba[..., :3].copy(), alpha=robot_pixel,
            optical_depth_m=depth, depth_valid=robot_pixel.copy(), component_id=component,
            T_world_flange=T_world_flange, T_world_adapter=T_world_adapter,
            T_world_hand=T_world_hand,
            body_id=visible_body, link_index=visible_link,
            link_frames_camera=link_frames_camera, link_names=link_names,
        )

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility wrapper. S2 products must consume ``frame_layers``."""
        layers = self.frame_layers(index)
        return layers.rgb, layers.alpha

    def close(self) -> None:
        self.b.disconnect(self.client)


def composite(clean_bgr: np.ndarray, robot_rgb: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    if clean_bgr.shape[:2] != robot_mask.shape or robot_rgb.shape[:2] != robot_mask.shape:
        raise ValueError("COMPOSITOR_DOMAIN_MISMATCH")
    result = clean_bgr.copy()
    result[robot_mask] = robot_rgb[robot_mask, ::-1]
    return result
