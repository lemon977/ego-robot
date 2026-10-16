"""Tianji 2x7 arm + KaiHand kinematics used by the takeover baseline.

Hand-root mount is the frozen visual proxy ``flange_L/R -> hand_l/r_base_link``
(7.52 cm along the flange axis). This equals the mount used by the accepted
Cards042 V2 render; the fresh v1 stage parented the same proxy on ``side_tool``,
which displaced every robot hand 14.5 cm from its arm target.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets
from chaoyang.pipeline.robot_wrist_kai_adapter import final_v3_mano_palm_basis


MOUNT_PROXY = Path("assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/"
                   "MOUNT_VISUAL_PROXY_CONTRACT.json")
FLANGE_MESH = Path("assets/robot/V2相机固定法兰PRO版本.STL")
SIDES = ("left", "right")
SUFFIX = ("L", "R")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_TERMINAL = {"thumb": 6, "index": 4, "middle": 4, "ring": 4, "pinky": 4}
# Links standing in for MANO (base, knuckle, joint) per finger; the tip comes from the mesh.
# Fingers: link2 carries the knuckle flexion axis, link3 PIP, link4 DIP.
# Thumb: link3 metacarpal base, link4 MCP, link6 IP.
KEYPOINT_LINKS = {"thumb": (3, 4, 6), "index": (2, 3, 4), "middle": (2, 3, 4),
                  "ring": (2, 3, 4), "pinky": (2, 3, 4)}
SHOULDER_HEIGHT = 1.121


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.asarray(((t * x * x + c, t * x * y - s * z, t * x * z + s * y),
                       (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
                       (t * x * z - s * y, t * y * z + s * x, t * z * z + c)))


class Chain:
    """Serial FK from a model root to a set of links, indexed by actuated joint order."""

    def __init__(self, model: Any, links: tuple[str, ...]):
        by_child = {joint.child: joint for joint in model.joints}
        order: list[Any] = []
        for link in links:
            path = []
            cursor = link
            while cursor != model.root_link:
                joint = by_child[cursor]
                path.append(joint)
                cursor = joint.parent
            for joint in reversed(path):
                if joint not in order:
                    order.append(joint)
        self.steps = []
        self.joint_names: list[str] = []
        for joint in order:
            index = None
            if joint.joint_type != "fixed":
                index = len(self.joint_names)
                self.joint_names.append(joint.name)
            axis = None if joint.axis is None else np.asarray(joint.axis, float) / np.linalg.norm(joint.axis)
            self.steps.append((joint.parent, joint.child, np.asarray(joint.origin, float), axis, index))
        self.root = model.root_link
        self.lower = np.asarray([next(j.lower for j in model.joints if j.name == n) for n in self.joint_names])
        self.upper = np.asarray([next(j.upper for j in model.joints if j.name == n) for n in self.joint_names])

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        frames = {self.root: np.eye(4)}
        for parent, child, origin, axis, index in self.steps:
            transform = frames[parent] @ origin
            if index is not None:
                motion = np.eye(4)
                motion[:3, :3] = _rodrigues(axis, float(q[index]))
                transform = transform @ motion
            frames[child] = transform
        return frames


@dataclass
class HandContract:
    chain: Chain
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    neutral: np.ndarray
    basis: np.ndarray          # palm basis (lateral, forward, normal) in hand-root frame
    palm_centre: np.ndarray    # midpoint of root origin and MCP centroid, hand-root frame
    mcp_scale: float
    prefix: str
    groups: dict[str, np.ndarray]
    fingers: dict[str, Chain]
    tips: dict[str, np.ndarray]  # fingertip offset in the terminal link frame


class Robot:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.assets = load_pinned_robot_assets(self.repo_root)
        proxy = json.loads((self.repo_root / MOUNT_PROXY).read_text(encoding="utf-8"))["proxy_transforms"]
        if proxy.get("matrix_direction") != "p_flange = T_flange_hand_root_proxy @ p_hand_root":
            raise ValueError("MOUNT_PROXY_DIRECTION_DRIFT")
        self.flange_root = np.asarray((proxy["left_flange_L_to_hand_l_base_link"],
                                       proxy["right_flange_R_to_hand_r_base_link"]), float)
        zero = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
        fk = forward_kinematics(self.assets.tianji, zero)
        # Raster parents the hand on ``<side>_tool``; this mount makes tool @ mount == flange @ proxy.
        self.tool_mounts = np.asarray([np.linalg.inv(fk[f"flange_{SUFFIX[s]}"]) @ fk[f"{SIDES[s]}_tool"]
                                       for s in range(2)])
        self.tool_mounts = np.asarray([np.linalg.inv(self.tool_mounts[s]) @ self.flange_root[s]
                                       for s in range(2)])
        self.arms = []
        for side in range(2):
            suffix = SUFFIX[side]
            chain = Chain(self.assets.tianji, (f"Link2_{suffix}", f"Link4_{suffix}",
                                               f"Link6_{suffix}", f"flange_{suffix}"))
            if tuple(chain.joint_names) != ARM_JOINT_NAMES[side]:
                raise ValueError("ARM_CHAIN_ORDER_DRIFT")
            self.arms.append(chain)
        self.arm_lower = np.asarray([chain.lower for chain in self.arms])
        self.arm_upper = np.asarray([chain.upper for chain in self.arms])
        self.shoulder = np.asarray([self.arms[s].fk(np.zeros(7))[f"Link2_{SUFFIX[s]}"][:3, 3] for s in range(2)])
        self.upper_arm = float(np.linalg.norm(
            self.arms[0].fk(np.zeros(7))["Link4_L"][:3, 3] - self.shoulder[0]))
        self.hands = [self._hand_contract(side) for side in range(2)]

    def arm_frames(self, side: int, q: np.ndarray) -> dict[str, np.ndarray]:
        frames = self.arms[side].fk(q)
        suffix = SUFFIX[side]
        return {"shoulder": frames[f"Link2_{suffix}"], "elbow": frames[f"Link4_{suffix}"],
                "wrist": frames[f"Link6_{suffix}"], "flange": frames[f"flange_{suffix}"],
                "root": frames[f"flange_{suffix}"] @ self.flange_root[side]}

    def _hand_contract(self, side: int) -> HandContract:
        model = (self.assets.left_hand, self.assets.right_hand)[side]
        prefix = ("hand_l", "hand_r")[side]
        links = tuple(f"{prefix}_{finger}_link{k}" for finger in FINGERS
                      for k in range(1, FINGER_TERMINAL[finger] + 1))
        chain = Chain(model, links)
        moving = tuple(j.name for j in model.joints if j.joint_type != "fixed")
        if tuple(chain.joint_names) != moving or len(moving) != 22:
            raise ValueError("KAIHAND_CHAIN_ORDER_DRIFT")
        neutral = 0.5 * (chain.lower + chain.upper)
        rest = np.clip(np.zeros(len(moving)), chain.lower, chain.upper)
        frames = chain.fk(rest)
        groups, fingers, start = {}, {}, 0
        for finger in FINGERS:
            fingers[finger] = Chain(model, tuple(f"{prefix}_{finger}_link{k}"
                                                 for k in range(1, FINGER_TERMINAL[finger] + 1)))
            width = len(fingers[finger].joint_names)
            groups[finger] = np.arange(start, start + width)
            if tuple(moving[start:start + width]) != tuple(fingers[finger].joint_names):
                raise ValueError("KAIHAND_FINGER_GROUP_DRIFT")
            start += width
        tips = {finger: self._tip_offset(model, frames, prefix, finger) for finger in FINGERS}
        knuckles = {finger: frames[f"{prefix}_{finger}_link{KEYPOINT_LINKS[finger][finger == 'thumb']}"][:3, 3]
                    for finger in FINGERS}
        dummy = np.zeros((21, 3))
        for index, finger in ((2, "thumb"), (5, "index"), (9, "middle"), (13, "ring"), (17, "pinky")):
            dummy[index] = knuckles[finger]
        basis = final_v3_mano_palm_basis(dummy, handedness=SIDES[side])
        mcp = np.asarray([knuckles[finger] for finger in FINGERS[1:]])
        return HandContract(chain=chain, names=moving, lower=chain.lower, upper=chain.upper,
                            neutral=neutral, basis=basis, palm_centre=0.5 * mcp.mean(axis=0),
                            mcp_scale=float(np.mean(np.linalg.norm(mcp @ basis, axis=1))),
                            prefix=prefix, groups=groups, fingers=fingers, tips=tips)

    @staticmethod
    def _tip_offset(model: Any, frames: dict[str, np.ndarray], prefix: str, finger: str) -> np.ndarray:
        """Distal-most mesh point of the terminal link, along its last phalanx, in that link's frame."""
        import trimesh

        links = KEYPOINT_LINKS[finger]
        terminal = f"{prefix}_{finger}_link{links[-1]}"
        before = f"{prefix}_{finger}_link{links[-2]}"
        visual = next(v for v in model.visuals if v.link == terminal)
        mesh = trimesh.load(visual.mesh_path, force="mesh")
        vertices = np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ (frames[terminal] @ visual.origin).T
        start, end = frames[before][:3, 3], frames[terminal][:3, 3]
        direction = (end - start) / np.linalg.norm(end - start)
        reach = float(np.max((vertices[:, :3] - end) @ direction))
        tip_root = end + reach * direction
        return (np.linalg.inv(frames[terminal]) @ np.r_[tip_root, 1.0])[:3]

    def finger_points(self, side: int, finger: str, q_finger: np.ndarray) -> np.ndarray:
        """Hand-root-frame anatomical points matching MANO (wrist, base, knuckle, joint, tip)."""
        contract = self.hands[side]
        frames = contract.fingers[finger].fk(q_finger)
        links = KEYPOINT_LINKS[finger]
        points = [np.zeros(3)] + [frames[f"{contract.prefix}_{finger}_link{k}"][:3, 3] for k in links]
        terminal = frames[f"{contract.prefix}_{finger}_link{links[-1]}"]
        points.append(terminal[:3, :3] @ contract.tips[finger] + terminal[:3, 3])
        return np.asarray(points)
