#!/usr/bin/env python3
from __future__ import annotations

"""Audit real Tianji/KaiHand URDF collision geometry on pinned trajectories.

This is the V7.1 R1 self-intersection half-gate.  It is deliberately separate
from camera z-buffer/occlusion and from robot-object contact.  Passing this
tool never grants Robot, control, contact, or physical deployment authority.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
TIANJI = ROOT / "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf"
HANDS = (
    ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)/urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
)
ARM_NAMES = tuple(tuple(f"Joint{index}_{suffix}" for index in range(1, 8)) for suffix in ("L", "R"))
SIDES = ("left", "right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_uniform_valid_frames(valid_side_frame: np.ndarray, count: int) -> list[int]:
    valid = np.asarray(valid_side_frame, dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError("valid_side_frame must be [2,T]")
    eligible = np.flatnonzero(np.all(valid, axis=0))
    if len(eligible) < count or count <= 0:
        raise ValueError(f"need {count} bilateral valid frames; found {len(eligible)}")
    indices = np.rint(np.linspace(0, len(eligible) - 1, count)).astype(int)
    selected = [int(eligible[index]) for index in indices]
    if len(set(selected)) != count:
        raise ValueError("uniform frame selection produced duplicates")
    return selected


def _transform_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("pose must be finite 4x4")
    if not np.allclose(value[3], (0, 0, 0, 1), atol=1e-9):
        raise ValueError("pose homogeneous row invalid")
    rotation = value[:3, :3]
    if np.linalg.det(rotation) < 0.999 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("pose rotation is not proper orthonormal")
    return value[:3, 3].tolist(), Rotation.from_matrix(rotation).as_quat().tolist()


def _link_names(bullet: Any, body: int, client: int) -> dict[int, str]:
    names = {-1: bullet.getBodyInfo(body, physicsClientId=client)[0].decode("utf-8")}
    for index in range(bullet.getNumJoints(body, physicsClientId=client)):
        names[index] = bullet.getJointInfo(body, index, physicsClientId=client)[12].decode("utf-8")
    return names


def _joint_indices(bullet: Any, body: int, client: int) -> dict[str, int]:
    return {
        bullet.getJointInfo(body, index, physicsClientId=client)[1].decode("utf-8"): index
        for index in range(bullet.getNumJoints(body, physicsClientId=client))
    }


def is_allowed_mount_contact(
    *, tianji_link: str, hand_link_index: int, side: str
) -> bool:
    suffix = "L" if side == "left" else "R"
    tool = "left_tool" if side == "left" else "right_tool"
    return hand_link_index == -1 and tianji_link in {f"Link7_{suffix}", f"flange_{suffix}", tool}


def _collision_shape_count(bullet: Any, body: int, client: int) -> int:
    return sum(
        bool(bullet.getCollisionShapeData(body, link, physicsClientId=client))
        for link in range(-1, bullet.getNumJoints(body, physicsClientId=client))
    )


def audit(
    *, session_id: str, arm_path: Path, hand_path: Path, frame_count: int,
    penetration_tolerance_m: float,
) -> dict[str, Any]:
    import pybullet as bullet

    with np.load(arm_path, allow_pickle=False) as source:
        arm = {key: np.asarray(source[key]) for key in source.files}
    with np.load(hand_path, allow_pickle=False) as source:
        hand = {key: np.asarray(source[key]) for key in source.files}
    q_arm = np.asarray(arm["q_arm"], dtype=np.float64)
    q_hand = np.asarray(hand["q_hand"], dtype=np.float64)
    valid_arm = np.asarray(arm["valid_side_frame"], dtype=bool)
    valid_hand = np.asarray(hand["valid_side_frame"], dtype=bool)
    if q_arm.ndim != 3 or q_arm.shape[1:] != (2, 7):
        raise ValueError(f"q_arm must be [T,2,7], got {q_arm.shape}")
    if q_hand.shape != (len(q_arm), 2, 22):
        raise ValueError(f"q_hand must be [T,2,22], got {q_hand.shape}")
    if valid_arm.shape != (2, len(q_arm)) or not np.array_equal(valid_arm, valid_hand):
        raise ValueError("arm/hand validity mismatch")
    if not np.array_equal(arm["source_frames"], hand["source_frames"]):
        raise ValueError("arm/hand source frame identity mismatch")
    roots = np.asarray(arm["T_actual_hand_root_world"], dtype=np.float64)
    if roots.shape != (len(q_arm), 2, 4, 4):
        raise ValueError("T_actual_hand_root_world shape mismatch")
    frames = select_uniform_valid_frames(valid_arm, frame_count)

    client = bullet.connect(bullet.DIRECT)
    try:
        flags = bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        robot = bullet.loadURDF(str(TIANJI.resolve()), useFixedBase=True, flags=flags, physicsClientId=client)
        hands = [
            bullet.loadURDF(str(path.resolve()), useFixedBase=True, flags=flags, physicsClientId=client)
            for path in HANDS
        ]
        bodies = [robot, *hands]
        names = [_link_names(bullet, body, client) for body in bodies]
        collision_shapes = [_collision_shape_count(bullet, body, client) for body in bodies]
        if any(count == 0 for count in collision_shapes):
            raise ValueError(f"URDF body lacks collision geometry: {collision_shapes}")
        robot_joints = _joint_indices(bullet, robot, client)
        hand_joints = [_joint_indices(bullet, body, client) for body in hands]
        if any(len(indices) != 22 for indices in hand_joints):
            raise ValueError("KaiHand moving-joint closure is not 22")

        base_position, base_orientation = _transform_pose(np.asarray(arm["T_world_base"]))
        bullet.resetBasePositionAndOrientation(robot, base_position, base_orientation, physicsClientId=client)
        frame_rows = []
        global_contacts: list[dict[str, Any]] = []
        for frame in frames:
            for side in range(2):
                for joint_name, value in zip(ARM_NAMES[side], q_arm[frame, side], strict=True):
                    bullet.resetJointState(robot, robot_joints[joint_name], float(value), physicsClientId=client)
                position, orientation = _transform_pose(roots[frame, side])
                bullet.resetBasePositionAndOrientation(hands[side], position, orientation, physicsClientId=client)
                ordered = sorted(hand_joints[side].items(), key=lambda item: item[1])
                for (_, joint_index), value in zip(ordered, q_hand[frame, side], strict=True):
                    bullet.resetJointState(hands[side], joint_index, float(value), physicsClientId=client)
            bullet.performCollisionDetection(physicsClientId=client)
            candidates: list[tuple[str, int, int, str, str, float]] = []
            for body_index, body in enumerate(bodies):
                for point in bullet.getContactPoints(body, body, physicsClientId=client):
                    candidates.append((f"self_{body_index}", body_index, body_index, names[body_index].get(point[3], str(point[3])), names[body_index].get(point[4], str(point[4])), float(point[8])))
            for first in range(len(bodies)):
                for second in range(first + 1, len(bodies)):
                    for point in bullet.getContactPoints(bodies[first], bodies[second], physicsClientId=client):
                        link_a = names[first].get(point[3], str(point[3]))
                        link_b = names[second].get(point[4], str(point[4]))
                        if first == 0 and second in (1, 2):
                            side = SIDES[second - 1]
                            if is_allowed_mount_contact(tianji_link=link_a, hand_link_index=int(point[4]), side=side):
                                continue
                        candidates.append((f"cross_{first}_{second}", first, second, link_a, link_b, float(point[8])))
            contacts = []
            seen = set()
            for pair, first, second, link_a, link_b, distance in candidates:
                penetration = max(0.0, -distance)
                key = (pair, link_a, link_b, round(distance, 12))
                if penetration <= penetration_tolerance_m or key in seen:
                    continue
                seen.add(key)
                contacts.append({
                    "pair": pair, "body_a": first, "body_b": second,
                    "link_a": link_a, "link_b": link_b,
                    "contact_distance_m": distance, "penetration_depth_m": penetration,
                })
            global_contacts.extend({"frame_id": frame, **item} for item in contacts)
            frame_rows.append({
                "frame_id": frame,
                "illegal_contact_count": len(contacts),
                "max_penetration_m": max((item["penetration_depth_m"] for item in contacts), default=0.0),
                "contacts": contacts,
            })
    finally:
        bullet.disconnect(client)

    passed = not global_contacts
    return {
        "schema_version": "robot-geometry-self-collision-audit-v71",
        "artifact_revision": "R7_1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": session_id,
        "status": "PASS_4FRAME_DEVELOPMENT_COLLISION_AUDIT" if passed and frame_count == 4 else ("PASS_DEVELOPMENT_COLLISION_AUDIT" if passed else "FAILED_QUALITY_C"),
        "selected_frames": frames,
        "collision_engine": "PYBULLET_URDF_COLLISION_GEOMETRY",
        "collision_shape_counts": {"tianji": collision_shapes[0], "kaihand_left": collision_shapes[1], "kaihand_right": collision_shapes[2]},
        "penetration_tolerance_m": penetration_tolerance_m,
        "illegal_contact_count": len(global_contacts),
        "max_penetration_m": max((item["penetration_depth_m"] for item in global_contacts), default=0.0),
        "frames": frame_rows,
        "inputs": {"arm_states": ref(arm_path), "hand_states": ref(hand_path), "tianji_urdf": ref(TIANJI), "kaihand_urdfs": [ref(path) for path in HANDS]},
        "gates": {"real_collision_geometry_loaded": True, "non_adjacent_self_intersection_absent": passed},
        "next_action": "RUN_24FRAME_AND_UNIFIED_ZBUFFER_AUDIT" if passed and frame_count == 4 else ("RUN_UNIFIED_ZBUFFER_AUDIT" if passed else "STOP_AND_DIAGNOSE_COLLISION_PAIRS"),
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Development collision audit of pinned digital URDF trajectories only. It does not include object contact or camera z-buffer and is not Robot/control/physical authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--arm-states", type=Path, required=True)
    parser.add_argument("--hand-states", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--penetration-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    value = audit(
        session_id=args.session_id,
        arm_path=args.arm_states.resolve(strict=True),
        hand_path=args.hand_states.resolve(strict=True),
        frame_count=args.frame_count,
        penetration_tolerance_m=args.penetration_tolerance_m,
    )
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "RESULT.json", value)
    print(json.dumps({"status": value["status"], "illegal_contact_count": value["illegal_contact_count"], "output": str((args.output_dir / 'RESULT.json').resolve())}, ensure_ascii=False))
    return 0 if value["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
