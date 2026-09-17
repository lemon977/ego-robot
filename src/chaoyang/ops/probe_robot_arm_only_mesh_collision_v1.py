#!/usr/bin/env python3
"""Audit one observed pose-only q against Tianji arm mesh, without hands."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import audit_robot_geometry_self_collision_v71 as collision  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402


def _closed(reference: dict) -> Path:
    path = Path(reference["path"])
    actual = artifact_ref(path)
    if actual["bytes"] != reference["bytes"] or actual["sha256"] != reference["sha256"]:
        raise RuntimeError(f"input SHA conflict: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-result", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("output must be fresh")
    branch = json.loads(args.branch_result.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    if branch["session_id"] != pilot["session_id"] or branch["source_frame"] != 0:
        raise RuntimeError("branch/pilot session or frame mismatch")
    if branch["inputs"]["pilot_result"]["sha256"] != artifact_ref(args.pilot_result)["sha256"]:
        raise RuntimeError("branch result does not bind pilot")
    if pilot["inputs"]["preset"]["sha256"] != artifact_ref(args.preset)["sha256"]:
        raise RuntimeError("pilot preset SHA mismatch")
    with np.load(_closed(pilot["outputs"]["states"]), allow_pickle=False) as loaded:
        frozen = {name: np.asarray(loaded[name]) for name in loaded.files}
    targets = np.asarray(frozen["T_target_hand_root_world"], np.float64)
    midpoint = 0.5 * (targets[0, 0, :3, 3] + targets[0, 1, :3, 3])
    frozen_base = np.asarray(frozen["T_world_base"], np.float64)
    base_targets = np.asarray(preset["base_midpoint_target_candidates_m"], np.float64)
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)

    import pybullet as bullet  # noqa: PLC0415

    client = bullet.connect(bullet.DIRECT)
    try:
        flags = bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        body = bullet.loadURDF(
            str(collision.TIANJI.resolve()), useFixedBase=True,
            flags=flags, physicsClientId=client,
        )
        names = collision._link_names(bullet, body, client)
        joints = collision._joint_indices(bullet, body, client)
        link_shapes = {
            names[index]: len(bullet.getCollisionShapeData(body, index, physicsClientId=client))
            for index in names
        }
        arm_link_shapes = {
            ("left", "right")[side]: {
                f"Link{number}_{('L', 'R')[side]}": link_shapes.get(
                    f"Link{number}_{('L', 'R')[side]}", 0
                ) for number in range(1, 8)
            } for side in range(2)
        }
        if any(count == 0 for links in arm_link_shapes.values() for count in links.values()):
            raise RuntimeError("an arm link lacks collision geometry")
        rows = []
        for candidate_index in (0, 2):
            world_base = frozen_base.copy()
            world_base[:3, 3] = midpoint - world_base[:3, :3] @ base_targets[candidate_index]
            position, orientation = collision._transform_pose(world_base)
            bullet.resetBasePositionAndOrientation(
                body, position, orientation, physicsClientId=client
            )
            pose_rows = []
            for side in range(2):
                source = next(
                    row for row in branch["rows"]
                    if row["base_candidate_index"] == candidate_index
                    and row["side"] == ("left", "right")[side]
                )
                chosen = source["pose_only_best"]
                q = np.asarray(chosen["q_arm"], np.float64)
                if q.shape != (7,) or np.any(q < lower[side] - 1e-9) or np.any(q > upper[side] + 1e-9):
                    raise RuntimeError("pose-only q violates URDF limits")
                for joint_name, value in zip(collision.ARM_NAMES[side], q, strict=True):
                    bullet.resetJointState(body, joints[joint_name], float(value), physicsClientId=client)
                pose_rows.append({
                    "side": ("left", "right")[side],
                    "position_mm": chosen["position_mm"],
                    "rotation_deg": chosen["rotation_deg"],
                    "legacy_branch_clearance_m_min": chosen["branch_clearance_m_min"],
                    "within_urdf_limits": True,
                })
            bullet.performCollisionDetection(physicsClientId=client)
            contacts = [
                {
                    "link_a": names.get(point[3], str(point[3])),
                    "link_b": names.get(point[4], str(point[4])),
                    "penetration_depth_m": float(-point[8]),
                }
                for point in bullet.getContactPoints(body, body, physicsClientId=client)
                if -point[8] > 1e-4
            ]
            rows.append({
                "base_candidate_index": candidate_index,
                "side_poses": pose_rows,
                "arm_only_illegal_contact_count": len(contacts),
                "arm_only_illegal_contacts": contacts,
            })
    finally:
        bullet.disconnect(client)
    result = {
        "schema_version": "rc1-arm-only-mesh-collision-probe-v1",
        "created_at": now_iso(),
        "status": "COMPLETED_DIAGNOSTIC",
        "session_id": pilot["session_id"],
        "source_frame": 0,
        "input_mode": "OFFLINE_VISUAL",
        "quality_status_unchanged": pilot["status"],
        "training_eligible": False,
        "authority": False,
        "engine": "PYBULLET_TIANJI_URDF_ARM_ONLY",
        "penetration_tolerance_m": 1e-4,
        "arm_link_collision_shape_counts": arm_link_shapes,
        "inputs": {
            "branch_result": artifact_ref(args.branch_result),
            "pilot_result": artifact_ref(args.pilot_result),
            "preset": artifact_ref(args.preset),
            "tianji_urdf": artifact_ref(collision.TIANJI),
            "robot_asset_pin": artifact_ref(ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
            "code": artifact_ref(Path(__file__)),
        },
        "rows": rows,
        "claim_limit": "Only Tianji arm-body contacts at one source frame are tested. KaiHand, adapter, robot-object collision, trajectory, target-pose hard pass and physical accuracy remain untested. A zero count cannot promote the quality-C pilot or replace the full digital collision audit.",
    }
    atomic_json(args.output, result)
    print(f"{pilot['session_id']}: {len(rows)} arm-only candidate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
