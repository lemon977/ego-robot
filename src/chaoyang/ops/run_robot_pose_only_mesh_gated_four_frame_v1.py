#!/usr/bin/env python3
"""Four-frame arm solver canary: preserve hard gates, replace branch proxy with mesh QA."""

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
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import audit_robot_geometry_self_collision_v71 as collision  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.ops import run_robot_pose_only_independent_arm_v6 as old_arm  # noqa: E402


def _closed(ref: dict) -> Path:
    path = Path(ref["path"])
    current = artifact_ref(path)
    if current["bytes"] != ref["bytes"] or current["sha256"] != ref["sha256"]:
        raise RuntimeError(f"input SHA conflict: {path}")
    return path


def _solve_pose_only(
    assets, side, target, lower, upper, seeds, previous, lo, hi,
):
    free = hi - lo > 1e-12
    attempts = []
    for seed_index, seed in enumerate(seeds):
        q = np.clip(np.asarray(seed, np.float64), lo, hi)

        def residual(candidate):
            pose = arm.old.official._pose_residual(
                arm.old.official._tool_fk(assets, side, candidate), target
            )
            continuity = np.empty(0) if previous is None else 0.01 * (candidate - previous)
            return np.concatenate((pose, continuity))

        if np.any(free):
            solution = least_squares(
                lambda values: residual(old_arm._fill_free(q, free, values)),
                q[free], bounds=(lo[free], hi[free]), max_nfev=500,
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            q = old_arm._fill_free(q, free, solution.x)
        position_mm, rotation_deg = arm.arm.error(assets, side, q, target)
        branch_clearance = float(np.min(arm.arm.margins(assets, side, q)))
        passed = position_mm <= 10.0 and rotation_deg <= 5.0
        score = (
            not passed,
            max(0.0, position_mm - 10.0) + max(0.0, rotation_deg - 5.0),
            position_mm + rotation_deg,
        )
        attempts.append((score, q, seed_index, position_mm, rotation_deg, branch_clearance))
    return min(attempts, key=lambda item: item[0])


def _arm_mesh_audit(q: np.ndarray, world_base: np.ndarray):
    import pybullet as bullet  # noqa: PLC0415

    client = bullet.connect(bullet.DIRECT)
    try:
        body = bullet.loadURDF(
            str(collision.TIANJI.resolve()), useFixedBase=True,
            flags=bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, physicsClientId=client,
        )
        names = collision._link_names(bullet, body, client)
        joint_indices = collision._joint_indices(bullet, body, client)
        for side in range(2):
            for index in range(1, 8):
                link = f"Link{index}_{('L', 'R')[side]}"
                indices = [number for number, name in names.items() if name == link]
                if len(indices) != 1 or not bullet.getCollisionShapeData(
                    body, indices[0], physicsClientId=client
                ):
                    raise RuntimeError(f"arm collision shape missing: {link}")
        position, orientation = collision._transform_pose(world_base)
        bullet.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=client)
        rows = []
        for frame in range(len(q)):
            for side in range(2):
                for name, value in zip(collision.ARM_NAMES[side], q[frame, side], strict=True):
                    bullet.resetJointState(
                        body, joint_indices[name], float(value), physicsClientId=client
                    )
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
                "frame": frame,
                "arm_only_illegal_contact_count": len(contacts),
                "contacts": contacts,
            })
        return rows
    finally:
        bullet.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=8)
    parser.add_argument("--wall-cap-s", type=int, default=900)
    args = parser.parse_args()
    if args.output_root.exists() or not 0 <= args.random_seeds <= 30:
        raise RuntimeError("fresh output root and bounded random starts required")
    if not 0 < args.wall_cap_s <= 1800:
        raise RuntimeError("wall cap out of range")
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    if pilot["status"] not in {"FAILED_QUALITY_C", "PASSED_DEVELOPMENT_NUMERIC"}:
        raise RuntimeError("closed four-frame pilot required")
    if pilot["summary"]["frames"] != 4:
        raise RuntimeError("pilot must bind exactly four frames")
    states_path = _closed(pilot["outputs"]["states"])
    with np.load(states_path, allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    source_q = np.asarray(source["q_arm"], np.float64)
    targets = np.asarray(source["T_target_hand_root_world"], np.float64)
    mounts = np.asarray(source["T_tool_hand_root"], np.float64)
    world_base = np.asarray(source["T_world_base"], np.float64)
    if source_q.shape != (4, 2, 7) or targets.shape != (4, 2, 4, 4):
        raise RuntimeError("pilot source shape mismatch")
    if not np.isfinite(targets).all() or not np.isfinite(source_q).all():
        raise RuntimeError("pilot has unknown/nonfinite first four frames")
    if not np.asarray(source["valid_side_frame"], bool).all():
        raise RuntimeError("pilot four-frame side validity is incomplete")
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    base_world = np.linalg.inv(world_base)
    rng = np.random.default_rng(41)
    q = np.full_like(source_q, np.nan)
    actual = np.full_like(targets, np.nan)
    rows = []
    started = time.monotonic()
    for side in range(2):
        previous = None
        previous_previous = None
        for frame in range(4):
            if time.monotonic() - started > args.wall_cap_s:
                raise TimeoutError("four-frame solver wall cap")
            target = base_world @ targets[frame, side] @ np.linalg.inv(mounts[side])
            if previous is None:
                lo, hi = lower[side].copy(), upper[side].copy()
            else:
                lo, hi = old_arm._one_step_viable_limits(
                    lower[side], upper[side], previous, previous_previous,
                    arm.temporal.ARM_SOLVER_STEP_LIMIT,
                )
            seeds = [source_q[frame, side], source_q[0, side]]
            if previous is not None:
                seeds.insert(0, previous)
            seeds += [0.5 * (lo + hi)]
            seeds += [rng.uniform(lo, hi) for _ in range(args.random_seeds)]
            _, fitted, seed_index, position_mm, rotation_deg, branch = _solve_pose_only(
                assets, side, target, lower[side], upper[side], seeds, previous, lo, hi,
            )
            q[frame, side] = fitted
            actual[frame, side] = world_base @ arm.old.official._tool_fk(
                assets, side, fitted
            ) @ mounts[side]
            rows.append({
                "frame": frame,
                "side": ("left", "right")[side],
                "position_mm": float(position_mm),
                "rotation_deg": float(rotation_deg),
                "target_pose_gate": bool(position_mm <= 10.0 and rotation_deg <= 5.0),
                "legacy_branch_clearance_m_diagnostic_only": branch,
                "seed_index": seed_index,
            })
            previous_previous, previous = previous, fitted
    velocity = np.max(np.abs(q[1:] - q[:-1]))
    acceleration = np.max(np.abs(q[2:] - 2 * q[1:-1] + q[:-2]))
    mesh_rows = _arm_mesh_audit(q, world_base)
    gates = {
        "target_pose_all_eight": all(row["target_pose_gate"] for row in rows),
        "urdf_limits": bool(np.all(q >= lower[None, :, :] - 1e-9) and np.all(
            q <= upper[None, :, :] + 1e-9
        )),
        "velocity": bool(velocity <= 0.12 + 1e-9),
        "acceleration": bool(acceleration <= 0.06 + 1e-9),
        "tianji_arm_only_mesh": all(
            row["arm_only_illegal_contact_count"] == 0 for row in mesh_rows
        ),
    }
    status = "PASS_ARM_ONLY_DEVELOPMENT" if all(gates.values()) else "FAILED_QUALITY_C"
    args.output_root.mkdir(parents=True)
    states_out = args.output_root / "ARM_ONLY_RESEARCH_STATES.npz"
    np.savez_compressed(
        states_out, q_arm=q, T_world_base=world_base, T_tool_hand_root=mounts,
        T_target_hand_root_world=targets, T_actual_hand_root_world=actual,
        valid_side_frame=source["valid_side_frame"],
        source_frames=source["source_frames"],
    )
    result = {
        "schema_version": "rc1-mesh-gated-four-frame-arm-v1",
        "created_at": now_iso(),
        "status": status,
        "session_id": pilot["session_id"],
        "frame_count": 4,
        "input_mode": "OFFLINE_VISUAL",
        "authority": False,
        "training_eligible": False,
        "full_robot_geometry_pass": False,
        "random_seed": 41,
        "random_seeds_per_row": args.random_seeds,
        "wall_seconds": round(time.monotonic() - started, 3),
        "metrics": {
            "velocity_rad_per_frame_max": float(velocity),
            "acceleration_rad_per_frame2_max": float(acceleration),
        },
        "gates": gates,
        "pose_rows": rows,
        "arm_mesh_rows": mesh_rows,
        "inputs": {
            "pilot_result": artifact_ref(args.pilot_result),
            "pilot_states": artifact_ref(states_path),
            "tianji_urdf": artifact_ref(collision.TIANJI),
            "robot_asset_pin": artifact_ref(ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
            "code": artifact_ref(Path(__file__)),
            "temporal_helper_code": artifact_ref(Path(old_arm.__file__)),
        },
        "output_states": artifact_ref(states_out),
        "claim_limit": "Four-frame Tianji arm-only development canary. Legacy branch plane is diagnostic, not ignored as a full collision gate; KaiHand, adapter, object, full-session temporal and physical safety are not covered. No Robot authority or causal training eligibility.",
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(f"{pilot['session_id']}: {status}; {sum(row['target_pose_gate'] for row in rows)}/8 target rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
