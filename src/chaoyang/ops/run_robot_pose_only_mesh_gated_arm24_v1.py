#!/usr/bin/env python3
"""Extend a passed four-frame arm-only canary to 24 frames without future state."""

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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.ops import run_robot_pose_only_independent_arm_v6 as old_arm  # noqa: E402
from chaoyang.ops import run_robot_pose_only_independent_placement_pilot_v1 as placement  # noqa: E402
from chaoyang.ops import run_robot_pose_only_mesh_gated_four_frame_v1 as four  # noqa: E402
from chaoyang.ops import audit_robot_geometry_self_collision_v71 as collision  # noqa: E402


def _closed(ref: dict) -> Path:
    path = Path(ref["path"])
    current = artifact_ref(path)
    if current["bytes"] != ref["bytes"] or current["sha256"] != ref["sha256"]:
        raise RuntimeError(f"input SHA conflict: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pilot-result", type=Path, required=True)
    parser.add_argument("--passed-four-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extra-random-seeds", type=int, default=30)
    parser.add_argument("--wall-cap-s", type=int, default=1800)
    args = parser.parse_args()
    if args.output_root.exists() or not 0 <= args.extra_random_seeds <= 30:
        raise RuntimeError("fresh output root and bounded extra seeds required")
    if not 0 < args.wall_cap_s <= 3600:
        raise RuntimeError("wall cap out of range")
    source_result = json.loads(args.source_pilot_result.read_text(encoding="utf-8"))
    four_result = json.loads(args.passed_four_result.read_text(encoding="utf-8"))
    if four_result["status"] != "PASS_ARM_ONLY_DEVELOPMENT" or four_result["frame_count"] != 4:
        raise RuntimeError("passed four-frame arm-only result required")
    if four_result["session_id"] != source_result["session_id"]:
        raise RuntimeError("session ID mismatch")
    if four_result["inputs"]["pilot_result"]["sha256"] != artifact_ref(args.source_pilot_result)["sha256"]:
        raise RuntimeError("four-frame result does not bind source pilot")
    with np.load(_closed(four_result["output_states"]), allow_pickle=False) as loaded:
        accepted = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(_closed(source_result["inputs"]["hawor_npz"]), allow_pickle=False) as loaded:
        hawor = {name: np.asarray(loaded[name]) for name in (
            "joints_3d_world", "c2w", "original_frame_indices"
        )}
    human = np.asarray(hawor["joints_3d_world"], np.float64)
    count = 24
    if len(hawor["c2w"]) < count or not np.array_equal(
        hawor["original_frame_indices"][:count], np.arange(count)
    ):
        raise RuntimeError("HaWoR 24-frame identity not contiguous")
    valid = np.isfinite(human[:, :count]).all(axis=(2, 3))
    if not valid.all():
        raise RuntimeError("this bounded 24-frame canary requires all hand observations")
    world_base = np.asarray(accepted["T_world_base"], np.float64)
    mounts = np.asarray(accepted["T_tool_hand_root"], np.float64)
    base_world = np.linalg.inv(world_base)
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    q = np.full((count, 2, 7), np.nan)
    target = np.full((count, 2, 4, 4), np.nan)
    actual = np.full_like(target, np.nan)
    q[:4] = accepted["q_arm"]
    target[:4] = accepted["T_target_hand_root_world"]
    actual[:4] = accepted["T_actual_hand_root_world"]
    rng = np.random.default_rng(41)
    started = time.monotonic()
    rows = list(four_result["pose_rows"])
    for side in range(2):
        anchor_rotation = target[0, side, :3, :3]
        palm_anchor = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
            human[side, 0], handedness=arm.shared.SIDES[side]
        )
        for frame in range(4, count):
            if time.monotonic() - started > args.wall_cap_s:
                raise TimeoutError("24-frame arm canary wall cap")
            palm = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
                human[side, frame], handedness=arm.shared.SIDES[side]
            )
            rotation = palm @ palm_anchor.T @ anchor_rotation
            root_target = placement._root_target(human[side, frame, 0], rotation)
            target[frame, side] = root_target
            tool_target = base_world @ root_target @ np.linalg.inv(mounts[side])
            previous, previous_previous = q[frame - 1, side], q[frame - 2, side]
            lo, hi = old_arm._one_step_viable_limits(
                lower[side], upper[side], previous, previous_previous,
                arm.temporal.ARM_SOLVER_STEP_LIMIT,
            )
            seeds = [previous, q[0, side], 0.5 * (lo + hi)]
            best = four._solve_pose_only(
                assets, side, tool_target, lower[side], upper[side], seeds,
                previous, lo, hi,
            )
            if best[0][0] and args.extra_random_seeds:
                seeds += [rng.uniform(lo, hi) for _ in range(args.extra_random_seeds)]
                best = four._solve_pose_only(
                    assets, side, tool_target, lower[side], upper[side], seeds,
                    previous, lo, hi,
                )
            _, fitted, seed_index, position_mm, rotation_deg, branch = best
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
    velocity = np.max(np.abs(q[1:] - q[:-1]))
    acceleration = np.max(np.abs(q[2:] - 2 * q[1:-1] + q[:-2]))
    mesh_rows = four._arm_mesh_audit(q, world_base)
    gates = {
        "target_pose_all_observed": all(row["target_pose_gate"] for row in rows),
        "urdf_limits": bool(np.all(q >= lower[None, :, :] - 1e-9) and np.all(
            q <= upper[None, :, :] + 1e-9
        )),
        "velocity": bool(velocity <= 0.12 + 1e-9),
        "acceleration": bool(acceleration <= 0.06 + 1e-9),
        "tianji_arm_only_mesh": all(
            row["arm_only_illegal_contact_count"] == 0 for row in mesh_rows
        ),
        "first_four_q_immutable": bool(np.allclose(q[:4], accepted["q_arm"], atol=1e-10)),
        "first_four_target_immutable": bool(np.allclose(
            target[:4], accepted["T_target_hand_root_world"], atol=1e-10
        )),
    }
    status = "PASS_ARM_ONLY_DEVELOPMENT" if all(gates.values()) else "FAILED_QUALITY_C"
    args.output_root.mkdir(parents=True)
    states_out = args.output_root / "ARM_ONLY_RESEARCH_STATES.npz"
    np.savez_compressed(
        states_out, q_arm=q, T_world_base=world_base, T_tool_hand_root=mounts,
        T_target_hand_root_world=target, T_actual_hand_root_world=actual,
        valid_side_frame=valid, source_frames=hawor["original_frame_indices"][:count],
    )
    result = {
        "schema_version": "rc1-mesh-gated-arm24-v1",
        "created_at": now_iso(),
        "status": status,
        "session_id": source_result["session_id"],
        "frame_count": count,
        "input_mode": "OFFLINE_VISUAL",
        "authority": False,
        "training_eligible": False,
        "full_robot_geometry_pass": False,
        "wall_seconds": round(time.monotonic() - started, 3),
        "metrics": {
            "target_pass_rows": sum(row["target_pose_gate"] for row in rows),
            "target_total_rows": len(rows),
            "velocity_rad_per_frame_max": float(velocity),
            "acceleration_rad_per_frame2_max": float(acceleration),
        },
        "gates": gates,
        "pose_rows": rows,
        "arm_mesh_rows": mesh_rows,
        "inputs": {
            "source_pilot_result": artifact_ref(args.source_pilot_result),
            "accepted_four_result": artifact_ref(args.passed_four_result),
            "hawor_npz": artifact_ref(Path(source_result["inputs"]["hawor_npz"]["path"])),
            "tianji_urdf": artifact_ref(collision.TIANJI),
            "robot_asset_pin": artifact_ref(ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
            "code": artifact_ref(Path(__file__)),
            "four_frame_solver_code": artifact_ref(Path(four.__file__)),
            "temporal_helper_code": artifact_ref(Path(old_arm.__file__)),
        },
        "output_states": artifact_ref(states_out),
        "claim_limit": "24-frame offline arm-only research result. HaWoR upstream uses full-sequence smoothing; KaiHand, adapter, object, full digital collision and causal input are not approved. No full Robot authority or training eligibility.",
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(f"{source_result['session_id']}: {status}; {result['metrics']['target_pass_rows']}/{len(rows)} target rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
