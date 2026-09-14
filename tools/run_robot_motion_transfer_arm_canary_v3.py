#!/usr/bin/env python3
"""CPU full-session arm canary with per-side first-observed anchoring.

This is a narrow successor to v2 for sessions whose frame 0 does not observe
both hands.  The Robot base remains fixed for the full session.  Each physical
side is anchored at its own first genuinely observed HaWoR frame; earlier and
interior missing rows remain UNKNOWN/NaN and temporal constraints never cross
a missing gap.  Numeric gates and placement semantics are unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import run_robot_motion_transfer_arm_canary_v2 as v2  # noqa: E402


def first_observed_per_side(valid: np.ndarray) -> np.ndarray:
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError(f"expected valid shape (2,N), got {valid.shape}")
    anchors = []
    for side in range(2):
        observed = np.flatnonzero(valid[side])
        if not len(observed):
            raise ValueError(f"{v2.SIDES[side]} has no observed frame")
        anchors.append(int(observed[0]))
    return np.asarray(anchors, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--accepted-states", type=Path, required=True)
    parser.add_argument("--accepted-hawor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-gain", type=float, default=1.0)
    parser.add_argument(
        "--task-base-backoff-m",
        type=float,
        default=0.0,
        help="Fixed task-level base translation along negative base X; shared for every frame.",
    )
    args = parser.parse_args()
    if args.motion_gain != 1.0:
        raise RuntimeError("v3 contract fixes translation motion gain to 1.0")
    for path in (args.hawor, args.hawor_result, args.accepted_states, args.accepted_hawor):
        path.resolve(strict=True)
    if args.session not in str(args.hawor.resolve()) or args.session not in str(args.hawor_result.resolve()):
        raise RuntimeError("same-session current HaWoR identity check failed")

    hawor = v2.load(args.hawor)
    accepted = v2.load(args.accepted_states)
    accepted_hawor = v2.load(args.accepted_hawor)
    n = len(hawor["c2w"])
    assets = v2.old.load_pinned_robot_assets(PROJECT)
    lower, upper = v2.taskfit.arm_limits(assets)
    accepted_q0 = np.asarray(accepted["q_arm"][0], dtype=np.float64)
    mounts = np.asarray(accepted["T_tool_hand_root"], dtype=np.float64)
    accepted_camera_base = np.asarray(accepted["T_camera_base"][0], dtype=np.float64)
    world_base = np.asarray(hawor["c2w"][0], dtype=np.float64) @ accepted_camera_base
    base_adjustment = np.eye(4, dtype=np.float64)
    base_adjustment[0, 3] = -args.task_base_backoff_m
    world_base = world_base @ base_adjustment
    base_world = np.linalg.inv(world_base)

    human = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    valid = np.isfinite(human).all(axis=(2, 3))
    anchor_frames = first_observed_per_side(valid)
    accepted_human = np.asarray(accepted_hawor["joints_3d_world"][:, 0, 0], dtype=np.float64)
    accepted_offsets_camera = np.empty((2, 3), dtype=np.float64)
    target = np.full((n, 2, 4, 4), np.nan, dtype=np.float64)
    q = np.full((n, 2, 7), np.nan, dtype=np.float64)
    actual = np.full_like(target, np.nan)
    rows = []

    for side in range(2):
        anchor_frame = int(anchor_frames[side])
        accepted_target_cam = (
            np.linalg.inv(accepted_hawor["c2w"][0]) @ accepted["T_target_hand_root_world"][0, side]
        )
        accepted_human_cam = np.linalg.inv(accepted_hawor["c2w"][0]) @ np.r_[accepted_human[side], 1.0]
        accepted_offsets_camera[side] = accepted_target_cam[:3, 3] - accepted_human_cam[:3]
        current_human_cam_anchor = (
            np.linalg.inv(hawor["c2w"][anchor_frame]) @ np.r_[human[side, anchor_frame, 0], 1.0]
        )
        anchor_position = (
            hawor["c2w"][anchor_frame]
            @ np.r_[current_human_cam_anchor[:3] + accepted_offsets_camera[side], 1.0]
        )[:3]
        accepted_root_world = (
            world_base @ v2.old.official._tool_fk(assets, side, accepted_q0[side]) @ mounts[side]
        )
        palm_anchor = v2.shared.wrist_adapter.final_v3_mano_palm_basis(
            human[side, anchor_frame], handedness=v2.shared.SIDES[side]
        )
        anchor_rotation = accepted_root_world[:3, :3]

        previous = None
        previous_previous = None
        for frame in range(n):
            if not valid[side, frame]:
                rows.append(
                    {"frame": frame, "side": v2.SIDES[side], "status": "UNKNOWN_MISSING_HAWOR", "pass": False}
                )
                previous = None
                previous_previous = None
                continue
            palm = v2.shared.wrist_adapter.final_v3_mano_palm_basis(
                human[side, frame], handedness=v2.shared.SIDES[side]
            )
            root = np.eye(4)
            root[:3, :3] = palm @ palm_anchor.T @ anchor_rotation
            root[:3, 3] = anchor_position + args.motion_gain * (
                human[side, frame, 0] - human[side, anchor_frame, 0]
            )
            target[frame, side] = root
            target_base = base_world @ root @ np.linalg.inv(mounts[side])
            if previous is None:
                seeds = [accepted_q0[side], 0.5 * (lower[side] + upper[side])]
                attempt = v2.solve_one(assets, side, target_base, lower[side], upper[side], seeds)
                policy = "SEGMENT_START_GLOBAL_MULTISEED"
            else:
                lo, hi = v2.temporal.bounded_limits(
                    lower[side], upper[side], previous, previous_previous, v2.temporal.ARM_SOLVER_STEP_LIMIT
                )
                seeds = [np.clip(previous, lo, hi), np.clip(accepted_q0[side], lo, hi)]
                attempt = v2.solve_one(
                    assets,
                    side,
                    target_base,
                    lower[side],
                    upper[side],
                    seeds,
                    previous=previous,
                    lo=lo,
                    hi=hi,
                )
                policy = "CONTIGUOUS_BOUNDED_MULTISEED"
            *_, chosen, seed_index, pos, rot, margins = attempt
            q[frame, side] = chosen
            actual[frame, side] = world_base @ v2.old.official._tool_fk(assets, side, chosen) @ mounts[side]
            passed = pos <= 10.0 and rot <= 5.0 and float(np.min(margins)) >= -1e-8
            rows.append(
                {
                    "frame": frame,
                    "side": v2.SIDES[side],
                    "status": "SOLVED" if passed else "HOLD_NUMERIC_GATE",
                    "position_mm": pos,
                    "rotation_deg": rot,
                    "branch_margins_m": margins.tolist(),
                    "policy": f"{policy}_{seed_index}",
                    "pass": bool(passed),
                }
            )
            previous_previous, previous = previous, chosen

    solved_rows = [row for row in rows if row["status"] != "UNKNOWN_MISSING_HAWOR"]
    failed_rows = [row for row in solved_rows if not row["pass"]]
    velocity_values = []
    acceleration_values = []
    for side in range(2):
        for frame in range(1, n):
            if valid[side, frame] and valid[side, frame - 1]:
                velocity_values.extend(np.abs(q[frame, side] - q[frame - 1, side]).tolist())
        for frame in range(2, n):
            if valid[side, frame] and valid[side, frame - 1] and valid[side, frame - 2]:
                acceleration_values.extend(
                    np.abs(q[frame, side] - 2 * q[frame - 1, side] + q[frame - 2, side]).tolist()
                )
    velocity_max = float(max(velocity_values, default=0.0))
    acceleration_max = float(max(acceleration_values, default=0.0))
    gates = {
        "all_observed_rows_pose_branch": len(failed_rows) == 0,
        "arm_velocity": velocity_max <= 0.12 + 1e-9,
        "arm_acceleration": acceleration_max <= 0.06 + 1e-9,
        "base_fixed": True,
        "motion_gain_exact_one": args.motion_gain == 1.0,
        "missing_frames_unknown_not_filled": bool(np.all(np.isnan(q[~valid.T]))),
        "per_side_anchor_is_observed": bool(np.all(valid[np.arange(2), anchor_frames])),
    }
    output_states = args.output.with_name("ARM_CANARY_STATES.npz")
    output_states.parent.mkdir(parents=True, exist_ok=True)
    tmp_states = output_states.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        tmp_states,
        q_arm=q,
        T_world_base=world_base,
        T_tool_hand_root=mounts,
        T_target_hand_root_world=target,
        T_actual_hand_root_world=actual,
        valid_side_frame=valid,
        source_frames=hawor["original_frame_indices"],
        accepted_offset_camera=accepted_offsets_camera,
        anchor_frame_per_side=anchor_frames,
    )
    os.replace(tmp_states, output_states)
    payload = {
        "schema_version": "robot-motion-transfer-arm-canary-v3",
        "created_at": v2.now(),
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY" if all(gates.values()) else "HOLD_NUMERIC_CANARY",
        "task": args.task,
        "session": args.session,
        "frame_count": n,
        "motion_gain": args.motion_gain,
        "lineage": {
            "hawor": {"path": str(args.hawor.resolve()), "sha256": v2.sha(args.hawor)},
            "hawor_result": {"path": str(args.hawor_result.resolve()), "sha256": v2.sha(args.hawor_result)},
            "accepted_states": {"path": str(args.accepted_states.resolve()), "sha256": v2.sha(args.accepted_states)},
            "accepted_hawor": {"path": str(args.accepted_hawor.resolve()), "sha256": v2.sha(args.accepted_hawor)},
            "algorithm_predecessor": {
                "path": str(Path(v2.__file__).resolve()),
                "sha256": v2.sha(Path(v2.__file__).resolve()),
            },
        },
        "contract": {
            "base": "one fixed world base for the full session; never dynamically moved",
            "task_base_backoff_m": args.task_base_backoff_m,
            "initial_anchor": "first genuinely observed frame independently for each side",
            "anchor_frame_per_side": anchor_frames.tolist(),
            "translation": "unit gain on human world wrist displacement from the same-side observed anchor",
            "orientation": "full relative MANO palm rotation from the same-side observed anchor",
            "missing": "UNKNOWN NaN, no interpolation, no hold, no temporal constraint across gap",
        },
        "metrics": {
            "observed_rows": len(solved_rows),
            "unknown_rows": int((~valid).sum()),
            "failed_rows": len(failed_rows),
            "position_mm_max": max((row["position_mm"] for row in solved_rows), default=float("nan")),
            "rotation_deg_max": max((row["rotation_deg"] for row in solved_rows), default=float("nan")),
            "branch_margin_m_min": min(
                (min(row["branch_margins_m"]) for row in solved_rows), default=float("nan")
            ),
            "velocity_rad_per_frame_max_contiguous": velocity_max,
            "acceleration_rad_per_frame2_max_contiguous": acceleration_max,
        },
        "gates": gates,
        "rows": rows,
        "output_states": {
            "path": str(output_states),
            "sha256": v2.sha(output_states),
            "bytes": output_states.stat().st_size,
        },
        "authority": False,
        "action_sidecar_published": False,
        "claim_limit": (
            "CPU numeric canary with per-side first-observed anchoring only; not visual approval, action "
            "sidecar, Robot authority, or physical contact validation."
        ),
    }
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "result": str(args.output),
                "sha256": v2.sha(args.output),
                "status": payload["status"],
                "metrics": payload["metrics"],
                "anchor_frame_per_side": anchor_frames.tolist(),
            }
        )
    )
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
