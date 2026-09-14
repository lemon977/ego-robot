#!/usr/bin/env python3
"""CPU full-session arm canary for reanchored unit-gain motion transfer.

The Robot base is fixed for the full session. Missing human-hand frames stay
UNKNOWN (NaN q/target/actual); the solver never interpolates or holds them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import diagnose_robot_scaled_temporal_arm_v3 as arm  # noqa: E402
from tools import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from tools import render_poker_static_closure_task_translation_successor as taskfit  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_same_side_world_temporal_review as temporal  # noqa: E402

SIDES = ("left", "right")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def solve_one(assets, side, target_base, lower, upper, seeds, previous=None, lo=None, hi=None):
    if lo is None:
        lo, hi = lower.copy(), upper.copy()
    attempts = []
    for seed_index, seed in enumerate(seeds):
        seed = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)

        def residual(x):
            pose = old.official._pose_residual(old.official._tool_fk(assets, side, x), target_base)
            branch = np.minimum(arm.margins(assets, side, x) - 0.003, 0.0) / 0.001
            continuity = np.empty(0) if previous is None else 0.01 * (x - previous)
            return np.concatenate((pose, branch, continuity))

        free = hi - lo > 1e-12
        x = seed.copy()
        if np.any(free):
            sol = least_squares(
                lambda y: residual(np.where(free, y, x)),
                x[free],
                bounds=(lo[free], hi[free]),
                max_nfev=500,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            x[free] = sol.x
        pos, rot = arm.error(assets, side, x, target_base)
        margins = arm.margins(assets, side, x)
        failed = not (pos <= 10.0 and rot <= 5.0 and float(np.min(margins)) >= -1e-8)
        score = (failed, max(0.0, -float(np.min(margins))), max(0.0, pos - 10.0) + max(0.0, rot - 5.0), pos + rot)
        attempts.append((*score, x, seed_index, pos, rot, margins))
    return min(attempts, key=lambda x: x[:4])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("chips", "poker"), required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--hawor", type=Path, required=True)
    ap.add_argument("--hawor-result", type=Path, required=True)
    ap.add_argument("--accepted-states", type=Path, required=True)
    ap.add_argument("--accepted-hawor", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--motion-gain", type=float, default=1.0)
    ap.add_argument("--task-base-backoff-m", type=float, default=0.0,
                    help="Fixed task-level base translation along negative base X; shared for every frame.")
    args = ap.parse_args()
    if args.motion_gain != 1.0:
        raise RuntimeError("v2 contract fixes translation motion gain to 1.0")
    for path in (args.hawor, args.hawor_result, args.accepted_states, args.accepted_hawor):
        path.resolve(strict=True)
    if args.session not in str(args.hawor.resolve()) or args.session not in str(args.hawor_result.resolve()):
        raise RuntimeError("same-session current HaWoR identity check failed")

    hawor = load(args.hawor)
    accepted = load(args.accepted_states)
    accepted_hawor = load(args.accepted_hawor)
    n = len(hawor["c2w"])
    assets = old.load_pinned_robot_assets(PROJECT)
    lower, upper = taskfit.arm_limits(assets)
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
    if not np.all(valid[:, 0]):
        raise RuntimeError("frame0 must be observed on both sides for reanchor")
    accepted_human = np.asarray(accepted_hawor["joints_3d_world"][:, 0, 0], dtype=np.float64)
    accepted_offsets_camera = np.empty((2, 3), dtype=np.float64)
    target = np.full((n, 2, 4, 4), np.nan, dtype=np.float64)
    q = np.full((n, 2, 7), np.nan, dtype=np.float64)
    actual = np.full_like(target, np.nan)
    rows = []

    for side in range(2):
        accepted_target_cam = np.linalg.inv(accepted_hawor["c2w"][0]) @ accepted["T_target_hand_root_world"][0, side]
        accepted_human_cam = np.linalg.inv(accepted_hawor["c2w"][0]) @ np.r_[accepted_human[side], 1.0]
        accepted_offsets_camera[side] = accepted_target_cam[:3, 3] - accepted_human_cam[:3]
        current_human_cam0 = np.linalg.inv(hawor["c2w"][0]) @ np.r_[human[side, 0, 0], 1.0]
        anchor_position = (hawor["c2w"][0] @ np.r_[current_human_cam0[:3] + accepted_offsets_camera[side], 1.0])[:3]
        accepted_root_world = world_base @ old.official._tool_fk(assets, side, accepted_q0[side]) @ mounts[side]
        palm0 = shared.wrist_adapter.final_v3_mano_palm_basis(human[side, 0], handedness=shared.SIDES[side])
        anchor_rotation = accepted_root_world[:3, :3]

        previous = None
        previous_previous = None
        for frame in range(n):
            if not valid[side, frame]:
                rows.append({"frame": frame, "side": SIDES[side], "status": "UNKNOWN_MISSING_HAWOR", "pass": False})
                previous = None
                previous_previous = None
                continue
            palm = shared.wrist_adapter.final_v3_mano_palm_basis(human[side, frame], handedness=shared.SIDES[side])
            root = np.eye(4)
            root[:3, :3] = palm @ palm0.T @ anchor_rotation
            root[:3, 3] = anchor_position + args.motion_gain * (human[side, frame, 0] - human[side, 0, 0])
            target[frame, side] = root
            target_base = base_world @ root @ np.linalg.inv(mounts[side])
            if previous is None:
                seeds = [accepted_q0[side], 0.5 * (lower[side] + upper[side])]
                attempt = solve_one(assets, side, target_base, lower[side], upper[side], seeds)
                policy = "SEGMENT_START_GLOBAL_MULTISEED"
            else:
                lo, hi = temporal.bounded_limits(lower[side], upper[side], previous, previous_previous, temporal.ARM_SOLVER_STEP_LIMIT)
                seeds = [np.clip(previous, lo, hi), np.clip(accepted_q0[side], lo, hi)]
                attempt = solve_one(assets, side, target_base, lower[side], upper[side], seeds, previous=previous, lo=lo, hi=hi)
                policy = "CONTIGUOUS_BOUNDED_MULTISEED"
            *_, chosen, seed_index, pos, rot, margins = attempt
            q[frame, side] = chosen
            actual[frame, side] = world_base @ old.official._tool_fk(assets, side, chosen) @ mounts[side]
            passed = pos <= 10.0 and rot <= 5.0 and float(np.min(margins)) >= -1e-8
            rows.append({
                "frame": frame, "side": SIDES[side], "status": "SOLVED" if passed else "HOLD_NUMERIC_GATE",
                "position_mm": pos, "rotation_deg": rot, "branch_margins_m": margins.tolist(),
                "policy": f"{policy}_{seed_index}", "pass": bool(passed),
            })
            previous_previous, previous = previous, chosen

    solved_rows = [r for r in rows if r["status"] != "UNKNOWN_MISSING_HAWOR"]
    failed_rows = [r for r in solved_rows if not r["pass"]]
    velocity_values = []
    acceleration_values = []
    for side in range(2):
        for frame in range(1, n):
            if valid[side, frame] and valid[side, frame - 1]:
                velocity_values.extend(np.abs(q[frame, side] - q[frame - 1, side]).tolist())
        for frame in range(2, n):
            if valid[side, frame] and valid[side, frame - 1] and valid[side, frame - 2]:
                acceleration_values.extend(np.abs(q[frame, side] - 2 * q[frame - 1, side] + q[frame - 2, side]).tolist())
    velocity_max = float(max(velocity_values, default=0.0))
    acceleration_max = float(max(acceleration_values, default=0.0))
    gates = {
        "all_observed_rows_pose_branch": len(failed_rows) == 0,
        "arm_velocity": velocity_max <= 0.12 + 1e-9,
        "arm_acceleration": acceleration_max <= 0.06 + 1e-9,
        "base_fixed": True,
        "motion_gain_exact_one": args.motion_gain == 1.0,
        "missing_frames_unknown_not_filled": bool(np.all(np.isnan(q[~valid.T]))),
    }
    output_states = args.output.with_name("ARM_CANARY_STATES.npz")
    output_states.parent.mkdir(parents=True, exist_ok=True)
    tmp_states = output_states.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        tmp_states, q_arm=q, T_world_base=world_base, T_tool_hand_root=mounts,
        T_target_hand_root_world=target, T_actual_hand_root_world=actual,
        valid_side_frame=valid, source_frames=hawor["original_frame_indices"],
        accepted_offset_camera=accepted_offsets_camera,
    )
    os.replace(tmp_states, output_states)
    payload = {
        "schema_version": "robot-motion-transfer-arm-canary-v2",
        "created_at": now(), "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY" if all(gates.values()) else "HOLD_NUMERIC_CANARY",
        "task": args.task, "session": args.session, "frame_count": n, "motion_gain": args.motion_gain,
        "lineage": {
            "hawor": {"path": str(args.hawor.resolve()), "sha256": sha(args.hawor)},
            "hawor_result": {"path": str(args.hawor_result.resolve()), "sha256": sha(args.hawor_result)},
            "accepted_states": {"path": str(args.accepted_states.resolve()), "sha256": sha(args.accepted_states)},
            "accepted_hawor": {"path": str(args.accepted_hawor.resolve()), "sha256": sha(args.accepted_hawor)},
        },
        "contract": {
            "base": "one fixed world base for the full session; never dynamically moved",
            "task_base_backoff_m": args.task_base_backoff_m,
            "initial_anchor": "new-session frame0 human wrist plus accepted per-task camera-space hand-root-minus-wrist offset",
            "translation": "unit gain on human world wrist displacement",
            "orientation": "full relative MANO palm rotation from session frame0",
            "missing": "UNKNOWN NaN, no interpolation, no hold, no temporal constraint across gap",
        },
        "metrics": {
            "observed_rows": len(solved_rows), "unknown_rows": int((~valid).sum()), "failed_rows": len(failed_rows),
            "position_mm_max": max((r["position_mm"] for r in solved_rows), default=float("nan")),
            "rotation_deg_max": max((r["rotation_deg"] for r in solved_rows), default=float("nan")),
            "branch_margin_m_min": min((min(r["branch_margins_m"]) for r in solved_rows), default=float("nan")),
            "velocity_rad_per_frame_max_contiguous": velocity_max,
            "acceleration_rad_per_frame2_max_contiguous": acceleration_max,
        },
        "gates": gates, "rows": rows,
        "output_states": {"path": str(output_states), "sha256": sha(output_states), "bytes": output_states.stat().st_size},
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "CPU numeric canary only; not visual approval, action sidecar, Robot authority, or physical contact validation.",
    }
    tmp = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"result": str(args.output), "sha256": sha(args.output), "status": payload["status"], "metrics": payload["metrics"]}))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
