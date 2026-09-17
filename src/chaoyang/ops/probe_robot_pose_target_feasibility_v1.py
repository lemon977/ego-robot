#!/usr/bin/env python3
"""Diagnose fixed-base IK failures without changing targets or quality gates."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.pipeline import robot_renderer_eevee_fullchain as renderer  # noqa: E402


def _position_only(assets, side, target, lower, upper, seeds):
    best = None
    for index, seed in enumerate(seeds):
        def residual(q):
            actual = arm.old.official._tool_fk(assets, side, q)
            position_mm = 1000.0 * (actual[:3, 3] - target[:3, 3])
            margin_penalty = np.minimum(arm.arm.margins(assets, side, q) - 0.003, 0.0) / 0.001
            return np.concatenate((position_mm, margin_penalty))

        solution = least_squares(
            residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=500
        )
        position_mm, rotation_deg = arm.arm.error(assets, side, solution.x, target)
        margins = arm.arm.margins(assets, side, solution.x)
        candidate = {
            "seed_index": index,
            "position_mm": float(position_mm),
            "rotation_deg": float(rotation_deg),
            "branch_clearance_m_min": float(np.min(margins)),
            "q_arm": solution.x.tolist(),
        }
        if best is None or (candidate["position_mm"], -candidate["branch_clearance_m_min"]) < (
            best["position_mm"], -best["branch_clearance_m_min"]
        ):
            best = candidate
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--random-seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.random_seeds <= 100 or args.output.exists() or args.output.is_symlink():
        raise RuntimeError("fresh output and bounded random seeds required")
    if args.session_id not in args.states.resolve().parts:
        raise RuntimeError("states path does not bind full session ID")
    with np.load(args.states, allow_pickle=False) as loaded:
        states = {name: np.asarray(loaded[name]) for name in loaded.files}
    q = np.asarray(states["q_arm"], np.float64)
    targets = np.asarray(states["T_target_hand_root_world"], np.float64)
    mounts = np.asarray(states["T_tool_hand_root"], np.float64)
    if q.ndim != 3 or q.shape[1:] != (2, 7) or targets.shape != (len(q), 2, 4, 4):
        raise RuntimeError("arm states and target shape mismatch")
    side = 0 if args.side == "left" else 1
    base_world = np.linalg.inv(states["T_world_base"])
    inverse_mount = np.linalg.inv(mounts[side])
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    rng = np.random.default_rng(24)
    started = time.monotonic()
    rows = []
    for frame in args.frames:
        if frame < 1 or frame >= len(q) or not np.isfinite(q[frame - 1, side]).all():
            raise RuntimeError(f"frame {frame} lacks a finite previous state")
        target = base_world @ targets[frame, side] @ inverse_mount
        if not np.isfinite(target).all():
            raise RuntimeError(f"frame {frame} target is nonfinite")
        seeds = [q[frame - 1, side], q[0, side], 0.5 * (lower[side] + upper[side])]
        seeds += [rng.uniform(lower[side], upper[side]) for _ in range(args.random_seeds)]
        full = arm.solve_one(
            assets, side, target, lower[side], upper[side], seeds,
            previous=None, lo=lower[side], hi=upper[side],
        )
        position_only = _position_only(assets, side, target, lower[side], upper[side], seeds)
        rows.append({
            "frame": frame,
            "side": args.side,
            "target_tool_position_base_m": target[:3, 3].tolist(),
            "full_pose_no_temporal": {
                "within_10mm_5deg_and_margins": not bool(full[0]),
                "seed_index": int(full[-4]),
                "position_mm": float(full[-3]),
                "rotation_deg": float(full[-2]),
                "branch_clearance_m_min": float(np.min(full[-1])),
                "q_arm": full[-5].tolist(),
            },
            "position_only_no_temporal": position_only,
        })
    pin = ROOT / renderer.ASSET_PIN_RELATIVE
    result = {
        "schema_version": "rc1-robot-pose-target-feasibility-v1",
        "created_at": now_iso(),
        "status": "COMPLETED_DIAGNOSTIC",
        "session_id": args.session_id,
        "input_mode": "OFFLINE_VISUAL",
        "authority": False,
        "training_eligible": False,
        "random_seed": 24,
        "random_starts_per_frame": args.random_seeds,
        "wall_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "arm_states": artifact_ref(args.states),
            "robot_asset_pin": artifact_ref(pin),
            "code": artifact_ref(Path(__file__)),
        },
        "rows": rows,
        "claim_limit": "Bounded multistart numerical diagnostic only. A failed search is not a mathematical proof of unreachable pose; position-only success is not full-pose success. Fixed base/mount and formal gates are unchanged. No causal or physical authority.",
    }
    atomic_json(args.output, result)
    print(f"{args.session_id}: {len(rows)} diagnostic frames in {result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
