#!/usr/bin/env python3
"""Compare fixed base presets with pose-only IK and frozen branch-clearance QA."""

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
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm  # noqa: E402
from chaoyang.pipeline import robot_renderer_eevee_fullchain as renderer  # noqa: E402


def _checked(ref: dict) -> Path:
    path = Path(ref["path"])
    actual = artifact_ref(path)
    if actual["bytes"] != ref["bytes"] or actual["sha256"] != ref["sha256"]:
        raise RuntimeError(f"input SHA conflict: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--wall-cap-s", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("output must be fresh")
    if not 0 <= args.random_seeds <= 100 or not 0 < args.wall_cap_s <= 1800:
        raise RuntimeError("probe budget out of bounds")
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    if pilot["status"] != "FAILED_QUALITY_C" or pilot["summary"]["frames"] != 4:
        raise RuntimeError("requires a closed quality-C four-frame pilot")
    if artifact_ref(args.preset)["sha256"] != pilot["inputs"]["preset"]["sha256"]:
        raise RuntimeError("preset differs from frozen pilot")
    with np.load(_checked(pilot["outputs"]["states"]), allow_pickle=False) as loaded:
        states = {name: np.asarray(loaded[name]) for name in loaded.files}
    q = np.asarray(states["q_arm"], np.float64)
    targets = np.asarray(states["T_target_hand_root_world"], np.float64)
    mounts = np.asarray(states["T_tool_hand_root"], np.float64)
    if q.shape != (4, 2, 7) or targets.shape != (4, 2, 4, 4):
        raise RuntimeError("pilot state shape mismatch")
    base_candidates = np.asarray(preset["base_midpoint_target_candidates_m"], np.float64)
    midpoint = 0.5 * (targets[0, 0, :3, 3] + targets[0, 1, :3, 3])
    frozen_base = np.asarray(states["T_world_base"], np.float64)
    chosen_index = int(pilot["chosen_candidate_index"])
    rebuilt_origin = midpoint - frozen_base[:3, :3] @ base_candidates[chosen_index]
    if not np.allclose(rebuilt_origin, frozen_base[:3, 3], atol=1e-8):
        raise RuntimeError("frozen base does not close with candidate preset")
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    rng = np.random.default_rng(21)
    started = time.monotonic()
    rows = []
    for candidate_index, base_target in enumerate(base_candidates):
        world_base = frozen_base.copy()
        world_base[:3, 3] = midpoint - world_base[:3, :3] @ base_target
        base_world = np.linalg.inv(world_base)
        for side in range(2):
            if time.monotonic() - started > args.wall_cap_s:
                raise TimeoutError("branch tradeoff wall cap")
            target = base_world @ targets[0, side] @ np.linalg.inv(mounts[side])
            seeds = [q[0, side], 0.5 * (lower[side] + upper[side])]
            seeds += [rng.uniform(lower[side], upper[side]) for _ in range(args.random_seeds)]
            candidates = []
            for seed_index, seed in enumerate(seeds):
                solution = least_squares(
                    lambda x: arm.old.official._pose_residual(
                        arm.old.official._tool_fk(assets, side, x), target
                    ),
                    np.clip(seed, lower[side], upper[side]),
                    bounds=(lower[side], upper[side]),
                    max_nfev=500,
                )
                position_mm, rotation_deg = arm.arm.error(assets, side, solution.x, target)
                clearance_m = float(np.min(arm.arm.margins(assets, side, solution.x)))
                candidates.append({
                    "seed_index": seed_index,
                    "position_mm": float(position_mm),
                    "rotation_deg": float(rotation_deg),
                    "branch_clearance_m_min": clearance_m,
                    "passes_pose_only": position_mm <= 10.0 and rotation_deg <= 5.0,
                    "passes_pose_and_clearance": (
                        position_mm <= 10.0 and rotation_deg <= 5.0 and clearance_m >= -1e-8
                    ),
                    "q_arm": solution.x.tolist(),
                })
            best = min(candidates, key=lambda row: (
                max(0.0, row["position_mm"] - 10.0) + max(0.0, row["rotation_deg"] - 5.0),
                row["position_mm"] + row["rotation_deg"],
            ))
            rows.append({
                "base_candidate_index": candidate_index,
                "side": ("left", "right")[side],
                "pose_only_best": best,
                "pose_only_pass_count": sum(row["passes_pose_only"] for row in candidates),
                "pose_and_clearance_pass_count": sum(
                    row["passes_pose_and_clearance"] for row in candidates
                ),
            })
    result = {
        "schema_version": "rc1-robot-branch-clearance-tradeoff-v1",
        "created_at": now_iso(),
        "status": "COMPLETED_DIAGNOSTIC",
        "session_id": pilot["session_id"],
        "source_frame": 0,
        "quality_status_unchanged": pilot["status"],
        "random_seed": 21,
        "random_starts_per_candidate_side": args.random_seeds,
        "wall_seconds": round(time.monotonic() - started, 3),
        "authority": False,
        "training_eligible": False,
        "inputs": {
            "pilot_result": artifact_ref(args.pilot_result),
            "pilot_states": artifact_ref(Path(pilot["outputs"]["states"]["path"])),
            "preset": artifact_ref(args.preset),
            "robot_asset_pin": artifact_ref(ROOT / renderer.ASSET_PIN_RELATIVE),
            "code": artifact_ref(Path(__file__)),
        },
        "rows": rows,
        "claim_limit": "Pose-only optimization deliberately omits branch-clearance penalty solely to diagnose tradeoff; its results are invalid for Robot geometry approval. No finding proves global infeasibility or authorizes relaxed collision gates, training, or physical deployment.",
    }
    atomic_json(args.output, result)
    print(f"{pilot['session_id']}: {len(rows)} base/side rows in {result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
