#!/usr/bin/env python3
"""Read-only multistart diagnosis of a closed four-frame Robot pilot."""

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
from chaoyang.ops import probe_robot_pose_target_feasibility_v1 as feasibility  # noqa: E402
from chaoyang.pipeline import robot_renderer_eevee_fullchain as renderer  # noqa: E402


def _closed(reference: dict) -> Path:
    path = Path(reference["path"])
    actual = artifact_ref(path)
    if actual["bytes"] != reference["bytes"] or actual["sha256"] != reference["sha256"]:
        raise RuntimeError(f"input SHA conflict: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=30)
    parser.add_argument("--wall-cap-s", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("output must be fresh")
    if not 0 <= args.random_seeds <= 100 or not 0 < args.wall_cap_s <= 1800:
        raise RuntimeError("probe budget out of bounds")
    pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
    if pilot["status"] != "FAILED_QUALITY_C" or pilot["summary"]["frames"] != 4:
        raise RuntimeError("probe requires a closed quality-C four-frame pilot")
    session_id = pilot["session_id"]
    states_path = _closed(pilot["outputs"]["states"])
    with np.load(states_path, allow_pickle=False) as loaded:
        states = {name: np.asarray(loaded[name]) for name in loaded.files}
    q = np.asarray(states["q_arm"], np.float64)
    targets = np.asarray(states["T_target_hand_root_world"], np.float64)
    mounts = np.asarray(states["T_tool_hand_root"], np.float64)
    if q.shape != (4, 2, 7) or targets.shape != (4, 2, 4, 4):
        raise RuntimeError("closed pilot states have unexpected shape")
    assets = arm.old.load_pinned_robot_assets(ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    base_world = np.linalg.inv(states["T_world_base"])
    rng = np.random.default_rng(24)
    started = time.monotonic()
    rows = []
    for side in range(2):
        inverse_mount = np.linalg.inv(mounts[side])
        for frame in (0, 1):
            if time.monotonic() - started > args.wall_cap_s:
                raise TimeoutError("multistart diagnostic wall cap")
            target = base_world @ targets[frame, side] @ inverse_mount
            if not np.isfinite(target).all():
                raise RuntimeError("nonfinite target")
            seeds = [q[frame, side], q[0, side], 0.5 * (lower[side] + upper[side])]
            seeds += [rng.uniform(lower[side], upper[side]) for _ in range(args.random_seeds)]
            full = arm.solve_one(
                assets, side, target, lower[side], upper[side], seeds,
                previous=None, lo=lower[side], hi=upper[side],
            )
            position_only = feasibility._position_only(
                assets, side, target, lower[side], upper[side], seeds
            )
            rows.append({
                "side": ("left", "right")[side],
                "frame": frame,
                "full_pose_no_temporal": {
                    "passed_10mm_5deg_and_branch_clearance": not bool(full[0]),
                    "position_mm": float(full[-3]),
                    "rotation_deg": float(full[-2]),
                    "branch_clearance_m_min": float(np.min(full[-1])),
                    "seed_index": int(full[-4]),
                    "q_arm": full[-5].tolist(),
                },
                "position_only_no_temporal": position_only,
            })
    result = {
        "schema_version": "rc1-robot-four-frame-multistart-probe-v1",
        "created_at": now_iso(),
        "status": "COMPLETED_DIAGNOSTIC",
        "session_id": session_id,
        "pilot_quality_status_unchanged": pilot["status"],
        "authority": False,
        "training_eligible": False,
        "input_mode": "OFFLINE_VISUAL",
        "random_seed": 24,
        "random_starts_per_row": args.random_seeds,
        "wall_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pilot_result": artifact_ref(args.pilot_result),
            "pilot_states": artifact_ref(states_path),
            "robot_asset_pin": artifact_ref(ROOT / renderer.ASSET_PIN_RELATIVE),
            "code": artifact_ref(Path(__file__)),
            "position_only_code": artifact_ref(Path(feasibility.__file__)),
        },
        "rows": rows,
        "claim_limit": "Four observed rows only; bounded multistart cannot prove global infeasibility. It neither retries nor supersedes the four-frame quality-C pilot and cannot authorize a full video or causal training.",
    }
    atomic_json(args.output, result)
    print(f"{session_id}: {len(rows)} rows; {result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
