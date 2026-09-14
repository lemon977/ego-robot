#!/usr/bin/env python3
"""Choose a forward or reverse feasible IK path per observed hand segment.

Reverse solving is finite lookahead: it anticipates a segment endpoint without
ever constraining across an UNKNOWN gap. Numeric thresholds are unchanged.
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import diagnose_robot_scaled_temporal_arm_v3 as arm  # noqa: E402
from tools import render_poker_static_closure_task_translation_successor as taskfit  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import render_same_side_world_temporal_review as temporal  # noqa: E402
from tools import run_robot_motion_transfer_arm_canary_v2 as forward_tool  # noqa: E402

SIDES = ("left", "right")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def segments(valid: np.ndarray) -> list[np.ndarray]:
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return []
    cuts = np.flatnonzero(np.diff(ids) != 1) + 1
    return [x for x in np.split(ids, cuts) if len(x)]


def solve_direction(assets, side, order, target, base_world, mounts, accepted_q0, lower, upper, seed_path):
    q = {}
    rows = []
    previous = None
    previous_previous = None
    for frame in order:
        target_base = base_world @ target[frame, side] @ np.linalg.inv(mounts[side])
        seed_frame = seed_path[frame, side] if np.isfinite(seed_path[frame, side]).all() else accepted_q0[side]
        if previous is None:
            seeds = [seed_frame, accepted_q0[side], 0.5 * (lower[side] + upper[side])]
            attempt = forward_tool.solve_one(assets, side, target_base, lower[side], upper[side], seeds)
            policy = "SEGMENT_ENDPOINT_GLOBAL"
        else:
            lo, hi = temporal.bounded_limits(lower[side], upper[side], previous, previous_previous, temporal.ARM_SOLVER_STEP_LIMIT)
            seeds = [np.clip(seed_frame, lo, hi), np.clip(previous, lo, hi), np.clip(accepted_q0[side], lo, hi)]
            attempt = forward_tool.solve_one(assets, side, target_base, lower[side], upper[side], seeds, previous=previous, lo=lo, hi=hi)
            policy = "SEGMENT_BOUNDED"
        *_, chosen, seed_index, pos, rot, margins = attempt
        passed = pos <= 10.0 and rot <= 5.0 and float(np.min(margins)) >= -1e-8
        q[int(frame)] = chosen
        rows.append({"frame": int(frame), "side": SIDES[side], "position_mm": pos, "rotation_deg": rot,
                     "branch_margins_m": margins.tolist(), "pass": bool(passed),
                     "policy": f"{policy}_SEED{seed_index}"})
        previous_previous, previous = previous, chosen
    rows.sort(key=lambda r: r["frame"])
    ordered = np.asarray([q[int(f)] for f in sorted(q)])
    vel = float(np.max(np.abs(np.diff(ordered, axis=0)))) if len(ordered) > 1 else 0.0
    acc = float(np.max(np.abs(np.diff(ordered, n=2, axis=0)))) if len(ordered) > 2 else 0.0
    score = (
        sum(not r["pass"] for r in rows),
        max((max(0.0, r["position_mm"] - 10.0) + max(0.0, r["rotation_deg"] - 5.0) for r in rows), default=0.0),
        max((r["position_mm"] + r["rotation_deg"] for r in rows), default=0.0),
        vel,
    )
    return q, rows, {"score": score, "velocity_max": vel, "acceleration_max": acc,
                     "all_gates": score[0] == 0 and vel <= 0.12 + 1e-9 and acc <= 0.06 + 1e-9}


def existing_forward_candidate(side, ids, seed_path, forward_rows):
    """Reuse the immutable forward path; only the reverse lookahead is new CPU work."""
    by_frame = {(int(r["frame"]), r["side"]): r for r in forward_rows if "position_mm" in r}
    rows = [dict(by_frame[(int(frame), SIDES[side])]) for frame in ids]
    q = {int(frame): seed_path[frame, side].copy() for frame in ids}
    ordered = seed_path[ids, side]
    vel = float(np.max(np.abs(np.diff(ordered, axis=0)))) if len(ordered) > 1 else 0.0
    acc = float(np.max(np.abs(np.diff(ordered, n=2, axis=0)))) if len(ordered) > 2 else 0.0
    score = (sum(not r["pass"] for r in rows),
             max((max(0.0, r["position_mm"] - 10.0) + max(0.0, r["rotation_deg"] - 5.0)
                  for r in rows), default=0.0),
             max((r["position_mm"] + r["rotation_deg"] for r in rows), default=0.0), vel)
    return q, rows, {"score": score, "velocity_max": vel, "acceleration_max": acc,
                     "all_gates": score[0] == 0 and vel <= 0.12 + 1e-9 and acc <= 0.06 + 1e-9,
                     "source": "immutable forward canary path"}


def reverse_candidate_or_infeasible(solve):
    """Convert deterministic temporal-bound emptiness into quality evidence.

    A reverse direction whose physical joint limits and previous-accepted
    motion bounds have an empty intersection is not a runtime failure.  It is
    an infeasible candidate direction.  The immutable forward direction can
    still be selected for the segment without loosening any threshold.
    """
    try:
        q, rows, metrics = solve()
    except temporal.TemporalReviewError as error:
        evidence = {
            "status": "HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS",
            "all_gates": False,
            "candidate_available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        return None, evidence
    return ("REVERSE_LOOKAHEAD", q, rows, metrics), metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("chips", "poker"), required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--forward-result", type=Path, required=True)
    ap.add_argument("--forward-states", type=Path, required=True)
    ap.add_argument("--accepted-states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.forward_result, args.forward_states, args.accepted_states):
        p.resolve(strict=True)
    if args.session not in str(args.forward_result.resolve()) or args.session not in str(args.forward_states.resolve()):
        raise RuntimeError("same-session forward input check failed")
    forward_result = json.load(args.forward_result.open())
    if forward_result["status"] not in ("PASS_NUMERIC_CANARY_NO_AUTHORITY", "HOLD_NUMERIC_CANARY"):
        raise RuntimeError("unexpected forward canary status")
    with np.load(args.forward_states, allow_pickle=False) as z:
        states = {k: np.asarray(z[k]) for k in z.files}
    with np.load(args.accepted_states, allow_pickle=False) as z:
        accepted = {k: np.asarray(z[k]) for k in z.files}
    target = states["T_target_hand_root_world"]
    valid = states["valid_side_frame"]
    seed_path = states["q_arm"]
    mounts = states["T_tool_hand_root"]
    world_base = states["T_world_base"]
    base_world = np.linalg.inv(world_base)
    n = len(target)
    assets = old.load_pinned_robot_assets(PROJECT)
    lower, upper = taskfit.arm_limits(assets)
    accepted_q0 = accepted["q_arm"][0]
    q = np.full((n, 2, 7), np.nan, dtype=np.float64)
    actual = np.full_like(target, np.nan)
    rows = []
    segment_reviews = []
    for side in range(2):
        for segment_index, ids in enumerate(segments(valid[side])):
            fq, fr, fm = existing_forward_candidate(side, ids, seed_path, forward_result["rows"])
            if fm["all_gates"]:
                candidates = [("FORWARD", fq, fr, fm)]
                rm = {"not_run": "immutable forward segment already passes every strict gate"}
            else:
                reverse, rm = reverse_candidate_or_infeasible(
                    lambda: solve_direction(
                        assets, side, ids[::-1], target, base_world, mounts,
                        accepted_q0, lower, upper, seed_path,
                    )
                )
                candidates = [("FORWARD", fq, fr, fm)]
                if reverse is not None:
                    candidates.append(reverse)
            direction, chosen_q, chosen_rows, chosen_metrics = min(candidates, key=lambda x: x[3]["score"])
            for frame, value in chosen_q.items():
                q[frame, side] = value
                actual[frame, side] = world_base @ old.official._tool_fk(assets, side, value) @ mounts[side]
            rows.extend(chosen_rows)
            segment_reviews.append({
                "side": SIDES[side], "segment_index": segment_index, "frame_start": int(ids[0]), "frame_end": int(ids[-1]),
                "selected_direction": direction, "selected": chosen_metrics,
                "forward": fm, "reverse_lookahead": rm,
            })
    for side in range(2):
        for frame in np.flatnonzero(~valid[side]):
            rows.append({"frame": int(frame), "side": SIDES[side], "status": "UNKNOWN_MISSING_HAWOR", "pass": False})
    solved = [r for r in rows if "position_mm" in r]
    failed = [r for r in solved if not r["pass"]]
    velocity = []
    acceleration = []
    for side in range(2):
        for ids in segments(valid[side]):
            qq = q[ids, side]
            if len(qq) > 1:
                velocity.extend(np.abs(np.diff(qq, axis=0)).ravel())
            if len(qq) > 2:
                acceleration.extend(np.abs(np.diff(qq, n=2, axis=0)).ravel())
    vel = float(max(velocity, default=0.0))
    acc = float(max(acceleration, default=0.0))
    gates = {"pose_branch_all_observed": len(failed) == 0, "velocity": vel <= 0.12 + 1e-9,
             "acceleration": acc <= 0.06 + 1e-9, "missing_unknown_not_filled": bool(np.all(np.isnan(q[~valid.T])))}
    state_path = args.output.with_name("ARM_BIDIRECTIONAL_STATES.npz")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state = state_path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp_state, q_arm=q, T_world_base=world_base, T_tool_hand_root=mounts,
                        T_target_hand_root_world=target, T_actual_hand_root_world=actual,
                        valid_side_frame=valid, source_frames=states["source_frames"])
    os.replace(tmp_state, state_path)
    payload = {
        "schema_version": "robot-arm-segment-bidirectional-lookahead-v3", "created_at": now(),
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY" if all(gates.values()) else "HOLD_NUMERIC_CANARY",
        "task": args.task, "session": args.session, "frame_count": n,
        "lineage": {"forward_result": {"path": str(args.forward_result.resolve()), "sha256": sha(args.forward_result)},
                    "forward_states": {"path": str(args.forward_states.resolve()), "sha256": sha(args.forward_states)},
                    "accepted_states": {"path": str(args.accepted_states.resolve()), "sha256": sha(args.accepted_states)}},
        "contract": "One feasible direction per observed contiguous segment; reverse direction is endpoint lookahead. Empty temporal-bound intersections are explicit HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS evidence and never loosen a threshold. No solve, hold, interpolation, or temporal constraint crosses UNKNOWN gaps.",
        "metrics": {"observed_rows": len(solved), "unknown_rows": int((~valid).sum()), "failed_rows": len(failed),
                    "infeasible_direction_count": sum(
                        review["reverse_lookahead"].get("status") == "HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS"
                        for review in segment_reviews
                    ),
                    "position_mm_max": max((r["position_mm"] for r in solved), default=float("nan")),
                    "rotation_deg_max": max((r["rotation_deg"] for r in solved), default=float("nan")),
                    "branch_margin_m_min": min((min(r["branch_margins_m"]) for r in solved), default=float("nan")),
                    "velocity_rad_per_frame_max_contiguous": vel, "acceleration_rad_per_frame2_max_contiguous": acc},
        "gates": gates, "segments": segment_reviews, "rows": sorted(rows, key=lambda r: (r["frame"], r["side"])),
        "output_states": {"path": str(state_path), "sha256": sha(state_path), "bytes": state_path.stat().st_size},
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "CPU numeric lookahead canary only; no visual, sidecar, Robot, contact, or training authority.",
    }
    tmp = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"result": str(args.output), "sha256": sha(args.output), "status": payload["status"], "metrics": payload["metrics"]}))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
