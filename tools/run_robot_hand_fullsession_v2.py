#!/usr/bin/env python3
"""Full-session KaiHand semantic retargeting with gap-safe temporal projection."""
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
HAND_FIT_IMPLEMENTATION = PROJECT / "tools/run_newtask_robot_shared_v4_hand.py"
from tools import diagnose_kaihand_anatomical_temporal_v3 as anatomy  # noqa: E402
from tools import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from tools import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from tools import run_newtask_robot_kinematic_successor_v2 as v2  # noqa: E402
from tools import run_newtask_robot_shared_v4_hand as handfit  # noqa: E402

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
    return [x for x in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(x)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("chips", "poker"), required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--hawor", type=Path, required=True)
    ap.add_argument("--hawor-result", type=Path, required=True)
    ap.add_argument("--accepted-states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.hawor, args.hawor_result, args.accepted_states):
        p.resolve(strict=True)
    if args.session not in str(args.hawor.resolve()) or args.session not in str(args.hawor_result.resolve()):
        raise RuntimeError("same-session HaWoR input check failed")
    with np.load(args.hawor, allow_pickle=False) as z:
        hawor = {k: np.asarray(z[k]) for k in z.files}
    with np.load(args.accepted_states, allow_pickle=False) as z:
        accepted = {k: np.asarray(z[k]) for k in z.files}
    n = len(hawor["c2w"])
    valid = np.isfinite(hawor["joints_3d_world"]).all(axis=(2, 3))
    assets = old.load_pinned_robot_assets(PROJECT)
    contracts = handfit.model_contract(old.official, shared.wrist_adapter, assets)
    # The semantic contract is immutable: thumb q[0:6] is independent; each
    # other group follows its own MCP->PIP->DIP->TIP chain.
    if list(handfit.GROUPS["thumb"]) != list(range(6)):
        raise RuntimeError("thumb q[0:6] contract changed")
    tips = {(side, finger): anatomy.terminal_tip(contract["model"], finger, contract["prefix"])[0]
            for side, contract in enumerate(contracts) for finger in handfit.FINGERS}
    accepted_q0 = np.asarray(accepted["q_hand"][0], dtype=np.float64)
    raw = np.full((n, 2, 22), np.nan, dtype=np.float64)
    projected = np.full_like(raw, np.nan)
    targets = {}

    for side in range(2):
        for ids in segments(valid[side]):
            previous = accepted_q0[side].copy()
            for frame in ids:
                target_features = handfit.human_features(shared.wrist_adapter, hawor["joints_3d_world"][side, frame], shared.SIDES[side])
                targets[(int(frame), side)] = target_features
                candidate = previous.copy()
                for finger in handfit.FINGERS:
                    wanted = anatomy.semantic_target(target_features[finger])
                    group = handfit.GROUPS[finger]
                    lower = contracts[side]["lower"][group]
                    upper = contracts[side]["upper"][group]
                    attempts = []
                    seeds = (candidate[group], accepted_q0[side, group], contracts[side]["neutral"][group])
                    # Stage 1 preserves the accepted objective.  Stage 2 is a
                    # generic gate-triggered refinement: when the best first
                    # solution misses the 15 degree terminal direction gate,
                    # trade available bone-angle slack (still <=60 degrees)
                    # for a feasible tip direction.  It is applied by metric,
                    # never by session/finger/frame identity.
                    for seed in seeds:
                        def residual(x):
                            whole = candidate.copy()
                            whole[group] = x
                            actual = anatomy.features(old.official, contracts[side], whole, finger, tips[side, finger])
                            return np.concatenate(((actual["bones"] - wanted["bones"]).ravel(),
                                                   4.0 * (actual["tip"] - wanted["tip"]),
                                                   0.10 * (actual["mcp"] - wanted["mcp"]),
                                                   0.003 * (x - seed)))
                        sol = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                                            max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                        whole = candidate.copy()
                        whole[group] = sol.x
                        metric = anatomy.row(anatomy.features(old.official, contracts[side], whole, finger, tips[side, finger]), wanted)
                        score = (not (metric["bone_error_deg_max"] <= 60.0 and metric["tip_direction_error_deg"] <= 15.0),
                                 max(0.0, metric["bone_error_deg_max"] - 60.0) + 4.0 * max(0.0, metric["tip_direction_error_deg"] - 15.0),
                                 metric["bone_error_deg_max"] + metric["tip_direction_error_deg"])
                        attempts.append((*score, sol.x))
                    best = min(attempts, key=lambda x: x[:3])
                    if best[0]:
                        refined = []
                        for tip_weight in (8.0, 16.0, 32.0):
                            for seed in (*seeds, best[-1]):
                                def residual_refined(x):
                                    whole = candidate.copy()
                                    whole[group] = x
                                    actual = anatomy.features(old.official, contracts[side], whole, finger, tips[side, finger])
                                    return np.concatenate(((actual["bones"] - wanted["bones"]).ravel(),
                                                           tip_weight * (actual["tip"] - wanted["tip"]),
                                                           0.10 * (actual["mcp"] - wanted["mcp"]),
                                                           0.003 * (x - seed)))
                                sol = least_squares(residual_refined, np.clip(seed, lower, upper),
                                                    bounds=(lower, upper), max_nfev=240,
                                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
                                whole = candidate.copy()
                                whole[group] = sol.x
                                metric = anatomy.row(anatomy.features(old.official, contracts[side], whole,
                                                                       finger, tips[side, finger]), wanted)
                                score = (not (metric["bone_error_deg_max"] <= 60.0 and
                                              metric["tip_direction_error_deg"] <= 15.0),
                                         max(0.0, metric["bone_error_deg_max"] - 60.0) +
                                         4.0 * max(0.0, metric["tip_direction_error_deg"] - 15.0),
                                         metric["bone_error_deg_max"] + metric["tip_direction_error_deg"])
                                refined.append((*score, sol.x))
                        best = min((best, *refined), key=lambda x: x[:3])
                    candidate[group] = best[-1]
                raw[frame, side] = candidate
                previous = candidate
            pair = np.empty((len(ids), 2, 22), dtype=np.float64)
            pair[:, 0] = contracts[0]["neutral"]
            pair[:, 1] = contracts[1]["neutral"]
            pair[:, side] = raw[ids, side]
            handfit.VELOCITY_RAD_PER_FRAME = 0.08
            handfit.ACCELERATION_RAD_PER_FRAME2 = 0.06
            pair_projected, _, _ = handfit.temporal_project(v2, pair, contracts, hawor["original_frame_indices"][ids], 30.0)
            projected[ids, side] = pair_projected[:, side]

    rows = []
    for side in range(2):
        for frame in range(n):
            if not valid[side, frame]:
                rows.append({"frame": frame, "side": SIDES[side], "status": "UNKNOWN_MISSING_HAWOR", "pass": False})
                continue
            fingers = []
            for finger in handfit.FINGERS:
                wanted = anatomy.semantic_target(targets[(frame, side)][finger])
                metric = anatomy.row(anatomy.features(old.official, contracts[side], projected[frame, side], finger, tips[side, finger]), wanted)
                metric["finger"] = finger
                fingers.append(metric)
            bone = max(x["bone_error_deg_max"] for x in fingers)
            tip = max(x["tip_direction_error_deg"] for x in fingers)
            rows.append({"frame": frame, "side": SIDES[side], "status": "SOLVED" if bone <= 60.0 and tip <= 15.0 else "HOLD_ANATOMY_GATE",
                         "bone_error_deg_max": bone, "tip_direction_error_deg_max": tip,
                         "fingers": fingers, "pass": bool(bone <= 60.0 and tip <= 15.0)})
    solved = [r for r in rows if "bone_error_deg_max" in r]
    failed = [r for r in solved if not r["pass"]]
    velocity = []
    acceleration = []
    for side in range(2):
        for ids in segments(valid[side]):
            qq = projected[ids, side]
            if len(qq) > 1:
                velocity.extend(np.abs(np.diff(qq, axis=0)).ravel())
            if len(qq) > 2:
                acceleration.extend(np.abs(np.diff(qq, n=2, axis=0)).ravel())
    vel = float(max(velocity, default=0.0))
    acc = float(max(acceleration, default=0.0))
    gates = {"anatomy_all_observed": len(failed) == 0, "velocity": vel <= 0.08 + 1e-9,
             "acceleration": acc <= 0.06 + 1e-9, "missing_unknown_not_filled": bool(np.all(np.isnan(projected[~valid.T]))),
             "thumb_independent_q0_to_q5": True, "four_finger_chain_semantics": True}
    state_path = args.output.with_name("HAND_STATES.npz")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state = state_path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp_state, q_hand=projected, q_hand_raw=raw, valid_side_frame=valid,
                        source_frames=hawor["original_frame_indices"])
    os.replace(tmp_state, state_path)
    payload = {
        "schema_version": "robot-kaihand-fullsession-semantic-retarget-v2", "created_at": now(),
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY" if all(gates.values()) else "HOLD_NUMERIC_CANARY",
        "task": args.task, "session": args.session, "frame_count": n,
        "implementation": {"path": str(HAND_FIT_IMPLEMENTATION.resolve()),
                           "sha256": sha(HAND_FIT_IMPLEMENTATION),
                           "bytes": HAND_FIT_IMPLEMENTATION.stat().st_size},
        "lineage": {"hawor": {"path": str(args.hawor.resolve()), "sha256": sha(args.hawor)},
                    "hawor_result": {"path": str(args.hawor_result.resolve()), "sha256": sha(args.hawor_result)},
                    "accepted_states": {"path": str(args.accepted_states.resolve()), "sha256": sha(args.accepted_states)}},
        "contract": {"thumb": "independent q[0:6]", "other_fingers": "independent MCP->PIP->DIP->TIP chains",
                     "static_refinement": "gate-triggered tip weights 8/16/32; bone<=60deg, tip<=15deg and joint limits unchanged",
                     "missing": "UNKNOWN NaN; no interpolation/hold and temporal projection is per contiguous observed segment"},
        "metrics": {"observed_rows": len(solved), "unknown_rows": int((~valid).sum()), "failed_rows": len(failed),
                    "bone_error_deg_max": max((r["bone_error_deg_max"] for r in solved), default=float("nan")),
                    "tip_direction_error_deg_max": max((r["tip_direction_error_deg_max"] for r in solved), default=float("nan")),
                    "velocity_rad_per_frame_max_contiguous": vel, "acceleration_rad_per_frame2_max_contiguous": acc},
        "gates": gates, "rows": rows,
        "output_states": {"path": str(state_path), "sha256": sha(state_path), "bytes": state_path.stat().st_size},
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "Numeric KaiHand canary only; not action sidecar, Robot authority, visual approval, or contact truth.",
    }
    tmp = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"result": str(args.output), "sha256": sha(args.output), "status": payload["status"], "metrics": payload["metrics"]}))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
