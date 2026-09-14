#!/usr/bin/env python3
"""Select forward/reverse temporal projection for each observed hand segment."""
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
    return [x for x in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(x)] if len(ids) else []


def temporal_candidate(raw: np.ndarray, ids: np.ndarray, side: int, contracts, reverse: bool) -> np.ndarray:
    order = ids[::-1] if reverse else ids
    pair = np.empty((len(order), 2, 22), dtype=np.float64)
    pair[:, 0] = contracts[0]["neutral"]
    pair[:, 1] = contracts[1]["neutral"]
    pair[:, side] = raw[order, side]
    handfit.VELOCITY_RAD_PER_FRAME = 0.08
    handfit.ACCELERATION_RAD_PER_FRAME2 = 0.06
    projected, _, _ = handfit.temporal_project(v2, pair, contracts, np.arange(len(order)), 30.0)
    values = projected[:, side]
    return values[::-1] if reverse else values


def audit(values, ids, side, hawor, contracts, tips):
    rows = []
    for i, frame in enumerate(ids):
        target_features = handfit.human_features(shared.wrist_adapter, hawor["joints_3d_world"][side, frame], shared.SIDES[side])
        fingers = []
        for finger in handfit.FINGERS:
            wanted = anatomy.semantic_target(target_features[finger])
            metric = anatomy.row(anatomy.features(old.official, contracts[side], values[i], finger,
                                                   tips[side, finger]), wanted)
            metric["finger"] = finger
            fingers.append(metric)
        bone = max(x["bone_error_deg_max"] for x in fingers)
        tip = max(x["tip_direction_error_deg"] for x in fingers)
        rows.append({"frame": int(frame), "side": SIDES[side], "bone_error_deg_max": bone,
                     "tip_direction_error_deg_max": tip, "fingers": fingers,
                     "pass": bool(bone <= 60.0 and tip <= 15.0)})
    vel = float(np.max(np.abs(np.diff(values, axis=0)))) if len(values) > 1 else 0.0
    acc = float(np.max(np.abs(np.diff(values, n=2, axis=0)))) if len(values) > 2 else 0.0
    failures = sum(not x["pass"] for x in rows)
    excess = max((max(0.0, x["bone_error_deg_max"] - 60.0) +
                  4.0 * max(0.0, x["tip_direction_error_deg_max"] - 15.0) for x in rows), default=0.0)
    score = (failures, excess,
             max((x["bone_error_deg_max"] + x["tip_direction_error_deg_max"] for x in rows), default=0.0),
             vel, acc)
    return rows, {"score": score, "failures": failures, "velocity_max": vel, "acceleration_max": acc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("chips", "poker"), required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--hawor", type=Path, required=True)
    ap.add_argument("--forward-result", type=Path, required=True)
    ap.add_argument("--forward-states", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.hawor, args.forward_result, args.forward_states):
        p.resolve(strict=True)
    if args.session not in str(args.hawor.resolve()) or args.session not in str(args.forward_result.resolve()):
        raise RuntimeError("same-session input check failed")
    with np.load(args.hawor, allow_pickle=False) as z:
        hawor = {k: np.asarray(z[k]) for k in z.files}
    with np.load(args.forward_states, allow_pickle=False) as z:
        state = {k: np.asarray(z[k]) for k in z.files}
    n = len(hawor["c2w"])
    raw = state["q_hand_raw"]
    valid = state["valid_side_frame"]
    assets = old.load_pinned_robot_assets(PROJECT)
    contracts = handfit.model_contract(old.official, shared.wrist_adapter, assets)
    if list(handfit.GROUPS["thumb"]) != list(range(6)):
        raise RuntimeError("thumb q[0:6] semantic contract changed")
    tips = {(side, finger): anatomy.terminal_tip(contract["model"], finger, contract["prefix"])[0]
            for side, contract in enumerate(contracts) for finger in handfit.FINGERS}
    q = np.full((n, 2, 22), np.nan, dtype=np.float64)
    rows, reviews = [], []
    for side in range(2):
        for segment_index, ids in enumerate(segments(valid[side])):
            candidates = []
            for direction, reverse in (("FORWARD", False), ("REVERSE_LOOKAHEAD", True)):
                values = temporal_candidate(raw, ids, side, contracts, reverse)
                candidate_rows, metrics = audit(values, ids, side, hawor, contracts, tips)
                candidates.append((direction, values, candidate_rows, metrics))
            direction, values, selected_rows, selected_metrics = min(candidates, key=lambda x: x[3]["score"])
            q[ids, side] = values
            rows.extend(selected_rows)
            reviews.append({"side": SIDES[side], "segment_index": segment_index,
                            "frame_start": int(ids[0]), "frame_end": int(ids[-1]),
                            "selected_direction": direction, "selected": selected_metrics,
                            "forward": candidates[0][3], "reverse_lookahead": candidates[1][3]})
    for side in range(2):
        for frame in np.flatnonzero(~valid[side]):
            rows.append({"frame": int(frame), "side": SIDES[side],
                         "status": "UNKNOWN_MISSING_HAWOR", "pass": False})
    solved = [r for r in rows if "bone_error_deg_max" in r]
    failed = [r for r in solved if not r["pass"]]
    vel, acc = [], []
    for side in range(2):
        for ids in segments(valid[side]):
            if len(ids) > 1:
                vel.extend(np.abs(np.diff(q[ids, side], axis=0)).ravel())
            if len(ids) > 2:
                acc.extend(np.abs(np.diff(q[ids, side], n=2, axis=0)).ravel())
    vel_max, acc_max = float(max(vel, default=0.0)), float(max(acc, default=0.0))
    gates = {"anatomy_all_observed": len(failed) == 0, "velocity": vel_max <= 0.08 + 1e-9,
             "acceleration": acc_max <= 0.06 + 1e-9,
             "missing_unknown_not_filled": bool(np.all(np.isnan(q[~valid.T]))),
             "thumb_independent_q0_to_q5": True, "four_finger_chain_semantics": True}
    state_path = args.output.with_name("HAND_BIDIRECTIONAL_STATES.npz")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_state = state_path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp_state, q_hand=q, q_hand_raw=raw, valid_side_frame=valid,
                        source_frames=state["source_frames"])
    os.replace(tmp_state, state_path)
    payload = {
        "schema_version": "robot-kaihand-segment-bidirectional-lookahead-v3", "created_at": now(),
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY" if all(gates.values()) else "HOLD_NUMERIC_CANARY",
        "task": args.task, "session": args.session, "frame_count": n,
        "implementation": {"path": str(HAND_FIT_IMPLEMENTATION.resolve()),
                           "sha256": sha(HAND_FIT_IMPLEMENTATION),
                           "bytes": HAND_FIT_IMPLEMENTATION.stat().st_size},
        "lineage": {"hawor": {"path": str(args.hawor.resolve()), "sha256": sha(args.hawor)},
                    "forward_result": {"path": str(args.forward_result.resolve()), "sha256": sha(args.forward_result)},
                    "forward_states": {"path": str(args.forward_states.resolve()), "sha256": sha(args.forward_states)}},
        "contract": "One temporal direction per observed segment; reverse is endpoint lookahead. UNKNOWN gaps are never crossed or filled. Thumb q[0:6] and per-finger MCP->PIP->DIP->TIP semantics unchanged.",
        "metrics": {"observed_rows": len(solved), "unknown_rows": int((~valid).sum()),
                    "failed_rows": len(failed),
                    "bone_error_deg_max": max((r["bone_error_deg_max"] for r in solved), default=float("nan")),
                    "tip_direction_error_deg_max": max((r["tip_direction_error_deg_max"] for r in solved), default=float("nan")),
                    "velocity_rad_per_frame_max_contiguous": vel_max,
                    "acceleration_rad_per_frame2_max_contiguous": acc_max},
        "gates": gates, "segments": reviews,
        "rows": sorted(rows, key=lambda r: (r["frame"], r["side"])),
        "output_states": {"path": str(state_path), "sha256": sha(state_path), "bytes": state_path.stat().st_size},
        "authority": False, "action_sidecar_published": False,
        "claim_limit": "Numeric KaiHand lookahead canary only; not action/contact/Robot authority or visual approval.",
    }
    tmp = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"result": str(args.output), "sha256": sha(args.output),
                      "status": payload["status"], "metrics": payload["metrics"]}))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
