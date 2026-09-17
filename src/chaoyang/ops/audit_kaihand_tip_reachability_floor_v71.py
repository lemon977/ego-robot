#!/usr/bin/env python3
"""Measure a deterministic KaiHand fingertip-direction reachability floor."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

from chaoyang.ops import diagnose_kaihand_anatomical_temporal_v3 as anatomy
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as old
from chaoyang.ops import run_newtask_robot_shared_v4_hand as handfit


ROOT = Path(__file__).resolve().parents[3]
SIDES = {"left": 0, "right": 1}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact(item: dict[str, Any], label: str) -> Path:
    path = Path(str(item["path"])).resolve(strict=True)
    if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"{label} bytes mismatch")
    if item.get("sha256") is not None and sha256(path) != item["sha256"]:
        raise RuntimeError(f"{label} SHA mismatch")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def conclusion(floor_deg: float, threshold_deg: float) -> str:
    return (
        "TARGET_OUTSIDE_SAMPLED_ROBOT_REACHABLE_SET"
        if floor_deg > threshold_deg + 1e-9
        else "TARGET_REACHABLE_UNDER_TIP_ONLY_OBJECTIVE"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-result", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--side", choices=tuple(SIDES), required=True)
    parser.add_argument("--finger", choices=handfit.FINGERS, required=True)
    parser.add_argument("--random-seeds", type=int, default=40)
    parser.add_argument("--threshold-deg", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.hand_result.resolve(strict=True)
    result = load(result_path)
    state_path = exact(result["output_states"], "hand states")
    hawor_path = exact(result["lineage"]["hawor"], "HaWoR")
    with np.load(state_path, allow_pickle=False) as archive:
        states = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(hawor_path, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}

    side = SIDES[args.side]
    frame = args.frame
    q = np.asarray(states["q_hand"][frame, side], dtype=np.float64)
    points = np.asarray(hawor["joints_3d_world"][side, frame], dtype=np.float64)
    if not np.isfinite(points).all() or not np.isfinite(q).all():
        raise RuntimeError("selected side/frame is not an observed finite row")

    assets = old.load_pinned_robot_assets(ROOT)
    contract = handfit.model_contract(old.official, shared.wrist_adapter, assets)[side]
    finger = args.finger
    group = handfit.GROUPS[finger]
    lower, upper = contract["lower"][group], contract["upper"][group]
    tip = anatomy.terminal_tip(contract["model"], finger, contract["prefix"])[0]
    wanted = anatomy.semantic_target(
        handfit.human_features(shared.wrist_adapter, points, shared.SIDES[side])[finger]
    )

    def features(x: np.ndarray) -> dict[str, np.ndarray]:
        whole = q.copy()
        whole[group] = x
        return anatomy.features(old.official, contract, whole, finger, tip)

    current = anatomy.row(features(q[group]), wanted)
    rng = np.random.default_rng(7)
    seeds = [q[group], contract["neutral"][group], lower, upper]
    seeds.extend(rng.uniform(lower, upper) for _ in range(args.random_seeds))
    attempts = []
    for index, seed in enumerate(seeds):
        solved = least_squares(
            lambda x: features(x)["tip"] - wanted["tip"],
            np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=1000,
            ftol=1e-13, xtol=1e-13, gtol=1e-13,
        )
        metric = anatomy.row(features(solved.x), wanted)
        attempts.append({
            "seed_index": index,
            "tip_direction_error_deg": metric["tip_direction_error_deg"],
            "bone_error_deg_max": metric["bone_error_deg_max"],
            "q": solved.x.tolist(),
        })
    best = min(attempts, key=lambda row: row["tip_direction_error_deg"])
    best_q = np.asarray(best["q"])
    at_limit = np.isclose(best_q, lower, atol=1e-6) | np.isclose(best_q, upper, atol=1e-6)
    payload = {
        "schema_version": "kaihand-tip-reachability-floor-v71-v1",
        "status": "PASSED_DEVELOPMENT_REACHABILITY_DIAGNOSIS",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": result.get("task"),
        "session": result.get("session"),
        "frame": frame,
        "side": args.side,
        "finger": finger,
        "threshold_deg": args.threshold_deg,
        "current": {
            "tip_direction_error_deg": current["tip_direction_error_deg"],
            "bone_error_deg_max": current["bone_error_deg_max"],
            "q": q[group].tolist(),
        },
        "tip_only_reachability_floor": best,
        "joint_lower": lower.tolist(),
        "joint_upper": upper.tolist(),
        "best_at_joint_limit": at_limit.tolist(),
        "all_best_joints_at_limit": bool(at_limit.all()),
        "attempt_count": len(attempts),
        "diagnosis": conclusion(best["tip_direction_error_deg"], args.threshold_deg),
        "inputs": {"hand_result": ref(result_path), "states": ref(state_path), "hawor": ref(hawor_path)},
        "code": ref(Path(__file__)),
        "authority": False,
        "claim_limit": "Deterministic sampled digital reachability diagnosis for one side/frame/finger; not proof of global physical reachability, human truth, Robot authority or calibration.",
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "diagnosis": payload["diagnosis"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
