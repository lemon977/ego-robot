#!/usr/bin/env python3
"""Per-session fixed-base solve for Exact78 P1.

No foreign-session camera/base/q/offset value is read.  The only static
attachment input is the global development-only mount proxy.  Numeric misses
are frame statuses and never abort the session.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm


SIDES = ("left", "right")
POSITION_GATE_MM = 10.0
ROTATION_GATE_DEG = 5.0
MOUNT_CONTRACT = (REPO_ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1"
                  / "excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json")


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def mounts() -> tuple[np.ndarray, dict]:
    value = json.loads(MOUNT_CONTRACT.read_text(encoding="utf-8"))
    if value.get("status") != "FROZEN_VISUAL_REVIEW_PROXY_NOT_CALIBRATION":
        raise RuntimeError("MOUNT_PROXY_AUTHORITY_DRIFT")
    proxy = value["proxy_transforms"]
    if proxy["matrix_direction"] != "p_flange = T_flange_hand_root_proxy @ p_hand_root":
        raise RuntimeError("MOUNT_PROXY_DIRECTION_DRIFT")
    result = np.asarray((proxy["left_flange_L_to_hand_l_base_link"],
                         proxy["right_flange_R_to_hand_r_base_link"]), dtype=np.float64)
    for matrix in result:
        if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10):
            raise RuntimeError("MOUNT_PROXY_ROTATION_DRIFT")
    return result, artifact_ref(MOUNT_CONTRACT)


def rz(degrees: float) -> np.ndarray:
    value = math.radians(degrees)
    return np.asarray(((math.cos(value), -math.sin(value), 0.0),
                       (math.sin(value), math.cos(value), 0.0),
                       (0.0, 0.0, 1.0)), dtype=np.float64)


def proper(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = position
    return value


def candidate_base(c2w: np.ndarray, midpoint_world: np.ndarray,
                   x: float, y: float, z: float, yaw: float) -> np.ndarray:
    # Camera x=right,y=down,z=forward; base x=forward,y=left,z=up.
    camera_base = np.asarray(((0.0, -1.0, 0.0),
                              (0.0, 0.0, -1.0),
                              (1.0, 0.0, 0.0)), dtype=np.float64) @ rz(yaw)
    rotation = c2w[:3, :3] @ camera_base
    translation = midpoint_world - rotation @ np.asarray((x, y, z), dtype=np.float64)
    return proper(translation, rotation)


def sample_ids(metric: np.ndarray, visual: np.ndarray, maximum: int = 8) -> np.ndarray:
    ids = np.flatnonzero(np.any(metric, axis=0))
    if not len(ids):
        ids = np.flatnonzero(np.any(visual, axis=0))
    if not len(ids):
        return np.asarray((0,), dtype=np.int64)
    indices = np.rint(np.linspace(0, len(ids) - 1, min(maximum, len(ids)))).astype(int)
    return np.unique(ids[indices])


def solve_global(assets, side: int, target_tool: np.ndarray,
                 lower: np.ndarray, upper: np.ndarray, seeds: list[np.ndarray]) -> tuple:
    return arm.solve_one(assets, side, target_tool, lower, upper, seeds)


def target_rotation(human: np.ndarray, side: int, frame: int, anchor: int,
                    anchor_rotation: np.ndarray) -> np.ndarray:
    current = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
        human[side, frame], handedness=arm.shared.SIDES[side])
    origin = arm.shared.wrist_adapter.final_v3_mano_palm_basis(
        human[side, anchor], handedness=arm.shared.SIDES[side])
    return current @ origin.T @ anchor_rotation


def evaluate_candidate(spec: tuple[float, float, float, float], hawor: dict, human: np.ndarray,
                       visual: np.ndarray, sample: np.ndarray, anchors: np.ndarray,
                       assets, mount: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                       neutral: np.ndarray) -> dict:
    bilateral = np.flatnonzero(visual[0] & visual[1])
    anchor_mid = int(bilateral[0]) if len(bilateral) else int(min(anchors))
    points = [human[side, anchors[side], 0] for side in (0, 1) if visual[side, anchors[side]]]
    midpoint = np.mean(points, axis=0)
    world_base = candidate_base(hawor["c2w"][anchor_mid], midpoint, *spec)
    base_world = np.linalg.inv(world_base)
    anchor_rotations = []
    offsets = np.zeros((2, 3), dtype=np.float64)
    anchor_q = np.zeros((2, 7), dtype=np.float64)
    for side in (0, 1):
        neutral_root = world_base @ arm.old.official._tool_fk(assets, side, neutral[side]) @ mount[side]
        anchor_rotation = neutral_root[:3, :3]
        anchor_rotations.append(anchor_rotation)
        wanted = proper(human[side, anchors[side], 0], anchor_rotation)
        attempt = solve_global(assets, side, base_world @ wanted @ np.linalg.inv(mount[side]),
                               lower[side], upper[side],
                               [neutral[side], .25 * lower[side] + .75 * upper[side],
                                .75 * lower[side] + .25 * upper[side]])
        *_, chosen, _, _, _, _ = attempt
        actual = world_base @ arm.old.official._tool_fk(assets, side, chosen) @ mount[side]
        offsets[side] = actual[:3, 3] - human[side, anchors[side], 0]
        anchor_q[side] = chosen
    rows = []
    for frame in sample:
        for side in (0, 1):
            if not visual[side, frame]:
                continue
            root = proper(human[side, frame, 0] + offsets[side],
                          target_rotation(human, side, int(frame), int(anchors[side]), anchor_rotations[side]))
            attempt = solve_global(
                assets, side, base_world @ root @ np.linalg.inv(mount[side]), lower[side], upper[side],
                [anchor_q[side], neutral[side], .25 * lower[side] + .75 * upper[side],
                 .75 * lower[side] + .25 * upper[side]])
            *_, chosen, _, position, rotation, margins = attempt
            passed = position <= POSITION_GATE_MM and rotation <= ROTATION_GATE_DEG and min(margins) >= -1e-8
            rows.append((bool(passed), float(position), float(rotation), float(min(margins))))
    passes = sum(row[0] for row in rows)
    score = (-passes, sum(not row[0] for row in rows),
             float(np.median([row[1] for row in rows])) if rows else 1e9,
             float(np.median([row[2] for row in rows])) if rows else 1e9,
             -float(min((row[3] for row in rows), default=-1e9)))
    return {"spec": list(spec), "T_world_base": world_base, "offsets": offsets,
            "anchor_q": anchor_q, "anchor_rotations": np.asarray(anchor_rotations),
            "sample_rows": len(rows), "sample_passed": passes, "score": score}


def classify(position: float, rotation: float, margin: float) -> str:
    if margin < -1e-8:
        return "BRANCH"
    if position > POSITION_GATE_MM:
        return "REACH"
    if rotation > ROTATION_GATE_DEG:
        return "ORIENT"
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    output.mkdir(parents=True)
    hawor_path = args.hawor.resolve(strict=True)
    if args.session_id not in str(hawor_path):
        raise RuntimeError("SAME_SESSION_HAWOR_REQUIRED")
    hawor = arrays(hawor_path)
    human = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    visual = np.asarray(hawor["visual_valid"], dtype=bool)
    metric = np.asarray(hawor["metric_observed"], dtype=bool)
    provenance = np.asarray(hawor["provenance"])
    frame_count = human.shape[1]
    mount, mount_ref = mounts()
    assets = arm.old.load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm.taskfit.arm_limits(assets)
    neutral = .5 * (lower + upper)
    anchors = np.asarray([int(np.flatnonzero(visual[side])[0]) if visual[side].any() else 0
                          for side in (0, 1)], dtype=np.int64)
    sample = sample_ids(metric, visual)
    coarse = [(x, y, z, yaw) for x in (0.50, 0.65) for y in (-0.05, 0.05)
              for z in (0.60, 0.75) for yaw in (-15.0, 0.0, 15.0)]
    candidates = [evaluate_candidate(value, hawor, human, visual, sample, anchors,
                                     assets, mount, lower, upper, neutral) for value in coarse]
    first = min(candidates, key=lambda row: row["score"])
    x, y, z, yaw = first["spec"]
    fine = [(x, y, z, yaw), (x-.025, y, z, yaw), (x+.025, y, z, yaw),
            (x, y-.025, z, yaw), (x, y+.025, z, yaw), (x, y, z-.025, yaw),
            (x, y, z+.025, yaw), (x, y, z, yaw-5.0), (x, y, z, yaw+5.0)]
    candidates.extend(evaluate_candidate(value, hawor, human, visual, sample, anchors,
                                         assets, mount, lower, upper, neutral) for value in fine[1:])
    chosen = min(candidates, key=lambda row: row["score"])
    world_base = chosen["T_world_base"]
    base_world = np.linalg.inv(world_base)
    offsets = chosen["offsets"]
    anchor_rotations = chosen["anchor_rotations"]

    q = np.zeros((frame_count, 2, 7), dtype=np.float64)
    target = np.zeros((frame_count, 2, 4, 4), dtype=np.float64)
    actual = np.zeros_like(target)
    rows = []
    for side in (0, 1):
        previous = None
        previous_previous = None
        for frame in range(frame_count):
            if not visual[side, frame]:
                selected = neutral[side]
                target[frame, side] = world_base @ arm.old.official._tool_fk(assets, side, selected) @ mount[side]
                actual[frame, side] = target[frame, side]
                q[frame, side] = selected
                rows.append({"frame": frame, "side": SIDES[side], "status": "NEUTRAL_NO_MOTION",
                             "pass": False, "metric_observed": False, "provenance": str(provenance[side, frame])})
                previous_previous, previous = previous, selected
                continue
            root = proper(human[side, frame, 0] + offsets[side],
                          target_rotation(human, side, frame, int(anchors[side]), anchor_rotations[side]))
            target[frame, side] = root
            tool = base_world @ root @ np.linalg.inv(mount[side])
            if previous is None:
                lo, hi = lower[side], upper[side]
                seeds = [chosen["anchor_q"][side], neutral[side],
                         .25 * lower[side] + .75 * upper[side], .75 * lower[side] + .25 * upper[side]]
            else:
                lo, hi = arm.temporal.bounded_limits(lower[side], upper[side], previous,
                                                     previous_previous, arm.temporal.ARM_SOLVER_STEP_LIMIT)
                seeds = [np.clip(previous, lo, hi), np.clip(chosen["anchor_q"][side], lo, hi),
                         np.clip(neutral[side], lo, hi)]
            attempt = arm.solve_one(assets, side, tool, lower[side], upper[side], seeds,
                                    previous=previous, lo=lo, hi=hi)
            *_, solved, seed_index, position, rotation, margins = attempt
            reason = classify(float(position), float(rotation), float(min(margins)))
            passed = reason == "PASS"
            selected = solved if passed or previous is None else previous
            q[frame, side] = selected
            actual[frame, side] = world_base @ arm.old.official._tool_fk(assets, side, selected) @ mount[side]
            status = "PASS" if passed else f"HOLD_{reason}"
            rows.append({"frame": frame, "side": SIDES[side], "status": status,
                         "position_mm": float(position), "rotation_deg": float(rotation),
                         "branch_margin_m_min": float(min(margins)), "seed_index": int(seed_index),
                         "pass": bool(passed), "metric_observed": bool(metric[side, frame]),
                         "provenance": str(provenance[side, frame])})
            previous_previous, previous = previous, selected

    metric_rows = [row for row in rows if row["metric_observed"]]
    visual_rows = [row for row in rows if row["status"] != "NEUTRAL_NO_MOTION"]
    state_path = output / "ARM_SESSION_BASE_STATES.npz"
    np.savez_compressed(state_path, q_arm=q, T_world_base=world_base,
                        T_tool_hand_root=mount, T_target_hand_root_world=target,
                        T_actual_hand_root_world=actual, valid_side_frame=visual,
                        metric_observed_side_frame=metric, source_frames=hawor["original_frame_indices"],
                        accepted_offset_world=offsets, anchor_frame_per_side=anchors,
                        per_frame_status=np.asarray([[next(row["status"] for row in rows
                                                          if row["frame"] == frame and row["side"] == SIDES[side])
                                                     for frame in range(frame_count)] for side in (0, 1)], dtype="<U24"))
    candidate_report = []
    for index, row in enumerate(candidates):
        candidate_report.append({"index": index, "x_y_z_yaw": row["spec"],
                                 "sample_rows": row["sample_rows"], "sample_passed": row["sample_passed"],
                                 "score": list(row["score"])})
    atomic_json(output / "BASE_SEARCH.json", {"sample_frames": sample.tolist(), "candidates": candidate_report,
                                              "selected_index": next(index for index, row in enumerate(candidates)
                                                                     if row is chosen),
                                              "selected_x_y_z_yaw": chosen["spec"],
                                              "accepted_offset_world": offsets.tolist()})
    failure_counts = Counter(row["status"] for row in visual_rows if not row["pass"])
    metric_pass = sum(row["pass"] for row in metric_rows)
    result = {
        "schema_version": "exact78-session-base-arm-v1", "status": "COMPLETE_PER_FRAME_STATUS",
        "session_id": args.session_id, "frame_count": frame_count,
        "inputs": {"hawor": artifact_ref(hawor_path), "hawor_result": artifact_ref(args.hawor_result),
                   "mount_proxy": mount_ref},
        "cold_start": {"cards042_fitted_products_read": False,
                       "foreign_q0_read": False, "foreign_base_read": False,
                       "foreign_offset_read": False, "neutral_seed_from_urdf_limits": True},
        "thresholds": {"position_mm_max": POSITION_GATE_MM, "rotation_deg_max": ROTATION_GATE_DEG,
                       "joint_margin_min": 0.0, "changed": False},
        "metrics": {"metric_observed_rows": len(metric_rows), "metric_passed_rows": metric_pass,
                    "metric_pass_fraction": float(metric_pass / len(metric_rows)) if metric_rows else None,
                    "visual_rows": len(visual_rows), "visual_passed_rows": sum(row["pass"] for row in visual_rows),
                    "failure_clusters": dict(failure_counts), "wall_seconds": time.monotonic() - started},
        "rows": rows,
        "outputs": {"states": artifact_ref(state_path), "base_search": artifact_ref(output / "BASE_SEARCH.json")},
        "visual_qualified": False, "training_eligible": False, "control_ground_truth": False,
        "claim_limit": "Per-session fixed-base offline visual IK with unchanged numeric gates; frame holds are diagnostics, not physical reachability or control authority.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "session": args.session_id,
                      "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
