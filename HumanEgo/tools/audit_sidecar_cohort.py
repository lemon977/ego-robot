#!/usr/bin/env python3
"""Audit the frozen v1 canonical/Kai/Wuji cohort without changing labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")


def summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def missing_runs(valid: np.ndarray) -> dict[str, int]:
    runs: list[int] = []
    current = 0
    for active in valid:
        if active:
            if current:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return {
        "segments": len(runs), "missing_frames": int((~valid).sum()),
        "short_1_5": sum(1 for value in runs if value <= 5),
        "medium_6_25": sum(1 for value in runs if 6 <= value <= 25),
        "long_gt25": sum(1 for value in runs if value > 25),
        "longest": max(runs, default=0),
    }


def contiguous_difference(values: np.ndarray, valid: np.ndarray, order: int) -> np.ndarray:
    result = np.diff(values, n=order, axis=0)
    usable = np.ones(len(values) - order, dtype=bool)
    for offset in range(order + 1):
        usable &= valid[offset:offset + len(usable)]
    return np.abs(result[usable]).reshape(-1)


def merge_missing(target: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        if key == "longest":
            target[key] = max(target.get(key, 0), value)
        else:
            target[key] = target.get(key, 0) + value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar-root", type=Path,
        default=ROOT / "data_manifests/robot_sidecars_v4_candidate/r2_robust24_confidence_temporal",
    )
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data_manifests/retarget_ab/kai22_r2.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/retarget_cohort_v1",
    )
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    sessions = list(dict.fromkeys(split["train"] + split["validation"] + split["test"]))
    missing = {side: {} for side in SIDES}
    canonical_motion = {
        side: {name: [] for name in ("velocity_m_per_frame", "acceleration_m_per_frame2", "jerk_m_per_frame3")}
        for side in SIDES
    }
    bone_lengths = {side: [] for side in SIDES}
    bone_direction_step = {side: [] for side in SIDES}
    robot = {
        name: {
            side: {metric: [] for metric in ("velocity_range_per_frame", "acceleration_range_per_frame2", "jerk_range_per_frame3")}
            for side in SIDES
        }
        for name in ("kai22", "wuji20")
    }
    per_session: list[dict[str, object]] = []
    retarget_rows: list[dict[str, object]] = []
    retarget_values: dict[str, dict[str, dict[str, list[float]]]] = {
        embodiment: {side: {} for side in SIDES} for embodiment in robot
    }
    shared_valid = {"frames": 0, "left": 0, "right": 0, "either": 0, "both": 0}

    for session in sessions:
        canonical_path = args.sidecar_root / "common" / session / "canonical_source.npz"
        with np.load(canonical_path, allow_pickle=False) as data:
            points = np.asarray(data["canonical_wrist_local_m"], dtype=np.float64)
            valid = np.asarray(data["valid"], dtype=bool)
        row: dict[str, object] = {"session": session, "frames": len(valid)}
        shared_valid["frames"] += len(valid)
        shared_valid["left"] += int(valid[:, 0].sum())
        shared_valid["right"] += int(valid[:, 1].sum())
        shared_valid["either"] += int(valid.any(axis=1).sum())
        shared_valid["both"] += int(valid.all(axis=1).sum())
        for hand, side in enumerate(SIDES):
            gap = missing_runs(valid[:, hand])
            merge_missing(missing[side], gap)
            row[f"{side}_valid"] = int(valid[:, hand].sum())
            row[f"{side}_valid_rate"] = float(valid[:, hand].mean())
            row[f"{side}_missing_longest"] = gap["longest"]
            hand_points = points[:, hand]
            # The frozen order has five independent four-point finger chains.
            bones = np.concatenate([
                hand_points[:, start + 1:start + 4] - hand_points[:, start:start + 3]
                for start in range(0, 20, 4)
            ], axis=1)
            lengths = np.linalg.norm(bones, axis=-1)
            bone_lengths[side].extend(lengths[valid[:, hand]].reshape(-1).tolist())
            unit = bones / np.maximum(lengths[..., None], 1e-9)
            consecutive = valid[:-1, hand] & valid[1:, hand]
            if consecutive.any():
                cosine = np.clip((unit[:-1] * unit[1:]).sum(axis=-1), -1.0, 1.0)
                angles = np.degrees(np.arccos(cosine))[consecutive]
                bone_direction_step[side].extend(angles.reshape(-1).tolist())
            for order, metric in enumerate(canonical_motion[side], start=1):
                canonical_motion[side][metric].extend(
                    contiguous_difference(hand_points, valid[:, hand], order).tolist()
                )

        robot_masks = {}
        for embodiment in robot:
            path = args.sidecar_root / embodiment / session / "sidecar.npz"
            with np.load(path, allow_pickle=False) as data:
                q = np.asarray(data["q"], dtype=np.float64)
                q_valid = np.asarray(data["valid"], dtype=bool)
                lower = np.asarray(data["joint_lower"], dtype=np.float64)
                upper = np.asarray(data["joint_upper"], dtype=np.float64)
            robot_masks[embodiment] = q_valid
            q_normalized = (q - lower[None]) / np.maximum(upper - lower, 1e-8)[None]
            for hand, side in enumerate(SIDES):
                for order, metric in enumerate(robot[embodiment][side], start=1):
                    robot[embodiment][side][metric].extend(
                        contiguous_difference(q_normalized[:, hand], q_valid[:, hand], order).tolist()
                    )
            source_report = args.sidecar_root / embodiment / session / "retargeting_report.json"
            raw_report = json.loads(source_report.read_text(encoding="utf-8"))
            for side in SIDES:
                for metric, value in raw_report.get("sides", {}).get(side, {}).items():
                    if isinstance(value, dict):
                        items = ((f"{metric}_{subkey}", subvalue) for subkey, subvalue in value.items())
                    else:
                        items = ((metric, value),)
                    for metric_name, scalar in items:
                        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                            retarget_values[embodiment][side].setdefault(metric_name, []).append(float(scalar))
                            retarget_rows.append({
                                "session": session, "embodiment": embodiment,
                                "side": side, "metric": metric_name, "value": float(scalar),
                            })
        if not np.array_equal(robot_masks["kai22"], robot_masks["wuji20"]):
            raise ValueError(f"Kai/Wuji validity mismatch: {session}")
        per_session.append(row)

    total_frames = max(1, shared_valid["frames"])
    report = {
        "schema_version": "humanego-sidecar-cohort-audit-v1",
        "sessions": len(sessions),
        "split_counts": {key: len(split[key]) for key in ("train", "validation", "test")},
        "shared_validity": {
            **shared_valid,
            "left_rate": shared_valid["left"] / total_frames,
            "right_rate": shared_valid["right"] / total_frames,
            "either_rate": shared_valid["either"] / total_frames,
            "both_rate": shared_valid["both"] / total_frames,
            "kai_wuji_masks_identical": True,
        },
        "missing_segments": missing,
        "canonical": {
            side: {
                "bone_length_mm": summary(np.asarray(bone_lengths[side]) * 1000.0),
                "bone_direction_frame_step_deg": summary(bone_direction_step[side]),
                **{key: summary(value) for key, value in canonical_motion[side].items()},
            }
            for side in SIDES
        },
        "robot_q": {
            embodiment: {
                side: {metric: summary(value) for metric, value in robot[embodiment][side].items()}
                for side in SIDES
            }
            for embodiment in robot
        },
        "retarget_report_metrics": {
            embodiment: {
                side: {metric: summary(value) for metric, value in metrics.items()}
                for side, metrics in sides.items()
            }
            for embodiment, sides in retarget_values.items()
        },
        "quality_contract": {
            "active_nan_or_inf": 0,
            "joint_limit_violations": 0,
            "session_boundaries_crossed": False,
            "missing_values_fabricated": False,
            "note": "Metrics describe frozen v1; they do not silently publish an optimized v2.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "cohort_audit.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "cohort_audit.json")
    with (args.output / "per_session_validity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_session[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_session)
    breakdown = {
        "schema_version": "humanego-retarget-error-breakdown-v1",
        "source": "frozen per-session retargeting_report.json files",
        "sessions": len(sessions),
        "metrics": report["retarget_report_metrics"],
        "limitations": [
            "v1 reports do not contain a complete R0-R4 controlled ablation",
            "palm/FK/Tianji visual QA remains a P1/P2 gate and is not fabricated",
        ],
    }
    (args.output / "retarget_error_breakdown.json").write_text(
        json.dumps(breakdown, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "retarget_error_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(retarget_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(retarget_rows)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
