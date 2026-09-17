#!/usr/bin/env python3
"""ROBOT-TARGET-10: prove world target construction without granting authority."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import (  # noqa: E402
    SIDES,
    artifact_ref,
    atomic_new_json,
    load_json,
    now_iso,
    proper_se3_mask,
    side_target_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--arm-result", type=Path, required=True)
    parser.add_argument("--arm-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detail = load_json(args.arm_result.resolve(strict=True))
    if detail.get("session") != args.session or detail.get("task") != args.task:
        raise SystemExit("arm result task/session identity mismatch")
    with np.load(args.hawor.resolve(strict=True), allow_pickle=False) as source:
        human = np.asarray(source["joints_3d_world"], dtype=np.float64)
        source_frames = np.asarray(source["original_frame_indices"], dtype=np.int64)
    with np.load(args.arm_states.resolve(strict=True), allow_pickle=False) as source:
        target = np.asarray(source["T_target_hand_root_world"], dtype=np.float64)
        actual = np.asarray(source["T_actual_hand_root_world"], dtype=np.float64)
        valid = np.asarray(source["valid_side_frame"], dtype=bool).T
        state_frames = np.asarray(source["source_frames"], dtype=np.int64)
        world_base = np.asarray(source["T_world_base"], dtype=np.float64)
    if human.shape[:3] != (2, len(target), 21) or target.shape != actual.shape:
        raise SystemExit("HaWoR/Robot frame or side shape mismatch")
    if not np.array_equal(source_frames, state_frames):
        raise SystemExit("HaWoR/Robot source frame identity mismatch")
    target_proper = proper_se3_mask(target, valid)
    actual_proper = proper_se3_mask(actual, valid)
    base_proper = bool(
        np.isfinite(world_base).all()
        and np.max(np.abs(world_base[:3, :3].T @ world_base[:3, :3] - np.eye(3))) <= 1e-4
        and abs(float(np.linalg.det(world_base[:3, :3])) - 1.0) <= 1e-4
        and np.max(np.abs(world_base[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) <= 1e-8
    )
    rows = []
    for side, name in enumerate(SIDES):
        metrics = side_target_metrics(human[side, :, 0], target[:, side], actual[:, side], valid[:, side])
        delta_p95 = metrics.get("human_to_target_delta_error_mm_p95")
        rows.append(
            {
                "side": name,
                "metrics": metrics,
                "target_preserves_human_translation": delta_p95 is not None and delta_p95 <= 1.0,
            }
        )
    gates = {
        "frame_identity": True,
        "fixed_world_base_proper_se3": base_proper,
        "all_observed_target_transforms_proper_se3": bool(np.all(target_proper[valid])),
        "all_observed_actual_transforms_proper_se3": bool(np.all(actual_proper[valid])),
        "human_translation_preserved_at_gain_one": all(row["target_preserves_human_translation"] for row in rows),
    }
    payload = {
        "schema_version": "robot-target-10-audit-v1",
        "artifact_revision": "R7_ROBOT_5_TARGET_10",
        "created_at": now_iso(),
        "task_id": "ROBOT-TARGET-10",
        "task": args.task,
        "session": args.session,
        "evidence_class": "DIAGNOSTIC_PROXY",
        "object6d_consumed": False,
        "workspace_bounds_consumed": False,
        "terminal_status": "PASSED" if all(gates.values()) else "FAILED_QUALITY_C",
        "robot_tier": "NONE",
        "hard_target_contract_pass": all(gates.values()),
        "soft_pose_similarity_pass": bool(detail.get("gates", {}).get("pose_branch_all_observed", False)),
        "gates": gates,
        "sides": rows,
        "inputs": {
            "hawor": artifact_ref(args.hawor),
            "arm_result": artifact_ref(args.arm_result),
            "arm_states": artifact_ref(args.arm_states),
        },
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "DIAGNOSTIC_PROXY for digital target-chain translation only. Object6D and actual "
            "Robot workspace bounds are not consumed; no clipping, collision, contact, control, "
            "or physical authority."
        ),
    }
    atomic_new_json(args.output.resolve(), payload)
    return 0 if payload["terminal_status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
