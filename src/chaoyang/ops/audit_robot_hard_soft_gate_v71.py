#!/usr/bin/env python3
"""Separate Robot hard feasibility from soft human-pose imitation residuals.

The legacy v5.2 terminal gate requires every observed arm/hand row to meet a
strict pose residual.  V7.1 defines joint/safety/temporal constraints as hard
and human-pose similarity as a soft target.  This audit does not supersede a
legacy C terminal or grant Robot authority; it identifies trajectories that
are safe enough for the next visual/z-buffer review despite soft residuals.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as robot  # noqa: E402
from chaoyang.ops import run_newtask_robot_shared_v4_hand as handfit  # noqa: E402
from chaoyang.ops.robot_target_reach_v75 import status_specific_artifacts  # noqa: E402


ARM_HARD_GATES = ("velocity", "acceleration", "missing_unknown_not_filled")
HAND_HARD_GATES = (
    "velocity",
    "acceleration",
    "missing_unknown_not_filled",
    "thumb_independent_q0_to_q5",
    "four_finger_chain_semantics",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def exact(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference["path"])).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256(path) != reference["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def classify_gates(
    arm_gates: dict[str, Any],
    hand_gates: dict[str, Any],
    collision_gates: dict[str, Any],
) -> dict[str, Any]:
    arm_hard = all(arm_gates.get(key) is True for key in ARM_HARD_GATES)
    hand_hard = all(hand_gates.get(key) is True for key in HAND_HARD_GATES)
    # ``real_collision_geometry_loaded`` is a legacy field name.  It only
    # means that the currently pinned URDF/mesh was loaded; it is not physical
    # ground-truth collision evidence.
    digital_collision_loaded = (
        collision_gates.get("digital_collision_geometry_loaded") is True
        or collision_gates.get("real_collision_geometry_loaded") is True
    )
    collision_hard = digital_collision_loaded and collision_gates.get(
        "non_adjacent_self_intersection_absent"
    ) is True
    strict_pose = (
        arm_gates.get("pose_branch_all_observed") is True
        and hand_gates.get("anatomy_all_observed") is True
    )
    hard = arm_hard and hand_hard and collision_hard
    if hard and strict_pose:
        status = "PASS_STRICT_POSE_AND_HARD_GEOMETRY_DEVELOPMENT"
    elif hard:
        status = "PASS_HARD_GEOMETRY_SOFT_POSE_REVIEW_REQUIRED"
    else:
        status = "FAILED_HARD_GEOMETRY"
    return {
        "status": status,
        "terminal_status": "PASSED" if hard else "FAILED_QUALITY_C",
        "robot_tier": "NONE",
        "candidate_robot_tier": "POSE_ONLY_VISUAL" if hard else "NONE",
        "hard_geometry_pass": hard,
        "strict_pose_match": strict_pose,
        "soft_pose_similarity_pass": strict_pose,
        "hard_gates": {
            "arm_temporal_and_unknown_contract": arm_hard,
            "hand_temporal_and_semantic_contract": hand_hard,
            "fullsession_collision_absent": collision_hard,
            "digital_collision_geometry_loaded": digital_collision_loaded,
        },
        "soft_gates": {
            "arm_pose_all_observed": arm_gates.get("pose_branch_all_observed") is True,
            "hand_anatomy_all_observed": hand_gates.get("anatomy_all_observed") is True,
        },
    }


def state_limits(arm_state: Path, hand_state: Path) -> dict[str, Any]:
    with np.load(arm_state, allow_pickle=False) as source:
        arm = {key: np.asarray(source[key]) for key in source.files}
    with np.load(hand_state, allow_pickle=False) as source:
        hand = {key: np.asarray(source[key]) for key in source.files}
    q_arm = np.asarray(arm["q_arm"], dtype=np.float64)
    q_hand = np.asarray(hand["q_hand"], dtype=np.float64)
    valid = np.asarray(arm["valid_side_frame"], dtype=bool)
    if q_arm.shape[1:] != (2, 7) or q_hand.shape != (len(q_arm), 2, 22):
        raise RuntimeError("unexpected arm/hand state shape")
    if not np.array_equal(valid, np.asarray(hand["valid_side_frame"], dtype=bool)):
        raise RuntimeError("arm/hand validity mismatch")
    assets = robot.load_pinned_robot_assets(Path(__file__).resolve().parents[3])
    arm_lower, arm_upper = robot.official._arm_limits(assets)
    contracts = handfit.model_contract(robot.official, __import__(
        "chaoyang.ops.render_poker_same_side_outward_frame0", fromlist=["wrist_adapter"]
    ).wrist_adapter, assets)
    finite = True
    limits = True
    for side in range(2):
        rows = np.flatnonzero(valid[side])
        finite &= bool(np.isfinite(q_arm[rows, side]).all() and np.isfinite(q_hand[rows, side]).all())
        limits &= bool(
            np.all(q_arm[rows, side] >= arm_lower[side] - 1e-9)
            and np.all(q_arm[rows, side] <= arm_upper[side] + 1e-9)
            and np.all(q_hand[rows, side] >= contracts[side]["lower"] - 1e-9)
            and np.all(q_hand[rows, side] <= contracts[side]["upper"] + 1e-9)
        )
    return {"finite_valid_states": finite, "urdf_joint_limits": limits}


def soft_failure_summary(hand_result: dict[str, Any], hand_state: Path) -> dict[str, Any]:
    with np.load(hand_state, allow_pickle=False) as source:
        q = np.asarray(source["q_hand"], dtype=np.float64)
    assets = robot.load_pinned_robot_assets(Path(__file__).resolve().parents[3])
    shared = __import__("chaoyang.ops.render_poker_same_side_outward_frame0", fromlist=["wrist_adapter"])
    contracts = handfit.model_contract(robot.official, shared.wrist_adapter, assets)
    side_index = {"left": 0, "right": 1}
    by_finger: dict[str, dict[str, int]] = {}
    total_failed_rows = 0
    saturated_failed_rows = 0
    for row in hand_result.get("rows", []):
        if row.get("pass") is not False or "fingers" not in row:
            continue
        total_failed_rows += 1
        row_saturated = False
        side = side_index[str(row["side"])]
        frame = int(row["frame"])
        for item in row["fingers"]:
            finger = str(item["finger"])
            if float(item["bone_error_deg_max"]) <= 60.0 and float(item["tip_direction_error_deg"]) <= 15.0:
                continue
            entry = by_finger.setdefault(f"{row['side']}:{finger}", {"failed_rows": 0, "rows_at_joint_limit": 0})
            entry["failed_rows"] += 1
            group = handfit.GROUPS[finger]
            values = q[frame, side, group]
            at_limit = bool(
                np.any(np.isclose(values, contracts[side]["lower"][group], atol=1e-5))
                or np.any(np.isclose(values, contracts[side]["upper"][group], atol=1e-5))
            )
            entry["rows_at_joint_limit"] += int(at_limit)
            row_saturated |= at_limit
        saturated_failed_rows += int(row_saturated)
    return {
        "failed_rows": total_failed_rows,
        "failed_rows_with_joint_limit_saturation": saturated_failed_rows,
        "by_side_finger": by_finger,
        "interpretation_limit": "Joint-limit saturation supports a morphology/workspace clipping diagnosis; it does not prove the human target or Robot pose is physically correct.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--collision-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch_root.resolve(strict=True)
    arm_path = batch / "arm_round2/RESULT.json"
    hand_path = batch / "hand_round2/RESULT.json"
    render_path = batch / "render/RESULT.json"
    arm = load(arm_path)
    hand = load(hand_path)
    render = load(render_path)
    arm_rows = {row["session"]: row for row in arm["sessions"]}
    hand_rows = {row["session"]: row for row in hand["sessions"]}
    review_rows = {row["session"]: row for row in render["sessions"]}
    if not (set(arm_rows) == set(hand_rows) == set(review_rows)):
        raise RuntimeError("arm/hand/render session sets differ")
    rows = []
    for session in sorted(arm_rows):
        arm_row = arm_rows[session]
        hand_row = hand_rows[session]
        collision_path = args.collision_root.resolve(strict=True) / session / "collision_full/RESULT.json"
        collision = load(collision_path)
        arm_result_ref, arm_states_ref = status_specific_artifacts(arm_row, kind="arm", session=session)
        hand_result_ref, hand_states_ref = status_specific_artifacts(hand_row, kind="hand", session=session)
        arm_detail_path = exact(arm_result_ref, f"{session}.arm_result")
        arm_state = exact(arm_states_ref, f"{session}.arm_states")
        hand_state = exact(hand_states_ref, f"{session}.hand_states")
        hand_detail_path = exact(hand_result_ref, f"{session}.hand_result")
        arm_detail = load(arm_detail_path)
        hand_detail = load(hand_detail_path)
        arm_gates = arm_row.get("gates") or arm_detail.get("gates")
        hand_gates = hand_row.get("gates") or hand_detail.get("gates")
        if not isinstance(arm_gates, dict) or not isinstance(hand_gates, dict):
            raise RuntimeError(f"{session}: final arm/hand gate dictionaries are required")
        classification = classify_gates(arm_gates, hand_gates, collision["gates"])
        state_gate = state_limits(arm_state, hand_state)
        classification["hard_gates"].update(state_gate)
        if not all(classification["hard_gates"].values()):
            classification["hard_geometry_pass"] = False
            classification["status"] = "FAILED_HARD_GEOMETRY"
        rows.append({
            "session": session,
            "task": arm_row["task"],
            **classification,
            "arm_soft_metrics": arm_row.get("metrics") or arm_detail.get("metrics"),
            "hand_soft_metrics": hand_row.get("metrics") or hand_detail.get("metrics"),
            "hand_soft_failure_diagnosis": soft_failure_summary(hand_detail, hand_state),
            "evidence": {
                "arm_result": arm_result_ref,
                "hand_result": hand_result_ref,
                "collision_full": ref(collision_path),
                "review_result": review_rows[session]["result"],
            },
        })
    output = args.output.resolve()
    atomic_new(output, {
        "schema_version": "robot-hard-vs-soft-gate-audit-v71-v1",
        "artifact_revision": "R7_3_DIAGNOSTIC",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASSED_DIAGNOSTIC_AUDIT",
        "counts": {
            "sessions": len(rows),
            "hard_geometry_pass": sum(row["hard_geometry_pass"] for row in rows),
            "strict_pose_match": sum(row["strict_pose_match"] for row in rows),
        },
        "inputs": {"arm": ref(arm_path), "hand": ref(hand_path), "render": ref(render_path)},
        "rows": rows,
        "decision": "Do not use strict all-frame human-pose residual as a hard feasibility gate. Preserve it as soft diagnostic and require unified z-buffer plus visual review before any visual authority.",
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Digital URDF hard-feasibility and soft-retarget diagnosis only; no Robot/contact/control/physical authority.",
    })
    print(json.dumps(load(output), ensure_ascii=False)[0:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
