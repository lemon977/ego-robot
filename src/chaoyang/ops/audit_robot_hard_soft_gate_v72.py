#!/usr/bin/env python3
"""Schema-compatible successor for the v7.1 Robot hard/soft audit.

The v7.1 audit expected generic arm gate names while the pinned v7.7 arm
solver emits task-specific aliases.  This successor changes only gate-name
resolution; thresholds, collision checks, state-limit checks and all evidence
loading remain delegated to v7.1.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


from typing import Any

from chaoyang.ops import audit_robot_hard_soft_gate_v71 as v71

_V71_CLASSIFY_GATES = v71.classify_gates

ARM_GATE_ALIASES = {
    "velocity": ("velocity", "arm_velocity"),
    "acceleration": ("acceleration", "arm_acceleration"),
    "missing_unknown_not_filled": (
        "missing_unknown_not_filled",
        "missing_frames_unknown_not_filled",
    ),
}


def _resolved_true(gates: dict[str, Any], aliases: tuple[str, ...]) -> bool:
    present = [gates[name] for name in aliases if name in gates]
    if not present:
        return False
    # Fail closed if two aliases disagree instead of selecting the convenient
    # spelling.
    return all(value is True for value in present)


def classify_gates(
    arm_gates: dict[str, Any],
    hand_gates: dict[str, Any],
    collision_gates: dict[str, Any],
) -> dict[str, Any]:
    result = _V71_CLASSIFY_GATES(arm_gates, hand_gates, collision_gates)
    arm_hard = all(
        _resolved_true(arm_gates, aliases)
        for aliases in ARM_GATE_ALIASES.values()
    )
    hand_hard = all(hand_gates.get(key) is True for key in v71.HAND_HARD_GATES)
    digital_collision_loaded = (
        collision_gates.get("digital_collision_geometry_loaded") is True
        or collision_gates.get("real_collision_geometry_loaded") is True
    )
    collision_hard = digital_collision_loaded and collision_gates.get(
        "non_adjacent_self_intersection_absent"
    ) is True
    hard = arm_hard and hand_hard and collision_hard
    strict_pose = (
        arm_gates.get("pose_branch_all_observed") is True
        and hand_gates.get("anatomy_all_observed") is True
    )
    result["status"] = (
        "PASS_STRICT_POSE_AND_HARD_GEOMETRY_DEVELOPMENT"
        if hard and strict_pose
        else "PASS_HARD_GEOMETRY_SOFT_POSE_REVIEW_REQUIRED"
        if hard
        else "FAILED_HARD_GEOMETRY"
    )
    result["terminal_status"] = "PASSED" if hard else "FAILED_QUALITY_C"
    result["hard_geometry_pass"] = hard
    result["strict_pose_match"] = strict_pose
    result["soft_pose_similarity_pass"] = strict_pose
    result["candidate_robot_tier"] = "POSE_ONLY_VISUAL" if hard else "NONE"
    result["hard_gates"]["arm_temporal_and_unknown_contract"] = arm_hard
    result["hard_gates"]["hand_temporal_and_semantic_contract"] = hand_hard
    result["hard_gates"]["fullsession_collision_absent"] = collision_hard
    result["hard_gates"]["digital_collision_geometry_loaded"] = digital_collision_loaded
    result["gate_schema_resolution"] = {
        "arm_aliases": {name: list(aliases) for name, aliases in ARM_GATE_ALIASES.items()},
        "policy": "ALL_PRESENT_ALIASES_MUST_BE_TRUE_MISSING_FAILS_CLOSED",
    }
    return result


def main() -> int:
    v71.classify_gates = classify_gates
    return v71.main()


if __name__ == "__main__":
    raise SystemExit(main())
