#!/usr/bin/env python3
"""Add the per-side first-observed arm runner to a passed Robot preflight.

This is a no-IK code-closure successor.  It preserves every frozen session,
matrix, Clean, HaWoR, template, placement, and method-budget reference from the
passed v5.2 preflight and adds one versioned forward-arm implementation.  It is
used only when frame 0 is missing one side; it does not fill missing data or
consume a method round.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
ARM_V2 = PROJECT / "tools/run_robot_motion_transfer_arm_canary_v2.py"
ARM_V3 = PROJECT / "tools/run_robot_motion_transfer_arm_canary_v3.py"
ARM_ROUND2_WRAPPER = PROJECT / "tools/run_exact78_robot_arm_bidirectional_round2_v52.py"
ARM_ROUND2_TOOL = PROJECT / "tools/run_robot_arm_segment_bidirectional_v3.py"
HAND_ROUND1_WRAPPER = PROJECT / "tools/run_exact78_robot_hand_round1_v52.py"
HAND_ROUND2_WRAPPER = PROJECT / "tools/run_exact78_robot_hand_bidirectional_round2_v52.py"
HAND_ROUND1_TOOL = PROJECT / "tools/run_robot_hand_fullsession_v2.py"
HAND_ROUND2_TOOL = PROJECT / "tools/run_robot_hand_segment_bidirectional_v3.py"
HAND_FIT_IMPLEMENTATION = PROJECT / "tools/run_newtask_robot_shared_v4_hand.py"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.base_preflight)
    if base.get("status") != "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT":
        raise SystemExit("passed base Robot preflight required")
    if base.get("programs", {}).get(ARM_V2.name) != ref(ARM_V2):
        raise SystemExit("base preflight no longer matches the v2 predecessor")
    sessions = base.get("sessions", [])
    if not sessions or len({row.get("session") for row in sessions}) != len(sessions):
        raise SystemExit("non-empty unique base session set required")

    result = copy.deepcopy(base)
    result.update(
        schema_version="exact78-robot-ready-anchor-v3-preflight-v1",
        created_at=now(),
        status="PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT",
        predecessor_preflight=ref(args.base_preflight),
        preflight_program=ref(Path(__file__)),
        arm_forward_tool=ref(ARM_V3),
        arm_anchor_policy={
            "name": "FIRST_OBSERVED_PER_SIDE",
            "missing_before_anchor": "UNKNOWN_NAN",
            "missing_gap_policy": "NO_FILL_NO_HOLD_NO_TEMPORAL_CROSSING",
            "world_base": "FIXED_FULL_SESSION",
            "motion_gain": 1.0,
            "numeric_gates": "UNCHANGED_FROM_V2",
            "method_rounds_consumed": 0,
        },
        gpu_calls=0,
        authority=False,
        claim_limit=(
            "Code-closure successor only: per-side first-observed anchoring is available without filling "
            "missing frames or changing gates. No IK/render was run, no method round was consumed, and no "
            "Robot/contact/action/deployment authority is published."
        ),
    )
    result["programs"][ARM_V3.name] = ref(ARM_V3)
    # The base preflight predates the shared KaiHand implementation module.
    # Refresh the direct runners and pin their imported implementation so a
    # runtime hotfix cannot silently execute under an older code closure.
    for program in (
        ARM_ROUND2_WRAPPER,
        ARM_ROUND2_TOOL,
        HAND_ROUND1_WRAPPER,
        HAND_ROUND2_WRAPPER,
        HAND_ROUND1_TOOL,
        HAND_ROUND2_TOOL,
        HAND_FIT_IMPLEMENTATION,
    ):
        result["programs"][program.name] = ref(program)
    atomic_new(args.output, result)
    print(json.dumps({"status": result["status"], "output": ref(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
