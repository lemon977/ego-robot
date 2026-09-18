#!/usr/bin/env python3
"""Corrective executor after the V2 sidecar-schema-only stop."""

from __future__ import annotations

from pathlib import Path

from chaoyang.ops import run_0915_interaction_contact_robot_dev_v1 as implementation
from chaoyang.ops.run_0915_interaction_contact_robot_dev_v2 import load_frozen_inputs_v2


TASK_ID = "0915_interaction_contact_robot_dev_v3"
ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    implementation.TASK_ID = TASK_ID
    implementation.PHASE = "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V3"
    implementation.OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
    implementation.TERMINAL_RECEIPT = (
        ROOT / "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V3_RESULT.json"
    )
    implementation.load_frozen_inputs = load_frozen_inputs_v2
    implementation.__file__ = __file__
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
