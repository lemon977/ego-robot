#!/usr/bin/env python3
"""Finite P1 rerun after correcting arm postprocessor helper binding."""
from __future__ import annotations

import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_action_canary_v1 as base
from chaoyang.governance.common import REPO_ROOT


base.TASK = "exact78_general_p1_action_canary_v1_retry8_20260930"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.SELECTED = (
    ("play_cards_0902_042", "ab_regression_cards042"),
    ("get_potato_chips_0902_103", "ab_regression_chips103"),
    ("get_potato_chips_0901_013", "native_25fps_longest_chips"),
)
base.__file__ = str(Path(__file__).resolve())
os.environ["EXACT78_P1_MASTER_OPERATION"] = "run_exact78_general_p1_action_canary_v9"
os.environ["EXACT78_P1_QUALITY_MODIFICATION_ROUNDS"] = "2"


if __name__ == "__main__":
    raise SystemExit(base.main())
