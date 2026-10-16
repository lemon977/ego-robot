#!/usr/bin/env python3
"""Run Cards042 P1 causal bounded-IK correction canary."""
from __future__ import annotations

import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_action_canary_v1_retry10 as manifest_source
from chaoyang.governance import execute_exact78_general_p1_action_canary_v1 as base
from chaoyang.governance.common import REPO_ROOT


base.TASK = "exact78_general_p1_action_canary_v1_retry12_20260930"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.__file__ = str(Path(__file__).resolve())
os.environ["EXACT78_P1_MASTER_OPERATION"] = "run_exact78_general_p1_action_canary_v13"
os.environ["EXACT78_P1_QUALITY_MODIFICATION_ROUNDS"] = "1"
base.build_manifest = manifest_source.build_manifest


if __name__ == "__main__":
    raise SystemExit(base.main())
