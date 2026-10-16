#!/usr/bin/env python3
"""Retry the bounded three-session replay with collapsed-bound-safe integration."""

from __future__ import annotations

import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_cross_session_replay_v1 as base
from chaoyang.governance.common import REPO_ROOT


base.TASK = "exact78_general_p1_cross_session_replay_v1_retry1_20260930"
base.EXECUTION_REVISION = "EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1.json"
base.REVIEW = REPO_ROOT / "review/exact78_p1_cross_session_v1_retry1"
os.environ["EXACT78_P1_CROSS_SESSION_OPERATION"] = "run_exact78_general_p1_cross_session_replay_v2"
os.environ["EXACT78_P1_CROSS_SESSION_ARM_OPERATION"] = "run_exact78_session_base_arm_v9"
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    raise SystemExit(base.main())
