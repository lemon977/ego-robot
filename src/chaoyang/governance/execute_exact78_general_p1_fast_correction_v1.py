#!/usr/bin/env python3
"""Register and run the bounded Cards042 fast P1 correction."""
from __future__ import annotations

import json
import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_action_canary_v1 as base
from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json


base.TASK = "exact78_general_p1_fast_correction_v1_20260930"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.SELECTED = (("play_cards_0902_042", "fast_causal_correction_from_terminal_retry11"),)
base.__file__ = str(Path(__file__).resolve())
os.environ["EXACT78_P1_MASTER_OPERATION"] = "run_exact78_general_p1_fast_correction_v1"
os.environ["EXACT78_P1_QUALITY_MODIFICATION_ROUNDS"] = "1"

SOURCE = (REPO_ROOT / "_run/current/exact78_general_p1_action_canary_v1_retry11_20260930"
          / "attempts/attempt_0001/sessions/play_cards_0902_042")
original_build_manifest = base.build_manifest


def build_manifest() -> Path:
    path = original_build_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    row = manifest["sessions"][0]
    row["fast_correction_inputs"] = {
        "hawor": artifact_ref(SOURCE / "hawor_recovery/HAWOR_RECOVERED_VISUAL_MANO21.npz"),
        "hawor_result": artifact_ref(SOURCE / "hawor_recovery/RESULT.json"),
        "arm_states": artifact_ref(SOURCE / "arm/ARM_SESSION_BASE_STATES.npz"),
        "arm_result": artifact_ref(SOURCE / "arm/RESULT.json"),
        "hand_states": artifact_ref(SOURCE / "hand/HAND_STATES.npz"),
        "terminal_parent": artifact_ref(SOURCE.parents[1] / "TERMINAL_RECEIPT.json"),
    }
    manifest["selection_policy"] = "cards042_fast_correction_from_terminal_retry11_only"
    manifest["reuse_policy"] = "SHA-pinned same-session terminal intermediates; causal Arm and diagnostics fresh"
    atomic_json(path, manifest)
    return path


base.build_manifest = build_manifest


if __name__ == "__main__":
    raise SystemExit(base.main())
