#!/usr/bin/env python3
"""Run the Cards042 P1 visual-contract correction canary."""
from __future__ import annotations

import json
import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_action_canary_v1 as base
from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json


base.TASK = "exact78_general_p1_action_canary_v1_retry10_20260930"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.SELECTED = (("play_cards_0902_042", "mount_flange_palm_jump_correction"),)
base.__file__ = str(Path(__file__).resolve())
os.environ["EXACT78_P1_MASTER_OPERATION"] = "run_exact78_general_p1_action_canary_v11"
os.environ["EXACT78_P1_QUALITY_MODIFICATION_ROUNDS"] = "1"

RETRY8 = (REPO_ROOT / "_run/current/exact78_general_p1_action_canary_v1_retry8_20260930"
          / "attempts/attempt_0001")
original_build_manifest = base.build_manifest


def build_manifest() -> Path:
    path = original_build_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    row = manifest["sessions"][0]
    source = RETRY8 / "sessions" / row["session_id"]
    row["reuse_hawor"] = {
        "source_task": artifact_ref(RETRY8 / "RESULT.json"),
        "hawor_l0": {
            "HAWOR_RAW_MANO21.npz": artifact_ref(source / "hawor_l0/HAWOR_RAW_MANO21.npz"),
            "RESULT.json": artifact_ref(source / "hawor_l0/RESULT.json"),
        },
        "hawor_recovery": {
            "HAWOR_RECOVERED_VISUAL_MANO21.npz": artifact_ref(
                source / "hawor_recovery/HAWOR_RECOVERED_VISUAL_MANO21.npz"),
            "RESULT.json": artifact_ref(source / "hawor_recovery/RESULT.json"),
        },
    }
    manifest["selection_policy"] = "cards042_visual_contract_correction_only"
    manifest["reuse_policy"] = (
        "SHA-identical same-session retry8 HaWoR only; base/Arm/Hand/diagnostics fresh"
    )
    atomic_json(path, manifest)
    return path


base.build_manifest = build_manifest


if __name__ == "__main__":
    raise SystemExit(base.main())
