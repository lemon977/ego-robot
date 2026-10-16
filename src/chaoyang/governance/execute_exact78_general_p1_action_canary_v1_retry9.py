#!/usr/bin/env python3
"""Run the five remaining frozen P1 canaries with the corrected arm solver."""
from __future__ import annotations

import json
import os
from pathlib import Path

from chaoyang.governance import execute_exact78_general_p1_action_canary_v1 as base
from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json


base.TASK = "exact78_general_p1_action_canary_v1_retry9_20260930"
base.ROOT = REPO_ROOT / "_run/current" / base.TASK / "attempts/attempt_0001"
base.PACKET = REPO_ROOT / "tasks/current" / base.TASK / "TASK_PACKET.json"
base.SELECTED = (
    ("get_potato_chips_0901_010", "historical_role_grade_c_native_25fps"),
    ("get_potato_chips_0903_007", "longest_historical_two_hand_bbox_overlap_chips"),
    ("play_cards_0901_002", "historical_hawor_role_grade_c"),
    ("play_cards_0901_003", "longest_poker"),
    ("play_cards_0901_016", "longest_historical_two_hand_bbox_overlap_poker"),
)
base.__file__ = str(Path(__file__).resolve())
os.environ["EXACT78_P1_MASTER_OPERATION"] = "run_exact78_general_p1_action_canary_v10"
os.environ["EXACT78_P1_QUALITY_MODIFICATION_ROUNDS"] = "2"

RETRY6 = (REPO_ROOT / "_run/current/exact78_general_p1_action_canary_v1_retry6_20260930"
          / "attempts/attempt_0001")
FRESH_ONLY = {"get_potato_chips_0901_010"}
original_build_manifest = base.build_manifest


def build_manifest() -> Path:
    path = original_build_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest["sessions"]:
        session_id = row["session_id"]
        if session_id in FRESH_ONLY:
            row["reuse_hawor"] = None
            continue
        source = RETRY6 / "sessions" / session_id
        row["reuse_hawor"] = {
            "source_task": artifact_ref(RETRY6 / "RESULT.json"),
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
    manifest["selection_policy"] = "remaining_five_of_frozen_eight_after_retry8_correction_validation"
    manifest["reuse_policy"] = (
        "SHA-identical same-session retry6 HaWoR for four 30-fps sessions; "
        "0901_010 runs fresh native-fps HaWoR v2; all downstream stages run fresh"
    )
    atomic_json(path, manifest)
    return path


base.build_manifest = build_manifest


if __name__ == "__main__":
    raise SystemExit(base.main())
