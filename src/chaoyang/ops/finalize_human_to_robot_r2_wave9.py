#!/usr/bin/env python3
"""Append a side-resolved correction to the immutable Wave-3 R1 funnel."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, TASK, ref, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    prior_path = ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave3/RESULT.json"
    prior = load(prior_path)
    rows = []
    blocker_override = {
        "play_cards_0915_031": "BLOCKED_STRICT_METRIC_CONTACT_AUTHORITY",
        "get_potato_chips_0915_007": "BLOCKED_OBJECT6D_VISIBLE_PATCH_ABSENT",
        "get_potato_chips_0902_103": "BLOCKED_DEPTH_REGISTRATION_ABSENT",
        "play_cards_0902_042": "BLOCKED_DEPTH_REGISTRATION_ABSENT",
    }
    for row in prior["sessions"]:
        session = row["session_id"]
        sides = row["stages"]["timeline_hand_valid"]["side_frames"]
        valid_sides = [name for name, count in zip(("left", "right"), sides, strict=True) if count]
        rows.append({
            "session_id": session,
            "valid_r0_sides": valid_sides,
            "bilateral_quality_limitation": (
                "LEFT_ROLE_INCONCLUSIVE_DOES_NOT_BLOCK_RIGHT_R0"
                if session == "play_cards_0915_031" else None
            ),
            "corrected_first_r1_blocker": blocker_override[session],
            "eligible_metric_contact_windows": 0,
            "r1_windows_executed": 0,
            "r1_windows_adopted": 0,
            "unaffected": ["valid-side R0", "per-session candidate rendering", "Scene diagnostics"],
        })
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_CONTACT_R1_FUNNEL_ADDENDUM_V1",
        "task_id": TASK,
        "created_at": now(),
        "supersedes_interpretation_not_bytes": ref(prior_path),
        "sessions": rows,
        "r1_windows_screened_for_metric_evidence": 0,
        "r1_windows_executed": 0,
        "r1_windows_adopted": 0,
        "correction": "031 missing left evidence limits bilateral quality only; it does not erase the 102-frame right-side R0. Strict metric Contact authority remains the first R1 blocker.",
        "claim_limit": "No Contact or Robot R1 was manufactured; this addendum narrows blocker scope only.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    out = ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json"
    write_json(out, result)
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE9_CONTACT_FUNNEL_SCOPE_CORRECTION",
        "observed_at": now(),
        "contact_r1_addendum": ref(out),
        "r1_windows_executed": 0,
        "r1_windows_adopted": 0,
        "main_product_candidates": 2,
        "regression_product_candidates": 2,
        "product_quality_pass": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE9.json", progress)
    state_path = ATTEMPT / "lanes/lane1_scene/STATE.json"
    state = load(state_path)
    artifacts = state.get("latest_artifacts", [])
    item = ref(out)
    if not any(x.get("path") == item["path"] for x in artifacts if isinstance(x, dict)):
        artifacts.append(item)
    state.update(
        current_action="Four Scene candidates executed and rejected; synthetic occlusion contract passed, real occlusion remains UNKNOWN, and no metric Contact/R1 window is authorized.",
        latest_artifacts=artifacts,
        writer={"pid": os.getpid(), "proc_start_ticks": None},
        updated_at=now(),
    )
    write_json(state_path, state)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
