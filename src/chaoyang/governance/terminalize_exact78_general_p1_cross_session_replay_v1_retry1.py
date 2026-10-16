#!/usr/bin/env python3
"""Close the stopped P1 diagnostic replay without promoting its visual output."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
)


TASK = "exact78_general_p1_cross_session_replay_v1_retry1_20260930"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1.json"


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    if process_identity(int(row["pid"]))["alive"]:
        raise RuntimeError("WRITER_STILL_ALIVE")
    if RECEIPT.exists() or (ROOT / "RESULT.json").exists():
        raise RuntimeError("TERMINAL_RESULT_ALREADY_EXISTS")

    cards = ROOT / "sessions/play_cards_0902_042"
    diagnostics = load_json(cards / "diagnostics/RESULT.json")
    progress = load_json(ROOT / "PROGRESS.json")
    if len(progress["sessions"]) != 1 or progress["sessions"][0]["session_id"] != "play_cards_0902_042":
        raise RuntimeError("PARTIAL_PROGRESS_DRIFT")
    if diagnostics["metrics"]["frame_count"] != 171:
        raise RuntimeError("CARDS_FRAME_COUNT_DRIFT")
    overlay = cards / "diagnostics/play_cards_0902_042_RAW_ROBOT_OVERLAY.mp4"
    whole = cards / "diagnostics/play_cards_0902_042_INDEPENDENT_WHOLE_ROBOT.mp4"
    if not overlay.is_file() or not whole.is_file():
        raise RuntimeError("DIAGNOSTIC_VIDEO_MISSING")

    result = {
        "schema_version": "exact78-general-p1-cross-session-replay-result-v1",
        "task_id": TASK,
        "status": "CANCELLED_AFTER_VISUAL_REVIEW",
        "stop_reason": "User requested no further modifications after P1 diagnostic showed a dominant proximal arm and poorly visible robot hands.",
        "execution_complete": False,
        "visual_qualified": False,
        "quality_passed": False,
        "training_eligible": False,
        "sessions_requested": 3,
        "sessions_started": ["play_cards_0902_042", "get_potato_chips_0902_103"],
        "sessions_with_complete_diagnostics": 1,
        "sessions_visual_qualified": 0,
        "sessions_not_started": ["get_potato_chips_0901_013"],
        "cards042_metrics": {
            "visible_without_visual_action_side_frames": 0,
            "visible_side_frames": 342,
            "observed_arm_rows": 295,
            "static_ik_pass_rows": 277,
            "static_ik_pass_fraction": 0.9389830508474576,
            "smoothed_visual_pass_rows": 248,
            "smoothed_visual_pass_fraction": 0.8406779661016949,
            "decoded_diagnostic_frames": 171,
            "visual_qualified_frame_fraction": None,
            "visual_qualified_frame_fraction_reason": "Raw-overlay diagnostic is not a Clean+Robot composite and has no full per-frame visual gate.",
        },
        "visual_problem": {
            "session": "play_cards_0902_042",
            "inspected_frames": [139, 155],
            "observations": [
                "Raw human hands intentionally remain in the diagnostic overlay.",
                "A large proximal robot arm occupies the first-person center and obscures most of the robot hand.",
                "Frame-wise robot-pixel presence and arm reach do not establish acceptable hand visibility or composition.",
            ],
            "owner": "ARM_BASE_OR_RASTER_COMPOSITION_UNRESOLVED",
            "raw_robot_overlay": artifact_ref(overlay),
            "independent_whole_robot": artifact_ref(whole),
            "frame_manifest": artifact_ref(cards / "diagnostics/FRAME_MANIFEST.json"),
        },
        "partial_chip103_arm_log": artifact_ref(ROOT / "sessions/get_potato_chips_0902_103/arm.log"),
        "parent_v2_robot_result": artifact_ref(REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"),
        "adoption": "NOT_ADOPTED",
        "next_action": "Freeze V2 as Cards042 visual parent; do not start another solver/replay until a bounded V2-vs-P1 visual contract and review scope are agreed.",
        "deletion_authorized": False,
        "claim_limit": "Stopped P1 diagnostic only; not a general Raw-to-Robot baseline, visual result, training label, or control trajectory.",
    }
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(RECEIPT, {
        "schema_version": "exact78-general-p1-cross-session-replay-receipt-v1",
        "task_id": TASK,
        "status": "CANCELLED",
        "result": artifact_ref(ROOT / "RESULT.json"),
        "deletion_authorized": False,
        "claim_limit": result["claim_limit"],
    })

    row.update(
        status="CANCELLED", pid=None, proc_start_ticks=None,
        updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(ROOT / "RESULT.json"),
        terminal_receipt=artifact_ref(RECEIPT),
        quality="FAILED_VISUAL_REVIEW", adoption="NOT_ADOPTED", review="FRAMES_139_155_REJECTED",
    )
    state["next_task"] = None
    index = {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(
        packet_revision="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1_CANCELLED",
        plan_revision="EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        execution_revision="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1",
        status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_RETRY1_CANCELLED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    validation = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    if validation.returncode:
        raise RuntimeError(validation.stdout[-4000:] + validation.stderr[-4000:])
    print(json.dumps({"status": "CANCELLED_RECORDED", "governance_revision": published["governance_revision"], "result": artifact_ref(ROOT / "RESULT.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
