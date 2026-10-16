#!/usr/bin/env python3
"""Terminalize the stopped P1 replay after a reproduced integration failure."""

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


TASK = "exact78_general_p1_cross_session_replay_v1_20260930"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1.json"
PRIOR_GUARD = (
    REPO_ROOT
    / "_run/current/exact78_baseline_route_consolidation_v2_20260930/attempts/attempt_0001/ROBOT100_GUARD.json"
)


def index_copy() -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    identity = process_identity(int(row["pid"]))
    if identity["alive"]:
        raise RuntimeError("WRITER_STILL_ALIVE")
    if RECEIPT.exists() or (ROOT / "RESULT.json").exists():
        raise RuntimeError("FRESH_TERMINALIZATION_REQUIRED")

    cards = ROOT / "sessions/play_cards_0902_042"
    arm_result = cards / "arm/RESULT.json"
    error_video = cards / "play_cards_0902_042_ERROR_DIAGNOSTIC.mp4"
    if not arm_result.is_file() or not error_video.is_file():
        raise RuntimeError("EXPECTED_FAILURE_EVIDENCE_MISSING")
    log = cards / "arm.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    if "operands could not be broadcast together with shapes (7,) (6,) (7,)" not in text:
        raise RuntimeError("FAILURE_SIGNATURE_DRIFT")

    previous_guard = load_json(PRIOR_GUARD)["files"]
    current_guard = {path: artifact_ref(Path(path)) for path in previous_guard}
    if previous_guard != current_guard:
        raise RuntimeError("ROBOT100_GUARD_CHANGED")
    partials = []
    for path in sorted((ROOT / "sessions").rglob("*")):
        if path.is_file():
            partials.append(artifact_ref(path))
    result = {
        "schema_version": "exact78-general-p1-cross-session-replay-result-v1",
        "task_id": TASK,
        "status": "FAILED_INTEGRATION_FINAL",
        "execution_complete": False,
        "visual_qualified": False,
        "quality_passed": False,
        "training_eligible": False,
        "failure": {
            "owner": "INTEGRATION",
            "operation": "run_exact78_session_base_arm_v8",
            "signature": "collapsed bound invokes legacy seven-dimensional broadcast solver",
            "exception": "ValueError: operands could not be broadcast together with shapes (7,) (6,) (7,)",
            "first_reproduced_session": "play_cards_0902_042",
            "source_arm_result": artifact_ref(arm_result),
            "error_video": artifact_ref(error_video),
            "log": artifact_ref(log),
        },
        "partial_artifacts": partials,
        "sessions_started": ["play_cards_0902_042", "get_potato_chips_0902_103"],
        "sessions_completed": 0,
        "stop_reason": "Known deterministic integration failure reproduced; remaining sessions stopped to avoid wasted compute.",
        "next_action": "New v9 successor must bind the already-tested collapsed-bound-safe solve_one without changing models or numeric thresholds.",
        "robot100_unchanged": True,
        "deletion_authorized": False,
        "claim_limit": "Failed integration attempt only; no visual, quality, training, control or deployment claim.",
    }
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(
        RECEIPT,
        {
            "schema_version": "exact78-general-p1-cross-session-replay-receipt-v1",
            "task_id": TASK,
            "status": "FAILED_RUNTIME_FINAL",
            "result": artifact_ref(ROOT / "RESULT.json"),
            "robot100_unchanged": True,
            "deletion_authorized": False,
            "claim_limit": result["claim_limit"],
        },
    )
    row.update(
        status="FAILED_RUNTIME_FINAL",
        pid=None,
        proc_start_ticks=None,
        updated_at=now_iso(),
        heartbeat_at=now_iso(),
        result=artifact_ref(ROOT / "RESULT.json"),
        terminal_receipt=artifact_ref(RECEIPT),
        quality="FAILED_INTEGRATION_FINAL",
        adoption="NOT_ADOPTED",
        review="ERROR_VIDEO_ONLY",
    )
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(
        packet_revision="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_FAILED_TERMINAL",
        plan_revision="EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        execution_revision="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1",
        status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_FAILED_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    validation = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    if validation.returncode:
        raise RuntimeError(validation.stdout[-4000:] + validation.stderr[-4000:])
    print(json.dumps({
        "status": "FAILED_RUNTIME_FINAL_RECORDED",
        "governance_revision": published["governance_revision"],
        "result": artifact_ref(ROOT / "RESULT.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
