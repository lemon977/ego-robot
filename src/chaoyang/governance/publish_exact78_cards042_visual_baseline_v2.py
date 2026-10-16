#!/usr/bin/env python3
"""Publish the sole reproducible Exact78 Cards042 visual baseline V2."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

import cv2

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_cards042_visual_baseline_v2_publish_20260929"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"
POINTER = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"
LOCK = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json"
HANDOFF = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_ZH.md"
ROBOT_RESULT = (REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929"
                / "attempts/attempt_0001/robot/RESULT.json")

EXPECTED = {
    ROBOT_RESULT: "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1",
}


def index_body() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def decoded_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.size == 0:
                raise RuntimeError(f"EMPTY_DECODED_FRAME:{path}:{count}")
            count += 1
    finally:
        capture.release()
    return count


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), index_body()
    # A failed pre-commit publication can leave only its packet-index row.  It
    # is not a registered writer because the ledger and governance revision did
    # not advance; replace that exact row idempotently.
    stale_rows = [row for row in index["task_packets"] if row.get("task_id") == TASK]
    if stale_rows and not any(row.get("task_id") == TASK for row in state.get("tasks", [])):
        index["task_packets"] = [row for row in index["task_packets"] if row.get("task_id") != TASK]
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if RECEIPT.exists():
        raise RuntimeError("FRESH_PUBLICATION_REQUIRED")
    for path, digest in EXPECTED.items():
        if artifact_ref(path)["sha256"] != digest:
            raise RuntimeError(f"SOURCE_SHA_DRIFT:{path}")

    pointer = load_json(POINTER)
    branch_lock = load_json(LOCK)
    robot = load_json(ROBOT_RESULT)
    if pointer.get("baseline_id") != "EXACT78_CARDS042_VISUAL_BASELINE_V2":
        raise RuntimeError("POINTER_ID")
    if branch_lock.get("baseline_pointer") != "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json":
        raise RuntimeError("LOCK_POINTER")
    if robot.get("status") != "PASS_VISUAL_REVIEW_CANDIDATE" or not all(robot.get("hard_gates", {}).values()):
        raise RuntimeError("ROBOT_HARD_GATES")
    if robot.get("training_eligible") is not False or robot.get("scene_depth_consumed") is not False:
        raise RuntimeError("AUTHORITY_BOUNDARY")

    expected_video_frames = {"composite": 171, "review_4panel": 171, "external_robot": 171,
                             "contact_review": 16}
    video_checks = {}
    for name, frames in expected_video_frames.items():
        ref = pointer["selected_artifacts"][name if name != "composite" else "first_person_clean_robot"]
        path = Path(ref["path"])
        if artifact_ref(path) != ref:
            raise RuntimeError(f"VIDEO_SHA_DRIFT:{name}")
        actual = decoded_frames(path)
        if actual != frames:
            raise RuntimeError(f"VIDEO_FRAME_COUNT:{name}:{actual}")
        video_checks[name] = {"artifact": ref, "decoded_frames": actual}

    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"pointer": artifact_ref(POINTER), "branch_lock": artifact_ref(LOCK),
              "handoff": artifact_ref(HANDOFF), "robot_result": artifact_ref(ROBOT_RESULT)}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "video_checks": video_checks,
                                                "decision": "PUBLISH_SOLE_REPRODUCIBLE_V2"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Publish the sole reproducible Cards042 V2 visual baseline and retire V3 routing",
        "phase": "EXACT78_CARDS042_VISUAL_BASELINE_V2_PUBLICATION",
        "plan_revision": "EXACT78_CARDS042_VISUAL_BASELINE_V2",
        "execution_revision": "EXACT78_CARDS042_VISUAL_BASELINE_V2_PUBLISH",
        "session_id": "play_cards_0902_042",
        "deadline_at": (datetime.now().astimezone() + timedelta(minutes=30)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
                     "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json"],
        "write_set": [str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "docs/governance"],
        "prerequisites": ["NO_ACTIVE_WRITER", "FRESH_STAGE_RESULTS_SHA_FROZEN",
                          "ALL_SELECTED_VIDEOS_FULL_DECODE", "V3_ROBOT_ANCHOR_PIXEL_REGRESSION_PASS"],
        "required_outputs": ["V2_publication_receipt", "V2_current_document_authority",
                             "explicit_quality_boundaries"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 1800, "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "active_writer_conflict", "video_decode_failure",
                            "authority_boundary_drift", "one_attempt_terminal"],
        "claim_limit": ("One Cards042 reproducible development visual baseline. Clean is synthetic, contact is "
                        "2-D visual only, motion is selected frozen state, and no training/control/deployment authority exists."),
        "writer": writer, "inputs": inputs, "run_signature": signature,
        "commands": ["python -m chaoyang.governance.publish_exact78_cards042_visual_baseline_v2"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    ROOT.mkdir(parents=True, exist_ok=True)
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "play_cards_0902_042_documentation", "updated_at": now, "heartbeat_at": now,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "PENDING", "adoption": "PENDING", "review": "AI_FIXED_FRAMES_PASS_HUMAN_FULL_VIDEO_PENDING",
    })
    state["next_task"] = {"task_id": TASK, "session": "play_cards_0902_042_documentation",
                          "prerequisites": packet["prerequisites"], "expected_resource": "CPU_ONE_THREAD",
                          "stop_condition": "one terminal publication"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CARDS042_VISUAL_BASELINE_V2_RUNNING", status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    authority["claims"] = [claim for claim in authority.get("claims", []) if claim.get("scope") not in {
        "exact78/play_cards_0902_042/current_visual_adjustment_parent_v1",
        "exact78/play_cards_0902_042/current_visual_stage_version_lock_v1",
    }]
    publish_bundle(authority, state, event_type="EXACT78_CARDS042_VISUAL_BASELINE_V2_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)

    receipt_value = {
        "schema_version": "exact78-cards042-visual-baseline-v2-receipt",
        "task_id": TASK, "status": "PASSED", "recorded_at": now_iso(),
        "decision": "PUBLISH_V2_AS_SOLE_CURRENT_REPRODUCIBLE_DEVELOPMENT_VISUAL_BASELINE",
        "baseline_pointer": artifact_ref(POINTER), "branch_lock": artifact_ref(LOCK),
        "handoff": artifact_ref(HANDOFF), "robot_result": artifact_ref(ROBOT_RESULT),
        "video_checks": video_checks,
        "review": {"ai_fixed_frames": [0, 42, 76, 77, 82, 84, 87, 91, 96, 125, 137, 170],
                   "selected_robot_rgba_exact_to_user_accepted_v3_frames": [0, 42, 76, 77, 82, 84, 87, 91, 170],
                   "new_full_video_human_review": "PENDING"},
        "not_promoted": {"clean_standalone_quality": False, "physical_occlusion": False,
                         "training_eligible": False, "control_ground_truth": False,
                         "physical_deployment": False},
        "cleanup_authority": "Only a separate explicit allowlist receipt may remove retired dependencies.",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(RECEIPT, receipt_value)
    result = {"schema_version": "exact78-cards042-visual-baseline-v2-publication-result",
              "task_id": TASK, "status": "PASSED_CURRENT_V2_PUBLISHED",
              "baseline_pointer": artifact_ref(POINTER), "branch_lock": artifact_ref(LOCK),
              "publication_receipt": artifact_ref(RECEIPT), "video_checks": video_checks,
              "training_eligible": False, "physical_occlusion_authority": False,
              "claim_limit": packet["claim_limit"]}
    atomic_json(ROOT / "RESULT.json", result)

    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    authority.setdefault("claims", []).append({
        "claim": "Exact78 Cards042 V2 是当前唯一可路由、可重跑的单会话视觉开发基线。",
        "claim_limit": packet["claim_limit"], "scope": "exact78/play_cards_0902_042/current_visual_baseline_v2",
        "status": "DEVELOPMENT_EVIDENCE",
        "evidence": [artifact_ref(RECEIPT), artifact_ref(ROBOT_RESULT), artifact_ref(POINTER)],
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status="PASSED", pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ROOT / "RESULT.json"), terminal_receipt=artifact_ref(RECEIPT),
               quality="REPRODUCIBLE_DEVELOPMENT_VISUAL_BASELINE", adoption="CURRENT_V2",
               review="AI_FIXED_FRAMES_PASS_HUMAN_FULL_VIDEO_PENDING")
    state["next_task"] = None
    index = index_body()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_VISUAL_BASELINE_V2_TERMINAL",
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(authority, state, event_type="EXACT78_CARDS042_VISUAL_BASELINE_V2_PUBLISHED",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    validation = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                                cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                                capture_output=True, text=True)
    if validation.returncode:
        raise RuntimeError("POST_PUBLISH_INVALID:" + validation.stdout[-3000:] + validation.stderr[-3000:])
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"],
                      "result": artifact_ref(ROOT / "RESULT.json"),
                      "receipt": artifact_ref(RECEIPT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
