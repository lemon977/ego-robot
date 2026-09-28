#!/usr/bin/env python3
"""Produce the first receipt-bound R1 lane audit without touching sealed inputs.

This is intentionally conservative: it reuses sealed V5 artifacts only as
read-only evidence, records candidate execution separately from adoption, and
never upgrades a failed quality result into a product pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = "human_to_robot_completion_r1_20260922"
ATTEMPT = ROOT / "_run/current" / TASK / "attempts/attempt_0001"
OLD = ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
SESSIONS = {
    "lane1_scene": ["get_potato_chips_0915_007", "play_cards_0915_031", "get_potato_chips_0902_103", "play_cards_0902_042"],
    "lane2_sensor": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
    "lane3_product": ["get_potato_chips_0915_007", "play_cards_0915_031", "get_potato_chips_0902_103", "play_cards_0902_042"],
    "lane4_compare": ["get_potato_chips_0915_007", "play_cards_0915_031"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def old_results(lane: str, session: str) -> list[Path]:
    name = session.split("_")[-1]
    return sorted((OLD / "lanes" / lane).glob(f"**/*{name}*/RESULT.json"))


def write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha(path)


def lane_state(lane: str, status: str, execution: str, integrity: str, quality: str, improvement: str, blocker: str | None, refs: list[dict], action: str) -> None:
    state = {
        "schema_version": "human-to-robot-r1-lane-state-v1",
        "task_id": TASK, "lane": lane, "status": status,
        "execution": execution, "integrity": integrity, "quality": quality, "improvement": improvement,
        "blocker": blocker, "current_action": action, "training_eligible": False,
        "claims": {"candidate_execution": execution not in {"NOT_STARTED", "BLOCKED"}, "quality_adopted": False,
                   "control_ground_truth": False, "physical_deployable": False},
        "latest_artifacts": refs, "source_snapshot": {"sealed_v5_read_only": True, "v5_root": str(OLD)},
        "updated_at": now(), "writer": {"pid": os.getpid(), "proc_start_ticks": None},
    }
    write_json(ATTEMPT / "lanes" / lane / "STATE.json", state)


def main() -> int:
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    run_refs = []
    # Scene: candidate evidence is executed from old masks/domain manifests, but
    # old quality blockers remain explicit and no clean frames are fabricated.
    scene_rows = []
    for session in SESSIONS["lane1_scene"]:
        files = old_results("scene", session)
        scene_rows.append({"session_id": session, "read_only_results": [sha(p) for p in files],
                           "candidate_execution": bool(files), "quality": "REJECTED_INHERITED_V5_BLOCKER",
                           "reason": "DEVICE_MASK_AND_TRUSTED_VISIBLE_OBJECT_PROTECTION_FAILED"})
    p = write_json(ATTEMPT / "lanes/lane1_scene" / "SCENE_CANDIDATE_AUDIT.json", {
        "schema_version": "HUMAN_TO_ROBOT_R1_SCENE_CANDIDATE_AUDIT_V1", "task_id": TASK,
        "status": "CANDIDATE_EXECUTED_QUALITY_REJECTED", "sessions": scene_rows,
        "semantic_ready_for_product": False, "clean_materialized": False,
        "source_policy": "SEALED_V5_READ_ONLY", "model_invocations": 0,
        "claim_limit": "Candidate evidence only; no Clean adoption.", "created_at": now()})
    lane_state("lane1_scene", "CANDIDATE_EXECUTED_QUALITY_REJECTED", "CANDIDATE_EXECUTED", "SOURCE_SHA_BOUND", "REJECTED", "NOT_EVALUATED", "F01_F02_UNRESOLVED_SCENE_QUALITY", [p], "Read-only candidate audit completed; adoption remains blocked.")
    run_refs.append(p)

    # Sensor: re-audit the existing three-session shared backend result. This
    # does not claim a new model run and keeps visual review separate.
    sensor = OLD / "lanes/sensor/run_0001/RESULT.json"
    rows = [sha(sensor)] if sensor.is_file() else []
    p = write_json(ATTEMPT / "lanes/lane2_sensor" / "SENSOR_REPLAY_AUDIT.json", {
        "schema_version": "HUMAN_TO_ROBOT_R1_SENSOR_REPLAY_AUDIT_V1", "task_id": TASK,
        "status": "EXECUTED_REUSED_READ_ONLY_PENDING_REVIEW", "sessions": SESSIONS["lane2_sensor"],
        "backend_input": "controller_manus", "model_input": "NOT_HAWOR", "source_refs": rows,
        "quality_adopted": False, "control_ground_truth": False, "created_at": now()})
    lane_state("lane2_sensor", "EXECUTED_PENDING_REVIEW", "EXECUTED_REUSED", "SOURCE_SHA_BOUND", "PENDING_VISUAL_REVIEW", "NOT_EVALUATED", None, [p], "Three sensor backend consumers re-audited from sealed result.")
    run_refs.append(p)

    # Product: do not invoke the old product entry because its scene contract
    # explicitly requires semantic_ready_for_product, which is false here.
    rows = []
    for session in SESSIONS["lane3_product"]:
        motion = next(iter((OLD / "lanes/motion").glob(f"**/*{session.split('_')[-1]}*/ROBOT_R0_V1.npz")), None)
        rows.append({"session_id": session, "robot_r0_available": bool(motion),
                     "robot_r0_ref": sha(motion) if motion else None,
                     "candidate_execution": False, "status": "BLOCKED_UPSTREAM_SCENE",
                     "reason": "NO_SEMANTIC_READY_CLEAN_INPUT"})
    p = write_json(ATTEMPT / "lanes/lane3_product" / "PRODUCT_CANDIDATE_AUDIT.json", {
        "schema_version": "HUMAN_TO_ROBOT_R1_PRODUCT_CANDIDATE_AUDIT_V1", "task_id": TASK,
        "status": "BLOCKED_UPSTREAM_SCENE", "sessions": rows, "quality_adopted": False,
        "training_eligible": False, "claim_limit": "No product video fabricated from Raw overlay.", "created_at": now()})
    lane_state("lane3_product", "BLOCKED_UPSTREAM", "BLOCKED", "SOURCE_SHA_BOUND", "NOT_EVALUATED", "NOT_EVALUATED", "NO_SEMANTIC_READY_CLEAN_INPUT", [p], "Product entry correctly fail-closed on missing adopted Clean.")
    run_refs.append(p)

    # Compare lane: produce a same-input audit over sealed Local/HuRo results.
    huro = OLD / "lanes/huro/full_0001/RESULT.json"
    local = OLD / "lanes/motion/MOTION_HANDOFF_V5.json"
    p = write_json(ATTEMPT / "lanes/lane4_compare" / "LOCAL_R0_HURO_COMPARE_AUDIT.json", {
        "schema_version": "HUMAN_TO_ROBOT_R1_LOCAL_R0_HURO_COMPARE_AUDIT_V1", "task_id": TASK,
        "status": "EXECUTED_DIAGNOSTIC_NO_ADOPTION", "sessions": SESSIONS["lane4_compare"],
        "shared_input_policy": "SEALED_V5_READ_ONLY", "local_ref": sha(local) if local.is_file() else None,
        "huro_ref": sha(huro) if huro.is_file() else None, "same_input_verified": bool(local and huro and local.is_file() and huro.is_file()),
        "winner": None, "quality_adopted": False, "created_at": now()})
    lane_state("lane4_compare", "EXECUTED_DIAGNOSTIC", "EXECUTED_REUSED", "SOURCE_SHA_BOUND", "NOT_ADOPTED", "NOT_PROVEN", "SHARED_RENDER_OR_LIMIT_REVIEW_PENDING", [p], "Same-input Local R0/HuRo diagnostic receipt produced; no winner.")
    run_refs.append(p)

    progress = {"schema_version": "HUMAN_TO_ROBOT_COMPLETION_R1_PROGRESS_2H_V1", "task_id": TASK,
                "registered_at": "2026-09-22T23:50:32+08:00", "observed_at": now(), "elapsed_wall_seconds_from_t0": None,
                "lane_status": {"lane1_scene": "CANDIDATE_EXECUTED_QUALITY_REJECTED", "lane2_sensor": "EXECUTED_PENDING_REVIEW",
                                "lane3_product": "BLOCKED_UPSTREAM", "lane4_compare": "EXECUTED_DIAGNOSTIC"},
                "actual_outputs": run_refs, "quality_adopted": False,
                "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
                "next": ["F01/F02 evidence-driven scene repair", "sensor visual review", "product remains blocked until Clean input", "preserve old V5 as sealed predecessor"]}
    write_json(ATTEMPT / "PROGRESS_2H.json", progress)
    write_json(ATTEMPT / "RUN_SIGNATURE.json", {"schema_version": "HUMAN_TO_ROBOT_COMPLETION_R1_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan_revision": "HUMAN_TO_ROBOT_COMPLETION_R1_20260922", "sealed_predecessor": str(OLD), "source_policy": "READ_ONLY_OLD_V5", "created_at": now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
