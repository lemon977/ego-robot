#!/usr/bin/env python3
"""Publish R2 wave-4 actual Scene candidates without promoting their quality.

The four candidates have real ProPainter executions and complete review
videos, but the frozen Scene gate still rejects visible-object protection and
device-role evidence.  This publisher keeps execution, structure, quality and
adoption as independent axes and also removes stale lane blockers that were
closed by earlier waves.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import (
    ATTEMPT, TASK, ref, write_json,
)


CASES = {
    "031": ("play_cards_0915_031", 149, "main"),
    "007": ("get_potato_chips_0915_007", 378, "main"),
    "103": ("get_potato_chips_0902_103", 284, "regression"),
    "042": ("play_cards_0902_042", 171, "regression"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def update_state(lane: str, **updates) -> None:
    path = ATTEMPT / "lanes" / lane / "STATE.json"
    state = load(path)
    state.update(updates)
    state.update(writer={"pid": os.getpid(), "proc_start_ticks": None}, updated_at=now())
    write_json(path, state)


def main() -> int:
    rows = []
    artifacts = []
    for short, (session, expected, role) in CASES.items():
        result_path = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_wave4/RESULT.json"
        result = load(result_path)
        review = Path(result["review"]["video"]["path"])
        if result.get("session_id") != session or int(result.get("frame_count", -1)) != expected:
            raise ValueError(f"SESSION_OR_FRAME_MISMATCH:{short}")
        if result.get("execution") != "EXECUTED" or result.get("structure") != "PASS":
            raise ValueError(f"SCENE_STRUCTURE_NOT_PASS:{short}")
        if result.get("quality") != "REJECTED_QUALITY" or result.get("adoption") != "CANDIDATE_ONLY":
            raise ValueError(f"QUALITY_PROMOTION_FORBIDDEN:{short}")
        if result["review"].get("decoded_frames") != expected or not review.is_file():
            raise ValueError(f"REVIEW_INCOMPLETE:{short}")
        rows.append({
            "session_id": session,
            "role": role,
            "frame_count": expected,
            "execution": result["execution"],
            "structure": result["structure"],
            "quality": result["quality"],
            "adoption": result["adoption"],
            "result": ref(result_path),
            "review": ref(review),
            "known_quality_blockers": result["known_quality_blockers"],
        })
        artifacts.extend([result_path, review])

    scene_state_path = ATTEMPT / "lanes/lane1_scene/STATE.json"
    scene_state = load(scene_state_path)
    existing = scene_state.get("latest_artifacts", [])
    seen = {item.get("path") for item in existing if isinstance(item, dict)}
    for path in artifacts:
        item = ref(path)
        if item["path"] not in seen:
            existing.append(item)
            seen.add(item["path"])
    scene_state.update(
        status="RUNNING_WITH_LOCAL_BLOCKER",
        execution="EXECUTED",
        structure="PASS",
        quality="REJECTED_QUALITY",
        adoption="NOT_ADOPTED",
        latest_artifacts=existing,
        current_action="Four actual ProPainter candidates completed; visible-object protection and device-role quality remain rejected.",
        candidate_sessions=4,
        adopted_clean_sessions=0,
        blocker={
            "code": "OBJECT_PROTECTION_AND_DEVICE_EVIDENCE_INCOMPLETE",
            "missing": "trusted directly-visible object protection and complete device/cable role evidence",
            "consumer": "formal Clean and same-background product",
            "owner": "lane1_scene",
            "unblock_action": "validate frozen-window local device observations and directly-visible object protection without full-frame union",
            "affected": ["formal Clean adoption", "same-background product", "real-scene occlusion review"],
            "unaffected": ["Motion/R0", "Sensor backend", "Local/HuRo numeric comparison", "candidate review"],
        },
        writer={"pid": os.getpid(), "proc_start_ticks": None},
        updated_at=now(),
    )
    write_json(scene_state_path, scene_state)

    render = ATTEMPT / "lanes/lane4_compare/renderer_four_way_wave3/RESULT.json"
    update_state(
        "lane4_compare",
        status="RUNNING_WITH_LOCAL_BLOCKER",
        blocker={
            "code": "FINAL_SAME_BACKGROUND_COMPARISON_BLOCKED_CLEAN",
            "missing": "adopted same-session Clean",
            "consumer": "final same-background Local/HuRo visual comparison",
            "owner": "lane1_scene",
            "unblock_action": "obtain G2/G3 adopted Clean; numeric and renderer-isolation diagnostics are already complete",
            "affected": ["final same-background comparison"],
            "unaffected": ["same-target numeric diagnostic", "renderer four-way isolation"],
            "evidence": ref(render),
        },
        current_action="Renderer q/camera/background isolation passed; final same-background comparison waits only for adopted Clean.",
    )
    update_state(
        "lane3_sensor",
        status="EXECUTED_KINEMATIC_ONLY",
        blocker=None,
        current_action="097/098/101 common backend, FK and collision diagnostics completed at KINEMATIC_ONLY authority.",
    )

    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE4_ACTUAL_SCENE_CANDIDATES",
        "observed_at": now(),
        "scene_candidates": rows,
        "main_candidate_complete": 2,
        "main_quality_pass": 0,
        "regression_candidate_complete": 2,
        "regression_quality_pass": 0,
        "adopted_clean_sessions": 0,
        "product_sessions": 0,
        "model_execution_sessions": 4,
        "claim_limit": "Actual offline ProPainter candidates; no candidate is adopted Clean or product authority.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    path = ATTEMPT / "PROGRESS_WAVE4.json"
    if path.exists():
        raise FileExistsError(f"IMMUTABLE_PROGRESS_EXISTS:{path}")
    write_json(path, progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
