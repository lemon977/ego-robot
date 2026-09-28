#!/usr/bin/env python3
"""Publish the two 0902 same-recipe candidate products and resume evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, REPO, TASK, ref, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for short, expected in (("103", 284), ("042", 171)):
        root = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
        result_path, video = root / "RESULT.json", root / "robot_candidate.mp4"
        before = {"result": ref(result_path), "video": ref(video)}
        result = load(result_path)
        command = [sys.executable, "-B", "-m", "chaoyang.ops.run_human_to_robot_r2_candidate_product",
                   "--session", short]
        completed = subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": "src"},
                                   text=True, capture_output=True)
        after = {"result": ref(result_path), "video": ref(video)}
        if completed.returncode or before != after:
            raise RuntimeError(f"REGRESSION_PRODUCT_RESUME:{short}:{completed.stderr[-2000:]}")
        if result.get("decoded_frames") != expected or result.get("quality") != "REJECTED_QUALITY":
            raise ValueError(f"REGRESSION_PRODUCT_RESULT:{short}")
        rows.append({
            "session_id": result["session_id"], "frame_count": expected,
            "execution": result["execution"], "structure": result["structure"],
            "quality": result["quality"], "adoption": result["adoption"],
            "valid_side_frames_anatomical_left_right": result["valid_side_frames_anatomical_left_right"],
            "result": before["result"], "video": before["video"], "resume_same_bytes": True,
        })
    receipt = {
        "schema_version": "HUMAN_TO_ROBOT_R2_REGRESSION_PRODUCT_V1",
        "task_id": TASK, "created_at": now(), "sessions": rows,
        "candidate_complete": 2, "quality_pass": 0, "adopted": 0,
        "resume_same_bytes": True,
        "claim_limit": "0902 same-recipe offline candidates only; Scene quality and occlusion remain rejected/unknown.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    receipt_path = ATTEMPT / "lanes/lane2_motion/REGRESSION_PRODUCT_AND_RESUME_WAVE7.json"
    write_json(receipt_path, receipt)
    state_path = ATTEMPT / "lanes/lane2_motion/STATE.json"
    state = load(state_path)
    artifacts = state.get("latest_artifacts", [])
    seen = {item.get("path") for item in artifacts if isinstance(item, dict)}
    for path in (receipt_path, *(Path(row["video"]["path"]) for row in rows)):
        item = ref(path)
        if item["path"] not in seen:
            artifacts.append(item); seen.add(item["path"])
    state.update(
        current_action="Four per-session Robot candidates and same-byte resume checks completed; all remain quality-rejected because Scene/occlusion are not adopted.",
        candidate_product_sessions=4,
        adopted_product_sessions=0,
        latest_artifacts=artifacts,
        writer={"pid": os.getpid(), "proc_start_ticks": None}, updated_at=now(),
    )
    write_json(state_path, state)
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK, "stage": "WAVE7_0902_REGRESSION_PRODUCTS",
        "observed_at": now(), "regression_products": rows,
        "main_candidate_complete": 2, "regression_candidate_complete": 2,
        "all_product_candidate_complete": 4, "product_quality_pass": 0,
        "product_adopted": 0, "resume_same_bytes": True,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE7.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
