#!/usr/bin/env python3
"""Finalize per-session R2 candidate products and verify exact resume reuse."""
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
    for short, expected in (("031", 149), ("007", 378)):
        root = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
        result_path, video = root / "RESULT.json", root / "robot_candidate.mp4"
        result = load(result_path)
        before = {"result": ref(result_path), "video": ref(video)}
        command = [sys.executable, "-B", "-m", "chaoyang.ops.run_human_to_robot_r2_candidate_product",
                   "--session", short]
        completed = subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": "src"},
                                   text=True, capture_output=True)
        after = {"result": ref(result_path), "video": ref(video)}
        if completed.returncode or before != after:
            raise RuntimeError(f"PRODUCT_RESUME_RERAN_OR_FAILED:{short}:{completed.stderr[-2000:]}")
        if result.get("decoded_frames") != expected or result.get("adoption") != "CANDIDATE_ONLY":
            raise ValueError(f"PRODUCT_RESULT_STATE:{short}")
        rows.append({
            "session_id": result["session_id"],
            "execution": result["execution"],
            "structure": result["structure"],
            "quality": result["quality"],
            "adoption": result["adoption"],
            "decoded_frames": expected,
            "valid_side_frames_anatomical_left_right": result["valid_side_frames_anatomical_left_right"],
            "invalid_side_policy": result["invalid_side_policy"],
            "result": before["result"],
            "video": before["video"],
            "resume_same_bytes": True,
            "resume_command": command,
        })
    receipt = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PRODUCT_RESUME_V1",
        "task_id": TASK,
        "created_at": now(),
        "sessions": rows,
        "candidate_complete": 2,
        "quality_pass": 0,
        "adopted": 0,
        "resume_same_bytes": True,
        "claim_limit": "Per-session offline candidate join and same-byte resume only; rejected Clean and unknown occlusion remain.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    receipt_path = ATTEMPT / "lanes/lane2_motion/PRODUCT_CANDIDATE_AND_RESUME_WAVE6.json"
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
        status="EXECUTED_WITH_LOCAL_QUALITY_LIMIT",
        execution="EXECUTED",
        structure="PASS",
        quality="INCONCLUSIVE",
        adoption="CANDIDATE_ONLY",
        blocker={
            "code": "031_LEFT_OBSERVABILITY_INCONCLUSIVE",
            "missing": "independent side-resolved evidence for anatomical left",
            "consumer": "031 bilateral-motion quality claim only",
            "owner": "lane2_motion",
            "unblock_action": "obtain independent side-resolved evidence; do not infer from evaluated HaWoR",
            "affected": ["031 bilateral quality claim"],
            "unaffected": ["031 single-valid-side candidate", "007 bilateral candidate", "Sensor", "Scene"],
        },
        current_action="031 single-valid-side and 007 bilateral candidate products completed; same-byte resume passed.",
        latest_artifacts=artifacts,
        writer={"pid": os.getpid(), "proc_start_ticks": None},
        updated_at=now(),
    )
    write_json(state_path, state)
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE6_PER_SESSION_PRODUCT_AND_RESUME",
        "observed_at": now(),
        "products": rows,
        "main_candidate_complete": 2,
        "main_quality_pass": 0,
        "main_adopted": 0,
        "resume_same_bytes": True,
        "scene_model_candidate_sessions": 4,
        "scene_adopted_sessions": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE6.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
