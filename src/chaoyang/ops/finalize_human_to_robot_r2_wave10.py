#!/usr/bin/env python3
"""Publish same-background diagnostics and migrate strict product resumes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.ops.run_human_to_robot_r2_candidate_product import CASES, product_binding
from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, TASK, ref, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    resume_rows = []
    for short, case in CASES.items():
        root = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
        result_path, video_path = root / "RESULT.json", root / "robot_candidate.mp4"
        scene_root = ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{case['scene_wave']}"
        scene_path = scene_root / "RESULT.json"
        prep = load(scene_root / "SCENE_PREP_MANIFEST.json")
        domain_path = Path(prep["source_domain"]["path"])
        motion_path = ATTEMPT / f"lanes/lane2_motion/{case['motion']}/ROBOT_R0_V1.npz"
        binding = product_binding(scene_path=scene_path, robot_path=motion_path, domain_path=domain_path)
        binding.update({
            "task_id": TASK, "session_id": case["session"], "created_at": now(),
            "product_result": ref(result_path), "product_video": ref(video_path),
            "resume_policy": "REUSE_ONLY_WHEN_SIGNATURE_AND_OUTPUT_SHA_MATCH",
            "migration_basis": "Existing Wave6/7 output had prior same-byte resume evidence; render semantics unchanged, strict validation added.",
        })
        binding_path = root / "CACHE_BINDING_V2.json"
        if not binding_path.is_file():
            write_json(binding_path, binding)
        before = {"result": ref(result_path), "video": ref(video_path)}
        command = [sys.executable, "-B", "-m", "chaoyang.ops.run_human_to_robot_r2_candidate_product",
                   "--session", short]
        completed = subprocess.run(command, cwd=Path.cwd(), env={**os.environ, "PYTHONPATH": "src"},
                                   text=True, capture_output=True)
        after = {"result": ref(result_path), "video": ref(video_path)}
        if completed.returncode or before != after:
            raise RuntimeError(f"STRICT_RESUME_FAILED:{short}:{completed.stderr[-2000:]}")
        resume_rows.append({"session_id": case["session"], "strict_resume": "PASS",
                            "binding": ref(binding_path), **after})

    compare_rows = []
    for short in ("031", "007"):
        root = ATTEMPT / f"lanes/lane4_compare/same_rejected_background_{short}_wave10"
        result_path = root / "RESULT.json"
        result = load(result_path)
        if result.get("decoded_frames") != result.get("frame_count") or result.get("winner") is not None:
            raise ValueError(f"COMPARE_RESULT:{short}")
        compare_rows.append({
            "session_id": result["session_id"],
            "frame_count": result["frame_count"],
            "common_valid_side_frames": result["common_valid_side_frames"],
            "quality": result["quality"],
            "adoption": result["adoption"],
            "result": ref(result_path),
            "video": result["video"],
        })
    receipt = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE10_STRICT_RESUME_AND_LOCAL_HURO_DIAGNOSTIC",
        "observed_at": now(),
        "strict_product_resume": resume_rows,
        "same_rejected_background_comparisons": compare_rows,
        "compare_execution": "EXECUTED",
        "compare_structure": "PASS",
        "compare_quality": "INCONCLUSIVE_REJECTED_BACKGROUND",
        "compare_adoption": "NOT_ADOPTED",
        "winner": None,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    progress_path = ATTEMPT / "PROGRESS_WAVE10.json"
    write_json(progress_path, receipt)

    state_path = ATTEMPT / "lanes/lane4_compare/STATE.json"
    state = load(state_path)
    artifacts = state.get("latest_artifacts", [])
    existing = {item.get("path") for item in artifacts if isinstance(item, dict)}
    for row in compare_rows:
        for item in (row["result"], row["video"]):
            if item["path"] not in existing:
                artifacts.append(item); existing.add(item["path"])
    state.update(
        current_action="Two same-background Local/HuRo diagnostics executed and decoded; rejected Clean and unknown occlusion keep final comparison blocked and no winner is declared.",
        latest_artifacts=artifacts,
        writer={"pid": os.getpid(), "proc_start_ticks": None},
        updated_at=now(),
    )
    write_json(state_path, state)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
