#!/usr/bin/env python3
"""Publish the immutable T+2h receipt and adapter scene evidence."""
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
    adapter_path = ATTEMPT / "lanes/lane2_motion/assembly_scene_canary_wave11/RESULT.json"
    adapter = load(adapter_path)
    if not adapter.get("scene_object_loaded") or adapter.get("adoption") != "NOT_ADOPTED":
        raise ValueError("ADAPTER_CANARY_RESULT")
    state_path = ATTEMPT / "lanes/lane2_motion/STATE.json"
    state = load(state_path)
    artifacts = state.get("latest_artifacts", [])
    item = ref(adapter_path)
    if not any(row.get("path") == item["path"] for row in artifacts if isinstance(row, dict)):
        artifacts.append(item)
    blockers = state.get("blockers", [])
    assembly_blocker = {
        "code": "COMPLETE_ASSEMBLY_MEASUREMENT_ABSENT",
        "missing": "measured installation transform, Robot TCP, and camera/world-to-base calibration",
        "consumer": "complete assembly adoption and physical deployment",
        "owner": "external hardware calibration",
        "unblock_action": "capture and validate the missing measured transforms; do not infer them from STEP geometry",
        "affected": ["complete assembly adoption", "physical deployment"],
        "unaffected": ["offline R0 candidates", "Scene", "Sensor common backend", "diagnostic rendering"],
        "evidence": item,
    }
    blockers = [row for row in blockers if row.get("code") != assembly_blocker["code"]]
    blockers.append(assembly_blocker)
    state.update(
        blockers=blockers,
        current_action="Four content-bound product candidates complete; real adapter mesh loads in a diagnostic scene, while measured assembly transforms remain absent.",
        latest_artifacts=artifacts,
        writer={"pid": os.getpid(), "proc_start_ticks": None},
        updated_at=now(),
    )
    write_json(state_path, state)

    scene_receipts = [
        ATTEMPT / f"lanes/lane1_scene/clean_candidate_{short}_{wave}/RESULT.json"
        for short, wave in (("031", "wave5"), ("007", "wave4"), ("103", "wave4"), ("042", "wave4"))
    ]
    product_receipts = [
        ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6/RESULT.json"
        for short in ("031", "007", "103", "042")
    ]
    sensor_receipts = [
        ATTEMPT / f"lanes/lane3_sensor/run_{short}_{wave}/RESULT.json"
        for short, wave in (("097", "wave0"), ("098", "wave1"), ("101", "wave1"))
    ]
    compare_receipts = [
        ATTEMPT / f"lanes/lane4_compare/same_rejected_background_{short}_wave10/RESULT.json"
        for short in ("031", "007")
    ]
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_2H_V1",
        "task_id": TASK,
        "observed_at": now(),
        "elapsed_checkpoint": "T_PLUS_2H",
        "scene": {"executed": 4, "structure_pass": 4, "quality_pass": 0, "adopted": 0,
                  "receipts": [ref(path) for path in scene_receipts]},
        "motion": {"session_r0": 4, "product_candidates": 4, "product_quality_pass": 0,
                   "product_adopted": 0, "strict_resume_pass": 4,
                   "receipts": [ref(path) for path in product_receipts]},
        "sensor": {"common_backend_sessions": 3, "authority": "KINEMATIC_ONLY",
                   "receipts": [ref(path) for path in sensor_receipts]},
        "contact_robot_r1": {"eligible_metric_windows": 0, "executed": 0, "adopted": 0,
                             "receipt": ref(ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json")},
        "local_huro": {"numeric_sessions": 2, "same_candidate_background_videos": 2,
                       "winner": None, "receipts": [ref(path) for path in compare_receipts]},
        "occlusion": {"synthetic_contract": "PASS", "real_sessions": "UNKNOWN",
                      "receipt": ref(ATTEMPT / "lanes/lane1_scene/OCCLUSION_CONTRACT_WAVE8.json")},
        "assembly": {"candidate_mesh_loaded": True, "measured_installation": "ABSENT",
                     "receipt": item},
        "gpu": {"actual_model_invocations": 5, "observed_seconds": 976.646,
                "lease_status_after_runs": "RELEASED"},
        "remaining_work": [
            "quality blocker finalization and complete validation",
            "formal current-state/result publication",
        ],
        "quality_limits": [
            "device evidence is nonempty in only 7/149 and 7/378 primary-session frames",
            "all four Scene candidates remain REJECTED_QUALITY",
            "registered real-session occlusion and strict metric Contact are unavailable",
            "measured assembly transforms are absent",
        ],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    write_json(ATTEMPT / "PROGRESS_2H.json", progress)
    write_json(ATTEMPT / "PROGRESS_WAVE11.json", {**progress, "stage": "WAVE11_ADAPTER_AND_2H_RECEIPT"})
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
