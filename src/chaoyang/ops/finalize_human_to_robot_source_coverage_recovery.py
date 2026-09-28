"""Seal the bounded 007/031 source-recovery task without product promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.ops.run_human_to_robot_source_coverage_diagnostic import TASK, OUT

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
STATUS = REPO_ROOT / "docs/current/STATUS.json"
EVIDENCE = {
    "scene_diagnostic": OUT / "lanes/scene/source_coverage_v1/RESULT.json",
    "scene_rebound": OUT / "lanes/scene/rebound_007_window_v2/RESULT.json",
    "scene_candidate": OUT / "lanes/scene/rebound_clean_007_window_v1/RESULT.json",
    "scene_review": OUT / "lanes/scene/rebound_clean_007_window_v1/AI_VISUAL_REVIEW.json",
    "gpu_call": OUT / "lanes/scene/REBOUND_GPU_RECEIPT.json",
    "motion_diagnostic": OUT / "lanes/motion_product/target_source_v1/RESULT.json",
    "motion_projection": OUT / "lanes/motion_product/source_projection_031_v1/RESULT.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_SOLE_INDEX_ENTRY")
    for path in EVIDENCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if (OUT / "RESULT.json").exists():
        raise FileExistsError(OUT / "RESULT.json")
    source = load_json(EVIDENCE["scene_diagnostic"])
    rebound = load_json(EVIDENCE["scene_rebound"])
    candidate = load_json(EVIDENCE["scene_candidate"])
    review = load_json(EVIDENCE["scene_review"])
    gpu = load_json(EVIDENCE["gpu_call"])
    projection = load_json(EVIDENCE["motion_projection"])
    if source["frames_left_screen_model_mask_absent"] != [184, 185, 186, 187, 188]:
        raise RuntimeError("SCENE_FAILURE_REPRODUCTION_CHANGED")
    if min(row["new_left_screen_support_pixels"] for row in rebound["rows"]) <= 0:
        raise RuntimeError("REBOUND_INPUT_INCOMPLETE")
    if gpu.get("status") != "PASSED" or candidate.get("model_returncode") != 0:
        raise RuntimeError("CHANGED_INPUT_MODEL_NOT_EXECUTED")
    if review["quality"] != "REJECTED_QUALITY_BY_VISIBLE_COUNTEREXAMPLES":
        raise RuntimeError("QUALITY_WAS_NOT_INDEPENDENTLY_REJECTED")
    if projection["model_prediction_frames"] != 102 or projection["projected_wrist_inside_image_frames"] != 0:
        raise RuntimeError("031_PROJECTION_COUNTS_CHANGED")
    video = Path(candidate["review"]["path"])
    if artifact_ref(video)["sha256"] != candidate["review"]["sha256"]:
        raise RuntimeError("VIDEO_SHA_MISMATCH")
    if review["review_video_sha256"] != candidate["review"]["sha256"]:
        raise RuntimeError("AI_REVIEW_VIDEO_BINDING_CHANGED")
    capture = cv2.VideoCapture(str(video))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    if frames != 16:
        raise RuntimeError(f"VIDEO_DECODE_COUNT:{frames}")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_source_coverage_recovery.py",
         "tests/test_human_to_robot_product_first_cable.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_REGRESSION_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = OUT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation_path = OUT / "FINAL_VALIDATION.json"
    atomic_json(validation_path, {
        "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_CHECKED_STRUCTURE_NOT_PRODUCT_QUALITY",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": tests.returncode, "summary": tests.stdout.strip().splitlines()[-1]},
        "new_review_video": artifact_ref(video), "decoded_frames": frames,
        "scene_input_rebound": artifact_ref(EVIDENCE["scene_rebound"]),
        "scene_candidate_quality": review["quality"],
        "motion_projection": artifact_ref(EVIDENCE["motion_projection"]),
        "protected_root_snapshot_before_after": "NOT_ESTABLISHED_AT_T0",
        "claim_limit": "Validated new structured outputs, video and targeted regressions only; Clean and product quality remain rejected.",
    })
    finished = now_iso()
    blockers = {
        "scene": {
            "missing": "007 complete human/device removal: changed-input ProPainter still leaves hand-shaped residual, white line and ghost",
            "consumer": "accepted 007 Clean and full product", "owner": "scene",
            "unblock_action": "obtain a distinct, falsifiable removal method or independently supported background donors; do not rerun either rejected signature",
            "unaffected": ["old 378-frame candidate", "new traceable 16-frame input support", "other lanes"],
        },
        "motion_product": {
            "missing": "031 directly image-supported wrist target and left-hand motion; 102 model outputs have 0 projected wrist pixels in image",
            "consumer": "031 wrist IK quality and complete product", "owner": "motion_product",
            "unblock_action": "establish separate valid wrist-center authority or restrict product to evidence-supported local hand regions before new IK",
            "unaffected": ["old 149-frame timeline", "right-hand model predictions", "actual FK/renderer consistency", "007 product"],
        },
    }
    lanes = {
        "scene": {"evidence": [artifact_ref(EVIDENCE[key]) for key in ("scene_diagnostic", "scene_rebound", "scene_candidate", "scene_review", "gpu_call")],
                  "quality": "REJECTED_QUALITY_CLEAN", "improvement": "MISSING_LEFT_HAND_MODEL_SUPPORT_RECOVERED_NOT_INPAINT_QUALITY"},
        "motion_product": {"evidence": [artifact_ref(EVIDENCE[key]) for key in ("motion_diagnostic", "motion_projection")],
                           "quality": "TARGET_AUTHORITY_REJECTED_FOR_WRIST_IK", "improvement": "031_WRIST_PROJECTION_0_OF_102_IN_IMAGE_NO_NEW_IK"},
    }
    for name, details in lanes.items():
        atomic_json(OUT / "lanes" / name / "STATE.json", {
            "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_LANE_STATE_V1", "task_id": TASK,
            "lane": name, "status": "TERMINAL_WITH_QUALITY_GAPS", "execution": "EXECUTED",
            "structure": "PASS_FOR_EXECUTED_ARTIFACTS", "quality": details["quality"],
            "improvement": details["improvement"], "adoption": "NOT_ADOPTED",
            "evidence": details["evidence"], "blocker": blockers[name], "updated_at": finished,
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 11},
            "training_eligible": False, "control_ground_truth": False,
            "physical_deployable": False, "external_metric_authority": False,
        })
    prior = load_json(REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/RESULT.json")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_RECOVERY_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_WITH_QUALITY_GAPS", "task_terminal_status": "REJECTED_QUALITY",
        "terminal_at": finished, "event_driven_early_close": True,
        "early_close_reason": "Both frozen input/source branches have executed results or evidence-based quality rejection; no unchanged GPU retry, full Clean expansion or new IK is authorized in this packet.",
        "execution_summary": {
            "007_old_mask_left_region_absent_frames": source["frames_left_screen_model_mask_absent"],
            "007_rebound_mask_left_region_min_pixels": min(row["new_left_screen_support_pixels"] for row in rebound["rows"]),
            "007_new_changed_input_model_frames": 16,
            "007_new_clean_quality": review["quality"], "007_new_full_clean_frames": 0,
            "031_model_prediction_frames": 102,
            "031_projected_wrist_inside_image_frames": 0,
            "031_frame47_projected_joint_count_inside_image": projection["frame47_projected_joint_count_inside_image"],
            "031_new_ik_attempts": 0,
            "products_structure": prior["execution_summary"]["products_structure"],
            "products_quality": prior["execution_summary"]["products_quality"],
            "products_adopted": prior["execution_summary"]["products_adopted"],
            "prior_cleanup_logical_deleted_bytes": prior["execution_summary"]["cleanup_logical_deleted_bytes"],
            "new_deletions": 0,
        },
        "evidence": {key: artifact_ref(path) for key, path in EVIDENCE.items()},
        "lane_states": {name: artifact_ref(OUT / "lanes" / name / "STATE.json") for name in lanes},
        "targeted_tests": artifact_ref(test_path),
        "final_validation": artifact_ref(validation_path),
        "review_video": artifact_ref(video), "review_video_decoded_frames": frames,
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_EXECUTED_ARTIFACTS",
                         "quality": "REJECTED_QUALITY", "improvement": "INPUT_SUPPORT_AND_TARGET_AUTHORITY_DIAGNOSTIC_ONLY",
                         "adoption": "NOT_ADOPTED"},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "007 input mask support improved, but Clean still fails. 031 model-produced wrist is offscreen in all 102 frames and is not externally observed wrist truth. No product, Contact, training, control or deployment promotion.",
    }
    atomic_json(OUT / "RESULT.json", result)
    result_ref = artifact_ref(OUT / "RESULT.json")
    task.update(status="REJECTED_QUALITY", phase="SOURCE_COVERAGE_TERMINAL", attempt=1,
                pid=None, proc_start_ticks=None, gpu_id=None, heartbeat_at=None,
                updated_at=finished, result=result_ref,
                last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="INPUT_REPAIRED_BUT_CLEAN_REJECTED_AND_031_WRIST_AUTHORITY_UNSUPPORTED")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "REJECTED_QUALITY", "created_at": finished,
        "result": result_ref, "message": "007 rebound mask/model executed but Clean rejected; 031 wrist projection 0/102 in-image; product quality remains 0/4.",
    }])[-100:]
    current_status = load_json(STATUS)
    current_status.update({
        "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_TERMINAL_STATUS_V1",
        "latest_task": TASK, "result": result_ref, "result_status": result["status"],
        "next_task": None, "active_tasks": [], "current_index_status": "PASS_NO_ACTIVE_TASKS",
        "parent_task_id": "human_to_robot_product_first_cleanup_20260923", "parent_status": "REJECTED_QUALITY",
        "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"],
        "final_validation": artifact_ref(validation_path),
        "counts": {**current_status["counts"], **result["execution_summary"]},
        "generated_at": finished, "governance_revision": args.expected_revision + 1,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    atomic_json(STATUS, current_status)
    empty_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_RECOVERY_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_RECOVERY_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "Bounded 007/031 successor terminal rejected-quality; next algorithm scope requires another finite packet.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_SOURCE_COVERAGE_TERMINAL",
        expected_revision=args.expected_revision, task_packet_index_path=INDEX,
        task_packet_index_value=empty_index, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "REJECTED_QUALITY", "revision": published["governance_revision"],
                      "result": result_ref, "review_video": str(video)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
