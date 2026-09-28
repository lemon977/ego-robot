"""Publish the bounded Robot/Clean/Data result cycle without promoting quality."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
)

TASK = "human_to_robot_result_breakthrough_20260924"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_RECEIPT = REPO_ROOT / "tasks/receipts/HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_20260924_RESULT.json"
ROBOT = ATTEMPT / "lanes/robot"
CLEAN = ATTEMPT / "lanes/clean/POKER_076_091_INPUT_REPAIR"
DATA = ATTEMPT / "lanes/data"
CLEANUP = ATTEMPT / "lanes/cleanup/batch1"
DOCS = REPO_ROOT / "docs/current"


def _decode_frames(path: Path, expected: int) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"VIDEO_MISSING:{path}")
    probe = subprocess.run(
        ["/usr/bin/ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    count = int(probe.stdout.strip())
    if count != expected:
        raise RuntimeError(f"FRAME_COUNT_MISMATCH:{path}:{count}!={expected}")
    decode = subprocess.run(
        ["/usr/bin/ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    if decode.returncode != 0 or decode.stderr.strip():
        raise RuntimeError(f"FULL_DECODE_FAILED:{path}:{decode.stderr[-1000:]}")
    return {"artifact": artifact_ref(path), "decoded_frames": count, "decode_status": "PASS"}


def _check_ref(value: dict[str, object]) -> None:
    path = Path(str(value["path"]))
    actual = artifact_ref(path)
    if actual["sha256"] != value["sha256"] or actual["bytes"] != value["bytes"]:
        raise RuntimeError(f"SHA_OR_BYTES_MISMATCH:{path}")


def _lane(lane: str, quality: str, result: Path, blocker: dict | None, now: str) -> dict:
    return {
        "schema_version": "HUMAN_TO_ROBOT_RESULT_LANE_STATE_V1",
        "task_id": TASK,
        "lane": lane,
        "status": "TERMINAL",
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": quality,
        "adoption": "NOT_ADOPTED",
        "result": artifact_ref(result),
        "consumer": "OFFLINE_VISUAL_REVIEW" if lane != "data" else "EXISTING_FLOW_MATCHING_Q_ONLY",
        "next_action": "NONE_IN_THIS_FINITE_CYCLE",
        "blocker": blocker,
        "updated_at": now,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-revision", required=True, type=int)
    args = ap.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((x for x in state["tasks"] if x.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "RUNNING"}:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("NOT_CURRENT_TASK")
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0]["task_id"] != TASK:
        raise RuntimeError("PACKET_INDEX_NOT_UNIQUE")
    if (ATTEMPT / "RESULT.json").exists() or TASK_RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_RESULT_ALREADY_EXISTS")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") == "ACQUIRED" and str(lease.get("task_id", "")).startswith(TASK):
        raise RuntimeError("GPU_STILL_ACQUIRED")

    robot = load_json(ROBOT / "POKER_171_INDEPENDENT_FK_V3.json")
    robot_review = load_json(ROBOT / "POKER_171_FIXED_THIRD_PERSON_RESULT_V3.json")
    clean = load_json(CLEAN / "VISUAL_REVIEW.json")
    data = load_json(DATA / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json")
    cleanup = load_json(CLEANUP / "DELETE_RECEIPT.json")
    for row in (robot["candidate"], robot["independent_fk"], robot_review["video"],
                clean["candidate"], clean["video"], data["hand_motion"],
                data["backend"], data["source_hdf5"], cleanup["consumer_check"]):
        _check_ref(row)
    if (robot["frames"], robot["new_tolerance_pass"], robot["old_tolerance_pass"]) != (171, 339, 323):
        raise RuntimeError("ROBOT_NUMERIC_BINDING_CHANGED")
    if clean["quality"] != "REJECTED_QUALITY" or clean["full_171"] != "NOT_RUN_FIXED_WINDOW_FAILED":
        raise RuntimeError("CLEAN_QUALITY_CHANGED")
    if not data["interface_pass"] or data["label_quality_pass"] or data["qualified_h50_windows"] != 0:
        raise RuntimeError("DATA_QUALITY_CHANGED")
    if cleanup["execution"] != "ACTUALLY_PURGED" or len(cleanup["targets"]) != 19:
        raise RuntimeError("CLEANUP_NOT_ACTUALLY_PURGED")
    if cleanup["isolation_remaining_bytes"] != 0:
        raise RuntimeError("ISOLATION_NOT_EMPTY")
    with np.load(ROBOT / "POKER_171_CONSTRAINED_V3.npz", allow_pickle=False) as arr:
        if arr["q_arm"].shape != (171, 2, 7) or arr["q22"].shape != (171, 2, 22):
            raise RuntimeError("ROBOT_ARRAY_SHAPE_CHANGED")
        if not np.isfinite(arr["q_arm"]).all() or not np.isfinite(arr["q22"]).all():
            raise RuntimeError("ROBOT_ARRAY_NONFINITE")
        if not np.array_equal(arr["frame_id"], np.arange(171)):
            raise RuntimeError("ROBOT_FRAME_MAP_CHANGED")

    videos = {
        "robot_full": _decode_frames(ROBOT / "POKER_171_FIXED_THIRD_PERSON_CANDIDATE_V3.mp4", 171),
        "clean_window": _decode_frames(CLEAN / "POKER_076_091_OLD_NEW_CLEAN_REVIEW.mp4", 16),
    }
    for name in ("AI_WORK_ENTRY_ZH.md", "PLAN.md", "README_ZH.md", "RESULT_BREAKTHROUGH_VIDEO_INDEX_ZH.md"):
        if not (DOCS / name).is_file():
            raise RuntimeError(f"CURRENT_DOC_MISSING:{name}")
    archive = REPO_ROOT / "docs/archive/HUMAN_TO_ROBOT_CURRENT_TEXT_PRE_RESULT_20260924.zip"
    subprocess.run(["unzip", "-tq", str(archive)], capture_output=True, check=True, timeout=60)
    tests = ATTEMPT / "TARGETED_TESTS.xml"
    if not tests.is_file():
        raise RuntimeError("TARGETED_REGRESSION_RECEIPT_MISSING")
    test_root = ElementTree.parse(tests).getroot()
    suites = list(test_root.iter("testsuite"))
    if not suites or sum(int(x.get("tests", "0")) for x in suites) < 30:
        raise RuntimeError("TARGETED_REGRESSION_TOO_SMALL")
    if any(int(x.get("failures", "0")) or int(x.get("errors", "0")) for x in suites):
        raise RuntimeError("TARGETED_REGRESSION_FAILED")
    validation = {
        "schema_version": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_FINAL_VALIDATION_V1",
        "task_id": TASK,
        "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
        "videos": videos,
        "robot_array": artifact_ref(ROBOT / "POKER_171_CONSTRAINED_V3.npz"),
        "robot_q_fk_parity_m": robot["independent_fk_max_abs_m"],
        "robot_hard_gate": {"pass": 339, "denominator": 342, "status": "FAIL"},
        "clean": {"window_decoded": 16, "full_session_new": False, "quality": "REJECTED_QUALITY"},
        "sensor097": {"interface_pass": True, "label_quality_pass": False, "qualified_h50_windows": 0},
        "cleanup": {"actual_purged_objects": 19, "logical_deleted_bytes": cleanup["logical_deleted_bytes"],
                    "isolation_remaining_bytes": 0, "physical_reclaim": "UNKNOWN_SHARED_CPFS"},
        "historical_receipts": "UNCHANGED; HASH_CHECK_IN_CLEANUP_CONSUMER_RECEIPT",
        "targeted_regression": artifact_ref(tests),
        "archive": artifact_ref(archive),
        "navigation": artifact_ref(DOCS / "RESULT_BREAKTHROUGH_VIDEO_INDEX_ZH.md"),
    }
    atomic_json(ATTEMPT / "FINAL_VALIDATION.json", validation)
    now = now_iso()
    lane_specs = {
        "robot": ("REJECTED_QUALITY", ROBOT / "POKER_171_INDEPENDENT_FK_V3.json", {
            "code": "HAND_ROOT_GATE_3_OF_342_FAIL_AND_FINGER_JUMP_UNRESOLVED",
            "consumer": "POKER_FULL_PRODUCT", "owner": "Motion/Product",
            "resolution": "METHOD_LEVEL_RETARGET_OR_INPUT_REVIEW; NO_MORE_SAME_RECIPE_RETRIES",}),
        "clean": ("REJECTED_QUALITY", CLEAN / "VISUAL_REVIEW.json", {
            "code": "FIXED_WINDOW_HAND_RESIDUE_AND_CARD_INPAINT_INVALID",
            "consumer": "POKER_CLEAN_FULL_AND_PRODUCT", "owner": "Scene/Clean",
            "resolution": "METHOD_LEVEL_REMOVAL_OR_BETTER_GROUNDED_SUPPORT; NO_FULL_EXPANSION",}),
        "data": ("REJECTED_QUALITY", DATA / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json", {
            "code": "KAI22_LOCAL_LABEL_SHAPE_NOT_QUALIFIED",
            "consumer": "FLOW_MATCHING_KAI22_SUPERVISION", "owner": "Sensor/Data",
            "resolution": "INDEPENDENT_FINGER_LABEL_AUTHORITY_OR_CALIBRATION",}),
        "cleanup": ("PASS_SCOPE", CLEANUP / "DELETE_RECEIPT.json", None),
    }
    for lane, (quality, path, blocker) in lane_specs.items():
        atomic_json(ATTEMPT / "lanes" / lane / "STATE.json", _lane(lane, quality, path, blocker, now))
    result = {
        "schema_version": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_RESULT_V1",
        "task_id": TASK, "route": ROUTE, "cycle": 1,
        "status": "REJECTED_QUALITY", "terminal_at": now,
        "second_cycle": "NOT_STARTED_FIRST_CYCLE_NO_QUALIFIED_RECIPE",
        "counts": {"products_structure": "4/4", "products_quality": "0/4",
                   "products_adopted": "0/4", "qualified_label_windows": 0},
        "robot": {"execution": "FULL_171_EXECUTED", "structure": "PASS",
                  "quality": "REJECTED_QUALITY", "improvement": "POSE_PASS_323_TO_339_OF_342_AND_MAX_RIGHT_ARM_STEP_1.968_TO_1.041_RAD; P99_AND_FINGER_NOT_CLEARED",
                  "adoption": "NOT_ADOPTED", "numeric": artifact_ref(ROBOT / "POKER_171_INDEPENDENT_FK_V3.json"),
                  "video": videos["robot_full"]["artifact"]},
        "clean": {"execution": "REAL_MODEL_WINDOW_16_EXECUTED", "structure": "PASS",
                  "quality": "REJECTED_QUALITY", "full_171": "NOT_RUN_FIXED_WINDOW_FAILED",
                  "adoption": "NOT_ADOPTED", "review": artifact_ref(CLEAN / "VISUAL_REVIEW.json"),
                  "video": videos["clean_window"]["artifact"]},
        "data": {"execution": "REAL_RAW_TO_SAVED_TO_KAI22_AUDIT; PRIOR_REAL_CONSUMER_REUSED",
                 "interface_pass": True, "label_quality_pass": False,
                 "qualified_h50_windows": 0, "optimizer_steps": 0,
                 "receipt": artifact_ref(DATA / "SENSOR_097_LOCAL_KAI22_QUALIFICATION.json")},
        "cleanup": {"execution": "ACTUALLY_PURGED", "objects": 19,
                    "logical_deleted_bytes": cleanup["logical_deleted_bytes"],
                    "physical_reclaim": "UNKNOWN_SHARED_CPFS",
                    "receipt": artifact_ref(CLEANUP / "DELETE_RECEIPT.json")},
        "final_validation": artifact_ref(ATTEMPT / "FINAL_VALIDATION.json"),
        "video_navigation": artifact_ref(DOCS / "RESULT_BREAKTHROUGH_VIDEO_INDEX_ZH.md"),
        "code_binding": {name: artifact_ref(REPO_ROOT / path) for name, path in {
            "robot_producer": "src/chaoyang/ops/run_human_to_robot_result_robot.py",
            "robot_evaluator": "src/chaoyang/ops/evaluate_human_to_robot_result_robot.py",
            "clean_producer": "src/chaoyang/ops/run_human_to_robot_result_clean.py",
            "data_checker": "src/chaoyang/ops/run_human_to_robot_result_data.py",
            "cleanup": "src/chaoyang/ops/run_human_to_robot_result_cleanup.py",
        }.items()},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "Offline failed-quality evidence, not product adoption or qualified training labels. A second identical pipeline cycle is not authorized after all first-cycle quality candidates failed.",
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(TASK_RECEIPT, {"schema_version": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_TASK_RECEIPT_V1",
                               "task_id": TASK, "status": "REJECTED_QUALITY", "result": result_ref})
    task.update(status="REJECTED_QUALITY", phase="HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_TERMINAL",
                attempt=1, updated_at=now, heartbeat_at=None, pid=None, proc_start_ticks=None,
                gpu_id=None, result=result_ref, last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="ROBOT_CLEAN_AND_KAI22_LABEL_QUALITY_ALL_UNMET")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "REJECTED_QUALITY", "created_at": now,
        "message": "171-frame Robot candidate, 16-frame rejected Clean, 0 qualified label windows, and 19 actual cleanup purges.",
        "result": result_ref,
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1_TERMINAL",
                 "plan_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
                 "execution_revision": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CYCLE1",
                 "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [],
                 "claim_limit": "First result cycle terminated with quality gaps; method-level reassessment required, no automatic second cycle."}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_TERMINAL",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    atomic_json(DOCS / "STATUS.json", {
        "schema_version": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CURRENT_STATUS_V1",
        "generated_at": now_iso(), "governance_revision": published["governance_revision"],
        "status": "PASS_NO_ACTIVE_TASKS", "active_tasks": [], "latest_task": TASK,
        "latest_terminal_status": "REJECTED_QUALITY", "result": result_ref,
        "counts": result["counts"], "robot_full": "171_FRAME_CANDIDATE_REJECTED_QUALITY",
        "clean_full": "NOT_RUN_FIXED_WINDOW_REJECTED_QUALITY",
        "data_interface_pass": True, "data_label_quality_pass": False,
        "second_cycle": result["second_cycle"],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    print(json.dumps({"status": "REJECTED_QUALITY", "revision": published["governance_revision"],
                      "result": result_ref, "robot_frames": 171, "clean_window_frames": 16,
                      "qualified_label_windows": 0, "purged_objects": 19}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
