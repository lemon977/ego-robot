"""Publish the bounded shared-hand/Robot/Clean delivery result.

This finalizer is deliberately fail-closed.  It preserves the predecessor's
four-product denominator and does not promote a module-level improvement to a
product, training-label, control, or deployment claim.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

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


TASK = "human_to_robot_shared_hand_delivery_20260924"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
LANES = ATTEMPT / "lanes"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_RECEIPT = REPO_ROOT / "tasks/receipts/HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_20260924_RESULT.json"
NAVIGATION = REPO_ROOT / "docs/current/SHARED_HAND_DELIVERY_RESULT_ZH.md"
BASELINE_RESULT = (
    REPO_ROOT
    / "_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/RESULT.json"
)


def _check_ref(ref: dict[str, object]) -> None:
    actual = artifact_ref(Path(str(ref["path"])))
    if actual["sha256"] != ref.get("sha256") or (
        "bytes" in ref and actual["bytes"] != ref.get("bytes")
    ):
        raise RuntimeError(f"ARTIFACT_REF_MISMATCH:{ref.get('path')}")


def _decode_video(path: Path, expected: int) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"VIDEO_MISSING:{path}")
    probe = subprocess.run(
        [
            "/usr/bin/ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of",
            "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    frames = int(probe.stdout.strip())
    if frames != expected:
        raise RuntimeError(f"VIDEO_FRAME_COUNT:{path}:{frames}!={expected}")
    decoded = subprocess.run(
        ["/usr/bin/ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if decoded.returncode != 0 or decoded.stderr.strip():
        raise RuntimeError(f"VIDEO_DECODE_FAILED:{path}:{decoded.stderr[-1000:]}")
    return {"artifact": artifact_ref(path), "decoded_frames": frames, "status": "PASS"}


def _require_terminal_lane(state: dict[str, object], lane: str) -> None:
    if state.get("task_id") != TASK:
        raise RuntimeError(f"{lane.upper()}_TASK_ID")
    if state.get("next_action") not in {
        "Publisher consumes this terminal lane result; do not run the full stage.",
        "HAND_RECIPE_EXHAUSTED_REPORT_NUMERIC_FAILURE; DO_NOT_RUN_098_101_OR_POKER",
        "PUBLISH_TERMINAL_RESULT",
        "STOP_CLEAN_RECIPE_AND_DELIVER_REJECTION",
        "NONE_IN_THIS_FINITE_TASK",
        "NEXT_AUTHORIZED_CYCLE_ONLY_REMOVE_OUTPUT_TIMES_255; CHANGE_NOTHING_ELSE",
    }:
        raise RuntimeError(f"{lane.upper()}_NOT_TERMINAL:{state.get('next_action')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    if (ATTEMPT / "RESULT.json").exists() or TASK_RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_RESULT_ALREADY_EXISTS")

    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "RUNNING"}:
        raise RuntimeError("TASK_NOT_CURRENT_AND_LIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTED")
    index = load_json(INDEX)
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("PACKET_INDEX_NOT_UNIQUE")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") == "ACQUIRED" and str(lease.get("task_id", "")).startswith(TASK):
        raise RuntimeError("TASK_GPU_LEASE_STILL_ACQUIRED")

    robot_state = load_json(LANES / "robot/STATE.json")
    robot_result = load_json(LANES / "robot/RESULT.json")
    hand_state = load_json(LANES / "hand_data/STATE.json")
    hand_result = load_json(LANES / "hand_data/RESULT_REPAIR2.json")
    clean_state = load_json(LANES / "clean/STATE.json")
    _require_terminal_lane(robot_state, "robot")
    _require_terminal_lane(hand_state, "hand_data")
    _require_terminal_lane(clean_state, "clean")

    if robot_result.get("gates", {}).get("expansion_gate") is not False:
        raise RuntimeError("ROBOT_EXPANSION_GATE_CHANGED")
    if robot_result.get("gates", {}).get("full_171_executed") is not False:
        raise RuntimeError("ROBOT_FULL_UNEXPECTED")
    if hand_result.get("qualified_h50_windows") != 0:
        raise RuntimeError("HAND_H50_COUNT_CHANGED")
    if hand_result.get("label_quality_pass") is not False or hand_result.get("interface_pass") is not False:
        raise RuntimeError("HAND_QUALIFICATION_CHANGED")
    if hand_result.get("optimizer_steps") != 0:
        raise RuntimeError("OPTIMIZER_STEP_OCCURRED")
    if clean_state.get("quality") not in {
        "NOT_EVALUATED_IMPLEMENTATION_LIMIT",
        "NOT_EVALUATED_IMPLEMENTATION_INVALID",
        "REJECTED_IMPLEMENTATION",
        "REJECTED_QUALITY",
    }:
        raise RuntimeError(f"CLEAN_TERMINAL_QUALITY:{clean_state.get('quality')}")
    clean_ref = clean_state.get("result")
    if not isinstance(clean_ref, dict):
        raise RuntimeError("CLEAN_RESULT_REF_MISSING")
    _check_ref(clean_ref)
    clean_result = load_json(Path(str(clean_ref["path"])))
    if clean_result.get("full_171") not in {
        "NOT_RUN_FIXED_WINDOW_REJECTED",
        "NOT_RUN_IMPLEMENTATION_LIMIT",
        "PROHIBITED_FIXED_WINDOW_NOT_VALID",
    }:
        raise RuntimeError("CLEAN_FULL_SCOPE_CHANGED")

    for ref in (robot_result["window_result"], robot_result["window_candidate"],
                hand_result["sensor_097"]["states"], clean_ref):
        _check_ref(ref)

    baseline = load_json(BASELINE_RESULT)
    if baseline.get("counts", {}).get("products_quality") != "0/4":
        raise RuntimeError("BASELINE_PRODUCT_COUNT_CHANGED")
    robot_video_ref = baseline["robot"]["video"]
    _check_ref(robot_video_ref)
    robot_video = _decode_video(Path(str(robot_video_ref["path"])), 171)
    clean_review_ref = clean_result.get("visual_review")
    if not isinstance(clean_review_ref, dict):
        raise RuntimeError("CLEAN_VISUAL_REVIEW_REF_MISSING")
    _check_ref(clean_review_ref)
    clean_review = load_json(Path(str(clean_review_ref["path"])))
    clean_video_ref = clean_review.get("comparison_video")
    if not isinstance(clean_video_ref, dict):
        raise RuntimeError("CLEAN_REVIEW_VIDEO_REF_MISSING")
    _check_ref(clean_video_ref)
    clean_video = _decode_video(Path(str(clean_video_ref["path"])), 16)

    tests = ATTEMPT / "TARGETED_TESTS.xml"
    if not tests.is_file():
        raise RuntimeError("TARGETED_TESTS_MISSING")
    suites = list(ElementTree.parse(tests).getroot().iter("testsuite"))
    if not suites or sum(int(row.get("tests", "0")) for row in suites) < 20:
        raise RuntimeError("TARGETED_TEST_COUNT_TOO_SMALL")
    if any(int(row.get("failures", "0")) or int(row.get("errors", "0")) for row in suites):
        raise RuntimeError("TARGETED_TESTS_FAILED")
    if not NAVIGATION.is_file():
        raise RuntimeError("RESULT_NAVIGATION_MISSING")

    validation = {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_FINAL_VALIDATION_V1",
        "task_id": TASK,
        "status": "PASS_STRUCTURE_WITH_UNRESOLVED_QUALITY_AND_IMPLEMENTATION",
        "robot_reused_full_video": robot_video,
        "robot_new_window": artifact_ref(LANES / "robot/POKER_080_111_JOINT_TRAJECTORY_FIX1.json"),
        "clean_review_video": clean_video,
        "hand_data": {
            "sensor097_side_items": 330,
            "direction_pass_side_items": hand_result["sensor_097"]["direction_pass_side_frames"],
            "pinch_pass_side_items": hand_result["sensor_097"]["pinch_pass_side_frames"],
            "collision_pass_side_items": hand_result["sensor_097"]["collision_pass_side_frames"],
            "qualified_h50_windows": 0,
        },
        "targeted_tests": artifact_ref(tests),
        "navigation": artifact_ref(NAVIGATION),
    }
    atomic_json(ATTEMPT / "FINAL_VALIDATION.json", validation)
    now = now_iso()
    result = {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_RESULT_V1",
        "task_id": TASK,
        "route": ROUTE,
        "status": "REJECTED_QUALITY",
        "terminal_at": now,
        "counts": {
            "products_structure": "4/4",
            "products_quality": "0/4",
            "products_adopted": "0/4",
            "qualified_kai22_h50_windows": 0,
        },
        "robot": {
            "execution": "NEW_FIXED_WINDOW_EXECUTED; PRIOR_FULL_171_REUSED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY",
            "improvement": "NOT_ESTABLISHED; THREE_OF_FOUR_LOCAL_BLOCKS_FEASIBLE_BUT_NOT_SPLICED",
            "full_171_new": False,
            "window_result": robot_result["window_result"],
            "reused_full_video": robot_video["artifact"],
        },
        "clean": {
            "execution": "PROPANTER_TRACE_AND_REAL_LAMA_V2_V3_WINDOWS_EXECUTED",
            "structure": "PASS",
            "quality": clean_state["quality"],
            "method_quality": "NOT_EVALUATED_ADAPTER_OUTPUT_SCALE_INCORRECT",
            "full_171_new": False,
            "result": clean_ref,
            "review_video": clean_video["artifact"],
        },
        "hand_data": {
            "execution": "SENSOR097_165_FRAME_330_SIDE_ITEMS_EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY",
            "improvement": "DIRECTION_P50_118.58_TO_13.48_DEG; LEFT_DIRECTION_165_OF_165_PASS",
            "label_quality_pass": False,
            "interface_pass": False,
            "qualified_h50_windows": 0,
            "result": artifact_ref(LANES / "hand_data/RESULT_REPAIR2.json"),
        },
        "final_validation": artifact_ref(ATTEMPT / "FINAL_VALIDATION.json"),
        "navigation": artifact_ref(NAVIGATION),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": "Offline bounded module evidence. No product, training-label, user-adoption, control, deployment, or external metric promotion.",
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(TASK_RECEIPT, {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_TASK_RECEIPT_V1",
        "task_id": TASK,
        "status": "REJECTED_QUALITY",
        "result": result_ref,
    })
    atomic_json(LANES / "delivery/STATE.json", {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_DELIVERY_LANE_STATE_V1",
        "task_id": TASK,
        "lane": "delivery",
        "status": "TERMINAL",
        "execution": "PUBLISHED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY",
        "improvement": "REAL_INCREMENT_PRESERVED_WITHOUT_PROMOTION",
        "consumer_qualification": "NO_PRODUCT_OR_TRAINING_PROMOTION",
        "adoption": "NOT_ADOPTED",
        "next_action": "NONE_IN_THIS_FINITE_TASK",
        "dependencies": [robot_result["window_result"], clean_ref,
                         artifact_ref(LANES / "hand_data/RESULT_REPAIR2.json")],
        "evidence": [result_ref],
        "blocker": None,
        "updated_at": now,
    })

    task.update(
        status="REJECTED_QUALITY",
        phase="HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_TERMINAL",
        attempt=1,
        updated_at=now,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        result=result_ref,
        last_attempt_terminal="REJECTED_QUALITY",
        last_attempt_reason="ROBOT_WINDOW_AND_LOCAL_KAI22_QUALITY_FAILED; CLEAN_METHOD_NOT_EVALUATED_AFTER_ADAPTER_REPAIR_LIMIT",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "REJECTED_QUALITY",
        "created_at": now,
        "message": "Bounded Robot, shared-hand, and Clean work terminated without product or label promotion.",
        "result": result_ref,
    }])[-100:]
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_20260924",
        "execution_revision": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_20260924",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "Finite task terminal; no automatic successor or quality promotion.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_TERMINAL",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor_index,
    )
    atomic_json(REPO_ROOT / "docs/current/STATUS.json", {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_CURRENT_STATUS_V1",
        "generated_at": now_iso(),
        "governance_revision": published["governance_revision"],
        "status": "PASS_NO_ACTIVE_TASKS",
        "active_tasks": [],
        "latest_task": TASK,
        "latest_terminal_status": "REJECTED_QUALITY",
        "result": result_ref,
        "counts": result["counts"],
        "robot_qualified_scope": "NO_NEW_FULL_SCOPE; 3_OF_4_WINDOW_BLOCKS_FEASIBLE_NOT_ADOPTED",
        "clean_qualified_scope": "NONE; METHOD_QUALITY_NOT_EVALUATED_AFTER_ADAPTER_LIMIT",
        "data_label_quality_pass": False,
        "data_interface_pass": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    })
    print(json.dumps({
        "status": "REJECTED_QUALITY",
        "governance_revision": published["governance_revision"],
        "result": result_ref,
        "products_quality": "0/4",
        "products_adopted": "0/4",
        "qualified_h50_windows": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
