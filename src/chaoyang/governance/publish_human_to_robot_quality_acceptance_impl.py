"""Publish the bounded quality task's implementation into the live algorithm contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
    validate_artifact_ref,
)

TASK = "human_to_robot_quality_acceptance_20260924"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def _terminal(state: dict, revision: int) -> dict:
    """Close only after immutable failures and actual payloads have been checked."""
    if (ATTEMPT / "RESULT.json").exists() or (ATTEMPT / "FINAL_VALIDATION.json").exists():
        raise FileExistsError("QUALITY_TASK_TERMINAL_ALREADY_EXISTS")
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING"}:
        raise RuntimeError("QUALITY_TASK_NOT_ACTIVE")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("QUALITY_TASK_NOT_SOLE_PACKET")
    evidence = {
        "poker_quality": ATTEMPT / "lanes/motion_product/POKER_171_TECHNICAL_QUALITY.json",
        "poker_seed_suffix": ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.json",
        "poker_finger": ATTEMPT / "lanes/motion_product/POKER_104_105_FINGER_FIRST_LAYER.json",
        "chips_seed": ATTEMPT / "lanes/motion_product/CHIPS_SEED_COUNTERFACTUAL_0_160.json",
        "scene_review": ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/VISUAL_REVIEW.json",
        "stereo_g0": ATTEMPT / "lanes/scene/POKER_0902_STEREO_G0.json",
        "sensor_quality": ATTEMPT / "lanes/sensor/SENSOR_LOCAL_KAI22_QUALITY_AUDIT.json",
        "sensor_bones": ATTEMPT / "lanes/sensor/SENSOR_097_NATIVE_BONE_SEMANTICS.json",
        "sensor_forward": ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_REAL_RGB_FORWARD.json",
        "compare": REPO_ROOT / "_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json",
    }
    for path in evidence.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    quality = load_json(evidence["poker_quality"])
    scene = load_json(evidence["scene_review"])
    sensor = load_json(evidence["sensor_bones"])
    if quality["decision"]["quality"] != "REJECTED_QUALITY" or scene["quality"] != "REJECTED_QUALITY":
        raise RuntimeError("REJECTED_PRODUCT_WAS_PROMOTED")
    if sensor["quality"] != "INTERNAL_GEOMETRY_MISMATCH_LOCAL_LABEL_NOT_ADMITTED":
        raise RuntimeError("SENSOR_LABEL_WAS_PROMOTED")
    videos = {
        "poker_old_formal": (REPO_ROOT / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/robot.mp4", 171),
        "poker_clean_rebase": (ATTEMPT / "lanes/scene/POKER_HISTORICAL_CLEAN_EXACT_REBASE/CLEAN_EXACT_REBASE_CANDIDATE.mp4", 171),
        "poker_new_clean_review": (ATTEMPT / "lanes/scene/POKER_076_091_CARD_PROTECTED_CLEAN/POKER_076_091_RAW_OLD_SUPPORT_NEW_REVIEW.mp4", 16),
    }
    decoded = {}
    for name, (path, expected) in videos.items():
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1", str(path)],
                               capture_output=True, text=True, check=True)
        if probe.stdout.strip() != f"nb_read_frames={expected}":
            raise RuntimeError(f"VIDEO_FRAME_COUNT_INVALID:{name}:{probe.stdout.strip()}")
        decode = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "2",
                                 "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                                capture_output=True, text=True, check=False)
        if decode.returncode or decode.stderr.strip():
            raise RuntimeError(f"VIDEO_DECODE_FAILED:{name}:{decode.stderr[-500:]}")
        decoded[name] = {"video": artifact_ref(path), "decoded_frames": expected}
    arrays = {
        "poker_suffix": ATTEMPT / "lanes/motion_product/POKER_SEED_RECOVERY_SUFFIX_V1.npz",
        "sensor_qonly": ATTEMPT / "lanes/sensor/SENSOR_097_QONLY_H50_DIAGNOSTIC.npz",
    }
    reloaded = {}
    for name, path in arrays.items():
        with np.load(path, allow_pickle=False) as content:
            if name == "poker_suffix" and content["q_arm"].shape != (171, 2, 7):
                raise RuntimeError("POKER_SUFFIX_RELOAD_SHAPE")
            if name == "sensor_qonly" and content["action_future"].shape != (50, 62):
                raise RuntimeError("SENSOR_QONLY_RELOAD_SHAPE")
            reloaded[name] = {"array": artifact_ref(path), "keys": list(content.files)}
    tests = subprocess.run(["/usr/local/bin/python", "-B", "-m", "pytest", "-q",
                            "tests/unit/test_product_scene_gate_quality.py",
                            "tests/unit/test_full_robot_review_v2.py"],
                           cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if tests.returncode:
        raise RuntimeError(f"FINAL_TARGETED_TESTS_FAILED:{tests.stdout[-500:]}:{tests.stderr[-500:]}")
    validation = {"schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_FINAL_VALIDATION_V1",
                  "task_id": TASK, "status": "PASS_FOR_CHECKED_STRUCTURE_ONLY",
                  "videos": decoded, "arrays": reloaded,
                  "evidence": {key: artifact_ref(path) for key, path in evidence.items()},
                  "targeted_tests": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
                  "claim_limit": "Full decode, reload and target tests do not grant product quality, label quality or user adoption."}
    atomic_json(ATTEMPT / "FINAL_VALIDATION.json", validation)
    blockers = {
        "scene": {"missing": "accepted 171-frame Clean and same-session qualified occlusion; new protected 16-frame Clean still smears held card and leaves wristband",
                  "consumer": "Poker full technical product", "owner": "Scene",
                  "unblock_action": "obtain new independent support/reconstruction evidence; do not rerun the rejected frozen candidate",
                  "unaffected": ["saved 171-frame Clean diagnostic", "Sensor", "old Local/HuRo comparison"]},
        "motion_product": {"missing": "19 right-arm old tracking passes and 104-105 pinky continuity; seed-recovery candidate loses five pass frames",
                           "consumer": "Poker full technical product", "owner": "Motion/Product",
                           "unblock_action": "validate a distinct producer-level finger/arm correction against all frozen gates; do not bind rejected suffix",
                           "unaffected": ["old full q/FK and fixed-third-person playback", "Scene Clean review"]},
        "sensor": {"missing": "Kai22 local shape correspondence; native-to-robot bone scale varies by segment",
                   "consumer": "training label admission", "owner": "Sensor",
                   "unblock_action": "validate corrected 25-to-21/local retarget semantics and independent local shape quality",
                   "unaffected": ["native MANUS", "kinematic q/FK", "diagnostic forward with zero optimizer steps"]},
        "compare": {"missing": "qualified common method output for winner claim",
                    "consumer": "Local/HuRo comparative adoption", "owner": "Local/HuRo",
                    "unblock_action": "none in this task; preserve limited comparison and no new HuRo solver run",
                    "unaffected": ["old same-target metrics and video"]},
    }
    for lane in ("scene", "motion_product", "sensor", "compare"):
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        value = load_json(path)
        value.update(status="TERMINAL_WITH_QUALITY_GAPS", adoption="NOT_ADOPTED",
                     blocker=blockers[lane], next_action="FINITE_AUTHORIZED_ACTIONS_EXHAUSTED_OR_REJECTED",
                     updated_at=now_iso())
        atomic_json(path, value)
    terminal_at = now_iso()
    result = {"schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_RESULT_V1", "task_id": TASK,
              "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "REJECTED_QUALITY",
              "release_status": "INCOMPLETE", "terminal_at": terminal_at,
              "event_driven_early_close": True,
              "early_close_reason": "Poker full product retains independently rejected Motion, Scene, occlusion and visual gates; authorized one-shot fixes were tested and rejected. Remaining full Stereo batch cannot change current product pass; no same-signature retry, new IK, HuRo solve or model change is authorized.",
              "counts": {"products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
                         "sensor_training_admitted_labels": 0, "optimizer_steps": 0},
              "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_CHECKED_ARTIFACTS",
                               "quality": "REJECTED_QUALITY", "improvement": "LOCAL_SEED_STEP_ONLY_WITH_TRACKING_REGRESSION",
                               "adoption": "NOT_ADOPTED"},
              "evidence": {key: artifact_ref(path) for key, path in evidence.items()},
              "lane_states": {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json")
                              for lane in blockers},
              "final_validation": artifact_ref(ATTEMPT / "FINAL_VALIDATION.json"),
              "visual_index": artifact_ref(REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE/INDEX_ZH.md"),
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False,
              "claim_limit": "0/4 product quality and adoption; actual ProPainter and Flow Matching forward are diagnostic, not product or training success."}
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    receipt_path = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    atomic_json(receipt_path, {"schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_TASK_RECEIPT_V1",
                               "task_id": TASK, "status": "REJECTED_QUALITY", "result": result_ref,
                               "products_quality": "0/4", "products_adopted": "0/4"})
    task.update(status="REJECTED_QUALITY", phase="HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_TERMINAL", attempt=1,
                pid=None, proc_start_ticks=None, gpu_id=None, heartbeat_at=None,
                updated_at=terminal_at, result=result_ref,
                last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="FINITE_WORK_EXECUTED_WITH_PRODUCT_QUALITY_GAPS")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": TASK, "attempt": 1,
        "status": "REJECTED_QUALITY", "created_at": terminal_at, "result": result_ref,
        "message": "Finite quality task ended with 0/4 product quality, 0/4 adoption; new Poker Clean and motion candidates rejected; Sensor label not admitted."}])[-100:]
    successor_index = {"schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_20260924",
        "execution_revision": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_20260924",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "This finite task is terminal rejected-quality; no product or training admission was granted."}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_TERMINAL",
        expected_revision=revision, task_packet_index_path=INDEX,
        task_packet_index_value=successor_index, generator_path=Path(__file__))
    return {"status": "REJECTED_QUALITY", "governance_revision": published["governance_revision"],
            "result": result_ref}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if args.terminal:
        print(json.dumps(_terminal(state, args.expected_revision), ensure_ascii=False))
        return 0
    receipt = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_IMPLEMENTATION_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PUBLISHED", "governance_revision": receipt["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
