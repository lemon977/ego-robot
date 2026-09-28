"""Publish evidence-backed product-first progress; never promote rejected quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_product_first_cleanup_20260923"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
EVIDENCE = {
    "cable_prep": ROOT / "lanes/scene/cable_007/window_v1/RESULT.json",
    "cable_model": ROOT / "lanes/scene/cable_007/clean_window_v1/RESULT.json",
    "cable_review": ROOT / "lanes/scene/cable_007/quality_review_v1/RESULT.json",
    "patch": ROOT / "lanes/scene/object_patch_031/window_v1/RESULT.json",
    "product_007_window": ROOT / "lanes/motion_product/product_007/window_v1/RESULT.json",
    "frame47": ROOT / "lanes/motion_product/frame47_audit_031/v1/RESULT.json",
    "left_rgb": ROOT / "lanes/motion_product/left_rgb_review_031/v1/RESULT.json",
    "robot_correspondence": ROOT / "lanes/motion_product/robot_correspondence_031/window_v1/RESULT.json",
    "adapter_collision": ROOT / "lanes/motion_product/adapter_collision/window_v1/RESULT.json",
    "sensor_review": ROOT / "lanes/sensor/review_v1/RESULT.json",
    "compare": ROOT / "lanes/compare/common_old_contract_v1/RESULT.json",
    "cleanup_batch1": ROOT / "cleanup/batch1/DELETE_RECEIPT.json",
}


def lane(name: str, evidence: list[str], quality: str, improvement: str,
         blocker: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_LANE_STATE_V1",
        "task_id": TASK, "lane": name, "status": "READY",
        "execution": "EXECUTED_PARTIAL", "structure": "PASS_FOR_EXECUTED_ARTIFACTS",
        "quality": quality, "improvement": improvement, "adoption": "NOT_ADOPTED",
        "evidence": [artifact_ref(EVIDENCE[key]) for key in evidence],
        "blocker": blocker, "updated_at": now_iso(),
        "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 10},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    for path in EVIDENCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {
        "scene": lane("scene", ["cable_prep", "cable_model", "cable_review", "patch"],
            "REJECTED_QUALITY_FULL_CLEAN", "007_CABLE_LOCAL_WINDOW_IMPROVED_16_OF_16_NOT_FULL_CLEAN",
            {"missing": "full human/device removal and full-session cable identity",
             "consumer": "007 full Clean and adopted product", "owner": "scene",
             "unblock_action": "repair independently proven mask support; current fixed-window candidate must not be scaled unchanged",
             "unaffected": ["031 patch sampler", "old 007/031 candidates", "Robot R0"]}),
        "motion_product": lane("motion_product", ["product_007_window", "frame47", "left_rgb", "robot_correspondence", "adapter_collision"],
            "REJECTED_QUALITY_FORMAL_PRODUCTS", "007_NEW_16_FRAME_PRODUCT_CANDIDATE_AND_031_FRAME47_TARGET_OUTLIER_LOCALIZED",
            {"missing": "full Clean quality, 031 reliable target/left motion, approved adapter collision scope",
             "consumer": "four formal product adoption", "owner": "motion_product",
             "unblock_action": "keep current q/FK evidence, correct source only with independent proof; no new 031 IK in this budget",
             "unaffected": ["actual FK renderer", "Robot correspondence diagnostics", "007 fixed-window candidate"]}),
        "sensor": lane("sensor", ["sensor_review"], "INCONCLUSIVE_VISUAL_ALIGNMENT_USER_REVIEW_PENDING",
            "466_FRAME_REUSED_ARRAYS_AND_FULL_VIDEOS_INDEPENDENTLY_REVIEWED",
            {"missing": "independent alignment truth and user visual acceptance", "consumer": "sensor visual adoption",
             "owner": "sensor", "unblock_action": "review existing three full videos against frozen RGB frames; do not refit 101",
             "unaffected": ["native arrays", "saved q/FK", "kinematic-only claims"]}),
        "compare": lane("compare", ["compare"], "INCONCLUSIVE_NO_EXTERNAL_TRUTH",
            "OLD_COMMON_TARGET_FULL_METRICS_REFRESHED_NO_NEW_SOLVER_RUN",
            {"missing": "independent truth and accepted same-session Clean", "consumer": "method winner and formal visual comparison",
             "owner": "compare", "unblock_action": "retain limited old-common-contract result, no winner claim",
             "unaffected": ["old solver arrays", "per-session numeric diagnostics"]}),
    }
    for name, value in rows.items():
        atomic_json(ROOT / "lanes" / name / "STATE.json", value)
    cleanup = load_json(EVIDENCE["cleanup_batch1"])
    atomic_json(ROOT / "cleanup/DELETE_RECEIPT.json", {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_DELETE_SUMMARY_V1",
        "task_id": TASK, "batch1": artifact_ref(EVIDENCE["cleanup_batch1"]),
        "batch2": "NOT_YET_FROZEN_OR_PURGED", "logical_deleted_bytes": cleanup.get("logical_deleted_bytes"),
        "physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
        "claim_limit": "Only batch1 actually purged; isolated bytes are not counted as reclaimed.",
    })
    progress = ROOT / "PROGRESS_H1.json"
    if not progress.exists():
        atomic_json(progress, {
            "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_PROGRESS_V1", "task_id": TASK,
            "created_at": now_iso(), "checkpoint": "FIRST_EVIDENCE_WAVE_NOT_TERMINAL",
            "counts": {"formal_structure_pass": 4, "formal_quality_pass": 0,
                       "formal_adopted": 0, "007_new_clean_window_frames": 16,
                       "007_full_clean_quality_pass": False, "031_patch_sampled": 24,
                       "031_robot_correspondence_valid": 2207,
                       "contact_r1_executed": 0, "sensor_reused_frames_reviewed": 466,
                       "cleanup_batch1_logical_deleted_bytes": cleanup.get("logical_deleted_bytes")},
            "evidence": {key: artifact_ref(path) for key, path in EVIDENCE.items()},
            "lane_states": {name: artifact_ref(ROOT / "lanes" / name / "STATE.json") for name in rows},
            "authority": {"training_eligible": False, "control_ground_truth": False,
                          "physical_deployable": False, "external_metric_authority": False},
        })
    task = next(row for row in state["tasks"] if row.get("task_id") == TASK)
    task.update(status="PENDING", attempt=1, updated_at=now_iso(), heartbeat_at=None,
                pid=None, proc_start_ticks=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": now_iso(),
        "message": "007 actual 16-frame Clean/Product improved locally but rejected overall; three geometry consumers ran; batch1 obsolete payloads purged.",
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_FIRST_EVIDENCE_PUBLISHED",
        expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "PASS", "revision": published["governance_revision"],
                      "progress": str(progress)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
