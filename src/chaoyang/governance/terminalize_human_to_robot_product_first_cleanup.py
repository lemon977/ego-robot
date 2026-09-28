"""Seal the finite product-first run without promoting rejected product quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
    validate_artifact_ref,
)

TASK = "human_to_robot_product_first_cleanup_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT_OUT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"
EVIDENCE = {
    "007_prep": ROOT / "lanes/scene/cable_007/window_v1/RESULT.json",
    "007_clean": ROOT / "lanes/scene/cable_007/clean_window_v1/RESULT.json",
    "007_review": ROOT / "lanes/scene/cable_007/quality_review_v1/RESULT.json",
    "007_product_window": ROOT / "lanes/motion_product/product_007/window_v1/RESULT.json",
    "007_formal": ROOT / "lanes/motion_product/formal_007_current/attempt_0001/PRODUCT_RESULT.json",
    "007_resume": ROOT / "lanes/motion_product/formal_007_current/RESUME_RECEIPT.json",
    "031_card": ROOT / "lanes/scene/card_031/review_v1/RESULT.json",
    "031_frame47": ROOT / "lanes/motion_product/frame47_audit_031/v1/RESULT.json",
    "031_left_rgb": ROOT / "lanes/motion_product/left_rgb_review_031/v1/RESULT.json",
    "031_patch": ROOT / "lanes/scene/object_patch_031/window_v1/RESULT.json",
    "031_robot_correspondence": ROOT / "lanes/motion_product/robot_correspondence_031/window_v1/RESULT.json",
    "adapter_collision": ROOT / "lanes/motion_product/adapter_collision/window_v1/RESULT.json",
    "sensor": ROOT / "lanes/sensor/review_v1/RESULT.json",
    "compare": ROOT / "lanes/compare/common_old_contract_v1/RESULT.json",
    "cleanup": ROOT / "cleanup/DELETE_SUMMARY_B2.json",
    "large_inventory": ROOT / "cleanup/LARGE_OBJECT_INVENTORY.json",
    "post_cleanup": ROOT / "validation_after_cleanup_batch2/RESULT.json",
}


def _lane(name: str, quality: str, improvement: str, keys: tuple[str, ...], blocker: dict) -> dict:
    return {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_LANE_STATE_V1",
        "task_id": TASK, "lane": name, "status": "TERMINAL_WITH_QUALITY_GAPS",
        "execution": "EXECUTED", "structure": "PASS_FOR_EXECUTED_ARTIFACTS",
        "quality": quality, "improvement": improvement, "adoption": "NOT_ADOPTED",
        "evidence": [artifact_ref(EVIDENCE[key]) for key in keys],
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
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_SOLE_INDEX_ENTRY")
    for path in EVIDENCE.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if (ROOT / "RESULT.json").exists() or (ROOT / "FINAL_VALIDATION.json").exists() or RECEIPT_OUT.exists():
        raise FileExistsError("IMMUTABLE_TERMINAL_ALREADY_EXISTS")
    review = load_json(EVIDENCE["007_review"])
    formal = load_json(EVIDENCE["007_formal"])
    patch = load_json(EVIDENCE["031_patch"])
    correspondence = load_json(EVIDENCE["031_robot_correspondence"])
    cleanup = load_json(EVIDENCE["cleanup"])
    inventory = load_json(EVIDENCE["large_inventory"])
    post = load_json(EVIDENCE["post_cleanup"])
    if review["quality"] != "REJECTED_QUALITY_FULL_CLEAN" or formal["quality"] == "PASS":
        raise RuntimeError("007_QUALITY_UNEXPECTED_OR_PROMOTED")
    if cleanup["execution"] != "THREE_AUDITED_BATCHES_ACTUALLY_PURGED" or cleanup["logical_deleted_bytes"] != 5219025177:
        raise RuntimeError("CLEANUP_SUMMARY_MISMATCH")
    if post["status"] != "PASS_FOR_CHECKED_SCOPE" or len(post["slots"]) != 15:
        raise RuntimeError("POST_CLEANUP_VALIDATION_INVALID")
    if len(inventory["top20_subdirectories"]) != 20:
        raise RuntimeError("LARGE_OBJECT_INVENTORY_INCOMPLETE")
    for row in post["slots"]:
        if validate_artifact_ref(row["video"]):
            raise RuntimeError(f"VIDEO_REF_CHANGED:{row['slot_id']}")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q",
         "tests/test_human_to_robot_product_first_cable.py",
         "tests/test_human_to_robot_product_first_cleanup_batch1.py",
         "tests/test_object_patch_sample_v1.py",
         "tests/test_robot_rigid_correspondence_v1.py",
         "tests/test_human_to_robot_terminal_status.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_REGRESSION_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_log = ROOT / "FINAL_TARGETED_TESTS.log"
    if test_log.exists():
        raise FileExistsError(test_log)
    test_log.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_CHECKED_SCOPE",
        "video_count": len(post["slots"]), "post_cleanup": artifact_ref(EVIDENCE["post_cleanup"]),
        "targeted_tests": artifact_ref(test_log),
        "pytest": {"exit_code": tests.returncode, "summary": tests.stdout.strip().splitlines()[-1]},
        "hardlink_transaction_tests": "14_NOT_EVALUATED_ENV_CPFS_TMP_SEMANTICS",
        "protected_root_snapshot_before_after": "NOT_ESTABLISHED_AT_T0",
        "scope": "Current 15 video slots, retained forensic samples, environments, governed status and targeted code regressions; not product quality or a complete raw/processed/sealed pre/post Merkle proof.",
    }
    atomic_json(ROOT / "FINAL_VALIDATION.json", validation)
    blockers = {
        "scene": {"missing": "full hand/other-device removal and reliable all-session cable identity; 031 card edge remains blurred",
                  "consumer": "accepted Clean and product", "owner": "scene",
                  "unblock_action": "obtain independent full-region support and a falsifiable inpainting fix in a separately authorized task",
                  "unaffected": ["retained old Clean", "007 local cable diagnostic", "031 model-conditioned patch"]},
        "motion_product": {"missing": "031 reliable wrist target/left-hand motion, qualified adapter collision scope, accepted Clean",
                           "consumer": "031 complete motion and four product adoptions", "owner": "motion_product",
                           "unblock_action": "repair independent source/target evidence before new IK; do not rerun this task's rejected candidate",
                           "unaffected": ["actual q/FK rendering", "007 fixed-window candidate", "rigid correspondence diagnostic"]},
        "sensor": {"missing": "independent alignment truth and user visual acceptance",
                   "consumer": "sensor visual adoption", "owner": "sensor",
                   "unblock_action": "review retained three full-session videos against original RGB; 098/101 have no new labels",
                   "unaffected": ["native MANUS arrays", "saved robot q/FK", "kinematic diagnostics"]},
        "compare": {"missing": "accepted common background and external truth",
                    "consumer": "Local/HuRo winner claim", "owner": "compare",
                    "unblock_action": "freeze a new common target only with separate authorization; keep old-contract numeric comparison limited",
                    "unaffected": ["old common target metrics", "both retained solver arrays"]},
    }
    lanes = {
        "scene": _lane("scene", "REJECTED_QUALITY_FULL_CLEAN", "007_COMPLAINED_CABLE_LOCAL_PROXY_REDUCED_16_OF_16",
                       ("007_prep", "007_clean", "007_review", "031_card", "031_patch"), blockers["scene"]),
        "motion_product": _lane("motion_product", "REJECTED_QUALITY_PRODUCTS", "031_FRAME47_TARGET_OUTLIER_AND_CURRENT_007_SAME_SHA_ENTRY_VERIFIED",
                                ("007_product_window", "007_formal", "007_resume", "031_frame47", "031_left_rgb", "031_robot_correspondence", "adapter_collision"), blockers["motion_product"]),
        "sensor": _lane("sensor", "INCONCLUSIVE_VISUAL_ALIGNMENT_USER_REVIEW_PENDING", "THREE_REUSED_FULL_REVIEWS_466_FRAMES_RECHECKED",
                        ("sensor",), blockers["sensor"]),
        "compare": _lane("compare", "INCONCLUSIVE_NO_EXTERNAL_TRUTH", "OLD_COMMON_TARGET_METRICS_RECOMPUTED_NO_NEW_SOLVER",
                         ("compare",), blockers["compare"]),
    }
    for name, value in lanes.items():
        atomic_json(ROOT / "lanes" / name / "STATE.json", value)
    terminal_at = now_iso()
    result = {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_WITH_QUALITY_GAPS", "task_terminal_status": "REJECTED_QUALITY",
        "release_status": "INCOMPLETE", "terminal_at": terminal_at,
        "event_driven_early_close": True,
        "early_close_reason": "All frozen finite branches produced an executable result or evidence-based rejection; no same-signature quality retry, new IK, unsupported full Clean expansion, or unproved large-tree deletion is authorized.",
        "execution_summary": {
            "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
            "007_new_propaint_fixed_window_frames": 16,
            "007_complaint_cable_local_proxy_reduced_frames": review["bright_proxy_reduced_frames"],
            "007_full_clean_quality": review["quality"],
            "007_formal_current_video": "378_FRAMES_SAME_SHA_AS_OLD_NO_VISUAL_IMPROVEMENT",
            "031_new_ik_attempts": 0, "031_frame47_root_cause": "FROZEN_TARGET_JUMP_NOT_FK_RENDER_MISMATCH",
            "031_same_object_patch_valid": f"{patch['counts']['geometry_screened']}/{patch['counts']['qualification_checked']}",
            "031_robot_correspondence": "2207/2294_SAME_LINK_DEPTH_CONSISTENT",
            "adapter_collision": "CONTROLLED_AND_REAL_QUERIES_NO_FULL_MACHINE_PASS",
            "contact_strict_admissible": 0, "robot_r1_executed": 0,
            "sensor_frames_reused_reviewed": 466, "local_huro_winner": None,
            "video_slots_reloaded_and_decoded": 15,
            "cleanup_exact_targets_purged": 8,
            "cleanup_logical_deleted_bytes": cleanup["logical_deleted_bytes"],
            "cleanup_physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
            "remaining_root_allocated_bytes_at_inventory": inventory["root_allocated_bytes_at_inventory"],
        },
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_EXECUTED_ARTIFACTS",
                         "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE", "improvement": "LOCAL_CABLE_AND_GEOMETRY_DIAGNOSTIC_ONLY",
                         "adoption": "NOT_ADOPTED"},
        "reason_codes": ["007_FULL_CLEAN_RESIDUAL_HAND_AND_SECOND_LINE", "031_FROZEN_TARGET_OUTLIER_AND_LEFT_INPUT_MISSING",
                         "031_CARD_EDGE_QUALITY_REJECTED", "ADAPTER_COLLISION_SCOPE_INCOMPLETE",
                         "CONTACT_STRICT_AND_ROBOT_R1_NOT_ADMITTED", "SENSOR_USER_VISUAL_REVIEW_PENDING",
                         "LOCAL_HURO_NO_EXTERNAL_TRUTH", "LARGE_ENV_WORKTREES_RETAINED_NO_REPLACEMENT_PROOF"],
        "evidence": {name: artifact_ref(path) for name, path in EVIDENCE.items()},
        "lane_states": {name: artifact_ref(ROOT / "lanes" / name / "STATE.json") for name in lanes},
        "final_validation": artifact_ref(ROOT / "FINAL_VALIDATION.json"),
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "Finite offline product-first and audited cleanup. Current product quality is 0/4, adoption 0/4; local cable reduction, internal geometry consistency, model-conditioned patch and logical deletion do not grant Contact, external metric, training, control or deployment authority.",
    }
    atomic_json(ROOT / "RESULT.json", result)
    result_ref = artifact_ref(ROOT / "RESULT.json")
    atomic_json(RECEIPT_OUT, {"schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_RECEIPT_V1",
        "task_id": TASK, "status": "REJECTED_QUALITY", "result": result_ref,
        "products_structure": "4/4", "products_quality": "0/4", "products_adopted": "0/4",
        "cleanup_logical_deleted_bytes": cleanup["logical_deleted_bytes"], "video_slots": "15/15_CHECKED"})
    task.update(status="REJECTED_QUALITY", phase="PRODUCT_FIRST_TERMINAL", attempt=1,
                pid=None, proc_start_ticks=None, gpu_id=None, heartbeat_at=None,
                updated_at=terminal_at, result=result_ref,
                last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="FINITE_WORK_EXECUTED_WITH_PRODUCT_QUALITY_GAPS")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "REJECTED_QUALITY",
        "created_at": terminal_at, "result": result_ref,
        "message": "Product-first finite task terminal: 007 local cable improvement, three consumers, three audited purges, 0/4 product quality.",
    }])[-100:]
    successor_index = {"schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_PRODUCT_FIRST_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "Product-first task is terminal rejected-quality; new algorithm or cleanup scope needs a newly authorized finite task."}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_CLEANUP_TERMINAL",
        expected_revision=args.expected_revision, task_packet_index_path=INDEX,
        task_packet_index_value=successor_index, generator_path=Path(__file__))
    print(json.dumps({"status": "REJECTED_QUALITY", "revision": published["governance_revision"],
                      "result": result_ref, "cleanup_logical_deleted_bytes": cleanup["logical_deleted_bytes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
