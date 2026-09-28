"""Publish the latest explicitly supported Human-to-Robot terminal receipt.

The historical public entry point is retained, but S1 supersedes R2. No
directory mtime, filename ordering, or document text selects an authority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import (
    REPO_ROOT, artifact_ref, atomic_json, load_json, validate_artifact_ref,
)

TASK = "human_to_robot_root_cause_gated_r2_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUTPUT = REPO_ROOT / "docs/current/STATUS.json"
S1_TASK = "human_to_robot_quality_closure_s1_20260923"
S2_TASK = "human_to_robot_evidence_unlock_s2_20260923"
PRODUCT_FIRST_TASK = "human_to_robot_product_first_cleanup_20260923"
SOURCE_COVERAGE_TASK = "human_to_robot_source_coverage_recovery_20260923"
DEVICE_DONOR_TASK = "human_to_robot_007_device_donor_probe_20260923"
WEARABLE_TASK = "human_to_robot_007_wearable_sam_canary_20260923"
SEED_REFINE_TASK = "human_to_robot_007_wearable_seed_refine_20260923"
TABLE_DONOR_TASK = "human_to_robot_007_table_donor_expansion_20260923"
ATTACH_CLEAN_TASK = "human_to_robot_007_attachment_clean_canary_20260923"
MULTIDONOR_TABLE_TASK = "human_to_robot_007_multidonor_table_canary_20260923"
SENSOR_DISPLAY_TASK = "human_to_robot_sensor_display_correction_20260923"
SPATIAL_DONOR_TASK = "human_to_robot_007_spatial_donor_check_20260923"
CROSS_EYE_TASK = "human_to_robot_007_cross_eye_source_probe_20260923"
REGION_DTYPE_TASK = "human_to_robot_031_observability_dtype_fix_20260923"
SURFACE_DEPTH_TASK = "human_to_robot_031_surface_depth_diagnostic_20260923"
SURFACE_TEMPORAL_TASK = "human_to_robot_031_surface_depth_temporal_audit_20260923"
TEN_HOUR_DELIVERY_TASK = "human_to_robot_10h_delivery_20260924"
FOUR_LANE_INCREMENT_TASK = "human_to_robot_four_lane_quality_increment_20260924"
REPRESENTATIVE_BASELINE_TASK = "human_to_robot_representative_baseline_20260924"
QUALITY_ACCEPTANCE_TASK = "human_to_robot_quality_acceptance_20260924"
RESULT_BREAKTHROUGH_TASK = "human_to_robot_result_breakthrough_20260924"
SHARED_HAND_DELIVERY_TASK = "human_to_robot_shared_hand_delivery_20260924"
TASK_PRIORITY = (S2_TASK, S1_TASK, TASK)
TERMINAL_STATUSES = {"REJECTED_QUALITY", "FAILED_QUALITY_C"}
ACTIVE_STATUSES = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
SYNC_VALIDATION = (
    REPO_ROOT / "_run/current/human_to_robot_status_handoff_20260923/"
    "attempts/attempt_0001/REGRESSION.json"
)


def select_task(task_state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = {row.get("task_id"): row for row in task_state.get("tasks", [])}
    return next((rows[name] for name in TASK_PRIORITY if name in rows), None)


def build_active_s2_status(
    task_state: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    rows = {row.get("task_id"): row for row in task_state.get("tasks", [])}
    task = rows.get(S2_TASK)
    if task is None or task.get("status") not in ACTIVE_STATUSES:
        raise RuntimeError("S2_IS_NOT_ACTIVE")
    attempt = REPO_ROOT / f"_run/current/{S2_TASK}/attempts/attempt_0001"
    checkpoint_name = next(
        (
            name
            for name in ("H9", "H6", "H3")
            if (attempt / f"checkpoints/{name}_RESULT.json").is_file()
        ),
        None,
    )
    if checkpoint_name is None:
        raise RuntimeError("S2_ACTIVE_CHECKPOINT_MISSING")
    checkpoint = attempt / f"checkpoints/{checkpoint_name}_RESULT.json"
    active = [
        str(row["task_id"])
        for row in task_state.get("tasks", [])
        if row.get("status") in ACTIVE_STATUSES
    ]
    return {
        "schema_version": "HUMAN_TO_ROBOT_S2_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "current_index_status": "PASS",
        "status": f"ACTIVE_{checkpoint_name}_CHECKPOINT",
        "active_tasks": active,
        "next_task": task_state.get("next_task"),
        "parent_task_id": S2_TASK,
        "latest_task": S2_TASK,
        "parent_status": task.get("status"),
        "checkpoint": artifact_ref(checkpoint),
        "counts": {
            "formal_product_candidates_complete": 4,
            "formal_product_quality_pass": 0,
            "formal_product_adopted": 0,
            "sensor_sessions_reverified": 3,
            "local_huro_adapter_refresh_sessions": 2,
            "robot_r1_screened": 0,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
        },
        "quality_axes": {
            "execution": "EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY_MIXED_INCONCLUSIVE",
            "adoption": "NOT_ADOPTED",
        },
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Active S2 checkpoint projection only. Four structural candidates, zero "
            "quality passes and zero adoptions; Contact/Robot R1 screening was not completed, "
            "and no Contact, control or deployment authority was obtained."
        ),
    }


def build_active_product_first_status(
    task_state: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == PRODUCT_FIRST_TASK), None)
    if task is None or task.get("status") not in ACTIVE_STATUSES:
        raise RuntimeError("PRODUCT_FIRST_IS_NOT_ACTIVE")
    attempt = REPO_ROOT / f"_run/current/{PRODUCT_FIRST_TASK}/attempts/attempt_0001"
    progress = next((attempt / name for name in ("PROGRESS_CLEANUP_B2.json", "PROGRESS_CLEANUP_B1B.json", "PROGRESS_H2.json", "PROGRESS_H1.json")
                     if (attempt / name).is_file()), attempt / "PROGRESS_H1.json")
    if not progress.is_file():
        raise RuntimeError("PRODUCT_FIRST_PROGRESS_MISSING_NO_OLD_STATUS_FALLBACK")
    evidence = load_json(progress)
    if evidence.get("task_id") != PRODUCT_FIRST_TASK:
        raise RuntimeError("PRODUCT_FIRST_PROGRESS_TASK_MISMATCH")
    return {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "current_index_status": "PASS", "status": "ACTIVE_EVIDENCE_WAVE",
        "active_tasks": [row["task_id"] for row in task_state.get("tasks", [])
                         if row.get("status") in ACTIVE_STATUSES],
        "next_task": task_state.get("next_task"),
        "parent_task_id": PRODUCT_FIRST_TASK, "latest_task": PRODUCT_FIRST_TASK,
        "parent_status": task.get("status"), "progress": artifact_ref(progress),
        "counts": evidence["counts"],
        "quality_axes": {"execution": "EXECUTED_PARTIAL", "structure": "PASS_FOR_EXECUTED_ARTIFACTS",
                         "quality": "REJECTED_QUALITY_FULL_CLEAN_AND_PRODUCTS", "adoption": "NOT_ADOPTED"},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": "Current product-first partial progress only. 007 cable improved locally but full Clean rejected; four products quality 0/4 and adoption 0/4. Geometry diagnostics and cleanup do not promote Contact or product authority.",
    }


def build_terminal_product_first_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == PRODUCT_FIRST_TASK), None)
    if task is None or task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("PRODUCT_FIRST_IS_NOT_TERMINAL")
    attempt = REPO_ROOT / f"_run/current/{PRODUCT_FIRST_TASK}/attempts/attempt_0001"
    result_path = attempt / "RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("PRODUCT_FIRST_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("PRODUCT_FIRST_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != PRODUCT_FIRST_TASK or result.get("status") != "TERMINAL_WITH_QUALITY_GAPS":
        raise RuntimeError("PRODUCT_FIRST_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("PRODUCT_FIRST_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_CHECKED_SCOPE":
        raise RuntimeError("PRODUCT_FIRST_FINAL_VALIDATION_NOT_PASSED")
    return {
        "schema_version": "HUMAN_TO_ROBOT_PRODUCT_FIRST_TERMINAL_STATUS_V1",
        "generated_at": generated_at, "governance_revision": task_state.get("governance_revision"),
        "current_index_status": "PASS_NO_ACTIVE_TASKS", "active_tasks": [],
        "next_task": task_state.get("next_task"), "parent_task_id": PRODUCT_FIRST_TASK,
        "latest_task": PRODUCT_FIRST_TASK, "parent_status": task["status"],
        "result_status": result["status"], "result": artifact_ref(result_path),
        "final_validation": validation, "counts": result["execution_summary"],
        "quality_axes": result["quality_axes"],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
        "claim_limit": result["claim_limit"],
    }


def build_terminal_source_coverage_status(
    task_state: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == SOURCE_COVERAGE_TASK), None)
    if task is None or task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("SOURCE_COVERAGE_IS_NOT_TERMINAL")
    result_path = REPO_ROOT / f"_run/current/{SOURCE_COVERAGE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("SOURCE_COVERAGE_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("SOURCE_COVERAGE_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != SOURCE_COVERAGE_TASK or result.get("status") != "TERMINAL_WITH_QUALITY_GAPS":
        raise RuntimeError("SOURCE_COVERAGE_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("SOURCE_COVERAGE_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_CHECKED_STRUCTURE_NOT_PRODUCT_QUALITY":
        raise RuntimeError("SOURCE_COVERAGE_FINAL_VALIDATION_NOT_PASSED")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    return {
        "schema_version": "HUMAN_TO_ROBOT_SOURCE_COVERAGE_TERMINAL_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
        "active_tasks": active,
        "next_task": task_state.get("next_task"),
        "parent_task_id": SOURCE_COVERAGE_TASK,
        "latest_task": SOURCE_COVERAGE_TASK,
        "parent_status": task["status"],
        "result_status": result["status"],
        "result": artifact_ref(result_path),
        "final_validation": validation,
        "counts": result["execution_summary"],
        "quality_axes": result["quality_axes"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": result["claim_limit"],
    }


def build_device_donor_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == DEVICE_DONOR_TASK), None)
    if task is None:
        raise RuntimeError("DEVICE_DONOR_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active,
        "next_task": task_state.get("next_task"),
        "parent_task_id": DEVICE_DONOR_TASK,
        "latest_task": DEVICE_DONOR_TASK,
        "parent_status": task["status"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_PROBE",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "New CPU-only 007 device/donor probe registered; prior products remain quality 0/4 and adopted 0/4."}
    if task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("DEVICE_DONOR_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{DEVICE_DONOR_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("DEVICE_DONOR_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("DEVICE_DONOR_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != DEVICE_DONOR_TASK or result.get("status") != "TERMINAL_WITH_QUALITY_GAPS":
        raise RuntimeError("DEVICE_DONOR_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("DEVICE_DONOR_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_PROBE_STRUCTURE_NOT_PRODUCT_QUALITY":
        raise RuntimeError("DEVICE_DONOR_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_wearable_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == WEARABLE_TASK), None)
    if task is None:
        raise RuntimeError("WEARABLE_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active,
        "next_task": task_state.get("next_task"),
        "parent_task_id": WEARABLE_TASK,
        "latest_task": WEARABLE_TASK,
        "parent_status": task["status"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CANARY",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Pinned SAM one-frame wearable canary registered; prior products quality/adoption remain 0/4."}
    if task.get("status") not in {"PASSED", *TERMINAL_STATUSES}:
        raise RuntimeError("WEARABLE_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{WEARABLE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("WEARABLE_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("WEARABLE_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != WEARABLE_TASK or result.get("status") != "TERMINAL_SINGLE_FRAME":
        raise RuntimeError("WEARABLE_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("WEARABLE_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_SINGLE_FRAME_STRUCTURE_NOT_CLEAN_QUALITY":
        raise RuntimeError("WEARABLE_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_seed_refine_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == SEED_REFINE_TASK), None)
    if task is None:
        raise RuntimeError("SEED_REFINE_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SEED_REFINE_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": SEED_REFINE_TASK, "latest_task": SEED_REFINE_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CANARY",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "One-frame seed/refine task registered; prior products quality/adoption remain 0/4."}
    if task.get("status") not in {"PASSED", *TERMINAL_STATUSES}:
        raise RuntimeError("SEED_REFINE_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{SEED_REFINE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("SEED_REFINE_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("SEED_REFINE_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != SEED_REFINE_TASK or result.get("status") != "TERMINAL_SINGLE_FRAME":
        raise RuntimeError("SEED_REFINE_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("SEED_REFINE_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_SINGLE_FRAME_STRUCTURE_NOT_CLEAN_QUALITY":
        raise RuntimeError("SEED_REFINE_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_table_donor_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == TABLE_DONOR_TASK), None)
    if task is None:
        raise RuntimeError("TABLE_DONOR_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_TABLE_DONOR_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": TABLE_DONOR_TASK, "latest_task": TABLE_DONOR_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CPU_PROBE",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Fixed-grid table-only donor probe registered; product quality/adoption remain 0/4."}
    if task.get("status") not in {"PASSED", *TERMINAL_STATUSES}:
        raise RuntimeError("TABLE_DONOR_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{TABLE_DONOR_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("TABLE_DONOR_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("TABLE_DONOR_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != TABLE_DONOR_TASK or result.get("status") != "TERMINAL_FIXED_GRID":
        raise RuntimeError("TABLE_DONOR_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("TABLE_DONOR_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_FIXED_GRID_STRUCTURE_NOT_CLEAN_QUALITY":
        raise RuntimeError("TABLE_DONOR_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_attachment_clean_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == ATTACH_CLEAN_TASK), None)
    if task is None:
        raise RuntimeError("ATTACH_CLEAN_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_ATTACHMENT_CLEAN_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": ATTACH_CLEAN_TASK, "latest_task": ATTACH_CLEAN_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CANARY",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Changed-input 007 Clean canary registered; prior product quality/adoption remain 0/4."}
    if task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("ATTACH_CLEAN_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{ATTACH_CLEAN_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("ATTACH_CLEAN_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("ATTACH_CLEAN_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != ATTACH_CLEAN_TASK or result.get("status") != "TERMINAL_16_FRAME":
        raise RuntimeError("ATTACH_CLEAN_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("ATTACH_CLEAN_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_EXECUTION_STRUCTURE_NOT_CLEAN_QUALITY":
        raise RuntimeError("ATTACH_CLEAN_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_multidonor_table_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == MULTIDONOR_TABLE_TASK), None)
    if task is None:
        raise RuntimeError("MULTIDONOR_TABLE_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_MULTIDONOR_TABLE_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": MULTIDONOR_TABLE_TASK, "latest_task": MULTIDONOR_TABLE_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CANARY",
                "status": "REGISTERED_NOT_YET_QUALITY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Table-only 007 canary registered; prior product quality/adoption remain 0/4."}
    if task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("MULTIDONOR_TABLE_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{MULTIDONOR_TABLE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("MULTIDONOR_TABLE_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("MULTIDONOR_TABLE_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != MULTIDONOR_TABLE_TASK or result.get("status") != "TERMINAL_16_FRAME":
        raise RuntimeError("MULTIDONOR_TABLE_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("MULTIDONOR_TABLE_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_EXECUTION_STRUCTURE_NOT_CLEAN_QUALITY":
        raise RuntimeError("MULTIDONOR_TABLE_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_sensor_display_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == SENSOR_DISPLAY_TASK), None)
    if task is None:
        raise RuntimeError("SENSOR_DISPLAY_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": SENSOR_DISPLAY_TASK, "latest_task": SENSOR_DISPLAY_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task.get("status") in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_SENSOR_DISPLAY",
                "status": "REGISTERED_NOT_YET_VISUALLY_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "display_quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Sensor re-render registered; no new alignment or product quality claim."}
    if task.get("status") not in {"PASSED", *TERMINAL_STATUSES}:
        raise RuntimeError("SENSOR_DISPLAY_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{SENSOR_DISPLAY_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("SENSOR_DISPLAY_TERMINAL_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("SENSOR_DISPLAY_TERMINAL_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != SENSOR_DISPLAY_TASK or result.get("status") != "TERMINAL_466_FRAME":
        raise RuntimeError("SENSOR_DISPLAY_TERMINAL_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("SENSOR_DISPLAY_FINAL_VALIDATION_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_DISPLAY_STRUCTURE_NOT_SENSOR_ALIGNMENT":
        raise RuntimeError("SENSOR_DISPLAY_FINAL_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation, "counts": result["execution_summary"],
            "quality_axes": result["quality_axes"], "claim_limit": result["claim_limit"]}


def build_spatial_donor_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == SPATIAL_DONOR_TASK), None)
    if task is None:
        raise RuntimeError("SPATIAL_DONOR_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": SPATIAL_DONOR_TASK, "latest_task": SPATIAL_DONOR_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task["status"] in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_SPATIAL_DONOR",
                "status": "REGISTERED_NOT_YET_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Twelve donor pairs registered; no spatial or Clean quality conclusion yet."}
    if task["status"] != "PASSED":
        raise RuntimeError("SPATIAL_DONOR_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{SPATIAL_DONOR_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("SPATIAL_DONOR_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("SPATIAL_DONOR_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != SPATIAL_DONOR_TASK or result.get("status") != "TERMINAL_12_PAIRS":
        raise RuntimeError("SPATIAL_DONOR_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("SPATIAL_DONOR_VALIDATION_REF_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_SPATIAL_INTERPOLATION_DIAGNOSTIC_ONLY":
        raise RuntimeError("SPATIAL_DONOR_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation,
            "counts": {"pairs_checked": result["pairs_checked"],
                       "products_structure": result["products_structure"],
                       "products_quality": result["products_quality"],
                       "products_adopted": result["products_adopted"]},
            "quality_axes": {"execution": "EXECUTED", "structure": "PASS",
                             "source_interpolation": "DIAGNOSTIC_ONLY",
                             "clean_quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED"},
            "claim_limit": result["claim_limit"]}


def build_cross_eye_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == CROSS_EYE_TASK), None)
    if task is None:
        raise RuntimeError("CROSS_EYE_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_007_CROSS_EYE_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": CROSS_EYE_TASK, "latest_task": CROSS_EYE_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task["status"] in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_CROSS_EYE",
                "status": "REGISTERED_NOT_YET_EVALUATED",
                "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "Four cross-eye source frames registered; no Clean or product claim."}
    if task["status"] != "PASSED":
        raise RuntimeError("CROSS_EYE_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{CROSS_EYE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("CROSS_EYE_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("CROSS_EYE_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != CROSS_EYE_TASK or result.get("status") != "TERMINAL_4_FRAMES":
        raise RuntimeError("CROSS_EYE_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("CROSS_EYE_VALIDATION_REF_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_CROSS_EYE_SOURCE_DIAGNOSTIC_ONLY":
        raise RuntimeError("CROSS_EYE_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation,
            "counts": {"frames_checked": result["frames_checked"],
                       "supported_pixels": result["supported_pixels"],
                       "products_structure": result["products_structure"],
                       "products_quality": result["products_quality"],
                       "products_adopted": result["products_adopted"]},
            "quality_axes": {"execution": "EXECUTED", "structure": "PASS",
                             "source_support": "DIAGNOSTIC_ONLY",
                             "clean_quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED"},
            "claim_limit": result["claim_limit"]}


def build_region_dtype_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = next((row for row in task_state.get("tasks", [])
                 if row.get("task_id") == REGION_DTYPE_TASK), None)
    if task is None:
        raise RuntimeError("REGION_DTYPE_TASK_MISSING")
    active = [str(row["task_id"]) for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    base = {
        "schema_version": "HUMAN_TO_ROBOT_031_REGION_DTYPE_CURRENT_STATUS_V1",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "active_tasks": active, "next_task": task_state.get("next_task"),
        "parent_task_id": REGION_DTYPE_TASK, "latest_task": REGION_DTYPE_TASK,
        "parent_status": task["status"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    if task["status"] in ACTIVE_STATUSES:
        return {**base, "current_index_status": "ACTIVE_FINITE_031_REGION_DTYPE",
                "status": "REGISTERED_NOT_YET_EVALUATED", "task_packet": task.get("task_packet"),
                "counts": {"products_structure": "4/4", "products_quality": "0/4",
                           "products_adopted": "0/4"},
                "quality_axes": {"execution": "PENDING", "structure": "NOT_EVALUATED",
                                 "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
                "claim_limit": "031 uint16 region correction registered; no model or product conclusion yet."}
    if task["status"] != "PASSED":
        raise RuntimeError("REGION_DTYPE_TASK_STATUS_UNSUPPORTED")
    result_path = REPO_ROOT / f"_run/current/{REGION_DTYPE_TASK}/attempts/attempt_0001/RESULT.json"
    if not task.get("result") or validate_artifact_ref(task["result"]):
        raise RuntimeError("REGION_DTYPE_RESULT_REF_INVALID")
    if Path(task["result"]["path"]).resolve() != result_path.resolve():
        raise RuntimeError("REGION_DTYPE_RESULT_PATH_MISMATCH")
    result = load_json(result_path)
    if result.get("task_id") != REGION_DTYPE_TASK or result.get("status") != "TERMINAL_149_FRAME_REGION_CORRECTION":
        raise RuntimeError("REGION_DTYPE_RESULT_INVALID")
    validation = result.get("final_validation")
    if not validation or validate_artifact_ref(validation):
        raise RuntimeError("REGION_DTYPE_VALIDATION_REF_INVALID")
    if load_json(Path(validation["path"])).get("status") != "PASS_FOR_REGION_DENOMINATOR_CORRECTION_ONLY":
        raise RuntimeError("REGION_DTYPE_VALIDATION_NOT_PASSED")
    return {**base, "current_index_status": "ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
            "status": result["status"], "result": artifact_ref(result_path),
            "final_validation": validation,
            "counts": {"old_region_visible_frames": result["old_region_visible_frames"],
                       "corrected_region_visible_frames": result["corrected_region_visible_frames"],
                       "corrected_region_unknown_frames": result["corrected_region_unknown_frames"],
                       "products_structure": result["products_structure"],
                       "products_quality": result["products_quality"],
                       "products_adopted": result["products_adopted"]},
            "quality_axes": {"execution": "EXECUTED", "structure": "PASS",
                             "observability_denominator": "CORRECTED_REGION_ONLY",
                             "pose_quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED"},
            "claim_limit": result["claim_limit"]}


def build_status(task_state: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    task = select_task(task_state)
    if task is None:
        raise RuntimeError("HUMAN_TO_ROBOT_TASK_MISSING")
    if task.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError("LATEST_HUMAN_TO_ROBOT_TASK_NOT_TERMINAL")
    task_id = str(task["task_id"])
    attempt = REPO_ROOT / f"_run/current/{task_id}/attempts/attempt_0001"
    result_path = attempt / "RESULT.json"
    validation_path = attempt / "FINAL_VALIDATION.json"
    if task.get("result"):
        errors = validate_artifact_ref(task["result"])
        if errors or Path(task["result"]["path"]).resolve() != result_path.resolve():
            raise RuntimeError(f"TERMINAL_RESULT_BINDING_INVALID:{errors}")
    result = load_json(result_path)
    validation = load_json(validation_path)
    if result.get("task_id") != task_id:
        raise RuntimeError("TERMINAL_RESULT_TASK_MISMATCH")
    if result.get("final_validation"):
        errors = validate_artifact_ref(result["final_validation"])
        if errors or Path(result["final_validation"]["path"]).resolve() != validation_path.resolve():
            raise RuntimeError(f"TERMINAL_VALIDATION_BINDING_INVALID:{errors}")
    active = [row["task_id"] for row in task_state.get("tasks", [])
              if row.get("status") in ACTIVE_STATUSES]
    if task_id == S2_TASK:
        sealed_tests = {
            **validation["regression"],
            "scope": (
                "S2 current closure plus immutable pre-H3 regression evidence; "
                "the fourteen CPFS hardlink transaction tests remain not evaluated."
            ),
        }
    else:
        sealed_tests = {
            **validation["pytest"],
            "scope": "as recorded in immutable run validation; not the current code regression",
        }
    status = {
        "schema_version": "HUMAN_TO_ROBOT_TERMINAL_STATUS_V2",
        "generated_at": generated_at,
        "governance_revision": task_state.get("governance_revision"),
        "current_index_status": "PASS" if active or task_state.get("next_task") else "PASS_NO_ACTIVE_TASKS",
        "active_tasks": active,
        "next_task": task_state.get("next_task"),
        "parent_task_id": task_id,
        "latest_task": task_id,
        "parent_status": task.get("status"),
        "result_status": result.get("status"),
        "result": artifact_ref(result_path),
        "final_validation": artifact_ref(validation_path),
        "counts": {
            **result["execution_summary"],
            "validated_videos": validation["video_count"],
        },
        "sealed_validation_test_summary": sealed_tests,
        "quality_axes": result["quality_axes"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": result["claim_limit"],
    }
    if task_id in {S2_TASK, S1_TASK}:
        prior_path = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/RESULT.json"
        if prior_path.exists():
            prior = load_json(prior_path)
            status["preserved_r2"] = {
                "result": artifact_ref(prior_path),
                "counts": prior["execution_summary"],
                "scope": "prior R2 execution, not new S1 execution",
            }
        if task_id == S1_TASK and SYNC_VALIDATION.exists():
            verification = load_json(SYNC_VALIDATION)
            if verification.get("algorithm_task_id") != task_id:
                raise RuntimeError("SYNC_REGRESSION_TASK_MISMATCH")
            for name in ("junit", "log"):
                errors = validate_artifact_ref(verification[name])
                if errors:
                    raise RuntimeError(f"SYNC_REGRESSION_BINDING_INVALID:{errors}")
            status["latest_code_regression"] = {
                **verification["pytest"], "receipt": artifact_ref(SYNC_VALIDATION),
                "scope": "status synchronization code and full CPU regression; no model rerun",
            }
    return status


def publish_navigation_if_human_to_robot_r2(
    repo_root: Path, task_state: Mapping[str, Any], generated_at: str
) -> bool:
    from chaoyang.governance.four_stream_completion import publish_navigation
    if publish_navigation(repo_root, task_state, generated_at):
        return True
    if repo_root.resolve() != REPO_ROOT.resolve():
        raise RuntimeError("R2_REPO_ROOT_MISMATCH")
    # This task is selected by the ledger's current event, not by the mere
    # existence of an older task row. The older four-lane row remains in the
    # ledger indefinitely and must not take over the current projection.
    latest_event = next(
        (event for event in reversed(task_state.get("recent_events", []))
         if event.get("task_id")), None,
    )
    if latest_event and latest_event["task_id"] == SHARED_HAND_DELIVERY_TASK:
        task = next((row for row in task_state.get("tasks", [])
                     if row.get("task_id") == SHARED_HAND_DELIVERY_TASK), None)
        if task is None:
            raise RuntimeError("SHARED_HAND_DELIVERY_EVENT_WITHOUT_ROW")
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {
            "schema_version": "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "PASS" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active,
            "next_task": task_state.get("next_task"),
            "parent_task_id": SHARED_HAND_DELIVERY_TASK,
            "latest_task": SHARED_HAND_DELIVERY_TASK,
            "parent_status": task["status"],
            "task_packet": task.get("task_packet"),
            "counts": {"products_structure": "4/4", "products_quality": "0/4",
                       "products_adopted": "0/4", "qualified_kai22_h50_windows": 0},
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
            "claim_limit": "Bounded shared-hand terminal projection; module evidence does not promote product or training eligibility.",
        }
        if task["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_SHARED_HAND_DELIVERY"
        elif task["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result_ref = task.get("result")
            if not result_ref or validate_artifact_ref(result_ref):
                raise RuntimeError("SHARED_HAND_DELIVERY_TERMINAL_RESULT_INVALID")
            result = load_json(Path(result_ref["path"]))
            if result.get("task_id") != SHARED_HAND_DELIVERY_TASK:
                raise RuntimeError("SHARED_HAND_DELIVERY_RESULT_TASK_MISMATCH")
            status.update(
                status="PASS_NO_ACTIVE_TASKS",
                latest_terminal_status=task["status"],
                result=result_ref,
                counts=result["counts"],
                robot_qualified_scope="NO_NEW_FULL_SCOPE; 3_OF_4_WINDOW_BLOCKS_FEASIBLE_NOT_ADOPTED",
                clean_qualified_scope="NONE; METHOD_QUALITY_NOT_EVALUATED_AFTER_ADAPTER_LIMIT",
                data_label_quality_pass=result["hand_data"]["label_quality_pass"],
                data_interface_pass=result["hand_data"]["interface_pass"],
            )
        else:
            raise RuntimeError("SHARED_HAND_DELIVERY_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    if latest_event and latest_event["task_id"] == RESULT_BREAKTHROUGH_TASK:
        task = next((row for row in task_state.get("tasks", [])
                     if row.get("task_id") == RESULT_BREAKTHROUGH_TASK), None)
        if task is None:
            raise RuntimeError("RESULT_BREAKTHROUGH_EVENT_WITHOUT_ROW")
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {
            "schema_version": "HUMAN_TO_ROBOT_RESULT_BREAKTHROUGH_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "PASS" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active, "next_task": task_state.get("next_task"),
            "parent_task_id": RESULT_BREAKTHROUGH_TASK,
            "latest_task": RESULT_BREAKTHROUGH_TASK,
            "parent_status": task["status"],
            "task_packet": task.get("task_packet"),
            "counts": {"products_structure": "4/4", "products_quality": "0/4",
                       "products_adopted": "0/4", "qualified_label_windows": 0},
            "training_eligible": False, "control_ground_truth": False,
            "physical_deployable": False, "external_metric_authority": False,
            "claim_limit": "Executed Robot/Clean/Data attempts remain quality-rejected; structure and finite loss do not grant product or label adoption.",
        }
        if task["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_RESULT_BREAKTHROUGH"
        elif task["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result_ref = task.get("result")
            if not result_ref or validate_artifact_ref(result_ref):
                raise RuntimeError("RESULT_BREAKTHROUGH_TERMINAL_RESULT_INVALID")
            result = load_json(Path(result_ref["path"]))
            if result.get("task_id") != RESULT_BREAKTHROUGH_TASK:
                raise RuntimeError("RESULT_BREAKTHROUGH_RESULT_TASK_MISMATCH")
            status.update(
                status="PASS_NO_ACTIVE_TASKS",
                latest_terminal_status=task["status"],
                result=result_ref, counts=result["counts"],
                robot_full="171_FRAME_CANDIDATE_REJECTED_QUALITY",
                clean_full="NOT_RUN_FIXED_WINDOW_REJECTED_QUALITY",
                data_interface_pass=result["data"]["interface_pass"],
                data_label_quality_pass=result["data"]["label_quality_pass"],
                second_cycle=result["second_cycle"],
            )
        else:
            raise RuntimeError("RESULT_BREAKTHROUGH_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    if latest_event and latest_event["task_id"] == QUALITY_ACCEPTANCE_TASK:
        task = next((row for row in task_state.get("tasks", [])
                     if row.get("task_id") == QUALITY_ACCEPTANCE_TASK), None)
        if task is None:
            raise RuntimeError("QUALITY_ACCEPTANCE_EVENT_WITHOUT_ROW")
        attempt = REPO_ROOT / f"_run/current/{QUALITY_ACCEPTANCE_TASK}/attempts/attempt_0001"
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {
            "schema_version": "HUMAN_TO_ROBOT_QUALITY_ACCEPTANCE_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "PASS" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active, "next_task": task_state.get("next_task"),
            "parent_task_id": QUALITY_ACCEPTANCE_TASK,
            "latest_task": QUALITY_ACCEPTANCE_TASK,
            "parent_status": task["status"],
            "task_packet": task.get("task_packet"),
            "counts": {"products_structure": "4/4", "products_quality": "0/4",
                       "products_adopted": "0/4"},
            "training_eligible": False, "control_ground_truth": False,
            "physical_deployable": False, "external_metric_authority": False,
            "claim_limit": "Poker technical quality requires independent gates; user adoption needs explicit review of the bound SHA.",
        }
        progress = attempt / "PROGRESS.json"
        if progress.is_file():
            value = load_json(progress)
            if value.get("task_id") != QUALITY_ACCEPTANCE_TASK:
                raise RuntimeError("QUALITY_ACCEPTANCE_PROGRESS_TASK_DRIFT")
            status["progress"] = artifact_ref(progress)
            status["counts"] = value["product_counts"]
        if task["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_QUALITY_ACCEPTANCE"
        elif task["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result_ref = task.get("result")
            if not result_ref or validate_artifact_ref(result_ref):
                raise RuntimeError("QUALITY_ACCEPTANCE_TERMINAL_RESULT_INVALID")
            status.update(status="TERMINAL_QUALITY_ACCEPTANCE", result=result_ref)
        else:
            raise RuntimeError("QUALITY_ACCEPTANCE_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    if latest_event and latest_event["task_id"] == REPRESENTATIVE_BASELINE_TASK:
        representative = next(
            (row for row in task_state.get("tasks", [])
             if row.get("task_id") == REPRESENTATIVE_BASELINE_TASK), None,
        )
        if representative is None:
            raise RuntimeError("REPRESENTATIVE_TASK_EVENT_WITHOUT_ROW")
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {
            "schema_version": "HUMAN_TO_ROBOT_REPRESENTATIVE_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "PASS" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active,
            "next_task": task_state.get("next_task"),
            "parent_task_id": REPRESENTATIVE_BASELINE_TASK,
            "latest_task": REPRESENTATIVE_BASELINE_TASK,
            "parent_status": representative["status"],
            "task_packet": representative.get("task_packet"),
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
            "claim_limit": "Executable learning interface and decoded full-arm reviews do not prove stable Robot action, Clean quality or training-label eligibility.",
        }
        if representative["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_REPRESENTATIVE_BASELINE"
        elif representative["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result_ref = representative.get("result")
            if not result_ref or validate_artifact_ref(result_ref):
                raise RuntimeError("REPRESENTATIVE_TERMINAL_RESULT_INVALID")
            result = load_json(Path(result_ref["path"]))
            if result.get("task_id") != REPRESENTATIVE_BASELINE_TASK:
                raise RuntimeError("REPRESENTATIVE_RESULT_TASK_MISMATCH")
            status.update(
                status="TERMINAL_REPRESENTATIVE_BASELINE",
                result_status=result["status"],
                result=result_ref,
                latest_result=result_ref,
                counts=result["product_counts"],
                final_validation=result["validation"],
            )
        else:
            raise RuntimeError("REPRESENTATIVE_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    increment = next((row for row in task_state.get("tasks", [])
                      if row.get("task_id") == FOUR_LANE_INCREMENT_TASK), None)
    if increment is not None:
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        result_path = REPO_ROOT / f"_run/current/{FOUR_LANE_INCREMENT_TASK}/attempts/attempt_0001/RESULT.json"
        status = {
            "schema_version": "HUMAN_TO_ROBOT_FOUR_LANE_INCREMENT_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "ACTIVE_FINITE_FOUR_LANE_INCREMENT" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active, "next_task": task_state.get("next_task"),
            "parent_task_id": FOUR_LANE_INCREMENT_TASK,
            "latest_task": FOUR_LANE_INCREMENT_TASK,
            "parent_status": increment["status"],
            "task_packet": increment.get("task_packet"),
            "counts": {"products_structure": "4/4", "products_quality": "0/4",
                       "products_adopted": "0/4"},
            "training_eligible": False, "control_ground_truth": False,
            "physical_deployable": False, "external_metric_authority": False,
            "claim_limit": "CPU quality increments do not promote Clean, product or external authority.",
        }
        if increment["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_FOUR_LANE_INCREMENT"
        elif increment["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result = increment.get("result")
            if not result or validate_artifact_ref(result):
                raise RuntimeError("FOUR_LANE_INCREMENT_TERMINAL_RESULT_INVALID")
            status["status"] = "TERMINAL_FOUR_LANE_INCREMENT"
            status["result"] = result
        else:
            raise RuntimeError("FOUR_LANE_INCREMENT_STATUS_UNSUPPORTED")
        if result_path.is_file():
            status["latest_result"] = artifact_ref(result_path)
        atomic_json(OUTPUT, status)
        return True
    delivery_task = next((row for row in task_state.get("tasks", [])
                          if row.get("task_id") == TEN_HOUR_DELIVERY_TASK), None)
    if delivery_task is not None:
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        attempt = REPO_ROOT / f"_run/current/{TEN_HOUR_DELIVERY_TASK}/attempts/attempt_0001"
        delivery_path = attempt / "delivery/RESULT.json"
        status = {
            "schema_version": "HUMAN_TO_ROBOT_10H_DELIVERY_CURRENT_STATUS_V1",
            "generated_at": generated_at,
            "governance_revision": task_state.get("governance_revision"),
            "current_index_status": "ACTIVE_FINITE_DELIVERY" if active else "PASS_NO_ACTIVE_TASKS",
            "active_tasks": active,
            "next_task": task_state.get("next_task"),
            "parent_task_id": TEN_HOUR_DELIVERY_TASK,
            "latest_task": TEN_HOUR_DELIVERY_TASK,
            "parent_status": delivery_task["status"],
            "task_packet": delivery_task.get("task_packet"),
            "counts": {"products_structure": "4/4", "products_quality": "0/4",
                       "products_adopted": "0/4"},
            "training_eligible": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
            "claim_limit": "Full decode and 15-slot delivery do not imply Clean or product quality adoption.",
        }
        if delivery_path.is_file():
            delivery = load_json(delivery_path)
            if (delivery.get("task_id") != TEN_HOUR_DELIVERY_TASK
                    or delivery.get("validated_slots") != 15):
                raise RuntimeError("TEN_HOUR_DELIVERY_RESULT_INVALID")
            status["delivery"] = artifact_ref(delivery_path)
            status["counts"]["validated_video_slots"] = 15
        if delivery_task["status"] in ACTIVE_STATUSES:
            status["status"] = "ACTIVE_DELIVERY_IN_PROGRESS"
        elif delivery_task["status"] in TERMINAL_STATUSES | {"PASSED"}:
            result = delivery_task.get("result")
            if not result or validate_artifact_ref(result):
                raise RuntimeError("TEN_HOUR_TERMINAL_RESULT_INVALID")
            status["status"] = "TERMINAL_DELIVERY_QUALITY_REJECTED"
            status["result"] = result
        else:
            raise RuntimeError("TEN_HOUR_TASK_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    temporal_task = next((row for row in task_state.get("tasks", [])
                          if row.get("task_id") == SURFACE_TEMPORAL_TASK), None)
    if temporal_task is not None:
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_TEMPORAL_CURRENT_STATUS_V1",
                  "generated_at": generated_at, "governance_revision": task_state.get("governance_revision"),
                  "active_tasks": active, "next_task": task_state.get("next_task"),
                  "parent_task_id": SURFACE_TEMPORAL_TASK, "latest_task": SURFACE_TEMPORAL_TASK,
                  "parent_status": temporal_task["status"], "training_eligible": False,
                  "control_ground_truth": False, "physical_deployable": False,
                  "external_metric_authority": False,
                  "counts": {"products_structure": "4/4", "products_quality": "0/4",
                             "products_adopted": "0/4"},
                  "claim_limit": "031 temporal model/Stereo surface-Z diagnostic with visible-card UNKNOWN; not product quality."}
        if temporal_task["status"] in ACTIVE_STATUSES:
            status.update(current_index_status="ACTIVE_FINITE_031_SURFACE_TEMPORAL",
                          status="REGISTERED_NOT_YET_EVALUATED", task_packet=temporal_task.get("task_packet"))
        elif temporal_task["status"] == "PASSED":
            result_path = REPO_ROOT / f"_run/current/{SURFACE_TEMPORAL_TASK}/attempts/attempt_0001/RESULT.json"
            if not temporal_task.get("result") or validate_artifact_ref(temporal_task["result"]):
                raise RuntimeError("SURFACE_TEMPORAL_RESULT_REF_INVALID")
            result = load_json(result_path)
            if result.get("task_id") != SURFACE_TEMPORAL_TASK or result.get("status") != "TERMINAL_DEVELOPMENT_DIAGNOSTIC":
                raise RuntimeError("SURFACE_TEMPORAL_TERMINAL_RESULT_INVALID")
            validation = result.get("final_validation")
            if not validation or validate_artifact_ref(validation):
                raise RuntimeError("SURFACE_TEMPORAL_VALIDATION_REF_INVALID")
            status.update(current_index_status="ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
                          status=result["status"], result=artifact_ref(result_path),
                          final_validation=validation,
                          surface_summary={"frame_count": 149,
                                           "original_right_valid_frames": result["original_right_valid_frames"],
                                           "strict_object_guard_frames_with_samples": result["strict_object_guard_frames_with_samples"],
                                           "strict_object_guard_samples": result["strict_object_guard_samples"],
                                           "strict_frame_median_of_medians_m": result["strict_frame_median_of_medians_m"]})
        else:
            raise RuntimeError("SURFACE_TEMPORAL_TASK_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    surface_task = next((row for row in task_state.get("tasks", [])
                         if row.get("task_id") == SURFACE_DEPTH_TASK), None)
    if surface_task is not None:
        active = [str(row["task_id"]) for row in task_state.get("tasks", [])
                  if row.get("status") in ACTIVE_STATUSES]
        status = {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_CURRENT_STATUS_V1",
                  "generated_at": generated_at, "governance_revision": task_state.get("governance_revision"),
                  "active_tasks": active, "next_task": task_state.get("next_task"),
                  "parent_task_id": SURFACE_DEPTH_TASK, "latest_task": SURFACE_DEPTH_TASK,
                  "parent_status": surface_task["status"], "training_eligible": False,
                  "control_ground_truth": False, "physical_deployable": False,
                  "external_metric_authority": False,
                  "counts": {"products_structure": "4/4", "products_quality": "0/4",
                             "products_adopted": "0/4"},
                  "claim_limit": "031 same-pixel MANO/Stereo surface-Z diagnostic only, not wrist-centre or product accuracy."}
        if surface_task["status"] in ACTIVE_STATUSES:
            status.update(current_index_status="ACTIVE_FINITE_031_SURFACE_DEPTH",
                          status="REGISTERED_NOT_YET_EVALUATED", task_packet=surface_task.get("task_packet"))
        elif surface_task["status"] == "PASSED":
            result_path = REPO_ROOT / f"_run/current/{SURFACE_DEPTH_TASK}/attempts/attempt_0001/RESULT.json"
            if not surface_task.get("result") or validate_artifact_ref(surface_task["result"]):
                raise RuntimeError("SURFACE_DEPTH_RESULT_REF_INVALID")
            result = load_json(result_path)
            if result.get("task_id") != SURFACE_DEPTH_TASK or result.get("status") != "TERMINAL_DEVELOPMENT_DIAGNOSTIC":
                raise RuntimeError("SURFACE_DEPTH_TERMINAL_RESULT_INVALID")
            validation = result.get("final_validation")
            if not validation or validate_artifact_ref(validation):
                raise RuntimeError("SURFACE_DEPTH_VALIDATION_REF_INVALID")
            status.update(current_index_status="ACTIVE_OTHER_TASKS" if active else "PASS_NO_ACTIVE_TASKS",
                          status=result["status"], result=artifact_ref(result_path),
                          final_validation=validation,
                          surface_summary={"sampled_frames": result["sampled_frames"],
                                           "frames_with_surface_samples": result["frames_with_surface_samples"],
                                           "total_surface_samples": result["total_surface_samples"],
                                           "frame_median_of_signed_medians_m": result["frame_median_of_signed_medians_m"]})
        else:
            raise RuntimeError("SURFACE_DEPTH_TASK_STATUS_UNSUPPORTED")
        atomic_json(OUTPUT, status)
        return True
    region_task = next((row for row in task_state.get("tasks", [])
                        if row.get("task_id") == REGION_DTYPE_TASK), None)
    if region_task is not None:
        atomic_json(OUTPUT, build_region_dtype_status(task_state, generated_at))
        return True
    cross_eye_task = next((row for row in task_state.get("tasks", [])
                           if row.get("task_id") == CROSS_EYE_TASK), None)
    if cross_eye_task is not None:
        atomic_json(OUTPUT, build_cross_eye_status(task_state, generated_at))
        return True
    spatial_task = next((row for row in task_state.get("tasks", [])
                         if row.get("task_id") == SPATIAL_DONOR_TASK), None)
    if spatial_task is not None:
        atomic_json(OUTPUT, build_spatial_donor_status(task_state, generated_at))
        return True
    sensor_display_task = next((row for row in task_state.get("tasks", [])
                                if row.get("task_id") == SENSOR_DISPLAY_TASK), None)
    if sensor_display_task is not None:
        atomic_json(OUTPUT, build_sensor_display_status(task_state, generated_at))
        return True
    multidonor_task = next((row for row in task_state.get("tasks", [])
                            if row.get("task_id") == MULTIDONOR_TABLE_TASK), None)
    if multidonor_task is not None:
        atomic_json(OUTPUT, build_multidonor_table_status(task_state, generated_at))
        return True
    attachment_clean_task = next((row for row in task_state.get("tasks", [])
                                  if row.get("task_id") == ATTACH_CLEAN_TASK), None)
    if attachment_clean_task is not None:
        atomic_json(OUTPUT, build_attachment_clean_status(task_state, generated_at))
        return True
    table_task = next((row for row in task_state.get("tasks", [])
                       if row.get("task_id") == TABLE_DONOR_TASK), None)
    if table_task is not None:
        atomic_json(OUTPUT, build_table_donor_status(task_state, generated_at))
        return True
    seed_task = next((row for row in task_state.get("tasks", [])
                      if row.get("task_id") == SEED_REFINE_TASK), None)
    if seed_task is not None:
        atomic_json(OUTPUT, build_seed_refine_status(task_state, generated_at))
        return True
    wearable_task = next((row for row in task_state.get("tasks", [])
                          if row.get("task_id") == WEARABLE_TASK), None)
    if wearable_task is not None:
        atomic_json(OUTPUT, build_wearable_status(task_state, generated_at))
        return True
    donor_task = next((row for row in task_state.get("tasks", [])
                       if row.get("task_id") == DEVICE_DONOR_TASK), None)
    if donor_task is not None:
        atomic_json(OUTPUT, build_device_donor_status(task_state, generated_at))
        return True
    source_task = next((row for row in task_state.get("tasks", [])
                        if row.get("task_id") == SOURCE_COVERAGE_TASK), None)
    if source_task is not None:
        if source_task.get("status") in ACTIVE_STATUSES:
            # Missing current evidence must not silently expose an older terminal task.
            return False
        if source_task.get("status") in TERMINAL_STATUSES:
            atomic_json(OUTPUT, build_terminal_source_coverage_status(task_state, generated_at))
            return True
    active_product = next((row for row in task_state.get("tasks", [])
                           if row.get("task_id") == PRODUCT_FIRST_TASK
                           and row.get("status") in ACTIVE_STATUSES), None)
    if active_product is not None:
        atomic_json(OUTPUT, build_active_product_first_status(task_state, generated_at))
        return True
    terminal_product = next((row for row in task_state.get("tasks", [])
                             if row.get("task_id") == PRODUCT_FIRST_TASK
                             and row.get("status") in TERMINAL_STATUSES), None)
    if terminal_product is not None:
        atomic_json(OUTPUT, build_terminal_product_first_status(task_state, generated_at))
        return True
    task = select_task(task_state)
    if task is not None and task.get("task_id") == S2_TASK and task.get("status") in ACTIVE_STATUSES:
        atomic_json(OUTPUT, build_active_s2_status(task_state, generated_at))
        return True
    if task is None or task.get("status") not in TERMINAL_STATUSES:
        return False
    # A known successor with missing evidence must fail, never fall back to R2.
    atomic_json(OUTPUT, build_status(task_state, generated_at))
    return True
