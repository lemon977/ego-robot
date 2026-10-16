"""Close sealed Mask evidence and freeze bounded G3 work; never run models."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref, verify_sealed_stage

CPU = "inpainting_generalization_g3_freeze_engineering_20261013"
MASK = "inpainting_post_g1_text_retention_model_a01_20261013"
MODEL = "inpainting_generalization_g3_mask_a01_20261013"
RECOVERY = "inpainting_generalization_g3_freeze_publication_recovery_20261013"
KEYS = ["001", "003", "008", "100", "114", "050"]


def once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def packet():
    path = REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    value = load_json(path)
    if value["task_id"] != CPU or value["algorithm_execution_allowed"] is not False:
        raise ValueError("EXACT_CPU_PACKET_REQUIRED")
    for reference in value["inputs"].values():
        verify_ref(reference)
    return value, Path(value["output_dir"])


def close_text():
    value, output = packet()
    model_root = REPO_ROOT / "_run" / MASK / "attempt_0001"
    model = load_json(model_root / "MODEL_RESULT.json")
    terminal = load_json(model_root / "RESULT.json")
    post = load_json(model_root / "POST_VALIDATION.json")
    if (terminal["status"] != "PASSED" or not terminal["model_result_consumable"]
            or post["exit_code"] != 0 or model["completed_mask_sessions"] != 6):
        raise ValueError("COMPLETE_SIX_MASK_AND_POST_REQUIRED")
    left_path = model_root / "review_cpu/lane_001_003_008/RESULT.json"
    right_path = model_root / "review_cpu/lane_100_114_050/RESULT.json"
    left, right = load_json(left_path), load_json(right_path)
    if (left["totals"]["actual_viewed_native_frames"] != 18
            or right["actual_viewed_frames"] != 18
            or left["totals"]["unviewed_native_frames"] != 603
            or right["unviewed_frames"] != 394):
        raise ValueError("ACTUAL36_AND_UNVIEWED997_REQUIRED")
    first = {row["session_key"]: row for row in left["sessions"]}
    second = {row["session_key"]: row for row in right["session_reviews"]}
    rows = []
    for produced in model["sessions"]:
        key = produced["session_key"]
        verify_ref(produced["source"])
        verify_ref(produced["mask"])
        receipt_path = Path(produced["mask"]["path"])
        receipt = load_json(receipt_path)
        verify_sealed_stage(receipt, receipt["component_signature"], output_root=receipt_path.parent)
        if key in first:
            item = first[key]
            review_ref = item["review"]
            counts = {"HIGH": item["defect_counts"]["HIGH_session_clusters"],
                      "MED": item["defect_counts"]["MED_session_clusters"]}
        else:
            item = second[key]
            review_path = right_path.parent / item["path"]
            review_ref = artifact_ref(review_path)
            if review_ref["sha256"] != item["sha256"]:
                raise ValueError("ACTUAL_AI_REVIEW_SHA_DRIFT")
            counts = {name: item[name] for name in ("HIGH", "MED")}
        verify_ref(review_ref)
        review = load_json(Path(review_ref["path"]))
        frames = review.get("viewed_native_frames", review.get("viewed_frames", item.get("viewed_frames")))
        if not isinstance(frames, list) or len(frames) != 6 or len(set(frames)) != 6:
            raise ValueError("EXACT_SIX_ACTUAL_ANCHORS_REQUIRED:" + key)
        video = produced["review"]["video"]
        verify_ref(video)
        if produced["review"]["decoded_frames"] != produced["native_frames"]:
            raise ValueError("FULL_NATIVE_DECODE_REQUIRED")
        target = REPO_ROOT / "review/inpainting_optimization" / ("POST_TEXT_" + key + "_Raw_Mask.mp4")
        if target.exists():
            raise ValueError("SHALLOW_MASK_DELIVERY_ALREADY_EXISTS")
        os.link(video["path"], target)
        shallow = artifact_ref(target)
        if shallow["sha256"] != video["sha256"] or shallow["bytes"] != video["bytes"]:
            raise ValueError("SHALLOW_MASK_BYTES_CHANGED")
        rows.append({"session_key": key, "native_frames": produced["native_frames"],
            "source": produced["source"], "mask": produced["mask"], "review": review_ref,
            "actual_viewed_frames": frames, "unviewed_frames": produced["native_frames"] - len(frames),
            "shallow_raw_mask_video": shallow, "decoded_frames": produced["review"]["decoded_frames"],
            "visual_quality": "FAILED", "issue_counts": counts,
            "timebase_status": produced["review"]["timebase_status"],
            "clean_motion_render_composite": "NOT_EXECUTED_BY_THIS_MASK_EXPERIMENT"})
    if [row["session_key"] for row in rows] != KEYS:
        raise ValueError("FROZEN_SIX_ORDER_REQUIRED")
    normal = load_json(Path(rows[-1]["review"]["path"]))["normal_control"]
    if normal["status"] != "FAILED_LOCAL_VISIBLE_GLOVE_SUPPORT_REGRESSION":
        raise ValueError("KNOWN050_NORMAL_REGRESSION_MUST_REMAIN")
    audit_path = model_root / "audit_cpu/AUDIT_RESULT.json"
    audit_ref = artifact_ref(audit_path)
    parent_review_path = REPO_ROOT / "_run/inpainting_post_g1_controller_rgb_model_a01_20261013/attempt_0001/review_cpu/REVIEW_ROLLUP.json"
    reference = once(output / "TEXT_RETENTION_REVIEW_ROLLUP.json", {
        "schema": "inpainting-post-text-retention-final-review-v1", "created_at": now_iso(),
        "task_id": CPU, "producer_task": MASK, "producer_terminal": artifact_ref(model_root / "RESULT.json"),
        "producer_post": artifact_ref(model_root / "POST_VALIDATION.json"),
        "model_result": artifact_ref(model_root / "MODEL_RESULT.json"),
        "origin_input_resource_audit": audit_ref, "parent_review": artifact_ref(parent_review_path),
        "lane_reviews": [artifact_ref(left_path), artifact_ref(right_path)], "sessions": rows,
        "metrics": {"execution_complete_sessions": 6, "full_decoded_native_frames": 1033,
            "actual_ai_viewed_frames": 36, "unviewed_native_frames": 997,
            "visual_qualified_sessions": 0, "HIGH_session_clusters_parent": 12,
            "HIGH_session_clusters_candidate": sum(row["issue_counts"]["HIGH"] for row in rows),
            "MED_session_clusters_parent": 6,
            "MED_session_clusters_candidate": sum(row["issue_counts"]["MED"] for row in rows),
            "HIGH_reduction_percent": 0, "new_HIGH_normal_anchor_events": 1,
            "normal_control_passed": False, "manual_seed_reads": 0,
            "SAM_adapter_API_attempts": model["model_call_accounting"]["attempted_api_calls"],
            "completed_API_count": "NOT_INSTRUMENTED_SEALED_STAGE_COMPLETENESS_IS_SEPARATE"},
        "normal_control_050_127": normal,
        "sample_scope": "SIX_FIXED_ANCHORS_EACH_NOT_FULL_VISUAL_VIDEO_REVIEW",
        "hidden_regions": "UNKNOWN", "human_review": "PENDING", "physical_eye_verified": False,
        "candidate_adopted": False, "training_eligible": False, "control_ground_truth": False,
        "clean_robot": "NOT_EXECUTED_BY_THIS_MASK_EXPERIMENT", "files_deleted": 0,
        "next_action": "No TEXT-retention retry/adoption. Freeze unchanged recipe for legal G3 execution diagnostic; future role/UNKNOWN conflict work is another finite task, not holdout retuning."})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(post_g1_text_retention_model=artifact_ref(model_root / "MODEL_RESULT.json"),
        post_g1_text_retention_terminal=artifact_ref(model_root / "RESULT.json"),
        post_g1_text_retention_post=artifact_ref(model_root / "POST_VALIDATION.json"),
        post_g1_text_retention_final_review=reference, post_g1_text_retention_latest_facts=load_json(Path(reference["path"]))["metrics"],
        post_g1_text_retention_shallow_videos={row["session_key"]: row["shallow_raw_mask_video"] for row in rows})
    index["next_actions"].update(mask="TEXT candidate sealed6/1033, actual36/997 unviewed; HIGH12->12, MED6->5, newHIGH050f127; reject adoption, no same-recipe retry.",
        cross_domain="G3 separate CPU Source/dispatch freeze now; fixed039/007/199 plus selected0918/001-008, no quality-based swapping or holdout tuning.")
    atomic_json(index_path, index)
    return reference


def seal_engineering(junit):
    value, output = packet()
    if junit is None or not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_COMPLETE_JUNIT_REQUIRED")
    counts = {name: sum(int(row.attrib.get(name, 0)) for row in ET.parse(junit).getroot().iter("testsuite"))
              for name in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 898 or any(counts[name] for name in ("errors", "failures", "skipped")):
        raise ValueError("EXISTING898_PLUS_G3_COMPLETE_CPU_REGRESSION_REQUIRED")
    source_index_path = output / "lane_source/SOURCE_INDEX_G3.json"
    source_result_path = output / "lane_source/SOURCE_RESULT.json"
    source_index, source_result = load_json(source_index_path), load_json(source_result_path)
    if (source_index["schema"] != "inpainting-g3-frozen-source-manifest-v1"
            or source_result["sessions_selected"] != 11 or source_result["model_calls"] != 0):
        raise ValueError("FIXED11_ZERO_MODEL_SOURCE_REQUIRED")
    for row in source_index["sessions"]:
        if row["source_status"] == "READY":
            verify_ref(row["source_receipt"])
            receipt_path = Path(row["source_receipt"]["path"])
            receipt = load_json(receipt_path)
            verify_sealed_stage(receipt, row["source_signature"], output_root=receipt_path.parent)
            verify_ref(row["frame_map_ref"])
        elif row["source_status"] != "BLOCKED":
            raise ValueError("G3_SOURCE_STATUS_UNSUPPORTED")
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    runtime = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"]
    extra = [artifact_ref(Path(__file__).resolve()), artifact_ref(REPO_ROOT / "tests/unit/test_inpainting_candidate_g3_execution.py")]
    references = {ref["path"]: ref for ref in [*runtime, *extra]}
    snapshot = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in references.values():
            verify_ref(reference)
            archive.write(reference["path"], str(Path(reference["path"]).relative_to(REPO_ROOT)))
    lanes = [output / "lane_runtime/ENGINEERING_RESULT.json", output / "lane_pipeline/PIPELINE_READINESS_RESULT.json",
             output / "lane_gate/ROOT_HELPER_INTEGRATION_AUDIT_R02.json"]
    if not all(path.is_file() for path in lanes):
        raise ValueError("COMPLETE_BOUNDED_RUNTIME_AND_PIPELINE_LANES_REQUIRED")
    review_path = output / "TEXT_RETENTION_REVIEW_ROLLUP.json"
    review = load_json(review_path)
    if review["metrics"]["visual_qualified_sessions"] != 0 or review["metrics"]["new_HIGH_normal_anchor_events"] != 1:
        raise ValueError("ACTUAL_REJECTED_MASK_AND_NORMAL_REGRESSION_MUST_REMAIN")
    result_ref = once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-g3-source-execution-engineering-v1", "task_id": CPU,
        "run_signature": value["run_signature"], "completed_at": now_iso(), "cpu_tests": counts,
        "junit": artifact_ref(junit), "code": list(references.values()), "snapshot": artifact_ref(snapshot),
        "source_index": artifact_ref(source_index_path), "source_result": artifact_ref(source_result_path),
        "lane_results": [artifact_ref(path) for path in lanes],
        "text_retention_actual_review": artifact_ref(review_path),
        "root_050_actual_corroboration": artifact_ref(output / "ROOT_TEXT_MASK_050_REVIEW_R01.json"),
        "source_summary": {key: source_result[key] for key in ("sessions_selected", "ready_source_sessions",
            "locally_blocked_sessions", "sealed_source_frames", "verified_raw_files", "verified_raw_bytes")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "publication_successor": artifact_ref(REPO_ROOT / "tasks/current" / RECOVERY / "TASK_PACKET.json"),
        "parent_writer_disposition": "FAILED_RUNTIME_FINAL_OWNED_PARTIAL_PUBLICATION_RETAINED",
        "helper_review_scope": "R02 independently checked Source/Mask ABI before subsequent publication-provenance-only recovery binding. Actual final helper bytes are frozen in code/snapshot; G3 algorithm runtime matches final R02 hashes. No algorithm, recipe or schema change after R02.",
        "model_calls": 0, "files_deleted": 0, "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1",
        "algorithm_recipe_changed": False, "models_changed": False, "thresholds_changed": False,
        "visual_quality": "NOT_EVALUATED_G3_MODELS_NOT_RUN", "candidate_adopted": False,
        "training_eligible": False, "control_ground_truth": False,
        "next_task": MODEL})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(g3_engineering=result_ref, g3_source_index=artifact_ref(source_index_path),
                 g3_source_result=artifact_ref(source_result_path))
    index["next_actions"]["cross_domain"] = "CPU source/dispatch sealed. Register one fixed G3 Mask diagnostic after CPU terminal/POST0; not adoption or full Robot production."
    atomic_json(index_path, index)
    once(output / "MAINTENANCE_DONE.json", {"task_id": CPU, "run_signature": value["run_signature"],
        "status": "PASS", "artifacts": [result_ref, artifact_ref(junit), artifact_ref(snapshot),
            artifact_ref(source_index_path), artifact_ref(source_result_path), artifact_ref(review_path)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result_ref


def model_spec():
    value, output = packet()
    parent_cpu_result = load_json(output / "RESULT.json")
    recovered = (parent_cpu_result.get("status") == "FAILED_RUNTIME_FINAL"
                 and parent_cpu_result.get("scope_status") == "OWNED_PARTIAL_PUBLICATION_INTERRUPTED")
    completed_root = output / "recovery_0001" if recovered else output
    completed = load_json(completed_root / "RESULT.json")
    if (completed["scope_status"] != "PASS" or completed["status"] != "PASSED"
            or load_json(completed_root / "POST_VALIDATION.json")["exit_code"] != 0):
        raise ValueError("CLOSED_G3_CPU_AND_POST_REQUIRED")
    if recovered:
        successor = load_json(REPO_ROOT / "tasks/current" / RECOVERY / "TASK_PACKET.json")
        if (completed["task_id"] != RECOVERY or completed["run_signature"] != successor["run_signature"]
                or completed["evidence"] != artifact_ref(output / "MAINTENANCE_DONE.json")
                or completed["interrupted_parent_result"] != artifact_ref(output / "RESULT.json")
                or successor["inputs"]["interrupted_parent"] != artifact_ref(REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json")):
            raise ValueError("EXACT_G3_PUBLICATION_SUCCESSOR_REQUIRED")
    engineering = load_json(output / "ENGINEERING_RESULT.json")
    for reference in engineering["code"]:
        verify_ref(reference)
    parent_path = REPO_ROOT / "tasks/current" / MASK / "TASK_PACKET.json"
    parent = load_json(parent_path)
    parent_terminal = load_json(Path(value["inputs"]["retention_terminal"]["path"]))
    if parent["run_signature"] != parent_terminal["run_signature"]:
        raise ValueError("EXACT_SEALED_RETENTION_PARENT_REQUIRED")
    config = copy.deepcopy(parent["config"])
    config.update(candidate_id="G3_FROZEN_TEXT_RETENTION_A01",
                  algorithm_parameter_change="NONE_FROZEN_GLOBAL_RECIPE_CROSS_DOMAIN_DIAGNOSTIC")
    from chaoyang.governance.inpainting_candidate_registration import (
        visual_route_contract, VISUAL_ROUTE_KEY, G3_SCOPE, G3_SPEC_SCHEMA, G3_RESULT_SCHEMA,
        g3_session_inputs, g3_api_bound, validate_g3_experiment)
    runtime = [*visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"],
               artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    weights = parent["model_weights"]
    for reference in [*runtime, *weights]:
        verify_ref(reference)
    source_index_path = output / "lane_source/SOURCE_INDEX_G3.json"
    source_index = load_json(source_index_path)
    schema = copy.deepcopy(parent["schema"])
    schema["model_result"] = G3_RESULT_SCHEMA
    rows, maximum = [], 0
    calibration = {"metric_projection": "NOT_CONSUMED_MASK",
                   "source_eye_qualification": "PER_SESSION_SOURCE_RECEIPT_NOT_METRIC_PROJECTION"}
    for frozen in source_index["sessions"]:
        row = g3_session_inputs(frozen)
        if row["source_status"] == "READY":
            row["model_calls_max"] = g3_api_bound(row["native_frames"])
            row["mask_signature"] = canonical_sha({"source": row["source_receipt"],
                "code": runtime, "config": config, "weights": weights,
                "calibration": calibration, "schema": schema})
            maximum += row["model_calls_max"]
        else:
            row["model_calls_max"] = 0
        rows.append(row)
    if maximum <= 0:
        raise ValueError("NO_LEGAL_READY_SOURCE_KEEP_DIAGNOSTIC_NO_GPU")
    spec = {"spec_schema": G3_SPEC_SCHEMA, "experiment_scope": G3_SCOPE,
        "task_id": MODEL, "output_dir": str(REPO_ROOT / "_run" / MODEL / "attempt_0001"),
        "infrastructure": value["runtime_code"][1], "runtime_code": runtime,
        "model_weights": weights, "weights": copy.deepcopy(parent["weights"]),
        "physical_gpu_index": 0, "model_stages": ["Mask"], "config": config,
        "calibration": calibration, "schema": schema,
        "inputs": {"source_index": artifact_ref(source_index_path),
            "parent_recipe_packet": artifact_ref(parent_path), "frozen_selection": value["inputs"]["frozen_inputs"]},
        "engineering_evidence": {"packet": artifact_ref(REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"),
            "result": artifact_ref(output / "RESULT.json"), "post": artifact_ref(output / "POST_VALIDATION.json"),
            "completed_publication_result": artifact_ref(completed_root / "RESULT.json"),
            "completed_publication_post": artifact_ref(completed_root / "POST_VALIDATION.json"),
            "engineering": artifact_ref(output / "ENGINEERING_RESULT.json"),
            "source_result": artifact_ref(output / "lane_source/SOURCE_RESULT.json"),
            "rejected_development_review": artifact_ref(output / "TEXT_RETENTION_REVIEW_ROLLUP.json")},
        "budgets": {"candidates_max": 1, "wall_seconds_max": 7200, "gpu_count": 1,
            "gpu_seconds_max": 7200, "cpu_threads_per_worker": 2, "model_calls_max": maximum,
            "model_calls_unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION", "training_steps_max": 0,
            "output_bytes_estimate": 8589934592}, "sessions": rows,
        "evaluation_only": {"development_actual_review": artifact_ref(output / "TEXT_RETENTION_REVIEW_ROLLUP.json"),
                            "hidden_regions": "UNKNOWN", "human_review": "PENDING"},
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
                     "inpainting_optimization/G3_MASK_A01_SPEC.json"],
        "read_set_expansions": ["Only consume each frozen READY Source's current Raw RGB/native map, the unchanged global SAM recipe and frozen checkpoint. BLOCKED rows are diagnostic only with zero model calls. No old0918/199 Mask, seeds, Clean, q, annotations, hand fitting or task configuration.",
            "One frozen G3 execution diagnostic, not a new G1 recipe or adoption. Do not tune on this fixed challenge/cross-domain sample after results; preserve all failure, unknown and unviewed scope. Mask-only, not Clean/Robot, training or physical deployment."]}
    validate_g3_experiment(spec, repo_root=REPO_ROOT)
    return once(REPO_ROOT / "inpainting_optimization/G3_MASK_A01_SPEC.json", spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("close-text", "seal-engineering", "model-spec"))
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    print(close_text() if args.action == "close-text" else seal_engineering(args.junit)
          if args.action == "seal-engineering" else model_spec())


if __name__ == "__main__":
    main()
