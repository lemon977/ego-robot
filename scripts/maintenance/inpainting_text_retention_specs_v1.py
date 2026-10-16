"""Freeze finite post-G1 support-retention work without reviving old budgets."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref, verify_sealed_stage

ORIGINAL_CPU = "inpainting_post_g1_text_retention_engineering_20261013"
CPU = "inpainting_post_g1_text_retention_engineering_successor_20261013"
MODEL = "inpainting_post_g1_text_retention_model_a01_20261013"
SCOPE = "POST_G1_TEXT_WEARER_RETENTION_FAILURE"
STRATEGY = "PRESERVE_TEXT_WEARER_RGB_SUPPORT_V1"
DEV = ["001", "003", "008", "100", "114", "050"]
NATIVE = [238, 183, 200, 139, 145, 128]


def once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def successor_spec():
    packet_path = REPO_ROOT / "tasks/current" / ORIGINAL_CPU / "TASK_PACKET.json"
    packet = load_json(packet_path)
    original_output = Path(packet["output_dir"])
    terminal = load_json(original_output / "RESULT.json")
    post = load_json(original_output / "POST_VALIDATION.json")
    if (terminal["status"] != "PASSED" or terminal["scope_status"] != "BLOCKED"
            or post["exit_code"] != 0):
        raise ValueError("EXPLICIT_WRITE_SET_GAP_AND_CLEAN_TERMINAL_REQUIRED")
    spec = copy.deepcopy(load_json(Path(packet["inputs"]["spec"]["path"])))
    for reference in packet["inputs"].values():
        verify_ref(reference)
    spec["task_id"] = CPU
    spec["output_dir"] = str(REPO_ROOT / "_run" / CPU / "attempt_0001")
    # Future successors reserve administrative launch time; do not extend the
    # original deadline by however long CAS registration itself takes.
    spec["budgets"]["wall_seconds_max"] = min(7200, int((datetime.fromisoformat(
        packet["deadline_at"]) - datetime.now(timezone.utc)).total_seconds()) - 30)
    if spec["budgets"]["wall_seconds_max"] < 1:
        raise ValueError("ORIGINAL_CPU_TIMEBOX_EXHAUSTED")
    spec["objective"] += " Exact-write-set successor; original CPU hard deadline preserved, not a model attempt or third G1 recipe."
    spec["inputs"].update(write_set_audit=artifact_ref(original_output / "WRITE_SET_DEPENDENCY.json"),
        previous_cpu_packet=artifact_ref(packet_path),
        previous_cpu_terminal=artifact_ref(original_output / "RESULT.json"),
        previous_cpu_post=artifact_ref(original_output / "POST_VALIDATION.json"))
    spec["read_set"][3] = "inpainting_optimization/MASK_TEXT_RETENTION_CPU_S01_SPEC.json"
    spec["write_set"] = [item.replace(ORIGINAL_CPU, CPU) for item in spec["write_set"]]
    spec["write_set"] += [str(REPO_ROOT / "src/chaoyang/inpainting_candidate/model_stages.py"),
        str(REPO_ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py"),
        str(original_output / "lane_review")]
    spec["change_contract"]["original_cpu_deadline_at"] = packet["deadline_at"]
    spec["change_contract"]["partial_engineering"] = "IN_PLACE_OWNED_CODE_NOT_SEALED_STAGE_INPUT"
    return once(REPO_ROOT / "inpainting_optimization/MASK_TEXT_RETENTION_CPU_S01_SPEC.json", spec)


def packet():
    path = REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    value = load_json(path)
    if (value["task_id"] != CPU or value["algorithm_execution_allowed"] is not False
            or value["deletion_allowed"] is not False or value["budgets"]["gpu_count"] != 0):
        raise ValueError("EXACT_ZERO_MODEL_CPU_PACKET_REQUIRED")
    for reference in value["inputs"].values():
        verify_ref(reference)
    return path, value


def prepare():
    _, value = packet()
    output = Path(value["output_dir"])
    terminal = load_json(Path(value["inputs"]["composite_terminal"]["path"]))
    if terminal["status"] != "PASSED" or load_json(Path(value["inputs"]["composite_post"]["path"]))["exit_code"] != 0:
        raise ValueError("SEALED_FULL_COMPOSITE_AND_POST_REQUIRED")
    result = load_json(Path(value["inputs"]["composite_result"]["path"]))
    row = result["sessions"][0]
    reference = row["composite_receipt"]
    verify_ref(reference)
    receipt = load_json(Path(reference["path"]))
    refs = verify_sealed_stage(receipt, receipt["component_signature"], output_root=Path(reference["path"]).parent)
    report_ref = next(ref for ref in refs if Path(ref["path"]).name == "COMPOSITE_REPORT.json")
    report = load_json(Path(report_ref["path"]))
    if report["native_frames"] != 238 or report["qualified_trusted_occlusion_pixels"] != 0:
        raise ValueError("FULL_NATIVE238_AND_UNVERIFIED_GEOMETRY_REQUIRED")
    videos = {}
    for name, filename in (("final", "G2_001_Final_Robot.mp4"),
                           ("four_panel", "G2_001_Raw_Mask_Clean_Robot.mp4")):
        source = report["review"][name]
        if source["decoded_frames"] != 238:
            raise ValueError("COMPLETE_NATIVE_REVIEW_VIDEO_REQUIRED")
        verify_ref(source["video"])
        target = REPO_ROOT / "review/inpainting_optimization" / filename
        if target.exists():
            raise ValueError("IMMUTABLE_SHALLOW_VIDEO_ALREADY_EXISTS")
        os.link(source["video"]["path"], target)
        linked = artifact_ref(target)
        if linked["sha256"] != source["video"]["sha256"] or linked["bytes"] != source["video"]["bytes"]:
            raise ValueError("SHALLOW_VIDEO_BYTES_CHANGED")
        videos[name] = linked
    delivery = once(output / "COMPOSITE_DELIVERY.json", {
        "producer_terminal": value["inputs"]["composite_terminal"],
        "producer_post": value["inputs"]["composite_post"], "composite_receipt": reference,
        "report": report_ref, "videos": videos, "native_frames": 238,
        "geometry_unknown_pixels": report["geometry_unknown_pixels"],
        "object_first_fallback_pixels": report["object_first_fallback_pixels"],
        "model_support_pixels": report["model_support_pixels"],
        "qualified_trusted_occlusion_pixels": 0, "mask_quality": report["mask_quality"],
        "clean_quality": report["clean_quality"], "visual_quality": "PENDING_ACTUAL_AI_REVIEW",
        "geometry": "UNKNOWN_SCENE_DEPTH_ABSENT", "human_review": "PENDING",
        "hidden_regions": "UNKNOWN", "training_eligible": False, "control_ground_truth": False,
        "model_calls": 0, "render_calls": 0, "created_at": now_iso()})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(composite_execution=value["inputs"]["composite_terminal"],
        composite_validation=value["inputs"]["composite_post"], composite_delivery=delivery,
        composite_shallow_videos=videos)
    index["next_actions"].update(
        occlusion="001 full238 actual Composite sealed; two complete videos. Actual AI review must preserve sceneUNKNOWN, model-supportUNVERIFIED, Mask/CleanFAILED and119 MotionHIGH.",
        mask="Independent TEXT wearer support retention CPU engineering. No fresh Mask/model run yet; missing Controller RGB support is not fixed by retaining TEXT.")
    atomic_json(index_path, index)
    return delivery


def seal(junit):
    _, value = packet()
    output = Path(value["output_dir"])
    original = load_json(Path(value["inputs"]["previous_cpu_packet"]["path"]))
    if datetime.now(timezone.utc) >= datetime.fromisoformat(original["deadline_at"]):
        raise ValueError("ORIGINAL_CPU_HARD_DEADLINE_EXHAUSTED")
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in ET.parse(junit).getroot().iter("testsuite"))
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 817 or any(counts[k] for k in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_PREVIOUS_AND_RETENTION_CPU_REGRESSION_REQUIRED")
    reviewed = Path(original["output_dir"]) / "lane_review/AI_REVIEW.json"
    review = load_json(reviewed)
    if (review["actual_viewed_count"] != 12 or review["unviewed_native_frames_count"] != 226
            or review["human_review"] != "PENDING" or review["hidden_regions"] != "UNKNOWN"):
        raise ValueError("ACTUAL_COMPOSITE_REVIEW_REQUIRED")
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    refs = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"]
    refs += [artifact_ref(Path(__file__).resolve()), *[artifact_ref(REPO_ROOT / "tests/unit" / name)
        for name in ("test_inpainting_candidate_text_retention.py", "test_inpainting_candidate_text_retention_execution.py")]]
    archive_path = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            archive.write(reference["path"], str(Path(reference["path"]).relative_to(REPO_ROOT)))
    root_review = output / "ROOT_COMPOSITE_REVIEW_R01.json"
    counterfactual = output / "lane_counterfactual/RESULT.json"
    lane_evidence = [artifact_ref(path) for path in (
        output / "lane_mask/TEXT_RETENTION_ENGINEERING_RESULT.json",
        output / "lane_gate/ROOT_HELPER_INTEGRATION_AUDIT_R01.json",
        output / "lane_mask/CONTACT119_DIAGNOSIS_R01.json") if path.is_file()]
    result = once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-post-g1-text-retention-engineering-v1", "task_id": CPU,
        "run_signature": value["run_signature"], "completed_at": now_iso(),
        "experiment_scope": SCOPE, "single_variable": "text_wearer_retention",
        "candidate_value": STRATEGY, "cpu_tests": counts, "junit": artifact_ref(junit),
        "code": refs, "snapshot": artifact_ref(archive_path),
        "actual_composite_review": artifact_ref(reviewed),
        "root_composite_corroboration": artifact_ref(root_review),
        "counterfactual_audit": artifact_ref(counterfactual) if counterfactual.is_file() else None,
        "lane_evidence": lane_evidence,
        "composite_delivery": artifact_ref(output / "COMPOSITE_DELIVERY.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "counterfactual_is_fresh_mask": False, "models_called": 0, "files_deleted": 0,
        "g1_third_recipe": False, "original_cpu_deadline_at": original["deadline_at"],
        "candidate_adopted": False, "models_changed": False, "thresholds_changed": False,
        "next_task": MODEL})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(post_g1_text_retention_engineering=result,
        composite_actual_ai_review=artifact_ref(reviewed),
        composite_root_corroboration=artifact_ref(root_review))
    index["next_actions"]["occlusion"] = "001 Composite technical complete; actual12/226 unviewed,4HIGHclusters/2MED, sampledFAILED; model object support causes false device foreground. Keep sceneUNKNOWN/trusted0, humanPENDING."
    atomic_json(index_path, index)
    once(output / "MAINTENANCE_DONE.json", {"task_id": CPU, "run_signature": value["run_signature"],
        "status": "PASS", "artifacts": [result, artifact_ref(junit), artifact_ref(archive_path), artifact_ref(reviewed)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def model_spec():
    from chaoyang.governance.inpainting_candidate_registration import (
        visual_route_contract, VISUAL_ROUTE_KEY, validate_post_text_retention_experiment)
    packet_path, value = packet()
    output = Path(value["output_dir"])
    if (load_json(output / "RESULT.json")["scope_status"] != "PASS"
            or load_json(output / "POST_VALIDATION.json")["exit_code"] != 0):
        raise ValueError("SEALED_ENGINEERING_AND_POST_REQUIRED")
    engineering = load_json(output / "ENGINEERING_RESULT.json")
    for reference in engineering["code"]:
        verify_ref(reference)
    parent_terminal = load_json(Path(value["inputs"]["parent_terminal"]["path"]))
    parent_path = REPO_ROOT / "tasks/current" / parent_terminal["task_id"] / "TASK_PACKET.json"
    parent = load_json(parent_path)
    if parent["run_signature"] != parent_terminal["run_signature"]:
        raise ValueError("SEALED_PARENT_RECIPE_IDENTITY_DRIFT")
    config = copy.deepcopy(parent["config"])
    config.update(candidate_id="POST_G1_TEXT_WEARER_RETENTION_A01",
        algorithm_parameter_change="TEXT_HUMAN_GLOVE_SUPPORT_RETENTION_AT_POINT_REPLACEMENT",
        text_wearer_retention=STRATEGY)
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    runtime = [*route["code_closure"], artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    weights = parent["model_weights"]
    for reference in [*runtime, *weights]:
        verify_ref(reference)
    source_index = load_json(Path(value["inputs"]["source_index"]["path"]))
    if ([row["session_key"] for row in source_index["sessions"]] != DEV
            or [row["native_frames"] for row in source_index["sessions"]] != NATIVE):
        raise ValueError("FIXED_SIX_FULL_NATIVE_SOURCES_REQUIRED")
    spec = {"spec_schema": "inpainting-finite-g1-mask-spec-v1", "experiment_scope": SCOPE,
        "task_id": MODEL, "output_dir": str(REPO_ROOT / "_run" / MODEL / "attempt_0001"),
        "infrastructure": value["runtime_code"][1], "runtime_code": runtime,
        "model_weights": weights, "weights": {"files": weights}, "physical_gpu_index": 0,
        "model_stages": ["Mask"], "config": config, "calibration": parent["calibration"],
        "schema": parent["schema"], "inputs": {
            "v1_manifest": value["inputs"]["v1_manifest"], "source_index": value["inputs"]["source_index"],
            "parent_recipe_packet": artifact_ref(parent_path), "parent_terminal": value["inputs"]["parent_terminal"],
            "retention_cpu_packet": artifact_ref(packet_path),
            "retention_cpu_result": artifact_ref(output / "RESULT.json"),
            "retention_cpu_post": artifact_ref(output / "POST_VALIDATION.json"),
            "retention_engineering_result": artifact_ref(output / "ENGINEERING_RESULT.json")},
        "budgets": {"candidates_max": 1, "wall_seconds_max": 7200, "gpu_count": 1,
            "gpu_seconds_max": 7200, "cpu_threads_per_worker": 2, "model_calls_max": 215,
            "model_calls_unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION", "training_steps_max": 0,
            "output_bytes_estimate": 4294967296}, "sessions": [],
        "evaluation_only": {"completed_parent_model_result": parent_terminal["model_result"],
                            "parent_actual_review": value["inputs"]["parent_review"]},
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
                     "inpainting_optimization/MASK_TEXT_RETENTION_A01_SPEC.json"],
        "read_set_expansions": ["Fresh producer reads only current six sealed Raw Source stages and frozen code/weights/global recipe. No old proposals, Mask/Clean, annotation points or session fitting.",
            "Independent post-G1 TEXT support-retention failure-cluster hypothesis; not third G1 recipe, Controller retry or automatic V1 promotion. Evaluator may read old arrays only after new results seal."]}
    for row in source_index["sessions"]:
        reference = row["source_receipt"]
        verify_ref(reference)
        source = load_json(Path(reference["path"]))
        verify_sealed_stage(source, source["component_signature"], output_root=Path(reference["path"]).parent)
        signature = canonical_sha({"source": reference, "code": runtime, "config": config,
            "weights": weights, "calibration": spec["calibration"], "schema": spec["schema"]})
        spec["sessions"].append({"session_key": row["session_key"], "native_frames": row["native_frames"],
            "source_receipt": reference, "mask_signature": signature})
    validate_post_text_retention_experiment(spec, repo_root=REPO_ROOT)
    return once(REPO_ROOT / "inpainting_optimization/MASK_TEXT_RETENTION_A01_SPEC.json", spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("successor-spec", "prepare", "seal-engineering", "model-spec"))
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    print(successor_spec() if args.action == "successor-spec" else prepare() if args.action == "prepare"
        else seal(args.junit) if args.action == "seal-engineering" else model_spec(), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
