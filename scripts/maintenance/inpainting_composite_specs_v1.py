"""Freeze a pixel-local CPU candidate and keep image review separate from truth."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref, verify_sealed_stage

CPU = "inpainting_g2_composite_engineering_20261013"
EXECUTION = "inpainting_composite_001_a01_20261013"


def once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def packet():
    path = REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    value = load_json(path)
    if (value["task_id"] != CPU or value["algorithm_execution_allowed"] is not False
            or value["deletion_allowed"] is not False or value["budgets"]["gpu_count"] != 0):
        raise ValueError("EXACT_FINITE_COMPOSITE_ENGINEERING_PACKET_REQUIRED")
    for reference in value["inputs"].values():
        verify_ref(reference)
    return path, value


def stages(value):
    terminal = load_json(Path(value["inputs"]["geometry_terminal"]["path"]))
    if terminal["status"] != "PASSED" or load_json(Path(value["inputs"]["geometry_post"]["path"]))["exit_code"] != 0:
        raise ValueError("SEALED_GEOMETRY_AND_POST_REQUIRED")
    geometry = load_json(Path(value["inputs"]["geometry_result"]["path"]))
    if geometry["native_frames"] != 238 or geometry["completed_geometry_sessions"] != 1:
        raise ValueError("FIXED001_FULL238_REQUIRED")
    row = geometry["sessions"][0]
    references = {"Source": row["source"], "Render": row["render"],
        "Mask": value["inputs"]["mask_receipt"], "Clean": value["inputs"]["clean_receipt"]}
    signatures = {}
    for name, reference in references.items():
        verify_ref(reference)
        receipt_path = Path(reference["path"])
        receipt = load_json(receipt_path)
        count_ok = (receipt.get("native_frames") == 238 if name in ("Source", "Render")
                    else receipt.get("native_input_frames") == receipt.get("native_output_frames") == 238)
        if receipt["stage"] != name or not count_ok:
            raise ValueError("UPSTREAM_STAGE_OR_NATIVE_FRAME_DRIFT:" + name)
        signatures[name] = receipt["component_signature"]
        verify_sealed_stage(receipt, signatures[name], output_root=receipt_path.parent)
    return references, signatures


def prepare():
    _, value = packet()
    output = Path(value["output_dir"])
    references, _ = stages(value)
    review_paths = {
        "Mask": REPO_ROOT / "_run/inpainting_generalization_g1_mask_wearer_b_20261013/attempt_0001/review_cpu/lane_001/AI_REVIEW_R01.json",
        "Clean": REPO_ROOT / "_run/inpainting_generalization_g2_clean_ai_review_20261013/attempt_0001/REVIEW_RESULT.json"}
    summaries = {}
    for name, path in review_paths.items():
        review = load_json(path)
        verdict = review["ai_sample_visual_status"] if name == "Mask" else review["quality"]["visual_quality"]
        if verdict != "FAILED":
            raise ValueError("DECLARED_FAILED_UPSTREAM_REVIEW_REQUIRED:" + name)
        summaries[name] = once(output / (name.upper() + "_QUALITY_STATUS.json"), {
            "schema": "inpainting-upstream-quality-status-v1", "stage": name,
            "upstream_receipt": references[name], "visual_quality": verdict,
            "review_reference": artifact_ref(path), "human_review": "PENDING",
            "hidden_regions": "UNKNOWN", "pixel_algorithm_input": False,
            "claim_limit": "Compact assessment metadata only. No review points, masks or manual corrections supplied to the pixel producer."})
    render = load_json(Path(references["Render"]["path"]))
    report_ref = next(r for r in render["outputs"] if Path(r["path"]).name == "RENDER_REPORT.json")
    report = load_json(Path(report_ref["path"]))
    videos = {}
    for view, name in (("fpv", "G2_001_Raw_Robot_FPV.mp4"),
                       ("third_person", "G2_001_Robot_FullArm_ThirdPerson.mp4")):
        source = report["review"][view]["video"]
        verify_ref(source)
        target = REPO_ROOT / "review/inpainting_optimization" / name
        if target.exists():
            raise ValueError("IMMUTABLE_REVIEW_ALIAS_ALREADY_EXISTS:" + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source["path"], target)
        linked = artifact_ref(target)
        if linked["bytes"] != source["bytes"] or linked["sha256"] != source["sha256"]:
            raise ValueError("REVIEW_ALIAS_CONTENT_CHANGED")
        videos[view] = linked
    delivery = once(output / "GEOMETRY_DELIVERY.json", {
        "producer_result": value["inputs"]["geometry_terminal"],
        "producer_post": value["inputs"]["geometry_post"],
        "render_receipt": references["Render"], "videos": videos,
        "native_frames": report["native_frames"], "render_calls": report["render_calls"],
        "zbuffer_passes": report["zbuffer_passes"], "full_visual_review": False,
        "scene_geometry": "UNKNOWN", "clean": "NOT_CONSUMED_IN_RENDER",
        "model_calls": 0, "render_calls_in_this_helper": 0,
        "upstream_quality_status": summaries, "created_at": now_iso()})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(motion_render_execution=value["inputs"]["geometry_terminal"],
        motion_render_validation=value["inputs"]["geometry_post"], motion_render_delivery=delivery,
        motion_render_shallow_videos=videos)
    index["next_actions"].update(motion="001 full238 Motion/Render sealed; actual AI sampled review in current CPU packet. Preserve numerical FAIL and uncertain timebase.",
        occlusion="Thin Composite engineering over sealed Source/Mask/Clean/Render. Scene depth ABSENT and predicted object support not truth; actual CPU production separately registered.")
    atomic_json(index_path, index)
    return delivery


def route():
    from chaoyang.governance.inpainting_candidate_registration import composite_route_contract, COMPOSITE_ROUTE_KEY
    return composite_route_contract(REPO_ROOT)[COMPOSITE_ROUTE_KEY]


def seal(junit):
    path, value = packet()
    output = Path(value["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in ET.parse(junit).getroot().iter("testsuite"))
              for k in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 719 or any(counts[k] for k in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_PREVIOUS_AND_COMPOSITE_REGRESSION_REQUIRED")
    reviewed = output / "lane_review/AI_REVIEW.json"
    review = load_json(reviewed)
    if (review["native_frames"] != 238 or review["actual_viewed_count"] != 12
            or review["unviewed_native_frames_count"] != 226 or review["human_review"] != "PENDING"
            or review["hidden_regions"] != "UNKNOWN" or review["full_video_review"] is not False):
        raise ValueError("ACTUAL_BOUNDED_GEOMETRY_REVIEW_REQUIRED")
    corroboration = output / "ROOT_ACTUAL_REVIEW_R01.json"
    root_review = load_json(corroboration)
    if (root_review["actual_viewed_count"] != 12
            or root_review["unviewed_native_frames_count"] != 226
            or root_review["ai_sample_visual_status"] != "FAILED"):
        raise ValueError("INDEPENDENT_ROOT_IMAGE_CORROBORATION_REQUIRED")
    refs = route()["code_closure"] + [artifact_ref(Path(__file__).resolve())]
    refs += [artifact_ref(REPO_ROOT / "tests/unit" / name) for name in (
        "test_inpainting_candidate_composite.py", "test_inpainting_candidate_composite_execution.py")]
    snapshot = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            archive.write(reference["path"], str(Path(reference["path"]).relative_to(REPO_ROOT)))
    result = once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-composite-engineering-evidence-v1", "task_id": CPU,
        "run_signature": value["run_signature"], "completed_at": now_iso(),
        "cpu_tests": counts, "junit": artifact_ref(junit), "code": refs,
        "snapshot": artifact_ref(snapshot), "actual_geometry_review": artifact_ref(reviewed),
        "root_geometry_corroboration": artifact_ref(corroboration),
        "upstream_contract_audit": artifact_ref(output / "lane_composite/INTEGRATION_AUDIT_R01.json"),
        "geometry_delivery": artifact_ref(output / "GEOMETRY_DELIVERY.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "algorithm_calls": 0, "model_calls": 0, "gpu_count": 0, "files_deleted": 0,
        "candidate_adopted": False, "thresholds_changed": False, "next_task": EXECUTION})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(composite_engineering=result,
        motion_render_actual_review=artifact_ref(reviewed),
        motion_render_root_review=artifact_ref(corroboration))
    index["next_actions"].update(
        motion="001 native238 Motion/Render complete; actual12/226 unviewed;119 grasp-screen HIGH, sampledFAILED. Keep numericFAIL0 and prior/timebase limits separate.",
        occlusion="817 complete CPU regression PASS; actual full001 Composite requires new finite packet; no scene geometry qualification or known Mask/Clean upgrade.")
    atomic_json(index_path, index)
    once(output / "MAINTENANCE_DONE.json", {"task_id": CPU, "run_signature": value["run_signature"],
        "status": "PASS", "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot), artifact_ref(reviewed), artifact_ref(corroboration)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def execution_spec():
    from chaoyang.governance.inpainting_candidate_registration import (
        COMPOSITE_SPEC_SCHEMA, COMPOSITE_SCOPE, COMPOSITE_OPERATION)
    path, value = packet()
    output = Path(value["output_dir"])
    terminal = load_json(output / "RESULT.json")
    if terminal["status"] != "PASSED" or terminal["scope_status"] != "PASS" or load_json(output / "POST_VALIDATION.json")["exit_code"] != 0:
        raise ValueError("SEALED_ENGINEERING_AND_POST_REQUIRED")
    engineering = load_json(output / "ENGINEERING_RESULT.json")
    for reference in engineering["code"]:
        verify_ref(reference)
    upstream, signatures = stages(value)
    quality = {name: artifact_ref(output / (name.upper() + "_QUALITY_STATUS.json")) for name in ("Mask", "Clean")}
    config = {"scene_depth": "ABSENT", "display_policy": "RAW_MODEL_OBJECT_FIRST_DIAGNOSTIC",
        "image_size": [640, 480], "complete_robot": True}
    runtime = route()["code_closure"]
    calibration = {"scene_depth": "ABSENT", "scene_robot_depth_comparability": False,
        "robot_depth": "FROZEN_COMPLETE_ROBOT_CAMERA_Z_ONLY",
        "timebase": load_json(Path(upstream["Source"]["path"]))["timebase_status"]}
    schema = {"result": "inpainting-composite-batch-result-v1",
        "component": "inpainting-candidate-component-v1", "support": "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"}
    component = canonical_sha({"upstream": upstream, "upstream_signatures": signatures,
        "quality_metadata": quality, "config": config, "calibration": calibration, "schema": schema, "code": runtime})
    row = {"session_key": "001", "native_frames": 238, "composite_signature": component,
        "upstream_quality_evidence": quality}
    for name in upstream:
        row[name.lower() + "_receipt"] = upstream[name]
        row[name.lower() + "_signature"] = signatures[name]
    spec = {"spec_schema": COMPOSITE_SPEC_SCHEMA, "experiment_scope": COMPOSITE_SCOPE,
        "task_id": EXECUTION, "operation": COMPOSITE_OPERATION,
        "output_dir": str(REPO_ROOT / "_run" / EXECUTION / "attempt_0001"),
        "infrastructure": value["runtime_code"][1], "runtime_code": runtime,
        "inputs": {**{k: value["inputs"][k] for k in ("v1_manifest", "manus0916_pointer", "geometry_terminal", "geometry_post")},
            "engineering_packet": artifact_ref(path), "engineering_result": artifact_ref(output / "ENGINEERING_RESULT.json"),
            "engineering_terminal": artifact_ref(output / "RESULT.json"), "engineering_post": artifact_ref(output / "POST_VALIDATION.json")},
        "config": config, "calibration": calibration, "schema": schema, "stages": ["Composite"],
        "budgets": {"wall_seconds_max": 7200, "gpu_count": 0, "model_calls_max": 0,
            "training_steps_max": 0, "cpu_threads_per_worker": 2, "render_calls_max": 0,
            "frames_max": 238, "review_videos": 2, "candidates_max": 1, "output_bytes_estimate": 2147483648},
        "sessions": [row], "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json", "inpainting_optimization/G2_COMPOSITE_001_SPEC.json"],
        "read_set_expansions": ["Only declared sealed current Source/Mask/Clean/Render and compact quality metadata. No evaluation points or manual seeds enter the pixel producer."],
        "control_ground_truth": False, "candidate_adopted": False}
    return once(REPO_ROOT / "inpainting_optimization/G2_COMPOSITE_001_SPEC.json", spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal-engineering", "execution-spec"))
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    print(prepare() if args.action == "prepare" else seal(args.junit) if args.action == "seal-engineering" else execution_spec(), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
