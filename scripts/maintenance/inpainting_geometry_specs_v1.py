"""Seal CPU engineering and bind one independent, existing-core Geometry run.

No algorithm execution, model loading, new publisher or default adoption here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref, verify_sealed_stage

CPU = "inpainting_motion_render_engineering_20261013"
EXECUTION = "inpainting_motion_render_001_a01_20261013"


def once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def packet():
    path = REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    value = load_json(path)
    if (value["task_id"] != CPU or value["algorithm_execution_allowed"] is not False
            or value["deletion_allowed"] is not False or value["budgets"]["gpu_count"] != 0
            or value["budgets"]["model_calls_max"] != 0):
        raise ValueError("EXACT_FINITE_GEOMETRY_ENGINEERING_PACKET_REQUIRED")
    for pin in value["inputs"].values():
        verify_ref(pin)
    return path, value


def route():
    from chaoyang.governance.inpainting_candidate_registration import geometry_route_contract, GEOMETRY_ROUTE_KEY
    return geometry_route_contract(REPO_ROOT)[GEOMETRY_ROUTE_KEY]


def review_index():
    path, value = packet()
    root = REPO_ROOT / "_run/inpainting_post_g1_controller_rgb_model_a01_20261013/attempt_0001"
    review_ref = artifact_ref(root / "review_cpu/REVIEW_ROLLUP.json")
    reviewed = load_json(Path(review_ref["path"]))
    terminal = load_json(root / "RESULT.json")
    if (terminal["status"] != "PASSED" or load_json(root / "POST_VALIDATION.json")["exit_code"] != 0
            or reviewed["run_signature"] != terminal["run_signature"]
            or reviewed["native_frames"] != 1033 or reviewed["actual_AI_viewed_distinct_frames"] != 36):
        raise ValueError("SEALED_ACTUAL_CONTROLLER_REVIEW_REQUIRED")
    for row in reviewed["sessions"]:
        for key in ("ai_review", "native_comparison", "export", "review_video"):
            verify_ref(row[key])
    record = once(Path(value["output_dir"]) / "CONTROLLER_REVIEW_LINK.json", {
        "task_id": CPU, "run_signature": value["run_signature"],
        "producer_result": artifact_ref(root / "RESULT.json"),
        "post_validation": artifact_ref(root / "POST_VALIDATION.json"),
        "actual_review": review_ref, "next_action": "Independent Motion/Render engineering; not another text retry.",
        "model_calls": 0, "candidate_adopted": False, "created_at": now_iso()})
    target = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(target)
    index.update(post_controller_actual_review=review_ref, post_controller_review_link=record,
        post_controller_execution=artifact_ref(root / "RESULT.json"),
        post_controller_validation=artifact_ref(root / "POST_VALIDATION.json"),
        post_controller_model_result=artifact_ref(root / "MODEL_RESULT.json"),
        motion_render_engineering_task=artifact_ref(path),
        post_controller_latest_facts={key: reviewed[key] for key in (
            "completed_sessions", "native_frames", "actual_AI_viewed_distinct_frames",
            "unviewed_native_frames", "AI_sample_visual_qualified_sessions", "HIGH_session_clusters",
            "MED_session_clusters", "all_native_mask_arrays_equal_to_B", "algorithm_gain")})
    index["next_actions"].update(
        mask="Controller A01 complete six/full1033, identical to B; actual36 reviewed/997 unviewed,0of6 qualified,HIGH12/MED6. No adoption or further prompt retry.",
        motion="Independent CPU engineering RUNNING; existing raw bridge/retarget plus per-frame provenance. Actual001 separately registered after full tests seal.",
        occlusion="Full-link Render engineering RUNNING; actual buffers not yet produced. Scene depth UNKNOWN, Mask failure does not block Motion/Render.",
        publisher="Current engineering Task Packet and live governance INDEX authoritative; preserve V1,0916 and ICT.")
    atomic_json(target, index)
    return record


def seal(junit):
    path, value = packet()
    output = Path(value["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in ET.parse(junit).getroot().iter("testsuite"))
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 567 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_EXISTING_AND_GEOMETRY_REGRESSION_REQUIRED")
    refs = route()["code_closure"] + [artifact_ref(Path(__file__).resolve())]
    refs += [artifact_ref(REPO_ROOT / "tests/unit" / name) for name in (
        "test_inpainting_candidate_motion.py", "test_inpainting_candidate_render.py",
        "test_inpainting_candidate_geometry_execution.py")]
    snapshot = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for pin in refs:
            verify_ref(pin)
            archive.write(pin["path"], str(Path(pin["path"]).relative_to(REPO_ROOT)))
    review = REPO_ROOT / "_run/inpainting_post_g1_controller_rgb_model_a01_20261013/attempt_0001/review_cpu/REVIEW_ROLLUP.json"
    reviewed = load_json(review)
    if reviewed["native_frames"] != 1033 or reviewed["actual_AI_viewed_distinct_frames"] != 36:
        raise ValueError("ACTUAL_FIXED_CONTROLLER_REVIEW_REQUIRED")
    audits = [artifact_ref(output / relative) for relative in (
        "lane_motion_audit/SPEC_SEAL_AUDIT.json",
        "lane_render_audit/FINAL_EXECUTOR_BINDING_R01.json")]
    for pin in audits:
        verify_ref(pin)
    result = once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-motion-render-engineering-evidence-v1", "task_id": CPU,
        "run_signature": value["run_signature"], "completed_at": now_iso(),
        "cpu_tests": counts, "junit": artifact_ref(junit), "code": refs,
        "snapshot": artifact_ref(snapshot), "controller_actual_review": artifact_ref(review),
        "static_integration_audits": audits,
        "asset_closure": value["inputs"]["asset_closure"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "algorithm_calls": 0, "model_calls": 0, "gpu_count": 0, "files_deleted": 0,
        "candidate_adopted": False, "thresholds_changed": False,
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "visual_quality": "NOT_EVALUATED",
        "next_task": EXECUTION,
        "next_action": "Separately register CPU001/full238 Motion and full-link Render; Mask failures do not block this independent consumer."})
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(motion_render_engineering=result, motion_render_cpu_junit=artifact_ref(junit),
        post_controller_actual_review=artifact_ref(review),
        post_controller_latest_facts={key: reviewed[key] for key in (
            "completed_sessions", "native_frames", "actual_AI_viewed_distinct_frames",
            "unviewed_native_frames", "AI_sample_visual_qualified_sessions", "HIGH_session_clusters",
            "MED_session_clusters", "all_native_mask_arrays_equal_to_B", "algorithm_gain")})
    index["next_actions"].update(
        mask="Independent short Controller prompt executed six/full1033; arrays exactly B, 36 actually reviewed/997 unseen, 0of6 qualified. Seal no gain, no more prompt retry. New proposal strategy only in another bounded packet.",
        motion="Engineering sealed with existing native raw bridge/retarget. Register independent CPU001/full238 with immutable component/asset/camera signatures; preserve per-frame validity and TIMEBASE_UNVERIFIED.",
        occlusion="Execute complete_robot=True full-link Render in separate CPU task, seal RGBA/depth/link/triangle and fixed third-person diagnostic. Scene depth UNKNOWN; failed Mask does not block this consumer.",
        publisher="Controller MODEL_RESULT/RESULT/POST0 and actual six-review rollup sealed; geometry engineering full CPU regression is separate. Preserve V1,0916,ICT and follow live governance INDEX.")
    atomic_json(index_path, index)
    once(output / "MAINTENANCE_DONE.json", {"task_id": CPU, "run_signature": value["run_signature"],
        "status": "PASS", "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot), artifact_ref(review)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def execution_spec():
    from chaoyang.governance.inpainting_candidate_registration import (
        GEOMETRY_SPEC_SCHEMA, GEOMETRY_SCOPE, GEOMETRY_OPERATION)
    path, value = packet()
    output = Path(value["output_dir"])
    terminal = load_json(output / "RESULT.json")
    post = load_json(output / "POST_VALIDATION.json")
    if terminal["status"] != "PASSED" or terminal["scope_status"] != "PASS" or post["exit_code"] != 0:
        raise ValueError("SEALED_ENGINEERING_AND_POST_PASS_REQUIRED")
    engineering = load_json(output / "ENGINEERING_RESULT.json")
    for pin in engineering["code"]:
        verify_ref(pin)
    source_index = load_json(Path(value["inputs"]["source_index"]["path"]))
    source_row = next(row for row in source_index["sessions"] if row["session_key"] == "001")
    if source_row["native_frames"] != 238:
        raise ValueError("FIXED001_FULL238_REQUIRED")
    source_ref = source_row["source_receipt"]
    verify_ref(source_ref)
    source_path = Path(source_ref["path"])
    source = load_json(source_path)
    verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
    report_pin = next(pin for pin in source["outputs"] if Path(pin["path"]).name == "SOURCE_REPORT.json")
    report = load_json(Path(report_pin["path"]))
    manifest = load_json(Path(value["inputs"]["v1_manifest"]["path"]))
    camera_contract = {"v1_manifest": value["inputs"]["v1_manifest"], "camera_config": manifest["inputs"]["camera"]}
    camera_config = load_json(Path(camera_contract["camera_config"]["path"]))
    serial = report["camera"]["camera_device_serial"]
    selected = camera_config["devices"][serial]
    visual_evidence = {key: selected[key] for key in (
        "correction_evidence", "installation_correction_evidence") if key in selected}
    asset_audit = load_json(Path(value["inputs"]["asset_closure"]["path"]))
    if asset_audit["asset_count"] != 69 or asset_audit["issues"]:
        raise ValueError("FROZEN_ROBOT_ASSET_CLOSURE_REQUIRED")
    assets = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in asset_audit["assets"]]
    runtime = route()["code_closure"]
    for pin in [*runtime, *assets, *camera_contract.values(), *visual_evidence.values(),
                *report["actual_inputs"].values()]:
        verify_ref(pin)
    config = {"complete_robot": True, "image_size": [640, 480],
        "motion_rule": "FROZEN_V1_RETARGET_AUTOMATIC_BASE", "scene_depth": "ABSENT"}
    calibration = {"camera": report["actual_inputs"]["camera"],
        "installation_container": report["actual_inputs"]["installation_container"],
        "camera_contract": camera_contract, "metric_qualification": "NOT_VERIFIED",
        "frozen_visual_metadata_evidence": visual_evidence,
        "scene_depth": "ABSENT", "timebase": source["timebase_status"]}
    schema = {"result": "inpainting-geometry-batch-result-v1",
        "component": "inpainting-candidate-component-v1", "motion": "NATIVE_CONTROLLER_HEAD_MANUS_VISUAL_ROBOT",
        "render": "RGBA_CAMERA_Z_DEPTH_LINK_TRIANGLE_FULL_ROBOT"}
    common = {"source": source_ref, "raw_inputs": report["actual_inputs"],
        "camera_contract": camera_contract, "robot_assets": assets, "calibration": calibration, "schema": schema}
    # Motion and Render never depend on Mask or Clean; changed downstream code
    # cannot turn those old arrays into implicit new inputs.
    from chaoyang.inpainting_candidate.motion import CODE_CLOSURE as MOTION_CODE
    from chaoyang.inpainting_candidate.motion_source import PARENT_CODE as NATIVE_CODE
    motion_paths = {str(REPO_ROOT / relative) for relative in (*MOTION_CODE, *NATIVE_CODE,
        "src/chaoyang/inpainting_candidate/motion.py")}
    motion_code = [pin for pin in runtime if pin["path"] in motion_paths]
    if {pin["path"] for pin in motion_code} != motion_paths:
        raise ValueError("ACTUAL_MOTION_COMPONENT_CODE_CLOSURE_INCOMPLETE")
    motion_sig = canonical_sha({**common, "stage": "Motion", "code": motion_code,
        "config": {"motion_rule": config["motion_rule"]}})
    render_sig = canonical_sha({**common, "stage": "Render", "upstream_motion": motion_sig,
        "code": runtime, "config": config})
    spec = {"spec_schema": GEOMETRY_SPEC_SCHEMA, "experiment_scope": GEOMETRY_SCOPE,
        "task_id": EXECUTION, "operation": GEOMETRY_OPERATION,
        "output_dir": str(REPO_ROOT / "_run" / EXECUTION / "attempt_0001"),
        "infrastructure": value["runtime_code"][1],
        "inputs": {"v1_manifest": value["inputs"]["v1_manifest"],
            "source_index": value["inputs"]["source_index"], "asset_closure": value["inputs"]["asset_closure"],
            "engineering_packet": artifact_ref(path), "engineering_result": artifact_ref(output / "ENGINEERING_RESULT.json"),
            "engineering_terminal": artifact_ref(output / "RESULT.json"),
            "engineering_post": artifact_ref(output / "POST_VALIDATION.json")},
        "runtime_code": runtime, "robot_assets": assets, "config": config,
        "calibration": calibration, "schema": schema, "stages": ["Motion", "Render"],
        "budgets": {"wall_seconds_max": 7200, "gpu_count": 0, "model_calls_max": 0,
            "training_steps_max": 0, "cpu_threads_per_worker": 2, "render_frames_max": 238,
            "render_views": 2, "render_calls_max": 476,
            "candidates_max": 1, "output_bytes_estimate": 4294967296},
        "sessions": [{"session_key": "001", "native_frames": 238, "source_receipt": source_ref,
            "source_signature": source["component_signature"], "motion_signature": motion_sig,
            "render_signature": render_sig, "raw_inputs": report["actual_inputs"], "camera_contract": camera_contract}],
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
            "inpainting_optimization/MOTION_RENDER_001_SPEC.json"],
        "read_set_expansions": ["Only declared sealed Source, this capture raw refs, current V1 camera/installation metadata and frozen robot assets/core. No old q/Mask/Clean/seeds/placement."],
        "control_ground_truth": False, "candidate_adopted": False}
    return once(REPO_ROOT / "inpainting_optimization/MOTION_RENDER_001_SPEC.json", spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review-index", "seal-engineering", "execution-spec"))
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    result = (review_index() if args.action == "review-index" else
              seal(args.junit) if args.action == "seal-engineering" else execution_spec())
    print(result, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
