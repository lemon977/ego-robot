"""Candidate CPU metadata publisher; no model, retarget or render execution."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from chaoyang.governance.common import REPO_ROOT
from chaoyang.governance.inpainting_candidate_registration import (_active_packet, _snapshot,
    POST_SOURCE_AWARE_SCOPE, SOURCE_AWARE_SPEC_SCHEMA, SOURCE_AWARE_RETENTION_VETO, validate_post_source_aware_experiment,
    visual_route_contract, geometry_route_contract, GEOMETRY_ROUTE_KEY, VISUAL_ROUTE_KEY,
    validate_geometry_spec)
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_ref, verify_sealed_stage

TASK = "inpainting_native_role_engineering_20261013"
G3_TASK = "inpainting_generalization_g3_mask_a01_20261013"


def require(value, code):
    if not value:
        raise ValueError(code)


def load(path):
    return json.loads(Path(path).read_text())


def cpu_packet():
    _, state, index, _ = _snapshot(REPO_ROOT)
    packet = _active_packet(REPO_ROOT, state,
        [row for row in index["task_packets"] if row.get("execution_allowed") is True])
    require(packet["task_id"] == TASK and packet["execution_mode"] == "CPU_DEVELOPMENT_ONLY"
        and packet["budgets"]["gpu_count"] == packet["budgets"]["model_calls_max"] == 0,
        "OWN_FINITE_CPU_TASK_REQUIRED")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]),
        "CPU_DEADLINE")
    for reference in packet["inputs"].values():
        verify_ref(reference)
    return packet


def normalise_review(path, count):
    review = load(path)
    schema = review.get("schema")
    if schema == "inpainting-g3-actual-anchored-mask-ai-review-v1":
        frames = review["viewed_native_frames"]
        claimed = review["viewed_frames_count"]
        high, med = review["issue_counts"]["HIGH"], review["issue_counts"]["MED"]
        quality = review["visual_quality"]
    elif schema == "inpainting-g3-actual-ai-mask-review-v1":
        frames = review["actual_viewed_frames"]
        claimed = review["actual_viewed_count"]
        high, med = review["HIGH_cluster_count"], review["MED_cluster_count"]
        quality = review["ai_sample_visual_status"]
    elif schema == "inpainting-g3-mask-ai-actual-review-v1":
        frames = review["coverage"]["viewed_frames"]
        claimed = review["coverage"]["viewed_count"]
        high, med = review["high_count"], review["medium_count"]
        quality = review["visual_quality"]
    else:
        raise ValueError("REVIEW_SCHEMA_NOT_SUPPORTED:" + str(schema))
    expected = sorted({*((count - 1) * k // 5 for k in range(6)),
        (count - 1) // 2 - 1, (count - 1) // 2, (count - 1) // 2 + 1})
    require(sorted(frames) == expected and len(set(frames)) == claimed == 9,
        "EXACT_PRE_FROZEN_ACTUAL_VIEW_SCOPE_REQUIRED")
    require(type(high) is type(med) is int and min(high, med) >= 0,
        "ACTUAL_ISSUE_COUNTS_REQUIRED")
    require(not high or quality == "FAILED", "HIGH_MUST_BE_FAILED")
    require(review["hidden_regions"] == "UNKNOWN" and review["human_review"] == "PENDING"
        and review["training_eligible"] is False, "REVIEW_CANNOT_UPGRADE_QUALIFICATION")
    return {"reference": file_ref(path), "actual_viewed_frames": frames,
        "actual_viewed_count": claimed, "unviewed_native_frames": count - claimed,
        "HIGH": high, "MED": med, "visual_quality": quality,
        "hidden_regions": "UNKNOWN", "human_review": "PENDING"}


def publish_g3(revision):
    packet = cpu_packet()
    require(revision in {"R01", "R02"}, "FINITE_PUBLICATION_REVISION_REQUIRED")
    outer = load(packet["inputs"]["g3_terminal"]["path"])
    model = load(packet["inputs"]["g3_model"]["path"])
    post = load(packet["inputs"]["g3_post"]["path"])
    require(outer["task_id"] == G3_TASK and outer["status"] == "PASSED"
        and outer["model_result"] == packet["inputs"]["g3_model"] and post["exit_code"] == 0,
        "G3_ACTUAL_SEALED_EXECUTION_REQUIRED")
    rows = model["sessions"]
    require(len(rows) == outer["completed_mask_sessions"] == 11
        and sum(row["native_frames"] for row in rows) == 1400,
        "G3_FULL_FROZEN_11_1400_REQUIRED")
    model_root = Path(packet["inputs"]["g3_model"]["path"]).parent
    shallow = REPO_ROOT / "review/inpainting_optimization"
    delivered = []
    for row in rows:
        require(row["execution"] == "COMPLETE_MASK_ONLY", "MASK_NOT_COMPLETE")
        for name in ("source", "mask"):
            reference = row[name]
            verify_ref(reference)
            stage = load(reference["path"])
            require(stage["stage"] == name.capitalize(), "EXACT_STAGE_ROLE_REQUIRED")
            verify_sealed_stage(stage, stage["component_signature"],
                output_root=Path(reference["path"]).parent)
        video = row["review"]["video"]
        verify_ref(video)
        source = Path(video["path"])
        require(source.resolve(strict=True) == source and source.is_relative_to(model_root)
            and row["review"]["decoded_frames"] == row["native_frames"],
            "SIGNED_COMPLETE_VIDEO_REQUIRED")
        target = shallow / ("G3_" + row["session_key"] + "_Raw_Mask.mp4")
        if not target.exists():
            os.link(source, target)
        require(file_ref(target)["sha256"] == video["sha256"], "SHALLOW_VIDEO_COLLISION")
        key = row["session_key"]
        lane = ("lane_challenge" if key.startswith("pick_place_") else
            "lane_0918_a" if int(key.rsplit("_", 1)[1]) <= 4 else "lane_0918_b")
        review = model_root / "review_cpu" / lane / key / "AI_REVIEW.json"
        status = normalise_review(review, row["native_frames"]) if review.exists() else {
            "visual_quality": "NOT_REVIEWED", "actual_viewed_count": 0,
            "unviewed_native_frames": row["native_frames"], "HIGH": 0, "MED": 0}
        delivered.append({"session_key": key, "native_frames": row["native_frames"],
            "source": row["source"], "mask": row["mask"], "video": video,
            "shallow_video": file_ref(target), "review": status})
    result = {"schema": "inpainting-g3-cpu-shallow-delivery-v1", "revision": revision,
        "publisher_task_id": TASK, "publisher_run_signature": packet["run_signature"],
        "frozen_model_task": G3_TASK, "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1",
        "execution_complete_sessions": 11, "native_frames": 1400,
        "actual_adapter_API_attempts": model["model_call_accounting"]["attempted_api_calls"],
        "reviewed_sessions": sum(row["review"]["actual_viewed_count"] > 0 for row in delivered),
        "actual_unique_viewed_frames": sum(row["review"]["actual_viewed_count"] for row in delivered),
        "unviewed_native_frames": sum(row["review"]["unviewed_native_frames"] for row in delivered),
        "failed_sessions": sum(row["review"]["visual_quality"] == "FAILED" for row in delivered),
        "HIGH": sum(row["review"]["HIGH"] for row in delivered),
        "MED": sum(row["review"]["MED"] for row in delivered),
        "sessions": delivered, "new_models_or_algorithm_calls_by_publisher": 0,
        "Clean_Motion_Render_Composite": "NOT_EXECUTED_BY_G3_MASK_TASK",
        "candidate_adopted": False, "training_eligible": False,
        "human_review": "PENDING", "hidden_regions": "UNKNOWN",
        "full_video_visual_continuity": "NOT_EVALUATED",
        "root_repeat_views_not_double_counted": True,
        "execution": packet["inputs"]["g3_terminal"], "model": packet["inputs"]["g3_model"],
        "post": packet["inputs"]["g3_post"],
        "created_at": datetime.now(timezone.utc).isoformat()}
    output = Path(packet["output_dir"]) / ("G3_DELIVERY_" + revision + ".json")
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"delivery": file_ref(output),
        **{key: result[key] for key in ("reviewed_sessions", "actual_unique_viewed_frames",
            "unviewed_native_frames", "failed_sessions", "HIGH", "MED")}}, ensure_ascii=False))


def json_once(path, value):
    require(path.is_absolute() and path.is_relative_to(REPO_ROOT)
        and path.parent.resolve(strict=True) == path.parent, "REAL_PROJECT_METADATA_OUTPUT_REQUIRED")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def freeze_next_specs():
    """Freeze pending finite specs only. No registration or algorithm call."""
    packet = cpu_packet()
    parent_path = REPO_ROOT / "tasks/current/inpainting_post_g1_text_retention_model_a01_20261013/TASK_PACKET.json"
    parent = load(parent_path)
    parent_ref = file_ref(parent_path)
    parent_spec_ref = parent["inputs"]["spec"]
    verify_ref(parent_spec_ref)
    spec = copy.deepcopy(load(parent_spec_ref["path"]))
    spec.update(task_id="inpainting_post_g1_source_aware_mask_a01_20261013",
        experiment_scope=POST_SOURCE_AWARE_SCOPE, spec_schema=SOURCE_AWARE_SPEC_SCHEMA,
        output_dir=str(REPO_ROOT / "_run/inpainting_post_g1_source_aware_mask_a01_20261013/attempt_0001"))
    spec["inputs"] = {"source_index": parent["inputs"]["source_index"], "parent_recipe_packet": parent_ref}
    spec["evaluation_only"] = {
        "feeds_producer": False, "read_only_after_fresh_stage_sealed": True,
        "completed_parent_model_result": file_ref(Path(parent["output_dir"]) / "MODEL_RESULT.json"),
        "parent_actual_review": file_ref(REPO_ROOT /
            "_run/inpainting_generalization_g3_freeze_engineering_20261013/attempt_0001/TEXT_RETENTION_REVIEW_ROLLUP.json"),
        "review_scope": "Original pre-frozen36 development anchors, including the documented050/native127 normal-control regression; retain original known failures and all unviewed frames.",
        "benefit_target_not_training_gate": {"original_HIGH_clusters": 12, "target_max": 6,
            "normal_control_new_HIGH_allowed": 0, "protected_pixels_modified_allowed": 0}}
    spec["config"].update(candidate_id="POST_G1_SOURCE_AWARE_VETO_A01",
        algorithm_parameter_change="SOURCE_AWARE_RETENTION_VETO_ONLY",
        source_aware_retention_veto=SOURCE_AWARE_RETENTION_VETO)
    spec["runtime_code"] = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"]
    spec["read_set"] = ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
        "inpainting_optimization/SOURCE_AWARE_MASK_MODEL_SPEC.json", "inpainting_optimization/CONTINUE_HERE_ZH.md",
        "src/chaoyang/inpainting_candidate/model_worker.py", "src/chaoyang/inpainting_candidate/automatic_masks.py",
        "src/chaoyang/inpainting_candidate/model_stages.py"]
    spec["read_set_expansions"] = [
        "Read only exact Source receipt/report/Raw/model/code/manifest signatures; parent is recipe metadata only.",
        "No previous Mask/proposals/Clean, evaluator points, per-session fitting, new prompt or numerical setting.",
        "Evaluator may read frozen baseline outputs only after fresh producer stages are sealed. Hidden UNKNOWN, human PENDING and original quality FAIL remain. New Mask is not a new Clean/Robot product."]
    for row in spec["sessions"]:
        row["mask_signature"] = canonical_sha({"stage": "Mask", "source": row["source_receipt"],
            "models": spec["model_weights"], "runtime_code": spec["runtime_code"],
            "config": spec["config"], "schema": spec["schema"]})
    validate_post_source_aware_experiment(spec, repo_root=REPO_ROOT)
    index = load(packet["inputs"]["frozen_sources"]["path"])
    geometry_code = geometry_route_contract(REPO_ROOT)[GEOMETRY_ROUTE_KEY]["code_closure"]
    config = {"native_rule": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "camera_binding": "NOT_REQUESTED"}
    rows = []
    for source in index["sessions"]:
        require(source["source_status"] == "READY", "NATIVE_AUDIT_ONLY_EXACT_READY_SOURCE")
        sealed = load(source["source_receipt"]["path"])
        report_ref = next(ref for ref in sealed["outputs"] if Path(ref["path"]).name == "SOURCE_REPORT.json")
        report = load(report_ref["path"])
        row = {k: source[k] for k in ("session_key", "session_id", "native_frames", "source_receipt",
            "source_signature", "frame_map_ref")}
        row["raw_inputs"] = report["actual_inputs"]
        row["native_signature"] = canonical_sha({"stage": "MotionSource", "source": row["source_receipt"],
            "raw_inputs": row["raw_inputs"], "config": config, "runtime_code": geometry_code})
        rows.append(row)
    native = {"spec_schema": "inpainting-finite-native-geometry-spec-v1",
        "experiment_scope": "INPAINTING_NATIVE_MOTION_RENDER_CPU",
        "task_id": "inpainting_native_motion_render_audit_20261013",
        "operation": "run_inpainting_candidate_geometry_stages_v1",
        "output_dir": str(REPO_ROOT / "_run/inpainting_native_motion_render_audit_20261013/attempt_0001"),
        "stages": ["Native"], "sessions": rows, "config": config,
        "inputs": {"frozen_sources": packet["inputs"]["frozen_sources"]},
        "runtime_code": geometry_code, "infrastructure": load(packet["inputs"]["spec"]["path"])["infrastructure"],
        "calibration": {"metric_projection": "NOT_CONSUMED_NATIVE_ONLY", "camera_binding": "NOT_REQUESTED"},
        "schema": {"result": "inpainting-native-geometry-batch-result-v1"},
        "budgets": {"wall_seconds_max": 7200, "gpu_count": 0, "model_calls_max": 0,
            "training_steps_max": 0, "cpu_threads_per_worker": 2, "candidates_max": 1,
            "render_calls_max": 0, "zbuffer_passes_max": 0, "output_bytes_estimate": 33554432},
        "read_set": ["AGENTS.md", "inpainting_optimization/NATIVE_MOTION_AUDIT_SPEC.json",
            "inpainting_optimization/CONTINUE_HERE_ZH.md", "inpainting_v1/MANIFEST.json",
            "src/chaoyang/inpainting_candidate/motion_source.py", "src/chaoyang/inpainting_candidate/geometry_worker.py",
            "src/chaoyang/inpainting_candidate/native_contracts.py", "src/chaoyang/governance/inpainting_candidate_registration.py"],
        "read_set_expansions": ["Follow only exact current signed Raw/Source/map and implementation references. Native-only consumes no old q, fit, Mask/Clean, Robot assets or camera K. Read mapped actual Controller/Head/MANUS support, preserve missing validity. This finite metadata/array production does not validate full Robot, geometry or control."]}
    validate_geometry_spec(native, repo_root=REPO_ROOT)
    targets = [REPO_ROOT / "inpainting_optimization/SOURCE_AWARE_MASK_MODEL_SPEC.json",
        REPO_ROOT / "inpainting_optimization/NATIVE_MOTION_AUDIT_SPEC.json",
        Path(packet["output_dir"]) / "NEXT_FINITE_SPECS.json"]
    require(not any(path.exists() for path in targets), "PENDING_SPECS_IMMUTABLE_ALREADY_EXIST")
    source_aware_ref = json_once(targets[0], spec)
    native_ref = json_once(targets[1], native)
    result = {"schema": "inpainting-native-pending-spec-freeze-v1", "execution_allowed": False,
        "requires_separate_registration_after_current_cpu_task_sealed": True,
        "source_aware": source_aware_ref, "native_audit": native_ref,
        "model_execution_or_motion_render_calls": 0, "default_v1_unchanged": True,
        "publisher_task": packet["task_id"], "publisher_run_signature": packet["run_signature"]}
    print(json.dumps(json_once(Path(packet["output_dir"]) / "NEXT_FINITE_SPECS.json", result), ensure_ascii=False))


def freeze_native_runtime_revision():
    """Preserve the original pending spec; pin the runner's resource helper."""
    packet = cpu_packet()
    output = Path(packet["output_dir"])
    previous = load(output / "NEXT_FINITE_SPECS.json")
    for key in ("source_aware", "native_audit"):
        verify_ref(previous[key])
    validate_post_source_aware_experiment(load(previous["source_aware"]["path"]), repo_root=REPO_ROOT)
    native = copy.deepcopy(load(previous["native_audit"]["path"]))
    require(native["stages"] == ["Native"], "NATIVE_REVISION_ONLY_NO_RENDER")
    resource = file_ref(REPO_ROOT / "src/chaoyang/inpainting_candidate/resources.py")
    require(not any(ref["path"] == resource["path"] for ref in native["runtime_code"]),
        "NATIVE_PENDING_HELPER_ALREADY_SIGNED")
    native["runtime_code"] = sorted([*native["runtime_code"], resource], key=lambda ref: ref["path"])
    task_id = "inpainting_native_motion_render_audit_a02_20261013"
    native.update(task_id=task_id, output_dir=str(REPO_ROOT / "_run" / task_id / "attempt_0001"),
        supersedes_pending_spec=previous["native_audit"])
    native["read_set"] = ["inpainting_optimization/NATIVE_MOTION_AUDIT_SPEC_R02.json"
        if entry == "inpainting_optimization/NATIVE_MOTION_AUDIT_SPEC.json" else entry
        for entry in native["read_set"]]
    for row in native["sessions"]:
        row["native_signature"] = canonical_sha({"stage": "MotionSource", "source": row["source_receipt"],
            "raw_inputs": row["raw_inputs"], "config": native["config"], "runtime_code": native["runtime_code"]})
    validate_geometry_spec(native, repo_root=REPO_ROOT)
    route_refs = geometry_route_contract(REPO_ROOT)[GEOMETRY_ROUTE_KEY]["code_closure"]
    require(all(ref in native["runtime_code"] for ref in route_refs) and resource in native["runtime_code"],
        "NATIVE_REVISED_COMPLETE_ROUTE_AND_HELPER_REQUIRED")
    targets = [REPO_ROOT / "inpainting_optimization/NATIVE_MOTION_AUDIT_SPEC_R02.json",
        output / "NEXT_FINITE_SPECS_R02.json"]
    require(not any(path.exists() for path in targets), "NATIVE_PENDING_REVISION_ALREADY_EXISTS")
    native_ref = json_once(targets[0], native)
    result = {**previous, "native_audit": native_ref,
        "superseded_native_pending_spec": previous["native_audit"],
        "supersession_reason": "NATIVE_RUNTIME_RESOURCES_HELPER_NOT_PINNED",
        "resource_helper": resource, "algorithm_or_numerical_changes": 0}
    print(json.dumps(json_once(targets[1], result), ensure_ascii=False))


def candidate_code_refs():
    routes = {**visual_route_contract(REPO_ROOT), **geometry_route_contract(REPO_ROOT)}
    refs = {ref["path"]: ref for route in routes.values() for ref in route["code_closure"]}
    for path in (REPO_ROOT / "tests/unit").glob("test_inpainting_candidate*.py"):
        refs[str(path)] = file_ref(path)
    refs[str(Path(__file__).resolve())] = file_ref(Path(__file__).resolve())
    return [refs[path] for path in sorted(refs)]


def freeze_test_inputs(revision):
    packet = cpu_packet()
    require(revision in {"R01", "R02", "R03"}, "FINITE_TEST_FREEZE_REVISION_REQUIRED")
    result = {"schema": "inpainting-cpu-test-code-freeze-v1", "task_id": TASK,
        "run_signature": packet["run_signature"], "code_and_test_refs": candidate_code_refs(),
        "scope": "SYNTHETIC_CPU_TESTS_NOT_VISUAL_QUALIFICATION"}
    print(json.dumps(json_once(Path(packet["output_dir"]) / ("FULL_TEST_CODE_INPUTS_" + revision + ".json"), result), ensure_ascii=False))


def seal_engineering(junit, revision):
    packet = cpu_packet()
    output = Path(packet["output_dir"])
    junit = Path(junit).absolute()
    require(junit.resolve(strict=True) == junit and junit.is_relative_to(output), "OWN_FULL_JUNIT_REQUIRED")
    suites = ET.parse(junit).getroot()
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites.iter("testsuite"))
    skipped = sum(int(s.get("skipped", 0)) for s in suites.iter("testsuite"))
    cases = list(suites.iter("testcase"))
    modules = {str(case.get("classname", "")).rsplit(".", 1)[-1] for case in cases}
    expected = {p.stem for p in (REPO_ROOT / "tests/unit").glob("test_inpainting_candidate*.py")}
    require(cases and failed == skipped == 0 and expected <= modules, "FULL_CANDIDATE_CPU_REGRESSION_REQUIRED")
    require(revision in {"R01", "R02", "R03"}, "FINITE_TEST_FREEZE_REVISION_REQUIRED")
    freeze_path = output / ("FULL_TEST_CODE_INPUTS_" + revision + ".json")
    freeze = load(freeze_path)
    require(freeze["task_id"] == TASK and freeze["run_signature"] == packet["run_signature"], "TEST_CODE_FREEZE_IDENTITY")
    for ref in freeze["code_and_test_refs"]:
        verify_ref(ref)
    require(candidate_code_refs() == freeze["code_and_test_refs"], "CPU_CODE_CHANGED_AFTER_FINAL_TEST_FREEZE")
    lanes = {key: file_ref(output / lane / ("ENGINEERING_RESULT_R02.json"
        if key in {"Mask", "CleanComposite"} else "ENGINEERING_RESULT.json")) for key, lane in
        {"Mask": "lane_veto", "Geometry": "lane_geometry", "CleanComposite": "lane_clean_composite"}.items()}
    def verify_tree(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= set(value):
                verify_ref(value)
            else:
                for child in value.values():
                    verify_tree(child)
        elif isinstance(value, list):
            for child in value:
                verify_tree(child)
    for ref in lanes.values():
        value = load(ref["path"])
        require(value["task_id"] == TASK and value["run_signature"] == packet["run_signature"], "LANE_WRITER_TASK_DRIFT")
        verify_tree(value)
    next_specs = file_ref(output / ("NEXT_FINITE_SPECS_R02.json" if revision == "R03" else "NEXT_FINITE_SPECS.json"))
    verify_tree(load(next_specs["path"]))
    delivered = file_ref(output / "G3_DELIVERY_R02.json")
    summary = {"schema": "inpainting-native-role-cpu-engineering-result-v1", "task_id": TASK,
        "run_signature": packet["run_signature"], "status": "CPU_ENGINEERING_COMPLETE",
        "full_cpu_tests": {"reference": file_ref(junit), "passed": len(cases), "failed": 0, "skipped": 0},
        "code_freeze": file_ref(freeze_path), "lanes": lanes,
        "pending_independent_finite_specs": next_specs, "G3_actual_delivery": delivered,
        "algorithm_model_native_ik_render_calls": 0, "training_steps": 0, "files_deleted": 0,
        "default_v1_unchanged": True, "manus0916_unchanged": True, "ICT_unchanged": True,
        "visual_quality": "NOT_EVALUATED_ENGINEERING", "candidate_adopted": False,
        "known_reviewed_G3": {"sessions": 11, "native_frames": 1400, "actual_viewed_frames": 99,
            "unviewed_frames": 1301, "FAILED": 11, "HIGH": 28, "MED": 4},
        "next_action": "Close this CPU task, independently register full-native11 motion audit, then one six-development source-aware Mask model candidate. Ready downstream scopes remain independent; quality failure does not justify all199 or adoption.",
        "claim_limit": "CPU wrappers/provenance tests are not new production stages, visual improvement, training or physical Robot truth."}
    summary_ref = json_once(output / "ENGINEERING_RESULT.json", summary)
    docs = [file_ref(REPO_ROOT / relative) for relative in (
        "inpainting_optimization/CONTINUE_HERE_ZH.md", "inpainting_optimization/G3_STATUS_ZH.md",
        "inpainting_optimization/RESULT_INDEX.json", "review/inpainting_optimization/README_ZH.md")]
    marker = {"task_id": TASK, "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [summary_ref, *lanes.values(), file_ref(junit), delivered, next_specs, *docs],
        "actual_model_motion_render_calls": 0, "adopted": False}
    print(json.dumps(json_once(output / "MAINTENANCE_DONE.json", marker), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish-g3", "freeze-next", "freeze-native-runtime", "freeze-tests", "seal-engineering"])
    parser.add_argument("--revision", choices=["R01", "R02", "R03"])
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    if args.command == "publish-g3":
        publish_g3(args.revision)
    elif args.command == "freeze-next":
        require(args.revision is None, "SPEC_FREEZE_NO_DELIVERY_REVISION")
        freeze_next_specs()
    elif args.command == "freeze-native-runtime":
        require(args.revision is None, "NATIVE_REVISION_COMMAND_HAS_FIXED_FRESH_TARGETS")
        freeze_native_runtime_revision()
    elif args.command == "freeze-tests":
        freeze_test_inputs(args.revision)
    else:
        require(args.junit is not None, "FINAL_FULL_JUNIT_ARGUMENT_REQUIRED")
        seal_engineering(args.junit, args.revision)


if __name__ == "__main__":
    main()
