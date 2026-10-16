"""Finite G2 specifications and evidence sealing; no models or publisher here."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import verify_ref

PREP = "inpainting_generalization_g2_clean_engineering_20261013"
REPAIR = "inpainting_generalization_g2_cpu_publish_refresh_20261013"
ENGINEERING = "inpainting_generalization_g2_clean_engineering_successor_20261013"
MODEL = "inpainting_generalization_g2_clean_refs_a01_20261013"
RAW_REPAIR = "inpainting_generalization_g2_raw_input_runtime_repair_20261013"
MODEL_R02 = "inpainting_generalization_g2_clean_refs_a02_20261013"
BOOTSTRAP = "scripts/maintenance/inpainting_optimization_task_v1.py"


def checked_packet(path, task_id):
    packet = load_json(path)
    if (packet["task_id"] != task_id or packet["algorithm_execution_allowed"] is not False
            or packet["deletion_allowed"] is not False or packet["budgets"]["gpu_count"] != 0):
        raise ValueError("EXACT_CPU_PACKET_REQUIRED")
    for reference in packet["inputs"].values():
        verify_ref(reference)
    return packet


def write_once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def seal_preparation(path):
    packet = checked_packet(path, PREP)
    output = Path(packet["output_dir"])
    rollup = output / "G1_B_REVIEW_ROLLUP.json"
    review = load_json(rollup)
    snapshot = output / "PREPARATION_RUNTIME.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in (BOOTSTRAP, "src/chaoyang/governance/inpainting_candidate_registration.py",
                         "scripts/maintenance/inpainting_clean_specs_v1.py"):
            archive.write(REPO_ROOT / relative, relative)
    evidence = {
        "schema": "inpainting-g2-preparation-evidence-v1", "task_id": PREP,
        "run_signature": packet["run_signature"], "completed_at": now_iso(),
        "g1_b_review": artifact_ref(rollup), "g2_design": packet["inputs"]["clean_design"],
        "cpu_publisher": artifact_ref(REPO_ROOT / BOOTSTRAP),
        "prechange_runtime_snapshot": artifact_ref(snapshot),
        "scope_status": "PREPARATION_SEALED_ENGINEERING_REQUIRES_FRESH_PUBLISHER",
        "identified_risk": "The running CPU bootstrap imported candidate_registration during initial publication. Canonical route-function changes would otherwise leave later publications using its cached implementation, while fresh validators read the changed implementation.",
        "risk_status": "PREVENTED_BEFORE_CANONICAL_ROUTE_CHANGE",
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False,
        "visual_quality": "G1_B_REJECTED_G2_NOT_EVALUATED",
        "review_summary": {key: value for key, value in review.items()
                           if key not in {"sessions", "references", "artifacts"}},
        "next_action": "Narrow CPU bootstrap refresh task, then a fresh G2 engineering packet; no changes to baseline, shared publisher, thresholds or models.",
    }
    result = write_once(output / "PREPARATION_RESULT.json", evidence)
    write_once(output / "MAINTENANCE_DONE.json", {
        "task_id": PREP, "run_signature": packet["run_signature"], "status": "BLOCKED",
        "reason": "Preparation complete; canonical engineering deferred to a fresh bounded publisher after minimal cache-risk repair.",
        "artifacts": [result, artifact_ref(rollup), evidence["g2_design"], artifact_ref(snapshot)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False,
    })
    return result


def prepare_cpu_spec(mode):
    original = load_json(REPO_ROOT / "inpainting_optimization/G2_CLEAN_CPU_SPEC.json")
    previous_id = PREP if mode == "repair" else REPAIR
    previous_root = REPO_ROOT / "_run" / previous_id / "attempt_0001"
    previous_result = load_json(previous_root / "RESULT.json")
    post = load_json(previous_root / "POST_VALIDATION.json")
    if previous_result["status"] != "PASSED" or post["exit_code"] != 0:
        raise ValueError("NORMAL_TERMINAL_AND_GOVERNANCE_PASS_REQUIRED")
    task_id = REPAIR if mode == "repair" else ENGINEERING
    output = REPO_ROOT / "_run" / task_id / "attempt_0001"
    value = copy.deepcopy(original)
    value.update(task_id=task_id, output_dir=str(output))
    value["inputs"]["previous_cpu_result"] = artifact_ref(previous_root / "RESULT.json")
    value["inputs"]["previous_cpu_validation"] = artifact_ref(previous_root / "POST_VALIDATION.json")
    if mode == "repair":
        value["objective"] = "Prevent cached candidate-route publication in the existing finite CPU bootstrap by refreshing only candidate_registration before each existing publisher call. CPU synthetic regression only; no shared publisher/infra/V1 changes, models, thresholds, deletion or new queue."
        value["budgets"]["wall_seconds_max"] = 1800
        value["write_set"] = [str(output), str(REPO_ROOT / "tasks/current" / task_id),
            str(REPO_ROOT / BOOTSTRAP),
            str(REPO_ROOT / "tests/unit/test_inpainting_cpu_publish_refresh.py"),
            str(REPO_ROOT / "scripts/maintenance/inpainting_clean_specs_v1.py"),
            str(REPO_ROOT / "inpainting_optimization"),
            str(REPO_ROOT / "review/inpainting_optimization"),
            str(REPO_ROOT / "_run/inpainting_g2_cpu_publish_refresh_launch_20261013"),
            "Existing CAS publisher generated projections only"]
        value["read_set_expansions"].append("CPU bootstrap import-cache risk only: source, retained previous bytes and synthetic fake publisher tests. No live process patching.")
        value["change_contract"] = {"models_changed": False, "thresholds_changed": False,
            "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "shared_infrastructure_changed": False,
            "change": "Refresh candidate-registration module before existing CPU publisher calls only."}
        filename = "G2_CPU_PUBLISH_REFRESH_SPEC.json"
    else:
        value["objective"] = original["objective"] + " Uses the completed refresh repair; no G1 candidate is adopted."
        value["inputs"]["cpu_bootstrap"] = artifact_ref(REPO_ROOT / BOOTSTRAP)
        value["write_set"] = [item.replace(PREP, ENGINEERING).replace(
            "_run/inpainting_g2_clean_cpu_launch_20261013", "_run/inpainting_g2_clean_cpu_successor_launch_20261013")
            for item in original["write_set"]]
        filename = "G2_CLEAN_CPU_SUCCESSOR_SPEC.json"
    path = REPO_ROOT / "inpainting_optimization" / filename
    return write_once(path, value)


def seal_repair(path, junit):
    packet = checked_packet(path, REPAIR)
    output = Path(packet["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 3 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("CPU_REFRESH_TEST_PASS_REQUIRED")
    refs = [artifact_ref(REPO_ROOT / BOOTSTRAP), artifact_ref(REPO_ROOT / "tests/unit/test_inpainting_cpu_publish_refresh.py")]
    snapshot = output / "REPAIR_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            source = Path(reference["path"])
            archive.write(source, str(source.relative_to(REPO_ROOT)))
    result = write_once(output / "REPAIR_RESULT.json", {
        "schema": "inpainting-cpu-publish-refresh-evidence-v1", "task_id": REPAIR,
        "run_signature": packet["run_signature"], "completed_at": now_iso(),
        "status": "CPU_TESTS_PASS_FRESH_G2_ENGINEERING_NEXT", "cpu_tests": counts,
        "junit": artifact_ref(junit), "code": refs, "snapshot": artifact_ref(snapshot),
        "models_called": 0, "files_deleted": 0, "candidate_adopted": False,
        "shared_infrastructure_changed": False, "visual_quality": "NOT_EVALUATED",
        "next_action": "Register fresh G2 Clean CPU engineering; then separately register actual ProPainter execution.",
    })
    write_once(output / "MAINTENANCE_DONE.json", {"task_id": REPAIR,
        "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def seal_engineering(path, junit):
    packet = checked_packet(path, ENGINEERING)
    output = Path(packet["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 423 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_CANDIDATE_CPU_REGRESSION_REQUIRED")
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    if "Clean" not in route["model_stages"]:
        raise ValueError("CLEAN_CANDIDATE_ROUTE_REQUIRED")
    refs = [*route["code_closure"], artifact_ref(Path(__file__).resolve())]
    snapshot = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            source = Path(reference["path"])
            archive.write(source, str(source.relative_to(REPO_ROOT)))
    result = write_once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-g2-clean-engineering-evidence-v1", "task_id": ENGINEERING,
        "run_signature": packet["run_signature"], "completed_at": now_iso(),
        "status": "CPU_WIRING_COMPLETE_ACTUAL_PROPAINTER_NEXT", "cpu_tests": counts,
        "junit": artifact_ref(junit), "code": refs, "snapshot": artifact_ref(snapshot),
        "models_called": 0, "files_deleted": 0, "candidate_adopted": False,
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "visual_quality": "NOT_EVALUATED",
        "next_action": "Freeze actual 001/238 Source, Mask, native map, capture and weights in a separate finite GPU task; no new SAM inference.",
    })
    write_once(output / "MAINTENANCE_DONE.json", {"task_id": ENGINEERING,
        "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def seal_raw_input_repair(path, junit):
    packet = checked_packet(path, RAW_REPAIR)
    output = Path(packet["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if not counts["tests"] or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("RAW_INPUT_REPAIR_CPU_TEST_PASS_REQUIRED")
    paths = ["scripts/maintenance/inpainting_candidate_model_task_v1.py",
             "tests/unit/test_inpainting_candidate_model_execution.py",
             "scripts/maintenance/inpainting_clean_specs_v1.py"]
    refs = [artifact_ref(REPO_ROOT / item) for item in paths]
    snapshot = output / "REPAIR_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in paths:
            archive.write(REPO_ROOT / item, item)
    result = write_once(output / "REPAIR_RESULT.json", {
        "schema": "inpainting-g2-raw-input-repair-evidence-v1", "task_id": RAW_REPAIR,
        "run_signature": packet["run_signature"], "completed_at": now_iso(),
        "status": "CPU_REPAIR_COMPLETE_ACTUAL_PROPAINTER_NEXT", "cpu_tests": counts,
        "junit": artifact_ref(junit), "code": refs, "snapshot": artifact_ref(snapshot),
        "failed_launch": packet["inputs"]["failed_model_launch"],
        "failed_spec": packet["inputs"]["failed_model_spec"],
        "models_called": 0, "files_deleted": 0, "candidate_adopted": False,
        "shared_infrastructure_changed": False, "visual_quality": "NOT_EVALUATED",
        "next_action": "Create a fresh A02 spec with repaired runtime SHA; do not overwrite or execute the old A01 spec.",
    })
    write_once(output / "MAINTENANCE_DONE.json", {"task_id": RAW_REPAIR,
        "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return result


def prepare_clean_model_spec(*, repaired=False):
    from chaoyang.exact78_baseline.clean import SETTINGS
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_sealed_stage
    packet_path = REPO_ROOT / "tasks/current" / ENGINEERING / "TASK_PACKET.json"
    packet = checked_packet(packet_path, ENGINEERING)
    output = Path(packet["output_dir"])
    completed = load_json(output / "RESULT.json")
    post = load_json(output / "POST_VALIDATION.json")
    if completed["status"] != "PASSED" or completed["scope_status"] != "PASS" or post["exit_code"] != 0:
        raise ValueError("ENGINEERING_TERMINAL_AND_GOVERNANCE_PASS_REQUIRED")
    repair_refs = {}
    if repaired:
        repair_packet_path = REPO_ROOT / "tasks/current" / RAW_REPAIR / "TASK_PACKET.json"
        repair_packet = checked_packet(repair_packet_path, RAW_REPAIR)
        repair_output = Path(repair_packet["output_dir"])
        repair_result = load_json(repair_output / "RESULT.json")
        repair_post = load_json(repair_output / "POST_VALIDATION.json")
        if (repair_result["status"] != "PASSED" or repair_result["scope_status"] != "PASS"
                or repair_post["exit_code"] != 0):
            raise ValueError("RAW_INPUT_REPAIR_TERMINAL_AND_GOVERNANCE_PASS_REQUIRED")
        repair_refs = {"raw_input_repair_packet": artifact_ref(repair_packet_path),
                       "raw_input_repair_result": artifact_ref(repair_output / "RESULT.json"),
                       "raw_input_repair_validation": artifact_ref(repair_output / "POST_VALIDATION.json")}
    model_id = MODEL_R02 if repaired else MODEL
    spec_stem = "G2_CLEAN_A02" if repaired else "G2_CLEAN_A01"
    source_index_ref = packet["inputs"]["source_index"]
    index = load_json(Path(source_index_ref["path"]))
    row, = [row for row in index["sessions"] if row["session_key"] == "001"]
    source_ref = row["source_receipt"]
    source_path = Path(source_ref["path"])
    verify_ref(source_ref)
    source = load_json(source_path)
    verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
    if source["native_frames"] != 238:
        raise ValueError("FULL_238_NATIVE_SOURCE_REQUIRED")
    report = load_json(source_path.parent / "SOURCE_REPORT.json")
    mask_ref = packet["inputs"]["fixed_mask_001"]
    mask_path = Path(mask_ref["path"])
    mask = load_json(mask_path)
    verify_sealed_stage(mask, mask["component_signature"], output_root=mask_path.parent)
    raw_ref = report["actual_inputs"]["video"]
    if (mask["native_input_frames"] != 238 or mask["native_output_frames"] != 238
            or mask["actual_inputs"]["raw_source"] != raw_ref):
        raise ValueError("FULL_238_SAME_RAW_MASK_REQUIRED")
    for reference in report["actual_inputs"].values():
        verify_ref(reference)
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    runtime = [*route["code_closure"], artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    design = load_json(Path(packet["inputs"]["clean_design"]["path"]))
    weights = design["model_weights"]
    for reference in [*runtime, *weights]:
        verify_ref(reference)
    raft, = [ref for ref in weights if Path(ref["path"]).name == "raft-things.pth"]
    raw_rgb_ref = artifact_ref(source_path.parent / "RAW_RGB.npz")
    frame_map_ref = artifact_ref(source_path.parent / "FRAME_MAP.json")
    capture = {
        "schema": "inpainting-raw-raft-capture-config-v1", "session_id": report["session_id"],
        "native_shape": [238, 480, 640], "source_receipt": source_ref,
        "raw_source_ref": raw_ref, "raw_rgb_ref": raw_rgb_ref, "frame_map_ref": frame_map_ref,
        "raw_frames_sha256": mask["actual_inputs"]["frames_sha256"],
        "frame_map_payload_sha256": mask["actual_inputs"]["frame_map_sha256"],
        "raft_weight_ref": raft, "runtime_code": runtime, "raft_iter": SETTINGS["raft_iter"],
        "chunk_length": 12, "preprocessing": "RGB_UINT8_FLOAT32_B1_T3HW_DIV127_5_MINUS1",
        "background_registration": "ABSENT", "raw_flow_is_background_truth": False,
    }
    capture_ref = write_once(REPO_ROOT / "inpainting_optimization" / (spec_stem + "_CAPTURE_CONFIG.json"), capture)
    clean_config = {"capture_config_ref": capture_ref, "frozen_settings": dict(SETTINGS),
        "background_registration": "ABSENT", "reference_policy": "RAW_FB_DIAGNOSTIC_UNKNOWN_NO_QUALIFIED_SUPPORT",
        "evaluation_anchor_frames": [0, 50, 119, 132, 202, 237],
        "candidates": ["A", "B_IF_REFS_CHANGED"], "all_references_same_policy": "NOT_EXECUTED_NO_REFERENCE_CHANGE"}
    value = {
        "spec_schema": "inpainting-finite-g2-clean-spec-v1", "task_id": model_id,
        "output_dir": str(REPO_ROOT / "_run" / model_id / "attempt_0001"),
        "infrastructure": packet["runtime_code"][1],
        "inputs": {"v1_manifest": packet["inputs"]["v1_manifest"], "source_index": source_index_ref,
            "g2_cpu_packet": artifact_ref(packet_path), "g2_cpu_result": artifact_ref(output / "RESULT.json"),
            "g2_cpu_validation": artifact_ref(output / "POST_VALIDATION.json"),
            "fixed_mask_001": mask_ref, "capture_config": capture_ref, **repair_refs},
        "runtime_code": runtime, "model_weights": weights, "physical_gpu_index": 0,
        "model_stages": ["Clean"], "actual_stage_scope": "G2_CLEAN_REFERENCE_SELECTION_AB_ONLY",
        "config": {"candidate_id": "G2_FIXED_MASK_REFS_ONLY_A01", "stage": "G2_CLEAN_REFERENCE_SELECTION_AB_ONLY",
            "algorithm_parameter_change": "REFERENCE_INDICES_ONLY_IF_QUALIFIED_SUPPORT_EXISTS",
            "frozen_settings": dict(SETTINGS), "clean_config_sha256": canonical_sha(clean_config),
            "manual_seed_inputs": 0, "recipe_session_branches": 0},
        "clean_config": clean_config,
        "calibration": {"metric_projection": "NOT_CONSUMED_CLEAN",
            "source_eye_qualification": source["eye_route_status"], "background_registration": "ABSENT"},
        "schema": {"component": "inpainting-candidate-component-v1", "frame": "NATIVE_FRAME_MAP",
            "model_result": "inpainting-g2-clean-batch-result-v1"},
        "budgets": {"candidates_max": 2, "wall_seconds_max": 7200, "gpu_seconds_max": 7200,
            "gpu_count": 1, "cpu_threads_per_worker": 2, "model_calls_max": 2,
            "model_calls_unit": "PROPAINTER_INPAINT_API_INVOCATION", "training_steps_max": 0,
            "output_bytes_estimate": 4294967296},
        "sessions": [{"session_key": "001", "native_frames": 238, "source_receipt": source_ref,
            "mask_receipt": mask_ref, "raw_source_ref": raw_ref, "frame_map_ref": frame_map_ref}],
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
            "inpainting_optimization/" + spec_stem + "_SPEC.json", str(source_path.relative_to(REPO_ROOT)),
            str(mask_path.relative_to(REPO_ROOT)), "inpainting_optimization/" + spec_stem + "_CAPTURE_CONFIG.json"],
        "read_set_expansions": ["Frozen actual PP/RAFT code and weights; current001 Source/Mask/native map only. Review annotations and historical A masks are evaluator-only, not producer inputs.",
            "No scene/background qualification is inferred from mask complement or low raw flow residual. B is skipped when references are unchanged; sealed A remains independently reportable."],
    }
    return write_once(REPO_ROOT / "inpainting_optimization" / (spec_stem + "_SPEC.json"), value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal-preparation", "repair-spec", "engineering-spec", "seal-repair", "seal-engineering", "model-spec", "seal-raw-input-repair", "repaired-model-spec"))
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    if args.action == "seal-preparation":
        result = seal_preparation(args.packet)
    elif args.action == "seal-repair":
        result = seal_repair(args.packet, args.junit)
    elif args.action == "seal-engineering":
        result = seal_engineering(args.packet, args.junit)
    elif args.action == "model-spec":
        result = prepare_clean_model_spec()
    elif args.action == "seal-raw-input-repair":
        result = seal_raw_input_repair(args.packet, args.junit)
    elif args.action == "repaired-model-spec":
        result = prepare_clean_model_spec(repaired=True)
    else:
        result = prepare_cpu_spec("repair" if args.action == "repair-spec" else "engineering")
    print(result, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
