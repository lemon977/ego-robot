"""Seal bounded Controller engineering and freeze its independent model task.

This helper neither runs a model nor publishes governance. It reuses the
existing candidate executor, lease and registered operation.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref, verify_sealed_stage

CPU = "inpainting_post_g1_controller_prompt_engineering_20261013"
MODEL = "inpainting_post_g1_controller_rgb_model_a01_20261013"
SCOPE = "POST_G1_CONTROLLER_RGB_PROPOSAL_FAILURE"
DEV = ["001", "003", "008", "100", "114", "050"]
NATIVE = [238, 183, 200, 139, 145, 128]


def write_once(path, value):
    if path.exists():
        raise ValueError("IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    atomic_json(path, value)
    return artifact_ref(path)


def cpu_packet(path=None):
    path = path or REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    packet = load_json(path)
    if (packet["task_id"] != CPU or packet["algorithm_execution_allowed"] is not False
            or packet["deletion_allowed"] is not False or packet["budgets"]["gpu_count"] != 0
            or packet["budgets"]["model_calls_max"] != 0):
        raise ValueError("EXACT_CONTROLLER_CPU_PACKET_REQUIRED")
    for reference in packet["inputs"].values():
        verify_ref(reference)
    return packet


def seal_engineering(packet_path, junit):
    packet = cpu_packet(packet_path)
    output = Path(packet["output_dir"])
    if not junit.resolve(strict=True).is_relative_to(output):
        raise ValueError("TASK_JUNIT_REQUIRED")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    counts = {key: sum(int(row.attrib.get(key, 0)) for row in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 510 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_EXISTING_AND_CONTROLLER_CPU_REGRESSION_REQUIRED")
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    refs = [*route["code_closure"], artifact_ref(Path(__file__).resolve())]
    for relative in (
        "tests/unit/test_inpainting_candidate_automatic_masks.py",
        "tests/unit/test_inpainting_candidate_model_stages.py",
        "tests/unit/test_inpainting_candidate_model_execution.py",
        "tests/unit/test_inpainting_candidate_entry.py",
    ):
        refs.append(artifact_ref(REPO_ROOT / relative))
    snapshot = output / "ENGINEERING_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            source = Path(reference["path"])
            archive.write(source, str(source.relative_to(REPO_ROOT)))
    audit = output / "MOTION_RENDER_CONSUMER_AUDIT.json"
    next_packet = output / "NEXT_MOTION_RENDER_PACKET_PROPOSAL.json"
    result = write_once(output / "ENGINEERING_RESULT.json", {
        "schema": "inpainting-post-g1-controller-engineering-evidence-v1",
        "task_id": CPU, "run_signature": packet["run_signature"], "completed_at": now_iso(),
        "experiment_scope": SCOPE, "execution": "CPU_ENGINEERING_COMPLETE",
        "cpu_tests": counts, "junit": artifact_ref(junit), "code": refs,
        "snapshot": artifact_ref(snapshot),
        "motion_render_audit": artifact_ref(audit) if audit.is_file() else None,
        "next_motion_render_packet_proposal": artifact_ref(next_packet) if next_packet.is_file() else None,
        "single_variable": "controller_text_prompt",
        "control_value": "a handheld or hand-mounted VR controller", "candidate_value": "VR controller",
        "original_g1_unchanged": True, "g1_third_recipe": False,
        "models_called": 0, "files_deleted": 0, "candidate_adopted": False,
        "models_changed": False, "thresholds_changed": False,
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "visual_quality": "NOT_EVALUATED",
        "next_task": MODEL,
        "next_action": "Register one independent full-six Controller prompt experiment after terminal governance validation; evaluate against sealed B only after new results seal.",
    })
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(post_g1_controller_engineering=result,
                 post_g1_controller_cpu_junit=artifact_ref(junit),
                 motion_render_consumer_audit=artifact_ref(audit) if audit.is_file() else None)
    atomic_json(index_path, index)
    write_once(output / "MAINTENANCE_DONE.json", {
        "task_id": CPU, "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [result, artifact_ref(junit), artifact_ref(snapshot)],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False,
    })
    return result


def prepare_model_spec():
    from chaoyang.governance.inpainting_candidate_registration import (
        visual_route_contract, VISUAL_ROUTE_KEY, validate_post_controller_experiment,
    )
    packet_path = REPO_ROOT / "tasks/current" / CPU / "TASK_PACKET.json"
    packet = cpu_packet(packet_path)
    output = Path(packet["output_dir"])
    result = load_json(output / "RESULT.json")
    post = load_json(output / "POST_VALIDATION.json")
    if (result["status"] != "PASSED" or result["scope_status"] != "PASS"
            or post["exit_code"] != 0):
        raise ValueError("CPU_TERMINAL_AND_GOVERNANCE_PASS_REQUIRED")
    engineering = load_json(output / "ENGINEERING_RESULT.json")
    if engineering["experiment_scope"] != SCOPE or engineering["run_signature"] != packet["run_signature"]:
        raise ValueError("SEALED_CONTROLLER_ENGINEERING_REQUIRED")
    parent_ref = packet["inputs"]["g1_b_parent_packet"]
    parent = load_json(Path(parent_ref["path"]))
    index_ref = packet["inputs"]["source_index"]
    index = load_json(Path(index_ref["path"]))
    rows = index["sessions"]
    if ([row["session_key"] for row in rows] != DEV
            or [row["native_frames"] for row in rows] != NATIVE
            or index["native_frames"] != 1033):
        raise ValueError("FIXED_FULL_SIX_NATIVE_SOURCES_REQUIRED")
    config = copy.deepcopy(parent["config"])
    config.update(candidate_id="POST_G1_CONTROLLER_SHORT_CONCEPT_A01",
                  algorithm_parameter_change="GLOBAL_CONTROLLER_TEXT_CONCEPT_ONLY",
                  controller_text_prompt="VR controller")
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    runtime = [*route["code_closure"], artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    weights = parent["model_weights"]
    for reference in [*runtime, *weights]:
        verify_ref(reference)
    spec = {
        "spec_schema": "inpainting-finite-g1-mask-spec-v1",
        "experiment_scope": SCOPE, "task_id": MODEL,
        "output_dir": str(REPO_ROOT / "_run" / MODEL / "attempt_0001"),
        "infrastructure": packet["runtime_code"][1],
        "inputs": {
            "v1_manifest": packet["inputs"]["v1_manifest"], "source_index": index_ref,
            "parent_recipe_packet": parent_ref,
            "controller_cpu_packet": artifact_ref(packet_path),
            "controller_cpu_result": artifact_ref(output / "RESULT.json"),
            "controller_cpu_validation": artifact_ref(output / "POST_VALIDATION.json"),
            "controller_engineering_result": artifact_ref(output / "ENGINEERING_RESULT.json"),
        },
        "runtime_code": runtime, "model_weights": weights, "physical_gpu_index": 0,
        "model_stages": ["Mask"], "config": config,
        "calibration": parent["calibration"], "schema": parent["schema"],
        "budgets": {"candidates_max": 1, "wall_seconds_max": 7200, "gpu_count": 1,
                    "gpu_seconds_max": 7200, "cpu_threads_per_worker": 2,
                    "model_calls_max": 512, "model_calls_unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION",
                    "training_steps_max": 0, "output_bytes_estimate": 4294967296},
        "sessions": [],
        "evaluation_only": {"completed_b_model_result": packet["inputs"]["g1_b_parent_model_result"],
                            "g2_actual_review": packet["inputs"]["g2_actual_review"]},
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md",
                     "inpainting_v1/MANIFEST.json", "inpainting_optimization/CONTROLLER_RGB_A01_SPEC.json"],
        "read_set_expansions": [
            "Production consumes only current sealed six RGB Source stages and frozen code/weight/config. Parent packet supplies global recipe metadata only; no old Masks/Clean/manual seeds/Robot trajectories.",
            "This is an independent registered Controller failure-cluster experiment, not a third G1 candidate or successor. B masks and annotations are evaluator-only after new results seal.",
        ],
    }
    for row in rows:
        reference = row["source_receipt"]
        verify_ref(reference)
        path = Path(reference["path"])
        source = load_json(path)
        verify_sealed_stage(source, source["component_signature"], output_root=path.parent)
        if source["native_frames"] != row["native_frames"]:
            raise ValueError("NATIVE_SOURCE_COUNT_DRIFT")
        signature = canonical_sha({"source": reference, "code": runtime, "config": config,
                                   "weights": weights, "calibration": spec["calibration"], "schema": spec["schema"]})
        spec["sessions"].append({"session_key": row["session_key"], "source_receipt": reference,
                                 "native_frames": row["native_frames"], "mask_signature": signature})
    validate_post_controller_experiment(spec, repo_root=REPO_ROOT)
    reference = write_once(REPO_ROOT / "inpainting_optimization/CONTROLLER_RGB_A01_SPEC.json", spec)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(post_g1_controller_spec=reference)
    atomic_json(index_path, index)
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal-engineering", "model-spec"))
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    result = seal_engineering(args.packet, args.junit) if args.action == "seal-engineering" else prepare_model_spec()
    print(result, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
