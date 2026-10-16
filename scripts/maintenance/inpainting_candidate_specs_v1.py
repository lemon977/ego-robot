"""Generate narrow immutable candidate specs; no task registration or models."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.inpainting_candidate.contracts import canonical_sha, verify_ref

DEV = ("001", "003", "008", "100", "114", "050")
G1_ORIGINAL_DEADLINE = "2026-10-13T01:35:47.996565+00:00"
WEARER_CPU_TASK = "inpainting_generalization_g1_wearer_b_engineering_20261013"
WEARER_MODEL_TASK = "inpainting_generalization_g1_mask_wearer_b_20261013"


def wearer_cpu_packet(path):
    packet = load_json(path)
    if (packet["task_id"] != WEARER_CPU_TASK or packet["algorithm_execution_allowed"] is not False
            or packet["budgets"]["gpu_count"] != 0 or packet["budgets"]["model_calls_max"] != 0):
        raise ValueError("EXACT_WEARER_B_CPU_ENGINEERING_PACKET_REQUIRED")
    return packet


def _a_recipe(config):
    """Execution-reader repairs were A, not a segmentation candidate change."""
    value = {key: item for key, item in config.items()
             if key not in {"candidate_id", "algorithm_parameter_change"}}
    if value.get("mask_rule_variant") not in {None, "HUMAN_ONLY_A"}:
        raise ValueError("PARENTS_MUST_BE_SAME_HUMAN_ONLY_A_RECIPE")
    return value


def prepare_wearer_b_spec(packet_path):
    """Freeze the second RGB rule; old masks are evaluation metadata, not inputs."""
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    from chaoyang.inpainting_candidate.contracts import verify_sealed_stage
    packet = wearer_cpu_packet(packet_path)
    parent_ref = packet["inputs"]["completed_a_packet"]
    verify_ref(parent_ref)
    parent = load_json(Path(parent_ref["path"]))
    if parent["task_id"] != "inpainting_generalization_g1_mask_c02_successor_20261013":
        raise ValueError("EXACT_COMPLETED_A_SUCCESSOR_REQUIRED")
    parent_root = Path(parent["output_dir"])
    completed = load_json(parent_root / "RESULT.json")
    if (completed["status"] != "PASSED" or completed["model_result_consumable"] is not True
            or completed["task_id"] != parent["task_id"] or completed["run_signature"] != parent["run_signature"]):
        raise ValueError("A_EXECUTION_MUST_BE_COMPLETE_AND_CONSUMABLE")
    verify_ref(completed["model_result"])
    model = load_json(Path(completed["model_result"]["path"]))
    if (model["task_id"] != parent["task_id"] or model["run_signature"] != parent["run_signature"]
            or model["completed_mask_sessions"] != 5 or model["planned_sessions"] != 5
            or [row["session_key"] for row in model["sessions"]] != list(DEV[1:])
            or any(row["execution"] != "COMPLETE_MASK_ONLY" for row in model["sessions"])):
        raise ValueError("A_REMAINING_FIVE_COMPLETE_REQUIRED")
    history = []
    for candidate in ("c01", "c02"):
        task = "inpainting_generalization_g1_mask_" + candidate + "_20261013"
        old_packet = load_json(REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json")
        old_result = load_json(Path(old_packet["output_dir"]) / "RESULT.json")
        if (old_result["task_id"] != task or old_result["run_signature"] != old_packet["run_signature"]
                or old_result["status"] != "FAILED_RUNTIME_FINAL"):
            raise ValueError("EXACT_TERMINAL_A_HISTORY_REQUIRED")
        if (_a_recipe(old_packet["config"]) != _a_recipe(parent["config"])
                or old_packet["model_weights"] != parent["model_weights"]
                or old_packet["calibration"] != parent["calibration"] or old_packet["schema"] != parent["schema"]):
            raise ValueError("A_SETTINGS_WEIGHTS_OR_CALIBRATION_DRIFT")
        history.append((old_packet, old_result))
    c01, c01_result = history[0]
    c02, c02_result = history[1]
    log = (Path(c01["output_dir"]) / "WORKER.log").read_text(encoding="utf-8")
    if (c01_result["model_result"] is not None or c01_result["errors"] != ["MODEL_WORKER_EXIT:1"]
            or "ValueError: expected JSON object:" not in log or "Source/FRAME_MAP.json" not in log):
        raise ValueError("C01_DEMONSTRATED_PRE_API_SOURCE_MAP_FAILURE_REQUIRED")
    if c02_result["prior_model_api_calls"] != 30 or c02_result["completed_mask_sessions"] != 1:
        raise ValueError("C02_CUMULATIVE_API_EVIDENCE_REQUIRED")
    verify_ref(parent["resume_context"]["retained_session"])
    retained = load_json(Path(parent["resume_context"]["retained_session"]["path"]))
    if (retained["session_key"] != "001" or retained["execution"] != "COMPLETE_MASK_ONLY"
            or retained["mask"] != c02_result["retained_sealed_001"]
            or retained["source"] != c02["sessions"][0]["source_receipt"]):
        raise ValueError("A_RETAINED_001_COMPLETE_REQUIRED")
    all_rows = [retained, *model["sessions"]]
    if sum(row["native_frames"] for row in all_rows) != 1033:
        raise ValueError("A_FIXED_SIX_1033_NATIVE_FRAMES_REQUIRED")
    source_ref = packet["inputs"]["source_index"]
    verify_ref(source_ref)
    index = load_json(Path(source_ref["path"]))
    if ([row["session_key"] for row in index["sessions"]] != list(DEV) or index["native_frames"] != 1033
            or [row["source_receipt"] for row in index["sessions"]] != [row["source"] for row in all_rows]
            or [row["source_receipt"] for row in c01["sessions"]] != [row["source_receipt"] for row in index["sessions"]]
            or [row["source_receipt"] for row in c02["sessions"]] != [row["source_receipt"] for row in index["sessions"]]):
        raise ValueError("A_AND_B_EXACT_SAME_SIX_SOURCE_REFS_REQUIRED")
    for observed, planned in zip(all_rows, c02["sessions"]):
        verify_ref(observed["mask"])
        mask_path = Path(observed["mask"]["path"])
        mask = load_json(mask_path)
        signature = planned["mask_signature"] if observed["session_key"] == "001" else next(
            item["mask_signature"] for item in parent["sessions"] if item["session_key"] == observed["session_key"])
        verify_sealed_stage(mask, signature, output_root=mask_path.parent)
        verify_ref(observed["review"]["video"])
        if (mask["native_input_frames"] != observed["native_frames"]
                or mask["native_output_frames"] != observed["native_frames"]
                or observed["review"]["decoded_frames"] != observed["native_frames"]):
            raise ValueError("A_SEALED_COMPLETE_NATIVE_MASK_REQUIRED")
    accounting = model["model_call_accounting"]
    used = accounting["attempted_api_calls"]
    if (accounting["unit"] != "SAM_ADAPTER_INFERENCE_API_INVOCATION" or type(used) is not int
            or used != sum(accounting["by_api"].values()) or accounting["maximum"] != 19970
            or used <= 0 or used > accounting["maximum"]):
        raise ValueError("A_ACTUAL_CUMULATIVE_API_ACCOUNTING_REQUIRED")
    remaining_calls = 20000 - 30 - used
    deadline = datetime.fromisoformat(G1_ORIGINAL_DEADLINE)
    if datetime.fromisoformat(c01["deadline_at"]) != deadline:
        raise ValueError("ORIGINAL_G1_DEADLINE_DRIFT")
    remaining_seconds = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining_seconds <= 0 or remaining_calls <= 0:
        raise ValueError("ORIGINAL_G1_TIME_OR_API_BUDGET_EXHAUSTED")
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    runtime = [*route["code_closure"], artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    for ref in parent["model_weights"]:
        verify_ref(ref)
    config = {**parent["config"], "candidate_id": "G1_AUTOMATIC_RGB_MASK_GLOVE_WEARER_B",
        "algorithm_parameter_change": "RGB_GLOVE_INDEPENDENT_WEARER_BOOTSTRAP_ONLY",
        "mask_rule_variant": "RGB_GLOVE_WEARER_SUPPORT_B",
        "diagnostic_proposal_archive": "TEXT_POINT_PACKBITS_NOT_ALGORITHM_GAIN"}
    value = {"spec_schema": "inpainting-finite-g1-mask-spec-v1", "task_id": WEARER_MODEL_TASK,
        "output_dir": str(REPO_ROOT / "_run" / WEARER_MODEL_TASK / "attempt_0001"),
        "infrastructure": artifact_ref(REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"),
        "inputs": {"v1_manifest": packet["inputs"]["v1_manifest"], "source_index": source_ref,
                   "wearer_cpu_packet": artifact_ref(packet_path)},
        "runtime_code": runtime, "model_weights": parent["model_weights"],
        "physical_gpu_index": parent["gpu_mapping"]["physical_index"], "config": config,
        "calibration": parent["calibration"], "schema": parent["schema"],
        "original_g1_deadline_at": G1_ORIGINAL_DEADLINE,
        "budgets": {**parent["budgets"], "wall_seconds_max": min(28800, remaining_seconds),
                    "model_calls_max": remaining_calls}, "sessions": [],
        "prior_model_api_calls": {"c01": 0, "c02_interrupted": 30, "c02_successor": used, "cumulative": 30 + used},
        "evaluation_only": {"completed_a_result": artifact_ref(parent_root / "RESULT.json"),
                            "completed_a_model_result": completed["model_result"],
                            "retained_a_001_session": parent["resume_context"]["retained_session"]},
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
                     "inpainting_optimization/G1_MASK_WEARER_B_SPEC.json"],
        "read_set_expansions": ["Production reads exact same six sealed Source stages and frozen model/code closure only; no historical Mask/Clean/seed pixels.",
                                "A result/session references are evaluation-only metadata, not model producer inputs; glove wearer bootstrap is the sole changed segmentation rule."]}
    for row in index["sessions"]:
        verify_ref(row["source_receipt"])
        source_path = Path(row["source_receipt"]["path"])
        source = load_json(source_path)
        verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
        signature = canonical_sha({"source": row["source_receipt"], "code": runtime, "config": config,
            "weights": value["model_weights"], "calibration": value["calibration"], "schema": value["schema"]})
        value["sessions"].append({"session_key": row["session_key"], "source_receipt": row["source_receipt"], "mask_signature": signature})
    path = REPO_ROOT / "inpainting_optimization/G1_MASK_WEARER_B_SPEC.json"
    if path.exists():
        raise ValueError("IMMUTABLE_WEARER_B_SPEC_ALREADY_EXISTS")
    atomic_json(path, value)
    return artifact_ref(path)


def seal_wearer_b_engineering(packet_path):
    packet = wearer_cpu_packet(packet_path)
    output = Path(packet["output_dir"])
    suites = list(ET.parse(output / "CPU_TESTS.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 409 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_409_CASE_CPU_REGRESSION_REQUIRED")
    spec_path = REPO_ROOT / "inpainting_optimization/G1_MASK_WEARER_B_SPEC.json"
    spec = load_json(spec_path)
    if spec["task_id"] != WEARER_MODEL_TASK or spec["original_g1_deadline_at"] != G1_ORIGINAL_DEADLINE:
        raise ValueError("EXACT_FINITE_WEARER_B_SPEC_REQUIRED")
    rollup_path = output / "A_REVIEW_ROLLUP.json"
    rollup = load_json(rollup_path)
    if ([row.get("session_key", row.get("session")) for row in rollup["sessions"]] != list(DEV)
            or rollup["completed_sessions"] != 6 or rollup["native_frames"] != 1033):
        raise ValueError("SIX_ACTUAL_A_REVIEW_REFS_REQUIRED")
    for row in rollup["sessions"]:
        verify_ref(row["review"])
        review = load_json(Path(row["review"]["path"]))
        if (review["session"] != row["session_key"] or review["execution"] != "COMPLETE_MASK_ONLY"
                or not review.get("actual_viewed_frames") or review["hidden_regions"] != "UNKNOWN"
                or review["native_frames"] != row["native_frames"]):
            raise ValueError("ACTUAL_BOUND_A_SAMPLE_REVIEW_REQUIRED")
    def check_refs(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                verify_ref(value)
            else:
                for item in value.values(): check_refs(item)
        elif isinstance(value, list):
            for item in value: check_refs(item)
    check_refs(rollup)
    archive_path = output / "RUNTIME_SNAPSHOT.zip"
    refs = [*spec["runtime_code"], artifact_ref(Path(__file__))]
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            source = Path(reference["path"])
            archive.write(source, str(source.relative_to(REPO_ROOT)))
    value = {"schema": "inpainting-g1-wearer-b-engineering-evidence-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "execution": "CPU_ENGINEERING_COMPLETE",
        "cpu_tests": counts, "cpu_tests_ref": artifact_ref(output / "CPU_TESTS.xml"),
        "a_review_rollup": artifact_ref(rollup_path), "wearer_b_spec": artifact_ref(spec_path),
        "runtime_snapshot": artifact_ref(archive_path), "algorithm_changed": "RGB_GLOVE_INDEPENDENT_WEARER_BOOTSTRAP_ONLY",
        "diagnostic_archive_is_algorithm_gain": False, "model_weights_changed": False,
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "candidate_adopted": False,
        "original_g1_deadline_at": spec["original_g1_deadline_at"],
        "prior_model_api_calls": spec["prior_model_api_calls"], "remaining_model_api_budget": spec["budgets"]["model_calls_max"],
        "visual_quality": "B_NOT_EVALUATED_A_REVIEWS_SEPARATE", "model_calls": 0,
        "files_deleted": 0, "training_steps": 0, "external_writes": 0, "next_task": WEARER_MODEL_TASK}
    result_path = output / "ENGINEERING_RESULT.json"
    atomic_json(result_path, value)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(g1_wearer_b_engineering=artifact_ref(result_path), g1_spec=value["wearer_b_spec"], g1_a_review_rollup=value["a_review_rollup"])
    atomic_json(index_path, index)
    atomic_json(output / "MAINTENANCE_DONE.json", {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "status": "PASS", "artifacts": [artifact_ref(result_path), value["wearer_b_spec"], value["runtime_snapshot"],
                                          value["cpu_tests_ref"], value["a_review_rollup"]],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False, "visual_quality": "B_NOT_EVALUATED"})
    return artifact_ref(result_path)


def cpu_publication_packet(path):
    packet = load_json(path)
    if (not packet["task_id"].startswith("inpainting_generalization_g0_publication_")
            or packet["algorithm_execution_allowed"] is not False):
        raise ValueError("PUBLICATION_CPU_PACKET_REQUIRED")
    return packet


def publish_wrapper_bindings(packet_path):
    from chaoyang.inpainting_candidate.publication import WRAPPERS, HELPER, runtime_publication
    packet = cpu_publication_packet(packet_path)
    v1_path = REPO_ROOT / "inpainting_v1/MANIFEST.json"
    manifest = load_json(v1_path)
    originals = {ref["path"]: ref for ref in manifest["runtime_closure"]}
    parent_path = REPO_ROOT / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json"
    parent = load_json(parent_path)
    rows = []
    for relative in sorted(WRAPPERS):
        original = originals[str(REPO_ROOT / relative)]
        previous = parent["current_wrapper"] if relative.endswith("inpainting_v1_selection.py") else original
        rows.append({"relative_path": relative, "original": original, "parent": previous,
                     "current": artifact_ref(REPO_ROOT / relative), "role": "GOVERNANCE_RUNTIME_ONLY"})
    value = {"schema": "inpainting-v1-runtime-publication-r03",
        "baseline_id": "INPAINTING_RAW_TO_ROBOT_V1", "baseline_pointer": artifact_ref(v1_path),
        "parent_supplement": artifact_ref(parent_path), "wrapper_overrides": rows,
        "additional_runtime": [artifact_ref(REPO_ROOT / HELPER)],
        "task_packet": artifact_ref(packet_path), "algorithm_changed": False,
        "files_deleted": 0, "model_calls": 0, "training_steps": 0, "external_writes": 0,
        "claim_limit": "Candidate-only dispatch and governance wrapper binding; V1 core, default, source and historical results unchanged."}
    path = REPO_ROOT / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"
    if path.exists():
        raise ValueError("IMMUTABLE_R03_ALREADY_EXISTS")
    atomic_json(path, value)
    runtime_publication(REPO_ROOT, manifest)
    return artifact_ref(path)


def prepare_sources(packet_path):
    from chaoyang.inpainting_candidate.source import inspect_raw_session
    from chaoyang.inpainting_candidate.source_route import bind_adopted_visual_route
    from chaoyang.inpainting_candidate.source_frames import prepare_rgb_source
    packet = cpu_publication_packet(packet_path)
    out = Path(packet["output_dir"])
    refs = []
    code = [artifact_ref(REPO_ROOT / ("src/chaoyang/inpainting_candidate/" + name))
            for name in ("source.py", "source_route.py", "source_frames.py", "contracts.py")]
    v1 = artifact_ref(REPO_ROOT / "inpainting_v1/MANIFEST.json")
    for key in DEV:
        source = inspect_raw_session(Path("/path/to/private_storage/datasets/ego/pick_place_1008") / key)
        source = bind_adopted_visual_route(source, v1_manifest_ref=v1)
        signature = canonical_sha({"source": source["actual_inputs"], "adopted_route_contracts": source.get("adopted_route_contracts", []), "code": code,
            "config": {"output_size": [640, 480], "route": source["camera"]},
            "weights": "NOT_CONSUMED_SOURCE", "calibration": source["camera"],
            "schema": "inpainting-candidate-component-v1"})
        directory = out / "sessions" / key / "Source"
        stage = prepare_rgb_source(source, directory, component_signature=signature)
        refs.append({"session_key": key, "source_receipt": artifact_ref(directory / "RECEIPT.json"),
                     "native_frames": stage["native_frames"]})
        print({"source_sealed": key, "native_frames": stage["native_frames"]}, flush=True)
    summary = {"schema": "inpainting-six-source-stage-index-v1", "task_id": packet["task_id"],
               "sessions": refs, "native_frames": sum(row["native_frames"] for row in refs),
               "models_run": 0, "physical_eye_verified": False}
    atomic_json(out / "SOURCE_INDEX.json", summary)
    return artifact_ref(out / "SOURCE_INDEX.json")


def prepare_g1_spec(packet_path):
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    packet = cpu_publication_packet(packet_path)
    source_index = load_json(Path(packet["output_dir"]) / "SOURCE_INDEX.json")
    if [row["session_key"] for row in source_index["sessions"]] != list(DEV):
        raise ValueError("FIXED_DEV_REQUIRED")
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    weight = artifact_ref(REPO_ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt")
    config = {"stage": "G1_AUTOMATIC_MASK_ONLY", "image_size": [640, 480],
              "projection_hints": None, "manual_seed_inputs": 0, "recipe_session_branches": 0,
              "candidate_id": "G1_AUTOMATIC_RGB_MASK_C01"}
    schema = {"component": "inpainting-candidate-component-v1", "frame": "NATIVE_FRAME_MAP",
              "model_result": "inpainting-g1-mask-batch-result-v1"}
    calibration = {"metric_projection": "NOT_CONSUMED_MASK", "source_eye_qualification": "EXPLICIT_V1_FROZEN_VISUAL_ASSUMPTION"}
    sessions = []
    for row in source_index["sessions"]:
        verify_ref(row["source_receipt"])
        signature = canonical_sha({"source": row["source_receipt"], "code": route["code_closure"],
            "config": config, "weights": [weight], "calibration": calibration, "schema": schema})
        sessions.append({"session_key": row["session_key"], "source_receipt": row["source_receipt"], "mask_signature": signature})
    task = "inpainting_generalization_g1_mask_c01_20261013"
    result = {"spec_schema": "inpainting-finite-g1-mask-spec-v1", "task_id": task,
        "output_dir": str(REPO_ROOT / "_run" / task / "attempt_0001"),
        "infrastructure": artifact_ref(REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"),
        "inputs": {"v1_manifest": artifact_ref(REPO_ROOT / "inpainting_v1/MANIFEST.json"),
                   "source_index": artifact_ref(Path(packet["output_dir"]) / "SOURCE_INDEX.json")},
        "runtime_code": route["code_closure"], "model_weights": [weight],
        "physical_gpu_index": 0, "sessions": sessions, "config": config,
        "calibration": calibration, "schema": schema,
        "budgets": {"wall_seconds_max": 28800, "gpu_count": 1, "cpu_threads_per_worker": 2,
                    "candidates_max": 1, "model_calls_max": 20000,
                    "model_calls_unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION",
                    "output_bytes_estimate": 16 * 1024 ** 3, "training_steps_max": 0},
        "read_set": ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md",
                     "inpainting_v1/MANIFEST.json", "inpainting_optimization/G1_MASK_C01_SPEC.json"],
        "read_set_expansions": ["Only exact sealed Source and model/code closure pinned by this packet; no historical seed/Mask/Clean/motion/result reads"]}
    path = REPO_ROOT / "inpainting_optimization/G1_MASK_C01_SPEC.json"
    if path.exists():
        raise ValueError("IMMUTABLE_G1_SPEC_ALREADY_EXISTS")
    atomic_json(path, result)
    return artifact_ref(path)


def prepare_g1_successor(packet_path):
    """New immutable attempt after demonstrated integration failure, not retry in place."""
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    from chaoyang.inpainting_candidate.contracts import verify_sealed_stage
    packet = load_json(packet_path)
    if (packet["task_id"] != "inpainting_generalization_g1_c01_runtime_repair_20261013"
            or packet["algorithm_execution_allowed"] is not False):
        raise ValueError("FINITE_C01_CPU_REPAIR_PACKET_REQUIRED")
    original = load_json(Path(packet["inputs"]["original_candidate_spec"]["path"]))
    parent_packet_path = REPO_ROOT / "tasks/current" / original["task_id"] / "TASK_PACKET.json"
    parent = load_json(parent_packet_path)
    parent_result_path = Path(original["output_dir"]) / "RESULT.json"
    failed = load_json(parent_result_path)
    if (failed["status"] != "FAILED_RUNTIME_FINAL" or failed["model_result"] is not None
            or failed["errors"] != ["MODEL_WORKER_EXIT:1"]):
        raise ValueError("EXACT_C01_TERMINAL_INTEGRATION_FAILURE_REQUIRED")
    log = (Path(original["output_dir"]) / "WORKER.log").read_text(encoding="utf-8")
    if "ValueError: expected JSON object:" not in log or "Source/FRAME_MAP.json" not in log:
        raise ValueError("DEMONSTRATED_FRAME_MAP_LIST_FAILURE_REQUIRED")
    remaining = int((datetime.fromisoformat(parent["deadline_at"]) - datetime.now(timezone.utc)).total_seconds())
    if not 0 < remaining <= 28800:
        raise ValueError("G1_ORIGINAL_EIGHT_HOUR_TIMEBOX_EXHAUSTED")
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    for ref in original["model_weights"]:
        verify_ref(ref)
    value = {**original, "task_id": "inpainting_generalization_g1_mask_c02_20261013"}
    value["output_dir"] = str(REPO_ROOT / "_run" / value["task_id"] / "attempt_0001")
    value["runtime_code"] = route["code_closure"]
    value["config"] = {**original["config"], "candidate_id": "G1_AUTOMATIC_RGB_MASK_C02",
                       "algorithm_parameter_change": "NONE_SOURCE_LIST_READER_REPAIR_ONLY"}
    value["budgets"] = {**original["budgets"], "wall_seconds_max": remaining}
    value["original_g1_deadline_at"] = parent["deadline_at"]
    value["inputs"] = {**original["inputs"], "failed_c01_result": artifact_ref(parent_result_path),
                       "runtime_repair_packet": artifact_ref(packet_path)}
    value["sessions"] = []
    for row in original["sessions"]:
        verify_ref(row["source_receipt"])
        source_path = Path(row["source_receipt"]["path"])
        source = load_json(source_path)
        verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
        # This is the actual Source schema that C01 incorrectly treated as dict.
        import json
        mapping = json.loads((source_path.parent / "FRAME_MAP.json").read_text(encoding="utf-8"))
        if (not isinstance(mapping, list) or len(mapping) != source["native_frames"]
                or any(item.get("native_frame_index") != index
                       or item.get("decoded_display_index") != index for index, item in enumerate(mapping))):
            raise ValueError("SEALED_NATIVE_SOURCE_MAP_REQUIRED")
        signature = canonical_sha({"source": row["source_receipt"], "code": value["runtime_code"],
            "config": value["config"], "weights": value["model_weights"],
            "calibration": value["calibration"], "schema": value["schema"]})
        value["sessions"].append({**row, "mask_signature": signature})
    value["read_set"] = ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md",
                         "inpainting_v1/MANIFEST.json", "inpainting_optimization/G1_MASK_C02_SPEC.json"]
    path = REPO_ROOT / "inpainting_optimization/G1_MASK_C02_SPEC.json"
    if path.exists():
        raise ValueError("IMMUTABLE_C02_SPEC_ALREADY_EXISTS")
    atomic_json(path, value)
    return artifact_ref(path)


def seal_runtime_repair(packet_path):
    packet = load_json(packet_path)
    if (packet["task_id"] != "inpainting_generalization_g1_c01_runtime_repair_20261013"
            or packet["algorithm_execution_allowed"] is not False):
        raise ValueError("FINITE_C01_CPU_REPAIR_PACKET_REQUIRED")
    output = Path(packet["output_dir"])
    suites = list(ET.parse(output / "CPU_TESTS.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 361 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("COMPLETE_CPU_REGRESSION_AND_SOURCE_WORKER_TEST_REQUIRED")
    successor_path = REPO_ROOT / "inpainting_optimization/G1_MASK_C02_SPEC.json"
    successor = load_json(successor_path)
    refs = successor["runtime_code"]
    archive_path = output / "RUNTIME_SNAPSHOT.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            path = Path(reference["path"])
            archive.write(path, str(path.relative_to(REPO_ROOT)))
    value = {"schema": "inpainting-g1-c01-runtime-repair-evidence-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "execution": "CPU_REPAIR_COMPLETE",
        "failure": "Source FRAME_MAP JSON list was passed to governance dict-only loader",
        "change": "Read original ordered list with json.loads; retain native stage validations; save real strict SAM loader evidence",
        "algorithm_parameters_changed": False, "model_weights_changed": False,
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "cpu_tests": counts,
        "cpu_tests_ref": artifact_ref(output / "CPU_TESTS.xml"),
        "successor_spec": artifact_ref(successor_path), "runtime_snapshot": artifact_ref(archive_path),
        "model_calls": 0, "files_deleted": 0, "training_steps": 0,
        "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "original_g1_deadline_at": successor["original_g1_deadline_at"]}
    result_path = output / "REPAIR_RESULT.json"
    atomic_json(result_path, value)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(g1_c01_runtime_repair=artifact_ref(result_path), g1_spec=value["successor_spec"],
                 g1_failed_c01=successor["inputs"]["failed_c01_result"])
    atomic_json(index_path, index)
    atomic_json(output / "MAINTENANCE_DONE.json", {"task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [artifact_ref(result_path), value["runtime_snapshot"], value["successor_spec"], value["cpu_tests_ref"]],
        "model_calls": 0, "files_deleted": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False})
    return artifact_ref(result_path)


def prepare_g1_execution_resume(packet_path):
    from chaoyang.governance.inpainting_candidate_registration import visual_route_contract, VISUAL_ROUTE_KEY
    packet = load_json(packet_path)
    if (packet["task_id"] != "inpainting_generalization_g1_c02_executor_recovery_20261013"
            or packet["algorithm_execution_allowed"] is not False):
        raise ValueError("EXACT_CPU_EXECUTOR_RECOVERY_REQUIRED")
    parent_packet_ref = packet["inputs"]["interrupted_c02"]
    parent = load_json(Path(parent_packet_ref["path"]))
    original = load_json(Path(parent["inputs"]["spec"]["path"]))
    progress_path = Path(parent["output_dir"]) / "MODEL_PROGRESS.json"
    progress = load_json(progress_path)
    consumed = progress["model_call_accounting"]["attempted_api_calls"]
    if consumed != 30:
        raise ValueError("EXACT_C02_ATTEMPTED_API_ACCOUNTING_REQUIRED")
    route = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]
    runtime = [*route["code_closure"], artifact_ref(REPO_ROOT / "scripts/maintenance/inpainting_candidate_detach_v1.py")]
    # Only executor transport changed; all actual image/model recipe code and
    # model weights are the original frozen C02 bytes, not a third candidate.
    current = {ref["path"]: ref for ref in runtime}
    changed = [ref["path"] for ref in original["runtime_code"] if current.get(ref["path"]) != ref]
    allowed = str(REPO_ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py")
    if changed != [allowed]:
        raise ValueError("UNEXPECTED_C02_ALGORITHM_OR_VENDOR_CHANGE:" + str(changed))
    for reference in original["model_weights"]:
        verify_ref(reference)
    remaining = int((datetime.fromisoformat(original["original_g1_deadline_at"]) - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        raise ValueError("ORIGINAL_G1_TIMEBOX_EXHAUSTED")
    value = {**original, "task_id": "inpainting_generalization_g1_mask_c02_successor_20261013",
        "runtime_code": runtime, "inputs": {**original["inputs"], "failed_c02_result": packet["inputs"]["failed_c02_result"],
            "executor_recovery_packet": artifact_ref(packet_path)},
        "budgets": {**original["budgets"], "wall_seconds_max": remaining,
                    "model_calls_max": original["budgets"]["model_calls_max"] - consumed},
        "resume_context": {"parent_packet": parent_packet_ref,
            "failed_parent_result": packet["inputs"]["failed_c02_result"],
            "retained_session": artifact_ref(Path(parent["output_dir"]) / "sessions/001/SESSION_RESULT.json")},
        "sessions": []}
    value["output_dir"] = str(REPO_ROOT / "_run" / value["task_id"] / "attempt_0001")
    for row in original["sessions"][1:]:
        verify_ref(row["source_receipt"])
        signature = canonical_sha({"source": row["source_receipt"], "code": runtime,
            "config": value["config"], "weights": value["model_weights"],
            "calibration": value["calibration"], "schema": value["schema"]})
        value["sessions"].append({**row, "mask_signature": signature})
    value["read_set"] = ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md",
        "inpainting_v1/MANIFEST.json", "inpainting_optimization/G1_MASK_C02_SUCCESSOR_SPEC.json"]
    value["read_set_expansions"] = [
        "Exact sealed Source/model/code closure; retained complete C02/001 metadata proves skip only, never feeds another producer",
        "No C02/003 partial, historical seeds, old Mask/Clean/motion pixels; same C02 RGB algorithm/weights, original deadline and cumulative API cap"]
    path = REPO_ROOT / "inpainting_optimization/G1_MASK_C02_SUCCESSOR_SPEC.json"
    if path.exists():
        raise ValueError("IMMUTABLE_C02_EXECUTION_SUCCESSOR_SPEC_EXISTS")
    atomic_json(path, value)
    return artifact_ref(path)


def seal_executor_recovery(packet_path):
    packet = load_json(packet_path)
    if packet["task_id"] != "inpainting_generalization_g1_c02_executor_recovery_20261013":
        raise ValueError("EXACT_CPU_EXECUTOR_RECOVERY_REQUIRED")
    output = Path(packet["output_dir"])
    suites = list(ET.parse(output / "CPU_TESTS.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["tests"] < 365 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("SOURCE_WORKER_AND_TRANSPORT_CPU_REGRESSION_REQUIRED")
    successor_path = REPO_ROOT / "inpainting_optimization/G1_MASK_C02_SUCCESSOR_SPEC.json"
    successor = load_json(successor_path)
    with zipfile.ZipFile(output / "RUNTIME_SNAPSHOT.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in successor["runtime_code"]:
            verify_ref(reference)
            path = Path(reference["path"])
            archive.write(path, str(path.relative_to(REPO_ROOT)))
    original_root = Path(load_json(Path(successor["resume_context"]["parent_packet"]["path"]))["output_dir"])
    review_path = original_root / "review_cpu/lane_001_003_008/001/AI_REVIEW.json"
    value = {"schema": "inpainting-g1-c02-executor-recovery-evidence-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "execution": "CPU_EXECUTOR_REPAIR_COMPLETE",
        "cpu_tests": counts, "cpu_tests_ref": artifact_ref(output / "CPU_TESTS.xml"),
        "successor_spec": artifact_ref(successor_path), "runtime_snapshot": artifact_ref(output / "RUNTIME_SNAPSHOT.zip"),
        "retained_001_review": artifact_ref(review_path),
        "retained_001_mask_visual": "AI_SAMPLE_FAILED_HIGH2_MED1_USER_PENDING",
        "new_gpu_sessions": [row["session_key"] for row in successor["sessions"]],
        "algorithm_parameter_or_weights_changed": False, "new_candidate": False,
        "prior_model_api_calls": 30, "remaining_model_api_budget": successor["budgets"]["model_calls_max"],
        "original_g1_deadline_at": successor["original_g1_deadline_at"],
        "model_calls": 0, "files_deleted": 0, "generated_dead_lease_preserved": artifact_ref(output / "STALE_CARD_LEASE.json"),
        "candidate_adopted": False, "training_steps": 0}
    result_path = output / "EXECUTOR_REPAIR_RESULT.json"
    atomic_json(result_path, value)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(g1_c02_executor_recovery=artifact_ref(result_path), g1_spec=value["successor_spec"],
                 g1_c02_failed=successor["resume_context"]["failed_parent_result"],
                 g1_001_ai_review=value["retained_001_review"])
    atomic_json(index_path, index)
    atomic_json(output / "MAINTENANCE_DONE.json", {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "status": "PASS", "artifacts": [artifact_ref(result_path), value["successor_spec"],
            value["runtime_snapshot"], value["cpu_tests_ref"], value["retained_001_review"]],
        "model_calls": 0, "files_deleted": 0, "candidate_adopted": False})
    return artifact_ref(result_path)


def seal_publication(packet_path):
    packet = cpu_publication_packet(packet_path)
    output = Path(packet["output_dir"])
    parent_packet = load_json(Path(packet["inputs"]["interrupted_parent"]["path"])) if "interrupted_parent" in packet["inputs"] else packet
    parent_output = Path(parent_packet["output_dir"])
    source_index_path = parent_output / "SOURCE_INDEX.json"
    source_index = load_json(source_index_path)
    if [row["session_key"] for row in source_index["sessions"]] != list(DEV) or source_index["native_frames"] != 1033:
        raise ValueError("COMPLETE_NATIVE_SOURCE_REQUIRED")
    from chaoyang.inpainting_candidate.contracts import verify_sealed_stage
    for row in source_index["sessions"]:
        verify_ref(row["source_receipt"])
        source = load_json(Path(row["source_receipt"]["path"]))
        verify_sealed_stage(source, source["component_signature"], output_root=Path(row["source_receipt"]["path"]).parent)
        for reference in source["actual_inputs"]:
            verify_ref(reference)
    tests_path = parent_output / "CPU_TESTS.xml"
    suites = list(ET.parse(tests_path).getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    # pytest JUnit includes the nine CLI subtests in its 360 total cases.
    if counts["tests"] != 360 or any(counts[key] for key in ("errors", "failures", "skipped")):
        raise ValueError("PUBLISHED_CPU_REGRESSION_PASS_REQUIRED")
    g1_path = REPO_ROOT / "inpainting_optimization/G1_MASK_C01_SPEC.json"
    g1 = load_json(g1_path)
    refs = [*g1["runtime_code"], artifact_ref(REPO_ROOT / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json")]
    archive_path = output / "RUNTIME_SNAPSHOT.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for reference in refs:
            verify_ref(reference)
            path = Path(reference["path"])
            archive.write(path, str(path.relative_to(REPO_ROOT)))
    value = {"schema": "inpainting-g0-publication-evidence-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "execution": "CPU_PUBLICATION_COMPLETE",
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1", "cpu_tests": counts,
        "source_index": artifact_ref(source_index_path), "cpu_tests_ref": artifact_ref(tests_path),
        "g1_spec": artifact_ref(g1_path), "runtime_code": refs,
        "runtime_snapshot": artifact_ref(archive_path),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "git_qualification": "Base revision plus explicit byte snapshot; unrelated pre-existing dirty code remains uncommitted and unchanged.",
        "model_calls": 0, "files_deleted": 0, "training_steps": 0, "external_writes": 0,
        "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "interrupted_publication": artifact_ref(parent_output / "RESULT.json") if parent_output != output else None,
        "next_task": g1["task_id"]}
    result_path = output / "PUBLICATION_RESULT.json"
    atomic_json(result_path, value)
    index_path = REPO_ROOT / "inpainting_optimization/RESULT_INDEX.json"
    index = load_json(index_path)
    index.update(g0_publication=artifact_ref(result_path), g1_spec=value["g1_spec"], source_index=value["source_index"])
    atomic_json(index_path, index)
    atomic_json(output / "MAINTENANCE_DONE.json", {"task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "status": "PASS",
        "artifacts": [artifact_ref(result_path), value["runtime_snapshot"], value["source_index"], value["g1_spec"], value["cpu_tests_ref"]],
        "model_calls": 0, "files_deleted": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False})
    return artifact_ref(result_path)


def publication_spec():
    parent = load_json(REPO_ROOT / "inpainting_optimization/G0_INTEGRATION_SPEC.json")
    path = REPO_ROOT / "inpainting_optimization/G0_PUBLICATION_SPEC.json"
    if path.exists():
        raise ValueError("FRESH_PUBLICATION_SPEC_REQUIRED")
    task = "inpainting_generalization_g0_publication_20261013"
    value = {**parent, "task_id": task,
        "objective": "Publish candidate CPU submit and G1 visual operation routes with explicit R03 governance-wrapper bindings; prepare six full native Source stages, selective candidate Git seal, CPU regressions. Preserve V1 algorithm/data/weights and 0916. No real models.",
        "output_dir": str(REPO_ROOT / "_run" / task / "attempt_0001"),
        "write_set": ["src/chaoyang/cli/__init__.py",
            "src/chaoyang/governance/current_r3_contracts.py",
            "src/chaoyang/governance/inpainting_v1_selection.py",
            "src/chaoyang/ops/run_inpainting_candidate_cpu_admission_v1.py",
            "src/chaoyang/ops/run_inpainting_candidate_visual_stages_v1.py",
            "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json",
            "src/chaoyang/inpainting_candidate/", "tests/unit/test_inpainting_candidate_*.py",
            "scripts/maintenance/inpainting_candidate_*.py", "inpainting_optimization/",
            "_run/" + task + "/", "_run/inpainting_candidate_cpu_tests_publication/",
            "tasks/current/" + task + "/",
            "Project .git selective candidate/wrapper-only commit with unrelated staging preserved",
            "Existing CAS publisher generated governance projections only"],
        "read_set_expansions": [*parent["read_set_expansions"],
            "Read complete integration stage receipts and tests; source decode/count/time audit only, no model calls",
            "Inspect current wrapper references and explicit R02 supplement before new R03; no legacy adoption rewrite"],
        "inputs": {**parent["inputs"], "parent_wrapper_publication": artifact_ref(REPO_ROOT / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json")}}
    atomic_json(path, value)
    return artifact_ref(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-publication-spec", action="store_true")
    group.add_argument("--prepare-sources", action="store_true")
    group.add_argument("--prepare-g1-spec", action="store_true")
    group.add_argument("--publish-wrapper-bindings", action="store_true")
    group.add_argument("--seal-publication", action="store_true")
    group.add_argument("--prepare-g1-successor", action="store_true")
    group.add_argument("--seal-runtime-repair", action="store_true")
    group.add_argument("--prepare-g1-execution-resume", action="store_true")
    group.add_argument("--seal-executor-recovery", action="store_true")
    group.add_argument("--prepare-wearer-b", action="store_true")
    group.add_argument("--seal-wearer-b-engineering", action="store_true")
    parser.add_argument("--packet", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_publication_spec:
        result = publication_spec()
    elif args.packet is None:
        parser.error("--packet is required for publication work")
    elif args.prepare_sources:
        result = prepare_sources(args.packet)
    elif args.publish_wrapper_bindings:
        result = publish_wrapper_bindings(args.packet)
    elif args.seal_publication:
        result = seal_publication(args.packet)
    elif args.prepare_g1_successor:
        result = prepare_g1_successor(args.packet)
    elif args.seal_runtime_repair:
        result = seal_runtime_repair(args.packet)
    elif args.prepare_g1_execution_resume:
        result = prepare_g1_execution_resume(args.packet)
    elif args.seal_executor_recovery:
        result = seal_executor_recovery(args.packet)
    elif args.prepare_wearer_b:
        result = prepare_wearer_b_spec(args.packet)
    elif args.seal_wearer_b_engineering:
        result = seal_wearer_b_engineering(args.packet)
    else:
        result = prepare_g1_spec(args.packet)
    print(result, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
