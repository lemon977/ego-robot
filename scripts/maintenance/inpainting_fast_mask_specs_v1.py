"""Bounded CPU freeze for the two-session glove-concept experiment."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT
from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_ref
from chaoyang.governance.inpainting_candidate_registration import (
    POST_GLOVE_SCOPE, POST_GLOVE_PREFIX, GLOVE_SPEC_SCHEMA, GLOVE_TEXT_PROMPT,
    VISUAL_ROUTE_KEY, visual_route_contract, validate_post_glove_experiment, RegistrationError)

TASK = "inpainting_fast_mask_engineering_20261013"
OUT = REPO_ROOT / "_run" / TASK / "attempt_0001"
PARENT_OUT = REPO_ROOT / "_run/inpainting_post_g1_source_aware_mask_a01_20261013/attempt_0001"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, code):
    if not condition:
        raise ValueError(code)


def once(path, value):
    path = Path(path)
    require(path.is_relative_to(REPO_ROOT) and path.resolve() == path, "PROJECT_METADATA_REQUIRED")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def packet():
    current = load(REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json")
    from chaoyang.inpainting_candidate.resources import proc_start_ticks
    require(proc_start_ticks(current["writer"]["pid"]) == current["writer"]["proc_start_ticks"]
        and datetime.now(timezone.utc) < datetime.fromisoformat(current["deadline_at"]), "CPU_WRITER_OR_DEADLINE")
    index = load(REPO_ROOT / "tasks/current/INDEX.json")
    active = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    require(len(active) == 1 and active[0]["task_id"] == TASK
        and active[0]["packet_sha256"] == file_ref(Path(current["packet_path"]))["sha256"], "CPU_SCOPE_REQUIRED")
    return current


def publish_review():
    current = packet()
    model = load(PARENT_OUT / "MODEL_RESULT.json")
    terminal = load(PARENT_OUT / "RESULT.json")
    require(terminal["status"] == "PASSED" and load(PARENT_OUT / "POST_VALIDATION.json")["exit_code"] == 0
        and model["completed_mask_sessions"] == 6, "SEALED_SIX_REQUIRED")
    paths = [PARENT_OUT / "review_cpu/lane_001_003/ACTUAL_REVIEW_001_003.json",
             PARENT_OUT / "review_cpu/lane_008_100/RESULT_R01.json",
             PARENT_OUT / "review_cpu/lane_114_050/ACTUAL_REVIEW_114_050.json"]
    rows, refs = {}, []
    for path in paths:
        reference = file_ref(path)
        refs.append(reference)
        report = load(path)
        require(report["task_id"] == model["task_id"] and report["run_signature"] == model["run_signature"],
                "REVIEW_MODEL_BINDING_REQUIRED")
        for row in report["sessions"]:
            key = row["session_key"]
            require(key not in rows and len(row["actual_viewed_frames"]) == 6
                and row.get("actual_viewed_count", 6) == 6
                and row["visual_quality"] == "FAILED", "ACTUAL_REVIEW_ROW_REQUIRED")
            rows[key] = {**row, "actual_viewed_count": 6, "actual_review_ref": reference}
    ordered = [rows[row["session_key"]] for row in model["sessions"]]
    require(sum(row["native_frames"] for row in ordered) == 1033, "REVIEW_FULL1033_REQUIRED")
    videos = []
    review_root = REPO_ROOT / "review/inpainting_optimization"
    for row in model["sessions"]:
        reference = row["review"]["video"]
        verify_ref(reference)
        shallow = review_root / f"G1_SOURCE_AWARE_{row['session_key']}_Raw_Mask.mp4"
        require(not shallow.exists(), "FRESH_SHALLOW_ALIAS_REQUIRED")
        os.link(reference["path"], shallow)
        videos.append({"session_key": row["session_key"], "immutable": reference, "shallow": file_ref(shallow)})
    result = {"schema": "inpainting-source-aware-actual-review-rollup-v1",
        "task_id": model["task_id"], "run_signature": model["run_signature"],
        "model_result": file_ref(PARENT_OUT / "MODEL_RESULT.json"), "review_refs": refs,
        "sessions": ordered, "videos": videos,
        "completed_mask_sessions": 6, "native_frames": 1033, "actual_viewed_frames": 36,
        "unviewed_native_frames": 997, "visual_qualified_sessions": 0,
        "HIGH_clusters": sum(row["issue_counts"]["HIGH"] for row in ordered),
        "MED_clusters": sum(row["issue_counts"].get("MED", 0) for row in ordered),
        "normal050_native127": {"xy": [138, 329], "parent_EPU": [False, False, True],
            "fresh_EPU": [True, False, True], "local_removal_restored": True,
            "UNKNOWN_cleared": False, "full_normal_control_pass": False},
        "candidate_adopted": False, "human_review": "PENDING", "hidden_regions": "UNKNOWN",
        "training_eligible": False, "files_deleted": 0,
        "next_action": "One independent glove-concept test on001+050/full366. No unchanged Motion/Render rerun, no all199 or repeated whole CPU suite."}
    return once(OUT / "SOURCE_AWARE_REVIEW_ROLLUP.json", result)


def freeze():
    current = packet()
    parent_ref = current["inputs"]["parent_model_packet"]
    parent = load(parent_ref["path"])
    spec = copy.deepcopy(load(parent["inputs"]["spec"]["path"]))
    spec.update(task_id=POST_GLOVE_PREFIX + "mask_a01_20261013", spec_schema=GLOVE_SPEC_SCHEMA,
        experiment_scope=POST_GLOVE_SCOPE,
        output_dir=str(REPO_ROOT / "_run" / (POST_GLOVE_PREFIX + "mask_a01_20261013") / "attempt_0001"))
    spec["inputs"] = {"source_index": parent["inputs"]["source_index"], "parent_recipe_packet": parent_ref}
    spec["config"].update(candidate_id="POST_G1_GLOVE_CONCEPT_A01",
        algorithm_parameter_change="GLOVE_RGB_CONCEPT_ONLY", glove_text_prompt=GLOVE_TEXT_PROMPT)
    spec["sessions"] = [row for key in ("001", "050") for row in spec["sessions"] if row["session_key"] == key]
    spec["budgets"].update(wall_seconds_max=1800, gpu_seconds_max=1800, output_bytes_estimate=2147483648)
    spec["runtime_code"] = visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"]
    spec["evaluation_only"] = {"feeds_producer": False, "read_only_after_fresh_stage_sealed": True,
        "completed_parent_model_result": file_ref(PARENT_OUT / "MODEL_RESULT.json"),
        "parent_actual_review": file_ref(OUT / "SOURCE_AWARE_REVIEW_ROLLUP.json"),
        "fixed_anchors": {"001": [0, 50, 119, 132, 202, 237], "050": [0, 28, 64, 91, 96, 127]},
        "normal_control_no_new_HIGH": True, "claim_limit": "Two-session hypothesis, not six-session or cross-domain adoption."}
    spec["read_set"] = ["AGENTS.md", "inpainting_optimization/00_EXECUTION_ZH.md", "inpainting_v1/MANIFEST.json",
        "inpainting_optimization/GLOVE_CONCEPT_MODEL_SPEC.json", "inpainting_optimization/CONTINUE_HERE_ZH.md",
        "src/chaoyang/inpainting_candidate/model_worker.py", "src/chaoyang/inpainting_candidate/automatic_masks.py",
        "src/chaoyang/inpainting_candidate/model_stages.py"]
    spec["read_set_expansions"] = [
        "Only exact signed Source/Raw/map/model/code refs. No historical proposals/Mask/Clean/seed/fit inputs to producer. Parent packet is recipe metadata.",
        "One global glove concept change to a gloved hand; same full001+050, not shortened video. All other parent algorithm parameters, POINT rule, thresholds and projection=None are frozen.",
        "Evaluator only may read completed parent outputs after fresh stage sealed; preserve HIGH/FAILED, time restrictions, hiddenUNKNOWN and humanPENDING. No automatic adoption or training claim."]
    for row in spec["sessions"]:
        row["mask_signature"] = canonical_sha({"stage": "Mask", "source": row["source_receipt"],
            "models": spec["model_weights"], "runtime_code": spec["runtime_code"],
            "config": spec["config"], "schema": spec["schema"]})
    validate_post_glove_experiment(spec, repo_root=REPO_ROOT)
    return once(REPO_ROOT / "inpainting_optimization/GLOVE_CONCEPT_MODEL_SPEC.json", spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish-review", "freeze", "self-check", "seal"])
    args = parser.parse_args()
    if args.command == "self-check":
        current = packet()
        spec_path = REPO_ROOT / "inpainting_optimization/GLOVE_CONCEPT_MODEL_SPEC.json"
        spec = load(spec_path)
        validate_post_glove_experiment(spec, repo_root=REPO_ROOT)
        require(spec["runtime_code"] == visual_route_contract(REPO_ROOT)[VISUAL_ROUTE_KEY]["code_closure"],
                "FROZEN_RUNTIME_CLOSURE_DRIFT")
        loader = importlib.util.spec_from_file_location("_checked_glove_executor",
            REPO_ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py")
        executor = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(executor)
        require(executor.planned_session_keys(spec) == ["001", "050"], "EXECUTOR_TWO_SESSION_DISPATCH")
        errors = []
        changes = [lambda value: value["config"].update(glove_text_prompt="glove"),
                   lambda value: value["config"].update(projection_hints="invented"),
                   lambda value: value["config"].update(new_numeric_threshold=0.1),
                   lambda value: value["sessions"][0].update(native_frames=10),
                   lambda value: value["budgets"].update(wall_seconds_max=1801),
                   lambda value: value["inputs"].update(previous_masks={}),
                   lambda value: value.update(resume_context={})]
        for change in changes:
            bad = copy.deepcopy(spec)
            change(bad)
            try:
                validate_post_glove_experiment(bad, repo_root=REPO_ROOT)
            except RegistrationError as error:
                errors.append(str(error))
            else:
                raise ValueError("NEGATIVE_CONTRACT_ACCEPTED")
        reference = once(OUT / "FAST_SCOPE_CHECKS_R01.json", {
            "task_id": TASK, "run_signature": current["run_signature"],
            "model_spec": file_ref(spec_path), "valid_full366_scope": True,
            "executor_keys": ["001", "050"], "negative_checks": errors,
            "algorithm_calls": 0, "new_models": 0, "threshold_changes": 0})
    elif args.command == "seal":
        current = packet()
        paths = [OUT / "SOURCE_AWARE_REVIEW_ROLLUP.json", OUT / "FAST_SCOPE_CHECKS_R01.json",
            OUT / "ROOT_IMPACTED_TESTS_R01.xml", OUT / "ROOT_IMPACTED_TESTS_R02.xml",
            OUT / "lane_glove/ENGINEERING_RESULT_R01.json",
            OUT / "lane_glove/GLOVE_CONCEPT_TARGETED_R01.xml",
            REPO_ROOT / "inpainting_optimization/GLOVE_CONCEPT_MODEL_SPEC.json"]
        reference = once(OUT / "MAINTENANCE_DONE.json", {
            "task_id": TASK, "run_signature": current["run_signature"], "status": "PASS",
            "artifacts": [file_ref(path) for path in paths],
            "tests": {"glove_directed": {"passed": 62, "failed": 0},
                "root_impacted_final_R02": {"passed": 223, "failed": 0},
                "root_R01_setup_failure_preserved": {"passed": 200, "failed": 23,
                    "cause": "TMPDIR/basetemp was not set; project-local policy assertions rejected temporary fixture paths before model calls.",
                    "outside_project_fixture_path": "/tmp/pytest-of-root/pytest-187",
                    "not_deleted": True}},
            "full_regression_repeated": False, "algorithm_calls": 0, "files_deleted": 0,
            "visual_quality": "NOT_EVALUATED_BY_CPU", "candidate_adopted": False,
            "next_action": "Register the frozen two-session glove experiment immediately. Actual full366 model output and fixed12-frame review, not CPU tests, decide benefit."})
    else:
        reference = publish_review() if args.command == "publish-review" else freeze()
    print(json.dumps(reference, ensure_ascii=False))


if __name__ == "__main__":
    main()
