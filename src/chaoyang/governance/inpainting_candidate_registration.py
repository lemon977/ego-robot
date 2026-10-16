"""Finite CPU admission registration using the existing CAS Task Packet ledger.

No second queue, publisher daemon, GPU lease, model runner or default adoption.
The new CLI/operation/contract publication is intentionally a separate task.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_ref, verify_sealed_stage

OPERATION = "run_inpainting_candidate_cpu_admission_v1"
ROUTE_KEY = "inpainting_candidate_submit_v1"
VISUAL_OPERATION = "run_inpainting_candidate_visual_stages_v1"
VISUAL_ROUTE_KEY = "inpainting_candidate_visual_stages_v1"
CLEAN_SPEC_SCHEMA = "inpainting-finite-g2-clean-spec-v1"
CLEAN_MODEL_CALLS_UNIT = "PROPAINTER_INPAINT_API_INVOCATION"
CLEAN_STAGE_SCOPE = "G2_CLEAN_REFERENCE_SELECTION_AB_ONLY"
POST_CONTROLLER_SCOPE = "POST_G1_CONTROLLER_RGB_PROPOSAL_FAILURE"
POST_CONTROLLER_PHASE = "INPAINTING_POST_G1_CONTROLLER_RGB_PROPOSAL"
POST_CONTROLLER_PREFIX = "inpainting_post_g1_controller_"
POST_CONTROLLER_SESSION_KEYS = ["001", "003", "008", "100", "114", "050"]
POST_TEXT_RETENTION_SCOPE = "POST_G1_TEXT_WEARER_RETENTION_FAILURE"
POST_TEXT_RETENTION_PHASE = "INPAINTING_POST_G1_TEXT_WEARER_RETENTION"
POST_TEXT_RETENTION_PREFIX = "inpainting_post_g1_text_retention_"
TEXT_WEARER_RETENTION_RULE = "PRESERVE_TEXT_WEARER_RGB_SUPPORT_V1"
G3_SCOPE = "G3_FROZEN_CROSS_DOMAIN_MASK_DIAGNOSTIC"
G3_PHASE = "INPAINTING_G3_FROZEN_MASK_EXECUTION"
G3_PREFIX = "inpainting_generalization_g3_mask_"
G3_MANIFEST_SCHEMA = "inpainting-g3-frozen-source-manifest-v1"
G3_SPEC_SCHEMA = "inpainting-finite-g3-mask-spec-v1"
G3_RESULT_SCHEMA = "inpainting-g3-mask-batch-result-v1"
G3_API_BOUND_RULE = "SIX_TEXT_MAX16_TRACKS_MAX3_NATIVE_ANCHORS_MAX2_DIRECTIONS"
POST_SOURCE_AWARE_SCOPE = "POST_G1_SOURCE_AWARE_RETENTION_VETO"
POST_SOURCE_AWARE_PHASE = "INPAINTING_POST_G1_SOURCE_AWARE_RETENTION_VETO"
POST_SOURCE_AWARE_PREFIX = "inpainting_post_g1_source_aware_"
SOURCE_AWARE_SPEC_SCHEMA = "inpainting-finite-source-aware-mask-spec-v1"
SOURCE_AWARE_RETENTION_VETO = "SOURCE_AWARE_POINT_WEARER_V1"
POST_GLOVE_SCOPE = "POST_G1_GLOVE_CONCEPT"
POST_GLOVE_PHASE = "INPAINTING_POST_G1_GLOVE_CONCEPT"
POST_GLOVE_PREFIX = "inpainting_post_g1_glove_concept_"
GLOVE_SPEC_SCHEMA = "inpainting-finite-glove-concept-spec-v1"
GLOVE_TEXT_PROMPT = "a gloved hand"
GEOMETRY_OPERATION = "run_inpainting_candidate_geometry_stages_v1"
GEOMETRY_ROUTE_KEY = "inpainting_candidate_geometry_stages_v1"
GEOMETRY_SPEC_SCHEMA = "inpainting-finite-motion-render-spec-v1"
GEOMETRY_SCOPE = "G2_FULL_LINK_ROBOT_DEPTH_CPU"
GEOMETRY_PHASE = "INPAINTING_CANDIDATE_MOTION_RENDER_CPU"
COMPOSITE_OPERATION = "run_inpainting_candidate_composite_stage_v1"
COMPOSITE_ROUTE_KEY = "inpainting_candidate_composite_stage_v1"
COMPOSITE_SPEC_SCHEMA = "inpainting-finite-composite-spec-v1"
COMPOSITE_SCOPE = "G2_PIXEL_OCCLUSION_CPU"
COMPOSITE_PHASE = "INPAINTING_CANDIDATE_COMPOSITE_CPU"
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
WALL_SECONDS_MAX = 7200
RUNTIME_PATHS = (
    "src/chaoyang/cli/inpainting_submit.py", "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/__main__.py", "src/chaoyang/paths.py",
    "src/chaoyang/inpainting_candidate/__init__.py", "src/chaoyang/inpainting_candidate/entry.py",
    "src/chaoyang/inpainting_candidate/source.py", "src/chaoyang/inpainting_candidate/contracts.py",
    "src/chaoyang/inpainting_candidate/native_contracts.py",
    "src/chaoyang/governance/inpainting_candidate_registration.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py", "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    f"src/chaoyang/ops/{OPERATION}.py",
)


class RegistrationError(RuntimeError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RegistrationError(code)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "GOVERNANCE_JSON_OBJECT_REQUIRED")
    return value


def g3_api_bound(native_frames: int) -> int:
    """Existing six prompts/16-track/three-anchor/two-direction API ceiling.

    This is a resource ceiling, not predicted successful calls or a quality
    threshold. No algorithm module is imported before the operation gate.
    """
    _require(type(native_frames) is int and native_frames > 0, "G3_POSITIVE_NATIVE_FRAMES_REQUIRED")
    return 6 + 16 * min(3, native_frames) + (2 if native_frames > 1 else 1)


def g3_session_inputs(row: dict) -> dict:
    """Drop audit/capability metadata rather than feed it to the producer."""
    fields = {"session_key", "session_id", "raw_session_dir", "native_frames", "source_status"}
    fields.update({"source_receipt", "source_signature", "frame_map_ref"} if row.get("source_status") == "READY"
                  else {"failure_code", "diagnostic_ref"})
    _require(fields <= set(row), "G3_EXPLICIT_SOURCE_ROW_REQUIRED")
    return {key: row[key] for key in sorted(fields)}


def validate_g3_experiment(value: dict, *, repo_root: Path) -> bool:
    """Independent frozen cross-domain diagnostic, never a G1 retry/adoption.

    Selection/parent are signed metadata only. Producer inputs remain this
    manifest's complete Raw Source, never parent masks or evaluation points.
    """
    active = str(value.get("task_id", "")).startswith(G3_PREFIX)
    if not active:
        _require(value.get("experiment_scope") != G3_SCOPE and value.get("spec_schema") != G3_SPEC_SCHEMA,
                 "G3_INDEPENDENT_TASK_REQUIRED")
        return False
    config = value.get("config", {})
    _require(value.get("experiment_scope") == G3_SCOPE and value.get("spec_schema") == G3_SPEC_SCHEMA
        and value.get("model_stages") == ["Mask"] and isinstance(config, dict)
        and config.get("stage") == "G1_AUTOMATIC_MASK_ONLY", "G3_EXACT_MASK_DIAGNOSTIC_SCOPE_REQUIRED")
    _require(not any(k in value for k in ("resume_context", "original_g1_deadline_at")),
             "G3_FRESH_NOT_OLD_TIMEBOX_REQUIRED")
    bounds = value.get("budgets", {})
    _require(all(type(bounds.get(k)) is int and bounds[k] == expected for k, expected in
        {"candidates_max": 1, "gpu_count": 1, "cpu_threads_per_worker": 2, "training_steps_max": 0}.items())
        and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 28800
        and type(bounds.get("model_calls_max")) is int and bounds["model_calls_max"] >= 0
        and bounds.get("model_calls_unit") == "SAM_ADAPTER_INFERENCE_API_INVOCATION",
        "G3_ONE_FRESH_CANDIDATE_SINGLE_CARD_EIGHT_HOURS_REQUIRED")
    inputs = value.get("inputs", {})
    _require(isinstance(inputs, dict) and set(inputs) <= {"source_index", "parent_recipe_packet", "frozen_selection", "spec"}
        and {"source_index", "parent_recipe_packet", "frozen_selection"} <= set(inputs),
        "G3_ONLY_FROZEN_METADATA_INPUTS_REQUIRED")
    root = repo_root.resolve(strict=True)
    for reference in inputs.values():
        _require(isinstance(reference, dict), "G3_SIGNED_METADATA_REF_REQUIRED")
        p = Path(reference.get("path", ""))
        _require(p.is_absolute() and p.resolve(strict=True) == p and p.is_relative_to(root),
                 "G3_METADATA_REAL_PROJECT_REF_REQUIRED")
        verify_ref(reference)
    parent = _load(Path(inputs["parent_recipe_packet"]["path"]))
    manifest = _load(Path(inputs["source_index"]["path"]))
    selection = _load(Path(inputs["frozen_selection"]["path"]))
    _require(parent.get("task_id") == "inpainting_post_g1_text_retention_model_a01_20261013"
        and parent.get("experiment_scope") == POST_TEXT_RETENTION_SCOPE
        and parent.get("model_stages") == ["Mask"], "G3_EXACT_ACTUAL_RETENTION_PARENT_REQUIRED")
    ignored = {"candidate_id", "algorithm_parameter_change"}
    _require({k: v for k, v in config.items() if k not in ignored}
        == {k: v for k, v in parent.get("config", {}).items() if k not in ignored}
        and config.get("controller_text_prompt") == "VR controller"
        and config.get("text_wearer_retention") == TEXT_WEARER_RETENTION_RULE,
        "G3_FROZEN_GLOBAL_RECIPE_DRIFT")
    for domain in ("weights", "model_weights"):
        _require(value.get(domain) == parent.get(domain), "G3_FROZEN_WEIGHT_DRIFT:" + domain)
    _require(value.get("schema") == {**parent.get("schema", {}), "model_result": G3_RESULT_SCHEMA},
             "G3_FROZEN_RESULT_SCHEMA_REQUIRED")
    _require(value.get("calibration") == {"metric_projection": "NOT_CONSUMED_MASK",
        "source_eye_qualification": "PER_SESSION_SOURCE_RECEIPT_NOT_METRIC_PROJECTION"},
        "G3_NO_INHERITED_PARENT_EYE_QUALIFICATION")
    challenges = selection.get("known_challenges_not_blind", [])
    cross = selection.get("cross_domain_selected", [])
    _require(isinstance(challenges, list) and [r.get("session_id") for r in challenges] == ["039", "007", "199"]
        and isinstance(cross, list) and len(cross) <= 8, "G3_ORIGINAL_FIXED_SELECTION_REQUIRED")
    selected_keys = ["pick_place_" + r["session_id"] for r in challenges] + [r["identity"] for r in cross]
    rows, frozen = value.get("sessions"), manifest.get("sessions")
    _require(manifest.get("schema") == G3_MANIFEST_SCHEMA and isinstance(rows, list)
        and isinstance(frozen, list) and [r.get("session_key") for r in rows] == selected_keys
        and [r.get("session_key") for r in frozen] == selected_keys
        and len(set(selected_keys)) == len(selected_keys), "G3_FIXED_MANIFEST_IDENTITIES_REQUIRED")
    raw_refs = {r["expected"]["path"]: r["expected"] for r in selection.get("raw_file_checks", [])
                if isinstance(r, dict) and isinstance(r.get("expected"), dict)}
    for selected in cross:
        raw_refs.update({ref["path"]: ref for ref in selected.get("source_refs", {}).values()})
    total_bound = 0
    for index, (row, original) in enumerate(zip(rows, frozen)):
        allowed = {"session_key", "session_id", "raw_session_dir", "native_frames", "source_status",
            "source_receipt", "source_signature", "frame_map_ref", "failure_code", "diagnostic_ref",
            "mask_signature", "model_calls_max"}
        _require(set(row) <= allowed, "G3_FORBIDDEN_PRODUCER_INPUT")
        _require({k: v for k, v in row.items() if k not in {"mask_signature", "model_calls_max"}} == g3_session_inputs(original),
                 "G3_SIGNED_SOURCE_MANIFEST_DRIFT")
        count = row.get("native_frames")
        _require(type(count) is int and count > 0, "G3_FULL_NATIVE_FRAME_COUNT_REQUIRED")
        raw_dir = row.get("raw_session_dir")
        _require(isinstance(raw_dir, str) and Path(raw_dir).is_absolute(), "G3_ORIGINAL_RAW_DIRECTORY_REQUIRED")
        if index < 3:
            _require(count == challenges[index].get("frames")
                and Path(raw_dir).name == challenges[index]["session_id"]
                and str(Path(raw_dir) / "raw/vst.h264") in raw_refs,
                "G3_ORIGINAL_CHALLENGE_SOURCE_OR_COUNT_DRIFT")
        else:
            _require(raw_dir == cross[index - 3].get("session_dir")
                and row.get("session_id") == cross[index - 3].get("identity"), "G3_ORIGINAL_0918_SOURCE_SWAPPED")
        _require(row.get("source_status") in {"READY", "BLOCKED"}, "G3_SOURCE_STATUS_REQUIRED")
        if row["source_status"] == "BLOCKED":
            _require(isinstance(row.get("failure_code"), str) and bool(row["failure_code"])
                and isinstance(row.get("diagnostic_ref"), dict)
                and not any(k in row for k in ("source_receipt", "source_signature", "frame_map_ref", "mask_signature"))
                and row.get("model_calls_max", 0) == 0, "G3_BLOCKED_SOURCE_CANNOT_FAKE_SEALED")
            p = Path(row["diagnostic_ref"]["path"])
            _require(p.is_relative_to(root) and p.resolve(strict=True) == p, "G3_DIAGNOSTIC_REAL_PROJECT_REF_REQUIRED")
            verify_ref(row["diagnostic_ref"])
            _require(_load(p).get("status") != "SEALED_COMPLETE", "G3_BLOCKED_SOURCE_CANNOT_FAKE_SEALED")
            continue
        import re
        _require(re.fullmatch(r"[0-9a-f]{64}", str(row.get("mask_signature"))) is not None
            and row.get("model_calls_max") == g3_api_bound(count), "G3_SOURCE_SIGNATURE_OR_API_BOUND_REQUIRED")
        reference = row.get("source_receipt")
        _require(isinstance(reference, dict), "G3_READY_SEALED_SOURCE_REQUIRED")
        p = Path(reference.get("path", ""))
        _require(p.is_absolute() and p.is_relative_to(root) and p.resolve(strict=True) == p,
                 "G3_SOURCE_REAL_PROJECT_REF_REQUIRED")
        verify_ref(reference)
        source = _load(p)
        verify_sealed_stage(source, row.get("source_signature"), output_root=p.parent)
        _require(source.get("stage") == "Source" and source.get("native_frames") == count
            and source.get("session_id") == row.get("session_id")
            and row.get("frame_map_ref") in source.get("outputs", []), "G3_SEALED_SOURCE_NATIVE_IDENTITY_DRIFT")
        report_refs = [r for r in source["outputs"] if Path(r["path"]).name == "SOURCE_REPORT.json"]
        _require(len(report_refs) == 1, "G3_ACTUAL_SOURCE_REPORT_REQUIRED")
        report = _load(Path(report_refs[0]["path"]))
        actual = report.get("actual_inputs", {})
        _require(report.get("session_dir") == raw_dir and report.get("session_id") == row["session_id"]
            and report.get("native_sidecar_frames") == count and isinstance(actual, dict)
            and {"video", "manifest", "video_qpc", "video_wall"} <= set(actual)
            and all(ref == raw_refs.get(ref.get("path")) and Path(ref.get("path", "")).is_relative_to(Path(raw_dir))
                    for ref in actual.values())
            and actual["video"]["path"] == str(Path(raw_dir) / "raw/vst.h264"),
            "G3_SOURCE_NOT_ORIGINAL_FROZEN_RAW")
        for ref in actual.values():
            verify_ref(ref)
        _require(_load(Path(actual["manifest"]["path"])).get("session") == row["session_id"],
                 "G3_RAW_MANIFEST_IDENTITY_DRIFT")
        mapping = json.loads(Path(row["frame_map_ref"]["path"]).read_text(encoding="utf-8"))
        qpc_rows = [json.loads(line) for line in Path(actual["video_qpc"]["path"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        _require(len(qpc_rows) == count and all(type(v) is int for v in qpc_rows)
            and all(a < b for a, b in zip(qpc_rows, qpc_rows[1:])), "G3_FULL_ORIGINAL_QPC_RECORD_COUNT_REQUIRED")
        _require(isinstance(mapping, list) and len(mapping) == count
            and all(isinstance(m, dict) and all(m.get(k) == i for k in
                ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
                and m.get("qpc_declared_ns") == qpc_rows[i]
                for i, m in enumerate(mapping)), "G3_FULL_SOURCE_FRAME_MAP_REQUIRED")
        total_bound += row["model_calls_max"]
    _require(bounds["model_calls_max"] == total_bound, "G3_FROZEN_TOTAL_API_BOUND_REQUIRED")
    return True


def validate_post_controller_experiment(value: dict, *, repo_root: Path) -> bool:
    """Finite metadata guard; no algorithm imports or historical Mask reads.

    Old G1/Clean defaults cannot borrow the explicit experimental parameter.
    Parent metadata establishes the unchanged recipe and Source identities,
    not a second producer input or permission to restart G1's exhausted box.
    """
    config = value.get("config", {})
    _require(isinstance(config, dict), "CONTROLLER_CONFIG_OBJECT_REQUIRED")
    if str(value.get("task_id", "")).startswith(POST_SOURCE_AWARE_PREFIX):
        validate_post_source_aware_experiment(value, repo_root=repo_root)
        return False
    _require("source_aware_retention_veto" not in config, "SOURCE_AWARE_PARAMETER_REQUIRES_INDEPENDENT_TASK")
    if str(value.get("task_id", "")).startswith(G3_PREFIX):
        validate_g3_experiment(value, repo_root=repo_root)
        return False
    if value.get("task_id", "").startswith(POST_TEXT_RETENTION_PREFIX):
        validate_post_text_retention_experiment(value, repo_root=repo_root)
        return False  # Independent experiment, never reclassified Controller.
    post = value.get("task_id", "").startswith(POST_CONTROLLER_PREFIX)
    if not post:
        _require("controller_text_prompt" not in config and "text_wearer_retention" not in config
                 and "experiment_scope" not in value,
                 "CONTROLLER_PARAMETER_REQUIRES_INDEPENDENT_POST_G1_TASK")
        return False
    _require("text_wearer_retention" not in config,
             "TEXT_RETENTION_PARAMETER_REQUIRES_INDEPENDENT_POST_G1_TASK")
    _require(value.get("experiment_scope") == POST_CONTROLLER_SCOPE,
             "POST_CONTROLLER_EXPERIMENT_SCOPE_REQUIRED")
    _require(config.get("controller_text_prompt") == "VR controller",
             "POST_CONTROLLER_UNIQUE_GLOBAL_PROMPT_REQUIRED")
    _require(value.get("model_stages") == ["Mask"] and config.get("stage") == "G1_AUTOMATIC_MASK_ONLY",
             "POST_CONTROLLER_MASK_ONLY_SCOPE_REQUIRED")
    _require("resume_context" not in value and "original_g1_deadline_at" not in value,
             "POST_CONTROLLER_FRESH_NOT_G1_SUCCESSOR_REQUIRED")
    budgets = value.get("budgets", {})
    _require(type(budgets.get("candidates_max")) is int and budgets["candidates_max"] == 1
             and type(budgets.get("gpu_count")) is int and budgets["gpu_count"] == 1
             and type(budgets.get("wall_seconds_max")) is int and 0 < budgets["wall_seconds_max"] <= 28800
             and type(budgets.get("model_calls_max")) is int and budgets["model_calls_max"] > 0
             and budgets.get("model_calls_unit") == "SAM_ADAPTER_INFERENCE_API_INVOCATION",
             "POST_CONTROLLER_ONE_FRESH_CANDIDATE_FINITE_BOUNDS_REQUIRED")
    inputs = value.get("inputs", {})
    _require(isinstance(inputs.get("source_index"), dict)
             and isinstance(inputs.get("parent_recipe_packet"), dict),
             "POST_CONTROLLER_FROZEN_SOURCE_AND_RECIPE_REFS_REQUIRED")
    root = repo_root.resolve(strict=True)
    for ref in (inputs["source_index"], inputs["parent_recipe_packet"]):
        path = Path(ref.get("path", ""))
        _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(root),
                 "POST_CONTROLLER_METADATA_PROJECT_REAL_PATH_REQUIRED")
        verify_ref(ref)
    index = _load(Path(inputs["source_index"]["path"]))
    parent = _load(Path(inputs["parent_recipe_packet"]["path"]))
    _require(parent.get("task_id") == "inpainting_generalization_g1_mask_wearer_b_20261013"
             and parent.get("model_stages") == ["Mask"], "POST_CONTROLLER_EXACT_B_PARENT_RECIPE_REQUIRED")
    recipe_keys_ignored = {"candidate_id", "algorithm_parameter_change", "controller_text_prompt"}
    current_recipe = {key: item for key, item in config.items() if key not in recipe_keys_ignored}
    parent_recipe = {key: item for key, item in parent.get("config", {}).items() if key not in recipe_keys_ignored}
    _require(current_recipe == parent_recipe and bool(current_recipe), "POST_CONTROLLER_OTHER_RECIPE_CONFIG_DRIFT")
    _require(index.get("schema") == "inpainting-six-source-stage-index-v1"
             and index.get("native_frames") == 1033
             and isinstance(index.get("sessions"), list)
             and [row.get("session_key") for row in index["sessions"]] == POST_CONTROLLER_SESSION_KEYS
             and isinstance(value.get("sessions"), list)
             and [row.get("session_key") for row in value["sessions"]] == POST_CONTROLLER_SESSION_KEYS
             and [row.get("session_key") for row in parent.get("sessions", [])] == POST_CONTROLLER_SESSION_KEYS,
             "POST_CONTROLLER_FIXED_SIX_FULL_1033_REQUIRED")
    total = 0
    for row, frozen, original in zip(value["sessions"], index["sessions"], parent["sessions"]):
        _require(row.get("source_receipt") == frozen.get("source_receipt") == original.get("source_receipt"),
                 "POST_CONTROLLER_NATIVE_SOURCE_SWAPPED")
        _require(not any(key in row for key in ("mask_receipt", "historical_mask", "manual_seed", "frame_limit")),
                 "POST_CONTROLLER_FRESH_RAW_SOURCE_ONLY")
        reference = row["source_receipt"]
        verify_ref(reference)
        path = Path(reference["path"])
        _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(root),
                 "POST_CONTROLLER_SOURCE_STAGE_PROJECT_REAL_PATH_REQUIRED")
        source = _load(path)
        verify_sealed_stage(source, source.get("component_signature"), output_root=path.parent)
        count = source.get("native_frames")
        _require(source.get("stage") == "Source" and type(count) is int and count > 0
                 and count == frozen.get("native_frames")
                 and ("native_frames" not in row or row["native_frames"] == count),
                 "POST_CONTROLLER_NATIVE_FRAMES_SHORTENED")
        total += count
    _require(total == 1033, "POST_CONTROLLER_FIXED_SIX_FULL_1033_REQUIRED")
    return True


def validate_post_text_retention_experiment(value: dict, *, repo_root: Path) -> bool:
    """One finite retention-only successor experiment; parent is metadata only.

    Exact Source identities, unchanged prompt/recipe/weights and no manual
    inputs. This guard reads no historical proposal, Mask, Clean or q arrays.
    """
    config = value.get("config", {})
    _require(isinstance(config, dict), "TEXT_RETENTION_CONFIG_OBJECT_REQUIRED")
    if str(value.get("task_id", "")).startswith(POST_SOURCE_AWARE_PREFIX):
        validate_post_source_aware_experiment(value, repo_root=repo_root)
        return False
    _require("source_aware_retention_veto" not in config, "SOURCE_AWARE_PARAMETER_REQUIRES_INDEPENDENT_TASK")
    if str(value.get("task_id", "")).startswith(G3_PREFIX):
        validate_g3_experiment(value, repo_root=repo_root)
        return False
    post = value.get("task_id", "").startswith(POST_TEXT_RETENTION_PREFIX)
    if not post:
        _require("text_wearer_retention" not in config
            and value.get("experiment_scope") != POST_TEXT_RETENTION_SCOPE,
            "TEXT_RETENTION_PARAMETER_REQUIRES_INDEPENDENT_POST_G1_TASK")
        return False
    _require(value.get("experiment_scope") == POST_TEXT_RETENTION_SCOPE
        and config.get("text_wearer_retention") == TEXT_WEARER_RETENTION_RULE,
        "POST_TEXT_RETENTION_EXACT_SCOPE_AND_RULE_REQUIRED")
    _require(config.get("controller_text_prompt") == "VR controller"
        and value.get("model_stages") == ["Mask"] and config.get("stage") == "G1_AUTOMATIC_MASK_ONLY",
        "POST_TEXT_RETENTION_PARENT_PROMPT_AND_MASK_SCOPE_REQUIRED")
    _require(not any(key in value for key in ("resume_context", "original_g1_deadline_at"))
        and not any(key in config for key in ("manual_seed", "seed", "manual_points", "session_config", "frame_limit")),
        "POST_TEXT_RETENTION_FRESH_RAW_ONLY_REQUIRED")
    budgets = value.get("budgets", {})
    _require(all(type(budgets.get(key)) is int and budgets[key] == expected for key, expected in
        {"candidates_max": 1, "gpu_count": 1, "cpu_threads_per_worker": 2,
         "model_calls_max": 215, "training_steps_max": 0}.items())
        and type(budgets.get("wall_seconds_max")) is int and 0 < budgets["wall_seconds_max"] <= 7200
        and budgets.get("model_calls_unit") == "SAM_ADAPTER_INFERENCE_API_INVOCATION",
        "POST_TEXT_RETENTION_ONE_FRESH_CANDIDATE_215_API_TWO_HOURS_REQUIRED")
    root = repo_root.resolve(strict=True)
    inputs = value.get("inputs", {})
    _require(isinstance(inputs, dict)
        and isinstance(inputs.get("source_index"), dict)
        and isinstance(inputs.get("parent_recipe_packet"), dict)
        and not any(key in inputs for key in ("manual_seed", "mask_receipt", "historical_mask", "proposal_arrays", "evaluation_annotation")),
        "POST_TEXT_RETENTION_SOURCE_AND_PARENT_METADATA_REQUIRED")
    for reference in (inputs["source_index"], inputs["parent_recipe_packet"]):
        path = Path(reference.get("path", ""))
        _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(root),
            "POST_TEXT_RETENTION_METADATA_REAL_PROJECT_REF_REQUIRED")
        verify_ref(reference)
    parent = _load(Path(inputs["parent_recipe_packet"]["path"]))
    index = _load(Path(inputs["source_index"]["path"]))
    _require(parent.get("task_id") == "inpainting_post_g1_controller_rgb_model_a01_20261013"
        and parent.get("experiment_scope") == POST_CONTROLLER_SCOPE and parent.get("model_stages") == ["Mask"]
        and parent.get("config", {}).get("controller_text_prompt") == "VR controller"
        and "text_wearer_retention" not in parent.get("config", {}),
        "POST_TEXT_RETENTION_EXACT_CONTROLLER_A01_PARENT_REQUIRED")
    ignored = {"candidate_id", "algorithm_parameter_change", "text_wearer_retention"}
    current_config = {key: item for key, item in config.items() if key not in ignored}
    parent_config = {key: item for key, item in parent["config"].items() if key not in ignored}
    _require(current_config == parent_config and bool(current_config), "POST_TEXT_RETENTION_OTHER_RECIPE_CONFIG_DRIFT")
    for domain in ("weights", "model_weights", "calibration"):
        _require(value.get(domain) == parent.get(domain), "POST_TEXT_RETENTION_OTHER_FROZEN_DOMAIN_DRIFT:" + domain)
    rows = value.get("sessions")
    _require(index.get("schema") == "inpainting-six-source-stage-index-v1" and index.get("native_frames") == 1033
        and isinstance(rows, list) and isinstance(index.get("sessions"), list)
        and isinstance(parent.get("sessions"), list)
        and all([row.get("session_key") for row in group] == POST_CONTROLLER_SESSION_KEYS
            for group in (rows, index["sessions"], parent["sessions"])),
        "POST_TEXT_RETENTION_FIXED_SIX_FULL1033_REQUIRED")
    total = 0
    for row, frozen, original in zip(rows, index["sessions"], parent["sessions"]):
        _require(not any(key in row for key in ("mask_receipt", "historical_mask", "manual_seed", "seed", "frame_limit",
            "manual_points", "session_config", "proposal_arrays", "evaluation_annotation", "placement")),
            "POST_TEXT_RETENTION_FRESH_RAW_SOURCE_ONLY")
        reference = row.get("source_receipt")
        _require(reference == frozen.get("source_receipt") == original.get("source_receipt"),
            "POST_TEXT_RETENTION_NATIVE_SOURCE_SWAPPED")
        verify_ref(reference)
        path = Path(reference["path"])
        _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(root),
            "POST_TEXT_RETENTION_SOURCE_STAGE_REAL_PROJECT_REF_REQUIRED")
        source = _load(path)
        verify_sealed_stage(source, source.get("component_signature"), output_root=path.parent)
        count = source.get("native_frames")
        _require(source.get("stage") == "Source" and type(count) is int and count > 0
            and count == frozen.get("native_frames")
            and ("native_frames" not in row or row["native_frames"] == count),
            "POST_TEXT_RETENTION_NATIVE_FRAMES_SHORTENED")
        total += count
    _require(total == 1033, "POST_TEXT_RETENTION_FIXED_SIX_FULL1033_REQUIRED")
    return True


def validate_post_source_aware_experiment(value: dict, *, repo_root: Path) -> bool:
    """A new single-variable six-source candidate, never a hot G3 retune."""
    active = str(value.get("task_id", "")).startswith(POST_SOURCE_AWARE_PREFIX)
    config = value.get("config", {})
    _require(isinstance(config, dict), "SOURCE_AWARE_CONFIG_OBJECT_REQUIRED")
    if not active:
        _require("source_aware_retention_veto" not in config
            and value.get("experiment_scope") != POST_SOURCE_AWARE_SCOPE,
            "SOURCE_AWARE_PARAMETER_REQUIRES_INDEPENDENT_TASK")
        return False
    _require(value.get("experiment_scope") == POST_SOURCE_AWARE_SCOPE
        and value.get("spec_schema") == SOURCE_AWARE_SPEC_SCHEMA
        and config.get("source_aware_retention_veto") == SOURCE_AWARE_RETENTION_VETO
        and value.get("model_stages") == ["Mask"] and config.get("stage") == "G1_AUTOMATIC_MASK_ONLY",
        "SOURCE_AWARE_EXACT_OPT_IN_MASK_SCOPE_REQUIRED")
    _require(not any(k in value for k in ("resume_context", "original_g1_deadline_at")),
        "SOURCE_AWARE_FRESH_FINITE_TASK_REQUIRED")
    bounds = value.get("budgets", {})
    _require(all(type(bounds.get(k)) is int and bounds[k] == v for k, v in
        {"candidates_max": 1, "gpu_count": 1, "cpu_threads_per_worker": 2,
         "model_calls_max": 215, "training_steps_max": 0}.items())
        and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 7200
        and bounds.get("model_calls_unit") == "SAM_ADAPTER_INFERENCE_API_INVOCATION",
        "SOURCE_AWARE_ONE_CANDIDATE_215_API_TWO_HOURS_REQUIRED")
    root = Path(repo_root).resolve(strict=True)
    inputs = value.get("inputs", {})
    _require(isinstance(inputs, dict) and set(inputs) <= {"source_index", "parent_recipe_packet", "spec"}
        and {"source_index", "parent_recipe_packet"} <= set(inputs), "SOURCE_AWARE_METADATA_INPUTS_ONLY_REQUIRED")
    for reference in inputs.values():
        _require(isinstance(reference, dict), "SOURCE_AWARE_SIGNED_METADATA_REQUIRED")
        path = Path(reference.get("path", ""))
        _require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path,
            "SOURCE_AWARE_REAL_PROJECT_REF_REQUIRED")
        verify_ref(reference)
    parent = _load(Path(inputs["parent_recipe_packet"]["path"]))
    index = _load(Path(inputs["source_index"]["path"]))
    _require(parent.get("task_id") == "inpainting_post_g1_text_retention_model_a01_20261013"
        and parent.get("experiment_scope") == POST_TEXT_RETENTION_SCOPE and parent.get("model_stages") == ["Mask"],
        "SOURCE_AWARE_EXACT_COMPLETED_RETENTION_PARENT_REQUIRED")
    ignored = {"candidate_id", "algorithm_parameter_change", "source_aware_retention_veto"}
    _require({k: v for k, v in config.items() if k not in ignored}
        == {k: v for k, v in parent.get("config", {}).items() if k not in ignored},
        "SOURCE_AWARE_OTHER_ALGORITHM_PARAMETER_DRIFT")
    for domain in ("weights", "model_weights", "calibration", "schema"):
        _require(value.get(domain) == parent.get(domain), "SOURCE_AWARE_FROZEN_DOMAIN_DRIFT:" + domain)
    rows = value.get("sessions")
    _require(index.get("schema") == "inpainting-six-source-stage-index-v1" and index.get("native_frames") == 1033
        and isinstance(rows, list) and all(isinstance(group, list)
        and [row.get("session_key") for row in group] == POST_CONTROLLER_SESSION_KEYS
        for group in (rows, index.get("sessions"), parent.get("sessions"))), "SOURCE_AWARE_FIXED_SIX_FULL1033_REQUIRED")
    from chaoyang.inpainting_candidate.native_contracts import source_binding
    total = 0
    for row, frozen, original in zip(rows, index["sessions"], parent["sessions"]):
        _require(set(row) <= {"session_key", "source_receipt", "mask_signature", "native_frames"},
            "SOURCE_AWARE_RAW_SOURCE_ONLY_REQUIRED")
        reference = row.get("source_receipt")
        _require(reference == frozen.get("source_receipt") == original.get("source_receipt"),
            "SOURCE_AWARE_NATIVE_SOURCE_SWAPPED")
        verify_ref(reference)
        path = Path(reference["path"])
        source = _load(path)
        source_row = {"session_key": row["session_key"], "session_id": source["session_id"],
            "native_frames": frozen["native_frames"], "source_receipt": reference,
            "source_signature": source["component_signature"],
            "frame_map_ref": next(ref for ref in source["outputs"] if Path(ref["path"]).name == "FRAME_MAP.json")}
        source_binding(source_row, root)
        _require("native_frames" not in row or row["native_frames"] == source_row["native_frames"],
            "SOURCE_AWARE_FULL_NATIVE_FRAMES_REQUIRED")
        total += source_row["native_frames"]
    _require(total == 1033, "SOURCE_AWARE_FIXED_SIX_FULL1033_REQUIRED")
    return True


def validate_post_glove_experiment(value: dict, *, repo_root: Path) -> bool:
    """One concept change, two complete native sessions, no legacy relaxation."""
    config = value.get("config", {})
    _require(isinstance(config, dict), "GLOVE_CONFIG_OBJECT_REQUIRED")
    active = str(value.get("task_id", "")).startswith(POST_GLOVE_PREFIX)
    if not active:
        _require("glove_text_prompt" not in config and value.get("experiment_scope") != POST_GLOVE_SCOPE,
                 "GLOVE_PARAMETER_REQUIRES_INDEPENDENT_TASK")
        return False
    _require(value.get("experiment_scope") == POST_GLOVE_SCOPE
        and value.get("spec_schema") == GLOVE_SPEC_SCHEMA
        and value.get("model_stages") == ["Mask"]
        and config.get("stage") == "G1_AUTOMATIC_MASK_ONLY"
        and config.get("glove_text_prompt") == GLOVE_TEXT_PROMPT,
        "GLOVE_EXACT_SINGLE_PARAMETER_SCOPE_REQUIRED")
    _require(not any(key in value for key in ("resume_context", "original_g1_deadline_at")),
             "GLOVE_FRESH_FINITE_TASK_REQUIRED")
    bounds = value.get("budgets", {})
    _require(all(type(bounds.get(key)) is int and bounds[key] == number for key, number in
        {"candidates_max": 1, "gpu_count": 1, "cpu_threads_per_worker": 2,
         "model_calls_max": 215, "training_steps_max": 0}.items())
        and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 1800
        and bounds.get("model_calls_unit") == "SAM_ADAPTER_INFERENCE_API_INVOCATION",
        "GLOVE_SINGLE_CANDIDATE_30_MINUTE_BOUND_REQUIRED")
    root = Path(repo_root).resolve(strict=True)
    inputs = value.get("inputs", {})
    _require(isinstance(inputs, dict) and set(inputs) <= {"source_index", "parent_recipe_packet", "spec"}
        and {"source_index", "parent_recipe_packet"} <= set(inputs), "GLOVE_METADATA_INPUTS_ONLY_REQUIRED")
    for reference in inputs.values():
        _require(isinstance(reference, dict), "GLOVE_SIGNED_METADATA_REQUIRED")
        path = Path(reference.get("path", ""))
        _require(path.is_absolute() and path.is_relative_to(root) and path.resolve(strict=True) == path,
                 "GLOVE_REAL_PROJECT_REF_REQUIRED")
        verify_ref(reference)
    parent = _load(Path(inputs["parent_recipe_packet"]["path"]))
    index = _load(Path(inputs["source_index"]["path"]))
    _require(parent.get("task_id") == "inpainting_post_g1_source_aware_mask_a01_20261013"
        and parent.get("experiment_scope") == POST_SOURCE_AWARE_SCOPE
        and parent.get("model_stages") == ["Mask"], "GLOVE_EXACT_SOURCE_AWARE_PARENT_REQUIRED")
    ignored = {"candidate_id", "algorithm_parameter_change", "glove_text_prompt"}
    _require({key: item for key, item in config.items() if key not in ignored}
        == {key: item for key, item in parent.get("config", {}).items() if key not in ignored},
        "GLOVE_OTHER_ALGORITHM_PARAMETER_DRIFT")
    for domain in ("weights", "model_weights", "calibration", "schema"):
        _require(value.get(domain) == parent.get(domain), "GLOVE_FROZEN_DOMAIN_DRIFT:" + domain)
    rows = value.get("sessions")
    _require(index.get("schema") == "inpainting-six-source-stage-index-v1"
        and isinstance(rows, list) and [row.get("session_key") for row in rows] == ["001", "050"],
        "GLOVE_FIXED_TWO_FULL_NATIVE_REQUIRED")
    from chaoyang.inpainting_candidate.native_contracts import source_binding
    total = 0
    for row in rows:
        key = row["session_key"]
        _require(set(row) <= {"session_key", "source_receipt", "mask_signature", "native_frames"},
                 "GLOVE_RAW_SOURCE_ONLY_REQUIRED")
        frozen = next(item for item in index["sessions"] if item["session_key"] == key)
        original = next(item for item in parent["sessions"] if item["session_key"] == key)
        _require(row.get("source_receipt") == frozen["source_receipt"] == original["source_receipt"],
                 "GLOVE_NATIVE_SOURCE_SWAPPED")
        source = _load(Path(row["source_receipt"]["path"]))
        source_row = {"session_key": key, "session_id": source["session_id"],
            "native_frames": frozen["native_frames"], "source_receipt": row["source_receipt"],
            "source_signature": source["component_signature"],
            "frame_map_ref": next(ref for ref in source["outputs"] if Path(ref["path"]).name == "FRAME_MAP.json")}
        source_binding(source_row, root)
        _require(row.get("native_frames") == source_row["native_frames"], "GLOVE_FULL_NATIVE_FRAMES_REQUIRED")
        total += source_row["native_frames"]
    _require(total == 366, "GLOVE_FULL366_REQUIRED")
    return True


def _validate_native(kind: str, spec: dict, *, repo_root: Path, verify_inputs: bool = True) -> None:
    from chaoyang.inpainting_candidate import native_contracts
    try:
        getattr(native_contracts, "validate_native_" + kind + "_spec")(
            spec, repo_root=repo_root, verify_inputs=verify_inputs)
    except native_contracts.ContractError as error:
        raise RegistrationError(str(error)) from error


def route_contract(repo_root: Path) -> dict[str, Any]:
    """Additive hook for current_r3_contracts in the next publication task."""
    root = Path(repo_root).resolve(strict=True)
    if any(not (root / item).is_file() for item in RUNTIME_PATHS):
        return {}  # Draft files are not a runnable operation contract.
    return {ROUTE_KEY: {"recipe": "inpainting-raw-to-robot", "candidate_operation": OPERATION,
        "execution_status": "CANDIDATE_FENCED_CPU_ADMISSION_ONLY", "task_packet_required": True,
        "current_runnable_operation": None, "model_calls_authorized": 0,
        "render_calls_authorized": 0, "training_steps_authorized": 0,
        "code_closure": [file_ref(root / item) for item in RUNTIME_PATHS],
        "output_schema": "inpainting-candidate-cpu-admission-stage-v1",
        "motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
        "claim_limit": "CPU source/decode admission only; no Robot output, visual adoption, training or deployment qualification."}}


def mco_route_contract(repo_root: Path) -> dict[str, Any]:
    """Original-V1 finite M/C/O experiments; never change default adoption."""
    root = Path(repo_root).resolve(strict=True)
    operation = "run_inpainting_v1_mco_stage_v1"
    required = set(RUNTIME_PATHS) | {
        f"src/chaoyang/ops/{operation}.py",
        "scripts/maintenance/inpainting_v1_mco_task.py",
        "scripts/maintenance/inpainting_candidate_model_task_v1.py",
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "src/chaoyang/inpainting_candidate/v1_mco_worker.py",
        "src/chaoyang/inpainting_candidate/v1_mask_repair.py",
        "src/chaoyang/inpainting_candidate/v1_point_observer.py",
        "src/chaoyang/inpainting_candidate/v1_clean_ablation.py",
        "src/chaoyang/inpainting_candidate/v1_boundary_alpha.py",
        "src/chaoyang/inpainting_candidate/model_stages.py",
        "src/chaoyang/inpainting_candidate/model_worker.py",
        "src/chaoyang/inpainting_candidate/resources.py",
        "src/chaoyang/inpainting_candidate/source_frames.py",
        "src/chaoyang/inpainting_candidate/automatic_masks.py",
        "src/chaoyang/inpainting_candidate/automatic_v1_seeds.py",
        "src/chaoyang/inpainting_candidate/clean_references.py",
        "src/chaoyang/inpainting_candidate/clean_flow.py",
        "src/chaoyang/storage_relocation/__init__.py",
        "src/chaoyang/storage_relocation/pick_place_original199_v1.py",
        "src/chaoyang/humanego_ict/project_mount_paths.py",
        "src/chaoyang/ops/immutable_artifact_io.py",
        "src/chaoyang/pipeline/robot_renderer_cycles.py",
        "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
        "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
        "src/chaoyang/pipeline/robot_tool_definition_contract.py",
        "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
        "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
        "src/chaoyang/pipeline/exact78_visual_compositor_v1.py",
        "src/chaoyang/pipeline/pico_manus_motion_v2.py",
        "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
        "src/chaoyang/pipeline/__init__.py",
        "configs/systems/mask/configs/sam31_runtime_source_v1.json",
        "manifests/vendor.json",
    }
    # New finite compatible mode stays on the same registered operation,
    # publisher and card lease. Optional during bounded CPU implementation;
    # actual execution still requires its complete frozen component closure.
    additions = (
        "src/chaoyang/inpainting_candidate/v1_clean_pair.py",
        "src/chaoyang/inpainting_candidate/v1_mask_clean_causal.py",
        "src/chaoyang/inpainting_candidate/v1_clean_context.py",
        "src/chaoyang/inpainting_candidate/v1_clean_observer.py",
        "src/chaoyang/inpainting_candidate/v1_full_pixel_context.py",
        "src/chaoyang/inpainting_candidate/v1_device_proposals.py",
        "src/chaoyang/inpainting_candidate/v1_equipment_rgb.py",
        "src/chaoyang/inpainting_candidate/v1_context_exclusion.py",
        "src/chaoyang/inpainting_candidate/v1_context_builder.py",
        "src/chaoyang/inpainting_candidate/v1_resume_plan.py",
        "src/chaoyang/inpainting_candidate/v1_context_seam.py",
        "src/chaoyang/inpainting_candidate/v1_boundary_builder.py",
        "src/chaoyang/inpainting_candidate/v1_scene_donor.py",
        "src/chaoyang/inpainting_candidate/v1_scene_donor_worker.py",
        "src/chaoyang/inpainting_candidate/v1_scene_donor_local.py",
        "src/chaoyang/inpainting_candidate/v1_spatial_clean.py",
        "src/chaoyang/inpainting_candidate/automation.py",
        "src/chaoyang/inpainting_candidate/automation_worker.py",
        "src/chaoyang/inpainting_candidate/automation_mask.py",
        "src/chaoyang/inpainting_candidate/automation_paths.py",
        "scripts/maintenance/inpainting_v1_automation_specs.py",
    )
    required.update(relative for relative in additions if (root / relative).is_file())
    if any(not (root / relative).is_file() for relative in required):
        return {}
    paths = {root / relative for relative in required}
    for relative in ("src/chaoyang/exact78_baseline", "src/chaoyang/pick_place_1008",
                     "vendor/SAM3/sam3", "vendor/ProPainter/model",
                     "vendor/ProPainter/RAFT", "vendor/ProPainter/utils"):
        paths.update(path for path in (root / relative).rglob("*.py")
                     if "__pycache__" not in path.parts)
    paths.add(root / "vendor/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    if any(path.resolve(strict=True) != path or not path.is_relative_to(root) for path in paths):
        raise RegistrationError("MCO_RUNTIME_ALIAS_OR_PROJECT_ESCAPE")
    return {"inpainting_original_v1_mco_candidate_v1": {
        "operation": operation, "task_packet_required": True,
        "allowed_modes": ["Mask", "Clean", "Boundary", "Automation"],
        "execution_status": "FINITE_ORIGINAL_V1_CANDIDATE_ONLY",
        "current_runnable_operation": None,
        "code_closure": [file_ref(path) for path in sorted(paths)],
        "output_schema": "inpainting-v1-mco-stage-result-v1",
        "default_baseline": "INPAINTING_RAW_TO_ROBOT_V1",
        "candidate_adopted": False, "training_steps_authorized": 0,
        "model_calls_authorized": "ONLY_FINITE_PACKET_AND_ONE_CARD_LEASE",
        "render_calls_authorized": "ONLY_FINITE_CPU_BOUNDARY_PACKET",
        "claim_limit": "Separate Mask, Clean and display-boundary experiments; not geometry, quality approval, adoption or control labels.",
    }}


def visual_route_contract(repo_root: Path) -> dict[str, Any]:
    """Freeze the actual Mask/Clean worker seams and existing GPU runner.

    Missing operation files keep this a draft.  This function does not import
    any model or load checkpoint bytes.  Weights are separately signed in the
    finite packet, not mislabeled as Python code here.
    """
    root = Path(repo_root).resolve(strict=True)
    required = set(RUNTIME_PATHS) | {
        f"src/chaoyang/ops/{VISUAL_OPERATION}.py",
        "scripts/maintenance/inpainting_candidate_model_task_v1.py",
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "src/chaoyang/governance/exact78_operation_gate.py",
        "src/chaoyang/exact78_baseline/__init__.py",
        "src/chaoyang/exact78_baseline/mask.py", "src/chaoyang/exact78_baseline/clean.py",
        "src/chaoyang/pick_place_1008/__init__.py", "src/chaoyang/pick_place_1008/sam_point_adapter.py",
        "src/chaoyang/pipeline/__init__.py", "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
        "src/chaoyang/inpainting_candidate/model_worker.py", "src/chaoyang/inpainting_candidate/model_stages.py",
        "src/chaoyang/inpainting_candidate/resources.py", "src/chaoyang/inpainting_candidate/automatic_masks.py",
        "src/chaoyang/inpainting_candidate/clean_references.py", "src/chaoyang/inpainting_candidate/source_frames.py",
        "src/chaoyang/inpainting_candidate/clean_worker.py", "src/chaoyang/inpainting_candidate/clean_flow.py",
        "src/chaoyang/pick_place_1008/visual.py", "src/chaoyang/pipeline/exact78_visual_compositor_v1.py",
        "vendor/ProPainter/model/__init__.py", "vendor/ProPainter/model/propainter.py",
        "vendor/ProPainter/model/recurrent_flow_completion.py", "vendor/ProPainter/model/modules/flow_comp_raft.py",
        "vendor/ProPainter/model/modules/flow_loss_utils.py", "vendor/ProPainter/RAFT/__init__.py",
        "vendor/ProPainter/RAFT/raft.py",
        "vendor/SAM3/sam3/__init__.py", "vendor/SAM3/sam3/model_builder.py",
        "vendor/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "configs/systems/mask/configs/sam31_runtime_source_v1.json", "manifests/vendor.json",
    }
    if any(not (root / relative).is_file() for relative in required):
        return {**geometry_route_contract(root), **composite_route_contract(root)}
    auto_seed_paths = {
        "src/chaoyang/inpainting_candidate/automatic_v1_seeds.py",
        "src/chaoyang/ops/run_inpainting_auto_v1_seed_mask_v1.py",
        "scripts/maintenance/inpainting_auto_v1_seed_one_hour_v1.py",
        "src/chaoyang/pick_place_1008/source.py", "src/chaoyang/pick_place_1008/camera.py",
        "src/chaoyang/exact78_baseline/pico_source.py", "src/chaoyang/pipeline/pico_manus_motion_v2.py",
    }
    auto_seed_ready = all((root / relative).is_file() for relative in auto_seed_paths)
    if auto_seed_ready:
        required.update(auto_seed_paths)
    late_entry_paths = {
        "src/chaoyang/inpainting_candidate/late_entry_seeds.py",
        "src/chaoyang/ops/run_inpainting_late_entry_mask_v1.py",
        "scripts/maintenance/inpainting_late_entry_one_hour_v1.py",
    }
    late_entry_ready = auto_seed_ready and all((root / relative).is_file() for relative in late_entry_paths)
    if late_entry_ready:
        required.update(late_entry_paths)
    observer_paths = {
        "src/chaoyang/inpainting_candidate/controller_observer.py",
        "src/chaoyang/inpainting_candidate/controller_replay.py",
        "src/chaoyang/ops/run_inpainting_controller_observer_v1.py",
        "scripts/maintenance/inpainting_controller_observer_v1.py",
    }
    observer_ready = all((root / relative).is_file() for relative in observer_paths)
    if observer_ready:
        required.update(observer_paths)
    local_crop_paths = {
        "src/chaoyang/inpainting_candidate/local_crop.py",
        "src/chaoyang/inpainting_candidate/local_crop_worker.py",
        "src/chaoyang/ops/run_inpainting_local_crop_v1.py",
        "scripts/maintenance/inpainting_local_crop_v1.py",
    }
    local_crop_ready = observer_ready and all((root / relative).is_file() for relative in local_crop_paths)
    if local_crop_ready:
        required.update(local_crop_paths)
    neutral_object_paths = {
        "src/chaoyang/inpainting_candidate/neutral_object_prompts.py",
        "src/chaoyang/inpainting_candidate/neutral_object_worker.py",
        "src/chaoyang/ops/run_inpainting_neutral_object_mask_v1.py",
        "scripts/maintenance/inpainting_neutral_object_v1.py",
    }
    neutral_object_ready = auto_seed_ready and all((root / relative).is_file() for relative in neutral_object_paths)
    if neutral_object_ready:
        required.update(neutral_object_paths)
    paths = {root / relative for relative in required}
    paths.update(path for path in (root / "vendor/SAM3/sam3").rglob("*.py")
                 if "__pycache__" not in path.parts)
    for relative in ("vendor/ProPainter/model", "vendor/ProPainter/RAFT", "vendor/ProPainter/utils"):
        paths.update(path for path in (root / relative).rglob("*.py") if "__pycache__" not in path.parts)
    if any(path.resolve(strict=True) != path or not path.is_relative_to(root) for path in paths):
        raise RegistrationError("VISUAL_RUNTIME_ALIAS_OR_PROJECT_ESCAPE")
    return {**mco_route_contract(root), **geometry_route_contract(root), **composite_route_contract(root), VISUAL_ROUTE_KEY: {
        "candidate_operation": VISUAL_OPERATION, "execution_status": "CANDIDATE_FENCED_MASK_OR_CLEAN",
        "auto_v1_seed_operation": "run_inpainting_auto_v1_seed_mask_v1" if auto_seed_ready else None,
        "late_entry_operation": "run_inpainting_late_entry_mask_v1" if late_entry_ready else None,
        "controller_observer_operation": "run_inpainting_controller_observer_v1" if observer_ready else None,
        "local_crop_operation": "run_inpainting_local_crop_v1" if local_crop_ready else None,
        "neutral_object_operation": "run_inpainting_neutral_object_mask_v1" if neutral_object_ready else None,
        "task_packet_required": True, "current_runnable_operation": None,
        "model_calls_authorized": "ONLY_WITHIN_FROZEN_STAGE_PACKET_API_BUDGET",
        "model_calls_unit": "SAM_ADAPTER_INFERENCE_API_INVOCATION",
        "stage_model_calls_units": {"Mask": "SAM_ADAPTER_INFERENCE_API_INVOCATION", "Clean": CLEAN_MODEL_CALLS_UNIT},
        "render_calls_authorized": 0, "training_steps_authorized": 0,
        "code_closure": [file_ref(path) for path in sorted(paths)],
        "output_schema": "inpainting-g1-mask-batch-result-v1", "model_stages": ["Mask", "Clean"],
        "stage_output_schemas": {"Mask": "inpainting-g1-mask-batch-result-v1", "Clean": "inpainting-g2-clean-batch-result-v1"},
        "independent_mask_diagnostic": {"scope": G3_SCOPE, "result_schema": G3_RESULT_SCHEMA,
            "source_manifest_schema": G3_MANIFEST_SCHEMA, "candidate_adopted": False},
        "motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS_NOT_EXECUTED_IN_G1",
        "candidate_adopted": False,
        "native_clean_scope": "INPAINTING_NATIVE_BASELINE_CLEAN",
        "source_aware_mask_scope": POST_SOURCE_AWARE_SCOPE,
        "claim_limit": "Separately signed finite Mask or baseline native Clean candidates; no Robot output, visual adoption, training or deployment qualification.",
    }}


def geometry_route_contract(repo_root: Path) -> dict[str, Any]:
    """Add a separate zero-model CPU route, never borrow engineering rights."""
    root = Path(repo_root).resolve(strict=True)
    required = set(RUNTIME_PATHS) | {
        f"src/chaoyang/ops/{GEOMETRY_OPERATION}.py",
        "scripts/maintenance/inpainting_candidate_geometry_task_v1.py",
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "src/chaoyang/inpainting_candidate/geometry_worker.py",
        "src/chaoyang/inpainting_candidate/motion.py",
        "src/chaoyang/inpainting_candidate/render.py",
        "src/chaoyang/inpainting_candidate/motion_source.py",
        "src/chaoyang/inpainting_candidate/source_frames.py",
        "src/chaoyang/pick_place_1008/__init__.py",
        "src/chaoyang/pick_place_1008/source.py",
        "src/chaoyang/pick_place_1008/camera.py",
        "src/chaoyang/exact78_baseline/__init__.py",
        "src/chaoyang/exact78_baseline/pico_source.py",
        "src/chaoyang/exact78_baseline/retarget.py",
        "src/chaoyang/exact78_baseline/robot_model.py",
        "src/chaoyang/exact78_baseline/render.py",
        "src/chaoyang/pipeline/__init__.py",
        "src/chaoyang/pipeline/pico_manus_motion_v2.py",
        "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
        "src/chaoyang/pipeline/robot_renderer_cycles.py",
        "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
        "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
        "src/chaoyang/pipeline/robot_tool_definition_contract.py",
        "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
        "src/chaoyang/ops/immutable_artifact_io.py",
    }
    if any(not (root / item).is_file() for item in required):
        return {}
    paths = sorted(root / item for item in required)
    _require(all(path.resolve(strict=True) == path for path in paths),
             "GEOMETRY_RUNTIME_ALIAS_OR_PROJECT_ESCAPE")
    return {GEOMETRY_ROUTE_KEY: {
        "candidate_operation": GEOMETRY_OPERATION,
        "execution_status": "CANDIDATE_FENCED_CPU_MOTION_RENDER_ONLY",
        "task_packet_required": True, "current_runnable_operation": None,
        "model_calls_authorized": 0, "gpu_count": 0,
        "render_calls_authorized": "ONLY_WITHIN_SIGNED_NATIVE_PACKET_TWO_VIEWS_OR_ZERO_NATIVE_ONLY",
        "training_steps_authorized": 0, "candidate_adopted": False,
        "code_closure": [file_ref(path) for path in paths],
        "output_schema": "inpainting-geometry-batch-result-v1",
        "stages": ["Motion", "Render"],
        "native_scope": "INPAINTING_NATIVE_MOTION_RENDER_CPU",
        "native_stages": [["Native"], ["Motion", "Render"]],
        "claim_limit": "Independent finite Source-native CPU Motion/Robot or Native-only; local blockers and quality failures remain explicit; no Clean/Composite, adoption, training or deployment qualification.",
    }}


def geometry_signature(spec: dict, code_closure: list[dict]) -> str:
    """All original finite spec domains are signed, including raw and assets."""
    return canonical_sha({"spec": spec, "code_closure": code_closure})


def composite_route_contract(repo_root: Path) -> dict[str, Any]:
    """One independently registered pixel Composite over sealed stages only."""
    root = Path(repo_root).resolve(strict=True)
    required = set(RUNTIME_PATHS) | {
        f"src/chaoyang/ops/{COMPOSITE_OPERATION}.py",
        "scripts/maintenance/inpainting_candidate_composite_task_v1.py",
        "scripts/maintenance/inpainting_candidate_geometry_task_v1.py",
        "scripts/humanego_ict/production_resume_runner_v1.py",
        "src/chaoyang/inpainting_candidate/composite.py",
        "src/chaoyang/inpainting_candidate/composite_worker.py",
        "src/chaoyang/inpainting_candidate/occlusion.py",
        "src/chaoyang/inpainting_candidate/clean_references.py",
        "src/chaoyang/inpainting_candidate/resources.py",
        "src/chaoyang/inpainting_candidate/source_frames.py",
        "src/chaoyang/exact78_baseline/__init__.py",
        "src/chaoyang/exact78_baseline/clean.py",
        "src/chaoyang/ops/immutable_artifact_io.py",
    }
    if any(not (root / relative).is_file() for relative in required):
        return {}
    paths = sorted(root / relative for relative in required)
    _require(all(path.resolve(strict=True) == path for path in paths),
             "COMPOSITE_RUNTIME_ALIAS_OR_PROJECT_ESCAPE")
    return {COMPOSITE_ROUTE_KEY: {
        "candidate_operation": COMPOSITE_OPERATION,
        "execution_status": "CANDIDATE_FENCED_CPU_COMPOSITE_ONLY",
        "task_packet_required": True, "current_runnable_operation": None,
        "model_calls_authorized": 0, "gpu_count": 0, "render_calls_authorized": 0,
        "training_steps_authorized": 0, "candidate_adopted": False,
        "code_closure": [file_ref(path) for path in paths],
        "output_schema": "inpainting-composite-batch-result-v1", "stages": ["Composite"],
        "native_scope": "INPAINTING_NATIVE_COMPOSITE_CPU",
        "claim_limit": "Separately signed finite native diagnostic Composite; model-predicted raw support unverified, scene geometry UNKNOWN, upstream failures retained. No adoption/training/deployment.",
    }}


def composite_signature(spec: dict, code_closure: list[dict]) -> str:
    return canonical_sha({"spec": spec, "code_closure": code_closure})


def validate_composite_spec(spec: dict, *, repo_root: Path,
                            verify_inputs: bool = True) -> None:
    """No algorithm imports, model loading or quality threshold changes."""
    if spec.get("experiment_scope") == "INPAINTING_NATIVE_COMPOSITE_CPU":
        return _validate_native("composite", spec, repo_root=repo_root, verify_inputs=verify_inputs)
    root = Path(repo_root).resolve(strict=True)
    _require(spec.get("spec_schema") == COMPOSITE_SPEC_SCHEMA
        and spec.get("experiment_scope") == COMPOSITE_SCOPE
        and spec.get("operation") == COMPOSITE_OPERATION and spec.get("stages") == ["Composite"],
        "COMPOSITE_SPEC_SCOPE_REQUIRED")
    task_id = spec.get("task_id", "")
    _require(task_id.startswith("inpainting_composite_001_")
             and task_id.replace("_", "").isalnum(), "COMPOSITE_FIXED001_TASK_REQUIRED")
    bounds = spec.get("budgets", {})
    _require(all(type(bounds.get(k)) is int and bounds[k] == v for k, v in
        {"gpu_count": 0, "model_calls_max": 0, "training_steps_max": 0,
         "cpu_threads_per_worker": 2, "candidates_max": 1, "frames_max": 238,
         "review_videos": 2, "render_calls_max": 0}.items())
        and type(bounds.get("wall_seconds_max")) is int
        and 0 < bounds["wall_seconds_max"] <= WALL_SECONDS_MAX,
        "COMPOSITE_FINITE_ZERO_MODEL_CPU_BOUNDS_REQUIRED")
    _require(spec.get("config") == {"scene_depth": "ABSENT",
        "display_policy": "RAW_MODEL_OBJECT_FIRST_DIAGNOSTIC", "image_size": [640, 480],
        "complete_robot": True}, "COMPOSITE_FROZEN_DIAGNOSTIC_RECIPE_REQUIRED")
    _require(not spec.get("model_weights") and not any(key in spec for key in
        ("resume_context", "model_stages", "original_g1_deadline_at")), "COMPOSITE_NO_MODELS_OR_OLD_TASK_RESUME")
    _require(all(name in spec for name in ("inputs", "runtime_code", "calibration", "schema")),
             "COMPOSITE_SIGNATURE_DOMAINS_REQUIRED")
    rows = spec.get("sessions")
    _require(isinstance(rows, list) and len(rows) == 1 and rows[0].get("session_key") == "001"
        and type(rows[0].get("native_frames")) is int and rows[0]["native_frames"] == 238,
        "COMPOSITE_FIXED001_FULL238_REQUIRED")
    row = rows[0]
    _require(not any(key in row for key in ("manual_seed", "seed", "frame_limit", "placement", "robot_q",
        "historical_motion", "manual_placement", "scene_depth", "evaluation_annotation")),
        "COMPOSITE_FORBIDDEN_INPUT")
    import re
    stages = ("Source", "Mask", "Clean", "Render")
    _require(all(re.fullmatch(r"[0-9a-f]{64}", str(row.get(name.lower() + "_signature")))
                 for name in (*stages, "Composite")), "COMPOSITE_STAGE_SIGNATURES_REQUIRED")
    for stage in stages:
        reference = row.get(stage.lower() + "_receipt")
        _require(isinstance(reference, dict) and {"path", "bytes", "sha256"} <= reference.keys(),
                 "COMPOSITE_SIGNED_UPSTREAM_REF_REQUIRED:" + stage)
        path = Path(reference["path"])
        _require(path.is_absolute() and path.is_relative_to(root), "COMPOSITE_UPSTREAM_OUTSIDE_PROJECT")
    quality_refs = row.get("upstream_quality_evidence")
    _require(isinstance(quality_refs, dict) and set(quality_refs) == {"Mask", "Clean"},
             "COMPOSITE_COMPACT_UPSTREAM_QUALITY_REQUIRED")
    if not verify_inputs:
        return
    def verify_tree(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                verify_ref(value)
            else:
                for child in value.values():
                    verify_tree(child)
        elif isinstance(value, list):
            for child in value:
                verify_tree(child)
    verify_tree(spec)
    values = {}
    for stage in stages:
        reference = row[stage.lower() + "_receipt"]
        path = Path(reference["path"])
        _require(path.resolve(strict=True) == path, "COMPOSITE_UPSTREAM_ALIAS_FORBIDDEN")
        receipt = _load(path)
        verify_sealed_stage(receipt, row[stage.lower() + "_signature"], output_root=path.parent)
        _require(receipt.get("stage") == stage
            and (receipt.get("native_frames") == 238 or
                receipt.get("native_input_frames") == receipt.get("native_output_frames") == 238),
            "COMPOSITE_FULL_NATIVE_SEALED_UPSTREAM_REQUIRED:" + stage)
        values[stage] = receipt
    _require(values["Render"].get("source_receipt") == row["source_receipt"]
        and values["Render"].get("source_signature") == row["source_signature"]
        and values["Render"].get("complete_robot") is True,
        "COMPOSITE_RENDER_SOURCE_OR_FULL_ROBOT_DRIFT")
    _require(values["Clean"].get("actual_inputs", {}).get("mask_receipt") == row["mask_receipt"]
        and values["Clean"].get("actual_inputs", {}).get("raw_source") == values["Mask"].get("actual_inputs", {}).get("raw_source"),
        "COMPOSITE_CLEAN_MASK_RAW_INPUT_DRIFT")
    for stage, reference in quality_refs.items():
        verify_ref(reference)
        path = Path(reference["path"])
        _require(path.resolve(strict=True) == path and path.is_relative_to(root),
                 "COMPOSITE_COMPACT_QUALITY_REAL_PROJECT_REF_REQUIRED")
        quality = _load(path)
        _require(quality.get("schema") == "inpainting-upstream-quality-status-v1"
            and quality.get("stage") == stage
            and quality.get("upstream_receipt") == row[stage.lower() + "_receipt"]
            and quality.get("visual_quality") == "FAILED"
            and not any(key in quality for key in ("points", "annotations", "pixel_points", "manual_seed")),
            "COMPOSITE_KNOWN_UPSTREAM_FAILURE_METADATA_DRIFT")


def authorize_composite_operation(operation: str, args: list[str], *, repo_root: Path,
                                  full: bool = True) -> dict:
    """Fenced CPU only; reject before Composite/codec module imports."""
    _require(operation == COMPOSITE_OPERATION, "COMPOSITE_OPERATION_UNSUPPORTED")
    root = Path(repo_root).resolve(strict=True)
    _, state, index, contract = _snapshot(root)
    route = contract.get("special_status_contracts", {}).get(COMPOSITE_ROUTE_KEY)
    _require(isinstance(route, dict) and route.get("candidate_operation") == COMPOSITE_OPERATION
        and route.get("task_packet_required") is True and route.get("current_runnable_operation") is None
        and route.get("model_calls_authorized") == route.get("gpu_count") == route.get("render_calls_authorized") == 0,
        "COMPOSITE_ROUTE_ABSENT")
    if full:
        _require(route == composite_route_contract(root).get(COMPOSITE_ROUTE_KEY), "COMPOSITE_ROUTE_SHA_DRIFT")
    packet = _active_packet(root, state, [row for row in index["task_packets"] if row.get("execution_allowed") is True])
    expected_phase = ("INPAINTING_NATIVE_COMPOSITE_CPU"
        if packet.get("experiment_scope") == "INPAINTING_NATIVE_COMPOSITE_CPU" else COMPOSITE_PHASE)
    _require(packet.get("phase") == expected_phase
        and packet.get("execution_mode") == "CANDIDATE_COMPOSITE_CPU"
        and packet.get("algorithm_execution_allowed") is True and packet.get("operation") == COMPOSITE_OPERATION,
        "COMPOSITE_REGISTERED_EXECUTION_PACKET_REQUIRED")
    _require(route.get("code_closure") and all(reference in packet.get("runtime_code", []) for reference in route["code_closure"]),
        "COMPOSITE_CODE_CLOSURE_INCOMPLETE")
    expected = ["--packet", str(root / "tasks/current" / packet["task_id"] / "TASK_PACKET.json")]
    commands = packet.get("commands", [])
    tokens = shlex.split(commands[0]) if commands else []
    _require(args == expected and len(commands) == 1
        and tokens[1:6] == ["-B", "-m", "chaoyang.cli", "run", COMPOSITE_OPERATION]
        and tokens[6:] == args, "COMPOSITE_EXACT_COMMAND_REQUIRED")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    _require(_caller_is_writer_descendant(packet["writer"]["pid"], packet["writer"]["proc_start_ticks"], os.getpid()),
             "COMPOSITE_CALLER_NOT_FENCED_WRITER_DESCENDANT")
    _require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]), "COMPOSITE_FINITE_DEADLINE")
    reference = packet.get("inputs", {}).get("spec")
    _require(isinstance(reference, dict), "COMPOSITE_SIGNED_SPEC_REQUIRED")
    verify_ref(reference)
    spec = _load(Path(reference["path"]))
    validate_composite_spec(spec, repo_root=root, verify_inputs=full)
    for key, expected_value in spec.items():
        actual = packet.get(key)
        if key == "inputs":
            actual = {k: v for k, v in packet["inputs"].items() if k != "spec"}
        _require(actual == expected_value, "COMPOSITE_PACKET_SPEC_DRIFT:" + key)
    _require(composite_signature(spec, route["code_closure"]) == packet.get("run_signature"), "COMPOSITE_RUN_SIGNATURE_DRIFT")
    return packet


def validate_geometry_spec(spec: dict, *, repo_root: Path,
                           verify_inputs: bool = True) -> None:
    """Bounded CPU validation; no Motion/Render or model module imports."""
    if spec.get("experiment_scope") == "INPAINTING_NATIVE_MOTION_RENDER_CPU":
        return _validate_native("geometry", spec, repo_root=repo_root, verify_inputs=verify_inputs)
    root = Path(repo_root).resolve(strict=True)
    _require(spec.get("spec_schema") == GEOMETRY_SPEC_SCHEMA
             and spec.get("experiment_scope") == GEOMETRY_SCOPE
             and spec.get("operation") == GEOMETRY_OPERATION
             and spec.get("stages") == ["Motion", "Render"],
             "GEOMETRY_SPEC_SCOPE_REQUIRED")
    task_id = spec.get("task_id", "")
    _require(task_id.startswith("inpainting_motion_render_001_")
             and task_id.replace("_", "").isalnum(), "GEOMETRY_FIXED001_TASK_REQUIRED")
    bounds = spec.get("budgets", {})
    _require(all(type(bounds.get(k)) is int and bounds[k] == v for k, v in
        {"gpu_count": 0, "model_calls_max": 0, "training_steps_max": 0,
         "cpu_threads_per_worker": 2, "candidates_max": 1,
         "render_frames_max": 238, "render_views": 2, "render_calls_max": 476}.items())
        and type(bounds.get("wall_seconds_max")) is int
        and 0 < bounds["wall_seconds_max"] <= WALL_SECONDS_MAX,
        "GEOMETRY_FINITE_ZERO_MODEL_CPU_BOUNDS_REQUIRED")
    _require(spec.get("config") == {"complete_robot": True, "image_size": [640, 480],
             "motion_rule": "FROZEN_V1_RETARGET_AUTOMATIC_BASE", "scene_depth": "ABSENT"},
             "GEOMETRY_FROZEN_RECIPE_REQUIRED")
    _require(not spec.get("model_weights") and not any(key in spec for key in
             ("resume_context", "model_stages", "original_g1_deadline_at")),
             "GEOMETRY_NO_MODEL_OR_OLD_TASK_RESUME")
    _require(all(name in spec for name in ("inputs", "runtime_code", "calibration", "schema")),
             "GEOMETRY_SIGNATURE_DOMAINS_REQUIRED")
    rows = spec.get("sessions")
    _require(isinstance(rows, list) and len(rows) == 1
             and rows[0].get("session_key") == "001"
             and type(rows[0].get("native_frames")) is int and rows[0]["native_frames"] == 238,
             "GEOMETRY_FIXED001_FULL238_REQUIRED")
    row = rows[0]
    _require(not any(k in row for k in ("q", "robot_q", "historical_motion", "mask_receipt",
             "clean_receipt", "manual_seed", "seed", "manual_placement", "placement", "frame_limit")),
             "GEOMETRY_FORBIDDEN_DERIVED_INPUT")
    import re
    _require(all(re.fullmatch(r"[0-9a-f]{64}", str(row.get(k))) for k in
                 ("source_signature", "motion_signature", "render_signature")),
             "GEOMETRY_STAGE_SIGNATURES_REQUIRED")
    source_ref = row.get("source_receipt", {})
    camera = row.get("camera_contract", {})
    _require(isinstance(camera, dict) and set(camera) == {"v1_manifest", "camera_config"},
             "GEOMETRY_EXACT_FROZEN_CAMERA_CONTRACT_REQUIRED")
    raw = row.get("raw_inputs")
    _require(isinstance(raw, dict) and raw and not any(k in raw for k in
             ("q", "robot_q", "historical_motion", "mask", "clean", "manual_seed", "placement")),
             "GEOMETRY_RAW_INPUT_SET_REQUIRED")
    for reference in raw.values():
        _require(isinstance(reference, dict) and {"path", "bytes", "sha256"} <= reference.keys(),
                 "GEOMETRY_ACTUAL_RAW_FILE_REFS_REQUIRED")
        path = Path(reference["path"])
        _require(path.is_absolute() and not ({p.lower() for p in path.parts}
                  & {"_run", "review", "processed", "attempts"})
                 and path.name not in {"source.npz", "MASK_ARRAYS.npz", "MOTION.npz"},
                 "GEOMETRY_HISTORICAL_RAW_INPUT_FORBIDDEN")
    assets = spec.get("robot_assets")
    _require(isinstance(assets, list) and len(assets) == 69
             and len({r.get("path") for r in assets}) == 69,
             "GEOMETRY_FROZEN_69_ASSETS_REQUIRED")
    if not verify_inputs:
        return  # Frame-level authority still verifies packet/code/fence/command.
    def verify_tree(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                verify_ref(value)
            else:
                for child in value.values():
                    verify_tree(child)
        elif isinstance(value, list):
            for child in value:
                verify_tree(child)
    verify_tree(spec)
    for reference in [source_ref, *camera.values(), *assets]:
        p = Path(reference["path"])
        _require(p.resolve(strict=True) == p and p.is_relative_to(root),
                 "GEOMETRY_FROZEN_LOCAL_INPUT_REAL_PATH_REQUIRED")
    source_path = Path(source_ref["path"])
    source = _load(source_path)
    verify_sealed_stage(source, row["source_signature"], output_root=source_path.parent)
    reports = [ref for ref in source["outputs"] if Path(ref["path"]).name == "SOURCE_REPORT.json"]
    _require(source.get("stage") == "Source" and source.get("native_frames") == 238
             and len(reports) == 1, "GEOMETRY_SEALED_SOURCE_FULL238_REQUIRED")
    report = _load(Path(reports[0]["path"]))
    _require(report.get("actual_inputs") == raw, "GEOMETRY_RAW_INPUTS_NOT_CURRENT_SOURCE")
    closure_ref = spec["inputs"].get("asset_closure")
    _require(isinstance(closure_ref, dict), "GEOMETRY_ASSET_CLOSURE_REF_REQUIRED")
    closure = _load(Path(closure_ref["path"]))
    expected_assets = [{k: item[k] for k in ("path", "bytes", "sha256")}
                       for item in closure.get("assets", [])]
    _require(closure.get("asset_count") == 69
             and sorted(assets, key=lambda r: r["path"]) == sorted(expected_assets, key=lambda r: r["path"]),
             "GEOMETRY_ASSET_CLOSURE_DRIFT")


def authorize_geometry_operation(operation: str, args: list[str], *, repo_root: Path,
                                 full: bool = True) -> dict:
    """Before algorithm import, bind one current CPU packet and real writer.

    ``full=False`` is only a repeated in-worker authority check. Initial calls
    and stage boundaries use the default full input/asset verification.
    """
    _require(operation == GEOMETRY_OPERATION, "GEOMETRY_OPERATION_UNSUPPORTED")
    root = Path(repo_root).resolve(strict=True)
    _, state, index, contract = _snapshot(root)
    route = contract.get("special_status_contracts", {}).get(GEOMETRY_ROUTE_KEY)
    _require(isinstance(route, dict) and route.get("candidate_operation") == GEOMETRY_OPERATION
             and route.get("task_packet_required") is True and route.get("current_runnable_operation") is None
             and route.get("model_calls_authorized") == 0 and route.get("gpu_count") == 0,
             "GEOMETRY_ROUTE_ABSENT")
    if full:
        _require(route == geometry_route_contract(root).get(GEOMETRY_ROUTE_KEY),
                 "GEOMETRY_ROUTE_SHA_DRIFT")
    packet = _active_packet(root, state, [r for r in index["task_packets"] if r.get("execution_allowed") is True])
    _require(packet.get("phase") == GEOMETRY_PHASE
             and packet.get("execution_mode") == "CANDIDATE_GEOMETRY_CPU"
             and packet.get("algorithm_execution_allowed") is True
             and packet.get("operation") == GEOMETRY_OPERATION,
             "GEOMETRY_REGISTERED_EXECUTION_PACKET_REQUIRED")
    runtime = packet.get("runtime_code", [])
    _require(route.get("code_closure") and all(r in runtime for r in route["code_closure"]),
             "GEOMETRY_CODE_CLOSURE_INCOMPLETE")
    expected = ["--packet", str(root / "tasks/current" / packet["task_id"] / "TASK_PACKET.json")]
    tokens = shlex.split(packet.get("commands", [""])[0])
    _require(args == expected and len(packet.get("commands", [])) == 1
             and tokens[1:6] == ["-B", "-m", "chaoyang.cli", "run", GEOMETRY_OPERATION]
             and tokens[6:] == args, "GEOMETRY_EXACT_COMMAND_REQUIRED")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    writer = packet["writer"]
    _require(_caller_is_writer_descendant(writer["pid"], writer["proc_start_ticks"], os.getpid()),
             "GEOMETRY_CALLER_NOT_FENCED_WRITER_DESCENDANT")
    _require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]),
             "GEOMETRY_FINITE_DEADLINE")
    spec_ref = packet.get("inputs", {}).get("spec")
    _require(isinstance(spec_ref, dict), "GEOMETRY_SIGNED_SPEC_REQUIRED")
    verify_ref(spec_ref)
    spec = _load(Path(spec_ref["path"]))
    validate_geometry_spec(spec, repo_root=root, verify_inputs=full)
    for key, value in spec.items():
        actual = packet.get(key)
        if key == "inputs":
            actual = {k: v for k, v in packet["inputs"].items() if k != "spec"}
        _require(actual == value, "GEOMETRY_PACKET_SPEC_DRIFT:" + key)
    _require(geometry_signature(spec, route["code_closure"]) == packet.get("run_signature"),
             "GEOMETRY_RUN_SIGNATURE_DRIFT")
    return packet


def _snapshot(root: Path) -> tuple[dict, dict, dict, dict]:
    # Existing publisher holds LOCK_EX across its multi-file commit.  Reading
    # under the same LOCK_SH prevents a valid heartbeat becoming a false drift.
    with (root / "_run/current/locks/governance.lock").open("rb") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_SH)
        try:
            receipt = _load(root / "docs/governance/CURRENT_STATUS_RECEIPT.json")
            refs = receipt.get("files", {})
            required = {"task_state": root / "docs/governance/LONG_HORIZON_TASK_STATE.json",
                        "task_packet_index_current": root / "tasks/current/INDEX.json",
                        "algorithm_contract": root / "docs/governance/ALGORITHM_CONTRACT.json"}
            values = {}
            for name, path in required.items():
                reference = refs.get(name)
                _require(isinstance(reference, dict) and reference.get("path") == str(path), "CURRENT_RECEIPT_REFERENCE_ABSENT:" + name)
                verify_ref(reference)
                values[name] = _load(path)
                _require(values[name].get("governance_revision") == receipt.get("governance_revision")
                         and values[name].get("generation_id") == receipt.get("generation_id"), "CURRENT_REVISION_OR_GENERATION_DRIFT")
            return receipt, values["task_state"], values["task_packet_index_current"], values["algorithm_contract"]
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


def _route(root: Path, contract: dict) -> dict:
    expected = route_contract(root).get(ROUTE_KEY)
    actual = contract.get("special_status_contracts", {}).get(ROUTE_KEY)
    _require(expected is not None and actual == expected, "CPU_CANDIDATE_ROUTE_ABSENT_OR_SHA_DRIFT")
    return actual


def cpu_signature(request: dict, closure: list[dict]) -> str:
    from chaoyang.inpainting_candidate.entry import verify_request
    verify_request(request)
    return canonical_sha({"input_manifest": request, "code": closure,
        "config": {"cpu_threads": 2, "decode_probe": True, "gpu_count": 0},
        "weights": {"status": "NOT_CONSUMED_CPU_ADMISSION"},
        "calibration": [source.get("camera", {"status": "ABSENT"}) for source in request["sources"]],
        "schema": "inpainting-candidate-cpu-admission-stage-v1"})


def _paths(root: Path, task_id: str) -> tuple[Path, Path, Path]:
    return (root / "_run/current" / task_id / "attempts/attempt_0001",
            root / "tasks/current" / task_id / "TASK_PACKET.json",
            root / "tasks/receipts" / (task_id.upper() + ".json"))


def _live(pid: int, ticks: int) -> bool:
    from chaoyang.governance.common import process_identity
    try:
        return type(pid) is int and type(ticks) is int and pid > 1 and process_identity(pid)["start_ticks"] == ticks
    except (OSError, ValueError, IndexError, KeyError):
        return False


def _active_packet(root: Path, state: dict, allowed: list[dict]) -> dict:
    _require(len(allowed) == 1, "MULTIPLE_ACTIVE_PACKETS")
    row = allowed[0]
    task_id = row.get("task_id")
    _require(isinstance(task_id, str) and task_id.replace("_", "").isalnum(), "ACTIVE_TASK_ID_INVALID")
    _, path, _ = _paths(root, task_id)
    _require(row.get("packet_path") == str(path.relative_to(root)) and file_ref(path)["sha256"] == row.get("packet_sha256"), "ACTIVE_PACKET_SHA_OR_PATH_DRIFT")
    packet = _load(path)
    _require(packet.get("task_id") == task_id, "ACTIVE_PACKET_IDENTITY_DRIFT")
    output = Path(str(packet.get("output_dir", "")))
    writer_file = Path(str(packet.get("writer_file", "")))
    _require(output.is_absolute() and output.resolve() == output and output != root and output.is_relative_to(root)
             and writer_file == output / "WRITER.json", "ACTIVE_OUTPUT_OR_WRITER_FILE_OUTSIDE_PROJECT")
    _require(any((root / raw if not Path(raw).is_absolute() else Path(raw)).resolve() == output
                 for raw in packet.get("write_set", []) if isinstance(raw, str)), "ACTIVE_OUTPUT_NOT_IN_WRITE_SET")
    matches = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    _require(len(matches) == 1 and matches[0].get("status") == "RUNNING"
             and matches[0].get("task_packet") == file_ref(path)
             and matches[0].get("run_signature") == packet.get("run_signature")
             and (state.get("next_task") or {}).get("task_id") == task_id, "ACTIVE_PACKET_STATE_DRIFT")
    writer = packet.get("writer", {})
    _require(writer == _load(output / "WRITER.json") and writer.get("scope") == task_id
             and bool(writer.get("fencing_token")) and type(writer.get("executor_epoch")) is int
             and writer.get("executor_epoch") >= 1 and matches[0].get("pid") == writer.get("pid")
             and matches[0].get("proc_start_ticks") == writer.get("proc_start_ticks")
             and _live(writer.get("pid"), writer.get("proc_start_ticks")), "ACTIVE_WRITER_FENCE_DRIFT")
    for reference in packet.get("runtime_code", []):
        _require(Path(str(reference.get("path", ""))).is_absolute()
                 and Path(reference["path"]).resolve(strict=True).is_relative_to(root), "RUNTIME_CODE_OUTSIDE_PROJECT")
        verify_ref(reference)
    return packet


def plan_request(request: dict, *, repo_root: Path) -> dict:
    """Read-only same-signature attachment or registration readiness decision."""
    root = repo_root.resolve(strict=True)
    _, state, index, contract = _snapshot(root)
    route = _route(root, contract)
    signature = cpu_signature(request, route["code_closure"])
    task_id = "inpainting_cpu_admission_" + signature[:24]
    base = {"task_id": task_id, "run_signature": signature,
            "submission_input_sha256": request["submission_input_sha256"],
            "execution_started_here": False, "robot_video_produced": False,
            "visual_qualified": False, "training_eligible": False, "deployable": False}
    allowed = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    if allowed:
        packet = _active_packet(root, state, allowed)
        _require(packet.get("phase") == "INPAINTING_CANDIDATE_CPU_ADMISSION"
                 and packet.get("run_signature") == signature
                 and packet.get("inputs", {}).get("request_sha256") == canonical_sha(request), "ACTIVE_DIFFERENT_SIGNATURE_OR_WRITER_CONFLICT")
        return {**base, "status": "ATTACHED_TO_ACTIVE_CPU_ADMISSION"}
    _require(not state.get("next_task") and not any(item.get("status") in ACTIVE for item in state.get("tasks", [])), "ACTIVE_STATE_WITHOUT_PACKET_CONFLICT")
    prior = [item for item in state.get("tasks", []) if item.get("task_id") == task_id]
    output, packet_path, receipt_path = _paths(root, task_id)
    if prior:
        _require(len(prior) == 1 and prior[0].get("status") in {"PASSED", "FAILED_RUNTIME_FINAL"}, "PRIOR_NONTERMINAL_OR_AMBIGUOUS_TASK")
        references = (prior[0].get("task_packet"), prior[0].get("result"), prior[0].get("terminal_receipt"))
        for reference, path in zip(references, (packet_path, output / "RESULT.json", receipt_path)):
            _require(isinstance(reference, dict) and reference.get("path") == str(path), "TERMINAL_REFERENCE_ABSENT_OR_PATH_DRIFT")
            verify_ref(reference)
        packet, result, terminal = (_load(path) for path in (packet_path, output / "RESULT.json", receipt_path))
        _require(all(item.get("run_signature") == signature and item.get("task_id") == task_id for item in (packet, result, terminal))
                 and result.get("status") == prior[0].get("status") == terminal.get("status")
                 and terminal.get("result") == references[1], "TERMINAL_SIGNATURE_OR_STATUS_DRIFT")
        return {**base, "status": "ATTACHED_TO_TERMINAL_CPU_ADMISSION", "terminal_status": result["status"], "result": references[1]}
    _require(not any(path.exists() for path in (output, packet_path, receipt_path)), "UNSEALED_ATTEMPT_CONFLICT")
    return {**base, "status": "CPU_ADMISSION_READY_NOT_REGISTERED"}


def write_once_json(path: Path, value: dict) -> None:
    from chaoyang.governance.common import atomic_json
    _require(not path.exists(), "IMMUTABLE_OUTPUT_ALREADY_EXISTS:" + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def register(request: dict, *, repo_root: Path) -> dict:
    """Use the existing CAS publisher.  No direct uncontracted execution."""
    from chaoyang.governance.common import now_iso, process_identity, publish_bundle
    from chaoyang.governance.register_single_task_packet import _validate_packet
    root = repo_root.resolve(strict=True)
    # publish_bundle's authority paths belong to the real project only.
    from chaoyang.governance.common import REPO_ROOT
    _require(root == REPO_ROOT, "REGISTRATION_REAL_PROJECT_REQUIRED")
    decision = plan_request(request, repo_root=root)
    if decision["status"] != "CPU_ADMISSION_READY_NOT_REGISTERED":
        return decision
    receipt, state, index, contract = _snapshot(root)
    _require(not state.get("next_task") and not any(row.get("execution_allowed") is True for row in index["task_packets"]), "REGISTRATION_WINDOW_CHANGED")
    route = _route(root, contract)
    task_id = decision["task_id"]
    output, packet_path, terminal_path = _paths(root, task_id)
    request_path = output / "REQUEST.json"
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": task_id}
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", OPERATION,
               "--request", str(request_path), "--output-root", str(output)]
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Generic original Controller/Head+MANUS CPU source/decode admission only",
        "phase": "INPAINTING_CANDIDATE_CPU_ADMISSION", "execution_mode": "INPAINTING_CANDIDATE_CPU",
        "plan_revision": "INPAINTING_GENERALIZATION_199_0918_20261012", "execution_revision": task_id.upper(),
        "read_set": ["AGENTS.md", "inpainting_v1/MANIFEST.json", "inpainting_optimization/00_EXECUTION_ZH.md"],
        "read_set_expansions": ["Only actual raw source refs frozen in REQUEST.json and pinned runtime code"],
        "write_set": [str(output.relative_to(root)), str(packet_path.relative_to(root)), str(terminal_path.relative_to(root)),
                      "Existing CAS publisher generated governance projections only"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "CPU_ROUTE_PINNED", "RAW_SOURCE_SHA_PINNED"],
        "required_outputs": ["REQUEST.json", "ADMISSION_STAGE.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"cpu_threads_total": 2, "gpu_count": 0, "model_calls_max": 0, "render_calls_max": 0,
                    "training_steps_max": 0, "wall_seconds_max": WALL_SECONDS_MAX, "attempts_max": 1},
        "stop_conditions": ["deadline", "writer conflict", "input/code SHA drift", "one sealed CPU result"],
        "claim_limit": "CPU admission only, not Raw-to-Robot execution or visual/training/deployment qualification",
        "writer": writer, "writer_file": str(output / "WRITER.json"), "output_dir": str(output),
        "request_path": str(request_path), "run_signature": decision["run_signature"],
        "inputs": {"request_sha256": canonical_sha(request), "submission_input_sha256": request["submission_input_sha256"]},
        "runtime_code": route["code_closure"], "commands": [shlex.join(command)],
        "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "deletion_allowed": False, "algorithm_execution_allowed": False,
        "control_ground_truth": False, "physical_deployment_authorized": False}
    _require(not _validate_packet(packet), "CPU_TASK_PACKET_INVALID")
    _require(not any(path.exists() for path in (output, packet_path, terminal_path)), "FRESH_TASK_REQUIRED")
    write_once_json(request_path, request)
    write_once_json(output / "WRITER.json", writer)
    write_once_json(packet_path, packet)
    state, index = copy.deepcopy(state), copy.deepcopy(index)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "status": "RUNNING", "attempt": 1,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "updated_at": now_iso(),
        "heartbeat_at": now_iso(), "task_packet": file_ref(packet_path), "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_TWO_THREADS_NO_GPU"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(root)),
        "packet_sha256": file_ref(packet_path)["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    publish_bundle(_load(root / "docs/governance/CURRENT_AUTHORITY_INDEX.json"), state,
        event_type="INPAINTING_CPU_ADMISSION_REGISTERED", expected_revision=receipt["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=root / "tasks/current/INDEX.json", task_packet_index_value=index)
    return packet


def authorize_candidate_operation(operation: str, args: list[str], *, repo_root: Path) -> dict:
    """Exact finite CPU/visual gate, always before the model module is imported.

    VISUAL is a separately published route, not permission obtained by CPU
    admission.  The G1 worker/Task Packet must bind its actual model closure.
    """
    _require(operation in {OPERATION, VISUAL_OPERATION}, "INPAINTING_CANDIDATE_OPERATION_UNSUPPORTED")
    root = repo_root.resolve(strict=True)
    _, state, index, contract = _snapshot(root)
    if operation == OPERATION:
        route = _route(root, contract)
    else:
        route = contract.get("special_status_contracts", {}).get(VISUAL_ROUTE_KEY)
        expected_route = visual_route_contract(root).get(VISUAL_ROUTE_KEY)
        _require(expected_route is not None and route == expected_route
                 and isinstance(route, dict) and route.get("candidate_operation") == VISUAL_OPERATION
                 and route.get("task_packet_required") is True and route.get("current_runnable_operation") is None
                 and isinstance(route.get("code_closure"), list) and bool(route["code_closure"]), "VISUAL_CANDIDATE_ROUTE_ABSENT")
        for reference in route["code_closure"]:
            verify_ref(reference)
    packet = _active_packet(root, state, [row for row in index["task_packets"] if row.get("execution_allowed") is True])
    runtime = packet.get("runtime_code", [])
    _require(bool(runtime) and all(reference in runtime for reference in route["code_closure"])
             and any(reference.get("path") == str(root / "src/chaoyang/ops" / (operation + ".py")) for reference in runtime), "CANDIDATE_CODE_CLOSURE_INCOMPLETE")
    bounds = packet.get("budgets", {})
    if operation == OPERATION:
        _require(packet.get("phase") == "INPAINTING_CANDIDATE_CPU_ADMISSION"
                 and packet.get("execution_mode") == "INPAINTING_CANDIDATE_CPU"
                 and bounds.get("model_calls_max") == 0 and bounds.get("gpu_count") == 0
                 and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= WALL_SECONDS_MAX,
                 "CPU_CANDIDATE_PACKET_REQUIRED")
    else:
        native_clean = packet.get("experiment_scope") == "INPAINTING_NATIVE_BASELINE_CLEAN"
        if native_clean:
            _validate_native("clean", packet, repo_root=root)
        glove = False if native_clean else validate_post_glove_experiment(packet, repo_root=root)
        source_aware = False if native_clean or glove else validate_post_source_aware_experiment(packet, repo_root=root)
        g3 = False if native_clean or source_aware or glove else validate_g3_experiment(packet, repo_root=root)
        post_controller = False if native_clean or source_aware or g3 or glove else validate_post_controller_experiment(packet, repo_root=root)
        post_text_retention = False if native_clean or source_aware or g3 or glove else validate_post_text_retention_experiment(packet, repo_root=root)
        g1_bounds = (packet.get("task_id", "").startswith("inpainting_generalization_g1_")
                 and packet.get("model_stages", ["Mask"]) == ["Mask"]
                 and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
                 and type(bounds.get("gpu_count")) is int and 1 <= bounds["gpu_count"] <= 2
                 and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 28800
                 and type(bounds.get("model_calls_max")) is int and bounds["model_calls_max"] > 0
                 and type(bounds.get("candidates_max", 2)) is int and 1 <= bounds.get("candidates_max", 2) <= 2)
        g2_bounds = (packet.get("task_id", "").startswith("inpainting_generalization_g2_")
                 and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
                 and packet.get("spec_schema") == CLEAN_SPEC_SCHEMA and packet.get("model_stages") == ["Clean"]
                 and packet.get("actual_stage_scope") == CLEAN_STAGE_SCOPE
                 and type(bounds.get("gpu_count")) is int and bounds["gpu_count"] == 1
                 and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 28800
                 and type(bounds.get("model_calls_max")) is int and 1 <= bounds["model_calls_max"] <= 2
                 and bounds.get("model_calls_unit") == CLEAN_MODEL_CALLS_UNIT
                 and type(bounds.get("candidates_max", 2)) is int and 1 <= bounds.get("candidates_max", 2) <= 2
                 and isinstance(packet.get("sessions"), list) and len(packet["sessions"]) == 1
                 and packet["sessions"][0].get("session_key") == "001"
                 and type(packet["sessions"][0].get("native_frames")) is int
                 and packet["sessions"][0]["native_frames"] == 238 and not packet.get("resume_context"))
        post_bounds = (post_controller and packet.get("phase") == POST_CONTROLLER_PHASE
                       and packet.get("operation") == VISUAL_OPERATION
                       and packet.get("algorithm_execution_allowed") is True
                       and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        retention_bounds = (post_text_retention and packet.get("phase") == POST_TEXT_RETENTION_PHASE
            and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
            and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        g3_bounds = (g3 and packet.get("phase") == G3_PHASE and packet.get("operation") == VISUAL_OPERATION
            and packet.get("algorithm_execution_allowed") is True
            and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        native_clean_bounds = (native_clean and packet.get("phase") == "INPAINTING_NATIVE_BASELINE_CLEAN"
            and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
            and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        source_aware_bounds = (source_aware and packet.get("phase") == POST_SOURCE_AWARE_PHASE
            and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
            and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        glove_bounds = (glove and packet.get("phase") == POST_GLOVE_PHASE
            and packet.get("operation") == VISUAL_OPERATION and packet.get("algorithm_execution_allowed") is True
            and packet.get("execution_mode") == "CANDIDATE_VISUAL_MODEL")
        _require(g1_bounds or g2_bounds or post_bounds or retention_bounds or g3_bounds or native_clean_bounds or source_aware_bounds or glove_bounds,
                 "VISUAL_FINITE_G1_BOUNDS_OR_G2_CLEAN_BOUNDS_REQUIRED")
    commands = packet.get("commands", [])
    tokens = shlex.split(commands[0]) if len(commands) == 1 else []
    _require(len(tokens) >= 6 and tokens[1:6] == ["-B", "-m", "chaoyang.cli", "run", operation]
             and tokens[6:] == args, "CPU_COMMAND_DRIFT")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    writer = packet["writer"]
    _require(_caller_is_writer_descendant(writer["pid"], writer["proc_start_ticks"], os.getpid()), "CPU_CALLER_NOT_FENCED_WRITER_DESCENDANT")
    _require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]), "CPU_FINITE_DEADLINE")
    if operation == OPERATION:
        request = _load(Path(packet["request_path"]))
        _require(canonical_sha(request) == packet["inputs"]["request_sha256"]
                 and cpu_signature(request, packet["runtime_code"]) == packet["run_signature"], "CPU_REQUEST_OR_COMPONENT_SIGNATURE_DRIFT")
    else:
        def verify_declared(value):
            if isinstance(value, dict):
                if {"path", "bytes", "sha256"} <= value.keys():
                    verify_ref(value)
                else:
                    for child in value.values():
                        verify_declared(child)
            elif isinstance(value, list):
                for child in value:
                    verify_declared(child)
        for domain in ("inputs", "weights", "config", "calibration", "schema"):
            _require(domain in packet, "VISUAL_SIGNATURE_DOMAIN_ABSENT:" + domain)
            verify_declared(packet[domain])
        if packet.get("model_stages") == ["Clean"]:
            _require(isinstance(packet.get("clean_config"), dict), "G2_CLEAN_CONFIG_REQUIRED")
            verify_declared(packet["clean_config"])
    return packet


def authorize_cpu_operation(args: list[str], *, repo_root: Path) -> dict:
    return authorize_candidate_operation(OPERATION, args, repo_root=repo_root)


def _publish_update(packet: dict, *, terminal: dict | None = None) -> None:
    from chaoyang.governance.common import now_iso, publish_bundle
    root = Path(packet["output_dir"]).parents[4]
    receipt, state, index, _ = _snapshot(root)
    active = _active_packet(root, state, [row for row in index["task_packets"] if row.get("execution_allowed") is True])
    _require(active.get("task_id") == packet["task_id"] and active.get("run_signature") == packet["run_signature"]
             and active.get("writer") == packet["writer"], "PUBLISHING_WRITER_NO_LONGER_CURRENT")
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
    if terminal is not None:
        row.update(status=terminal["status"], pid=None, proc_start_ticks=None, result=terminal["result"], terminal_receipt=terminal["receipt"])
        next(item for item in index["task_packets"] if item["task_id"] == packet["task_id"]).update(
            execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
        state["next_task"] = None
    publish_bundle(_load(root / "docs/governance/CURRENT_AUTHORITY_INDEX.json"), state,
        event_type="INPAINTING_CPU_ADMISSION_SEALED" if terminal else "INPAINTING_CPU_ADMISSION_HEARTBEAT",
        expected_revision=receipt["governance_revision"], generator_path=Path(__file__),
        task_packet_index_path=root / "tasks/current/INDEX.json", task_packet_index_value=index)


def run_registered(packet: dict) -> dict:
    """One bounded existing-CLI child.  Writer remains this submit process."""
    output = Path(packet["output_dir"])
    _require(packet["writer"]["pid"] == os.getpid() and _live(os.getpid(), packet["writer"]["proc_start_ticks"]), "CPU_WRITER_IDENTITY_DRIFT")
    cache = output / "cache"
    cache.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(output.parents[4] / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TMPDIR": str(cache), "TMP": str(cache), "TEMP": str(cache), "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"), "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"), "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions")})
    child = None
    error = None
    try:
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(shlex.split(packet["commands"][0]), cwd=output.parents[4], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            beat = time.monotonic()
            while child.poll() is None:
                _require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]), "CPU_FINITE_DEADLINE")
                if time.monotonic() - beat >= 60:
                    _publish_update(packet)
                    beat = time.monotonic()
                time.sleep(0.25)
            _require(child.returncode == 0, "CPU_ADMISSION_CHILD_EXIT:" + str(child.returncode))
        stage = _load(output / "ADMISSION_STAGE.json")
        _require(stage.get("status") == "SEALED_COMPLETE_CPU_ADMISSION" and stage.get("run_signature") == packet["run_signature"], "CPU_STAGE_NOT_SEALED")
        for reference in stage["sources"]:
            verify_ref(reference)
    except Exception as exc:
        error = type(exc).__name__ + ":" + str(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
    result = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
              "status": "PASSED" if error is None else "FAILED_RUNTIME_FINAL", "scope": "CPU_ADMISSION_ONLY",
              "error": error, "models_called": 0, "robot_video_produced": False, "visual_qualified": False,
              "training_eligible": False, "deployable": False, "candidate_adopted": False,
              "stage": file_ref(output / "ADMISSION_STAGE.json") if error is None else None}
    write_once_json(output / "RESULT.json", result)
    root = output.parents[4]
    terminal_path = _paths(root, packet["task_id"])[2]
    terminal = {**result, "result": file_ref(output / "RESULT.json")}
    write_once_json(terminal_path, terminal)
    _publish_update(packet, terminal={"status": result["status"], "result": terminal["result"], "receipt": file_ref(terminal_path)})
    return {**result, "result": terminal["result"], "execution_started_here": True}


def submit_request(request: dict, *, repo_root: Path, preview: bool = False) -> dict:
    if preview:
        from chaoyang.inpainting_candidate.entry import verify_request
        verify_request(request)
        return {"status": "READ_ONLY_CPU_REQUEST_PREVIEW", "submission_input_sha256": request["submission_input_sha256"],
                "sessions": len(request["sources"]), "request": request, "execution_started_here": False,
                "robot_video_produced": False, "visual_qualified": False, "training_eligible": False, "deployable": False}
    decision = plan_request(request, repo_root=repo_root)
    if decision["status"] != "CPU_ADMISSION_READY_NOT_REGISTERED":
        return decision
    packet = register(request, repo_root=repo_root)
    if packet.get("status", "").startswith("ATTACHED"):
        return packet
    return run_registered(packet)


def submit_v1_compatible_request(request: dict, *, repo_root: Path, preview: bool = False,
                                 wall_seconds: int = 3600, execution_context: dict | None = None,
                                 clean_reference_policy: str | None = None,
                                 resume_manifest_path: Path | None = None,
                                 clean_write_policy: str | None = None) -> dict:
    """Thin full-flow submission through the existing MCO finite executor.

    This is intentionally not raw-auto, a new queue, or baseline adoption.
    No algorithm is imported here; the MCO operation fence owns execution.
    """
    import importlib.util
    from chaoyang.humanego_ict.project_mount_paths import PROJECT_ALIASES, resolve_input_path
    from chaoyang.inpainting_candidate.automation import build_v1_request, read_reference, verify_v1_request
    root = Path(repo_root).resolve(strict=True)
    verify_v1_request(request, project_root=root)
    # The current shared ledger uses the CPFS spelling. Both aliases must be
    # proved to be the same physical project before selecting that mount.
    execution_root = PROJECT_ALIASES[1] if PROJECT_ALIASES[1].is_dir() else root
    resolve_input_path(root, execution_root)
    if execution_root != root:
        selection = request.get('selection_manifest')
        request = build_v1_request(project_root=execution_root,
            session=Path(request['sessions'][0]['raw_session_dir']) if selection is None else None,
            manifest=resolve_input_path(selection['path'], execution_root) if selection is not None else None,
            repair_rounds=request.get('repair_rounds', 1))
    source = execution_root / 'scripts/maintenance/inpainting_v1_automation_specs.py'
    definition = importlib.util.spec_from_file_location('_v1_compatible_spec_builder', source)
    builder = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(builder)
    if clean_reference_policy is not None:
        _require(clean_reference_policy=='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'and request.get('repair_rounds',1)in(2,3),
                 'AUTOMATION_CANDIDATE_CLEAN_POLICY_OR_ROUNDS')
        if execution_context is None:
            execution_context=builder.clean_reference_execution_context(execution_root,wall_seconds=wall_seconds)
    elif request.get('repair_rounds', 1) in (1, 2, 3) and execution_context is None:
        execution_context = builder.temporal_execution_context(execution_root, wall_seconds=wall_seconds,
            repair_rounds=request.get('repair_rounds', 1))
    _require(execution_context is not None or wall_seconds == 3600, 'AUTOMATION_WALL_BUDGET_REQUIRES_NEW_EXECUTION_CONTEXT')
    context = builder._execution_context(execution_root, execution_context) if execution_context is not None else None
    submission_root = Path(context['output_root']) if context else execution_root / '_run/inpainting_v1_automation_20261014'
    submission = submission_root / 'submissions' / request['submission_input_sha256']
    request_path = submission / 'REQUEST.json'
    data = (json.dumps(request, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    import hashlib
    request_ref = {'path': str(request_path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    extra={'clean_reference_policy':clean_reference_policy}if clean_reference_policy is not None else {}
    if clean_write_policy is not None:extra['clean_write_policy']=clean_write_policy
    if resume_manifest_path is not None:
        local=resolve_input_path(Path(resume_manifest_path).absolute(),execution_root)
        _require(local.is_relative_to(execution_root),'AUTOMATION_PROJECT_RESUME_MANIFEST_REQUIRED')
        extra['resume_manifest']=file_ref(local)
    spec = builder.compatible_spec(execution_root, request, request_ref, execution_context=context,**extra)
    decision = {'task_id': spec['task_id'], 'mode': 'v1-compatible',
        'producer_policy': request['producer_policy'], 'original_v1_seeds_read': True,
        'raw_auto_proved': False, 'robot_video_produced': False, 'candidate_adopted': False,
        'training_eligible': False, 'component_signatures': spec['component_signatures']}
    if clean_reference_policy is not None:
        decision.update(clean_reference_policy=clean_reference_policy,
            stage_reuse={sid:sorted(refs)for sid,refs in spec.get('reuse_stages',{}).items()})
    if resume_manifest_path is not None:
        decision.update(resume_manifest=spec['inputs']['resume_manifest'],
            stage_reuse={sid:sorted(refs)for sid,refs in spec.get('reuse_stages',{}).items()},
            resume_decisions=spec['resume_plan'].get('decisions',[]))
    if clean_write_policy is not None:
        decision.update(clean_write_policy=clean_write_policy,
            clean_work_stage_reuse=sorted(spec.get('clean_work_reuse',{})))
    if preview:
        return {**decision, 'status': 'FULL_FLOW_READY_PREVIEW_ONLY', 'execution_started_here': False}
    receipt = _load(execution_root / 'docs/governance/CURRENT_STATUS_RECEIPT.json')
    state = _load(Path(receipt['files']['task_state']['path']))
    index = _load(Path(receipt['files']['task_packet_index_current']['path']))
    allowed = [row for row in index['task_packets'] if row.get('execution_allowed') is True]
    packet_path = execution_root / 'tasks/current' / spec['task_id'] / 'TASK_PACKET.json'
    if allowed:
        _require(len(allowed) == 1 and allowed[0]['task_id'] == spec['task_id'], 'ACTIVE_DIFFERENT_SIGNATURE_OR_WRITER_CONFLICT')
        packet = _load(packet_path)
        verify_ref(file_ref(packet_path))
        _require(allowed[0].get('packet_sha256') == file_ref(packet_path)['sha256']
            and packet.get('mode') == 'Automation' and packet.get('runtime_code') == spec['runtime_code']
            and packet.get('component_signatures') == spec['component_signatures']
            and read_reference(packet['inputs']['request']) == request,
            'AUTOMATION_ACTIVE_REQUEST_OR_CODE_DRIFT')
        writer = packet['writer']
        _require(_live(writer['pid'], writer['proc_start_ticks']), 'AUTOMATION_ACTIVE_WRITER_NOT_LIVE')
        return {**decision, 'status': 'ATTACHED_TO_ACTIVE_V1_COMPATIBLE', 'execution_started_here': False}
    _require(not state.get('next_task') and not any(r.get('status') in ACTIVE for r in state.get('tasks', [])),
             'ACTIVE_STATE_WITHOUT_PACKET_CONFLICT')
    if packet_path.exists():
        packet = _load(packet_path)
        _require(packet.get('runtime_code') == spec['runtime_code']
            and packet.get('component_signatures') == spec['component_signatures']
            and read_reference(packet['inputs']['request']) == request,
            'AUTOMATION_TERMINAL_SIGNATURE_DRIFT')
        result = Path(packet['output_dir']) / 'RESULT.json'
        _require(result.is_file(), 'AUTOMATION_UNSEALED_ATTEMPT_REJECTED')
        terminal = _load(result)
        _require(terminal.get('task_id') == spec['task_id'] and terminal.get('run_signature') == packet['run_signature'],
                 'AUTOMATION_TERMINAL_IDENTITY_DRIFT')
        status = terminal.get('status')
        _require(isinstance(status, str) and (status in {'PASSED', 'FAILED', 'TIMEOUT', 'TIMED_OUT', 'DEADLINE_EXCEEDED'}
                 or status.startswith('FAILED_')), 'AUTOMATION_RESULT_NOT_TERMINAL')
        robot_video_produced = False
        if status == 'PASSED':
            model_ref = terminal.get('model_result')
            _require(terminal.get('model_result_consumable') is True and isinstance(model_ref, dict),
                     'AUTOMATION_TERMINAL_MODEL_REQUIRED')
            try:
                verify_ref(model_ref)
                _require(Path(model_ref['path']).resolve(strict=True)
                         == (Path(packet['output_dir']) / 'MODEL_RESULT.json').resolve(strict=True),
                         'AUTOMATION_TERMINAL_MODEL_PATH_DRIFT')
                model = _load(Path(model_ref['path']))
            except (OSError, ValueError, KeyError, TypeError, RegistrationError) as error:
                raise RegistrationError('AUTOMATION_TERMINAL_MODEL_REFERENCE_REJECTED') from error
            _require(model.get('schema') == 'inpainting-v1-mco-stage-result-v1'
                     and model.get('status') == 'SEALED_COMPLETE' and model.get('mode') == 'Automation'
                     and model.get('task_id') == packet['task_id']
                     and model.get('run_signature') == packet['run_signature'],
                     'AUTOMATION_TERMINAL_MODEL_NOT_SEALED_OR_BOUND')
            rows = model.get('sessions')
            planned = request['sessions']
            _require(isinstance(rows, list) and len(rows) == len(planned)
                     and all(isinstance(row, dict)
                             and row.get('session_key') == original['session_id']
                             and row.get('native_frames') == original['native_frames']
                             and row.get('native_output_frames') == original['native_frames']
                             and row.get('completed_frame_indices') == list(range(original['native_frames']))
                             and row.get('execution_complete') is True
                             for row, original in zip(rows, planned)),
                     'AUTOMATION_TERMINAL_MODEL_EXECUTION_INCOMPLETE')
            robot_video_produced = model.get('robot_video_produced') is True
        return {**decision, 'status': 'ATTACHED_TO_TERMINAL_V1_COMPATIBLE', 'terminal_status': status,
                'result': file_ref(result), 'robot_video_produced': robot_video_produced,
                'execution_started_here': False}
    # File creation is admission, not model inference. A competing submit can
    # race here but the existing MCO CAS/lease gate permits only one writer.
    if request_path.exists():
        _require(request_path.read_bytes() == data, 'AUTOMATION_REQUEST_COLLISION')
    else:
        request_path.parent.mkdir(parents=True, exist_ok=True)
        with request_path.open('xb') as stream:
            stream.write(data)
    spec_path = submission / (spec['task_id'] + '_SPEC.json')
    if spec_path.exists():
        _require(_load(spec_path) == spec, 'AUTOMATION_SPEC_COLLISION')
    else:
        write_once_json(spec_path, spec)
    launcher = (source if context else execution_root / '_run/inpainting_v1_automation_20261014/bootstrap/launch.py')
    _require(launcher.is_file(), 'EXISTING_MOUNT_COMPAT_LAUNCHER_REQUIRED')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(execution_root / 'src'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    command = ([sys.executable, '-B', str(launcher), '--execute-spec', str(spec_path)] if context else
               [sys.executable, '-B', str(launcher), 'mco', '--spec', str(spec_path)])
    completed = subprocess.run(command, cwd=execution_root, env=env, check=False)
    return {**decision, 'status': 'EXECUTED_V1_COMPATIBLE' if completed.returncode == 0 else 'FAILED_V1_COMPATIBLE',
            'execution_started_here': True, 'executor_exit_code': completed.returncode,
            'result': file_ref(Path(spec['output_dir']) / 'RESULT.json')
                      if (Path(spec['output_dir']) / 'RESULT.json').is_file() else None,
            'robot_video_produced': completed.returncode == 0}


def authorize_automation_operation(args: list[str], *, repo_root: Path) -> dict:
    """Delegate to the existing MCO before-import writer/command/SHA fence."""
    import importlib.util
    source = Path(repo_root) / 'scripts/maintenance/inpainting_v1_mco_task.py'
    definition = importlib.util.spec_from_file_location('_v1_automation_existing_gate', source)
    runner = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    packet = runner.authorize_operation(args)
    _require(packet.get('mode') == 'Automation', 'AUTOMATION_REGISTERED_MODE_REQUIRED')
    return packet
