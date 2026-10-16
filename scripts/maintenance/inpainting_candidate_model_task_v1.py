"""Finite Mask/Clean GPU executor using existing Task Packet/CAS/card lease.

Source stages must already be sealed; this runner performs no manual seed,
retarget, adoption, cleanup or training operation. It dispatches exactly
one registered chaoyang run operation, not an implicit algorithm entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.inpainting_candidate.resources import (CANDIDATE_CARD_LOCK_ROOT, DeviceMapping,
    GpuCardLease, read_gpu_inventory)
from chaoyang.inpainting_candidate.resources import ResourceError
from chaoyang.inpainting_candidate.resources import proc_start_ticks
from chaoyang.inpainting_candidate.contracts import verify_ref, verify_sealed_stage
from chaoyang.governance.inpainting_candidate_registration import (CLEAN_SPEC_SCHEMA, CLEAN_MODEL_CALLS_UNIT,
    CLEAN_STAGE_SCOPE, POST_CONTROLLER_SCOPE, POST_CONTROLLER_PHASE, POST_CONTROLLER_PREFIX,
    validate_post_controller_experiment, POST_TEXT_RETENTION_SCOPE, POST_TEXT_RETENTION_PHASE,
    POST_TEXT_RETENTION_PREFIX, TEXT_WEARER_RETENTION_RULE, validate_post_text_retention_experiment,
    G3_SCOPE, G3_PHASE, G3_PREFIX, G3_SPEC_SCHEMA, G3_RESULT_SCHEMA, validate_g3_experiment,
    POST_SOURCE_AWARE_SCOPE, POST_SOURCE_AWARE_PHASE, POST_SOURCE_AWARE_PREFIX,
    SOURCE_AWARE_RETENTION_VETO, SOURCE_AWARE_SPEC_SCHEMA, validate_post_source_aware_experiment,
    POST_GLOVE_SCOPE, POST_GLOVE_PHASE, POST_GLOVE_PREFIX, GLOVE_SPEC_SCHEMA,
    GLOVE_TEXT_PROMPT, validate_post_glove_experiment)
from chaoyang.inpainting_candidate.native_contracts import (
    NATIVE_CLEAN_SCOPE, NATIVE_CLEAN_SCHEMA, validate_native_clean_spec)

MODEL_CALLS_UNIT = "SAM_ADAPTER_INFERENCE_API_INVOCATION"
EXPECTED_SESSION_KEYS = ["001", "003", "008", "100", "114", "050"]


def planned_session_keys(value):
    """Only proven complete stage evidence can justify the fixed remaining five.

    Retained001 is evaluation evidence, never input to another Mask producer.
    New masks carry their own current execution/component signatures.
    """
    keys = [row["session_key"] for row in value["sessions"]]
    if value.get("experiment_scope") == NATIVE_CLEAN_SCOPE:
        validate_native_clean_spec(value, repo_root=REPO_ROOT)
        return keys
    if validate_post_glove_experiment(value, repo_root=REPO_ROOT):
        return keys
    if validate_post_source_aware_experiment(value, repo_root=REPO_ROOT):
        return keys
    if validate_g3_experiment(value, repo_root=REPO_ROOT):
        return keys
    if validate_post_text_retention_experiment(value, repo_root=REPO_ROOT):
        return keys
    if validate_post_controller_experiment(value, repo_root=REPO_ROOT):
        return keys
    if value.get("spec_schema") == CLEAN_SPEC_SCHEMA or value.get("model_stages") == ["Clean"]:
        if (value.get("spec_schema") != CLEAN_SPEC_SCHEMA
                or not value.get("task_id", "").startswith("inpainting_generalization_g2_")
                or keys != ["001"] or len(value["sessions"]) != 1
                or type(value["sessions"][0].get("native_frames")) is not int
                or value["sessions"][0]["native_frames"] != 238 or value.get("resume_context")
                or value.get("actual_stage_scope") != CLEAN_STAGE_SCOPE):
            raise ValueError("G2_FIXED_001_FULL_238_REQUIRED")
        return keys
    context = value.get("resume_context")
    if context is None:
        if keys != EXPECTED_SESSION_KEYS:
            raise ValueError("G1_FIXED_SIX_FULL_SESSIONS_REQUIRED")
        return keys
    if keys != EXPECTED_SESSION_KEYS[1:]:
        raise ValueError("G1_EXACT_PROVEN_REMAINING_FIVE_REQUIRED")
    for name in ("failed_parent_result", "parent_packet", "retained_session"):
        verify_ref(context[name])
    failed = load_json(Path(context["failed_parent_result"]["path"]))
    parent = load_json(Path(context["parent_packet"]["path"]))
    retained = load_json(Path(context["retained_session"]["path"]))
    if (parent["task_id"] != "inpainting_generalization_g1_mask_c02_20261013"
            or failed["task_id"] != parent["task_id"] or failed["run_signature"] != parent["run_signature"]
            or failed["status"] != "FAILED_RUNTIME_FINAL" or failed["completed_mask_sessions"] != 1
            or failed["planned_sessions"] != 6
            or retained["session_key"] != "001" or retained["execution"] != "COMPLETE_MASK_ONLY"
            or retained["native_frames"] != 238
            or retained["source"] != parent["sessions"][0]["source_receipt"]):
        raise ValueError("G1_RETAINED_COMPLETE_STAGE_IDENTITY_REQUIRED")
    verify_ref(retained["source"])
    source_path = Path(retained["source"]["path"])
    source = load_json(source_path)
    verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
    if retained["mask"] != failed["retained_sealed_001"]:
        raise ValueError("G1_RETAINED_MASK_NOT_EXACT_FAILED_PARENT_EVIDENCE")
    verify_ref(retained["mask"])
    stage_path = Path(retained["mask"]["path"])
    stage = load_json(stage_path)
    verify_sealed_stage(stage, parent["sessions"][0]["mask_signature"], output_root=stage_path.parent)
    verify_ref(retained["review"]["video"])
    if (stage["native_input_frames"] != 238 or stage["native_output_frames"] != 238
            or retained["review"]["decoded_frames"] != 238):
        raise ValueError("G1_RETAINED_FULL_NATIVE_STAGE_REQUIRED")
    if any(row["source_receipt"] != original["source_receipt"]
           for row, original in zip(value["sessions"], parent["sessions"][1:])):
        raise ValueError("G1_REMAINING_NATIVE_SOURCE_DRIFT")
    return keys


def effective_deadline(spec, started):
    if spec.get("task_id", "").startswith((POST_CONTROLLER_PREFIX, POST_TEXT_RETENTION_PREFIX,
                                           POST_SOURCE_AWARE_PREFIX, POST_GLOVE_PREFIX, G3_PREFIX, "inpainting_native_clean_")):
        if "resume_context" in spec or "original_g1_deadline_at" in spec:
            if spec["task_id"].startswith((POST_SOURCE_AWARE_PREFIX, POST_GLOVE_PREFIX, "inpainting_native_clean_")):
                raise ValueError("NATIVE_OR_SOURCE_AWARE_FRESH_TIMEBOX_REQUIRED")
            if spec["task_id"].startswith(G3_PREFIX):
                raise ValueError("G3_FRESH_NOT_OLD_TIMEBOX_REQUIRED")
            raise ValueError("POST_RETENTION_FRESH_NOT_G1_SUCCESSOR_REQUIRED" if
                spec["task_id"].startswith(POST_TEXT_RETENTION_PREFIX) else
                "POST_CONTROLLER_FRESH_NOT_G1_SUCCESSOR_REQUIRED")
    deadline = started + timedelta(seconds=spec["budgets"]["wall_seconds_max"])
    if spec.get("original_g1_deadline_at"):
        deadline = min(deadline, datetime.fromisoformat(spec["original_g1_deadline_at"]))
    if deadline <= started:
        raise ValueError("G1_ORIGINAL_TIMEBOX_EXHAUSTED")
    return deadline


def own_row(packet, state, index):
    """Candidate fence that permits resource waiting; permanent infra unchanged."""
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == packet["task_id"]]
    active = [row for row in state.get("tasks", []) if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    pointers = [row for row in index.get("task_packets", []) if row.get("execution_allowed") is True]
    writer = packet["writer"]
    packet_ref = artifact_ref(Path(packet["packet_path"]))
    if (len(rows) != 1 or len(active) != 1 or active[0] is not rows[0]
            or rows[0].get("status") not in {"RUNNING", "WAIT_GPU_RESOURCE"}
            or rows[0].get("pid") != writer["pid"] or rows[0].get("proc_start_ticks") != writer["proc_start_ticks"]
            or rows[0].get("run_signature") != packet["run_signature"] or rows[0].get("task_packet") != packet_ref
            or (state.get("next_task") or {}).get("task_id") != packet["task_id"]
            or len(pointers) != 1 or pointers[0].get("task_id") != packet["task_id"]
            or pointers[0].get("packet_sha256") != packet_ref["sha256"]
            or load_json(Path(packet["packet_path"])) != packet
            or load_json(Path(packet["writer_file"])) != writer
            or proc_start_ticks(writer["pid"]) != writer["proc_start_ticks"]):
        raise RuntimeError("CANDIDATE_WRITER_FENCE_DRIFT")
    return rows[0], pointers[0]


def review_decode_count(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-threads", "2", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=180, check=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("MASK_REVIEW_ONE_VIDEO_STREAM_REQUIRED")
    return int(streams[0]["nb_read_frames"])


def verify_session_inputs(infra, row, *, clean):
    """Derived stages stay project-only; signed Raw may be a read-only NAS file."""
    if row.get("source_status") == "BLOCKED" or row.get("upstream_status") == "BLOCKED":
        infra._verify(row["diagnostic_ref"])
        return
    infra._verify(row["source_receipt"])
    if not clean:
        return
    for name in ("mask_receipt", "frame_map_ref"):
        infra._verify(row[name])
    verify_ref(row["raw_source_ref"])
    source_path = Path(row["source_receipt"]["path"])
    source = load_json(source_path)
    verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
    reports = [ref for ref in source["outputs"] if Path(ref["path"]).name == "SOURCE_REPORT.json"]
    if len(reports) != 1 or row["frame_map_ref"] not in source["outputs"]:
        raise ValueError("G2_RAW_SOURCE_SEALED_REPORT_OR_MAP_REQUIRED")
    report = load_json(Path(reports[0]["path"]))
    if report.get("actual_inputs", {}).get("video") != row["raw_source_ref"]:
        raise ValueError("G2_RAW_SOURCE_NOT_CURRENT_SEALED_SOURCE")


def validate_model_result(packet, *, decoder=review_decode_count):
    """No partial batch or self-reported video count can pass execution sealing."""
    root = Path(packet["output_dir"])
    reference = artifact_ref(root / "MODEL_RESULT.json")
    value = load_json(Path(reference["path"]))
    if packet.get("model_stages") == ["Clean"]:
        # This CPU validator verifies the actual sealed outputs and decodes; it
        # neither imports a model nor treats a no-op B as a second model run.
        from chaoyang.inpainting_candidate.clean_worker import validate_clean_result
        planned_session_keys(packet)
        validate_clean_result(packet, value, decoder=decoder)
        return reference
    expected = packet["sessions"]
    rows = value.get("sessions")
    keys = planned_session_keys(packet)
    post_controller = packet.get("experiment_scope") == POST_CONTROLLER_SCOPE
    post_retention = packet.get("experiment_scope") == POST_TEXT_RETENTION_SCOPE
    source_aware = packet.get("experiment_scope") == POST_SOURCE_AWARE_SCOPE
    glove = packet.get("experiment_scope") == POST_GLOVE_SCOPE
    g3 = packet.get("experiment_scope") == G3_SCOPE
    expected_ready = sum(row.get("source_status") == "READY" for row in expected) if g3 else len(expected)
    expected_blocked = len(expected) - expected_ready
    if ([row["session_key"] for row in expected] != keys
            or value.get("schema") != (G3_RESULT_SCHEMA if g3 else "inpainting-g1-mask-batch-result-v1")
            or value.get("task_id") != packet["task_id"] or value.get("run_signature") != packet["run_signature"]
            or value.get("stage_scope") != "G1_AUTOMATIC_MASK_ONLY"
            or value.get("planned_sessions") != len(expected) or value.get("completed_mask_sessions") != expected_ready
            or not isinstance(rows, list) or [row.get("session_key") for row in rows] != keys
            or value.get("candidate_adopted") is not False or value.get("visual_quality") != "NOT_REVIEWED"):
        raise ValueError("MODEL_RESULT_BATCH_BINDING_OR_COMPLETENESS_FAILED")
    if post_controller and value.get("experiment_scope") != POST_CONTROLLER_SCOPE:
        raise ValueError("POST_CONTROLLER_RESULT_EXPERIMENT_SCOPE_DRIFT")
    if post_retention and value.get("experiment_scope") != POST_TEXT_RETENTION_SCOPE:
        raise ValueError("POST_RETENTION_RESULT_EXPERIMENT_SCOPE_DRIFT")
    if source_aware and value.get("experiment_scope") != POST_SOURCE_AWARE_SCOPE:
        raise ValueError("SOURCE_AWARE_RESULT_EXPERIMENT_SCOPE_DRIFT")
    if glove and value.get("experiment_scope") != POST_GLOVE_SCOPE:
        raise ValueError("GLOVE_RESULT_EXPERIMENT_SCOPE_DRIFT")
    if g3 and (value.get("experiment_scope") != G3_SCOPE or expected_ready == 0
        or value.get("ready_source_sessions") != expected_ready
        or value.get("blocked_source_sessions") != expected_blocked
        or value.get("execution_diagnostic_complete") is not True):
        raise ValueError("G3_RESULT_READY_BLOCKED_SCOPE_OR_COMPLETENESS_FAILED")
    accounting = value.get("model_call_accounting", {})
    counts = accounting.get("by_api", {})
    if (accounting.get("unit") != MODEL_CALLS_UNIT or packet["budgets"].get("model_calls_unit") != MODEL_CALLS_UNIT
            or set(counts) != {"track_text", "point_add_prompt", "point_propagate_in_video"}
            or any(type(count) is not int or count < 0 for count in counts.values())
            or accounting.get("attempted_api_calls") != sum(counts.values())
            or accounting.get("maximum") != packet["budgets"]["model_calls_max"]
            or not 0 < sum(counts.values()) <= packet["budgets"]["model_calls_max"]):
        raise ValueError("MODEL_RESULT_CALL_ACCOUNTING_FAILED")
    for planned, row in zip(expected, rows):
        if g3 and planned.get("source_status") == "BLOCKED":
            verify_ref(planned["diagnostic_ref"])
            if (row.get("execution") != "BLOCKED_SOURCE" or row.get("source_status") != "BLOCKED"
                or row.get("session_id") != planned["session_id"]
                or row.get("native_frames") != planned["native_frames"]
                or row.get("failure_code") != planned["failure_code"]
                or row.get("diagnostic_ref") != planned["diagnostic_ref"]
                or row.get("experiment_scope") != G3_SCOPE
                or row.get("model_api_calls_attempted") != 0 or row.get("model_calls_unit") != MODEL_CALLS_UNIT
                or row.get("manual_seed_reads") != 0 or row.get("control_ground_truth") is not False
                or row.get("training_eligible") is not False or row.get("partial_consumable") is not False
                or row.get("visual_quality") != "NOT_REVIEWED"
                or any(k in row for k in ("mask", "review", "source"))):
                raise ValueError("G3_BLOCKED_RESULT_CANNOT_FAKE_EXECUTION")
            continue
        source_ref = planned["source_receipt"]
        verify_ref(source_ref)
        source = load_json(Path(source_ref["path"]))
        source_folder = Path(source_ref["path"]).parent
        verify_sealed_stage(source, source["component_signature"], output_root=source_folder)
        source_report = load_json(source_folder / "SOURCE_REPORT.json")
        count = source["native_frames"]
        if (row.get("execution") != "COMPLETE_MASK_ONLY" or row.get("source") != source_ref
                or row.get("session_id") != source_report["session_id"] or row.get("native_frames") != count
                or row.get("manual_seed_reads") != 0 or row.get("control_ground_truth") is not False
                or row.get("training_eligible") is not False or row.get("visual_quality") != "NOT_REVIEWED"
                or row.get("model_calls_unit") != MODEL_CALLS_UNIT
                or type(row.get("model_api_calls_attempted")) is not int or row["model_api_calls_attempted"] <= 0):
            raise ValueError("MODEL_RESULT_SESSION_IDENTITY_OR_SCOPE_FAILED:" + planned["session_key"])
        mask_ref = row["mask"]
        expected_mask = root / "sessions" / planned["session_key"] / "Mask/RECEIPT.json"
        if mask_ref.get("path") != str(expected_mask):
            raise ValueError("MODEL_RESULT_MASK_PATH_DRIFT")
        verify_ref(mask_ref)
        mask = load_json(expected_mask)
        verify_sealed_stage(mask, planned["mask_signature"], output_root=expected_mask.parent)
        if (mask.get("native_input_frames") != count or mask.get("native_output_frames") != count
                or mask.get("manual_points_submitted") != 0 or mask.get("producer_manual_seed_reads") != 0):
            raise ValueError("MODEL_RESULT_MASK_NATIVE_FRAMES_OR_MANUAL_INPUT_FAILED")
        if post_controller or post_retention or source_aware or g3 or glove:
            experiment_scope = (POST_GLOVE_SCOPE if glove else G3_SCOPE if g3 else POST_SOURCE_AWARE_SCOPE if source_aware else
                                POST_TEXT_RETENTION_SCOPE if post_retention else POST_CONTROLLER_SCOPE)
            if (row.get("experiment_scope") != experiment_scope
                    or mask.get("actual_inputs", {}).get("controller_text_prompt") != "VR controller"):
                raise ValueError("POST_CONTROLLER_ACTUAL_MODEL_PROMPT_OR_SCOPE_DRIFT")
            text_refs = [ref for ref in mask["outputs"] if Path(ref["path"]).name == "SAM_TEXT.json"]
            if len(text_refs) != 1:
                raise ValueError("POST_CONTROLLER_ACTUAL_TEXT_CALLS_REQUIRED")
            text_calls = load_json(Path(text_refs[0]["path"])).get("calls", [])
            controller_calls = [call for call in text_calls if call.get("role") == "controller"]
            if len(controller_calls) != 1 or controller_calls[0].get("prompt") != "VR controller":
                raise ValueError("POST_CONTROLLER_ACTUAL_MODEL_PROMPT_OR_SCOPE_DRIFT")
        if post_retention or source_aware or g3 or glove:
            if mask.get("actual_inputs", {}).get("text_wearer_retention") != TEXT_WEARER_RETENTION_RULE:
                raise ValueError("POST_RETENTION_ACTUAL_RULE_REQUIRED")
            point_refs = [ref for ref in mask["outputs"] if Path(ref["path"]).name == "SAM_POINT.json"]
            if len(point_refs) != 1:
                raise ValueError("POST_RETENTION_ACTUAL_REFINEMENT_REQUIRED")
            point = load_json(Path(point_refs[0]["path"]))
            if point.get("text_wearer_retention") != TEXT_WEARER_RETENTION_RULE:
                raise ValueError("POST_RETENTION_ACTUAL_REFINEMENT_RULE_DRIFT")
        if (source_aware or glove) and mask.get("actual_inputs", {}).get("source_aware_retention_veto") != SOURCE_AWARE_RETENTION_VETO:
            raise ValueError("SOURCE_AWARE_ACTUAL_RULE_REQUIRED")
        if glove:
            if mask.get("actual_inputs", {}).get("glove_text_prompt") != GLOVE_TEXT_PROMPT:
                raise ValueError("GLOVE_ACTUAL_CONCEPT_REQUIRED")
            glove_calls = [call for call in text_calls if call.get("role") == "glove"]
            if len(glove_calls) != 1 or glove_calls[0].get("prompt") != GLOVE_TEXT_PROMPT:
                raise ValueError("GLOVE_ACTUAL_MODEL_CONCEPT_DRIFT")
        if g3:
            source_map = json.loads(Path(planned["frame_map_ref"]["path"]).read_text(encoding="utf-8"))
            mask_map_refs = [ref for ref in mask["outputs"] if Path(ref["path"]).name == "FRAME_MAP.json"]
            if (row["model_api_calls_attempted"] > planned["model_calls_max"]
                or len(mask_map_refs) != 1
                or json.loads(Path(mask_map_refs[0]["path"]).read_text(encoding="utf-8")) != source_map
                or mask.get("actual_inputs", {}).get("raw_source") != source_report["actual_inputs"]["video"]):
                raise ValueError("G3_ACTUAL_SOURCE_MAP_OR_API_BOUND_DRIFT")
        review = row["review"]
        video = review["video"]
        if video.get("path") != str(expected_mask.parent.parent / "raw_mask_review.mp4"):
            raise ValueError("MODEL_RESULT_REVIEW_PATH_DRIFT")
        verify_ref(video)
        if review.get("decoded_frames") != count or decoder(Path(video["path"])) != count:
            raise ValueError("MODEL_RESULT_REVIEW_DECODE_COUNT_FAILED")
    if sum(row["model_api_calls_attempted"] for row in rows) != accounting["attempted_api_calls"]:
        raise ValueError("MODEL_RESULT_SESSION_CALL_TOTAL_FAILED")
    return reference


def stop_owned_child(child):
    """Only the Popen child session created by this runner; retain lease if live."""
    if child is None or child.poll() is not None:
        return
    if os.getpgid(child.pid) != child.pid:
        raise RuntimeError("OWN_CHILD_SESSION_IDENTITY_CHANGED")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # An owned GPU computation may not outlive its finite Task Packet.
        if child.poll() is None and os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGKILL)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("OWN_CHILD_STILL_RUNNING_STOP_REQUIRED") from error


@contextmanager
def lease_until_child_stops(lease, child_holder):
    if not lease._held:
        lease.acquire()
    try:
        yield lease
    finally:
        # Do not unlock the GPU before a running model process is really dead.
        # If stop cannot be confirmed, keep its explicit lease file for recovery.
        stop_owned_child(child_holder[0])
        lease.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve(strict=True)
    if not spec_path.is_relative_to(REPO_ROOT / "inpainting_optimization"):
        raise ValueError("CANDIDATE_SPEC_LOCATION_REQUIRED")
    spec = load_json(spec_path)
    native_clean = spec.get("experiment_scope") == NATIVE_CLEAN_SCOPE
    if native_clean:
        validate_native_clean_spec(spec, repo_root=REPO_ROOT)
    clean = native_clean or spec.get("spec_schema") == CLEAN_SPEC_SCHEMA
    glove = False if native_clean else validate_post_glove_experiment(spec, repo_root=REPO_ROOT)
    source_aware = False if native_clean or glove else validate_post_source_aware_experiment(spec, repo_root=REPO_ROOT)
    g3 = False if native_clean or source_aware or glove else validate_g3_experiment(spec, repo_root=REPO_ROOT)
    post_retention = False if g3 or native_clean or source_aware or glove else validate_post_text_retention_experiment(spec, repo_root=REPO_ROOT)
    post_controller = False if g3 or native_clean or source_aware or glove else validate_post_controller_experiment(spec, repo_root=REPO_ROOT)
    unit = CLEAN_MODEL_CALLS_UNIT if clean else MODEL_CALLS_UNIT
    if (spec["budgets"].get("model_calls_unit") != unit
            or type(spec["budgets"].get("model_calls_max")) is not int or spec["budgets"]["model_calls_max"] < (0 if g3 or native_clean else 1)
            or (clean and not native_clean and spec["budgets"]["model_calls_max"] > 2)):
        raise ValueError("G2_MODEL_CALL_BUDGET_UNIT_REQUIRED" if clean else "G1_MODEL_CALL_BUDGET_UNIT_REQUIRED")
    task_id = spec["task_id"]
    prefix = (POST_GLOVE_PREFIX if glove else "inpainting_native_clean_" if native_clean else POST_SOURCE_AWARE_PREFIX if source_aware else
              G3_PREFIX if g3 else POST_TEXT_RETENTION_PREFIX if post_retention else POST_CONTROLLER_PREFIX if post_controller else
              "inpainting_generalization_g2_" if clean else "inpainting_generalization_g1_")
    if not task_id.startswith(prefix) or not task_id.replace("_", "").isalnum():
        raise ValueError("G1_FINITE_TASK_ID_REQUIRED")
    if not 0 < spec["budgets"]["wall_seconds_max"] <= 28800 or spec["budgets"]["gpu_count"] != 1:
        raise ValueError("G1_FINITE_SINGLE_CARD_REQUIRED")
    root = Path(spec["output_dir"]).resolve()
    if root.exists() or not root.is_relative_to(REPO_ROOT / "_run"):
        raise ValueError("G1_FRESH_OUTPUT_REQUIRED")
    planned_session_keys(spec)
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    if artifact_ref(infra_path) != spec["infrastructure"]:
        raise ValueError("EXISTING_INFRASTRUCTURE_SHA_REQUIRED")
    module = importlib.util.spec_from_file_location("_existing_candidate_model_publisher", infra_path)
    infra = importlib.util.module_from_spec(module)
    module.loader.exec_module(infra)
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    if check.returncode:
        raise ValueError("GOVERNANCE_REQUIRED:" + check.stdout[-1500:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    for ref in [*spec["inputs"].values(), *spec["runtime_code"], *spec["model_weights"]]:
        infra._verify(ref)
    for row in spec["sessions"]:
        verify_session_inputs(infra, row, clean=clean)
    mapping = DeviceMapping.from_inventory(spec["physical_gpu_index"], read_gpu_inventory())
    space = os.statvfs(REPO_ROOT)
    if space.f_bavail * space.f_frsize < spec["budgets"]["output_bytes_estimate"]:
        raise ValueError("G1_OUTPUT_CAPACITY_UNPROVEN")
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    if packet_path.exists():
        raise ValueError("G1_FRESH_PACKET_REQUIRED")
    writer = infra._writer_identity(task_id, 1)
    started = datetime.now(timezone.utc)
    event_prefix = (POST_GLOVE_PHASE if glove else NATIVE_CLEAN_SCOPE if native_clean else POST_SOURCE_AWARE_PHASE if source_aware else
                    "INPAINTING_G3_FROZEN_MASK" if g3 else "INPAINTING_POST_G1_TEXT_WEARER_RETENTION" if post_retention else
                    "INPAINTING_POST_G1_CONTROLLER" if post_controller else
                    "INPAINTING_G2_CLEAN" if clean else "INPAINTING_G1_MASK")
    operation = "run_inpainting_candidate_visual_stages_v1"
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", operation, "--packet", str(packet_path)]
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": ("Independent variable-native unchanged ProPainter baseline Clean; Source/Mask-bound, local upstream blockers, no reference experiment or adoption."
                      if native_clean else "Independent source-aware TEXT-retention-veto failure-cluster; one flag on the frozen six/full1033 parent recipe, no G1 revival or adoption."
                      if source_aware else "Independent G3 frozen cross-domain Mask diagnostic: original039/007/199 and path-frozen0918; unchanged TEXT retention recipe, per-session Source blockers, no quality total gate or adoption."
                      if g3 else "Independent post-G1 text wearer retention failure-cluster: one text support retention change on the same six/full1033 Sources; no new model calls, G1 timebox reset or baseline adoption."
                      if post_retention else "Independent post-G1 Controller RGB proposal failure-cluster: one global text concept change on the same six/full1033 Sources; no G1 timebox reset or baseline adoption."
                      if post_controller else "Fixed001/full238 same-Mask Clean reference-selection-only A/B; no baseline adoption."
                      if clean else "Fixed six full-session automatic RGB mask candidate; no manual seeds or session overrides."),
        "phase": NATIVE_CLEAN_SCOPE if native_clean else POST_SOURCE_AWARE_PHASE if source_aware else G3_PHASE if g3 else POST_TEXT_RETENTION_PHASE if post_retention else POST_CONTROLLER_PHASE if post_controller else "INPAINTING_G2_CLEAN_REFERENCE_AB" if clean else "INPAINTING_G1_AUTOMATIC_MASK", "plan_revision": "INPAINTING_GENERALIZATION_199_0918_20261012",
        "execution_revision": task_id.upper(), "execution_mode": "CANDIDATE_VISUAL_MODEL",
        "operation": operation, "commands": [shlex.join(command)], "command_argv": command,
        "read_set": spec["read_set"], "read_set_expansions": spec.get("read_set_expansions", []),
        "write_set": [str(root), str(packet_path.parent), str(CANDIDATE_CARD_LOCK_ROOT), "Existing CAS publisher generated projections only"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "INPUT_CODE_WEIGHT_SHA_PINNED", "PROJECT_CAPACITY_ESTABLISHED"],
        "required_outputs": ["MODEL_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": spec["budgets"], "model_stages": ["Clean"] if clean else ["Mask"],
        "deletion_allowed": False, "algorithm_execution_allowed": True,
        "stop_conditions": ["deadline", "writer conflict", "source/code/model SHA drift", "lease loss", "forbidden producer input"],
        "claim_limit": ("Native unchanged baseline Clean execution only. BLOCKED is not executed; offline Clean pixels are not geometry or training truth. No adoption."
                        if native_clean else "Independent source-aware wearer candidate. Frozen prompts/models/thresholds unchanged; no adoption, Clean/Robot or training qualification."
                        if source_aware else "G3 frozen execution diagnosis only: Source BLOCKED is not a completed Mask; failed development visual quality is preserved, not a total execution gate. No recipe changes, Clean/Robot, adoption, training or deployment qualification."
                        if g3 else "Independent post-G1 text wearer retention candidate, not a G1 successor; Controller A01 is evaluation-only. Retained text is not point observation. No Clean/Robot, adoption, training or deployment qualification."
                        if post_retention else "Independent post-G1 Controller proposal candidate, not a G1 successor; original B is evaluation-only. No Clean/Robot, adoption, training or deployment qualification."
                        if post_controller else "Candidate offline Clean execution only; no Robot, visual adoption, training or deployment claim."
                        if clean else "Candidate Mask execution only; no Clean/Robot, visual adoption, training or deployment claim."),
        "inputs": {**spec["inputs"], "spec": artifact_ref(spec_path)},
        "runtime_code": spec["runtime_code"], "model_weights": spec["model_weights"],
        "weights": {"files": spec["model_weights"]},
        "config": spec["config"], "calibration": spec["calibration"], "schema": spec["schema"],
        "sessions": spec["sessions"], "gpu_mapping": {"physical_index": mapping.physical_index, "uuid": mapping.uuid, "logical_index": 0},
        "writer": writer, "writer_file": str(root / "WRITER.json"), "packet_path": str(packet_path), "output_dir": str(root),
        "execution_started_at": started.isoformat(),
        "deadline_at": effective_deadline(spec, started).isoformat(),
        "control_ground_truth": False, "physical_deployment_authorized": False}
    if clean:
        packet.update(spec_schema=NATIVE_CLEAN_SCHEMA if native_clean else CLEAN_SPEC_SCHEMA, actual_stage_scope=spec["actual_stage_scope"],
                      clean_config=spec["clean_config"])
    if native_clean:
        packet["experiment_scope"] = NATIVE_CLEAN_SCOPE
    if glove:
        packet.update(experiment_scope=POST_GLOVE_SCOPE, spec_schema=GLOVE_SPEC_SCHEMA,
            phase=POST_GLOVE_PHASE,
            objective="Single glove RGB concept experiment on full-native001+050; unchanged models, thresholds, POINT rule and other parent parameters.",
            claim_limit="Two-session candidate execution, not adoption or full Robot/quality/training qualification. Glove TEXT and dependent POINT are actual inference, not cached claims.")
    if source_aware:
        packet.update(experiment_scope=POST_SOURCE_AWARE_SCOPE, spec_schema=SOURCE_AWARE_SPEC_SCHEMA)
    if post_controller:
        packet["experiment_scope"] = POST_CONTROLLER_SCOPE
    if post_retention:
        packet["experiment_scope"] = POST_TEXT_RETENTION_SCOPE
    if g3:
        packet.update(experiment_scope=G3_SCOPE, spec_schema=G3_SPEC_SCHEMA)
    if spec.get("resume_context"):
        packet["resume_context"] = spec["resume_context"]
    if spec.get("original_g1_deadline_at"):
        packet["original_g1_deadline_at"] = spec["original_g1_deadline_at"]
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    if errors := _validate_packet(packet):
        raise ValueError("G1_PACKET_INVALID:" + str(errors))
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    ref = infra._json_once(packet_path, packet)
    infra._json_once(root / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 0, "status": "WAIT_GPU_RESOURCE", "model_calls": 0,
        "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "task_packet": ref, "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CANDIDATE_GPU_CLEAN_SINGLE_CARD" if clean else "CANDIDATE_GPU_MASK_SINGLE_CARD"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, event_prefix + "_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "run_signature": packet["run_signature"]}), flush=True)
    errors, exit_code, child = [], None, None
    child_holder = [None]
    try:
        candidate_lease = GpuCardLease(CANDIDATE_CARD_LOCK_ROOT, mapping, task_id=task_id,
                         run_signature=packet["run_signature"], fencing_token=writer["fencing_token"])
        while not candidate_lease._held:
            try:
                candidate_lease.acquire()
            except ResourceError as error:
                if "WAIT_GPU_RESOURCE" not in str(error):
                    raise
                if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
                    raise RuntimeError("WAIT_GPU_RESOURCE_FINITE_DEADLINE") from error
                receipt, state, index = infra._snapshot()
                row, _ = own_row(packet, state, index)
                row.update(status="WAIT_GPU_RESOURCE", updated_at=now_iso(), heartbeat_at=now_iso(), model_calls=0)
                infra._publish(state, index, event_prefix + "_WAIT_GPU_RESOURCE", expected_revision=receipt["governance_revision"])
                time.sleep(10)
        with lease_until_child_stops(candidate_lease, child_holder) as lease:
            receipt, state, index = infra._snapshot()
            row, _ = own_row(packet, state, index)
            row.update(status="RUNNING", attempt=1, updated_at=now_iso(), heartbeat_at=now_iso())
            infra._publish(state, index, event_prefix + "_CARD_ACQUIRED", expected_revision=receipt["governance_revision"])
            infra._json_once(root / "CARD_LEASE.json", lease.owner)
            with (root / "WORKER.log").open("xb") as log:
                env = mapping.worker_environment()
                env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUNBUFFERED": "1"})
                child = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child_holder[0] = child
                heartbeat = time.monotonic()
                while child.poll() is None:
                    lease.assert_owned()
                    if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
                        raise RuntimeError("G1_FINITE_DEADLINE")
                    if time.monotonic() - heartbeat >= 60:
                        receipt, state, index = infra._snapshot()
                        row, _ = own_row(packet, state, index)
                        row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                        infra._publish(state, index, event_prefix + "_HEARTBEAT", expected_revision=receipt["governance_revision"])
                        heartbeat = time.monotonic()
                    time.sleep(1)
                exit_code = child.returncode
            if exit_code:
                errors.append("MODEL_WORKER_EXIT:" + str(exit_code))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
        if child is not None and child.poll() is None:
            # Lease is retained by lease_until_child_stops on this path.
            raise RuntimeError("CANNOT_SEAL_WITH_LIVE_GPU_CHILD") from error
    for ref in spec["inputs"].values():
        try:
            infra._verify(ref)
        except Exception as error:
            errors.append("PROTECTED_INPUT_FINAL_CHECK:" + str(error))
    receipt, state, index = infra._snapshot()
    row, pointer = own_row(packet, state, index)
    inner = artifact_ref(root / "MODEL_RESULT.json") if (root / "MODEL_RESULT.json").is_file() else None
    if not errors and inner:
        try:
            inner = validate_model_result(packet)
        except Exception as error:
            errors.append("MODEL_RESULT_FINAL_VERIFICATION:" + type(error).__name__ + ":" + str(error))
    result = {"schema": "inpainting-native-clean-executor-result-v1" if native_clean else "inpainting-g3-mask-executor-result-v1" if g3 else "inpainting-g2-clean-executor-result-v1" if clean else "inpainting-g1-mask-executor-result-v1", "task_id": task_id, "run_signature": packet["run_signature"],
        "status": ("BLOCKED_RESOURCE" if child is None and any("WAIT_GPU_RESOURCE" in e for e in errors)
                   else "PASSED" if not errors and inner else "FAILED_RUNTIME_FINAL"), "errors": errors,
        "worker_exit_code": exit_code, "model_result": inner, "model_result_consumable": not errors and inner is not None,
        "visual_quality": "NOT_EVALUATED",
        "candidate_adopted": False, "training_steps": 0, "files_deleted": 0, "completed_at": now_iso()}
    if g3:
        actual = load_json(root / "MODEL_RESULT.json") if inner else {}
        result.update(experiment_scope=G3_SCOPE, planned_sessions=len(packet["sessions"]),
            completed_mask_sessions=actual.get("completed_mask_sessions", 0),
            blocked_source_sessions=actual.get("blocked_source_sessions", 0),
            execution_claim="FROZEN_READY_SCOPE_ONLY_NOT_ALL_SOURCES_OR_VISUAL_QUALIFICATION")
    row.update(status=result["status"], result=infra._json_once(root / "RESULT.json", result),
               pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, event_prefix + "_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=300)
    infra._json_once(root / "POST_VALIDATION.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if result["status"] == "PASSED" and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
