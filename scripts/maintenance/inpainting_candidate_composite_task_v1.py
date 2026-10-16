"""One separately registered finite CPU Composite run using existing CAS infra.

No GPU lease, model, second queue, scene depth invention or baseline promotion.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import zipfile

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance.inpainting_candidate_registration import (
    COMPOSITE_OPERATION, COMPOSITE_ROUTE_KEY, COMPOSITE_PHASE,
    composite_route_contract, composite_signature, validate_composite_spec)
from chaoyang.inpainting_candidate.contracts import verify_ref, verify_sealed_stage
from chaoyang.inpainting_candidate.resources import proc_start_ticks
from chaoyang.inpainting_candidate.native_contracts import (
    NATIVE_COMPOSITE_SCOPE, validate_native_composite_spec)


def _require(value, code):
    if not value:
        raise ValueError(code)


def own_row(packet, state, index):
    rows = [r for r in state.get("tasks", []) if r.get("task_id") == packet["task_id"]]
    active = [r for r in state.get("tasks", []) if r.get("status") in
              {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    pointers = [r for r in index.get("task_packets", []) if r.get("execution_allowed") is True]
    writer = packet["writer"]
    pin = artifact_ref(Path(packet["packet_path"]))
    _require(len(rows) == len(active) == len(pointers) == 1 and active[0] is rows[0]
        and rows[0].get("status") == "RUNNING"
        and rows[0].get("pid") == writer["pid"]
        and rows[0].get("proc_start_ticks") == writer["proc_start_ticks"]
        and rows[0].get("run_signature") == packet["run_signature"]
        and rows[0].get("task_packet") == pin
        and (state.get("next_task") or {}).get("task_id") == packet["task_id"]
        and pointers[0].get("task_id") == packet["task_id"]
        and pointers[0].get("packet_sha256") == pin["sha256"]
        and load_json(Path(packet["packet_path"])) == packet
        and load_json(Path(packet["writer_file"])) == writer
        and proc_start_ticks(writer["pid"]) == writer["proc_start_ticks"],
        "COMPOSITE_WRITER_FENCE_DRIFT")
    return rows[0], pointers[0]


def review_decode_count(path):
    proc = subprocess.run(["ffprobe", "-v", "error", "-threads", "2", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=180)
    streams = json.loads(proc.stdout)["streams"]
    _require(len(streams) == 1, "COMPOSITE_REVIEW_ONE_VIDEO_STREAM_REQUIRED")
    return int(streams[0]["nb_read_frames"])


def _outputs(reference, stage, signature, *, native_frames=238):
    verify_ref(reference)
    path = Path(reference["path"])
    value = load_json(path)
    _require(value.get("stage") == stage and (value.get("native_frames") == native_frames
        or value.get("native_input_frames") == value.get("native_output_frames") == native_frames),
        "COMPOSITE_STAGE_OR_NATIVE_COUNT_DRIFT:" + stage)
    refs = verify_sealed_stage(value, signature, output_root=path.parent)
    direct = [r for r in refs if Path(r["path"]).parent == path.parent]
    names = {Path(r["path"]).name: r for r in direct}
    _require(len(names) == len(direct), "COMPOSITE_OUTPUT_NAMES_AMBIGUOUS")
    return value, names


def _json_any(reference):
    verify_ref(reference)
    return json.loads(Path(reference["path"]).read_text())


def _validate_arrays(reference, source_refs, mask_refs, clean_refs, render_refs, mapping, *, native_frames=238):
    """Actual full native payload and preservation checks, not reported counts."""
    import hashlib
    import numpy as np
    verify_ref(reference)
    n, h, w = native_frames, 480, 640
    expected = {"final_rgb": ((n,h,w,3), "u"),
        "geometry_decision": ((n,h,w), "u"), "display_policy": ((n,h,w), "u"),
        "dense_robot_owner": ((n,h,w), "i"), "source_link_labels": ((n,h,w), "i"),
        "model_predicted_object_support": ((n,h,w), "b"), "mask_unknown": ((n,h,w), "b"),
        "qpc_ns": ((n,), "i")}
    with zipfile.ZipFile(reference["path"]) as archive:
        for name, (shape, kind) in expected.items():
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                actual, _, dtype = np.lib.format._read_array_header(stream, version)
                _require(actual == shape and dtype.kind == kind and not dtype.hasobject,
                    "COMPOSITE_NATIVE_ARRAY_SHAPE_OR_DTYPE:" + name)
                exact = np.dtype(np.uint8 if kind == "u" else np.bool_ if kind == "b"
                    else np.int64 if name == "qpc_ns" else np.int32)
                _require(dtype == exact, "COMPOSITE_NATIVE_ARRAY_EXACT_DTYPE:" + name)
                _require(archive.getinfo(name + ".npy").file_size
                    == stream.tell() + int(np.prod(actual)) * dtype.itemsize,
                    "COMPOSITE_NATIVE_ARRAY_PAYLOAD_TRUNCATED:" + name)
    with np.load(source_refs["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
        raw = archive["rgb"]
    _require(raw.shape == (n,h,w,3) and raw.dtype == np.uint8, "COMPOSITE_CURRENT_RAW_RGB_DOMAIN")
    digest = hashlib.sha256()
    digest.update(str(raw.shape).encode("ascii"))
    digest.update(raw.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(raw).data)
    frame_sha = digest.hexdigest()
    with np.load(mask_refs["MASK_ARRAYS.npz"]["path"], allow_pickle=False) as archive:
        masks = {key: archive[key] for key in ("protected_visible", "effective_removal", "human",
                                              "equipment", "object_observed", "unknown")}
    with np.load(clean_refs["CLEAN_ARRAYS.npz"]["path"], allow_pickle=False) as archive:
        clean, effective, protected = archive["clean_rgb"], archive["effective_removal"], archive["protected_visible"]
    _require(clean.shape == raw.shape and clean.dtype == raw.dtype
        and all(value.shape == (n,h,w) and value.dtype == np.bool_ for value in masks.values())
        and effective.shape == protected.shape == (n,h,w)
        and effective.dtype == protected.dtype == np.bool_
        and np.array_equal(effective, masks["effective_removal"])
        and np.array_equal(protected, masks["protected_visible"]), "COMPOSITE_FROZEN_P_OR_E_DRIFT")
    with np.load(render_refs["ROBOT_BUFFERS.npz"]["path"], allow_pickle=False) as archive:
        rgba, labels = archive["rgba"], archive["link_id"]
    _require(rgba.shape == (n,h,w,4) and rgba.dtype == np.uint8
        and labels.shape == (n,h,w) and labels.dtype == np.int32
        and np.array_equal(rgba[...,3] > 0, labels >= 0)
        and np.isin(rgba[...,3], [0,255]).all(), "COMPOSITE_RENDER_ALPHA_LABEL_DOMAIN")
    with np.load(reference["path"], allow_pickle=False) as archive:
        _require(np.array_equal(archive["qpc_ns"], [f["qpc_declared_ns"] for f in mapping]),
            "COMPOSITE_QPC_ARRAY_DRIFT")
        _require(not np.any(archive["geometry_decision"]), "COMPOSITE_UNKNOWN_GEOMETRY_MUST_REMAIN_UNKNOWN")
        _require(np.array_equal(archive["source_link_labels"], labels), "COMPOSITE_SOURCE_LINK_LABEL_DRIFT")
        _require(np.array_equal(archive["mask_unknown"], masks["unknown"]), "COMPOSITE_MASK_UNKNOWN_DROPPED")
        support = (masks["protected_visible"] & masks["object_observed"] & ~masks["unknown"]
            & ~masks["human"] & ~masks["equipment"] & ~effective)
        _require(np.array_equal(archive["model_predicted_object_support"], support),
            "COMPOSITE_MODEL_OBJECT_SUPPORT_DRIFT")
        policy = archive["display_policy"]
        final = archive["final_rgb"]
        owners = archive["dense_robot_owner"]
        lookup = json.loads((Path(reference["path"]).parent / "LABEL_LOOKUP.json").read_text())
        owner_lookup = {-1: -1, **{item["source_link_label"]: item["dense_owner"] for item in lookup["entries"]}}
        for index in range(n):
            for label in np.unique(labels[index]):
                _require(int(label) in owner_lookup and np.all(owners[index][labels[index] == label] == owner_lookup[int(label)]),
                    "COMPOSITE_DENSE_OWNER_SPARSE_LABEL_DRIFT")
            _require(not np.any(clean[index][protected[index]] != raw[index][protected[index]])
                and not np.any(clean[index][~effective[index]] != raw[index][~effective[index]]),
                "COMPOSITE_CLEAN_PRESERVATION_ZERO_CHANGE_FAILED")
            robot = labels[index] >= 0
            fallback = robot & support[index] & ~masks["human"][index] & ~masks["equipment"][index] & ~effective[index]
            wanted = np.where(robot, 4, 0).astype(np.uint8)
            wanted[fallback] = 3
            _require(np.array_equal(policy[index], wanted), "COMPOSITE_PIXEL_POLICY_OR_EXCLUSION_DRIFT")
            draw = robot & ~fallback
            _require(np.array_equal(final[index][draw], rgba[index,...,:3][draw])
                and np.array_equal(final[index][fallback], raw[index][fallback])
                and np.array_equal(final[index][~robot], clean[index][~robot]),
                "COMPOSITE_FINAL_ACTUAL_PIXEL_PROVENANCE_DRIFT")
    return frame_sha


def validate_composite_result(packet, *, decoder=review_decode_count):
    if packet.get("experiment_scope") == NATIVE_COMPOSITE_SCOPE:
        return validate_native_composite_result(packet, decoder=decoder)
    root = Path(packet["output_dir"])
    pin = artifact_ref(root / "COMPOSITE_RESULT.json")
    value = load_json(Path(pin["path"]))
    rows = value.get("sessions")
    _require(value.get("schema") == "inpainting-composite-batch-result-v1"
        and value.get("task_id") == packet["task_id"] and value.get("run_signature") == packet["run_signature"]
        and value.get("status") == "PASSED"
        and value.get("planned_sessions") == value.get("completed_composite_sessions") == 1
        and value.get("native_frames") == 238
        and all(type(value.get(k)) is int and value[k] == 0 for k in ("model_calls", "gpu_count", "render_calls", "training_steps"))
        and value.get("candidate_adopted") is False and value.get("training_eligible") is False
        and value.get("control_ground_truth") is False and value.get("visual_quality") == "NOT_REVIEWED"
        and isinstance(rows, list) and len(rows) == 1, "COMPOSITE_BATCH_BINDING_OR_COMPLETENESS_FAILED")
    row, planned = rows[0], packet["sessions"][0]
    _require(row.get("session_key") == "001" and row.get("native_frames") == 238
        and row.get("execution") == "COMPLETE_COMPOSITE"
        and row.get("geometry") == "UNKNOWN_SCENE_DEPTH_ABSENT"
        and row.get("object_support_qualification") == "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"
        and row.get("mask_quality") == row.get("clean_quality") == "FAILED"
        and row.get("visual_quality") == "NOT_REVIEWED" and row.get("human_review") == "PENDING"
        and row.get("training_eligible") is False and row.get("control_ground_truth") is False,
        "COMPOSITE_SESSION_QUALIFICATION_OR_IDENTITY_DRIFT")
    upstream = {}
    for name in ("Source", "Mask", "Clean", "Render"):
        reference = planned[name.lower() + "_receipt"]
        _require(row.get(name.lower()) == reference, "COMPOSITE_CURRENT_UPSTREAM_IDENTITY_DRIFT")
        upstream[name] = _outputs(reference, name, planned[name.lower() + "_signature"])
    source, source_refs = upstream["Source"]
    mapping = _json_any(source_refs["FRAME_MAP.json"])
    _require(isinstance(mapping,list) and len(mapping) == 238, "COMPOSITE_FULL_NATIVE_MAP_REQUIRED")
    for index, frame in enumerate(mapping):
        _require(all(type(frame.get(k)) is int and frame[k] == index for k in
            ("native_frame_index","decoded_display_index","model_input_frame","output_frame"))
            and type(frame.get("qpc_declared_ns")) is int, "COMPOSITE_NATIVE_MAP_IDENTITY_DRIFT")
    _require(all(b["qpc_declared_ns"] > a["qpc_declared_ns"] for a,b in zip(mapping,mapping[1:])),
        "COMPOSITE_QPC_BACKWARD_OR_DUPLICATE")
    for _, refs in upstream.values():
        _require(_json_any(refs["FRAME_MAP.json"]) == mapping, "COMPOSITE_UPSTREAM_MAP_DRIFT")
    reference = row.get("composite_receipt")
    _require(isinstance(reference,dict) and Path(reference["path"]) == root / "sessions/001/Composite/RECEIPT.json"
        and row.get("composite") == reference, "COMPOSITE_STAGE_WRITE_SET_IDENTITY_DRIFT")
    stage, refs = _outputs(reference,"Composite",planned["composite_signature"])
    for name in ("source","mask","clean","render"):
        _require(stage.get(name + "_receipt") == planned[name + "_receipt"]
            and stage.get(name + "_signature") == planned[name + "_signature"], "COMPOSITE_STAGE_UPSTREAM_SIGNATURE_DRIFT")
    _require(stage.get("session_id") == source.get("session_id")
        and stage.get("upstream_quality_evidence") == planned["upstream_quality_evidence"]
        and stage.get("control_ground_truth") is False and stage.get("training_eligible") is False
        and stage.get("qualified_trusted_occlusion_pixels") == 0,
        "COMPOSITE_STAGE_UNVERIFIED_QUALIFICATION_DRIFT")
    _require(_json_any(refs["FRAME_MAP.json"]) == mapping, "COMPOSITE_OUTPUT_MAP_DRIFT")
    states = _json_any(refs["FRAME_STATUS.json"])
    _require(isinstance(states,list) and len(states) == 238, "COMPOSITE_FRAME_STATUS_FULL238_REQUIRED")
    for index, state in enumerate(states):
        _require(state.get("native_frame_index") == index and state.get("source_frame_map") == mapping[index],
            "COMPOSITE_FRAME_STATUS_NATIVE_IDENTITY_DRIFT")
    report = _json_any(refs["COMPOSITE_REPORT.json"])
    _require(report.get("geometry") == "UNKNOWN_SCENE_DEPTH_ABSENT"
        and report.get("object_support_qualification") == "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"
        and report.get("qualified_trusted_occlusion_pixels") == 0
        and report.get("mask_quality") == report.get("clean_quality") == "FAILED"
        and report.get("visual_quality") == "NOT_REVIEWED"
        and report.get("control_ground_truth") is False and report.get("training_eligible") is False,
        "COMPOSITE_REPORT_UPSTREAM_FAILURE_OR_QUALIFICATION_DRIFT")
    lookup = _json_any(refs["LABEL_LOOKUP.json"])
    entries = lookup.get("entries")
    _require(isinstance(entries,list) and all(item.get("dense_owner") == i
        and type(item.get("source_link_label")) is int and all(isinstance(item.get(k),str) and item[k] for k in
            ("instance_id","part_id","link_id")) for i,item in enumerate(entries))
        and len({item["source_link_label"] for item in entries}) == len(entries),
        "COMPOSITE_DENSE_LABEL_LOOKUP_INVALID")
    frame_sha = _validate_arrays(refs["COMPOSITE_ARRAYS.npz"], source_refs, upstream["Mask"][1],
        upstream["Clean"][1], upstream["Render"][1], mapping)
    _require(upstream["Mask"][0].get("actual_inputs",{}).get("frames_sha256") == frame_sha
        and upstream["Clean"][0].get("actual_inputs",{}).get("frames_sha256") == frame_sha,
        "COMPOSITE_CURRENT_RAW_PIXEL_SHA_DRIFT")
    videos = {"FINAL_CANDIDATE.mp4", "RAW_MASK_CLEAN_ROBOT_REVIEW.mp4"}
    _require({name for name in refs if name.endswith(".mp4")} == videos, "COMPOSITE_TWO_FULL_REVIEW_VIDEOS_REQUIRED")
    for filename in videos:
        _require(decoder(Path(refs[filename]["path"])) == 238, "COMPOSITE_REVIEW_FULL_ACTUAL_DECODE_REQUIRED")
    return pin


def validate_native_composite_result(packet, *, decoder=review_decode_count):
    """Independent sealed/map/pixel/video checks for variable-N READY rows."""
    validate_native_composite_spec(packet, repo_root=REPO_ROOT)
    root = Path(packet["output_dir"])
    pin = artifact_ref(root / "COMPOSITE_RESULT.json")
    value = load_json(Path(pin["path"]))
    rows = value.get("sessions", [])
    plans = packet["sessions"]
    _require(value.get("schema") == "inpainting-native-composite-batch-result-v1"
        and value.get("experiment_scope") == NATIVE_COMPOSITE_SCOPE
        and value.get("task_id") == packet["task_id"] and value.get("run_signature") == packet["run_signature"]
        and [r.get("session_key") for r in rows] == [r["session_key"] for r in plans]
        and value.get("planned_sessions") == len(plans), "NATIVE_COMPOSITE_BATCH_IDENTITY_DRIFT")
    completed, blocked, frames = 0, 0, 0
    for planned, row in zip(plans, rows):
        n = planned["native_frames"]
        _require(row.get("native_frames") == n and row.get("session_id") == planned["session_id"]
            and row.get("experiment_scope") == NATIVE_COMPOSITE_SCOPE
            and row.get("training_eligible") is False and row.get("control_ground_truth") is False
            and row.get("human_review") == "PENDING" and row.get("visual_quality") == "NOT_REVIEWED",
            "NATIVE_COMPOSITE_ROW_IDENTITY_OR_QUALIFICATION_DRIFT")
        if planned["upstream_status"] == "BLOCKED":
            verify_ref(planned["diagnostic_ref"])
            _require(row.get("execution") == "BLOCKED_UPSTREAM"
                and row.get("failure_code") == planned["failure_code"]
                and row.get("diagnostic_ref") == planned["diagnostic_ref"]
                and all(row.get(k) == 0 for k in ("model_calls", "render_calls", "gpu_count"))
                and not any(k in row for k in ("composite", "composite_receipt", "review", "video")),
                "NATIVE_BLOCKED_COMPOSITE_CANNOT_FAKE_STAGE")
            blocked += 1
            continue
        _require(row.get("execution") == "COMPLETE_COMPOSITE"
            and row.get("geometry") == "UNKNOWN_SCENE_DEPTH_ABSENT"
            and row.get("object_support_qualification") == "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"
            and row.get("qualified_trusted_occlusion_pixels") == 0,
            "NATIVE_COMPOSITE_READY_EXECUTION_OR_GEOMETRY_DRIFT")
        upstream = {}
        for name in ("Source", "Mask", "Clean", "Render"):
            reference = planned[name.lower() + "_receipt"]
            _require(row.get(name.lower()) == reference, "NATIVE_COMPOSITE_UPSTREAM_REFERENCE_DRIFT")
            upstream[name] = _outputs(reference, name, planned[name.lower() + "_signature"], native_frames=n)
        source, source_refs = upstream["Source"]
        mapping = _json_any(source_refs["FRAME_MAP.json"])
        _require(len(mapping) == n and all(_json_any(refs["FRAME_MAP.json"]) == mapping
            for _, refs in upstream.values()), "NATIVE_COMPOSITE_FULL_UPSTREAM_MAP_DRIFT")
        reference = row.get("composite_receipt")
        _require(isinstance(reference, dict) and row.get("composite") == reference
            and Path(reference["path"]) == root / "sessions" / planned["session_key"] / "Composite/RECEIPT.json",
            "NATIVE_COMPOSITE_WRITE_SET_DRIFT")
        stage, refs = _outputs(reference, "Composite", planned["composite_signature"], native_frames=n)
        qualities = {name: (_json_any(planned["upstream_quality_evidence"][name])["visual_quality"]
            if name in planned["upstream_quality_evidence"] else
            upstream[name][0].get("visual_quality", "NOT_EVALUATED")) for name in ("Mask", "Clean")}
        _require(stage.get("session_id") == source["session_id"]
            and stage.get("upstream_quality_evidence") == planned["upstream_quality_evidence"]
            and stage.get("control_ground_truth") is False and stage.get("training_eligible") is False
            and stage.get("qualified_trusted_occlusion_pixels") == 0,
            "NATIVE_COMPOSITE_STAGE_QUALIFICATION_DRIFT")
        for name in ("source", "mask", "clean", "render"):
            _require(stage.get(name + "_receipt") == planned[name + "_receipt"]
                and stage.get(name + "_signature") == planned[name + "_signature"],
                "NATIVE_COMPOSITE_STAGE_UPSTREAM_SIGNATURE_DRIFT")
        _require(_json_any(refs["FRAME_MAP.json"]) == mapping, "NATIVE_COMPOSITE_OUTPUT_MAP_DRIFT")
        states = _json_any(refs["FRAME_STATUS.json"])
        _require(len(states) == n and all(s.get("native_frame_index") == i
            and s.get("source_frame_map") == mapping[i] for i, s in enumerate(states)),
            "NATIVE_COMPOSITE_FULL_FRAME_STATUS_DRIFT")
        report = _json_any(refs["COMPOSITE_REPORT.json"])
        _require(row.get("mask_quality") == report.get("mask_quality") == qualities["Mask"]
            and row.get("clean_quality") == report.get("clean_quality") == qualities["Clean"]
            and report.get("geometry") == "UNKNOWN_SCENE_DEPTH_ABSENT"
            and report.get("object_support_qualification") == "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"
            and report.get("qualified_trusted_occlusion_pixels") == 0
            and report.get("control_ground_truth") is False and report.get("training_eligible") is False
            and report.get("visual_quality") == "NOT_REVIEWED", "NATIVE_COMPOSITE_UPSTREAM_QUALITY_UPGRADE")
        lookup = _json_any(refs["LABEL_LOOKUP.json"]).get("entries")
        _require(isinstance(lookup, list) and all(v.get("dense_owner") == i
            and type(v.get("source_link_label")) is int and all(isinstance(v.get(k), str) and v[k]
            for k in ("instance_id", "part_id", "link_id")) for i, v in enumerate(lookup))
            and len({v["source_link_label"] for v in lookup}) == len(lookup),
            "NATIVE_COMPOSITE_OWNER_LOOKUP_INVALID")
        raw_sha = _validate_arrays(refs["COMPOSITE_ARRAYS.npz"], source_refs, upstream["Mask"][1],
            upstream["Clean"][1], upstream["Render"][1], mapping, native_frames=n)
        _require(all(upstream[name][0].get("actual_inputs", {}).get("frames_sha256") == raw_sha
            for name in ("Mask", "Clean")), "NATIVE_COMPOSITE_CURRENT_RAW_PIXEL_SHA_DRIFT")
        videos = {"FINAL_CANDIDATE.mp4", "RAW_MASK_CLEAN_ROBOT_REVIEW.mp4"}
        _require({name for name in refs if name.endswith(".mp4")} == videos
            and all(decoder(Path(refs[name]["path"])) == n for name in videos),
            "NATIVE_COMPOSITE_FULL_VIDEO_DECODE_FAILED")
        completed += 1
        frames += n
    _require(completed > 0 and value.get("status") == "PASSED"
        and value.get("execution_diagnostic_complete") is True
        and value.get("completed_sessions") == value.get("completed_composite_sessions") == completed
        and value.get("blocked_sessions") == blocked and value.get("failed_sessions") == 0
        and value.get("native_frames") == frames
        and value.get("planned_native_frames") == sum(r["native_frames"] for r in plans)
        and all(type(value.get(k)) is int and value[k] == 0 for k in ("model_calls", "render_calls", "gpu_count", "training_steps", "files_deleted"))
        and value.get("candidate_adopted") is False and value.get("training_eligible") is False
        and value.get("control_ground_truth") is False and value.get("visual_quality") == "NOT_REVIEWED",
        "NATIVE_COMPOSITE_COUNTS_OR_FALSE_COMPLETENESS")
    return pin


def _stop_child(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)
    _require(child.poll() is not None, "COMPOSITE_LIVE_CHILD_CANNOT_SEAL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path = args.spec.absolute()
    _require(spec_path.resolve(strict=True) == spec_path
             and spec_path.is_relative_to(REPO_ROOT / "inpainting_optimization"),
             "COMPOSITE_PROJECT_REAL_SPEC_REQUIRED")
    spec = load_json(spec_path)
    validate_composite_spec(spec, repo_root=REPO_ROOT)
    native = spec.get("experiment_scope") == NATIVE_COMPOSITE_SCOPE
    route = composite_route_contract(REPO_ROOT).get(COMPOSITE_ROUTE_KEY)
    _require(route and all(r in spec["runtime_code"] for r in route["code_closure"]),
             "COMPOSITE_FROZEN_RUNTIME_CLOSURE_REQUIRED")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    _require(artifact_ref(infra_path) == spec["infrastructure"], "COMPOSITE_EXISTING_INFRASTRUCTURE_SHA_REQUIRED")
    loader = importlib.util.spec_from_file_location("_existing_composite_publisher", infra_path)
    infra = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(infra)
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    _require(check.returncode == 0, "COMPOSITE_GOVERNANCE_REQUIRED:" + check.stdout[-1500:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    output = Path(spec["output_dir"])
    _require(output.is_absolute() and output.resolve() == output and not output.exists()
             and output.is_relative_to(REPO_ROOT / "_run"), "COMPOSITE_FRESH_PROJECT_OUTPUT_REQUIRED")
    task_id = spec["task_id"]
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    _require(not packet_path.exists(), "COMPOSITE_FRESH_PACKET_REQUIRED")
    space = os.statvfs(REPO_ROOT)
    _require(space.f_bavail * space.f_frsize >= spec["budgets"]["output_bytes_estimate"],
             "COMPOSITE_OUTPUT_CAPACITY_UNPROVEN")
    writer = infra._writer_identity(task_id, 1)
    started = datetime.now(timezone.utc)
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", COMPOSITE_OPERATION, "--packet", str(packet_path)]
    packet = copy.deepcopy(spec)
    packet.update(schema_version="chaoyang-r22-task-packet-v1", phase=NATIVE_COMPOSITE_SCOPE if native else COMPOSITE_PHASE,
        execution_mode="CANDIDATE_COMPOSITE_CPU", execution_revision=task_id.upper(),
        objective=("Independent variable-native zero-model Composite with local upstream blockers."
            if native else "Independent full001/native238 zero-model CPU pixel Composite diagnostic."),
        plan_revision="INPAINTING_GENERALIZATION_199_0918_20261012",
        writer=writer, writer_file=str(output / "WRITER.json"), packet_path=str(packet_path),
        commands=[shlex.join(command)], command_argv=command,
        required_outputs=["COMPOSITE_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        prerequisites=["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "INPUT_CODE_CAMERA_ASSET_SHA_PINNED"],
        write_set=[str(output), str(packet_path.parent), "Existing CAS publisher generated projections only"],
        deletion_allowed=False, algorithm_execution_allowed=True, physical_deployment_authorized=False,
        execution_started_at=started.isoformat(),
        deadline_at=(started + timedelta(seconds=spec["budgets"]["wall_seconds_max"])).isoformat(),
        stop_conditions=["deadline", "writer conflict", "input/code/camera/asset SHA drift", "input integrity error"],
        claim_limit="Candidate pixel Composite only; scene UNKNOWN and known upstream FAILED remain unchanged; no adoption/training/deployment.",
        run_signature=composite_signature(spec, route["code_closure"]))
    packet["inputs"]["spec"] = artifact_ref(spec_path)
    _require(not _validate_packet(packet), "COMPOSITE_TASK_PACKET_INVALID")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    packet_ref = infra._json_once(packet_path, packet)
    infra._json_once(output / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "status": "RUNNING", "attempt": 1,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "updated_at": now_iso(), "heartbeat_at": now_iso(), "task_packet": packet_ref,
        "run_signature": packet["run_signature"], "model_calls": 0})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_TWO_THREADS_NO_GPU_MODELS"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "INPAINTING_COMPOSITE_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "run_signature": packet["run_signature"]}), flush=True)
    cache = output / "executor_cache"
    cache.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(REPO_ROOT / "src"), PYTHONUNBUFFERED="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    child, exit_code, errors = None, None, []
    try:
        with (output / "WORKER.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            beat = time.monotonic()
            while child.poll() is None:
                _require(datetime.now(timezone.utc) < datetime.fromisoformat(packet["deadline_at"]),
                         "COMPOSITE_FINITE_DEADLINE")
                if time.monotonic() - beat >= 60:
                    receipt, state, index = infra._snapshot()
                    row, _ = own_row(packet, state, index)
                    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                    infra._publish(state, index, "INPAINTING_COMPOSITE_HEARTBEAT",
                                   expected_revision=receipt["governance_revision"])
                    beat = time.monotonic()
                time.sleep(0.5)
            exit_code = child.returncode
            _require(exit_code == 0, "COMPOSITE_WORKER_EXIT:" + str(exit_code))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        _stop_child(child)
    inner = None
    try:
        validate_composite_spec(spec, repo_root=REPO_ROOT)
        if not errors:
            inner = validate_composite_result(packet)
    except Exception as error:
        errors.append("COMPOSITE_FINAL_VERIFICATION:" + type(error).__name__ + ":" + str(error))
    receipt, state, index = infra._snapshot()
    row, pointer = own_row(packet, state, index)
    result = {"schema": "inpainting-composite-executor-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED" if not errors and inner else "FAILED_RUNTIME_FINAL",
        "errors": errors, "worker_exit_code": exit_code, "composite_result": inner,
        "composite_result_consumable": not errors and inner is not None,
        "model_calls": 0, "gpu_count": 0, "render_calls": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "training_eligible": False, "training_steps": 0, "files_deleted": 0, "completed_at": now_iso()}
    row.update(status=result["status"], result=infra._json_once(output / "RESULT.json", result),
               pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, "INPAINTING_COMPOSITE_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(output / "POST_VALIDATION.json", {"exit_code": check.returncode,
        "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if result["status"] == "PASSED" and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
