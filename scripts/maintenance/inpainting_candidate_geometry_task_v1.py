"""One separately registered finite CPU Geometry run using existing CAS infra.

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
    GEOMETRY_OPERATION, GEOMETRY_ROUTE_KEY, GEOMETRY_PHASE,
    geometry_route_contract, geometry_signature, validate_geometry_spec)
from chaoyang.inpainting_candidate.contracts import verify_ref, verify_sealed_stage
from chaoyang.inpainting_candidate.resources import proc_start_ticks


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
        "GEOMETRY_WRITER_FENCE_DRIFT")
    return rows[0], pointers[0]


def review_decode_count(path):
    proc = subprocess.run(["ffprobe", "-v", "error", "-threads", "2", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=180)
    streams = json.loads(proc.stdout)["streams"]
    _require(len(streams) == 1, "GEOMETRY_REVIEW_ONE_VIDEO_STREAM_REQUIRED")
    return int(streams[0]["nb_read_frames"])


def _outputs(reference, stage, signature):
    verify_ref(reference)
    path = Path(reference["path"])
    value = load_json(path)
    _require(value.get("stage") == stage and value.get("native_frames") == 238,
             "GEOMETRY_STAGE_OR_NATIVE_COUNT_DRIFT:" + stage)
    refs = verify_sealed_stage(value, signature, output_root=path.parent)
    # Native bridge artifacts remain signed and verified, but are not the
    # Motion consumer's root FRAME_STATUS/FRAME_MAP.  Never collapse nested
    # artifacts by basename (Native/FRAME_STATUS.json is a distinct ABI).
    direct = [r for r in refs if Path(r["path"]).parent == path.parent]
    names = {Path(r["path"]).name: r for r in direct}
    _require(len(names) == len(direct), "GEOMETRY_OUTPUT_NAMES_AMBIGUOUS")
    for pin in value.get("actual_inputs", []):
        verify_ref(pin)
    return value, names


def _json_any(reference):
    verify_ref(reference)
    return json.loads(Path(reference["path"]).read_text())


def _validate_buffers(reference, qpc):
    """Validate actual NPZ structure/payload lengths without a 1.5GB copy.

    Depth visibility itself is checked by the signed producer before sealing;
    this independently checks full native arrays rather than receipt counts.
    """
    import numpy as np
    verify_ref(reference)
    expected = {"rgba": ((238, 480, 640, 4), "u"),
                "depth_m": ((238, 480, 640), "f"),
                "link_id": ((238, 480, 640), "i"),
                "triangle_id": ((238, 480, 640), "i"),
                "qpc_ns": ((238,), "i"), "K": ((3, 3), "f"),
                "c2w": ((238, 4, 4), "f"), "T_world_base": ((4, 4), "f")}
    with zipfile.ZipFile(reference["path"]) as archive:
        for name, (shape, kind) in expected.items():
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                actual, _, dtype = np.lib.format._read_array_header(stream, version)
                _require(actual == shape and dtype.kind == kind and not dtype.hasobject,
                         "GEOMETRY_NATIVE_BUFFER_SHAPE_OR_DTYPE:" + name)
                _require(archive.getinfo(name + ".npy").file_size
                    == stream.tell() + int(np.prod(actual)) * dtype.itemsize,
                    "GEOMETRY_NATIVE_BUFFER_PAYLOAD_TRUNCATED:" + name)
    with np.load(reference["path"], allow_pickle=False) as archive:
        _require(np.array_equal(archive["qpc_ns"], qpc), "GEOMETRY_BUFFER_QPC_DRIFT")


def validate_geometry_result(packet, *, decoder=review_decode_count):
    """Execution completeness is separate from per-frame numerical FAIL."""
    if packet.get("experiment_scope") == "INPAINTING_NATIVE_MOTION_RENDER_CPU":
        return validate_native_geometry_result(packet, decoder=decoder)
    root = Path(packet["output_dir"])
    pin = artifact_ref(root / "GEOMETRY_RESULT.json")
    value = load_json(Path(pin["path"]))
    rows = value.get("sessions")
    _require(value.get("schema") == "inpainting-geometry-batch-result-v1"
        and value.get("task_id") == packet["task_id"]
        and value.get("run_signature") == packet["run_signature"]
        and value.get("status") == "PASSED"
        and value.get("planned_sessions") == value.get("completed_geometry_sessions") == 1
        and value.get("native_frames") == 238
        and type(value.get("model_calls")) is int and value["model_calls"] == 0
        and type(value.get("gpu_count")) is int and value["gpu_count"] == 0
        and value.get("candidate_adopted") is False
        and value.get("training_eligible") is False
        and value.get("control_ground_truth") is False
        and value.get("visual_quality") == "NOT_REVIEWED"
        and isinstance(rows, list) and len(rows) == 1,
        "GEOMETRY_BATCH_BINDING_OR_COMPLETENESS_FAILED")
    row, planned = rows[0], packet["sessions"][0]
    _require(row.get("session_key") == "001" and row.get("native_frames") == 238
        and row.get("execution") == "COMPLETE_MOTION_RENDER"
        and row.get("source") == planned["source_receipt"]
        and row.get("geometry") == "VISUAL_ROBOT_ONLY_SCENE_UNKNOWN"
        and row.get("mask") == row.get("clean") == "NOT_CONSUMED"
        and row.get("composite") == "NOT_EXECUTED"
        and row.get("visual_quality") == "NOT_REVIEWED"
        and row.get("human_review") == "PENDING"
        and row.get("training_eligible") is False and row.get("control_ground_truth") is False,
        "GEOMETRY_SESSION_QUALIFICATION_OR_IDENTITY_DRIFT")
    source, source_refs = _outputs(planned["source_receipt"], "Source", planned["source_signature"])
    mapping = _json_any(source_refs["FRAME_MAP.json"])
    _require(isinstance(mapping, list) and len(mapping) == 238,
             "GEOMETRY_NATIVE_MAP_FULL238_REQUIRED")
    for index, frame in enumerate(mapping):
        _require(all(type(frame.get(k)) is int and frame[k] == index for k in
             ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
             and type(frame.get("qpc_declared_ns")) is int,
             "GEOMETRY_NATIVE_MAP_CHANGED")
    qpc = [frame["qpc_declared_ns"] for frame in mapping]
    _require(all(b > a for a, b in zip(qpc, qpc[1:])), "GEOMETRY_NATIVE_QPC_DUPLICATE_OR_BACKWARD")
    receipts = {}
    for name in ("Motion", "Render"):
        reference = row.get(name.lower() + "_receipt")
        _require(isinstance(reference, dict) and Path(reference["path"])
                 == root / "sessions/001" / name / "RECEIPT.json",
                 "GEOMETRY_STAGE_WRITE_SET_IDENTITY:" + name)
        stage, refs = _outputs(reference, name, planned[name.lower() + "_signature"])
        _require(stage.get("source_receipt") == planned["source_receipt"]
             and stage.get("source_signature") == planned["source_signature"]
             and stage.get("session_id") == source.get("session_id")
             and stage.get("model_calls") == 0 and stage.get("control_ground_truth") is False
             and stage.get("training_eligible") is False
             and _json_any(refs["FRAME_MAP.json"]) == mapping,
             "GEOMETRY_STAGE_CURRENT_SOURCE_OR_MAP_DRIFT:" + name)
        states = _json_any(refs["FRAME_STATUS.json"])
        _require(isinstance(states, list) and len(states) == 238,
                 "GEOMETRY_FRAME_STATUS_FULL238_REQUIRED")
        for index, state in enumerate(states):
            _require(state.get("native_frame_index") == index and state.get("source_frame_map") == mapping[index],
                     "GEOMETRY_FRAME_STATUS_NATIVE_IDENTITY_DRIFT")
        receipts[name] = (stage, refs)
    render, refs = receipts["Render"]
    _require(render.get("motion_receipt") == row["motion_receipt"]
         and render.get("motion_signature") == planned["motion_signature"]
         and render.get("complete_robot") is True and render.get("render_views") == 2
         and render.get("render_calls") == packet["budgets"]["render_calls_max"] == 476,
         "GEOMETRY_RENDER_FULL_LINK_OR_CALL_ACCOUNTING_DRIFT")
    report = _json_any(refs["RENDER_REPORT.json"])
    _require(report.get("scene_geometry") == "UNKNOWN_NOT_CONSUMED"
         and report.get("complete_robot") is True and report.get("render_calls") == 476
         and report.get("render_views") == 2 and report.get("model_calls") == 0,
         "GEOMETRY_SCENE_DEPTH_OR_RENDER_REPORT_DRIFT")
    _validate_buffers(refs["ROBOT_BUFFERS.npz"], qpc)
    videos = {"RAW_ROBOT_FPV_REVIEW.mp4", "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4"}
    _require({k for k in refs if k.endswith(".mp4")} == videos,
             "GEOMETRY_TWO_FULL_REVIEW_VIDEOS_REQUIRED")
    for filename in videos:
        _require(decoder(Path(refs[filename]["path"])) == 238,
                 "GEOMETRY_REVIEW_FULL_ACTUAL_DECODE_REQUIRED")
    return pin


def _native_outputs(reference, stage, signature, count):
    """Role-aware native branch; the original fixed238 consumer stays exact."""
    from chaoyang.inpainting_candidate.geometry_worker import _direct_outputs
    value = _json_any(reference)
    _require(value.get("stage") == stage and value.get("native_frames") == count,
             "NATIVE_GEOMETRY_STAGE_OR_NATIVE_COUNT_DRIFT:" + stage)
    names = _direct_outputs(value, signature, reference["path"])
    for pin in value.get("actual_inputs", []):
        verify_ref(pin)
    return value, names


def _validate_native_buffers(reference, qpc, image_size=(640, 480)):
    import numpy as np
    verify_ref(reference)
    count, (width, height) = len(qpc), image_size
    expected = {"rgba": ((count, height, width, 4), "u"),
                "depth_m": ((count, height, width), "f"),
                "link_id": ((count, height, width), "i"),
                "triangle_id": ((count, height, width), "i"),
                "qpc_ns": ((count,), "i"), "K": ((3, 3), "f"),
                "c2w": ((count, 4, 4), "f"), "T_world_base": ((4, 4), "f")}
    with zipfile.ZipFile(reference["path"]) as archive:
        for name, (shape, kind) in expected.items():
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                actual, _, dtype = np.lib.format._read_array_header(stream, version)
                _require(actual == shape and dtype.kind == kind and not dtype.hasobject,
                         "NATIVE_GEOMETRY_BUFFER_SHAPE_OR_DTYPE:" + name)
                _require(archive.getinfo(name + ".npy").file_size
                    == stream.tell() + int(np.prod(actual)) * dtype.itemsize,
                    "NATIVE_GEOMETRY_BUFFER_PAYLOAD_TRUNCATED:" + name)
    with np.load(reference["path"], allow_pickle=False) as archive:
        _require(np.array_equal(archive["qpc_ns"], qpc), "NATIVE_GEOMETRY_BUFFER_QPC_DRIFT")


def validate_native_geometry_result(packet, *, decoder=review_decode_count):
    """All rows must finish diagnostics, not all rows must produce Robot video."""
    from chaoyang.inpainting_candidate.contracts import canonical_sha
    from chaoyang.inpainting_candidate.geometry_worker import verify_native_source, verify_native_receipt
    root = Path(packet["output_dir"])
    pin = artifact_ref(root / "GEOMETRY_RESULT.json")
    value = load_json(Path(pin["path"]))
    rows = value.get("sessions")
    planned = packet["sessions"]
    stages = packet["stages"]
    _require(stages in (["Native"], ["Motion", "Render"])
        and value.get("schema") == "inpainting-native-geometry-batch-result-v1"
        and value.get("task_id") == packet["task_id"] and value.get("run_signature") == packet["run_signature"]
        and value.get("status") == "PASSED" and isinstance(rows, list) and len(rows) == len(planned)
        and value.get("planned_sessions") == value.get("completed_sessions")
            == value.get("completed_native_sessions") == len(planned)
        and value.get("native_frames") == sum(r["native_frames"] for r in planned)
        and all(type(value.get(k)) is int and value[k] == 0 for k in ("model_calls", "gpu_count", "training_steps", "files_deleted"))
        and value.get("candidate_adopted") is False and value.get("control_ground_truth") is False
        and value.get("training_eligible") is False and value.get("visual_quality") == "NOT_REVIEWED",
        "NATIVE_GEOMETRY_BATCH_BINDING_OR_COMPLETENESS_FAILED")
    complete = blocked = render_calls = zbuffer_passes = 0
    for row, plan in zip(rows, planned):
        source, _, mapping = verify_native_source(plan)
        count = plan["native_frames"]
        qpc = [r["qpc_declared_ns"] for r in mapping]
        session = root / "sessions" / plan["session_key"]
        _require(row.get("session_key") == plan["session_key"]
            and row.get("session_id") == plan["session_id"] and row.get("native_frames") == count
            and row.get("source") == plan["source_receipt"] and row.get("frame_map_ref") == plan["frame_map_ref"]
            and row.get("timebase_status") == source.get("timebase_status")
            and row.get("mask") == row.get("clean") == "NOT_CONSUMED" and row.get("composite") == "NOT_EXECUTED"
            and row.get("visual_quality") == "NOT_REVIEWED" and row.get("human_review") == "PENDING"
            and row.get("training_eligible") is False and row.get("control_ground_truth") is False,
            "NATIVE_GEOMETRY_SESSION_IDENTITY_OR_QUALIFICATION_DRIFT")
        _require(Path(row["session_result"]["path"]) == session / "SESSION_RESULT.json"
            and _json_any(row["session_result"]) == {k:v for k,v in row.items() if k != "session_result"},
            "NATIVE_GEOMETRY_SESSION_RESULT_BINDING_DRIFT")
        if stages == ["Native"]:
            _require(Path(row["native_receipt"]["path"]) == session / "Native/RECEIPT.json"
                and row.get("execution") == "COMPLETE_NATIVE_ONLY" and row.get("render_status") == "NOT_REQUESTED"
                and row.get("motion_receipt") is None and row.get("render_receipt") is None
                and row.get("render_calls") == row.get("zbuffer_passes") == 0
                and row.get("geometry") == "UNKNOWN", "NATIVE_GEOMETRY_NATIVE_ONLY_MISCLAIM")
            verify_native_receipt(row["native_receipt"], plan["native_signature"], plan, mapping, native_only=True)
            continue
        motion_ref = row.get("motion_receipt")
        _require(isinstance(motion_ref, dict) and Path(motion_ref["path"]) == session / "Motion/RECEIPT.json",
                 "NATIVE_GEOMETRY_MOTION_WRITE_SET_DRIFT")
        motion, mrefs = _native_outputs(motion_ref, "Motion", plan["motion_signature"], count)
        native_signature = canonical_sha({"stage": "MotionSource", "Motion": plan["motion_signature"], "Source": plan["source_signature"]})
        _require(Path(row["native_receipt"]["path"]) == session / "Motion/Native/RECEIPT.json",
                 "NATIVE_GEOMETRY_NESTED_NATIVE_WRITE_SET_DRIFT")
        verify_native_receipt(row["native_receipt"], native_signature, plan, mapping, native_only=False)
        for stage, refs in ((motion, mrefs),):
            _require(stage.get("source_receipt") == plan["source_receipt"]
                and stage.get("source_signature") == plan["source_signature"] and stage.get("session_id") == plan["session_id"]
                and stage.get("model_calls") == 0 and stage.get("control_ground_truth") is False
                and stage.get("training_eligible") is False and _json_any(refs["FRAME_MAP.json"]) == mapping,
                "NATIVE_GEOMETRY_MOTION_SOURCE_OR_MAP_DRIFT")
            states = _json_any(refs["FRAME_STATUS.json"])
            _require(isinstance(states, list) and len(states) == count
                and all(type(s.get("native_frame_index")) is int and s["native_frame_index"] == i
                    and s.get("source_frame_map") == mapping[i] for i,s in enumerate(states)),
                "NATIVE_GEOMETRY_MOTION_FRAME_STATUS_DRIFT")
        if motion.get("kinematic_actions_generated") is False:
            _require(row.get("execution") == "COMPLETE_NATIVE_LOCAL_RENDER_BLOCKED" and row.get("render_status") == "BLOCKED"
                and row.get("render_receipt") is None and row.get("render_calls") == row.get("zbuffer_passes") == 0
                and isinstance(motion.get("failure_code"), str) and row.get("failure_code") == motion["failure_code"]
                and row.get("geometry") == "UNKNOWN", "NATIVE_GEOMETRY_LOCAL_BLOCKER_MISCLAIM")
            diagnostic_ref = row["render_diagnostic"]
            _require(Path(diagnostic_ref["path"]) == session / "RENDER_BLOCKED.json", "NATIVE_GEOMETRY_BLOCKED_DIAGNOSTIC_WRITE_SET_DRIFT")
            diagnostic = _json_any(diagnostic_ref)
            _require(diagnostic.get("status") == "BLOCKED" and diagnostic.get("failure_code") == motion["failure_code"]
                and diagnostic.get("native_frames") == count and diagnostic.get("source_receipt") == plan["source_receipt"]
                and diagnostic.get("motion_receipt") == motion_ref and diagnostic.get("render_calls") == diagnostic.get("zbuffer_passes") == 0,
                "NATIVE_GEOMETRY_BLOCKED_DIAGNOSTIC_DRIFT")
            import numpy as np
            with np.load(mrefs["MOTION.npz"]["path"], allow_pickle=False) as arrays:
                _require(set(arrays.files) == {"qpc_ns"} and np.array_equal(arrays["qpc_ns"], qpc),
                         "NATIVE_GEOMETRY_BLOCKED_ACTION_FICTION")
            blocked += 1
            continue
        _require(motion.get("kinematic_actions_generated") is True and row.get("execution") == "COMPLETE_MOTION_RENDER"
            and row.get("render_status") == "COMPLETE" and row.get("geometry") == "VISUAL_ROBOT_ONLY_SCENE_UNKNOWN",
            "NATIVE_GEOMETRY_ACTION_OR_RENDER_STATUS_DRIFT")
        render_ref = row.get("render_receipt")
        _require(isinstance(render_ref, dict) and Path(render_ref["path"]) == session / "Render/RECEIPT.json",
                 "NATIVE_GEOMETRY_RENDER_WRITE_SET_DRIFT")
        render, refs = _native_outputs(render_ref, "Render", plan["render_signature"], count)
        _require(render.get("source_receipt") == plan["source_receipt"] and render.get("source_signature") == plan["source_signature"]
            and render.get("session_id") == plan["session_id"] and render.get("motion_receipt") == motion_ref
            and render.get("motion_signature") == plan["motion_signature"] and render.get("complete_robot") is True
            and render.get("render_views") == 2 and render.get("model_calls") == 0
            and render.get("control_ground_truth") is False and render.get("training_eligible") is False
            and render.get("render_calls") == row.get("render_calls") == 2 * count
            and render.get("zbuffer_passes") == row.get("zbuffer_passes") == 4 * count
            and _json_any(refs["FRAME_MAP.json"]) == mapping,
            "NATIVE_GEOMETRY_RENDER_BINDING_OR_CALL_ACCOUNTING_DRIFT")
        states = _json_any(refs["FRAME_STATUS.json"])
        _require(isinstance(states, list) and len(states) == count
            and all(s.get("native_frame_index") == i and s.get("source_frame_map") == mapping[i]
                    for i,s in enumerate(states)), "NATIVE_GEOMETRY_RENDER_FRAME_STATUS_DRIFT")
        report = _json_any(refs["RENDER_REPORT.json"])
        _require(report.get("scene_geometry") == "UNKNOWN_NOT_CONSUMED" and report.get("complete_robot") is True
            and report.get("render_views") == 2 and report.get("render_calls") == 2 * count
            and report.get("zbuffer_passes") == 4 * count and report.get("model_calls") == 0,
            "NATIVE_GEOMETRY_RENDER_REPORT_DRIFT")
        _validate_native_buffers(refs["ROBOT_BUFFERS.npz"], qpc, tuple(packet["config"]["image_size"]))
        videos = {"RAW_ROBOT_FPV_REVIEW.mp4", "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4"}
        _require({k for k in refs if k.endswith(".mp4")} == videos, "NATIVE_GEOMETRY_TWO_NATIVE_REVIEW_VIDEOS_REQUIRED")
        for filename in videos:
            _require(decoder(Path(refs[filename]["path"])) == count, "NATIVE_GEOMETRY_ACTUAL_VIDEO_DECODE_COUNT_DRIFT")
        complete += 1
        render_calls += 2 * count
        zbuffer_passes += 4 * count
    _require(value.get("completed_geometry_sessions") == complete and value.get("blocked_render_sessions") == blocked
        and value.get("render_calls") == render_calls and value.get("zbuffer_passes") == zbuffer_passes
        and render_calls <= packet["budgets"]["render_calls_max"]
        and zbuffer_passes <= packet["budgets"]["zbuffer_passes_max"], "NATIVE_GEOMETRY_BATCH_ACTUAL_COUNTS_DRIFT")
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
    _require(child.poll() is not None, "GEOMETRY_LIVE_CHILD_CANNOT_SEAL")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path = args.spec.absolute()
    _require(spec_path.resolve(strict=True) == spec_path
             and spec_path.is_relative_to(REPO_ROOT / "inpainting_optimization"),
             "GEOMETRY_PROJECT_REAL_SPEC_REQUIRED")
    spec = load_json(spec_path)
    validate_geometry_spec(spec, repo_root=REPO_ROOT)
    route = geometry_route_contract(REPO_ROOT).get(GEOMETRY_ROUTE_KEY)
    _require(route and all(r in spec["runtime_code"] for r in route["code_closure"]),
             "GEOMETRY_FROZEN_RUNTIME_CLOSURE_REQUIRED")
    infra_path = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
    _require(artifact_ref(infra_path) == spec["infrastructure"], "GEOMETRY_EXISTING_INFRASTRUCTURE_SHA_REQUIRED")
    loader = importlib.util.spec_from_file_location("_existing_geometry_publisher", infra_path)
    infra = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(infra)
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    _require(check.returncode == 0, "GEOMETRY_GOVERNANCE_REQUIRED:" + check.stdout[-1500:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    output = Path(spec["output_dir"])
    _require(output.is_absolute() and output.resolve() == output and not output.exists()
             and output.is_relative_to(REPO_ROOT / "_run"), "GEOMETRY_FRESH_PROJECT_OUTPUT_REQUIRED")
    task_id = spec["task_id"]
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    _require(not packet_path.exists(), "GEOMETRY_FRESH_PACKET_REQUIRED")
    space = os.statvfs(REPO_ROOT)
    _require(space.f_bavail * space.f_frsize >= spec["budgets"]["output_bytes_estimate"],
             "GEOMETRY_OUTPUT_CAPACITY_UNPROVEN")
    writer = infra._writer_identity(task_id, 1)
    started = datetime.now(timezone.utc)
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", GEOMETRY_OPERATION, "--packet", str(packet_path)]
    packet = copy.deepcopy(spec)
    native_scope = spec.get("experiment_scope") == "INPAINTING_NATIVE_MOTION_RENDER_CPU"
    packet.update(schema_version="chaoyang-r22-task-packet-v1", phase=GEOMETRY_PHASE,
        execution_mode="CANDIDATE_GEOMETRY_CPU", execution_revision=task_id.upper(),
        objective=("Independent source-native zero-model CPU motion audit and optional full-link visual Render diagnostic."
                   if native_scope else "Independent full001/native238 zero-model CPU Motion and full-link Render diagnostic."),
        plan_revision="INPAINTING_GENERALIZATION_199_0918_20261012",
        writer=writer, writer_file=str(output / "WRITER.json"), packet_path=str(packet_path),
        commands=[shlex.join(command)], command_argv=command,
        required_outputs=["GEOMETRY_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        prerequisites=["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "INPUT_CODE_CAMERA_ASSET_SHA_PINNED"],
        write_set=[str(output), str(packet_path.parent), "Existing CAS publisher generated projections only"],
        deletion_allowed=False, algorithm_execution_allowed=True, physical_deployment_authorized=False,
        execution_started_at=started.isoformat(),
        deadline_at=(started + timedelta(seconds=spec["budgets"]["wall_seconds_max"])).isoformat(),
        stop_conditions=["deadline", "writer conflict", "input/code/camera/asset SHA drift", "input integrity error"],
        claim_limit="Candidate visual Robot/depth diagnostic only; frame numerical FAIL does not abort the session; no Clean/Composite/adoption/training/deployment.",
        run_signature=geometry_signature(spec, route["code_closure"]))
    packet["inputs"]["spec"] = artifact_ref(spec_path)
    _require(not _validate_packet(packet), "GEOMETRY_TASK_PACKET_INVALID")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    packet_ref = infra._json_once(packet_path, packet)
    infra._json_once(output / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": GEOMETRY_PHASE, "status": "RUNNING", "attempt": 1,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "updated_at": now_iso(), "heartbeat_at": now_iso(), "task_packet": packet_ref,
        "run_signature": packet["run_signature"], "model_calls": 0})
    state["next_task"] = {"task_id": task_id, "expected_resource": "CPU_TWO_THREADS_NO_GPU_MODELS"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "INPAINTING_GEOMETRY_REGISTERED", expected_revision=receipt["governance_revision"])
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
                         "GEOMETRY_FINITE_DEADLINE")
                if time.monotonic() - beat >= 60:
                    receipt, state, index = infra._snapshot()
                    row, _ = own_row(packet, state, index)
                    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
                    infra._publish(state, index, "INPAINTING_GEOMETRY_HEARTBEAT",
                                   expected_revision=receipt["governance_revision"])
                    beat = time.monotonic()
                time.sleep(0.5)
            exit_code = child.returncode
            _require(exit_code == 0, "GEOMETRY_WORKER_EXIT:" + str(exit_code))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        _stop_child(child)
    inner = None
    try:
        validate_geometry_spec(spec, repo_root=REPO_ROOT)
        if not errors:
            inner = validate_geometry_result(packet)
    except Exception as error:
        errors.append("GEOMETRY_FINAL_VERIFICATION:" + type(error).__name__ + ":" + str(error))
    receipt, state, index = infra._snapshot()
    row, pointer = own_row(packet, state, index)
    result = {"schema": "inpainting-geometry-executor-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "status": "PASSED" if not errors and inner else "FAILED_RUNTIME_FINAL",
        "errors": errors, "worker_exit_code": exit_code, "geometry_result": inner,
        "geometry_result_consumable": not errors and inner is not None,
        "model_calls": 0, "gpu_count": 0, "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "training_eligible": False, "training_steps": 0, "files_deleted": 0, "completed_at": now_iso()}
    row.update(status=result["status"], result=infra._json_once(output / "RESULT.json", result),
               pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, "INPAINTING_GEOMETRY_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    infra._json_once(output / "POST_VALIDATION.json", {"exit_code": check.returncode,
        "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if result["status"] == "PASSED" and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
