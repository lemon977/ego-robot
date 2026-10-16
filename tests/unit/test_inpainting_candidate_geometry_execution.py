"""Finite CPU Geometry gates/receipts, using synthetic producers only.

No FK, Robot raster, neural model, GPU lease, NAS writes or registration.
Small synthetic artifacts test provenance; they are not visual evidence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np
import pytest

from chaoyang.governance import inpainting_candidate_registration as reg
from chaoyang.inpainting_candidate.contracts import file_ref

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/maintenance/inpainting_candidate_geometry_task_v1.py"
_loader = importlib.util.spec_from_file_location("_geometry_cpu_runner_tests", SCRIPT)
runner = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(runner)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_ref(path)


def blob(path, value=b"CPU synthetic artifact; not a model or video"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return file_ref(path)


def seal(path, stage, signature, outputs, **extra):
    return put(path, {"schema": "inpainting-candidate-component-v1", "stage": stage,
        "status": "SEALED_COMPLETE", "component_signature": signature,
        "native_frames": 238, "outputs": outputs, **extra})


@pytest.fixture
def finite(tmp_path, monkeypatch):
    """Raw NAS identities are fake metadata; verify wrapper never reads NAS."""
    raw = {"controller_head": {"path": "/path/to/private_storage/001/raw/aligned.h5",
                              "bytes": 1, "sha256": "f" * 64}}
    original_verify = reg.verify_ref
    verified_raw = []
    def verify(reference):
        if reference["path"].startswith("/path/to/private_storage/"):
            assert reference == raw["controller_head"]
            verified_raw.append(reference)
        else:
            original_verify(reference)
    monkeypatch.setattr(reg, "verify_ref", verify)
    mapping = [{"native_frame_index": i, "decoded_display_index": i,
                "model_input_frame": i, "output_frame": i,
                "qpc_declared_ns": 1000000000 + i * 33333333} for i in range(238)]
    source_dir = tmp_path / "Source"
    outputs = [put(source_dir / "SOURCE_REPORT.json", {"actual_inputs": raw, "session_id": "raw_001"}),
               put(source_dir / "FRAME_MAP.json", mapping), blob(source_dir / "RAW_RGB.npz")]
    source = seal(source_dir / "RECEIPT.json", "Source", "a" * 64, outputs,
                  session_id="raw_001", timebase_status="TIMEBASE_UNVERIFIED")
    camera = {"v1_manifest": blob(tmp_path / "metadata/MANIFEST.json"),
              "camera_config": blob(tmp_path / "metadata/CAMERA.json")}
    assets = [blob(tmp_path / "assets" / (str(i) + ".mesh")) for i in range(69)]
    asset_ref = put(tmp_path / "ASSET_CLOSURE.json", {"asset_count": 69, "assets": assets})
    code = blob(tmp_path / "runtime/cpu.py")
    value = {"spec_schema": reg.GEOMETRY_SPEC_SCHEMA, "experiment_scope": reg.GEOMETRY_SCOPE,
        "task_id": "inpainting_motion_render_001_fixture", "operation": reg.GEOMETRY_OPERATION,
        "output_dir": str(tmp_path / "candidate"), "inputs": {"asset_closure": asset_ref},
        "runtime_code": [code], "robot_assets": assets,
        "calibration": {"scene_depth": "ABSENT", "camera": camera},
        "schema": {"result": "inpainting-geometry-batch-result-v1"},
        "config": {"complete_robot": True, "image_size": [640, 480],
                   "motion_rule": "FROZEN_V1_RETARGET_AUTOMATIC_BASE", "scene_depth": "ABSENT"},
        "stages": ["Motion", "Render"], "budgets": {"wall_seconds_max": 7200,
            "gpu_count": 0, "model_calls_max": 0, "training_steps_max": 0,
            "cpu_threads_per_worker": 2, "render_frames_max": 238, "render_views": 2,
            "render_calls_max": 476, "candidates_max": 1},
        "sessions": [{"session_key": "001", "native_frames": 238,
            "source_receipt": source, "source_signature": "a" * 64,
            "motion_signature": "b" * 64, "render_signature": "c" * 64,
            "raw_inputs": raw, "camera_contract": camera}]}
    return tmp_path, value, mapping, verified_raw


def test_full_spec_verifies_current_raw_assets_and_sealed_source(finite):
    root, value, _, checked = finite
    reg.validate_geometry_spec(value, repo_root=root)
    assert len(checked) == 1


@pytest.mark.parametrize("domain", ["calibration", "config", "schema", "robot_assets", "runtime_code", "sessions"])
def test_all_domains_signed(finite, domain):
    _, value, _, _ = finite
    changed = copy.deepcopy(value)
    if isinstance(changed[domain], dict):
        changed[domain]["synthetic_mutation"] = True
    else:
        changed[domain].append({"synthetic_mutation": True})
    assert reg.geometry_signature(value, value["runtime_code"]) != reg.geometry_signature(changed, value["runtime_code"])


@pytest.mark.parametrize("kind", ["schema", "scope", "op", "stage", "short", "session", "gpu", "model",
    "train", "threads", "calls", "views", "candidates", "wall", "boolean", "config", "depth",
    "q", "mask", "clean", "seed", "placement", "frame_limit", "resume", "weights", "old_deadline",
    "bad_signature", "old_raw", "raw_role", "asset_count", "asset_duplicate"])
def test_spec_negative_contracts(finite, kind):
    root, value, _, _ = finite
    value = copy.deepcopy(value)
    row = value["sessions"][0]
    if kind == "schema": value["spec_schema"] = "other"
    elif kind == "scope": value["experiment_scope"] = "old_g1"
    elif kind == "op": value["operation"] = reg.VISUAL_OPERATION
    elif kind == "stage": value["stages"] = ["Mask", "Motion", "Render"]
    elif kind == "short": row["native_frames"] = 237
    elif kind == "session": row["session_key"] = "003"
    elif kind == "gpu": value["budgets"]["gpu_count"] = 1
    elif kind == "model": value["budgets"]["model_calls_max"] = 1
    elif kind == "train": value["budgets"]["training_steps_max"] = 1
    elif kind == "threads": value["budgets"]["cpu_threads_per_worker"] = 3
    elif kind == "calls": value["budgets"]["render_calls_max"] = 238
    elif kind == "views": value["budgets"]["render_views"] = 1
    elif kind == "candidates": value["budgets"]["candidates_max"] = 2
    elif kind == "wall": value["budgets"]["wall_seconds_max"] = 7201
    elif kind == "boolean": value["budgets"]["gpu_count"] = False
    elif kind == "config": value["config"]["task_name"] = "pick_place"
    elif kind == "depth": value["config"]["scene_depth"] = "INFERRED_FROM_CLEAN"
    elif kind == "q": row["robot_q"] = {}
    elif kind == "mask": row["mask_receipt"] = {}
    elif kind == "clean": row["clean_receipt"] = {}
    elif kind == "seed": row["manual_seed"] = []
    elif kind == "placement": row["placement"] = [0, 0, 0]
    elif kind == "frame_limit": row["frame_limit"] = 6
    elif kind == "resume": value["resume_context"] = {}
    elif kind == "weights": value["model_weights"] = [{"fake": True}]
    elif kind == "old_deadline": value["original_g1_deadline_at"] = None
    elif kind == "bad_signature": row["motion_signature"] = "unknown"
    elif kind == "old_raw": row["raw_inputs"]["controller_head"]["path"] = "/path/to/private_storage/001/source.npz"
    elif kind == "raw_role": row["raw_inputs"]["robot_q"] = row["raw_inputs"]["controller_head"]
    elif kind == "asset_count": value["robot_assets"].pop()
    elif kind == "asset_duplicate": value["robot_assets"][1] = value["robot_assets"][0]
    with pytest.raises(reg.RegistrationError):
        reg.validate_geometry_spec(value, repo_root=root, verify_inputs=False)


def test_full_spec_source_raw_drift_is_not_accepted(finite):
    root, value, _, _ = finite
    report_path = Path(value["sessions"][0]["source_receipt"]["path"]).parent / "SOURCE_REPORT.json"
    put(report_path, {"actual_inputs": {}, "session_id": "raw_001"})
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        reg.validate_geometry_spec(value, repo_root=root)


@pytest.fixture
def authorized(finite, monkeypatch):
    root, spec, _, _ = finite
    spec_ref = put(root / "inpainting_optimization/SPEC.json", spec)
    packet = copy.deepcopy(spec)
    writer = {"pid": 1234, "proc_start_ticks": 5678, "executor_epoch": 1,
              "fencing_token": "cpu-fixture-only", "scope": spec["task_id"]}
    path = root / "tasks/current" / spec["task_id"] / "TASK_PACKET.json"
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", reg.GEOMETRY_OPERATION,
               "--packet", str(path)]
    packet.update(phase=reg.GEOMETRY_PHASE, execution_mode="CANDIDATE_GEOMETRY_CPU",
        algorithm_execution_allowed=True, writer=writer, writer_file=str(root / "candidate/WRITER.json"),
        packet_path=str(path), commands=[shlex.join(command)], write_set=[str(root / "candidate")],
        deadline_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        run_signature=reg.geometry_signature(spec, spec["runtime_code"]))
    packet["inputs"]["spec"] = spec_ref
    put(Path(packet["writer_file"]), writer)
    packet_ref = put(path, packet)
    row = {"task_id": spec["task_id"], "status": "RUNNING", "pid": 1234,
        "proc_start_ticks": 5678, "run_signature": packet["run_signature"], "task_packet": packet_ref}
    state = {"tasks": [row], "next_task": {"task_id": spec["task_id"]}}
    index = {"task_packets": [{"task_id": spec["task_id"], "execution_allowed": True,
        "packet_path": str(path.relative_to(root)), "packet_sha256": packet_ref["sha256"]}]}
    route = {"candidate_operation": reg.GEOMETRY_OPERATION, "task_packet_required": True,
        "current_runnable_operation": None, "model_calls_authorized": 0, "gpu_count": 0,
        "code_closure": spec["runtime_code"]}
    contract = {"special_status_contracts": {reg.GEOMETRY_ROUTE_KEY: route}}
    monkeypatch.setattr(reg, "_snapshot", lambda unused: ({}, state, index, contract))
    monkeypatch.setattr(reg, "geometry_route_contract", lambda unused: {reg.GEOMETRY_ROUTE_KEY: route})
    monkeypatch.setattr(reg, "_live", lambda pid, ticks: pid == 1234 and ticks == 5678)
    from chaoyang.governance import exact78_operation_gate
    monkeypatch.setattr(exact78_operation_gate, "_caller_is_writer_descendant", lambda *args: True)
    return root, packet, state, index, route


def test_actual_gate_accepts_one_exact_fenced_command(authorized):
    root, packet, _, _, _ = authorized
    assert reg.authorize_geometry_operation(reg.GEOMETRY_OPERATION,
        ["--packet", packet["packet_path"]], repo_root=root) == packet
    assert reg.authorize_geometry_operation(reg.GEOMETRY_OPERATION,
        ["--packet", packet["packet_path"]], repo_root=root, full=False) == packet


@pytest.mark.parametrize("kind", ["no_packet", "double", "state", "writer", "sha", "dead", "caller",
    "phase", "mode", "algorithm", "command", "deadline", "sig", "config", "source", "route", "runtime"])
def test_actual_gate_rejects_before_algorithm(authorized, monkeypatch, kind):
    root, packet, state, index, route = authorized
    if kind == "no_packet": index["task_packets"] = []
    elif kind == "double": index["task_packets"].append(copy.deepcopy(index["task_packets"][0]))
    elif kind == "state": state["tasks"][0]["status"] = "PASSED"
    elif kind == "writer": put(Path(packet["writer_file"]), {"different": True})
    elif kind == "sha": index["task_packets"][0]["packet_sha256"] = "e" * 64
    elif kind == "dead": monkeypatch.setattr(reg, "_live", lambda *a: False)
    elif kind == "caller":
        from chaoyang.governance import exact78_operation_gate
        monkeypatch.setattr(exact78_operation_gate, "_caller_is_writer_descendant", lambda *a: False)
    elif kind == "route": route["gpu_count"] = 1
    elif kind == "runtime": Path(packet["runtime_code"][0]["path"]).write_bytes(b"drift")
    else:
        if kind == "phase": packet["phase"] = "INPAINTING_G0_CANDIDATE_CPU_ENGINEERING"
        elif kind == "mode": packet["execution_mode"] = "CPU_DEVELOPMENT_ONLY"
        elif kind == "algorithm": packet["algorithm_execution_allowed"] = False
        elif kind == "command": packet["commands"] = [packet["commands"][0] + " --extra true"]
        elif kind == "deadline": packet["deadline_at"] = "2020-01-01T00:00:00+00:00"
        elif kind == "sig": packet["run_signature"] = "e" * 64
        elif kind == "config": packet["config"]["scene_depth"] = "CLEAN"
        elif kind == "source": packet["sessions"][0]["source_signature"] = "d" * 64
        reference = put(Path(packet["packet_path"]), packet)
        state["tasks"][0].update(task_packet=reference, run_signature=packet["run_signature"])
        index["task_packets"][0]["packet_sha256"] = reference["sha256"]
    with pytest.raises((reg.RegistrationError, ValueError)):
        reg.authorize_geometry_operation(reg.GEOMETRY_OPERATION,
            ["--packet", packet["packet_path"]], repo_root=root)


def test_own_row_requires_actual_writer_and_one_running_packet(authorized, monkeypatch):
    _, packet, state, index, _ = authorized
    monkeypatch.setattr(runner, "proc_start_ticks", lambda pid: 5678)
    assert runner.own_row(packet, state, index)[0] is state["tasks"][0]
    state["tasks"][0]["proc_start_ticks"] = 123
    with pytest.raises(ValueError, match="FENCE_DRIFT"):
        runner.own_row(packet, state, index)


def test_operation_refuses_before_worker_import(monkeypatch):
    from chaoyang.ops import run_inpainting_candidate_geometry_stages_v1 as op
    from chaoyang.inpainting_candidate import geometry_worker
    events = []
    def deny(*args, **kwargs):
        events.append("gate")
        raise reg.RegistrationError("no packet")
    monkeypatch.setattr(reg, "authorize_geometry_operation", deny)
    monkeypatch.setattr(geometry_worker, "main", lambda *a: pytest.fail("worker reached"))
    with pytest.raises(reg.RegistrationError, match="no packet"):
        op.main(["--packet", "/invalid"])
    assert events == ["gate"]


def synthetic_motion(packet, mapping, **kwargs):
    folder = kwargs["output_dir"]
    source = packet["sessions"][0]
    common = {"session_id": "raw_001", "source_receipt": source["source_receipt"],
        "source_signature": source["source_signature"], "model_calls": 0,
        "control_ground_truth": False, "training_eligible": False, "actual_inputs": []}
    statuses = [{"native_frame_index": i, "source_frame_map": frame,
                 "left": {"quality": "FAIL"}, "right": {"quality": "UNKNOWN"}}
                for i, frame in enumerate(mapping)]
    outputs = [put(folder / "FRAME_MAP.json", mapping), put(folder / "FRAME_STATUS.json", statuses),
               put(folder / "MOTION_REPORT.json", {"native_frames": 238})]
    # Real Motion ABI has nested homonyms; root consumers must not overwrite.
    outputs.append(put(folder / "Native/FRAME_STATUS.json", {"different_native_bridge_schema": True}))
    reference = seal(folder / "RECEIPT.json", "Motion", kwargs["component_signature"], outputs, **common)
    return json.loads(Path(reference["path"]).read_text())


def synthetic_render(packet, mapping, **kwargs):
    folder = kwargs["output_dir"]
    row = packet["sessions"][0]
    statuses = [{"native_frame_index": i, "source_frame_map": frame, "motion_quality": "FAIL"}
                for i, frame in enumerate(mapping)]
    outputs = [put(folder / "FRAME_MAP.json", mapping), put(folder / "FRAME_STATUS.json", statuses),
        blob(folder / "ROBOT_BUFFERS.npz", b"CPU tiny buffer fixture, separate header tests below"),
        put(folder / "RENDER_REPORT.json", {"scene_geometry": "UNKNOWN_NOT_CONSUMED",
            "complete_robot": True, "render_calls": 476, "render_views": 2, "model_calls": 0}),
        blob(folder / "RAW_ROBOT_FPV_REVIEW.mp4"), blob(folder / "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4")]
    reference = seal(folder / "RECEIPT.json", "Render", kwargs["component_signature"], outputs,
        session_id="raw_001", source_receipt=row["source_receipt"], source_signature=row["source_signature"],
        motion_receipt=kwargs["motion_receipt"], motion_signature=kwargs["motion_signature"],
        model_calls=0, control_ground_truth=False, training_eligible=False,
        complete_robot=True, render_views=2, render_calls=476, actual_inputs=[])
    kwargs["authorization"]()
    return json.loads(Path(reference["path"]).read_text())


@pytest.fixture
def completed(authorized, finite, monkeypatch):
    root, packet, _, _, _ = authorized
    mapping = finite[2]
    from chaoyang.governance import common
    from chaoyang.inpainting_candidate import geometry_worker, motion, render
    monkeypatch.setattr(common, "REPO_ROOT", root)
    monkeypatch.setattr(reg, "authorize_geometry_operation", lambda *a, **kw: packet)
    monkeypatch.setattr(motion, "build_motion_stage", lambda **kw: synthetic_motion(packet, mapping, **kw))
    monkeypatch.setattr(render, "build_render_stage", lambda **kw: synthetic_render(packet, mapping, **kw))
    # Worker intentionally sets process env/cache; restore all of them after test.
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    for name in ("PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        monkeypatch.setenv(name, __import__("os").environ.get(name, ""))
    assert geometry_worker.main(["--packet", packet["packet_path"]]) == 0
    def tiny_buffer_check(reference, actual_qpc):
        assert Path(reference["path"]).read_bytes().startswith(b"CPU tiny")
        assert actual_qpc == [frame["qpc_declared_ns"] for frame in mapping]
    monkeypatch.setattr(runner, "_validate_buffers", tiny_buffer_check)
    result = json.loads((Path(packet["output_dir"]) / "GEOMETRY_RESULT.json").read_text())
    return packet, result, mapping


def test_worker_thin_integration_seals238_and_keeps_frame_fail(completed):
    packet, result, _ = completed
    assert result["model_calls"] == result["gpu_count"] == result["training_steps"] == 0
    assert result["completed_geometry_sessions"] == 1
    calls = []
    reference = runner.validate_geometry_result(packet, decoder=lambda path: calls.append(path.name) or 238)
    assert reference["path"].endswith("GEOMETRY_RESULT.json")
    assert set(calls) == {"RAW_ROBOT_FPV_REVIEW.mp4", "ROBOT_FULL_ARM_THIRD_PERSON_REVIEW.mp4"}
    stage, direct = runner._outputs(result["sessions"][0]["motion_receipt"], "Motion", "b" * 64)
    assert len(stage["outputs"]) == 4
    assert Path(direct["FRAME_STATUS.json"]["path"]).parent.name == "Motion"
    assert json.loads(Path(direct["FRAME_STATUS.json"]["path"]).read_text())[0]["left"]["quality"] == "FAIL"


@pytest.mark.parametrize("kind", ["task", "sig", "partial", "model", "gpu", "adopted", "training", "ground",
    "visual", "frames", "row", "source", "mask", "clean", "composite", "row_quality", "human", "path", "decode"])
def test_result_receipts_cannot_be_self_reported_success(completed, kind):
    packet, result, _ = completed
    row = result["sessions"][0]
    if kind == "task": result["task_id"] = "other"
    elif kind == "sig": result["run_signature"] = "d" * 64
    elif kind == "partial": result["completed_geometry_sessions"] = 0
    elif kind == "model": result["model_calls"] = 1
    elif kind == "gpu": result["gpu_count"] = 1
    elif kind == "adopted": result["candidate_adopted"] = True
    elif kind == "training": result["training_eligible"] = True
    elif kind == "ground": result["control_ground_truth"] = True
    elif kind == "visual": result["visual_quality"] = "PASSED"
    elif kind == "frames": result["native_frames"] = 237
    elif kind == "row": row["session_key"] = "003"
    elif kind == "source": row["source"] = row["motion_receipt"]
    elif kind == "mask": row["mask"] = "PASSED"
    elif kind == "clean": row["clean"] = "PASSED"
    elif kind == "composite": row["composite"] = "PASSED"
    elif kind == "row_quality": row["visual_quality"] = "PASSED"
    elif kind == "human": row["human_review"] = "PASSED"
    elif kind == "path": row["motion_receipt"] = row["render_receipt"]
    put(Path(packet["output_dir"]) / "GEOMETRY_RESULT.json", result)
    with pytest.raises((ValueError, KeyError)):
        runner.validate_geometry_result(packet, decoder=lambda path: 237 if kind == "decode" else 238)


def test_signed_native_bridge_output_drift_still_rejected(completed):
    packet, result, _ = completed
    motion = Path(result["sessions"][0]["motion_receipt"]["path"]).parent
    put(motion / "Native/FRAME_STATUS.json", {"drift": True})
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        runner.validate_geometry_result(packet, decoder=lambda path: 238)


def test_bad_native_buffer_header_rejected_without_full_decompression(tmp_path):
    path = tmp_path / "bad_buffers.npz"
    np.savez_compressed(path, rgba=np.zeros((1, 2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="NATIVE_BUFFER_SHAPE_OR_DTYPE:rgba"):
        runner._validate_buffers(file_ref(path), [1] * 238)


def test_geometry_route_requires_closure_but_not_nonexistent_stages(tmp_path):
    assert reg.geometry_route_contract(tmp_path) == {}
    root = SCRIPT.parents[2]
    route = reg.geometry_route_contract(root)[reg.GEOMETRY_ROUTE_KEY]
    assert route["gpu_count"] == route["model_calls_authorized"] == 0
    assert not any(ref["path"].endswith("/inpainting_candidate/stages.py") for ref in route["code_closure"])
