"""CPU synthetic Composite gate/lifecycle tests: no production/model/render."""
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

ROOT = Path(__file__).resolve().parents[2]
def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(value)
    return value

geometry_tests = module("_shared_geometry_fixture", Path(__file__).with_name("test_inpainting_candidate_geometry_execution.py"))
finite = geometry_tests.finite
put, blob, seal = geometry_tests.put, geometry_tests.blob, geometry_tests.seal
runner = module("_composite_cpu_runner", ROOT / "scripts/maintenance/inpainting_candidate_composite_task_v1.py")


@pytest.fixture
def complete_inputs(finite):
    root, old_spec, mapping, _ = finite
    source_row = old_spec["sessions"][0]
    source = source_row["source_receipt"]
    raw_ref = source_row["raw_inputs"]["controller_head"]
    refs = {}
    for stage, signature in (("Mask", "b" * 64), ("Clean", "c" * 64), ("Render", "d" * 64)):
        folder = root / stage
        outputs = [put(folder / "FRAME_MAP.json", mapping), put(folder / "FRAME_STATUS.json",
            [{"native_frame_index": i, "source_frame_map": frame, "quality": "FAILED"}
             for i, frame in enumerate(mapping)])]
        if stage == "Mask":
            outputs.append(blob(folder / "MASK_ARRAYS.npz"))
        elif stage == "Clean":
            outputs.append(blob(folder / "CLEAN_ARRAYS.npz"))
        else:
            outputs.extend([blob(folder / "ROBOT_BUFFERS.npz"), put(folder / "LABEL_IDENTITY.json",
                {"2000": {"instance_id": "left_kaihand", "part_id": "palm", "link_id": "base_link"}})])
        actual = {"raw_source": raw_ref, "frames_sha256": "9" * 64}
        if stage == "Clean": actual["mask_receipt"] = refs["Mask"]
        reference = seal(folder / "RECEIPT.json", stage, signature, outputs,
            session_id="raw_001", actual_inputs=actual, source_receipt=source,
            source_signature="a" * 64, complete_robot=True, control_ground_truth=False, training_eligible=False)
        if stage in {"Mask", "Clean"}:
            value = json.loads(Path(reference["path"]).read_text())
            value.pop("native_frames")
            value.update(native_input_frames=238, native_output_frames=238)
            reference = put(Path(reference["path"]), value)
        refs[stage] = reference
    quality = {name: put(root / (name + "_QUALITY_STATUS.json"), {
        "schema": "inpainting-upstream-quality-status-v1", "stage": name,
        "upstream_receipt": refs[name], "visual_quality": "FAILED",
        "human_review": "PENDING", "hidden_regions": "UNKNOWN", "review_reference": {"sha256": "8" * 64}})
        for name in ("Mask", "Clean")}
    row = {"session_key": "001", "native_frames": 238, "source_receipt": source,
        "source_signature": "a" * 64, "mask_receipt": refs["Mask"], "mask_signature": "b" * 64,
        "clean_receipt": refs["Clean"], "clean_signature": "c" * 64,
        "render_receipt": refs["Render"], "render_signature": "d" * 64,
        "composite_signature": "e" * 64, "upstream_quality_evidence": quality}
    value = {"spec_schema": reg.COMPOSITE_SPEC_SCHEMA, "experiment_scope": reg.COMPOSITE_SCOPE,
        "task_id": "inpainting_composite_001_fixture", "operation": reg.COMPOSITE_OPERATION,
        "output_dir": str(root / "composite_candidate"), "inputs": {},
        "runtime_code": old_spec["runtime_code"], "calibration": {"scene_depth": "ABSENT"},
        "schema": {"result": "inpainting-composite-batch-result-v1"}, "stages": ["Composite"],
        "config": {"scene_depth": "ABSENT", "display_policy": "RAW_MODEL_OBJECT_FIRST_DIAGNOSTIC",
                   "image_size": [640, 480], "complete_robot": True},
        "budgets": {"gpu_count": 0, "model_calls_max": 0, "training_steps_max": 0, "render_calls_max": 0,
            "cpu_threads_per_worker": 2, "candidates_max": 1, "frames_max": 238,
            "review_videos": 2, "wall_seconds_max": 7200}, "sessions": [row]}
    return root, value, mapping


def test_current_full_upstream_references_and_count_abi(complete_inputs):
    root, spec, _ = complete_inputs
    reg.validate_composite_spec(spec, repo_root=root)


@pytest.mark.parametrize("kind", ["scope", "schema", "op", "stage", "session", "short", "gpu", "model", "render",
    "train", "threads", "wall", "bool", "views", "config", "depth", "seed", "points", "placement", "q", "resume", "sig", "quality"])
def test_bounded_spec_rejects_wrong_or_manual_input(complete_inputs, kind):
    root, spec, _ = complete_inputs
    spec = copy.deepcopy(spec)
    row = spec["sessions"][0]
    if kind == "scope": spec["experiment_scope"] = reg.GEOMETRY_SCOPE
    elif kind == "schema": spec["spec_schema"] = reg.GEOMETRY_SPEC_SCHEMA
    elif kind == "op": spec["operation"] = reg.GEOMETRY_OPERATION
    elif kind == "stage": spec["stages"] = ["Render", "Composite"]
    elif kind == "session": row["session_key"] = "003"
    elif kind == "short": row["native_frames"] = 237
    elif kind == "gpu": spec["budgets"]["gpu_count"] = 1
    elif kind == "model": spec["budgets"]["model_calls_max"] = 1
    elif kind == "render": spec["budgets"]["render_calls_max"] = 476
    elif kind == "train": spec["budgets"]["training_steps_max"] = 1
    elif kind == "threads": spec["budgets"]["cpu_threads_per_worker"] = 3
    elif kind == "wall": spec["budgets"]["wall_seconds_max"] = 7201
    elif kind == "bool": spec["budgets"]["gpu_count"] = False
    elif kind == "views": spec["budgets"]["review_videos"] = 1
    elif kind == "config": spec["config"]["paper_cup"] = True
    elif kind == "depth": spec["config"]["scene_depth"] = "CLEAN"
    elif kind == "seed": row["manual_seed"] = []
    elif kind == "points": row["evaluation_annotation"] = {}
    elif kind == "placement": row["placement"] = []
    elif kind == "q": row["robot_q"] = {}
    elif kind == "resume": spec["resume_context"] = None
    elif kind == "sig": row["composite_signature"] = "wrong"
    elif kind == "quality": row["upstream_quality_evidence"] = {}
    with pytest.raises(reg.RegistrationError):
        reg.validate_composite_spec(spec, repo_root=root, verify_inputs=False)


def test_known_failure_cannot_be_relabelled_pass(complete_inputs):
    root, spec, _ = complete_inputs
    row = spec["sessions"][0]
    reference = row["upstream_quality_evidence"]["Clean"]
    value = json.loads(Path(reference["path"]).read_text())
    value["visual_quality"] = "PASS"
    row["upstream_quality_evidence"]["Clean"] = put(Path(reference["path"]), value)
    with pytest.raises(reg.RegistrationError, match="KNOWN_UPSTREAM_FAILURE"):
        reg.validate_composite_spec(spec, repo_root=root)


@pytest.fixture
def authorized(complete_inputs, monkeypatch):
    root, spec, mapping = complete_inputs
    packet = copy.deepcopy(spec)
    path = root / "tasks/current" / spec["task_id"] / "TASK_PACKET.json"
    writer = {"pid": 1234, "proc_start_ticks": 5678, "executor_epoch": 1,
              "fencing_token": "fixture_only", "scope": spec["task_id"]}
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", reg.COMPOSITE_OPERATION,
               "--packet", str(path)]
    packet.update(phase=reg.COMPOSITE_PHASE, execution_mode="CANDIDATE_COMPOSITE_CPU",
        algorithm_execution_allowed=True, writer=writer, writer_file=str(Path(spec["output_dir"]) / "WRITER.json"),
        packet_path=str(path), commands=[shlex.join(command)], write_set=[spec["output_dir"]],
        deadline_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        run_signature=reg.composite_signature(spec, spec["runtime_code"]))
    packet["inputs"]["spec"] = put(root / "inpainting_optimization/SPEC.json", spec)
    put(Path(packet["writer_file"]), writer)
    reference = put(path, packet)
    state = {"tasks": [{"task_id": spec["task_id"], "status": "RUNNING", "pid": 1234,
        "proc_start_ticks": 5678, "run_signature": packet["run_signature"], "task_packet": reference}],
        "next_task": {"task_id": spec["task_id"]}}
    index = {"task_packets": [{"task_id": spec["task_id"], "execution_allowed": True,
        "packet_path": str(path.relative_to(root)), "packet_sha256": reference["sha256"]}]}
    route = {"candidate_operation": reg.COMPOSITE_OPERATION, "task_packet_required": True,
        "current_runnable_operation": None, "gpu_count": 0, "model_calls_authorized": 0,
        "render_calls_authorized": 0, "code_closure": spec["runtime_code"]}
    monkeypatch.setattr(reg, "_snapshot", lambda *a: ({}, state, index,
        {"special_status_contracts": {reg.COMPOSITE_ROUTE_KEY: route}}))
    monkeypatch.setattr(reg, "composite_route_contract", lambda *a: {reg.COMPOSITE_ROUTE_KEY: route})
    monkeypatch.setattr(reg, "_live", lambda pid, ticks: (pid,ticks) == (1234,5678))
    from chaoyang.governance import exact78_operation_gate
    monkeypatch.setattr(exact78_operation_gate, "_caller_is_writer_descendant", lambda *a: True)
    return root, packet, state, index, mapping


def test_one_exact_fenced_composite_operation(authorized):
    root, packet, _, _, _ = authorized
    assert reg.authorize_composite_operation(reg.COMPOSITE_OPERATION,
        ["--packet",packet["packet_path"]],repo_root=root) == packet


@pytest.mark.parametrize("kind", ["no_packet", "double", "state", "writer", "sha", "dead", "caller", "runtime",
    "phase", "mode", "command", "deadline", "sig", "config", "source"])
def test_gate_negative_before_algorithm_import(authorized, monkeypatch, kind):
    root, packet, state, index, _ = authorized
    if kind == "no_packet": index["task_packets"] = []
    elif kind == "double": index["task_packets"].append(copy.deepcopy(index["task_packets"][0]))
    elif kind == "state": state["tasks"][0]["status"] = "PASSED"
    elif kind == "writer": put(Path(packet["writer_file"]), {"different": True})
    elif kind == "sha": index["task_packets"][0]["packet_sha256"] = "f" * 64
    elif kind == "dead": monkeypatch.setattr(reg,"_live",lambda *a:False)
    elif kind == "caller":
        from chaoyang.governance import exact78_operation_gate
        monkeypatch.setattr(exact78_operation_gate,"_caller_is_writer_descendant",lambda *a:False)
    elif kind == "runtime": Path(packet["runtime_code"][0]["path"]).write_bytes(b"drift")
    else:
        if kind == "phase": packet["phase"] = reg.GEOMETRY_PHASE
        elif kind == "mode": packet["execution_mode"] = "CPU_DEVELOPMENT_ONLY"
        elif kind == "command": packet["commands"][0] += " --extra yes"
        elif kind == "deadline": packet["deadline_at"] = "2020-01-01T00:00:00+00:00"
        elif kind == "sig": packet["run_signature"] = "f" * 64
        elif kind == "config": packet["config"]["scene_depth"] = "CLEAN"
        elif kind == "source": packet["sessions"][0]["source_signature"] = "f" * 64
        reference = put(Path(packet["packet_path"]),packet)
        state["tasks"][0].update(task_packet=reference,run_signature=packet["run_signature"])
        index["task_packets"][0]["packet_sha256"] = reference["sha256"]
    with pytest.raises((reg.RegistrationError,ValueError)):
        reg.authorize_composite_operation(reg.COMPOSITE_OPERATION,["--packet",packet["packet_path"]],repo_root=root)


def test_op_gate_before_importing_worker(monkeypatch):
    from chaoyang.ops import run_inpainting_candidate_composite_stage_v1 as op
    from chaoyang.inpainting_candidate import composite_worker
    monkeypatch.setattr(composite_worker,"main",lambda *a:pytest.fail("worker reached"))
    def denied(*a,**kw): raise reg.RegistrationError("missing finite packet")
    monkeypatch.setattr(reg,"authorize_composite_operation",denied)
    with pytest.raises(reg.RegistrationError,match="missing finite"):
        op.main(["--packet","/invalid"])


@pytest.fixture
def completed(authorized, monkeypatch):
    root, packet, _, _, mapping = authorized
    from chaoyang.governance import common
    from chaoyang.inpainting_candidate import composite_worker, composite
    monkeypatch.setattr(common,"REPO_ROOT",root)
    monkeypatch.setattr(reg,"authorize_composite_operation",lambda *a,**kw:packet)
    def provider(**kwargs):
        folder = kwargs["output_dir"]
        row = packet["sessions"][0]
        report = {"geometry":"UNKNOWN_SCENE_DEPTH_ABSENT",
            "object_support_qualification":"MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED",
            "qualified_trusted_occlusion_pixels":0,"mask_quality":"FAILED","clean_quality":"FAILED",
            "visual_quality":"NOT_REVIEWED","control_ground_truth":False,"training_eligible":False}
        outputs = [put(folder/"FRAME_MAP.json",mapping),put(folder/"FRAME_STATUS.json",
            [{"native_frame_index":i,"source_frame_map":f} for i,f in enumerate(mapping)]),
            put(folder/"COMPOSITE_REPORT.json",report),put(folder/"LABEL_LOOKUP.json",{"entries":[{
                "dense_owner":0,"source_link_label":2000,"instance_id":"left_kaihand","part_id":"palm","link_id":"base_link"}]}),
            blob(folder/"COMPOSITE_ARRAYS.npz"),blob(folder/"FINAL_CANDIDATE.mp4"),blob(folder/"RAW_MASK_CLEAN_ROBOT_REVIEW.mp4")]
        fields = {key:row[key] for key in row if key.endswith("_receipt") or key.endswith("_signature")}
        fields.pop("composite_signature")
        fields.update(session_id="raw_001",upstream_quality_evidence=row["upstream_quality_evidence"],
            control_ground_truth=False,training_eligible=False,qualified_trusted_occlusion_pixels=0)
        reference = seal(folder/"RECEIPT.json","Composite",kwargs["component_signature"],outputs,**fields)
        kwargs["authorization"]()
        return json.loads(Path(reference["path"]).read_text())
    monkeypatch.setattr(composite,"build_composite_stage",provider)
    monkeypatch.setattr(tempfile,"tempdir",tempfile.tempdir)
    monkeypatch.setattr(sys,"dont_write_bytecode",sys.dont_write_bytecode)
    for name in ("PYTHONDONTWRITEBYTECODE","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS","CUDA_VISIBLE_DEVICES","TMPDIR","TMP","TEMP","XDG_CACHE_HOME","NUMBA_CACHE_DIR"):
        monkeypatch.setenv(name,__import__("os").environ.get(name,""))
    assert composite_worker.main(["--packet",packet["packet_path"]]) == 0
    monkeypatch.setattr(runner,"_validate_arrays",lambda *a:"9"*64)
    result = json.loads((Path(packet["output_dir"])/"COMPOSITE_RESULT.json").read_text())
    return packet,result


def test_worker_complete_zero_model_zero_render_and_preserved_fail(completed):
    packet,result = completed
    assert result["model_calls"] == result["gpu_count"] == result["render_calls"] == 0
    assert result["sessions"][0]["mask_quality"] == result["sessions"][0]["clean_quality"] == "FAILED"
    decoded = []
    assert runner.validate_composite_result(packet,decoder=lambda p:decoded.append(p.name) or 238)
    assert len(decoded) == 2


@pytest.mark.parametrize("kind",["task","sig","partial","gpu","model","render","train","adopt","visual",
    "row","frames","mask_quality","clean_quality","geometry","support","human","source","decode"])
def test_completed_result_never_proves_visual_or_training_pass(completed,kind):
    packet,result = completed
    row=result["sessions"][0]
    if kind=="task": result["task_id"]="other"
    elif kind=="sig": result["run_signature"]="f"*64
    elif kind=="partial": result["completed_composite_sessions"]=0
    elif kind=="gpu": result["gpu_count"]=1
    elif kind=="model": result["model_calls"]=1
    elif kind=="render": result["render_calls"]=476
    elif kind=="train": result["training_steps"]=1
    elif kind=="adopt": result["candidate_adopted"]=True
    elif kind=="visual": result["visual_quality"]="PASSED"
    elif kind=="row": row["session_key"]="003"
    elif kind=="frames": row["native_frames"]=237
    elif kind=="mask_quality": row["mask_quality"]="PASSED"
    elif kind=="clean_quality": row["clean_quality"]="PASSED"
    elif kind=="geometry": row["geometry"]="VERIFIED"
    elif kind=="support": row["object_support_qualification"]="TRUSTED"
    elif kind=="human": row["human_review"]="PASSED"
    elif kind=="source": row["source"]=row["mask"]
    put(Path(packet["output_dir"])/"COMPOSITE_RESULT.json",result)
    with pytest.raises(ValueError):
        runner.validate_composite_result(packet,decoder=lambda p:237 if kind=="decode" else 238)


def test_bad_composite_payload_header_rejected_without_model(tmp_path):
    path=tmp_path/"bad.npz"
    np.savez_compressed(path,final_rgb=np.zeros((1,2,2,3),np.uint8))
    with pytest.raises(ValueError,match="NATIVE_ARRAY_SHAPE_OR_DTYPE:final_rgb"):
        runner._validate_arrays(file_ref(path),{},{},{},{},[])


def test_additive_composite_route_has_no_model_render_or_adoption(tmp_path):
    assert reg.composite_route_contract(tmp_path)=={}
    route=reg.composite_route_contract(ROOT)[reg.COMPOSITE_ROUTE_KEY]
    assert route["model_calls_authorized"] == route["render_calls_authorized"] == route["gpu_count"] == 0
    assert route["current_runnable_operation"] is None
    assert route["candidate_adopted"] is False
