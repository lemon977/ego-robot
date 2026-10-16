"""CPU contract tests only; synthetic bytes, no model/IK/render execution."""
from __future__ import annotations

import copy
import json
import importlib.util
from pathlib import Path

import pytest

from chaoyang.governance import inpainting_candidate_registration as reg
from chaoyang.inpainting_candidate import native_contracts as native
from chaoyang.inpainting_candidate.contracts import ContractError, file_ref


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_ref(path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    n = 3
    raw = {key: {"path": "/path/to/private_storage/001/raw/" + name, "bytes": 1, "sha256": "f" * 64}
        for key, name in {"video": "vst.h264", "video_qpc": "vst.qpc", "video_wall": "vst.wall",
                          "manifest": "manifest.json"}.items()}
    mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
        "output_frame": i, "qpc_declared_ns": 1000 + i * 33} for i in range(n)]
    directory = tmp_path / "Source"
    outputs = [put(directory / "FRAME_MAP.json", mapping),
        put(directory / "RAW_RGB.npz", {"synthetic": "not image or visual evidence"}),
        put(directory / "SOURCE_REPORT.json", {"session_id": "raw001", "native_sidecar_frames": n,
            "actual_inputs": raw})]
    receipt = put(directory / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1",
        "stage": "Source", "status": "SEALED_COMPLETE", "component_signature": "a" * 64,
        "session_id": "raw001", "native_frames": n, "outputs": outputs})
    original_signed, original_read = native.signed_ref, Path.read_text

    def signed(reference, root, *, verify=True, local=True):
        if isinstance(reference, dict) and str(reference.get("path", "")).startswith("/path/to/private_storage/"):
            assert not local and reference in raw.values()
            return Path(reference["path"])
        return original_signed(reference, root, verify=verify, local=local)

    def read(path, *args, **kwargs):
        if str(path) == raw["video_qpc"]["path"]:
            return "\n".join(json.dumps(m["qpc_declared_ns"]) for m in mapping)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(native, "signed_ref", signed)
    monkeypatch.setattr(Path, "read_text", read)
    row = {"session_key": "001", "session_id": "raw001", "native_frames": n,
        "source_receipt": receipt, "source_signature": "a" * 64, "frame_map_ref": outputs[0],
        "raw_inputs": raw, "native_signature": "b" * 64}
    spec = {"spec_schema": native.NATIVE_GEOMETRY_SCHEMA, "experiment_scope": native.NATIVE_GEOMETRY_SCOPE,
        "task_id": "inpainting_native_motion_render_fixture", "operation": reg.GEOMETRY_OPERATION,
        "output_dir": str(tmp_path / "Output"), "stages": ["Native"], "sessions": [row],
        "inputs": {}, "runtime_code": [], "calibration": {"metric_projection": "NOT_REQUESTED"},
        "schema": {"result": "inpainting-native-geometry-batch-result-v1"},
        "config": {"native_rule": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "camera_binding": "NOT_REQUESTED"},
        "budgets": {"gpu_count": 0, "model_calls_max": 0, "training_steps_max": 0,
            "cpu_threads_per_worker": 2, "candidates_max": 1, "wall_seconds_max": 7200,
            "render_calls_max": 0, "zbuffer_passes_max": 0}}
    return tmp_path, spec, row, mapping


def test_native_only_full_source_verified_without_model_assets_or_camera(source):
    root, spec, _, _ = source
    native.validate_native_geometry_spec(spec, repo_root=root)
    reg.validate_geometry_spec(spec, repo_root=root)


@pytest.mark.parametrize("change", ["old_scope", "old_schema", "old_task", "model", "gpu", "training",
    "threads", "wall", "manual", "q", "mask", "clean", "camera", "assets", "per_row_stages",
    "render", "zbuffer", "short", "identity", "signature", "duplicate", "outside", "resume", "config"])
def test_native_only_negative_contracts(source, change):
    root, spec, _, _ = source
    spec = copy.deepcopy(spec)
    row = spec["sessions"][0]
    if change == "old_scope": spec["experiment_scope"] = reg.GEOMETRY_SCOPE
    elif change == "old_schema": spec["spec_schema"] = reg.GEOMETRY_SPEC_SCHEMA
    elif change == "old_task": spec["task_id"] = "inpainting_motion_render_001_fixture"
    elif change == "model": spec["budgets"]["model_calls_max"] = 1
    elif change == "gpu": spec["budgets"]["gpu_count"] = 1
    elif change == "training": spec["budgets"]["training_steps_max"] = 1
    elif change == "threads": spec["budgets"]["cpu_threads_per_worker"] = 3
    elif change == "wall": spec["budgets"]["wall_seconds_max"] = 28801
    elif change == "manual": row["manual_seed"] = {}
    elif change == "q": row["q"] = []
    elif change == "mask": row["mask_receipt"] = {}
    elif change == "clean": row["clean_receipt"] = {}
    elif change == "camera": row["camera_contract"] = {}
    elif change == "assets": spec["robot_assets"] = [{"fitted": "old"}]
    elif change == "per_row_stages": row["stages"] = ["Native"]
    elif change == "render": spec["budgets"]["render_calls_max"] = 1
    elif change == "zbuffer": spec["budgets"]["zbuffer_passes_max"] = 1
    elif change == "short": row["native_frames"] = 2
    elif change == "identity": row["session_id"] = "other"
    elif change == "signature": row["native_signature"] = "not signed"
    elif change == "duplicate": spec["sessions"].append(row)
    elif change == "outside": spec["output_dir"] = "/path/to/private_storage"
    elif change == "resume": spec["resume_context"] = {}
    elif change == "config": spec["config"]["camera_binding"] = "COPY_OLD_K"
    with pytest.raises((ContractError, reg.RegistrationError)):
        reg.validate_geometry_spec(spec, repo_root=root)


@pytest.mark.parametrize("key,value", [("decoded_display_index", 0), ("model_input_frame", 0),
    ("output_frame", 0), ("qpc_declared_ns", 0)])
def test_equal_frame_count_does_not_hide_reordered_or_wrong_map(source, key, value):
    root, _, row, mapping = source
    mapping[1][key] = value
    row["frame_map_ref"] = put(Path(row["frame_map_ref"]["path"]), mapping)
    source_path = Path(row["source_receipt"]["path"])
    receipt = native.load(source_path)
    receipt["outputs"][0] = row["frame_map_ref"]
    row["source_receipt"] = put(source_path, receipt)
    with pytest.raises(ContractError, match="QPC|MAP"):
        native.source_binding(row, root)


def test_partial_source_rejected_even_with_new_outer_sha(source):
    root, _, row, _ = source
    path = Path(row["source_receipt"]["path"])
    receipt = native.load(path)
    receipt["status"] = "PARTIAL"
    row["source_receipt"] = put(path, receipt)
    with pytest.raises(ContractError):
        native.source_binding(row, root)


@pytest.mark.parametrize("scope,schema,prefix,operation,gpu", [
    (native.NATIVE_CLEAN_SCOPE, native.NATIVE_CLEAN_SCHEMA, "inpainting_native_clean_", reg.VISUAL_OPERATION, 1),
    (native.NATIVE_COMPOSITE_SCOPE, native.NATIVE_COMPOSITE_SCHEMA, "inpainting_native_composite_", reg.COMPOSITE_OPERATION, 0)])
def test_local_blocked_rows_are_explicit_not_fake_success(source, scope, schema, prefix, operation, gpu):
    root, spec, _, _ = source
    spec.update(experiment_scope=scope, spec_schema=schema, task_id=prefix + "fixture", operation=operation)
    spec["budgets"]["gpu_count"] = gpu
    diagnostic = put(root / "DIAGNOSTIC.json", {"status": "BLOCKED", "failure_code": "RENDER_NOT_PRODUCED",
        "session_id": "0918_001", "native_frames": 125})
    spec["sessions"] = [{"session_key": "0918_001", "session_id": "0918_001", "native_frames": 125,
        "upstream_status": "BLOCKED", "failure_code": "RENDER_NOT_PRODUCED", "diagnostic_ref": diagnostic}]
    rows, _ = native._base(spec, root, scope=scope, schema=schema, prefix=prefix,
        operation=operation, gpu=gpu, verify=True)
    assert rows[0]["upstream_status"] == "BLOCKED"
    spec["sessions"][0]["diagnostic_ref"] = put(Path(diagnostic["path"]), {"status": "SEALED_COMPLETE"})
    with pytest.raises(ContractError, match="FAKE_SEALED"):
        native._base(spec, root, scope=scope, schema=schema, prefix=prefix, operation=operation, gpu=gpu, verify=True)


@pytest.mark.parametrize("scope", ["old_g1", reg.POST_TEXT_RETENTION_SCOPE, reg.G3_SCOPE])
def test_source_aware_opt_in_cannot_be_smuggled_into_old_scope(tmp_path, scope):
    packet = {"task_id": "unregistered_or_old", "experiment_scope": scope,
        "config": {"source_aware_retention_veto": reg.SOURCE_AWARE_RETENTION_VETO}}
    with pytest.raises(reg.RegistrationError, match="INDEPENDENT_TASK"):
        reg.validate_post_source_aware_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("domain", ["config", "schema", "calibration", "sessions", "runtime_code"])
def test_native_run_signature_binds_real_spec_domains(source, domain):
    _, spec, _, _ = source
    changed = copy.deepcopy(spec)
    if isinstance(changed[domain], dict):
        changed[domain]["synthetic_change"] = True
    else:
        changed[domain].append({"synthetic_change": True})
    assert reg.geometry_signature(spec, []) != reg.geometry_signature(changed, [])


def source_aware_fixture(tmp_path, monkeypatch):
    loader = importlib.util.spec_from_file_location("_native_source_aware_parent_fixture",
        Path(__file__).with_name("test_inpainting_candidate_text_retention_execution.py"))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    packet, contract = module.retention_gate_fixture(tmp_path, monkeypatch)
    index_path = Path(packet["inputs"]["source_index"]["path"])
    index = native.load(index_path)
    for planned, frozen in zip(packet["sessions"], index["sessions"]):
        source_path = Path(planned["source_receipt"]["path"])
        receipt = native.load(source_path)
        receipt["session_id"] = "cpu_fake_" + planned["session_key"]
        # Isolated recipe gate fixture: the map contents are not passed as
        # production evidence. Full actual map verification is tested above.
        receipt["outputs"].append(put(source_path.parent / "FRAME_MAP.json", []))
        reference = put(source_path, receipt)
        planned["source_receipt"] = frozen["source_receipt"] = reference
    packet["inputs"]["source_index"] = put(index_path, index)
    parent = put(tmp_path / "TEXT_PARENT.json", copy.deepcopy(packet))
    packet.update(task_id="inpainting_post_g1_source_aware_fixture",
        experiment_scope=reg.POST_SOURCE_AWARE_SCOPE, phase=reg.POST_SOURCE_AWARE_PHASE,
        spec_schema=reg.SOURCE_AWARE_SPEC_SCHEMA)
    packet["inputs"]["parent_recipe_packet"] = parent
    packet["config"].update(source_aware_retention_veto=reg.SOURCE_AWARE_RETENTION_VETO,
        candidate_id="SOURCE_AWARE_FIXTURE", algorithm_parameter_change="ONE_VARIABLE")
    bound = []

    def mock_source_binding(row, root, *, verify=True):
        # Source binding itself is independently exercised using real tiny
        # sealed bytes above. This gate fixture isolates recipe/task metadata.
        assert verify and root == tmp_path and row["source_signature"]
        bound.append(row)
        return None, None, None

    monkeypatch.setattr(native, "source_binding", mock_source_binding)
    return packet, contract, bound


def test_source_aware_exact_new_scope_preserves_parent_and_six_sources(tmp_path, monkeypatch):
    packet, _, bound = source_aware_fixture(tmp_path, monkeypatch)
    assert reg.validate_post_source_aware_experiment(packet, repo_root=tmp_path)
    assert len(bound) == 6 and sum(row["native_frames"] for row in bound) == 1033
    assert not reg.validate_post_controller_experiment(packet, repo_root=tmp_path)
    assert not reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION,
        ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("change", ["flag", "prompt", "retention", "scope", "phase", "model_budget",
    "new_seed", "new_mask", "swap_source", "models", "schema", "old_deadline", "two_candidates"])
def test_source_aware_single_variable_and_finite_gate(tmp_path, monkeypatch, change):
    packet, _, _ = source_aware_fixture(tmp_path, monkeypatch)
    if change == "flag": packet["config"]["source_aware_retention_veto"] = "UNION_UNVERIFIED"
    elif change == "prompt": packet["config"]["controller_text_prompt"] = "new controller"
    elif change == "retention": packet["config"]["text_wearer_retention"] = None
    elif change == "scope": packet["experiment_scope"] = reg.G3_SCOPE
    elif change == "phase": packet["phase"] = reg.G3_PHASE
    elif change == "model_budget": packet["budgets"]["model_calls_max"] = 216
    elif change == "new_seed": packet["sessions"][0]["manual_seed"] = "forbidden"
    elif change == "new_mask": packet["sessions"][0]["mask_receipt"] = {}
    elif change == "swap_source": packet["sessions"][0]["source_receipt"] = packet["sessions"][1]["source_receipt"]
    elif change == "models": packet["model_weights"] = []
    elif change == "schema": packet["schema"]["changed"] = True
    elif change == "old_deadline": packet["original_g1_deadline_at"] = None
    elif change == "two_candidates": packet["budgets"]["candidates_max"] = 2
    with pytest.raises(reg.RegistrationError):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)
