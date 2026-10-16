"""Frozen G3 metadata/dispatch tests; providers, device authority and codec fake."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from chaoyang.governance import inpainting_candidate_registration as reg
from chaoyang.inpainting_candidate.contracts import file_ref


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = _load("_g3_entry_fixtures", "test_inpainting_candidate_entry.py")
execution = _load("_g3_execution_fixtures", "test_inpainting_candidate_model_execution.py")
runner, worker = execution.runner, execution.worker
source_worker_batch = execution.source_worker_batch


def g3_fixture(tmp_path, monkeypatch, *, source_fixture=None):
    packet, contract = entry.visual_gate_fixture(tmp_path, monkeypatch)
    config = {"candidate_id": "G3", "algorithm_parameter_change": "NONE_FROZEN_CROSS_DOMAIN_DIAGNOSTIC",
        "stage": "G1_AUTOMATIC_MASK_ONLY", "mask_rule_variant": "RGB_GLOVE_WEARER_SUPPORT_B",
        "image_size": [640, 480], "manual_seed_inputs": 0, "recipe_session_branches": 0,
        "projection_hints": None, "controller_text_prompt": "VR controller",
        "text_wearer_retention": reg.TEXT_WEARER_RETENTION_RULE}
    if source_fixture is not None:
        packet = source_fixture.packet
        old_source_folder = Path(packet["sessions"][0]["source_receipt"]["path"]).parent
        source_folder = tmp_path / "source_g3/039"
        source_folder.mkdir(parents=True)
        import shutil
        shutil.copyfile(old_source_folder / "RAW_RGB.npz", source_folder / "RAW_RGB.npz")
        execution.put(source_folder / "FRAME_MAP.json", json.loads((old_source_folder / "FRAME_MAP.json").read_text()))
        execution.put(source_folder / "SOURCE_REPORT.json", json.loads((old_source_folder / "SOURCE_REPORT.json").read_text()))
        count = len(source_fixture.expected["001"][0])
        source_fixture.expected["pick_place_039"] = source_fixture.expected["001"]
    else:
        source_folder = tmp_path / "source/039"
        source_folder.mkdir(parents=True)
        count = 2
        (source_folder / "RAW_RGB.npz").write_bytes(b"CPU metadata fixture; not neural output")
        execution.put(source_folder / "FRAME_MAP.json", [{"native_frame_index": i,
            "decoded_display_index": i, "model_input_frame": i, "output_frame": i,
            "qpc_declared_ns": 1_000_000_000 + i * 40_000_000} for i in range(count)])
    raw_dir = tmp_path / "raw/039"
    raw_dir.mkdir(parents=True)
    raw = {}
    for name in ("video", "manifest", "video_qpc", "video_wall"):
        path = raw_dir / ("raw/vst.h264" if name == "video" else name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("\n".join(str(1_000_000_000 + i * 40_000_000) for i in range(count)) + "\n").encode()
                         if name in {"video_qpc", "video_wall"} else b"synthetic immutable raw identity " + name.encode())
        if name == "manifest":
            path.write_text(json.dumps({"session": "pick_place_039"}))
        raw[name] = file_ref(path)
    previous = (json.loads((source_folder / "SOURCE_REPORT.json").read_text())
                if (source_folder / "SOURCE_REPORT.json").exists() else {})
    previous.update(session_id="pick_place_039", session_dir=str(raw_dir), actual_inputs=raw,
                    native_sidecar_frames=count)
    execution.put(source_folder / "SOURCE_REPORT.json", previous)
    source_ref = execution.put(source_folder / "RECEIPT.json", {
        "schema": "inpainting-candidate-component-v1", "stage": "Source", "status": "SEALED_COMPLETE",
        "component_signature": "c" * 64, "session_id": "pick_place_039", "native_frames": count,
        "actual_inputs": list(raw.values()), "outputs": [file_ref(source_folder / name) for name in
            ("RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json")]})
    rows = [{"session_key": "pick_place_039", "session_id": "pick_place_039", "raw_session_dir": str(raw_dir),
        "native_frames": count, "source_status": "READY", "source_receipt": source_ref,
        "source_signature": "c" * 64, "frame_map_ref": file_ref(source_folder / "FRAME_MAP.json")}]
    cross = []
    for key, frames in [("pick_place_007", 3), ("pick_place_199", 4)] + [(f"cards_0918_{i:03}", 2) for i in range(1, 9)]:
        raw_session_dir = tmp_path / "raw" / key
        # Challenge selection identifies its original three-digit source folder.
        if key.startswith("pick_place_"):
            raw_session_dir = tmp_path / "raw" / key[-3:]
            raw_refs = file_ref(raw["video"]["path"])
            raw_refs["path"] = str(raw_session_dir / "raw/vst.h264")
        else:
            cross.append({"identity": key, "session_dir": str(raw_session_dir), "source_refs": {}})
        diagnostic = execution.put(tmp_path / "diagnostics" / (key + ".json"),
            {"status": "BLOCKED_PHYSICAL_EYE_UNRESOLVED", "model_calls": 0})
        rows.append({"session_key": key, "session_id": key, "raw_session_dir": str(raw_session_dir),
            "native_frames": frames, "source_status": "BLOCKED", "failure_code": "BLOCKED_PHYSICAL_EYE_UNRESOLVED",
            "diagnostic_ref": diagnostic})
    selection_raw = [{"expected": reference} for reference in raw.values()]
    for row in rows[1:3]:
        selection_raw.append({"expected": {**raw["video"], "path": str(Path(row["raw_session_dir"]) / "raw/vst.h264")}})
    selection = execution.put(tmp_path / "SELECTION.json", {"known_challenges_not_blind": [
        {"session_id": key, "frames": frames} for key, frames in (("039", count), ("007", 3), ("199", 4))],
        "cross_domain_selected": cross, "raw_file_checks": selection_raw})
    index = execution.put(tmp_path / "SOURCE_INDEX_G3.json", {"schema": reg.G3_MANIFEST_SCHEMA,
        "sessions": [{**row, "visual_quality": "NOT_EVALUATED", "projection_qualified": False} for row in rows]})
    parent = execution.put(tmp_path / "RETENTION_PACKET.json", {
        "task_id": "inpainting_post_g1_text_retention_model_a01_20261013",
        "experiment_scope": reg.POST_TEXT_RETENTION_SCOPE, "model_stages": ["Mask"],
        "config": {**config, "candidate_id": "PARENT"},
        "weights": {"files": packet.get("model_weights", [])}, "model_weights": packet.get("model_weights", [])})
    planned = [{**reg.g3_session_inputs(row), **({"mask_signature": "d" * 64,
        "model_calls_max": reg.g3_api_bound(row["native_frames"])} if row["source_status"] == "READY" else {})}
        for row in rows]
    packet.update(task_id=reg.G3_PREFIX + "a01_20261013", experiment_scope=reg.G3_SCOPE,
        phase=reg.G3_PHASE, execution_mode="CANDIDATE_VISUAL_MODEL", spec_schema=reg.G3_SPEC_SCHEMA,
        model_stages=["Mask"], config=config, sessions=planned,
        model_weights=packet.get("model_weights", []),
        weights={"files": packet.get("model_weights", [])},
        schema={"model_result": reg.G3_RESULT_SCHEMA},
        calibration={"metric_projection": "NOT_CONSUMED_MASK",
            "source_eye_qualification": "PER_SESSION_SOURCE_RECEIPT_NOT_METRIC_PROJECTION"},
        inputs={"source_index": index, "parent_recipe_packet": parent, "frozen_selection": selection},
        budgets={"candidates_max": 1, "gpu_count": 1, "cpu_threads_per_worker": 2,
            "training_steps_max": 0, "model_calls_max": reg.g3_api_bound(count),
            "model_calls_unit": worker.MODEL_CALLS_UNIT, "wall_seconds_max": 28800})
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    if source_fixture is not None:
        execution.put(Path(packet["packet_path"]), packet)
        monkeypatch.setattr(reg, "authorize_candidate_operation", lambda *a, **k: packet)
    return packet, contract


def test_g3_fixed_recipe_new_scope_blocked_independent_gate(tmp_path, monkeypatch):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    assert reg.validate_g3_experiment(packet, repo_root=tmp_path)
    assert not reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)
    assert not reg.validate_post_controller_experiment(packet, repo_root=tmp_path)
    assert len(runner.planned_session_keys(packet)) == 11
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("field,value", [("experiment_scope", reg.POST_TEXT_RETENTION_SCOPE),
    ("spec_schema", "inpainting-six-source-stage-index-v1"), ("model_stages", ["Clean"]),
    ("resume_context", {}), ("original_g1_deadline_at", None), ("phase", "INPAINTING_G1_AUTOMATIC_MASK")])
def test_g3_wrong_scope_and_old_timebox_refused(tmp_path, monkeypatch, field, value):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet[field] = value
    with pytest.raises(reg.RegistrationError):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("controller_text_prompt", "controller"),
    ("text_wearer_retention", None), ("manual_points", [[1, 1]]),
    ("manual_seed_inputs", 1), ("image_size", [320, 240]), ("task_name", "cards")])
def test_g3_recipe_no_session_tuning_or_annotations(tmp_path, monkeypatch, field, value):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet["config"][field] = value
    with pytest.raises(reg.RegistrationError, match="RECIPE_DRIFT"):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("candidates_max", 2), ("gpu_count", 2), ("model_calls_max", 57),
    ("training_steps_max", 1), ("wall_seconds_max", 28801), ("cpu_threads_per_worker", 3),
    ("model_calls_unit", "NEURAL_FORWARD"), ("model_calls_max", True)])
def test_g3_finite_budget_exact_original_rule_ceiling(tmp_path, monkeypatch, field, value):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet["budgets"][field] = value
    with pytest.raises(reg.RegistrationError):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["manual_seed", "manual_points", "mask_receipt", "historical_mask", "placement", "frame_limit"])
def test_g3_forbidden_producer_inputs_refused(tmp_path, monkeypatch, field):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet["sessions"][0][field] = "forbidden historical input"
    with pytest.raises(reg.RegistrationError, match="FORBIDDEN_PRODUCER_INPUT"):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("kind", ["source_swap", "count", "map", "manifest_order", "extra_source", "source_status"])
def test_g3_fixed_source_manifest_cannot_be_swapped_or_shortened(tmp_path, monkeypatch, kind):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    if kind == "source_swap": packet["sessions"][0]["source_receipt"] = packet["sessions"][1]["diagnostic_ref"]
    elif kind == "count": packet["sessions"][0]["native_frames"] -= 1
    elif kind == "map": packet["sessions"][0]["frame_map_ref"] = packet["sessions"][1]["diagnostic_ref"]
    elif kind == "manifest_order": packet["sessions"].reverse()
    elif kind == "extra_source": packet["sessions"].append(copy.deepcopy(packet["sessions"][-1]))
    else: packet["sessions"][1]["source_status"] = "READY"
    with pytest.raises(reg.RegistrationError):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


def test_g3_no_packet_refused_before_algorithm_import(tmp_path, monkeypatch):
    g3_fixture(tmp_path, monkeypatch)
    def denied(*a, **k):
        raise reg.RegistrationError("NO_ACTIVE_PACKET")
    monkeypatch.setattr(reg, "_active_packet", denied)
    with pytest.raises(reg.RegistrationError, match="NO_ACTIVE_PACKET"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


def test_g3_native_length_api_ceiling_and_fresh_deadline():
    assert [reg.g3_api_bound(n) for n in (1, 2, 3, 1400)] == [23, 40, 56, 56]
    started = datetime.now(timezone.utc)
    assert runner.effective_deadline({"task_id": reg.G3_PREFIX + "a01", "budgets": {"wall_seconds_max": 28800}}, started) == started + timedelta(hours=8)


def test_g3_actual_worker_skips_blocked_rows_and_retains_native_receipts(source_worker_batch, tmp_path, monkeypatch):
    packet, _ = g3_fixture(tmp_path, monkeypatch, source_fixture=source_worker_batch)
    result = worker.run(packet)
    assert result["schema"] == reg.G3_RESULT_SCHEMA
    assert result["completed_mask_sessions"] == result["ready_source_sessions"] == 1
    assert result["blocked_source_sessions"] == 10 and result["planned_sessions"] == 11
    assert result["execution_diagnostic_complete"] is True
    assert result["model_call_accounting"]["attempted_api_calls"] == len(source_worker_batch.sam.calls) == 6
    for row in result["sessions"][1:]:
        assert row["execution"] == "BLOCKED_SOURCE" and row["model_api_calls_attempted"] == 0
        assert not (Path(packet["output_dir"]) / "sessions" / row["session_key"] / "Mask").exists()
    assert runner.validate_model_result(packet, decoder=lambda p: 2)["path"].endswith("MODEL_RESULT.json")


@pytest.mark.parametrize("kind", ["blocked_calls", "fake_mask", "blocked_count", "scope", "short_decode"])
def test_g3_actual_outputs_cannot_claim_blocked_success(source_worker_batch, tmp_path, monkeypatch, kind):
    packet, _ = g3_fixture(tmp_path, monkeypatch, source_fixture=source_worker_batch)
    result = worker.run(packet)
    if kind == "blocked_calls": result["sessions"][1]["model_api_calls_attempted"] = 1
    elif kind == "fake_mask": result["sessions"][1]["mask"] = result["sessions"][0]["mask"]
    elif kind == "blocked_count": result["blocked_source_sessions"] = 0
    elif kind == "scope": result["experiment_scope"] = reg.POST_TEXT_RETENTION_SCOPE
    # The real CPU producer sealed MODEL_RESULT read-only. Preserve it; the
    # negative fixture substitutes the parsed result, not production bytes.
    original_load = runner.load_json
    model_result_path = Path(packet["output_dir"]) / "MODEL_RESULT.json"
    monkeypatch.setattr(runner, "load_json", lambda path: copy.deepcopy(result)
        if Path(path) == model_result_path else original_load(path))
    with pytest.raises(ValueError):
        runner.validate_model_result(packet, decoder=lambda p: 1 if kind == "short_decode" else 2)


def test_g3_all_blocked_constructs_no_model_and_not_execution_complete(source_worker_batch, tmp_path, monkeypatch):
    from chaoyang.inpainting_candidate import model_stages
    packet, _ = g3_fixture(tmp_path, monkeypatch, source_fixture=source_worker_batch)
    row = packet["sessions"][0]
    diagnostic = execution.put(tmp_path / "ALL_BLOCKED.json", {"status": "BLOCKED_SOURCE"})
    packet["sessions"][0] = {k: v for k, v in row.items() if k in
        {"session_key", "session_id", "raw_session_dir", "native_frames"}}
    packet["sessions"][0].update(source_status="BLOCKED", diagnostic_ref=diagnostic, failure_code="BLOCKED_SOURCE")
    packet["budgets"]["model_calls_max"] = 0
    monkeypatch.setattr(model_stages, "load_frozen_models", lambda *a, **k: pytest.fail("all blocked constructed model"))
    result = worker.run(packet)
    assert result["completed_mask_sessions"] == 0 and result["blocked_source_sessions"] == 11
    assert result["model_call_accounting"]["attempted_api_calls"] == 0
    assert result["execution_diagnostic_complete"] is False


@pytest.mark.parametrize("domain", ["weights", "model_weights", "calibration"])
def test_g3_weights_and_eye_qualification_cannot_be_borrowed_or_changed(tmp_path, monkeypatch, domain):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet[domain] = {"changed": True}
    with pytest.raises(reg.RegistrationError):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


def test_g3_resigned_manifest_cannot_swap_original_0918_directory(tmp_path, monkeypatch):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    packet["sessions"][3]["raw_session_dir"] = packet["sessions"][4]["raw_session_dir"]
    index_path = Path(packet["inputs"]["source_index"]["path"])
    manifest = json.loads(index_path.read_text())
    manifest["sessions"][3]["raw_session_dir"] = packet["sessions"][3]["raw_session_dir"]
    packet["inputs"]["source_index"] = execution.put(index_path, manifest)
    with pytest.raises(reg.RegistrationError, match="0918_SOURCE_SWAPPED"):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


def test_g3_blocked_diagnostic_cannot_pretend_sealed_source(tmp_path, monkeypatch):
    packet, _ = g3_fixture(tmp_path, monkeypatch)
    row = packet["sessions"][1]
    fake_sealed = execution.put(tmp_path / "FAKE_SOURCE.json", {"stage": "Source", "status": "SEALED_COMPLETE"})
    row["diagnostic_ref"] = fake_sealed
    index_path = Path(packet["inputs"]["source_index"]["path"])
    manifest = json.loads(index_path.read_text())
    manifest["sessions"][1]["diagnostic_ref"] = fake_sealed
    packet["inputs"]["source_index"] = execution.put(index_path, manifest)
    with pytest.raises(reg.RegistrationError, match="CANNOT_FAKE_SEALED"):
        reg.validate_g3_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["resume_context", "original_g1_deadline_at"])
def test_g3_deadline_not_restarted_g1(field):
    with pytest.raises(ValueError, match="G3_FRESH_NOT_OLD_TIMEBOX"):
        runner.effective_deadline({"task_id": reg.G3_PREFIX + "a01", field: None,
            "budgets": {"wall_seconds_max": 28800}}, datetime.now(timezone.utc))

