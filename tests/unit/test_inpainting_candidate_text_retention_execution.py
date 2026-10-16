"""Finite independent text-retention gate/dispatch tests, CPU fake providers only."""
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


entry = _load("_retention_entry_fixtures", "test_inpainting_candidate_entry.py")
execution = _load("_retention_execution_fixtures", "test_inpainting_candidate_model_execution.py")
runner = execution.runner
batch = execution.batch
source_worker_batch = execution.source_worker_batch


def retention_gate_fixture(tmp_path, monkeypatch):
    packet, contract = entry.controller_gate_fixture(tmp_path, monkeypatch)
    parent = tmp_path / "CONTROLLER_A01_PACKET.json"
    parent.write_text(json.dumps(copy.deepcopy(packet)))
    packet.update(task_id="inpainting_post_g1_text_retention_model_a01_20261013",
        experiment_scope=reg.POST_TEXT_RETENTION_SCOPE, phase=reg.POST_TEXT_RETENTION_PHASE)
    packet["inputs"]["parent_recipe_packet"] = file_ref(parent)
    packet["config"].update(candidate_id="POST_G1_TEXT_WEARER_RETENTION_A01",
        algorithm_parameter_change="TEXT_HUMAN_GLOVE_SUPPORT_RETENTION_AT_POINT_REPLACEMENT",
        text_wearer_retention=reg.TEXT_WEARER_RETENTION_RULE)
    packet["budgets"].update(candidates_max=1, gpu_count=1, cpu_threads_per_worker=2,
        model_calls_max=215, training_steps_max=0, wall_seconds_max=7200)
    return packet, contract


def test_retention_exact_scope_fixed_sources_independent_phase(tmp_path, monkeypatch):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    assert reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)
    assert not reg.validate_post_controller_experiment(packet, repo_root=tmp_path)
    assert runner.planned_session_keys(packet) == runner.EXPECTED_SESSION_KEYS
    assert reg.authorize_candidate_operation(reg.VISUAL_OPERATION,
        ["--fixture", "synthetic"], repo_root=tmp_path) is packet


@pytest.mark.parametrize("field,value", [
    ("experiment_scope", reg.POST_CONTROLLER_SCOPE), ("phase", reg.POST_CONTROLLER_PHASE),
    ("phase", "INPAINTING_G1_AUTOMATIC_MASK"), ("model_stages", ["Clean"]),
    ("resume_context", {}), ("original_g1_deadline_at", None),
    ("task_id", "inpainting_generalization_g1_mask_third_candidate"),
])
def test_retention_wrong_or_restarted_task_rejected(tmp_path, monkeypatch, field, value):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet[field] = value
    with pytest.raises(reg.RegistrationError):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [
    ("text_wearer_retention", None), ("text_wearer_retention", "UNION_ALL_MASKS"),
    ("controller_text_prompt", "a virtual reality controller held in a human hand"),
    ("controller_text_prompt", {"001": "VR controller"}), ("manual_seed", "seed.json"),
    ("manual_seed_inputs", 1), ("image_size", [320, 240]), ("task_name", "pick_place"),
])
def test_retention_only_one_rule_parent_prompt_and_recipe(tmp_path, monkeypatch, field, value):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet["config"][field] = value
    with pytest.raises(reg.RegistrationError):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("field,value", [("candidates_max", 2), ("gpu_count", 2),
    ("cpu_threads_per_worker", 3), ("model_calls_max", 214), ("model_calls_max", 216),
    ("model_calls_max", True), ("training_steps_max", 1), ("wall_seconds_max", 7201),
    ("wall_seconds_max", 0), ("model_calls_unit", "NEURAL_FORWARD_PASSES")])
def test_retention_exact_finite_budget(tmp_path, monkeypatch, field, value):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet["budgets"][field] = value
    with pytest.raises(reg.RegistrationError, match="215_API_TWO_HOURS"):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("domain", ["weights", "model_weights", "calibration"])
def test_retention_frozen_domains_cannot_change(tmp_path, monkeypatch, domain):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet[domain] = {"changed": True}
    with pytest.raises(reg.RegistrationError, match="FROZEN_DOMAIN_DRIFT"):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["mask_receipt", "manual_seed", "proposal_arrays", "placement", "frame_limit"])
def test_retention_forbidden_producer_inputs(tmp_path, monkeypatch, field):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet["sessions"][0][field] = "evaluation-only-not-producer"
    with pytest.raises(reg.RegistrationError, match="FRESH_RAW_SOURCE_ONLY"):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


def test_retention_swapped_source_rejected(tmp_path, monkeypatch):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    packet["sessions"][0]["source_receipt"] = packet["sessions"][1]["source_receipt"]
    with pytest.raises(reg.RegistrationError, match="SOURCE_SWAPPED"):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


def test_retention_resigned_short_source_rejected(tmp_path, monkeypatch):
    packet, _ = retention_gate_fixture(tmp_path, monkeypatch)
    source_path = Path(packet["sessions"][0]["source_receipt"]["path"])
    source = json.loads(source_path.read_text())
    source["native_frames"] = 237
    changed = execution.put(source_path, source)
    packet["sessions"][0]["source_receipt"] = changed
    for name in ("source_index", "parent_recipe_packet"):
        path = Path(packet["inputs"][name]["path"])
        metadata = json.loads(path.read_text())
        metadata["sessions"][0]["source_receipt"] = changed
        if name == "source_index":
            metadata["sessions"][0]["native_frames"] = 237
        packet["inputs"][name] = execution.put(path, metadata)
    with pytest.raises(reg.RegistrationError, match="FULL1033"):
        reg.validate_post_text_retention_experiment(packet, repo_root=tmp_path)


def test_retention_no_packet_refused_before_algorithm_import(tmp_path, monkeypatch):
    retention_gate_fixture(tmp_path, monkeypatch)
    def denied(*args):
        raise reg.RegistrationError("NO_ACTIVE_PACKET")
    monkeypatch.setattr(reg, "_active_packet", denied)
    with pytest.raises(reg.RegistrationError, match="NO_ACTIVE_PACKET"):
        reg.authorize_candidate_operation(reg.VISUAL_OPERATION, ["--fixture", "synthetic"], repo_root=tmp_path)


@pytest.mark.parametrize("field", ["resume_context", "original_g1_deadline_at"])
def test_retention_independent_deadline_no_old_box(field):
    spec = {"task_id": "inpainting_post_g1_text_retention_model_a01_20261013",
            "budgets": {"wall_seconds_max": 7200}, field: None}
    with pytest.raises(ValueError, match="NOT_G1_SUCCESSOR"):
        runner.effective_deadline(spec, datetime.now(timezone.utc))


def test_retention_deadline_exact_new_finite_two_hours():
    started = datetime.now(timezone.utc)
    spec = {"task_id": "inpainting_post_g1_text_retention_model_a01_20261013",
            "budgets": {"wall_seconds_max": 7200}}
    assert runner.effective_deadline(spec, started) == started + timedelta(seconds=7200)


def retention_result_fixture(batch, monkeypatch):
    packet, result = execution.post_controller_batch_fixture(batch)
    root = Path(packet["output_dir"])
    parent = execution.put(root / "CONTROLLER_A01_PACKET.json", copy.deepcopy(packet))
    packet.update(task_id="inpainting_post_g1_text_retention_model_a01_20261013",
                  experiment_scope=reg.POST_TEXT_RETENTION_SCOPE)
    packet["inputs"]["parent_recipe_packet"] = parent
    packet["config"]["text_wearer_retention"] = reg.TEXT_WEARER_RETENTION_RULE
    packet["budgets"].update(model_calls_max=215, cpu_threads_per_worker=2, training_steps_max=0)
    result.update(task_id=packet["task_id"], experiment_scope=reg.POST_TEXT_RETENTION_SCOPE)
    result["model_call_accounting"]["maximum"] = 215
    for row in result["sessions"]:
        row["experiment_scope"] = reg.POST_TEXT_RETENTION_SCOPE
        path = Path(row["mask"]["path"])
        mask = json.loads(path.read_text())
        mask["actual_inputs"]["text_wearer_retention"] = reg.TEXT_WEARER_RETENTION_RULE
        mask["outputs"].append(execution.put(path.parent / "SAM_POINT.json",
            {"text_wearer_retention": reg.TEXT_WEARER_RETENTION_RULE, "prepared_proposals": 0}))
        row["mask"] = execution.put(path, mask)
    execution.save_batch(packet, result)
    monkeypatch.setattr(runner, "REPO_ROOT", root.parent)
    return packet, result


def _decoder(path):
    return dict(zip(runner.EXPECTED_SESSION_KEYS, (238, 183, 200, 139, 145, 128)))[path.parent.name]


def test_retention_actual_receipt_and_refinement_binding_no_fabricated_origin_arrays(batch, monkeypatch):
    packet, result = retention_result_fixture(batch, monkeypatch)
    assert runner.validate_model_result(packet, decoder=_decoder)["path"].endswith("MODEL_RESULT.json")
    # No legal wearer proposal is allowed to remain UNKNOWN, without fake origin arrays.
    assert result["completed_mask_sessions"] == 6


@pytest.mark.parametrize("kind", ["scope", "row_scope", "receipt_rule", "refinement_rule", "missing_refinement", "prompt"])
def test_retention_result_cannot_self_report_rule(batch, monkeypatch, kind):
    packet, result = retention_result_fixture(batch, monkeypatch)
    row = result["sessions"][0]
    if kind == "scope":
        result["experiment_scope"] = reg.POST_CONTROLLER_SCOPE
    elif kind == "row_scope":
        row["experiment_scope"] = reg.POST_CONTROLLER_SCOPE
    else:
        path = Path(row["mask"]["path"])
        mask = json.loads(path.read_text())
        if kind == "receipt_rule":
            mask["actual_inputs"]["text_wearer_retention"] = None
        elif kind == "prompt":
            mask["actual_inputs"]["controller_text_prompt"] = "controller"
        elif kind == "missing_refinement":
            mask["outputs"] = [ref for ref in mask["outputs"] if Path(ref["path"]).name != "SAM_POINT.json"]
        else:
            new_ref = execution.put(path.parent / "SAM_POINT.json", {"text_wearer_retention": "MUTATED"})
            mask["outputs"] = [new_ref if Path(ref["path"]).name == "SAM_POINT.json" else ref for ref in mask["outputs"]]
        row["mask"] = execution.put(path, mask)
    execution.save_batch(packet, result)
    with pytest.raises(ValueError, match="RETENTION|PROMPT_OR_SCOPE"):
        runner.validate_model_result(packet, decoder=_decoder)


def test_worker_passes_rule_and_preserves_budget_no_additional_calls(source_worker_batch):
    fixture = source_worker_batch
    fixture.packet["experiment_scope"] = reg.POST_TEXT_RETENTION_SCOPE
    fixture.packet["config"] = {"controller_text_prompt": "VR controller",
        "text_wearer_retention": reg.TEXT_WEARER_RETENTION_RULE}
    result = execution.worker.run(fixture.packet)
    assert result["completed_mask_sessions"] == 6
    assert result["experiment_scope"] == reg.POST_TEXT_RETENTION_SCOPE
    assert result["model_call_accounting"]["attempted_api_calls"] == 36
    for row in result["sessions"]:
        assert row["experiment_scope"] == reg.POST_TEXT_RETENTION_SCOPE
        folder = Path(row["mask"]["path"]).parent
        receipt = json.loads((folder / "RECEIPT.json").read_text())
        point = json.loads((folder / "SAM_POINT.json").read_text())
        text = json.loads((folder / "SAM_TEXT.json").read_text())
        assert receipt["actual_inputs"]["text_wearer_retention"] == reg.TEXT_WEARER_RETENTION_RULE
        assert point["text_wearer_retention"] == reg.TEXT_WEARER_RETENTION_RULE
        assert [row["prompt"] for row in text["calls"] if row["role"] == "controller"] == ["VR controller"]


def test_worker_old_default_has_no_retention_receipt_fields(source_worker_batch):
    fixture = source_worker_batch
    result = execution.worker.run(fixture.packet)
    assert result["model_call_accounting"]["attempted_api_calls"] == 36
    assert "experiment_scope" not in result
    for row in result["sessions"]:
        folder = Path(row["mask"]["path"]).parent
        receipt = json.loads((folder / "RECEIPT.json").read_text())
        assert "text_wearer_retention" not in receipt["actual_inputs"]
        assert "text_wearer_retention" not in json.loads((folder / "SAM_POINT.json").read_text())
