"""Bounded successor front-end tests; no real model, inference or GPU."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from chaoyang.humanego_ict import production_resume_cli as cli
from chaoyang.humanego_ict.production_training_cli import make_training_config
from chaoyang.humanego_ict.production_resume_gate import PARENT_INPUT_ROLES


def config(world=1):
    ref = {"path": "/synthetic/not-production", "bytes": 1, "sha256": "a" * 64}
    value = make_training_config(3000, manifest_ref=ref, pretrained_ref=ref,
        pretrained_config_ref=ref, world_size=world)
    value.update(schema_version="pick-place-humanego-ict-production-resume-config-v1",
        initialization="RESUME_COMPLETE_TRAINING_STATE",
        resume_parent_bindings={name: dict(ref) for name in PARENT_INPUT_ROLES})
    return value


def args(tmp_path, value):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(value))
    return ["train", "--task-packet", str(tmp_path / "packet.json"), "--manifest", str(tmp_path / "manifest.json"),
        "--config", str(cfg), "--checkpoint", str(tmp_path / "old.pt"),
        "--output-dir", str(tmp_path / "output"), "--branch", "raw"]


def forbid_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Torch must not be imported before successful task admission")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_unregistered_resume_rejected_before_torch(tmp_path, monkeypatch, capsys):
    argv = args(tmp_path, config())
    monkeypatch.setenv("WORLD_SIZE", "1")
    forbid_torch(monkeypatch)

    def reject(*a, **k):
        raise ValueError("UNREGISTERED_SYNTHETIC_TASK")

    monkeypatch.setattr(cli, "authorize_production_resume", reject)
    assert cli.main(argv) == 2
    value = json.loads(capsys.readouterr().err)
    assert value["reason"] == "UNREGISTERED_SYNTHETIC_TASK"
    assert not value["model_loaded"] and not value["training_started"]


def test_fresh_configuration_cannot_be_silently_resumed(tmp_path, monkeypatch, capsys):
    value = config()
    value["initialization"] = "FRESH_SHARED_PRETRAINED_NEW_IO"
    forbid_torch(monkeypatch)
    assert cli.main(args(tmp_path, value)) == 2
    assert "EXPLICIT_COMPLETE_STATE_RESUME_CONFIG_REQUIRED" in capsys.readouterr().err


def test_eight_gpu_parent_change_not_called_exact_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(cli, "authorize_production_resume", lambda *a, **k: {})
    forbid_torch(monkeypatch)
    assert cli.main(args(tmp_path, config(8))) == 2
    assert "EXACT_SINGLE_H20_PARENT_TOPOLOGY_REQUIRED" in capsys.readouterr().err


def test_final_result_reports_only_new_real_updates():
    packet = {"branch": "raw", "task_id": "synthetic-successor", "run_signature": "a" * 64,
        "inputs": {"resume_checkpoint": {"sha256": "b" * 64}}}
    before = {"optimizer_updates": 4377}
    result = cli.final_result(before, packet, 4376, {"sha256": "c" * 64}, {})
    assert before == {"optimizer_updates": 4377}
    assert result["parent_optimizer_updates"] == 4376 and result["new_optimizer_updates"] == 1
    assert not result["control_ground_truth"] and not result["physical_deployment_authorized"]
    assert result["initialization"] == "RESUME_COMPLETE_TRAINING_STATE"
    for updates in (0, 4375, 4376):
        with pytest.raises(ValueError, match="NEW_REAL_OPTIMIZER"):
            cli.final_result({"optimizer_updates": updates}, packet, 4376, {}, {})


def test_signatures_preserve_source_and_bind_successor():
    inputs = {key: {"sha256": str(i) * 64} for i, key in enumerate(
        ("production_manifest", "training_config", "pretrained", "pretrained_config", "storage_capacity_observation"), 1)}
    packet = {"inputs": inputs, "run_signature": "a" * 64, "branch": "robotized", "runtime_code": ["synthetic"]}
    source = {"prepared_pair_ref": inputs["production_manifest"]}
    before = copy.deepcopy(packet)
    signature = cli.lineage_signatures(packet, source)
    assert packet == before
    assert signature["data"] == source["prepared_pair_ref"] and signature["branch"] == "robotized"
    assert signature["run_signature"] == packet["run_signature"] and not signature["control_ground_truth"]


def runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts/humanego_ict/production_resume_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_synthetic_test_resume_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wall", [1501, 43200])
def test_finite_publisher_only_changes_declared_resume_metadata_and_wall(wall):
    runner = runner_module()
    parent = config()
    parent.pop("resume_parent_bindings")
    parent.update(schema_version="pick-place-humanego-ict-production-training-config-v1",
        initialization="FRESH_SHARED_PRETRAINED_NEW_IO")
    before = copy.deepcopy(parent)
    refs = config()["resume_parent_bindings"]
    successor = runner.make_resume_config(parent, refs, wall)
    assert parent == before
    assert successor["initialization"] == "RESUME_COMPLETE_TRAINING_STATE"
    assert successor["training"]["wall_seconds_max"] == wall - 1500
    projected = copy.deepcopy(successor)
    projected.pop("resume_parent_bindings")
    projected["initialization"] = before["initialization"]
    projected["schema_version"] = before["schema_version"]
    projected["training"]["wall_seconds_max"] = before["training"]["wall_seconds_max"]
    assert projected == before


@pytest.mark.parametrize("wall", [True, 0, 1500, 43201])
def test_publisher_rejects_unbounded_or_export_starved_config(wall):
    with pytest.raises(ValueError, match="FINITE_RESUME_AND_EXPORT"):
        runner_module().make_resume_config(config(), {}, wall)


@pytest.mark.parametrize("raw_exit, expected_calls", [(0, 2), (2, 1)])
def test_explicit_pair_only_registers_next_branch_after_normal_seal(monkeypatch, raw_exit, expected_calls):
    runner = runner_module()
    calls = []

    def registered(argv):
        calls.append(list(argv))
        return raw_exit if len(calls) == 1 else 0

    monkeypatch.setattr(runner, "register_main", registered)
    result = runner.main(["--continue-pair", "--raw-parent-task-packet", "/synthetic/raw.json",
        "--robotized-parent-task-packet", "/synthetic/robotized.json", "--wall-seconds", "43200", "--attempt", "1"])
    assert result == raw_exit
    assert len(calls) == expected_calls
    assert calls[0][calls[0].index("--branch") + 1] == "raw"
    if expected_calls == 2:
        assert calls[1][calls[1].index("--branch") + 1] == "robotized"
        assert calls[1][calls[1].index("--parent-task-packet") + 1] == "/synthetic/robotized.json"
    for argv in calls:
        assert argv[argv.index("--wall-seconds") + 1] == "43200"
