"""Synthetic governance admission only: no GPUs, model or real dataset calls."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from chaoyang.humanego_ict import joint26_runtime as gate
from chaoyang.humanego_ict.joint26_contract import PREPARE, PREPARE_MODE, PREPARE_ROUTE


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return gate.file_ref(path)


@pytest.fixture
def signed(tmp_path, monkeypatch):
    """Fake receipt binds fake artifacts; ancestry is mocked explicitly."""
    root = tmp_path / "synthetic_project"
    root.mkdir()
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "_is_descendant", lambda writer: True)
    lock_path = root / "_run/current/locks/governance.lock"
    lock_path.parent.mkdir(parents=True); lock_path.touch()
    roles = ("source_manifest", "source_map", "profile", "geometry_index", "client_source")
    inputs = {name: dump(root / (name + ".json"), {"synthetic": True}) for name in roles}
    code = dump(root / "synthetic_code.json", {"synthetic": True})
    path = root / "tasks/current/synthetic_joint26/TASK_PACKET.json"
    writer = {"pid": 1, "proc_start_ticks": 2, "fencing_token": "synthetic", "executor_epoch": 1}
    writer_file = root / "writer.json"; dump(writer_file, writer)
    output = root / "synthetic_output"
    arguments = ["--task-packet", str(path)]
    for name in roles:
        arguments += ["--" + name.replace("_", "-"), inputs[name]["path"]]
    arguments += ["--output", str(output)]
    import shlex
    packet = {"task_id": "synthetic_joint26", "execution_mode": PREPARE_MODE,
        "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        "run_signature": "synthetic_signature", "writer": writer, "writer_file": str(writer_file),
        "runtime_code": [code], "inputs": inputs, "output_dir": str(output),
        "commands": [shlex.join(["python", "-m", "chaoyang.cli", "run", PREPARE, *arguments])],
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "budgets": {"gpu_count": 0, "samples_max": 3, "geometry_calls_per_window_max": 1000}}
    contract = {"special_status_contracts": {PREPARE_ROUTE: {"candidate_operation": PREPARE,
        "task_packet_required": True, "physical_deployment_authorized": False, "code_closure": [code]}}}
    state = {"tasks": [{"task_id": packet["task_id"], "status": "RUNNING", "run_signature": packet["run_signature"],
        "pid": 1, "proc_start_ticks": 2}]}
    index = {"task_packets": []}
    receipt_path = root / "docs/governance/CURRENT_STATUS_RECEIPT.json"

    def publish():
        packet_ref = dump(path, packet)
        index["task_packets"] = [{"task_id": packet["task_id"], "execution_allowed": True,
            "packet_sha256": packet_ref["sha256"], "packet_path": str(path.relative_to(root))}]
        refs = {"algorithm_contract": dump(root / "contract.json", contract),
                "task_state": dump(root / "state.json", state),
                "task_packet_index_current": dump(root / "index.json", index)}
        dump(receipt_path, {"files": refs})
    publish()
    return packet, state, contract, inputs, path, output, publish


def admit(signed):
    packet, state, contract, inputs, path, output, publish = signed
    return gate.authorize_joint26_task(path, phase="prepare",
        input_paths={k: v["path"] for k, v in inputs.items()}, output_dir=output)


def test_exact_signed_cpu_prepare(signed):
    assert admit(signed)["task_id"] == "synthetic_joint26"


def test_static_initialization_defaults_and_explicit_computational_budget():
    assert gate.preparation_budgets({}) == {
        "current_settle_steps_max": 60, "geometry_calls_per_window_max": 1000}
    assert gate.preparation_budgets({"current_initialization": {
        "mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY", "settle_steps_max": 300,
        "geometry_calls_per_window_max": 5000}}) == {
            "current_settle_steps_max": 300, "geometry_calls_per_window_max": 5000}


@pytest.mark.parametrize("policy,code", [
    (None, "STATIC_INITIALIZATION_POLICY"),
    ({"mode": "READ_FUTURE_TARGETS"}, "CURRENT_ONLY_STATIC"),
    ({"position_tolerance_m": 0.1}, "STATIC_INITIALIZATION_POLICY"),
    ({"settle_steps_max": True}, "INITIALIZATION_STEPS"),
    ({"settle_steps_max": 0}, "INITIALIZATION_STEPS"),
    ({"settle_steps_max": 301}, "INITIALIZATION_STEPS"),
    ({"settle_steps_max": 30.0}, "INITIALIZATION_STEPS"),
    ({"geometry_calls_per_window_max": True}, "GEOMETRY_CALL_BUDGET"),
    ({"geometry_calls_per_window_max": 0}, "GEOMETRY_CALL_BUDGET"),
    ({"geometry_calls_per_window_max": 5001}, "GEOMETRY_CALL_BUDGET"),
    ({"geometry_calls_per_window_max": 1000.0}, "GEOMETRY_CALL_BUDGET"),
    ({"settle_steps_max": 300, "geometry_calls_per_window_max": 1000}, "GEOMETRY_BUDGET_INSUFFICIENT"),
])
def test_static_initialization_budget_never_changes_quality_or_reads_future(policy, code):
    with pytest.raises(ValueError, match=code):
        gate.preparation_budgets({"current_initialization": policy})


def test_expanded_static_effort_must_be_bound_in_both_profile_and_packet(signed):
    packet, _, _, inputs, _, _, publish = signed
    profile_path = Path(inputs["profile"]["path"])
    inputs["profile"] = dump(profile_path, {"current_initialization": {
        "mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY", "settle_steps_max": 300,
        "geometry_calls_per_window_max": 5000}})
    publish()
    with pytest.raises(ValueError, match="CPU_LABEL_BUDGET"):
        admit(signed)
    packet["budgets"].update(current_settle_steps_max=300, geometry_calls_per_window_max=5000)
    publish()
    assert admit(signed)["budgets"]["current_settle_steps_max"] == 300
    packet["budgets"]["current_settle_steps_max"] = True
    publish()
    with pytest.raises(ValueError, match="CPU_LABEL_BUDGET"):
        admit(signed)


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.update(execution_mode="CPU_DEVELOPMENT_ONLY"), "REAL_TASK_REQUIRED"),
    (lambda p: p.update(deadline_at="2020-01-01T00:00:00+00:00"), "DEADLINE_EXPIRED"),
    (lambda p: p.update(commands=[]), "COMMAND_NOT_PINNED"),
    (lambda p: p.update(physical_deployment_authorized=True), "RESEARCH_FLAGS"),
    (lambda p: p["budgets"].update(geometry_calls_per_window_max=2000), "CPU_LABEL_BUDGET"),
])
def test_reject_packet_deviation(signed, mutation, code):
    mutation(signed[0]); signed[-1]()
    with pytest.raises(ValueError, match=code):
        admit(signed)


def static_mesh_profile():
    return {"collision_validation": "FROZEN_STL_MESH_AUDIT_V1",
            "current_initialization": {"mode": "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY",
                "settle_steps_max": 300, "geometry_calls_per_window_max": 5000,
                "static_position_cost": 100.0,
                "static_orientation_cost": 11.459155902616466, "static_qp_damping": 1e-6}}


def test_static_mesh_options_are_explicit_and_original_defaults_remain():
    assert gate.preparation_options({}) == {
        "current_initialization_mode": "STATIC_NUMERICAL_CURRENT_TARGET_ONLY",
        "collision_validation": "ORIGINAL_CAPSULE_V1",
        "training_admission_policy": "STRICT_MODEL_NUMERICS_V1"}
    assert gate.preparation_options(static_mesh_profile()) == {
        "current_initialization_mode": "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY",
        "collision_validation": "FROZEN_STL_MESH_AUDIT_V1",
        "training_admission_policy": "STRICT_MODEL_NUMERICS_V1"}
    assert gate.preparation_budgets(static_mesh_profile()) == {
        "current_settle_steps_max": 300, "geometry_calls_per_window_max": 5000}


@pytest.mark.parametrize("key,value", [
    ("static_position_cost", True), ("static_position_cost", 99.0),
    ("static_orientation_cost", 0.12), ("static_qp_damping", 0.1),
])
def test_static_parameters_cannot_change_after_recipe_freeze(key, value):
    profile = static_mesh_profile()
    profile["current_initialization"][key] = value
    with pytest.raises(ValueError, match="FROZEN_STATIC_TASK_PARAMETERS"):
        gate.preparation_budgets(profile)


def test_static_mode_requires_all_parameters():
    profile = static_mesh_profile()
    del profile["current_initialization"]["static_qp_damping"]
    with pytest.raises(ValueError, match="FROZEN_STATIC_TASK_PARAMETERS"):
        gate.preparation_options(profile)


@pytest.mark.parametrize("mode", [None, True, [], "IGNORE_COLLISION"])
def test_unknown_collision_representation_is_rejected(mode):
    with pytest.raises(ValueError, match="COLLISION_REPRESENTATION"):
        gate.preparation_options({"collision_validation": mode})


def test_static_mesh_options_must_match_signed_task(signed):
    packet, _, _, inputs, _, _, publish = signed
    inputs["profile"] = dump(Path(inputs["profile"]["path"]), static_mesh_profile())
    packet["budgets"].update(current_settle_steps_max=300, geometry_calls_per_window_max=5000)
    publish()
    with pytest.raises(ValueError, match="SIGNED_PREPARATION_OPTIONS_DRIFT"):
        admit(signed)
    packet["preparation_options"] = gate.preparation_options(static_mesh_profile())
    publish()
    assert admit(signed)["preparation_options"] == packet["preparation_options"]
    packet["preparation_options"]["collision_validation"] = "ORIGINAL_CAPSULE_V1"
    publish()
    with pytest.raises(ValueError, match="SIGNED_PREPARATION_OPTIONS_DRIFT"):
        admit(signed)


def ego_profile():
    profile = static_mesh_profile()
    profile["training_admission_policy"] = "EGO_DERIVED_STRUCTURAL_V1"
    profile["collision_validation"] = "ORIGINAL_CAPSULE_V1"
    profile["current_initialization"]["settle_steps_max"] = 60
    return profile


def test_ego_admission_is_explicit_without_mesh_or_fk_quality_gate():
    assert gate.preparation_options(ego_profile()) == {
        "current_initialization_mode": "STATIC_FRAME_TASK_CURRENT_TARGET_ONLY",
        "collision_validation": "ORIGINAL_CAPSULE_V1",
        "training_admission_policy": "EGO_DERIVED_STRUCTURAL_V1"}
    assert gate.preparation_budgets(ego_profile()) == {
        "current_settle_steps_max": 60, "geometry_calls_per_window_max": 5000}


@pytest.mark.parametrize("policy", [None, True, [], "EGO_IGNORE_ALL_INPUTS"])
def test_unknown_training_admission_policy_rejected(policy):
    with pytest.raises(ValueError, match="TRAINING_ADMISSION_POLICY"):
        gate.preparation_options({"training_admission_policy": policy})


def test_ego_cannot_make_mesh_an_implicit_training_prerequisite():
    profile = ego_profile()
    profile["collision_validation"] = "FROZEN_STL_MESH_AUDIT_V1"
    with pytest.raises(ValueError, match="EGO_FIXED_RETARGET_RECIPE"):
        gate.preparation_options(profile)


def test_ego_computation_budget_does_not_change_quality_thresholds():
    profile = ego_profile()
    profile["current_initialization"]["settle_steps_max"] = 300
    with pytest.raises(ValueError, match="EGO_FIXED_CURRENT_COMPUTATION"):
        gate.preparation_budgets(profile)


def test_ego_policy_is_signature_bound_not_enabled_by_missing_field(signed):
    packet, _, _, inputs, _, _, publish = signed
    inputs["profile"] = dump(Path(inputs["profile"]["path"]), ego_profile())
    packet["budgets"].update(current_settle_steps_max=60, geometry_calls_per_window_max=5000)
    publish()
    with pytest.raises(ValueError, match="SIGNED_PREPARATION_OPTIONS_DRIFT"):
        admit(signed)
    packet["preparation_options"] = gate.preparation_options(ego_profile())
    publish()
    assert admit(signed)["preparation_options"]["training_admission_policy"] == "EGO_DERIVED_STRUCTURAL_V1"
    del packet["preparation_options"]["training_admission_policy"]
    publish()
    with pytest.raises(ValueError, match="SIGNED_PREPARATION_OPTIONS_DRIFT"):
        admit(signed)


def test_wrong_state_signature(signed):
    signed[1]["tasks"][0]["run_signature"] = "not_owned"; signed[-1]()
    with pytest.raises(ValueError, match="LIVE_OWNED"):
        admit(signed)


def test_missing_route(signed):
    signed[2]["special_status_contracts"] = {}; signed[-1]()
    with pytest.raises(ValueError, match="ROUTE_NOT_REGISTERED"):
        admit(signed)


def test_input_sha_drift(signed):
    Path(signed[3]["profile"]["path"]).write_text('{"synthetic":"changed"}')
    with pytest.raises(ValueError, match="ARTIFACT_SHA_DRIFT"):
        admit(signed)


def test_writer_fence_drift(signed):
    dump(Path(signed[0]["writer_file"]), {"fencing_token": "other"})
    with pytest.raises(ValueError, match="FENCING_TOKEN_DRIFT"):
        admit(signed)


def test_wrong_command_input_path(signed):
    signed[0]["commands"][0] += " --extra ignored"; signed[-1]()
    with pytest.raises(ValueError, match="COMMAND_NOT_PINNED"):
        admit(signed)


def test_symlink_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    target = tmp_path / "target.json"; target.write_text("{}")
    link = tmp_path / "link.json"; link.symlink_to(target)
    with pytest.raises(ValueError, match="SYMLINK"):
        gate._ordinary(link)


def test_runner_default_config_and_topology():
    path = Path(__file__).resolve().parents[2] / "scripts/humanego_ict/joint26_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_synthetic_joint26_runner", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = module.default_config(8, 43200)
    assert config["training"]["micro_batch"] == 8
    assert config["training"]["max_updates"] == 20000
    assert config["model"]["horizon"] == 30 and config["model"]["action_dim"] == 26
    assert module.default_config(1, 43200)["training"]["micro_batch"] == 64
    assert module.default_config(8, 43200, 10000)["training"]["max_updates"] == 10000
    with pytest.raises(ValueError, match="GLOBAL_BATCH"):
        module.default_config(3, 43200)


@pytest.mark.parametrize("steps", [0, -1, True, 2.5])
def test_runner_rejects_invalid_requested_steps(steps):
    path = Path(__file__).resolve().parents[2] / "scripts/humanego_ict/joint26_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_joint26_budget_runner", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="POSITIVE_INTEGER_STEPS"):
        module.default_config(8, 43200, steps)


def test_runner_cpu_config_resolution_and_explicit_budget_conflict(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/humanego_ict/joint26_runner_v1.py"
    spec = importlib.util.spec_from_file_location("_joint26_config_runner", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = SimpleNamespace(steps=1234, config=None, nproc=8, wall_seconds=43200)
    cfg, document = module.resolve_training_config(args)
    assert cfg.max_updates == document["training"]["max_updates"] == 1234
    config_path = tmp_path / "configured_training.json"
    config_path.write_text(json.dumps(document))
    args.config = config_path
    args.steps = None
    assert module.resolve_training_config(args)[0].max_updates == 1234
    args.steps = 5000
    with pytest.raises(ValueError, match="CLI_STEPS_CONFIG_BUDGET_MISMATCH"):
        module.resolve_training_config(args)
    args.steps = 1234
    assert module.resolve_training_config(args)[0].max_updates == 1234


@pytest.fixture
def signed_training(signed):
    import shlex
    import socket
    from chaoyang.humanego_ict.joint26_contract import EXECUTION_MODE, ROUTE, TRAIN
    packet, state, contract, _, path, output, publish = signed
    root = path.parents[3]
    inputs = {"paired_manifest": dump(root / "paired.json", {"synthetic": True}),
              "training_config": dump(root / "config.json", {
                  "schema": "humanego-ict-joint26-training-config-v1",
                  "training": {"max_updates": 1234}})}
    closure = packet["runtime_code"]
    contract["special_status_contracts"] = {ROUTE: {"candidate_operation": TRAIN,
        "task_packet_required": True, "physical_deployment_authorized": False,
        "code_closure": closure}}
    arguments = ["train", "--task-packet", str(path), "--manifest", inputs["paired_manifest"]["path"],
        "--config", inputs["training_config"]["path"], "--output", str(output), "--branch", "raw"]
    lease_path = root / "GPU_LEASE.json"
    writer = packet["writer"]
    dump(lease_path, {"status": "ACQUIRED", "task_id": packet["task_id"],
        "hostname": socket.gethostname(), "pid": writer["pid"],
        "process_startticks": writer["proc_start_ticks"], "fencing_token": writer["fencing_token"],
        "executor_epoch": writer["executor_epoch"], "gpu_id": 0,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()})
    packet.update(execution_mode=EXECUTION_MODE, inputs=inputs, branch="raw", allocated_gpu_ids=[0, 1],
        gpu_lease_verified=True, gpu_lease_path=str(lease_path),
        budgets={"gpu_count": 2, "optimizer_updates_max": 1234},
        commands=[shlex.join(["python", "-m", "chaoyang.cli", "run", TRAIN, *arguments])])
    publish()
    return packet, inputs, path, output, publish


def admit_training(signed_training):
    packet, inputs, path, output, publish = signed_training
    return gate.authorize_joint26_training(path, inputs["paired_manifest"]["path"],
        inputs["training_config"]["path"], output, "raw", 2)


def test_signed_non_20k_training_budget_is_admitted(signed_training):
    assert admit_training(signed_training)["budgets"]["optimizer_updates_max"] == 1234


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, 20000])
def test_signed_training_budget_drift_or_invalid_type_rejected(signed_training, budget):
    signed_training[0]["budgets"]["optimizer_updates_max"] = budget
    signed_training[-1]()
    with pytest.raises(ValueError, match="UPDATE_OR_DDP_BUDGET_DRIFT"):
        admit_training(signed_training)
