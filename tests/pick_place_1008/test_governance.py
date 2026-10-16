import json

import pytest

from chaoyang.governance import execute_pick_place_1008_v1 as executor
from chaoyang.governance.exact78_visual_selection_v1 import (
    PICK_PLACE_OPERATIONS, TAKEOVER_OPERATIONS, filter_selected_operations,
)


def contract():
    return {"special_status_contracts": {
        "exact78_visual_baseline_v1": {"execution_policy": {
            "enabled": True, "baseline_id": "EXACT78_VISUAL_BASELINE_V1",
            "scope": "chips_cards_tracker_0901_0902",
            "allowed_operations": sorted(TAKEOVER_OPERATIONS)}},
        "pick_place_1008_candidate_v1": {
            "execution_status": "CANDIDATE_FENCED_PICK_PLACE_TASK_PACKET_ONLY",
            "scope": "pick_place_1008", "plan_revision": "PICK_PLACE_1008_RAW_TO_ROBOT_V1",
            "candidate_operations": sorted(PICK_PLACE_OPERATIONS), "task_packet_required": True,
            "training_steps_authorized": 0, "current_runnable_operation": None,
            "code_closure": [{"path": "/project/code.py", "sha256": "a" * 64, "bytes": 1}]}}}


def test_selection_adds_only_two_fenced_candidates():
    retired = "run_exact78_general_p1_action_canary_v10"
    supplied = set(TAKEOVER_OPERATIONS | PICK_PLACE_OPERATIONS) | {retired, "run_robot100_v16"}
    assert filter_selected_operations(supplied, contract()) == set(
        TAKEOVER_OPERATIONS | PICK_PLACE_OPERATIONS) | {"run_robot100_v16"}


@pytest.mark.parametrize("override", [
    {"task_packet_required": False}, {"scope": "0916"},
    {"candidate_operations": ["run_exact78_takeover_baseline_v1"]},
    {"current_runnable_operation": "run_exact78_pick_place_raw_to_robot_candidate_v1"},
    {"code_closure": []}, {"plan_revision": "OLD_PLAN"},
])
def test_malformed_candidate_contract_fails_closed(override):
    value = contract()
    value["special_status_contracts"]["pick_place_1008_candidate_v1"].update(override)
    assert filter_selected_operations(set(TAKEOVER_OPERATIONS | PICK_PLACE_OPERATIONS),
                                      value) == set(TAKEOVER_OPERATIONS)


def test_candidate_absent_does_not_reopen_legacy_or_admit_without_contract():
    value = contract()
    del value["special_status_contracts"]["pick_place_1008_candidate_v1"]
    assert filter_selected_operations(set(TAKEOVER_OPERATIONS | PICK_PLACE_OPERATIONS),
                                      value) == set(TAKEOVER_OPERATIONS)
    assert filter_selected_operations(set(PICK_PLACE_OPERATIONS), {}) == set()


def test_executor_spec_rejects_unknown_operation_and_owned_output(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "SPEC_ROOT", tmp_path)
    spec = {key: [] for key in executor.SPEC_FIELDS}
    spec.update(task_id="exact78_pick_place_test", operation="run_exact78_pick_place_legacy",
                args=[], budgets={}, read_set=[], required_outputs=[])
    path = tmp_path / "exact78_pick_place_test.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="OPERATION_NOT_ALLOWED"):
        executor.load_spec(path)
    spec["operation"] = "run_exact78_pick_place_audit_candidate_v1"
    spec["args"] = ["--output-root=/outside"]
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match="OUTPUT_ROOT_IS_EXECUTOR_OWNED"):
        executor.load_spec(path)


def test_gpu_resource_exit_needs_explicit_prealgorithm_evidence(tmp_path):
    stage = tmp_path / "candidate"
    stage.mkdir()
    paths = {"stage": stage}
    assert not executor._resource_wait_evidence(paths, 75, None)
    receipt = stage / "RESOURCE_WAIT.json"
    receipt.write_text(json.dumps({"status": "WAIT_GPU_RESOURCE", "algorithm_started": True}))
    assert not executor._resource_wait_evidence(paths, 75, None)
    receipt.write_text(json.dumps({"status": "WAIT_GPU_RESOURCE", "algorithm_started": False}))
    assert executor._resource_wait_evidence(paths, 75, None)
    assert not executor._resource_wait_evidence(paths, 1, None)


def test_gpu_wait_status_restore(monkeypatch):
    reasons = iter(["CENTRAL_GPU_LEASE_UNAVAILABLE", None])
    status = []
    monkeypatch.setattr(executor, "_gpu_wait_reason", lambda packet: next(reasons))
    monkeypatch.setattr(executor, "_resource_status",
                        lambda packet, state, reason: status.append(state))
    monkeypatch.setattr(executor.time, "sleep", lambda duration: None)
    executor.wait_for_gpu_resource({}, executor.time.monotonic() + 20)
    assert status == ["WAIT_GPU_RESOURCE", "RUNNING"]


def test_recovery_refuses_still_live_fenced_worker(tmp_path, monkeypatch):
    packet = {"task_id": "exact78_pick_place_test", "run_signature": "signed",
              "commands": ["PYTHONPATH=src python -B -m chaoyang.cli run op"],
              "writer": {"pid": 1, "proc_start_ticks": 3}}
    worker = {"task_id": packet["task_id"], "run_signature": "signed", "pid": 100,
              "proc_start_ticks": 20, "command": executor._packet_command_tokens(packet),
              "packet_command": packet["commands"][0], "writer": packet["writer"]}
    (tmp_path / "WORKER.json").write_text(json.dumps(worker))
    monkeypatch.setattr(executor, "process_identity", lambda pid: {"alive": True, "start_ticks": 20})
    with pytest.raises(RuntimeError, match="WORKER_ALIVE"):
        executor._verify_recovery_worker(packet, {"output": tmp_path})
    monkeypatch.setattr(executor, "process_identity", lambda pid: {"alive": False, "start_ticks": None})
    assert executor._verify_recovery_worker(packet, {"output": tmp_path})["original_worker_stopped"]


def test_recovery_keeps_existing_required_index_ref_and_missing_outputs(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    index = candidate / "INDEX.json"
    index.write_text('{"sessions":[]}')
    outputs, errors = executor._collect_required_outputs(
        {"required_outputs": ["candidate/INDEX.json", "candidate/RESULT.json"]}, {"output": tmp_path})
    assert outputs["candidate/INDEX.json"]["sha256"] == executor.artifact_ref(index)["sha256"]
    assert errors == ["PICK_PLACE_REQUIRED_OUTPUT_MISSING:candidate/RESULT.json"]


def test_worker_journal_binds_frozen_command_and_rejects_missing_evidence(tmp_path, monkeypatch):
    packet = {"task_id": "exact78_pick_place_journal", "run_signature": "signed",
              "commands": ["PYTHONPATH=src python -B -m chaoyang.cli run operation"],
              "writer": {"pid": 1, "proc_start_ticks": 2}}
    monkeypatch.setattr(executor, "_paths", lambda task_id: {"output": tmp_path})
    command = executor._packet_command_tokens(packet)
    executor._write_worker_identity(packet, 99, 100, command)
    value = json.loads((tmp_path / "WORKER.json").read_text())
    assert value["pid"] == 99 and value["proc_start_ticks"] == 100
    assert value["command"] == command and value["run_signature"] == "signed"
    with pytest.raises(RuntimeError, match="ALREADY_EXISTS"):
        executor._write_worker_identity(packet, 99, 100, command)
    value["run_signature"] = "changed"
    (tmp_path / "WORKER.json").write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="IDENTITY_DRIFT"):
        executor._verify_recovery_worker(packet, {"output": tmp_path})
    (tmp_path / "WORKER.json").unlink()
    (tmp_path / "worker.log").write_text("")
    with pytest.raises(RuntimeError, match="IDENTITY_UNPROVEN"):
        executor._verify_recovery_worker(packet, {"output": tmp_path})
