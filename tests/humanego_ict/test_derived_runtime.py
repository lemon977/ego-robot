"""Admission must precede models and must not weaken the original route."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

from chaoyang.humanego_ict import derived_runtime as runtime
from chaoyang.humanego_ict.derived_contract import INFER, PREPARE
from chaoyang.humanego_ict.runtime_gate import file_ref


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_ref(path)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    root = tmp_path
    task_id = "bounded-derived-cpu-test"
    packet_path = root / "tasks/current" / task_id / "TASK_PACKET.json"
    request = root / "inputs/request.json"
    request_ref = save(request, {"schema": "fixture-only"})
    output = root / "_run/candidate"
    closure = [save(root / "frozen-code.json", {"immutable": True})]
    writer = {"pid": 100, "proc_start_ticks": 123, "executor_epoch": 1,
              "fencing_token": "test-nonce", "scope": task_id}
    writer_file = output / "WRITER.json"
    save(writer_file, writer)
    args = ["--task-packet", str(packet_path), "--request", str(request),
            "--output-dir", str(output)]
    packet = {"task_id": task_id, "execution_mode": "HUMANEGO_ICT_DERIVED_LABEL_PREPARATION",
              "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
              "writer": writer, "writer_file": str(writer_file), "run_signature": "signature",
              "inputs": {"request": request_ref}, "output_dir": str(output),
              "runtime_code": closure,
              "commands": ["python -m chaoyang.cli run " + PREPARE + " " + " ".join(args)]}
    save(packet_path, packet)
    index_path = root / "tasks/current/INDEX.json"
    state_path = root / "docs/governance/state.json"
    contract_path = root / "docs/governance/contract.json"
    save(contract_path, {"special_status_contracts": {runtime.ROUTE: {
        "candidate_operations": [PREPARE, INFER], "code_closure": closure,
        "task_packet_required": True, "control_ground_truth": False,
        "physical_deployment_authorized": False}}})
    state = {"tasks": [{"task_id": task_id, "status": "RUNNING", "run_signature": "signature",
                        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"]}]}
    save(state_path, state)

    def bind():
        save(index_path, {"task_packets": [{"task_id": task_id, "execution_allowed": True,
                                           "packet_path": str(packet_path.relative_to(root)),
                                           "packet_sha256": file_ref(packet_path)["sha256"]}]})
        save(root / "docs/governance/CURRENT_STATUS_RECEIPT.json", {"files": {
            "algorithm_contract": file_ref(contract_path), "task_state": file_ref(state_path),
            "task_packet_index_current": file_ref(index_path)}})

    bind()
    monkeypatch.setattr(runtime, "_is_descendant", lambda _: True)
    return root, packet_path, request, output, args, packet, bind


def admit(fixture, operation=PREPARE):
    root, packet_path, request, output, args, _, _ = fixture
    return runtime.authorize_operation(root, packet_path, operation, args, request=request, output=output)


def test_exact_finite_cpu_prepare_admitted(admitted):
    assert admit(admitted)["admitted_authorization_expires_at"] > datetime.now(timezone.utc).timestamp()


@pytest.mark.parametrize("field,value,error", [
    ("execution_mode", "CPU_DEVELOPMENT_ONLY", "OPERATION_TASK_MODE"),
    ("commands", [], "COMMAND_NOT_PINNED"),
    ("runtime_code", [], "CODE_CLOSURE"),
    ("deadline_at", "2001-01-01T00:00:00+00:00", "EXPIRED"),
    ("deadline_at", "2001-01-01T00:00:00", "EXPIRED"),
])
def test_packet_rejects_development_and_drift(admitted, field, value, error):
    _, path, _, _, _, packet, bind = admitted
    packet[field] = value
    save(path, packet)
    bind()
    with pytest.raises(ValueError, match=error):
        admit(admitted)


def test_no_writer_bypass(admitted, monkeypatch):
    monkeypatch.setattr(runtime, "_is_descendant", lambda _: False)
    with pytest.raises(ValueError, match="WRITER_ANCESTRY"):
        admit(admitted)


def test_no_request_substitution(admitted):
    admitted[2].write_text('{"schema":"changed"}')
    with pytest.raises(ValueError, match="REQUEST_SHA"):
        admit(admitted)


def test_no_inference_on_cpu_preparation_packet(admitted):
    with pytest.raises(ValueError, match="TASK_MODE"):
        admit(admitted, INFER)


def test_infer_without_lease_rejected_before_consumer(admitted):
    _, path, _, _, args, packet, bind = admitted
    packet["execution_mode"] = "HUMANEGO_ICT_REVO2_OFFLINE_INFERENCE"
    packet["commands"] = ["python -m chaoyang.cli run " + INFER + " " + " ".join(args)]
    save(path, packet)
    bind()
    with pytest.raises(KeyError, match="gpu_lease_path"):
        admit(admitted, INFER)


def test_immutable_output_and_symlink_rejection(tmp_path):
    path = tmp_path / "_run/result.json"
    runtime.write_once(path, {"a": 1}, tmp_path)
    runtime.write_once(path, {"a": 1}, tmp_path)
    with pytest.raises(ValueError, match="IMMUTABLE"):
        runtime.write_once(path, {"a": 2}, tmp_path)
    link = tmp_path / "link"
    link.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="SYMLINK"):
        runtime.write_once(link / "other.json", {}, tmp_path)


def test_missing_packet_does_not_import_model(tmp_path):
    was_imported = "chaoyang.humanego_ict.model" in sys.modules
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.authorize_operation(tmp_path, tmp_path / "missing.json", PREPARE, [],
                                    request=tmp_path / "request.json", output=tmp_path / "_run")
    assert ("chaoyang.humanego_ict.model" in sys.modules) == was_imported


def test_model_reservation_is_one_shot_even_without_result(admitted):
    packet = admit(admitted)
    ref = {"path": "test-policy", "sha256": "a" * 64, "bytes": 0}
    runtime._claim_model_invocation(packet, ref, 5)
    with pytest.raises(ValueError, match="ALREADY_RESERVED"):
        runtime._claim_model_invocation(packet, ref, 5)


def test_build_labels_requires_finite_unique_anchor_budget(tmp_path):
    packet = {"task_id": "test", "budgets": {"label_windows_max": 1},
              "admitted_authorization_expires_at": datetime.now(timezone.utc).timestamp() + 60}
    for anchors in ([0, 1], [0, 0], [True], [], [-1]):
        with pytest.raises(ValueError, match="ANCHOR_BUDGET"):
            runtime.execute_preparation_request({"schema": "humanego-ict-derived-preparation-request-v1",
                                                 "mode": "BUILD_LABELS", "anchor_rows": anchors}, packet, tmp_path)
