"""Read-only prep and negative tests; no CPU candidate is executed."""

from pathlib import Path

import pytest

from chaoyang.governance import execute_exact78_chips001_first_anchor_display_cpu_v1 as executor
from chaoyang.governance import publish_exact78_chips001_first_anchor_display_cpu_v1 as publisher
from chaoyang.ops import run_exact78_chips001_first_anchor_display_cpu_v1 as operation


def test_sealed_fresh_ik_is_present_and_bound_to_hawor():
    value = operation.check_only()
    assert value["status"] == "PREP_ONLY_PINNED_FRESH_IK_AVAILABLE_NOT_EXECUTED"
    assert value["frame_count"] == 799
    assert value["frame_side_rows"] == 1598
    assert value["hawor_prediction_rows"] == 1598
    assert value["fresh_v2_ik_states_sha256"] == operation.PINNED["states"][1]
    assert value["old_robot_q_action_input"] is False


def test_only_structural_v2_ik_is_an_arm_q_input():
    values = [relative for relative, _ in operation.PINNED.values()]
    assert sum(path.endswith("ROBOT_CAMERA_DIAGNOSTIC_STATES.npz") for path in values) == 1
    assert all("_v1/attempts" not in path for path in values)
    assert all("0915" not in path and "0916" not in path for path in values)


def test_pinned_sha_drift_fails_before_array_read(monkeypatch):
    relative, _ = operation.PINNED["states"]
    monkeypatch.setitem(operation.PINNED, "states", (relative, "0" * 64))
    with pytest.raises(RuntimeError, match="PINNED_INPUT_SHA_DRIFT:states"):
        operation.check_only()


def test_old_robot_consumption_claim_fails_closed(monkeypatch):
    original = operation._json
    target = str(operation.ROOT / operation.PINNED["v2_candidate"][0])

    def altered(ref):
        value = original(ref)
        if ref["path"] == target:
            value = {**value, "old_robot_state_consumed": True}
        return value

    monkeypatch.setattr(operation, "_json", altered)
    with pytest.raises(RuntimeError, match="FRESH_V2_HAWOR_CHAIN_DRIFT"):
        operation.check_only()


def test_missing_per_frame_q_is_an_explicit_blocker(monkeypatch):
    original = operation.np.load

    class MissingQ:
        files = ("q_hand", "prediction_valid", "original_frame_indices", "fps_nominal")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def load(path, **kwargs):
        if str(path) == str(operation.ROOT / operation.PINNED["states"][0]):
            return MissingQ()
        return original(path, **kwargs)

    monkeypatch.setattr(operation.np, "load", load)
    with pytest.raises(RuntimeError, match="FRESH_PER_FRAME_IK_Q_UNAVAILABLE"):
        operation.check_only()


def test_execution_gate_precedes_any_candidate_work(monkeypatch):
    calls = []

    def denied(*_):
        calls.append("gate")
        raise RuntimeError("DENIED_BY_FENCE")

    monkeypatch.setattr(operation, "authorize_exact78_operation", denied)
    monkeypatch.setattr(operation, "_active_packet", lambda: calls.append("active"))
    with pytest.raises(RuntimeError, match="DENIED_BY_FENCE"):
        operation.run_candidate(operation.OUTPUT)
    assert calls == ["gate"]
    assert not operation.OUTPUT.exists()


def test_draft_packet_is_finite_and_not_published():
    packet = executor.packet_value()
    assert packet["draft_status"] == "PREP_ONLY_NOT_REGISTERED_NOT_EXECUTABLE"
    assert packet["task_id"] == operation.TASK_ID
    assert packet["config"]["frame_count"] == 799
    assert packet["config"]["old_robot_q_action_input"] is False
    assert packet["budgets"]["session_attempts_max"] == 1
    assert packet["budgets"]["gpu_count"] == 0
    assert packet["budgets"]["model_calls"] == 0
    assert packet["budgets"]["render_calls"] == 0
    assert packet["registered_at"] is None
    assert packet["writer"] is None
    assert packet["fenced_writer_required_at_registration"] is True
    assert len(packet["read_set"]) <= 8
    assert all(Path(ref["path"]).is_file() for ref in packet["runtime_code"])
    assert not executor.PACKET_PATH.exists()
    assert not operation.OUTPUT.exists()


def test_check_only_rejects_output_path_without_writing():
    with pytest.raises(RuntimeError, match="CHECK_ONLY_OUTPUT_FORBIDDEN"):
        operation.main(["--check-only", "--output-root", str(operation.OUTPUT)])
    assert not operation.OUTPUT.exists()


def test_route_is_not_staged_and_check_only_never_publishes():
    value = publisher.check_only()
    assert value["status"] == "PREP_ONLY_NOT_REGISTERED_NOT_EXECUTED"
    assert value["route_readiness"] == "CHIPS001_ROUTE_NOT_STAGED_IN_CURRENT_R3_AND_CLI"
    assert value["active_packet_written"] is False
    assert not executor.PACKET_PATH.exists()
    assert not executor.ATTEMPT.exists()


def test_route_contract_and_packet_code_closure_are_exact():
    packet = executor.packet_value()
    route = publisher.route_contract(packet["runtime_code"])
    assert route["candidate_operation"] == operation.OPERATION
    assert route["code_closure"] == packet["runtime_code"]
    assert route["render_calls_authorized"] == 0
    assert route["current_runnable_operation"] is None


def test_finalized_packet_has_one_live_fence_and_900_second_deadline():
    packet = executor.finalize_packet(executor.packet_value())
    from datetime import datetime

    duration = (datetime.fromisoformat(packet["deadline_at"])
                - datetime.fromisoformat(packet["registered_at"])).total_seconds()
    assert duration == 900
    assert packet["writer"]["proc_start_ticks"] is not None
    assert packet["writer"]["fencing_token"]
    assert packet["budgets"]["session_attempts_max"] == 1
    assert packet["run_signature"]
    assert not executor.PACKET_PATH.exists()


def test_publisher_route_failure_precedes_all_writes(monkeypatch):
    def denied(_packet):
        raise RuntimeError("UNSTAGED_ROUTE")

    monkeypatch.setattr(publisher, "route_readiness", denied)
    with pytest.raises(RuntimeError, match="UNSTAGED_ROUTE"):
        publisher.publish()
    assert not executor.ATTEMPT.exists()
    assert not executor.PACKET_PATH.exists()
    assert not executor.RECEIPT_PATH.exists()


def test_heartbeat_and_worker_fence_precede_mutation(monkeypatch):
    calls = []

    def denied(_packet):
        calls.append("fence")
        raise RuntimeError("FENCE_DENIED")

    monkeypatch.setattr(executor, "verify_fence", denied)
    monkeypatch.setattr(executor, "publish_bundle", lambda *_a, **_k: calls.append("publish"))
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *_a, **_k: calls.append("spawn"))
    with pytest.raises(RuntimeError, match="FENCE_DENIED"):
        executor.heartbeat({}, 60.0)
    with pytest.raises(RuntimeError, match="FENCE_DENIED"):
        executor.run_worker({})
    assert calls == ["fence", "fence"]
    assert not (executor.ATTEMPT / "cache").exists()


def test_terminal_write_requires_fence_and_missing_candidate_is_failure(monkeypatch):
    candidate, errors = executor.verify_candidate({})
    assert candidate is None
    assert errors == ["CHIPS001_CANDIDATE_MISSING"]
    monkeypatch.setattr(executor, "verify_fence", lambda _packet: (_ for _ in ()).throw(
        RuntimeError("FENCE_DENIED")))
    with pytest.raises(RuntimeError, match="FENCE_DENIED"):
        executor.finish({}, 1, executor.ATTEMPT / "worker.log", "worker_failed")
    assert not (executor.ATTEMPT / "RESULT.json").exists()
    assert not executor.RECEIPT_PATH.exists()


def test_worker_group_identity_drift_never_signals(monkeypatch):
    class Worker:
        pid = 4242

        def poll(self):
            return None

    signaled = []
    monkeypatch.setattr(executor, "process_identity", lambda _pid: {"start_ticks": 19})
    monkeypatch.setattr(executor.os, "getpgid", lambda _pid: 4242)
    monkeypatch.setattr(executor.os, "killpg", lambda *_args: signaled.append(True))
    with pytest.raises(executor.WorkerNotStoppedError, match="GROUP_FENCE_DRIFT"):
        executor._stop_fenced_group(Worker(), 20)
    assert signaled == []
