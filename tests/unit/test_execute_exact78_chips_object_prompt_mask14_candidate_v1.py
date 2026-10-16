"""No-write gates for the unregistered finite Mask14 executor."""

from __future__ import annotations

import json

import pytest

from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design
from chaoyang.governance import execute_exact78_chips_object_prompt_mask14_candidate_v1 as executor
from chaoyang.ops import run_exact78_chips_object_prompt_mask14_candidate_v1 as runner


def test_packet_closure_is_six_object_calls() -> None:
    packet = design.packet_value()
    assert packet["config"]["new_object_prompt"] == "a yellow potato chip"
    assert packet["config"]["mask13_config"]["other_role_prompts"]["chips_object"] == "a potato chip"
    assert packet["budgets"] == {"workers_max": 1, "cpu_threads_total": 2,
                                 "gpu_count": 1, "model_calls_max": 6,
                                 "training_steps": 0, "wall_seconds_max": 1800,
                                 "session_attempts_max": 1}
    assert len(packet["inputs"]["anchors"]) == 6
    assert packet["config"]["clean_calls_authorized"] == 0
    assert packet["config"]["render_calls_authorized"] == 0


def test_absent_route_is_dry_run_and_register_rejects_before_writes(monkeypatch,
                                                                     capsys) -> None:
    packet = design.packet_value()
    monkeypatch.setattr(executor, "route_readiness", lambda _: "DRY_RUN_NOT_ROUTABLE")
    monkeypatch.setattr(design, "packet_value", lambda: packet)
    assert executor.main(["--check-only"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "DRY_RUN_NOT_ROUTABLE"
    assert printed["registered"] is False and printed["gpu_called"] is False
    assert not executor.ATTEMPT.exists()
    assert not executor.PACKET.exists()
    assert not executor.RECEIPT.exists()
    with pytest.raises(RuntimeError, match="DRY_RUN_NOT_ROUTABLE"):
        executor.main(["--register-and-run"])
    assert not executor.ATTEMPT.exists()
    assert not executor.PACKET.exists()
    assert not executor.RECEIPT.exists()


def test_exact_staged_route_can_publish_with_packet_in_one_cas(monkeypatch) -> None:
    from chaoyang import cli
    from chaoyang.governance import current_r3_contracts

    packet = design.packet_value()
    exact = design.route_contract(packet["runtime_code"])
    monkeypatch.setattr(cli, "_maintained_operations", lambda: set())
    monkeypatch.setattr(executor, "load_json", lambda path: {
        "special_status_contracts": {}
    } if path == executor.ALGORITHM_CONTRACT_PATH else {})
    monkeypatch.setattr(current_r3_contracts, "build_algorithm_contract",
                        lambda *_args: {"special_status_contracts": {design.ROUTE_KEY: exact}})
    assert executor.route_readiness(packet) == "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"

    monkeypatch.setattr(current_r3_contracts, "build_algorithm_contract",
                        lambda *_args: {"special_status_contracts": {
                            design.ROUTE_KEY: {**exact, "model_calls_authorized": 7}}})
    with pytest.raises(RuntimeError, match="STAGED_ROUTE_OR_CLI_CODE_SHA_DRIFT"):
        executor.route_readiness(packet)


def test_direct_op_requires_active_packet_before_output_write() -> None:
    output = executor.ATTEMPT / "candidate"
    assert not output.exists()
    with pytest.raises(RuntimeError, match="EXACT_ACTIVE_PACKET"):
        runner._active_packet(output)
    assert not output.exists()
