"""Read-only, CPU-only negative checks for the Prompt14 CAS publisher."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from unittest.mock import patch

import pytest

from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep
from chaoyang.governance import execute_exact78_vst_finger_sleeve_prompt14_candidate_v1 as executor
from chaoyang.governance import publish_exact78_vst_finger_sleeve_prompt14_candidate_v1 as publisher


def test_exact_packet_is_one_role_three_frames_and_900_seconds() -> None:
    packet = publisher.packet_value()
    assert packet["inputs"] == prep.frozen_inputs()
    assert packet["anchor_frames"] == [0, 75, 149]
    assert packet["single_change"] == {"role": "finger_sleeve",
                                       "from": prep.OLD_PROMPT, "to": prep.NEW_PROMPT}
    assert packet["budgets"]["model_calls_max"] == 3
    assert packet["budgets"]["session_attempts_max"] == 1
    assert packet["commands"] == [executor.expected_command()]
    assert len(packet["read_set"]) == 8
    assert len(packet["runtime_code"]) == len(executor.RUNTIME_RELATIVE)
    assert (datetime.fromisoformat(packet["deadline_at"])
            - datetime.fromisoformat(packet["registered_at"])).total_seconds() == 900
    assert packet["run_signature"] == executor.run_signature(packet)
    publisher.validate_for_publication(packet)


def test_h20_requires_complete_900_second_dispatch_window() -> None:
    edge = publisher.H20_NO_NEW_HEAVY_DISPATCH - timedelta(seconds=900)
    assert publisher.dispatch_gate(edge) == "DISPATCH_WINDOW_OPEN"
    assert publisher.dispatch_gate(edge + timedelta(microseconds=1)) == "H20_HEAVY_DISPATCH_CLOSED"
    assert publisher.dispatch_gate(publisher.H20_NO_NEW_HEAVY_DISPATCH) == "H20_HEAVY_DISPATCH_CLOSED"
    with patch.object(publisher, "_now", return_value=publisher.H20_NO_NEW_HEAVY_DISPATCH):
        packet = publisher.packet_value()
        assert publisher.readiness(packet) == "H20_HEAVY_DISPATCH_CLOSED"
        with pytest.raises(RuntimeError, match="DEADLINE_DRIFT"):
            publisher.validate_for_publication(packet)


def test_packet_deadline_is_capped_by_h24() -> None:
    late = publisher.H24_HARD_DEADLINE - timedelta(seconds=100)
    with patch.object(publisher, "_now", return_value=late):
        packet = publisher.packet_value()
    assert datetime.fromisoformat(packet["deadline_at"]) == publisher.H24_HARD_DEADLINE


@pytest.mark.parametrize("mutate,code", [
    (lambda p: p["config"].update(output_prob_thresh=0.4), "PACKET_VALUE_DRIFT"),
    (lambda p: p["budgets"].update(model_calls_max=4), "PACKET_VALUE_DRIFT"),
    (lambda p: p["runtime_code"].pop(), "PACKET_VALUE_DRIFT"),
    (lambda p: p.update(run_signature="0" * 64), "RUN_SIGNATURE_DRIFT"),
    (lambda p: p["writer"].update(fencing_token="short"), "WRITER_FENCE_DRIFT"),
])
def test_publication_rejects_packet_drift_before_writes(mutate, code) -> None:
    packet = publisher.packet_value()
    mutate(packet)
    with pytest.raises(RuntimeError, match=code):
        publisher.validate_for_publication(packet)


def test_active_conflict_precedes_unstaged_route() -> None:
    packet = publisher.packet_value()
    state = {"next_task": {"task_id": "other"}, "tasks": []}
    index = {"task_packets": []}
    with (patch.object(publisher, "load_json", side_effect=[state, index]),
          patch.object(publisher, "route_readiness") as route):
        assert publisher.readiness(packet) == "ACTIVE_CONFLICT"
    route.assert_not_called()


def test_route_not_staged_is_explicit_and_blocks_registration() -> None:
    packet = publisher.packet_value()
    from chaoyang.governance import current_r3_contracts

    with (patch.object(publisher, "load_json", side_effect=[
            {"special_status_contracts": {}}, {}, {}]),
          patch.object(current_r3_contracts, "build_algorithm_contract",
                       return_value={"special_status_contracts": {}})):
        assert publisher.route_readiness(packet) == "NOT_ROUTABLE_UNTIL_ROUTE_STAGED"
    with patch.object(publisher, "readiness", return_value="NOT_ROUTABLE_UNTIL_ROUTE_STAGED"):
        with pytest.raises(RuntimeError, match="REGISTER_BLOCKED:NOT_ROUTABLE_UNTIL_ROUTE_STAGED"):
            publisher.register(packet)


def test_check_only_does_not_publish_or_run(capsys: pytest.CaptureFixture[str]) -> None:
    with (patch.object(publisher, "register") as register,
          patch.object(publisher, "publish_and_run") as run):
        assert publisher.main(["--check-only"]) == 0
    register.assert_not_called()
    run.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] in {
        "ACTIVE_CONFLICT",
        "NOT_ROUTABLE_UNTIL_ROUTE_STAGED",
        "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS",
    }
    assert output["registered"] is False and output["gpu_called"] is False
    assert output["heartbeat_seconds"] == 20
