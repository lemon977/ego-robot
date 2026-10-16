"""CPU-only gates for the unregistered Prompt14 GPU operation and executor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep
from chaoyang.governance import execute_exact78_vst_finger_sleeve_prompt14_candidate_v1 as executor
from chaoyang.ops import run_exact78_vst_finger_sleeve_prompt14_candidate_v1 as operation


def _sample_packet() -> dict:
    inputs = prep.frozen_inputs()
    return {
        "task_id": prep.TASK_ID, "execution_mode": "EXACT78_CANDIDATE",
        "candidate_operation": prep.OPERATION, "route_key": prep.ROUTE_KEY,
        "source_session": prep.SOURCE_SESSION, "anchor_frames": list(prep.FRAMES),
        "single_change": {"role": prep.ROLE, "from": prep.OLD_PROMPT, "to": prep.NEW_PROMPT},
        "inputs": inputs, "config": {"output_prob_thresh": 0.5, "instance_score_gate": 0.5},
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "model_calls_max": 3, "render_calls": 0, "clean_calls": 0,
                    "training_steps": 0, "wall_seconds_max": 900,
                    "session_attempts_max": 1},
        "commands": [executor.expected_command()],
        "runtime_code": executor.runtime_code(),
        "write_set": [str(prep.ATTEMPT.relative_to(executor.REPO_ROOT))],
        "deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        "writer": {"scope": prep.TASK_ID, "pid": 1234, "proc_start_ticks": 5678,
                   "executor_epoch": 1, "fencing_token": "a" * 48},
    }


def test_one_role_and_exact_three_frames() -> None:
    packet = _sample_packet()
    operation.validate_one_change(packet, packet["inputs"])
    assert [row["frame"] for row in packet["inputs"]["anchors"]] == [0, 75, 149]
    assert {name for name in packet["inputs"]["anchors"][0]["candidate_prompts"]
            if packet["inputs"]["anchors"][0]["candidate_prompts"][name]
            != packet["inputs"]["anchors"][0]["old_prompts"][name]} == {"finger_sleeve"}


@pytest.mark.parametrize("change", [
    lambda p: p["inputs"]["anchors"][1]["candidate_prompts"].update(yellow_cable="new cable"),
    lambda p: p["config"].update(output_prob_thresh=0.4),
    lambda p: p["budgets"].update(model_calls_max=4),
    lambda p: p.update(anchor_frames=[0, 75]),
])
def test_operation_rejects_scope_drift(change) -> None:
    packet = _sample_packet()
    change(packet)
    with pytest.raises(RuntimeError, match="PROMPT14_"):
        operation.validate_one_change(packet, prep.frozen_inputs())


def test_executor_rejects_runtime_sha_and_expired_deadline_before_writer() -> None:
    packet = _sample_packet()
    packet["run_signature"] = executor.run_signature(packet)
    packet["runtime_code"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="PACKET_OR_RUNTIME_SHA_DRIFT"):
        executor.verify_packet(packet)
    packet = _sample_packet()
    packet["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    packet["run_signature"] = executor.run_signature(packet)
    with pytest.raises(RuntimeError, match="DEADLINE_EXPIRED"):
        executor.verify_packet(packet)


def test_executor_requires_live_fenced_writer_before_gpu() -> None:
    packet = _sample_packet()
    packet["run_signature"] = executor.run_signature(packet)
    with patch.object(executor, "load_json", return_value={"pid": -1}):
        with pytest.raises(RuntimeError, match="DOUBLE_WRITER_OR_FENCE_DRIFT"):
            executor.verify_packet(packet)


def test_check_only_is_read_only_and_not_registered(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(executor, "run_registered") as run:
        assert executor.main(["--check-only"]) == 0
    run.assert_not_called()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "READY_FOR_SEPARATE_CAS_PUBLICATION"
    assert payload["anchor_frames"] == [0, 75, 149]
    assert payload["registered"] is False and payload["gpu_called"] is False


def test_direct_operation_denied_before_cuda_import() -> None:
    from chaoyang.governance.exact78_operation_gate import Exact78GateDenied

    with pytest.raises(Exact78GateDenied):
        operation.run_candidate(output_root=prep.ATTEMPT / "candidate")


def test_deadline_cleanup_rejects_unowned_gpu_child() -> None:
    with patch.object(executor, "GPU_LEASE", Path("/nonexistent/prompt14-lease")):
        assert executor._terminate_owned_gpu_child(1234, 5678) is False
