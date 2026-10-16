"""CPU-only checks for the unregistered C0 publisher/worker handoff."""

from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path

from chaoyang.governance import execute_exact78_c0_cleanup_proof_v1 as worker
from chaoyang.governance import register_exact78_c0_cleanup_proof_v1 as wrapper
from chaoyang.governance.common import artifact_ref, canonical_bytes, process_identity


def test_packet_pins_exact_worker_command_and_fenced_pid() -> None:
    deadline = datetime.now().astimezone() + timedelta(hours=3)
    inputs = {"candidate_manifest": {"path": "/project/candidates.json", "bytes": 1,
                                     "sha256": "a" * 64},
              "registration_code": artifact_ref(wrapper.SCRIPT)}
    scope = {"file_globs": ["docs/**/*.json"], "max_text_file_bytes": 1024,
             "max_total_text_bytes": 2048, "max_reference_files": 10}
    packet = wrapper.packet_value(inputs, scope, deadline)
    assert packet["commands"] == [wrapper.COMMAND]
    assert packet["runtime_code"] == [artifact_ref(worker.SCRIPT)]
    assert packet["writer"]["pid"] == os.getpid()
    assert packet["writer"]["proc_start_ticks"] == process_identity(os.getpid())["start_ticks"]
    assert packet["writer"]["executor_epoch"] == 1
    assert len(packet["writer"]["fencing_token"]) == 48
    assert packet["campaign_deadline_at"] == deadline.isoformat()
    assert datetime.fromisoformat(packet["deadline_at"]) < deadline
    expected = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": packet["runtime_code"],
        "reference_scan_scope": scope, "schema": "exact78-c0-proof-v1",
        "weights": "NOT_CONSUMED", "calibration": "NOT_CONSUMED",
    })).hexdigest()
    assert packet["run_signature"] == expected


def test_packet_write_set_is_project_local_and_c0_only() -> None:
    packet = wrapper.packet_value({}, {"file_globs": []},
                                  datetime.now().astimezone() + timedelta(hours=3))
    assert len(packet["read_set"]) <= 8
    for relative in packet["write_set"]:
        assert not Path(relative).is_absolute()
        assert ".." not in Path(relative).parts
    assert packet["write_set"] == [
        str(wrapper.OUT.relative_to(wrapper.REPO_ROOT)),
        str(wrapper.PACKET.relative_to(wrapper.REPO_ROOT)),
        str(wrapper.RECEIPT.relative_to(wrapper.REPO_ROOT)),
        "tasks/current/INDEX.json", "docs/governance",
    ]
    assert packet["budgets"]["gpu_count"] == 0
    assert packet["budgets"]["session_attempts_max"] == 1
