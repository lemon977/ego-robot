from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.v52_contracts import validate_task_packet
from chaoyang.governance.v71_contracts import (
    build_artifact_revision,
    publish_new_revision,
    stale_for_latest_revision,
    summarize_terminal_closure,
    validate_artifact_revision,
)


def _artifact():
    return build_artifact_revision(
        artifact_id="clean.play_cards_001",
        artifact_revision="R7_0",
        input_artifacts=[("raw.play_cards_001", "R7_0")],
        input_manifest_sha="a" * 64,
        producer_signature="run-signature-001",
        payload={"status": "PASSED"},
    )


def _packet():
    return {
        "schema_version": "exact78-task-packet-v1",
        "task_id": "fixture",
        "objective": "fixture",
        "read_set": [],
        "write_set": [],
        "prerequisites": [],
        "gates": [],
        "budgets": {"cpu_seconds": 1, "gpu_seconds": 0, "wall_seconds": 1},
        "attempt_max": 1,
        "stop_condition": "one terminal",
        "output_contract": ["RESULT.json"],
        "claim_limit": "test only",
        "executor_epoch": 1,
        "fencing": {"pid_startticks_required": True, "immutable_final": True},
    }


def test_artifact_revision_matches_schema_and_pairs_inputs():
    artifact = _artifact()
    schema = json.loads(
        (ROOT / "contracts/immutable_artifact_revision_v71.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(artifact)
    assert validate_artifact_revision(artifact) == []
    assert artifact["input_artifact_ids"] == ["raw.play_cards_001"]
    assert artifact["input_artifact_revisions"] == ["R7_0"]


def test_successor_is_new_and_old_revision_becomes_stale_only_in_projection(tmp_path):
    old = _artifact()
    path = tmp_path / "clean-r7_0.json"
    publish_new_revision(path, old)
    original_bytes = path.read_bytes()

    with pytest.raises(FileExistsError):
        publish_new_revision(path, old)

    stale = stale_for_latest_revision(old)
    successor = build_artifact_revision(
        artifact_id="clean.play_cards_001.r7_1",
        artifact_revision="R7_1",
        input_artifacts=[("raw.play_cards_001", "R7_0")],
        input_manifest_sha="b" * 64,
        producer_signature="run-signature-002",
        supersedes_artifact_id=old["artifact_id"],
    )
    publish_new_revision(tmp_path / "clean-r7_1.json", successor)
    assert path.read_bytes() == original_bytes
    assert old["validity"] == "VALID_FOR_PINNED_REVISION"
    assert stale["validity"] == "STALE_FOR_LATEST_REVISION"


def test_input_ids_and_revisions_must_be_one_to_one():
    artifact = _artifact()
    artifact["input_artifact_revisions"] = []
    assert "input artifact ids and revisions must have equal lengths" in validate_artifact_revision(artifact)


def test_task_packet_revision_policy_is_optional_and_backward_compatible():
    packet = _packet()
    assert validate_task_packet(packet) == []
    packet["artifact_revision_contract"] = {
        "required_revision_status": "VALID_FOR_PINNED_REVISION",
        "forbid_in_place_overwrite": True,
    }
    assert validate_task_packet(packet) == []


def test_task_packet_rejects_policy_that_allows_in_place_overwrite():
    packet = _packet()
    packet["artifact_revision_contract"] = {
        "required_revision_status": "VALID_FOR_PINNED_REVISION",
        "forbid_in_place_overwrite": False,
    }
    assert "artifact_revision_contract.forbid_in_place_overwrite must be true" in validate_task_packet(packet)


def test_task_packet_accepts_claim_time_concrete_revision_pins():
    packet = _packet()
    packet["artifact_revision_contract"] = {
        "required_revision_status": "VALID_FOR_PINNED_REVISION",
        "forbid_in_place_overwrite": True,
        "input_artifact_ids": ["raw.play_cards_001"],
        "input_artifact_revisions": ["R7_0"],
        "input_manifest_sha": "c" * 64,
        "output_artifact_revision": "R7_1",
    }
    assert validate_task_packet(packet) == []


@pytest.mark.parametrize(
    "negative",
    [
        "FAILED_QUALITY_C",
        "FAILED_RUNTIME_FINAL",
        "BLOCKED_PREREQ",
        "BLOCKED_RESOURCE",
        "BLOCKED_EXTERNAL",
        "BLOCKED_REFERENCE_PROOF",
        "UNKNOWN_VERIFICATION_REQUIRED",
        "CANCELLED",
    ],
)
def test_negative_terminal_closes_stage_without_authorizing_downstream(negative):
    closure = summarize_terminal_closure(
        [{"task_id": "a", "status": "PASSED"}, {"task_id": "b", "status": negative}],
        ["a", "b"],
    )
    assert closure["stage_terminal_complete"] is True
    assert closure["downstream_authorized"] is False


def test_only_all_passed_unique_denominator_authorizes_downstream():
    closure = summarize_terminal_closure(
        [{"task_id": "a", "status": "PASSED"}, {"task_id": "b", "status": "PASSED"}],
        ["a", "b"],
    )
    assert closure["stage_terminal_complete"] is True
    assert closure["downstream_authorized"] is True

    duplicate = summarize_terminal_closure(
        [{"task_id": "a", "status": "PASSED"}, {"task_id": "a", "status": "PASSED"}],
        ["a", "b"],
    )
    assert duplicate["stage_terminal_complete"] is False
    assert duplicate["downstream_authorized"] is False
    assert duplicate["duplicate_task_ids"] == ["a"]


def test_running_or_empty_denominator_never_authorizes_downstream():
    running = summarize_terminal_closure([{"task_id": "a", "status": "RUNNING"}], ["a"])
    assert running["stage_terminal_complete"] is False
    assert running["downstream_authorized"] is False
    assert running["non_terminal_task_ids"] == ["a"]

    empty = summarize_terminal_closure([], [])
    assert empty["stage_terminal_complete"] is True
    assert empty["downstream_authorized"] is False
