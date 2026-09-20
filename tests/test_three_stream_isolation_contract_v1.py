from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/workspace/code/chaoyang")


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_three_streams_are_not_cross_promoted() -> None:
    contract = _load("docs/governance/ALGORITHM_CONTRACT.json")
    registry = _load("docs/governance/CURRENT_BASELINE_REGISTRY_V2.json")
    special = contract["special_status_contracts"]

    assert special["handle_data_cleaning_v4_0916"]["downstream_execution"] == (
        "FORBIDDEN_IN_THIS_TASK"
    )
    assert special["robot_quality_recovery_0915_15h_v21"]["0916_consumption"] == (
        "FORBIDDEN"
    )
    assert special["mask_baseline"]["selection_policy"] == "SAM3.1_ONLY_USER_LOCKED"
    assert special["mask_baseline"]["challengers"] == []

    serialized_registry = json.dumps(registry, ensure_ascii=False)
    assert "play_cards_0916_101" not in serialized_registry
    assert "research/wiyh-hand-depth-v1" not in serialized_registry
    assert registry["parallel_pipelines"][0]["cohort"].startswith("0909/0910")


def test_0915_source_group_current_provenance_preserves_historical_claim() -> None:
    contract = _load("docs/governance/ALGORITHM_CONTRACT.json")
    recovery = contract["special_status_contracts"]["robot_quality_recovery_0915_15h_v21"]
    authorization = _load("tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json")
    inventory = _load("tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json")

    assert authorization["session_recording_fact"]["source"] == "USER_CONFIRMED"
    assert inventory["source_group_status"] == (
        "PASS_USER_CONFIRMED_AND_METADATA_CORROBORATED"
    )
    assert recovery["source_group_authority"] == (
        "USER_CONFIRMED_AND_METADATA_CORROBORATED"
    )
    assert recovery["historical_v21_execution_claim"] == (
        "METADATA_VERIFIED_NO_USER_ATTESTATION"
    )
    assert len(recovery["source_group_authority_evidence"]) == 2


def test_current_execution_is_one_four_stream_parent() -> None:
    index = _load("tasks/current/INDEX.json")
    assert index["status"] == "PASS"
    assert len(index["task_packets"]) == 1
    parent = index["task_packets"][0]
    assert parent["task_id"] == "four_stream_pretraining_baseline_v32"
    assert parent["execution_allowed"] is True
    assert parent["weights"] == "ABSENT"

    contract = _load("docs/governance/ALGORITHM_CONTRACT.json")
    four_stream = contract["special_status_contracts"]["four_stream_pretraining_baseline_v32"]
    assert len(set(four_stream["lane_roots"].values())) == 4
    assert four_stream["control_ground_truth"] is False
    assert four_stream["physical_deployment_authorized"] is False
