from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from tools.validate_project_contracts import (
    mask_benchmark_fixture,
    mask_benchmark_semantic_errors,
    mask_evidence_fixture,
    mask_evidence_semantic_errors,
    mask_independent_qa_fixture,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return json.loads((ROOT / "contracts" / name).read_text(encoding="utf-8"))


def _errors(schema_name: str, payload: dict) -> list[str]:
    schema = _schema(schema_name)
    Draft202012Validator.check_schema(schema)
    return [error.message for error in Draft202012Validator(schema).iter_errors(payload)]


def test_mask_benchmark_is_45_to_60_houb_frames_with_sealed_blind_partitions() -> None:
    benchmark = mask_benchmark_fixture()
    assert not _errors("mask_benchmark.schema.json", benchmark)
    assert not mask_benchmark_semantic_errors(benchmark)

    unfrozen = copy.deepcopy(benchmark)
    unfrozen["artifact_state"] = "DRAFT_UNFROZEN"
    unfrozen["human_review_ref"] = None
    assert _errors("mask_benchmark.schema.json", unfrozen)

    wrong_cross_session = copy.deepcopy(benchmark)
    wrong_cross_session["partitions"]["cross_session_blind"][0]["session_id"] = "grap_a_cap_004"
    assert _errors("mask_benchmark.schema.json", wrong_cross_session)

    duplicate = copy.deepcopy(benchmark)
    duplicate["partitions"]["same_session_blind"][0]["frame_index"] = 0
    assert mask_benchmark_semantic_errors(duplicate)


def test_mask_candidate_requires_versioned_producer_identity() -> None:
    candidate = mask_evidence_fixture()
    assert not _errors("mask_evidence.schema.json", candidate)

    candidate.pop("producer_identity")
    assert _errors("mask_evidence.schema.json", candidate)


def test_mask_candidate_025_is_parameterized_and_fail_closed() -> None:
    pipeline = yaml.safe_load((ROOT / "contracts/pipeline_contract_v1.yaml").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "contracts/manifests/verified_source_manifest_v1.json").read_text(encoding="utf-8"))
    candidate = mask_evidence_fixture(
        session_id="grap_a_cap_025",
        product_line="EXACT78_R2_VISUAL_DOMAIN",
        frame_count=501,
    )
    assert not _errors("mask_evidence.schema.json", candidate)
    assert not mask_evidence_semantic_errors(candidate, pipeline, source)

    forged_source = copy.deepcopy(candidate)
    forged_source["source_session_digest"] = "f" * 64
    assert mask_evidence_semantic_errors(forged_source, pipeline, source)

    unregistered_model = copy.deepcopy(candidate)
    unregistered_model["model"]["weight_path"] = "unregistered/checkpoint.pt"
    assert mask_evidence_semantic_errors(unregistered_model, pipeline, source)


def test_auditor_a1_requires_frozen_benchmark_and_complete_coverage() -> None:
    qa = mask_independent_qa_fixture()
    assert not _errors("mask_independent_qa.schema.json", qa)

    wrong_auditor = copy.deepcopy(qa)
    wrong_auditor["auditor_identity"]["auditor_version"] = "auditor_a2"
    assert _errors("mask_independent_qa.schema.json", wrong_auditor)

    incomplete = copy.deepcopy(qa)
    incomplete["coverage"] = {
        "candidate_scope": "DIAGNOSTIC_15_FRAME",
        "benchmark_frame_count": 15,
        "evaluated_partitions": ["DEVELOPMENT_004"],
        "all_benchmark_frames_evaluated": False,
    }
    assert _errors("mask_independent_qa.schema.json", incomplete)


def test_contract_stops_v10_and_keeps_all_execution_closed() -> None:
    pipeline = yaml.safe_load((ROOT / "contracts/pipeline_contract_v1.yaml").read_text(encoding="utf-8"))
    profile = yaml.safe_load((ROOT / "contracts/project_profile_v1.yaml").read_text(encoding="utf-8"))
    graph = json.loads((ROOT / "docs/architecture/DATA_REFERENCE_GRAPH_v1.json").read_text(encoding="utf-8"))

    assert pipeline["calibration_execution_authorized"] is False
    assert pipeline["formal_production_allowed"] is False
    assert profile["calibration_execution_ready"] is False
    assert profile["execution_ready"] is False
    assert profile["formal_clean_enabled"] is False
    assert graph["readiness"]["calibration_execution_authorized"] is False
    assert pipeline["mask_experiment_governance"]["heuristic_line_terminal_version"] == "v9"
    assert pipeline["mask_experiment_governance"]["v10_heuristic_authorized"] is False
    auditor = profile["mask_experiment_governance"]["auditor"]
    benchmark = profile["mask_experiment_governance"]["benchmark"]
    label_interface = profile["mask_experiment_governance"]["label_interface"]
    assert auditor["freeze_state"] == "FROZEN_HUMAN_REVIEW_POLICY_DIAGNOSTIC_ONLY_NO_AUTOMATIC_PASS"
    assert auditor["claim_limit"] == "BENCHMARK_DIAGNOSTIC_RANKING_ONLY_NOT_PRODUCTION_PASS"
    assert auditor["threshold_profile_ref"]["empirical_noise_floor"] is None
    assert auditor["threshold_profile_ref"]["automatic_pass_thresholds"] is None
    assert auditor["advancement_decision_policy_ref"]["automatic_pass_allowed"] is False
    assert auditor["freeze_ref"]["sha256"] == "d2cd400a5ffb7e30f365f1ae3af21fb916b31302e64034ca8ba75c36c001fd9c"
    assert benchmark["state"] == "V2_FROZEN_HUMAN_REVIEWED_WITH_0_255_ADAPTER"
    assert benchmark["frozen_ref"]["sha256"] == "09f94084a84cd0ac0d3e93a3311da2b41c85788742029e89347b18afb6ed073b"
    assert benchmark["frozen_ref"]["binary_values"] == [0, 255]
    assert benchmark["staging_v1"]["pilot_review"]["state"] == "PASS_PILOT_ONLY_NOT_FROZEN"
    assert benchmark["staging_v2"]["frame_count"] == 45
    assert benchmark["staging_v2"]["live_label_counts"].startswith("DERIVE_AT_RUNTIME")
    assert "locked_migrated_labels" not in benchmark["staging_v2"]
    assert "remaining_labels" not in benchmark["staging_v2"]
    assert benchmark["staging_v2"]["held_out_final_blind_session"] == "grap_a_cap_149"
    assert label_interface["unique_atomic_label_freezer_state"] == "T2_FROZEN_0_255_ADAPTER_ORIGINAL_T0_0_1_PRESERVED"
    assert label_interface["derived_t0_manifest"]["freeze_ref_issued"] is True
    assert label_interface["derived_t0_manifest"]["original_binary_values"] == [0, 1]
    assert label_interface["derived_t0_manifest"]["consumer_allowed"] is False
    assert label_interface["frozen_adapter"]["binary_values"] == [0, 255]
    assert label_interface["frozen_adapter"]["reverse_palette_exact_frames"] == 45
    assert label_interface["independent_second_pass"]["state"] == "USER_WAIVED_0_OF_8"
    assert label_interface["independent_second_pass"]["noise_floor"] == "UNAVAILABLE_USER_WAIVED"
    missing_noise_floor = profile["mask_experiment_governance"]["missing_noise_floor_policy"]
    assert missing_noise_floor["allowed_routes"] == ["HUMAN_REVIEW_POLICY"]
    assert missing_noise_floor["blind_unsealed"] is False
    assert missing_noise_floor["t2_authorized"] is True
    assert missing_noise_floor["b3_authorized"] is False
    assert missing_noise_floor["automatic_pass_allowed"] is False
    assert pipeline["mask_experiment_governance"]["missing_noise_floor_policy"] == missing_noise_floor
    for contract in (pipeline, profile):
        registry = contract["reject_hold_evidence_registry"]
        assert registry["state"] == "IMPLEMENTED_REVIEW_ONLY_NOT_A_FORMAL_ARTIFACT"
        assert registry["registry_ref"] == "_run/rejected_visual_review_v1/MANIFEST.json"
        assert registry["registry_sha256"] == "9ae293385f4f0f5cafec8f3d66d8a2ad5403759abc0f749f6d32413c3da5abcf"
        assert registry["index_sha256"] == "edd14102eb994023fb1b0f54afa383bbea413d82748ea089218dbd3aa1abe888"
        assert registry["asset_count"] == 13
        assert registry["required_for"] == ["REJECT", "HOLD"]
        assert registry["review_may_mutate_upstream_or_decision"] is False
        assert registry["formal_consumer_allowed"] is False
        assert registry["t2_or_blind_unsealed"] is False
        assert registry["b1_second_pass_required_by_registry"] is False
    assert label_interface["round_trip_exactness_required"] is True
    historical = pipeline["mask_experiment_governance"]["historical_evidence"]
    assert all(
        body["comparison_status"].endswith("NOT_RANKABLE")
        for key, body in historical.items()
        if key in {"v3", "v8", "v9"}
    )
    assert historical["producer_p1_r3"]["comparison_status"] == "HOLD_T1_FORENSIC_PIXEL_AND_SCHEMA_FAILURE"
    assert historical["producer_p1_r3"]["formal_consumer_allowed"] is False


def test_live_v9_hold_evidence_is_digest_bound_and_not_a_mask_candidate() -> None:
    run_path = ROOT / "_run/g2_004_mask_clean_candidate_v9/run_manifest.json"
    qa_path = ROOT / "_run/g2_004_mask_clean_candidate_v9/qa/INDEPENDENT_QA.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == "b943797a905739530ed69bb50bb65b71be992a8a413e5d08f3ad9c2695ee8953"
    assert hashlib.sha256(qa_path.read_bytes()).hexdigest() == "00df88188d783e509912f9708ac06b6aa2f351fc04140b9f2e0fdd1eefe70310"
    assert run["diagnostic_status"] == "DIAGNOSTIC_HOLD"
    assert run["full_h20_authorized"] is False
    assert run["mask_producer_started"] is False
    assert qa["machine_checks"]["left"]["held_frames"] == 15
    assert qa["machine_checks"]["left"]["final_sleeve_prompt_recall"] == 0.4
    assert qa["machine_checks"]["left"]["final_sleeve_prompt_recall_threshold"] == 0.6
    assert qa["machine_checks"]["raw_second_sam_candidate_files"] == 0


def test_validator_reports_consistent_governance_hold() -> None:
    report = validate(ROOT)
    assert report["overall_verdict"] == "PASS_G1_CONTRACT_CONSISTENCY_G2_GOVERNANCE_HOLD"
    assert report["summary"] == {"checks_pass": 22, "checks_fail": 0, "errors": 0, "warnings": 0}
    assert report["g2_entry"]["contract_authorized_for_calibration"] is False
