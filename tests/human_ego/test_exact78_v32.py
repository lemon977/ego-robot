from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.human_ego import exact78_v32 as contract
from chaoyang.human_ego.tools import build_exact78_pair_ledgers_v32 as builder
from chaoyang.human_ego.tools import train_visual_aux_future2d_v53 as trainer


ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "manifests/human_ego/exact78_cohort_split_v32.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def readiness() -> dict:
    return {
        "production_readiness": {
            "stages": [
                {"stage": stage, "status": "PASS", "evidence_sha256": "e" * 64}
                for stage in contract.PRODUCTION_STAGES[:-1]
            ],
            "next_stage": "PAIR_LEDGER_VALIDATION",
        },
        "robotized_producer": {
            "producer_kind": "CAUSAL_PIXEL_COMPOSITOR",
            "actual_production": True,
            "copy_or_rename_only": False,
            "producer_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "input_sha256": "3" * 64,
        },
    }


def suffix_payload() -> dict:
    return {
        "schema_version": contract.SUFFIX_SCHEMA,
        "status": "PASS",
        "causal_current_inputs_authorized": True,
        "independent_replay": True,
        "production_runs": {
            "full": {
                "run_id": "full-1",
                "cache_root": "/tmp/exact78-full-cache",
                "actual_production": True,
                "producer_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "input_sha256": "3" * 64,
            },
            "prefix": {
                "run_id": "prefix-1",
                "cache_root": "/tmp/exact78-prefix-cache",
                "actual_production": True,
                "producer_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "input_sha256": "4" * 64,
            },
        },
        "fields": {
            name: {
                "suffix_invariant": True,
                "effective_temporal_authority": "CAUSAL_CURRENT",
            }
            for name in contract.REQUIRED_SUFFIX_FIELDS
        },
    }


def test_checked_in_cohort_freezes_156_without_replacement() -> None:
    payload = load(COHORT)
    indexed = contract.validate_cohort_split(payload)
    assert len(indexed) == 156
    assert sum(row["split"] == "train" for row in indexed.values()) == 59
    assert sum(row["split"] == "validation" for row in indexed.values()) == 77
    assert sum(row["split"] == "development_final" for row in indexed.values()) == 20
    assert indexed["get_potato_chips_0902_002"]["split"] == "validation"
    assert indexed["play_cards_0903_245"]["split"] == "development_final"
    assert payload["development_final_policy"] == "EXPOSED_NOT_FIRST_BLIND_TEST"


def test_cohort_rejects_source_group_leakage() -> None:
    payload = load(COHORT)
    payload["source_groups"][2]["split"] = "train"
    with pytest.raises(contract.ContractError):
        contract.validate_cohort_split(payload)


def test_current_rebind_preserves_missing_and_ignores_extra_candidate() -> None:
    frozen = contract.validate_cohort_split(load(COHORT))
    rows = []
    missing = "get_potato_chips_0901_001"
    for session, item in frozen.items():
        if session == missing:
            continue
        rows.append(
            {
                "session_id": session,
                "task": item["task"],
                "date": item["capture_date"],
                "status": "PASS_RAW_PRESENT",
                "source_group_candidate": item["source_group_id"],
                "raw_path": f"/current/{session}",
                "clip_manifest": {
                    "path": f"/current/{session}/clip_manifest.json",
                    "bytes": 1,
                    "sha256": "a" * 64,
                },
            }
        )
    rows.append(
        {
            "session_id": "get_potato_chips_0901_999",
            "task": "chips",
            "date": "0901",
            "status": "PASS_RAW_PRESENT",
            "source_group_candidate": "extra",
        }
    )
    audit, rebound, unresolved = builder.validate_current_cohort(
        COHORT, {"rows": rows}
    )
    assert unresolved == [missing]
    assert rebound[missing]["current_rebind_status"] == "UNRESOLVED_NO_SUBSTITUTION"
    assert audit["substitution_count"] == 0
    assert audit["unexpected_candidates_consumed"] is False
    assert audit["unexpected_current_candidates"] == ["get_potato_chips_0901_999"]


def test_pair_readiness_rejects_renamed_copy_and_accepts_real_difference() -> None:
    raw = [
        {"path": "/pair/raw/000.png", "sha256": "a" * 64},
        {"path": "/pair/raw/001.png", "sha256": "b" * 64},
    ]
    copied = [
        {"path": "/pair/robot/000.png", "sha256": "a" * 64},
        {"path": "/pair/robot/001.png", "sha256": "b" * 64},
    ]
    with pytest.raises(contract.ContractError, match="byte-identical renamed"):
        contract.validate_pair_production_readiness(
            readiness(), raw_frame_refs=raw, robotized_frame_refs=copied
        )
    copied[1]["sha256"] = "c" * 64
    report = contract.validate_pair_production_readiness(
        readiness(), raw_frame_refs=raw, robotized_frame_refs=copied
    )
    assert report["byte_different_frame_count"] == 1
    assert report["robotized_is_copy_or_rename"] is False


def test_suffix_replay_requires_isolated_caches_and_real_runs() -> None:
    payload = suffix_payload()
    contract.validate_suffix_receipt(payload)
    payload["production_runs"]["prefix"]["cache_root"] = payload[
        "production_runs"
    ]["full"]["cache_root"]
    with pytest.raises(contract.ContractError, match="distinct run IDs and cache roots"):
        contract.validate_suffix_receipt(payload)


def test_raw_only_contract_is_explicitly_non_ab() -> None:
    payload = load(ROOT / "configs/human_ego/exact78_raw_only_dev_v1.json")
    contract.validate_raw_only_contract(payload)
    payload["formal_ab_allowed"] = True
    with pytest.raises(contract.ContractError, match="formal_ab_allowed"):
        contract.validate_raw_only_contract(payload)


def _artifact(path: Path) -> dict:
    return trainer.artifact_ref(path)


def test_trainer_consumes_v32_split_not_v31_hardcoded_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This 0902 identity was historically called train.  V3.2 freezes it as
    # validation, and validate_ledger must enforce that at its real boundary.
    session = "get_potato_chips_0902_002"
    manifest = tmp_path / session / "VISUAL_AUX_SESSION_MANIFEST.json"
    dump(
        manifest,
        {
            "session_id": session,
            "task": "chips",
            "split": "train",
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True, "raw_robotized_valid_mask_shared": True},
        },
    )
    other_session = "get_potato_chips_0901_001"
    other = tmp_path / other_session / "VISUAL_AUX_SESSION_MANIFEST.json"
    dump(
        other,
        {
            "session_id": other_session,
            "task": "chips",
            "split": "validation",
            "input_mode": "CAUSAL_TRAINING_INPUT",
            "causal_proof": {"enabled": True, "raw_robotized_valid_mask_shared": True},
        },
    )
    hardset = tmp_path / "hardset.json"
    hardset.write_text("{}\n", encoding="utf-8")
    lineage = tmp_path / "clip_manifest.json"
    lineage.write_text("{}\n", encoding="utf-8")
    suffix = tmp_path / "suffix.json"
    dump(suffix, suffix_payload())
    cohort_index = contract.validate_cohort_split(load(COHORT))

    def binding(identity: str, path: Path, declared_split: str) -> dict:
        return {
            "session_id": identity,
            "source_group_id": cohort_index[identity]["source_group_id"],
            "split": declared_split,
            "source_frame": "BUNDLE_FRAME_IDS",
            "timestamp": "BUNDLE_TIMESTAMPS",
            "current_source_lineage": str(lineage.resolve()),
            "source_lineage_receipt": _artifact(lineage),
            "input_sha256": trainer.sha256(path),
            "status": "RESOLVED_EXACT_FROZEN_IDENTITY",
        }

    receipt_rows = []
    for identity, path in ((session, manifest), (other_session, other)):
        receipt_rows.append(
            {
                "session_id": identity,
                "bundle_manifest_sha256": trainer.sha256(path),
                "audit": _artifact(suffix),
            }
        )
    ledger = {
        "schema_version": contract.PAIR_LEDGER_SCHEMA,
        "status": "PASS_EXACT78_PAIR_LEDGER_V32",
        "contract": contract.PAIR_CONTRACT,
        "release_id": "EXACT78_V32_DEVELOPMENT",
        "task": "chips",
        "branch": "HUMAN_RAW_RGB",
        "seed": 7,
        "input_mode": "CAUSAL_TRAINING_INPUT",
        "image_domain": "SOURCE_DOMAIN_BOUND_PER_BUNDLE",
        "units": "RGB_UINT8_AND_NORMALIZED_IMAGE_XY",
        "side": "BUNDLE_DECLARED_PHYSICAL_SIDE",
        "validity": "EXPLICIT_SHARED_VALID_MASK_NO_FILL",
        "observability": "OBSERVED_AND_INFERRED_EXPLICITLY_SEPARATED",
        "temporal_authority": "CAUSAL_CURRENT_INPUT_NONCAUSAL_TARGET",
        "producer_identity": {
            "producer_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "input_sha256": "3" * 64,
        },
        "cohort_authority": _artifact(COHORT),
        "source_bindings": [
            binding(session, manifest, "train"),
            binding(other_session, other, "validation"),
        ],
        "suffix_invariance_receipts": receipt_rows,
        "pair_production_readiness": [
            {"session_id": session},
            {"session_id": other_session},
        ],
        "train": [_artifact(manifest)],
        "validation": [_artifact(other)],
        "development_final": [],
        "occlusion_hardset": _artifact(hardset),
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "external_metric_authority": False,
    }
    ledger["pair_dataset_signature"] = trainer.pair_dataset_signature(ledger)
    ledger_path = tmp_path / "ledger.json"
    dump(ledger_path, ledger)
    monkeypatch.setattr(
        trainer,
        "validate_bundle",
        lambda path: {"eligible_h50_window_count": 1, "eligible_h50_starts": [0]},
    )
    monkeypatch.setattr(trainer, "_validate_manifest_training_gate", lambda *args: None)

    with pytest.raises(RuntimeError, match="outside its frozen cohort task/split"):
        trainer.validate_ledger(ledger_path)


def test_builder_v32_ledgers_bind_manifest_split_and_pair_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}\n", encoding="utf-8")
    admitted = []
    for split, session in (
        ("train", "get_potato_chips_0901_001"),
        ("validation", "get_potato_chips_0902_002"),
    ):
        manifest = tmp_path / session / "VISUAL_AUX_SESSION_MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        lineage = tmp_path / session / "clip_manifest.json"
        lineage.write_text("{}\n", encoding="utf-8")
        audit = tmp_path / session / "suffix.json"
        dump(audit, suffix_payload())
        admitted.append(
            {
                "session_id": session,
                "task": "chips",
                "split": split,
                "source_group_id": f"group-{split}",
                "source_lineage": _artifact(lineage),
                "manifest": _artifact(manifest),
                "suffix_audit": _artifact(audit),
                "pair_pixel_audit": {
                    "raw_frames": 1,
                    "robotized_frames": 1,
                    "byte_different_frame_count": 1,
                    "robotized_is_copy_or_rename": False,
                },
                "eligible_h50_starts": [0],
                "eligible_h50_window_count": 1,
            }
        )
    monkeypatch.setattr(trainer, "validate_paired_ledgers", lambda *args: {})
    result, blockers = builder._task_outputs(
        "chips",
        admitted,
        tmp_path / "out",
        _artifact(cohort),
        {
            "producer_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "input_sha256": "3" * 64,
        },
    )
    assert blockers == []
    raw = load(Path(result["ledgers"]["HUMAN_RAW_RGB"]["path"]))
    robot = load(Path(result["ledgers"]["ROBOTIZED_RGB"]["path"]))
    assert raw["schema_version"] == contract.PAIR_LEDGER_SCHEMA
    assert raw["train"] == robot["train"]
    assert raw["validation"] == robot["validation"]
    assert raw["pair_production_readiness"] == robot["pair_production_readiness"]
    assert raw["pair_dataset_signature"] == robot["pair_dataset_signature"]
