from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import yaml

from tools.validate_project_contracts import (
    clean_coverage_hold_fixture,
    mask_clean_review_fixture,
    mask_independent_qa_fixture,
    object_texture_donor_fixture,
    reveal_labels_hold_fixture,
    temporal_atlas_fixture,
    validate,
    verified_reveal_policy_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "contracts" / name).read_text(encoding="utf-8"))


def _errors(schema_name: str, instance: dict[str, object]) -> list[str]:
    validator = Draft202012Validator(_schema(schema_name))
    return [error.message for error in validator.iter_errors(instance)]


def test_new_clean_schemas_meta_validate_and_formal_execution_stays_closed() -> None:
    for name in (
        "verified_reveal_policy.schema.json",
        "object_texture_donor_manifest.schema.json",
        "temporal_atlas_manifest.schema.json",
        "mask_independent_qa.schema.json",
        "mask_clean_review_package.schema.json",
        "reveal_labels.schema.json",
        "clean_layers.schema.json",
        "clean_coverage.schema.json",
    ):
        Draft202012Validator.check_schema(_schema(name))

    pipeline = yaml.safe_load((ROOT / "contracts/pipeline_contract_v1.yaml").read_text())
    profile = yaml.safe_load((ROOT / "contracts/project_profile_v1.yaml").read_text())
    graph = json.loads((ROOT / "docs/architecture/DATA_REFERENCE_GRAPH_v1.json").read_text())
    assert pipeline["formal_production_allowed"] is False
    assert pipeline["formal_clean_enabled"] is False
    assert pipeline["execution_modes"]["FORMAL_PRODUCTION"]["authorized"] is False
    assert pipeline["execution_modes"]["FORMAL_PRODUCTION"]["clean_enabled"] is False
    assert profile["execution_ready"] is False
    assert profile["formal_production_allowed"] is False
    assert profile["formal_clean_enabled"] is False
    assert graph["readiness"]["formal_production_allowed"] is False
    assert graph["readiness"]["formal_clean_enabled"] is False


def test_synthetic_policy_and_donors_cannot_impersonate_formal_inputs() -> None:
    policy = verified_reveal_policy_fixture()
    assert not _errors("verified_reveal_policy.schema.json", policy)
    synthetic_policy = copy.deepcopy(policy)
    synthetic_policy.update(
        execution_mode="SYNTHETIC_DRY_RUN",
        artifact_state="SYNTHETIC_TEST_ONLY",
        formal_consumable=True,
    )
    assert _errors("verified_reveal_policy.schema.json", synthetic_policy)

    object_donor = object_texture_donor_fixture()
    assert not _errors("object_texture_donor_manifest.schema.json", object_donor)
    synthetic_object = copy.deepcopy(object_donor)
    synthetic_object.update(
        execution_mode="SYNTHETIC_DRY_RUN",
        artifact_state="SYNTHETIC_TEST_ONLY",
        formal_consumable=True,
    )
    assert _errors("object_texture_donor_manifest.schema.json", synthetic_object)

    atlas = temporal_atlas_fixture()
    assert not _errors("temporal_atlas_manifest.schema.json", atlas)
    synthetic_atlas = copy.deepcopy(atlas)
    synthetic_atlas.update(
        execution_mode="SYNTHETIC_DRY_RUN",
        artifact_state="SYNTHETIC_TEST_ONLY",
        formal_consumable=True,
    )
    assert _errors("temporal_atlas_manifest.schema.json", synthetic_atlas)


def test_cross_identity_donors_and_fake_mask_qa_are_rejected() -> None:
    object_donor = object_texture_donor_fixture()
    object_donor["identity_checks"]["background_identity_overlap_pixel_count"] = 1
    assert _errors("object_texture_donor_manifest.schema.json", object_donor)

    atlas = temporal_atlas_fixture()
    atlas["identity_checks"]["protected_object_overlap_pixel_count"] = 1
    assert _errors("temporal_atlas_manifest.schema.json", atlas)

    qa = mask_independent_qa_fixture()
    assert not _errors("mask_independent_qa.schema.json", qa)
    qa["checks"]["anatomy_coverage"]["result"] = "FAIL"
    assert _errors("mask_independent_qa.schema.json", qa)


def test_overlap_accounting_and_hold_evidence_are_fail_closed() -> None:
    reveal = reveal_labels_hold_fixture()
    assert not _errors("reveal_labels.schema.json", reveal)
    reveal["frames"][0]["post_route_identity"][
        "object_background_overlap_pixel_count"
    ] = 1
    assert _errors("reveal_labels.schema.json", reveal)

    clean = clean_coverage_hold_fixture()
    assert not _errors("clean_coverage.schema.json", clean)
    clean["frames"][0]["hold"] = False
    assert _errors("clean_coverage.schema.json", clean)


def test_review_package_requires_digest_bound_independent_mask_qa() -> None:
    package = mask_clean_review_fixture()
    assert not _errors("mask_clean_review_package.schema.json", package)

    missing = copy.deepcopy(package)
    missing.pop("mask_independent_qa_ref")
    assert _errors("mask_clean_review_package.schema.json", missing)

    wrong_owner = copy.deepcopy(package)
    wrong_owner["mask_independent_qa_ref"]["producer"] = "mask_producer"
    assert _errors("mask_clean_review_package.schema.json", wrong_owner)

    wrong_schema = copy.deepcopy(package)
    wrong_schema["mask_independent_qa_ref"]["schema_version"] = "failure-attribution-v1"
    assert _errors("mask_clean_review_package.schema.json", wrong_schema)

    synthetic_impersonation = copy.deepcopy(package)
    synthetic_impersonation["artifact_state"] = "SYNTHETIC_TEST_ONLY"
    assert _errors("mask_clean_review_package.schema.json", synthetic_impersonation)


def test_project_validator_checks_clean_boundary_and_dependency_acyclicity() -> None:
    report = validate(ROOT)
    checks = {entry["id"]: entry for entry in report["checks"]}
    assert checks["clean_review_contract_boundary"]["result"] == "PASS"
    assert checks["clean_pre_review_dependency_acyclic"]["result"] == "PASS"
    assert checks["fail_closed_schema_negative_tests"]["result"] == "PASS"
    assert report["g2_entry"]["formal_production_allowed"] is False
