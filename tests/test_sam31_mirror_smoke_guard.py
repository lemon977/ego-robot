from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import sam31_mirror_smoke_guard as guard


ROOT = Path(__file__).resolve().parents[1]


GOOD_ENV = {
    "python": "3.12.5",
    "pytorch": "2.7.1+cu126",
    "torch_cuda_build": "12.6",
    "cuda_toolkit_nvcc": "12.6",
    "gpu_queried": False,
    "cuda_initialized": False,
    "network_used": False,
}


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    return project


def test_default_request_is_fail_closed_without_owner_acceptance(tmp_path: Path) -> None:
    report = guard.evaluate_request(guard.default_request(), make_project(tmp_path), GOOD_ENV)
    assert report["status"] == "HOLD_PENDING_EXPLICIT_MIRROR_AND_SAM_LICENSE_ACCEPTANCE"
    assert report["effective_download_allowed"] is False
    assert report["effective_gpu_allowed"] is False
    assert report["downloads_performed"] == 0
    assert report["gpu_operations_performed"] == 0


def test_checked_in_default_request_exactly_matches_guard_policy() -> None:
    checked_in = json.loads((ROOT / "archive/legacy/task_cards/SAM31_MIRROR_SMOKE_REQUEST_V1.json").read_text())
    assert checked_in == guard.default_request()


@pytest.mark.parametrize("field", ["download_allowed", "gpu_allowed"])
def test_capability_cannot_be_enabled_by_editing_request_only(tmp_path: Path, field: str) -> None:
    request = guard.default_request()
    request[field] = True
    report = guard.evaluate_request(request, make_project(tmp_path), GOOD_ENV)
    assert report["status"] == "HOLD_INVALID_OR_UNAUTHORIZED_REQUEST"
    assert any(f"{field} must remain false" in error for error in report["errors"])
    assert report[f"effective_{field}"] is False


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("pins", "mirror_revision", "main"),
        ("pins", "weight_bytes", 1),
        ("pins", "weight_sha256", "0" * 64),
        ("environment_requirements", "python_min", "3.11"),
        ("output_policy", "root_template", "assets/models/sam3_1"),
        ("output_policy", "exclusive_create", False),
    ],
)
def test_immutable_policy_tampering_is_rejected(
    tmp_path: Path, section: str, field: str, bad_value: object
) -> None:
    request = guard.default_request()
    request[section][field] = bad_value
    report = guard.evaluate_request(request, make_project(tmp_path), GOOD_ENV)
    assert report["status"] == "HOLD_INVALID_OR_UNAUTHORIZED_REQUEST"
    assert report["effective_download_allowed"] is False
    assert report["effective_gpu_allowed"] is False


def test_environment_below_official_baseline_holds(tmp_path: Path) -> None:
    request = guard.default_request()
    environment = dict(GOOD_ENV)
    environment.update(
        python="3.11.11",
        pytorch="2.6.0+cu124",
        torch_cuda_build="12.4",
        cuda_toolkit_nvcc="12.4",
    )
    report = guard.evaluate_request(request, make_project(tmp_path), environment)
    assert report["status"] == "HOLD_PENDING_EXPLICIT_MIRROR_AND_SAM_LICENSE_ACCEPTANCE"
    assert not all(report["environment_checks"].values())
    assert "ENV_PYTHON_BELOW_OR_MISSING" in report["blockers"]
    assert "ENV_CUDA_TOOLKIT_NVCC_BELOW_OR_MISSING" in report["blockers"]


def test_t0_probe_must_not_claim_network_or_gpu_use(tmp_path: Path) -> None:
    request = guard.default_request()
    environment = dict(GOOD_ENV)
    environment["network_used"] = True
    environment["gpu_queried"] = True
    report = guard.evaluate_request(request, make_project(tmp_path), environment)
    assert report["status"] == "HOLD_INVALID_OR_UNAUTHORIZED_REQUEST"
    assert any("must not use network" in error for error in report["errors"])
    assert any("must not query a GPU" in error for error in report["errors"])


def test_torch_cuda_suffix_parser() -> None:
    assert guard._torch_cuda_from_version("2.9.1+cu128") == "12.8"
    assert guard._torch_cuda_from_version("2.7.0+cu126") == "12.6"
    assert guard._torch_cuda_from_version("2.7.0") is None


def test_fake_acceptance_without_auditable_record_is_rejected(tmp_path: Path) -> None:
    request = guard.default_request()
    request["owner_acceptance"] = {
        "accepted": True,
        "accepted_by": "project-owner",
        "accepted_at": "2026-08-28T10:00:00+08:00",
        "authorization_ref": "archive/audits/user_authorizations/missing.json",
        "authorization_sha256": "0" * 64,
    }
    request["download_allowed"] = True
    request["gpu_allowed"] = True
    report = guard.evaluate_request(request, make_project(tmp_path), GOOD_ENV)
    assert report["status"] == "HOLD_INVALID_OR_UNAUTHORIZED_REQUEST"
    assert report["owner_acceptance_verified"] is False
    assert report["effective_download_allowed"] is False
    assert report["effective_gpu_allowed"] is False


def test_exact_owner_record_only_unlocks_separate_future_task(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    acceptance_path = project / "archive/audits/user_authorizations/accept.json"
    acceptance_path.parent.mkdir(parents=True)
    accepted_at = "2026-08-28T10:00:00+08:00"
    record = {
        "schema_version": guard.ACCEPTANCE_SCHEMA,
        "accepted_by": "project-owner",
        "accepted_at": accepted_at,
        "non_official_mirror_accepted": True,
        "sam_license_accepted": True,
        "official_code_commit": guard.OFFICIAL_CODE_COMMIT,
        "mirror_revision": guard.MIRROR_REVISION,
        "weight_sha256": guard.WEIGHT_SHA256,
        "statement": guard.ACCEPTANCE_STATEMENT,
    }
    acceptance_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    request = guard.default_request()
    request["owner_acceptance"] = {
        "accepted": True,
        "accepted_by": "project-owner",
        "accepted_at": accepted_at,
        "authorization_ref": "archive/audits/user_authorizations/accept.json",
        "authorization_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
    }
    request["download_allowed"] = True
    request["gpu_allowed"] = True
    report = guard.evaluate_request(request, project, GOOD_ENV)
    assert report["status"] == "READY_FOR_SEPARATE_AUTHORIZED_SMOKE_TASK_NOT_EXECUTED"
    assert report["owner_acceptance_verified"] is True
    assert report["downloads_performed"] == 0
    assert report["gpu_operations_performed"] == 0


@pytest.mark.parametrize("run_id", ["../escape", "x/y", "UPPER", "a", ".hidden"])
def test_unsafe_run_ids_are_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(guard.GuardError):
        guard._validate_run_root(make_project(tmp_path), run_id)


def test_report_is_o_excl_and_second_write_preserves_first(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    first = {"status": "FIRST"}
    report_path = guard.write_preflight_report(project, "sam31_preflight_v1", first)
    before = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        guard.write_preflight_report(project, "sam31_preflight_v1", {"status": "SECOND"})
    assert report_path.read_bytes() == before
    assert json.loads(before)["status"] == "FIRST"


def test_failed_staging_is_deleted_only_with_matching_owner_marker(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    guard.write_preflight_report(project, "sam31_preflight_v1", {"status": "HOLD"})
    staging, marker_sha = guard.prepare_owned_staging(
        project, "sam31_preflight_v1", "a" * 64
    )
    (staging / "partial.bin").write_bytes(b"partial")
    result = guard.cleanup_failed_owned_staging(project, "sam31_preflight_v1", marker_sha)
    assert result == "FAILED_STAGING_DELETED_OWNER_VERIFIED"
    assert not staging.exists()


def test_unowned_failed_staging_is_preserved_and_held(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    report = guard.write_preflight_report(project, "sam31_preflight_v1", {"status": "HOLD"})
    staging = report.parent / guard.STAGING_DIRNAME
    staging.mkdir()
    (staging / "unknown.bin").write_bytes(b"do-not-delete")
    result = guard.cleanup_failed_owned_staging(
        project, "sam31_preflight_v1", "b" * 64
    )
    assert result == "HOLD_PRESERVE_UNOWNED_STAGING"
    assert (staging / "unknown.bin").read_bytes() == b"do-not-delete"


def test_marker_digest_mismatch_preserves_failed_staging(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    guard.write_preflight_report(project, "sam31_preflight_v1", {"status": "HOLD"})
    staging, _ = guard.prepare_owned_staging(project, "sam31_preflight_v1", "a" * 64)
    result = guard.cleanup_failed_owned_staging(
        project, "sam31_preflight_v1", "b" * 64
    )
    assert result == "HOLD_PRESERVE_OWNER_MARKER_SHA_MISMATCH"
    assert staging.is_dir()


def test_run_parent_symlink_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    (project / "_run").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(guard.GuardError, match="ordinary directory"):
        guard._validate_run_root(project, "sam31_preflight_v1")
