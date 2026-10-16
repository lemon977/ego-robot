"""Portable profile/execution policy tests; no recursive pytest or model calls."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("synthetic_cpu_profile_runner", ROOT / "scripts/maintenance/run_project_checks_v1.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
CHECK = RUN.CHECK


def test_portable_server_profiles_are_explicit_disjoint_and_complete():
    assert len(CHECK.profile_files("portable")) == 2
    assert len(CHECK.profile_files("server")) == 4
    assert not set(CHECK.PORTABLE_TESTS) & set(CHECK.SERVER_TESTS)
    assert all(name.startswith("tests/maintenance/") for name in CHECK.PORTABLE_TESTS)
    assert all(name.startswith("tests/unit/") for name in CHECK.SERVER_TESTS)


def test_server_profile_requires_historical_evidence_even_in_clean_clone(tmp_path):
    with pytest.raises(CHECK.CheckError, match="SERVER_EVIDENCE_ABSENT"):
        CHECK.require_server_evidence(tmp_path)
    assert not list(tmp_path.iterdir())


def test_server_profile_fixed_batch04_is_not_auto_latest(tmp_path):
    path = tmp_path / CHECK.SERVER_PACKET
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(CHECK.CheckError, match="SERVER_EVIDENCE_SHA_DRIFT"):
        CHECK.require_server_evidence(tmp_path)
    assert "batch04" in CHECK.SERVER_PACKET
    assert CHECK.SERVER_PACKET_SHA == "2931068f54f0a311718716f3d952d0280e82af64d154e6166cf0d38b52ce3420"


def test_missing_pytest_is_environment_failure_never_install(monkeypatch, tmp_path):
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(CHECK.CheckError, match="ENV_PYTEST_ABSENT"):
        RUN.run_profile(tmp_path, "portable")
    assert not list(tmp_path.iterdir())


def test_environment_snapshot_is_metadata_not_gpu_claim():
    result = RUN.environment_snapshot()
    assert result["interpreter"]
    assert set(result["software_versions"]) == {"pytest", "jsonschema"}
    assert result["model_environment"] == "NOT_CHECKED_NO_MODEL_IMPORT"
    assert "not GPU health" in result["claim_limit"]


@pytest.mark.parametrize("value", [0, -1, 301, 3600])
def test_timeout_is_finite_and_bounded(value):
    with pytest.raises(CHECK.CheckError, match="CPU_TIMEOUT_OUT_OF_BOUNDS"):
        RUN.bounded_timeout(value)


@pytest.mark.parametrize("value", [1, 180, 300])
def test_accepted_timeout(value):
    assert RUN.bounded_timeout(value) == value


def test_output_cannot_escape_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(CHECK.CheckError, match="PATH_SCOPE_OR_SYMLINK"):
        RUN.output_parent(root, tmp_path / "outside")
    with pytest.raises(CHECK.CheckError, match="PATH_SCOPE_OR_SYMLINK"):
        RUN.output_parent(root, root)


def test_output_cannot_follow_symlink(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    (root / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(CHECK.CheckError, match="PATH_SCOPE_OR_SYMLINK"):
        RUN.output_parent(root, root / "alias")


def test_test_environment_uses_one_cpu_no_plugins_no_bytecode_and_project_paths(tmp_path):
    root = tmp_path / "project"
    attempt = root / "local_attempt"
    env = RUN.pytest_environment(root, attempt)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == env["PYTEST_ADDOPTS"] == env["PYTEST_PLUGINS"] == ""
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX"):
        assert attempt in Path(env[key]).parents


def test_explicit_command_never_recurses_into_tmp_trees(tmp_path):
    command = RUN.pytest_command(tmp_path, "portable", tmp_path / "attempt")
    assert command[:5] == [RUN.sys.executable, "-B", "-m", "pytest", "-q"]
    assert command[-2:] == [str(tmp_path / name) for name in CHECK.PORTABLE_TESTS]
    assert str(tmp_path / "tests") not in command
    assert "--basetemp" in command and "cache_dir=" + str(tmp_path / "attempt/pytest_cache") in command


def test_signature_and_counts_do_not_fabricate_test_success():
    assert RUN.signature({"b": 2, "a": 1}) == RUN.signature({"a": 1, "b": 2})
    assert RUN.signature({"a": 1}) != RUN.signature({"a": 2})
    assert RUN.test_counts("no tests ran in 0.01s") == {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    assert RUN.test_counts("44 passed, 2 failed in 1.00s")["failed"] == 2


def test_profile_declaration_does_not_execute(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("declaration must not read server evidence or launch tests")
    monkeypatch.setattr(CHECK, "inspect_project", forbidden)
    monkeypatch.setattr(RUN.subprocess, "run", forbidden)
    assert CHECK.main(["--test-profile", "server"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "TEST_PROFILE_DECLARED_NOT_EXECUTED"
    assert value["requires_server_evidence"] is True
    assert not value["execution_authorization"]


def test_profile_declaration_missing_environment_has_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(CHECK, "environment_report", lambda required: {"errors": ["ENV_PYTEST_ABSENT"]})
    assert CHECK.main(["--test-profile", "portable"]) == 2
    assert json.loads(capsys.readouterr().out)["environment"]["errors"] == ["ENV_PYTEST_ABSENT"]
