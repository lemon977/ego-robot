from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import launch_clean_spatial12_bounded_v5 as launcher


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_local_authority_fixture(
    project: Path, *, authority_patch: dict[str, object] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    environment_root = project / launcher.ENVIRONMENT_ROOT_RELATIVE
    python_path = project / launcher.LOCAL_PYTHON_RELATIVE
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"synthetic-local-python")
    python_path.chmod(0o755)
    cv2_init = environment_root / "lib/python3.11/site-packages/cv2/__init__.py"
    cv2_init.parent.mkdir(parents=True)
    cv2_init.write_bytes(b"synthetic-local-cv2")
    snapshot = project / launcher.SNAPSHOT_MANIFEST_RELATIVE
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"synthetic-snapshot-manifest")
    clean_launcher = project / launcher.LOCAL_LAUNCHER_RELATIVE
    clean_launcher.parent.mkdir(parents=True, exist_ok=True)
    clean_launcher.write_bytes(b"#!/bin/sh\nexec local-python \"$@\"\n")
    clean_launcher.chmod(0o755)
    v4_path = project / launcher.V4_LAUNCHER_RELATIVE
    v4_path.write_bytes(b"synthetic-v4")

    launcher_sha = launcher.sha256(clean_launcher)
    snapshot_sha = launcher.sha256(snapshot)
    python_sha = launcher.sha256(python_path)
    cv2_sha = launcher.sha256(cv2_init)
    authority = {
        "schema_version": "chaoyang-clean-environment-authority-v1",
        "status": "PASS_LOCAL_CLEAN_ENVIRONMENT_PROMOTED",
        "canonical_runtime_authorized": True,
        "environment_id": launcher.ENVIRONMENT_ID,
        "launch": {
            "command": [launcher.LOCAL_LAUNCHER_RELATIVE.as_posix()],
            "activation_required": False,
            "fallback_to_external_conda_forbidden": True,
            "inherited_pythonpath_forbidden": True,
            "project_relative_environment_root": launcher.ENVIRONMENT_ROOT_RELATIVE.as_posix(),
            "project_relative_python": launcher.LOCAL_PYTHON_RELATIVE.as_posix(),
            "absolute_python": str(project / launcher.LOCAL_PYTHON_RELATIVE),
            "snapshot_manifest": launcher.SNAPSHOT_MANIFEST_RELATIVE.as_posix(),
        },
        "runtime": {
            "python_version": "3.11.15",
            "python_sha256": python_sha,
            "opencv_version": "4.11.0",
            "opencv_init_sha256": cv2_sha,
            "numpy_version": "1.26.4",
            "scipy_version": "1.16.2",
            "pillow_version": "11.3.0",
            "cuda_hidden_by_launcher": True,
        },
        "snapshot": {
            "source_role": "PROVENANCE_ONLY_NOT_A_RUNTIME_DEPENDENCY",
            "source_path": "/provenance-only/not-runtime",
            "manifest_sha256": snapshot_sha,
            "tree_content_sha256": launcher.TREE_CONTENT_SHA256,
            "post_publish_full_content_verify": "PASS_67676_UNIQUE_INODES",
        },
        "tools": {
            "launcher": {
                "path": launcher.LOCAL_LAUNCHER_RELATIVE.as_posix(),
                "sha256": launcher_sha,
            }
        },
        "frozen_v3_v4_unchanged": {
            "v4_launcher": {"sha256": launcher.sha256(v4_path)}
        },
    }
    if authority_patch:
        authority.update(authority_patch)
    authority_path = project / launcher.ENVIRONMENT_AUTHORITY_RELATIVE
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    authority_sha = launcher.sha256(authority_path)
    lock = {
        "schema_version": "chaoyang-clean-local-environment-lock-v1",
        "status": "PASS_LOCAL_CLEAN_ENVIRONMENT_PROMOTED",
        "environment_id": launcher.ENVIRONMENT_ID,
        "environment_root": launcher.ENVIRONMENT_ROOT_RELATIVE.as_posix(),
        "python": launcher.LOCAL_PYTHON_RELATIVE.as_posix(),
        "launcher": [launcher.LOCAL_LAUNCHER_RELATIVE.as_posix()],
        "authority": {
            "path": launcher.ENVIRONMENT_AUTHORITY_RELATIVE.as_posix(),
            "sha256": authority_sha,
        },
        "snapshot": {
            "manifest": launcher.SNAPSHOT_MANIFEST_RELATIVE.as_posix(),
            "manifest_sha256": snapshot_sha,
            "tree_content_sha256": launcher.TREE_CONTENT_SHA256,
        },
    }
    lock_path = project / launcher.ENVIRONMENT_LOCK_RELATIVE
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    hashes = {
        "authority": authority_sha,
        "lock": launcher.sha256(lock_path),
        "launcher": launcher_sha,
        "snapshot": snapshot_sha,
        "python": python_sha,
        "cv2": cv2_sha,
        "v4": launcher.sha256(v4_path),
    }
    paths = {
        "python": str(python_path),
        "cv2": str(cv2_init),
        "numpy": str(environment_root / "numpy.py"),
        "scipy": str(environment_root / "scipy.py"),
        "pillow": str(environment_root / "PIL.py"),
    }
    for key in ("numpy", "scipy", "pillow"):
        Path(paths[key]).write_bytes(key.encode())
    return hashes, paths


def validate_fixture(project: Path, hashes: dict[str, str]) -> dict[str, object]:
    original = {
        "LOCAL_PYTHON_SHA256": launcher.LOCAL_PYTHON_SHA256,
        "OPENCV_INIT_SHA256": launcher.OPENCV_INIT_SHA256,
        "V4_LAUNCHER_SHA256": launcher.V4_LAUNCHER_SHA256,
    }
    launcher.LOCAL_PYTHON_SHA256 = hashes["python"]
    launcher.OPENCV_INIT_SHA256 = hashes["cv2"]
    launcher.V4_LAUNCHER_SHA256 = hashes["v4"]
    try:
        return launcher.validate_local_environment_authority(
            project,
            expected_authority_sha=hashes["authority"],
            expected_lock_sha=hashes["lock"],
            expected_launcher_sha=hashes["launcher"],
            expected_snapshot_manifest_sha=hashes["snapshot"],
        )
    finally:
        for name, value in original.items():
            setattr(launcher, name, value)


def local_environment(project: Path) -> dict[str, str]:
    root = project / launcher.ENVIRONMENT_ROOT_RELATIVE
    return {
        "CHA0YANG_PROJECT_ROOT": str(project),
        "CHA0YANG_CLEAN_ENV_ROOT": str(root),
        "CONDA_PREFIX": str(root),
        "CONDA_DEFAULT_ENV": "chaoyang-clean-track4world-py311-v1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
    }


def test_real_promoted_authority_and_v4_pin_validate() -> None:
    evidence = launcher.validate_local_environment_authority()
    assert evidence["status"] == "PASS_PROJECT_LOCAL_CLEAN_ENVIRONMENT_AUTHORITY"
    assert evidence["external_runtime_dependency"] is False
    assert launcher.sha256(
        launcher.PROJECT / launcher.V4_LAUNCHER_RELATIVE
    ) == launcher.V4_LAUNCHER_SHA256


def test_synthetic_authority_validates_without_returning_provenance_source(
    tmp_path: Path,
) -> None:
    hashes, _ = make_local_authority_fixture(tmp_path)
    evidence = validate_fixture(tmp_path, hashes)
    serialized = json.dumps(evidence)
    assert evidence["environment_root"] == launcher.ENVIRONMENT_ROOT_RELATIVE.as_posix()
    assert "provenance-only" not in serialized
    assert evidence["external_runtime_dependency"] is False


def test_authority_semantic_drift_fails_even_with_matching_file_hash(
    tmp_path: Path,
) -> None:
    hashes, _ = make_local_authority_fixture(tmp_path)
    path = tmp_path / launcher.ENVIRONMENT_AUTHORITY_RELATIVE
    authority = json.loads(path.read_text(encoding="utf-8"))
    authority["canonical_runtime_authorized"] = False
    path.write_text(json.dumps(authority), encoding="utf-8")
    hashes["authority"] = launcher.sha256(path)
    lock_path = tmp_path / launcher.ENVIRONMENT_LOCK_RELATIVE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["authority"]["sha256"] = hashes["authority"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    hashes["lock"] = launcher.sha256(lock_path)
    with pytest.raises(launcher.CleanV5LaunchHold, match="authority contract drift"):
        validate_fixture(tmp_path, hashes)


@pytest.mark.parametrize("target", ["authority", "lock", "launcher"])
def test_authority_lock_or_launcher_hash_drift_fails(
    tmp_path: Path, target: str
) -> None:
    hashes, _ = make_local_authority_fixture(tmp_path)
    path = {
        "authority": tmp_path / launcher.ENVIRONMENT_AUTHORITY_RELATIVE,
        "lock": tmp_path / launcher.ENVIRONMENT_LOCK_RELATIVE,
        "launcher": tmp_path / launcher.LOCAL_LAUNCHER_RELATIVE,
    }[target]
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(launcher.CleanV5LaunchHold, match="hash drift"):
        validate_fixture(tmp_path, hashes)


def test_external_runtime_or_module_origin_is_rejected(tmp_path: Path) -> None:
    hashes, paths = make_local_authority_fixture(tmp_path)
    validate_fixture(tmp_path, hashes)
    environment = local_environment(tmp_path)
    origins = {key: paths[key] for key in ("cv2", "numpy", "scipy", "pillow")}
    with pytest.raises(launcher.CleanV5LaunchHold, match="external Python runtime"):
        launcher.validate_runtime_origins(
            tmp_path,
            environment,
            executable="/external/miniconda/bin/python",
            prefix=tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE,
            sys_path=[str(tmp_path)],
            module_origins=origins,
            versions=launcher.EXPECTED_RUNTIME_VERSIONS,
            ld_library_path=str(tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE / "lib"),
        )
    origins["opencv"] = "/external/site-packages/cv2/__init__.py"
    with pytest.raises(launcher.CleanV5LaunchHold, match="external or unavailable"):
        launcher.validate_runtime_origins(
            tmp_path,
            environment,
            executable=paths["python"],
            prefix=tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE,
            sys_path=[str(tmp_path)],
            module_origins=origins,
            versions=launcher.EXPECTED_RUNTIME_VERSIONS,
            ld_library_path=str(tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE / "lib"),
        )


def test_inherited_pythonpath_and_origin_drift_fail_closed(tmp_path: Path) -> None:
    hashes, paths = make_local_authority_fixture(tmp_path)
    validate_fixture(tmp_path, hashes)
    environment = local_environment(tmp_path)
    environment["PYTHONPATH"] = "/external/site-packages"
    with pytest.raises(launcher.CleanV5LaunchHold, match="PYTHONPATH"):
        launcher.validate_runtime_origins(
            tmp_path,
            environment,
            executable=paths["python"],
            prefix=tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE,
            sys_path=[str(tmp_path)],
            module_origins={key: paths[key] for key in ("cv2", "numpy", "scipy", "pillow")},
            versions=launcher.EXPECTED_RUNTIME_VERSIONS,
            ld_library_path=str(tmp_path / launcher.ENVIRONMENT_ROOT_RELATIVE / "lib"),
        )


def test_fixed_argv_uses_only_project_local_launcher_and_v5_staging() -> None:
    argv = launcher.fixed_base_argv()
    assert argv[0] == str(launcher.PROJECT / launcher.LOCAL_LAUNCHER_RELATIVE)
    assert argv[argv.index("--output") + 1] == str(
        launcher.PROJECT / launcher.STAGING_RELATIVE
    )
    assert argv[argv.index("--donor-authority") + 1] == str(
        launcher.PROJECT / launcher.v4.RUN_RELATIVE / "DONOR_ARCHIVE_AUTHORITY.json"
    )
    assert "producer.v5-local-env-held-capacity-staging" in " ".join(argv)
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "miniconda3" not in source
    assert "/external/" not in source


def test_frozen_public_capacity_and_clean_argv_use_project_local_python() -> None:
    manifest_path = (
        launcher.PROJECT
        / "tasks/chips/runs/clean/20260903_chips001_clean_spatial12_launch_v5"
        / "CPU_PREPARED_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    argv = manifest["public_argv"]
    local_launcher = str(launcher.PROJECT / launcher.LOCAL_LAUNCHER_RELATIVE)
    assert argv[0] == local_launcher
    separator = argv.index("--")
    assert argv[separator + 1] == local_launcher
    assert "/usr/local/bin/python" not in manifest["public_command"]
