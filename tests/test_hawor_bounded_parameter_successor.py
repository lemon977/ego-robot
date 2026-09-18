from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py"
SPEC = importlib.util.spec_from_file_location("hawor_bounded_parameter_successor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def prepare_runtime_guard_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    environment = tmp_path / "hawor-py310-v1"
    marker = environment / ".chaoyang-hardlink-snapshot/manifest.json.gz"
    smplx = environment / "lib/python3.10/site-packages/smplx/__init__.py"
    marker.parent.mkdir(parents=True)
    smplx.parent.mkdir(parents=True)
    marker.write_bytes(b"manifest")
    smplx.write_text("", encoding="utf-8")
    monkeypatch.setattr(MODULE, "EXPECTED_HAWOR_ENV_ROOT", environment)
    monkeypatch.setattr(MODULE.sys, "prefix", str(environment))
    monkeypatch.setenv("CHA0YANG_HAWOR_ENV_ROOT", str(environment))
    monkeypatch.setattr(MODULE.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(smplx)))
    return environment


def test_runtime_guard_accepts_published_launcher_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    environment = prepare_runtime_guard_fixture(monkeypatch, tmp_path)
    result = MODULE.validate_runtime_environment()
    assert result["status"] == "PASS_FIXED_HAWOR_RUNTIME"
    assert result["environment_root"] == str(environment.resolve())


def test_runtime_guard_rejects_system_python_before_input_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prepare_runtime_guard_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(MODULE.sys, "prefix", "/usr/local")
    with pytest.raises(RuntimeError, match="WRONG_RUNTIME_ENTRYPOINT"):
        MODULE.validate_runtime_environment()


def test_segments_never_bridge_missing_frames() -> None:
    mask = np.asarray([True, True, False, True, False, False, True, True, True])
    assert MODULE.contiguous_true_segments(mask) == [(0, 2), (3, 4), (6, 9)]


def test_whittaker_reduces_second_difference() -> None:
    values = np.asarray([[0.0], [1.0], [-0.8], [1.2], [0.0]])
    fitted = MODULE.whittaker(values, np.ones(5), strength=3.0)
    assert np.linalg.norm(np.diff(fitted, n=2, axis=0)) < np.linalg.norm(np.diff(values, n=2, axis=0))


def test_rotation_fit_and_backtrack_remain_on_so3() -> None:
    angles = np.asarray([0.0, 0.15, -0.12, 0.2, 0.05])
    matrices = Rotation.from_rotvec(np.column_stack((np.zeros(5), np.zeros(5), angles))).as_matrix()
    proposal = MODULE.smooth_rotations(matrices, np.ones(5), strength=1.5)
    candidate = MODULE.interpolate_rotations(matrices, proposal, alpha=0.35)
    products = np.transpose(candidate, (0, 2, 1)) @ candidate
    assert np.max(np.abs(products - np.eye(3))) < 1e-10
    assert np.min(np.linalg.det(candidate)) > 0.999999
