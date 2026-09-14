from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from tools.probe_eevee_next_headless_throughput import (
    EXPECTED_EGL_VENDOR_BYTES,
    EXPECTED_EGL_VENDOR_SHA256,
    FULL_FRAME_COUNT,
    ProbeError,
    RESOLUTIONS,
    create_owned_output,
    extrapolate_seconds,
    fifty_hour_impact,
    read_ordinary,
    verify_file,
)


def test_resolution_and_glvnd_contract_are_exact() -> None:
    assert RESOLUTIONS == (("240x320", 320, 240), ("1280x960", 1280, 960))
    project = Path(__file__).resolve().parents[1]
    verify_file(
        project / "tools/glvnd/10_nvidia.json",
        EXPECTED_EGL_VENDOR_BYTES,
        EXPECTED_EGL_VENDOR_SHA256,
    )


def test_t17_extrapolation_and_50h_impact_are_exact() -> None:
    assert extrapolate_seconds(0.1) == pytest.approx(3231.8)
    impact = fifty_hour_impact()
    assert impact["t17_cycles_serial_hours"] == pytest.approx(
        10.0 * FULL_FRAME_COUNT / 3600.0
    )
    assert impact["t17_cycles_meets_50h"] is False
    assert impact["t17_cycles_exceeds_50h_by_hours"] == pytest.approx(
        10.0 * FULL_FRAME_COUNT / 3600.0 - 50.0
    )


def test_extrapolation_rejects_bool_nonfinite_and_nonpositive() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ProbeError):
            extrapolate_seconds(value)
    with pytest.raises(ProbeError, match="exact int"):
        extrapolate_seconds(0.1, True)


def test_verify_file_checks_ordinary_bytes_and_sha(tmp_path: Path) -> None:
    payload = b"pinned-robot-asset"
    path = tmp_path / "asset.stl"
    path.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    identity = verify_file(path, len(payload), expected_sha)
    assert identity.bytes == len(payload)
    assert identity.sha256 == expected_sha
    with pytest.raises(ProbeError, match="identity drift"):
        verify_file(path, len(payload) + 1, expected_sha)


def test_read_ordinary_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ProbeError, match="cannot open ordinary file"):
        read_ordinary(link)


def test_owned_output_is_new_contained_and_o_excl(tmp_path: Path) -> None:
    (tmp_path / "_run").mkdir()
    token = "a" * 64
    component, owner = create_owned_output(tmp_path, "eevee_probe_v1", token)
    assert component == tmp_path / "_run/eevee_probe_v1/eevee_throughput_probe"
    assert owner.is_file() and not owner.is_symlink()
    assert os.stat(owner).st_size > 0
    with pytest.raises(FileExistsError):
        create_owned_output(tmp_path, "eevee_probe_v1", token)


def test_owned_output_rejects_escape_and_symlinked_run_root(tmp_path: Path) -> None:
    real = tmp_path / "real_run"
    real.mkdir()
    (tmp_path / "_run").symlink_to(real, target_is_directory=True)
    with pytest.raises(ProbeError, match="ordinary directory"):
        create_owned_output(tmp_path, "eevee_probe_v1", "b" * 64)
    with pytest.raises(ProbeError, match="invalid run id"):
        create_owned_output(tmp_path, "../escape", "b" * 64)
