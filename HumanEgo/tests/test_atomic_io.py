from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.atomic_io import atomic_directory, atomic_write, atomic_write_json


def test_atomic_json_replaces_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("old", encoding="utf-8")
    atomic_write_json(path, {"status": "complete"})
    assert path.read_text(encoding="utf-8") == '{\n  "status": "complete"\n}\n'
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_failed_atomic_write_preserves_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "result.bin"
    path.write_bytes(b"old")

    def fail(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        atomic_write(path, fail)
    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_atomic_directory_never_publishes_partial_tree(tmp_path: Path) -> None:
    destination = tmp_path / "frozen"

    def fail(staging: Path) -> None:
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        atomic_directory(destination, fail)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))

    atomic_directory(
        destination,
        lambda staging: (staging / "complete").write_text("ok", encoding="utf-8"),
    )
    assert (destination / "complete").read_text(encoding="utf-8") == "ok"


def test_atomic_write_rejects_writer_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    destination = tmp_path / "result.bin"

    with pytest.raises(RuntimeError, match="ordinary file"):
        atomic_write(destination, lambda temporary: temporary.symlink_to(outside))
    assert not destination.exists()
    assert outside.read_bytes() == b"outside"


def test_atomic_directory_rejects_symlink_member(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    destination = tmp_path / "published"

    def build(staging: Path) -> None:
        (staging / "link.bin").symlink_to(outside)

    with pytest.raises(RuntimeError, match="non-ordinary file"):
        atomic_directory(destination, build)
    assert not destination.exists()


def test_atomic_directory_rejects_symlink_staging_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "published"

    def replace_root_with_symlink(staging: Path) -> None:
        staging.rmdir()
        staging.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="staging root"):
        atomic_directory(destination, replace_root_with_symlink)

    assert not destination.exists()
    assert outside.is_dir()
    assert not destination.parent.joinpath(f".{destination.name}.tmp-{os.getpid()}").exists()
