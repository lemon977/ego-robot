from __future__ import annotations

from pathlib import Path
import pytest

from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch1 import (
    _tree, isolate_verified, rollback_verified,
)


def test_changed_content_invalidates_delete_authorization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"old")
    identity = _tree(source)
    (source / "data.bin").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="OBJECT_CHANGED"):
        isolate_verified(source, tmp_path / "isolated", identity)
    assert source.is_dir()


def test_rollback_never_overwrites_new_same_path_content(tmp_path: Path) -> None:
    source, isolated = tmp_path / "source", tmp_path / "isolated"
    source.mkdir()
    (source / "data.bin").write_bytes(b"old")
    identity = _tree(source)
    isolate_verified(source, isolated, identity)
    source.mkdir()
    (source / "data.bin").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="ROLLBACK_BLOCKED"):
        rollback_verified(source, isolated, identity)
    assert (source / "data.bin").read_bytes() == b"new"
    assert (isolated / "data.bin").read_bytes() == b"old"
