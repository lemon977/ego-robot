from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chaoyang.ops.verify_three_stream_handoff_v1 import (
    ARCHIVE_TASK_PREFIX,
    OLD_TASK_PREFIX,
    current_robot30_path,
    validate_declared,
)


def test_robot30_path_rewrite_is_exact() -> None:
    historical = OLD_TASK_PREFIX + "control/run/video.mp4"
    assert str(current_robot30_path(historical)) == ARCHIVE_TASK_PREFIX + "control/run/video.mp4"


def test_robot30_path_rewrite_rejects_unexpected_prefix() -> None:
    with pytest.raises(RuntimeError, match="unexpected historical prefix"):
        current_robot30_path("/tmp/video.mp4")


def test_declared_artifact_requires_bytes_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"verified")
    declared = {
        "bytes": len(b"verified"),
        "sha256": hashlib.sha256(b"verified").hexdigest(),
    }
    assert validate_declared(path, declared) == []
    path.write_bytes(b"drifted")
    errors = validate_declared(path, declared)
    assert any("sha mismatch" in error for error in errors)
