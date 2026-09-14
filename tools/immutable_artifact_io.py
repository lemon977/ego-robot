"""Small no-follow immutable artifact reader used by Robot asset loaders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


@dataclass(frozen=True)
class VerifiedBytes:
    path: Path
    payload: bytes
    bytes: int
    sha256: str

    def evidence_ref(self) -> dict[str, Any]:
        return {"path": str(self.path), "bytes": self.bytes, "sha256": self.sha256}


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def read_bytes_nofollow(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    allowed_root: Path | str,
) -> VerifiedBytes:
    root = Path(allowed_root).resolve(strict=True)
    candidate = Path(path)
    resolved = candidate.resolve(strict=True)
    if not _inside(resolved, root):
        raise RuntimeError(f"artifact escapes allowed root: {candidate}")
    if candidate.is_symlink() or resolved.is_symlink():
        raise RuntimeError(f"symlink artifact forbidden: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"regular file required: {resolved}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block); digest.update(block); total += len(block)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if expected_bytes is not None and total != int(expected_bytes):
        raise RuntimeError(f"artifact byte count drift: {resolved}")
    if expected_sha256 is not None and observed != str(expected_sha256):
        raise RuntimeError(f"artifact SHA256 drift: {resolved}")
    return VerifiedBytes(resolved, b"".join(chunks), total, observed)


def read_json_nofollow(
    path: Path | str,
    *,
    allowed_root: Path | str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[Any, VerifiedBytes]:
    record = read_bytes_nofollow(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        allowed_root=allowed_root,
    )
    try:
        value = json.loads(record.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid immutable JSON: {record.path}: {exc}") from exc
    return value, record
