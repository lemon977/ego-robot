"""Small same-filesystem atomic writers used by formal HumanEgo tools."""
from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Callable


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular_nofollow(path: Path) -> int:
    """Open one existing ordinary file without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"atomic output is not an ordinary file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _fsync_regular_tree(root: Path) -> None:
    """Reject links/special files and durably flush a privately built tree."""
    root_mode = os.lstat(root).st_mode
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise RuntimeError(f"atomic staging root is not an ordinary directory: {root}")
    directories = [root]
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise RuntimeError(f"atomic directory contains a non-directory: {child}")
            directories.append(child)
        for name in filenames:
            child = current_path / name
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"atomic directory contains a non-ordinary file: {child}")
            descriptor = _open_regular_nofollow(child)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Write, fsync and replace a file without exposing partial contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if os.path.lexists(temporary):
        raise FileExistsError(f"stale atomic-write temporary exists: {temporary}")
    try:
        writer(temporary)
        mode = os.lstat(temporary).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError(f"writer did not create an ordinary file: {temporary}")
        descriptor = _open_regular_nofollow(temporary)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    atomic_write(path, write)


def atomic_directory(path: str | Path, builder: Callable[[Path], None]) -> None:
    """Build a new directory privately and publish it with one atomic rename."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"immutable output directory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if os.path.lexists(staging):
        raise FileExistsError(f"stale atomic-directory staging exists: {staging}")
    staging.mkdir()
    try:
        builder(staging)
        _fsync_regular_tree(staging)
        os.replace(staging, path)
        _fsync_directory(path.parent)
    except BaseException:
        if os.path.lexists(staging) and (
            stat.S_ISLNK(os.lstat(staging).st_mode)
            or not stat.S_ISDIR(os.lstat(staging).st_mode)
        ):
            staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging)
        raise
