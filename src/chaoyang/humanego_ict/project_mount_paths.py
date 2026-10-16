"""Strict read aliases for the two known mounts of this project.

Signed references keep their original path and digest. Only the path used to
read an input changes; output paths must belong to the active project root.
"""
from __future__ import annotations

from pathlib import Path
import re


PROJECT_ALIASES = (
    Path("/path/to/chaoyang"),
    Path("/path/to/private_workspace/code/chaoyang"),
)


def _path(value):
    try:
        path = Path(value)
    except (TypeError, ValueError) as error:
        raise ValueError("PROJECT_ONLY_PATH_REQUIRED") from error
    if ".." in path.parts:
        raise ValueError("PROJECT_ONLY_PATH_REQUIRED")
    return path


def _no_symlinks(path):
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("PROJECT_SYMLINK_FORBIDDEN")


def _root(root):
    root = _path(root)
    if not root.is_absolute():
        raise ValueError("ABSOLUTE_CANONICAL_PROJECT_ROOT_REQUIRED")
    _no_symlinks(root)
    if not root.is_dir():
        raise ValueError("PROJECT_ROOT_DIRECTORY_REQUIRED")
    return root.resolve(strict=True)


def _check_alias_identity():
    """Reject two accessible mounts when one is a different project copy."""
    identities = []
    for alias in PROJECT_ALIASES:
        alias = _path(alias)
        if not alias.is_absolute():
            raise ValueError("ABSOLUTE_CANONICAL_PROJECT_ROOT_REQUIRED")
        _no_symlinks(alias)
        try:
            observed = alias.stat()
        except FileNotFoundError:
            continue
        if not alias.is_dir():
            raise ValueError("PROJECT_ROOT_DIRECTORY_REQUIRED")
        identities.append((observed.st_dev, observed.st_ino))
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("PROJECT_MOUNT_ALIAS_IDENTITY_CONFLICT")


def resolve_input_path(value, root, *, exists=True):
    """Locate an ordinary input without modifying its signed path.

    ``exists=False`` only translates and validates a path. The project root
    itself is a permitted directory for consumers constructing child paths.
    """
    root = _root(root)
    path = _path(value)
    _check_alias_identity()
    if not path.is_absolute():
        path = root / path
    elif not path.is_relative_to(root):
        for alias in PROJECT_ALIASES:
            if path.is_relative_to(alias):
                path = root / path.relative_to(alias)
                break
        else:
            raise ValueError("PROJECT_ONLY_PATH_REQUIRED")
    _no_symlinks(path)
    if not path.is_relative_to(root):
        raise ValueError("PROJECT_ONLY_PATH_REQUIRED")
    if exists and not (path.is_file() or path == root and path.is_dir()):
        raise ValueError("PROJECT_ORDINARY_INPUT_REQUIRED:" + str(path))
    return path


def resolve_output_path(value, root):
    """Validate a local output; historical input aliases never grant writes."""
    root = _root(root)
    path = _path(value)
    _check_alias_identity()
    if not path.is_absolute():
        path = root / path
    if not path.is_relative_to(root):
        raise ValueError("PROJECT_ONLY_PATH_REQUIRED")
    _no_symlinks(path)
    return path


def normalize_ref(ref, root):
    """Return a shallow local-path copy of a well-formed SHA reference."""
    if (not isinstance(ref, dict) or not {"path", "bytes", "sha256"} <= ref.keys()
            or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
            or type(ref["bytes"]) is not int or ref["bytes"] < 0
            or not isinstance(ref["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("EXACT_SHA_REFERENCE_REQUIRED")
    local = dict(ref)
    local["path"] = str(resolve_input_path(ref["path"], root))
    return local


def reference_equal(a, b, root):
    """Compare local identity and signed size/digest, preserving both refs."""
    left, right = normalize_ref(a, root), normalize_ref(b, root)
    return all(left[key] == right[key] for key in ("path", "bytes", "sha256"))
