"""Fail-closed validation for frozen HumanEgo split and sidecar inputs.

The project treats a split manifest as a complete contract, not merely as a
convenient list of sessions.  Callers must therefore verify its partition,
declared sidecar root and every consumed sidecar digest before using data.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping


SPLIT_ROLES = ("train", "validation", "test")
WITHHELD_FINAL_TEST_SPLIT_SCHEMA = "humanego-eligible68-split-v2"
SESSION_ID_PATTERN = re.compile(r"grap_a_cap_[0-9]{3}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUNTIME_CHECKPOINT_AUTHORITY_SCHEMA = "humanego-runtime-checkpoint-authority-v1"
TRAINING_COMPLETE_SCHEMA = "humanego-training-complete-v1"
ISOLATED_EVALUATION_METRIC_KEYS = frozenset({
    "batches",
    "frames",
    "done_acc",
    "pos_err_k1_m",
    "pos_err_kK_m",
    "rot_err_k1_deg",
    "rot_err_kK_deg",
    "pos_err_w_m",
    "rot_err_w_deg",
    "grasp_f1_k1",
    "grasp_f1_kK",
    "grasp_f1_w",
    "joint_mae_k1_normalized",
    "joint_mae_kK_normalized",
    "joint_mae_w_normalized",
    "zero_state_ratio",
})


def _withholds_final_test_sidecars(split: Mapping[str, Any]) -> bool:
    return split.get("schema_version") == WITHHELD_FINAL_TEST_SPLIT_SCHEMA


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON semantics instead of platform-dependent pretty formatting."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinks in an existing absolute path, including its parents."""
    path = path.absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise FileNotFoundError(current) from None
        if stat.S_ISLNK(mode):
            raise ValueError(f"formal path contains a symlink component: {current}")


def _open_directory_chain_nofollow(path: Path, *, label: str) -> int:
    """Hold one absolute directory by walking every component with O_NOFOLLOW."""
    path = path.absolute()
    if path != Path(os.path.normpath(path)):
        raise ValueError(f"{label} path is not canonical: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be an ordinary directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_beneath_directory(
    root_descriptor: int,
    relative_path: Path,
    *,
    label: str,
) -> tuple[int, int]:
    """Open an ordinary file beneath a held root without traversing a symlink."""
    if relative_path.is_absolute() or not relative_path.parts or any(
        component in {"", ".", ".."} for component in relative_path.parts
    ):
        raise ValueError(f"{label} relative path is invalid: {relative_path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.dup(root_descriptor)
    try:
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            relative_path.parts[-1],
            file_flags,
            dir_fd=parent_descriptor,
        )
        return parent_descriptor, descriptor
    except BaseException:
        os.close(parent_descriptor)
        raise


def _verify_open_regular_path_binding(
    lexical_path: Path,
    parent_descriptor: int,
    file_state: os.stat_result,
    *,
    label: str,
    phase: str,
) -> None:
    """Require both the held parent entry and live absolute path to bind one FD."""
    try:
        held_entry = os.stat(
            lexical_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _reject_symlink_components(lexical_path)
        live_entry = os.lstat(lexical_path)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} path changed {phase}: {lexical_path}") from error
    expected_identity = (file_state.st_dev, file_state.st_ino)
    if (
        (held_entry.st_dev, held_entry.st_ino) != expected_identity
        or (live_entry.st_dev, live_entry.st_ino) != expected_identity
    ):
        raise ValueError(f"{label} path changed {phase}: {lexical_path}")


def _verify_open_directory_path_binding(
    lexical_path: Path,
    directory_state: os.stat_result,
    *,
    label: str,
    phase: str,
) -> None:
    """Require a held ordinary directory to remain bound to its absolute path."""
    try:
        _reject_symlink_components(lexical_path)
        live_entry = os.lstat(lexical_path)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} path changed {phase}: {lexical_path}") from error
    if (
        not stat.S_ISDIR(live_entry.st_mode)
        or (live_entry.st_dev, live_entry.st_ino)
        != (directory_state.st_dev, directory_state.st_ino)
    ):
        raise ValueError(f"{label} path changed {phase}: {lexical_path}")


def _open_absolute_regular_nofollow(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, int, int]:
    """Open an absolute ordinary-file candidate through a stable dirfd chain."""
    lexical_path = Path(path).absolute()
    if lexical_path != Path(os.path.normpath(lexical_path)):
        raise ValueError(f"{label} path is not canonical: {lexical_path}")
    relative_path = Path(*lexical_path.parts[1:])
    anchor_descriptor = _open_directory_chain_nofollow(
        Path(lexical_path.anchor),
        label=f"{label} filesystem anchor",
    )
    try:
        parent_descriptor, descriptor = _open_regular_beneath_directory(
            anchor_descriptor,
            relative_path,
            label=label,
        )
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(
            f"{label} must be an ordinary file with no symlink component: "
            f"{lexical_path}"
        ) from error
    finally:
        os.close(anchor_descriptor)
    return lexical_path, parent_descriptor, descriptor


def _read_ordinary_file_bytes(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, bytes]:
    """Read one canonical ordinary file once through an ``O_NOFOLLOW`` FD.

    Path validation, file identity, byte capture and stability checks all refer
    to the same open file description.  Callers that hash and parse a manifest
    must parse the returned bytes rather than reopening ``path``.
    """
    lexical_path, parent_descriptor, descriptor = _open_absolute_regular_nofollow(
        path,
        label=label,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be an ordinary file: {lexical_path}")
        if before.st_nlink != 1:
            raise ValueError(f"{label} must not be hard-linked: {lexical_path}")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            before,
            label=label,
            phase="while opening",
        )
        resolved = lexical_path.resolve(strict=True)
        if lexical_path != resolved:
            raise ValueError(f"{label} path is not canonical: {lexical_path}")

        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label} changed while being read: {lexical_path}")
        encoded = b"".join(blocks)
        if len(encoded) != after.st_size:
            raise ValueError(f"{label} changed while being read: {lexical_path}")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            after,
            label=label,
            phase="while being read",
        )
        return resolved, encoded
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def read_ordinary_file_bytes(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, bytes]:
    """Public same-FD reader for manifests that do not yet have a reference."""
    return _read_ordinary_file_bytes(path, label=label)


def read_isolated_evaluation_metrics(
    path: str | Path,
    *,
    label: str,
) -> dict[str, int | float]:
    """Read and validate one worker metric object from a single FD snapshot."""
    _, encoded = _read_ordinary_file_bytes(path, label=label)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != ISOLATED_EVALUATION_METRIC_KEYS:
        raise ValueError(f"{label} metric schema drift")
    for key in ("batches", "frames"):
        value = payload[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} {key} must be a positive integer")
    for key in ISOLATED_EVALUATION_METRIC_KEYS - {"batches", "frames"}:
        value = payload[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} {key} must be finite")
    return dict(payload)


def read_file_reference(
    reference: Mapping[str, Any],
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> tuple[Path, bytes]:
    """Validate and read a reference without reopening between hash and parse."""
    with open_verified_file_reference(
        reference,
        allowed_roots=allowed_roots,
        label=label,
    ) as (path, stream):
        encoded = stream.read()
    return path, encoded


def validate_file_reference(
    reference: Mapping[str, Any],
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> Path:
    """Validate a canonical path/bytes/SHA reference under one approved root."""
    path, _ = read_file_reference(
        reference,
        allowed_roots=allowed_roots,
        label=label,
    )
    return path


def validate_directory_path(
    path: str | Path,
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> Path:
    """Validate a canonical ordinary directory with no symlink components."""
    lexical_path = Path(path).absolute()
    _reject_symlink_components(lexical_path)
    if not stat.S_ISDIR(os.lstat(lexical_path).st_mode):
        raise ValueError(f"{label} must be an ordinary directory: {lexical_path}")
    resolved = lexical_path.resolve(strict=True)
    if lexical_path != resolved:
        raise ValueError(f"{label} path is not canonical: {lexical_path}")
    roots = [Path(root).resolve(strict=True) for root in allowed_roots]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} escapes approved roots: {resolved}")
    return resolved


def sha256_file(path: str | Path) -> str:
    lexical_path, parent_descriptor, descriptor = _open_absolute_regular_nofollow(
        path,
        label="SHA256 source",
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"SHA256 source must be a singly-linked ordinary file: {lexical_path}")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            before,
            label="SHA256 source",
            phase="while opening",
        )
        value = hashlib.sha256()
        observed_bytes = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed_bytes += len(block)
            value.update(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"
        )
        if (
            observed_bytes != after.st_size
            or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        ):
            raise ValueError(f"SHA256 source changed while being read: {lexical_path}")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            after,
            label="SHA256 source",
            phase="while being read",
        )
        return value.hexdigest()
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def file_reference(path: str | Path) -> dict[str, Any]:
    """Return the canonical ordinary-file identity used by formal manifests."""
    lexical_path, parent_descriptor, descriptor = _open_absolute_regular_nofollow(
        path,
        label="formal reference",
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"formal reference must be a singly-linked ordinary file: {lexical_path}"
            )
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            before,
            label="formal reference",
            phase="while opening",
        )
        resolved = lexical_path.resolve(strict=True)
        if lexical_path != resolved:
            raise ValueError(f"formal reference path is not canonical: {lexical_path}")
        digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed_bytes += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"
        )
        if (
            observed_bytes != after.st_size
            or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        ):
            raise ValueError(f"formal reference changed while being read: {lexical_path}")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            after,
            label="formal reference",
            phase="while being read",
        )
        return {
            "path": str(resolved),
            "bytes": observed_bytes,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def open_verified_file_reference(
    reference: Mapping[str, Any],
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> Iterator[tuple[Path, BinaryIO]]:
    """Prehash, consume and posthash one referenced file through one stable FD."""
    if not isinstance(reference, Mapping):
        raise ValueError(f"{label} reference must be an object")
    raw_path = reference.get("path")
    expected_bytes = reference.get("bytes")
    expected_sha256 = reference.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} reference path must be canonical and absolute")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        raise ValueError(f"{label} reference bytes must be positive")
    if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ValueError(f"{label} reference SHA256 is invalid")
    lexical_path = Path(raw_path)
    if lexical_path != Path(os.path.normpath(lexical_path)):
        raise ValueError(f"{label} path is not canonical: {lexical_path}")
    roots: list[Path] = []
    for value in allowed_roots:
        lexical_root = Path(value).absolute()
        _reject_symlink_components(lexical_root)
        root = lexical_root.resolve(strict=True)
        if lexical_root != root:
            raise ValueError(f"{label} approved root is not canonical: {lexical_root}")
        roots.append(root)
    containing_roots = [root for root in roots if lexical_path.is_relative_to(root)]
    if not containing_roots:
        raise ValueError(f"{label} reference escapes approved roots: {lexical_path}")
    approved_root = max(containing_roots, key=lambda root: len(root.parts))
    root_descriptor = _open_directory_chain_nofollow(
        approved_root,
        label=f"{label} approved root",
    )
    root_state = os.fstat(root_descriptor)
    try:
        _reject_symlink_components(approved_root)
        current_root = os.lstat(approved_root)
        if (current_root.st_dev, current_root.st_ino) != (
            root_state.st_dev,
            root_state.st_ino,
        ):
            raise ValueError(f"{label} approved root changed while opening")
        parent_descriptor, descriptor = _open_regular_beneath_directory(
            root_descriptor,
            lexical_path.relative_to(approved_root),
            label=label,
        )
    except FileNotFoundError:
        os.close(root_descriptor)
        raise
    except OSError as error:
        os.close(root_descriptor)
        raise ValueError(
            f"{label} path contains a symlink component or changed while opening: "
            f"{lexical_path}"
        ) from error
    except BaseException:
        os.close(root_descriptor)
        raise
    stream: BinaryIO | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"{label} must be a singly-linked ordinary file; hard-linked files "
                f"are forbidden: {lexical_path}"
            )
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            before,
            label=label,
            phase="while opening",
        )
        resolved = lexical_path.resolve(strict=True)
        if lexical_path != resolved:
            raise ValueError(f"{label} path is not canonical: {lexical_path}")

        def digest_open_description() -> tuple[int, str]:
            os.lseek(descriptor, 0, os.SEEK_SET)
            value = hashlib.sha256()
            size = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                size += len(block)
                value.update(block)
            return size, value.hexdigest()

        pre_size, pre_digest = digest_open_description()
        prehash = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"
        )
        if pre_size != expected_bytes:
            raise ValueError(
                f"{label} byte-size mismatch: {pre_size} != {expected_bytes}"
            )
        if pre_digest != expected_sha256:
            raise ValueError(
                f"{label} SHA256 mismatch: {pre_digest} != {expected_sha256}"
            )
        if pre_size != prehash.st_size or any(
            getattr(before, field) != getattr(prehash, field)
            for field in stable_fields
        ):
            raise ValueError(f"{label} changed during preverification")
        _verify_open_regular_path_binding(
            lexical_path,
            parent_descriptor,
            prehash,
            label=label,
            phase="before consumption",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(os.dup(descriptor), "rb")
        try:
            yield resolved, stream
        finally:
            stream.close()
            stream = None
            post_size, post_digest = digest_open_description()
            after = os.fstat(descriptor)
            if (
                post_size != expected_bytes
                or post_digest != expected_sha256
                or any(
                    getattr(prehash, field) != getattr(after, field)
                    for field in stable_fields
                )
            ):
                raise ValueError(f"{label} changed during consumption")
            _verify_open_regular_path_binding(
                lexical_path,
                parent_descriptor,
                after,
                label=label,
                phase="during consumption",
            )
            _reject_symlink_components(approved_root)
            current_root = os.lstat(approved_root)
            if (current_root.st_dev, current_root.st_ino) != (
                root_state.st_dev,
                root_state.st_ino,
            ):
                raise ValueError(f"{label} approved root changed during consumption")
    finally:
        if stream is not None:
            stream.close()
        os.close(descriptor)
        os.close(parent_descriptor)
        os.close(root_descriptor)


def load_verified_torch_checkpoint(
    reference: Mapping[str, Any],
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
    map_location: str = "cpu",
) -> tuple[Path, Any]:
    """Deserialize only bytes already verified from the same stable FD."""
    import torch

    with open_verified_file_reference(
        reference,
        allowed_roots=allowed_roots,
        label=label,
    ) as (path, stream):
        payload = torch.load(stream, map_location=map_location, weights_only=True)
    return path, payload


def runtime_checkpoint_authority_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).absolute()
    return checkpoint_path.with_name(f"{checkpoint_path.name}.authority.json")


def _runtime_checkpoint_authority_material(
    checkpoint_path: str | Path,
    *,
    run_manifest_reference: Mapping[str, Any],
    dataset_stats_reference: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], bytes]:
    checkpoint_path = Path(checkpoint_path).absolute()
    checkpoint_reference = file_reference(checkpoint_path)
    authority_path = runtime_checkpoint_authority_path(checkpoint_path)
    parent = authority_path.parent.absolute()
    if parent != Path(os.path.normpath(parent)):
        raise ValueError("runtime checkpoint authority parent is not canonical")
    if authority_path != parent / authority_path.name:
        raise ValueError("runtime checkpoint authority path is not canonical")
    payload = {
        "schema_version": RUNTIME_CHECKPOINT_AUTHORITY_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "checkpoint_ref": checkpoint_reference,
        "run_manifest_ref": dict(run_manifest_reference),
        "dataset_stats_ref": dict(dataset_stats_reference),
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return authority_path, parent, checkpoint_reference, encoded


def _authority_staging_name(authority_path: Path) -> str:
    return f".{authority_path.name}.staging"


def _directory_entry_state(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        state = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(state.st_mode):
        raise ValueError(f"runtime checkpoint authority transaction entry is not regular: {name}")
    return state


def _read_authority_transaction_entry(
    parent_descriptor: int,
    name: str,
    *,
    expected_links: int,
) -> tuple[bytes, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        before = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != expected_links
            or (before.st_dev, before.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            raise ValueError("runtime checkpoint authority transaction identity drift")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
        entry_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stable_fields = (
            "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"
        )
        encoded = b"".join(blocks)
        if (
            len(encoded) != after.st_size
            or any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            )
            or (after.st_dev, after.st_ino)
            != (entry_after.st_dev, entry_after.st_ino)
        ):
            raise ValueError("runtime checkpoint authority transaction changed while read")
        return encoded, after
    finally:
        os.close(descriptor)


def _recover_runtime_checkpoint_authority_transaction(
    authority_path: Path,
    parent: Path,
    parent_descriptor: int,
    parent_state: os.stat_result,
    expected_encoded: bytes,
) -> bool:
    """Finish only an exact fsynced staging/link transaction after a crash."""
    staging_name = _authority_staging_name(authority_path)
    staging_state = _directory_entry_state(parent_descriptor, staging_name)
    if staging_state is None:
        return False
    authority_state = _directory_entry_state(parent_descriptor, authority_path.name)
    if authority_state is None:
        encoded, staging_state = _read_authority_transaction_entry(
            parent_descriptor,
            staging_name,
            expected_links=1,
        )
        if encoded != expected_encoded:
            raise ValueError("staged runtime checkpoint authority content drift")
        _verify_open_directory_path_binding(
            parent,
            parent_state,
            label="runtime checkpoint authority parent",
            phase="before staged recovery publication",
        )
        os.link(
            staging_name,
            authority_path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
        authority_state = _directory_entry_state(
            parent_descriptor,
            authority_path.name,
        )
    if authority_state is None or (
        authority_state.st_dev,
        authority_state.st_ino,
    ) != (staging_state.st_dev, staging_state.st_ino):
        raise ValueError("runtime checkpoint authority transaction paths diverged")
    encoded, linked_state = _read_authority_transaction_entry(
        parent_descriptor,
        authority_path.name,
        expected_links=2,
    )
    if encoded != expected_encoded or linked_state.st_nlink != 2:
        raise ValueError("linked runtime checkpoint authority content drift")
    _verify_open_directory_path_binding(
        parent,
        parent_state,
        label="runtime checkpoint authority parent",
        phase="before staged transaction cleanup",
    )
    os.unlink(staging_name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    final_encoded, final_state = _read_authority_transaction_entry(
        parent_descriptor,
        authority_path.name,
        expected_links=1,
    )
    if final_encoded != expected_encoded or (
        final_state.st_dev,
        final_state.st_ino,
    ) != (linked_state.st_dev, linked_state.st_ino):
        raise ValueError("published runtime checkpoint authority content drift")
    _verify_open_directory_path_binding(
        parent,
        parent_state,
        label="runtime checkpoint authority parent",
        phase="after staged transaction recovery",
    )
    _reject_symlink_components(authority_path)
    return True


def write_runtime_checkpoint_authority(
    checkpoint_path: str | Path,
    *,
    run_manifest_reference: Mapping[str, Any],
    dataset_stats_reference: Mapping[str, Any],
) -> Path:
    """Transactionally and exclusively publish one checkpoint authority."""
    authority_path, parent, checkpoint_reference, encoded = (
        _runtime_checkpoint_authority_material(
            checkpoint_path,
            run_manifest_reference=run_manifest_reference,
            dataset_stats_reference=dataset_stats_reference,
        )
    )
    staging_name = _authority_staging_name(authority_path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = _open_directory_chain_nofollow(
        parent,
        label="runtime checkpoint authority parent",
    )
    parent_state = os.fstat(parent_descriptor)
    descriptor: int | None = None
    created = False
    published = False
    staging_identity: tuple[int, int] | None = None
    try:
        _verify_open_directory_path_binding(
            parent,
            parent_state,
            label="runtime checkpoint authority parent",
            phase="before creation",
        )
        if _directory_entry_state(parent_descriptor, authority_path.name) is not None:
            raise FileExistsError(f"runtime checkpoint authority exists: {authority_path}")
        if _directory_entry_state(parent_descriptor, staging_name) is not None:
            raise FileExistsError(
                f"runtime checkpoint authority staging exists: {parent / staging_name}"
            )
        with open_verified_file_reference(
            checkpoint_reference,
            allowed_roots=[parent],
            label="runtime checkpoint",
        ):
            descriptor = os.open(
                staging_name,
                flags,
                0o444,
                dir_fd=parent_descriptor,
            )
            created = True
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("short write while creating checkpoint authority")
                offset += written
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            state = os.fstat(descriptor)
            staging_identity = (state.st_dev, state.st_ino)
            held_entry = os.stat(
                staging_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _verify_open_directory_path_binding(
                parent,
                parent_state,
                label="runtime checkpoint authority parent",
                phase="during staging creation",
            )
            if (
                not stat.S_ISREG(state.st_mode)
                or state.st_nlink != 1
                or state.st_size != len(encoded)
                or (state.st_dev, state.st_ino)
                != (held_entry.st_dev, held_entry.st_ino)
            ):
                raise ValueError(
                    "runtime checkpoint authority staging changed while being written"
                )
            os.fsync(parent_descriptor)
            published = _recover_runtime_checkpoint_authority_transaction(
                authority_path,
                parent,
                parent_descriptor,
                parent_state,
                encoded,
            )
            if not published:
                raise RuntimeError("runtime checkpoint authority staging was not published")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created and not published:
            try:
                current_staging = _directory_entry_state(
                    parent_descriptor,
                    staging_name,
                )
                current_authority = _directory_entry_state(
                    parent_descriptor,
                    authority_path.name,
                )
                if (
                    current_staging is not None
                    and current_authority is None
                    and staging_identity
                    == (current_staging.st_dev, current_staging.st_ino)
                ):
                    os.unlink(staging_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return authority_path


def read_runtime_checkpoint_authority(
    checkpoint_path: str | Path,
    *,
    authority_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).absolute()
    authority_path = runtime_checkpoint_authority_path(checkpoint_path)
    if authority_reference is None:
        _, encoded = _read_ordinary_file_bytes(
            authority_path,
            label="runtime checkpoint authority",
        )
    else:
        if not isinstance(authority_reference, Mapping):
            raise ValueError("runtime checkpoint authority reference must be an object")
        if authority_reference.get("path") != str(authority_path):
            raise ValueError("runtime checkpoint authority reference path drift")
        _, encoded = read_file_reference(
            authority_reference,
            allowed_roots=[checkpoint_path.parent],
            label="frozen runtime checkpoint authority",
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime checkpoint authority is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "immutable",
        "no_fallback",
        "checkpoint_ref",
        "run_manifest_ref",
        "dataset_stats_ref",
    }:
        raise ValueError("runtime checkpoint authority schema drift")
    if (
        payload.get("schema_version") != RUNTIME_CHECKPOINT_AUTHORITY_SCHEMA
        or payload.get("immutable") is not True
        or payload.get("no_fallback") is not True
    ):
        raise ValueError("runtime checkpoint authority is not immutable/no-fallback")
    checkpoint_reference = payload.get("checkpoint_ref")
    if (
        not isinstance(checkpoint_reference, Mapping)
        or checkpoint_reference.get("path") != str(checkpoint_path)
    ):
        raise ValueError("runtime checkpoint authority binds another checkpoint")
    for field in ("run_manifest_ref", "dataset_stats_ref"):
        if not isinstance(payload.get(field), Mapping):
            raise ValueError(f"runtime checkpoint authority lacks {field}")
    return payload


def ensure_runtime_checkpoint_authority(
    checkpoint_path: str | Path,
    *,
    run_manifest_reference: Mapping[str, Any],
    dataset_stats_reference: Mapping[str, Any],
) -> Path:
    """Create or exactly reuse a crash-surviving checkpoint authority."""
    checkpoint_path = Path(checkpoint_path).absolute()
    authority_path, parent, checkpoint_reference, expected_encoded = (
        _runtime_checkpoint_authority_material(
            checkpoint_path,
            run_manifest_reference=run_manifest_reference,
            dataset_stats_reference=dataset_stats_reference,
        )
    )
    parent_descriptor = _open_directory_chain_nofollow(
        parent,
        label="runtime checkpoint authority recovery parent",
    )
    try:
        parent_state = os.fstat(parent_descriptor)
        _verify_open_directory_path_binding(
            parent,
            parent_state,
            label="runtime checkpoint authority recovery parent",
            phase="before recovery",
        )
        with open_verified_file_reference(
            checkpoint_reference,
            allowed_roots=[parent],
            label="runtime checkpoint recovery checkpoint",
        ):
            recovered = _recover_runtime_checkpoint_authority_transaction(
                authority_path,
                parent,
                parent_descriptor,
                parent_state,
                expected_encoded,
            )
    finally:
        os.close(parent_descriptor)
    if not recovered:
        try:
            write_runtime_checkpoint_authority(
                checkpoint_path,
                run_manifest_reference=run_manifest_reference,
                dataset_stats_reference=dataset_stats_reference,
            )
        except FileExistsError:
            # A prior completion attempt may have published the exact current
            # authority before this process reached its exclusive create.
            pass
    authority_reference = file_reference(authority_path)
    authority = read_runtime_checkpoint_authority(
        checkpoint_path,
        authority_reference=authority_reference,
    )
    expected_checkpoint = file_reference(checkpoint_path)
    if authority.get("checkpoint_ref") != expected_checkpoint:
        raise ValueError("existing checkpoint authority binds stale checkpoint bytes")
    if authority.get("run_manifest_ref") != dict(run_manifest_reference):
        raise ValueError("existing checkpoint authority binds another run manifest")
    if authority.get("dataset_stats_ref") != dict(dataset_stats_reference):
        raise ValueError("existing checkpoint authority binds other dataset statistics")
    return authority_path


def load_frozen_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    embodiment: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a frozen best checkpoint only after its freeze authority verifies it."""
    checkpoint_path = Path(checkpoint_path).absolute()
    freeze_path = checkpoint_path.parent / "freeze_manifest.json"
    _, freeze_encoded = _read_ordinary_file_bytes(
        freeze_path,
        label="frozen checkpoint authority",
    )
    try:
        freeze = json.loads(freeze_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen checkpoint authority is not valid JSON") from error
    if not isinstance(freeze, dict):
        raise ValueError("frozen checkpoint authority must be an object")
    if (
        freeze.get("schema_version") != "humanego-frozen-best-v2"
        or freeze.get("immutable") is not True
        or freeze.get("no_fallback") is not True
        or freeze.get("status") != "frozen"
        or freeze.get("embodiment") != embodiment
    ):
        raise ValueError("frozen checkpoint authority status/schema drift")
    checkpoint_reference = freeze.get("checkpoint_ref")
    if (
        not isinstance(checkpoint_reference, Mapping)
        or checkpoint_reference.get("path") != str(checkpoint_path)
    ):
        raise ValueError("frozen checkpoint authority binds another checkpoint")
    loaded_path, payload = load_verified_torch_checkpoint(
        checkpoint_reference,
        allowed_roots=[checkpoint_path.parent],
        label="frozen best checkpoint",
    )
    if loaded_path != checkpoint_path or not isinstance(payload, dict):
        raise ValueError("frozen best checkpoint payload/path is invalid")
    return payload, dict(checkpoint_reference), freeze


def _session_hash(split: Mapping[str, Any], embodiment: str, session_id: str) -> str:
    try:
        expected = split["sessions"][session_id]["embodiments"][embodiment][
            "sidecar_sha256"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"split has no frozen {embodiment} sidecar hash for {session_id}"
        ) from error
    if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
        raise ValueError(f"invalid sidecar SHA256 for {session_id}: {expected!r}")
    return expected


def load_frozen_split(
    split_path: str | Path,
    embodiment: str,
    *,
    sidecar_root: str | Path | None = None,
    verify_sidecars: bool = False,
    verify_roles: Iterable[str] | None = None,
    split_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a complete, disjoint frozen session split.

    When ``sidecar_root`` is provided it must resolve to the exact root declared
    by the manifest. ``verify_sidecars`` hashes all roles unless
    ``verify_roles`` explicitly narrows phase-2 reads. Role-limited verification
    keeps eligible68 final-test bytes unopened during training and development
    evaluation. An empty test role is legal only when explicitly unavailable.
    """
    split_path = Path(split_path).absolute()
    if split_payload is None:
        split_path, encoded = _read_ordinary_file_bytes(
            split_path,
            label="frozen split",
        )
        try:
            split = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid frozen split JSON: {split_path}") from error
    else:
        # The eligible68 wrapper has already read, hashed and parsed this exact
        # payload.  Reopening the path here would reintroduce a phase1/phase2
        # replacement window and could admit forbidden session identities.
        try:
            split = json.loads(
                json.dumps(split_payload, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("validated frozen split payload is not JSON") from error
    if not isinstance(split, dict):
        raise ValueError("frozen split must be a JSON object")

    if verify_roles is None:
        roles_to_verify = frozenset(SPLIT_ROLES) if verify_sidecars else frozenset()
    else:
        roles_to_verify = frozenset(verify_roles)
        unknown_roles = sorted(roles_to_verify - set(SPLIT_ROLES))
        if unknown_roles:
            raise ValueError(f"unknown sidecar verification roles: {unknown_roles}")
        if roles_to_verify and not verify_sidecars:
            raise ValueError("verify_roles requires verify_sidecars=True")

    role_sets: dict[str, set[str]] = {}
    ordered_sessions: list[str] = []
    for role in SPLIT_ROLES:
        values = split.get(role)
        empty_test_is_explicit = (
            role == "test"
            and values == []
            and split.get("independent_test_policy") == "UNAVAILABLE_NOT_REPURPOSED"
        )
        if not isinstance(values, list) or (not values and not empty_test_is_explicit):
            raise ValueError(f"split role {role!r} must be a non-empty list")
        if any(
            not isinstance(value, str) or not SESSION_ID_PATTERN.fullmatch(value)
            for value in values
        ):
            raise ValueError(f"split role {role!r} contains an invalid session id")
        if len(values) != len(set(values)):
            raise ValueError(f"split role {role!r} contains duplicate sessions")
        role_sets[role] = set(values)
        ordered_sessions.extend(values)

    for index, left in enumerate(SPLIT_ROLES):
        for right in SPLIT_ROLES[index + 1 :]:
            overlap = sorted(role_sets[left] & role_sets[right])
            if overlap:
                raise ValueError(f"split leakage between {left}/{right}: {overlap}")

    sessions = split.get("sessions")
    if not isinstance(sessions, dict):
        raise ValueError("split sessions metadata must be an object")
    frozen_ids = set(ordered_sessions)
    if set(sessions) != frozen_ids:
        missing = sorted(frozen_ids - set(sessions))
        extra = sorted(set(sessions) - frozen_ids)
        raise ValueError(
            f"split sessions metadata mismatch; missing={missing}, extra={extra}"
        )

    declared_value = split.get("sidecar_root")
    if not isinstance(declared_value, str) or not declared_value:
        raise ValueError("split must declare a non-empty sidecar_root")
    declared_root = Path(declared_value)
    if not declared_root.is_absolute():
        declared_root = split_path.parent / declared_root
    _reject_symlink_components(declared_root.absolute())
    declared_root = declared_root.resolve(strict=True)
    if sidecar_root is not None and Path(sidecar_root).resolve() != declared_root:
        raise ValueError(
            "configured sidecar root differs from the frozen split: "
            f"{Path(sidecar_root).resolve()} != {declared_root}"
        )

    for role in SPLIT_ROLES:
        for session_id in split[role]:
            expected = _session_hash(split, embodiment, session_id)
            if role not in roles_to_verify:
                continue
            sidecar = declared_root / embodiment / session_id / "sidecar.npz"
            _reject_symlink_components(sidecar)
            if not stat.S_ISREG(os.lstat(sidecar).st_mode):
                raise ValueError(f"frozen sidecar must be an ordinary file: {sidecar}")
            observed = sha256_file(sidecar)
            if observed != expected:
                raise ValueError(
                    f"frozen sidecar SHA256 mismatch for {session_id}: "
                    f"{observed} != {expected}"
                )
    return split


def sessions_for_role(split: Mapping[str, Any], role: str) -> list[str]:
    if role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {role!r}")
    values = split.get(role)
    empty_test_is_explicit = (
        role == "test"
        and values == []
        and split.get("independent_test_policy") == "UNAVAILABLE_NOT_REPURPOSED"
    )
    if not isinstance(values, list) or (not values and not empty_test_is_explicit):
        raise ValueError(f"split role {role!r} is missing or empty")
    return list(values)


def validate_run_manifest(
    run_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    embodiment: str,
    *,
    require_runtime_fields: bool = False,
) -> None:
    """Require exact train/validation/test sidecar lineage in a formal run."""
    if run_manifest.get("embodiment") != embodiment:
        raise ValueError("run manifest embodiment mismatch")
    for field, role in (
        ("train_sessions", "train"),
        ("validation_sessions", "validation"),
    ):
        observed = run_manifest.get(field)
        expected = sessions_for_role(split, role)
        if observed != expected:
            raise ValueError(f"run manifest {field} differs from frozen {role}")

    expected_training = {
        session_id: _session_hash(split, embodiment, session_id)
        for role in ("train", "validation")
        for session_id in sessions_for_role(split, role)
    }
    if run_manifest.get("sidecars") != expected_training:
        raise ValueError("run manifest training sidecar lineage is incomplete or changed")
    expected_evaluation = {
        session_id: _session_hash(split, embodiment, session_id)
        for session_id in sessions_for_role(split, "test")
    }
    if run_manifest.get("evaluation_sidecars") != expected_evaluation:
        raise ValueError("run manifest test sidecar lineage is incomplete or changed")
    if require_runtime_fields:
        if run_manifest.get("schema_version") != "humanego-formal-run-v2":
            raise ValueError("formal Trainer requires humanego-formal-run-v2")
        split_ref = run_manifest.get("split_ref")
        if not isinstance(split_ref, Mapping) or not isinstance(
            split_ref.get("path"), str
        ):
            raise ValueError("formal run manifest has no split_ref")
        split_path, split_encoded = read_file_reference(
            split_ref,
            allowed_roots=[Path(split_ref["path"]).parent],
            label="formal split",
        )
        try:
            referenced_split = json.loads(split_encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("formal split reference is not valid JSON") from error
        if referenced_split != split:
            raise ValueError("run manifest split_ref content differs from loaded split")
        if run_manifest.get("split_sha256") != split_ref.get("sha256"):
            raise ValueError("legacy split_sha256 differs from split_ref")
        config_ref = run_manifest.get("config_ref")
        if not isinstance(config_ref, Mapping):
            raise ValueError("formal run manifest has no config_ref")
        if run_manifest.get("config_sha256") != config_ref.get("sha256"):
            raise ValueError("legacy config_sha256 differs from config_ref")
        resolved_config = run_manifest.get("resolved_config")
        if not isinstance(resolved_config, Mapping):
            raise ValueError("formal run manifest has no resolved_config")
        if run_manifest.get("resolved_config_sha256") != canonical_json_sha256(
            resolved_config
        ):
            raise ValueError("run manifest resolved_config SHA256 mismatch")
        sidecar_root_value = split.get("sidecar_root")
        if (
            not isinstance(sidecar_root_value, str)
            or not Path(sidecar_root_value).is_absolute()
        ):
            raise ValueError("formal split must declare a canonical sidecar root")
        sidecar_root = validate_directory_path(
            sidecar_root_value,
            allowed_roots=[Path(sidecar_root_value).anchor],
            label="frozen sidecar root",
        )
        referenced_roles = (
            ("train", "validation")
            if _withholds_final_test_sidecars(split)
            else SPLIT_ROLES
        )
        all_session_ids = [
            session_id
            for role in referenced_roles
            for session_id in sessions_for_role(split, role)
        ]
        sidecar_refs = run_manifest.get("sidecar_refs")
        if not isinstance(sidecar_refs, Mapping) or set(sidecar_refs) != set(
            all_session_ids
        ):
            raise ValueError("run manifest sidecar references are incomplete or changed")
        for session_id in all_session_ids:
            sidecar_path = validate_file_reference(
                sidecar_refs[session_id],
                allowed_roots=[sidecar_root],
                label=f"{session_id} sidecar",
            )
            expected_path = (
                sidecar_root / embodiment / session_id / "sidecar.npz"
            ).resolve(strict=True)
            if sidecar_path != expected_path:
                raise ValueError(
                    f"run manifest sidecar path differs from frozen split: {session_id}"
                )
            if sidecar_refs[session_id].get("sha256") != _session_hash(
                split, embodiment, session_id
            ):
                raise ValueError(
                    f"run manifest sidecar SHA256 differs from frozen split: {session_id}"
                )
        production_value = split.get("production_root")
        if not isinstance(production_value, str) or not Path(production_value).is_absolute():
            raise ValueError("frozen split must declare a canonical production root")
        production_root = validate_directory_path(
            production_value,
            allowed_roots=[Path(production_value).anchor],
            label="frozen production root",
        )
        adapter_paths = run_manifest.get("adapter_paths")
        expected_adapters = {
            session_id: str(
                validate_directory_path(
                    split["sessions"][session_id]["adapter"],
                    allowed_roots=[production_root],
                    label=f"{session_id} frozen adapter",
                )
            )
            for role in ("train", "validation")
            for session_id in sessions_for_role(split, role)
        }
        if adapter_paths != expected_adapters:
            raise ValueError("run manifest adapter paths differ from the frozen split")
        out_dir = resolved_config.get("out_dir")
        if not isinstance(out_dir, str):
            raise ValueError("resolved config has no formal run directory")
        validate_run_root_binding(out_dir, run_manifest)


def validate_run_root_binding(
    run_directory: str | Path,
    run_manifest: Mapping[str, Any],
    *,
    expected_run_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate the caller-approved root and exact directory of one formal run."""
    root_value = run_manifest.get("run_root")
    directory_value = run_manifest.get("run_directory")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("formal run manifest must declare an absolute run_root")
    if not isinstance(directory_value, str) or not Path(directory_value).is_absolute():
        raise ValueError("formal run manifest must declare an absolute run_directory")
    root = validate_directory_path(
        root_value,
        allowed_roots=[Path(root_value).anchor],
        label="formal run root",
    )
    if root.parent == Path(root.anchor):
        raise ValueError("formal run root must name a project namespace")
    if expected_run_root is not None:
        expected = validate_directory_path(
            expected_run_root,
            allowed_roots=[Path(expected_run_root).absolute().anchor],
            label="caller-approved run root",
        )
        if root != expected:
            raise ValueError("formal run root differs from the caller-approved root")
    directory = validate_directory_path(
        run_directory,
        allowed_roots=[root],
        label="formal training run directory",
    )
    declared_directory = Path(directory_value)
    if declared_directory != directory:
        raise ValueError("formal run directory differs from the run manifest")
    if directory.parent != root:
        raise ValueError("formal run directory must be an immediate child of run_root")
    return root, directory


def build_formal_run_manifest(
    *,
    split: Mapping[str, Any],
    split_path: str | Path,
    config_path: str | Path,
    resolved_config: Mapping[str, Any],
    embodiment: str,
    sidecar_root: str | Path,
    production_root: str | Path,
    run_root: str | Path,
    run_directory: str | Path,
    base_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the one canonical ``humanego-formal-run-v2`` field set.

    The function does not authorize a selector or a formal run.  It only binds
    the already-reviewed split/config/sidecars/adapters into one manifest.
    """
    protected = {
        "schema_version",
        "embodiment",
        "split_ref",
        "split_sha256",
        "config_ref",
        "config_sha256",
        "resolved_config",
        "resolved_config_sha256",
        "train_sessions",
        "validation_sessions",
        "sidecars",
        "evaluation_sidecars",
        "sidecar_refs",
        "adapter_paths",
        "run_root",
        "run_directory",
    }
    overlap = protected.intersection(base_fields)
    if overlap:
        raise ValueError(f"base fields may not override formal fields: {sorted(overlap)}")
    split_path, split_encoded = _read_ordinary_file_bytes(
        split_path,
        label="formal split",
    )
    config_path, config_encoded = _read_ordinary_file_bytes(
        config_path,
        label="reviewed config",
    )
    split_reference = {
        "path": str(split_path),
        "bytes": len(split_encoded),
        "sha256": hashlib.sha256(split_encoded).hexdigest(),
    }
    config_reference = {
        "path": str(config_path),
        "bytes": len(config_encoded),
        "sha256": hashlib.sha256(config_encoded).hexdigest(),
    }
    try:
        disk_split = json.loads(split_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal split is not valid JSON") from error
    if disk_split != split:
        raise ValueError("supplied split differs from split_path content")
    sidecar_root = validate_directory_path(
        sidecar_root,
        allowed_roots=[Path(sidecar_root).absolute().anchor],
        label="formal sidecar root",
    )
    declared_sidecar = split.get("sidecar_root")
    if not isinstance(declared_sidecar, str) or Path(declared_sidecar).resolve() != sidecar_root:
        raise ValueError("formal sidecar root differs from split")
    production_root = validate_directory_path(
        production_root,
        allowed_roots=[Path(production_root).absolute().anchor],
        label="formal production root",
    )
    declared_production = split.get("production_root")
    if (
        not isinstance(declared_production, str)
        or Path(declared_production).resolve() != production_root
    ):
        raise ValueError("formal production root differs from split")
    if not Path(run_root).is_absolute() or not Path(run_directory).is_absolute():
        raise ValueError("formal run root and directory must be absolute")
    run_root = validate_directory_path(
        run_root,
        allowed_roots=[Path(run_root).absolute().anchor],
        label="formal run root",
    )
    run_directory = validate_directory_path(
        run_directory,
        allowed_roots=[run_root],
        label="formal run directory",
    )
    if run_root.parent == Path(run_root.anchor):
        raise ValueError("formal run root must name a project namespace")
    if run_directory.parent != run_root:
        raise ValueError("formal run directory must be an immediate child of run_root")

    train_sessions = sessions_for_role(split, "train")
    validation_sessions = sessions_for_role(split, "validation")
    test_sessions = sessions_for_role(split, "test")
    training_sessions = train_sessions + validation_sessions
    referenced_sessions = (
        training_sessions
        if _withholds_final_test_sidecars(split)
        else training_sessions + test_sessions
    )
    sidecar_refs: dict[str, dict[str, Any]] = {}
    for session_id in referenced_sessions:
        sidecar = sidecar_root / embodiment / session_id / "sidecar.npz"
        reference = file_reference(sidecar)
        if reference["sha256"] != _session_hash(split, embodiment, session_id):
            raise ValueError(f"sidecar differs from frozen split: {session_id}")
        sidecar_refs[session_id] = reference
    adapter_paths = {
        session_id: str(
            validate_directory_path(
                split["sessions"][session_id]["adapter"],
                allowed_roots=[production_root],
                label=f"{session_id} formal adapter",
            )
        )
        for session_id in train_sessions + validation_sessions
    }
    resolved = json.loads(json.dumps(resolved_config, allow_nan=False))
    manifest = {
        **dict(base_fields),
        "schema_version": "humanego-formal-run-v2",
        "embodiment": embodiment,
        "split_ref": split_reference,
        "split_sha256": split_reference["sha256"],
        "config_ref": config_reference,
        "config_sha256": config_reference["sha256"],
        "resolved_config": resolved,
        "resolved_config_sha256": canonical_json_sha256(resolved),
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "sidecars": {
            session_id: sidecar_refs[session_id]["sha256"]
            for session_id in training_sessions
        },
        "evaluation_sidecars": {
            session_id: _session_hash(split, embodiment, session_id)
            for session_id in test_sessions
        },
        "sidecar_refs": sidecar_refs,
        "adapter_paths": adapter_paths,
        "run_root": str(run_root),
        "run_directory": str(run_directory),
    }
    return manifest


def validate_resolved_config(
    resolved_config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    *,
    config_root: str | Path,
) -> Path:
    """Bind the reviewed YAML and the complete post-CLI effective config."""
    config_path, config_encoded = read_file_reference(
        run_manifest.get("config_ref", {}),
        allowed_roots=[config_root],
        label="reviewed config",
    )
    expected = run_manifest.get("resolved_config")
    if not isinstance(expected, Mapping):
        raise ValueError("run manifest must embed the canonical resolved config")
    observed_value = json.loads(json.dumps(resolved_config, allow_nan=False))
    expected_value = json.loads(json.dumps(expected, allow_nan=False))
    if observed_value != expected_value:
        raise ValueError("effective Trainer config differs from the run manifest")
    observed_sha256 = canonical_json_sha256(observed_value)
    if run_manifest.get("resolved_config_sha256") != observed_sha256:
        raise ValueError("canonical resolved config SHA256 mismatch")
    if run_manifest.get("config_sha256") != hashlib.sha256(config_encoded).hexdigest():
        raise ValueError("run manifest reviewed-config SHA256 mismatch")
    return config_path


def validate_formal_runtime_contract(
    resolved_config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    embodiment: str,
    *,
    config_root: str | Path,
) -> Path:
    """Use the same v2 field validation in launcher tests and all consumers."""
    validate_run_manifest(
        run_manifest,
        split,
        embodiment,
        require_runtime_fields=True,
    )
    return validate_resolved_config(
        resolved_config,
        run_manifest,
        config_root=config_root,
    )


def validate_checkpoint_bundle(
    checkpoint_path: str | Path,
    payload: Mapping[str, Any],
    split_path: str | Path,
    embodiment: str,
    *,
    return_dataset_stats: bool = False,
) -> Path | tuple[Path, dict[str, Any]]:
    """Validate one frozen bundle and optionally return its verified stats.

    The optional payload is parsed from the exact ``dataset_stats.json`` bytes
    whose digest is checked below.  Formal evaluators must use that payload
    instead of reopening the path after validation.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_symlink():
        raise ValueError(f"frozen checkpoint must be an ordinary file: {checkpoint_path}")
    checkpoint_path = checkpoint_path.resolve()
    bundle = checkpoint_path.parent
    freeze_path = bundle / "freeze_manifest.json"
    run_manifest_path = bundle / "run_manifest.json"
    stats_path = bundle / "dataset_stats.json"
    for path in (freeze_path, run_manifest_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise ValueError(f"frozen checkpoint metadata must be ordinary: {path}")
    try:
        _, freeze_encoded = _read_ordinary_file_bytes(
            freeze_path,
            label="freeze manifest",
        )
        _, run_manifest_encoded = _read_ordinary_file_bytes(
            run_manifest_path,
            label="frozen run manifest",
        )
        _, stats_encoded = _read_ordinary_file_bytes(
            stats_path,
            label="frozen dataset statistics",
        )
        _, split_encoded = _read_ordinary_file_bytes(
            split_path,
            label="frozen checkpoint split",
        )
        freeze = json.loads(freeze_encoded)
        disk_manifest = json.loads(run_manifest_encoded)
        dataset_stats = json.loads(stats_encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen checkpoint bundle JSON") from error
    if not isinstance(dataset_stats, dict):
        raise ValueError("frozen dataset statistics must be a JSON object")
    if freeze.get("status") != "frozen" or freeze.get("embodiment") != embodiment:
        raise ValueError("freeze manifest status/embodiment mismatch")
    checks = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_sha256": hashlib.sha256(split_encoded).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(run_manifest_encoded).hexdigest(),
        "dataset_stats_sha256": hashlib.sha256(stats_encoded).hexdigest(),
    }
    for field, observed in checks.items():
        if freeze.get(field) != observed:
            raise ValueError(f"freeze manifest {field} mismatch")
    payload_stats_sha256 = payload.get("dataset_stats_sha256")
    if (
        not isinstance(payload_stats_sha256, str)
        or not SHA256_PATTERN.fullmatch(payload_stats_sha256)
        or payload_stats_sha256 != checks["dataset_stats_sha256"]
    ):
        raise ValueError(
            "checkpoint payload/dataset statistics SHA256 mismatch"
        )
    if payload.get("run_manifest") != disk_manifest:
        raise ValueError("checkpoint/on-disk run manifest mismatch")
    if return_dataset_stats:
        return stats_path, dataset_stats
    return stats_path


def validate_session_paths(
    paths: Iterable[str | Path],
    expected_ids: list[str],
    *,
    expected_paths: Mapping[str, str | Path] | None = None,
    allowed_root: str | Path | None = None,
) -> None:
    """Validate adapter paths passed from the reviewed launcher to the trainer."""
    lexical_paths = [Path(path).absolute() for path in paths]
    for path in lexical_paths:
        _reject_symlink_components(path)
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            raise ValueError(f"invalid HumanEgo adapter directory: {path}")
    paths = [path.resolve(strict=True) for path in lexical_paths]
    observed_ids = [path.parent.name for path in paths]
    if observed_ids != expected_ids:
        raise ValueError(
            f"session paths differ from run manifest: {observed_ids} != {expected_ids}"
        )
    for path in paths:
        if path.name != "09_humanego_adapter" or not path.is_dir():
            raise ValueError(f"invalid HumanEgo adapter directory: {path}")
    if allowed_root is not None:
        root = Path(allowed_root).resolve(strict=True)
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError("session adapter path escapes the approved production root")
    if expected_paths is not None:
        expected = [Path(expected_paths[session_id]).resolve(strict=True) for session_id in expected_ids]
        if paths != expected:
            raise ValueError("session adapter paths differ from frozen manifest paths")
