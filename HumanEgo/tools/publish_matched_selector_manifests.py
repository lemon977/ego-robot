#!/usr/bin/env python3
"""Publish the currently defined RAW/ROBOT_RGB matched-selector twins.

This publisher is deliberately limited to the two product lines accepted by
``utils.source_contract``.  Adding another arm requires a new, reviewed source
contract schema; it must not be inferred from a directory or a caller field.

The two inputs are already-shaped ``humanego-selector-manifest-v1`` record
manifests.  This tool verifies their exact references and every nested source
reference, then publishes canonical copies and the paired-kept ledger.  It does
not confer trust on a RobotRGB authority: the authority must exactly equal the
owner-reviewed pin compiled into ``source_contract``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "HumanEgo") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "HumanEgo"))

from utils import source_contract  # noqa: E402
from utils.frozen_contract import open_verified_file_reference  # noqa: E402


RAW_NAME = source_contract.PAIRED_RAW_SELECTOR_NAME
ROBOT_NAME = source_contract.PAIRED_ROBOT_SELECTOR_NAME
PAIRED_NAME = source_contract.PAIRED_TERMINAL_NAME
PAIRED_PRIVATE_NAME = source_contract.PAIRED_COMPLETE_PAYLOAD_ALIAS_NAME
TWIN_PRODUCT_LINES = ("RAW", "ROBOT_RGB")
H50_WINDOW_LENGTH = 51

_REF_KEYS = {"path", "bytes", "sha256"}
_RAW_SELECTOR_KEYS = {
    "schema_version",
    "immutable",
    "no_fallback",
    "product_line",
    "image_name",
    "artifact_root",
    "selector_root",
    "sessions",
}
_ROBOT_SELECTOR_KEYS = _RAW_SELECTOR_KEYS | {"robot_rgb_lineage_authority_ref"}


@dataclass(frozen=True)
class FileSnapshot:
    reference: dict[str, Any]
    allowed_roots: tuple[str, ...]
    label: str
    binding: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class OutputRecord:
    reference: dict[str, Any]
    binding: tuple[int, int, int, int]


@dataclass
class Preflight:
    project_root: Path
    split_payload: dict[str, Any]
    raw_payload: dict[str, Any]
    robot_payload: dict[str, Any]
    snapshots: list[FileSnapshot]
    reviewed_authority_ref: dict[str, Any]
    total_frames: int


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_json_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_exact_reference(
    value: Mapping[str, Any] | None, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REF_KEYS:
        raise ValueError(f"{label} must be an exact path/bytes/SHA256 reference")
    reference = dict(value)
    path_value = reference.get("path")
    size = reference.get("bytes")
    digest = reference.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(path_value)
    if path != Path(os.path.normpath(path)):
        raise ValueError(f"{label} path is not canonical")
    forbidden = {
        part.casefold() for part in path.parts
    } & source_contract.FORBIDDEN_SOURCE_COMPONENTS
    if forbidden:
        raise ValueError(
            f"{label} enters a forbidden source namespace: {sorted(forbidden)}"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{label} bytes must be a positive integer")
    if (
        not isinstance(digest, str)
        or source_contract.SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return reference


def _fd_binding(state: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
        int(state.st_nlink),
    )


def _read_snapshot(
    reference_value: Mapping[str, Any] | None,
    *,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> tuple[bytes, FileSnapshot]:
    reference = _require_exact_reference(reference_value, label=label)
    roots = tuple(str(Path(root).resolve(strict=True)) for root in allowed_roots)
    with open_verified_file_reference(
        reference,
        allowed_roots=[Path(root) for root in roots],
        label=label,
    ) as (_, stream):
        before = os.fstat(stream.fileno())
        encoded = stream.read()
        after = os.fstat(stream.fileno())
        if _fd_binding(before) != _fd_binding(after):
            raise ValueError(f"{label} changed on its held file descriptor")
        binding = _fd_binding(after)
    return encoded, FileSnapshot(reference, roots, label, binding)


def _reverify_snapshots(snapshots: Sequence[FileSnapshot]) -> None:
    for snapshot in snapshots:
        _, observed = _read_snapshot(
            snapshot.reference,
            allowed_roots=snapshot.allowed_roots,
            label=f"{snapshot.label} terminal recheck",
        )
        if observed.binding != snapshot.binding:
            raise ValueError(f"{snapshot.label} named identity changed after preflight")


def _session_order() -> tuple[str, ...]:
    frozen = tuple(
        session_id
        for session_id in source_contract.ELIGIBLE68_ORDER
        if session_id in source_contract.ELIGIBLE68
    )
    if len(frozen) == len(source_contract.ELIGIBLE68) and set(frozen) == set(
        source_contract.ELIGIBLE68
    ):
        return frozen
    return tuple(sorted(source_contract.ELIGIBLE68))


def _reviewed_authority() -> dict[str, Any]:
    value = source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF
    if value is None:
        raise RuntimeError(
            f"{source_contract.ROBOT_RGB_LINEAGE_HOLD}: owner-reviewed exact "
            "RobotRGB aggregate authority pin is not configured"
        )
    return _require_exact_reference(value, label="owner-reviewed RobotRGB authority")


def _validate_selector_top_level(
    payload: Mapping[str, Any], *, product_line: str
) -> None:
    expected = _RAW_SELECTOR_KEYS if product_line == "RAW" else _ROBOT_SELECTOR_KEYS
    if set(payload) != expected:
        raise ValueError(f"{product_line} selector-record input schema fields drift")
    if payload.get("product_line") != product_line:
        raise ValueError(f"{product_line} selector-record product line drift")


def _register_owner(
    snapshot: FileSnapshot,
    *,
    owner: str,
    path_owners: dict[str, str],
    inode_owners: dict[tuple[int, int], str],
) -> None:
    path = str(snapshot.reference["path"])
    inode = snapshot.binding[:2]
    previous_path = path_owners.get(path)
    if previous_path is not None and previous_path != owner:
        raise ValueError(
            f"nested source path aliases logical records: {previous_path}, {owner}"
        )
    previous_inode = inode_owners.get(inode)
    if previous_inode is not None and previous_inode != owner:
        raise ValueError(
            f"nested source inode aliases logical records: {previous_inode}, {owner}"
        )
    path_owners[path] = owner
    inode_owners[inode] = owner


def _preflight(
    *,
    split_reference: Mapping[str, Any],
    raw_selector_records_reference: Mapping[str, Any],
    robot_selector_records_reference: Mapping[str, Any],
    embodiment: str,
    project_root: Path,
) -> Preflight:
    # This trust decision is intentionally first: a missing owner pin must
    # produce HOLD before any output namespace can be created.
    reviewed_authority = _reviewed_authority()
    snapshots: list[FileSnapshot] = []

    split_encoded, split_snapshot = _read_snapshot(
        split_reference,
        allowed_roots=[project_root],
        label="eligible68-v2 split",
    )
    split_payload = _parse_json_object(split_encoded, label="eligible68-v2 split")
    observed_split = source_contract.validate_eligible68_split_phase1(
        split_snapshot.reference["path"],
        embodiment,
        project_root=project_root,
    )
    if observed_split != split_payload:
        raise ValueError("eligible68 split path changed around phase-1 validation")
    snapshots.append(split_snapshot)

    raw_encoded, raw_snapshot = _read_snapshot(
        raw_selector_records_reference,
        allowed_roots=[project_root],
        label="RAW selector-record input",
    )
    robot_encoded, robot_snapshot = _read_snapshot(
        robot_selector_records_reference,
        allowed_roots=[project_root],
        label="RobotRGB selector-record input",
    )
    raw_payload = _parse_json_object(raw_encoded, label="RAW selector-record input")
    robot_payload = _parse_json_object(
        robot_encoded, label="RobotRGB selector-record input"
    )
    _validate_selector_top_level(raw_payload, product_line="RAW")
    _validate_selector_top_level(robot_payload, product_line="ROBOT_RGB")

    declared_authority = _require_exact_reference(
        robot_payload.get("robot_rgb_lineage_authority_ref"),
        label="RobotRGB selector-record authority",
    )
    if declared_authority != reviewed_authority:
        raise RuntimeError(
            f"{source_contract.ROBOT_RGB_LINEAGE_HOLD}: selector-record authority "
            "differs from the owner-reviewed exact pin"
        )

    raw_root = Path(str(raw_payload["artifact_root"]))
    robot_root = Path(str(robot_payload["artifact_root"]))
    raw_validated = source_contract.validate_selector_manifest_reference(
        raw_snapshot.reference,
        artifact_root=raw_root,
        project_root=project_root,
    )
    robot_validated = source_contract.validate_selector_manifest_reference(
        robot_snapshot.reference,
        artifact_root=robot_root,
        project_root=project_root,
    )
    if raw_validated["product_line"] != "RAW":
        raise ValueError("RAW selector-record input did not validate as RAW")
    if robot_validated["product_line"] != "ROBOT_RGB":
        raise ValueError("RobotRGB selector-record input did not validate as RobotRGB")
    snapshots.extend((raw_snapshot, robot_snapshot))

    authority_path = Path(str(reviewed_authority["path"]))
    _, authority_snapshot = _read_snapshot(
        reviewed_authority,
        allowed_roots=[authority_path.parent],
        label="owner-reviewed RobotRGB aggregate authority",
    )
    snapshots.append(authority_snapshot)

    path_owners: dict[str, str] = {}
    inode_owners: dict[tuple[int, int], str] = {}
    raw_records = raw_validated["normalized_records"]
    robot_records = robot_validated["normalized_records"]
    total_frames = 0
    for session_id in _session_order():
        frame_count = source_contract.ELIGIBLE68_FRAME_COUNTS[session_id]
        total_frames += frame_count
        for frame_index in range(frame_count):
            frame_key = f"{frame_index:05d}"
            logical_key = f"{session_id}/{frame_key}"
            raw = raw_records[session_id][frame_key]
            robot = robot_records[session_id][frame_key]
            raw_metadata = _require_exact_reference(
                raw["metadata"], label=f"{logical_key} RAW metadata"
            )
            robot_metadata = _require_exact_reference(
                robot["metadata"], label=f"{logical_key} RobotRGB metadata"
            )
            if raw_metadata != robot_metadata:
                raise ValueError(f"RAW/Robot metadata differs: {logical_key}")
            raw_image = _require_exact_reference(
                raw["image"], label=f"{logical_key} RAW image"
            )
            robot_image = _require_exact_reference(
                robot["image"], label=f"{logical_key} RobotRGB output"
            )
            if (
                raw_image == robot_image
                or raw_image["path"] == robot_image["path"]
                or raw_image["sha256"] == robot_image["sha256"]
            ):
                raise ValueError(f"RAW/Robot image domains alias: {logical_key}")

            _, metadata_snapshot = _read_snapshot(
                raw_metadata,
                allowed_roots=[project_root],
                label=f"{logical_key} metadata",
            )
            _, raw_image_snapshot = _read_snapshot(
                raw_image,
                allowed_roots=[Path(str(raw_validated["selector_root"]))],
                label=f"{logical_key} RAW image",
            )
            _, robot_image_snapshot = _read_snapshot(
                robot_image,
                allowed_roots=[Path(str(robot_validated["selector_root"]))],
                label=f"{logical_key} RobotRGB output",
            )
            _register_owner(
                metadata_snapshot,
                owner=f"metadata:{logical_key}",
                path_owners=path_owners,
                inode_owners=inode_owners,
            )
            _register_owner(
                raw_image_snapshot,
                owner=f"RAW:{logical_key}",
                path_owners=path_owners,
                inode_owners=inode_owners,
            )
            _register_owner(
                robot_image_snapshot,
                owner=f"ROBOT_RGB:{logical_key}",
                path_owners=path_owners,
                inode_owners=inode_owners,
            )
            snapshots.extend(
                (metadata_snapshot, raw_image_snapshot, robot_image_snapshot)
            )

    _reverify_snapshots(snapshots)
    if (
        source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF is None
        or dict(source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF)
        != reviewed_authority
    ):
        raise RuntimeError(
            f"{source_contract.ROBOT_RGB_LINEAGE_HOLD}: owner-reviewed authority "
            "pin changed during preflight"
        )
    return Preflight(
        project_root=project_root,
        split_payload=split_payload,
        raw_payload=raw_payload,
        robot_payload=robot_payload,
        snapshots=snapshots,
        reviewed_authority_ref=reviewed_authority,
        total_frames=total_frames,
    )


def _open_directory_chain(path: Path, *, label: str) -> int:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(f"{label} must be canonical and absolute")
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
        state = os.fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} is not an ordinary directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity(state: os.stat_result) -> tuple[int, int]:
    return int(state.st_dev), int(state.st_ino)


def _fresh_directory_inventory(descriptor: int, *, label: str) -> set[str]:
    """Enumerate through a new open-file-description with offset zero.

    CPFS/FUSE ``getdents`` offsets belong to the open-file-description.  A
    repeated ``listdir(held_dirfd)`` can therefore return an empty suffix even
    while the directory is unchanged.  Opening ``.`` relative to the held FD
    preserves the accepted inode but gives every inventory pass a fresh offset.
    """
    held_before = os.fstat(descriptor)
    if not stat.S_ISDIR(held_before.st_mode):
        raise ValueError(f"{label} held FD is not an ordinary directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fresh_descriptor = os.open(".", flags, dir_fd=descriptor)
    try:
        fresh_before = os.fstat(fresh_descriptor)
        if not stat.S_ISDIR(fresh_before.st_mode) or _directory_identity(
            fresh_before
        ) != _directory_identity(held_before):
            raise ValueError(f"{label} fresh inventory FD identity drift")
        names = set(os.listdir(fresh_descriptor))
        fresh_after = os.fstat(fresh_descriptor)
        held_after = os.fstat(descriptor)
        if _directory_identity(fresh_after) != _directory_identity(
            held_before
        ) or _directory_identity(held_after) != _directory_identity(held_before):
            raise ValueError(f"{label} identity changed during inventory")
        return names
    finally:
        os.close(fresh_descriptor)


def _verify_named_directory(path: Path, descriptor: int, *, label: str) -> None:
    held = os.fstat(descriptor)
    try:
        named = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} named path changed") from error
    if not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(named.st_mode):
        raise ValueError(f"{label} is not an ordinary directory")
    if _directory_identity(held) != _directory_identity(named):
        raise ValueError(f"{label} named path identity changed")


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _prepare_output_path(output_root: Path, project_root: Path) -> tuple[Path, Path]:
    lexical = output_root.absolute()
    if lexical != Path(os.path.normpath(lexical)):
        raise ValueError("output root must be canonical and absolute")
    if not lexical.is_relative_to(project_root):
        raise ValueError("output root must remain inside the project root")
    if lexical.is_relative_to(project_root / "_run"):
        raise ValueError("formal selector outputs cannot be published under _run")
    forbidden = {
        part.casefold() for part in lexical.parts
    } & source_contract.FORBIDDEN_SOURCE_COMPONENTS
    if forbidden:
        raise ValueError(f"output root enters forbidden namespace: {sorted(forbidden)}")
    parent = lexical.parent
    resolved_parent = parent.resolve(strict=True)
    if parent != resolved_parent:
        raise ValueError("output parent must be canonical and contain no symlink")
    if _lexists(lexical):
        raise FileExistsError(f"immutable output root already exists: {lexical}")
    return lexical, parent


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("short write while publishing manifest")
        offset += written


def _output_binding(state: os.stat_result) -> tuple[int, int, int, int]:
    return int(state.st_dev), int(state.st_ino), int(state.st_size), int(state.st_nlink)


def _write_exclusive(
    root_descriptor: int,
    root_path: Path,
    name: str,
    encoded: bytes,
) -> OutputRecord:
    record, descriptor = _write_exclusive_held(
        root_descriptor,
        root_path,
        name,
        encoded,
    )
    os.close(descriptor)
    return record


def _write_exclusive_held(
    root_descriptor: int,
    root_path: Path,
    name: str,
    encoded: bytes,
) -> tuple[OutputRecord, int]:
    if name != Path(name).name or name in {"", ".", ".."}:
        raise ValueError(f"invalid output leaf name: {name!r}")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o400, dir_fd=root_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"output leaf is not singly-linked regular: {name}")
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or _output_binding(after) != _output_binding(named)
            or after.st_size != len(encoded)
        ):
            raise ValueError(f"output leaf identity drift: {name}")
        binding = _output_binding(after)
    except BaseException:
        os.close(descriptor)
        raise
    return (
        OutputRecord(
            reference={
                "path": str(root_path / name),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            binding=binding,
        ),
        descriptor,
    )


def _digest_held_output(
    descriptor: int, expected: OutputRecord, *, expected_links: int
) -> tuple[int, int, int, int]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != expected_links
    ):
        raise ValueError("held paired payload mode/link-count drift")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        size += len(block)
        digest.update(block)
    after = os.fstat(descriptor)
    if (
        _output_binding(before) != _output_binding(after)
        or size != expected.reference["bytes"]
        or digest.hexdigest() != expected.reference["sha256"]
    ):
        raise ValueError("held paired payload content drift")
    return _output_binding(after)


def _link_terminal_from_held(
    root_descriptor: int,
    root_path: Path,
    held_descriptor: int,
    private_record: OutputRecord,
) -> tuple[OutputRecord, OutputRecord]:
    before = _digest_held_output(held_descriptor, private_record, expected_links=1)
    named_private = os.stat(
        PAIRED_PRIVATE_NAME,
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    if _output_binding(named_private) != before:
        raise ValueError("paired payload alias changed before terminal hardlink")

    # ``/proc/self/fd`` binds the link source to the already verified open file
    # description.  link(2)/linkat(2) is atomic and refuses an existing target;
    # there is intentionally no rename or copy fallback.
    try:
        os.link(
            f"/proc/self/fd/{held_descriptor}",
            PAIRED_NAME,
            dst_dir_fd=root_descriptor,
            follow_symlinks=True,
        )
    except FileExistsError:
        raise FileExistsError(
            f"terminal manifest already exists: {PAIRED_NAME}"
        ) from None
    except OSError as error:
        raise RuntimeError(
            "atomic held-FD hardlink publication is unsupported; root remains "
            "private and nonterminal"
        ) from error
    os.fsync(root_descriptor)

    linked = os.fstat(held_descriptor)
    named_private = os.stat(
        PAIRED_PRIVATE_NAME,
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    named_terminal = os.stat(
        PAIRED_NAME,
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    binding = _output_binding(linked)
    if (
        not stat.S_ISREG(linked.st_mode)
        or stat.S_IMODE(linked.st_mode) != 0o444
        or linked.st_nlink != 2
        or _output_binding(named_private) != binding
        or _output_binding(named_terminal) != binding
    ):
        raise ValueError("paired terminal hardlink identity drift")
    linked_private = OutputRecord(
        reference=dict(private_record.reference),
        binding=binding,
    )
    terminal = OutputRecord(
        reference={
            **private_record.reference,
            "path": str(root_path / PAIRED_NAME),
        },
        binding=binding,
    )
    _digest_held_output(held_descriptor, terminal, expected_links=2)
    return linked_private, terminal


def _read_output_leaf(
    root_descriptor: int,
    root_path: Path,
    name: str,
    expected: OutputRecord,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=root_descriptor)
    try:
        before = os.fstat(descriptor)
        named = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != expected.binding[3]
            or _output_binding(before) != expected.binding
            or _output_binding(named) != expected.binding
        ):
            raise ValueError(f"published output identity drift: {name}")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        live = os.lstat(root_path / name)
        if (
            _output_binding(after) != expected.binding
            or _output_binding(live) != expected.binding
            or size != expected.reference["bytes"]
            or digest.hexdigest() != expected.reference["sha256"]
        ):
            raise ValueError(f"published output content drift: {name}")
    finally:
        os.close(descriptor)


def _verify_output_tree(
    root_descriptor: int,
    root_path: Path,
    records: Mapping[str, OutputRecord],
    *,
    expected_mode: int,
) -> None:
    _verify_named_directory(root_path, root_descriptor, label="output root")
    root_state = os.fstat(root_descriptor)
    if stat.S_IMODE(root_state.st_mode) != expected_mode:
        raise ValueError("output root mode drift")
    observed_names = _fresh_directory_inventory(root_descriptor, label="output root")
    if observed_names != set(records):
        raise ValueError(
            f"output root inventory drift: {sorted(observed_names)} != "
            f"{sorted(records)}"
        )
    for name, record in records.items():
        _read_output_leaf(root_descriptor, root_path, name, record)


def _paired_payload(preflight: Preflight, records: Mapping[str, OutputRecord]) -> dict:
    sessions: dict[str, dict[str, list[int]]] = {}
    for session_id in _session_order():
        raw_frames = {
            int(key) for key in preflight.raw_payload["sessions"][session_id]["frames"]
        }
        robot_frames = {
            int(key)
            for key in preflight.robot_payload["sessions"][session_id]["frames"]
        }
        paired_frames = raw_frames & robot_frames
        expected_frames = set(
            range(source_contract.ELIGIBLE68_FRAME_COUNTS[session_id])
        )
        if paired_frames != expected_frames:
            raise ValueError(f"asymmetric frame deletion is forbidden: {session_id}")
        starts = [
            start
            for start in range(
                source_contract.ELIGIBLE68_FRAME_COUNTS[session_id]
                - H50_WINDOW_LENGTH
                + 1
            )
            if set(range(start, start + H50_WINDOW_LENGTH)).issubset(paired_frames)
        ]
        if not starts:
            raise ValueError(f"no symmetric H50 window survives: {session_id}")
        sessions[session_id] = {
            "frames": sorted(paired_frames),
            "window_starts": starts,
        }
    return {
        "schema_version": source_contract.PAIRED_SCHEMA,
        "immutable": True,
        "no_fallback": True,
        "cohort": "eligible68",
        "selection_policy": "SYMMETRIC_FRAME_AND_WINDOW_INTERSECTION",
        "unresolved_policy": ("RETAIN_PAIRED_FRAME_REGARDLESS_OF_DOMAIN_UNRESOLVED"),
        "terminal_publication": {
            "profile": source_contract.PAIRED_TERMINAL_PUBLICATION_PROFILE,
            "terminal_name": PAIRED_NAME,
            "permanent_alias_name": PAIRED_PRIVATE_NAME,
            "required_nlink": 2,
        },
        "selector_refs": {
            "RAW": dict(records[RAW_NAME].reference),
            "ROBOT_RGB": dict(records[ROBOT_NAME].reference),
        },
        "artifact_roots": {
            "RAW": str(preflight.raw_payload["artifact_root"]),
            "ROBOT_RGB": str(preflight.robot_payload["artifact_root"]),
        },
        "sessions": sessions,
    }


def _assert_pin_stable(preflight: Preflight) -> None:
    value = source_contract.REVIEWED_ROBOT_RGB_LINEAGE_AUTHORITY_REF
    if value is None or dict(value) != preflight.reviewed_authority_ref:
        raise RuntimeError(
            f"{source_contract.ROBOT_RGB_LINEAGE_HOLD}: owner-reviewed authority "
            "pin changed during publication"
        )


def publish_matched_selector_manifests(
    *,
    split_reference: Mapping[str, Any],
    raw_selector_records_reference: Mapping[str, Any],
    robot_selector_records_reference: Mapping[str, Any],
    output_root: str | Path,
    embodiment: str,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify exact twin inputs and publish two selectors plus paired ledger.

    The output root is itself the private stage (mode 0700).  It is reserved
    with ``mkdirat`` and every leaf is created with ``O_EXCL``.  The complete
    paired payload alias is retained permanently and atomically hard-linked
    from its held FD into the terminal name last.  An existing terminal is
    never overwritten and unsupported hardlink semantics fail closed.

    Once the output root is created this function never unlinks, removes, or
    renames it on failure.  A failed root stays private and nonterminal for
    audit; a retry must use a fresh absent root.
    """
    resolved_project_root = Path(project_root).resolve(strict=True)
    preflight = _preflight(
        split_reference=split_reference,
        raw_selector_records_reference=raw_selector_records_reference,
        robot_selector_records_reference=robot_selector_records_reference,
        embodiment=embodiment,
        project_root=resolved_project_root,
    )
    output_path, output_parent = _prepare_output_path(
        Path(output_root), resolved_project_root
    )

    # One final complete input pass occurs before creating even the private
    # output root.  Trust/payload failures therefore leave output ABSENT.
    _assert_pin_stable(preflight)
    _reverify_snapshots(preflight.snapshots)

    parent_descriptor = _open_directory_chain(output_parent, label="output parent")
    root_descriptor: int | None = None
    paired_descriptor: int | None = None
    try:
        _verify_named_directory(output_parent, parent_descriptor, label="output parent")
        try:
            os.mkdir(output_path.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            raise FileExistsError(
                f"immutable output root already exists: {output_path}"
            ) from None
        os.fsync(parent_descriptor)
        root_descriptor = os.open(
            output_path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        _verify_named_directory(output_path, root_descriptor, label="output root")
        if _fresh_directory_inventory(root_descriptor, label="new output root"):
            raise ValueError("new private output root is not empty")

        records: dict[str, OutputRecord] = {}
        records[RAW_NAME] = _write_exclusive(
            root_descriptor,
            output_path,
            RAW_NAME,
            _canonical_json(preflight.raw_payload),
        )
        records[ROBOT_NAME] = _write_exclusive(
            root_descriptor,
            output_path,
            ROBOT_NAME,
            _canonical_json(preflight.robot_payload),
        )
        paired_payload = _paired_payload(preflight, records)
        paired_encoded = _canonical_json(paired_payload)
        paired_private, paired_descriptor = _write_exclusive_held(
            root_descriptor,
            output_path,
            PAIRED_PRIVATE_NAME,
            paired_encoded,
        )

        # The public consumer rejects the permanent alias by design.  Before
        # terminal commit we instead verify its exact JSON bytes, all input
        # snapshots, and the complete private output inventory.  The formal
        # consumer validation occurs only after the hardlink pair and mode 0555
        # root are complete.
        if (
            _parse_json_object(paired_encoded, label="paired-kept complete payload")
            != paired_payload
        ):
            raise ValueError("paired-kept payload JSON self-check drift")
        _assert_pin_stable(preflight)
        _reverify_snapshots(preflight.snapshots)
        _verify_output_tree(
            root_descriptor,
            output_path,
            {**records, PAIRED_PRIVATE_NAME: paired_private},
            expected_mode=0o700,
        )
        _verify_named_directory(output_parent, parent_descriptor, label="output parent")

        # This is the sole terminal commit.  There is no rename/copy fallback.
        paired_alias, paired_record = _link_terminal_from_held(
            root_descriptor,
            output_path,
            paired_descriptor,
            paired_private,
        )
        records[PAIRED_PRIVATE_NAME] = paired_alias
        records[PAIRED_NAME] = paired_record
        _verify_output_tree(
            root_descriptor,
            output_path,
            records,
            expected_mode=0o700,
        )

        _assert_pin_stable(preflight)
        _reverify_snapshots(preflight.snapshots)
        _verify_output_tree(
            root_descriptor,
            output_path,
            records,
            expected_mode=0o700,
        )
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
        os.fsync(parent_descriptor)
        _verify_output_tree(
            root_descriptor,
            output_path,
            records,
            expected_mode=0o555,
        )
        # Formal postlink validation uses the narrowly scoped source-contract
        # wrapper for this fixed terminal/alias pair.  Generic references remain
        # singly-linked, and passing the alias path as the paired input fails.
        binding = source_contract.require_manifest_selector_ready(
            records[RAW_NAME].reference,
            records[PAIRED_NAME].reference,
            artifact_root=preflight.raw_payload["artifact_root"],
            project_root=resolved_project_root,
        )
        _digest_held_output(paired_descriptor, paired_record, expected_links=2)
        _verify_output_tree(
            root_descriptor,
            output_path,
            records,
            expected_mode=0o555,
        )
        return {
            "status": "ARTIFACT_EXISTS",
            "schema_scope": "TWIN_ONLY_V1_RAW_AND_ROBOT_RGB",
            "third_arm_defined": False,
            "output_root": str(output_path),
            "split_reference": dict(split_reference),
            "frame_count": preflight.total_frames,
            "session_count": len(source_contract.ELIGIBLE68),
            "selector_references": {
                "RAW": dict(records[RAW_NAME].reference),
                "ROBOT_RGB": dict(records[ROBOT_NAME].reference),
            },
            "paired_kept_reference": dict(records[PAIRED_NAME].reference),
            "permanent_terminal_alias_ref": dict(
                records[PAIRED_PRIVATE_NAME].reference
            ),
            "product_lines": list(TWIN_PRODUCT_LINES),
            "validator_product_line": binding["product_line"],
            "postlink_source_contract_revalidation": True,
            "terminal_manifest_source_retained": True,
            "terminal_hardlink_nlink": 2,
            "terminal_manifest_last": True,
            "no_overwrite": True,
        }
    finally:
        if paired_descriptor is not None:
            os.close(paired_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)


def _parse_reference_argument(value: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"{label} must be JSON") from error
    try:
        return _require_exact_reference(payload, label=label)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-ref-json",
        required=True,
        type=lambda value: _parse_reference_argument(value, label="split reference"),
    )
    parser.add_argument(
        "--raw-selector-records-ref-json",
        required=True,
        type=lambda value: _parse_reference_argument(
            value, label="RAW selector-record reference"
        ),
    )
    parser.add_argument(
        "--robot-selector-records-ref-json",
        required=True,
        type=lambda value: _parse_reference_argument(
            value, label="RobotRGB selector-record reference"
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        report = publish_matched_selector_manifests(
            split_reference=arguments.split_ref_json,
            raw_selector_records_reference=arguments.raw_selector_records_ref_json,
            robot_selector_records_reference=(
                arguments.robot_selector_records_ref_json
            ),
            output_root=arguments.output_root,
            embodiment=arguments.embodiment,
            project_root=arguments.project_root,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
