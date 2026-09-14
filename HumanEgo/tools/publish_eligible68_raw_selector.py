#!/usr/bin/env python3
"""Build the formal eligible68 RAW selector from fixed, byte-real inputs.

The producer never discovers sessions or frames from a directory.  It uses the
literal eligible68 order/counts in ``utils.source_contract`` and derives every
metadata and original RAW path from that closed world.

``unresolved=false`` has one intentionally narrow meaning here: the selected
original RAW RGB PNG exists, is byte-stable, and fully decodes as uint8
1280x960 three-channel pixels.  It makes no claim about HaWoR, action labels,
hand-side validity, Object6D, or any MISSING/OUTSIDE status in another domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "HumanEgo") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "HumanEgo"))

from utils import source_contract  # noqa: E402


OUTPUT_NAME = source_contract.RAW_SELECTOR_STANDALONE_TERMINAL_NAME
PRIVATE_NAME = source_contract.RAW_SELECTOR_STANDALONE_ALIAS_NAME
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 960
EXPECTED_CHANNELS = 3


@dataclass
class RootLease:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    label: str

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass(frozen=True)
class FileSnapshot:
    root: RootLease
    relative_path: Path
    reference: dict[str, Any]
    binding: tuple[int, int, int, int, int, int, int]
    label: str


@dataclass(frozen=True)
class ObsSymlinkSnapshot:
    production_root: RootLease
    relative_path: Path
    expected_target: Path
    link_text: str
    binding: tuple[int, int, int, int, int, int, int]
    label: str


@dataclass
class SelectorPlan:
    project_root: RootLease
    production_root: RootLease
    raw_root: RootLease
    payload: dict[str, Any]
    encoded: bytes
    snapshots: list[FileSnapshot]
    symlinks: list[ObsSymlinkSnapshot]
    frame_count: int

    def close(self) -> None:
        self.raw_root.close()
        self.production_root.close()
        self.project_root.close()


@dataclass
class HumanEgoLease:
    project_root: RootLease
    descriptor: int
    identity: tuple[int, int]
    path: Path

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass
class OutputParentLease:
    humanego: HumanEgoLease
    outputs_descriptor: int
    outputs_identity: tuple[int, int]
    path: Path

    def close(self) -> None:
        os.close(self.outputs_descriptor)


@dataclass(frozen=True)
class OutputRecord:
    reference: dict[str, Any]
    binding: tuple[int, int, int, int, int]


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


def _forbidden_components(path: Path) -> set[str]:
    return {
        part.casefold() for part in path.parts
    } & source_contract.FORBIDDEN_SOURCE_COMPONENTS


def _reject_forbidden_path(path: Path, *, label: str) -> None:
    forbidden = _forbidden_components(path)
    if forbidden:
        raise ValueError(f"{label} enters forbidden namespace: {sorted(forbidden)}")


def _reject_derived_pixels(path: Path, *, label: str) -> None:
    lowered = [part.casefold() for part in path.parts]
    if any("clean" in part or "inpaint" in part for part in lowered):
        raise ValueError(f"{label} points to CLEAN/inpaint pixels: {path}")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_chain(path: Path, *, label: str) -> int:
    lexical = path.absolute()
    if not lexical.is_absolute() or lexical != Path(os.path.normpath(lexical)):
        raise ValueError(f"{label} must be canonical and absolute")
    descriptor = os.open(lexical.anchor, _directory_flags())
    try:
        for component in lexical.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        state = os.fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode):
            raise ValueError(f"{label} is not an ordinary directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_root(path_value: str | Path, *, label: str) -> RootLease:
    path = Path(path_value).absolute()
    if path != Path(os.path.normpath(path)):
        raise ValueError(f"{label} must be canonical and absolute")
    _reject_forbidden_path(path, label=label)
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} contains a symlink component")
    descriptor = _open_directory_chain(path, label=label)
    try:
        held = os.fstat(descriptor)
        named = os.lstat(path)
        identity = (int(held.st_dev), int(held.st_ino))
        if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != identity:
            raise ValueError(f"{label} named identity drift")
        return RootLease(path, descriptor, identity, label)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_root(lease: RootLease) -> None:
    held = os.fstat(lease.descriptor)
    named = os.lstat(lease.path)
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (held.st_dev, held.st_ino) != lease.identity
        or (named.st_dev, named.st_ino) != lease.identity
    ):
        raise ValueError(f"{lease.label} identity changed")


def _validate_relative(relative_path: Path, *, label: str) -> None:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"{label} relative path is invalid: {relative_path}")


def _open_relative_parent(
    root_descriptor: int, relative_parent: Path, *, label: str
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in relative_parent.parts:
            if component in {"", "."}:
                continue
            if component == "..":
                raise ValueError(f"{label} attempts parent traversal")
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _file_binding(state: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
        int(state.st_nlink),
        int(state.st_mode),
    )


def _read_snapshot(
    root: RootLease,
    relative_path: Path,
    *,
    label: str,
    expected_reference: Mapping[str, Any] | None = None,
) -> tuple[bytes, FileSnapshot]:
    _validate_relative(relative_path, label=label)
    _verify_root(root)
    parent_descriptor = _open_relative_parent(
        root.descriptor,
        relative_path.parent,
        label=f"{label} parent",
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            relative_path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        held_entry = os.stat(
            relative_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        lexical_path = root.path / relative_path
        live_entry = os.lstat(lexical_path)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (held_entry.st_dev, held_entry.st_ino) != identity
            or (live_entry.st_dev, live_entry.st_ino) != identity
            or lexical_path.resolve(strict=True) != lexical_path
        ):
            raise ValueError(f"{label} is not one canonical singly-linked file")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
        encoded = b"".join(blocks)
        after = os.fstat(descriptor)
        held_after = os.stat(
            relative_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        live_after = os.lstat(lexical_path)
        binding = _file_binding(after)
        if (
            _file_binding(before) != binding
            or len(encoded) != after.st_size
            or (held_after.st_dev, held_after.st_ino) != identity
            or (live_after.st_dev, live_after.st_ino) != identity
        ):
            raise ValueError(f"{label} changed during same-FD read")
        reference = {
            "path": str(lexical_path),
            "bytes": len(encoded),
            "sha256": digest.hexdigest(),
        }
        if expected_reference is not None and dict(expected_reference) != reference:
            raise ValueError(f"{label} exact reference changed")
        _verify_root(root)
        return encoded, FileSnapshot(
            root=root,
            relative_path=relative_path,
            reference=reference,
            binding=binding,
            label=label,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _symlink_binding(state: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return _file_binding(state)


def _inspect_obs_symlink(
    production_root: RootLease,
    relative_path: Path,
    *,
    expected_target: Path,
    label: str,
) -> ObsSymlinkSnapshot:
    _validate_relative(relative_path, label=label)
    parent_descriptor = _open_relative_parent(
        production_root.descriptor,
        relative_path.parent,
        label=f"{label} parent",
    )
    try:
        before = os.stat(
            relative_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        live = os.lstat(production_root.path / relative_path)
        if (
            not stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (live.st_dev, live.st_ino)
        ):
            raise ValueError(f"{label} is not the expected singly-linked symlink")
        link_text = os.readlink(relative_path.name, dir_fd=parent_descriptor)
        link_path = Path(link_text)
        if not link_path.is_absolute() and ".." in link_path.parts:
            raise ValueError(f"{label} relative symlink attempts parent escape")
        _reject_derived_pixels(link_path, label=f"{label} target")
        candidate = (
            link_path
            if link_path.is_absolute()
            else (production_root.path / relative_path.parent / link_path)
        )
        if candidate.resolve(strict=True) != expected_target:
            raise ValueError(f"{label} resolves to the wrong original RAW frame")
        after = os.stat(
            relative_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _symlink_binding(before) != _symlink_binding(after)
            or os.readlink(relative_path.name, dir_fd=parent_descriptor) != link_text
        ):
            raise ValueError(f"{label} changed during inspection")
        _verify_root(production_root)
        return ObsSymlinkSnapshot(
            production_root=production_root,
            relative_path=relative_path,
            expected_target=expected_target,
            link_text=link_text,
            binding=_symlink_binding(after),
            label=label,
        )
    finally:
        os.close(parent_descriptor)


def _reverify_obs_symlink(snapshot: ObsSymlinkSnapshot) -> None:
    observed = _inspect_obs_symlink(
        snapshot.production_root,
        snapshot.relative_path,
        expected_target=snapshot.expected_target,
        label=f"{snapshot.label} terminal recheck",
    )
    if observed.binding != snapshot.binding or observed.link_text != snapshot.link_text:
        raise ValueError(f"{snapshot.label} identity changed after preflight")


def _session_order() -> tuple[str, ...]:
    order = tuple(
        session_id
        for session_id in source_contract.ELIGIBLE68_ORDER
        if session_id in source_contract.ELIGIBLE68
    )
    if len(order) != len(source_contract.ELIGIBLE68) or set(order) != set(
        source_contract.ELIGIBLE68
    ):
        raise RuntimeError("source_contract eligible68 literal order/set drift")
    forbidden = set(order) & set(source_contract.FORBIDDEN10)
    if forbidden:
        raise RuntimeError(f"eligible68 literal enters forbidden cohort: {forbidden}")
    return order


def _metadata_relative(session_id: str, frame_key: str) -> Path:
    return (
        Path(session_id)
        / "09_humanego_adapter"
        / "preprocess"
        / "all_data"
        / frame_key
        / "training_data.json"
    )


def _adapter_rgb_relative(session_id: str, frame_key: str) -> Path:
    return _metadata_relative(session_id, frame_key).parent / "rgb.png"


def _raw_relative(session_id: str, frame_key: str) -> Path:
    return Path(session_id) / "preprocess" / "all_data" / frame_key / "rgb.png"


def _parse_metadata_rgb_path(encoded: bytes, *, label: str) -> str:
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("obs"), dict):
        raise ValueError(f"{label} lacks an obs object")
    value = payload["obs"].get("rgb_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} lacks obs.rgb_path")
    return value


def _inspect_metadata_rgb_binding(
    *,
    production_root: RootLease,
    raw_root: RootLease,
    session_id: str,
    frame_key: str,
    metadata_encoded: bytes,
) -> ObsSymlinkSnapshot | None:
    logical_key = f"{session_id}/{frame_key}"
    expected_raw = raw_root.path / _raw_relative(session_id, frame_key)
    adapter_relative = _adapter_rgb_relative(session_id, frame_key)
    adapter_rgb = production_root.path / adapter_relative
    rgb_value = _parse_metadata_rgb_path(
        metadata_encoded,
        label=f"{logical_key} final_v3 metadata",
    )
    rgb_path = Path(rgb_value)
    if not rgb_path.is_absolute() or rgb_path != Path(os.path.normpath(rgb_path)):
        raise ValueError(f"{logical_key} obs.rgb_path must be canonical and absolute")
    _reject_forbidden_path(rgb_path, label=f"{logical_key} obs.rgb_path")
    _reject_derived_pixels(rgb_path, label=f"{logical_key} obs.rgb_path")
    if rgb_path == adapter_rgb:
        return _inspect_obs_symlink(
            production_root,
            adapter_relative,
            expected_target=expected_raw,
            label=f"{logical_key} adapter rgb.png",
        )
    if rgb_path == expected_raw:
        if expected_raw.resolve(strict=True) != expected_raw:
            raise ValueError(f"{logical_key} direct obs.rgb_path is not original RAW")
        return None
    raise ValueError(
        f"{logical_key} obs.rgb_path is neither the frame symlink "
        "nor expected original RAW path"
    )


def _validate_png(encoded: bytes, *, label: str) -> None:
    if not encoded.startswith(PNG_SIGNATURE):
        raise ValueError(f"{label} is not PNG bytes")
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError(f"{label} cannot be fully decoded")
    if decoded.dtype != np.uint8 or decoded.shape != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        EXPECTED_CHANNELS,
    ):
        raise ValueError(
            f"{label} must decode as uint8 1280x960 three-channel RGB pixels"
        )


def _register_source_owner(
    snapshot: FileSnapshot,
    *,
    owner: str,
    path_owners: dict[str, str],
    inode_owners: dict[tuple[int, int], str],
) -> None:
    path = str(snapshot.reference["path"])
    inode = snapshot.binding[:2]
    previous_path = path_owners.get(path)
    previous_inode = inode_owners.get(inode)
    if previous_path is not None and previous_path != owner:
        raise ValueError(f"RAW selector source path aliases {previous_path}, {owner}")
    if previous_inode is not None and previous_inode != owner:
        raise ValueError(f"RAW selector source inode aliases {previous_inode}, {owner}")
    path_owners[path] = owner
    inode_owners[inode] = owner


def _build_plan(
    *,
    project_root: RootLease,
    production_root: str | Path,
    raw_artifact_root: str | Path,
) -> SelectorPlan:
    _verify_root(project_root)
    project = project_root.path
    production = _open_root(production_root, label="final_v3 production root")
    raw: RootLease | None = None
    try:
        if not production.path.is_relative_to(
            project
        ) or production.path.is_relative_to(project / "_run"):
            raise ValueError("final_v3 production root must be formal project input")
        raw = _open_root(raw_artifact_root, label="original RAW artifact root")
        sessions: dict[str, dict[str, Any]] = {}
        snapshots: list[FileSnapshot] = []
        symlinks: list[ObsSymlinkSnapshot] = []
        path_owners: dict[str, str] = {}
        inode_owners: dict[tuple[int, int], str] = {}
        total_frames = 0
        for session_id in _session_order():
            frame_count = source_contract.ELIGIBLE68_FRAME_COUNTS[session_id]
            frames: dict[str, dict[str, Any]] = {}
            for frame_index in range(frame_count):
                frame_key = f"{frame_index:05d}"
                logical_key = f"{session_id}/{frame_key}"
                metadata_relative = _metadata_relative(session_id, frame_key)
                raw_relative = _raw_relative(session_id, frame_key)
                metadata_encoded, metadata_snapshot = _read_snapshot(
                    production,
                    metadata_relative,
                    label=f"{logical_key} final_v3 metadata",
                )
                raw_encoded, raw_snapshot = _read_snapshot(
                    raw,
                    raw_relative,
                    label=f"{logical_key} original RAW rgb.png",
                )
                _validate_png(raw_encoded, label=f"{logical_key} original RAW rgb.png")
                symlink = _inspect_metadata_rgb_binding(
                    production_root=production,
                    raw_root=raw,
                    session_id=session_id,
                    frame_key=frame_key,
                    metadata_encoded=metadata_encoded,
                )
                if symlink is not None:
                    symlinks.append(symlink)
                _register_source_owner(
                    metadata_snapshot,
                    owner=f"metadata:{logical_key}",
                    path_owners=path_owners,
                    inode_owners=inode_owners,
                )
                _register_source_owner(
                    raw_snapshot,
                    owner=f"RAW:{logical_key}",
                    path_owners=path_owners,
                    inode_owners=inode_owners,
                )
                snapshots.extend((metadata_snapshot, raw_snapshot))
                frames[frame_key] = {
                    "metadata": dict(metadata_snapshot.reference),
                    "image": dict(raw_snapshot.reference),
                    "unresolved": False,
                }
                total_frames += 1
            sessions[session_id] = {"frames": frames}
        expected_total = sum(source_contract.ELIGIBLE68_FRAME_COUNTS.values())
        if total_frames != expected_total:
            raise RuntimeError(
                "RAW selector denominator differs from eligible68 literal"
            )
        payload = {
            "schema_version": source_contract.SELECTOR_SCHEMA,
            "immutable": True,
            "no_fallback": True,
            "product_line": "RAW",
            "image_name": "rgb.png",
            "artifact_root": str(raw.path),
            "selector_root": str(raw.path),
            "sessions": sessions,
        }
        return SelectorPlan(
            project_root=project_root,
            production_root=production,
            raw_root=raw,
            payload=payload,
            encoded=_canonical_json(payload),
            snapshots=snapshots,
            symlinks=symlinks,
            frame_count=total_frames,
        )
    except BaseException:
        if raw is not None:
            raw.close()
        production.close()
        raise


def _reverify_sources(plan: SelectorPlan) -> None:
    _verify_root(plan.project_root)
    _verify_root(plan.production_root)
    _verify_root(plan.raw_root)
    for snapshot in plan.snapshots:
        _, observed = _read_snapshot(
            snapshot.root,
            snapshot.relative_path,
            label=f"{snapshot.label} terminal recheck",
            expected_reference=snapshot.reference,
        )
        if observed.binding != snapshot.binding:
            raise ValueError(f"{snapshot.label} identity changed after preflight")
    for symlink in plan.symlinks:
        _reverify_obs_symlink(symlink)
    _verify_root(plan.production_root)
    _verify_root(plan.raw_root)
    _verify_root(plan.project_root)


def _verify_internal_payload(plan: SelectorPlan) -> None:
    try:
        decoded = json.loads(plan.encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("internally generated RAW selector JSON is invalid") from error
    if decoded != plan.payload:
        raise ValueError("internally generated RAW selector JSON roundtrip drift")
    source_contract._validate_standalone_raw_selector_payload(
        decoded,
        artifact_root=plan.raw_root.path,
        project_root=plan.project_root.path,
    )


def _fresh_directory_inventory(descriptor: int, *, label: str) -> set[str]:
    held = os.fstat(descriptor)
    if not stat.S_ISDIR(held.st_mode):
        raise ValueError(f"{label} held FD is not a directory")
    fresh = os.open(".", _directory_flags(), dir_fd=descriptor)
    try:
        fresh_state = os.fstat(fresh)
        if (fresh_state.st_dev, fresh_state.st_ino) != (held.st_dev, held.st_ino):
            raise ValueError(f"{label} fresh inventory FD identity drift")
        names = set(os.listdir(fresh))
        after = os.fstat(fresh)
        if (after.st_dev, after.st_ino) != (held.st_dev, held.st_ino):
            raise ValueError(f"{label} changed during inventory")
        return names
    finally:
        os.close(fresh)


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _open_humanego(project_root: RootLease) -> HumanEgoLease:
    _verify_root(project_root)
    descriptor = os.open("HumanEgo", _directory_flags(), dir_fd=project_root.descriptor)
    try:
        held = os.fstat(descriptor)
        named = os.stat(
            "HumanEgo", dir_fd=project_root.descriptor, follow_symlinks=False
        )
        live = os.lstat(project_root.path / "HumanEgo")
        identity = (int(held.st_dev), int(held.st_ino))
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(live.st_mode)
            or (named.st_dev, named.st_ino) != identity
            or (live.st_dev, live.st_ino) != identity
        ):
            raise ValueError("held HumanEgo directory identity drift")
        return HumanEgoLease(
            project_root=project_root,
            descriptor=descriptor,
            identity=identity,
            path=project_root.path / "HumanEgo",
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_humanego(lease: HumanEgoLease) -> None:
    _verify_root(lease.project_root)
    held = os.fstat(lease.descriptor)
    named = os.stat(
        "HumanEgo", dir_fd=lease.project_root.descriptor, follow_symlinks=False
    )
    live = os.lstat(lease.path)
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or (held.st_dev, held.st_ino) != lease.identity
        or (named.st_dev, named.st_ino) != lease.identity
        or (live.st_dev, live.st_ino) != lease.identity
    ):
        raise ValueError("held HumanEgo directory was replaced")


def _open_output_parent(humanego: HumanEgoLease) -> OutputParentLease:
    _verify_humanego(humanego)
    outputs_descriptor: int | None = None
    try:
        outputs_descriptor = os.open(
            "outputs", _directory_flags(), dir_fd=humanego.descriptor
        )
        outputs_state = os.fstat(outputs_descriptor)
        outputs_named = os.stat(
            "outputs", dir_fd=humanego.descriptor, follow_symlinks=False
        )
        outputs_live = os.lstat(humanego.path / "outputs")
        outputs_identity = (int(outputs_state.st_dev), int(outputs_state.st_ino))
        if (
            not stat.S_ISDIR(outputs_state.st_mode)
            or not stat.S_ISDIR(outputs_named.st_mode)
            or not stat.S_ISDIR(outputs_live.st_mode)
            or (outputs_named.st_dev, outputs_named.st_ino) != outputs_identity
            or (outputs_live.st_dev, outputs_live.st_ino) != outputs_identity
        ):
            raise ValueError("held HumanEgo/outputs identity drift")
        return OutputParentLease(
            humanego=humanego,
            outputs_descriptor=outputs_descriptor,
            outputs_identity=outputs_identity,
            path=humanego.path / "outputs",
        )
    except BaseException:
        if outputs_descriptor is not None:
            os.close(outputs_descriptor)
        raise


def _verify_output_parent(lease: OutputParentLease) -> None:
    _verify_humanego(lease.humanego)
    outputs_held = os.fstat(lease.outputs_descriptor)
    outputs_named = os.stat(
        "outputs", dir_fd=lease.humanego.descriptor, follow_symlinks=False
    )
    outputs_live = os.lstat(lease.path)
    if (
        not stat.S_ISDIR(outputs_held.st_mode)
        or not stat.S_ISDIR(outputs_named.st_mode)
        or not stat.S_ISDIR(outputs_live.st_mode)
        or (outputs_held.st_dev, outputs_held.st_ino) != lease.outputs_identity
        or (outputs_named.st_dev, outputs_named.st_ino) != lease.outputs_identity
        or (outputs_live.st_dev, outputs_live.st_ino) != lease.outputs_identity
    ):
        raise ValueError("held HumanEgo/outputs directory was replaced")


def _prepare_output_path(
    output_root: str | Path, humanego: HumanEgoLease
) -> tuple[Path, OutputParentLease]:
    output = Path(output_root).absolute()
    if output != Path(os.path.normpath(output)):
        raise ValueError("output root must be canonical and absolute")
    approved = humanego.path / "outputs"
    if output.parent != approved:
        raise ValueError("output root must be a direct child of HumanEgo/outputs")
    if not source_contract.RAW_SELECTOR_STANDALONE_ROOT_PATTERN.fullmatch(output.name):
        raise ValueError("output root does not use the standalone RAW wrapper name")
    _reject_forbidden_path(output, label="RAW selector output root")
    parent_lease = _open_output_parent(humanego)
    try:
        _verify_output_child_absent(parent_lease, output)
    except BaseException:
        parent_lease.close()
        raise
    return output, parent_lease


def _verify_output_child_absent(parent_lease: OutputParentLease, output: Path) -> None:
    if output.parent != parent_lease.path:
        raise ValueError("RAW selector output no longer has the held output parent")
    _verify_output_parent(parent_lease)
    try:
        os.stat(
            output.name,
            dir_fd=parent_lease.outputs_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"immutable RAW selector output root exists: {output}")
    if _lexists(output):
        raise FileExistsError(f"immutable RAW selector output root exists: {output}")
    _verify_output_parent(parent_lease)


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("short write while publishing RAW selector")
        offset += written


def _output_binding(state: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_nlink),
        int(state.st_mode),
    )


def _write_manifest(
    root_descriptor: int,
    output_root: Path,
    encoded: bytes,
    *,
    name: str,
) -> tuple[OutputRecord, int]:
    descriptor: int | None = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
        dir_fd=root_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("RAW selector output is not singly-linked regular")
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        binding = _output_binding(after)
        if (
            not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o444
            or after.st_nlink != 1
            or after.st_size != len(encoded)
            or _output_binding(named) != binding
        ):
            raise ValueError("RAW selector output identity/mode drift")
        record = OutputRecord(
            reference={
                "path": str(output_root / name),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            binding=binding,
        )
        # A chmod does not revoke write access from an already-open writer.
        # Close it before commit, then retain only a verified read-only FD as
        # the hardlink source.
        os.close(descriptor)
        descriptor = None
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        if _output_binding(os.fstat(descriptor)) != record.binding:
            raise ValueError("RAW selector alias changed during read-only reopen")
        return record, descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _verify_manifest(
    parent_lease: OutputParentLease,
    root_descriptor: int,
    root_identity: tuple[int, int],
    output_root: Path,
    descriptor: int,
    record: OutputRecord,
    *,
    name: str,
    expected_root_mode: int,
) -> None:
    _verify_output_parent(parent_lease)
    root_held = os.fstat(root_descriptor)
    root_named = os.stat(
        output_root.name,
        dir_fd=parent_lease.outputs_descriptor,
        follow_symlinks=False,
    )
    root_live = os.lstat(output_root)
    if (
        not stat.S_ISDIR(root_held.st_mode)
        or not stat.S_ISDIR(root_named.st_mode)
        or not stat.S_ISDIR(root_live.st_mode)
        or (root_held.st_dev, root_held.st_ino) != root_identity
        or (root_named.st_dev, root_named.st_ino) != root_identity
        or (root_live.st_dev, root_live.st_ino) != root_identity
        or stat.S_IMODE(root_held.st_mode) != expected_root_mode
        or stat.S_IMODE(root_named.st_mode) != expected_root_mode
        or stat.S_IMODE(root_live.st_mode) != expected_root_mode
    ):
        raise ValueError("RAW selector output root identity drift")
    held = os.fstat(descriptor)
    named = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    live = os.lstat(output_root / name)
    if (
        _output_binding(held) != record.binding
        or _output_binding(named) != record.binding
        or _output_binding(live) != record.binding
        or stat.S_IMODE(held.st_mode) != 0o444
    ):
        raise ValueError("RAW selector output named identity drift")
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
        _output_binding(after) != record.binding
        or size != record.reference["bytes"]
        or digest.hexdigest() != record.reference["sha256"]
    ):
        raise ValueError("RAW selector output content drift")
    if _fresh_directory_inventory(
        root_descriptor, label="RAW selector output root"
    ) != {name}:
        raise ValueError("RAW selector output inventory drift")
    _verify_output_parent(parent_lease)


def _commit_terminal_from_held(
    root_descriptor: int,
    manifest_descriptor: int,
) -> None:
    """Atomically expose the terminal name as the publication's final action."""
    try:
        os.link(
            f"/proc/self/fd/{manifest_descriptor}",
            OUTPUT_NAME,
            dst_dir_fd=root_descriptor,
            follow_symlinks=True,
        )
    except FileExistsError:
        raise FileExistsError(
            f"immutable RAW selector terminal already exists: {OUTPUT_NAME}"
        ) from None
    except OSError as error:
        raise RuntimeError(
            "atomic held-FD hardlink publication is unsupported; the retained "
            "alias is nonterminal"
        ) from error


def _linked_output_binding(record: OutputRecord) -> tuple[int, int, int, int, int]:
    return (
        record.binding[0],
        record.binding[1],
        record.binding[2],
        2,
        record.binding[4],
    )


def _verify_committed_terminal_pair(
    parent_lease: OutputParentLease,
    root_descriptor: int,
    root_identity: tuple[int, int],
    output_root: Path,
    manifest_descriptor: int,
    alias_record: OutputRecord,
) -> None:
    """Bind the committed pair to both held and canonical namespaces."""
    _verify_output_parent(parent_lease)
    root_held = os.fstat(root_descriptor)
    root_named = os.stat(
        output_root.name,
        dir_fd=parent_lease.outputs_descriptor,
        follow_symlinks=False,
    )
    root_live = os.lstat(output_root)
    if (
        not stat.S_ISDIR(root_held.st_mode)
        or not stat.S_ISDIR(root_named.st_mode)
        or not stat.S_ISDIR(root_live.st_mode)
        or (root_held.st_dev, root_held.st_ino) != root_identity
        or (root_named.st_dev, root_named.st_ino) != root_identity
        or (root_live.st_dev, root_live.st_ino) != root_identity
        or stat.S_IMODE(root_held.st_mode) != 0o555
        or stat.S_IMODE(root_named.st_mode) != 0o555
        or stat.S_IMODE(root_live.st_mode) != 0o555
    ):
        raise ValueError("committed RAW selector root identity drift")

    expected_binding = _linked_output_binding(alias_record)
    held = os.fstat(manifest_descriptor)
    alias_named = os.stat(
        PRIVATE_NAME,
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    terminal_named = os.stat(
        OUTPUT_NAME,
        dir_fd=root_descriptor,
        follow_symlinks=False,
    )
    alias_live = os.lstat(output_root / PRIVATE_NAME)
    terminal_live = os.lstat(output_root / OUTPUT_NAME)
    if (
        _output_binding(held) != expected_binding
        or _output_binding(alias_named) != expected_binding
        or _output_binding(terminal_named) != expected_binding
        or _output_binding(alias_live) != expected_binding
        or _output_binding(terminal_live) != expected_binding
        or not stat.S_ISREG(held.st_mode)
        or stat.S_IMODE(held.st_mode) != 0o444
    ):
        raise ValueError("committed RAW selector terminal pair identity drift")

    os.lseek(manifest_descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(manifest_descriptor, 1024 * 1024)
        if not block:
            break
        size += len(block)
        digest.update(block)
    if (
        _output_binding(os.fstat(manifest_descriptor)) != expected_binding
        or size != alias_record.reference["bytes"]
        or digest.hexdigest() != alias_record.reference["sha256"]
    ):
        raise ValueError("committed RAW selector terminal pair content drift")
    if _fresh_directory_inventory(
        root_descriptor, label="committed RAW selector output root"
    ) != {PRIVATE_NAME, OUTPUT_NAME}:
        raise ValueError("committed RAW selector output inventory drift")
    _verify_output_parent(parent_lease)


def _postcommit_validate(
    *,
    parent_lease: OutputParentLease,
    root_descriptor: int,
    root_identity: tuple[int, int],
    output_root: Path,
    manifest_descriptor: int,
    alias_record: OutputRecord,
    terminal_reference: Mapping[str, Any],
    artifact_root: Path,
    project_root: Path,
) -> None:
    """Run durability, namespace, and public-consumer checks after commit."""
    os.fsync(root_descriptor)
    os.fsync(parent_lease.outputs_descriptor)
    _verify_committed_terminal_pair(
        parent_lease,
        root_descriptor,
        root_identity,
        output_root,
        manifest_descriptor,
        alias_record,
    )
    source_contract.validate_selector_manifest_reference(
        terminal_reference,
        artifact_root=artifact_root,
        project_root=project_root,
    )
    _verify_committed_terminal_pair(
        parent_lease,
        root_descriptor,
        root_identity,
        output_root,
        manifest_descriptor,
        alias_record,
    )


def _canonical_consumer_accepts(
    terminal_reference: Mapping[str, Any],
    *,
    artifact_root: Path,
    project_root: Path,
) -> bool:
    """Independently classify the only canonical consumer reference."""
    try:
        source_contract.validate_selector_manifest_reference(
            terminal_reference,
            artifact_root=artifact_root,
            project_root=project_root,
        )
    except Exception:
        return False
    return True


def _close_descriptors(
    descriptors: Sequence[int | None], *, suppress_errors: bool
) -> None:
    """Close every FD; a committed publication must never be downgraded by close."""
    first_error: BaseException | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except BaseException as error:  # pragma: no branch - defensive commit fence
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def light_check_eligible68_raw_selector_inputs(
    *,
    project_root: str | Path,
    production_root: str | Path,
    raw_artifact_root: str | Path,
    estimate_session: str,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the bounded preflight used before the 28,265-frame full check.

    Every fixed metadata record and ``obs.rgb_path`` binding is checked.  RAW
    bytes are read and decoded only for each session's first/last frame and all
    frames of the explicitly named estimate session.  This function cannot
    publish and its report is deliberately not a selector-validity claim.
    """
    cv2.setNumThreads(1)
    started = time.monotonic()
    project_lease = _open_root(project_root, label="project root")
    humanego_lease: HumanEgoLease | None = None
    parent_lease: OutputParentLease | None = None
    production: RootLease | None = None
    raw: RootLease | None = None
    try:
        humanego_lease = _open_humanego(project_lease)
        prepared_output: tuple[Path, OutputParentLease] | None = None
        if output_root is not None:
            prepared_output = _prepare_output_path(output_root, humanego_lease)
            parent_lease = prepared_output[1]
        order = _session_order()
        if estimate_session not in order:
            raise ValueError("estimate session must be one literal eligible68 session")
        production = _open_root(production_root, label="final_v3 production root")
        if not production.path.is_relative_to(
            project_lease.path
        ) or production.path.is_relative_to(project_lease.path / "_run"):
            raise ValueError("final_v3 production root must be formal project input")
        raw = _open_root(raw_artifact_root, label="original RAW artifact root")
        snapshots: list[FileSnapshot] = []
        symlinks: list[ObsSymlinkSnapshot] = []
        path_owners: dict[str, str] = {}
        inode_owners: dict[tuple[int, int], str] = {}
        metadata_count = 0
        sampled_raw_count = 0
        sampled_raw_bytes = 0
        estimate_raw_count = 0
        estimate_raw_bytes = 0
        raw_read_seconds = 0.0
        for session_id in order:
            frame_count = source_contract.ELIGIBLE68_FRAME_COUNTS[session_id]
            sampled_indices = {0, frame_count - 1}
            if session_id == estimate_session:
                sampled_indices.update(range(frame_count))
            for frame_index in range(frame_count):
                frame_key = f"{frame_index:05d}"
                logical_key = f"{session_id}/{frame_key}"
                metadata_encoded, metadata_snapshot = _read_snapshot(
                    production,
                    _metadata_relative(session_id, frame_key),
                    label=f"{logical_key} final_v3 metadata",
                )
                symlink = _inspect_metadata_rgb_binding(
                    production_root=production,
                    raw_root=raw,
                    session_id=session_id,
                    frame_key=frame_key,
                    metadata_encoded=metadata_encoded,
                )
                if symlink is not None:
                    symlinks.append(symlink)
                _register_source_owner(
                    metadata_snapshot,
                    owner=f"metadata:{logical_key}",
                    path_owners=path_owners,
                    inode_owners=inode_owners,
                )
                snapshots.append(metadata_snapshot)
                metadata_count += 1
                if frame_index not in sampled_indices:
                    continue
                raw_started = time.monotonic()
                raw_encoded, raw_snapshot = _read_snapshot(
                    raw,
                    _raw_relative(session_id, frame_key),
                    label=f"{logical_key} sampled original RAW rgb.png",
                )
                raw_read_seconds += time.monotonic() - raw_started
                _validate_png(
                    raw_encoded,
                    label=f"{logical_key} sampled original RAW rgb.png",
                )
                _register_source_owner(
                    raw_snapshot,
                    owner=f"RAW:{logical_key}",
                    path_owners=path_owners,
                    inode_owners=inode_owners,
                )
                snapshots.append(raw_snapshot)
                sampled_raw_count += 1
                sampled_raw_bytes += int(raw_snapshot.reference["bytes"])
                if session_id == estimate_session:
                    estimate_raw_count += 1
                    estimate_raw_bytes += int(raw_snapshot.reference["bytes"])
        expected_metadata_count = sum(source_contract.ELIGIBLE68_FRAME_COUNTS.values())
        if metadata_count != expected_metadata_count:
            raise RuntimeError("light preflight denominator drift")
        for snapshot in snapshots:
            _, observed = _read_snapshot(
                snapshot.root,
                snapshot.relative_path,
                label=f"{snapshot.label} light terminal recheck",
                expected_reference=snapshot.reference,
            )
            if observed.binding != snapshot.binding:
                raise ValueError(f"{snapshot.label} identity changed after light scan")
        for symlink in symlinks:
            _reverify_obs_symlink(symlink)
        _verify_root(project_lease)
        _verify_root(production)
        _verify_root(raw)
        _verify_humanego(humanego_lease)
        if parent_lease is not None:
            assert prepared_output is not None
            _verify_output_child_absent(parent_lease, prepared_output[0])
        _verify_humanego(humanego_lease)
        mib = estimate_raw_bytes / (1024 * 1024)
        return {
            "status": "LIGHT_CHECK_ONLY_PASS_NO_SELECTOR_CLAIM",
            "cpu_only": True,
            "session_count": len(order),
            "metadata_count": metadata_count,
            "obs_rgb_binding_count": metadata_count,
            "adapter_rgb_symlink_count": len(symlinks),
            "sampled_raw_png_count": sampled_raw_count,
            "sampled_raw_bytes": sampled_raw_bytes,
            "estimate_session": estimate_session,
            "estimate_session_raw_png_count": estimate_raw_count,
            "estimate_session_raw_bytes": estimate_raw_bytes,
            "initial_raw_read_seconds": raw_read_seconds,
            "initial_raw_mib_per_second": (
                mib / raw_read_seconds if raw_read_seconds > 0 else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "full_selector_validated": False,
            "output_written": False,
        }
    finally:
        if raw is not None:
            raw.close()
        if production is not None:
            production.close()
        if parent_lease is not None:
            parent_lease.close()
        if humanego_lease is not None:
            humanego_lease.close()
        project_lease.close()


def publish_eligible68_raw_selector(
    *,
    project_root: str | Path,
    production_root: str | Path,
    raw_artifact_root: str | Path,
    output_root: str | Path | None = None,
    check_only: bool = False,
) -> dict[str, Any]:
    """Check every literal frame and optionally publish one immutable selector."""
    cv2.setNumThreads(1)
    project_lease = _open_root(project_root, label="project root")
    plan: SelectorPlan | None = None
    humanego_lease: HumanEgoLease | None = None
    parent_lease: OutputParentLease | None = None
    committed = False
    try:
        humanego_lease = _open_humanego(project_lease)
        prepared_output: tuple[Path, OutputParentLease] | None = None
        if output_root is not None:
            prepared_output = _prepare_output_path(output_root, humanego_lease)
            parent_lease = prepared_output[1]
        elif not check_only:
            raise ValueError("formal publication requires --output-root")
        plan = _build_plan(
            project_root=project_lease,
            production_root=production_root,
            raw_artifact_root=raw_artifact_root,
        )
        _verify_internal_payload(plan)
        _reverify_sources(plan)
        if check_only:
            _verify_humanego(humanego_lease)
            if parent_lease is not None:
                assert prepared_output is not None
                _verify_output_child_absent(parent_lease, prepared_output[0])
            _verify_humanego(humanego_lease)
            return {
                "status": "CHECK_ONLY_PASS_NO_OUTPUT_WRITTEN",
                "cpu_only": True,
                "session_count": len(source_contract.ELIGIBLE68),
                "frame_count": plan.frame_count,
                "metadata_count": plan.frame_count,
                "raw_png_count": plan.frame_count,
                "adapter_rgb_symlink_count": len(plan.symlinks),
                "selector_schema": source_contract.SELECTOR_SCHEMA,
                "selector_payload_bytes": len(plan.encoded),
                "selector_payload_sha256": hashlib.sha256(plan.encoded).hexdigest(),
                "unresolved_semantics": (
                    "FALSE_MEANS_ONLY_SELECTED_RAW_RGB_COMPLETE_AND_DECODABLE"
                ),
                "output_written": False,
            }

        assert prepared_output is not None
        assert parent_lease is not None
        output, _ = prepared_output
        payload_reference_fields = {
            "bytes": len(plan.encoded),
            "sha256": hashlib.sha256(plan.encoded).hexdigest(),
        }
        alias_reference = {
            "path": str(output / PRIVATE_NAME),
            **payload_reference_fields,
        }
        terminal_reference = {
            "path": str(output / OUTPUT_NAME),
            **payload_reference_fields,
        }
        report = {
            "status": "ARTIFACT_EXISTS",
            "cpu_only": True,
            "session_count": len(source_contract.ELIGIBLE68),
            "frame_count": plan.frame_count,
            "selector_manifest_ref": terminal_reference,
            "permanent_terminal_alias_ref": alias_reference,
            "unresolved_semantics": (
                "FALSE_MEANS_ONLY_SELECTED_RAW_RGB_COMPLETE_AND_DECODABLE"
            ),
            "output_root": str(output),
            "output_root_mode": "0555",
            "manifest_mode": "0444",
            "terminal_publication_profile": (
                source_contract.RAW_SELECTOR_STANDALONE_PUBLICATION_PROFILE
            ),
            "terminal_manifest_source_retained": True,
            "terminal_hardlink_nlink": 2,
            "terminal_manifest_last": True,
            "terminal_commit_then_canonical_classification": True,
            "precommit_source_contract_semantic_validation": True,
            "postcommit_source_contract_revalidation": True,
            "postcommit_recovered": False,
            "rc2_implies_canonical_consumer_reject": True,
            "no_overwrite": True,
            "no_delete_or_rename": True,
        }
        recovered_report = {
            **report,
            "status": "ARTIFACT_EXISTS_POSTCOMMIT_RECOVERED",
            "postcommit_recovered": True,
        }
        root_descriptor: int | None = None
        manifest_descriptor: int | None = None
        try:
            _verify_output_parent(parent_lease)
            os.mkdir(output.name, 0o700, dir_fd=parent_lease.outputs_descriptor)
            os.fsync(parent_lease.outputs_descriptor)
            root_descriptor = os.open(
                output.name,
                _directory_flags(),
                dir_fd=parent_lease.outputs_descriptor,
            )
            root_state = os.fstat(root_descriptor)
            root_named = os.stat(
                output.name,
                dir_fd=parent_lease.outputs_descriptor,
                follow_symlinks=False,
            )
            named_root = os.lstat(output)
            root_identity = (int(root_state.st_dev), int(root_state.st_ino))
            if (
                not stat.S_ISDIR(root_named.st_mode)
                or not stat.S_ISDIR(named_root.st_mode)
                or (root_named.st_dev, root_named.st_ino) != root_identity
                or (named_root.st_dev, named_root.st_ino) != root_identity
                or stat.S_IMODE(root_state.st_mode) != 0o700
                or _fresh_directory_inventory(
                    root_descriptor, label="new RAW selector output root"
                )
            ):
                raise ValueError("new RAW selector output root is not empty/private")
            _verify_output_parent(parent_lease)
            alias_record, manifest_descriptor = _write_manifest(
                root_descriptor,
                output,
                plan.encoded,
                name=PRIVATE_NAME,
            )
            if alias_record.reference != alias_reference:
                raise ValueError("planned RAW selector alias reference drift")
            _verify_manifest(
                parent_lease,
                root_descriptor,
                root_identity,
                output,
                manifest_descriptor,
                alias_record,
                name=PRIVATE_NAME,
                expected_root_mode=0o700,
            )
            os.fchmod(root_descriptor, 0o555)
            os.fsync(root_descriptor)
            os.fsync(parent_lease.outputs_descriptor)
            if stat.S_IMODE(os.fstat(root_descriptor).st_mode) != 0o555:
                raise ValueError("RAW selector output root final mode drift")
            _verify_manifest(
                parent_lease,
                root_descriptor,
                root_identity,
                output,
                manifest_descriptor,
                alias_record,
                name=PRIVATE_NAME,
                expected_root_mode=0o555,
            )
            # The public loader intentionally rejects the permanent alias before
            # commit.  Run its exact semantic core against the held payload now;
            # external consumers run the full terminal-pair loader after commit.
            _verify_internal_payload(plan)
            # This is the terminal all-source pass: every one of the literal
            # 28,265 metadata/RAW references is reopened and rehashed.
            _reverify_sources(plan)
            os.fsync(root_descriptor)
            os.fsync(parent_lease.outputs_descriptor)
            _verify_manifest(
                parent_lease,
                root_descriptor,
                root_identity,
                output,
                manifest_descriptor,
                alias_record,
                name=PRIVATE_NAME,
                expected_root_mode=0o555,
            )
            # The hardlink is the terminal commit.  Any later exception is
            # classified against the canonical public consumer: an accepted
            # canonical reference recovers to rc0, while rc2 is reserved for a
            # canonical reference that the consumer independently rejects.
            _commit_terminal_from_held(root_descriptor, manifest_descriptor)
            committed = True
            _postcommit_validate(
                parent_lease=parent_lease,
                root_descriptor=root_descriptor,
                root_identity=root_identity,
                output_root=output,
                manifest_descriptor=manifest_descriptor,
                alias_record=alias_record,
                terminal_reference=terminal_reference,
                artifact_root=plan.raw_root.path,
                project_root=plan.project_root.path,
            )
            return report
        except Exception:
            if _canonical_consumer_accepts(
                terminal_reference,
                artifact_root=plan.raw_root.path,
                project_root=plan.project_root.path,
            ):
                committed = True
                return recovered_report
            raise
        finally:
            _close_descriptors(
                (manifest_descriptor, root_descriptor),
                suppress_errors=committed,
            )
    finally:
        outer_descriptors: list[int | None] = []
        if parent_lease is not None:
            outer_descriptors.append(parent_lease.outputs_descriptor)
        if humanego_lease is not None:
            outer_descriptors.append(humanego_lease.descriptor)
        if plan is not None:
            outer_descriptors.extend(
                (
                    plan.raw_root.descriptor,
                    plan.production_root.descriptor,
                    plan.project_root.descriptor,
                )
            )
        else:
            outer_descriptors.append(project_lease.descriptor)
        _close_descriptors(outer_descriptors, suppress_errors=committed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--production-root", required=True, type=Path)
    parser.add_argument("--raw-artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    check_group = parser.add_mutually_exclusive_group()
    check_group.add_argument("--check-only", action="store_true")
    check_group.add_argument("--light-check-only", action="store_true")
    parser.add_argument("--estimate-session")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.light_check_only:
            if arguments.estimate_session is None:
                raise ValueError("--light-check-only requires --estimate-session")
            report = light_check_eligible68_raw_selector_inputs(
                project_root=arguments.project_root,
                production_root=arguments.production_root,
                raw_artifact_root=arguments.raw_artifact_root,
                estimate_session=arguments.estimate_session,
                output_root=arguments.output_root,
            )
        else:
            if arguments.estimate_session is not None:
                raise ValueError("--estimate-session is only valid with light check")
            report = publish_eligible68_raw_selector(
                project_root=arguments.project_root,
                production_root=arguments.production_root,
                raw_artifact_root=arguments.raw_artifact_root,
                output_root=arguments.output_root,
                check_only=arguments.check_only,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
