#!/usr/bin/env python3
"""Build once or validate the immutable D1 robot-asset identity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import trimesh


ROBOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROBOT_ROOT.parents[1]
PIN_PATH = ROBOT_ROOT / "ROBOT_ASSET_PIN.json"
TIANJI_ROOT = ROBOT_ROOT / "tianji" / "marvin_description"
KAIHAND_ROOT = ROBOT_ROOT / "kaihand" / "packages"
URDFS = (
    TIANJI_ROOT / "urdf" / "marvin_CCS_m6.urdf",
    KAIHAND_ROOT
    / "KaiBot-Dexhand shell-URDF-L-260624(1620)"
    / "urdf"
    / "KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
    KAIHAND_ROOT
    / "KaiBot-Dexhand shell-URDF-R-260424(1430)"
    / "urdf"
    / "KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
)


class AssetPinError(RuntimeError):
    pass


def _read_regular(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise AssetPinError(f"not an ordinary file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise AssetPinError(f"opened non-regular asset: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise AssetPinError(f"asset changed during open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _project_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _asset_files() -> list[Path]:
    paths: list[Path] = []
    for root in (TIANJI_ROOT, KAIHAND_ROOT):
        for path in root.rglob("*"):
            if path.is_symlink():
                raise AssetPinError(f"symlink asset is forbidden: {path}")
            if path.is_file():
                paths.append(path)
            elif not path.is_dir():
                raise AssetPinError(f"special asset node is forbidden: {path}")
    return sorted(paths, key=_project_path)


def _resolve_mesh(urdf: Path, reference: str) -> Path:
    if reference.startswith("package://marvin_description/"):
        path = TIANJI_ROOT / reference.removeprefix("package://marvin_description/")
    elif reference.startswith("package://"):
        raise AssetPinError(f"unknown package URI: {reference}")
    else:
        path = (urdf.parent / reference).resolve()
    path = path.resolve()
    allowed = (TIANJI_ROOT.resolve(), KAIHAND_ROOT.resolve())
    if not any(path == root or root in path.parents for root in allowed):
        raise AssetPinError(f"mesh escapes pinned roots: {reference} -> {path}")
    if path.is_symlink() or not path.is_file():
        raise AssetPinError(f"missing/non-regular mesh reference: {reference} -> {path}")
    return path


def _mesh_counts(path: Path) -> tuple[int, int]:
    loaded = trimesh.load_mesh(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise AssetPinError(f"empty mesh scene: {path}")
        vertices = sum(len(mesh.vertices) for mesh in loaded.geometry.values())
        faces = sum(len(mesh.faces) for mesh in loaded.geometry.values())
    else:
        vertices, faces = len(loaded.vertices), len(loaded.faces)
    if vertices <= 0 or faces <= 0:
        raise AssetPinError(f"empty mesh: {path}")
    return int(vertices), int(faces)


def build_manifest() -> dict[str, Any]:
    files = []
    for path in _asset_files():
        payload = _read_regular(path)
        files.append(
            {
                "path": _project_path(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    urdf_summaries = []
    all_resolved_meshes: set[Path] = set()
    for urdf in URDFS:
        root = ET.fromstring(_read_regular(urdf))
        refs = sorted({element.attrib["filename"] for element in root.findall(".//mesh")})
        resolved = [_resolve_mesh(urdf, reference) for reference in refs]
        all_resolved_meshes.update(resolved)
        urdf_summaries.append(
            {
                "path": _project_path(urdf),
                "robot_name": root.attrib.get("name"),
                "link_count": len(root.findall("link")),
                "joint_count": len(root.findall("joint")),
                "mesh_reference_count": len(refs),
                "resolved_meshes": [_project_path(path) for path in resolved],
            }
        )

    mesh_summaries = []
    for path in sorted(all_resolved_meshes, key=_project_path):
        vertices, triangles = _mesh_counts(path)
        mesh_summaries.append(
            {
                "path": _project_path(path),
                "vertices": vertices,
                "triangles": triangles,
            }
        )

    calibration = ROBOT_ROOT / "tianji" / "calibration"
    calibration_files = sorted(path for path in calibration.rglob("*") if path.is_file())
    return {
        "schema_version": "robot-asset-pin-v1",
        "status": "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION",
        "scope": {
            "roots": [_project_path(TIANJI_ROOT), _project_path(KAIHAND_ROOT)],
            "mutation_permitted": False,
            "scale_topology_urdf_fk_frozen": True,
        },
        "files": files,
        "urdfs": urdf_summaries,
        "meshes": mesh_summaries,
        "summary": {
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files),
            "urdf_count": len(urdf_summaries),
            "link_count": sum(row["link_count"] for row in urdf_summaries),
            "joint_count": sum(row["joint_count"] for row in urdf_summaries),
            "unique_referenced_mesh_count": len(mesh_summaries),
            "vertex_count": sum(row["vertices"] for row in mesh_summaries),
            "triangle_count": sum(row["triangles"] for row in mesh_summaries),
            "all_mesh_references_resolved": True,
            "all_meshes_loaded_by_trimesh": True,
        },
        "calibration": {
            "path": _project_path(calibration),
            "present": calibration.is_dir(),
            "file_count": len(calibration_files),
            "status": "NEED_IF_FK_REQUIRES_CALIBRATION" if not calibration_files else "PRESENT",
        },
    }


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_new(manifest: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(PIN_PATH, flags, 0o644)
    try:
        payload = _canonical(manifest)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-new", action="store_true")
    args = parser.parse_args()
    observed = build_manifest()
    if args.write_new:
        write_new(observed)
        print(f"WROTE {PIN_PATH}")
        return
    expected = json.loads(_read_regular(PIN_PATH))
    if expected != observed:
        raise AssetPinError("ROBOT_ASSET_PIN.json differs from current asset bytes/geometry")
    print(json.dumps(observed["summary"], sort_keys=True))
    print(json.dumps(observed["calibration"], sort_keys=True))


if __name__ == "__main__":
    main()
