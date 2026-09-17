#!/usr/bin/env python3
"""Audit the received KaiHand adapter STEP without inferring installation pose."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid


POINT = re.compile(
    r"CARTESIAN_POINT\s*\([^,]*,\s*\(\s*"
    r"([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)\s*\)"
)
SOLID = re.compile(r"MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'")
CYLINDER = re.compile(r"CYLINDRICAL_SURFACE\s*\([^,]*,\s*#[0-9]+\s*,\s*([-+0-9.Ee]+)\s*\)")
CIRCLE = re.compile(r"CIRCLE\s*\([^,]*,\s*#[0-9]+\s*,\s*([-+0-9.Ee]+)\s*\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _unique(values: list[float]) -> list[float]:
    return sorted({round(value, 9) for value in values})


def parse_step_text(text: str) -> dict[str, Any]:
    if not text.startswith("ISO-10303-21;") or "END-ISO-10303-21;" not in text:
        raise ValueError("not a complete ISO-10303-21 exchange file")
    schema_match = re.search(r"FILE_SCHEMA\s*\(\(\s*'([^']+)'", text)
    if schema_match is None:
        raise ValueError("STEP FILE_SCHEMA missing")
    millimetre = bool(re.search(
        r"LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\(\s*\*\s*\)\s*"
        r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", text,
    ))
    points = [tuple(float(value) for value in match.groups()) for match in POINT.finditer(text)]
    if not points:
        raise ValueError("STEP has no Cartesian points")
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return {
        "syntax": "ISO-10303-21",
        "file_schema": schema_match.group(1),
        "length_unit": "millimetre" if millimetre else "UNRESOLVED",
        "entity_record_count": len(re.findall(r"(?m)^#[0-9]+\s*=", text)),
        "manifold_solid_brep_names": SOLID.findall(text),
        "closed_shell_count": len(re.findall(r"\bCLOSED_SHELL\s*\(", text)),
        "advanced_face_count": len(re.findall(r"\bADVANCED_FACE\s*\(", text)),
        "cartesian_point_count": len(points),
        "declared_control_point_envelope": {
            "minimum": minimum,
            "maximum": maximum,
            "extent": [maximum[axis] - minimum[axis] for axis in range(3)],
            "unit": "millimetre" if millimetre else "UNRESOLVED",
            "claim_limit": "Envelope of declared CARTESIAN_POINT records; not a decoded/transformed B-rep bounding box.",
        },
        "cylindrical_surface_count": len(CYLINDER.findall(text)),
        "declared_cylindrical_radii": _unique([float(value) for value in CYLINDER.findall(text)]),
        "circle_count": len(CIRCLE.findall(text)),
        "declared_circle_radii": _unique([float(value) for value in CIRCLE.findall(text)]),
    }


def decode_with_cadquery(path: Path, mesh_path: Path) -> dict[str, Any]:
    try:
        import cadquery as cq  # type: ignore
    except ImportError:
        return {
            "status": "BLOCKED_EXTERNAL_CADQUERY_RUNTIME_ABSENT",
            "decoder": "cadquery",
            "review_mesh": None,
        }
    shape = cq.importers.importStep(str(path))
    solids = list(shape.solids().vals())
    if not solids:
        raise RuntimeError("CadQuery decoded zero solids")
    box = shape.val().BoundingBox()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(mesh_path), tolerance=0.15, angularTolerance=0.2)
    return {
        "status": "PASS_DECODED_AND_REVIEW_MESH_EXPORTED",
        "decoder": "cadquery",
        "decoder_version": getattr(cq, "__version__", "UNKNOWN"),
        "solid_count": len(solids),
        "decoded_bbox": {
            "minimum": [box.xmin, box.ymin, box.zmin],
            "maximum": [box.xmax, box.ymax, box.zmax],
            "extent": [box.xlen, box.ylen, box.zlen],
            "unit": "millimetre",
        },
        "review_mesh": {
            "path": str(mesh_path.resolve()),
            "bytes": mesh_path.stat().st_size,
            "sha256": sha256(mesh_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    step = args.step.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable output required: {output}")
    output.mkdir(parents=True)
    text = step.read_text(encoding="latin-1")
    static = parse_step_text(text)
    decoded = decode_with_cadquery(step, output / "KAI_HAND_ADAPTER_REVIEW.stl")
    receipt = {
        "schema_version": "kaihand-adapter-step-audit-v1",
        "status": "PRESENT_CANDIDATE_GEOMETRY",
        "source": {"path": str(step), "bytes": step.stat().st_size,
                   "sha256": sha256(step)},
        "static_step_evidence": static,
        "decoder": decoded,
        "interface_evidence": {
            "status": "CANDIDATE_GEOMETRY_ONLY",
            "hole_or_interface_classification": "NOT_INFERRED_FROM_RADII_ALONE",
            "installation_transform": "ABSENT",
            "robot_tcp": "ABSENT",
            "camera_world_to_base": "ABSENT",
        },
        "claim_limit": (
            "Received STEP candidate geometry only. Declared radii are not "
            "classified as mounting holes, and no measured mount, TCP, "
            "camera/world-to-base transform, manufacturing release, or "
            "physical deployment authority is inferred."
        ),
    }
    atomic_json(output / "RESULT.json", receipt)
    print(json.dumps({"status": receipt["status"],
                      "decoder": decoded["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
