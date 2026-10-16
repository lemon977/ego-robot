#!/usr/bin/env python3
"""Render Exact78 diagnostics with the frozen orange flange-ring proxy."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import trimesh

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.ops import render_exact78_action_diagnostics_v1 as previous
from chaoyang.pipeline import exact78_robot_raster_v1 as raster


RING = (REPO_ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1"
        / "excluded_reference_only/FLANGE_RING_VISUAL_PROXY_CONTRACT.json")


def annulus_local_triangles(geometry: dict) -> np.ndarray:
    length = float(geometry["length_m"])
    mesh = trimesh.creation.annulus(
        r_min=float(geometry["inner_radius_m"]),
        r_max=float(geometry["outer_radius_m"]),
        height=length,
        sections=int(geometry["sections"]),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 2] += length * 0.5
    return vertices[np.asarray(mesh.faces, dtype=np.int64)]


def main() -> int:
    contract = json.loads(RING.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_DEVELOPMENT_VISUALIZATION_ONLY":
        raise RuntimeError("FLANGE_RING_PROXY_AUTHORITY_DRIFT")
    geometry = contract["geometry"]
    local = annulus_local_triangles(geometry)
    color = tuple(int(value) for value in geometry["render_color_bgr"])
    old_flange = previous.FLANGE
    old_loader = raster.naturalv2_local_triangles
    old_color = raster.FLANGE_IVORY
    previous.FLANGE = RING
    raster.naturalv2_local_triangles = lambda _path: local.copy()
    raster.FLANGE_IVORY = color
    try:
        code = previous.main()
    finally:
        previous.FLANGE = old_flange
        raster.naturalv2_local_triangles = old_loader
        raster.FLANGE_IVORY = old_color
    if code != 0:
        return code
    output = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
    result_path = output / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema_version"] = "exact78-action-diagnostics-v2"
    result["inputs"]["flange_visual_proxy"] = artifact_ref(RING)
    result["flange_visualization"] = {
        "kind": "annular_cylinder",
        "color_bgr": list(color),
        "real_adapter_cad": False,
        "watermark": geometry["watermark"],
    }
    result["claim_limit"] = (
        "Raw-overlay and independent-view motion diagnostics with a visualization-only "
        "flange ring; no Clean, occlusion, contact, control, measured adapter or physical claim."
    )
    atomic_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
