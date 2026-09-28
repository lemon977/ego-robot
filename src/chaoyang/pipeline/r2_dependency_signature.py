"""Content-addressed dependency bindings for R2 session products.

The dependency graph is intentionally smaller than the task scheduler.  It
answers which *artifact* must be recomputed when one frozen input changes and
prevents a same-name file from being treated as a valid resume.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


STAGE_INPUTS: dict[str, frozenset[str]] = {
    "scene_clean": frozenset({"raw", "scene_masks", "scene_config"}),
    "motion_r0": frozenset({"raw", "roi", "motion_config", "robot_asset"}),
    "occlusion": frozenset({"scene_depth", "robot_depth", "occlusion_config"}),
    "product_render": frozenset({
        "scene_clean", "motion_r0", "camera_domain", "robot_asset",
        "occlusion", "render_config",
    }),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_product_binding(
    *,
    files: Mapping[str, Path],
    render_config: Mapping[str, object],
    occlusion_status: str,
) -> dict[str, object]:
    required = {"scene_clean", "motion_r0", "camera_domain", "robot_asset", "renderer_code"}
    missing = required - set(files)
    if missing:
        raise ValueError(f"missing product binding inputs: {sorted(missing)}")
    file_rows = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
        for name, path in sorted(files.items())
    }
    identity = {
        "files": {name: row["sha256"] for name, row in file_rows.items()},
        "render_config": dict(render_config),
        "occlusion_status": occlusion_status,
    }
    return {
        "schema_version": "HUMAN_TO_ROBOT_R2_PRODUCT_CACHE_BINDING_V1",
        "files": file_rows,
        "render_config": dict(render_config),
        "occlusion_status": occlusion_status,
        "signature_sha256": stable_json_sha256(identity),
    }


def affected_stages(changed_inputs: set[str]) -> set[str]:
    """Return the exact transitive invalidation set for named input changes."""

    affected: set[str] = set()
    available = set(changed_inputs)
    while True:
        newly = {
            stage for stage, dependencies in STAGE_INPUTS.items()
            if stage not in affected and dependencies & available
        }
        if not newly:
            return affected
        affected.update(newly)
        available.update(newly)
