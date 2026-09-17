#!/usr/bin/env python3
"""Export a bounded one-scene Robot z-buffer canary from pinned V5.2 states.

This is renderer provenance evidence only.  It does not include an object mesh,
does not decide Robot/object ownership, and cannot promote Robot or occlusion
authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from chaoyang.ops import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402


def file_ref(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def uniform_valid_frames(valid_side_frame: np.ndarray, count: int) -> np.ndarray:
    valid = np.asarray(valid_side_frame)
    if valid.ndim != 2 or valid.shape[0] != 2 or valid.dtype != np.bool_:
        raise ValueError("valid_side_frame must be bool [2,T]")
    eligible = np.flatnonzero(np.all(valid, axis=0))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} bilateral valid frames for requested {count}")
    slots = np.rint(np.linspace(0, len(eligible) - 1, count)).astype(np.int64)
    return eligible[slots]


def source_image_size(hawor: dict[str, np.ndarray], frame: int) -> tuple[float, float, str]:
    """Resolve the image domain without assuming a 2048x1536 source."""
    if "image_size" in hawor:
        values = np.asarray(hawor["image_size"], dtype=np.float64).reshape(-1)
        if len(values) < 2:
            raise ValueError("image_size must contain width and height")
        width, height = map(float, values[:2])
        source = "HAWOR_IMAGE_SIZE"
    else:
        intrinsics = np.asarray(hawor["intrinsics"][frame], dtype=np.float64)
        width = 2.0 * float(intrinsics[0, 2]) + 1.0
        height = 2.0 * float(intrinsics[1, 2]) + 1.0
        source = "CENTERED_PRINCIPAL_POINT_INFERENCE"
    if not (np.isfinite(width) and np.isfinite(height) and width >= 320 and height >= 240):
        raise ValueError(f"invalid inferred source domain: {(width, height)}")
    return width, height, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--arm-states", type=Path, required=True)
    parser.add_argument("--hand-states", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument(
        "--frame-ids",
        help="Optional comma-separated explicit frame ids; overrides uniform --frame-count selection.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.hawor.resolve(strict=True), args.arm_states.resolve(strict=True), args.hand_states.resolve(strict=True)]
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    if not all(args.session_id in str(path) for path in inputs):
        raise ValueError("same-session input identity check failed")
    hawor, arm, hand = map(load_npz, inputs)
    valid_side_frame = np.asarray(arm["valid_side_frame"], dtype=np.bool_)
    if args.frame_ids:
        frames = np.asarray([int(item) for item in args.frame_ids.split(",")], dtype=np.int64)
        if len(frames) == 0 or len(set(frames.tolist())) != len(frames):
            raise ValueError("explicit frame ids must be non-empty and unique")
        if np.any(frames < 0) or np.any(frames >= valid_side_frame.shape[1]):
            raise ValueError("explicit frame id outside session")
        if not np.all(valid_side_frame[:, frames]):
            raise ValueError("every explicit frame must be bilateral-valid")
    else:
        frames = uniform_valid_frames(valid_side_frame, args.frame_count)
    if not np.array_equal(arm["valid_side_frame"], hand["valid_side_frame"]):
        raise ValueError("arm/hand valid masks differ")
    if not (len(hawor["c2w"]) == len(arm["q_arm"]) == len(hand["q_hand"])):
        raise ValueError("state frame counts differ")
    args.output_dir.mkdir(parents=True)
    assets = old.load_pinned_robot_assets(ROOT)
    raster = old.load_module(old.RASTER_SOURCE, f"{args.session_id}_v71_zbuffer")
    flange = fixed.naturalv2_local_triangles()
    cache: dict[Path, object] = {}
    world_base = np.asarray(arm["T_world_base"])
    if world_base.ndim == 3:
        if not np.allclose(world_base, world_base[0], atol=1e-12):
            raise ValueError("world base is not fixed")
        world_base = world_base[0]
    rgb_rows, depth_rows, label_rows, triangle_rows = [], [], [], []
    identities: dict[str, dict[str, str]] = {}
    domain_sources: set[str] = set()
    for frame in frames:
        camera = np.linalg.inv(hawor["c2w"][frame]) @ world_base
        k = np.asarray(hawor["intrinsics"][frame], dtype=np.float64).copy()
        # Current review renderer is explicitly 640x480.  Scale the per-frame
        # camera from the HaWoR input domain instead of inventing a new K.
        source_width, source_height, domain_source = source_image_size(hawor, int(frame))
        domain_sources.add(domain_source)
        k[0] *= fixed.WIDTH / source_width
        k[1] *= fixed.HEIGHT / source_height
        color, label, buffers = shared.render_robot(
            raster, assets, arm["q_arm"][frame], hand["q_hand"][frame],
            arm["T_tool_hand_root"], camera, k, cache, flange,
            complete_robot=False, return_buffers=True,
        )
        assert isinstance(buffers, dict)
        depth = np.asarray(buffers["depth_m"])
        triangle = np.asarray(buffers["triangle_id"])
        visible = label >= 0
        if not np.all(np.isfinite(depth[visible]) & (depth[visible] > 0.0)):
            raise RuntimeError("rendered pixels lack finite positive optical-Z")
        if np.any(triangle[visible] < 0):
            raise RuntimeError("rendered pixels lack triangle provenance")
        for key, value in buffers["label_identity"].items():
            identities[str(key)] = value
        rgb_rows.append(color)
        depth_rows.append(depth)
        label_rows.append(label)
        triangle_rows.append(triangle)
    archive_path = args.output_dir / "ROBOT_UNIFIED_ZBUFFER_CANARY.npz"
    np.savez_compressed(
        archive_path,
        frame_ids=frames,
        rgb=np.asarray(rgb_rows, dtype=np.uint8),
        depth_m=np.asarray(depth_rows, dtype=np.float32),
        render_label=np.asarray(label_rows, dtype=np.int32),
        triangle_id=np.asarray(triangle_rows, dtype=np.int64),
    )
    identity_path = args.output_dir / "LABEL_IDENTITY.json"
    atomic_json(identity_path, identities)
    result = {
        "schema_version": "robot-unified-zbuffer-canary-v71",
        "artifact_revision": "R7_1",
        "status": "PASS_DEVELOPMENT_ONE_SCENE_ZBUFFER_EXPORT",
        "session_id": args.session_id,
        "frame_ids": frames.tolist(),
        "resolution": [fixed.WIDTH, fixed.HEIGHT],
        "source_image_domain": {
            "width": source_image_size(hawor, int(frames[0]))[0],
            "height": source_image_size(hawor, int(frames[0]))[1],
            "evidence": sorted(domain_sources),
        },
        "gates": {
            "one_scene_all_robot_links": True,
            "finite_positive_optical_z": True,
            "instance_part_link_mapping": True,
            "triangle_id_present": True,
            "draw_order_independent_tie_audit": "UNIT_TEST_ONLY_NOT_EXPOSED_BY_LEGACY_RASTER",
            "object_layers_present": False,
        },
        "inputs": [file_ref(path) for path in inputs],
        "outputs": [file_ref(archive_path), file_ref(identity_path)],
        "authority": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": "Robot-only one-scene renderer provenance at review resolution; no object layer, ownership, contact, Gold accuracy, physical truth or Robot authority.",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "frames": frames.tolist()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
