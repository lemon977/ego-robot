#!/usr/bin/env python3
"""Adapt legacy rectified-camera Object6D poses into the selected RGB camera.

This does not modify or promote the source Object6D artifact.  It emits a
development-only immutable sidecar so Robot/Contact experiments cannot silently
mix the rectified-depth camera with the selected-left RGB c2w trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def adapt_poses(
    t_rectified_object: np.ndarray,
    rectified_to_selected: np.ndarray,
    c2w_selected: np.ndarray,
    frame_indices: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if t_rectified_object.ndim != 3 or t_rectified_object.shape[1:] != (4, 4):
        raise ValueError("T_object_to_camera must have shape [N,4,4]")
    if rectified_to_selected.shape != (4, 4):
        raise ValueError("registration transform must have shape [4,4]")
    if c2w_selected.ndim != 3 or c2w_selected.shape[1:] != (4, 4):
        raise ValueError("selected camera-to-world must have shape [M,4,4]")
    if len(frame_indices) != len(t_rectified_object) or len(valid) != len(frame_indices):
        raise ValueError("frame identity closure mismatch")

    selected = np.einsum("ij,njk->nik", rectified_to_selected, t_rectified_object)
    world = np.tile(np.eye(4, dtype=np.float64), (len(selected), 1, 1))
    for local, absolute in enumerate(frame_indices.astype(np.int64)):
        if not valid[local]:
            continue
        camera = c2w_selected[local] if len(c2w_selected) == len(selected) else c2w_selected[int(absolute)]
        world[local] = camera @ selected[local]
    return selected, world


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--object6d", type=Path, required=True)
    parser.add_argument("--registration-authority", type=Path, required=True)
    parser.add_argument("--camera-to-world", type=Path, required=True)
    parser.add_argument("--camera-to-world-key", default="c2w")
    parser.add_argument("--artifact-revision", default="R7_1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = (args.object6d, args.registration_authority, args.camera_to_world)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"no-clobber output exists: {args.output_dir}")
    partial = args.output_dir.with_name(f".{args.output_dir.name}.partial")
    if partial.exists():
        raise FileExistsError(f"partial output exists: {partial}")
    partial.mkdir(parents=True)

    with np.load(args.object6d, allow_pickle=False) as source:
        frame_indices = source["frame_indices"].astype(np.int64)
        valid = source["valid"].astype(bool)
        t_rectified = source["T_object_to_camera"].astype(np.float64)
        legacy_world = source["T_object_to_world"].astype(np.float64)
    with np.load(args.registration_authority, allow_pickle=False) as registration:
        rectified_to_selected = registration[
            "T_stereo_rectified_camera_to_selected_camera"
        ].astype(np.float64)
    with np.load(args.camera_to_world, allow_pickle=False) as camera:
        c2w_selected = camera[args.camera_to_world_key].astype(np.float64)

    selected, corrected_world = adapt_poses(
        t_rectified, rectified_to_selected, c2w_selected, frame_indices, valid
    )
    if valid.any() and not (
        np.isfinite(selected[valid]).all() and np.isfinite(corrected_world[valid]).all()
    ):
        raise RuntimeError("corrected valid poses contain non-finite values")

    output = partial / "OBJECT6D_SELECTED_CAMERA_ADAPTER.npz"
    np.savez_compressed(
        output,
        frame_indices=frame_indices,
        valid=valid,
        T_object_to_rectified_camera=t_rectified,
        T_object_to_selected_camera=selected,
        T_object_to_world_corrected=corrected_world,
        T_object_to_world_legacy=legacy_world,
        T_stereo_rectified_camera_to_selected_camera=rectified_to_selected,
        source_coordinate_domain=np.asarray("STEREO_RECTIFIED_DEPTH_CAMERA"),
        target_coordinate_domain=np.asarray("SELECTED_LEFT_RGB_CAMERA"),
        control_ground_truth=np.asarray(False),
    )
    valid_delta = np.linalg.norm(
        corrected_world[valid, :3, 3] - legacy_world[valid, :3, 3], axis=1
    ) if valid.any() else np.asarray([], dtype=np.float64)
    output_reference = artifact(output)
    output_reference["path"] = str((args.output_dir / output.name).resolve())
    result = {
        "schema_version": "object6d-selected-camera-adapter-result-v1",
        "artifact_revision": args.artifact_revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DEVELOPMENT_COORDINATE_ADAPTER",
        "session_id": args.session_id,
        "coordinate_contract": (
            "T_selected_object = T_stereo_rectified_camera_to_selected_camera "
            "@ T_rectified_object; T_world_object = selected_camera_c2w @ T_selected_object"
        ),
        "valid_frames": int(valid.sum()),
        "registration_rotation_deg": rotation_angle_deg(rectified_to_selected[:3, :3]),
        "registration_translation_m": rectified_to_selected[:3, 3].tolist(),
        "legacy_vs_corrected_world_translation_m": {
            "median": float(np.median(valid_delta)) if valid_delta.size else None,
            "max": float(np.max(valid_delta)) if valid_delta.size else None,
        },
        "inputs": [artifact(path) for path in inputs],
        "outputs": [output_reference],
        "authorized_scopes": ["ROBOT_CONTACT_DEVELOPMENT_COORDINATE_INPUT"],
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Coordinate-domain correction only. It does not validate Object6D against external truth, "
            "does not promote source Object6D, and does not grant Contact/Robot/physical authority."
        ),
    }
    atomic_json(partial / "RESULT.json", result)
    os.replace(partial, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
