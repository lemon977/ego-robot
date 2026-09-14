#!/usr/bin/env python3
"""Build one fresh CPU-only HaWoR + object-ICT diagnostic canary.

The resulting directory is deliberately not a training bundle.  It freezes
one source session, a canonical HaWoR entity overlay, and the review-only
AUTO_ESTIMATED object sidecar so the real dataloader path can be replayed
without granting the object estimate training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


PROJECT = Path("/mnt/workspace/code/chaoyang")
V2_BUILDER = PROJECT / "NOW/daemon/tools/build_robot_humanego_ict_bundle_v2.py"
SPEC = importlib.util.spec_from_file_location("hawor_ict_builder_v2", V2_BUILDER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {V2_BUILDER}")
hawor_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hawor_builder)


DEFAULT_THRESHOLDS = {
    "AUTO_CV_RAY_HAWOR_FINGERTIP_Z": 0.65,
    "AUTO_CV_STATIC_SIZE_PRIOR": 0.60,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--mps-path", type=Path, required=True)
    parser.add_argument("--object-run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    source_mps = args.mps_path.resolve(strict=True)
    object_run = args.object_run.resolve(strict=True)
    destination = args.bundle.absolute()
    if destination.exists():
        raise SystemExit(f"refusing to overwrite canary bundle: {destination}")
    if source_mps.name != args.session:
        raise ValueError("source MPS basename must equal --session")
    source_json = object_run / "AUTO_ESTIMATED_OBJECT_STATE.json"
    source_npz = object_run / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    manifest = json.loads(source_json.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "auto-estimated-object-state-review-v2"
        or manifest.get("session") != args.session
        or manifest.get("task") != args.task
        or bool(manifest.get("consumption_authorized"))
    ):
        raise ValueError("object review sidecar manifest contract mismatch")
    if manifest.get("npz", {}).get("sha256") != sha256_file(source_npz):
        raise ValueError("object review NPZ differs from its manifest")

    destination.mkdir(parents=True)
    hawor_root = destination / "hawor_v3_sidecars"
    hawor_row = hawor_builder.build_session(
        session=args.session,
        task=args.task,
        base_bundle=Path("/"),
        destination=hawor_root,
        mps_path=source_mps,
    )
    object_dir = destination / "object_state_sidecars" / args.session
    object_dir.mkdir(parents=True)
    object_json = object_dir / source_json.name
    object_npz = object_dir / source_npz.name
    shutil.copy2(source_json, object_json)
    shutil.copy2(source_npz, object_npz)

    contract = {
        "schema_version": "humanego-object-ict-v3-canary-v1",
        "task": args.task,
        "session": args.session,
        "training_allowed": False,
        "purpose": "two-session CPU diagnostic replay only",
        "mps_path": str(source_mps),
        "hawor_v3_sidecar_root": str(hawor_root.resolve()),
        "object_state_sidecar_root": str(
            (destination / "object_state_sidecars").resolve()
        ),
        "object_state_consumption_mode": "diagnostic_estimated",
        "object_state_confidence_thresholds": DEFAULT_THRESHOLDS,
        "ict_contract": {
            "single_hand": False,
            "ict_dim": 29,
            "max_ict": 8,
            "frame_mode": "camera_frame",
            "translation_unit": "metre",
            "camera_coordinate_system": "x_right_y_down_z_forward",
            "token_order": [
                "left_hand=1", "right_hand=2", "anchor_manipulated=3",
                "other_static_fixture=4", "PAD=0",
            ],
            "object_pose9": "camera xyz metres + rotation6d before stats normalization",
            "unknown_policy": "UNKNOWN, unlisted provenance, or confidence below threshold => PAD",
        },
        "authority": {
            "object_schema": manifest["schema_version"],
            "object_consumption_authorized": False,
            "formal_object6d_valid": False,
            "claim_limit": (
                "AUTO_ESTIMATED diagnostic tokens only; hand-depth and image-yaw "
                "proxies are not measured Object6D"
            ),
        },
        "sources": {
            "hawor": hawor_row,
            "object_json": identity(object_json),
            "object_npz": identity(object_npz),
        },
    }
    contract_path = destination / "CANARY_CONTRACT.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "schema_version": "humanego-object-ict-v3-canary-freeze-v1",
        "training_allowed": False,
        "files": {
            "CANARY_CONTRACT.json": identity(contract_path),
            f"hawor_v3_sidecars/{args.session}/entities_hawor_v3.npz": identity(
                hawor_root / args.session / "entities_hawor_v3.npz"
            ),
            f"object_state_sidecars/{args.session}/{object_json.name}": identity(
                object_json
            ),
            f"object_state_sidecars/{args.session}/{object_npz.name}": identity(
                object_npz
            ),
        },
    }
    freeze_path = destination / "freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE_CPU_CANARY_BUNDLE",
        "bundle": str(destination.resolve()),
        "contract": str(contract_path.resolve()),
        "freeze": str(freeze_path.resolve()),
        "training_allowed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
