#!/usr/bin/env python3
"""Build and epoch-0 validate explicit low-weight auto-Object Grade-B canaries.

This does not relabel an estimate as formal Object6D.  It copies the existing
RGB/HaWoR/stereo-derived review sidecar, optionally replaces Poker's visible
card trajectory with the reviewed V5b estimate, and fills only missing anchor
frames with explicitly labelled bounded temporal estimates.  UNKNOWN remains
PAD in the loader; therefore a session is Grade B only if every anchor frame
has a real, provenance-labelled estimate after this operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
sys.path.insert(0, str(HUMANEGO))
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from utils.utils_math import normalize_o6d, rotmat_to_o6d  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def projected_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    candidate = (1.0 - alpha) * left + alpha * right
    u, _, vt = np.linalg.svd(candidate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def fill_anchor_estimates(values: dict[str, np.ndarray]) -> int:
    transforms = values["T_object_to_camera"]
    valid = values["valid"]
    confidence = values["confidence"]
    provenance = values["provenance"]
    good = np.flatnonzero(valid[:, 0])
    if good.size == 0:
        raise RuntimeError("no anchor estimate exists; temporal Grade B is impossible")
    filled = 0
    for frame in np.flatnonzero(~valid[:, 0]):
        before = good[good < frame]
        after = good[good > frame]
        if before.size and after.size:
            left, right = int(before[-1]), int(after[0])
            alpha = (frame - left) / (right - left)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = (
                (1.0 - alpha) * transforms[left, 0, :3, 3]
                + alpha * transforms[right, 0, :3, 3]
            )
            transform[:3, :3] = projected_rotation(
                transforms[left, 0, :3, :3], transforms[right, 0, :3, :3], alpha
            )
            source_confidence = min(confidence[left, 0], confidence[right, 0])
            source = "AUTO_TEMPORAL_INTERPOLATED_RGB_HAWOR_OBJECT_POSE"
        else:
            source_index = int(before[-1] if before.size else after[0])
            transform = transforms[source_index, 0].copy()
            source_confidence = confidence[source_index, 0]
            source = "AUTO_TEMPORAL_BOUNDARY_PROPAGATED_RGB_HAWOR_OBJECT_POSE"
        transforms[frame, 0] = transform
        valid[frame, 0] = True
        confidence[frame, 0] = max(0.20, float(source_confidence) * 0.55)
        provenance[frame, 0] = source
        filled += 1
    return filled


def apply_poker_v5b(values: dict[str, np.ndarray], path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        valid = np.asarray(archive["valid"], dtype=bool)
        transforms = np.asarray(archive["T_object_to_camera"], dtype=np.float64)
    used = 0
    for local in np.flatnonzero(valid):
        frame = int(indices[local])
        if not 0 <= frame < len(values["valid"]):
            continue
        values["T_object_to_camera"][frame, 0] = transforms[local]
        values["valid"][frame, 0] = True
        values["confidence"][frame, 0] = 0.78
        values["provenance"][frame, 0] = "AUTO_V5B_RGB_STEREO_CARD_TRAJECTORY"
        used += 1
    return used


def rebuild_pose9(values: dict[str, np.ndarray]) -> None:
    transforms = values["T_object_to_camera"]
    valid = values["valid"]
    pose9 = np.full((len(valid), 2, 9), np.nan, dtype=np.float32)
    for frame, obj in np.argwhere(valid):
        transform = transforms[frame, obj]
        pose9[frame, obj, :3] = transform[:3, 3]
        pose9[frame, obj, 3:] = normalize_o6d(rotmat_to_o6d(transform[:3, :3]))
    values["object_pose9_camera"] = pose9
    values["formal_object6d_valid"] = np.zeros_like(valid, dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--poker-v5b", type=Path)
    args = parser.parse_args()
    source = args.source_bundle.resolve(strict=True)
    destination = args.output_bundle.absolute()
    if destination.exists():
        raise SystemExit(f"no-clobber output exists: {destination}")
    contract = json.loads((source / "CANARY_CONTRACT.json").read_text(encoding="utf-8"))
    session = contract["session"]
    mps_path = Path(contract["mps_path"]).resolve(strict=True)
    source_object = source / "object_state_sidecars" / session
    source_hawor = source / "hawor_v3_sidecars" / session / "entities_hawor_v3.npz"
    source_npz = source_object / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    source_json = source_object / "AUTO_ESTIMATED_OBJECT_STATE.json"
    source_manifest = json.loads(source_json.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != "auto-estimated-object-state-review-v2"
        or bool(source_manifest.get("consumption_authorized"))
        or source_manifest.get("npz", {}).get("sha256") != digest(source_npz)
    ):
        raise RuntimeError("source must be digest-bound review-only auto estimate")
    with np.load(source_npz, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    # Avoid silently truncating the explicit temporal/V5b provenance labels to
    # the source archive's historical U40 dtype.
    values["provenance"] = values["provenance"].astype("<U96")
    v5b_used = 0
    if args.poker_v5b is not None:
        v5b_used = apply_poker_v5b(values, args.poker_v5b.resolve(strict=True))
    filled = fill_anchor_estimates(values)
    rebuild_pose9(values)
    if not bool(values["valid"][:, 0].all()):
        raise RuntimeError("Grade-B anchor is not nonzero on every frame")

    object_dir = destination / "object_state_sidecars" / session
    hawor_dir = destination / "hawor_v3_sidecars" / session
    object_dir.mkdir(parents=True)
    hawor_dir.mkdir(parents=True)
    output_npz = object_dir / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    np.savez_compressed(output_npz, **values)
    shutil.copy2(source_hawor, hawor_dir / source_hawor.name)
    thresholds = {
        "AUTO_CV_RAY_HAWOR_FINGERTIP_Z": 0.18,
        "AUTO_CV_STATIC_SIZE_PRIOR": 0.18,
        "AUTO_STATIC_FIXTURE_PROPAGATION": 0.18,
        "AUTO_TEMPORAL_INTERPOLATED_RGB_HAWOR_OBJECT_POSE": 0.18,
        "AUTO_TEMPORAL_BOUNDARY_PROPAGATED_RGB_HAWOR_OBJECT_POSE": 0.18,
        "AUTO_V5B_RGB_STEREO_CARD_TRAJECTORY": 0.70,
    }
    output_json = object_dir / "AUTO_ESTIMATED_OBJECT_STATE.json"
    grade_b_manifest = {
        "schema_version": "humanego-auto-estimated-object-grade-b-v1",
        "task": args.task, "session": session,
        "quality_grade": "B", "training_weight": 0.25,
        "consumption_authorized": True, "claims_formal_object6d": False,
        "array_contract": source_manifest["array_contract"],
        "confidence_thresholds": thresholds,
        "unknown_policy": "UNKNOWN or below-threshold remains PAD; no identity/zero pose substitution",
        "per_frame_anchor_required": True,
        "source_provenance": {
            "review_manifest": ref(source_json), "review_npz": ref(source_npz),
            "poker_v5b": ref(args.poker_v5b) if args.poker_v5b else None,
            "temporal_fill_count": filled, "v5b_direct_frame_count": v5b_used,
            "claim_limit": "automatic RGB/HaWoR/stereo/task-prior estimate; not measured or formal Object6D",
        },
        "npz": ref(output_npz),
    }
    atomic_json(output_json, grade_b_manifest)

    hawor_npz = hawor_dir / "entities_hawor_v3.npz"
    loader = FlowMatchingDataloader(
        sessions=[MPSSessions(str(mps_path))], pred_horizon=1, single_hand=False,
        max_ict=8, img_name=None, centric_mode="object_centric",
        frame_mode="camera_frame", hand_tracking_method="hawor_v3",
        hawor_v3_sidecar_root=str(destination / "hawor_v3_sidecars"),
        hawor_v3_sha256_by_session={session: digest(hawor_npz)},
        object_state_sidecar_root=str(destination / "object_state_sidecars"),
        object_state_npz_sha256_by_session={session: digest(output_npz)},
        object_state_json_sha256_by_session={session: digest(output_json)},
        object_state_consumption_mode="training_estimated_grade_b",
        object_state_confidence_thresholds=thresholds,
        use_pcd_features=False, use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False, use_aux_temporal_contrastive=False,
        enable_augmentation=False,
    )
    types_by_frame = []
    for path in loader.samples:
        frame = loader._read_frame(path)
        state, _, mask = loader._build_ict(frame, loader._get_T_w2ref(frame))
        types = state[mask, 0].tolist()
        if 3.0 not in types:
            raise RuntimeError(f"epoch0 loader missing anchor token: {path}")
        types_by_frame.append(types)
    first = loader[0]
    report_path = destination / "EPOCH0_LOADER_REPORT.json"
    atomic_json(report_path, {
        "schema_version": "humanego-grade-b-object-epoch0-loader-report-v1",
        "status": "PASS_REAL_LOADER_EPOCH0",
        "task": args.task, "session": session, "frames_checked": len(types_by_frame),
        "anchor_token_frames": sum(3.0 in values for values in types_by_frame),
        "all_pad_object_frames": sum(not ({3.0, 4.0} & set(values)) for values in types_by_frame),
        "first_batch_object_training_weight": float(first["object_training_weight"]),
        "admission": loader.object_state_admission_report(),
        "unknown_not_faked": True, "heldout_included": False,
        "files": {"manifest": ref(output_json), "npz": ref(output_npz), "hawor": ref(hawor_npz)},
    })
    freeze_path = destination / "FREEZE.json"
    atomic_json(freeze_path, {
        "schema_version": "humanego-grade-b-object-canary-freeze-v1",
        "training_allowed": True, "quality_grade": "B", "training_weight": 0.25,
        "heldout_included": False,
        "files": {"object_json": ref(output_json), "object_npz": ref(output_npz),
                  "hawor": ref(hawor_npz), "epoch0": ref(report_path)},
    })
    print(json.dumps({"status": "PASS", "bundle": str(destination), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
