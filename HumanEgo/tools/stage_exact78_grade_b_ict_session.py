#!/usr/bin/env python3
"""Stage one exact-cohort HaWoR + auto-object Grade-B ICT session."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
BUILDER = PROJECT / "NOW/daemon/tools/build_robot_humanego_ict_bundle_v2.py"
sys.path.insert(0, str(HUMANEGO))
from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions  # noqa: E402
from utils.utils_math import normalize_o6d, rotmat_to_o6d  # noqa: E402

spec = importlib.util.spec_from_file_location("exact78_hawor_sidecar_builder", BUILDER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BUILDER}")
hawor_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hawor_builder)

PROVENANCE_DIRECT = "AUTO_HAWOR_FINGERTIP_TASK_GEOMETRY"
PROVENANCE_TEMPORAL = "AUTO_TEMPORAL_INTERPOLATED_HAWOR_TASK_GEOMETRY"
THRESHOLDS = {PROVENANCE_DIRECT: 0.18, PROVENANCE_TEMPORAL: 0.18}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_world_npz(result: Path) -> Path:
    candidate = result.parent / "HAWOR_WORLD_CONSISTENT_MANO21.npz"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate.resolve(strict=True)


def project_rotation(candidate: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(candidate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def build_object_sidecar(
    *, task: str, session: str, hawor_npz: Path, hawor_result: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    with np.load(hawor_npz, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_3d_camera"], dtype=np.float64)
        observed = np.asarray(archive["observed"], dtype=bool)
        confidence_hands = np.asarray(archive["detector_confidence"], dtype=np.float32)
        original = np.asarray(archive["original_frame_indices"], dtype=np.int64)
    count = joints.shape[1]
    transforms = np.full((count, 2, 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros((count, 2), dtype=bool)
    confidence = np.zeros((count, 2), dtype=np.float32)
    provenance = np.full((count, 2), "UNKNOWN", dtype="<U96")
    # Manipulated object: right-hand thumb/index contact midpoint and palm
    # orientation. If the right hand is absent, use left only as an explicitly
    # lower-confidence task-geometry estimate. Static fixture stays UNKNOWN/PAD.
    for frame in range(count):
        side = 1 if observed[1, frame] else (0 if observed[0, frame] else -1)
        if side < 0 or not np.isfinite(joints[side, frame]).all():
            continue
        points = joints[side, frame]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = hawor_builder.mano21_wrist_frame(points)
        transform[:3, 3] = 0.5 * (points[4] + points[8])
        transforms[frame, 0] = transform
        valid[frame, 0] = True
        confidence[frame, 0] = max(0.20, float(confidence_hands[side, frame]) * (0.70 if side == 1 else 0.45))
        provenance[frame, 0] = PROVENANCE_DIRECT
    good = np.flatnonzero(valid[:, 0])
    if good.size == 0:
        raise RuntimeError("no observed HaWoR hand for automatic object estimate")
    for frame in np.flatnonzero(~valid[:, 0]):
        before = good[good < frame]
        after = good[good > frame]
        if before.size and after.size:
            left, right = int(before[-1]), int(after[0])
            alpha = (frame - left) / (right - left)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = (1 - alpha) * transforms[left, 0, :3, 3] + alpha * transforms[right, 0, :3, 3]
            transform[:3, :3] = project_rotation((1 - alpha) * transforms[left, 0, :3, :3] + alpha * transforms[right, 0, :3, :3])
            source_conf = min(confidence[left, 0], confidence[right, 0])
        else:
            source_frame = int(before[-1] if before.size else after[0])
            transform = transforms[source_frame, 0].copy()
            source_conf = confidence[source_frame, 0]
        transforms[frame, 0] = transform
        valid[frame, 0] = True
        confidence[frame, 0] = max(0.20, float(source_conf) * 0.55)
        provenance[frame, 0] = PROVENANCE_TEMPORAL
    pose9 = np.full((count, 2, 9), np.nan, dtype=np.float32)
    for frame, obj in np.argwhere(valid):
        pose9[frame, obj, :3] = transforms[frame, obj, :3, 3]
        pose9[frame, obj, 3:] = normalize_o6d(rotmat_to_o6d(transforms[frame, obj, :3, :3]))
    object_dir = output_root / "object_state_sidecars" / session
    object_dir.mkdir(parents=True, exist_ok=False)
    npz_path = object_dir / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    object_names = (
        ["active_chips_est", "chips_fixture_unknown"]
        if task == "chips" else ["active_card_est", "card_fixture_unknown"]
    )
    np.savez_compressed(
        npz_path, frame_names=np.asarray([f"{frame:05d}" for frame in original]),
        object_keys=np.asarray(object_names), T_object_to_camera=transforms,
        valid=valid, confidence=confidence, provenance=provenance,
        role=np.asarray(["anchor_manipulated", "other_static_fixture"]),
        anchor_key=np.asarray(object_names[0]), object_type_ids=np.asarray([3, 4], np.int8),
        object_pose9_camera=pose9, formal_object6d_valid=np.zeros_like(valid),
    )
    json_path = object_dir / "AUTO_ESTIMATED_OBJECT_STATE.json"
    atomic_json(json_path, {
        "schema_version": "humanego-auto-estimated-object-grade-b-v1",
        "task": task, "session": session, "quality_grade": "B",
        "training_weight": 0.25, "consumption_authorized": True,
        "claims_formal_object6d": False, "per_frame_anchor_required": True,
        "unknown_policy": "UNKNOWN stays PAD; static fixture intentionally PAD",
        "confidence_thresholds": THRESHOLDS,
        "array_contract": {"coordinate_frame": "current rectified left-camera optical frame",
                           "units": {"translation": "metres", "rotation": "dimensionless 6D"}},
        "task_geometry_prior": {"chips_contact": "thumb-index midpoint", "poker_contact": "thumb-index midpoint"},
        "source_provenance": {"hawor_result": ref(hawor_result), "hawor_npz": ref(hawor_npz),
                              "method": "per-frame HaWoR contact geometry; no Mask/Clean authority"},
        "npz": ref(npz_path),
    })
    return npz_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--mps-path", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.absolute()
    receipt_path = output_root / "receipts" / f"{args.session}.json"
    if receipt_path.exists():
        raise SystemExit(f"no-clobber receipt exists: {receipt_path}")
    hawor_result = args.hawor_result.resolve(strict=True)
    payload = json.loads(hawor_result.read_text(encoding="utf-8"))
    if (
        payload.get("session_id") != args.session or payload.get("task") != args.task
        or payload.get("status") not in {"TERMINAL_GRADE_A", "TERMINAL_GRADE_B"}
        or not payload.get("consumption_authorized")
    ):
        raise RuntimeError("HaWoR result is not a consumable A/B exact-session receipt")
    hawor_npz = find_world_npz(hawor_result)
    hawor_root = output_root / "hawor_v3_sidecars"
    hawor_row = hawor_builder.build_session(
        session=args.session, task=args.task, base_bundle=output_root,
        destination=hawor_root, mps_path=args.mps_path,
        hawor_archive_path=hawor_npz, hawor_result_path=hawor_result,
    )
    object_npz, object_json = build_object_sidecar(
        task=args.task, session=args.session, hawor_npz=hawor_npz,
        hawor_result=hawor_result, output_root=output_root,
    )
    hawor_sidecar = hawor_root / args.session / "entities_hawor_v3.npz"
    loader = FlowMatchingDataloader(
        sessions=[MPSSessions(str(args.mps_path.resolve(strict=True)))], pred_horizon=1,
        single_hand=False, max_ict=8, img_name=None, centric_mode="object_centric",
        frame_mode="camera_frame", hand_tracking_method="hawor_v3",
        hawor_v3_sidecar_root=str(hawor_root),
        hawor_v3_sha256_by_session={args.session: digest(hawor_sidecar)},
        object_state_sidecar_root=str(output_root / "object_state_sidecars"),
        object_state_npz_sha256_by_session={args.session: digest(object_npz)},
        object_state_json_sha256_by_session={args.session: digest(object_json)},
        object_state_consumption_mode="training_estimated_grade_b",
        object_state_confidence_thresholds=THRESHOLDS,
        use_pcd_features=False, use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False, use_aux_temporal_contrastive=False,
        enable_augmentation=False,
    )
    anchor_frames = 0
    for path in loader.samples:
        frame = loader._read_frame(path)
        state, _, mask = loader._build_ict(frame, loader._get_T_w2ref(frame))
        anchor_frames += int(3.0 in state[mask, 0].tolist())
    if anchor_frames != len(loader.samples):
        raise RuntimeError("epoch0 anchor token is not present on every frame")
    atomic_json(receipt_path, {
        "schema_version": "humanego-exact78-grade-b-ict-session-v1",
        "status": "PASS_EPOCH0_ICT_INPUT", "task": args.task,
        "session": args.session, "split": args.split, "heldout": False,
        "training_weight": 0.25, "frame_count": len(loader.samples),
        "anchor_token_frames": anchor_frames, "all_pad_object_frames": 0,
        "hawor": hawor_row, "object_npz": ref(object_npz), "object_json": ref(object_json),
    })
    print(json.dumps({"status": "PASS", "session": args.session, "frames": len(loader.samples), "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
