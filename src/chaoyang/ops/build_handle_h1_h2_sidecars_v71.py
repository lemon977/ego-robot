#!/usr/bin/env python3
"""Materialize V7.1 H1 canonical-hand and H2 tactile sidecars.

This is a read-only adapter over the acquisition-aligned HDF5 snapshots named
by the H0 ledger.  It never writes below the dataset root.  Every H0 row gets
one terminal row, so a missing Controller wrist blocks H1/Robot eligibility
without blocking the independent H2 tactile audit.

H1 is deliberately named ``HAND21_FROM_MANUS_CONTROLLER``.  It is a lossy,
explicit 25-to-21 landmark adapter anchored by the same-session calibrated
Controller wrist; it is neither HaWoR/MANO nor anatomical/external truth.

H2 preserves raw integer taxels, side/finger identity, timestamps, validity
and sample age.  No pressure-to-force calibration, object identity, palm
taxels, or contact threshold is invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


STAGES = {"h1", "h2"}
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANUS25_NAMES = (
    "Hand_Invalid",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_IP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_IP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_IP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_IP", "Pinky_DIP", "Pinky_TIP",
)
# Custom 21-point observation: Controller wrist plus MCP/PIP/DIP/TIP for each
# digit.  The MANUS-only intermediate ``*_IP`` node is intentionally omitted.
HAND21_NAMES = (
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
HAND21_FROM_MANUS25 = (-1, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24)
H1_STATES = {
    "PASSED", "BLOCKED_SOURCE", "BLOCKED_PREREQ_CONTROLLER_WRIST",
    "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL",
}
H2_STATES = {"PASSED", "BLOCKED_SOURCE", "FAILED_QUALITY_C", "FAILED_RUNTIME_FINAL"}


class SidecarContractError(RuntimeError):
    """A session cannot satisfy a sidecar contract without fabrication."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SidecarContractError(f"expected JSON object: {path}")
    return value


def _source_contract(row: dict[str, Any]) -> tuple[Path, str, Path | None]:
    target_text = row.get("target")
    if not isinstance(target_text, str):
        raise SidecarContractError("H0 row has no published target")
    target = Path(target_text)
    clip_path = target / "clip_manifest.json"
    if not clip_path.is_file():
        raise SidecarContractError(f"missing clip manifest: {clip_path}")
    clip = _load_object(clip_path)
    acquisition = clip.get("acquisition_aligned_hdf5_contract", {})
    hdf5_text = acquisition.get("path")
    hdf5_sha = acquisition.get("sha256")
    if not isinstance(hdf5_text, str) or not isinstance(hdf5_sha, str) or len(hdf5_sha) != 64:
        raise SidecarContractError("missing acquisition-aligned HDF5 path/SHA")
    hdf5_path = Path(hdf5_text)
    if not hdf5_path.is_file():
        raise SidecarContractError(f"source HDF5 missing: {hdf5_path}")
    files = clip.get("files", {})
    controller = files.get("controller_poses")
    controller_path = target / controller if isinstance(controller, str) else None
    return hdf5_path, hdf5_sha, controller_path


def _read_controller_wrist(path: Path, n: int, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise SidecarContractError(f"Controller wrist sidecar missing: {path}")
    world = np.empty((n, 2, 4, 4), dtype=np.float64)
    camera = np.empty_like(world)
    count = 0
    with path.open() as stream:
        for count, line in enumerate(stream, start=1):
            if count > n:
                raise SidecarContractError("Controller sidecar has extra frames")
            record = json.loads(line)
            index = count - 1
            if record.get("frame_index") != index or record.get("timeStampNs") != int(timestamps[index]):
                raise SidecarContractError(f"Controller frame identity mismatch at {index}")
            hands = record.get("hands", {})
            for side_index, side in enumerate(SIDES):
                hand = hands.get(side, {})
                world[index, side_index] = np.asarray(hand.get("T_wrist_to_world"), dtype=np.float64)
                camera[index, side_index] = np.asarray(hand.get("T_wrist_to_camera"), dtype=np.float64)
    if count != n:
        raise SidecarContractError(f"Controller line count mismatch: {count}/{n}")
    if not np.isfinite(world).all() or not np.isfinite(camera).all():
        raise SidecarContractError("non-finite Controller wrist transform")
    if not np.allclose(world[:, :, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise SidecarContractError("invalid homogeneous wrist transform")
    return world, camera


def _transform(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return np.einsum("tsij,tskj->tski", transforms[:, :, :3, :3], points) + transforms[:, :, None, :3, 3]


def materialize_h1(hdf5_path: Path, controller_path: Path, output: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as source:
        names = tuple(json.loads(str(source.attrs.get("joint_names", "[]"))))
        if names != MANUS25_NAMES or int(source.attrs.get("n_joint", -1)) != 25:
            raise SidecarContractError("MANUS25 joint contract mismatch")
        timestamps = np.asarray(source["timestamp_ns"], dtype=np.int64)
        n = len(timestamps)
        local25 = np.stack(
            [np.asarray(source[f"{side}_hand_joints"], dtype=np.float32) for side in SIDES], axis=1
        )
        valid = np.stack(
            [np.asarray(source[f"{side}_hand_valid"], dtype=bool) for side in SIDES], axis=1
        )
        source_rows = np.asarray(source["source_row_idx"], dtype=np.int64)
        video_frames = np.asarray(source["video_frame_idx"], dtype=np.int32)
        segments = np.asarray(source["segment_id"], dtype=np.int32)
    if local25.shape != (n, 2, 25, 3):
        raise SidecarContractError(f"MANUS25 shape mismatch: {local25.shape}")
    world_t, camera_t = _read_controller_wrist(controller_path, n, timestamps)
    local21 = np.zeros((n, 2, 21, 3), dtype=np.float32)
    local21[:, :, 1:] = local25[:, :, list(HAND21_FROM_MANUS25[1:])]
    world21 = _transform(local21.astype(np.float64), world_t).astype(np.float32)
    camera21 = _transform(local21.astype(np.float64), camera_t).astype(np.float32)
    joint_valid = np.broadcast_to(valid[:, :, None], (n, 2, 21)).copy()
    if not np.isfinite(local21[joint_valid]).all() or not np.isfinite(world21[joint_valid]).all():
        raise SidecarContractError("non-finite valid canonical hand point")
    np.savez_compressed(
        output,
        schema_version=np.asarray("handle-hand21-manus-controller-v7.1"),
        observation_type=np.asarray("HAND21_FROM_MANUS_CONTROLLER"),
        timestamp_ns=timestamps,
        source_row_idx=source_rows,
        video_frame_idx=video_frames,
        segment_id=segments,
        hand_valid=valid,
        joint_valid=joint_valid,
        joint_xyz_wrist_local_m=local21,
        joint_xyz_world_m=world21,
        joint_xyz_camera_m=camera21,
        T_wrist_to_world=world_t,
        T_wrist_to_camera=camera_t,
        hand21_from_manus25=np.asarray(HAND21_FROM_MANUS25, dtype=np.int16),
    )
    return {"frames": n, "valid_hand_frames": int(valid.sum()), "both_valid_frames": int(valid.all(axis=1).sum())}


def materialize_h2(hdf5_path: Path, output: Path) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as source:
        timestamps = np.asarray(source["timestamp_ns"], dtype=np.int64)
        n = len(timestamps)
        grids = np.stack(
            [np.asarray(source[f"{side}_tactile_fingers"], dtype=np.int16) for side in SIDES], axis=1
        )
        active = np.stack(
            [np.asarray(source[f"{side}_tactile_fingers_active_mask"], dtype=bool) for side in SIDES], axis=0
        )
        valid = np.stack(
            [np.asarray(source[f"{side}_tactile_valid"], dtype=bool) for side in SIDES], axis=1
        )
        offsets = np.stack(
            [np.asarray(source[f"{side}_tactile_offset_ms"], dtype=np.float32) for side in SIDES], axis=1
        )
        record_seq = np.stack(
            [np.asarray(source[f"{side}_tactile_record_seq"], dtype=np.int64) for side in SIDES], axis=1
        )
        stream_seq = np.stack(
            [np.asarray(source[f"{side}_tactile_stream_seq"], dtype=np.int64) for side in SIDES], axis=1
        )
        source_rows = np.asarray(source["source_row_idx"], dtype=np.int64)
        video_frames = np.asarray(source["video_frame_idx"], dtype=np.int32)
        segments = np.asarray(source["segment_id"], dtype=np.int32)
        gate_ms = float(source.attrs.get("tactile_gate_ms", 40.0))
        mapping_schema = str(source.attrs.get("tactile_finger_mapping_schema", ""))
        palm_present = bool(source.attrs.get("tactile_palm_present", False))
    if grids.shape != (n, 2, 5, 4, 8) or active.shape != (2, 5, 4, 8):
        raise SidecarContractError(f"tactile grid contract mismatch: {grids.shape}, {active.shape}")
    if palm_present:
        raise SidecarContractError("V7.1 five-fingertip contract unexpectedly reports palm taxels")
    if mapping_schema != "hs13_five_fingertip_4x8_four_4x7_v1":
        raise SidecarContractError(f"unknown tactile mapping schema: {mapping_schema}")
    if not valid.all():
        raise SidecarContractError("HDF5 complete-frame contract contains invalid tactile rows")
    if np.any(np.abs(offsets[valid]) > gate_ms + 1e-4):
        raise SidecarContractError("tactile sample age exceeds source gate")
    taxel_valid = valid[:, :, None, None, None] & active[None]
    np.savez_compressed(
        output,
        schema_version=np.asarray("handle-tactile-five-finger-sidecar-v7.1"),
        timestamp_ns=timestamps,
        source_row_idx=source_rows,
        video_frame_idx=video_frames,
        segment_id=segments,
        side_valid=valid,
        sample_offset_ms=offsets,
        record_seq=record_seq,
        stream_seq=stream_seq,
        finger_grid_raw_int16=grids,
        finger_active_mask=active,
        finger_valid_mask=taxel_valid,
    )
    return {
        "frames": n,
        "valid_side_frames": int(valid.sum()),
        "max_abs_sample_age_ms": float(np.max(np.abs(offsets))) if n else 0.0,
        "tactile_gate_ms": gate_ms,
    }


def _blocked_row(row: dict[str, Any], stage: str) -> dict[str, Any]:
    status = "BLOCKED_SOURCE"
    return {
        "session_id": row["session_id"], "dataset_id": row["dataset_id"], "task": row["task"],
        "frame_count": row.get("frame_count"), "status": status,
        "primary_blocker": "H0_BLOCKED_SOURCE", "artifact": None,
        "downstream": (
            {"hand_observation_eligible": False, "robot_geometry_eligible": False}
            if stage == "h1" else {"tactile_event_timing_eligible": False}
        ),
    }


def build_stage(h0_path: Path, output_root: Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(stage)
    if output_root.exists():
        raise FileExistsError(f"immutable output already exists: {output_root}")
    h0 = _load_object(h0_path)
    rows = h0.get("rows")
    if not isinstance(rows, list) or h0.get("row_count") != len(rows):
        raise SidecarContractError("invalid H0 ledger rows")
    staging = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    sidecar_root = staging / "sidecars"
    sidecar_root.mkdir(parents=True)
    terminal_rows: list[dict[str, Any]] = []
    try:
        for row in rows:
            if row.get("admission") == "BLOCKED_SOURCE":
                terminal_rows.append(_blocked_row(row, stage))
                continue
            session_id = str(row["session_id"])
            try:
                hdf5_path, hdf5_sha, controller_path = _source_contract(row)
                relative = Path(str(row["task"])) / f"{session_id}.npz"
                artifact_path = sidecar_root / relative
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                if stage == "h1":
                    if not row.get("eligibility", {}).get("controller_wrist", {}).get("eligible"):
                        terminal_rows.append({
                            **_blocked_row(row, stage),
                            "status": "BLOCKED_PREREQ_CONTROLLER_WRIST",
                            "primary_blocker": "CONTROLLER_TO_WRIST_NOT_PROVEN",
                        })
                        continue
                    if controller_path is None:
                        raise SidecarContractError("Controller sidecar path absent")
                    metrics = materialize_h1(hdf5_path, controller_path, artifact_path)
                    downstream = {"hand_observation_eligible": True, "robot_geometry_eligible": True}
                else:
                    metrics = materialize_h2(hdf5_path, artifact_path)
                    downstream = {"tactile_event_timing_eligible": True}
                terminal_rows.append({
                    "session_id": session_id, "dataset_id": row["dataset_id"], "task": row["task"],
                    "frame_count": row.get("frame_count"), "status": "PASSED", "primary_blocker": None,
                    "artifact": {
                        "path": str((output_root / "sidecars" / relative).resolve()),
                        "bytes": artifact_path.stat().st_size,
                        "sha256": _sha256(artifact_path),
                    }, "source_hdf5": {
                        "path": str(hdf5_path.resolve()), "sha256": hdf5_sha,
                    },
                    "metrics": metrics, "downstream": downstream,
                })
            except SidecarContractError as exc:
                terminal_rows.append({
                    "session_id": session_id, "dataset_id": row["dataset_id"], "task": row["task"],
                    "frame_count": row.get("frame_count"), "status": "FAILED_QUALITY_C",
                    "primary_blocker": str(exc), "artifact": None,
                    "downstream": (
                        {"hand_observation_eligible": False, "robot_geometry_eligible": False}
                        if stage == "h1" else {"tactile_event_timing_eligible": False}
                    ),
                })
            except Exception as exc:  # preserve a finite row terminal instead of aborting the cohort
                terminal_rows.append({
                    "session_id": session_id, "dataset_id": row["dataset_id"], "task": row["task"],
                    "frame_count": row.get("frame_count"), "status": "FAILED_RUNTIME_FINAL",
                    "primary_blocker": f"{type(exc).__name__}: {exc}", "artifact": None,
                    "downstream": (
                        {"hand_observation_eligible": False, "robot_geometry_eligible": False}
                        if stage == "h1" else {"tactile_event_timing_eligible": False}
                    ),
                })
        states = Counter(item["status"] for item in terminal_rows)
        ledger_name = "CANONICAL_HAND_LEDGER.json" if stage == "h1" else "TACTILE_SIDECAR_LEDGER.json"
        schema_version = "handle-canonical-hand-ledger-v7.1" if stage == "h1" else "handle-tactile-sidecar-ledger-v7.1"
        claim_limit = (
            "HAND21_FROM_MANUS_CONTROLLER is a Controller-wrist-anchored MANUS adapter; not HaWoR, "
            "MANO, anatomical truth, external depth truth, or Robot control truth."
            if stage == "h1" else
            "Raw five-finger taxels support synchronized contact-event timing analysis only; values are "
            "not calibrated force, object identity, palm tactile, or physical contact ground truth."
        )
        ledger = {
            "schema_version": schema_version,
            "artifact_revision": "R7_0",
            "validity": "VALID_FOR_PINNED_REVISION",
            "generated_at": _utc_now(),
            "stage": stage,
            "source_h0": _evidence(h0_path),
            "row_count": len(terminal_rows),
            "summary": {"by_status": dict(sorted(states.items())), "total_frames": sum(int(r.get("frame_count") or 0) for r in terminal_rows)},
            "rows": terminal_rows,
            "claim_limit": claim_limit,
        }
        if stage == "h1":
            ledger["observation_contract"] = {
                "name": "HAND21_FROM_MANUS_CONTROLLER",
                "side_order": list(SIDES), "joint_names": list(HAND21_NAMES),
                "manus25_source_names": list(MANUS25_NAMES),
                "hand21_from_manus25": list(HAND21_FROM_MANUS25),
                "omitted_source_nodes": ["Hand_Invalid", "Index_IP", "Middle_IP", "Ring_IP", "Pinky_IP"],
                "wrist_source": "same-session Controller plus recorded controller_to_wrist_calibration",
                "coordinate_views": ["wrist_local_m", "world_m", "camera_m"],
            }
        else:
            ledger["observation_contract"] = {
                "side_order": list(SIDES), "finger_order": list(FINGERS),
                "grid_shape": [5, 4, 8], "value_dtype": "int16_raw",
                "validity": "side_valid AND finger_active_mask", "timestamp_clock": "PICO timeStampNs",
                "palm_present": False, "event_threshold": "ABSENT_REQUIRES_CALIBRATION",
            }
        ledger_path = staging / ledger_name
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
        csv_path = staging / ledger_name.replace(".json", ".csv")
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "session_id", "dataset_id", "task", "frame_count", "status", "primary_blocker", "artifact_path"
            ])
            writer.writeheader()
            for item in terminal_rows:
                writer.writerow({
                    "session_id": item["session_id"], "dataset_id": item["dataset_id"], "task": item["task"],
                    "frame_count": item.get("frame_count"), "status": item["status"],
                    "primary_blocker": item.get("primary_blocker"),
                    "artifact_path": (item.get("artifact") or {}).get("path"),
                })
        result = {
            "schema_version": "handle-h1-h2-result-v7.1", "stage": stage, "status": "PASSED",
            "artifact_revision": "R7_0", "validity": "VALID_FOR_PINNED_REVISION",
            "generated_at": _utc_now(), "row_count": len(terminal_rows),
            "terminal_count": len(terminal_rows), "summary": ledger["summary"],
            "claim_limit": claim_limit,
        }
        # RESULT references are finalized after the directory is atomically published.
        (staging / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        os.rename(staging, output_root)
        # Add immutable output evidence without changing the already-published ledger.
        return ledger
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    ledger = build_stage(args.h0.resolve(), args.output.resolve(), args.stage)
    print(json.dumps({"stage": args.stage, "row_count": ledger["row_count"], "summary": ledger["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
