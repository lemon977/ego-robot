#!/usr/bin/env python3
"""Run CPU-only observed-surface Object6D for the frozen Robot15h W0.

The runner consumes only consumer-admitted SAM3.1 task-object masks and the
FoundationStereo optical-Z archives in the encoded physical-left domain.  It
publishes directly observed, finite surface support for the two playing-card
sessions.  A missing task-object mask is an upstream blocker, not an empty
Object6D result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import cv2
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import WINDOW_RUN_ID, build_packet
from chaoyang.pipeline.object6d_planar_observability_v2 import (
    DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    PlanarFrameInput,
    estimate_planar_frame,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_geometry_object6d_wave0_v1"
PHASE = "ROBOT15H_OBJECT6D_W0_VISIBLE_PATCH"
AUTHORITY = "DEVELOPMENT_VISIBLE_SURFACE_ONLY_NON_CONTROL_NON_DEPLOYABLE"
AUTHORIZED_SCOPE = "VISUAL_OBJECT6D_CANDIDATE_INPUT"
IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
DEPTH_SHAPE = (480, 640)
MASK_SHAPE = (960, 1280)
INSTANCE_IDS = ("playing_card_00", "playing_card_01", "playing_card_02")
INVENTORY = ROOT / (
    "_run/current/0915_robot15h_window_start_inventory_v1/attempts/"
    "attempt_0001/BATCH_MANIFEST.json"
)
DEPTH_ROOT = ROOT / (
    "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/"
    "attempts/attempt_0001"
)
MASK_ROOT = ROOT / (
    "_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/"
    "attempts/attempt_0001"
)
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_OBJECT6D_V1"
RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_GEOMETRY_OBJECT6D_WAVE0_V1_RESULT.json"
PIXEL_MAP = {
    "type": "ANALYTIC_PIXEL_CENTER_2X",
    "formula_x": "x_mask = 2*x_depth + 0.5",
    "formula_y": "y_mask = 2*y_depth + 0.5",
    "source": "ORIGINAL_PHYSICAL_LEFT_640x480",
    "target": IMAGE_DOMAIN + "_1280x960",
    "lens_undistortion_applied": False,
    "lens_remap_applied": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": sha256(candidate),
    }


def projected_ref(existing: Path, destination: Path) -> dict[str, Any]:
    """Describe bytes in staging at their post-rename immutable location."""

    existing = existing.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "bytes": existing.stat().st_size,
        "sha256": sha256(existing),
    }


def verify_ref(value: Mapping[str, Any]) -> Path:
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise RuntimeError(f"artifact reference drift: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def w0_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in inventory.get("sessions", []) if row.get("wave") == "W0"]
    if inventory.get("cohort_denominator") != 220 or len(rows) != 4:
        raise RuntimeError("frozen 0915 W0 must contain exactly four of 220 sessions")
    identities = [str(row.get("session_id")) for row in rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate W0 session identity")
    return rows


def rows_by_session(batch: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    rows = batch.get("sessions")
    if not isinstance(rows, list):
        raise RuntimeError("upstream batch lacks session rows")
    result = {str(row.get("session_id")): row for row in rows if isinstance(row, Mapping)}
    if set(result) != set(expected):
        raise RuntimeError("upstream batch session identity differs from frozen W0")
    return result


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("task packet differs from frozen weights-ABSENT spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError(f"{TASK_ID} is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if (
        route is None
        or route.get("execution_allowed") is not True
        or route.get("packet_sha256") != sha256(packet_path)
    ):
        raise RuntimeError(f"{TASK_ID} is not SHA-bound routable")
    return packet, packet_path


def validate_writer_claim(
    path: Path, *, signature_sha: str, executor_epoch: int, fencing_sha: str,
) -> dict[str, Any]:
    claim = load_json(path)
    pid = claim.get("pid")
    ticks = claim.get("proc_start_ticks")
    if (
        claim.get("schema_version") != "0915-robot15h-object6d-writer-claim-v1"
        or claim.get("task_id") != TASK_ID
        or claim.get("status") != "CLAIMED"
        or claim.get("weights") != "ABSENT"
        or claim.get("gpu_used") is not False
        or claim.get("unique_write_root") != str(OUTPUT.resolve())
        or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("fencing_token_sha256") != fencing_sha
        or not isinstance(pid, int)
        or not isinstance(ticks, int)
        or process_start_ticks(pid) != ticks
    ):
        raise RuntimeError("Object6D writer claim/fence mismatch")
    return claim


def heartbeat() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chaoyang.governance.heartbeat_task",
            "--task-id",
            TASK_ID,
            "--pid",
            str(os.getpid()),
            "--status",
            "RUNNING",
            "--phase",
            PHASE,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


class PackedMask:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.packed = np.asarray(archive["packed"], np.uint8)
            self.frame_count = int(archive["frame_count"])
            self.height = int(archive["height"])
            self.width = int(archive["width"])
            self.bitorder = str(archive["bitorder"])
        expected = (self.frame_count, (self.height * self.width + 7) // 8)
        if self.packed.shape != expected or self.bitorder != "big":
            raise RuntimeError(f"packed semantic mask contract drift: {path}")

    def unpack(self, frame_index: int) -> np.ndarray:
        return np.unpackbits(
            self.packed[frame_index], bitorder="big", count=self.height * self.width,
        ).reshape(self.height, self.width).astype(bool)


def depth_to_mask_map() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(DEPTH_SHAPE, dtype=np.float64)
    mapping = np.stack((2.0 * xx + 0.5, 2.0 * yy + 0.5), axis=-1)
    valid = (
        (mapping[..., 0] >= 0.0)
        & (mapping[..., 0] < MASK_SHAPE[1])
        & (mapping[..., 1] >= 0.0)
        & (mapping[..., 1] < MASK_SHAPE[0])
    )
    return mapping, valid


def validate_depth_session(
    batch_row: Mapping[str, Any], inventory_row: Mapping[str, Any],
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    if batch_row.get("status") != "PASSED" or not isinstance(batch_row.get("result"), Mapping):
        return None, None, "DEPTH_NOT_CONSUMER_ADMITTED"
    result_path = verify_ref(batch_row["result"])
    result = load_json(result_path)
    frame_count = int(inventory_row["frame_count"])
    if (
        result.get("status") != "PASSED"
        or result.get("session_id") != inventory_row["session_id"]
        or result.get("frame_count") != frame_count
        or result.get("consumption_authorized") is not True
        or result.get("authorized_scopes") != [AUTHORIZED_SCOPE]
        or result.get("external_accuracy") != "UNVERIFIED"
        or result.get("strict_metric_contact_authorized") is not False
        or result.get("source_mutated") is not False
    ):
        return None, None, "DEPTH_NOT_CONSUMER_ADMITTED"
    root = result_path.parent
    adapter = load_json(root / "ADAPTER_CONTRACT.json")
    contract = load_json(root / "DEPTH_CONTRACT.json")
    alignment = load_json(root / "RGB_ALIGNMENT_QA.json")
    summary = load_json(root / "DEPTH_SUMMARY.json")
    if (
        adapter.get("lens_remap_applied") is not False
        or adapter.get("lens_undistortion_applied") is not False
        or adapter.get("output_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or adapter.get("output_spatial_unflip") is not True
        or adapter.get("camera_swap") is not False
        or adapter.get("physical_source_indices") != {"left": 1, "right": 0}
        or contract.get("depth_reference") != DEPTH_REFERENCE
        or contract.get("frame_geometry") != [640, 480]
        or contract.get("frame_count") != frame_count
        or contract.get("consumption_authorized") is not True
        or contract.get("occluded_or_hidden_geometry") != "INVALID_NOT_COMPLETED"
        or alignment.get("depth_grid_domain") != "ORIGINAL_PHYSICAL_LEFT_640x480"
        or alignment.get("maximum_absolute_channel_error") != 0
        or alignment.get("mismatched_pixels") != 0
        or summary.get("frame_count") != frame_count
        or summary.get("consumption_authorized") is not True
    ):
        return None, None, "DEPTH_ENCODED_DOMAIN_QA_FAILED"
    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        return None, None, "DEPTH_FRAME_MANIFEST_INCOMPLETE"
    references = {
        "result": ref(result_path),
        "adapter_contract": ref(root / "ADAPTER_CONTRACT.json"),
        "depth_contract": ref(root / "DEPTH_CONTRACT.json"),
        "rgb_alignment_qa": ref(root / "RGB_ALIGNMENT_QA.json"),
        "depth_summary": ref(root / "DEPTH_SUMMARY.json"),
    }
    return root, {"frames": frames, "references": references}, None


def validate_mask_session(
    batch: Mapping[str, Any], batch_row: Mapping[str, Any], inventory_row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    if (
        batch.get("object_mask_consumer_allowed") is not True
        or batch_row.get("object_mask_consumer_allowed") is not True
        or batch_row.get("status") != "PASSED_VISIBLE_OBJECT_MASK_PROXY"
        or batch_row.get("task") != "playing_cards"
    ):
        return None, "NO_INDEPENDENT_CONSUMER_ADMITTED_TASK_OBJECT_MASK"
    if batch_row.get("frame_count") != inventory_row.get("frame_count"):
        return None, "OBJECT_MASK_FRAME_AXIS_DRIFT"
    result_path = verify_ref(batch_row["result"])
    result = load_json(result_path)
    manifest_path = verify_ref(batch_row["instance_manifest"])
    manifest = load_json(manifest_path)
    if (
        result.get("object_mask_consumer_allowed") is not True
        or manifest.get("image_domain") != IMAGE_DOMAIN
        or manifest.get("manual_coordinates_or_boxes") is not False
        or manifest.get("union_mask_created") is not False
        or manifest.get("support_tray_or_bowl_is_task_object") is not False
    ):
        return None, "OBJECT_MASK_CONTRACT_DRIFT"
    by_id = {str(row.get("instance_id")): row for row in manifest.get("instances", [])}
    if set(by_id) != set(INSTANCE_IDS):
        return None, "CARD_INSTANCE_IDENTITY_SET_DRIFT"
    masks: dict[str, PackedMask] = {}
    states: dict[str, list[dict[str, Any]]] = {}
    references: dict[str, Any] = {
        "session_result": ref(result_path),
        "instance_manifest": ref(manifest_path),
    }
    for instance_id in INSTANCE_IDS:
        row = by_id[instance_id]
        if (
            row.get("object_mask_consumer_allowed") is not True
            or row.get("visible_surface_only") is not True
            or row.get("hidden_shape_or_extent_inferred") is not False
        ):
            return None, f"OBJECT_MASK_INSTANCE_NOT_ADMITTED:{instance_id}"
        mask_path = verify_ref(row["semantic_archive"])
        state_path = verify_ref(row["state_ledger"])
        mask = PackedMask(mask_path)
        ledger = load_json(state_path)
        frame_rows = ledger.get("frames")
        if (
            mask.frame_count != int(inventory_row["frame_count"])
            or (mask.height, mask.width) != MASK_SHAPE
            or not isinstance(frame_rows, list)
            or len(frame_rows) != mask.frame_count
            or [item.get("frame_id") for item in frame_rows] != list(range(mask.frame_count))
        ):
            return None, f"OBJECT_MASK_FRAME_AXIS_DRIFT:{instance_id}"
        masks[instance_id] = mask
        states[instance_id] = frame_rows
        references[f"semantic_mask_{instance_id}"] = ref(mask_path)
        references[f"temporal_state_{instance_id}"] = ref(state_path)
    return {
        "masks": masks,
        "states": states,
        "references": references,
        "input_video": verify_ref(batch_row["input_video"]),
    }, None


def load_depth_frame(
    value: Mapping[str, Any], expected_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if value.get("frame") != expected_index or not isinstance(value.get("artifact"), Mapping):
        raise RuntimeError(f"depth frame manifest discontinuity at {expected_index}")
    path = verify_ref(value["artifact"])
    with np.load(path, allow_pickle=False) as archive:
        frame_id = int(archive["frame_id"])
        depth = np.asarray(archive["depth_m"], np.float64)
        valid = np.asarray(archive["valid"], bool)
        intrinsics = np.asarray(archive["physical_left_intrinsics"], np.float64)
        reference = str(archive["depth_reference"])
    if (
        frame_id != expected_index
        or depth.shape != DEPTH_SHAPE
        or valid.shape != DEPTH_SHAPE
        or intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or reference != DEPTH_REFERENCE
    ):
        raise RuntimeError(f"encoded physical-left depth contract drift: {path}")
    return depth, valid, intrinsics


def registered_support(
    mask: np.ndarray, depth: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray,
    mapping: np.ndarray, registration_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_map = np.isfinite(mapping).all(axis=2)
    base = valid & registration_valid & finite_map & np.isfinite(depth) & (depth > 0.0)
    source_y, source_x = np.nonzero(base)
    support = np.zeros(depth.shape, bool)
    if not len(source_x):
        return support, np.empty((0, 3), np.float64)
    mapped = np.rint(mapping[source_y, source_x]).astype(np.int64)
    in_bounds = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < mask.shape[1])
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < mask.shape[0])
    )
    selected = np.zeros(len(source_x), bool)
    selected[in_bounds] = mask[mapped[in_bounds, 1], mapped[in_bounds, 0]]
    xx = source_x[selected]
    yy = source_y[selected]
    support[yy, xx] = True
    z = depth[yy, xx]
    x = (xx.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return support, np.column_stack((x, y, z))


def finite_patch_record(
    support: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, archive_row: int,
) -> dict[str, Any]:
    yy, xx = np.nonzero(support)
    if len(xx) < 24:
        return {
            "observability": "UNOBSERVABLE",
            "estimate": None,
            "residual": None,
            "reason": "INSUFFICIENT_DIRECT_VISIBLE_DEPTH_SUPPORT",
            "semantics": "EXACT_DIRECT_VISIBLE_FINITE_SUPPORT_NOT_FULL_OBJECT_EXTENT",
        }
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, np.int32)
    largest_fraction = float(areas.max() / len(xx)) if len(areas) else 0.0
    contours, _hierarchy = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else np.empty((0, 2), np.int32)
    if len(boundary) > 64:
        indices = np.linspace(0, len(boundary) - 1, 64).round().astype(np.int64)
        boundary = boundary[indices]
    boundary_xyz: list[list[float]] = []
    for px, py in boundary:
        z = float(depth[int(py), int(px)])
        boundary_xyz.append([
            (float(px) - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (float(py) - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ])
    return {
        "observability": "OBSERVABLE_DIRECT_VISIBLE_FINITE_SUPPORT",
        "estimate": {
            "archive_row": int(archive_row),
            "support_pixel_count": int(len(xx)),
            "depth_pixel_bbox_xyxy_inclusive": [
                int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()),
            ],
            "boundary_depth_uv": boundary.astype(int).tolist(),
            "boundary_xyz_m": boundary_xyz,
        },
        "residual": {
            "connected_component_count": int(max(count - 1, 0)),
            "largest_component_fraction": largest_fraction,
        },
        "reason": None,
        "semantics": "EXACT_DIRECT_VISIBLE_FINITE_SUPPORT_NOT_FULL_OBJECT_EXTENT",
    }


def temporally_unify_axes(frames: list[dict[str, Any]]) -> None:
    previous_normal: np.ndarray | None = None
    previous_axis: np.ndarray | None = None
    for row in frames:
        normal = row["plane_normal"]
        if normal["observability"] != "UNOBSERVABLE":
            vector = np.asarray(normal["estimate"]["unit_xyz"], np.float64)
            flipped = previous_normal is not None and float(np.dot(previous_normal, vector)) < 0.0
            if flipped:
                vector *= -1.0
            normal["estimate"]["unit_xyz"] = vector.tolist()
            normal["estimate"]["sign_convention"] = (
                "TEMPORALLY_CONTINUOUS_SEEDED_Z_NONPOSITIVE"
            )
            normal["estimate"]["temporal_sign_flipped"] = bool(flipped)
            previous_normal = vector
        rotation = row["inplane_rotation"]
        if rotation["observability"] != "UNOBSERVABLE":
            axis = np.asarray(rotation["estimate"]["axis_unit_xyz"], np.float64)
            flipped = previous_axis is not None and float(np.dot(previous_axis, axis)) < 0.0
            if flipped:
                axis *= -1.0
            rotation["estimate"]["axis_unit_xyz"] = axis.tolist()
            rotation["estimate"]["axis_sign_convention"] = (
                "TEMPORALLY_CONTINUOUS_PI_EQUIVALENT"
            )
            rotation["estimate"]["temporal_axis_flipped"] = bool(flipped)
            previous_axis = axis


def summarize_objects(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = ("center_xyz", "plane_normal", "inplane_rotation", "finite_visible_patch")
    for obj in objects:
        frames = obj["frames"]
        result[str(obj["instance_id"])] = {
            "frame_count": len(frames),
            "observable": {
                field: sum(row[field]["observability"] != "UNOBSERVABLE" for row in frames)
                for field in fields
            },
            "mask_state_counts": dict(Counter(str(row["mask_state"]) for row in frames)),
            "normal_temporal_sign_flips": sum(
                bool(row["plane_normal"].get("estimate", {}).get("temporal_sign_flipped"))
                for row in frames
                if isinstance(row["plane_normal"].get("estimate"), Mapping)
            ),
            "inplane_temporal_axis_flips": sum(
                bool(row["inplane_rotation"].get("estimate", {}).get("temporal_axis_flipped"))
                for row in frames
                if isinstance(row["inplane_rotation"].get("estimate"), Mapping)
            ),
        }
    return result


def decode_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = 0
    width = height = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        count += 1
    capture.release()
    return {
        "frame_count": count,
        "expected_frame_count": expected_frames,
        "full_decode": count == expected_frames,
        "width": width,
        "height": height,
    }


def _project(point: Sequence[float], intrinsics: np.ndarray) -> tuple[int, int] | None:
    x, y, z = (float(v) for v in point)
    if not math.isfinite(z) or z <= 0.0:
        return None
    return (
        int(round(2.0 * (intrinsics[0, 0] * x / z + intrinsics[0, 2]) + 0.5)),
        int(round(2.0 * (intrinsics[1, 1] * y / z + intrinsics[1, 2]) + 0.5)),
    )


def render_review(
    source: Path,
    destination: Path,
    masks: Mapping[str, PackedMask],
    objects: Sequence[Mapping[str, Any]],
    intrinsics_by_frame: Sequence[np.ndarray],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (height, width) != MASK_SHAPE:
        capture.release()
        raise RuntimeError(f"physical-left review geometry drift: {(width, height)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("failed to open Object6D review writer")
    colors = [(70, 220, 70), (60, 170, 255), (230, 100, 230)]
    object_by_id = {str(obj["instance_id"]): obj for obj in objects}
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        overlay = image.copy()
        for color, instance_id in zip(colors, INSTANCE_IDS):
            mask = masks[instance_id].unpack(frame_index)
            overlay[mask] = (
                0.72 * overlay[mask].astype(np.float32) + 0.28 * np.asarray(color)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            row = object_by_id[instance_id]["frames"][frame_index]
            center_record = row["center_xyz"]
            if center_record["observability"] != "UNOBSERVABLE":
                center = np.asarray(center_record["estimate"]["xyz_m"], np.float64)
                uv = _project(center, intrinsics_by_frame[frame_index])
                if uv is not None:
                    cv2.circle(overlay, uv, 5, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        overlay, instance_id[-2:], (uv[0] + 7, uv[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA,
                    )
                    for field, axis_color in (
                        ("plane_normal", (20, 20, 240)),
                        ("inplane_rotation", (240, 220, 40)),
                    ):
                        record = row[field]
                        if record["observability"] == "UNOBSERVABLE":
                            continue
                        key = "unit_xyz" if field == "plane_normal" else "axis_unit_xyz"
                        axis = np.asarray(record["estimate"][key], np.float64)
                        end = _project(center + 0.04 * axis, intrinsics_by_frame[frame_index])
                        if end is not None:
                            cv2.arrowedLine(overlay, uv, end, axis_color, 2, cv2.LINE_AA, 0, 0.2)
        cv2.putText(
            overlay, f"frame {frame_index:04d} | observed finite patches only",
            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, "physical-left encoded resize-only | no undistortion | no hidden extent",
            (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(overlay)
        frame_index += 1
    capture.release()
    writer.release()
    decoded = decode_video(destination, len(intrinsics_by_frame))
    if not decoded["full_decode"]:
        raise RuntimeError(f"Object6D review failed full decode: {destination}")
    return {**decoded, "fps": fps}


def run_card_session(
    inventory_row: Mapping[str, Any], depth: Mapping[str, Any], mask: Mapping[str, Any],
    output: Path, visual: Path,
) -> dict[str, Any]:
    session_id = str(inventory_row["session_id"])
    task = str(inventory_row["task"])
    frame_count = int(inventory_row["frame_count"])
    final = output / "sessions" / task / session_id
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"fresh session output required: {final}")
    stage = output / f".session-staging-{session_id}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    mapping, mapping_valid = depth_to_mask_map()
    objects: list[dict[str, Any]] = []
    packed_support: list[np.ndarray] = []
    intrinsics_by_frame: list[np.ndarray] = []
    depth_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame_index, frame_row in enumerate(depth["frames"]):
        depth_cache.append(load_depth_frame(frame_row, frame_index))
        intrinsics_by_frame.append(depth_cache[-1][2])
    for instance_index, instance_id in enumerate(INSTANCE_IDS):
        frames: list[dict[str, Any]] = []
        supports: list[np.ndarray] = []
        for frame_index, (depth_m, valid, intrinsics) in enumerate(depth_cache):
            state = mask["states"][instance_id][frame_index]
            admitted = state.get("semantic_admitted") is True
            tracking_state = str(state.get("tracking_state"))
            mask_state = tracking_state if admitted and tracking_state in {
                "seeded", "tracked", "reseeded",
            } else "unknown"
            semantic_mask = mask["masks"][instance_id].unpack(frame_index)
            support, _points = registered_support(
                semantic_mask, depth_m, valid, intrinsics, mapping, mapping_valid,
            )
            if mask_state == "unknown":
                support[:] = False
            planar = estimate_planar_frame(PlanarFrameInput(
                frame_index=frame_index,
                mask=semantic_mask,
                mask_state=mask_state,
                visibility_state=DIRECT_VISIBILITY if mask_state != "unknown" else "UNKNOWN",
                depth_m=depth_m,
                depth_valid=valid,
                depth_intrinsics=intrinsics,
                depth_to_mask_xy=mapping,
                registration_valid=mapping_valid,
                depth_reference=DEPTH_REFERENCE,
                registration_authority=MASK_REGISTRATION_AUTHORITY,
            ))
            frames.append({
                **planar,
                "finite_visible_patch": finite_patch_record(
                    support, depth_m, intrinsics,
                    archive_row=instance_index * frame_count + frame_index,
                ),
                "full_extent": {
                    "observability": "UNOBSERVABLE",
                    "estimate": None,
                    "residual": None,
                    "reason": "FULL_OBJECT_BOUNDARY_AND_THICKNESS_UNPROVEN",
                    "semantics": "HIDDEN_EXTENT_NOT_COMPLETED",
                },
            })
            supports.append(np.packbits(support.reshape(-1), bitorder="big"))
        temporally_unify_axes(frames)
        packed_support.append(np.stack(supports))
        objects.append({
            "instance_id": instance_id,
            "entity_role": "PHYSICAL_TASK_OBJECT",
            "geometry_class": "PLAYING_CARD_DIRECT_VISIBLE_FINITE_PATCH",
            "frames": frames,
        })
    support_stage = stage / "VISIBLE_PATCH_SUPPORT.npz"
    atomic_npz(
        support_stage,
        packed=np.concatenate(packed_support, axis=0),
        instance_ids=np.asarray(INSTANCE_IDS),
        frame_count=np.asarray(frame_count, np.int32),
        height=np.asarray(DEPTH_SHAPE[0], np.int32),
        width=np.asarray(DEPTH_SHAPE[1], np.int32),
        bitorder=np.asarray("big"),
        row_layout=np.asarray("INSTANCE_MAJOR_FLAT_ROW"),
        semantics=np.asarray("EXACT_DIRECT_VISIBLE_DEPTH_SUPPORT_ONLY"),
    )
    final_support = final / support_stage.name
    summary = summarize_objects(objects)
    inputs = {
        "depth": depth["references"],
        "task_object_mask": mask["references"],
        "depth_to_mask_mapping": PIXEL_MAP,
    }
    document = {
        "schema_version": "0915-robot15h-object6d-visible-patch-session-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": "COMPLETED_DEVELOPMENT_VISIBLE_SURFACE",
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "center_xyz_semantics": "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_FIXED_CENTER",
        "normal_sign_policy": "TEMPORALLY_CONTINUOUS_SEEDED_Z_NONPOSITIVE",
        "inplane_axis_semantics": "PI_PERIODIC_VISIBLE_SUPPORT_AXIS",
        "finite_patch_semantics": "EXACT_PACKED_DIRECT_VISIBLE_DEPTH_SUPPORT_NOT_FULL_EXTENT",
        "visible_patch_support_archive": projected_ref(support_stage, final_support),
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "object_thickness_inferred": False,
        "full_object_extent_inferred": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
        "inputs": inputs,
        "objects": objects,
        "support_entities": [{
            "instance_id": "black_card_tray",
            "observability": "UNKNOWN",
            "geometry_authority": "NONE",
            "reason": "INDEPENDENT_SUPPORT_MASK_NOT_AVAILABLE",
        }],
        "summary": summary,
        "claim_limit": (
            "Directly visible finite physical-left optical-Z support only. Visible "
            "centroids are not object-fixed centres; hidden extent, thickness, Contact, "
            "control, training, deployment and external metric accuracy are not claimed."
        ),
    }
    atomic_json(stage / "OBJECT6D_VISIBLE_PATCH.json", document)
    review_stage = stage / "OBJECT6D_REVIEW.mp4"
    review_decode = render_review(
        mask["input_video"], review_stage, mask["masks"], objects, intrinsics_by_frame,
    )
    observable_patch_counts = {
        instance_id: summary[instance_id]["observable"]["finite_visible_patch"]
        for instance_id in INSTANCE_IDS
    }
    passed = all(count > 0 for count in observable_patch_counts.values())
    status = "PASSED_DEVELOPMENT_VISIBLE_SURFACE" if passed else "REJECTED_QUALITY"
    result = {
        "schema_version": "0915-robot15h-object6d-visible-patch-session-result-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "task": task,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": status,
        "object6d_consumer_allowed": passed,
        "authority": AUTHORITY,
        "weights": "ABSENT",
        "gpu_used": False,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "lens_undistortion_applied": False,
        "lens_remap_applied": False,
        "center_xyz_is_object_fixed_center": False,
        "hidden_geometry_inferred": False,
        "object_thickness_inferred": False,
        "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "observable_patch_counts": observable_patch_counts,
        "object6d": projected_ref(
            stage / "OBJECT6D_VISIBLE_PATCH.json", final / "OBJECT6D_VISIBLE_PATCH.json",
        ),
        "visible_patch_support": projected_ref(support_stage, final_support),
        "review": {
            **review_decode,
            "video": projected_ref(review_stage, final / "OBJECT6D_REVIEW.mp4"),
        },
        "source_mutated": False,
    }
    atomic_json(stage / "RESULT.json", result)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    published = load_json(final / "RESULT.json")
    verify_ref(published["object6d"])
    verify_ref(published["visible_patch_support"])
    verify_ref(published["review"]["video"])
    shallow = visual / f"{session_id}_OBJECT6D_REVIEW.mp4"
    try:
        os.link(final / "OBJECT6D_REVIEW.mp4", shallow)
    except OSError:
        shutil.copy2(final / "OBJECT6D_REVIEW.mp4", shallow)
    shallow_reference = ref(shallow)
    if shallow_reference["sha256"] != published["review"]["video"]["sha256"]:
        raise RuntimeError("shallow Object6D review differs from atomic session review")
    return {
        "session_id": session_id,
        "task": task,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": status,
        "first_blocker": None if passed else "NO_INSTANCE_WITH_OBSERVABLE_FINITE_PATCH",
        "depth_consumer_admitted": True,
        "object_mask_consumer_admitted": True,
        "object6d_attempted": True,
        "object6d_artifact_emitted": True,
        "object6d_consumer_allowed": passed,
        "review_video_emitted": True,
        "observable_patch_counts": observable_patch_counts,
        "result": ref(final / "RESULT.json"),
        "object6d": ref(final / "OBJECT6D_VISIBLE_PATCH.json"),
        "review": {**review_decode, "video": shallow_reference},
    }


def blocked_row(
    inventory_row: Mapping[str, Any], *, depth_ok: bool, mask_ok: bool, blocker: str,
) -> dict[str, Any]:
    return {
        "session_id": inventory_row["session_id"],
        "task": inventory_row["task"],
        "source_group": inventory_row["source_group"],
        "frame_count": int(inventory_row["frame_count"]),
        "status": (
            "BLOCKED_UPSTREAM_DEPTH" if not depth_ok else "BLOCKED_UPSTREAM_OBJECT_MASK"
        ),
        "first_blocker": blocker,
        "depth_consumer_admitted": depth_ok,
        "object_mask_consumer_admitted": mask_ok,
        "object6d_attempted": False,
        "object6d_artifact_emitted": False,
        "object6d_consumer_allowed": False,
        "review_video_emitted": False,
    }


def summarize_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "attempted": 0,
        "passed": 0,
        "quality_rejected": 0,
        "blocked_upstream": 0,
        "failed_runtime": 0,
        "unrun": 0,
    }
    for row in rows:
        status = str(row["status"])
        counts["attempted"] += int(row.get("object6d_attempted") is True)
        counts["passed"] += int(status.startswith("PASSED"))
        counts["quality_rejected"] += int(status == "REJECTED_QUALITY")
        counts["blocked_upstream"] += int(status.startswith("BLOCKED_UPSTREAM"))
        counts["failed_runtime"] += int(status == "FAILED_RUNTIME")
    counts["unrun"] = counts["total"] - counts["attempted"] - counts["blocked_upstream"]
    return counts


def validate_fixed_paths(output: Path, visual: Path, receipt: Path) -> None:
    if output.resolve() != OUTPUT.resolve():
        raise RuntimeError(f"output root must equal {OUTPUT}")
    if visual.resolve() != VISUAL.resolve():
        raise RuntimeError(f"visual root must equal {VISUAL}")
    if receipt.resolve() != RECEIPT.resolve():
        raise RuntimeError(f"receipt must equal {RECEIPT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    validate_fixed_paths(output, visual, receipt)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    packet, packet_path = validate_route()
    required_inputs = [
        INVENTORY, DEPTH_ROOT / "BATCH_RESULT.json", MASK_ROOT / "BATCH_RESULT.json",
    ]
    for path in required_inputs:
        if not path.is_file():
            raise RuntimeError(f"required upstream artifact missing: {path}")
    signature_payload = {
        "schema_version": "0915-robot15h-object6d-wave0-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "depth_to_mask_mapping": PIXEL_MAP,
        "task_packet": ref(packet_path),
        "inputs": [ref(path) for path in required_inputs],
        "code": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/pipeline/object6d_planar_observability_v2.py"),
            ref(ROOT / "src/chaoyang/pipeline/object6d_planar_observability_v1.py"),
        ],
        "config": {
            "instances": list(INSTANCE_IDS),
            "finite_patch_support": "EXACT_PACKED_DIRECT_VISIBLE_DEPTH_PIXELS",
            "hidden_geometry": "INVALID_NOT_COMPLETED",
            "normal_sign": "TEMPORALLY_CONTINUOUS_SEEDED_Z_NONPOSITIVE",
        },
    }
    signature = {
        **signature_payload,
        "run_signature_sha256": canonical_sha(signature_payload),
    }
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    fencing_sha = hashlib.sha256(args.fencing_token.encode()).hexdigest()
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-object6d-writer-claim-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "CLAIMED",
        "weights": "ABSENT",
        "gpu_used": False,
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": fencing_sha,
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output),
    })
    validate_writer_claim(
        output / "CLAIM.json", signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch, fencing_sha=fencing_sha,
    )
    heartbeat()

    inventory_rows = w0_rows(load_json(INVENTORY))
    identities = [str(row["session_id"]) for row in inventory_rows]
    depth_batch = load_json(DEPTH_ROOT / "BATCH_RESULT.json")
    mask_batch = load_json(MASK_ROOT / "BATCH_RESULT.json")
    depth_rows = rows_by_session(depth_batch, identities)
    mask_rows = rows_by_session(mask_batch, identities)
    rows: list[dict[str, Any]] = []
    for inventory_row in inventory_rows:
        session_id = str(inventory_row["session_id"])
        depth_root, depth_info, depth_blocker = validate_depth_session(
            depth_rows[session_id], inventory_row,
        )
        mask_info, mask_blocker = validate_mask_session(
            mask_batch, mask_rows[session_id], inventory_row,
        )
        if depth_info is None:
            rows.append(blocked_row(
                inventory_row, depth_ok=False, mask_ok=mask_info is not None,
                blocker=str(depth_blocker),
            ))
        elif mask_info is None:
            rows.append(blocked_row(
                inventory_row, depth_ok=True, mask_ok=False, blocker=str(mask_blocker),
            ))
        else:
            assert depth_root is not None
            try:
                rows.append(run_card_session(
                    inventory_row, depth_info, mask_info, output, visual,
                ))
            except Exception as error:
                rows.append({
                    "session_id": inventory_row["session_id"],
                    "task": inventory_row["task"],
                    "source_group": inventory_row["source_group"],
                    "frame_count": int(inventory_row["frame_count"]),
                    "status": "FAILED_RUNTIME",
                    "first_blocker": f"{type(error).__name__}:{error}",
                    "depth_consumer_admitted": True,
                    "object_mask_consumer_admitted": True,
                    "object6d_attempted": True,
                    "object6d_artifact_emitted": False,
                    "object6d_consumer_allowed": False,
                    "review_video_emitted": False,
                })
        heartbeat()
    counts = summarize_batch(rows)
    quality_status = (
        "PASSED_PARTIAL_W0_DEVELOPMENT_VISIBLE_SURFACE"
        if counts["passed"] > 0 and counts["failed_runtime"] == 0
        else "REJECTED_NO_OBJECT6D_SESSION"
    )
    ledger = {
        "schema_version": "0915-robot15h-object6d-wave0-ledger-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "geometry_quality_status": quality_status,
        "weights": "ABSENT",
        "gpu_used": False,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "rows": rows,
        "counts": counts,
        "all_unknown_artifact_emitted": False,
        "hidden_geometry_inferred": False,
        "strict_metric_contact_authorized": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "source_mutated": False,
    }
    ledger_path = output / "OBJECT6D_ELIGIBILITY_LEDGER.json"
    atomic_json(ledger_path, ledger)
    batch = {
        "schema_version": "0915-robot15h-object6d-wave0-batch-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "geometry_quality_status": quality_status,
        "counts": counts,
        "sessions": rows,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "lens_undistortion_applied": False,
        "lens_remap_applied": False,
        "all_unknown_artifact_emitted": False,
        "hidden_geometry_inferred": False,
        "strict_metric_contact_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    atomic_json(output / "METRICS.json", {
        "schema_version": "0915-robot15h-object6d-wave0-metrics-v1",
        "task_id": TASK_ID,
        "session_count": len(rows),
        "counts": counts,
        "object_instance_count": sum(
            len(row.get("observable_patch_counts", {})) for row in rows
        ),
        "observable_finite_patch_frames": sum(
            sum(row.get("observable_patch_counts", {}).values()) for row in rows
        ),
        "r0_affected": False,
        "r2_independent_path_affected": False,
    })
    atomic_json(visual / "INDEX.json", {
        "schema_version": "0915-robot15h-object6d-wave0-visual-index-v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "videos": [row["review"]["video"] for row in rows if row.get("review_video_emitted")],
    })
    top_status = "PASSED" if counts["failed_runtime"] == 0 else "FAILED_RUNTIME_FINAL"
    result = {
        "schema_version": "0915-robot15h-object6d-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "geometry_quality_status": quality_status,
        "counts": counts,
        "weights": "ABSENT",
        "gpu_used": False,
        "authority": AUTHORITY,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "lens_undistortion_applied": False,
        "lens_remap_applied": False,
        "hidden_geometry_inferred": False,
        "strict_metric_contact_authorized": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "ledger": ref(ledger_path),
        "batch_result": ref(output / "BATCH_RESULT.json"),
        "metrics": ref(output / "METRICS.json"),
        "visual_index": ref(visual / "INDEX.json"),
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "source_mutated": False,
        "claim_limit": packet["claim_limit"],
    }
    validate_writer_claim(
        output / "CLAIM.json", signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch, fencing_sha=fencing_sha,
    )
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-object6d-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({"status": top_status, "counts": counts, "result": str(output / "RESULT.json")}))
    return 0 if top_status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
