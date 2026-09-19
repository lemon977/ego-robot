#!/usr/bin/env python3
"""Build evidence-bounded W0 hand/object Interaction in the encoded image domain.

Only direct-observed HaWoR 2D fingertip projections admitted by the same-frame
SAM3.1 hand-role mask are associated with local FoundationStereo visible
surfaces. HaWoR absolute Z, inferred gaps, Removal, contact fitting and empirical
distance offsets are deliberately excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from chaoyang.pipeline.interaction_contact_robot_dev_v1 import (
    FiniteSurfacePatch,
    point_to_finite_patch,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_interaction_occlusion_v1"
PHASE = "ROBOT15H_INTERACTION_W0_DIRECT_OBSERVED_VISIBLE_SURFACE"
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"
IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
MASK_SHAPE = (960, 1280)
DEPTH_SHAPE = (480, 640)
HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OBJECTS = ("playing_card_00", "playing_card_01", "playing_card_02")
MIN_LARGEST_COMPONENT_FRACTION = 0.98
INVENTORY = ROOT / (
    "_run/current/0915_robot15h_window_start_inventory_v1/attempts/"
    "attempt_0001/BATCH_MANIFEST.json"
)
HAWOR_ROOT = ROOT / (
    "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001/hawor"
)
DEPTH_ROOT = ROOT / (
    "_run/current/0915_robot15h_foundationstereo_wave0_recovery_v1/"
    "attempts/attempt_0001"
)
MASK_ROOT = ROOT / (
    "_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/"
    "attempts/attempt_0001"
)
HAND_ROOT = ROOT / (
    "_run/current/0915_robot15h_sam31_temporal_identity_quality_v2/"
    "attempts/attempt_0001"
)
OBJECT_ROOT = ROOT / (
    "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001"
)
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_INTERACTION_V1"
RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_INTERACTION_OCCLUSION_V1_RESULT.json"
CLAIM_LIMIT = (
    "Development interaction evidence from direct-observed HaWoR 2D projections, "
    "consumer-admitted same-frame SAM3.1 hand roles, Stereo visible surfaces and "
    "finite directly visible object patches only. It is "
    "not anatomical fingertip ground truth, strict Contact, control, training, "
    "deployment, hidden geometry, or external metric accuracy authority."
)


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
    source = existing.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
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
    if len({str(row.get("session_id")) for row in rows}) != 4:
        raise RuntimeError("duplicate W0 session identity")
    return rows


def rows_by_session(batch: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    key = "results" if isinstance(batch.get("results"), list) else "sessions"
    rows = batch.get(key)
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
        raise RuntimeError("task packet differs from frozen weights-ABSENT specification")
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
    pid, ticks = claim.get("pid"), claim.get("proc_start_ticks")
    if (
        claim.get("schema_version") != "0915-robot15h-interaction-writer-claim-v1"
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
        raise RuntimeError("Interaction writer claim/fence mismatch")
    return claim


def heartbeat() -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(os.getpid()),
            "--status", "RUNNING", "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:]
        )


class PackedMask:
    def __init__(self, path: Path, expected_frames: int) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.packed = np.asarray(archive["packed"], np.uint8)
            self.frame_count = int(archive["frame_count"])
            self.height = int(archive["height"])
            self.width = int(archive["width"])
            self.bitorder = str(archive["bitorder"])
        expected_shape = (expected_frames, (MASK_SHAPE[0] * MASK_SHAPE[1] + 7) // 8)
        if (
            self.frame_count != expected_frames
            or (self.height, self.width) != MASK_SHAPE
            or self.bitorder != "big"
            or self.packed.shape != expected_shape
        ):
            raise RuntimeError(f"packed task-object mask contract drift: {path}")

    def unpack(self, frame_index: int) -> np.ndarray:
        return np.unpackbits(
            self.packed[frame_index], bitorder="big", count=self.height * self.width,
        ).reshape(self.height, self.width).astype(bool)


def load_depth(path: Path, expected_frame: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if (
        int(result["frame_id"]) != expected_frame
        or str(result["depth_reference"]) != DEPTH_REFERENCE
        or result["depth_m"].shape != DEPTH_SHAPE
        or result["valid"].shape != DEPTH_SHAPE
        or result["lr_consistent"].shape != DEPTH_SHAPE
        or result["lr_residual_px"].shape != DEPTH_SHAPE
        or result["physical_left_intrinsics"].shape != (3, 3)
    ):
        raise RuntimeError(f"Depth frame contract drift: {path}")
    return result


def backproject(pixel_uv: Sequence[float], depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    x, y = (float(value) for value in pixel_uv)
    z = float(depth_m)
    return np.asarray([
        (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    ])


def sample_finger_associated_visible_surface(
    *,
    source_pixel_uv: Sequence[float],
    direct_observed: bool,
    provenance: str,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    lr_consistent: np.ndarray,
    lr_residual_px: np.ndarray,
    intrinsics: np.ndarray,
    hand_mask: np.ndarray,
    hand_semantic_admitted: bool,
    hand_consumer_allowed: bool,
    object_union: np.ndarray,
    associated_hand: str,
    associated_finger: str,
    radius_depth_px: int = 4,
) -> dict[str, Any]:
    """Associate a direct 2D finger projection with a local visible Stereo surface.

    The hand-role mask is an admission gate, not anatomical ground truth.  The
    associated surface therefore remains role-unknown and is never promoted to
    anatomical fingertip truth.
    """

    base = {
        "associated_hand": associated_hand,
        "associated_finger": associated_finger,
        "surface_role": "unknown",
        "fingertip_surface_observation": False,
        "association_semantics": (
            "DIRECT_OBSERVED_HAWOR_2D_PLUS_SAM31_HAND_ROLE_TO_LOCAL_STEREO_VISIBLE_SURFACE;"
            "NOT_ANATOMICAL_TIP_GROUND_TRUTH"
        ),
    }
    if hand_mask.shape != MASK_SHAPE or object_union.shape != MASK_SHAPE:
        raise RuntimeError("hand/object semantic mask geometry drift")
    if not hand_consumer_allowed:
        return {**base, "status": "UNKNOWN", "reason": "HAND_ROLE_NOT_CONSUMER_ADMITTED"}
    if not direct_observed or provenance != "OBSERVED":
        return {**base, "status": "UNKNOWN", "reason": "HAWOR_NOT_DIRECT_OBSERVED"}
    if not hand_semantic_admitted:
        return {**base, "status": "UNKNOWN", "reason": "HAND_ROLE_FRAME_NOT_SEMANTIC_ADMITTED"}
    pixel = np.asarray(source_pixel_uv, np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return {**base, "status": "UNKNOWN", "reason": "NONFINITE_PROJECTED_FINGER"}
    depth_uv = (pixel - 0.5) / 2.0
    x0, y0 = np.rint(depth_uv).astype(int)
    if not (0 <= x0 < DEPTH_SHAPE[1] and 0 <= y0 < DEPTH_SHAPE[0]):
        return {
            **base, "status": "UNKNOWN", "reason": "PROJECTED_FINGER_OUT_OF_FRAME",
            "source_pixel_uv": pixel.tolist(),
        }
    x_min, x_max = max(0, x0 - radius_depth_px), min(DEPTH_SHAPE[1], x0 + radius_depth_px + 1)
    y_min, y_max = max(0, y0 - radius_depth_px), min(DEPTH_SHAPE[0], y0 + radius_depth_px + 1)
    local_y, local_x = np.mgrid[y_min:y_max, x_min:x_max]
    local_roi = (local_x - x0) ** 2 + (local_y - y0) ** 2 <= radius_depth_px ** 2
    ys, xs = local_y[local_roi], local_x[local_roi]
    sx = np.clip(np.rint(2.0 * xs + 0.5).astype(int), 0, MASK_SHAPE[1] - 1)
    sy = np.clip(np.rint(2.0 * ys + 0.5).astype(int), 0, MASK_SHAPE[0] - 1)
    hand_pixels = hand_mask[sy, sx]
    object_pixels = object_union[sy, sx]
    admitted = (
        depth_valid[ys, xs]
        & lr_consistent[ys, xs]
        & np.isfinite(depth_m[ys, xs])
        & (depth_m[ys, xs] > 0.0)
        & hand_pixels
        & ~object_pixels
    )
    roi_count = int(len(xs))
    admitted_count = int(np.count_nonzero(admitted))
    object_fraction = float(np.mean(object_pixels)) if roi_count else 1.0
    hand_purity = float(np.mean(hand_pixels)) if roi_count else 0.0
    quality: dict[str, Any] = {
        "roi_pixel_count": roi_count,
        "admitted_pixel_count": admitted_count,
        "object_mask_fraction": object_fraction,
        "hand_mask_purity": hand_purity,
        "independent_hand_role_mask_available": True,
        "hand_semantic_admitted": True,
    }
    if admitted_count < 6 or object_fraction > 0.35 or hand_purity < 0.45:
        return {
            **base, "status": "UNKNOWN", "reason": "LOCAL_HAND_ROLE_DEPTH_OR_OBJECT_GATE_FAILED",
            "source_pixel_uv": pixel.tolist(), "association_quality": quality,
        }
    selected_x, selected_y = xs[admitted], ys[admitted]
    selected_depth = np.asarray(depth_m[selected_y, selected_x], np.float64)
    median = float(np.median(selected_depth))
    mad = float(1.4826 * np.median(np.abs(selected_depth - median)))
    span = float(np.percentile(selected_depth, 90) - np.percentile(selected_depth, 10))
    quality.update({
        "local_depth_robust_sigma_m": mad,
        "local_depth_p90_p10_span_m": span,
        "lr_residual_median_px": float(np.median(lr_residual_px[selected_y, selected_x])),
    })
    if mad > 0.008 or span > 0.025:
        return {
            **base, "status": "UNKNOWN", "reason": "LOCAL_DEPTH_DISCONTINUITY",
            "source_pixel_uv": pixel.tolist(), "association_quality": quality,
        }
    distances = (selected_x - depth_uv[0]) ** 2 + (selected_y - depth_uv[1]) ** 2
    best = int(np.argmin(distances))
    sample_uv = np.asarray([float(selected_x[best]), float(selected_y[best])])
    near = distances <= max(float(distances[best]) + 4.0, 4.0)
    sample_depth = float(np.median(selected_depth[near]))
    point = backproject(sample_uv, sample_depth, intrinsics)
    return {
        **base,
        "status": "OBSERVED_VISIBLE_SURFACE",
        "reason": None,
        "surface_point_xyz": point.tolist(),
        "source_pixel_uv": pixel.tolist(),
        "sample_pixel_uv_depth_domain": sample_uv.tolist(),
        "association_quality": quality,
        "depth_quality": {
            "optical_z_m": sample_depth,
            "depth_reference": DEPTH_REFERENCE,
            "lr_consistency_required": True,
            "external_metric_accuracy": "UNVERIFIED",
        },
    }


def finite_patch_from_frame(frame: Mapping[str, Any]) -> FiniteSurfacePatch | None:
    fields = ("center_xyz", "plane_normal", "inplane_rotation", "finite_visible_patch")
    if any(frame.get(field, {}).get("observability") == "UNOBSERVABLE" for field in fields):
        return None
    try:
        # Object6D's stored boundary is the largest directly observed connected
        # support component.  Reject fragmented support instead of convexifying
        # across components.  Its registered_valid_depth_fraction currently has
        # a known denominator-domain mismatch and is never an admission gate here.
        patch_residual = frame["finite_visible_patch"]["residual"]
        if float(patch_residual["largest_component_fraction"]) < MIN_LARGEST_COMPONENT_FRACTION:
            return None
        center = np.asarray(frame["center_xyz"]["estimate"]["xyz_m"], np.float64)
        normal = np.asarray(frame["plane_normal"]["estimate"]["unit_xyz"], np.float64)
        normal /= np.linalg.norm(normal)
        axis_u = np.asarray(
            frame["inplane_rotation"]["estimate"]["axis_unit_xyz"], np.float64,
        )
        axis_u -= normal * float(np.dot(axis_u, normal))
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        axis_v /= np.linalg.norm(axis_v)
        boundary = np.asarray(
            frame["finite_visible_patch"]["estimate"]["boundary_xyz_m"], np.float64,
        )
        relative = boundary - center
        hull = np.column_stack((relative @ axis_u, relative @ axis_v))
        if boundary.ndim != 2 or boundary.shape[0] < 3 or not np.isfinite(boundary).all():
            return None
        residual = float(frame["plane_normal"]["residual"]["p90_plane_distance_m"])
        return FiniteSurfacePatch(
            center_xyz=center,
            normal_xyz=normal,
            axis_u_xyz=axis_u,
            axis_v_xyz=axis_v,
            hull_uv_m=hull,
            plane_residual_p90_m=residual,
            registered_valid_depth_fraction=float(frame["registered_valid_depth_fraction"]),
            source_mask_pixel_count=int(frame["mask_pixel_count"]),
        )
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return None


def object_pixel_distance(
    mask: np.ndarray, uv: Sequence[float], distance_map: np.ndarray | None = None,
) -> tuple[float | None, bool]:
    pixel = np.asarray(uv, np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return None, False
    x, y = np.rint(pixel).astype(int)
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return None, False
    if mask[y, x]:
        return 0.0, True
    distance = (
        cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        if distance_map is None else distance_map
    )
    if distance.shape != mask.shape:
        raise RuntimeError("object-mask distance map geometry drift")
    return float(distance[y, x]), False


def pair_observation(
    *, frame_id: int, timestamp_s: float, hand: str, finger: str, object_id: str,
    sample: Mapping[str, Any], object_frame: Mapping[str, Any], object_mask: np.ndarray,
    object_distance_map: np.ndarray | None = None,
) -> dict[str, Any]:
    pixel_distance, overlaps_object = object_pixel_distance(
        object_mask, sample.get("source_pixel_uv", [math.nan, math.nan]),
        object_distance_map,
    )
    row: dict[str, Any] = {
        "frame_id": frame_id,
        "timestamp_s": timestamp_s,
        "hand_id": hand,
        "finger_id": finger,
        "object_id": object_id,
        "pair_key": f"{hand}:{finger}:{object_id}",
        "finger_associated_visible_surface_point": dict(sample),
        "object_visibility": str(object_frame.get("visibility_state", "UNKNOWN")),
        "object_mask_state": str(object_frame.get("mask_state", "unknown")),
        "two_d_adjacency": {
            "distance_to_visible_object_mask_px": pixel_distance,
            "adjacent_within_20px": pixel_distance is not None and pixel_distance <= 20.0,
            "projected_finger_inside_object_mask": overlaps_object,
        },
        "occlusion_evidence": (
            "POSSIBLE_HAND_OBJECT_OCCLUSION_OR_CONTACT_BOUNDARY"
            if overlaps_object
            else "VISIBLE_SEPARATE_OR_UNKNOWN"
        ),
        "metric_interaction_state": "UNKNOWN",
        "metric": None,
        "short_gap_inferred": False,
        "contact_authority": "NONE",
    }
    patch = finite_patch_from_frame(object_frame)
    if sample.get("status") != "OBSERVED_VISIBLE_SURFACE" or patch is None:
        row["metric_blocker"] = (
            "FINGER_VISIBLE_SURFACE_UNKNOWN"
            if sample.get("status") != "OBSERVED_VISIBLE_SURFACE"
            else "OBJECT_FINITE_VISIBLE_PATCH_UNKNOWN"
        )
        return row
    metric = point_to_finite_patch(sample["surface_point_xyz"], patch)
    relative_z = float(sample["surface_point_xyz"][2]) - float(patch.center_xyz[2])
    if relative_z < -0.003:
        ordering = "FINGER_ASSOCIATED_SURFACE_CLOSER_TO_CAMERA"
    elif relative_z > 0.003:
        ordering = "OBJECT_VISIBLE_CENTER_CLOSER_TO_CAMERA"
    else:
        ordering = "RELATIVE_Z_WITHIN_3MM_INTERNAL_BAND"
    row.update({
        "metric_interaction_state": "DIRECT_VISIBLE_SURFACE_RELATION",
        "metric": {
            **metric,
            "relative_z_m": relative_z,
            "depth_ordering": ordering,
            "object_plane_residual_p90_m": patch.plane_residual_p90_m,
            "finger_surface_role": "unknown",
            "external_metric_accuracy": "UNVERIFIED",
            "finite_patch_support_source": "EXPLICIT_LARGEST_DIRECT_VISIBLE_SUPPORT_BOUNDARY",
            "finite_patch_component_count": int(
                object_frame["finite_visible_patch"]["residual"]["connected_component_count"]
            ),
            "finite_patch_largest_component_fraction": float(
                object_frame["finite_visible_patch"]["residual"]["largest_component_fraction"]
            ),
            "registered_valid_depth_fraction_used_as_quality_gate": False,
        },
        "metric_blocker": None,
    })
    return row


def add_relative_motion_diagnostics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_key"])].append(row)
    for sequence in grouped.values():
        sequence.sort(key=lambda item: int(item["frame_id"]))
        previous: dict[str, Any] | None = None
        for row in sequence:
            diagnostic = {
                "status": "UNKNOWN",
                "semantics": "SAME_FRAME_RELATIVE_GEOMETRY_DIAGNOSTIC_NOT_CONTACT",
                "approach_supported": False,
                "co_motion_supported": False,
            }
            if row["metric"] is not None and previous is not None and previous["metric"] is not None:
                if int(row["frame_id"]) == int(previous["frame_id"]) + 1:
                    change = float(row["metric"]["finite_patch_distance_m"]) - float(
                        previous["metric"]["finite_patch_distance_m"]
                    )
                    diagnostic = {
                        "status": "OBSERVED_CONSECUTIVE",
                        "semantics": "SAME_FRAME_RELATIVE_GEOMETRY_DIAGNOSTIC_NOT_CONTACT",
                        "finite_patch_distance_change_m": change,
                        "approach_supported": change <= -0.002,
                        "co_motion_supported": abs(change) <= 0.003,
                    }
            row["relative_motion"] = diagnostic
            previous = row if row["metric"] is not None else None


def decode_video(path: Path, expected_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    count = width = height = 0
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


def render_review(
    source: Path, destination: Path, frame_count: int,
    masks: Mapping[str, PackedMask], hand_roles: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame_id"])].append(row)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (height, width) != MASK_SHAPE:
        capture.release()
        raise RuntimeError(f"review source is not physical-left 1280x960: {(width, height)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("failed to open Interaction review writer")
    colors = {
        "playing_card_00": (50, 220, 70),
        "playing_card_01": (50, 170, 250),
        "playing_card_02": (220, 90, 220),
    }
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        overlay = image.copy()
        for object_id in OBJECTS:
            mask = masks[object_id].unpack(index)
            color = colors[object_id]
            overlay[mask] = (
                0.78 * overlay[mask].astype(np.float32) + 0.22 * np.asarray(color)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 1)
        for hand, color in (("left", (30, 230, 255)), ("right", (255, 190, 30))):
            role = hand_roles[hand]
            if role["consumer_allowed"] is not True:
                continue
            if role["states"][index].get("semantic_admitted") is not True:
                continue
            hand_mask = role["mask"].unpack(index)
            overlay[hand_mask] = (
                0.86 * overlay[hand_mask].astype(np.float32) + 0.14 * np.asarray(color)
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                hand_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 1)
        samples: dict[tuple[str, str], Mapping[str, Any]] = {}
        metric_count = 0
        inside_count = 0
        for row in by_frame[index]:
            sample = row["finger_associated_visible_surface_point"]
            key = (str(row["hand_id"]), str(row["finger_id"]))
            samples[key] = sample
            if row["metric"] is not None:
                metric_count += 1
                inside_count += int(row["metric"]["inside_visible_patch"])
        for (hand, _finger), sample in samples.items():
            uv = sample.get("source_pixel_uv")
            if uv is None:
                continue
            point = tuple(np.rint(uv).astype(int))
            color = (30, 230, 255) if hand == "left" else (255, 190, 30)
            cv2.circle(overlay, point, 4, color, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (8, 8), (760, 78), (0, 0, 0), -1)
        cv2.putText(
            overlay, f"frame {index:04d} | direct HaWoR 2D -> Stereo visible surface",
            (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, f"metric pair rows {metric_count} | inside finite patch {inside_count} | Contact: NONE",
            (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(overlay)
        index += 1
    capture.release()
    writer.release()
    decoded = decode_video(destination, frame_count)
    if not decoded["full_decode"]:
        raise RuntimeError(f"Interaction review failed full decode: {destination}")
    return {**decoded, "fps": fps}


def consumer_allowed_hand_sides(batch_row: Mapping[str, Any]) -> list[str]:
    """Return admitted anatomical sides without interpreting session status text."""

    admitted: list[str] = []
    for row in batch_row.get("side_results", []):
        side = row.get("side")
        role = row.get("role")
        if row.get("consumer_allowed") is True and side in (0, 1):
            expected_role = f"{HANDS[int(side)]}_hand"
            if role == expected_role:
                admitted.append(HANDS[int(side)])
    return admitted


def load_hand_roles(
    batch_row: Mapping[str, Any], inventory_row: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load only per-side hand roles explicitly admitted by quality-v2."""

    frame_count = int(inventory_row["frame_count"])
    result_path = verify_ref(batch_row["result"])
    result = load_json(result_path)
    if (
        result.get("session_id") != inventory_row["session_id"]
        or result.get("frame_count") != frame_count
        or result.get("side_results") != batch_row.get("side_results")
        or result.get("source_mutated") is not False
        or result.get("physical_deployment_authorized") is not False
        or result.get("training_eligible") is not False
    ):
        raise RuntimeError("SAM hand quality-v2 session contract drift")
    admitted = consumer_allowed_hand_sides(result)
    if not admitted:
        raise RuntimeError("NO_CONSUMER_ADMITTED_HAND_ROLE")
    roles: dict[str, dict[str, Any]] = {}
    references: dict[str, Any] = {"session_result": ref(result_path)}
    by_side = {int(row["side"]): row for row in result["side_results"] if "side" in row}
    for side_index, hand in enumerate(HANDS):
        row = by_side.get(side_index)
        if row is None or row.get("consumer_allowed") is not True:
            roles[hand] = {"consumer_allowed": False, "mask": None, "states": []}
            continue
        if (
            row.get("role") != f"{hand}_hand"
            or row.get("status") != "PASS_HAND_DIRECT_OBSERVED_PROXY"
            or row.get("quality", {}).get("direct_observed_admission_is_authoritative_gate") is not True
        ):
            raise RuntimeError(f"consumer-admitted hand role contract drift: {hand}")
        mask_path = verify_ref(row["semantic_archive"])
        state_path = verify_ref(row["state_ledger"])
        state_doc = load_json(state_path)
        states = state_doc.get("frames")
        if (
            state_doc.get("role") != f"{hand}_hand"
            or state_doc.get("empty_semantic_mask") != "UNKNOWN_NOT_ABSENT"
            or not isinstance(states, list)
            or len(states) != frame_count
            or [item.get("frame_id") for item in states] != list(range(frame_count))
        ):
            raise RuntimeError(f"hand semantic state axis drift: {hand}")
        roles[hand] = {
            "consumer_allowed": True,
            "mask": PackedMask(mask_path, frame_count),
            "states": states,
        }
        references[f"{hand}_semantic_mask"] = ref(mask_path)
        references[f"{hand}_state_ledger"] = ref(state_path)
    return roles, references


def load_card_inputs(
    inventory_row: Mapping[str, Any], hawor_row: Mapping[str, Any],
    depth_row: Mapping[str, Any], mask_row: Mapping[str, Any], object_row: Mapping[str, Any],
    hand_row: Mapping[str, Any],
) -> dict[str, Any]:
    frame_count = int(inventory_row["frame_count"])
    if (
        hawor_row.get("frame_count") != frame_count
        or hawor_row.get("task") != "playing_cards"
        or depth_row.get("status") != "PASSED"
        or mask_row.get("status") != "PASSED_VISIBLE_OBJECT_MASK_PROXY"
        or mask_row.get("object_mask_consumer_allowed") is not True
        or object_row.get("status") != "PASSED_DEVELOPMENT_VISIBLE_SURFACE"
        or object_row.get("object6d_consumer_allowed") is not True
    ):
        raise RuntimeError("playing-card upstream admission contract failed")
    hawor_path = verify_ref(hawor_row["npz"])
    with np.load(hawor_path, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    if (
        hawor["joints_2d"].shape != (2, frame_count, 21, 2)
        or hawor["observed"].shape != (2, frame_count)
        or hawor["provenance"].shape != (2, frame_count)
        or list(hawor["anatomical_side_names"].astype(str)) != list(HANDS)
        or list(hawor["mano_tip_indices"].astype(int)) != [4, 8, 12, 16, 20]
        or not np.array_equal(hawor["original_frame_indices"], np.arange(frame_count))
    ):
        raise RuntimeError("HaWoR direct-observation axis contract drift")
    depth_result = load_json(verify_ref(depth_row["result"]))
    depth_session_root = Path(depth_row["result"]["path"]).parent
    depth_summary_path = depth_session_root / "DEPTH_SUMMARY.json"
    depth_summary = load_json(depth_summary_path)
    depth_frames = depth_summary.get("frames")
    if (
        depth_result.get("consumption_authorized") is not True
        or depth_result.get("strict_metric_contact_authorized") is not False
        or not isinstance(depth_frames, list)
        or len(depth_frames) != frame_count
    ):
        raise RuntimeError("Depth consumer contract drift")
    object_path = verify_ref(object_row["object6d"])
    object_doc = load_json(object_path)
    if (
        object_doc.get("frame_count") != frame_count
        or object_doc.get("hidden_geometry_inferred") is not False
        or object_doc.get("contact_authority") != "NONE"
        or {str(item.get("instance_id")) for item in object_doc.get("objects", [])}
        != set(OBJECTS)
    ):
        raise RuntimeError("Object6D finite visible patch contract drift")
    object_by_id = {str(item["instance_id"]): item for item in object_doc["objects"]}
    manifest = load_json(verify_ref(mask_row["instance_manifest"]))
    manifest_by_id = {str(item["instance_id"]): item for item in manifest.get("instances", [])}
    if set(manifest_by_id) != set(OBJECTS):
        raise RuntimeError("independent task-object identity set drift")
    masks: dict[str, PackedMask] = {}
    mask_refs: dict[str, Any] = {"manifest": ref(verify_ref(mask_row["instance_manifest"]))}
    for object_id in OBJECTS:
        item = manifest_by_id[object_id]
        if item.get("object_mask_consumer_allowed") is not True:
            raise RuntimeError(f"task-object mask not admitted: {object_id}")
        path = verify_ref(item["semantic_archive"])
        masks[object_id] = PackedMask(path, frame_count)
        mask_refs[object_id] = ref(path)
    video = verify_ref(mask_row["input_video"])
    hand_roles, hand_references = load_hand_roles(hand_row, inventory_row)
    return {
        "hawor": hawor,
        "depth_frames": depth_frames,
        "objects": object_by_id,
        "masks": masks,
        "hand_roles": hand_roles,
        "video": video,
        "references": {
            "hawor_direct_observation": ref(hawor_path),
            "depth_result": ref(verify_ref(depth_row["result"])),
            "depth_summary": ref(depth_summary_path),
            "object6d_visible_patch": ref(object_path),
            "task_object_masks": mask_refs,
            "sam31_hand_quality_v2": hand_references,
            "physical_left_video": ref(video),
        },
    }


def run_card_session(
    inventory_row: Mapping[str, Any], inputs: Mapping[str, Any],
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
    hawor = inputs["hawor"]
    fps = float(hawor["fps"])
    rows: list[dict[str, Any]] = []
    observed_sample_counts = Counter()
    for frame_id in range(frame_count):
        depth_row = inputs["depth_frames"][frame_id]
        if depth_row.get("frame") != frame_id:
            raise RuntimeError(f"Depth frame manifest discontinuity: {frame_id}")
        depth_path = verify_ref(depth_row["artifact"])
        depth = load_depth(depth_path, frame_id)
        frame_masks = {object_id: inputs["masks"][object_id].unpack(frame_id) for object_id in OBJECTS}
        distance_maps = {
            object_id: cv2.distanceTransform(
                (~frame_masks[object_id]).astype(np.uint8), cv2.DIST_L2, 3,
            )
            for object_id in OBJECTS
        }
        object_union = np.logical_or.reduce(list(frame_masks.values()))
        samples: dict[tuple[str, str], dict[str, Any]] = {}
        for hand_index, hand in enumerate(HANDS):
            direct = bool(hawor["observed"][hand_index, frame_id])
            provenance = str(hawor["provenance"][hand_index, frame_id])
            hand_role = inputs["hand_roles"][hand]
            hand_allowed = hand_role["consumer_allowed"] is True
            hand_semantic_admitted = bool(
                hand_allowed and hand_role["states"][frame_id].get("semantic_admitted") is True
            )
            hand_mask = (
                hand_role["mask"].unpack(frame_id)
                if hand_allowed else np.zeros(MASK_SHAPE, bool)
            )
            for finger, tip_index in zip(FINGERS, [4, 8, 12, 16, 20]):
                sample = sample_finger_associated_visible_surface(
                    source_pixel_uv=hawor["joints_2d"][hand_index, frame_id, tip_index],
                    direct_observed=direct,
                    provenance=provenance,
                    depth_m=depth["depth_m"],
                    depth_valid=depth["valid"],
                    lr_consistent=depth["lr_consistent"],
                    lr_residual_px=depth["lr_residual_px"],
                    intrinsics=depth["physical_left_intrinsics"],
                    hand_mask=hand_mask,
                    hand_semantic_admitted=hand_semantic_admitted,
                    hand_consumer_allowed=hand_allowed,
                    object_union=object_union,
                    associated_hand=hand,
                    associated_finger=finger,
                )
                samples[(hand, finger)] = sample
                if sample["status"] == "OBSERVED_VISIBLE_SURFACE":
                    observed_sample_counts[f"{hand}:{finger}"] += 1
        for hand in HANDS:
            for finger in FINGERS:
                for object_id in OBJECTS:
                    rows.append(pair_observation(
                        frame_id=frame_id,
                        timestamp_s=float(hawor["original_frame_indices"][frame_id]) / fps,
                        hand=hand,
                        finger=finger,
                        object_id=object_id,
                        sample=samples[(hand, finger)],
                        object_frame=inputs["objects"][object_id]["frames"][frame_id],
                        object_mask=frame_masks[object_id],
                        object_distance_map=distance_maps[object_id],
                    ))
    add_relative_motion_diagnostics(rows)
    metric_rows = sum(row["metric"] is not None for row in rows)
    inside_rows = sum(bool(row.get("metric", {}).get("inside_visible_patch")) for row in rows if row["metric"] is not None)
    document = {
        "schema_version": "0915-robot15h-interaction-evidence-session-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "pair_denominator": frame_count * len(HANDS) * len(FINGERS) * len(OBJECTS),
        "status": "COMPLETED_DEVELOPMENT_INTERACTION_EVIDENCE",
        "authority": AUTHORITY,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "fixed_pair_axes": {
            "hands": list(HANDS), "fingers": list(FINGERS), "objects": list(OBJECTS),
        },
        "finger_surface_semantics": (
            "SAM_HAND_ADMITTED_FINGER_ASSOCIATED_VISIBLE_SURFACE_POINT_NOT_ANATOMICAL_FINGERTIP"
        ),
        "alignment": {
            "status": "NOT_REQUIRED_FOR_STEREO_ASSOCIATED_VISIBLE_SURFACE",
            "hawor_absolute_z_consumed": False,
            "metric_wrist_translation_authorized": False,
            "constant_48mm_bias_subtracted": False,
            "contact_fitting_alignment_used": False,
        },
        "metric_policy": {
            "short_gap_inferred_consumed": False,
            "removal_consumed": False,
            "infinite_plane_contact_used": False,
            "finite_direct_visible_patch_only": True,
            "external_metric_accuracy": "UNVERIFIED",
        },
        "inputs": inputs["references"],
        "rows": rows,
        "summary": {
            "observed_finger_surface_samples_by_pair": dict(observed_sample_counts),
            "metric_pair_rows": metric_rows,
            "inside_finite_patch_rows": inside_rows,
            "strict_contact_rows": 0,
        },
        "strict_metric_contact_authorized": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "claim_limit": CLAIM_LIMIT,
    }
    evidence_stage = stage / "INTERACTION_EVIDENCE.json"
    atomic_json(evidence_stage, document)
    review_stage = stage / "INTERACTION_REVIEW.mp4"
    review = render_review(
        inputs["video"], review_stage, frame_count, inputs["masks"],
        inputs["hand_roles"], rows,
    )
    passed = metric_rows > 0
    result = {
        "schema_version": "0915-robot15h-interaction-session-result-v1",
        "task_id": TASK_ID,
        "session_id": session_id,
        "task": task,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": "PASSED_DEVELOPMENT_INTERACTION_EVIDENCE" if passed else "REJECTED_QUALITY",
        "first_blocker": None if passed else "NO_DIRECT_VISIBLE_SURFACE_RELATION",
        "interaction_attempted": True,
        "interaction_consumer_allowed": passed,
        "metric_pair_rows": metric_rows,
        "inside_finite_patch_rows": inside_rows,
        "strict_metric_contact_authorized": False,
        "authority": AUTHORITY,
        "weights": "ABSENT",
        "gpu_used": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "source_mutated": False,
        "interaction_evidence": projected_ref(evidence_stage, final / evidence_stage.name),
        "review": {
            **review,
            "video": projected_ref(review_stage, final / review_stage.name),
        },
    }
    atomic_json(stage / "RESULT.json", result)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    published = load_json(final / "RESULT.json")
    verify_ref(published["interaction_evidence"])
    verify_ref(published["review"]["video"])
    shallow = visual / f"{session_id}_INTERACTION_REVIEW.mp4"
    try:
        os.link(final / "INTERACTION_REVIEW.mp4", shallow)
    except OSError:
        shutil.copy2(final / "INTERACTION_REVIEW.mp4", shallow)
    shallow_reference = ref(shallow)
    if shallow_reference["sha256"] != published["review"]["video"]["sha256"]:
        raise RuntimeError("shallow Interaction review differs from atomic session review")
    return {
        "session_id": session_id,
        "task": task,
        "source_group": inventory_row["source_group"],
        "frame_count": frame_count,
        "status": published["status"],
        "first_blocker": published["first_blocker"],
        "interaction_attempted": True,
        "interaction_artifact_emitted": True,
        "interaction_consumer_allowed": passed,
        "metric_pair_rows": metric_rows,
        "inside_finite_patch_rows": inside_rows,
        "result": ref(final / "RESULT.json"),
        "interaction_evidence": ref(final / "INTERACTION_EVIDENCE.json"),
        "review": {**review, "video": shallow_reference},
    }


def blocked_row(
    inventory_row: Mapping[str, Any], blocker: str, *, status: str = "BLOCKED_UPSTREAM",
) -> dict[str, Any]:
    return {
        "session_id": inventory_row["session_id"],
        "task": inventory_row["task"],
        "source_group": inventory_row["source_group"],
        "frame_count": int(inventory_row["frame_count"]),
        "status": status,
        "first_blocker": blocker,
        "interaction_attempted": False,
        "interaction_artifact_emitted": False,
        "interaction_consumer_allowed": False,
        "metric_pair_rows": 0,
        "inside_finite_patch_rows": 0,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(rows), "attempted": 0, "passed": 0, "quality_rejected": 0,
        "blocked_upstream": 0, "failed_runtime": 0, "unrun": 0,
    }
    for row in rows:
        status = str(row["status"])
        counts["attempted"] += int(row.get("interaction_attempted") is True)
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
    output, visual, receipt = (
        args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve(),
    )
    validate_fixed_paths(output, visual, receipt)
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    packet, packet_path = validate_route()
    required_inputs = [
        INVENTORY,
        HAWOR_ROOT / "BATCH_RESULT.json",
        HAND_ROOT / "BATCH_RESULT.json",
        HAND_ROOT / "RESULT.json",
        DEPTH_ROOT / "BATCH_RESULT.json",
        MASK_ROOT / "BATCH_RESULT.json",
        OBJECT_ROOT / "BATCH_RESULT.json",
    ]
    for path in required_inputs:
        if not path.is_file():
            raise RuntimeError(f"required upstream artifact missing: {path}")
    signature_payload = {
        "schema_version": "0915-robot15h-interaction-wave0-run-signature-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": args.executor_epoch,
        "weights": "ABSENT",
        "gpu_used": False,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "task_packet": ref(packet_path),
        "inputs": [ref(path) for path in required_inputs],
        "code": [
            ref(Path(__file__)),
            ref(ROOT / "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py"),
        ],
        "config": {
            "hands": list(HANDS), "fingers": list(FINGERS), "objects": list(OBJECTS),
            "hawor_absolute_z_consumed": False,
            "short_gap_inferred_consumed": False,
            "constant_48mm_bias_subtracted": False,
            "contact_fitting_alignment_used": False,
            "removal_consumed": False,
        },
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    fencing_sha = hashlib.sha256(args.fencing_token.encode()).hexdigest()
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-robot15h-interaction-writer-claim-v1",
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
    hawor_rows = rows_by_session(load_json(HAWOR_ROOT / "BATCH_RESULT.json"), identities)
    hand_batch = load_json(HAND_ROOT / "BATCH_RESULT.json")
    hand_rows = rows_by_session(hand_batch, identities)
    depth_rows = rows_by_session(load_json(DEPTH_ROOT / "BATCH_RESULT.json"), identities)
    mask_rows = rows_by_session(load_json(MASK_ROOT / "BATCH_RESULT.json"), identities)
    object_batch = load_json(OBJECT_ROOT / "BATCH_RESULT.json")
    object_rows = rows_by_session(object_batch, identities)
    rows: list[dict[str, Any]] = []
    for inventory_row in inventory_rows:
        session_id = str(inventory_row["session_id"])
        object_row = object_rows[session_id]
        if object_row.get("object6d_consumer_allowed") is not True:
            rows.append(blocked_row(
                inventory_row,
                str(object_row.get("first_blocker") or "OBJECT6D_NOT_CONSUMER_ADMITTED"),
                status="BLOCKED_UPSTREAM_OBJECT6D",
            ))
        elif not consumer_allowed_hand_sides(hand_rows[session_id]):
            rows.append(blocked_row(
                inventory_row,
                "NO_CONSUMER_ADMITTED_SAM31_HAND_ROLE",
                status="BLOCKED_UPSTREAM_HAND_ROLE",
            ))
        else:
            try:
                inputs = load_card_inputs(
                    inventory_row, hawor_rows[session_id], depth_rows[session_id],
                    mask_rows[session_id], object_row, hand_rows[session_id],
                )
                rows.append(run_card_session(inventory_row, inputs, output, visual))
            except Exception as error:
                rows.append({
                    **blocked_row(inventory_row, f"{type(error).__name__}:{error}"),
                    "status": "FAILED_RUNTIME",
                    "interaction_attempted": True,
                })
        heartbeat()
    counts = summarize(rows)
    quality = (
        "PASSED_PARTIAL_W0_DEVELOPMENT_INTERACTION_EVIDENCE"
        if counts["passed"] > 0 and counts["failed_runtime"] == 0
        else "REJECTED_NO_INTERACTION_SESSION"
    )
    ledger = {
        "schema_version": "0915-robot15h-interaction-evidence-ledger-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "interaction_quality_status": quality,
        "weights": "ABSENT",
        "gpu_used": False,
        "rows": rows,
        "counts": counts,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "hawor_absolute_z_consumed": False,
        "short_gap_inferred_consumed": False,
        "removal_consumed": False,
        "strict_metric_contact_authorized": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "source_mutated": False,
    }
    ledger_path = output / "INTERACTION_EVIDENCE_LEDGER.json"
    atomic_json(ledger_path, ledger)
    batch = {
        "schema_version": "0915-robot15h-interaction-wave0-batch-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "interaction_quality_status": quality,
        "counts": counts,
        "sessions": rows,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "strict_metric_contact_authorized": False,
    }
    atomic_json(output / "BATCH_RESULT.json", batch)
    atomic_json(output / "METRICS.json", {
        "schema_version": "0915-robot15h-interaction-wave0-metrics-v1",
        "task_id": TASK_ID,
        "session_count": len(rows),
        "counts": counts,
        "metric_pair_rows": sum(int(row.get("metric_pair_rows", 0)) for row in rows),
        "inside_finite_patch_rows": sum(
            int(row.get("inside_finite_patch_rows", 0)) for row in rows
        ),
        "strict_contact_rows": 0,
        "r0_affected": False,
        "r2_independent_path_affected": False,
    })
    atomic_json(visual / "INDEX.json", {
        "schema_version": "0915-robot15h-interaction-wave0-visual-index-v1",
        "task_id": TASK_ID,
        "status": "COMPLETE",
        "videos": [row["review"]["video"] for row in rows if "review" in row],
    })
    top_status = "PASSED" if counts["failed_runtime"] == 0 else "FAILED_RUNTIME_FINAL"
    result = {
        "schema_version": "0915-robot15h-interaction-wave0-result-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "interaction_quality_status": quality,
        "counts": counts,
        "weights": "ABSENT",
        "gpu_used": False,
        "authority": AUTHORITY,
        "coordinate_domain": DEPTH_REFERENCE,
        "image_domain": IMAGE_DOMAIN,
        "hawor_absolute_z_consumed": False,
        "short_gap_inferred_consumed": False,
        "constant_48mm_bias_subtracted": False,
        "contact_fitting_alignment_used": False,
        "removal_consumed": False,
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
        "claim_limit": packet.get("claim_limit", CLAIM_LIMIT),
    }
    validate_writer_claim(
        output / "CLAIM.json", signature_sha=signature["run_signature_sha256"],
        executor_epoch=args.executor_epoch, fencing_sha=fencing_sha,
    )
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-interaction-wave0-run-receipt-v1",
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "status": top_status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt),
        "gpu_used": False,
    })
    print(json.dumps({
        "status": top_status, "counts": counts, "result": str(output / "RESULT.json"),
    }))
    return 0 if top_status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
