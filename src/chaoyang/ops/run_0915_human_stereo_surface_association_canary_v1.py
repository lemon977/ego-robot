#!/usr/bin/env python3
"""Diagnose 0915 Human/Stereo surface association without rerunning models.

The previous development canary compared a MANO joint centre with a camera-visible
surface.  Those are different physical points, especially at the thumb and little
finger.  This bounded successor materialises the already frozen MANO parameters,
compares front-visible MANO vertices with direct Stereo surfaces on a frozen
non-contact split, and independently rechecks fingertip-associated surfaces against
finite visible object patches.  It never fits on object/contact evidence and never
relaxes the 5 mm Contact gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

PROJECT = Path(__file__).resolve().parents[3]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.interaction_contact_robot_dev_v1 import (
    CONTACT_DISTANCE_M,
    DEPTH_SHAPE,
    MASK_SHAPE,
    backproject_pixels,
    build_finite_surface_patch,
    build_pair_windows,
    classify_contact_candidate,
    fit_human_stereo_ray_depth_alignment,
    mask_to_depth_uv,
    point_to_finite_patch,
    sample_finger_associated_visible_surface,
)


TASK_ID = "0915_human_stereo_surface_association_canary_v1"
PHASE = "0915_HUMAN_STEREO_SURFACE_ASSOCIATION_CANARY_V1"
SESSION_ID = "play_cards_0915_001"
FRAME_COUNT = 150
OBJECTS = ("playing_card_00", "playing_card_01", "playing_card_02")
HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_INDICES = (4, 8, 12, 16, 20)
HAND_MASKS = {"left": "left_hand_00", "right": "right_hand_00"}
SLEEVE_MASKS = {
    "left": "left_finger_sleeve_cluster_00",
    "right": "right_finger_sleeve_cluster_00",
}
DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"

HAWOR = PROJECT / (
    "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/"
    "bounded_output_guarded_identity_fixed/play_cards_0915_001/"
    "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
)
SAM_ROOT = PROJECT / "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
DEPTH_ROOT = PROJECT / "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001"
OBJECT_ROOT = PROJECT / "_run/current/0915_planar_object6d_observability_canary_v2/attempts/attempt_0001"
MANO_ROOT = PROJECT / "assets/models/vendor/hawor/mano"
HAWOR_ROOT = PROJECT / "vendor/HaWoR"
HAWOR_LAUNCHER = PROJECT / "src/chaoyang/ops/hawor_python.sh"
OUTPUT = PROJECT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = PROJECT / "docs/current/visuals/0915_HUMAN_STEREO_SURFACE_ASSOCIATION_CANARY_V1"
TERMINAL_RECEIPT = PROJECT / "tasks/receipts/0915_HUMAN_STEREO_SURFACE_ASSOCIATION_CANARY_V1_RESULT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = PROJECT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current task packet differs from frozen specification")
    state = load_json(PROJECT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    index = load_json(PROJECT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize this task")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA differs from current route")
    return packet, packet_path


def heartbeat() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", PHASE],
        cwd=PROJECT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


class PackedMask:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            self.packed = np.asarray(archive["packed"], np.uint8)
            metadata = (
                int(archive["frame_count"]), int(archive["height"]),
                int(archive["width"]), str(archive["bitorder"]),
            )
        if metadata != (FRAME_COUNT, MASK_SHAPE[0], MASK_SHAPE[1], "big"):
            raise RuntimeError(f"mask domain drift: {path}: {metadata}")
        self.path = path

    def frame(self, index: int) -> np.ndarray:
        return np.unpackbits(
            self.packed[index], bitorder="big", count=MASK_SHAPE[0] * MASK_SHAPE[1],
        ).reshape(MASK_SHAPE).astype(bool)


def load_masks() -> tuple[dict[str, PackedMask], dict[str, list[str]], list[dict[str, Any]]]:
    manifest_path = SAM_ROOT / "ROLE_MANIFEST.json"
    temporal_path = SAM_ROOT / "TEMPORAL_STATE_LEDGER.json"
    manifest = load_json(manifest_path)
    temporal = load_json(temporal_path)
    if manifest.get("image_domain") != "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP":
        raise RuntimeError("SAM input is not the frozen physical-left encoded domain")
    by_id = {str(row["instance_id"]): row for row in manifest["instances"]}
    states = {
        str(row["instance_id"]): [str(item["state"]) for item in row["frames"]]
        for row in temporal["instances"]
    }
    names = list(OBJECTS) + list(HAND_MASKS.values()) + list(SLEEVE_MASKS.values())
    masks: dict[str, PackedMask] = {}
    refs = [ref(manifest_path), ref(temporal_path)]
    for name in names:
        row = by_id.get(name)
        if row is None:
            raise RuntimeError(f"sealed SAM instance missing: {name}")
        path = SAM_ROOT / str(row["mask_archive"])
        masks[name] = PackedMask(path)
        refs.append(ref(path))
        if len(states.get(name, [])) != FRAME_COUNT:
            raise RuntimeError(f"SAM state axis drift: {name}")
    return masks, states, refs


def admitted_mask(
    masks: Mapping[str, PackedMask], states: Mapping[str, list[str]], name: str, frame: int,
) -> np.ndarray:
    if states[name][frame] == "unknown":
        return np.zeros(MASK_SHAPE, bool)
    return masks[name].frame(frame)


def depth_frame(index: int) -> dict[str, np.ndarray]:
    path = DEPTH_ROOT / f"frames/{index:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if (
        int(result["frame_id"]) != index
        or str(result["depth_reference"]) != DEPTH_REFERENCE
        or result["depth_m"].shape != DEPTH_SHAPE
    ):
        raise RuntimeError(f"Depth frame contract drift: {path}")
    return result


def mask_boundary_distance(mask: np.ndarray, pixel_uv: Sequence[float]) -> float | None:
    value = np.asarray(pixel_uv, np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or not np.any(mask):
        return None
    x, y = np.rint(value).astype(int)
    if not (0 <= x < MASK_SHAPE[1] and 0 <= y < MASK_SHAPE[0]):
        return None
    if mask[y, x]:
        return 0.0
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[y, x])


def center_connected_surface_sample(
    *,
    source_pixel_uv_1280: Sequence[float],
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    lr_consistent: np.ndarray,
    lr_residual_px: np.ndarray,
    intrinsics: np.ndarray,
    hand_mask_1280: np.ndarray,
    sleeve_mask_1280: np.ndarray | None,
    object_union_1280: np.ndarray,
    associated_hand: str,
    associated_finger: str,
    radius_depth_px: int = 4,
    maximum_anchor_distance_px: float = 2.25,
    maximum_component_depth_delta_m: float = 0.015,
) -> dict[str, Any]:
    """Return only the depth component connected to the projected fingertip anchor."""

    pixel = np.asarray(source_pixel_uv_1280, np.float64)
    depth = np.asarray(depth_m, np.float64)
    valid = np.asarray(depth_valid, bool)
    lr_ok = np.asarray(lr_consistent, bool)
    residual = np.asarray(lr_residual_px, np.float64)
    hand = np.asarray(hand_mask_1280, bool)
    objects = np.asarray(object_union_1280, bool)
    sleeve = None if sleeve_mask_1280 is None else np.asarray(sleeve_mask_1280, bool)
    base = {
        "associated_hand": associated_hand,
        "associated_finger": associated_finger,
        "fingertip_surface_observation": False,
    }
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return {**base, "status": "UNKNOWN", "reason": "NONFINITE_PROJECTED_FINGER"}
    center = mask_to_depth_uv(pixel)
    x0, y0 = np.rint(center).astype(int)
    if not (0 <= x0 < DEPTH_SHAPE[1] and 0 <= y0 < DEPTH_SHAPE[0]):
        return {**base, "status": "UNKNOWN", "reason": "PROJECTED_FINGER_OUT_OF_FRAME"}
    x1, x2 = max(0, x0 - radius_depth_px), min(DEPTH_SHAPE[1], x0 + radius_depth_px + 1)
    y1, y2 = max(0, y0 - radius_depth_px), min(DEPTH_SHAPE[0], y0 + radius_depth_px + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    circle = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius_depth_px ** 2
    sx = np.clip(np.rint(2.0 * xx + 0.5).astype(int), 0, MASK_SHAPE[1] - 1)
    sy = np.clip(np.rint(2.0 * yy + 0.5).astype(int), 0, MASK_SHAPE[0] - 1)
    semantic_hand = hand[sy, sx]
    semantic_object = objects[sy, sx]
    admitted = (
        circle & valid[yy, xx] & lr_ok[yy, xx] & np.isfinite(depth[yy, xx])
        & (depth[yy, xx] > 0) & semantic_hand & ~semantic_object
    )
    roi_count = int(np.count_nonzero(circle))
    hand_purity = float(np.mean(semantic_hand[circle])) if roi_count else 0.0
    object_fraction = float(np.mean(semantic_object[circle])) if roi_count else 1.0
    ay, ax = np.nonzero(admitted)
    if ax.size == 0:
        return {
            **base, "status": "UNKNOWN", "reason": "NO_ADMITTED_ANCHOR",
            "source_pixel_uv": pixel.tolist(),
            "association_quality": {"roi_pixel_count": roi_count, "hand_mask_purity": hand_purity,
                                    "object_mask_fraction": object_fraction, "anchor_distance_px": None},
        }
    gx, gy = xx[ay, ax], yy[ay, ax]
    distances = np.hypot(gx - center[0], gy - center[1])
    anchor_index = int(np.argmin(distances))
    anchor_distance = float(distances[anchor_index])
    if anchor_distance > maximum_anchor_distance_px:
        return {
            **base, "status": "UNKNOWN", "reason": "ADMITTED_SURFACE_TOO_FAR_FROM_PROJECTION",
            "source_pixel_uv": pixel.tolist(),
            "association_quality": {"roi_pixel_count": roi_count, "hand_mask_purity": hand_purity,
                                    "object_mask_fraction": object_fraction,
                                    "anchor_distance_px": anchor_distance},
        }
    anchor_x, anchor_y = int(gx[anchor_index]), int(gy[anchor_index])
    anchor_depth = float(depth[anchor_y, anchor_x])
    compatible = admitted & (np.abs(depth[yy, xx] - anchor_depth) <= maximum_component_depth_delta_m)
    labels_count, labels = cv2.connectedComponents(compatible.astype(np.uint8), connectivity=8)
    del labels_count
    label = int(labels[anchor_y - y1, anchor_x - x1])
    component = labels == label if label > 0 else np.zeros_like(compatible)
    cy, cx = np.nonzero(component)
    component_count = int(cx.size)
    if component_count < 6 or hand_purity < 0.45 or object_fraction > 0.20:
        return {
            **base, "status": "UNKNOWN", "reason": "CENTER_COMPONENT_ADMISSION_FAILED",
            "source_pixel_uv": pixel.tolist(),
            "association_quality": {
                "roi_pixel_count": roi_count, "component_pixel_count": component_count,
                "hand_mask_purity": hand_purity, "object_mask_fraction": object_fraction,
                "anchor_distance_px": anchor_distance,
            },
        }
    selected_x, selected_y = xx[cy, cx], yy[cy, cx]
    selected_depth = depth[selected_y, selected_x]
    median_depth = float(np.median(selected_depth))
    mad = float(1.4826 * np.median(np.abs(selected_depth - median_depth)))
    span = float(np.percentile(selected_depth, 90) - np.percentile(selected_depth, 10))
    lr_values = residual[selected_y, selected_x]
    finite_lr = lr_values[np.isfinite(lr_values)]
    surface_role = "skin"
    if sleeve is not None and float(np.mean(sleeve[sy[component], sx[component]])) >= 0.50:
        surface_role = "sleeve"
    point = backproject_pixels(
        np.asarray([[anchor_x, anchor_y]], np.float64), np.asarray([median_depth]), intrinsics,
    )[0]
    qualified = bool(
        surface_role == "skin" and hand_purity >= 0.70 and object_fraction <= 0.05
        and component_count >= 8 and anchor_distance <= 1.5 and mad <= 0.008 and span <= 0.015
    )
    return {
        **base, "status": "OBSERVED_VISIBLE_SURFACE", "reason": None,
        "source_pixel_uv": pixel.tolist(), "sample_pixel_uv_depth_domain": [anchor_x, anchor_y],
        "surface_point_xyz": point.tolist(), "surface_role": surface_role,
        "association_quality": {
            "roi_pixel_count": roi_count, "component_pixel_count": component_count,
            "hand_mask_purity": hand_purity, "object_mask_fraction": object_fraction,
            "anchor_distance_px": anchor_distance,
            "center_connected_component_required": True,
        },
        "depth_quality": {
            "local_depth_median_m": median_depth, "local_depth_robust_sigma_m": mad,
            "local_depth_p90_p10_m": span,
            "lr_residual_median_px": float(np.median(finite_lr)) if finite_lr.size else None,
            "lr_consistent_fraction": 1.0,
        },
        "fingertip_surface_observation": qualified,
    }


def fit_surface_alignment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit one bounded session-static affine ray-depth map on frozen non-contact rows."""

    admitted = [
        row for row in rows
        if row.get("whole_frame_hand_contact_excluded") is False
        and int(row.get("support_count", 0)) >= 20
        and np.isfinite([row.get("mano_surface_depth_m"), row.get("stereo_surface_depth_m")]).all()
    ]
    train = [row for row in admitted if int(row["frame_id"]) % 5 != 0]
    holdout = [row for row in admitted if int(row["frame_id"]) % 5 == 0]
    base = {
        "schema_version": "HUMAN_STEREO_ALIGNMENT_CHECK_V2",
        "fit_scope": "SESSION_STATIC_MANO_SURFACE_TO_STEREO_SURFACE_RAY_DEPTH_SCALE_OFFSET",
        "fit_data_policy": "WHOLE_FRAME_HAND_NON_CONTACT_FRONT_VISIBLE_SURFACE_ONLY",
        "contact_or_object_fit_used": False,
        "fixed_48mm_bias_used": False,
        "train_row_count": len(train), "holdout_row_count": len(holdout),
    }
    if len(train) < 30 or len(holdout) < 8:
        return {**base, "status": "BLOCKED_INSUFFICIENT_NON_CONTACT_SURFACE_EVIDENCE",
                "metric_translation_authorized": False, "fit": None, "heldout": None}
    x = np.asarray([row["mano_surface_depth_m"] for row in train], np.float64)
    y = np.asarray([row["stereo_surface_depth_m"] for row in train], np.float64)
    keep = np.ones(x.size, bool)
    scale, offset = 1.0, 0.0
    for _ in range(4):
        design = np.column_stack((x[keep], np.ones(int(np.count_nonzero(keep)))))
        estimate, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        scale = float(np.clip(estimate[0], 0.8, 1.2))
        offset = float(np.clip(estimate[1], -0.15, 0.15))
        residual = y - (scale * x + offset)
        robust = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        next_keep = np.abs(residual - np.median(residual)) <= max(0.010, 3.0 * robust)
        if np.array_equal(next_keep, keep) or int(np.count_nonzero(next_keep)) < 30:
            break
        keep = next_keep
    hx = np.asarray([row["mano_surface_depth_m"] for row in holdout], np.float64)
    hy = np.asarray([row["stereo_surface_depth_m"] for row in holdout], np.float64)
    residual = np.abs(hy - (scale * hx + offset))
    heldout = {
        "median_abs_residual_m": float(np.median(residual)),
        "p90_abs_residual_m": float(np.percentile(residual, 90)),
        "max_abs_residual_m": float(np.max(residual)),
        "frame_ids": sorted({int(row["frame_id"]) for row in holdout}),
        "row_ids_sha256": canonical_sha([
            [int(row["frame_id"]), str(row["hand_id"])] for row in holdout
        ]),
    }
    passed = bool(
        0.8 <= scale <= 1.2 and abs(offset) <= 0.15
        and heldout["median_abs_residual_m"] <= 0.015
        and heldout["p90_abs_residual_m"] <= 0.030
    )
    return {
        **base, "status": "PASS_DEVELOPMENT_ALIGNMENT" if passed else "REJECTED_HELDOUT_ALIGNMENT",
        "metric_translation_authorized": passed,
        "fit": {"scale": scale, "offset_m": offset, "retained_train_rows": int(np.count_nonzero(keep))},
        "heldout": heldout,
    }


def _mano_worker(hawor_path: Path) -> int:
    expected = (PROJECT / "_run/current/environments/hawor-py310-v1").resolve()
    if Path(sys.prefix).resolve() != expected or os.environ.get("CHA0YANG_HAWOR_ENV_ROOT") != str(expected):
        raise RuntimeError("WRONG_RUNTIME_ENTRYPOINT: MANO materialisation requires hawor_python.sh")
    import torch
    sys.path.insert(0, str(HAWOR_ROOT))
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    with np.load(hawor_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    vertices = np.full((2, FRAME_COUNT, 778, 3), np.nan, np.float32)
    prior = Path.cwd()
    try:
        os.chdir(HAWOR_ROOT)
        for side, mano in ((0, run_mano_left), (1, run_mano)):
            selected = np.flatnonzero(data["observed"][side])
            roots = torch.from_numpy(data["root_orient_camera"][side, selected][None].astype(np.float32))
            poses = torch.from_numpy(data["hand_pose_rotmat"][side, selected][None].astype(np.float32))
            trans = torch.from_numpy(data["root_translation_camera"][side, selected][None].astype(np.float32))
            betas = torch.from_numpy(data["betas"][side, selected][None].astype(np.float32))
            output = mano(
                trans, rotation_matrix_to_angle_axis(roots), rotation_matrix_to_angle_axis(poses),
                betas=betas, use_cuda=False,
            )["vertices"][0].detach().cpu().numpy()
            vertices[side, selected] = output
    finally:
        os.chdir(prior)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, vertices_camera=vertices, observed=data["observed"])
    sys.stdout.buffer.write(buffer.getvalue())
    return 0


def materialize_mano_vertices() -> tuple[np.ndarray, dict[str, Any]]:
    completed = subprocess.run(
        [str(HAWOR_LAUNCHER), str(Path(__file__).resolve()), "--mano-worker", str(HAWOR)],
        cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError("MANO materialisation failed: " + completed.stderr.decode("utf-8", "replace")[-4000:])
    with np.load(io.BytesIO(completed.stdout), allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices_camera"], np.float32)
        observed = np.asarray(archive["observed"], bool)
    if vertices.shape != (2, FRAME_COUNT, 778, 3) or observed.shape != (2, FRAME_COUNT):
        raise RuntimeError("MANO materialisation shape drift")
    return vertices, {
        "runtime": "PINNED_HAWOR_PY310_CPU", "model_inference_performed": False,
        "source": "FROZEN_HAWOR_MANO_PARAMETERS", "gpu_used": False,
        "finite_vertex_side_frames": int(np.all(np.isfinite(vertices), axis=(2, 3)).sum()),
    }


def mano_surface_row(
    *, frame: int, hand: str, vertices: np.ndarray, intrinsics_1280: np.ndarray,
    depth: Mapping[str, np.ndarray], hand_mask: np.ndarray, object_union: np.ndarray,
    whole_frame_hand_contact_excluded: bool,
) -> dict[str, Any]:
    base = {
        "frame_id": frame, "hand_id": hand,
        "whole_frame_hand_contact_excluded": whole_frame_hand_contact_excluded,
        "source_kind": "FROZEN_MANO_FRONT_VISIBLE_SURFACE_TO_DIRECT_STEREO",
    }
    if whole_frame_hand_contact_excluded:
        return {**base, "status": "EXCLUDED_CONTACT_NEAR_FRAME_HAND", "support_count": 0}
    xyz = np.asarray(vertices, np.float64)
    finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz = xyz[finite]
    if xyz.size == 0:
        return {**base, "status": "UNKNOWN_NO_MANO_VERTICES", "support_count": 0}
    projected = (intrinsics_1280 @ xyz.T).T
    uv1280 = projected[:, :2] / projected[:, 2:3]
    uv_depth = (uv1280 - 0.5) / 2.0
    rounded = np.rint(uv_depth).astype(int)
    inside = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < DEPTH_SHAPE[1])
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < DEPTH_SHAPE[0])
    )
    xyz, uv_depth, rounded = xyz[inside], uv_depth[inside], rounded[inside]
    if rounded.size == 0:
        return {**base, "status": "UNKNOWN_NO_PROJECTED_VERTICES", "support_count": 0}
    linear = rounded[:, 1] * DEPTH_SHAPE[1] + rounded[:, 0]
    order = np.lexsort((xyz[:, 2], linear))
    linear_ordered = linear[order]
    first = np.concatenate(([True], linear_ordered[1:] != linear_ordered[:-1]))
    selected = order[first]
    xyz, uv_depth, rounded = xyz[selected], uv_depth[selected], rounded[selected]
    sx = np.clip(np.rint(2.0 * rounded[:, 0] + 0.5).astype(int), 0, MASK_SHAPE[1] - 1)
    sy = np.clip(np.rint(2.0 * rounded[:, 1] + 0.5).astype(int), 0, MASK_SHAPE[0] - 1)
    object_guard = cv2.dilate(object_union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    yy, xx = rounded[:, 1], rounded[:, 0]
    admitted = (
        hand_mask[sy, sx] & ~object_guard[sy, sx]
        & np.asarray(depth["valid"], bool)[yy, xx]
        & np.asarray(depth["lr_consistent"], bool)[yy, xx]
        & np.isfinite(depth["depth_m"][yy, xx]) & (depth["depth_m"][yy, xx] > 0)
    )
    mano_z = xyz[admitted, 2]
    stereo_z = np.asarray(depth["depth_m"], np.float64)[yy[admitted], xx[admitted]]
    physically_bounded = np.abs(stereo_z - mano_z) <= 0.15
    mano_z, stereo_z = mano_z[physically_bounded], stereo_z[physically_bounded]
    if mano_z.size < 20:
        return {**base, "status": "UNKNOWN_INSUFFICIENT_DIRECT_SURFACE_SUPPORT",
                "support_count": int(mano_z.size)}
    return {
        **base, "status": "OBSERVED_SURFACE_PAIR", "support_count": int(mano_z.size),
        "mano_surface_depth_m": float(np.median(mano_z)),
        "stereo_surface_depth_m": float(np.median(stereo_z)),
        "raw_median_signed_difference_m": float(np.median(stereo_z - mano_z)),
        "raw_pair_abs_difference_p90_m": float(np.percentile(np.abs(stereo_z - mano_z), 90)),
        "sampling": "FRONTMOST_PROJECTED_MANO_VERTEX_PER_DEPTH_PIXEL_EXACT_STEREO_PIXEL",
        "object_exclusion_radius_1280_px": 15,
    }


def render_diagnostic(metrics: Mapping[str, Any], path: Path) -> None:
    canvas = np.full((720, 1280, 3), 246, np.uint8)
    cv2.putText(canvas, "0915 Human/Stereo surface association", (46, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
    old = metrics["joint_center_baseline"]
    new = metrics["mano_surface_candidate"]
    lines = [
        "Frozen inputs: HaWoR MANO + SAM3.1 + encoded FoundationStereo + Object6D v2",
        "No model rerun / no Removal-Clean / no object-contact fitting / no 48 mm bias",
        f"Joint-centre baseline holdout P90: {old['heldout_p90_mm']:.2f} mm",
        f"MANO-surface candidate holdout P90: {new['heldout_p90_mm']:.2f} mm",
        f"MANO-surface candidate holdout median: {new['heldout_median_mm']:.2f} mm",
        f"Alignment: {new['status']}",
        f"Centre-connected tip observations: {metrics['center_connected_tip_observations']}",
        f"Closest finite visible-patch distance: {metrics['closest_finite_patch_mm']:.2f} mm",
        f"5-frame Contact windows: {metrics['admitted_contact_windows']}",
        f"R1 decision: {metrics['r1_status']}",
    ]
    y = 125
    for index, line in enumerate(lines):
        color = (25, 25, 25) if index < 2 else (55, 55, 55)
        cv2.putText(canvas, line, (55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 50
    cv2.imwrite(str(path), canvas)


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--mano-worker":
        return _mano_worker(Path(sys.argv[2]).resolve(strict=True))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token", required=True)
    args = parser.parse_args()
    packet, packet_path = validate_route()
    output, visual, receipt = args.output_root.resolve(), args.visual_root.resolve(), args.receipt.resolve()
    if output != OUTPUT.resolve() or visual != VISUAL.resolve() or receipt != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("fixed output/visual/receipt namespace mismatch")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive executor epoch and fencing token >=16 characters required")
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    output.mkdir(parents=True)

    masks, states, sam_refs = load_masks()
    object_doc = load_json(OBJECT_ROOT / "OBJECT6D_OBSERVABILITY_V2.json")
    object_result = load_json(OBJECT_ROOT / "RESULT.json")
    depth_result = load_json(DEPTH_ROOT / "RESULT.json")
    if object_result.get("status") != "PASSED" or depth_result.get("status") != "PASSED":
        raise RuntimeError("frozen Depth/Object6D is not terminal PASSED")
    with np.load(HAWOR, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    observed = np.asarray(hawor["observed"], bool)
    provenance = np.asarray(hawor["provenance"])
    if not np.all(provenance[observed] == "BOUNDED_PARAMETER_FIT") or not np.all(provenance[~observed] == "MISSING"):
        raise RuntimeError("HaWoR direct-observation provenance drift")
    vertices, materialisation = materialize_mano_vertices()

    input_paths = [
        HAWOR, MANO_ROOT / "MANO_LEFT.pkl", MANO_ROOT / "MANO_RIGHT.pkl",
        DEPTH_ROOT / "RESULT.json", DEPTH_ROOT / "DEPTH_CONTRACT.json",
        OBJECT_ROOT / "RESULT.json", OBJECT_ROOT / "OBJECT6D_OBSERVABILITY_V2.json",
        *[DEPTH_ROOT / f"frames/{index:06d}.npz" for index in range(FRAME_COUNT)],
    ]
    input_refs = [ref(path) for path in input_paths] + sam_refs
    input_snapshot = {row["path"]: row["sha256"] for row in input_refs}
    signature_payload = {
        "schema_version": "0915-human-stereo-surface-association-run-signature-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "executor_epoch": args.executor_epoch,
        "weights": "ABSENT", "frame_count": FRAME_COUNT, "gpu_used": False,
        "model_policy": "FROZEN_PARAMETERS_MANO_GEOMETRY_MATERIALISATION_ONLY_NO_MODEL_RERUN",
        "contact_distance_gate_m": CONTACT_DISTANCE_M,
        "forbidden_inputs": ["Removal", "Clean", "short_gap_inferred", "PICO26_HAND_RESULTS", "archive"],
        "task_packet": ref(packet_path), "inputs": input_refs,
        "code": [ref(Path(__file__)), ref(PROJECT / "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py")],
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-human-stereo-surface-association-writer-claim-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "attempt_id": output.name,
        "status": "CLAIMED", "weights": "ABSENT", "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()), "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"], "unique_write_root": str(output),
        "task_packet": ref(packet_path),
    }
    atomic_json(output / "CLAIM.json", claim)
    heartbeat()

    object_by_id = {row["instance_id"]: row for row in object_doc["objects"]}
    surface_rows: list[dict[str, Any]] = []
    old_alignment_rows: list[dict[str, Any]] = []
    old_tip_count = 0
    new_tip_count = 0
    old_observed_count = 0
    new_observed_count = 0
    recheck_rows: list[dict[str, Any]] = []
    prior_distance: dict[tuple[str, str, str], float] = {}
    timestamps = np.arange(FRAME_COUNT, dtype=np.float64) / float(hawor["fps"])

    for frame in range(FRAME_COUNT):
        depth = depth_frame(frame)
        k_depth = np.asarray(depth["physical_left_intrinsics"], np.float64)
        object_masks = {name: admitted_mask(masks, states, name, frame) for name in OBJECTS}
        object_union = np.logical_or.reduce(list(object_masks.values()))
        patches: dict[str, Any] = {}
        for object_id in OBJECTS:
            row = object_by_id[object_id]["frames"][frame]
            center, plane, axis = row["center_xyz"], row["plane_normal"], row["inplane_rotation"]
            if center["observability"] == "UNOBSERVABLE" or plane["observability"] == "UNOBSERVABLE":
                patches[object_id] = None
            else:
                patches[object_id] = build_finite_surface_patch(
                    center_xyz=center["estimate"]["xyz_m"], normal_xyz=plane["estimate"]["unit_xyz"],
                    axis_hint_xyz=None if axis["observability"] == "UNOBSERVABLE" else axis["estimate"]["axis_unit_xyz"],
                    mask_1280=object_masks[object_id], depth_m=depth["depth_m"],
                    depth_valid=np.asarray(depth["valid"], bool) & np.asarray(depth["lr_consistent"], bool),
                    intrinsics=k_depth, plane_residual_p90_m=float(plane["residual"]["p90_plane_distance_m"]),
                    registered_valid_depth_fraction=float(row["registered_valid_depth_fraction"]),
                )
        for side, hand in enumerate(HANDS):
            hand_mask = admitted_mask(masks, states, HAND_MASKS[hand], frame)
            sleeve_mask = admitted_mask(masks, states, SLEEVE_MASKS[hand], frame)
            near_hand = False
            if observed[side, frame]:
                for joint in TIP_INDICES:
                    distances = [
                        value for mask in object_masks.values()
                        if (value := mask_boundary_distance(mask, hawor["joints_2d"][side, frame, joint])) is not None
                    ]
                    near_hand = near_hand or bool(distances and min(distances) < 30.0)
            surface_rows.append(mano_surface_row(
                frame=frame, hand=hand, vertices=vertices[side, frame],
                intrinsics_1280=np.asarray(hawor["intrinsics"][frame], np.float64),
                depth=depth, hand_mask=hand_mask, object_union=object_union,
                whole_frame_hand_contact_excluded=near_hand,
            ))
            for finger, joint in zip(FINGERS, TIP_INDICES):
                if not observed[side, frame]:
                    old_sample = new_sample = {"status": "UNKNOWN", "reason": "HAWOR_NOT_DIRECT_OBSERVED",
                                                  "fingertip_surface_observation": False}
                else:
                    common = dict(
                        source_pixel_uv_1280=hawor["joints_2d"][side, frame, joint],
                        depth_m=depth["depth_m"], depth_valid=depth["valid"],
                        lr_consistent=depth["lr_consistent"], lr_residual_px=depth["lr_residual_px"],
                        intrinsics=k_depth, hand_mask_1280=hand_mask, sleeve_mask_1280=sleeve_mask,
                        object_union_1280=object_union, associated_hand=hand, associated_finger=finger,
                    )
                    old_sample = sample_finger_associated_visible_surface(**common, anatomical_tip_evidence=True)
                    new_sample = center_connected_surface_sample(**common)
                old_observed_count += old_sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
                new_observed_count += new_sample.get("status") == "OBSERVED_VISIBLE_SURFACE"
                old_tip_count += old_sample.get("fingertip_surface_observation") is True
                new_tip_count += new_sample.get("fingertip_surface_observation") is True
                if old_sample.get("fingertip_surface_observation") is True:
                    distances = [
                        value for mask in object_masks.values()
                        if (value := mask_boundary_distance(mask, old_sample["source_pixel_uv"])) is not None
                    ]
                    if not distances or min(distances) >= 30.0:
                        old_alignment_rows.append({
                            "frame_id": frame, "hand_id": hand, "finger_id": finger,
                            "source_kind": "NON_CONTACT_VISIBLE_HAND_SURFACE", "near_task_object": False,
                            "contact_or_object_fit_used": False,
                            "hawor_ray_depth_m": float(hawor["joints_3d_camera"][side, frame, joint, 2]),
                            "stereo_surface_depth_m": float(old_sample["surface_point_xyz"][2]),
                        })
                for object_id in OBJECTS:
                    key = (hand, finger, object_id)
                    patch = patches[object_id]
                    if new_sample.get("status") != "OBSERVED_VISIBLE_SURFACE" or patch is None:
                        state, finite_distance, inside = "UNKNOWN", None, False
                    else:
                        distance = point_to_finite_patch(new_sample["surface_point_xyz"], patch)
                        finite_distance = float(distance["finite_patch_distance_m"])
                        inside = bool(distance["inside_visible_patch"])
                        previous = prior_distance.get(key)
                        approach = previous is not None and previous - finite_distance >= 0.001
                        classification = classify_contact_candidate(
                            finite_patch_distance_m=finite_distance, inside_visible_patch=inside,
                            local_depth_robust_sigma_m=new_sample["depth_quality"]["local_depth_robust_sigma_m"],
                            plane_residual_p90_m=patch.plane_residual_p90_m,
                            lr_residual_median_px=new_sample["depth_quality"]["lr_residual_median_px"],
                            approach_supported=approach, co_motion_supported=False, tactile_supported=False,
                        )
                        state = classification["state"]
                        prior_distance[key] = finite_distance
                    recheck_rows.append({
                        "frame_id": frame, "timestamp_s": float(timestamps[frame]),
                        "hand_id": hand, "finger_id": finger, "object_id": object_id,
                        "state": state, "finite_patch_distance_m": finite_distance,
                        "inside_visible_patch": inside,
                    })
        if frame in {49, 99}:
            heartbeat()

    old_alignment = fit_human_stereo_ray_depth_alignment(old_alignment_rows)
    alignment = fit_surface_alignment(surface_rows)
    alignment.update({
        "session_id": SESSION_ID, "coordinate_domain": DEPTH_REFERENCE,
        "surface_correspondence": "MANO_FRONT_VISIBLE_VERTEX_TO_EXACT_DIRECT_STEREO_PIXEL",
        "contact_holdout_policy": "WHOLE_FRAME_HAND_EXCLUDED_WHEN_ANY_TIP_WITHIN_30PX_OF_TASK_OBJECT",
        "heldout_split": "frame_id_mod_5_equals_0",
        "mano_materialisation": materialisation,
        "joint_center_baseline": {
            "status": old_alignment["status"], "train_row_count": old_alignment["train_row_count"],
            "holdout_row_count": old_alignment["holdout_row_count"],
            "fit": old_alignment["fit"], "heldout": old_alignment["heldout"],
            "semantics": "DIAGNOSTIC_MISMATCH_JOINT_CENTRE_VS_VISIBLE_SURFACE",
        },
    })
    windows = build_pair_windows(recheck_rows)
    finite = [row["finite_patch_distance_m"] for row in recheck_rows if row["finite_patch_distance_m"] is not None]
    state_counts = dict(Counter(row["state"] for row in recheck_rows))
    admitted_windows = [row for row in windows if row["terminal"] == "ADMITTED_LOCAL_WINDOW"]
    comparison = {
        "schema_version": "FINGERTIP_SURFACE_ASSOCIATION_COMPARISON_V1",
        "session_id": SESSION_ID, "fixed_side_finger_denominator": FRAME_COUNT * len(HANDS) * len(FINGERS),
        "old": {"visible_surface_count": int(old_observed_count), "qualified_tip_count": int(old_tip_count),
                "sampler": "UNCONSTRAINED_LOCAL_MEDIAN"},
        "candidate": {"visible_surface_count": int(new_observed_count), "qualified_tip_count": int(new_tip_count),
                      "sampler": "PROJECTED_ANCHOR_CENTER_CONNECTED_DEPTH_COMPONENT"},
        "coverage_not_used_as_alignment_fit_target": True,
        "contact_or_object_fit_used": False,
    }
    recheck = {
        "schema_version": "INTERACTION_CONTACT_RECHECK_V1", "session_id": SESSION_ID,
        "fixed_record_denominator": len(recheck_rows), "distance_gate_m": CONTACT_DISTANCE_M,
        "distance_gate_was_uncertainty_expanded": False,
        "finite_visible_patch_required": True, "contact_or_object_fit_used": False,
        "state_counts": state_counts, "windows": windows,
        "admitted_window_count": len(admitted_windows),
        "closest_finite_patch_distance_m": min(finite) if finite else None,
        "inside_visible_patch_count": sum(row["inside_visible_patch"] for row in recheck_rows),
        "closest_rows": sorted(
            [row for row in recheck_rows if row["finite_patch_distance_m"] is not None],
            key=lambda row: row["finite_patch_distance_m"],
        )[:20],
        "tactile_reconsumed": False,
        "claim_limit": "Geometry-only recheck; no Contact truth or probability.",
    }
    blockers: list[str] = []
    if alignment.get("metric_translation_authorized") is not True:
        blockers.append("HUMAN_STEREO_SURFACE_ALIGNMENT_NOT_ADMITTED")
    if not admitted_windows:
        blockers.append("NO_FIXED_PAIR_FIVE_FRAME_CONTACT_WINDOW_AT_UNCHANGED_5MM_GATE")
    if admitted_windows:
        blockers.append("KAIHAND_PAD_TO_HUMAN_WRIST_FRAME_ALIGNMENT_UNVERIFIED")
    decision = {
        "schema_version": "0915_SURFACE_ASSOCIATION_DOWNSTREAM_DECISION_V1",
        "session_id": SESSION_ID, "r0_baseline_remains_authoritative": True,
        "r1_status": "BLOCKED_LOCAL_EVIDENCE" if blockers else "READY_FOR_SEPARATE_R1_SUCCESSOR",
        "first_blocker": blockers[0] if blockers else None, "blockers": blockers,
        "q22_or_wrist_modified": False, "r2_status": "NOT_RUN_R1_NOT_EXECUTED",
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }

    association = {
        "schema_version": "MANO_SURFACE_ASSOCIATION_V1", "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT, "coordinate_domain": DEPTH_REFERENCE,
        "records": surface_rows, "contact_or_object_fit_used": False,
        "fixed_48mm_bias_used": False, "short_gap_inferred_consumed": False,
        "removal_or_clean_consumed": False, "mano_materialisation": materialisation,
    }
    atomic_json(output / "MANO_SURFACE_ASSOCIATION_V1.json", association)
    atomic_json(output / "HUMAN_STEREO_ALIGNMENT_CHECK_V2.json", alignment)
    atomic_json(output / "FINGERTIP_SURFACE_ASSOCIATION_COMPARISON_V1.json", comparison)
    atomic_json(output / "INTERACTION_CONTACT_RECHECK_V1.json", recheck)
    atomic_json(output / "DOWNSTREAM_DECISION_V1.json", decision)

    old_p90 = float(old_alignment["heldout"]["p90_abs_residual_m"]) * 1000.0
    new_p90 = float(alignment["heldout"]["p90_abs_residual_m"]) * 1000.0 if alignment.get("heldout") else float("inf")
    new_median = float(alignment["heldout"]["median_abs_residual_m"]) * 1000.0 if alignment.get("heldout") else float("inf")
    metrics = {
        "schema_version": "0915-human-stereo-surface-association-metrics-v1",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "joint_center_baseline": {"status": old_alignment["status"], "heldout_p90_mm": old_p90,
                                  "heldout_median_mm": float(old_alignment["heldout"]["median_abs_residual_m"]) * 1000.0},
        "mano_surface_candidate": {"status": alignment["status"], "heldout_p90_mm": new_p90,
                                   "heldout_median_mm": new_median},
        "center_connected_tip_observations": int(new_tip_count),
        "closest_finite_patch_mm": (min(finite) * 1000.0 if finite else None),
        "admitted_contact_windows": len(admitted_windows), "contact_state_counts": state_counts,
        "r1_status": decision["r1_status"], "r1_first_blocker": decision["first_blocker"],
        "authority": AUTHORITY,
    }
    atomic_json(output / "METRICS.json", metrics)
    visual.mkdir(parents=True, exist_ok=False)
    render_diagnostic(metrics, visual / "SURFACE_ALIGNMENT_DIAGNOSTIC.png")
    readme = f"""# 0915 Human/Stereo 表面关联诊断

固定样本为 `play_cards_0915_001` 的 150 帧。HaWoR、SAM3.1、FoundationStereo 和 Object6D v2 均未重跑；MANO 只从冻结参数在固定 CPU 环境中重建表面。

- 旧 joint-centre ↔ visible-surface hold-out P90：`{old_p90:.2f} mm`。
- MANO-surface ↔ Stereo-surface hold-out P90：`{new_p90:.2f} mm`，状态 `{alignment['status']}`。
- 中心连通指尖表面资格：`{new_tip_count}/{FRAME_COUNT * len(HANDS) * len(FINGERS)}`；旧方法为 `{old_tip_count}`。
- 不变 `5 mm`、有限可见牌面门下最近距离：`{metrics['closest_finite_patch_mm']:.2f} mm`；五帧窗口 `{len(admitted_windows)}`。
- R1：`{decision['r1_status']}`；首阻塞 `{decision['first_blocker']}`。本任务没有修改 q22 或 wrist。

该结果只说明开发级几何关联是否自洽，不是接触真值、外部毫米精度、控制真值或部署授权。没有使用统一 48 mm 偏置，没有使用物体/Contact 数据拟合对齐，也没有读取 Removal/Clean。
"""
    (visual / "README_ZH.md").write_text(readme, encoding="utf-8")

    for raw, expected in input_snapshot.items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"read-only frozen input changed: {path}")
    result = {
        "schema_version": "0915-human-stereo-surface-association-result-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "status": "PASSED",
        "weights": "ABSENT", "source_mutated": False, "gpu_used": False,
        "model_rerun_performed": False, "removal_or_clean_consumed": False,
        "alignment_status": alignment["status"], "contact_window_count": len(admitted_windows),
        "r1_status": decision["r1_status"], "control_ground_truth": False,
        "physical_deployment_authorized": False, "calibration_authority": "DEVELOPMENT_ONLY",
        "artifacts": {
            "surface_association": ref(output / "MANO_SURFACE_ASSOCIATION_V1.json"),
            "alignment": ref(output / "HUMAN_STEREO_ALIGNMENT_CHECK_V2.json"),
            "finger_comparison": ref(output / "FINGERTIP_SURFACE_ASSOCIATION_COMPARISON_V1.json"),
            "contact_recheck": ref(output / "INTERACTION_CONTACT_RECHECK_V1.json"),
            "downstream_decision": ref(output / "DOWNSTREAM_DECISION_V1.json"),
            "metrics": ref(output / "METRICS.json"),
            "visual_readme": ref(visual / "README_ZH.md"),
            "diagnostic_png": ref(visual / "SURFACE_ALIGNMENT_DIAGNOSTIC.png"),
        },
        "run_signature": ref(output / "RUN_SIGNATURE.json"), "writer_claim": ref(output / "CLAIM.json"),
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-human-stereo-surface-association-run-receipt-v1",
        "task_id": TASK_ID, "status": "PASSED", "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt), "gpu_used": False,
    })
    print(json.dumps({"status": "PASSED", "alignment": alignment["status"],
                      "contact_windows": len(admitted_windows), "r1": decision["r1_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
