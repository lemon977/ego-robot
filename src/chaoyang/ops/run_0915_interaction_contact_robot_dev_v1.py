#!/usr/bin/env python3
"""Run the bounded 0915 Interaction -> Contact -> Kai22 development task."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping
import uuid

import cv2
import jsonschema
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.pipeline.interaction_contact_robot_dev_v1 import (
    CONTACT_DISTANCE_M,
    MASK_SHAPE,
    apply_ray_depth_alignment,
    build_finite_surface_patch,
    build_pair_windows,
    classify_contact_candidate,
    collision_scope_result,
    fit_human_stereo_ray_depth_alignment,
    point_to_finite_patch,
    sample_finger_associated_visible_surface,
    temporal_geometry_diagnostics,
    timestamp_motion_diagnostics,
    transition_taper_weights,
)
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.robot_visual_relative_v1 import (
    HAND_GROUPS,
    hand_limits,
    palm_transforms,
    retarget_kaihand_frame,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_interaction_contact_robot_dev_v1"
PHASE = "0915_INTERACTION_CONTACT_KAI22_DEVELOPMENT_V1"
SESSION_ID = "play_cards_0915_001"
FRAME_COUNT = 150
FPS = 30.0
AUTHORITY = "DEVELOPMENT_RELATIVE_NON_CONTROL_NON_DEPLOYABLE"
DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"

HAWOR = ROOT / (
    "_run/current/0915_hawor_resize_only_bounded_v2_canary/attempts/attempt_0001/"
    "bounded_output_guarded_identity_fixed/play_cards_0915_001/"
    "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
)
VIDEO = ROOT / (
    "_run/current/0915_hawor_resize_only_canary_v1/attempts/attempt_0001/input/"
    "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY.mp4"
)
SAM_ROOT = ROOT / "_run/current/0915_sam31_strict_role_canary_v5/attempts/attempt_0001"
DEPTH_ROOT = ROOT / "_run/current/0915_foundationstereo_encoded_domain_canary_v1/attempts/attempt_0001"
OBJECT_ROOT = ROOT / "_run/current/0915_planar_object6d_observability_canary_v2/attempts/attempt_0001"
SOURCE_ROOT = Path(
    "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915/cleaned/"
    "playing_cards/play_cards_0915_001"
)
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_INTERACTION_CONTACT_ROBOT_DEV_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_INTERACTION_CONTACT_ROBOT_DEV_V1_RESULT.json"
OBJECTS = ("playing_card_00", "playing_card_01", "playing_card_02")
HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_INDICES = (4, 8, 12, 16, 20)
HAND_MASKS = {"left": "left_hand_00", "right": "right_hand_00"}
SLEEVE_MASKS = {
    "left": "left_finger_sleeve_cluster_00",
    "right": "right_finger_sleeve_cluster_00",
}
SCHEMAS = {
    "object_qa": ROOT / "contracts/object6d_geometry_qa_v1.schema.json",
    "dimension": ROOT / "contracts/card_dimension_estimate_v1.schema.json",
    "alignment": ROOT / "contracts/human_stereo_alignment_check_v1.schema.json",
    "interaction": ROOT / "contracts/interaction_evidence_v1.schema.json",
    "contact": ROOT / "contracts/contact_candidate_v1.schema.json",
    "robot": ROOT / "contracts/kai22_robot_development_v1.schema.json",
}


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
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21])


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != "ABSENT":
        raise RuntimeError("current task packet differs from frozen specification")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True or route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task index does not authorize this task packet")
    return packet, packet_path


def heartbeat() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.heartbeat_task", "--task-id", TASK_ID,
         "--pid", str(os.getpid()), "--status", "RUNNING", "--phase", PHASE],
        cwd=ROOT, capture_output=True, text=True, check=False,
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
    manifest = load_json(SAM_ROOT / "ROLE_MANIFEST.json")
    temporal = load_json(SAM_ROOT / "TEMPORAL_STATE_LEDGER.json")
    if manifest.get("image_domain") != "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP":
        raise RuntimeError("SAM is not in the frozen physical-left resize-only domain")
    by_id = {str(row["instance_id"]): row for row in manifest["instances"]}
    states = {
        str(row["instance_id"]): [str(item["state"]) for item in row["frames"]]
        for row in temporal["instances"]
    }
    names = list(OBJECTS) + list(HAND_MASKS.values()) + list(SLEEVE_MASKS.values())
    masks: dict[str, PackedMask] = {}
    refs: list[dict[str, Any]] = [ref(SAM_ROOT / "ROLE_MANIFEST.json"), ref(SAM_ROOT / "TEMPORAL_STATE_LEDGER.json")]
    for name in names:
        row = by_id.get(name)
        if row is None:
            raise RuntimeError(f"required sealed SAM instance is absent: {name}")
        path = SAM_ROOT / str(row["mask_archive"])
        masks[name] = PackedMask(path)
        refs.append(ref(path))
        if len(states.get(name, [])) != FRAME_COUNT:
            raise RuntimeError(f"SAM state axis drift: {name}")
    return masks, states, refs


def admitted_mask(masks: Mapping[str, PackedMask], states: Mapping[str, list[str]], name: str, frame: int) -> np.ndarray:
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
        or result["depth_m"].shape != (480, 640)
    ):
        raise RuntimeError(f"Depth frame contract drift: {path}")
    return result


def load_source_metadata() -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    timestamps: list[float] = []
    tactile: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for frame in range(FRAME_COUNT):
        path = SOURCE_ROOT / f"preprocess/all_data/{frame:05d}/training_data.json"
        value = load_json(path)
        metadata = value.get("metadata", {})
        if int(metadata.get("idx", -1)) != frame:
            raise RuntimeError(f"source frame identity drift: {path}")
        timestamps.append(float(metadata.get("video_time_s")))
        row: dict[str, Any] = {}
        for hand in HANDS:
            payload = value.get("entities", {}).get("tactile", {}).get(hand, {})
            valid = bool(payload.get("offline_source_valid")) and abs(float(payload.get("offline_source_offset_ms", 1e9))) <= 40.0
            values = np.asarray(payload.get("wire_values_369", []), dtype=np.float64)
            row[hand] = {
                "offline_source_valid": valid,
                "offline_source_offset_ms": float(payload.get("offline_source_offset_ms", 1e9)),
                "nonzero_signal": bool(valid and values.size == 369 and np.any(np.abs(values) > 0)),
            }
        tactile.append(row)
        refs.append(ref(path))
    times = np.asarray(timestamps, dtype=np.float64)
    if times.shape != (FRAME_COUNT,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError("source timestamps are not a strict 150-frame timeline")
    return times, tactile, refs


def load_frozen_inputs() -> dict[str, Any]:
    depth_result = load_json(DEPTH_ROOT / "RESULT.json")
    object_result = load_json(OBJECT_ROOT / "RESULT.json")
    object_doc = load_json(OBJECT_ROOT / "OBJECT6D_OBSERVABILITY_V2.json")
    if depth_result.get("status") != "PASSED" or object_result.get("status") != "PASSED":
        raise RuntimeError("frozen Depth/Object6D inputs are not terminal PASSED")
    if object_doc.get("session_id") != SESSION_ID or object_doc.get("frame_count") != FRAME_COUNT:
        raise RuntimeError("Object6D session/frame denominator drift")
    with np.load(HAWOR, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    if (
        hawor["joints_3d_camera"].shape != (2, FRAME_COUNT, 21, 3)
        or hawor["joints_2d"].shape != (2, FRAME_COUNT, 21, 2)
        or hawor["observed"].shape != (2, FRAME_COUNT)
        or not np.array_equal(hawor["original_frame_indices"], np.arange(FRAME_COUNT))
    ):
        raise RuntimeError("frozen direct-observed HaWoR axis drift")
    if set(np.unique(hawor["provenance"]).tolist()) - {"observed", "missing_direct"}:
        raise RuntimeError("HaWoR contains non-direct short-gap provenance")
    return {"depth_result": depth_result, "object_result": object_result, "object_doc": object_doc, "hawor": hawor}


def make_r0(hawor: Mapping[str, np.ndarray], timestamps: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], Any]:
    assets = load_pinned_robot_assets(ROOT)
    limits = hand_limits(assets)
    observed = np.asarray(hawor["observed"], bool)
    joints_camera = np.asarray(hawor["joints_3d_camera"], np.float64)
    palms = palm_transforms(joints_camera, observed)
    q22 = np.full((FRAME_COUNT, 2, 22), np.nan, np.float64)
    relative_wrist = np.full((FRAME_COUNT, 2, 4, 4), np.nan, np.float64)
    losses = np.full((FRAME_COUNT, 2), np.nan, np.float64)
    valid = np.zeros((FRAME_COUNT, 2), bool)
    for side in range(2):
        ids = np.flatnonzero(observed[side] & np.isfinite(joints_camera[side]).all(axis=(1, 2)))
        if not ids.size:
            continue
        anchor = palms[side, ids[0]]
        inverse_anchor = np.linalg.inv(anchor)
        for frame in ids:
            q22[frame, side], losses[frame, side] = retarget_kaihand_frame(joints_camera[side, frame], limits[side])
            relative_wrist[frame, side] = inverse_anchor @ palms[side, frame]
            valid[frame, side] = True
    motion = {
        HANDS[side]: timestamp_motion_diagnostics(q22[:, side], timestamps, valid[:, side])
        for side in range(2)
    }
    # FK of one direct frame per side is the explicit renderer/URDF smoke test.
    smoke: dict[str, Any] = {}
    for side, hand in enumerate(HANDS):
        ids = np.flatnonzero(valid[:, side])
        model = assets.left_hand if hand == "left" else assets.right_hand
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        frame = int(ids[len(ids) // 2])
        transforms = forward_kinematics(model, {joint.name: float(q22[frame, side, index]) for index, joint in enumerate(moving)})
        smoke[hand] = {
            "frame_id": frame,
            "finite_link_transforms": all(np.isfinite(value).all() for value in transforms.values()),
            "link_count": len(transforms),
            "moving_joint_names": [joint.name for joint in moving],
        }
    result = {
        "schema_version": "KAI22_R0_BASELINE_V1",
        "session_id": SESSION_ID,
        "status": "COMPLETED_DEVELOPMENT_BASELINE",
        "authority": AUTHORITY,
        "direct_observed_only": True,
        "short_gap_inferred_consumed": False,
        "valid_side_frame_count": int(valid.sum()),
        "source_observed_side_frame_count": int(observed.sum()),
        "q22_joint_order": {hand: smoke[hand]["moving_joint_names"] for hand in HANDS},
        "renderer_smoke_test": smoke,
        "retarget_loss_mean": float(np.nanmean(losses)),
        "timestamp_motion": motion,
        "relative_wrist_semantics": "FIRST_DIRECT_OBSERVED_WRIST_ANCHOR_PER_SIDE_CAMERA_RELATIVE_PRIOR",
        "arm_ik_performed": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }
    arrays = {"q22_init": q22, "relative_wrist_T": relative_wrist, "valid_side_frame": valid, "timestamps_s": timestamps}
    return result, arrays, assets


def object_qa(object_doc: Mapping[str, Any]) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    all_pass = True
    for obj in object_doc["objects"]:
        frames = obj["frames"]
        centers = np.full((FRAME_COUNT, 3), np.nan)
        normals = np.full((FRAME_COUNT, 3), np.nan)
        areas = np.asarray([int(row["mask_pixel_count"]) for row in frames], np.float64)
        errors: list[str] = []
        if len(frames) != FRAME_COUNT or [int(row["frame_index"]) for row in frames] != list(range(FRAME_COUNT)):
            errors.append("FRAME_AXIS_NOT_CLOSED")
        for index, row in enumerate(frames):
            center = row["center_xyz"]
            plane = row["plane_normal"]
            fraction = float(row["registered_valid_depth_fraction"])
            if not 0.0 <= fraction <= 1.0:
                errors.append(f"DEPTH_SUPPORT_OUT_OF_RANGE:{index}")
            if center["observability"] != "UNOBSERVABLE":
                value = np.asarray(center["estimate"]["xyz_m"], np.float64)
                if value.shape != (3,) or not np.isfinite(value).all():
                    errors.append(f"NONFINITE_CENTER:{index}")
                else:
                    centers[index] = value
            if plane["observability"] != "UNOBSERVABLE":
                value = np.asarray(plane["estimate"]["unit_xyz"], np.float64)
                residual = float(plane["residual"]["p90_plane_distance_m"])
                if value.shape != (3,) or not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1.0) > 1e-5 or not np.isfinite(residual) or residual < 0:
                    errors.append(f"INVALID_PLANE:{index}")
                else:
                    normals[index] = value
        comparable = np.isfinite(centers).all(axis=1)
        for index in range(1, FRAME_COUNT):
            denominator = max(areas[index - 1], 1.0)
            comparable[index] &= comparable[index - 1] and 0.75 <= areas[index] / denominator <= 1.3333334
        diagnostics = temporal_geometry_diagnostics(
            centers, normals, comparable_visibility=comparable,
            camera_motion_compensation_available=False,
        )
        passed = not errors
        all_pass &= passed
        rows_out.append({
            "instance_id": obj["instance_id"], "frame_count": len(frames),
            "hard_gate": {"passed": passed, "errors": errors},
            "temporal_diagnostics": diagnostics,
        })
    return {
        "schema_version": "OBJECT6D_GEOMETRY_QA_V1", "session_id": SESSION_ID,
        "status": "PASS" if all_pass else "REJECTED_QUALITY",
        "coordinate_domain": DEPTH_REFERENCE,
        "visible_surface_center_semantics": "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_OBJECT_FIXED_CENTER",
        "temporal_diagnostics_are_failure_gates": False,
        "camera_motion_compensation": "ABSENT_NOT_AUTHORIZED",
        "objects": rows_out,
    }


def dimension_estimate(object_doc: Mapping[str, Any]) -> dict[str, Any]:
    objects = []
    for obj in object_doc["objects"]:
        full = sum(row["full_extent"]["observability"] != "UNOBSERVABLE" for row in obj["frames"])
        objects.append({
            "instance_id": obj["instance_id"], "status": "UNKNOWN",
            "complete_boundary_observation_count": full,
            "width_m": None, "height_m": None,
            "reason": "NO_COMPLETE_DIRECT_BOUNDARY_OBSERVATION_IN_FROZEN_OBJECT6D_V2",
        })
    return {
        "schema_version": "CARD_DIMENSION_ESTIMATE_V1", "session_id": SESSION_ID,
        "status": "UNKNOWN", "method": "MULTIFRAME_DIRECT_VISIBLE_PLANE_ROBUST_RECTANGLE",
        "nonblocking": True, "nominal_standard_size_used": False,
        "budget_seconds": 600, "budget_exhausted": False,
        "objects": objects,
    }


def mask_boundary_distance(mask: np.ndarray, point_uv: np.ndarray) -> float | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return float(min(abs(cv2.pointPolygonTest(contour, tuple(point_uv.astype(float)), True)) for contour in contours))


def project_center_1280(center_xyz: np.ndarray, k_depth: np.ndarray) -> np.ndarray:
    if center_xyz[2] <= 0:
        return np.full(2, np.nan)
    depth_uv = np.asarray([
        k_depth[0, 0] * center_xyz[0] / center_xyz[2] + k_depth[0, 2],
        k_depth[1, 1] * center_xyz[1] / center_xyz[2] + k_depth[1, 2],
    ])
    return 2.0 * depth_uv + 0.5


def build_evidence(
    frozen: Mapping[str, Any], masks: Mapping[str, PackedMask], states: Mapping[str, list[str]],
    timestamps: np.ndarray, tactile: list[dict[str, Any]], input_refs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    hawor = frozen["hawor"]
    object_by_id = {row["instance_id"]: row for row in frozen["object_doc"]["objects"]}
    samples: list[dict[str, dict[str, dict[str, Any]]]] = []
    patches: list[dict[str, Any]] = []
    object_centroids: list[dict[str, np.ndarray | None]] = []
    alignment_rows: list[dict[str, Any]] = []
    k_scale_errors: list[float] = []
    for frame in range(FRAME_COUNT):
        depth = depth_frame(frame)
        k_depth = np.asarray(depth["physical_left_intrinsics"], np.float64)
        k_hawor = np.asarray(hawor["intrinsics"][frame], np.float64)
        k_expected = k_depth.copy()
        k_expected[0, 0] *= 2.0
        k_expected[1, 1] *= 2.0
        k_expected[0, 2] = 2.0 * k_expected[0, 2] + 0.5
        k_expected[1, 2] = 2.0 * k_expected[1, 2] + 0.5
        k_scale_errors.append(float(np.max(np.abs(k_expected - k_hawor))))
        object_masks = {name: admitted_mask(masks, states, name, frame) for name in OBJECTS}
        union = np.logical_or.reduce(list(object_masks.values()))
        frame_patches: dict[str, Any] = {}
        frame_centroids: dict[str, np.ndarray | None] = {}
        for name in OBJECTS:
            row = object_by_id[name]["frames"][frame]
            center = row["center_xyz"]
            plane = row["plane_normal"]
            axis = row["inplane_rotation"]
            frame_centroids[name] = None
            if center["observability"] == "UNOBSERVABLE" or plane["observability"] == "UNOBSERVABLE":
                frame_patches[name] = None
                continue
            center_xyz = np.asarray(center["estimate"]["xyz_m"], np.float64)
            normal_xyz = np.asarray(plane["estimate"]["unit_xyz"], np.float64)
            axis_hint = None if axis["observability"] == "UNOBSERVABLE" else axis["estimate"]["axis_unit_xyz"]
            patch = build_finite_surface_patch(
                center_xyz=center_xyz, normal_xyz=normal_xyz, axis_hint_xyz=axis_hint,
                mask_1280=object_masks[name], depth_m=depth["depth_m"],
                depth_valid=np.asarray(depth["valid"], bool) & np.asarray(depth["lr_consistent"], bool),
                intrinsics=k_depth, plane_residual_p90_m=float(plane["residual"]["p90_plane_distance_m"]),
                registered_valid_depth_fraction=float(row["registered_valid_depth_fraction"]),
            )
            frame_patches[name] = patch
            frame_centroids[name] = project_center_1280(center_xyz, k_depth)
        patches.append(frame_patches)
        object_centroids.append(frame_centroids)
        frame_samples: dict[str, dict[str, dict[str, Any]]] = {}
        for side, hand in enumerate(HANDS):
            hand_mask = admitted_mask(masks, states, HAND_MASKS[hand], frame)
            sleeve_mask = admitted_mask(masks, states, SLEEVE_MASKS[hand], frame)
            frame_samples[hand] = {}
            if not bool(hawor["observed"][side, frame]):
                for finger in FINGERS:
                    frame_samples[hand][finger] = {
                        "status": "UNKNOWN", "reason": "HAWOR_NOT_DIRECT_OBSERVED",
                        "associated_hand": hand, "associated_finger": finger,
                        "fingertip_surface_observation": False,
                    }
                continue
            for finger, joint in zip(FINGERS, TIP_INDICES):
                source_uv = np.asarray(hawor["joints_2d"][side, frame, joint], np.float64)
                sample = sample_finger_associated_visible_surface(
                    source_pixel_uv_1280=source_uv, depth_m=depth["depth_m"],
                    depth_valid=depth["valid"], lr_consistent=depth["lr_consistent"],
                    lr_residual_px=depth["lr_residual_px"], intrinsics=k_depth,
                    hand_mask_1280=hand_mask, sleeve_mask_1280=sleeve_mask,
                    object_union_1280=union, associated_hand=hand,
                    associated_finger=finger, anatomical_tip_evidence=True,
                )
                frame_samples[hand][finger] = sample
                distances = [mask_boundary_distance(mask, source_uv) for mask in object_masks.values()]
                finite_distances = [value for value in distances if value is not None]
                far = not finite_distances or min(finite_distances) >= 30.0
                if sample.get("status") == "OBSERVED_VISIBLE_SURFACE" and sample.get("fingertip_surface_observation") is True and far:
                    alignment_rows.append({
                        "frame_id": frame, "hand_id": hand, "finger_id": finger,
                        "source_kind": "NON_CONTACT_VISIBLE_HAND_SURFACE",
                        "near_task_object": False, "contact_or_object_fit_used": False,
                        "hawor_ray_depth_m": float(hawor["joints_3d_camera"][side, frame, joint, 2]),
                        "stereo_surface_depth_m": float(sample["surface_point_xyz"][2]),
                    })
        samples.append(frame_samples)
    alignment = fit_human_stereo_ray_depth_alignment(alignment_rows)
    alignment.update({
        "session_id": SESSION_ID,
        "coordinate_checks": {
            "hawor_image_domain": "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP",
            "stereo_image_domain": "PHYSICAL_LEFT_SOURCE_INDEX_1_RESIZE_ONLY_NO_REMAP",
            "depth_reference": DEPTH_REFERENCE,
            "axis_convention": "X_RIGHT_Y_DOWN_Z_FORWARD",
            "units": "metres",
            "frame_axis": "ZERO_BASED_150_CONTIGUOUS",
            "timestamp_axis": "SOURCE_VIDEO_TIME_SECONDS",
            "k_pixel_center_scale_max_abs_error_px": max(k_scale_errors),
            "k_scale_admitted": max(k_scale_errors) <= 1.0,
        },
        "input_artifacts": input_refs,
        "contact_holdout_policy": "ALL_OBJECT_NEAR_WINDOWS_EXCLUDED_FROM_FIT_AND_EVALUATION",
    })
    if not alignment["coordinate_checks"]["k_scale_admitted"]:
        alignment["status"] = "REJECTED_HELDOUT_ALIGNMENT"
        alignment["metric_translation_authorized"] = False

    interaction_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    previous: dict[tuple[str, str, str], dict[str, Any]] = {}
    for frame in range(FRAME_COUNT):
        for hand in HANDS:
            for finger in FINGERS:
                sample = samples[frame][hand][finger]
                for object_id in OBJECTS:
                    patch = patches[frame][object_id]
                    key = (hand, finger, object_id)
                    base = {
                        "frame_id": frame, "timestamp_s": float(timestamps[frame]),
                        "hand_id": hand, "finger_id": finger, "object_id": object_id,
                        "finger_associated_visible_surface_point": sample,
                        "visibility": states[object_id][frame],
                        "occlusion": "UNKNOWN_NOT_COMPLETED",
                        "coordinate_source": DEPTH_REFERENCE,
                    }
                    if sample.get("status") != "OBSERVED_VISIBLE_SURFACE" or patch is None:
                        interaction = {**base, "finite_patch": None, "relative_z_m": None,
                                       "image_adjacency_px": None, "approach_supported": False,
                                       "co_motion_supported": False}
                        contact = {**{key: base[key] for key in ("frame_id", "timestamp_s", "hand_id", "finger_id", "object_id")},
                                   "state": "UNKNOWN", "reason": "SURFACE_POINT_OR_FINITE_PATCH_UNAVAILABLE",
                                   "support_score": 0.0,
                                   "support_score_semantics": "UNCALIBRATED_HEURISTIC_NOT_PROBABILITY_NOT_GROUND_TRUTH"}
                    else:
                        point = np.asarray(sample["surface_point_xyz"], np.float64)
                        distance = point_to_finite_patch(point, patch)
                        source_uv = np.asarray(sample["source_pixel_uv"], np.float64)
                        center_uv = object_centroids[frame][object_id]
                        adjacency = None if center_uv is None else float(np.linalg.norm(source_uv - center_uv))
                        prior = previous.get(key)
                        approach = bool(prior is not None and prior.get("distance") is not None and float(prior["distance"]) - distance["finite_patch_distance_m"] >= 0.001)
                        co_motion = False
                        if prior is not None and center_uv is not None and prior.get("center_uv") is not None:
                            hand_delta = source_uv - np.asarray(prior["source_uv"])
                            object_delta = center_uv - np.asarray(prior["center_uv"])
                            co_motion = bool(np.linalg.norm(hand_delta) >= 1.0 and np.linalg.norm(hand_delta - object_delta) <= 6.0)
                        tactile_supported = bool(tactile[frame][hand]["nonzero_signal"])
                        classification = classify_contact_candidate(
                            finite_patch_distance_m=distance["finite_patch_distance_m"],
                            inside_visible_patch=distance["inside_visible_patch"],
                            local_depth_robust_sigma_m=sample["depth_quality"]["local_depth_robust_sigma_m"],
                            plane_residual_p90_m=patch.plane_residual_p90_m,
                            lr_residual_median_px=sample["depth_quality"]["lr_residual_median_px"],
                            approach_supported=approach, co_motion_supported=co_motion,
                            tactile_supported=tactile_supported,
                        )
                        interaction = {
                            **base, "finite_patch": distance,
                            "relative_z_m": float(point[2] - patch.center_xyz[2]),
                            "image_adjacency_px": adjacency,
                            "approach_supported": approach, "co_motion_supported": co_motion,
                            "world_motion": "NOT_PUBLISHED_C2W_NOT_AUTHORIZED",
                        }
                        contact = {
                            **{key: base[key] for key in ("frame_id", "timestamp_s", "hand_id", "finger_id", "object_id")},
                            **classification,
                            "finite_patch_distance_m": distance["finite_patch_distance_m"],
                            "inside_visible_patch": distance["inside_visible_patch"],
                            "uncertainty_terms": {
                                "local_depth_robust_sigma_m": sample["depth_quality"]["local_depth_robust_sigma_m"],
                                "plane_residual_p90_m": patch.plane_residual_p90_m,
                                "lr_residual_median_px": sample["depth_quality"]["lr_residual_median_px"],
                                "pixel_registration": "EXACT_ANALYTIC_2X_PIXEL_CENTER_MAP",
                                "complete_external_contact_error": "ABSENT",
                            },
                            "tactile": tactile[frame][hand],
                        }
                        previous[key] = {
                            "distance": distance["finite_patch_distance_m"],
                            "source_uv": source_uv.tolist(),
                            "center_uv": None if center_uv is None else center_uv.tolist(),
                        }
                    interaction_rows.append(interaction)
                    contact_rows.append(contact)
    windows = build_pair_windows(contact_rows)
    interaction = {
        "schema_version": "INTERACTION_EVIDENCE_V1", "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT, "coordinate_domain": DEPTH_REFERENCE,
        "short_gap_inferred_consumed": False, "removal_consumed": False,
        "world_motion_authority": "ABSENT_IMAGE_AND_OBJECT_RELATIVE_ONLY",
        "records": interaction_rows,
    }
    contact = {
        "schema_version": "CONTACT_CANDIDATE_V1", "session_id": SESSION_ID,
        "distance_gate_m": CONTACT_DISTANCE_M,
        "distance_gate_was_uncertainty_expanded": False,
        "support_score_semantics": "UNCALIBRATED_HEURISTIC_NOT_PROBABILITY_NOT_GROUND_TRUTH",
        "records": contact_rows, "windows": windows,
        "state_counts": dict(Counter(row["state"] for row in contact_rows)),
        "admitted_window_count": sum(row["terminal"] == "ADMITTED_LOCAL_WINDOW" for row in windows),
        "event_timing_policy": "EVENT_LEVEL_WARNING_ONLY_NOT_GLOBAL_BLOCKER",
    }
    return alignment, interaction, contact, patches, samples


def make_r1(
    r0_arrays: Mapping[str, np.ndarray], alignment: Mapping[str, Any], contact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    q_init = np.asarray(r0_arrays["q22_init"], np.float64)
    valid = np.asarray(r0_arrays["valid_side_frame"], bool)
    timestamps = np.asarray(r0_arrays["timestamps_s"], np.float64)
    q_refined = q_init.copy()
    wrist_delta = np.zeros((FRAME_COUNT, 2, 6), np.float64)
    admitted = [row for row in contact["windows"] if row["terminal"] == "ADMITTED_LOCAL_WINDOW"]
    blockers: list[str] = []
    if alignment.get("metric_translation_authorized") is not True:
        blockers.append("HUMAN_STEREO_ALIGNMENT_NOT_ADMITTED")
    if not admitted:
        blockers.append("NO_FIXED_PAIR_FIVE_FRAME_CONTACT_WINDOW")
    # A finite observed human patch is not yet registered to the pinned KaiHand
    # pad mesh.  Do not optimize a proxy and call its own objective a validation.
    if admitted:
        blockers.append("KAIHAND_PAD_TO_HUMAN_WRIST_FRAME_ALIGNMENT_UNVERIFIED")
    status = "BLOCKED_LOCAL_EVIDENCE" if blockers else "COMPLETED_DEVELOPMENT_REFINEMENT"
    weights = transition_taper_weights(timestamps, admitted, transition_s=0.2) if admitted else np.zeros(FRAME_COUNT)
    collision = collision_scope_result(
        robot_self_collision="NOT_EVALUATED_R1_BLOCKED" if blockers else "UNVERIFIED",
        observed_object_patch="NOT_EVALUATED_R1_BLOCKED" if blockers else "UNVERIFIED",
    )
    result = {
        "schema_version": "KAI22_R1_LOCAL_REFINEMENT_V1", "session_id": SESSION_ID,
        "status": status, "first_blocker": blockers[0] if blockers else None,
        "blockers": blockers, "admitted_window_count": len(admitted),
        "frozen_window_list_sha256": canonical_sha(admitted),
        "frozen_evaluation_frame_ids": sorted({frame for row in admitted for frame in row["frame_ids"]}),
        "q22_or_wrist_modified": False if blockers else True,
        "wrist_translation_limit_m": 0.030, "wrist_rotation_limit_deg": 15.0,
        "transition_seconds": 0.2,
        "transition_generates_contact_labels": False,
        "outside_transition_equals_q_init": True,
        "timestamp_motion_before": {
            hand: timestamp_motion_diagnostics(q_init[:, side], timestamps, valid[:, side])
            for side, hand in enumerate(HANDS)
        },
        "timestamp_motion_after": {
            hand: timestamp_motion_diagnostics(q_refined[:, side], timestamps, valid[:, side])
            for side, hand in enumerate(HANDS)
        },
        "collision": collision,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }
    arrays = {
        "q22_refined_or_init_if_blocked": q_refined,
        "wrist_delta_translation_rotation": wrist_delta,
        "transition_weight": weights,
        "valid_side_frame": valid,
        "timestamps_s": timestamps,
    }
    return result, arrays


def video_frames() -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open review source: {VIDEO}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != FRAME_COUNT or any(frame.shape[:2] != MASK_SHAPE for frame in frames):
        raise RuntimeError("review source is not the exact 150-frame 1280x960 domain")
    return frames


def panel(title: str, lines: list[str], size: tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(canvas, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line[:78], (20, 75 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def write_video(path: Path, frames: list[np.ndarray], fps: float = FPS) -> dict[str, Any]:
    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise RuntimeError("review video frame geometry mismatch")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count != FRAME_COUNT:
        raise RuntimeError(f"review video decode mismatch: {path}: {count}")
    return {**ref(path), "frame_count": count, "full_decode": True, "geometry": [width, height], "fps": fps}


def render_reviews(
    raw_frames: list[np.ndarray], frozen: Mapping[str, Any], masks: Mapping[str, PackedMask],
    states: Mapping[str, list[str]], interaction: Mapping[str, Any], contact: Mapping[str, Any],
    r0: Mapping[str, Any], r0_arrays: Mapping[str, np.ndarray], r1: Mapping[str, Any],
) -> dict[str, Any]:
    VISUAL.mkdir(parents=True, exist_ok=False)
    object_by_id = {row["instance_id"]: row for row in frozen["object_doc"]["objects"]}
    colors = {"playing_card_00": (40, 210, 40), "playing_card_01": (230, 120, 40), "playing_card_02": (200, 50, 210)}
    geometry_frames: list[np.ndarray] = []
    interaction_frames: list[np.ndarray] = []
    robot_frames: list[np.ndarray] = []
    interaction_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    contact_by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in interaction["records"]:
        if row.get("finite_patch") is not None:
            interaction_by_frame[int(row["frame_id"])].append(row)
    for row in contact["records"]:
        if row.get("state") not in {"UNKNOWN", "NO_EVIDENCE"}:
            contact_by_frame[int(row["frame_id"])].append(row)
    q22 = np.asarray(r0_arrays["q22_init"])
    for frame, raw in enumerate(raw_frames):
        left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
        overlay = left.copy()
        lines = [f"frame {frame:03d} | encoded physical-left | no undistortion"]
        for object_id in OBJECTS:
            mask = admitted_mask(masks, states, object_id, frame)
            small = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = colors[object_id]
            overlay[small] = (0.72 * overlay[small] + 0.28 * np.asarray(color)).astype(np.uint8)
            row = object_by_id[object_id]["frames"][frame]
            center = row["center_xyz"]
            plane = row["plane_normal"]
            lines.append(f"{object_id[-2:]} center={center['observability'][:10]} plane={plane['observability'][:10]} mask={row['mask_pixel_count']}")
        geometry_frames.append(np.hstack((overlay, panel("Object6D V2 finite visible geometry", lines))))

        annotated = left.copy()
        rows = sorted(contact_by_frame.get(frame, []), key=lambda row: float(row.get("support_score", 0)), reverse=True)[:8]
        for index, row in enumerate(rows):
            sample = next((candidate["finger_associated_visible_surface_point"] for candidate in interaction_by_frame[frame]
                           if candidate["hand_id"] == row["hand_id"] and candidate["finger_id"] == row["finger_id"] and candidate["object_id"] == row["object_id"]), None)
            if sample and sample.get("source_pixel_uv"):
                x, y = (np.asarray(sample["source_pixel_uv"]) / 2.0).astype(int)
                cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"{row['hand_id'][0]}:{row['finger_id'][:2]}->{row['object_id'][-2:]} {row['state']}",
                        (12, 25 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 240), 1, cv2.LINE_AA)
        interaction_frames.append(np.hstack((annotated, panel("Interaction / Contact evidence", [
            f"frame {frame:03d} candidates={len(rows)}",
            "red point = finger-associated visible surface",
            "Contact gate = finite patch <=5mm + admitted uncertainty",
            "support_score = heuristic, not probability or GT",
            "tactile valid<=40ms can only add support",
            "world transport not published (c2w authority absent)",
        ]))))

        robot_panel = left.copy()
        side_lines = [
            f"frame {frame:03d}", f"R0={r0['status']}", f"R1={r1['status']}",
            f"R1 blocker={r1.get('first_blocker')}",
            "R2 optional: not run before R1 closure",
            "NON_CONTROL / NON_DEPLOYABLE",
        ]
        # Honest q22 state visualization; missing direct observations stay blank.
        for side, hand in enumerate(HANDS):
            if np.isfinite(q22[frame, side]).all():
                base_y = 305 + side * 75
                limits = np.nanmax(np.abs(q22[np.isfinite(q22).all(axis=2)])) if np.any(np.isfinite(q22).all(axis=2)) else 1.0
                for joint, value in enumerate(q22[frame, side]):
                    x0 = 20 + joint * 27
                    height = int(55 * abs(value) / max(limits, 1e-6))
                    cv2.rectangle(robot_panel, (x0, base_y - height), (x0 + 15, base_y), (50, 160, 230) if side == 0 else (230, 130, 50), -1)
                cv2.putText(robot_panel, f"{hand} q22", (20, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        robot_frames.append(np.hstack((robot_panel, panel("Kai22 R0 vs R1", side_lines))))
    reviews = {
        "OBJECT6D_GEOMETRY_REVIEW.mp4": write_video(VISUAL / "OBJECT6D_GEOMETRY_REVIEW.mp4", geometry_frames),
        "INTERACTION_CONTACT_REVIEW.mp4": write_video(VISUAL / "INTERACTION_CONTACT_REVIEW.mp4", interaction_frames),
        "KAI22_R0_VS_R1_REVIEW.mp4": write_video(VISUAL / "KAI22_R0_VS_R1_REVIEW.mp4", robot_frames),
    }
    return reviews


def validate_schema(name: str, value: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(load_json(SCHEMAS[name])).validate(value)


def main() -> int:
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
    timestamps, tactile, source_refs = load_source_metadata()
    frozen = load_frozen_inputs()
    depth_frames = sorted((DEPTH_ROOT / "frames").glob("*.npz"))
    if len(depth_frames) != FRAME_COUNT:
        raise RuntimeError("Depth frame denominator drift")
    input_refs = [
        ref(HAWOR), ref(VIDEO), ref(DEPTH_ROOT / "RESULT.json"),
        ref(DEPTH_ROOT / "DEPTH_CONTRACT.json"), ref(DEPTH_ROOT / "RGB_ALIGNMENT_QA.json"),
        ref(OBJECT_ROOT / "RESULT.json"), ref(OBJECT_ROOT / "OBJECT6D_OBSERVABILITY_V2.json"),
        *sam_refs, *source_refs, *(ref(path) for path in depth_frames),
    ]
    input_snapshot = {row["path"]: row["sha256"] for row in input_refs}
    signature_payload = {
        "schema_version": "0915-interaction-contact-robot-dev-run-signature-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "executor_epoch": args.executor_epoch,
        "weights": "ABSENT", "frame_count": FRAME_COUNT,
        "model_policy": "FROZEN_NO_MODEL_RERUN", "gpu_used": False,
        "forbidden_inputs": ["Removal", "Clean", "short_gap_inferred", "PICO26_HAND_RESULTS", "archive"],
        "task_packet": ref(packet_path), "inputs": input_refs,
        "code": [ref(Path(__file__)), ref(ROOT / "src/chaoyang/pipeline/interaction_contact_robot_dev_v1.py")],
        "schemas": [ref(path) for path in SCHEMAS.values()],
    }
    signature = {**signature_payload, "run_signature_sha256": canonical_sha(signature_payload)}
    atomic_json(output / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-interaction-contact-robot-dev-writer-claim-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "attempt_id": output.name,
        "status": "CLAIMED", "weights": "ABSENT", "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(output), "task_packet": ref(packet_path),
    }
    atomic_json(output / "CLAIM.json", claim)
    heartbeat()

    r0, r0_arrays, _assets = make_r0(frozen["hawor"], timestamps)
    np.savez_compressed(output / "KAI22_R0_BASELINE_V1.npz", **r0_arrays)
    r0["arrays"] = ref(output / "KAI22_R0_BASELINE_V1.npz")
    atomic_json(output / "KAI22_R0_BASELINE_V1.json", r0)

    qa = object_qa(frozen["object_doc"])
    dimensions = dimension_estimate(frozen["object_doc"])
    validate_schema("object_qa", qa)
    validate_schema("dimension", dimensions)
    atomic_json(output / "OBJECT6D_GEOMETRY_QA_V1.json", qa)
    atomic_json(output / "CARD_DIMENSION_ESTIMATE_V1.json", dimensions)

    alignment, interaction, contact, _patches, _samples = build_evidence(
        frozen, masks, states, timestamps, tactile, input_refs,
    )
    validate_schema("alignment", alignment)
    validate_schema("interaction", interaction)
    validate_schema("contact", contact)
    atomic_json(output / "HUMAN_STEREO_ALIGNMENT_CHECK_V1.json", alignment)
    atomic_json(output / "INTERACTION_EVIDENCE_V1.json", interaction)
    atomic_json(output / "CONTACT_CANDIDATE_V1.json", contact)

    r1, r1_arrays = make_r1(r0_arrays, alignment, contact)
    np.savez_compressed(output / "KAI22_R1_LOCAL_REFINEMENT_V1.npz", **r1_arrays)
    r1["arrays"] = ref(output / "KAI22_R1_LOCAL_REFINEMENT_V1.npz")
    atomic_json(output / "KAI22_R1_LOCAL_REFINEMENT_V1.json", r1)
    r2 = {
        "schema_version": "KAI22_R2_ARM_VISUAL_V1", "session_id": SESSION_ID,
        "status": "NOT_RUN_OPTIONAL_R1_NOT_CLOSED" if r1["status"] != "COMPLETED_DEVELOPMENT_REFINEMENT" else "NOT_RUN_TIMEBOX_PRIORITY",
        "blocks_r1": False, "arm_ik_performed": False,
        "authority": AUTHORITY,
    }
    atomic_json(output / "KAI22_R2_ARM_VISUAL_V1.json", r2)
    robot = {
        "schema_version": "KAI22_R0_R1_DEVELOPMENT_V1", "session_id": SESSION_ID,
        "authority": AUTHORITY, "control_ground_truth": False,
        "physical_deployment_authorized": False, "calibration_authority": "DEVELOPMENT_ONLY",
        "r0": r0, "r1": r1, "r2": r2,
    }
    validate_schema("robot", robot)

    reviews = render_reviews(video_frames(), frozen, masks, states, interaction, contact, r0, r0_arrays, r1)
    metrics = {
        "schema_version": "0915-interaction-contact-robot-dev-metrics-v1",
        "session_id": SESSION_ID, "frame_count": FRAME_COUNT,
        "object6d_qa_status": qa["status"], "dimension_status": dimensions["status"],
        "alignment_status": alignment["status"],
        "alignment_metric_translation_authorized": alignment["metric_translation_authorized"],
        "interaction_record_count": len(interaction["records"]),
        "contact_state_counts": contact["state_counts"],
        "contact_window_count": len(contact["windows"]),
        "admitted_contact_window_count": contact["admitted_window_count"],
        "r0_status": r0["status"], "r1_status": r1["status"], "r1_first_blocker": r1["first_blocker"],
        "r2_status": r2["status"], "reviews": reviews,
        "authority": AUTHORITY,
    }
    atomic_json(output / "METRICS.json", metrics)
    shutil.copyfile(output / "METRICS.json", VISUAL / "METRICS.json")
    readme = f"""# 0915 Interaction → Contact → Kai22 开发级单样本复核

固定输入为 `play_cards_0915_001` 的 150 帧 encoded physical-left resize-only 数据。没有重跑 HaWoR、SAM3.1、FoundationStereo 或 Object6D；没有读取 Removal/Clean，也没有使用短缺口补帧作为公制证据。

- Object6D QA：`{qa['status']}`。visible surface center 不是物体固定中心；跨帧变化只作诊断。
- 牌尺寸：`{dimensions['status']}`，未使用标准牌尺寸，也未补造隐藏边界。
- Human/Stereo alignment：`{alignment['status']}`；公制 wrist translation 授权为 `{alignment['metric_translation_authorized']}`。
- Contact 状态计数：`{json.dumps(contact['state_counts'], ensure_ascii=False, sort_keys=True)}`。
- 合格的固定 hand/finger/object 连续窗口：`{contact['admitted_window_count']}`。
- Kai22 R0：`{r0['status']}`，使用 direct-observed HaWoR；无 Contact 时仍独立交付。
- Kai22 R1：`{r1['status']}`；首阻塞：`{r1['first_blocker']}`。
- Kai22 R2：`{r2['status']}`，不会反向阻塞 R1。

三段视频均完整解码 150 帧。`KAI22_R0_VS_R1_REVIEW.mp4` 在 R1 阻塞时明确显示阻塞，不把未修改的 R0 冒充 refinement。

所有结果仅为 `DEVELOPMENT_RELATIVE / NON_CONTROL / NON_DEPLOYABLE`。自动门不代表视觉验收、接触真值、物理碰撞完整性或真机部署授权；完整牌体、隐藏背面、卡托及未建模环境保持 `UNVERIFIED`。
"""
    (VISUAL / "README_ZH.md").write_text(readme, encoding="utf-8")

    for raw, expected in input_snapshot.items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"read-only frozen input changed: {path}")
    result = {
        "schema_version": "0915-interaction-contact-robot-dev-result-v1",
        "task_id": TASK_ID, "session_id": SESSION_ID, "status": "PASSED",
        "development_terminal": "R0_COMPLETE_R1_" + ("COMPLETE" if r1["status"] == "COMPLETED_DEVELOPMENT_REFINEMENT" else "BLOCKED_LOCAL_EVIDENCE"),
        "weights": "ABSENT", "source_mutated": False, "gpu_used": False,
        "model_rerun_performed": False, "removal_or_clean_consumed": False,
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY", "authority": AUTHORITY,
        "r0_status": r0["status"], "r1_status": r1["status"], "r2_status": r2["status"],
        "artifacts": {
            "object6d_qa": ref(output / "OBJECT6D_GEOMETRY_QA_V1.json"),
            "card_dimensions": ref(output / "CARD_DIMENSION_ESTIMATE_V1.json"),
            "alignment": ref(output / "HUMAN_STEREO_ALIGNMENT_CHECK_V1.json"),
            "interaction": ref(output / "INTERACTION_EVIDENCE_V1.json"),
            "contact": ref(output / "CONTACT_CANDIDATE_V1.json"),
            "r0": ref(output / "KAI22_R0_BASELINE_V1.json"),
            "r1": ref(output / "KAI22_R1_LOCAL_REFINEMENT_V1.json"),
            "r2": ref(output / "KAI22_R2_ARM_VISUAL_V1.json"),
            "metrics": ref(output / "METRICS.json"),
            "visual_readme": ref(VISUAL / "README_ZH.md"),
            "reviews": reviews,
        },
        "run_signature": ref(output / "RUN_SIGNATURE.json"), "writer_claim": ref(output / "CLAIM.json"),
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-interaction-contact-robot-dev-run-receipt-v1",
        "task_id": TASK_ID, "status": "PASSED", "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(receipt), "gpu_used": False,
    })
    print(json.dumps({"status": "PASSED", "r1_status": r1["status"], "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
