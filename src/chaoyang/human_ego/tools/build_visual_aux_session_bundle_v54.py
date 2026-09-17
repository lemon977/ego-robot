#!/usr/bin/env python3
"""Build one paired future-2D Visual Aux session bundle.

The Robot review is development-only.  Object/Robot overlap is preserved from
Clean and excluded from *both* branches with one shared training mask.  This is
therefore a fail-closed Visual Aux input, not occlusion/contact authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))


import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

# Keep this CPU-only data builder below the active Clean/Robot executors.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.human_ego.tools.build_visual_retarget_projection_candidate_v52 import (  # noqa: E402
    future_arrays,
    project,
)
from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import validate_bundle  # noqa: E402
from chaoyang.ops import render_poker_same_side_outward_frame0 as shared  # noqa: E402
from chaoyang.ops import render_poker_symmetric_chirality_flange_successor as old  # noqa: E402
from chaoyang.ops import render_poker_v2c03_p3_naturalv2_successor as fixed  # noqa: E402


WIDTH, HEIGHT = 640, 480
LABEL_SCHEMA = "exact78-visual-aux-labels-v52-v1"
MIN_FORMAL_VALID_PIXEL_FRACTION = 0.70
SILVER_READY_STATUS = "VISUAL_OCCLUSION_SILVER_READY"
COMPOSITOR_READY_STATUS = "PASS_CAUSAL_TRAINING_COMPOSITOR"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ref(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    display = str(path.relative_to(relative_to.resolve())) if relative_to is not None else str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact(item: dict[str, Any], label: str) -> Path:
    path = Path(item["path"]).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}: ordinary file required")
    if path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256"):
        raise RuntimeError(f"{label}: byte/SHA mismatch")
    return path


def exact_from(root: Path, item: dict[str, Any], label: str) -> Path:
    path = Path(item["path"])
    if not path.is_absolute():
        path = root / path
    return exact({**item, "path": str(path)}, label)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _same_artifact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "bytes", "sha256"))


def validate_silver_compositor_receipts(
    *,
    silver_path: Path,
    compositor_path: Path,
    task: str,
    session: str,
    frame_count: int,
    robot_ref: dict[str, Any],
    clean_ref: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind a causal compositor pass to one fail-closed Silver result.

    This is intentionally stricter than merely seeing a ``READY`` string.  It
    prevents a bundle from silently upgrading the older overlap-exclusion
    development path into a formal training input.
    """

    silver_path = silver_path.resolve(strict=True)
    compositor_path = compositor_path.resolve(strict=True)
    silver_ref = ref(silver_path)
    compositor_ref = ref(compositor_path)
    silver = load(silver_path)
    if silver.get("schema_version") != "OCCLUSION_SILVER_R3":
        raise RuntimeError("formal OCCLUSION_SILVER_R3 receipt required")
    if silver.get("status") != SILVER_READY_STATUS or silver.get("authority_level") != "SILVER":
        raise RuntimeError("Occlusion Silver receipt is not ready")
    if (
        silver.get("task") != task
        or silver.get("session") != session
        or silver.get("frame_count") != frame_count
    ):
        raise RuntimeError("Occlusion Silver task/session/frame identity mismatch")
    if silver.get("accuracy_reported") is not False or silver.get("external_accuracy") != "UNKNOWN":
        raise RuntimeError("Silver receipt must not claim external accuracy")
    numeric_gates = {
        "known_decision_coverage": lambda value: value >= 0.70,
        "unknown_pixel_ratio": lambda value: value <= 0.30,
        "unknown_contact_frame_ratio": lambda value: value <= 0.20,
        "max_unknown_run": lambda value: value <= 5,
        "protected_retention": lambda value: value >= 0.99,
        "pixel_provenance_coverage": lambda value: value >= 0.70,
    }
    for key, predicate in numeric_gates.items():
        value = silver.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not predicate(value):
            raise RuntimeError(f"Occlusion Silver quality gate failed: {key}")
    for key in (
        "temporal_consistency_pass",
        "zbuffer_consistency_pass",
        "byte_exact_outside_authorized_band",
        "causal_donor_pass",
    ):
        if silver.get(key) is not True:
            raise RuntimeError(f"Occlusion Silver quality gate failed: {key}")

    compositor = load(compositor_path)
    if compositor.get("schema_version") != "ROBOTIZED_COMPOSITOR_CAUSAL_V1_RESULT":
        raise RuntimeError("formal causal compositor receipt required")
    if compositor.get("status") != COMPOSITOR_READY_STATUS:
        raise RuntimeError("causal compositor receipt is not ready")
    if (
        compositor.get("task") != task
        or compositor.get("session") != session
        or compositor.get("frame_count") != frame_count
    ):
        raise RuntimeError("compositor task/session/frame identity mismatch")
    if (
        compositor.get("input_mode") != "CAUSAL_TRAINING_INPUT"
        or compositor.get("training_authorized") is not True
        or compositor.get("pixel_source_gate_pass") is not True
        or compositor.get("control_ground_truth") is not False
    ):
        raise RuntimeError("compositor is not a causal, pixel-source-gated training result")
    bound_silver = compositor.get("occlusion_silver")
    if not isinstance(bound_silver, dict) or not _same_artifact(bound_silver, silver_ref):
        raise RuntimeError("compositor does not bind the exact Silver receipt")
    inputs = compositor.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("compositor input receipt map is required")
    if not isinstance(inputs.get("robot_result"), dict) or not _same_artifact(
        inputs["robot_result"], robot_ref
    ):
        raise RuntimeError("compositor Robot result binding mismatch")
    if not isinstance(inputs.get("clean_result"), dict) or not _same_artifact(
        inputs["clean_result"], clean_ref
    ):
        raise RuntimeError("compositor Clean result binding mismatch")
    # Resolve every claimed reference after comparing it.  Matching strings are
    # insufficient if the underlying file has subsequently changed.
    exact(bound_silver, "compositor-bound Silver receipt")
    exact(inputs["robot_result"], "compositor-bound Robot result")
    exact(inputs["clean_result"], "compositor-bound Clean result")
    return {"occlusion_silver_result": silver_ref, "compositor_result": compositor_ref}


def bundle_producer_signature(
    *,
    robot_ref: dict[str, Any],
    clean_ref: dict[str, Any],
    silver_ref: dict[str, Any],
    compositor_ref: dict[str, Any],
    split: str,
    h50_eligibility_mode: str,
    minimum_valid_pixel_fraction: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": "VISUAL_AUX_BUNDLE_PRODUCER_SIGNATURE_V1",
        "builder_code": ref(Path(__file__)),
        "robot_result": robot_ref,
        "clean_result": clean_ref,
        "occlusion_silver_result": silver_ref,
        "compositor_result": compositor_ref,
        "split": split,
        "seed": 7,
        "h50_eligibility_mode": h50_eligibility_mode,
        "minimum_valid_pixel_fraction": minimum_valid_pixel_fraction,
    }
    return {"payload": payload, "sha256": canonical_sha256(payload)}


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def object_union(row: dict[str, Any], source_shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(source_shape, dtype=bool)
    candidates = [
        value
        for key, value in row.items()
        if isinstance(value, dict)
        and (key.startswith("physical_object_") or key.startswith("task_object"))
        and "path" in value
    ]
    for item in candidates:
        path = exact(item, "task-object mask")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != source_shape:
            raise RuntimeError("task-object mask decode/shape mismatch")
        output |= image > 0
    return output


def write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"PNG write failed: {path}")


def scale_uv_to_training_domain(
    uv: np.ndarray, source_width: int, source_height: int
) -> np.ndarray:
    """Map pixel-centre coordinates without moving the last source pixel out of bounds."""
    output = np.asarray(uv, dtype=np.float32).copy()
    output[..., 0] *= (WIDTH - 1) / max(source_width - 1, 1)
    output[..., 1] *= (HEIGHT - 1) / max(source_height - 1, 1)
    return output


def causal_real_donor_validity(
    source_kind: np.ndarray, source_frame: np.ndarray, target_frame: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep current/past evidence and reject future temporal donors."""

    kind = np.asarray(source_kind)
    donor_frame = np.asarray(source_frame)
    if kind.shape != donor_frame.shape or kind.ndim != 2:
        raise RuntimeError("real-donor source-map shape mismatch")
    causal_donor = (kind == 1) & (donor_frame <= int(target_frame))
    valid = (kind == 0) | (kind == 2) | causal_donor
    return valid, causal_donor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-review-result", type=Path, required=True)
    parser.add_argument("--clean-result", type=Path, required=True)
    parser.add_argument("--occlusion-silver-result", type=Path, required=True)
    parser.add_argument("--compositor-result", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "heldout"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--causal-training", action="store_true")
    parser.add_argument(
        "--h50-eligibility-mode",
        choices=(
            "BOTH_ENDPOINTS_40_OF_50",
            "ANY_ENDPOINT_40_OF_50",
            "ANY_ENDPOINT_TIMESTEP_40_OF_50",
        ),
        default="ANY_ENDPOINT_TIMESTEP_40_OF_50",
    )
    parser.add_argument(
        "--min-valid-pixel-fraction",
        type=float,
        default=MIN_FORMAL_VALID_PIXEL_FRACTION,
    )
    args = parser.parse_args()

    if not MIN_FORMAL_VALID_PIXEL_FRACTION <= args.min_valid_pixel_fraction <= 1.0:
        raise RuntimeError(
            f"min-valid-pixel-fraction must be in [{MIN_FORMAL_VALID_PIXEL_FRACTION},1]"
        )

    robot_path = args.robot_review_result.resolve(strict=True)
    clean_path = args.clean_result.resolve(strict=True)
    robot = load(robot_path)
    clean = load(clean_path)
    numeric_review = robot.get("status") == "PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW"
    pose_only = robot.get("status") == "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY"
    if not (numeric_review or pose_only):
        raise RuntimeError("numeric-pass review or current pose-only Robot result is required")
    if robot.get("authority") is not False or robot.get("action_sidecar_published") is not False:
        raise RuntimeError("Robot result claim boundary mismatch")
    if pose_only and (
        robot.get("control_ground_truth") is not False
        or robot.get("metric_object_geometry") is not False
        or robot.get("contact_frame_valid") is not False
    ):
        raise RuntimeError("pose-only Robot result claim boundary mismatch")
    metric_clean = clean.get("status") == "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B"
    visual_tier_clean = clean.get("status") == "PASS_CAUSAL_REAL_DONOR_VISUAL_TIER_GRADE_B"
    if not metric_clean and not (args.causal_training and visual_tier_clean):
        raise RuntimeError("compatible Grade-B visual Clean result is required")
    if clean.get("downstream_authorized") is not True:
        raise RuntimeError("Clean result is not downstream-authorized")
    task, session = robot["task"], robot["session"]
    frame_count = int(robot["frame_count"])
    if clean.get("task") != task or clean.get("session") != session or clean.get("frame_count") != frame_count:
        raise RuntimeError("Robot/Clean identity mismatch")
    robot_ref = ref(robot_path)
    clean_ref = ref(clean_path)
    receipt_refs = validate_silver_compositor_receipts(
        silver_path=args.occlusion_silver_result,
        compositor_path=args.compositor_result,
        task=task,
        session=session,
        frame_count=frame_count,
        robot_ref=robot_ref,
        clean_ref=clean_ref,
    )
    producer_signature = bundle_producer_signature(
        robot_ref=robot_ref,
        clean_ref=clean_ref,
        silver_ref=receipt_refs["occlusion_silver_result"],
        compositor_ref=receipt_refs["compositor_result"],
        split=args.split,
        h50_eligibility_mode=args.h50_eligibility_mode,
        minimum_valid_pixel_fraction=args.min_valid_pixel_fraction,
    )

    if numeric_review:
        lineage = robot["lineage"]
        hawor_path = exact(lineage["hawor_npz"], "HaWoR temporal")
        arm_path = exact(lineage["arm_states"], "arm states")
        hand_path = exact(lineage["hand_states"], "hand states")
        source_path = exact(lineage["source_video"], "source video")
        hawor, arm, hand = arrays(hawor_path), arrays(arm_path), arrays(hand_path)
        q_arm = np.asarray(arm["q_arm"], dtype=np.float64)
        q_hand = np.asarray(hand["q_hand"], dtype=np.float64)
        valid = np.asarray(arm["valid_side_frame"], dtype=bool)
        if valid.shape != (2, frame_count) or not np.array_equal(valid, hand["valid_side_frame"]):
            raise RuntimeError("arm/hand validity mismatch")
        valid_tf = valid.T
        world_base = np.asarray(arm["T_world_base"], dtype=np.float64)
        mounts = np.asarray(arm["T_tool_hand_root"], dtype=np.float64)
        camera_metadata_source = "NUMERIC_REVIEW_TEMPORAL_HAWOR"
    else:
        pose_root = robot_path.parent
        lineage_path = exact_from(pose_root, robot["outputs"]["lineage"], "pose-only lineage")
        pose_lineage = load(lineage_path)
        inputs = pose_lineage["input_manifest"]
        temporal_ref = inputs["hawor_chain"]["temporal_output"]
        temporal_path = Path(temporal_ref["path"])
        if temporal_path.is_file():
            hawor_path = exact(temporal_ref, "HaWoR temporal")
            camera_metadata_source = "POSE_ONLY_LINEAGE_TEMPORAL_HAWOR"
        else:
            # The Robot trajectory is already frozen in the sidecar.  c2w/K are
            # camera metadata, so the byte-closed bounded HaWoR input is a
            # legitimate fallback when a retired temporal copy was removed.
            hawor_path = exact(
                inputs["hawor_chain"]["frozen_npz"], "HaWoR camera metadata"
            )
            camera_metadata_source = "POSE_ONLY_LINEAGE_FROZEN_HAWOR_CAMERA_METADATA_ONLY"
        source_path = exact(inputs["source_video"], "source video")
        sidecar_path = exact_from(
            pose_root, robot["outputs"]["trajectory_sidecar"], "visual Robot trajectory sidecar"
        )
        hawor, sidecar = arrays(hawor_path), arrays(sidecar_path)
        if bool(np.asarray(sidecar["control_ground_truth"]).item()):
            raise RuntimeError("visual Robot trajectory cannot be control ground truth")
        q_arm = np.asarray(sidecar["q_arm_rad"], dtype=np.float64)
        q_hand = np.asarray(sidecar["q_hand_rad"], dtype=np.float64)
        valid_tf = np.asarray(sidecar["valid_side_frame"], dtype=bool)
        if valid_tf.shape != (frame_count, 2):
            raise RuntimeError("pose-only side validity shape mismatch")
        world_base = np.asarray(sidecar["T_world_base"], dtype=np.float64)
        mounts = np.asarray(sidecar["T_tool_hand_root"], dtype=np.float64)
    clean_video = (
        exact(clean["artifacts"]["clean_synthetic_master"], "Clean master")
        if not args.causal_training else None
    )
    donor_manifest = None
    if args.causal_training:
        donor_manifest_path = exact(
            clean["inputs"]["real_donor_source_manifest"], "real donor source manifest"
        )
        donor_manifest = load(donor_manifest_path)
        if donor_manifest.get("session") != session or len(donor_manifest.get("frames", [])) != frame_count:
            raise RuntimeError("real donor manifest identity mismatch")
    mask_manifest_path = exact(clean["inputs"]["mask_frame_manifest"], "Mask frame manifest")
    mask_manifest = load(mask_manifest_path)
    if mask_manifest.get("session") != session or len(mask_manifest.get("frames", [])) != frame_count:
        raise RuntimeError("Mask manifest identity mismatch")

    frames = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
    if not np.array_equal(frames, np.arange(frame_count, dtype=np.int64)):
        raise RuntimeError("full contiguous frame identity required")
    if world_base.ndim == 3:
        if not np.allclose(world_base, world_base[0], atol=1e-12, rtol=0):
            raise RuntimeError("Robot base moved within session")
        world_base = world_base[0]
    assets = old.load_pinned_robot_assets(PROJECT)
    raster = old.load_module(old.RASTER_SOURCE, f"visual_aux_v54_{session}")
    flange = fixed.naturalv2_local_triangles()
    cache: dict[Path, object] = {}

    destination = args.output_root.resolve() / session
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"immutable output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{session}.partial.", dir=destination.parent))
    raw_root, robotized_root = staging / "human_raw_rgb", staging / "robotized_rgb"
    raw_root.mkdir()
    robotized_root.mkdir()
    raw_selector_rows, robot_selector_rows = [], []
    shared_valid = np.ones((frame_count, HEIGHT, WIDTH), dtype=bool)
    hand_roots_world = np.full((frame_count, 2, 4, 4), np.nan, dtype=np.float64)
    source_capture = cv2.VideoCapture(str(source_path))
    clean_capture = None if args.causal_training else cv2.VideoCapture(str(clean_video))
    if not source_capture.isOpened() or (clean_capture is not None and not clean_capture.isOpened()):
        raise RuntimeError("source/Clean video decode open failed")
    overlap_pixels = 0
    future_donor_pixels_rejected = 0
    unsupported_pixels = 0
    try:
        for frame in range(frame_count):
            ok_raw, raw_native = source_capture.read()
            if not ok_raw or raw_native is None:
                raise RuntimeError(f"source RGB missing frame {frame}")
            if args.causal_training:
                donor_row = donor_manifest["frames"][frame]
                donor_native = cv2.imread(
                    str(exact(donor_row["clean_rgb"], "real donor clean RGB")), cv2.IMREAD_COLOR
                )
                source_map_path = exact(donor_row["pixel_source_map"], "real donor source map")
                with np.load(source_map_path, allow_pickle=False) as source_map:
                    kind = source_map["source_kind"].astype(np.uint8)
                    donor_frame = source_map["source_frame"].astype(np.int64)
                if donor_native is None or donor_native.shape != raw_native.shape:
                    raise RuntimeError("real donor RGB decode/shape mismatch")
                causal_native_valid, causal_donor = causal_real_donor_validity(kind, donor_frame, frame)
                clean_native = raw_native.copy()
                clean_native[causal_donor] = donor_native[causal_donor]
                future_donor_pixels_rejected += int(((kind == 1) & (donor_frame > frame)).sum())
                unsupported_pixels += int((kind == 3).sum())
            else:
                ok_clean, clean_native = clean_capture.read()
                causal_native_valid = np.ones(raw_native.shape[:2], dtype=bool)
                if not ok_clean or clean_native is None:
                    raise RuntimeError(f"Clean missing frame {frame}")
            if clean_native is None:
                raise RuntimeError(f"source/Clean missing frame {frame}")
            source_h, source_w = raw_native.shape[:2]
            raw_rgb = cv2.resize(raw_native, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            clean_rgb = cv2.resize(clean_native, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            shared_valid[frame] &= cv2.resize(
                causal_native_valid.astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            intrinsics = np.asarray(hawor["intrinsics"][frame], dtype=np.float64).copy()
            intrinsics[0] *= WIDTH / source_w
            intrinsics[1] *= HEIGHT / source_h
            camera_base = np.linalg.inv(hawor["c2w"][frame]) @ world_base
            if valid_tf[frame].all():
                robot_rgb, robot_label, _ = shared.render_robot(
                    raster,
                    assets,
                    q_arm[frame],
                    q_hand[frame],
                    mounts,
                    camera_base,
                    intrinsics,
                    cache,
                    flange,
                    complete_robot=False,
                )
                robot_mask = robot_label >= 0
            else:
                robot_rgb = np.zeros_like(clean_rgb)
                robot_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
                shared_valid[frame] = False
            union_native = object_union(mask_manifest["frames"][frame], (source_h, source_w))
            object_mask = cv2.resize(
                union_native.astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            contested = cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            contested &= cv2.dilate(robot_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            shared_valid[frame, contested] = False
            overlap_pixels += int(contested.sum())
            robotized = clean_rgb.copy()
            robotized[robot_mask & ~contested] = robot_rgb[robot_mask & ~contested]
            shared_valid[frame, robot_mask & ~contested] = True

            raw_file = raw_root / f"{frame:06d}.png"
            robot_file = robotized_root / f"{frame:06d}.png"
            write_png(raw_file, raw_rgb)
            write_png(robot_file, robotized)
            raw_selector_rows.append({"frame_id": frame, "rgb": ref(raw_file, staging)})
            robot_selector_rows.append({"frame_id": frame, "rgb": ref(robot_file, staging)})

            values = {name: 0.0 for names in old.official.ARM_JOINT_NAMES for name in names}
            for side in range(2):
                if not valid_tf[frame, side]:
                    continue
                values.update(
                    dict(zip(old.official.ARM_JOINT_NAMES[side], q_arm[frame, side], strict=True))
                )
            fk = old.official.forward_kinematics(assets.tianji, values)
            if valid_tf[frame, 0]:
                hand_roots_world[frame, 0] = world_base @ fk["left_tool"] @ mounts[0]
            if valid_tf[frame, 1]:
                hand_roots_world[frame, 1] = world_base @ fk["right_tool"] @ mounts[1]
    finally:
        source_capture.release()
        if clean_capture is not None:
            clean_capture.release()

    uv, point_valid = project(
        np.asarray(hawor["c2w"], dtype=np.float64),
        np.asarray(hawor["intrinsics"], dtype=np.float64),
        hand_roots_world,
        valid_tf,
        source_w,
        source_h,
    )
    original, future_valid, current_valid, starts = future_arrays(
        uv, point_valid, args.h50_eligibility_mode
    )
    # Store labels in the 640x480 training domain.  Pixel coordinates are an
    # inclusive [0, size-1] domain; using WIDTH/source_width would map the
    # final source pixel slightly beyond 1.0 after normalization.
    original = scale_uv_to_training_domain(original, source_w, source_h)
    normalized = original.copy()
    normalized[..., 0] /= WIDTH - 1
    normalized[..., 1] /= HEIGHT - 1
    valid_pixel_fraction_by_frame = shared_valid.reshape(frame_count, -1).mean(axis=1)
    current_valid &= valid_pixel_fraction_by_frame >= args.min_valid_pixel_fraction
    starts = [frame for frame in starts if bool(current_valid[frame])]

    labels_path = staging / "VISUAL_AUX_LABELS.npz"
    np.savez_compressed(
        labels_path,
        schema_version_utf8=np.frombuffer(LABEL_SCHEMA.encode(), dtype=np.uint8),
        frame_ids=np.arange(frame_count, dtype=np.int64),
        future_2d_xy_original=original.astype(np.float32),
        future_2d_xy_normalized=normalized.astype(np.float32),
        future_2d_valid=future_valid.astype(bool),
        current_frame_valid=current_valid.astype(bool),
        rgb_training_valid_mask=shared_valid,
        label_source_utf8=np.frombuffer(b"VISUAL_RETARGET_PROJECTION", dtype=np.uint8),
        control_ground_truth=np.asarray(False, dtype=bool),
        physical_deployment_authorized=np.asarray(False, dtype=bool),
        image_width=np.asarray(WIDTH, dtype=np.int64),
        image_height=np.asarray(HEIGHT, dtype=np.int64),
        normalization_utf8=np.frombuffer(b"x/(W-1),y/(H-1); range=[0,1]", dtype=np.uint8),
    )
    selectors = {}
    for branch, rows in (("HUMAN_RAW_RGB", raw_selector_rows), ("ROBOTIZED_RGB", robot_selector_rows)):
        selector = staging / f"{branch}_SELECTOR.json"
        selector.write_text(
            json.dumps(
                {"branch": branch, "session_id": session, "frame_ids": list(range(frame_count)), "frames": rows},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        selectors[branch] = selector
    manifest = {
        "schema_version": "exact78-visual-aux-session-bundle-v52-v1",
        "task": task,
        "session_id": session,
        "split": args.split,
        "seed": 7,
        "pred_horizon": 50,
        "h50_eligibility_mode": args.h50_eligibility_mode,
        "frame_count": frame_count,
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "label_source": "VISUAL_RETARGET_PROJECTION",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "only_branch_variable": "RGB_BYTES",
        "producer_signature": producer_signature,
        "input_mode": "CAUSAL_TRAINING_INPUT" if args.causal_training else "OFFLINE_BIDIRECTIONAL_VISUALIZATION",
        "labels": ref(labels_path, staging),
        "labels_sha_shared_by_branches": sha256(labels_path),
        "eligible_h50_starts": starts,
        "branches": {branch: {"selector": ref(path, staging)} for branch, path in selectors.items()},
        "inputs": {
            "robot_result": robot_ref,
            "clean_result": clean_ref,
            **receipt_refs,
        },
        "clean_tier": clean.get("clean_tier", "METRIC_WAVE0"),
        "camera_metadata_source": camera_metadata_source,
        "occlusion_policy": {
            "status": "SILVER_BOUND_CAUSAL_COMPOSITOR_PASS",
            "contested_pixels": overlap_pixels,
            "training_valid_same_for_both_branches": True,
            "chips_instance_policy": "instances loaded independently then unioned only for conservative exclusion, never identity/contact inference",
        },
        "valid_pixel_gate": {
            "minimum_fraction": args.min_valid_pixel_fraction,
            "per_frame_fraction": valid_pixel_fraction_by_frame.tolist(),
            "eligible_frames_meet_minimum": bool(
                current_valid.any()
                and np.all(
                    valid_pixel_fraction_by_frame[current_valid]
                    >= args.min_valid_pixel_fraction
                )
            ),
        },
        "causal_proof": {
            "enabled": bool(args.causal_training),
            "rule": "temporal donor source_frame <= target_frame; ProPainter pixels excluded",
            "future_donor_pixels_rejected": future_donor_pixels_rejected,
            "unsupported_or_propainter_pixels_excluded": unsupported_pixels,
            "raw_robotized_valid_mask_shared": True,
        },
        "claim_limit": "Silver-bound causal Visual Aux bundle; Silver is not Gold accuracy. No Robot action, policy or deployment authority.",
    }
    manifest_path = staging / "VISUAL_AUX_SESSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = validate_bundle(staging)
    eligible = (
        report["eligible_h50_window_count"] > 0
        and report["rgb_valid_pixel_fraction"] >= args.min_valid_pixel_fraction
    )
    result_path = staging / "RESULT.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "exact78-visual-aux-session-bundle-result-v54-v1",
                "status": (
                    "PASS_VISUAL_AUX_BUNDLE_SILVER_BOUND"
                    if eligible
                    else (
                        "BLOCKED_PREREQ_ZERO_H50_WINDOWS"
                        if report["eligible_h50_window_count"] == 0
                        else "BLOCKED_PREREQ_VALID_PIXEL_COVERAGE"
                    )
                ),
                "task": task,
                "session": session,
                "split": args.split,
                "eligible_h50_windows": report["eligible_h50_window_count"],
                "rgb_valid_pixel_fraction": report["rgb_valid_pixel_fraction"],
                "manifest": ref(manifest_path, staging),
                "producer_signature": producer_signature,
                "inputs": manifest["inputs"],
                "claim_limit": manifest["claim_limit"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(staging, destination)
    print(json.dumps({"status": "PASS" if eligible else "BLOCKED_PREREQ", "session": session, "windows": len(starts), "output": str(destination)}))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
