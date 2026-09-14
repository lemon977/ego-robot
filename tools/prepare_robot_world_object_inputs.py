#!/usr/bin/env python3
"""Prepare bounded Robot/Object6D compositor inputs for two 0902 sessions.

This is a CPU-only, fail-closed preparation runner.  It deliberately does not
promote the stereo-constrained role trajectory to formal Object6D authority and
does not consume Clean.  It produces:

* a 24-frame shared-v4 functional-retarget trajectory driven by the explicit
  world-consistent HaWoR artifact;
* analytic cuboid near/far optical-Z and a conservative analytic visibility
  candidate at the compositor's fixed 2x grid;
* task-agnostic per-frame depth and render-request manifests; and
* exact Kai named-pad/cuboid distance, object/hand triangle SAT and
  distinct-finger self-SAT evidence.

The result remains PREPARED/HOLD unless the measured role trajectory itself
satisfies the 0--3 mm named-pad contact gate.  A neutral-background CPU review
is diagnostic only and is never a RobotRGB deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline import robot_wrist_kai_adapter as adapter  # noqa: E402
from pipeline.object_occlusion_geometry import (  # noqa: E402
    ObjectGeometry,
    project_object_depth,
)
from pipeline.robot_contact_geometry import (  # noqa: E402
    select_kaihand_pad_patch,
    triangle_triangle_intersects_sat,
)
from pipeline.robot_renderer_eevee_fullchain import (  # noqa: E402
    load_pinned_robot_assets,
)


CANARY_PATH = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
V2_PATH = PROJECT / "tools/run_newtask_robot_kinematic_successor_v2.py"
V4_PATH = PROJECT / "tools/run_newtask_robot_shared_v4_hand.py"
RASTER_PATH = (
    PROJECT
    / "_run/robot_004_mano21_10frame_cpu_zbuffer_review_20260902_v1/run_zbuffer_review.py"
)
PROTOCOL_PATH = (
    PROJECT
    / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v4/PROTOCOL_V4.json"
)
PBR_PATH = PROJECT / "systems/robot/configs/robot_pbr_palette_004ref_v1.json"
FOUNDATIONSTEREO_AUTHORITY = (
    PROJECT / "systems/foundationstereo/environment_authority.json"
)
FOUNDATIONSTEREO_LOCK = (
    PROJECT / "systems/foundationstereo/local_environment_lock.json"
)
FOUNDATIONSTEREO_LAUNCHER = PROJECT / "tools/foundationstereo_python.sh"
FOUNDATIONSTEREO_CHECKPOINT = (
    PROJECT
    / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
)
ANALYTIC_DEPTH_RUNNER = PROJECT / "pipeline/object_occlusion_geometry.py"

CONFIG = {
    "chips": {
        "session": "get_potato_chips_0902_027",
        "hawor": PROJECT
        / "data/processed/chips/get_potato_chips_0902_027/hawor/20260907_world_consistent_v1/HAWOR_OPTIMIZED_MANO21_WORLD_CONSISTENT.npz",
        "object6d": PROJECT
        / "tasks/chips/runs/object6d/20260907_chips027_stereo_constrained_trajectory_v2/OBJECT6D_STEREO_CONSTRAINED_TRAJECTORY.npz",
        "object6d_result": PROJECT
        / "tasks/chips/runs/object6d/20260907_chips027_stereo_constrained_trajectory_v2/RESULT.json",
        "old_scene": PROJECT
        / "tasks/chips/runs/robot/20260906_get_potato_chips_0902_027_visual_v1/interpolated/VISUAL_FULL799_SCENE.npz",
        "accepted_scene": PROJECT
        / "tasks/chips/runs/robot/20260904_get_potato_chips_0901_004_visual_v1/interpolated/VISUAL_FULL799_SCENE.npz",
        "operated_object_id": "chip_0",
        "output": PROJECT
        / "tasks/chips/runs/robot/20260907_get_potato_chips_0902_027_world_object_prepared_v1",
    },
    "poker": {
        "session": "play_cards_0902_017",
        "hawor": PROJECT
        / "data/processed/poker/play_cards_0902_017/hawor/20260907_world_consistent_v1/HAWOR_OPTIMIZED_MANO21_WORLD_CONSISTENT.npz",
        "object6d": PROJECT
        / "tasks/poker/runs/object6d/20260907_poker017_stereo_constrained_trajectory_v2/OBJECT6D_STEREO_CONSTRAINED_TRAJECTORY.npz",
        "object6d_result": PROJECT
        / "tasks/poker/runs/object6d/20260907_poker017_stereo_constrained_trajectory_v2/RESULT.json",
        "old_scene": PROJECT
        / "tasks/poker/runs/robot/20260906_play_cards_0902_017_visual_v1/interpolated/VISUAL_FULL799_SCENE.npz",
        "accepted_scene": PROJECT
        / "tasks/chips/runs/robot/20260904_get_potato_chips_0901_004_visual_v1/interpolated/VISUAL_FULL799_SCENE.npz",
        "operated_object_id": "card_0",
        "output": PROJECT
        / "tasks/poker/runs/robot/20260907_play_cards_0902_017_world_object_prepared_v1",
    },
}

SIDE_NAMES = ("left", "right")
PAD_SUFFIXES = {
    "thumb_distal": "thumb_link6",
    "index_distal": "index_link4",
    "middle_distal": "middle_link4",
    "ring_distal": "ring_link4",
    "pinky_distal": "pinky_link4",
}
FRAME_COUNT = 24
WIDTH = 320
HEIGHT = 240
SUPERSAMPLE = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_depth_dependency_authority(output: Path, task: str, session: str) -> Path:
    """Bind analytic depth now and the pinned stereo dependency for later Mask."""

    for path in (
        FOUNDATIONSTEREO_AUTHORITY,
        FOUNDATIONSTEREO_LOCK,
        FOUNDATIONSTEREO_LAUNCHER,
        FOUNDATIONSTEREO_CHECKPOINT,
        ANALYTIC_DEPTH_RUNNER,
    ):
        path.resolve(strict=True)
    path = output / "DEPTH_DEPENDENCY_AUTHORITY.json"
    atomic_json(
        path,
        {
            "schema_version": "robot-prepared-depth-dependency-authority-v1",
            "status": "ANALYTIC_OBJECT_DEPTH_READY_SCENE_DEPTH_NOT_RUN",
            "task_id": task,
            "session_id": session,
            "analytic_object_depth": {
                "runner": artifact(ANALYTIC_DEPTH_RUNNER),
                "storage": "OPTICAL_AXIS_CAMERA_Z_M_FLOAT32",
                "geometry": "box_xyz role cuboid from object_dimensions_m",
                "used_in_this_cpu_preparation": True,
            },
            "foundationstereo_scene_depth": {
                "environment_authority": artifact(FOUNDATIONSTEREO_AUTHORITY),
                "environment_lock": artifact(FOUNDATIONSTEREO_LOCK),
                "launcher": artifact(FOUNDATIONSTEREO_LAUNCHER),
                "checkpoint": artifact(FOUNDATIONSTEREO_CHECKPOINT),
                "used_in_this_cpu_preparation": False,
                "gpu_depth_run_started": False,
                "required_future_role": "independent metric scene-occluder depth for visible-object Mask authority",
            },
            "claim_limit": "The project-local FoundationStereo environment/checkpoint is pinned but no GPU depth inference is run here. Analytic object near/far does not by itself establish scene visibility.",
        },
    )
    return path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def box_mesh(size_xyz_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size_xyz_m, dtype=np.float64) / 2.0
    x, y, z = half
    vertices = np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def box_sdf(points_object: np.ndarray, size_xyz_m: np.ndarray) -> np.ndarray:
    q = np.abs(np.asarray(points_object, dtype=np.float64)) - np.asarray(
        size_xyz_m, dtype=np.float64
    ) / 2.0
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(
        np.max(q, axis=-1), 0.0
    )


def world_to_object(points_world: np.ndarray, t_object_world: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points_world, dtype=np.float64) - t_object_world[:3, 3]
    ) @ t_object_world[:3, :3]


def select_window(
    world: np.ndarray,
    valid: np.ndarray,
    t_object_world: np.ndarray,
    dimensions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick a valid contiguous window closest to either MANO21 hand."""

    points = np.transpose(world, (1, 0, 2, 3))
    distances = np.empty((len(valid), 2, 21), dtype=np.float64)
    for frame in range(len(valid)):
        local = world_to_object(points[frame], t_object_world[frame])
        distances[frame] = box_sdf(local, dimensions)
    nearest = distances.min(axis=(1, 2))
    candidates: list[tuple[float, float, int]] = []
    for start in range(len(valid) - FRAME_COUNT + 1):
        if bool(np.all(valid[start : start + FRAME_COUNT])):
            block = nearest[start : start + FRAME_COUNT]
            candidates.append((float(np.median(block)), float(np.min(block)), start))
    if not candidates:
        raise RuntimeError("no contiguous 24-frame valid Object6D window")
    median_distance, minimum_distance, start = min(candidates)
    selected = np.arange(start, start + FRAME_COUNT, dtype=np.int64)
    return selected, {
        "policy": "VALID_CONTIGUOUS24_MINIMIZE_MEDIAN_MANO21_TO_ROLE_CUBOID_SDF",
        "local_frame_start": start,
        "local_frame_stop_inclusive": start + FRAME_COUNT - 1,
        "mano21_surface_distance_median_m": median_distance,
        "mano21_surface_distance_min_m": minimum_distance,
    }


def link_visual(model: Any, suffix: str, cache: dict[Path, tuple[np.ndarray, np.ndarray]]) -> Any:
    matches = [visual for visual in model.visuals if visual.link.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one KaiHand visual for {suffix}, got {len(matches)}")
    if matches[0].mesh_path not in cache:
        raise RuntimeError("named-pad mesh missing from pinned cache")
    return matches[0]


def named_pad_points_world(
    model: Any,
    chirality: str,
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    hand_fk: dict[str, np.ndarray],
    hand_root_world: np.ndarray,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    visual = link_visual(model, PAD_SUFFIXES[name], cache)
    vertices, faces = cache[visual.mesh_path]
    patch = select_kaihand_pad_patch(visual.link, chirality, vertices, faces)
    matrix = hand_root_world @ hand_fk[visual.link] @ visual.origin
    return transform(matrix, patch.face_centroids), {
        "name": name,
        "link": visual.link,
        "rule": "PINNED_KAI_CAD_LINK_LOCAL_ANATOMICAL_PATCH",
        "patch_triangles": int(len(patch.face_indices)),
    }


def target_roots_and_features(
    v4: Any,
    contracts: list[dict[str, Any]],
    world: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, list[list[dict[str, dict[str, np.ndarray]]]]]:
    roots = np.empty((FRAME_COUNT, 2, 4, 4), dtype=np.float64)
    targets = []
    for slot, frame in enumerate(selected):
        frame_features = []
        for physical in range(2):
            human = v4.PHYSICAL_TO_HUMAN[physical]
            points = world[human, frame]
            frame_features.append(v4.human_features(adapter, points, SIDE_NAMES[human]))
            roots[slot, physical] = np.eye(4, dtype=np.float64)
            roots[slot, physical, :3, :3] = (
                adapter.final_v3_mano_palm_basis(
                    points, handedness=SIDE_NAMES[human]
                )
                @ contracts[physical]["basis"].T
            )
            roots[slot, physical, :3, 3] = points[0]
        targets.append(frame_features)
    return roots, targets


def pose_audit(
    canary: Any,
    assets: Any,
    base: np.ndarray,
    q_arm: np.ndarray,
    mounts: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for slot in range(FRAME_COUNT):
        for side in range(2):
            actual = (
                base
                @ official._tool_fk(assets, side, q_arm[slot, side])
                @ mounts[side]
            )
            position_mm, rotation_deg = canary.pose_error(
                actual, targets[slot, side], official
            )
            rows.append(
                {
                    "frame_slot": slot,
                    "physical_side": SIDE_NAMES[side],
                    "position_mm": position_mm,
                    "rotation_deg": rotation_deg,
                    "pass_10mm_5deg": bool(
                        position_mm <= 10.0 and rotation_deg <= 5.0
                    ),
                }
            )
    return {
        "rows": rows,
        "position_mm_max": float(max(row["position_mm"] for row in rows)),
        "position_mm_median": float(np.median([row["position_mm"] for row in rows])),
        "rotation_deg_max": float(max(row["rotation_deg"] for row in rows)),
        "rotation_deg_median": float(
            np.median([row["rotation_deg"] for row in rows])
        ),
        "pass_rows": sum(row["pass_10mm_5deg"] for row in rows),
        "required_rows": len(rows),
        "all_pass": bool(all(row["pass_10mm_5deg"] for row in rows)),
    }


def geometry_and_contact(
    *,
    canary: Any,
    assets: Any,
    contracts: list[dict[str, Any]],
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    base: np.ndarray,
    mounts: np.ndarray,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    object_world: np.ndarray,
    dimensions: np.ndarray,
) -> dict[str, Any]:
    vertices, faces = box_mesh(dimensions)
    rows = []
    for slot in range(FRAME_COUNT):
        placed_object = transform(object_world[slot], vertices)
        object_triangles = placed_object[faces]
        side_rows = []
        pad_rows = []
        for side, contract in enumerate(contracts):
            model = contract["model"]
            hand_fk = official.forward_kinematics(
                model,
                dict(zip(contract["names"], q_hand[slot, side], strict=True)),
            )
            hand_root = (
                base
                @ official._tool_fk(assets, side, q_arm[slot, side])
                @ mounts[side]
            )
            by_link = canary.transformed_triangles(model, hand_fk, cache, hand_root)
            hand_triangles = np.concatenate(tuple(by_link.values()), axis=0)
            intersects, candidates = canary.triangle_pair_intersects(
                hand_triangles, object_triangles, triangle_triangle_intersects_sat
            )
            side_rows.append(
                {
                    "physical_side": SIDE_NAMES[side],
                    "dense_object_hand_sat_pass": not intersects,
                    "intersection": bool(intersects),
                    "hand_triangles": int(len(hand_triangles)),
                    "object_triangles": int(len(object_triangles)),
                    "sat_candidates_tested_until_terminal": int(candidates),
                }
            )
            for pad_name in PAD_SUFFIXES:
                points_world, pad = named_pad_points_world(
                    model,
                    SIDE_NAMES[side],
                    cache,
                    hand_fk,
                    hand_root,
                    pad_name,
                )
                sdf = box_sdf(
                    world_to_object(points_world, object_world[slot]), dimensions
                )
                pad_rows.append(
                    {
                        "physical_side": SIDE_NAMES[side],
                        **pad,
                        "signed_sdf_m_min": float(np.min(sdf)),
                        "signed_sdf_m_median": float(np.median(sdf)),
                        "signed_sdf_m_abs_min": float(np.min(np.abs(sdf))),
                        "penetrating_samples": int(np.count_nonzero(sdf < 0.0)),
                    }
                )
        active = min(
            pad_rows,
            key=lambda row: (
                row["signed_sdf_m_abs_min"],
                row["physical_side"],
                row["name"],
            ),
        )
        contact_pass = bool(
            active["signed_sdf_m_min"] >= 0.0
            and active["signed_sdf_m_min"] <= 0.003
            and all(row["dense_object_hand_sat_pass"] for row in side_rows)
        )
        rows.append(
            {
                "frame_slot": slot,
                "active_pad": active,
                "active_pad_contact_0_to_3mm_pass": contact_pass,
                "sides": side_rows,
            }
        )
    contact_distances = [row["active_pad"]["signed_sdf_m_min"] for row in rows]
    dense_pass = bool(
        all(
            side["dense_object_hand_sat_pass"]
            for row in rows
            for side in row["sides"]
        )
    )
    return {
        "geometry": "ROLE_CUBOID_FROM_OBJECT6D_object_dimensions_m",
        "unit": "m",
        "rows": rows,
        "dense_object_hand_sat_all48_pass": dense_pass,
        "contact_all24_pass": bool(
            all(row["active_pad_contact_0_to_3mm_pass"] for row in rows)
        ),
        "active_pad_signed_sdf_m_min": float(np.min(contact_distances)),
        "active_pad_signed_sdf_m_median": float(np.median(contact_distances)),
        "active_pad_signed_sdf_m_max": float(np.max(contact_distances)),
        "contact_target_m": [0.0, 0.003],
        "method": "PINNED_NAMED_PAD_TO_ANALYTIC_BOX_SDF_PLUS_AABB_BROADPHASE_EXACT_TRIANGLE_SAT",
    }


def render_review(
    *,
    raster: Any,
    canary: Any,
    assets: Any,
    contracts: list[dict[str, Any]],
    base: np.ndarray,
    mounts: np.ndarray,
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    c2w: np.ndarray,
    camera_k: np.ndarray,
    object_depth_entries: list[dict[str, Any]],
    frame_ids: np.ndarray,
    session: str,
    output: Path,
) -> Path:
    """Render a small neutral-background diagnostic; no Clean is read."""

    video_path = output / "ROBOT_PREPARED_HOLD_NEUTRAL24_NONFINAL.mp4"
    temporary = output / "_review_frames"
    temporary.mkdir()
    visual_cache: dict[Path, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    high_width, high_height = WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE
    for slot in range(FRAME_COUNT):
        k = camera_k[slot]
        k2 = k.copy()
        k2[0, 0] *= SUPERSAMPLE
        k2[1, 1] *= SUPERSAMPLE
        k2[0, 2] = (k[0, 2] + 0.5) * SUPERSAMPLE - 0.5
        k2[1, 2] = (k[1, 2] + 0.5) * SUPERSAMPLE - 0.5
        camera_base = np.linalg.inv(c2w[slot]) @ base
        values = {
            name: 0.0 for names in official.ARM_JOINT_NAMES for name in names
        }
        for side in range(2):
            values.update(
                {
                    name: float(value)
                    for name, value in zip(
                        official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True
                    )
                }
            )
        arm_fk = official.forward_kinematics(assets.tianji, values)
        geometry = []
        for side, suffix in ((0, "L"), (1, "R")):
            tool = "left_tool" if side == 0 else "right_tool"
            arm = raster.camera_triangles(
                assets.tianji,
                arm_fk,
                camera_base,
                lambda link, suffix=suffix, tool=tool: link.endswith(f"_{suffix}")
                or link == tool,
                visual_cache,
            )
            geometry.append(arm[:3])
            hand_root = camera_base @ arm_fk[tool] @ mounts[side]
            hand_fk = official.forward_kinematics(
                contracts[side]["model"],
                dict(
                    zip(
                        contracts[side]["names"], q_hand[slot, side], strict=True
                    )
                ),
            )
            hand = raster.camera_triangles(
                contracts[side]["model"],
                hand_fk,
                hand_root,
                lambda _link: True,
                visual_cache,
            )
            geometry.append((hand[0], hand[1], hand[2] + 100 * (side + 1)))
        triangles = np.concatenate([row[0] for row in geometry if len(row[0])])
        colors = np.concatenate([row[1] for row in geometry if len(row[0])])
        labels = np.concatenate([row[2] for row in geometry if len(row[0])])
        _z, rendered, link_id = raster.rasterize_zbuffer(
            triangles,
            colors,
            labels,
            float(k2[0, 0]),
            float(k2[1, 1]),
            float(k2[0, 2]),
            float(k2[1, 2]),
            high_width,
            high_height,
        )
        panel = np.full((high_height, high_width, 3), (42, 44, 48), np.uint8)
        robot = link_id >= 0
        panel[robot] = rendered[robot]
        depth_path = Path(object_depth_entries[slot]["bundle"]["path"])
        with np.load(depth_path, allow_pickle=False) as depth:
            object_mask = np.asarray(depth["visible_object_mask_2x"], dtype=bool)
        object_only = object_mask & ~robot
        panel[object_only] = (30, 195, 245)
        both = object_mask & robot
        panel[both] = np.clip(
            0.55 * panel[both] + 0.45 * np.asarray((30, 195, 245)), 0, 255
        ).astype(np.uint8)
        cv2.rectangle(panel, (0, 0), (high_width, 88), (0, 0, 0), -1)
        cv2.putText(
            panel,
            f"{session} | f{int(frame_ids[slot]):05d} | PREPARED/HOLD",
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.59,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "NON-FINAL | NEUTRAL BACKGROUND | NO CLEAN CONSUMED",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (0, 120, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "YELLOW=analytic role cuboid visibility candidate",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )
        frame_path = temporary / f"{slot:06d}.png"
        if not cv2.imwrite(str(frame_path), panel):
            raise RuntimeError(f"cannot write review frame {frame_path}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "12",
        "-i",
        str(temporary / "%06d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(command, check=True)
    for path in temporary.iterdir():
        path.unlink()
    temporary.rmdir()
    return video_path


def prepare(task: str, *, no_review: bool) -> dict[str, Any]:
    started = time.time()
    config = CONFIG[task]
    output = config["output"]
    if output.exists():
        raise RuntimeError(f"fresh output required: {output}")
    output.mkdir(parents=True)

    required = (
        config["hawor"],
        config["object6d"],
        config["object6d_result"],
        config["old_scene"],
        config["accepted_scene"],
        PROTOCOL_PATH,
        PBR_PATH,
        FOUNDATIONSTEREO_AUTHORITY,
        FOUNDATIONSTEREO_LOCK,
        FOUNDATIONSTEREO_LAUNCHER,
        FOUNDATIONSTEREO_CHECKPOINT,
    )
    for path in required:
        path.resolve(strict=True)
    source_authority = json.loads(config["object6d_result"].read_text())
    if source_authority.get("consumption_authorized") is not False:
        raise RuntimeError("expected Object6D role trajectory to remain non-formal")

    hawor = arrays(config["hawor"])
    object6d = arrays(config["object6d"])
    old = arrays(config["old_scene"])
    accepted = arrays(config["accepted_scene"])
    world = np.asarray(hawor["joints_3d_world"], dtype=np.float64)
    c2w_all = np.asarray(hawor["c2w"], dtype=np.float64)
    intrinsics_all = np.asarray(hawor["intrinsics"], dtype=np.float64)
    frame_count = world.shape[1]
    if world.shape != (2, frame_count, 21, 3):
        raise RuntimeError("world-consistent MANO21 shape mismatch")
    if object6d["T_object_to_world"].shape != (frame_count, 4, 4):
        raise RuntimeError("Object6D/HaWoR frame-count mismatch")
    world_error = float(
        np.max(
            np.abs(
                c2w_all @ object6d["T_object_to_camera"]
                - object6d["T_object_to_world"]
            )
        )
    )
    if world_error > 1e-5:
        raise RuntimeError(f"Object6D camera/world mismatch {world_error}")
    dimensions = np.asarray(object6d["object_dimensions_m"], dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise RuntimeError("invalid object_dimensions_m")
    selected, selection = select_window(
        world,
        np.asarray(object6d["valid"], dtype=bool),
        np.asarray(object6d["T_object_to_world"], dtype=np.float64),
        dimensions,
    )
    original_frames = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
    frame_ids = original_frames[selected]
    fps = float(np.asarray(hawor["fps"]).item())
    timestamps = np.rint((frame_ids + 1) * 1e9 / fps).astype(np.int64)

    v4 = load_module(V4_PATH, f"robot_shared_v4_{task}")
    v2 = load_module(V2_PATH, f"robot_shared_v2_{task}")
    canary = load_module(CANARY_PATH, f"robot_canary_{task}")
    assets = load_pinned_robot_assets(PROJECT)
    mounts, mount_authority = v4.pinned_mounts()
    mount_004_residual = float(np.max(np.abs(mounts - accepted["T_tool_hand"])))
    if mount_004_residual > 1e-12:
        raise RuntimeError("pinned mount differs from accepted chips004 style")
    contracts = v4.model_contract(official, adapter, assets)
    targets, hand_features = target_roots_and_features(v4, contracts, world, selected)
    old_pose = pose_audit(
        canary,
        assets,
        np.asarray(old["T_world_rig"], dtype=np.float64),
        np.asarray(old["q_arm"], dtype=np.float64)[selected],
        mounts,
        targets,
    )
    base, q_arm, arm = v4.arm_candidate(
        official,
        v2,
        canary,
        assets,
        mounts,
        np.asarray(old["q_arm"], dtype=np.float64)[selected],
        np.asarray(old["T_world_rig"], dtype=np.float64),
        targets,
        frame_ids,
        fps,
    )
    protocol = json.loads(PROTOCOL_PATH.read_text())
    with np.load(v4.THUMB, allow_pickle=False) as archive:
        thumb_rotations = np.asarray(archive["rotations"], dtype=np.float64)
    hand_cache = canary.mesh_cache_for_models(
        (contracts[0]["model"], contracts[1]["model"])
    )
    selected_hand = v4.run_weight_candidate(
        official,
        adapter,
        v2,
        canary,
        contracts,
        hand_features,
        thumb_rotations,
        np.asarray(old["q_hand"], dtype=np.float64)[selected],
        frame_ids,
        fps,
        protocol["selected_weights"],
        hand_cache,
        triangle_triangle_intersects_sat,
    )
    q_hand = selected_hand["q_selected"]
    corrected_pose = pose_audit(canary, assets, base, q_arm, mounts, targets)

    object_world = np.asarray(object6d["T_object_to_world"], dtype=np.float64)[selected]
    contact = geometry_and_contact(
        canary=canary,
        assets=assets,
        contracts=contracts,
        cache=hand_cache,
        base=base,
        mounts=mounts,
        q_arm=q_arm,
        q_hand=q_hand,
        object_world=object_world,
        dimensions=dimensions,
    )

    source_width = int(round(2.0 * intrinsics_all[0, 0, 2] + 1.0))
    source_height = int(round(2.0 * intrinsics_all[0, 1, 2] + 1.0))
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("cannot infer source resolution from intrinsics")
    camera_k = intrinsics_all[selected].copy()
    camera_k[:, 0] *= WIDTH / source_width
    camera_k[:, 1] *= HEIGHT / source_height
    c2w = c2w_all[selected]
    camera_path = output / "CAMERA_K_C2W.npz"
    np.savez_compressed(
        camera_path,
        frame_id=frame_ids.astype(np.int64),
        timestamp_ns=timestamps,
        K=camera_k,
        T_camera_to_world=c2w,
    )

    geometry_path = output / "OBJECT_GEOMETRY.json"
    atomic_json(
        geometry_path,
        {
            "schema_version": "robot-role-cuboid-geometry-v1",
            "status": "PREPARED_APPROX_FROM_OBJECT6D_DIMENSIONS",
            "task_id": task,
            "session_id": config["session"],
            "object_id": config["operated_object_id"],
            "kind": "box_xyz",
            "size_xyz_m": dimensions.tolist(),
            "unit": "m",
            "claim_limit": "Role cuboid only; not a full per-instance task registry or formal Object6D geometry authority.",
        },
    )
    geometry_ref = artifact(geometry_path)
    object_path = output / "OPERATED_OBJECT6D.npz"
    np.savez_compressed(
        object_path,
        frame_id=frame_ids.astype(np.int64),
        timestamp_ns=timestamps,
        object_ids=np.asarray([config["operated_object_id"]]),
        T_object_to_camera=np.asarray(object6d["T_object_to_camera"])[selected, None],
        T_world_object=object_world[:, None],
        valid=np.ones((FRAME_COUNT, 1), dtype=np.bool_),
        confidence=np.asarray(object6d["confidence"], dtype=np.float64)[selected, None],
        provenance=np.full((FRAME_COUNT, 1), "MODEL_FIT", dtype="U16"),
        geometry_sha256=np.asarray([geometry_ref["sha256"]], dtype="U64"),
    )
    object_ref = artifact(object_path)
    camera_ref = artifact(camera_path)

    depth_root = output / "depth_frames"
    depth_root.mkdir()
    depth_entries = []
    depth_stats = []
    for slot, local_frame in enumerate(selected):
        depth = project_object_depth(
            ObjectGeometry(
                "box_xyz",
                transform_object_to_camera=np.asarray(
                    object6d["T_object_to_camera"][local_frame], dtype=np.float64
                ),
                box_size_xyz_m=dimensions,
            ),
            camera_k[slot],
            HEIGHT,
            WIDTH,
        )
        finite = depth.amodal_mask
        near = np.asarray(depth.near_m, dtype=np.float32)
        far = np.asarray(depth.far_m, dtype=np.float32)
        index = np.zeros(finite.shape, dtype=np.int16)
        index[finite] = 1
        # No independent scene-occluder Mask exists for these sessions.  The
        # only non-invented candidate is the analytic role support itself.
        visible = np.asarray(finite, dtype=np.bool_)
        if not np.array_equal(np.isfinite(near), np.isfinite(far)):
            raise RuntimeError("analytic near/far support mismatch")
        if np.any(near[finite] <= 0.0) or np.any(far[finite] < near[finite]):
            raise RuntimeError("analytic near/far is not positive ordered optical-Z")
        path = depth_root / f"{int(frame_ids[slot]):06d}.npz"
        np.savez_compressed(
            path,
            frame_id=np.asarray(frame_ids[slot], dtype=np.int64),
            object_depth_near_m_2x=near,
            object_depth_far_m_2x=far,
            object_index_2x=index,
            visible_object_mask_2x=visible,
        )
        depth_entries.append({"frame_id": int(frame_ids[slot]), "bundle": artifact(path)})
        depth_stats.append(
            {
                "frame_id": int(frame_ids[slot]),
                "visible_subpixels": int(np.count_nonzero(visible)),
                "near_m_min": float(np.min(near[finite])),
                "far_m_max": float(np.max(far[finite])),
                "ordered": True,
            }
        )
    depth_manifest_path = output / "OBJECT_DEPTH_FRAMES.json"
    atomic_json(
        depth_manifest_path,
        {
            "schema_version": "chaoyang-object-depth-frame-manifest-v1",
            "mode": "FORMAL_CANDIDATE",
            "task_id": task,
            "session_id": config["session"],
            "frame_count": FRAME_COUNT,
            "resolution": {"width": WIDTH, "height": HEIGHT},
            "depth_storage": "OPTICAL_AXIS_CAMERA_Z_M_FLOAT32",
            "visible_mask_storage": "BOOL_ANALYTIC_ROLE_SUPPORT_PENDING_INDEPENDENT_SCENE_MASK",
            "object_index_storage": "INT16_ZERO_BACKGROUND_POSITIVE_REGISTRY",
            "supersample": SUPERSAMPLE,
            "object_index_registry": [
                {"index": 1, "object_id": config["operated_object_id"]}
            ],
            "bindings": {
                "object6d_sha256": object_ref["sha256"],
                "camera_sha256": camera_ref["sha256"],
                "source_object6d_sha256": artifact(config["object6d"])["sha256"],
                "source_world_consistent_hawor_sha256": artifact(config["hawor"])[
                    "sha256"
                ],
            },
            "frames": depth_entries,
            "claim_limit": "Metric analytic role-cuboid near/far is valid; visible mask equals analytic support until an independent scene-occluder Mask is admitted.",
        },
    )

    trajectory_path = output / "ROBOT_TRAJECTORY_WORLD_CONSISTENT_PREPARED.npz"
    np.savez_compressed(
        trajectory_path,
        frame_id=frame_ids.astype(np.int64),
        timestamp_ns=timestamps,
        q_arm=q_arm,
        q_hand=q_hand,
        T_world_robot_base=base,
        T_camera_robot_base=np.linalg.inv(c2w) @ base,
        T_tool_hand=mounts,
        target_wrist_world=targets,
        source_local_frames=selected,
        source_world_frames=frame_ids,
        fps=np.asarray(fps, dtype=np.float64),
    )
    trajectory_ref = artifact(trajectory_path)

    contact_path = output / "NONPENETRATION_CONTACT_PREPARED_HOLD.json"
    atomic_json(
        contact_path,
        {
            "schema_version": "robot-nonpenetration-contact-prepared-v1",
            "status": "PASS_PREP_GEOMETRY_HOLD_CONTACT",
            "task_id": task,
            "session_id": config["session"],
            "frame_count": FRAME_COUNT,
            "frame_ids": frame_ids.tolist(),
            "bindings": {
                "trajectory_sha256": trajectory_ref["sha256"],
                "object6d_sha256": object_ref["sha256"],
                "geometry_sha256": geometry_ref["sha256"],
            },
            "functional_retarget": {
                "hand_feature_summary": selected_hand["feature_summary"],
                "hand_temporal": selected_hand["temporal"],
                "distinct_finger_self_sat": selected_hand[
                    "exact_distinct_finger_sat"
                ],
            },
            "object_contact": contact,
            "claim_limit": "Exact CPU geometry evidence for a non-authorized role trajectory; failure of named-pad contact keeps all Robot composite routes HOLD.",
        },
    )
    contact_ref = artifact(contact_path)

    render_request_path = output / "ROBOT_RENDER_FRAMES_PREPARED.json"
    atomic_json(
        render_request_path,
        {
            "schema_version": "chaoyang-robot-render-frame-request-prepared-v1",
            "status": "PREPARED_HOLD_NOT_RENDER_AUTHORITY",
            "task_id": task,
            "session_id": config["session"],
            "frame_count": FRAME_COUNT,
            "resolution": {"width": WIDTH, "height": HEIGHT},
            "supersample": SUPERSAMPLE,
            "required_bundle_keys": [
                "frame_id",
                "robot_bgr_2x",
                "robot_range_m_2x",
                "robot_alpha_2x",
            ],
            "required_range_storage": "EUCLIDEAN_CAMERA_RANGE_M_FLOAT32",
            "required_alpha_storage": "STRAIGHT_COVERAGE_FLOAT32_0_TO_1",
            "pbr_config": artifact(PBR_PATH),
            "bindings": {
                "trajectory_sha256": trajectory_ref["sha256"],
                "camera_sha256": camera_ref["sha256"],
                "nonpenetration_contact_sha256": contact_ref["sha256"],
            },
            "frames": [
                {
                    "frame_id": int(frame_id),
                    "output_bundle_path": str(
                        output / "robot_render_frames" / f"{int(frame_id):06d}.npz"
                    ),
                }
                for frame_id in frame_ids
            ],
            "claim_limit": "Task-agnostic renderer request only; bundles are intentionally absent while contact/Clean authority is HOLD.",
        },
    )

    frame_manifest_path = output / "COMPOSITOR_FRAME_INPUTS_PREPARED.json"
    dependency_path = write_depth_dependency_authority(output, task, config["session"])
    atomic_json(
        frame_manifest_path,
        {
            "schema_version": "robot-clean-compositor-frame-input-prepared-v1",
            "status": "PREPARED_HOLD",
            "task_id": task,
            "session_id": config["session"],
            "frame_count": FRAME_COUNT,
            "coordinate_contract": {
                "length_unit": "m",
                "matrix_convention": "column_vectors_left_multiply",
                "camera_axes": "+X_RIGHT_+Y_DOWN_+Z_FORWARD",
                "object_depth": "OPTICAL_AXIS_CAMERA_Z_M",
                "robot_depth_required": "EUCLIDEAN_CAMERA_RANGE_M",
                "supersample": SUPERSAMPLE,
            },
            "bindings": {
                "camera": camera_ref,
                "object6d": object_ref,
                "trajectory": trajectory_ref,
                "object_depth_frames": artifact(depth_manifest_path),
                "robot_render_request": artifact(render_request_path),
                "nonpenetration_contact": contact_ref,
                "depth_dependency_authority": artifact(dependency_path),
            },
            "frames": [
                {
                    "slot": slot,
                    "frame_id": int(frame_ids[slot]),
                    "depth_bundle": depth_entries[slot]["bundle"],
                    "robot_render_bundle_expected": str(
                        output
                        / "robot_render_frames"
                        / f"{int(frame_ids[slot]):06d}.npz"
                    ),
                }
                for slot in range(FRAME_COUNT)
            ],
            "missing_for_compose_robot_clean_general": [
                "formal full-task Object6D registry coverage",
                "independent visible-object Mask authority",
                "formal Clean FFV1 master and provenance",
                "contact/nonpenetration admission",
                "PBR robot frame bundles",
            ],
            "claim_limit": "Prepared frame identity/bindings only. Not a compose_robot_clean_general execution manifest.",
        },
    )

    review_ref = None
    if not no_review:
        raster = load_module(RASTER_PATH, f"robot_cpu_raster_{task}")
        review = render_review(
            raster=raster,
            canary=canary,
            assets=assets,
            contracts=contracts,
            base=base,
            mounts=mounts,
            q_arm=q_arm,
            q_hand=q_hand,
            c2w=c2w,
            camera_k=camera_k,
            object_depth_entries=depth_entries,
            frame_ids=frame_ids,
            session=config["session"],
            output=output,
        )
        review_ref = artifact(review)

    arm_pass = bool(arm["pose"]["all_pass"] and arm["temporal"]["pass"])
    hand_pass = bool(
        selected_hand["feature_summary"]["all_pass"]
        and selected_hand["temporal"]["pass"]
        and selected_hand["exact_distinct_finger_sat"]["pass"]
    )
    contact_pass = bool(
        contact["contact_all24_pass"]
        and contact["dense_object_hand_sat_all48_pass"]
    )
    status = (
        "PREPARED_HOLD_OBJECT6D_AUTHORITY_CONTACT_CLEAN"
        if arm_pass and hand_pass
        else "PREPARED_HOLD_KINEMATIC_AND_DOWNSTREAM_AUTHORITY"
    )
    result_path = output / "RESULT.json"
    result = {
        "schema_version": "robot-world-object-input-preparation-v1",
        "status": status,
        "task_id": task,
        "session_id": config["session"],
        "formal_robot_ready": False,
        "compose_execution_allowed": False,
        "clean_consumed": False,
        "gpu_calls": 0,
        "source_bindings": {
            "world_consistent_hawor": artifact(config["hawor"]),
            "stereo_constrained_object6d": artifact(config["object6d"]),
            "stereo_constrained_object6d_result": artifact(
                config["object6d_result"]
            ),
            "old_scene_audit_only": artifact(config["old_scene"]),
            "accepted_chips004_mount_style": artifact(config["accepted_scene"]),
            "shared_v4_protocol": artifact(PROTOCOL_PATH),
            "mount_authority": mount_authority,
        },
        "selection": {**selection, "frame_ids": frame_ids.tolist()},
        "coordinate_gates": {
            "object_camera_world_max_abs_error": world_error,
            "pass_at_1e_5": bool(world_error <= 1e-5),
            "mount_vs_chips004_max_abs_error": mount_004_residual,
            "mount_style_exact_match": bool(mount_004_residual <= 1e-12),
        },
        "trajectory": {
            "old_scene_vs_world_consistent_wrist": old_pose,
            "corrected_vs_world_consistent_wrist": corrected_pose,
            "shared_joint_refinement": arm,
            "functional_retarget": {
                "weights": selected_hand["weights"],
                "feature_summary": selected_hand["feature_summary"],
                "temporal": selected_hand["temporal"],
                "distinct_finger_self_sat": selected_hand[
                    "exact_distinct_finger_sat"
                ],
                "neutral_homotopy_selected_alpha": selected_hand[
                    "neutral_homotopy_selected_alpha"
                ],
            },
        },
        "object_depth": {
            "geometry": "analytic box_xyz",
            "unit": "m",
            "storage": "OPTICAL_AXIS_CAMERA_Z_M_FLOAT32",
            "fixed_2x": True,
            "positive_ordered_all_frames": True,
            "stats": depth_stats,
            "visible_mask_status": "HOLD_EQUALS_ANALYTIC_SUPPORT_PENDING_INDEPENDENT_SCENE_OCCLUDER_MASK",
        },
        "object6d_authority_scope": {
            "source_consumption_authorized": False,
            "mask_canary_input_authorized": True,
            "robot_formal_consumption_authorized": False,
            "scope": "STEREO_CONSTRAINED_ROLE_TRAJECTORY_ONLY_NOT_FULL_PER_INSTANCE_OBJECT6D",
        },
        "future_formal_entry": {
            "near_far_frame_schema_and_optical_z_implementation_reusable": True,
            "current_numeric_depth_bundles_formal_eligible": False,
            "promotion_conditions": [
                "full per-instance Object6D and task-registry geometry authority",
                "independent visible-object Mask/scene-occluder authority using the pinned FoundationStereo dependency",
                "all-frame object/hand named-pad contact and nonpenetration admission",
                "formal Clean and PBR Robot render-frame authority",
            ],
        },
        "contact_nonpenetration": contact,
        "gates": {
            "G0_explicit_world_consistent_hawor_binding": "PASS",
            "G1_explicit_stereo_constrained_role_object6d_binding": "PASS_PREP_ONLY_NOT_FORMAL_AUTHORITY",
            "G2_analytic_near_far_optical_z_all24": "PASS",
            "G3_independent_visible_object_mask": "HOLD_NO_MASK_AUTHORITY",
            "G4_chips004_mount_style_and_flange_hand_gap": "PASS_DEVELOPMENT_MOUNT",
            "G5_shared_joint_refinement_pose_temporal": "PASS" if arm_pass else "HOLD",
            "G6_functional_retarget_limits_temporal_self_sat": "PASS" if hand_pass else "HOLD",
            "G7_kai_named_pad_contact_0_to_3mm": "PASS" if contact_pass else "HOLD_OBJECT_HAND_ROLE_TRAJECTORY_DISAGREEMENT",
            "G8_clean_authority": "HOLD_NOT_CONSUMED_BY_DESIGN",
            "G9_formal_compositor_execution": "HOLD",
        },
        "artifacts": {
            "camera": camera_ref,
            "operated_object6d": object_ref,
            "object_geometry": geometry_ref,
            "object_depth_frames": artifact(depth_manifest_path),
            "depth_dependency_authority": artifact(dependency_path),
            "trajectory": trajectory_ref,
            "nonpenetration_contact": contact_ref,
            "robot_render_request_frames": artifact(render_request_path),
            "compositor_frame_inputs": artifact(frame_manifest_path),
            "neutral_background_review": review_ref,
        },
        "hold_reasons": [
            "The stereo-constrained v2 input explicitly has consumption_authorized=false and is only a role trajectory, not full per-instance Object6D.",
            "The analytic visible mask has no independent scene-occluder Mask authority.",
            "Kai named-pad contact must be 0--3 mm for all selected frames; the measured role trajectory does not satisfy that gate."
            if not contact_pass
            else "Contact passes the bounded window but still lacks formal Object6D/Mask/Clean authority.",
            "No formal Clean was consumed and no final Robot video was published.",
        ],
        "claim_limit": "CPU-only 24-frame Robot corrected-input preparation. PREPARED/HOLD; neutral review is non-final; no current/manifest/Clean/Robot publication mutation.",
        "wall_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    return {
        "task": task,
        "status": status,
        "output": str(output),
        "result": artifact(result_path),
        "trajectory": trajectory_ref,
        "depth_manifest": artifact(depth_manifest_path),
        "frame_manifest": artifact(frame_manifest_path),
        "review": review_ref,
        "contact_sdf_m": {
            key: contact[key]
            for key in (
                "active_pad_signed_sdf_m_min",
                "active_pad_signed_sdf_m_median",
                "active_pad_signed_sdf_m_max",
            )
        },
        "gpu_calls": 0,
    }


def annotate_existing(task: str) -> dict[str, Any]:
    """Add pinned depth dependencies to a pack generated before that binding."""

    config = CONFIG[task]
    output = config["output"].resolve(strict=True)
    result_path = (output / "RESULT.json").resolve(strict=True)
    frame_manifest_path = (
        output / "COMPOSITOR_FRAME_INPUTS_PREPARED.json"
    ).resolve(strict=True)
    dependency_path = write_depth_dependency_authority(output, task, config["session"])
    dependency_ref = artifact(dependency_path)
    frame_manifest = json.loads(frame_manifest_path.read_text())
    frame_manifest["bindings"]["depth_dependency_authority"] = dependency_ref
    atomic_json(frame_manifest_path, frame_manifest)
    result = json.loads(result_path.read_text())
    result["object6d_authority_scope"] = {
        "source_consumption_authorized": False,
        "mask_canary_input_authorized": True,
        "robot_formal_consumption_authorized": False,
        "scope": "STEREO_CONSTRAINED_ROLE_TRAJECTORY_ONLY_NOT_FULL_PER_INSTANCE_OBJECT6D",
    }
    result["future_formal_entry"] = {
        "near_far_frame_schema_and_optical_z_implementation_reusable": True,
        "current_numeric_depth_bundles_formal_eligible": False,
        "promotion_conditions": [
            "full per-instance Object6D and task-registry geometry authority",
            "independent visible-object Mask/scene-occluder authority using the pinned FoundationStereo dependency",
            "all-frame object/hand named-pad contact and nonpenetration admission",
            "formal Clean and PBR Robot render-frame authority",
        ],
    }
    result["artifacts"]["depth_dependency_authority"] = dependency_ref
    result["artifacts"]["compositor_frame_inputs"] = artifact(frame_manifest_path)
    atomic_json(result_path, result)
    return {
        "task": task,
        "status": result["status"],
        "depth_dependency_authority": dependency_ref,
        "frame_manifest": artifact(frame_manifest_path),
        "updated_result": artifact(result_path),
        "gpu_calls": 0,
    }


def render_existing(task: str) -> dict[str, Any]:
    """Add the optional neutral review to one already prepared fresh pack."""

    config = CONFIG[task]
    output = config["output"].resolve(strict=True)
    result_path = (output / "RESULT.json").resolve(strict=True)
    review_path = output / "ROBOT_PREPARED_HOLD_NEUTRAL24_NONFINAL.mp4"
    if review_path.exists():
        raise RuntimeError(f"review already exists: {review_path}")
    trajectory = arrays(output / "ROBOT_TRAJECTORY_WORLD_CONSISTENT_PREPARED.npz")
    camera = arrays(output / "CAMERA_K_C2W.npz")
    depth_manifest = json.loads((output / "OBJECT_DEPTH_FRAMES.json").read_text())
    v4 = load_module(V4_PATH, f"robot_shared_v4_review_{task}")
    canary = load_module(CANARY_PATH, f"robot_canary_review_{task}")
    raster = load_module(RASTER_PATH, f"robot_cpu_raster_review_{task}")
    assets = load_pinned_robot_assets(PROJECT)
    contracts = v4.model_contract(official, adapter, assets)
    path = render_review(
        raster=raster,
        canary=canary,
        assets=assets,
        contracts=contracts,
        base=np.asarray(trajectory["T_world_robot_base"], dtype=np.float64),
        mounts=np.asarray(trajectory["T_tool_hand"], dtype=np.float64),
        q_arm=np.asarray(trajectory["q_arm"], dtype=np.float64),
        q_hand=np.asarray(trajectory["q_hand"], dtype=np.float64),
        c2w=np.asarray(camera["T_camera_to_world"], dtype=np.float64),
        camera_k=np.asarray(camera["K"], dtype=np.float64),
        object_depth_entries=depth_manifest["frames"],
        frame_ids=np.asarray(camera["frame_id"], dtype=np.int64),
        session=config["session"],
        output=output,
    )
    result = json.loads(result_path.read_text())
    result["artifacts"]["neutral_background_review"] = artifact(path)
    atomic_json(result_path, result)
    return {
        "task": task,
        "status": result["status"],
        "review": artifact(path),
        "updated_result": artifact(result_path),
        "gpu_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    parser.add_argument("--annotate-existing", action="store_true")
    args = parser.parse_args()
    if args.render_existing and args.annotate_existing:
        raise RuntimeError("existing-pack modes are mutually exclusive")
    if args.annotate_existing:
        if args.no_review:
            raise RuntimeError("--annotate-existing and --no-review are mutually exclusive")
        value = annotate_existing(args.task)
    elif args.render_existing:
        if args.no_review:
            raise RuntimeError("--render-existing and --no-review are mutually exclusive")
        value = render_existing(args.task)
    else:
        value = prepare(args.task, no_review=args.no_review)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
