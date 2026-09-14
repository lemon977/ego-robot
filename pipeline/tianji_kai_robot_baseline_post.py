"""Post-IK audit and Chinese review for the Tianji + KaiHand baseline.

This module is imported only after the unified runner's strict preflight.  It
keeps link identity through exact triangle/cuboid SAT, evaluates all five
object-independent KaiHand CAD pad patches, validates the sequential joint
trajectory, checks the development NaturalV2 mount, and renders every source
frame with analytic Object6D near/far occlusion.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline import robot_scene_state_cpu as official  # noqa: E402
from pipeline.object_occlusion_geometry import (  # noqa: E402
    ObjectGeometry,
    project_object_depth,
)
from pipeline.robot_pbr_palette import load_shared_robot_palette  # noqa: E402
from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: E402
from tools import prepare_robot_world_object_inputs as prepared  # noqa: E402


CANARY = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
RASTER = PROJECT / "_run/robot_004_mano21_10frame_cpu_zbuffer_review_20260902_v1/run_zbuffer_review.py"
CAD = PROJECT / "tools/render_robot_scene_cpu_cad.py"
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
SIDE_NAMES = ("left", "right")
PAD_NAMES = ("thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal")
BASE_WIDTH = 320
BASE_HEIGHT = 240
SUPERSAMPLE = 2
MAX_COLLISION_ROW_FRACTION_GRADE_B = 0.02
MAX_PAD_PENETRATION_M_GRADE_B = 0.003


class RobotPostError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RobotPostError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RobotPostError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def triangle_box_sat(triangles_object: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact 13-axis triangle/AABB SAT for all triangles in one named link."""

    triangles = np.asarray(triangles_object, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise RobotPostError("link triangles must have shape (N,3,3)")
    overlap = np.all(
        (triangles.min(axis=1) <= half[None] + 1e-12)
        & (triangles.max(axis=1) >= -half[None] - 1e-12),
        axis=1,
    )
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    normal = np.cross(edges[:, 0], edges[:, 1])
    axes = [normal]
    basis = np.eye(3, dtype=np.float64)
    for edge in range(3):
        for axis in range(3):
            axes.append(np.cross(edges[:, edge], basis[axis]))
    for candidate in axes:
        active = np.einsum("ni,ni->n", candidate, candidate) > 1e-24
        projection = np.einsum("nti,ni->nt", triangles, candidate)
        radius = np.einsum("ni,i->n", np.abs(candidate), half)
        separated = active & (
            (projection.min(axis=1) > radius + 1e-12)
            | (projection.max(axis=1) < -radius - 1e-12)
        )
        overlap &= ~separated
    return overlap


def _in_object(points: np.ndarray, object_world: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - object_world[:3, 3]) @ object_world[:3, :3]


def _trajectory_audit(
    scene: Mapping[str, np.ndarray], assets: Any, result: Mapping[str, Any]
) -> dict[str, Any]:
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    q_arm = np.asarray(scene["q_arm"], dtype=np.float64)
    q_hand = np.asarray(scene["q_hand"], dtype=np.float64)
    count = len(frames)
    if not np.array_equal(frames, np.arange(count)):
        raise RobotPostError("kinematic trajectory does not cover exact full-session IDs")
    if q_arm.ndim != 3 or q_arm.shape[:2] != (count, 2):
        raise RobotPostError("arm trajectory shape mismatch")
    if q_hand.shape != (count, 2, 22):
        raise RobotPostError("KaiHand trajectory must be (N,2,22)")
    if not np.isfinite(q_arm).all() or not np.isfinite(q_hand).all():
        raise RobotPostError("trajectory contains NaN/Inf")
    arm_lower, arm_upper = official._arm_limits(assets)
    hand_bounds = []
    for model in (assets.left_hand, assets.right_hand):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        if len(moving) != 22:
            raise RobotPostError("pinned KaiHand does not have exactly 22 finite-limit DoF")
        hand_bounds.append(
            (
                np.asarray([joint.lower for joint in moving], dtype=np.float64),
                np.asarray([joint.upper for joint in moving], dtype=np.float64),
            )
        )
    hand_lower = np.stack([row[0] for row in hand_bounds])
    hand_upper = np.stack([row[1] for row in hand_bounds])
    arm_limits = bool(np.all(q_arm >= arm_lower[None] - 1e-9) and np.all(q_arm <= arm_upper[None] + 1e-9))
    hand_limits = bool(np.all(q_hand >= hand_lower[None] - 1e-9) and np.all(q_hand <= hand_upper[None] + 1e-9))
    arm_step = float(np.max(np.abs(np.diff(q_arm, axis=0)))) if count > 1 else 0.0
    hand_step = float(np.max(np.abs(np.diff(q_hand, axis=0)))) if count > 1 else 0.0
    previous = result.get("previous_accepted_contract", {})
    accepted_rows = previous.get("frames", [])
    accepted = bool(
        len(accepted_rows) == count
        and all(row.get("accepted") is True for row in accepted_rows)
        and accepted_rows[0].get("static_first_frame_exception") is True
        and all(
            row.get("seed_policy") == "PREVIOUS_ACCEPTED_ONLY"
            for row in accepted_rows[1:]
        )
    )
    branch_jump = bool(
        previous.get("branch_jump") is not False
        or arm_step > 0.12 + 1e-9
        or hand_step > 0.08 + 1e-9
    )
    return {
        "schema_version": "tianji-kai-fullsession-trajectory-audit-v1",
        "frame_count": count,
        "frame_ids_exact": True,
        "finite": True,
        "urdf_limits": {"arm_pass": arm_limits, "hand_pass": hand_limits},
        "first_frame_policy": "STATIC_IK_FULL_URDF_LIMITS",
        "later_frame_seed_policy": "PREVIOUS_ACCEPTED_ONLY",
        "all_frames_accepted": accepted,
        "arm_step_max_rad": arm_step,
        "arm_step_limit_rad": 0.12,
        "hand_step_max_rad": hand_step,
        "hand_step_limit_rad": 0.08,
        "branch_jump": branch_jump,
        "pass": bool(arm_limits and hand_limits and accepted and not branch_jump),
    }


def _arm_link_triangles(
    assets: Any,
    canary: Any,
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    arm_fk: dict[str, np.ndarray],
    base: np.ndarray,
    side: int,
) -> dict[str, np.ndarray]:
    all_links = canary.transformed_triangles(assets.tianji, arm_fk, cache, base)
    suffix = "L" if side == 0 else "R"
    tool = "left_tool" if side == 0 else "right_tool"
    return {
        link: triangles
        for link, triangles in all_links.items()
        if link.endswith(f"_{suffix}") or link == tool
    }


def _collision_contact_audit(
    scene: Mapping[str, np.ndarray], object6d: Mapping[str, np.ndarray], assets: Any,
    canary: Any,
) -> dict[str, Any]:
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    q_arm = np.asarray(scene["q_arm"], dtype=np.float64)
    q_hand = np.asarray(scene["q_hand"], dtype=np.float64)
    base = np.asarray(scene["T_world_rig"], dtype=np.float64)
    mounts = np.asarray(scene["T_tool_hand"], dtype=np.float64)
    object_world = np.asarray(object6d["T_object_to_world"], dtype=np.float64)
    size_key = "object_size_m" if "object_size_m" in object6d else "object_dimensions_m"
    sizes = np.asarray(object6d[size_key], dtype=np.float64)
    if sizes.shape == (3,):
        sizes = np.repeat(sizes[None], len(frames), axis=0)
    if sizes.shape != (len(frames), 3):
        raise RobotPostError("Object6D size trajectory mismatch")
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = [
        tuple(joint.name for joint in model.joints if joint.joint_type != "fixed")
        for model in hand_models
    ]
    cache = canary.mesh_cache_for_models((assets.tianji, *hand_models))

    # The denominator is derived once from the pinned URDF link identities and
    # reused for every frame.  Missing links are an error, never a skipped row.
    neutral_arm = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
    neutral_fk = official.forward_kinematics(assets.tianji, neutral_arm)
    arm_link_names = [
        tuple(sorted(_arm_link_triangles(assets, canary, cache, neutral_fk, np.eye(4), side)))
        for side in range(2)
    ]
    hand_link_names = []
    for side, model in enumerate(hand_models):
        neutral_hand = official.forward_kinematics(
            model, dict.fromkeys(hand_names[side], 0.0)
        )
        hand_link_names.append(
            tuple(sorted(canary.transformed_triangles(model, neutral_hand, cache)))
        )
    fixed_links = [arm_link_names[side] + hand_link_names[side] for side in range(2)]

    collision_rows: list[dict[str, Any]] = []
    pad_rows: list[dict[str, Any]] = []
    finger_self_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for slot, frame_id in enumerate(frames):
        arm_values = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
        for side in range(2):
            arm_values.update(
                dict(zip(official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True))
            )
        arm_fk = official.forward_kinematics(assets.tianji, arm_values)
        frame_collision_count = 0
        frame_pad_values = []
        for side, model in enumerate(hand_models):
            tool = "left_tool" if side == 0 else "right_tool"
            hand_root = base @ arm_fk[tool] @ mounts[side]
            hand_fk = official.forward_kinematics(
                model, dict(zip(hand_names[side], q_hand[slot, side], strict=True))
            )
            by_link = _arm_link_triangles(assets, canary, cache, arm_fk, base, side)
            by_link.update(canary.transformed_triangles(model, hand_fk, cache, hand_root))
            if tuple(sorted(by_link)) != tuple(sorted(fixed_links[side])):
                raise RobotPostError("per-link collision denominator drifted")
            for link in fixed_links[side]:
                triangles = by_link[link]
                local = _in_object(triangles.reshape(-1, 3), object_world[slot]).reshape(-1, 3, 3)
                hits = triangle_box_sat(local, sizes[slot] / 2.0)
                intersecting = int(np.count_nonzero(hits))
                frame_collision_count += int(intersecting > 0)
                collision_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "physical_side": SIDE_NAMES[side],
                        "link": link,
                        "triangle_denominator": int(len(hits)),
                        "intersecting_triangles": intersecting,
                        "collision": bool(intersecting),
                    }
                )
            for pad_name in PAD_NAMES:
                points, pad = prepared.named_pad_points_world(
                    model, SIDE_NAMES[side], cache, hand_fk, hand_root, pad_name
                )
                sdf = prepared.box_sdf(_in_object(points, object_world[slot]), sizes[slot])
                minimum = float(np.min(sdf))
                nearest = float(sdf[np.argmin(np.abs(sdf))])
                frame_pad_values.append(nearest)
                pad_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "physical_side": SIDE_NAMES[side],
                        "pad": pad_name.removesuffix("_distal"),
                        "link": pad["link"],
                        "patch_triangle_denominator": pad["patch_triangles"],
                        "signed_sdf_m_min": minimum,
                        "signed_sdf_m_nearest": nearest,
                        "contact_0_to_3mm": bool(0.0 <= nearest <= 0.003),
                        "penetration_over_3mm": bool(minimum < -MAX_PAD_PENETRATION_M_GRADE_B),
                    }
                )
            self_row = canary.distinct_finger_self_sat(
                model,
                q_hand[slot, side],
                hand_names[side],
                official,
                cache,
                canary.triangle_triangle_intersects_sat,
            )
            self_row.update(
                {"frame_id": int(frame_id), "physical_side": SIDE_NAMES[side]}
            )
            finger_self_rows.append(self_row)
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "colliding_link_count": frame_collision_count,
                "any_named_pad_contact": bool(any(0.0 <= value <= 0.003 for value in frame_pad_values)),
                "nearest_named_pad_signed_sdf_m": float(min(frame_pad_values, key=abs)),
            }
        )
    expected_collision_rows = len(frames) * sum(len(value) for value in fixed_links)
    expected_pad_rows = len(frames) * 2 * 5
    if len(collision_rows) != expected_collision_rows or len(pad_rows) != expected_pad_rows:
        raise RobotPostError("fixed collision/contact denominator is incomplete")
    colliding_rows = sum(row["collision"] for row in collision_rows)
    collision_fraction = colliding_rows / expected_collision_rows
    excessive_penetration = sum(row["penetration_over_3mm"] for row in pad_rows)
    self_collision_rows = sum(not row["pass"] for row in finger_self_rows)
    contact_frames = sum(row["any_named_pad_contact"] for row in frame_rows)
    grade_b_pass = bool(
        collision_fraction <= MAX_COLLISION_ROW_FRACTION_GRADE_B
        and excessive_penetration == 0
        and self_collision_rows == 0
    )
    return {
        "schema_version": "tianji-kai-fixed-denominator-collision-contact-v1",
        "method": {
            "object_collision": "EXACT_PER_LINK_TRIANGLE_CUBOID_13_AXIS_SAT",
            "self_collision": "EXACT_DISTINCT_FINGER_TRIANGLE_TRIANGLE_SAT",
            "contact": "FIVE_OBJECT_INDEPENDENT_KAI_CAD_LOCAL_ANATOMICAL_PAD_PATCHES",
            "sampled_sdf_role": "CONTACT_DIAGNOSTIC_ONLY_NOT_COLLISION_AUTHORITY",
        },
        "fixed_denominator": {
            "frames": len(frames),
            "sides": 2,
            "links_per_side": [len(value) for value in fixed_links],
            "collision_rows_expected": expected_collision_rows,
            "collision_rows_observed": len(collision_rows),
            "pads_per_side": 5,
            "pad_rows_expected": expected_pad_rows,
            "pad_rows_observed": len(pad_rows),
        },
        "thresholds": {
            "max_collision_row_fraction_grade_b": MAX_COLLISION_ROW_FRACTION_GRADE_B,
            "max_pad_penetration_m_grade_b": MAX_PAD_PENETRATION_M_GRADE_B,
        },
        "summary": {
            "colliding_link_rows": int(colliding_rows),
            "collision_row_fraction": collision_fraction,
            "excessive_pad_penetration_rows": int(excessive_penetration),
            "distinct_finger_self_collision_rows": int(self_collision_rows),
            "named_pad_contact_frames": int(contact_frames),
            "named_pad_contact_frame_fraction": contact_frames / len(frames),
        },
        "frame_rows": frame_rows,
        "collision_rows": collision_rows,
        "pad_rows": pad_rows,
        "distinct_finger_self_collision_rows": finger_self_rows,
        "grade_b_hard_gate_pass": grade_b_pass,
    }


def _mount_audit(
    spec: Mapping[str, Any], scene: Mapping[str, np.ndarray], assets: Any, canary: Any
) -> dict[str, Any]:
    authority_path = Path(spec["assets"]["mount_authority"]["path"])
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    expected = np.stack(
        [np.asarray(authority["T_tool_hand"][side], dtype=np.float64) for side in SIDE_NAMES]
    )
    observed = np.asarray(scene["T_tool_hand"], dtype=np.float64)
    residual = float(np.max(np.abs(expected - observed)))
    if residual > 1e-12:
        raise RobotPostError("kinematic trajectory did not consume the specified mount authority")
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    q_arm = np.asarray(scene["q_arm"], dtype=np.float64)
    q_hand = np.asarray(scene["q_hand"], dtype=np.float64)
    base = np.asarray(scene["T_world_rig"], dtype=np.float64)
    models = (assets.left_hand, assets.right_hand)
    names = [tuple(j.name for j in model.joints if j.joint_type != "fixed") for model in models]
    cache = canary.mesh_cache_for_models(models)
    rows = []
    for slot, frame_id in enumerate(frames):
        values = {name: 0.0 for group in official.ARM_JOINT_NAMES for name in group}
        for side in range(2):
            values.update(dict(zip(official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True)))
        arm_fk = official.forward_kinematics(assets.tianji, values)
        for side, suffix in ((0, "L"), (1, "R")):
            tool = "left_tool" if side == 0 else "right_tool"
            flange = base @ arm_fk[f"flange_{suffix}"]
            hand_root = base @ arm_fk[tool] @ observed[side]
            span_world = hand_root[:3, 3] - flange[:3, 3]
            distance = float(np.linalg.norm(span_world))
            if distance <= 0:
                raise RobotPostError("NaturalV2 flange/hand-root span is degenerate")
            axis_world = span_world / distance
            span_flange = flange[:3, :3].T @ span_world
            direction_error = float(
                np.degrees(np.arccos(np.clip(span_flange[2] / np.linalg.norm(span_flange), -1.0, 1.0)))
            )
            hand_fk = official.forward_kinematics(
                models[side], dict(zip(names[side], q_hand[slot, side], strict=True))
            )
            by_link = canary.transformed_triangles(models[side], hand_fk, cache, hand_root)
            base_points = np.concatenate(
                [triangles.reshape(-1, 3) for link, triangles in by_link.items() if "base_link" in link]
            )
            backward_extent = float(np.min((base_points - hand_root[:3, 3]) @ axis_world))
            surface_gap = 0.012 + backward_extent
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "physical_side": SIDE_NAMES[side],
                    "native_plus_z_direction_error_deg": direction_error,
                    "naturalv2_visual_clearance_m": 0.012,
                    "hand_base_backward_extent_m": backward_extent,
                    "separating_axis_surface_gap_m": surface_gap,
                    "direction_pass": bool(direction_error <= 0.1),
                    "nonembedded_pass": bool(surface_gap >= 0.005),
                }
            )
    passed = bool(all(row["direction_pass"] and row["nonembedded_pass"] for row in rows))
    return {
        "schema_version": "tianji-kai-naturalv2-development-mount-audit-v1",
        "mount_matrix_max_abs_residual": residual,
        "fixed_denominator": {"frames": len(frames), "sides": 2, "rows": len(rows)},
        "rows": rows,
        "direction_error_deg_max": max(row["native_plus_z_direction_error_deg"] for row in rows),
        "surface_gap_m_min": min(row["separating_axis_surface_gap_m"] for row in rows),
        "pass": passed,
        "development_only": True,
        "formal_consumer_allowed": False,
        "claim_limit": "12 mm NaturalV2 visual clearance and direction only; not calibrated mechanical deployment authority.",
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RobotPostError("Chinese font is unavailable")


def _header(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    value = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(value)
    draw.rectangle((0, 0, value.width, 76), fill=(0, 0, 0))
    font = _font(19)
    for row, (text, bgr) in enumerate(lines):
        draw.text((10, 4 + row * 24), text, font=font, fill=tuple(reversed(bgr)))
    return cv2.cvtColor(np.asarray(value), cv2.COLOR_RGB2BGR)


def _tint(colors: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if not len(colors):
        return colors
    light = np.clip(colors.astype(np.float64).mean(axis=1, keepdims=True) / 190.0, 0.52, 1.0)
    return np.clip(np.asarray(target)[None] * light, 0, 255).astype(np.uint8)


def _read_clean_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    extra, _ = capture.read()
    capture.release()
    if len(frames) != count or extra:
        raise RobotPostError("Clean video full decode count changed after preflight")
    return frames


def _render_review(
    spec: Mapping[str, Any], output: Path, scene: Mapping[str, np.ndarray],
    object6d: Mapping[str, np.ndarray], assets: Any, canary: Any,
    audit: Mapping[str, Any], trajectory: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    raster = _load_module(RASTER, "tianji_kai_baseline_raster")
    cad = _load_module(CAD, "tianji_kai_baseline_cad")
    palette = load_shared_robot_palette(PROJECT)["cpu_preview"]["bgr_uint8"]
    arm_color = tuple(palette["ARM_SHELL_MATTE_WHITE"])
    hand_color = tuple(palette["KAIHAND_SHELL_STEEL_BLUE"])
    connector_color = tuple(palette["CONNECTOR_FLANGE_IVORY"])
    frames = np.asarray(scene["source_frames"], dtype=np.int64)
    q_arm = np.asarray(scene["q_arm"], dtype=np.float64)
    q_hand = np.asarray(scene["q_hand"], dtype=np.float64)
    base = np.asarray(scene["T_world_rig"], dtype=np.float64)
    mounts = np.asarray(scene["T_tool_hand"], dtype=np.float64)
    c2w = np.asarray(scene["c2w"], dtype=np.float64)
    intrinsics = np.asarray(scene["intrinsics"], dtype=np.float64)
    t_object_camera = np.asarray(object6d["T_object_to_camera"], dtype=np.float64)
    size_key = "object_size_m" if "object_size_m" in object6d else "object_dimensions_m"
    sizes = np.asarray(object6d[size_key], dtype=np.float64)
    if sizes.shape == (3,):
        sizes = np.repeat(sizes[None], len(frames), axis=0)
    raw_root = Path(spec["inputs"]["raw_frame_root"])
    clean_frames = _read_clean_frames(Path(spec["inputs"]["clean"]["video"]["path"]), len(frames))
    frame_root = output / "review_frames"
    frame_root.mkdir()
    visual_cache: dict[Path, Any] = {}
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = [tuple(j.name for j in model.joints if j.joint_type != "fixed") for model in hand_models]
    frame_metrics = []
    for slot, frame_id in enumerate(frames):
        raw_path = raw_root / "preprocess/all_data" / f"{int(frame_id):05d}" / "rgb.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RobotPostError(f"missing selected RGB frame {raw_path}")
        source_h, source_w = raw.shape[:2]
        width, height = BASE_WIDTH * SUPERSAMPLE, BASE_HEIGHT * SUPERSAMPLE
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
        clean = cv2.resize(clean_frames[slot], (width, height), interpolation=cv2.INTER_AREA)
        k_base = intrinsics[slot].copy()
        k_base[0] *= BASE_WIDTH / source_w
        k_base[1] *= BASE_HEIGHT / source_h
        k_render = k_base.copy()
        k_render[0, 0] *= SUPERSAMPLE
        k_render[1, 1] *= SUPERSAMPLE
        k_render[0, 2] = (k_render[0, 2] + 0.5) * SUPERSAMPLE - 0.5
        k_render[1, 2] = (k_render[1, 2] + 0.5) * SUPERSAMPLE - 0.5
        camera_base = np.linalg.inv(c2w[slot]) @ base
        values = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
        for side in range(2):
            values.update(dict(zip(official.ARM_JOINT_NAMES[side], q_arm[slot, side], strict=True)))
        arm_fk = official.forward_kinematics(assets.tianji, values)
        geometry = []
        for side, suffix in ((0, "L"), (1, "R")):
            tool = "left_tool" if side == 0 else "right_tool"
            arm = raster.camera_triangles(
                assets.tianji, arm_fk, camera_base,
                lambda link, suffix=suffix, tool=tool: link.endswith(f"_{suffix}") or link == tool,
                visual_cache,
            )
            geometry.append((arm[0], _tint(arm[1], arm_color), arm[2]))
            hand_root = camera_base @ arm_fk[tool] @ mounts[side]
            connector = cad.connector_triangles(camera_base @ arm_fk[f"flange_{suffix}"], hand_root, 0.012)
            geometry.append((connector[0], np.full((len(connector[0]), 3), connector_color, np.uint8), connector[2] + 1000 + side))
            hand_fk = official.forward_kinematics(
                hand_models[side], dict(zip(hand_names[side], q_hand[slot, side], strict=True))
            )
            hand = raster.camera_triangles(
                hand_models[side], hand_fk, hand_root, lambda _link: True, visual_cache
            )
            geometry.append((hand[0], _tint(hand[1], hand_color), hand[2] + 2000 + side))
        triangles = np.concatenate([row[0] for row in geometry if len(row[0])])
        colors = np.concatenate([row[1] for row in geometry if len(row[0])])
        labels = np.concatenate([row[2] for row in geometry if len(row[0])])
        robot_z, robot_rgb, link_id = raster.rasterize_zbuffer(
            triangles, colors, labels,
            float(k_render[0, 0]), float(k_render[1, 1]),
            float(k_render[0, 2]), float(k_render[1, 2]), width, height,
        )
        object_depth = project_object_depth(
            ObjectGeometry(
                kind="box_xyz",
                transform_object_to_camera=t_object_camera[slot],
                box_size_xyz_m=sizes[slot],
            ),
            k_base, BASE_HEIGHT, BASE_WIDTH,
        )
        robot = link_id >= 0
        object_mask = object_depth.amodal_mask
        object_front = robot & object_mask & (robot_z >= object_depth.near_m - 1e-4)
        robot_visible = robot & ~object_front
        composite = clean.copy()
        composite[robot_visible] = np.clip(
            0.12 * clean[robot_visible] + 0.88 * robot_rgb[robot_visible], 0, 255
        ).astype(np.uint8)
        object_outline = cv2.Canny(object_mask.astype(np.uint8) * 255, 80, 160) > 0
        composite[object_outline] = (0, 220, 255)
        frame_row = audit["frame_rows"][slot]
        pass_frame = frame_row["colliding_link_count"] == 0
        diagnostic = composite.copy()
        diagnostic = _header(
            diagnostic,
            [
                (f"帧 {int(frame_id):05d}｜Robot接触/碰撞", (255, 255, 255)),
                (f"碰撞link={frame_row['colliding_link_count']}｜最近指腹={frame_row['nearest_named_pad_signed_sdf_m']*1000:.2f}mm", (60, 220, 80) if pass_frame else (40, 80, 255)),
                ("逐link精确SAT；五指CAD指腹；黄色=物体遮挡轮廓", (120, 220, 255)),
            ],
        )
        raw_panel = _header(raw, [(f"原始RGB｜帧 {int(frame_id):05d}", (255,255,255)), ("输入身份已绑定SHA", (80,220,100)), ("", (255,255,255))])
        clean_panel = _header(clean, [("Clean结果", (255,255,255)), ("无支撑区域保留原图；物体保护", (80,220,100)), ("", (255,255,255))])
        robot_panel = _header(composite, [("Tianji双臂 + 双KaiHand", (255,255,255)), ("Object6D near/far遮挡已应用", (80,220,100)), ("开发mount：不可用于真机部署", (40,80,255))])
        canvas = np.concatenate((np.concatenate((raw_panel, clean_panel), axis=1), np.concatenate((robot_panel, diagnostic), axis=1)), axis=0)
        path = frame_root / f"{slot:06d}.png"
        if not cv2.imwrite(str(path), canvas):
            raise RobotPostError(f"cannot write review frame {slot}")
        frame_metrics.append(
            {
                "frame_id": int(frame_id),
                "robot_pixels": int(np.count_nonzero(robot)),
                "object_occludes_robot_pixels": int(np.count_nonzero(object_front)),
                "robot_visible_pixels": int(np.count_nonzero(robot_visible)),
            }
        )
    video = output / spec["review"]["video_name"]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(spec["session"]["fps"]),
        "-i", str(frame_root / "%06d.png"), "-an", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", str(video),
    ]
    subprocess.run(command, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(video)],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    decoded = int(probe.stdout.strip())
    verify = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if decoded != len(frames) or verify.returncode != 0:
        raise RobotPostError("full-session Chinese review video decode failed")
    manifest = {
        "schema_version": "tianji-kai-robot-baseline-review-video-v1",
        "language": "zh-CN",
        "frame_count": len(frames),
        "decoded_frame_count": decoded,
        "fps": float(spec["session"]["fps"]),
        "panels": ["raw", "clean", "robot", "contact_collision"],
        "object6d_near_far_occlusion_applied": True,
        "mount_claim": "DEVELOPMENT_ONLY_NOT_DEPLOYABLE",
        "frames": frame_metrics,
    }
    _atomic_json(output / "REVIEW_MANIFEST.json", manifest)
    return video, manifest


def audit_and_render(
    spec: Mapping[str, Any], output: Path, kinematic_result: Mapping[str, Any]
) -> dict[str, Any]:
    scene = _arrays(Path(kinematic_result["outputs"]["scene"]["path"]))
    object6d = _arrays(Path(spec["inputs"]["object6d"]["npz"]["path"]))
    assets = load_pinned_robot_assets(PROJECT)
    canary = _load_module(CANARY, "tianji_kai_baseline_canary_post")
    trajectory = _trajectory_audit(scene, assets, kinematic_result)
    collision = _collision_contact_audit(scene, object6d, assets, canary)
    mount = _mount_audit(spec, scene, assets, canary)
    _atomic_json(output / "TRAJECTORY_AUDIT.json", trajectory)
    _atomic_json(output / "COLLISION_CONTACT_AUDIT.json", collision)
    _atomic_json(output / "NATURALV2_MOUNT_AUDIT.json", mount)
    video, review = _render_review(
        spec, output, scene, object6d, assets, canary, collision, trajectory
    )
    hard_pass = bool(
        trajectory["pass"]
        and collision["grade_b_hard_gate_pass"]
        and mount["pass"]
        and review["decoded_frame_count"] == spec["session"]["frame_count"]
    )
    return {
        "status": "PASS_GRADE_B_DEVELOPMENT_VISUALIZATION" if hard_pass else "FAIL_GRADE_C_ROBOT_HARD_GATE",
        "hard_gates_pass": hard_pass,
        "trajectory_audit": _artifact(output / "TRAJECTORY_AUDIT.json"),
        "collision_contact": _artifact(output / "COLLISION_CONTACT_AUDIT.json"),
        "mount_audit": _artifact(output / "NATURALV2_MOUNT_AUDIT.json"),
        "review_video": _artifact(video),
    }
