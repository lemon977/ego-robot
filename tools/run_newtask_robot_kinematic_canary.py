#!/usr/bin/env python3
"""Generic CPU-only HaWoR -> Tianji/KaiHand kinematic admission canary.

This is deliberately a visual/kinematic diagnostic.  It consumes one compact
full-session HaWoR MANO21 artifact, derives exactly one session-static rig from
the first 24 bilateral direct observations, runs actual Tianji FK/IK and a
shared unit-bone KaiHand retarget, and emits compact 12/24-frame CAD sheets.
It never consumes an old CAP004 base/q trajectory or task object geometry.
Without independent Object6D, Mask and Clean authority it must remain HOLD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
import trimesh  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
MOUNT_PREFLIGHT = PROJECT / "_run/gpt_robot_kaihand_rootplane_mount_cpu_successor_20260831_v1/CPU_PREFLIGHT.json"
MOUNT_MANIFEST = PROJECT / "_run/gpt_robot_kaihand_rootplane_mount_cpu_successor_20260831_v1/RUN_MANIFEST.json"
RASTER_SOURCE = PROJECT / "_run/robot_004_mano21_10frame_cpu_zbuffer_review_20260902_v1/run_zbuffer_review.py"
EXPECTED_MOUNT_PREFLIGHT_SHA = "424f7a708ee7fa01d24d37d3f1b0dce780a601b73a8075a31d2c307cf850cf25"
EXPECTED_MOUNT_MANIFEST_SHA = "e9cad69a42a37ed11ed04a69f28a2fe644c30509adf2b26d841433e2d7b944e8"
EXPECTED_THUMB_ADAPTER_SHA256 = "89d121a8ca869e9c3e37318e9e10e4a1eb3ef731a996554e00f32f7e73b01132"
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
MANO_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
FINGERS = tuple(MANO_CHAINS)
HAND_GROUPS = {
    "thumb": np.arange(0, 6),
    "index": np.arange(6, 10),
    "middle": np.arange(10, 14),
    "ring": np.arange(14, 18),
    "pinky": np.arange(18, 22),
}
HAND_TERMINALS = {"thumb": 6, "index": 4, "middle": 4, "ring": 4, "pinky": 4}
SIDE_NAMES = ("left", "right")
# The established crossed embodiment contract is explicit: human left drives
# physical robot right, human right drives physical robot left.
HUMAN_TO_PHYSICAL = {0: 1, 1: 0}
PHYSICAL_TO_HUMAN = {1: 0, 0: 1}
WIDTH, HEIGHT = 1280, 960
GRADEB_SUCCESSOR_SCHEMA = "tianji-kai-gradeb-cross-chirality-successor-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def chordal_mean(rotations: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(np.sum(rotations, axis=0))
    return left @ np.diag((1.0, 1.0, np.linalg.det(left @ right_t))) @ right_t


def robust_transform(rows: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = chordal_mean(np.asarray([row[:3, :3] for row in rows]))
    result[:3, 3] = np.median(np.asarray([row[:3, 3] for row in rows]), axis=0)
    return result


def rigid_initializer(source: list[np.ndarray], target: list[np.ndarray]) -> np.ndarray:
    source_direction = source[0][:3, 3] - source[1][:3, 3]
    target_direction = target[0][:3, 3] - target[1][:3, 3]
    source_direction /= np.linalg.norm(source_direction)
    target_direction /= np.linalg.norm(target_direction)
    covariance = 4.0 * np.outer(target_direction, source_direction)
    for source_root, target_root in zip(source, target, strict=True):
        covariance += target_root[:3, :3] @ source_root[:3, :3].T
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ np.diag((1.0, 1.0, np.linalg.det(left @ right_t))) @ right_t
    source_mid = 0.5 * (source[0][:3, 3] + source[1][:3, 3])
    target_mid = 0.5 * (target[0][:3, 3] + target[1][:3, 3])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_mid - rotation @ source_mid
    return result


def rotation_error_deg(matrix: np.ndarray, official: Any) -> float:
    return float(np.degrees(np.linalg.norm(official._rotation_vector(matrix))))


def pose_error(actual: np.ndarray, target: np.ndarray, official: Any) -> tuple[float, float]:
    delta = np.linalg.inv(target) @ actual
    return float(np.linalg.norm(delta[:3, 3]) * 1000.0), rotation_error_deg(delta[:3, :3], official)


def load_gradeb_successor_contract(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on the single bounded Chips034 development successor."""
    contract = json.loads(path.resolve(strict=True).read_text())
    if contract.get("schema_version") != GRADEB_SUCCESSOR_SCHEMA:
        raise RuntimeError("Grade-B successor contract schema mismatch")
    if (contract.get("task_id"), contract.get("session_id")) != (args.task_id, args.session_id):
        raise RuntimeError("Grade-B successor contract identity mismatch")
    scope = contract.get("scope", {})
    identity = contract.get("identity", {})
    wrist = contract.get("wrist_basis_policy", {})
    hand = contract.get("hand_retarget", {})
    placement = contract.get("frame0_static_placement", {})
    endpoint = contract.get("endpoint_gate_after_placement", {})
    temporal = contract.get("temporal_hard_gates", {})
    revision = contract.get("successor_revision")
    formal_integration = revision == "V2_ONE_STEP_STOP_VIABILITY_FORMAL_INTEGRATION"
    scope_required = (
        (
            args.window_frames >= 24,
            args.prefer_earliest_window,
            scope.get("kind") == "FULL_SESSION_FORMAL_DEVELOPMENT_VISUALIZATION_METHOD",
            scope.get("minimum_frame_count") == 24,
            scope.get("source_first_frame") == 0,
            scope.get("formal_robot_execution_requires_fresh_preflight") is True,
            scope.get("formal_deployment_authorized") is False,
            scope.get("clean_consumed_at_contract_creation") is False,
            scope.get("retry_or_threshold_search_authorized") is False,
        )
        if formal_integration
        else (
            args.window_frames == 24,
            args.prefer_earliest_window,
            scope.get("kind") == "24_FRAME_CPU_DEVELOPMENT_VISUALIZATION_CANARY_ONLY",
            scope.get("source_first_frame") == 0,
            scope.get("formal_robot_execution_authorized") is False,
            scope.get("formal_deployment_authorized") is False,
            scope.get("clean_consumed") is False,
            scope.get("retry_or_threshold_search_authorized") is False,
        )
    )
    required = (
        args.task_id == "chips",
        args.session_id == "get_potato_chips_0902_034",
        args.strict_previous_accepted,
        args.fixed_previous_ik_branch,
        args.first_frame_static_exception,
        not args.fast_hand_retarget,
        not args.skip_expensive_self_collision,
        args.arm_step_limit == 0.12,
        args.hand_step_limit == 0.08,
        *scope_required,
        identity.get("human_left_drives_physical_right") is True,
        identity.get("human_right_drives_physical_left") is True,
        wrist.get("name") == "SHARED_ANATOMICAL_SEMANTIC_BASIS_RETAIN_HUMAN_HANDEDNESS",
        wrist.get("physical_handedness_substitution_for_human_target") is False,
        hand.get("method") == "NONLINEAR_PER_FINGER_LEAST_SQUARES",
        hand.get("fast_analytic_allowed") is False,
        hand.get("hundred_degree_relaxation_allowed") is False,
        placement.get("source") == "FIRST_VALID_BILATERAL_DIRECT_FRAME_ONLY",
        placement.get("freeze_for_session") is True,
        placement.get("development_visualization_only") is True,
        placement.get("real_robot_measurement_or_calibration") is False,
        endpoint.get("position_max_mm") == 40.0,
        endpoint.get("rotation_max_deg") == 20.0,
        temporal.get("later_frame_seed") == "PREVIOUS_ACCEPTED_ONLY",
        temporal.get("arm_step_max_rad") == 0.12,
        temporal.get("hand_step_max_rad") == 0.08,
        temporal.get("arm_second_difference_max_rad") == 0.06,
        temporal.get("hand_second_difference_max_rad") == 0.06,
        temporal.get("urdf_limits_required") is True,
        temporal.get("branch_jump_forbidden") is True,
        temporal.get("collision_gate_relaxation_allowed") is False,
    )
    if not all(required):
        raise RuntimeError("Grade-B successor invocation or frozen contract mismatch")
    if revision in {
        "V2_ONE_STEP_STOP_VIABILITY",
        "V2_ONE_STEP_STOP_VIABILITY_FORMAL_INTEGRATION",
    }:
        if formal_integration:
            validated = contract.get("validated_canary", {})
            for name in ("readiness_result", "canary_result"):
                reference = validated.get(name, {})
                validated_path = Path(str(reference.get("path", ""))).resolve(strict=True)
                if (
                    validated_path.stat().st_size != reference.get("bytes")
                    or sha256(validated_path) != reference.get("sha256")
                ):
                    raise RuntimeError(f"Grade-B formal integration {name} drift")
            if (
                validated.get("accepted_frames") != 24
                or validated.get("visual_kinematic_candidate") is not True
                or temporal.get("full_fixed_denominator_per_link_collision_required_in_postprocessor") is not True
                or temporal.get("five_pad_contact_metrics_required_in_postprocessor") is not True
            ):
                raise RuntimeError("Grade-B formal integration evidence drift")
        else:
            predecessor = contract.get("predecessor_failure", {})
            predecessor_path = Path(str(predecessor.get("path", ""))).resolve(strict=True)
            if (
                predecessor_path.stat().st_size != predecessor.get("bytes")
                or sha256(predecessor_path) != predecessor.get("sha256")
            ):
                raise RuntimeError("Grade-B viability successor predecessor receipt drift")
        if not (
            temporal.get("arm_solver_effective_step_max_rad") == 0.06
            and temporal.get("hand_solver_effective_step_max_rad") == 0.06
            and temporal.get("one_step_stop_viability_required") is True
        ):
            raise RuntimeError("Grade-B viability successor contract drift")
    return contract


def bounded_temporal_limits(
    lower: np.ndarray,
    upper: np.ndarray,
    previous: np.ndarray,
    previous_previous: np.ndarray | None,
    step_limit: float,
    acceleration_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect URDF, previous-step and optional second-difference bounds."""
    bounded_lower = np.maximum(lower, previous - step_limit)
    bounded_upper = np.minimum(upper, previous + step_limit)
    if previous_previous is not None:
        predicted = 2.0 * previous - previous_previous
        bounded_lower = np.maximum(bounded_lower, predicted - acceleration_limit)
        bounded_upper = np.minimum(bounded_upper, predicted + acceleration_limit)
    if np.any(bounded_lower > bounded_upper):
        raise RuntimeError("empty previous-accepted temporal bounds")
    return bounded_lower, bounded_upper


def gradeb_hand_shape_gate(detail: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    thresholds = contract["hand_retarget"]
    absolute_max = max(float(row["max_bone_error_deg"]) for row in detail.values())
    mean_max = max(float(row["mean_bone_error_deg"]) for row in detail.values())
    distal_max = max(float(row["bone_errors_deg"][-1]) for row in detail.values())
    return {
        "absolute_bone_error_deg_max": absolute_max,
        "per_finger_mean_error_deg_max": mean_max,
        "distal_bone_error_deg_max": distal_max,
        "absolute_bone_error_limit_deg": float(thresholds["absolute_bone_error_max_deg"]),
        "per_finger_mean_error_limit_deg": float(thresholds["per_finger_mean_error_max_deg"]),
        "distal_bone_error_limit_deg": float(thresholds["distal_bone_error_max_deg"]),
        "pass": bool(
            absolute_max <= float(thresholds["absolute_bone_error_max_deg"])
            and mean_max <= float(thresholds["per_finger_mean_error_max_deg"])
            and distal_max <= float(thresholds["distal_bone_error_max_deg"])
        ),
    }


def derive_frame0_static_visual_placement(
    tool_world: np.ndarray,
    target_root: np.ndarray,
    nominal_mount: np.ndarray,
    official: Any,
    translation_cap_m: float,
    rotation_cap_deg: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Derive one session-static visual placement and audit it against caps."""
    effective_mount = np.linalg.inv(tool_world) @ target_root
    if not np.isfinite(effective_mount).all():
        raise RuntimeError("non-finite frame0-derived development placement")
    delta = np.linalg.inv(nominal_mount) @ effective_mount
    delta_translation_m = float(np.linalg.norm(delta[:3, 3]))
    delta_rotation_deg = rotation_error_deg(delta[:3, :3], official)
    if delta_translation_m > translation_cap_m + 1e-12:
        raise RuntimeError("frame0-derived placement exceeds frozen translation cap")
    if delta_rotation_deg > rotation_cap_deg + 1e-9:
        raise RuntimeError("frame0-derived placement exceeds frozen rotation cap")
    post_position_mm, post_rotation_deg = pose_error(tool_world @ effective_mount, target_root, official)
    return effective_mount, {
        "delta_translation_m": delta_translation_m,
        "delta_rotation_deg": delta_rotation_deg,
        "post_position_mm": post_position_mm,
        "post_rotation_deg": post_rotation_deg,
    }


def load_corrected_mounts() -> tuple[np.ndarray, dict[str, Any]]:
    if sha256(MOUNT_PREFLIGHT) != EXPECTED_MOUNT_PREFLIGHT_SHA:
        raise RuntimeError("corrected mount preflight SHA drift")
    if sha256(MOUNT_MANIFEST) != EXPECTED_MOUNT_MANIFEST_SHA:
        raise RuntimeError("corrected mount manifest SHA drift")
    preflight = json.loads(MOUNT_PREFLIGHT.read_text())
    manifest = json.loads(MOUNT_MANIFEST.read_text())
    if preflight.get("status") != "PASS_CPU_REVIEW_READY" or manifest.get("status") != "PASS_CPU_REVIEW_READY":
        raise RuntimeError("corrected mount CPU review gate is not pass")
    if preflight.get("development_only") is not True or preflight.get("formal_consumer_allowed") is not False:
        raise RuntimeError("corrected mount claim limits drift")
    mounts = np.stack([np.asarray(preflight["sides"][side]["T_tool_hand_exact"], dtype=np.float64) for side in SIDE_NAMES])
    if not np.array_equal(mounts, np.asarray(manifest["T_tool_hand"], dtype=np.float64)):
        raise RuntimeError("corrected mount files disagree")
    return mounts, {"preflight": evidence(MOUNT_PREFLIGHT), "manifest": evidence(MOUNT_MANIFEST), "formal_consumer_allowed": False}


def mesh_cache_for_models(models: tuple[Any, ...]) -> dict[Path, tuple[np.ndarray, np.ndarray]]:
    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for model in models:
        for visual in model.visuals:
            if visual.mesh_path in cache:
                continue
            mesh = trimesh.load_mesh(visual.mesh_path, process=False, force="mesh")
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            triangles = vertices[faces]
            twice_area = np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            cache[visual.mesh_path] = (vertices, faces[twice_area > 1e-9])
    return cache


def transformed_triangles(model: Any, fk: dict[str, np.ndarray], cache: dict[Path, tuple[np.ndarray, np.ndarray]], root: np.ndarray | None = None) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {}
    base = np.eye(4, dtype=np.float64) if root is None else root
    for visual in model.visuals:
        vertices, faces = cache[visual.mesh_path]
        placed = transform(base @ fk[visual.link] @ visual.origin, vertices)
        result.setdefault(visual.link, []).append(placed[faces])
    return {link: np.concatenate(rows) for link, rows in result.items()}


def triangle_pair_intersects(first: np.ndarray, second: np.ndarray, sat) -> tuple[bool, int]:
    first_min, first_max = first.min(axis=1), first.max(axis=1)
    second_min, second_max = second.min(axis=1), second.max(axis=1)
    candidate_count = 0
    for start in range(0, len(first), 192):
        stop = min(start + 192, len(first))
        overlap = np.all(
            (first_max[start:stop, None] >= second_min[None])
            & (second_max[None] >= first_min[start:stop, None]),
            axis=2,
        )
        for local, other in np.argwhere(overlap):
            candidate_count += 1
            if sat(first[start + int(local)], second[int(other)]):
                return True, candidate_count
    return False, candidate_count


def distinct_finger_self_sat(model: Any, q: np.ndarray, hand_names: tuple[str, ...], official: Any, cache: dict[Path, tuple[np.ndarray, np.ndarray]], sat) -> dict[str, Any]:
    fk = official.forward_kinematics(model, dict(zip(hand_names, q, strict=True)))
    triangles = transformed_triangles(model, fk, cache)
    by_finger: dict[str, list[tuple[str, np.ndarray]]] = {finger: [] for finger in FINGERS}
    for link, rows in triangles.items():
        for finger in FINGERS:
            if f"_{finger}_link" in link:
                by_finger[finger].append((link, rows))
                break
    cell_size = 0.006

    def occupied_cells(low: np.ndarray, high: np.ndarray):
        first = np.floor(low / cell_size).astype(np.int64)
        last = np.floor(high / cell_size).astype(np.int64)
        for x in range(int(first[0]), int(last[0]) + 1):
            for y in range(int(first[1]), int(last[1]) + 1):
                for z in range(int(first[2]), int(last[2]) + 1):
                    yield (x, y, z)

    hashes: dict[str, dict[tuple[int, int, int], list[int]]] = {}
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link, rows in triangles.items():
        low, high = rows.min(axis=1), rows.max(axis=1)
        table: dict[tuple[int, int, int], list[int]] = {}
        for index, (triangle_low, triangle_high) in enumerate(zip(low, high, strict=True)):
            for cell in occupied_cells(triangle_low, triangle_high):
                table.setdefault(cell, []).append(index)
        hashes[link] = table
        bounds[link] = (low, high)

    tested_pairs, tested_candidates = 0, 0
    for first_offset, first_name in enumerate(FINGERS):
        for second_name in FINGERS[first_offset + 1 :]:
            for first_link, first_rows in by_finger[first_name]:
                first_lo, first_hi = first_rows.min(axis=(0, 1)), first_rows.max(axis=(0, 1))
                for second_link, second_rows in by_finger[second_name]:
                    second_lo, second_hi = second_rows.min(axis=(0, 1)), second_rows.max(axis=(0, 1))
                    if np.any(first_hi < second_lo) or np.any(second_hi < first_lo):
                        continue
                    tested_pairs += 1
                    second_triangle_low, second_triangle_high = bounds[second_link]
                    table = hashes[second_link]
                    for first_triangle in first_rows:
                        low, high = first_triangle.min(axis=0), first_triangle.max(axis=0)
                        candidate_set: set[int] = set()
                        for cell in occupied_cells(low, high):
                            candidate_set.update(table.get(cell, ()))
                        if not candidate_set:
                            continue
                        candidates = np.fromiter(candidate_set, dtype=np.int64)
                        candidates = candidates[
                            np.all(second_triangle_high[candidates] >= low, axis=1)
                            & np.all(second_triangle_low[candidates] <= high, axis=1)
                        ]
                        for candidate in candidates:
                            tested_candidates += 1
                            if sat(first_triangle, second_rows[int(candidate)]):
                                return {"pass": False, "collision_pair": [first_link, second_link], "broadphase_link_pairs": tested_pairs, "sat_triangle_candidates_tested": tested_candidates}
    return {"pass": True, "collision_pair": None, "broadphase_link_pairs": tested_pairs, "sat_triangle_candidates_tested": tested_candidates}


def hand_finger_directions(model: Any, q: np.ndarray, finger: str, hand_names: tuple[str, ...], basis: np.ndarray, official: Any, adapter: Any, prefix: str) -> np.ndarray:
    fk = official.forward_kinematics(model, dict(zip(hand_names, q, strict=True)))
    points = [np.zeros(3, dtype=np.float64)]
    for number in range(1, HAND_TERMINALS[finger] + 1):
        points.append(fk[f"{prefix}_{finger}_link{number}"][:3, 3])
    return adapter.resample_polyline_unit_directions(np.asarray(points), bone_count=4) @ basis


def make_sheet(panels: list[np.ndarray], columns: int, panel_size: tuple[int, int]) -> np.ndarray:
    width, height = panel_size
    rows = int(np.ceil(len(panels) / columns))
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        resized = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = resized
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=("chips", "poker"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=24)
    parser.add_argument("--sparse-review", action="store_true")
    parser.add_argument("--skip-expensive-self-collision", action="store_true")
    parser.add_argument("--fast-hand-retarget", action="store_true")
    parser.add_argument("--fixed-previous-ik-branch", action="store_true")
    parser.add_argument("--first-frame-static-exception", action="store_true")
    parser.add_argument("--strict-previous-accepted", action="store_true")
    parser.add_argument("--arm-step-limit", type=float, default=0.12)
    parser.add_argument("--hand-step-limit", type=float, default=0.08)
    parser.add_argument("--thumb-adapter", type=Path)
    parser.add_argument(
        "--accepted-provenance",
        action="append",
        default=["OBSERVED"],
        help="Observed HaWoR provenance label admitted by this invocation",
    )
    parser.add_argument("--prefer-earliest-window", action="store_true")
    parser.add_argument("--gradeb-successor-contract", type=Path)
    args = parser.parse_args()
    if not (0.0 < args.arm_step_limit <= 0.12):
        parser.error("--arm-step-limit must be in (0, 0.12]")
    if not (0.0 < args.hand_step_limit <= 0.12):
        parser.error("--hand-step-limit must be in (0, 0.12]")
    if args.strict_previous_accepted and not args.fixed_previous_ik_branch:
        parser.error("--strict-previous-accepted requires --fixed-previous-ik-branch")
    successor_contract_path = (
        args.gradeb_successor_contract.resolve(strict=True)
        if args.gradeb_successor_contract is not None
        else None
    )
    successor_contract = (
        load_gradeb_successor_contract(successor_contract_path, args)
        if successor_contract_path is not None
        else None
    )
    successor_temporal = (
        successor_contract["temporal_hard_gates"]
        if successor_contract is not None
        else {}
    )
    arm_solver_step_limit = float(
        successor_temporal.get("arm_solver_effective_step_max_rad", args.arm_step_limit)
    )
    hand_solver_step_limit = float(
        successor_temporal.get("hand_solver_effective_step_max_rad", args.hand_step_limit)
    )
    started = time.time()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh output root required: {output}")
    output.mkdir(parents=True)

    sys.path.insert(0, str(PROJECT))
    from pipeline import robot_scene_state_cpu as official  # noqa: PLC0415
    from pipeline import robot_wrist_kai_adapter as adapter  # noqa: PLC0415
    from pipeline.robot_contact_geometry import triangle_triangle_intersects_sat  # noqa: PLC0415
    from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: PLC0415

    if args.thumb_adapter is not None:
        thumb_path = args.thumb_adapter.resolve(strict=True)
        if sha256(thumb_path) != EXPECTED_THUMB_ADAPTER_SHA256:
            raise RuntimeError("shared thumb adapter SHA drift")
        with np.load(thumb_path, allow_pickle=False) as thumb_archive:
            thumb_rotations = np.asarray(thumb_archive["rotations"], dtype=np.float64)
        if thumb_rotations.shape != (2, 3, 3) or not np.isfinite(thumb_rotations).all():
            raise RuntimeError("shared thumb adapter rotation shape mismatch")
        original_hand_finger_directions = hand_finger_directions

        def corrected_hand_finger_directions(
            model, q, finger, hand_names_value, basis, official_value, adapter_value, prefix
        ):
            directions = original_hand_finger_directions(
                model, q, finger, hand_names_value, basis, official_value,
                adapter_value, prefix,
            )
            if finger == "thumb":
                physical = 0 if prefix == "hand_l" else 1
                directions = directions @ thumb_rotations[physical].T
            return directions

        globals()["hand_finger_directions"] = corrected_hand_finger_directions

    raster = load_module(RASTER_SOURCE, "newtask_robot_cpu_raster")
    hawor_path = args.hawor_npz.resolve(strict=True)
    hawor_result_path = args.hawor_result.resolve(strict=True)
    admission_path = args.source_admission.resolve(strict=True)
    raw_root = args.raw_root.resolve(strict=True)
    data = arrays(hawor_path)
    result_upstream = json.loads(hawor_result_path.read_text())
    source_admission = json.loads(admission_path.read_text())
    frame_count = int(data["joints_3d_world"].shape[1])
    expected_shape = (2, frame_count, 21, 3)
    if data["joints_3d_world"].shape != expected_shape or data["joints_3d_camera"].shape != expected_shape:
        raise RuntimeError("HaWoR MANO21 geometry shape mismatch")
    if tuple(str(value) for value in data["mano_joint_names"].tolist()) != MANO_NAMES:
        raise RuntimeError("HaWoR MANO21 named order mismatch")
    if int(data["mano_wrist_index"]) != 0 or tuple(data["mano_tip_indices"].tolist()) != (4, 8, 12, 16, 20):
        raise RuntimeError("HaWoR MANO21 index authority mismatch")
    provenance = np.asarray(data["provenance"])
    observed = np.asarray(data["observed"], dtype=bool)
    accepted_provenance = tuple(dict.fromkeys(args.accepted_provenance))
    direct = observed & np.isin(provenance, accepted_provenance)
    if direct.shape != (2, frame_count):
        raise RuntimeError("HaWoR direct provenance shape mismatch")
    initializer_frames = np.arange(min(24, frame_count), dtype=np.int32)
    initializer_pass = bool(len(initializer_frames) == 24 and np.all(direct[:, initializer_frames]))
    if not initializer_pass:
        payload = {
            "schema_version": "newtask-robot-kinematic-admission-v1",
            "status": "HOLD_FIRST24_NOT_BILATERAL_DIRECT",
            "task_id": args.task_id,
            "session_id": args.session_id,
            "initializer_frames": initializer_frames.tolist(),
            "direct_counts": [int(np.count_nonzero(direct[side, initializer_frames])) for side in range(2)],
            "formal_robot_ready": False,
            "object6d_available": False,
            "contact_claim": False,
        }
        atomic_json(output / "RESULT.json", payload)
        return 2

    assets = load_pinned_robot_assets(PROJECT)
    mounts, mount_authority = load_corrected_mounts()
    arm_lower, arm_upper = official._arm_limits(assets)
    arm_neutral = 0.5 * (arm_lower + arm_upper)
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names: list[tuple[str, ...]] = []
    hand_lower: list[np.ndarray] = []
    hand_upper: list[np.ndarray] = []
    hand_basis: list[np.ndarray] = []
    prefixes = ("hand_l", "hand_r")
    for physical, model in enumerate(hand_models):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        names = tuple(joint.name for joint in moving)
        lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
        upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
        zero_fk = official.forward_kinematics(model, dict(zip(names, 0.5 * (lower + upper), strict=True)))
        dummy = np.zeros((21, 3), dtype=np.float64)
        prefix = prefixes[physical]
        dummy[2] = zero_fk[f"{prefix}_thumb_link1"][:3, 3]
        dummy[5] = zero_fk[f"{prefix}_index_link1"][:3, 3]
        dummy[9] = zero_fk[f"{prefix}_middle_link1"][:3, 3]
        dummy[17] = zero_fk[f"{prefix}_pinky_link1"][:3, 3]
        hand_names.append(names)
        hand_lower.append(lower)
        hand_upper.append(upper)
        hand_basis.append(
            adapter.final_v3_mano_palm_basis(
                dummy, handedness=SIDE_NAMES[physical]
            )
        )
    hand_neutral = np.stack([0.5 * (hand_lower[side] + hand_upper[side]) for side in range(2)])

    world = np.asarray(data["joints_3d_world"], dtype=np.float64)
    c2w = np.asarray(data["c2w"], dtype=np.float64)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    original_frames = np.asarray(data["original_frame_indices"], dtype=np.int32)
    fps = float(np.asarray(data["fps"]).item())

    def target_root(frame: int, human: int) -> np.ndarray:
        points = world[human, frame]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = adapter.final_v3_mano_palm_basis(points, handedness=SIDE_NAMES[human]) @ hand_basis[HUMAN_TO_PHYSICAL[human]].T
        result[:3, 3] = points[0]
        return result

    robust_targets = [robust_transform([target_root(int(frame), human) for frame in initializer_frames]) for human in range(2)]
    source_roots = [official._tool_fk(assets, physical, arm_neutral[physical]) @ mounts[physical] for physical in range(2)]
    target_by_physical = [robust_targets[PHYSICAL_TO_HUMAN[physical]] for physical in range(2)]
    world_rig = rigid_initializer(source_roots, target_by_physical)

    window_frames = int(args.window_frames)
    if window_frames < 24 or window_frames > frame_count:
        raise RuntimeError("window-frames must be within [24, frame_count]")
    wrist_step = np.max(np.linalg.norm(np.diff(world[:, :, 0], axis=1), axis=2), axis=0)
    center = window_frames // 2 if args.prefer_earliest_window else (int(np.argmax(wrist_step) + 1) if len(wrist_step) else 0)
    start = max(0, min(center - window_frames // 2, frame_count - window_frames))
    temporal_frames = np.arange(start, start + window_frames, dtype=np.int32)
    if not np.all(direct[:, temporal_frames]):
        candidates = [offset for offset in range(0, frame_count - window_frames + 1) if np.all(direct[:, offset : offset + window_frames])]
        if not candidates:
            raise RuntimeError(f"no bilateral direct {window_frames}-frame temporal window")
        start = min(candidates, key=lambda offset: (abs((offset + window_frames // 2) - center), offset))
        temporal_frames = np.arange(start, start + window_frames, dtype=np.int32)
    review_slots = np.unique(np.linspace(0, window_frames - 1, min(12, window_frames), dtype=np.int32))

    nominal_mounts = mounts.copy()
    frame0_arm: np.ndarray | None = None
    placement_audit: list[dict[str, Any]] = []
    if successor_contract is not None:
        if int(temporal_frames[0]) != 0 or int(original_frames[temporal_frames[0]]) != 0:
            raise RuntimeError("Grade-B successor placement must derive from source frame 0")
        effective_mounts = nominal_mounts.copy()
        frame0_rows: list[np.ndarray] = []
        placement_caps = successor_contract["frame0_static_placement"]
        for physical in range(2):
            human = PHYSICAL_TO_HUMAN[physical]
            target = target_root(0, human)
            target_tool_nominal = target @ np.linalg.inv(nominal_mounts[physical])
            q_position, n_position = official._solve_one_arm_position_only(
                assets,
                side=physical,
                base=world_rig,
                target_tool=target_tool_nominal,
                initial_q=arm_neutral[physical],
                lower=arm_lower[physical],
                upper=arm_upper[physical],
            )
            q_nominal, n_pose = official._solve_one_arm(
                assets,
                side=physical,
                base=world_rig,
                target_tool=target_tool_nominal,
                initial_q=q_position,
                lower=arm_lower[physical],
                upper=arm_upper[physical],
            )
            tool_world = world_rig @ official._tool_fk(assets, physical, q_nominal)
            nominal_actual = tool_world @ nominal_mounts[physical]
            pre_position_mm, pre_rotation_deg = pose_error(nominal_actual, target, official)
            effective_mount, placement_metrics = derive_frame0_static_visual_placement(
                tool_world,
                target,
                nominal_mounts[physical],
                official,
                float(placement_caps["delta_from_nominal_translation_max_m"]),
                float(placement_caps["delta_from_nominal_rotation_max_deg"]),
            )
            delta = np.linalg.inv(nominal_mounts[physical]) @ effective_mount
            effective_mounts[physical] = effective_mount
            frame0_rows.append(q_nominal)
            placement_audit.append(
                {
                    "physical_side": SIDE_NAMES[physical],
                    "human_side": SIDE_NAMES[human],
                    "source_frame": 0,
                    "nominal_mount": nominal_mounts[physical].tolist(),
                    "effective_session_static_mount": effective_mount.tolist(),
                    "delta_from_nominal": delta.tolist(),
                    "delta_translation_m": placement_metrics["delta_translation_m"],
                    "delta_rotation_deg": placement_metrics["delta_rotation_deg"],
                    "pre_placement_ik_residual": {
                        "position_mm": pre_position_mm,
                        "rotation_deg": pre_rotation_deg,
                    },
                    "post_placement_residual": {
                        "position_mm": placement_metrics["post_position_mm"],
                        "rotation_deg": placement_metrics["post_rotation_deg"],
                    },
                    "solver_evaluations": int(n_position + n_pose),
                    "development_visualization_only": True,
                    "real_robot_measurement_or_calibration": False,
                }
            )
        mounts = effective_mounts
        frame0_arm = np.asarray(frame0_rows, dtype=np.float64)

    def retarget_hand(
        frame: int,
        human: int,
        previous: np.ndarray,
        previous_previous: np.ndarray | None,
        *,
        static_initial: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        physical = HUMAN_TO_PHYSICAL[human]
        target = adapter.final_v3_mano_unit_bones_local(world[human, frame], handedness=SIDE_NAMES[human])
        q = np.clip(previous.copy(), hand_lower[physical], hand_upper[physical])
        detail: dict[str, Any] = {}
        for finger in FINGERS:
            group = HAND_GROUPS[finger]
            if args.fast_hand_retarget:
                target_directions = target[finger]
                bend = np.arccos(np.clip(np.sum(target_directions[:-1] * target_directions[1:], axis=1), -1.0, 1.0))
                closure = float(np.clip(np.mean(bend) / (0.55 * np.pi), 0.0, 1.0))
                desired = hand_lower[physical][group] + closure * (hand_upper[physical][group] - hand_lower[physical][group])
                if static_initial:
                    q[group] = np.clip(
                        desired,
                        hand_lower[physical][group],
                        hand_upper[physical][group],
                    )
                else:
                    q[group] = np.clip(
                        previous[group]
                        + np.clip(
                            desired - previous[group],
                            -hand_solver_step_limit,
                            hand_solver_step_limit,
                        ),
                        hand_lower[physical][group],
                        hand_upper[physical][group],
                    )
                actual = hand_finger_directions(hand_models[physical], q, finger, hand_names[physical], hand_basis[physical], official, adapter, prefixes[physical])
                degrees = np.degrees(np.arccos(np.clip(np.sum(actual * target_directions, axis=1), -1.0, 1.0)))
                detail[finger] = {"mean_bone_error_deg": float(np.mean(degrees)), "max_bone_error_deg": float(np.max(degrees)), "bone_errors_deg": degrees.tolist(), "seed": "previous_bounded_analytic", "nfev": 0, "score": float(np.linalg.norm(actual - target_directions))}
                continue
            def residual(local: np.ndarray) -> np.ndarray:
                candidate = q.copy()
                candidate[group] = local
                actual = hand_finger_directions(hand_models[physical], candidate, finger, hand_names[physical], hand_basis[physical], official, adapter, prefixes[physical])
                return np.concatenate(((actual - target[finger]).ravel(), 0.04 * (local - previous[group])))
            attempts = []
            solve_lower = hand_lower[physical][group]
            solve_upper = hand_upper[physical][group]
            if not static_initial:
                solve_lower, solve_upper = bounded_temporal_limits(
                    solve_lower,
                    solve_upper,
                    previous[group],
                    None if previous_previous is None else previous_previous[group],
                    hand_solver_step_limit,
                    0.06,
                )
            seeds = (("previous", q[group]),) if args.strict_previous_accepted else (
                ("previous", q[group]),
                ("limit_midpoint", hand_neutral[physical, group]),
            )
            for seed_name, seed in seeds:
                solved = least_squares(
                    residual,
                    np.clip(seed, solve_lower, solve_upper),
                    bounds=(solve_lower, solve_upper),
                    max_nfev=100,
                )
                attempts.append((float(np.linalg.norm(residual(solved.x))), seed_name, solved.x.copy(), int(solved.nfev)))
            score, seed_name, selected, nfev = min(attempts, key=lambda row: (row[0], row[1]))
            q[group] = selected
            actual = hand_finger_directions(hand_models[physical], q, finger, hand_names[physical], hand_basis[physical], official, adapter, prefixes[physical])
            degrees = np.degrees(np.arccos(np.clip(np.sum(actual * target[finger], axis=1), -1.0, 1.0)))
            detail[finger] = {"mean_bone_error_deg": float(np.mean(degrees)), "max_bone_error_deg": float(np.max(degrees)), "bone_errors_deg": degrees.tolist(), "seed": seed_name, "nfev": nfev, "score": score}
        return q, detail

    previous_arm = arm_neutral.copy()
    previous_hand = hand_neutral.copy()
    previous_previous_arm: np.ndarray | None = None
    previous_previous_hand: np.ndarray | None = None
    pose_position_limit_mm = 40.0 if successor_contract is not None else 10.0
    pose_rotation_limit_deg = 20.0 if successor_contract is not None else 5.0
    q_arm_rows: list[np.ndarray] = []
    q_hand_rows: list[np.ndarray] = []
    pose_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    accepted_frame_rows: list[dict[str, Any]] = []
    for trajectory_slot, frame in enumerate(temporal_frames):
        static_initial = bool(
            trajectory_slot == 0 and args.first_frame_static_exception
        )
        frame_arm, frame_hand = [], []
        for physical in range(2):
            human = PHYSICAL_TO_HUMAN[physical]
            target = target_root(int(frame), human)
            target_tool = target @ np.linalg.inv(mounts[physical])
            attempts = []
            if successor_contract is not None and static_initial:
                if frame0_arm is None:
                    raise RuntimeError("missing frozen frame0 arm solution")
                q_arm = frame0_arm[physical].copy()
                actual = world_rig @ official._tool_fk(assets, physical, q_arm) @ mounts[physical]
                position_mm, rotation_deg = pose_error(actual, target, official)
                score = (position_mm / 10.0) ** 2 + (rotation_deg / 10.0) ** 2
                seed_name = "FROZEN_FRAME0_NOMINAL_IK_WITH_DATA_DERIVED_VISUAL_PLACEMENT"
                evaluations = 0
            else:
                arm_seeds = (("previous_fixed_branch", previous_arm[physical]),) if args.fixed_previous_ik_branch else (("previous", previous_arm[physical]), ("limit_midpoint", arm_neutral[physical]))
                for seed_name, seed in arm_seeds:
                    bound_to_previous = args.fixed_previous_ik_branch and not static_initial
                    if bound_to_previous and successor_contract is not None:
                        solve_lower, solve_upper = bounded_temporal_limits(
                            arm_lower[physical],
                            arm_upper[physical],
                            previous_arm[physical],
                            None if previous_previous_arm is None else previous_previous_arm[physical],
                            arm_solver_step_limit,
                            0.06,
                        )
                    else:
                        solve_lower = np.maximum(
                            arm_lower[physical],
                            previous_arm[physical] - args.arm_step_limit,
                        ) if bound_to_previous else arm_lower[physical]
                        solve_upper = np.minimum(
                            arm_upper[physical],
                            previous_arm[physical] + args.arm_step_limit,
                        ) if bound_to_previous else arm_upper[physical]
                    q_position, n_position = official._solve_one_arm_position_only(assets, side=physical, base=world_rig, target_tool=target_tool, initial_q=np.clip(seed, solve_lower, solve_upper), lower=solve_lower, upper=solve_upper)
                    q, n_pose = official._solve_one_arm(assets, side=physical, base=world_rig, target_tool=target_tool, initial_q=q_position, lower=solve_lower, upper=solve_upper)
                    actual = world_rig @ official._tool_fk(assets, physical, q) @ mounts[physical]
                    position_mm, rotation_deg = pose_error(actual, target, official)
                    score = (position_mm / 10.0) ** 2 + (rotation_deg / 10.0) ** 2 + 0.02 * float(np.linalg.norm(q - previous_arm[physical]))
                    attempts.append((score, seed_name, q, position_mm, rotation_deg, n_position + n_pose))
                score, seed_name, q_arm, position_mm, rotation_deg, evaluations = min(attempts, key=lambda row: (row[0], row[1]))
            q_hand, hand_detail = retarget_hand(
                int(frame),
                human,
                previous_hand[physical],
                None if previous_previous_hand is None else previous_previous_hand[physical],
                static_initial=static_initial,
            )
            arm_limit_pass = bool(np.all(q_arm >= arm_lower[physical] - 1e-9) and np.all(q_arm <= arm_upper[physical] + 1e-9))
            hand_limit_pass = bool(np.all(q_hand >= hand_lower[physical] - 1e-9) and np.all(q_hand <= hand_upper[physical] + 1e-9))
            pose_rows.append({"local_frame": int(frame), "source_frame": int(original_frames[frame]), "physical_side": SIDE_NAMES[physical], "human_side": SIDE_NAMES[human], "position_mm": position_mm, "rotation_deg": rotation_deg, "pose_position_limit_mm": pose_position_limit_mm, "pose_rotation_limit_deg": pose_rotation_limit_deg, "pose_gate_pass": bool(position_mm <= pose_position_limit_mm and rotation_deg <= pose_rotation_limit_deg), "pose_10mm_5deg_pass": bool(position_mm <= 10.0 and rotation_deg <= 5.0), "arm_limits_pass": arm_limit_pass, "seed": seed_name, "solver_evaluations": int(evaluations), "score": score})
            all_errors = [value for details in hand_detail.values() for value in (details["mean_bone_error_deg"], details["max_bone_error_deg"])]
            shape_audit = gradeb_hand_shape_gate(hand_detail, successor_contract) if successor_contract is not None else None
            hand_rows.append({"local_frame": int(frame), "source_frame": int(original_frames[frame]), "physical_side": SIDE_NAMES[physical], "human_side": SIDE_NAMES[human], "limits_pass": hand_limit_pass, "mean_and_max_error_deg_max": float(max(all_errors)), "shape_gate_pass": bool(shape_audit["pass"] if shape_audit is not None else max(all_errors) <= 45.0), "shape_gate": shape_audit, "fingers": hand_detail})
            frame_arm.append(q_arm)
            frame_hand.append(q_hand)
        candidate_arm = np.asarray(frame_arm)
        candidate_hand = np.asarray(frame_hand)
        frame_pose_rows = pose_rows[-2:]
        frame_hand_rows = hand_rows[-2:]
        arm_step = 0.0 if static_initial else float(
            np.max(np.abs(candidate_arm - previous_arm))
        )
        hand_step = 0.0 if static_initial else float(
            np.max(np.abs(candidate_hand - previous_hand))
        )
        arm_second_difference = 0.0 if previous_previous_arm is None else float(
            np.max(np.abs(candidate_arm - 2.0 * previous_arm + previous_previous_arm))
        )
        hand_second_difference = 0.0 if previous_previous_hand is None else float(
            np.max(np.abs(candidate_hand - 2.0 * previous_hand + previous_previous_hand))
        )
        accepted = bool(
            all(
                row["pose_gate_pass"] and row["arm_limits_pass"]
                for row in frame_pose_rows
            )
            and all(
                row["limits_pass"] and row["shape_gate_pass"]
                for row in frame_hand_rows
            )
            and arm_step <= args.arm_step_limit + 1e-9
            and hand_step <= args.hand_step_limit + 1e-9
            and (
                successor_contract is None
                or (
                    arm_second_difference <= 0.06 + 1e-9
                    and hand_second_difference <= 0.06 + 1e-9
                )
            )
        )
        accepted_frame_rows.append(
            {
                "trajectory_slot": trajectory_slot,
                "local_frame": int(frame),
                "source_frame": int(original_frames[frame]),
                "static_first_frame_exception": static_initial,
                "seed_policy": "PREVIOUS_ACCEPTED_ONLY",
                "arm_step_max_rad": arm_step,
                "hand_step_max_rad": hand_step,
                "arm_solver_effective_step_limit_rad": arm_solver_step_limit,
                "hand_solver_effective_step_limit_rad": hand_solver_step_limit,
                "arm_second_difference_max_rad": arm_second_difference,
                "hand_second_difference_max_rad": hand_second_difference,
                "accepted": accepted,
            }
        )
        if args.strict_previous_accepted and not accepted:
            side_diagnostics = []
            for physical, (pose_row, hand_row) in enumerate(
                zip(frame_pose_rows, frame_hand_rows, strict=True)
            ):
                human = PHYSICAL_TO_HUMAN[physical]
                target = target_root(int(frame), human)
                actual = (
                    world_rig
                    @ official._tool_fk(assets, physical, candidate_arm[physical])
                    @ mounts[physical]
                )
                failed_gates = []
                if not pose_row["pose_gate_pass"]:
                    failed_gates.append("POSE_FROZEN_ENDPOINT_GATE")
                if not pose_row["arm_limits_pass"]:
                    failed_gates.append("ARM_URDF_LIMITS")
                if not hand_row["limits_pass"]:
                    failed_gates.append("HAND_URDF_LIMITS")
                if not hand_row["shape_gate_pass"]:
                    failed_gates.append("HAND_SHAPE_FROZEN_GRADEB_GATE")
                side_diagnostics.append(
                    {
                        "physical_side": SIDE_NAMES[physical],
                        "human_side": SIDE_NAMES[human],
                        "failed_gates": failed_gates,
                        "pose": pose_row,
                        "hand": hand_row,
                        "target_root_world": target.tolist(),
                        "actual_root_world": actual.tolist(),
                        "candidate_arm_q": candidate_arm[physical].tolist(),
                        "candidate_hand_q": candidate_hand[physical].tolist(),
                        "previous_arm_q": previous_arm[physical].tolist(),
                        "previous_hand_q": previous_hand[physical].tolist(),
                    }
                )
            atomic_json(
                output / "REJECTED_FRAME_DIAGNOSTIC.json",
                {
                    "schema_version": "tianji-kai-rejected-frame-diagnostic-v1",
                    "task_id": args.task_id,
                    "session_id": args.session_id,
                    "trajectory_slot": trajectory_slot,
                    "local_frame": int(frame),
                    "source_frame": int(original_frames[frame]),
                    "static_first_frame_exception": static_initial,
                    "arm_step_max_rad": arm_step,
                    "hand_step_max_rad": hand_step,
                    "arm_step_limit_rad": args.arm_step_limit,
                    "hand_step_limit_rad": args.hand_step_limit,
                    "arm_second_difference_max_rad": arm_second_difference,
                    "hand_second_difference_max_rad": hand_second_difference,
                    "second_difference_limit_rad": 0.06 if successor_contract is not None else None,
                    "world_rig": world_rig.tolist(),
                    "human_to_physical": {"left": "right", "right": "left"},
                    "gradeb_successor_contract": evidence(successor_contract_path) if successor_contract_path is not None else None,
                    "frame0_static_placement_audit": placement_audit,
                    "claim_limit": "Frame0-data-derived static development visualization placement; not a real-robot measurement, calibration, or deployment authority. Pre/post residuals are separate.",
                    "side_diagnostics": side_diagnostics,
                    "accepted": False,
                },
            )
            raise RuntimeError(
                "strict previous-accepted frame gate failed at "
                f"source frame {int(original_frames[frame])}"
            )
        if trajectory_slot >= 1:
            previous_previous_arm = previous_arm.copy()
            previous_previous_hand = previous_hand.copy()
        previous_arm = candidate_arm
        previous_hand = candidate_hand
        q_arm_rows.append(previous_arm.copy())
        q_hand_rows.append(previous_hand.copy())
        print(json.dumps({"event": "SOLVED", "frame": int(frame), "source_frame": int(original_frames[frame])}), flush=True)

    q_arm = np.asarray(q_arm_rows, dtype=np.float64)
    q_hand = np.asarray(q_hand_rows, dtype=np.float64)
    arm_velocity = np.diff(q_arm, axis=0)
    hand_velocity = np.diff(q_hand, axis=0)
    arm_acceleration = np.diff(q_arm, n=2, axis=0)
    hand_acceleration = np.diff(q_hand, n=2, axis=0)
    temporal = {
        "fps": fps,
        "arm_velocity_max_rad_per_frame": float(np.max(np.abs(arm_velocity))),
        "arm_velocity_max_rad_per_second": float(np.max(np.abs(arm_velocity)) * fps),
        "hand_velocity_max_rad_per_frame": float(np.max(np.abs(hand_velocity))),
        "hand_velocity_max_rad_per_second": float(np.max(np.abs(hand_velocity)) * fps),
        "arm_acceleration_max_rad_per_frame2": float(np.max(np.abs(arm_acceleration))),
        "hand_acceleration_max_rad_per_frame2": float(np.max(np.abs(hand_acceleration))),
        "arm_velocity_limit_rad_per_frame": args.arm_step_limit,
        "hand_velocity_limit_rad_per_frame": args.hand_step_limit,
        "arm_solver_effective_velocity_limit_rad_per_frame": arm_solver_step_limit,
        "hand_solver_effective_velocity_limit_rad_per_frame": hand_solver_step_limit,
        "acceleration_limit_rad_per_frame2": 0.06,
        "one_step_stop_viability_by_velocity_cap": bool(
            arm_solver_step_limit <= 0.06 and hand_solver_step_limit <= 0.06
        ),
    }
    temporal["pass"] = bool(
        temporal["arm_velocity_max_rad_per_frame"] <= args.arm_step_limit + 1e-9
        and temporal["hand_velocity_max_rad_per_frame"] <= args.hand_step_limit + 1e-9
        and max(
            temporal["arm_acceleration_max_rad_per_frame2"],
            temporal["hand_acceleration_max_rad_per_frame2"],
        ) <= 0.06
    )

    self_sat_rows = []
    if not args.skip_expensive_self_collision:
        mesh_cache = mesh_cache_for_models((assets.tianji, *hand_models))
        for slot in review_slots:
            frame = temporal_frames[int(slot)]
            for physical in range(2):
                row = distinct_finger_self_sat(hand_models[physical], q_hand[int(slot), physical], hand_names[physical], official, mesh_cache, triangle_triangle_intersects_sat)
                row.update({"local_frame": int(frame), "source_frame": int(original_frames[frame]), "physical_side": SIDE_NAMES[physical]})
                self_sat_rows.append(row)

    panels: dict[int, np.ndarray] = {}
    overlay_root = output / "robot_overlay_frames"
    overlay_root.mkdir()
    visual_cache: dict[Path, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    render_slots = review_slots if args.sparse_review else np.arange(window_frames, dtype=np.int32)
    for slot in render_slots:
        frame = temporal_frames[int(slot)]
        source_frame = int(original_frames[frame])
        image_path = raw_root / "preprocess/all_data" / f"{source_frame:05d}" / "rgb.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"missing RGB frame: {image_path}")
        source_height, source_width = image.shape[:2]
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = WIDTH / float(source_width), HEIGHT / float(source_height)
        k = intrinsics[frame].copy()
        k[0] *= scale_x
        k[1] *= scale_y
        camera_base = np.linalg.inv(c2w[frame]) @ world_rig
        geometry = []
        values = {name: 0.0 for names in official.ARM_JOINT_NAMES for name in names}
        for physical in range(2):
            values.update({name: float(value) for name, value in zip(official.ARM_JOINT_NAMES[physical], q_arm[slot, physical], strict=True)})
        arm_fk = official.forward_kinematics(assets.tianji, values)
        for physical, suffix in ((0, "L"), (1, "R")):
            tool = "left_tool" if physical == 0 else "right_tool"
            arm_triangles, arm_colors, arm_labels, _ = raster.camera_triangles(assets.tianji, arm_fk, camera_base, lambda link, suffix=suffix, tool=tool: link.endswith(f"_{suffix}") or link == tool, visual_cache)
            geometry.append((arm_triangles, arm_colors, arm_labels))
            hand_root = camera_base @ arm_fk[tool] @ mounts[physical]
            hand_fk = official.forward_kinematics(hand_models[physical], dict(zip(hand_names[physical], q_hand[slot, physical], strict=True)))
            hand_triangles, hand_colors, hand_labels, _ = raster.camera_triangles(hand_models[physical], hand_fk, hand_root, lambda _link: True, visual_cache)
            geometry.append((hand_triangles, hand_colors, hand_labels + 100 * (physical + 1)))
        triangles = np.concatenate([row[0] for row in geometry if len(row[0])])
        colors = np.concatenate([row[1] for row in geometry if len(row[0])])
        labels = np.concatenate([row[2] for row in geometry if len(row[0])])
        _, rendered, link_id = raster.rasterize_zbuffer(triangles, colors, labels, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), WIDTH, HEIGHT)
        alpha = link_id >= 0
        overlay = image.copy()
        overlay[alpha] = np.clip(0.18 * image[alpha] + 0.82 * rendered[alpha], 0, 255).astype(np.uint8)
        uv = np.asarray(data["joints_2d"][:, frame], dtype=np.float64) * np.asarray((scale_x, scale_y))
        for human in range(2):
            for chain in MANO_CHAINS.values():
                points = np.rint(uv[human, np.asarray(chain)]).astype(np.int32)
                cv2.polylines(overlay, [points], False, (70, 255, 80), 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (WIDTH, 86), (0, 0, 0), -1)
        cv2.putText(overlay, f"{args.task_id} {args.session_id} | source f{source_frame:05d} | CPU KINEMATIC CANARY", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "NO_OBJECT6D | NO CONTACT CLAIM | CLEAN PENDING", (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 80, 255), 2, cv2.LINE_AA)
        mount_label = (
            "FRAME0-DERIVED STATIC VISUAL PLACEMENT | NOT REAL-ROBOT CALIBRATION/DEPLOYMENT"
            if successor_contract is not None
            else "GREEN=HaWoR MANO21 | CAD=actual FK + development-only corrected mount"
        )
        cv2.putText(overlay, mount_label, (12, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1, cv2.LINE_AA)
        panels[int(slot)] = overlay
        if not cv2.imwrite(str(overlay_root / f"{source_frame:05d}.png"), overlay):
            raise RuntimeError(f"failed to write Robot overlay frame {source_frame}")

    sheet12_path = output / "ROBOT_KINEMATIC_REVIEW12.png"
    sheet24_path = output / "ROBOT_KINEMATIC_TEMPORAL24.png"
    if not cv2.imwrite(str(sheet12_path), make_sheet([panels[int(index)] for index in review_slots], 4, (480, 360))):
        raise RuntimeError("failed to write 12-frame sheet")
    ordered_panels = [panels[int(index)] for index in render_slots]
    if not cv2.imwrite(str(sheet24_path), make_sheet(ordered_panels, 6, (320, 240))):
        raise RuntimeError("failed to write temporal sheet")
    overlay_manifest_path = output / "ROBOT_OVERLAY_FRAME_MANIFEST.json"
    atomic_json(overlay_manifest_path, {
        "schema_version": "robot-overlay-frame-manifest-v1",
        "session_id": args.session_id,
        "source_frames": [int(original_frames[temporal_frames[int(index)]]) for index in render_slots],
        "sparse_review": bool(args.sparse_review),
        "frames": [evidence(overlay_root / f"{int(original_frames[temporal_frames[int(index)]]):05d}.png") for index in render_slots],
    })
    scene_path = output / "KINEMATIC_CANARY24.npz"
    np.savez_compressed(scene_path, local_frames=temporal_frames, source_frames=original_frames[temporal_frames], q_arm=q_arm, q_hand=q_hand, T_world_rig=world_rig, T_tool_hand=mounts, T_tool_hand_nominal=nominal_mounts, c2w=c2w[temporal_frames], intrinsics=intrinsics[temporal_frames], mano_joint_names=np.asarray(MANO_NAMES), human_to_physical=np.asarray((1, 0), dtype=np.int32))

    pose_pass = bool(all(row["pose_gate_pass"] and row["arm_limits_pass"] for row in pose_rows))
    hand_pass = bool(all(row["limits_pass"] and row["shape_gate_pass"] for row in hand_rows))
    self_sat_pass = bool(self_sat_rows and all(row["pass"] for row in self_sat_rows))
    gates = {
        "G0_source_terminal": source_admission.get("combined", {}).get(
            "upstream_terminal_status", "UNKNOWN"
        ),
        "G1_mano21_order_and_direct_provenance": "PASS",
        "G2_first24_bilateral_direct_initializer": "PASS",
        "G3_session_static_initializer": "PASS_VISUAL_SURROGATE_NOT_CALIBRATION_AUTHORITY",
        "G4_arm_actual_fk_pose_limits": "PASS" if pose_pass else "HOLD_IK_POSE_OR_LIMIT",
        "G5_kai_unit_bone_retarget_limits": "PASS" if hand_pass else "HOLD_HAND_SHAPE_OR_LIMIT",
        "G6_contiguous_velocity_acceleration": "PASS" if temporal["pass"] else "HOLD_TEMPORAL_LIMIT",
        "G7_distinct_finger_exact_visual_triangle_self_sat_12frame": "PASS" if self_sat_pass else ("NOT_EVALUATED_PRODUCTION_USES_LINK_PROXY" if args.skip_expensive_self_collision else "HOLD_SELF_COLLISION"),
        "G8_object6d_contact_nonpenetration": "NOT_EVALUATED_NO_OBJECT6D",
        "G9_mask_clean_authority": "NOT_EVALUATED_MASK_AND_CLEAN_PENDING",
    }
    payload = {
        "schema_version": "newtask-robot-kinematic-admission-v1",
        "status": "HOLD_NO_OBJECT6D_CONTACT_MASK_CLEAN_AUTHORITY",
        "task_id": args.task_id,
        "session_id": args.session_id,
        "formal_robot_ready": False,
        "visual_kinematic_candidate": bool(pose_pass and hand_pass and temporal["pass"] and self_sat_pass),
        "gates": gates,
        "initializer": {"method": "FIRST24_BILATERAL_DIRECT_MANO21_ONE_SESSION_STATIC_RIG_FROM_LIMIT_MIDPOINT_NEUTRAL", "frames": initializer_frames.tolist(), "T_world_rig": world_rig.tolist(), "copied_cap004_base": False, "copied_cap004_q": False, "per_frame_base": False},
        "gradeb_successor_contract": evidence(successor_contract_path) if successor_contract_path is not None else None,
        "frame0_static_placement": {
            "enabled": successor_contract is not None,
            "frozen_for_session": successor_contract is not None,
            "source_frame": 0 if successor_contract is not None else None,
            "audit": placement_audit,
            "development_visualization_only": successor_contract is not None,
            "real_robot_measurement_or_calibration": False,
            "pre_and_post_residuals_reported_separately": successor_contract is not None,
        },
        "side_contract": {"mano_side_axis": ["anatomical_left", "anatomical_right"], "human_to_physical": {"left": "right", "right": "left"}, "image_x_order_inference": False},
        "accepted_hawor_provenance": list(accepted_provenance),
        "mount_authority": mount_authority,
        "thumb_adapter": evidence(args.thumb_adapter.resolve(strict=True)) if args.thumb_adapter is not None else None,
        "neutral_q_authority": "DETERMINISTIC_PINNED_URDF_JOINT_LIMIT_MIDPOINT_NOT_OLD_SESSION_Q",
        "temporal_window": {"local_frames": temporal_frames.tolist(), "source_frames": original_frames[temporal_frames].tolist(), "selection": "24_CONTIGUOUS_DIRECT_FRAMES_NEAREST_MAX_BILATERAL_WRIST_STEP"},
        "pose_rows": pose_rows,
        "hand_rows": hand_rows,
        "temporal": temporal,
        "previous_accepted_contract": {
            "enabled": bool(args.strict_previous_accepted),
            "first_frame_policy": (
                "STATIC_IK_EXCEPTION_FULL_URDF_LIMITS"
                if args.first_frame_static_exception
                else "BOUNDED_FROM_NEUTRAL"
            ),
            "later_frame_seed_policy": "PREVIOUS_ACCEPTED_ONLY",
            "arm_step_limit_rad_per_frame": args.arm_step_limit,
            "hand_step_limit_rad_per_frame": args.hand_step_limit,
            "arm_solver_effective_step_limit_rad_per_frame": arm_solver_step_limit,
            "hand_solver_effective_step_limit_rad_per_frame": hand_solver_step_limit,
            "one_step_stop_viability_by_velocity_cap": bool(
                arm_solver_step_limit <= 0.06 and hand_solver_step_limit <= 0.06
            ),
            "frames": accepted_frame_rows,
            "branch_jump": bool(
                any(not row["accepted"] for row in accepted_frame_rows)
            ),
        },
        "self_sat_rows": self_sat_rows,
        "object6d_available": False,
        "contact_claim": False,
        "nonpenetration_claim": False,
        "clean_consumed": False,
        "mask_consumed": False,
        "full_arm_self_collision": "NOT_EVALUATED_NO_PINNED_COLLISION_MESH_SEMANTIC_EXCLUSION_SET",
        "upstream": {"hawor_npz": evidence(hawor_path), "hawor_result": evidence(hawor_result_path), "source_admission": evidence(admission_path), "hawor_numeric_candidate": result_upstream.get("gates", {}).get("robot_numeric_candidate")},
        "outputs": {"scene": evidence(scene_path), "review12": evidence(sheet12_path), "temporal24": evidence(sheet24_path), "overlay_frame_manifest": evidence(overlay_manifest_path)},
        "claim_limit": "CPU actual-FK and KaiHand visual-mesh kinematic admission only. Any frame0-derived static placement is development visualization only, not a real-robot measurement, calibration, or deployment authority; post-placement residual is not original IK accuracy. No metric Object6D, contact, object nonpenetration, full-arm collision, Clean, formal RobotRGB or training claim.",
        "gpu_calls": 0,
        "wall_seconds": time.time() - started,
        "producer": evidence(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", payload)
    print(json.dumps({"event": "COMPLETE", "status": payload["status"], "visual_kinematic_candidate": payload["visual_kinematic_candidate"], "review12": str(sheet12_path), "temporal24": str(sheet24_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
