#!/usr/bin/env python3
"""Build one fail-closed HumanEgo+ICT bundle from one current E2E session.

This is deliberately not an exact78 builder.  It admits exactly one of the two
20260908 baseline sessions and separates train/validation by disjoint temporal
H50 support inside that session.  The validation split is optimization
monitoring only; it is not a cross-session or heldout generalization claim.

No GPU work or training is performed here.  When a current Robot A/B authority
does not exist, ``--wait-receipt`` publishes a small honest wait receipt and
returns successfully without creating a bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
HUMANEGO = PROJECT / "HumanEgo"
CONTROL = PROJECT / "tasks/control/runs/20260908_two_task_e2e_baseline_v1"
AUTHORITY = CONTROL / "BASELINE_AUTHORITY.json"
REBIND = CONTROL / "TRAINING_RECIPE_REBIND_CONTRACT.json"
HAWOR_BUILDER = PROJECT / "NOW/daemon/tools/build_robot_humanego_ict_bundle_v2.py"
HORIZON = 50
ROLE_MIN_FRAMES = HORIZON + 1
TASKS = {
    "chips": "get_potato_chips_0902_034",
    "poker": "play_cards_0902_042",
}
OBJECT_DIRECT = "VISUAL_RGB_MASK_CORRECTED_STEREO_DEPTH"
OBJECT_PROPAGATED = "VISUAL_OCCLUSION_WORLD_POSE_PROPAGATED"
OBJECT_THRESHOLDS = {OBJECT_DIRECT: 0.20, OBJECT_PROPAGATED: 0.20}
FORBIDDEN_DATA_LINEAGE_MARKERS = (
    "exact78",
    "fresh78",
    "/humanego/artifacts/newtask_robot_bundles/",
    "fail_forward_prod",
    "fallback",
    "robot c",
    "robot_c",
)


class BundleHold(RuntimeError):
    """A fail-closed, user-actionable admission hold."""


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ref(path: Path, *, published: Path | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise BundleHold(f"HOLD_NOT_ORDINARY_FILE:{path}")
    return {
        "path": str((published or path).absolute()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BundleHold(f"HOLD_JSON_OBJECT_REQUIRED:{path}")
    return value


def verify_ref(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(str(value["path"])).resolve(strict=True)
        expected_bytes = int(value["bytes"])
        expected_sha = str(value["sha256"])
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise BundleHold(f"HOLD_INVALID_REFERENCE:{label}") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected_bytes
        or sha256(path) != expected_sha
    ):
        raise BundleHold(f"HOLD_REFERENCE_DRIFT:{label}:{path}")
    return path


def verify_pin(value: Mapping[str, Any], label: str) -> Path:
    """Verify a rebind pin; historical pins may omit the redundant byte count."""
    try:
        path = Path(str(value["path"])).resolve(strict=True)
        expected_sha = str(value["sha256"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise BundleHold(f"HOLD_INVALID_PIN:{label}") from exc
    if not path.is_file() or path.is_symlink() or sha256(path) != expected_sha:
        raise BundleHold(f"HOLD_PIN_DRIFT:{label}:{path}")
    if "bytes" in value and path.stat().st_size != int(value["bytes"]):
        raise BundleHold(f"HOLD_PIN_SIZE_DRIFT:{label}:{path}")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise BundleHold(f"HOLD_NO_CLOBBER:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_ordinary(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise BundleHold(f"HOLD_SOURCE_NOT_ORDINARY:{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
    if destination.stat().st_nlink != 1:
        raise BundleHold(f"HOLD_BUNDLE_NLINK_NOT_ONE:{destination}")


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _assert_current_data_lineage(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = str(path).lower()
        marker = next(
            (value for value in FORBIDDEN_DATA_LINEAGE_MARKERS if value in lowered),
            None,
        )
        if marker is not None:
            raise BundleHold(
                f"HOLD_OLD_EXACT78_C_OR_FALLBACK_DATA_LINEAGE:{marker}:{path}"
            )


def _authority_robot_row(
    authority: Mapping[str, Any], *, task: str, robot_result: Path
) -> dict[str, Any]:
    robot = authority.get("stage_authorities", {}).get("robot", {})
    expected = robot_result.resolve(strict=True)
    candidates: list[dict[str, Any]] = []
    direct = robot.get(task) if isinstance(robot, dict) else None
    if isinstance(direct, dict):
        candidates.append(direct)
    if isinstance(robot, dict):
        candidates.extend(_walk_dicts(robot))
    seen: set[int] = set()
    for row in candidates:
        if id(row) in seen:
            continue
        seen.add(id(row))
        path_value = row.get("result")
        result_path = None
        result_sha = row.get("result_sha256")
        if isinstance(path_value, str):
            result_path = Path(path_value)
        elif isinstance(path_value, dict):
            result_path = Path(str(path_value.get("path", "")))
            result_sha = path_value.get("sha256", result_sha)
        try:
            resolved = result_path.resolve(strict=True) if result_path else None
        except OSError:
            continue
        if resolved != expected:
            continue
        if result_sha != sha256(expected):
            raise BundleHold("HOLD_AUTHORITY_ROBOT_RESULT_SHA_MISMATCH")
        if row.get("grade") not in {"A", "B"}:
            raise BundleHold("HOLD_AUTHORITY_ROBOT_NOT_GRADE_A_OR_B")
        if row.get("downstream_authorized") is not True:
            raise BundleHold("HOLD_AUTHORITY_ROBOT_NOT_DOWNSTREAM_AUTHORIZED")
        return row
    raise BundleHold("WAIT_FRESH_ROBOT_A_OR_B_AUTHORITY_BINDING")


def _verified_stage_review_input(review: Mapping[str, Any], name: str) -> Path:
    value = review.get("inputs", {}).get(name)
    if not isinstance(value, dict):
        raise BundleHold(f"HOLD_ROBOT_REVIEW_MISSING_INPUT:{name}")
    path = verify_ref(value, f"Robot review {name}")
    stage = read_json(path)
    stage_grade = stage.get("grade")
    stage_status = str(stage.get("status", ""))
    authorized = stage.get("consumption_authorized")
    result_has_grade = (
        stage_grade in {"A", "B"}
        or "GRADE_A" in stage_status
        or "GRADE_B" in stage_status
    )
    if not result_has_grade:
        sibling_review = path.parent / "AGENT_REVIEW.json"
        if not sibling_review.is_file() or sibling_review.is_symlink():
            raise BundleHold(f"HOLD_UPSTREAM_NOT_GRADE_A_OR_B:{name}")
        review = read_json(sibling_review)
        if (
            review.get("grade") not in {"A", "B"}
            or review.get("downstream_authorized") is not True
        ):
            raise BundleHold(f"HOLD_UPSTREAM_REVIEW_NOT_GRADE_A_OR_B:{name}")
    if authorized is False:
        raise BundleHold(f"HOLD_UPSTREAM_NOT_AUTHORIZED:{name}")
    return path


def validate_robot_lineage(
    *, task: str, session: str, robot_result_path: Path, authority_path: Path
) -> dict[str, Any]:
    authority = read_json(authority_path.resolve(strict=True))
    authority_row = _authority_robot_row(
        authority, task=task, robot_result=robot_result_path
    )
    robot_result = read_json(robot_result_path)
    if (
        robot_result.get("schema_version") != "tianji-kai-robot-baseline-result-v1"
        or robot_result.get("task") != task
        or robot_result.get("session_id") != session
        or robot_result.get("grade") not in {"A", "B"}
        or robot_result.get("consumption_authorized") is not True
        or robot_result.get("hard_gates_pass") is not True
    ):
        raise BundleHold("WAIT_FRESH_ROBOT_A_OR_B_RESULT")
    review_path = robot_result_path.parent / "AGENT_REVIEW.json"
    review = read_json(review_path.resolve(strict=True))
    if (
        review.get("stage") != "ROBOT"
        or review.get("task") != task
        or review.get("session") != session
        or review.get("grade") not in {"A", "B"}
        or review.get("downstream_authorized") is not True
    ):
        raise BundleHold("HOLD_ROBOT_AGENT_REVIEW_NOT_A_OR_B")
    result_ref = review.get("artifacts", {}).get("result")
    if not isinstance(result_ref, dict) or verify_ref(result_ref, "Robot review result") != robot_result_path.resolve(strict=True):
        raise BundleHold("HOLD_ROBOT_REVIEW_RESULT_BINDING")
    upstream = {
        name: _verified_stage_review_input(review, name)
        for name in ("hawor_result", "mask_result", "object6d_result", "clean_result")
    }
    preflight = read_json(verify_ref(robot_result["preflight"], "Robot preflight"))
    spec = read_json(verify_ref(preflight["spec"], "Robot session spec"))
    if spec.get("session", {}).get("session_id") != session:
        raise BundleHold("HOLD_ROBOT_SPEC_SESSION_MISMATCH")
    for name, source in upstream.items():
        spec_name = "object6d" if name == "object6d_result" else name.removesuffix("_result")
        spec_ref = spec.get("inputs", {}).get(spec_name, {}).get("result")
        if not isinstance(spec_ref, dict) or verify_ref(spec_ref, f"Robot spec {name}") != source:
            raise BundleHold(f"HOLD_ROBOT_REVIEW_SPEC_LINEAGE_MISMATCH:{name}")
    kinematic_result_path = verify_ref(
        robot_result["kinematic_result"], "Robot kinematic result"
    )
    kinematic = read_json(kinematic_result_path)
    scene_path = verify_ref(kinematic["outputs"]["scene"], "Robot scene")
    hawor_npz = verify_ref(spec["inputs"]["hawor"]["npz"], "HaWoR NPZ")
    object_npz = verify_ref(spec["inputs"]["object6d"]["npz"], "Object6D NPZ")
    raw_video = verify_ref(spec["inputs"]["raw_video"], "raw video")
    mps_path = raw_video.parent
    raw_all_data = mps_path / "preprocess/all_data"
    if not raw_all_data.is_dir():
        raise BundleHold(f"HOLD_RAW_MPS_METADATA_ROOT_MISSING:{raw_all_data}")
    _assert_current_data_lineage((
        robot_result_path,
        review_path,
        kinematic_result_path,
        scene_path,
        upstream["hawor_result"],
        hawor_npz,
        upstream["mask_result"],
        upstream["object6d_result"],
        object_npz,
        upstream["clean_result"],
        raw_video,
        raw_all_data,
    ))
    return {
        "authority": authority,
        "authority_row": authority_row,
        "robot_result": robot_result,
        "robot_review": review,
        "robot_result_path": robot_result_path.resolve(strict=True),
        "robot_review_path": review_path.resolve(strict=True),
        "preflight_path": Path(robot_result["preflight"]["path"]).resolve(strict=True),
        "spec_path": Path(preflight["spec"]["path"]).resolve(strict=True),
        "kinematic_result_path": kinematic_result_path,
        "scene_path": scene_path,
        "hawor_result_path": upstream["hawor_result"],
        "hawor_npz": hawor_npz,
        "object6d_result_path": upstream["object6d_result"],
        "object6d_npz": object_npz,
        "mask_result_path": upstream["mask_result"],
        "clean_result_path": upstream["clean_result"],
        "raw_video": raw_video,
        "fps": float(spec["session"]["fps"]),
        "mps_path": mps_path,
        "raw_all_data": raw_all_data,
    }


def longest_true_span(values: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = 0
    flat = np.asarray(values, dtype=bool).reshape(-1)
    for index in range(len(flat) + 1):
        if index < len(flat) and flat[index]:
            continue
        if index - start > best[1] - best[0]:
            best = (start, index)
        start = index + 1
    return best


def temporal_split(valid: np.ndarray, frames: np.ndarray) -> dict[str, Any]:
    if valid.ndim != 2 or valid.shape[1] != 2 or len(valid) != len(frames):
        raise BundleHold("HOLD_TEMPORAL_SPLIT_INPUT_SHAPE")
    if not np.array_equal(frames, np.arange(len(frames), dtype=frames.dtype)):
        raise BundleHold("HOLD_ROBOT_FRAMES_NOT_EXACT_CONTIGUOUS_FULL_SESSION")
    start, stop = longest_true_span(valid.all(axis=1))
    length = stop - start
    if length < 2 * ROLE_MIN_FRAMES:
        raise BundleHold(
            f"HOLD_NO_TWO_DISJOINT_H50_TEMPORAL_BLOCKS:{length}<102"
        )
    train_length = int(np.floor(length * 2.0 / 3.0))
    train_length = max(ROLE_MIN_FRAMES, min(train_length, length - ROLE_MIN_FRAMES))
    train = (start, start + train_length)
    validation = (train[1], stop)

    def starts(span: tuple[int, int]) -> list[int]:
        first, exclusive = span
        return list(range(first, exclusive - HORIZON))

    train_starts = starts(train)
    validation_starts = starts(validation)
    train_support = set(range(train[0], train[1]))
    validation_support = set(range(validation[0], validation[1]))
    if not train_starts or not validation_starts or train_support & validation_support:
        raise BundleHold("HOLD_TEMPORAL_SPLIT_OVERLAP_OR_EMPTY")
    return {
        "policy": "LONGEST_DUAL_VALID_RUN_CHRONOLOGICAL_TWO_THIRDS_ONE_THIRD",
        "horizon": HORIZON,
        "train": {
            "frame_start": train[0],
            "frame_stop_exclusive": train[1],
            "support_frame_count": len(train_support),
            "window_starts": train_starts,
        },
        "validation": {
            "frame_start": validation[0],
            "frame_stop_exclusive": validation[1],
            "support_frame_count": len(validation_support),
            "window_starts": validation_starts,
        },
        "support_intersection": [],
        "validation_claim": (
            "WITHIN_SESSION_OPTIMIZATION_MONITORING_ONLY_NOT_CROSS_SESSION_"
            "GENERALIZATION_OR_CANONICAL_HELDOUT"
        ),
    }


def _load_hawor_builder():
    spec = importlib.util.spec_from_file_location(
        "single_session_hawor_builder", HAWOR_BUILDER
    )
    if spec is None or spec.loader is None:
        raise BundleHold("HOLD_CANNOT_IMPORT_HAWOR_SIDECAR_BUILDER")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rotation6d(rotation: np.ndarray) -> np.ndarray:
    # Preserve HumanEgo.utils.utils_math.rotmat_to_o6d byte semantics:
    # row-major flatten of the first two rotation columns.
    return np.asarray(rotation, dtype=np.float32)[:, :2].reshape(-1)


def build_robot_sidecar(
    *,
    scene_path: Path,
    hawor_sidecar: Path,
    destination: Path,
    session: str,
    fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not np.isfinite(fps) or fps <= 0:
        raise BundleHold(f"HOLD_ROBOT_SPEC_INVALID_FPS:{fps}")
    with np.load(scene_path, allow_pickle=False) as scene:
        required = {"source_frames", "q_hand", "human_to_physical"}
        if not required.issubset(scene.files):
            raise BundleHold(f"HOLD_ROBOT_SCENE_FIELDS:{sorted(required-set(scene.files))}")
        frames = np.asarray(scene["source_frames"], dtype=np.int64)
        q_hand = np.asarray(scene["q_hand"], dtype=np.float32)
        human_to_physical = np.asarray(scene["human_to_physical"], dtype=np.int64)
    count = len(frames)
    if (
        q_hand.shape != (count, 2, 22)
        or human_to_physical.tolist() != [1, 0]
        or not np.isfinite(q_hand).all()
    ):
        raise BundleHold("HOLD_ROBOT_SCENE_Q_OR_SIDE_MAPPING")
    with np.load(hawor_sidecar, allow_pickle=False) as hawor:
        names = np.asarray(hawor["frame_names"]).astype(str)
        wrist_human = np.asarray(hawor["T_hand_to_camera"], dtype=np.float64)
        valid_human = np.asarray(hawor["valid"], dtype=bool)
        confidence_human = np.asarray(hawor["confidence"], dtype=np.float32)
        grasp_human = np.asarray(hawor["grasp"], dtype=np.float32)
    if (
        len(names) != count
        or wrist_human.shape != (count, 2, 4, 4)
        or not np.array_equal(frames, np.asarray([int(value) for value in names]))
    ):
        raise BundleHold("HOLD_ROBOT_HAWOR_FRAME_IDENTITY")
    wrist = wrist_human[:, ::-1].copy()
    valid = valid_human[:, ::-1].copy()
    confidence = confidence_human[:, ::-1].copy()
    grasp = grasp_human[:, ::-1].copy()

    from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    assets = load_pinned_robot_assets(PROJECT)
    moving = []
    for model in (assets.left_hand, assets.right_hand):
        joints = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        if len(joints) != 22:
            raise BundleHold("HOLD_PINNED_KAIHAND_NOT_22_DOF")
        moving.append(joints)
    joint_names = np.asarray([[joint.name for joint in side] for side in moving])
    lower = np.asarray([[joint.lower for joint in side] for side in moving], dtype=np.float32)
    upper = np.asarray([[joint.upper for joint in side] for side in moving], dtype=np.float32)
    if np.any(q_hand < lower[None] - 1e-6) or np.any(q_hand > upper[None] + 1e-6):
        raise BundleHold("HOLD_ROBOT_Q_HAND_OUTSIDE_PINNED_URDF_LIMITS")
    hand_step = float(np.max(np.abs(np.diff(q_hand, axis=0)))) if count > 1 else 0.0
    if hand_step > 0.08 + 1e-6:
        raise BundleHold(f"HOLD_ROBOT_HAND_STEP:{hand_step}")
    wrist9 = np.empty((count, 2, 9), dtype=np.float32)
    for frame in range(count):
        for side in range(2):
            wrist9[frame, side, :3] = wrist[frame, side, :3, 3]
            wrist9[frame, side, 3:] = _rotation6d(wrist[frame, side, :3, :3])
    timestamps = np.arange(count, dtype=np.int64) * int(round(1e9 / fps))
    failure_reason = np.full((count, 2), "", dtype="<U1")
    hand_object = np.full((count, 2, 4, 4), np.nan, dtype=np.float64)
    canonical_sha = hashlib.sha256(
        ("|".join(joint_names.reshape(-1).tolist())).encode("utf-8")
    ).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray("humanego-robot-sidecar-v1"),
        embodiment=np.asarray("kai22"),
        session_id=np.asarray(session),
        frame_names=names,
        timestamps_ns=timestamps,
        wrist_9d=wrist9,
        wrist_T_camera=wrist,
        q=q_hand,
        valid=valid,
        confidence=confidence,
        grasp=grasp,
        joint_names=joint_names,
        joint_lower=lower,
        joint_upper=upper,
        canonical_sha256=np.asarray(canonical_sha),
        retarget_confidence=confidence,
        failure_reason=failure_reason,
        hand_object_T=hand_object,
        translation_unit=np.asarray("metre"),
        q_unit=np.asarray("radian"),
        ik_warm_start_used=np.asarray(True),
        achieved_robot_fk=np.asarray(True),
        human_to_physical=np.asarray([1, 0], dtype=np.int8),
        source_scene_sha256=np.asarray(sha256(scene_path)),
        source_hawor_sidecar_sha256=np.asarray(sha256(hawor_sidecar)),
    )
    os.replace(temporary, destination)
    from preprocess.retarget_labels.schema import EMBODIMENTS, validate_sidecar

    validation = validate_sidecar(destination, EMBODIMENTS["kai22"])
    split = temporal_split(valid, frames)
    return validation, split


def build_object_sidecar(
    *, source: Path, destination_dir: Path, published_dir: Path,
    session: str, grade: str,
) -> tuple[Path, Path, dict[str, Any]]:
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "frame_indices", "valid", "observed", "visibility",
            "T_object_to_camera",
        }
        if not required.issubset(archive.files):
            raise BundleHold(f"HOLD_OBJECT6D_FIELDS:{sorted(required-set(archive.files))}")
        frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        source_valid = np.asarray(archive["valid"], dtype=bool)
        observed = np.asarray(archive["observed"], dtype=bool)
        visibility = np.asarray(archive["visibility"], dtype=np.float32)
        source_transform = np.asarray(archive["T_object_to_camera"], dtype=np.float64)
    count = len(frames)
    if (
        not np.array_equal(frames, np.arange(count))
        or source_valid.shape != (count,)
        or observed.shape != (count,)
        or visibility.shape != (count,)
        or source_transform.shape != (count, 4, 4)
        or not source_valid.all()
    ):
        raise BundleHold("HOLD_OBJECT6D_NOT_DENSE_FULL_SESSION")
    transforms = np.full((count, 2, 4, 4), np.nan, dtype=np.float64)
    transforms[:, 0] = source_transform
    valid = np.zeros((count, 2), dtype=bool)
    valid[:, 0] = True
    formal_valid = np.zeros_like(valid)
    confidence = np.zeros((count, 2), dtype=np.float32)
    confidence[:, 0] = np.where(observed, np.maximum(visibility, 0.50), 0.25)
    provenance = np.full((count, 2), "UNKNOWN", dtype="<U64")
    provenance[:, 0] = np.where(observed, OBJECT_DIRECT, OBJECT_PROPAGATED)
    pose9 = np.full((count, 2, 9), np.nan, dtype=np.float32)
    for frame in range(count):
        pose9[frame, 0, :3] = transforms[frame, 0, :3, 3]
        pose9[frame, 0, 3:] = _rotation6d(transforms[frame, 0, :3, :3])
    destination_dir.mkdir(parents=True, exist_ok=False)
    npz_path = destination_dir / "AUTO_ESTIMATED_OBJECT_STATE.npz"
    np.savez_compressed(
        npz_path,
        frame_names=np.asarray([f"{value:05d}" for value in frames]),
        object_keys=np.asarray(["physical_object_0", "other_static_fixture_unknown"]),
        T_object_to_camera=transforms,
        valid=valid,
        confidence=confidence,
        provenance=provenance,
        role=np.asarray(["anchor_manipulated", "other_static_fixture"]),
        anchor_key=np.asarray("physical_object_0"),
        object_type_ids=np.asarray([3, 4], dtype=np.int8),
        object_pose9_camera=pose9,
        formal_object6d_valid=formal_valid,
    )
    json_path = destination_dir / "AUTO_ESTIMATED_OBJECT_STATE.json"
    manifest = {
        "schema_version": "humanego-auto-estimated-object-grade-b-v1",
        "session": session,
        "quality_grade": grade,
        "training_weight": 0.25 if grade == "B" else 1.0,
        "consumption_authorized": True,
        "claims_formal_object6d": False,
        "per_frame_anchor_required": True,
        "confidence_thresholds": OBJECT_THRESHOLDS,
        "array_contract": {
            "coordinate_frame": "current rectified left-camera optical frame",
            "units": {"translation": "metres", "rotation": "dimensionless 6D"},
        },
        "source_provenance": {
            "object6d_npz": ref(source),
            "method": (
                "current visual Object6D direct RGB-mask+corrected-depth poses plus "
                "explicit occlusion world-pose propagation; no contact authority"
            ),
        },
        "npz": ref(npz_path, published=published_dir / npz_path.name),
    }
    write_json(json_path, manifest)
    return npz_path, json_path, manifest


def _recipe_snapshot(path: Path = REBIND) -> dict[str, Any]:
    contract = read_json(path.resolve(strict=True))
    for group in ("source_of_truth", "implementation_pins"):
        for label, value in contract[group].items():
            if isinstance(value, dict) and "path" in value:
                verify_pin(value, f"recipe {group}.{label}")
    recipe = contract.get("frozen_effective_recipe", {})
    required = {
        "pred_horizon": 50,
        "batch_size": 64,
        "eval_batch_size": 64,
        "epochs_in_recipe": 400,
        "maximum_durable_epoch_per_task": 180,
        "seed": 7,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "w_flow": 1.0,
        "w_pos": 5.0,
        "w_rot": 1.0,
        "w_hand_joint": 5.0,
        "w_joint_limit": 0.1,
        "w_velocity": 0.1,
        "w_done": 0.2,
        "persistent_workers": True,
        "worker_cap": 8,
        "prefetch_factor": 4,
    }
    for key, expected in required.items():
        if recipe.get(key) != expected:
            raise BundleHold(f"HOLD_RECIPE_REBIND_FIELD_DRIFT:{key}")
    return contract


def materialize(
    *,
    task: str,
    session: str,
    lineage: Mapping[str, Any],
    bundle: Path,
    authority_path: Path | None = None,
) -> None:
    if bundle.exists() or bundle.is_symlink():
        raise BundleHold(f"HOLD_NO_CLOBBER:{bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.", dir=bundle.parent))
    try:
        authority_source = (authority_path or AUTHORITY).resolve(strict=True)
        authority_snapshot = temporary / "AUTHORITY_SNAPSHOT.json"
        recipe_contract_snapshot = temporary / "TRAINING_RECIPE_REBIND_CONTRACT_SNAPSHOT.json"
        copy_ordinary(authority_source, authority_snapshot)
        copy_ordinary(REBIND, recipe_contract_snapshot)
        recipe = _recipe_snapshot(recipe_contract_snapshot)
        _authority_robot_row(
            read_json(authority_snapshot),
            task=task,
            robot_result=lineage["robot_result_path"],
        )
        hawor_root = temporary / "hawor_v3_sidecars"
        hawor_builder = _load_hawor_builder()
        hawor_row = hawor_builder.build_session(
            session=session,
            task=task,
            base_bundle=temporary,
            destination=hawor_root,
            mps_path=lineage["mps_path"],
            hawor_archive_path=lineage["hawor_npz"],
            hawor_result_path=lineage["hawor_result_path"],
        )
        hawor_sidecar = hawor_root / session / "entities_hawor_v3.npz"
        hawor_row = {
            **hawor_row,
            "entity_sidecar": ref(
                hawor_sidecar,
                published=bundle / "hawor_v3_sidecars" / session
                / "entities_hawor_v3.npz",
            ),
        }
        robot_sidecar = temporary / "sidecars/kai22" / session / "sidecar.npz"
        robot_validation, split = build_robot_sidecar(
            scene_path=lineage["scene_path"],
            hawor_sidecar=hawor_sidecar,
            destination=robot_sidecar,
            session=session,
            fps=lineage["fps"],
        )
        grade = str(lineage["robot_result"]["grade"])
        # The visual Object6D authority remains a development estimate even
        # when the downstream Robot receipt is A.  Keep the previous recipe's
        # explicit Grade-B object weight and never promote it to contact truth.
        object_grade = "B"
        object_npz, object_json, object_manifest = build_object_sidecar(
            source=lineage["object6d_npz"],
            destination_dir=temporary / "object_state_sidecars" / session,
            published_dir=bundle / "object_state_sidecars" / session,
            session=session,
            grade=object_grade,
        )

        required_frames = sorted(set(
            range(split["train"]["frame_start"], split["train"]["frame_stop_exclusive"])
        ) | set(
            range(
                split["validation"]["frame_start"],
                split["validation"]["frame_stop_exclusive"],
            )
        ))
        adapter = temporary / "production" / session / "09_humanego_adapter"
        selector_frames: dict[str, Any] = {}
        for frame in required_frames:
            name = f"{frame:05d}"
            source = lineage["raw_all_data"] / name
            destination = adapter / "preprocess/all_data" / name
            copy_ordinary(source / "training_data.json", destination / "training_data.json")
            copy_ordinary(source / "rgb.png", destination / "rgb.png")
            selector_frames[name] = {
                "metadata": ref(
                    destination / "training_data.json",
                    published=bundle / destination.relative_to(temporary) / "training_data.json",
                ),
                "image": ref(
                    destination / "rgb.png",
                    published=bundle / destination.relative_to(temporary) / "rgb.png",
                ),
                "unresolved": False,
            }

        sidecar_sha = sha256(robot_sidecar)
        hawor_sha = sha256(hawor_sidecar)
        object_npz_sha = sha256(object_npz)
        object_json_sha = sha256(object_json)
        split_payload = {
            "schema_version": "humanego-two-task-single-session-temporal-split-v1",
            "task": task,
            "session": session,
            "seed": 7,
            "split_unit": "non_overlapping_temporal_h50_support_within_one_session",
            "temporal_split": split,
            "production_root": str((bundle / "production").absolute()),
            "sidecar_root": str((bundle / "sidecars").absolute()),
            "hawor_v3_sidecar_root": str((bundle / "hawor_v3_sidecars").absolute()),
            "object_state_sidecar_root": str((bundle / "object_state_sidecars").absolute()),
            "heldout_contract": {
                "canonical_heldout_consumed": False,
                "cross_session_heldout_exists": False,
                "claim": split["validation_claim"],
            },
            "ict_contract": {
                "hand_tracking_method": "hawor_v3",
                "single_hand": False,
                "ict_dim": 29,
                "frame_mode": "camera_frame",
                "translation_unit": "metre",
                "camera_coordinate_system": "x_right_y_down_z_forward",
                "object_token_policy": "VISUAL_OBJECT6D_GRADE_B_WEIGHT_0_25_NO_CONTACT",
            },
            "motion_contract": {
                "embodiment": "kai22",
                "sidecar_sha256": sidecar_sha,
                "source": "CURRENT_TIANJI_KAI_ROBOT_A_OR_B_SCENE_Q_HAND",
                "wrist_source": "CURRENT_BOUNDED_V2_HAWOR_CAMERA_FRAME",
                "arm_consumed_by_model": False,
                "contact_aux_weight": 0.0,
                "human_to_physical": {"left": "right", "right": "left"},
            },
            "object_state_contract": {
                "consumption_mode": "training_estimated_grade_b",
                "quality_grade": object_grade,
                "training_weight": object_manifest["training_weight"],
                "confidence_thresholds": OBJECT_THRESHOLDS,
                "npz_sha256": object_npz_sha,
                "json_sha256": object_json_sha,
            },
        }
        selector = {
            "schema_version": "humanego-newtask-robot-selector-v1",
            "product_line": "RAW_RGB_CURRENT_E2E_GRADED_ROBOT_ACTION",
            "image_name": "rgb.png",
            "artifact_root": str((bundle / "production").absolute()),
            "selector_root": str((bundle / "production").absolute()),
            "sessions": {session: {"frames": selector_frames}},
        }
        paired = {
            "schema_version": "humanego-single-session-role-windows-v1",
            "session": session,
            "horizon": HORIZON,
            "roles": {
                role: {
                    "window_starts": split[role]["window_starts"],
                    "support_frames": list(range(
                        split[role]["frame_start"],
                        split[role]["frame_stop_exclusive"],
                    )),
                }
                for role in ("train", "validation")
            },
        }
        source_lineage = {
            "schema_version": "two-task-single-session-e2e-lineage-v1",
            "run_id": "20260908_two_task_e2e_baseline_v1",
            "task": task,
            "session": session,
            "robot_grade": grade,
            "bundle_builder": ref(Path(__file__)),
            "authority": ref(
                authority_snapshot,
                published=bundle / "AUTHORITY_SNAPSHOT.json",
            ),
            "live_authority_at_build": ref(authority_source),
            "robot_result": ref(lineage["robot_result_path"]),
            "robot_agent_review": ref(lineage["robot_review_path"]),
            "robot_preflight": ref(lineage["preflight_path"]),
            "robot_spec": ref(lineage["spec_path"]),
            "robot_kinematic_result": ref(lineage["kinematic_result_path"]),
            "robot_scene": ref(lineage["scene_path"]),
            "raw_video": ref(lineage["raw_video"]),
            "hawor_result": ref(lineage["hawor_result_path"]),
            "hawor_npz": ref(lineage["hawor_npz"]),
            "mask_result": ref(lineage["mask_result_path"]),
            "object6d_result": ref(lineage["object6d_result_path"]),
            "object6d_npz": ref(lineage["object6d_npz"]),
            "clean_result": ref(lineage["clean_result_path"]),
            "claim_limit": (
                "Current A/B E2E baseline optimization data only; no contact, "
                "deployment, cross-session generalization, or canonical heldout claim."
            ),
        }
        frame_manifest = {
            "schema_version": "humanego-single-session-frame-manifest-v1",
            "session": session,
            "frames": selector_frames,
            "ordinary_files_nlink_one": all(
                (temporary / Path(row[key]["path"]).relative_to(bundle)).stat().st_nlink == 1
                for row in selector_frames.values()
                for key in ("metadata", "image")
            ),
        }
        audit = {
            "schema_version": "humanego-two-task-single-session-bundle-audit-v1",
            "status": "PASS_CPU_BUILD_NO_TRAINING",
            "task": task,
            "session": session,
            "robot_grade": grade,
            "robot_sidecar_validation": robot_validation,
            "hawor_sidecar": hawor_row,
            "temporal_split": split,
            "train_windows": len(split["train"]["window_starts"]),
            "validation_windows": len(split["validation"]["window_starts"]),
            "canonical_heldout_consumed": False,
            "training_started": False,
            "ordinary_files_nlink_one": True,
            "claim_limit": split["validation_claim"],
        }
        recipe_payload = {
            "schema_version": "two-task-single-session-effective-recipe-v1",
            "source_rebind_contract": ref(
                recipe_contract_snapshot,
                published=bundle / "TRAINING_RECIPE_REBIND_CONTRACT_SNAPSHOT.json",
            ),
            "live_rebind_contract_at_build": ref(REBIND),
            "source_previous_manifests": recipe["source_of_truth"],
            "implementation_pins": recipe["implementation_pins"],
            "effective_recipe": recipe["frozen_effective_recipe"],
            "policy": (
                "Model, pretrained, optimizer, losses, augmentation, effective "
                "batch=64, seed=7 and 400-epoch recipe are preserved; durable "
                "execution is capped at epoch 180."
            ),
        }
        payloads = {
            "split.json": split_payload,
            "selector_records.json": selector,
            "paired_windows.json": paired,
            "sidecars.json": {session: sidecar_sha},
            "hawor_v3_sidecars.json": {session: hawor_sha},
            "object_state_sidecars.json": {
                session: {"npz_sha256": object_npz_sha, "json_sha256": object_json_sha}
            },
            "SOURCE_LINEAGE.json": source_lineage,
            "FRAME_MANIFEST.json": frame_manifest,
            "RECIPE.json": recipe_payload,
            "AUDIT.json": audit,
        }
        for filename, payload in payloads.items():
            write_json(temporary / filename, payload)
        freeze = {
            "schema_version": "humanego-two-task-single-session-freeze-v1",
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task": task,
            "session": session,
            "training_started": False,
            "bundle_files": {
                name: ref(temporary / name, published=bundle / name)
                for name in (
                    *payloads,
                    "AUTHORITY_SNAPSHOT.json",
                    "TRAINING_RECIPE_REBIND_CONTRACT_SNAPSHOT.json",
                )
            },
            "payload_artifacts": {
                "robot_sidecar": ref(
                    robot_sidecar,
                    published=bundle / robot_sidecar.relative_to(temporary),
                ),
                "hawor_sidecar": ref(
                    hawor_sidecar,
                    published=bundle / hawor_sidecar.relative_to(temporary),
                ),
                "object_state_npz": ref(
                    object_npz,
                    published=bundle / object_npz.relative_to(temporary),
                ),
                "object_state_json": ref(
                    object_json,
                    published=bundle / object_json.relative_to(temporary),
                ),
            },
            "claim_limit": split["validation_claim"],
        }
        write_json(temporary / "freeze.json", freeze)
        for path in temporary.rglob("*"):
            if path.is_symlink():
                raise BundleHold(f"HOLD_BUNDLE_SYMLINK:{path}")
            if path.is_file() and path.stat().st_nlink != 1:
                raise BundleHold(f"HOLD_BUNDLE_NLINK_NOT_ONE:{path}")
        os.replace(temporary, bundle)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def wait_receipt(
    *, task: str, session: str, authority_path: Path, destination: Path
) -> None:
    authority = read_json(authority_path.resolve(strict=True))
    robot = authority.get("stage_authorities", {}).get("robot", {})
    payload = {
        "schema_version": "two-task-single-session-training-wait-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "WAIT_FRESH_ROBOT_A_OR_B",
        "task": task,
        "session": session,
        "training_started": False,
        "bundle_created": False,
        "authority": ref(authority_path),
        "observed_robot_status": robot.get("status") if isinstance(robot, dict) else None,
        "required_next": (
            "BASELINE_AUTHORITY.stage_authorities.robot.<task> must bind an exact "
            "Grade A/B downstream-authorized Robot RESULT.json and AGENT_REVIEW.json."
        ),
        "claim_limit": "Recipe/package readiness only; no training or checkpoint exists.",
    }
    atomic_json(destination, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--robot-result", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--wait-receipt", type=Path)
    args = parser.parse_args()
    if args.session != TASKS[args.task]:
        raise BundleHold("HOLD_TASK_SESSION_NOT_BASELINE_PAIR")
    if args.robot_result is None:
        if args.wait_receipt is None or args.bundle is not None:
            parser.error("without --robot-result, provide only --wait-receipt")
        wait_receipt(
            task=args.task,
            session=args.session,
            authority_path=args.authority,
            destination=args.wait_receipt.absolute(),
        )
        print(json.dumps({
            "status": "WAIT_FRESH_ROBOT_A_OR_B",
            "receipt": str(args.wait_receipt.absolute()),
            "training_started": False,
        }))
        return 0
    if args.bundle is None or args.wait_receipt is not None:
        parser.error("with --robot-result, provide only --bundle")
    lineage = validate_robot_lineage(
        task=args.task,
        session=args.session,
        robot_result_path=args.robot_result.resolve(strict=True),
        authority_path=args.authority.resolve(strict=True),
    )
    materialize(
        task=args.task,
        session=args.session,
        lineage=lineage,
        bundle=args.bundle.absolute(),
        authority_path=args.authority,
    )
    print(json.dumps({
        "status": "PASS_CPU_BUILD_NO_TRAINING",
        "bundle": str(args.bundle.absolute()),
        "training_started": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
