#!/usr/bin/env python3
"""Freeze and orchestrate visual Object6D with a mandatory Chinese review video.

Real Object6D observations are limited to decoded RGB, an independent RGB object
mask and authorized corrected metric stereo depth.  Camera poses are extracted
from immutable PICO ``training_data.json`` records into a camera-only bundle.
Hand joints, HaWoR object guesses, pinch points and Robot state are not accepted.

The normal execution path is no-clobber and transactional at the orchestration
directory level: numeric Object6D and its full-frame review must both succeed
before publication.  This tool also provides CPU-only preparation/validation and
synthetic tests that do not execute a real Object6D candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import run_visual_fixed_instance_object6d as numeric  # noqa: E402

SCRIPT = Path(__file__).resolve()
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
EXPECTED_FONT_SHA256 = "79c18ebe7b811951e8311bad7103ebeae8c337ed9988ea69e8a78a66cfe029b9"
EXPECTED_SPEC_KEYS = {
    "schema_version", "session_id", "task", "frame_count", "selected_rgb",
    "depth_root", "depth_result", "depth_frame_manifest",
    "registration_authority", "rgb_object_mask_manifest", "camera_to_world",
    "camera_to_world_key",
}
OPTIONAL_SPEC_KEYS = {"leading_unobserved_policy", "unobserved_pose_policy"}
CAMERA_ONLY_KEYS = {
    "frame_indices", "c2w", "selected_rgb_intrinsics",
    "source_training_data_sha256",
}
FORBIDDEN_TOKENS = ("hand", "pinch", "hawor", "mano", "robot", "joint")
REVIEW_CONTRACT = {
    "schema_version": "visual-object6d-mandatory-review-contract-v1",
    "language": "zh-CN",
    "layout": "1920x720: original RGB | mask+pose overlay | fixed-scale metric audit",
    "frame_policy": "one review frame for every Object6D frame, same order and frame_id",
    "fps": 30.0,
    "codec": "mp4v",
    "fixed_depth_range_m": [0.1, 3.0],
    "pose_axis_length_m": 0.03,
    "units": ["metre", "millimetre", "pixel", "unitless visibility"],
    "required_fields": [
        "original RGB", "RGB object mask", "pose XYZ axes", "observed near/far",
        "analytic near/far", "physical instance identity", "valid", "visibility",
    ],
    "watermark": "候选/HOLD：未经人工审核，不授权 Robot 接触",
    "forbidden_inputs": ["hand joints", "pinch points", "HaWoR object inference", "Robot state"],
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode() + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def published_artifact(current: Path, published: Path) -> dict[str, Any]:
    return {"path": str(published.resolve(strict=False)), "bytes": current.stat().st_size, "sha256": sha256(current)}


def verify_artifact(value: dict[str, Any], label: str) -> Path:
    if set(value) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label}: artifact schema mismatch")
    path = Path(value["path"])
    if not path.is_file():
        raise RuntimeError(f"{label}: missing {path}")
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != value["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA closure mismatch")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_forbidden_token(value: Any) -> list[str]:
    findings: list[str] = []

    def visit(node: Any, route: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = str(key).lower()
                if any(token in lowered for token in FORBIDDEN_TOKENS):
                    findings.append(f"{route}.{key}")
                visit(child, f"{route}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{route}[{index}]")

    visit(value, "$spec")
    return findings


def read_video_frame(cap: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"selected RGB decode failed at frame {frame_id}")
    return frame


def validate_spec(spec_path: Path, deep: bool = True) -> dict[str, Any]:
    spec = load_json(spec_path)
    errors: list[str] = []
    actual_keys = set(spec)
    if not EXPECTED_SPEC_KEYS.issubset(actual_keys) or not actual_keys.issubset(
        EXPECTED_SPEC_KEYS | OPTIONAL_SPEC_KEYS
    ):
        errors.append(
            f"SPEC_KEYS required={sorted(EXPECTED_SPEC_KEYS)} "
            f"optional={sorted(OPTIONAL_SPEC_KEYS)} actual={sorted(spec)}"
        )
    if spec.get("leading_unobserved_policy", "PROPAGATE_NEAREST") not in {
        "PROPAGATE_NEAREST",
        "INVALID_UNTIL_VISUAL_IDENTITY_ONSET",
    }:
        errors.append("LEADING_UNOBSERVED_POLICY")
    if spec.get("unobserved_pose_policy", "PROPAGATE_VISUALIZATION") not in {
        "PROPAGATE_VISUALIZATION",
        "KEEP_INVALID",
    }:
        errors.append("UNOBSERVED_POSE_POLICY")
    if spec.get("schema_version") != "visual-fixed-instance-object6d-input-v1":
        errors.append("SPEC_SCHEMA")
    if spec.get("task") not in numeric.GEOMETRY:
        errors.append("TASK")
    forbidden = contains_forbidden_token(spec)
    if forbidden:
        errors.append("FORBIDDEN_INPUT_KEYS:" + ",".join(forbidden))
    refs: dict[str, Path] = {}
    for key in (
        "selected_rgb", "depth_result", "depth_frame_manifest",
        "registration_authority", "rgb_object_mask_manifest", "camera_to_world",
    ):
        try:
            refs[key] = verify_artifact(spec[key], key)
        except Exception as error:
            errors.append(str(error))
    if errors:
        return {"spec": artifact(spec_path), "status": "HOLD", "errors": errors}

    depth_result = load_json(refs["depth_result"])
    depth_manifest = load_json(refs["depth_frame_manifest"])
    mask_manifest = load_json(refs["rgb_object_mask_manifest"])
    frame_count = int(spec["frame_count"])
    if depth_result.get("status") != "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION":
        errors.append("DEPTH_STATUS")
    if not depth_result.get("consumption_authorized"):
        errors.append("DEPTH_NOT_AUTHORIZED")
    if "VISUAL_OBJECT6D_CANDIDATE_INPUT" not in depth_result.get("authorized_scopes", []):
        errors.append("DEPTH_SCOPE")
    if depth_result.get("session_id") != spec["session_id"]:
        errors.append("DEPTH_SESSION")
    if depth_manifest.get("session_id") != spec["session_id"]:
        errors.append("DEPTH_MANIFEST_SESSION")
    if mask_manifest.get("session_id") != spec["session_id"] or mask_manifest.get("task") != spec["task"]:
        errors.append("MASK_SESSION_TASK")
    mask_rows = mask_manifest.get("frames", [])
    depth_rows = {int(row["frame_id"]): row for row in depth_manifest.get("frames", [])}
    if len(mask_rows) != frame_count:
        errors.append("MASK_FRAME_COUNT")
    frame_ids = np.asarray([int(row["frame_id"]) for row in mask_rows], dtype=np.int32)
    if len(set(frame_ids.tolist())) != frame_count or any(int(frame) not in depth_rows for frame in frame_ids):
        errors.append("MASK_DEPTH_FRAME_IDENTITY")
    if Path(spec["depth_root"]).resolve() != refs["depth_result"].parent.resolve():
        errors.append("DEPTH_ROOT_BINDING")
    if mask_manifest.get("input", {}).get("selected_rgb") != spec["selected_rgb"]:
        errors.append("MASK_SELECTED_RGB_BINDING")

    try:
        with np.load(refs["registration_authority"], allow_pickle=False) as registration:
            k_depth = registration["stereo_rectified_depth_formula_intrinsics"].astype(np.float64)
            k_selected = registration["selected_rgb_intrinsics"].astype(np.float64)
            h_depth_to_selected = registration["H_depth_pixel_to_selected_rgb"].astype(np.float64)
        if not np.array_equal(k_depth, np.asarray([[320., 0., 319.5], [0., 320., 239.5], [0., 0., 1.]])):
            errors.append("CORRECTED_DEPTH_K")
        if h_depth_to_selected.shape != (3, 3) or not np.isfinite(h_depth_to_selected).all():
            errors.append("REGISTRATION_H")
    except Exception as error:
        errors.append(f"REGISTRATION_LOAD:{error}")
        k_selected = np.empty((0, 0))

    try:
        with np.load(refs["camera_to_world"], allow_pickle=False) as camera:
            if set(camera.files) != CAMERA_ONLY_KEYS:
                errors.append(f"CAMERA_ONLY_KEYS:{sorted(camera.files)}")
            c2w = camera[spec["camera_to_world_key"]].astype(np.float64)
            camera_frames = camera["frame_indices"].astype(np.int32)
            camera_k = camera["selected_rgb_intrinsics"].astype(np.float64)
            source_sha = camera["source_training_data_sha256"]
        if spec["camera_to_world_key"] != "c2w":
            errors.append("CAMERA_KEY")
        if c2w.shape != (frame_count, 4, 4) or not np.isfinite(c2w).all():
            errors.append("CAMERA_C2W_SHAPE_FINITE")
        if not np.array_equal(camera_frames, frame_ids):
            errors.append("CAMERA_FRAME_IDENTITY")
        if camera_k.shape != (frame_count, 3, 3) or not np.allclose(camera_k, k_selected, atol=1e-9):
            errors.append("CAMERA_K_REGISTRATION_BINDING")
        if source_sha.shape != (frame_count,):
            errors.append("CAMERA_SOURCE_SHA_COUNT")
        if c2w.shape == (frame_count, 4, 4):
            rotations = c2w[:, :3, :3]
            orthogonal_error = float(np.max(np.abs(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3))))
            determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
            last_row_error = float(np.max(np.abs(c2w[:, 3, :] - np.asarray([0., 0., 0., 1.]))))
            if max(orthogonal_error, determinant_error, last_row_error) > 1e-6:
                errors.append("CAMERA_NOT_SE3")
        else:
            orthogonal_error = determinant_error = last_row_error = float("nan")
    except Exception as error:
        errors.append(f"CAMERA_LOAD:{error}")
        orthogonal_error = determinant_error = last_row_error = float("nan")

    checked_rgb = checked_masks = checked_depth = 0
    if deep and not errors:
        cap = cv2.VideoCapture(str(refs["selected_rgb"]))
        try:
            for row in mask_rows:
                frame_id = int(row["frame_id"])
                rgb = read_video_frame(cap, frame_id)
                if numeric.rgb_digest(rgb) != row["source_rgb_decoded_sha256"]:
                    errors.append(f"RGB_MASK_DECODE_SHA:{frame_id}")
                    break
                checked_rgb += 1
                try:
                    mask_path = verify_artifact(row["mask"], f"mask:{frame_id}")
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != rgb.shape[:2]:
                        errors.append(f"MASK_SHAPE:{frame_id}")
                        break
                    checked_masks += 1
                except Exception as error:
                    errors.append(str(error))
                    break
                depth_row = depth_rows[frame_id]
                depth_path = Path(spec["depth_root"]) / depth_row["relative_path"]
                if not depth_path.is_file() or depth_path.stat().st_size != depth_row["bytes"] or sha256(depth_path) != depth_row["sha256"]:
                    errors.append(f"DEPTH_BUNDLE_CLOSURE:{frame_id}")
                    break
                with np.load(depth_path, allow_pickle=False) as bundle:
                    required = {"frame_id", "disparity_px", "depth_m", "valid", "scaled_intrinsics", "input_closure_sha256"}
                    if not required.issubset(bundle.files):
                        errors.append(f"DEPTH_BUNDLE_KEYS:{frame_id}")
                        break
                    if int(bundle["frame_id"]) != frame_id or bundle["depth_m"].shape != (480, 640):
                        errors.append(f"DEPTH_BUNDLE_FRAME_SHAPE:{frame_id}")
                        break
                checked_depth += 1
        finally:
            cap.release()

    return {
        "spec": artifact(spec_path),
        "session_id": spec["session_id"],
        "task": spec["task"],
        "frame_count": frame_count,
        "frame_range": [int(frame_ids.min()), int(frame_ids.max())] if len(frame_ids) else None,
        "status": "PASS_REAL_INPUT_CLOSURE" if not errors else "HOLD",
        "errors": errors,
        "checked": {"decoded_rgb": checked_rgb, "mask_artifacts": checked_masks, "depth_bundles": checked_depth},
        "camera_only_keys": sorted(CAMERA_ONLY_KEYS),
        "camera_se3_max_errors": {
            "orthogonal": orthogonal_error,
            "determinant": determinant_error,
            "last_row": last_row_error,
        },
        "forbidden_hand_pinch_robot_inputs": False,
    }


def prepare_real_specs(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"no-clobber real spec root exists: {output_root}")
    config = load_json(config_path)
    if config.get("schema_version") != "visual-object6d-real-source-config-v1":
        raise RuntimeError("real source config schema mismatch")
    output_root.mkdir(parents=True)
    session_rows = []
    for case in config["sessions"]:
        for key in ("selected_rgb", "depth_result", "depth_frame_manifest", "registration_authority", "rgb_object_mask_manifest"):
            verify_artifact(case[key], f"config:{case['session_id']}:{key}")
        mask_manifest = load_json(Path(case["rgb_object_mask_manifest"]["path"]))
        frames = np.asarray([int(row["frame_id"]) for row in mask_manifest["frames"]], dtype=np.int32)
        c2w_rows = []
        k_rows = []
        source_shas = []
        source_rows = []
        for frame in frames.tolist():
            source = Path(case["camera_metadata_root"]) / f"{frame:05d}" / "training_data.json"
            source_ref = artifact(source)
            record = load_json(source)
            metadata = record["metadata"]
            if int(metadata["idx"]) != frame:
                raise RuntimeError(f"camera metadata frame mismatch: {source}")
            c2w_rows.append(np.asarray(metadata["c2w"], dtype=np.float64))
            k_rows.append(np.asarray(metadata["k"], dtype=np.float64))
            source_shas.append(source_ref["sha256"])
            source_rows.append({"frame_id": frame, **source_ref})
        c2w = np.stack(c2w_rows)
        k_selected = np.stack(k_rows)
        session_root = output_root / case["session_id"]
        session_root.mkdir()
        camera_path = session_root / "CAMERA_C2W_ONLY.npz"
        atomic_npz(
            camera_path,
            frame_indices=frames,
            c2w=c2w,
            selected_rgb_intrinsics=k_selected,
            source_training_data_sha256=np.asarray(source_shas, dtype="<U64"),
        )
        binding = {
            "schema_version": "camera-c2w-only-binding-v1",
            "created_at": now(),
            "session_id": case["session_id"],
            "frame_indices": frames.tolist(),
            "camera_bundle": artifact(camera_path),
            "camera_bundle_keys": sorted(CAMERA_ONLY_KEYS),
            "source_kind": "IMMUTABLE_PICO_PREPROCESS_TRAINING_DATA_METADATA_ONLY",
            "source_rows": source_rows,
            "hand_pinch_hawor_robot_arrays_written": False,
            "claim_limit": "Camera pose/K timeline only; not hand, object, contact, or task ground truth.",
        }
        binding_path = session_root / "CAMERA_C2W_BINDING.json"
        atomic_json(binding_path, binding)
        spec = {
            "schema_version": "visual-fixed-instance-object6d-input-v1",
            "session_id": case["session_id"],
            "task": case["task"],
            "frame_count": int(len(frames)),
            "selected_rgb": case["selected_rgb"],
            "depth_root": str(Path(case["depth_result"]["path"]).parent),
            "depth_result": case["depth_result"],
            "depth_frame_manifest": case["depth_frame_manifest"],
            "registration_authority": case["registration_authority"],
            "rgb_object_mask_manifest": case["rgb_object_mask_manifest"],
            "camera_to_world": artifact(camera_path),
            "camera_to_world_key": "c2w",
        }
        if "leading_unobserved_policy" in case:
            spec["leading_unobserved_policy"] = case["leading_unobserved_policy"]
        if "unobserved_pose_policy" in case:
            spec["unobserved_pose_policy"] = case["unobserved_pose_policy"]
        spec_path = session_root / "INPUT_SPEC.json"
        atomic_json(spec_path, spec)
        validation = validate_spec(spec_path, deep=True)
        if validation["status"] != "PASS_REAL_INPUT_CLOSURE":
            raise RuntimeError(f"fresh spec failed closure: {validation}")
        session_rows.append({
            "session_id": case["session_id"],
            "task": case["task"],
            "input_spec": artifact(spec_path),
            "camera_binding": artifact(binding_path),
            "validation": validation,
        })
    result = {
        "schema_version": "visual-object6d-fresh-real-specs-result-v1",
        "created_at": now(),
        "status": "PASS_TWO_REAL_INPUT_CLOSURES",
        "mode": "CPU_PREPARE_AND_VALIDATE_ONLY",
        "source_config": artifact(config_path),
        "sessions": session_rows,
        "object6d_producer_executed": False,
        "review_renderer_executed_on_real_data": False,
        "execution_authorized": False,
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def text_lines(canvas: np.ndarray, lines: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]]) -> np.ndarray:
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    cache: dict[int, ImageFont.FreeTypeFont] = {}
    for value, xy, size, bgr in lines:
        if size not in cache:
            cache[size] = ImageFont.truetype(str(FONT), size)
        draw.text(xy, value, font=cache[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def project_pose_axes(transform: np.ndarray, k_depth: np.ndarray, h_depth_to_selected: np.ndarray) -> list[np.ndarray] | None:
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    offsets = np.eye(3) * REVIEW_CONTRACT["pose_axis_length_m"]
    points = np.vstack([origin, origin + offsets @ rotation.T])
    if np.any(points[:, 2] <= 0):
        return None
    depth_pixels = np.column_stack([
        k_depth[0, 0] * points[:, 0] / points[:, 2] + k_depth[0, 2],
        k_depth[1, 1] * points[:, 1] / points[:, 2] + k_depth[1, 2],
        np.ones(4),
    ])
    selected = (h_depth_to_selected @ depth_pixels.T).T
    selected = selected[:, :2] / selected[:, 2:3]
    return [selected[0], selected[1], selected[2], selected[3]]


def draw_fixed_depth_gauge(panel: np.ndarray, observed: np.ndarray, analytic: np.ndarray) -> None:
    low, high = REVIEW_CONTRACT["fixed_depth_range_m"]
    x0, x1, y = 60, 575, 342
    cv2.line(panel, (x0, y), (x1, y), (170, 170, 170), 2, cv2.LINE_AA)
    for value, label in ((low, "0.1m"), (1.0, "1.0m"), (2.0, "2.0m"), (high, "3.0m")):
        x = int(round(x0 + (value - low) / (high - low) * (x1 - x0)))
        cv2.line(panel, (x, y - 7), (x, y + 7), (180, 180, 180), 1)
        cv2.putText(panel, label, (x - 18, y + 28), cv2.FONT_HERSHEY_SIMPLEX, .42, (190, 190, 190), 1, cv2.LINE_AA)
    for values, colour, yy in ((observed, (0, 220, 255), y - 12), (analytic, (255, 180, 70), y + 12)):
        if np.isfinite(values).all():
            xa, xb = [int(round(x0 + (float(np.clip(v, low, high)) - low) / (high - low) * (x1 - x0))) for v in values]
            cv2.line(panel, (xa, yy), (xb, yy), colour, 5, cv2.LINE_AA)


def render_review(
    spec_path: Path,
    numeric_root: Path,
    output_path: Path,
    *,
    agent_baseline_review: bool = False,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"no-clobber review exists: {output_path}")
    if not FONT.is_file() or sha256(FONT) != EXPECTED_FONT_SHA256:
        raise RuntimeError("Chinese font closure mismatch")
    validation = validate_spec(spec_path, deep=False)
    if validation["status"] == "HOLD":
        raise RuntimeError(f"review spec invalid: {validation['errors']}")
    spec = load_json(spec_path)
    manifest_path = numeric_root / "FRAME_MANIFEST.json"
    trajectory_path = numeric_root / "VISUAL_FIXED_INSTANCE_OBJECT6D.npz"
    result_path = numeric_root / "RESULT.json"
    for path in (manifest_path, trajectory_path, result_path):
        if not path.is_file():
            raise RuntimeError(f"numeric output incomplete: {path}")
    numeric_result = load_json(result_path)
    if (
        numeric_result.get("consumption_authorized") is not True
        or numeric_result.get("authorized_scopes") != ["CLEAN_VISUAL_BASELINE_INPUT"]
        or numeric_result.get("robot_contact_authorized") is not False
    ):
        raise RuntimeError("numeric candidate authority differs from Clean-only scope")
    rows = load_json(manifest_path)["frames"]
    mask_rows = load_json(Path(spec["rgb_object_mask_manifest"]["path"]))["frames"]
    if len(rows) != spec["frame_count"] or len(mask_rows) != spec["frame_count"]:
        raise RuntimeError("review frame closure mismatch")
    with np.load(trajectory_path, allow_pickle=False) as values:
        frame_ids = values["frame_indices"].astype(np.int32)
        valid = values["valid"].astype(bool)
        direct_observed = values["observed"].astype(bool)
        visibility = values["visibility"].astype(np.float64)
        instance = values["physical_instance_id"].astype(np.int32)
        transforms = values["T_object_to_camera"].astype(np.float64)
        observed_nf = values["observed_near_far_optical_z_m"].astype(np.float64)
        analytic = values["analytic_near_far_optical_z_m"].astype(np.float64)
    with np.load(Path(spec["registration_authority"]["path"]), allow_pickle=False) as registration:
        k_depth = registration["stereo_rectified_depth_formula_intrinsics"].astype(np.float64)
        h_depth_to_selected = registration["H_depth_pixel_to_selected_rgb"].astype(np.float64)
    review_contract = dict(REVIEW_CONTRACT)
    if agent_baseline_review:
        review_contract["watermark"] = "Baseline Grade B 自动审核候选；不授权真机接触/部署"
        review_contract["blocking_gate"] = "AGENT_A_B_C_NO_HUMAN_BLOCKING_GATE"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), review_contract["fps"], (1920, 720))
    if not writer.isOpened():
        raise RuntimeError("cannot open review video writer")
    cap = cv2.VideoCapture(spec["selected_rgb"]["path"])
    frame_digests = []
    try:
        for local, frame_id in enumerate(frame_ids.tolist()):
            rgb = read_video_frame(cap, frame_id)
            original = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
            mask_path = verify_artifact(mask_rows[local]["mask"], f"review-mask:{frame_id}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            overlay = rgb.copy()
            tint = np.zeros_like(overlay)
            tint[:] = (210, 60, 210)
            overlay[mask] = np.rint(overlay[mask] * 0.48 + tint[mask] * 0.52).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
            if valid[local]:
                axes = project_pose_axes(transforms[local], k_depth, h_depth_to_selected)
                if axes is not None:
                    points = [tuple(np.rint(point).astype(int)) for point in axes]
                    for endpoint, colour in zip(points[1:], ((0, 0, 255), (0, 255, 0), (255, 80, 0))):
                        cv2.line(overlay, points[0], endpoint, colour, 5, cv2.LINE_AA)
                        cv2.circle(overlay, endpoint, 6, colour, -1, cv2.LINE_AA)
                    cv2.circle(overlay, points[0], 7, (255, 255, 255), -1, cv2.LINE_AA)
            overlay = cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA)
            info = np.full((480, 640, 3), (28, 28, 30), np.uint8)
            draw_fixed_depth_gauge(info, observed_nf[local], analytic[local])
            canvas = np.full((720, 1920, 3), (16, 16, 18), np.uint8)
            canvas[70:550, 0:640] = original
            canvas[70:550, 640:1280] = overlay
            canvas[70:550, 1280:1920] = info
            identity = rows[local].get("identity_label", mask_rows[local].get("observed_identity", "UNKNOWN"))
            if direct_observed[local]:
                observed_text = f"{observed_nf[local,0]:.4f}–{observed_nf[local,1]:.4f} m ({observed_nf[local,0]*1000:.1f}–{observed_nf[local,1]*1000:.1f} mm)"
                direct_text = "是"
            elif valid[local]:
                observed_text = "无（遮挡；未伪造直接深度）"
                direct_text = "否（仅时序姿态）"
            else:
                observed_text = "无（实例身份尚未建立）"
                direct_text = "否（不提供6D姿态）"
            analytic_text = "无" if not np.isfinite(analytic[local]).all() else f"{analytic[local,0]:.4f}–{analytic[local,1]:.4f} m"
            lines = [
                (f"Object6D 中文逐帧审查 · {spec['session_id']} · 帧 {frame_id}", (24, 17), 30, (240, 240, 240)),
                ("原始 RGB", (20, 84), 25, (245, 245, 245)),
                ("RGB 物体 mask + 30mm 姿态轴", (660, 84), 25, (245, 245, 245)),
                ("固定公制审计（量程 0.1–3.0m）", (1300, 84), 25, (245, 245, 245)),
                (f"物理实例: {int(instance[local])} / {identity}", (1320, 145), 24, (230, 230, 230)),
                (f"有效: {'是' if valid[local] else '否'}", (1320, 190), 24, (80, 230, 120) if valid[local] else (80, 80, 240)),
                (f"直接观测: {direct_text}", (1320, 225), 22, (80, 230, 120) if direct_observed[local] else (40, 175, 245)),
                (f"可见度: {visibility[local]:.4f}", (1320, 257), 22, (230, 230, 230)),
                (f"观测 near/far: {observed_text}", (1320, 292), 19, (0, 220, 255)),
                (f"解析 near/far: {analytic_text}", (1320, 323), 19, (255, 180, 70)),
                ("黄=观测区间；蓝=固定尺寸解析区间", (1330, 448), 20, (210, 210, 210)),
                ("姿态轴: X红 / Y绿 / Z蓝；轴长固定 30 mm", (655, 566), 22, (225, 225, 225)),
                (review_contract["watermark"], (35, 625), 31, (30, 80, 245)),
                ("禁止输入：手关节 / pinch / HaWoR物体猜测 / Robot状态", (850, 632), 23, (80, 180, 245)),
            ]
            canvas = text_lines(canvas, lines)
            writer.write(canvas)
            frame_digests.append({"frame_id": frame_id, "review_frame_sha256": array_sha(canvas)})
    finally:
        cap.release()
        writer.release()
    check = cv2.VideoCapture(str(output_path))
    decoded_frames = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(check.get(cv2.CAP_PROP_FPS))
    width = int(round(check.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(check.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    check.release()
    if decoded_frames != len(rows) or abs(fps - review_contract["fps"]) > 0.01 or (width, height) != (1920, 720):
        raise RuntimeError("review video decode contract failed")
    manifest = {
        "schema_version": "visual-object6d-mandatory-review-result-v1",
        "created_at": now(),
        "status": "PASS_REVIEW_MATERIAL_AGENT_BASELINE" if agent_baseline_review else "PASS_REVIEW_MATERIAL_HOLD_HUMAN",
        "session_id": spec["session_id"],
        "review_contract": review_contract,
        "video": artifact(output_path),
        "decoded": {"frames": decoded_frames, "fps": fps, "width": width, "height": height},
        "observation_summary": {
            "direct_observed_frames": int(direct_observed.sum()),
            "temporally_propagated_frames": int((valid & ~direct_observed).sum()),
            "invalid_pre_identity_frames": int((~valid).sum()),
            "direct_observed_fraction_active_segment": float(
                direct_observed[valid].mean()
            ) if valid.any() else 0.0,
        },
        "frame_digests": frame_digests,
        "font": artifact(FONT),
        "numeric_result": artifact(result_path),
        "human_review_pass": False,
        "agent_baseline_review": agent_baseline_review,
        "consumption_authorized": False,
        "robot_contact_authorized": False,
    }
    atomic_json(output_path.parent / "REVIEW_MANIFEST.json", manifest)
    return manifest


def orchestrate(spec_path: Path, output: Path, *, agent_baseline_review: bool = False) -> dict[str, Any]:
    spec_path = spec_path.resolve(strict=True)
    output = output.resolve(strict=False)
    validation = validate_spec(spec_path, deep=True)
    if validation["status"] != "PASS_REAL_INPUT_CLOSURE":
        raise RuntimeError(f"input closure HOLD: {validation}")
    if output.exists():
        raise FileExistsError(f"no-clobber output exists: {output}")
    partial = output.parent / f".{output.name}.orchestration.partial"
    if partial.exists():
        raise FileExistsError(f"orchestration partial exists: {partial}")
    partial.mkdir(parents=True)
    numeric_root = partial / "numeric"
    numeric_result = numeric.produce(spec_path, numeric_root)
    review_path = partial / "OBJECT6D_中文逐帧审查.mp4"
    review = render_review(
        spec_path,
        numeric_root,
        review_path,
        agent_baseline_review=agent_baseline_review,
    )
    # The numeric producer publishes inside the orchestration staging root.
    # Rebind its artifact references to the final transaction path before the
    # parent directory is atomically renamed, so no current receipt points at
    # a vanished ``.orchestration.partial`` directory.
    numeric_result = load_json(numeric_root / "RESULT.json")
    numeric_result["artifacts"] = {
        "trajectory": published_artifact(
            numeric_root / "VISUAL_FIXED_INSTANCE_OBJECT6D.npz",
            output / "numeric/VISUAL_FIXED_INSTANCE_OBJECT6D.npz",
        ),
        "frame_manifest": published_artifact(
            numeric_root / "FRAME_MANIFEST.json",
            output / "numeric/FRAME_MANIFEST.json",
        ),
    }
    atomic_json(numeric_root / "RESULT.json", numeric_result)
    result = {
        "schema_version": "visual-object6d-mandatory-review-orchestration-result-v1",
        "created_at": now(),
        "status": "PASS_NUMERIC_AND_AGENT_BASELINE_REVIEW_MATERIAL" if agent_baseline_review else "PASS_NUMERIC_AND_REVIEW_MATERIAL_HOLD_HUMAN",
        "input_spec": artifact(spec_path),
        "numeric_status": numeric_result["status"],
        "numeric_result": published_artifact(numeric_root / "RESULT.json", output / "numeric/RESULT.json"),
        "review_status": review["status"],
        "review_video": published_artifact(review_path, output / review_path.name),
        "review_manifest": published_artifact(partial / "REVIEW_MANIFEST.json", output / "REVIEW_MANIFEST.json"),
        "review_contract": review["review_contract"],
        "transactional_publication": True,
        "human_review_pass": False,
        "agent_baseline_review": agent_baseline_review,
        "consumption_authorized": False,
        "robot_contact_authorized": False,
    }
    atomic_json(partial / "RESULT.json", result)
    os.replace(partial, output)
    return load_json(output / "RESULT.json")


def synthetic_test(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"no-clobber synthetic root exists: {output_root}")
    output_root.mkdir(parents=True)
    rows = []
    for task in ("chips", "poker"):
        case_root = output_root / f"source_{task}"
        spec_path, _unused = numeric.make_synthetic_case(case_root, task)
        spec = load_json(spec_path)
        frames = int(spec["frame_count"])
        session_id = spec["session_id"]
        depth_root = Path(spec["depth_root"])
        depth_manifest_path = Path(spec["depth_frame_manifest"]["path"])
        depth_manifest = load_json(depth_manifest_path)
        for row in depth_manifest["frames"]:
            frame_id = int(row["frame_id"])
            depth_path = depth_root / row["relative_path"]
            with np.load(depth_path, allow_pickle=False) as source:
                depth_m = source["depth_m"].astype(np.float32)
                valid = source["valid"].astype(bool)
            disparity = np.where(valid, 320.0 * 0.0637716504026918 / depth_m, np.nan).astype(np.float32)
            atomic_npz(
                depth_path,
                frame_id=np.asarray(frame_id, dtype=np.int32),
                disparity_px=disparity,
                depth_m=depth_m,
                valid=valid,
                scaled_intrinsics=np.asarray([[320., 0., 319.5], [0., 320., 239.5], [0., 0., 1.]]),
                input_closure_sha256=np.asarray("synthetic"),
            )
            ref = artifact(depth_path)
            row["bytes"] = ref["bytes"]
            row["sha256"] = ref["sha256"]
        depth_manifest["session_id"] = session_id
        atomic_json(depth_manifest_path, depth_manifest)
        depth_result_path = Path(spec["depth_result"]["path"])
        atomic_json(depth_result_path, {
            "schema_version": "synthetic-exact78-corrected-dense-depth-result-v1",
            "status": "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION",
            "consumption_authorized": True,
            "authorized_scopes": ["VISUAL_OBJECT6D_CANDIDATE_INPUT"],
            "session_id": session_id,
            "task": task,
        })
        mask_manifest_path = Path(spec["rgb_object_mask_manifest"]["path"])
        mask_manifest = load_json(mask_manifest_path)
        mask_manifest["session_id"] = session_id
        mask_manifest["task"] = task
        mask_manifest["input"] = {"selected_rgb": spec["selected_rgb"]}
        atomic_json(mask_manifest_path, mask_manifest)
        camera_path = Path(spec["camera_to_world"]["path"])
        with np.load(camera_path, allow_pickle=False) as source:
            c2w = source["c2w"].astype(np.float64)
        selected_k = np.tile(np.asarray([[640., 0., 639.5], [0., 640., 479.5], [0., 0., 1.]])[None], (frames, 1, 1))
        atomic_npz(
            camera_path,
            frame_indices=np.arange(frames, dtype=np.int32),
            c2w=c2w,
            selected_rgb_intrinsics=selected_k,
            source_training_data_sha256=np.asarray(["synthetic"] * frames),
        )
        registration_path = Path(spec["registration_authority"]["path"])
        half = np.asarray([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
        atomic_npz(
            registration_path,
            H_selected_rgb_to_depth_pixel=np.linalg.inv(half),
            H_depth_pixel_to_selected_rgb=half,
            selected_rgb_intrinsics=selected_k[0],
            stereo_rectified_depth_formula_intrinsics=np.asarray([[320., 0., 319.5], [0., 320., 239.5], [0., 0., 1.]]),
        )
        spec["camera_to_world"] = artifact(camera_path)
        spec["registration_authority"] = artifact(registration_path)
        spec["depth_result"] = artifact(depth_result_path)
        spec["depth_frame_manifest"] = artifact(depth_manifest_path)
        spec["rgb_object_mask_manifest"] = artifact(mask_manifest_path)
        atomic_json(spec_path, spec)
        output = output_root / f"orchestrated_{task}"
        result = orchestrate(spec_path, output)
        rows.append({
            "task": task,
            "status": result["status"],
            "result": artifact(output / "RESULT.json"),
            "review": artifact(output / "OBJECT6D_中文逐帧审查.mp4"),
            "decoded": load_json(output / "REVIEW_MANIFEST.json")["decoded"],
        })
    result = {
        "schema_version": "visual-object6d-mandatory-review-synthetic-test-v1",
        "created_at": now(),
        "status": "PASS",
        "gpu_used": False,
        "real_object6d_executed": False,
        "hand_pinch_hawor_robot_used": False,
        "sessions": rows,
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def validate_many(spec_paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"no-clobber validation output exists: {output}")
    rows = [validate_spec(path, deep=True) for path in spec_paths]
    result = {
        "schema_version": "visual-object6d-fresh-input-validation-v1",
        "created_at": now(),
        "status": "PASS" if all(row["status"] == "PASS_REAL_INPUT_CLOSURE" for row in rows) else "HOLD",
        "mode": "CPU_VALIDATE_ONLY",
        "sessions": rows,
        "producer": artifact(SCRIPT),
        "numeric_object6d_executed": False,
        "review_renderer_executed_on_real_data": False,
        "execution_authorized": False,
    }
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-real-specs", type=Path)
    parser.add_argument("--real-spec-output", type=Path)
    parser.add_argument("--validate-only", action="append", type=Path, default=[])
    parser.add_argument("--validate-only-output", type=Path)
    parser.add_argument("--synthetic-test", type=Path)
    parser.add_argument("--execute-formal-candidate", action="store_true")
    parser.add_argument("--agent-baseline-review", action="store_true")
    parser.add_argument("--input-spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_real_specs:
        if not args.real_spec_output:
            raise SystemExit("--real-spec-output is required")
        result = prepare_real_specs(args.prepare_real_specs, args.real_spec_output)
    elif args.validate_only:
        if not args.validate_only_output:
            raise SystemExit("--validate-only-output is required")
        result = validate_many(args.validate_only, args.validate_only_output)
    elif args.synthetic_test:
        result = synthetic_test(args.synthetic_test)
    elif args.execute_formal_candidate:
        if not args.input_spec or not args.output:
            raise SystemExit("--input-spec and --output are required")
        result = orchestrate(
            args.input_spec,
            args.output,
            agent_baseline_review=args.agent_baseline_review,
        )
    else:
        raise SystemExit("choose preparation, validate-only, synthetic-test, or explicit formal execution")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not str(result.get("status", "")).startswith("HOLD") else 2


if __name__ == "__main__":
    raise SystemExit(main())
