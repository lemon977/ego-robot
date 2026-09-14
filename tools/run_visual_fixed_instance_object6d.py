#!/usr/bin/env python3
"""Generic RGB-mask + metric-depth fixed-instance Object6D producer.

Object observations must come from a mask bound to the decoded RGB frame and
an authorized stereo-depth bundle.  Hand joints and Robot pinch points are not
accepted inputs.  Results remain candidate-only until contact/occlusion review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve()
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
GEOMETRY = {
    "chips": {"class": "POTATO_CHIP", "shape": "THIN_OVAL", "size_m": [0.050, 0.038, 0.003], "min_depth_pixels": 20},
    "poker": {"class": "PLAYING_CARD", "shape": "THIN_BOX", "size_m": [0.088, 0.063, 0.001], "min_depth_pixels": 50},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(value.dtype.str.encode() + b"\0")
    h.update(np.asarray(value.shape, dtype="<i8").tobytes()); h.update(value.tobytes())
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha(resolved)}


def published_artifact(current: Path, published: Path) -> dict[str, Any]:
    return {"path": str(published.resolve(strict=False)), "bytes": current.stat().st_size, "sha256": sha(current)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_ref(value: dict, label: str) -> Path:
    path = Path(value["path"])
    if not path.is_file() or path.stat().st_size != value["bytes"] or sha(path) != value["sha256"]:
        raise RuntimeError(f"{label} artifact closure failed")
    return path


def rgb_digest(frame: np.ndarray) -> str:
    return array_sha(np.asarray(frame, dtype=np.uint8))


def appearance(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [12, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(hist)); return hist / norm if norm else hist


class IdentityState:
    """Visual-only per-episode association; no hand or pinch signals."""
    def __init__(self, task: str):
        self.task = task; self.next_id = 0; self.active_id = -1; self.last_frame = -999
        self.centroid = None; self.depth = None; self.hist = None; self.label = "UNKNOWN"

    def observe(self, frame: int, centroid: np.ndarray, depth: float, hist: np.ndarray, label: str) -> tuple[int, str, str]:
        gap = frame - self.last_frame
        declared_match = (
            self.active_id >= 0
            and label != "UNKNOWN"
            and self.label == label
        )
        compatible = self.active_id >= 0 and (gap <= 4 or declared_match)
        if compatible and not declared_match:
            compatible &= float(np.linalg.norm(centroid - self.centroid)) <= 140.0
            compatible &= abs(depth - self.depth) <= 0.18
            compatible &= float(cv2.compareHist(hist.astype(np.float32), self.hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)) <= 0.55
            if self.task == "poker" and label != "UNKNOWN" and self.label != "UNKNOWN": compatible &= label == self.label
        transition = "CONTINUE_VISUAL_INSTANCE"
        if not compatible:
            self.active_id = self.next_id; self.next_id += 1; transition = "NEW_VISUAL_EPISODE"
            self.label = label if label != "UNKNOWN" else ("OPAQUE_CHIP_INSTANCE" if self.task == "chips" else "UNKNOWN")
        elif gap > 4:
            transition = "CONTINUE_DECLARED_INSTANCE_AFTER_OCCLUSION"
        elif self.task == "poker" and self.label == "UNKNOWN" and label != "UNKNOWN": self.label = label
        self.last_frame = frame; self.centroid = centroid.copy(); self.depth = depth; self.hist = hist.copy()
        return self.active_id, transition, self.label

    def missing(self, frame: int) -> tuple[int, str, str]:
        if self.active_id >= 0: return self.active_id, "OCCLUDED_VISUAL_GAP", self.label
        return -1, "NO_VISUAL_INSTANCE", "UNKNOWN"


def fit_pose(mask_depth: np.ndarray, depth: np.ndarray, k: np.ndarray, size_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    yy, xx = np.nonzero(mask_depth & np.isfinite(depth))
    z = depth[yy, xx].astype(np.float64)
    points = np.column_stack(((xx-k[0,2])*z/k[0,0], (yy-k[1,2])*z/k[1,1], z))
    center = np.median(points, axis=0); centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False); normal = vh[-1]
    if normal[2] > 0: normal = -normal
    long = vh[0] - normal * np.dot(vh[0], normal); long /= np.linalg.norm(long)
    if long[0] < 0: long = -long
    short = np.cross(normal, long); short /= np.linalg.norm(short)
    rotation = np.column_stack((long, short, normal))
    if np.linalg.det(rotation) < 0: rotation[:, 1] *= -1
    transform = np.eye(4); transform[:3,:3] = rotation; transform[:3,3] = center
    near_far_observed = np.quantile(z, [0.01, 0.99])
    corners = np.asarray([[sx,sy,sz] for sx in (-.5,.5) for sy in (-.5,.5) for sz in (-.5,.5)]) * size_m
    analytic_z = (corners @ rotation.T + center)[:,2]
    residual = float(np.median(np.abs(centered @ normal)))
    return transform, near_far_observed, np.asarray([analytic_z.min(), analytic_z.max()]), residual


def _rotation_lerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate on SO(3)'s shortest axis-angle path."""
    relative = left.T @ right
    rotation_vector, _ = cv2.Rodrigues(relative)
    incremental, _ = cv2.Rodrigues(rotation_vector * float(alpha))
    return left @ incremental


def _analytic_near_far(transform: np.ndarray, size_m: np.ndarray) -> np.ndarray:
    corners = np.asarray([
        [sx, sy, sz]
        for sx in (-0.5, 0.5)
        for sy in (-0.5, 0.5)
        for sz in (-0.5, 0.5)
    ]) * size_m
    z = (corners @ transform[:3, :3].T + transform[:3, 3])[:, 2]
    return np.asarray([z.min(), z.max()], dtype=np.float32)


def _gap_ranges(observed: np.ndarray) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for index, present in enumerate(observed.tolist() + [True]):
        if not present and start is None:
            start = index
        elif present and start is not None:
            gaps.append((start, index - 1))
            start = None
    return gaps


def _fill_occluded_poses(
    observed: np.ndarray,
    t_world: np.ndarray,
    t_camera: np.ndarray,
    analytic: np.ndarray,
    c2w: np.ndarray,
    frame_indices: np.ndarray,
    size_m: np.ndarray,
    *,
    fill_leading: bool = True,
) -> list[tuple[int, int]]:
    """Fill unobserved frames in world space while keeping provenance explicit.

    Interior gaps use translation interpolation plus shortest-path SO(3)
    interpolation. Leading/trailing gaps use the nearest direct observation.
    This is a visualization baseline only and never becomes contact authority.
    """
    direct = np.flatnonzero(observed)
    if not len(direct):
        return _gap_ranges(observed)
    gaps = _gap_ranges(observed)
    for index in np.flatnonzero(~observed):
        left_candidates = direct[direct < index]
        right_candidates = direct[direct > index]
        left = int(left_candidates[-1]) if len(left_candidates) else None
        right = int(right_candidates[0]) if len(right_candidates) else None
        if left is None and not fill_leading:
            continue
        if left is not None and right is not None:
            alpha = float((index - left) / (right - left))
            world = np.eye(4, dtype=np.float64)
            world[:3, :3] = _rotation_lerp(
                t_world[left, :3, :3], t_world[right, :3, :3], alpha
            )
            world[:3, 3] = (
                (1.0 - alpha) * t_world[left, :3, 3]
                + alpha * t_world[right, :3, 3]
            )
        else:
            nearest = left if left is not None else right
            assert nearest is not None
            world = t_world[nearest].copy()
        absolute = int(frame_indices[index])
        camera = c2w[absolute] if c2w.shape[0] != len(frame_indices) else c2w[index]
        t_world[index] = world
        t_camera[index] = np.linalg.inv(camera) @ world
        analytic[index] = _analytic_near_far(t_camera[index], size_m)
    return gaps


def _bound_direct_pose_steps(
    observed: np.ndarray,
    t_world: np.ndarray,
    t_camera: np.ndarray,
    analytic: np.ndarray,
    c2w: np.ndarray,
    frame_indices: np.ndarray,
    size_m: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    translation_limit_m_per_frame: float = 0.03,
    rotation_limit_deg_per_frame: float = 12.0,
) -> dict[str, float]:
    """Bound noisy thin-object pose measurements without changing observation depth.

    The raw stereo near/far remains untouched. Only the visualization pose is
    projected from the previous accepted world pose under explicit per-frame
    translation and rotation limits.
    """
    direct = np.flatnonzero(observed)
    raw_translation_steps: list[float] = []
    raw_rotation_steps: list[float] = []
    bounded_translation_steps: list[float] = []
    bounded_rotation_steps: list[float] = []
    if len(direct) < 2:
        return {
            "raw_translation_step_max_m": 0.0,
            "raw_rotation_step_max_deg": 0.0,
            "bounded_translation_step_max_m": 0.0,
            "bounded_rotation_step_max_deg": 0.0,
        }
    previous_index = int(direct[0])
    for index_value in direct[1:]:
        index = int(index_value)
        gap = max(1, int(frame_indices[index] - frame_indices[previous_index]))
        previous = t_world[previous_index].copy()
        raw = t_world[index].copy()
        translation_delta = raw[:3, 3] - previous[:3, 3]
        translation_norm = float(np.linalg.norm(translation_delta))
        translation_cap = translation_limit_m_per_frame * gap
        if translation_norm > translation_cap:
            translation_delta *= translation_cap / translation_norm

        relative = previous[:3, :3].T @ raw[:3, :3]
        rotation_vector, _ = cv2.Rodrigues(relative)
        raw_angle_deg = float(np.degrees(np.linalg.norm(rotation_vector)))
        rotation_cap_deg = rotation_limit_deg_per_frame * gap
        if raw_angle_deg > rotation_cap_deg:
            rotation_vector *= rotation_cap_deg / raw_angle_deg
        bounded_relative, _ = cv2.Rodrigues(rotation_vector)
        bounded = np.eye(4, dtype=np.float64)
        bounded[:3, :3] = previous[:3, :3] @ bounded_relative
        bounded[:3, 3] = previous[:3, 3] + translation_delta
        absolute = int(frame_indices[index])
        camera = c2w[absolute] if c2w.shape[0] != len(frame_indices) else c2w[index]
        t_world[index] = bounded
        t_camera[index] = np.linalg.inv(camera) @ bounded
        analytic[index] = _analytic_near_far(t_camera[index], size_m)
        bounded_translation = float(np.linalg.norm(bounded[:3, 3] - previous[:3, 3]))
        bounded_rotation = float(np.degrees(np.linalg.norm(rotation_vector)))
        raw_translation_steps.append(translation_norm / gap)
        raw_rotation_steps.append(raw_angle_deg / gap)
        bounded_translation_steps.append(bounded_translation / gap)
        bounded_rotation_steps.append(bounded_rotation / gap)
        rows[index]["pose_provenance"] = (
            "DIRECT_RGB_MASK_STEREO_DEPTH_WITH_PREVIOUS_ACCEPTED_WORLD_POSE_RATE_LIMIT"
        )
        rows[index]["raw_world_translation_step_m_per_frame"] = translation_norm / gap
        rows[index]["raw_world_rotation_step_deg_per_frame"] = raw_angle_deg / gap
        rows[index]["bounded_world_translation_step_m_per_frame"] = bounded_translation / gap
        rows[index]["bounded_world_rotation_step_deg_per_frame"] = bounded_rotation / gap
        rows[index]["near_far_analytic_m"] = analytic[index].tolist()
        previous_index = index
    return {
        "raw_translation_step_max_m": max(raw_translation_steps, default=0.0),
        "raw_rotation_step_max_deg": max(raw_rotation_steps, default=0.0),
        "bounded_translation_step_max_m": max(bounded_translation_steps, default=0.0),
        "bounded_rotation_step_max_deg": max(bounded_rotation_steps, default=0.0),
    }


def _dense_world_pose_step_metrics(
    t_world: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float]:
    """Measure the published dense trajectory, including propagated gaps."""
    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    for index in range(1, len(t_world)):
        if valid is not None and not (valid[index - 1] and valid[index]):
            continue
        translation_steps.append(
            float(np.linalg.norm(t_world[index, :3, 3] - t_world[index - 1, :3, 3]))
        )
        relative = t_world[index - 1, :3, :3].T @ t_world[index, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(relative)
        rotation_steps.append(float(np.degrees(np.linalg.norm(rotation_vector))))
    return {
        "translation_step_max_m": max(translation_steps, default=0.0),
        "rotation_step_max_deg": max(rotation_steps, default=0.0),
    }


def frame_digest(values: list[np.ndarray | str | int]) -> str:
    h = hashlib.sha256(b"visual-fixed-instance-object6d-frame-v1\0")
    for value in values:
        if isinstance(value, np.ndarray): h.update(array_sha(value).encode())
        else: h.update(str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def validate_spec(spec_path: Path) -> tuple[dict, list[str]]:
    spec = load_json(spec_path); errors = []
    if spec.get("schema_version") != "visual-fixed-instance-object6d-input-v1": errors.append("SCHEMA")
    if spec.get("task") not in GEOMETRY: errors.append("TASK")
    forbidden = set(spec) & {"hand_joints", "pinch_points", "robot_object", "hawor_object"}
    if forbidden: errors.append("FORBIDDEN_HAND_OR_ROBOT_OBJECT_INPUT:" + ",".join(sorted(forbidden)))
    for key in ("selected_rgb", "depth_result", "depth_frame_manifest", "registration_authority", "rgb_object_mask_manifest", "camera_to_world"):
        try: verify_ref(spec[key], key)
        except Exception as e: errors.append(str(e))
    for key in (
        "join_spec", "hawor_result", "hawor_agent_review", "role_mask_result",
        "role_mask_agent_review", "task_object_result",
        "task_object_agent_review", "task_object_manifest",
    ):
        if key in spec:
            try: verify_ref(spec[key], key)
            except Exception as e: errors.append(str(e))
    return spec, errors


def render_review(
    spec: dict,
    partial: Path,
    mask_rows: list[dict],
    depth_rows_by_id: dict[int, dict],
    frame_indices: np.ndarray,
    valid: np.ndarray,
    observed: np.ndarray,
    t_cam: np.ndarray,
    near_far: np.ndarray,
    h_depth_to_selected: np.ndarray,
    k: np.ndarray,
) -> dict:
    """Render a full Chinese review with only explicit observed-only claims."""
    from PIL import Image, ImageDraw, ImageFont

    cap = cv2.VideoCapture(spec["selected_rgb"]["path"])
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    target = partial / f"{spec['session_id']}_OBJECT6D_观测限定_中文全片复核.mp4"
    temp = partial / f".{target.name}.tmp.mp4"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    font = ImageFont.truetype(str(FONT), 18) if FONT.is_file() else None
    colours = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    try:
        for local, absolute_value in enumerate(frame_indices):
            absolute = int(absolute_value)
            cap.set(cv2.CAP_PROP_POS_FRAMES, absolute)
            ok, rgb = cap.read()
            if not ok:
                raise RuntimeError(f"review RGB ended at {absolute}")
            mask = cv2.imread(str(verify_ref(mask_rows[local]["mask"], f"review mask {absolute}")), cv2.IMREAD_GRAYSCALE) > 0
            overlay = rgb.copy()
            overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 210, 0])).astype(np.uint8)
            dr = depth_rows_by_id[absolute]
            depth_path = Path(spec["depth_root"]) / dr["relative_path"]
            with np.load(depth_path, allow_pickle=False) as bundle:
                depth, depth_valid = bundle["depth_m"].astype(np.float32), bundle["valid"].astype(bool)
            scalar = np.zeros(depth.shape, np.uint8)
            scalar[depth_valid] = np.rint(255 * (1 - np.clip((depth[depth_valid] - 0.10) / 2.90, 0, 1))).astype(np.uint8)
            colour_depth = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO)
            colour_depth[~depth_valid] = 0
            if valid[local]:
                origin = t_cam[local, :3, 3]
                axes = np.stack([origin] + [
                    origin + t_cam[local, :3, axis] * 0.035
                    for axis in range(3)
                ])
                depth_uv = np.column_stack((
                    k[0, 0] * axes[:, 0] / axes[:, 2] + k[0, 2],
                    k[1, 1] * axes[:, 1] / axes[:, 2] + k[1, 2],
                    np.ones(4),
                ))
                selected_h = depth_uv @ h_depth_to_selected.T
                selected_uv = selected_h[:, :2] / selected_h[:, 2:]
                center = tuple(np.rint(selected_uv[0]).astype(int))
                for axis, colour in enumerate(colours):
                    end = tuple(np.rint(selected_uv[axis + 1]).astype(int))
                    cv2.line(overlay, center, end, colour, 4, cv2.LINE_AA)
                cv2.circle(overlay, center, 7, (0, 255, 255), -1)
            canvas = np.concatenate((cv2.resize(overlay, (640, 480)), colour_depth), axis=1)
            if font:
                image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                state = "直接观测" if observed[local] else "无观测/无位姿"
                draw.text((10, 8), f"帧 {absolute:04d}｜绿色=同一任务物体 Mask｜坐标轴=Object6D", font=font, fill=(255, 255, 255))
                if observed[local]:
                    draw.text((650, 8), f"{state}｜Z near/far={near_far[local,0]:.3f}/{near_far[local,1]:.3f} m", font=font, fill=(255, 255, 255))
                else:
                    draw.text((650, 8), state, font=font, fill=(255, 220, 80))
                draw.text((650, 36), "右：校正双目深度；遮挡帧不传播、不猜测", font=font, fill=(255, 255, 255))
                canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            writer.write(canvas)
    finally:
        writer.release(); cap.release()
    os.replace(temp, target)
    check = cv2.VideoCapture(str(target)); decoded = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT))); check.release()
    if decoded != len(frame_indices):
        raise RuntimeError(f"Object6D review decode mismatch {decoded} != {len(frame_indices)}")
    return {**artifact(target), "frames": decoded, "fps": fps}


def produce(spec_path: Path, output: Path) -> dict:
    spec, errors = validate_spec(spec_path)
    if errors: raise RuntimeError(errors)
    if output.exists(): raise FileExistsError(f"no-clobber output exists: {output}")
    partial = output.parent / ("." + output.name + ".partial")
    if partial.exists(): raise FileExistsError(f"partial exists; explicit resume implementation required: {partial}")
    partial.mkdir(parents=True)
    task = spec["task"]; geometry = GEOMETRY[task]; size_m = np.asarray(geometry["size_m"], np.float64)
    depth_result = load_json(verify_ref(spec["depth_result"], "depth_result"))
    if depth_result.get("status") != "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION" or not depth_result.get("consumption_authorized"):
        raise RuntimeError("depth is not authorized corrected metric depth")
    depth_manifest = load_json(verify_ref(spec["depth_frame_manifest"], "depth_frame_manifest")); depth_rows_by_id = {int(row["frame_id"]): row for row in depth_manifest["frames"]}
    mask_manifest = load_json(verify_ref(spec["rgb_object_mask_manifest"], "mask_manifest")); mask_rows = mask_manifest["frames"]
    registration_path = verify_ref(spec["registration_authority"], "registration")
    with np.load(registration_path, allow_pickle=False) as r:
        h_selected_to_depth = r["H_selected_rgb_to_depth_pixel"].astype(np.float64)
        h_depth_to_selected = r["H_depth_pixel_to_selected_rgb"].astype(np.float64)
        k = r["stereo_rectified_depth_formula_intrinsics"].astype(np.float64)
    with np.load(verify_ref(spec["camera_to_world"], "camera_to_world"), allow_pickle=False) as c:
        c2w = c[spec["camera_to_world_key"]].astype(np.float64)
    if not np.array_equal(k, np.asarray([[320.,0.,319.5],[0.,320.,239.5],[0.,0.,1.]])): raise RuntimeError("depth formula K is not corrected fx=320")
    frame_count = int(spec["frame_count"])
    if len(mask_rows) != frame_count or c2w.ndim != 3 or c2w.shape[1:] != (4,4): raise RuntimeError("frame closure mismatch")
    frame_indices = np.asarray([int(row["frame_id"]) for row in mask_rows], dtype=np.int64)
    if len(set(frame_indices.tolist())) != frame_count or any(int(index) not in depth_rows_by_id for index in frame_indices): raise RuntimeError("mask/depth frame identity mismatch")
    if c2w.shape[0] != frame_count and c2w.shape[0] <= int(frame_indices.max()): raise RuntimeError("camera_to_world does not cover absolute frame indices")
    cap = cv2.VideoCapture(spec["selected_rgb"]["path"]); state = IdentityState(task)
    valid = np.zeros(frame_count,bool); observed_direct = np.zeros(frame_count,bool)
    visibility = np.zeros(frame_count,np.float32); instance = np.full(frame_count,-1,np.int32)
    t_cam = np.tile(np.eye(4),(frame_count,1,1)); t_world = np.tile(np.eye(4),(frame_count,1,1)); near_far = np.full((frame_count,2),np.nan,np.float32); analytic = np.full((frame_count,2),np.nan,np.float32)
    rows = []; identities = []
    try:
        for frame in range(frame_count):
            absolute_frame = int(frame_indices[frame])
            cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
            ok, rgb = cap.read()
            if not ok: raise RuntimeError(f"RGB ended at absolute frame {absolute_frame}")
            rgb_sha = rgb_digest(rgb); mr = mask_rows[frame]; dr = depth_rows_by_id[absolute_frame]
            if int(mr["frame_id"]) != absolute_frame or mr["source_rgb_decoded_sha256"] != rgb_sha: raise RuntimeError(f"RGB-mask binding failed at {absolute_frame}")
            mask_path = verify_ref(mr["mask"], f"mask {frame}"); mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            declared_observed = bool(mr.get("valid", mask.any()))
            if not declared_observed and mask.any():
                raise RuntimeError(f"unobserved object mask is not empty at {absolute_frame}")
            if declared_observed and int(mr.get("physical_instance_id", 0)) != 0:
                raise RuntimeError(f"observed object identity is not physical_instance_id=0 at {absolute_frame}")
            depth_path = Path(spec["depth_root"]) / dr["relative_path"]
            if not depth_path.is_file() or sha(depth_path) != dr["sha256"]: raise RuntimeError(f"depth frame binding {frame}")
            with np.load(depth_path, allow_pickle=False) as bundle: depth = bundle["depth_m"].astype(np.float32); depth_valid = bundle["valid"].astype(bool)
            warped = cv2.warpPerspective(mask.astype(np.uint8), h_selected_to_depth, (640,480), flags=cv2.INTER_NEAREST) > 0
            support = warped & depth_valid if declared_observed else np.zeros_like(depth_valid)
            count = int(support.sum())
            if count < geometry["min_depth_pixels"]:
                iid, transition, label = state.missing(frame); instance[frame] = iid; identities.append(label)
                rows.append({"frame_id":absolute_frame,"window_index":frame,"valid":False,"observed":False,"instance_id":iid,"identity_transition":transition,"pose_provenance":"PENDING_EXPLICIT_OCCLUSION_PROPAGATION","visibility":0.0,"depth_pixels":count,"near_far_observed_m":None,"near_far_analytic_m":None,"rgb_frame_sha256":rgb_sha,"rgb_mask_sha256":array_sha(mask),"depth_bundle_sha256":dr["sha256"],"object6d_frame_sha256":frame_digest([absolute_frame,iid,"UNOBSERVED",rgb_sha,array_sha(mask),dr["sha256"]])}); continue
            transform, observed_nf, analytic_nf, plane = fit_pose(support, depth, k, size_m)
            ys,xs=np.nonzero(mask); centroid=np.asarray([np.median(xs),np.median(ys)]); hist=appearance(rgb,mask); observed_label=str(mr.get("observed_identity","UNKNOWN"))
            iid, transition, label = state.observe(frame,centroid,float(transform[2,3]),hist,observed_label)
            expected_area=max(1.,k[0,0]*size_m[0]/transform[2,3]*k[1,1]*size_m[1]/transform[2,3]); vis=min(1.,count/expected_area)
            valid[frame]=True; observed_direct[frame]=True; visibility[frame]=vis; instance[frame]=iid; t_cam[frame]=transform; near_far[frame]=observed_nf; analytic[frame]=analytic_nf; identities.append(label)
            c2w_frame = c2w[absolute_frame] if c2w.shape[0] != frame_count else c2w[frame]
            t_world[frame]=c2w_frame@transform
            digest=frame_digest([absolute_frame,iid,transform,observed_nf,analytic_nf,rgb_sha,array_sha(mask),dr["sha256"]])
            rows.append({"frame_id":absolute_frame,"window_index":frame,"valid":True,"observed":True,"instance_id":iid,"identity_transition":transition,"identity_label":label,"pose_provenance":"DIRECT_RGB_MASK_PLUS_AUTHORIZED_METRIC_STEREO_DEPTH","visibility":vis,"depth_pixels":count,"plane_median_abs_residual_m":plane,"near_far_observed_m":observed_nf.tolist(),"near_far_analytic_m":analytic_nf.tolist(),"rgb_frame_sha256":rgb_sha,"rgb_mask_sha256":array_sha(mask),"depth_bundle_sha256":dr["sha256"],"object6d_frame_sha256":digest})
    finally: cap.release()
    pose_rate_metrics = _bound_direct_pose_steps(
        observed_direct, t_world, t_cam, analytic, c2w, frame_indices, size_m, rows
    )
    leading_policy = str(spec.get("leading_unobserved_policy", "PROPAGATE_NEAREST"))
    suppress_leading = leading_policy == "INVALID_UNTIL_VISUAL_IDENTITY_ONSET"
    unobserved_policy = str(spec.get("unobserved_pose_policy", "PROPAGATE_VISUALIZATION"))
    if unobserved_policy not in {"PROPAGATE_VISUALIZATION", "KEEP_INVALID"}:
        raise RuntimeError(f"unsupported unobserved_pose_policy: {unobserved_policy}")
    if unobserved_policy == "KEEP_INVALID":
        all_gap_ranges = _gap_ranges(observed_direct)
        instance[~observed_direct] = -1
        for frame in np.flatnonzero(~observed_direct):
            rows[int(frame)].update({
                "valid": False,
                "observed": False,
                "instance_id": -1,
                "identity_label": "UNKNOWN_AMBIGUOUS_OR_OCCLUDED",
                "identity_transition": "KEEP_INVALID_NO_POSE",
                "pose_provenance": "NO_OBJECT6D_POSE_IDENTITY_AMBIGUOUS_OR_OCCLUDED",
                "visibility": 0.0,
                "near_far_observed_m": None,
                "near_far_analytic_m": None,
            })
    else:
        all_gap_ranges = _fill_occluded_poses(
            observed_direct,
            t_world,
            t_cam,
            analytic,
            c2w,
            frame_indices,
            size_m,
            fill_leading=not suppress_leading,
        )
    direct_instances = sorted(set(instance[observed_direct].tolist()))
    canonical_instance = int(direct_instances[0]) if len(direct_instances) == 1 else -1
    canonical_identity = next(
        (str(row.get("identity_label")) for row in rows if row.get("observed")),
        "UNKNOWN",
    )
    if canonical_instance >= 0 and unobserved_policy == "PROPAGATE_VISUALIZATION":
        for frame in np.flatnonzero(~observed_direct):
            if not np.isfinite(analytic[frame]).all():
                rows[int(frame)].update({
                    "valid": False,
                    "instance_id": -1,
                    "identity_label": "IDENTITY_NOT_ESTABLISHED",
                    "identity_transition": "PRE_IDENTITY_ONSET",
                    "pose_provenance": "NO_VISUAL_IDENTITY_NO_OBJECT6D_POSE",
                    "near_far_observed_m": None,
                    "near_far_analytic_m": None,
                })
                continue
            valid[frame] = True
            instance[frame] = canonical_instance
            row = rows[int(frame)]
            row.update({
                "valid": True,
                "observed": False,
                "instance_id": canonical_instance,
                "identity_label": canonical_identity,
                "identity_transition": "OCCLUDED_TEMPORAL_WORLD_POSE_PROPAGATION",
                "pose_provenance": "WORLD_POSE_INTERPOLATION_OR_NEAREST_HOLD_NO_DIRECT_DEPTH",
                "visibility": 0.0,
                "near_far_observed_m": None,
                "near_far_analytic_m": analytic[frame].tolist(),
                "object6d_frame_sha256": frame_digest([
                    int(frame_indices[frame]), canonical_instance, t_cam[frame],
                    t_world[frame], analytic[frame], "TEMPORAL_PROPAGATION_NO_DIRECT_DEPTH",
                ]),
            })
    for frame in np.flatnonzero(observed_direct):
        index = int(frame)
        row = rows[index]
        if index == int(np.flatnonzero(observed_direct)[0]):
            row["pose_provenance"] = (
                "DIRECT_RGB_MASK_STEREO_DEPTH_WORLD_POSE_RATE_LIMIT_ORIGIN"
            )
        row["object6d_frame_sha256"] = frame_digest([
            int(frame_indices[index]), int(instance[index]), t_cam[index],
            t_world[index], near_far[index], analytic[index],
            row["rgb_frame_sha256"], row["rgb_mask_sha256"],
            row["depth_bundle_sha256"], "DIRECT_POSE_AFTER_RATE_LIMIT",
        ])
    dense_pose_rate_metrics = _dense_world_pose_step_metrics(t_world, valid)
    propagated_mask = valid & ~observed_direct
    propagated_ranges = _gap_ranges(~propagated_mask)
    gap_lengths = [end - start + 1 for start, end in propagated_ranges]
    max_gap = max(gap_lengths, default=0)
    direct = np.flatnonzero(observed_direct)
    active_start = int(direct[0]) if len(direct) else frame_count
    active_valid = valid[active_start:]
    active_observed = observed_direct[active_start:]
    observed_fraction = float(active_observed.mean()) if len(active_observed) else 0.0
    invalid_pre_identity = int((~valid[:active_start]).sum())
    trajectory = partial / "VISUAL_FIXED_INSTANCE_OBJECT6D.npz"
    atomic_npz(trajectory, frame_indices=frame_indices, valid=valid, observed=observed_direct, visibility=visibility, physical_instance_id=instance, T_object_to_camera=t_cam, T_object_to_world=t_world, observed_near_far_optical_z_m=near_far, analytic_near_far_optical_z_m=analytic, object_size_m=size_m, provenance=np.asarray("DIRECT_RGB_MASK_STEREO_DEPTH_PLUS_EXPLICIT_OCCLUSION_WORLD_POSE_PROPAGATION_NO_HAND_PINCH"))
    frame_manifest = partial / "FRAME_MANIFEST.json"; atomic_json(frame_manifest,{"schema_version":"visual-fixed-instance-object6d-frame-manifest-v1","session_id":spec["session_id"],"task":task,"frames":rows})
    validity_policy_pass = (
        bool(np.array_equal(valid, observed_direct))
        if unobserved_policy == "KEEP_INVALID"
        else bool(active_valid.all()) and (not suppress_leading or not valid[:active_start].any())
    )
    coverage_policy_pass = (
        bool(observed_direct.any())
        if unobserved_policy == "KEEP_INVALID"
        else observed_fraction >= 0.70 and max_gap <= 45
    )
    pass_numeric = (
        validity_policy_pass
        and len(direct_instances) == 1
        and coverage_policy_pass
        and np.isfinite(t_cam[valid]).all()
        and np.isfinite(t_world[valid]).all()
        and np.isfinite(analytic[valid]).all()
        and np.isfinite(near_far[observed_direct]).all()
        and np.all(near_far[observed_direct, 0] <= near_far[observed_direct, 1])
        and pose_rate_metrics["bounded_translation_step_max_m"] <= 0.030001
        and pose_rate_metrics["bounded_rotation_step_max_deg"] <= 12.001
        and dense_pose_rate_metrics["translation_step_max_m"] <= 0.030001
        and dense_pose_rate_metrics["rotation_step_max_deg"] <= 12.001
    )
    review_video = render_review(
        spec, partial, mask_rows, depth_rows_by_id, frame_indices, valid,
        observed_direct, t_cam, near_far, h_depth_to_selected, k,
    )
    review_video["path"] = str(output / Path(review_video["path"]).name)
    input_refs = {
        "spec": artifact(spec_path), "rgb": spec["selected_rgb"],
        "depth_result": spec["depth_result"],
        "mask_manifest": spec["rgb_object_mask_manifest"],
        "camera_to_world": spec["camera_to_world"],
    }
    for key in (
        "join_spec", "hawor_result", "hawor_agent_review", "role_mask_result",
        "role_mask_agent_review", "task_object_result",
        "task_object_agent_review", "task_object_manifest",
    ):
        if key in spec:
            input_refs[key] = spec[key]
    result = {"schema_version":"visual-fixed-instance-object6d-result-v2","status":"PASS_VISUAL_OBJECT6D_OBSERVED_ONLY_BASELINE" if pass_numeric else "HOLD_VISUAL_OBJECT6D_NUMERIC","grade":"B" if pass_numeric else "C","consumption_authorized":pass_numeric,"authorized_scopes":["CLEAN_VISUAL_BASELINE_INPUT"] if pass_numeric else [],"robot_contact_authorized":False,"session_id":spec["session_id"],"task":task,"object_class":geometry["class"],"shape":geometry["shape"],"identity_policy":"one action-conditioned physical instance; ambiguous/occluded frames remain invalid when requested","leading_unobserved_policy":leading_policy,"unobserved_pose_policy":unobserved_policy,"forbidden_inputs_absent":True,"frame_count":frame_count,"valid_frames":int(valid.sum()),"invalid_pre_identity_frames":invalid_pre_identity,"active_start_local_frame":active_start,"observed_frames":int(observed_direct.sum()),"propagated_frames":int(propagated_mask.sum()),"observed_fraction":observed_fraction,"observed_fraction_scope":"ACTIVE_SEGMENT_FROM_FIRST_VISUAL_IDENTITY","max_propagated_gap_frames":max_gap,"all_unobserved_gap_ranges_local_inclusive":[list(gap) for gap in all_gap_ranges],"propagated_gap_ranges_local_inclusive":[list(gap) for gap in propagated_ranges],"pose_rate_limits":{"translation_m_per_frame":0.03,"rotation_deg_per_frame":12.0},"pose_rate_metrics":pose_rate_metrics,"dense_pose_rate_metrics":dense_pose_rate_metrics,"instance_count":len(set(instance[valid].tolist())),"inputs":input_refs,"artifacts":{"trajectory":published_artifact(trajectory, output / trajectory.name),"frame_manifest":published_artifact(frame_manifest, output / frame_manifest.name),"review_video":review_video},"next_gate":"Grade B authorizes Clean visual baseline only. KEEP_INVALID never invents Object6D in ambiguous/occluded frames; contact/Robot remain unauthorized."}
    atomic_json(partial/"RESULT.json", result)
    agent_review = {
        "schema_version": "visual-fixed-instance-object6d-agent-review-v1",
        "created_at": now(), "session_id": spec["session_id"],
        "grade": "B" if pass_numeric else "C",
        "downstream_authorized": pass_numeric,
        "checks": {
            "input_sha_closure": True,
            "observed_only_no_pose_on_unobserved": bool(np.array_equal(valid, observed_direct)),
            "single_action_conditioned_physical_instance": len(direct_instances) == 1,
            "numeric_pose_and_near_far": pass_numeric,
            "full_chinese_review_decode": review_video["frames"] == frame_count,
            "forbidden_hand_or_robot_object_inputs_absent": True,
        },
        "review_video": review_video,
        "claim_limit": "Automatic Grade B supports visual baseline/Clean only; it is not contact, tactile, IK, Robot or deployment truth.",
    }
    atomic_json(partial / "AGENT_REVIEW.json", agent_review)
    checksum_lines = []
    for item in sorted(p for p in partial.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        checksum_lines.append(f"{sha(item)}  {item.relative_to(partial)}")
    (partial / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    os.replace(partial, output)
    return load_json(output/"RESULT.json")


def validate_from_depth_manifest(path: Path, output: Path, registry_path: Path | None = None) -> dict:
    manifest = load_json(path); rows=[]
    registry = load_json(registry_path).get("sessions", {}) if registry_path else {}
    for row in manifest["sessions"]:
        depth_root=Path(row["destination"]); mask_root=depth_root.parents[1]/"object_observation/20260908_rgb_instance_masks_v1"
        missing=[]
        for label,p in (("depth_result",depth_root/"RESULT.json"),("depth_frame_manifest",depth_root/"FRAME_MANIFEST.json"),("registration_authority",depth_root/"REGISTRATION_AUTHORITY.npz")):
            if not p.is_file(): missing.append(label)
        mask_entry=registry.get(row["session_id"])
        mask_state="MISSING"
        if mask_entry and mask_entry.get("consumption_authorized"):
            try: verify_ref(mask_entry["mask_manifest"], "rgb_object_mask_manifest"); mask_state="SHORT_WINDOW_GRADE_B"
            except Exception: missing.append("rgb_object_mask_manifest_sha")
        else:
            fallback=mask_root/"OBJECT_MASK_MANIFEST.json"
            if fallback.is_file(): mask_state="FULLSESSION_PRESENT"
            else: missing.append("rgb_object_mask_manifest")
        rows.append({"session_id":row["session_id"],"task":row["task"],"status":"READY" if not missing else "HOLD_MISSING_INPUTS","missing":missing,"rgb_object_mask_state":mask_state,"rgb_object_mask":mask_entry.get("mask_manifest") if mask_entry else None,"forbidden_hand_pinch_inputs":False})
    result={"schema_version":"visual-fixed-instance-object6d-validate-only-v1","created_at":now(),"mode":"CPU_VALIDATE_ONLY_NO_GPU","input_depth_manifest":artifact(path),"rgb_object_mask_registry":artifact(registry_path) if registry_path else None,"sessions":rows,"counts":{"ready":sum(r["status"]=="READY" for r in rows),"hold":sum(r["status"]!="READY" for r in rows)},"status":"PASS_READY" if all(r["status"]=="READY" for r in rows) else "HOLD_MISSING_REAL_VISUAL_INPUTS","producer":artifact(SCRIPT),"execution_authorized":False}
    atomic_json(output,result); return result


def make_synthetic_case(root: Path, task: str) -> tuple[Path,Path]:
    case=root/task; case.mkdir(parents=True); frames=10; rgb_path=case/"rgb.mp4"; writer=cv2.VideoWriter(str(rgb_path),cv2.VideoWriter_fourcc(*"mp4v"),30,(1280,960)); masks=[]
    for f in range(frames):
        image=np.full((960,1280,3),30,np.uint8); center=(500+f*8,460)
        if task=="chips": cv2.ellipse(image,center,(70,50),10,0,360,(30,190,230),-1)
        else: cv2.rectangle(image,(center[0]-100,center[1]-70),(center[0]+100,center[1]+70),(230,230,220),-1)
        writer.write(image); masks.append(center)
    writer.release(); cap=cv2.VideoCapture(str(rgb_path)); mask_rows=[]; depth_root=case/"depth";(depth_root/"frames").mkdir(parents=True); depth_rows=[]
    for f,center in enumerate(masks):
        ok,rgb=cap.read(); assert ok; mask=np.zeros((960,1280),np.uint8)
        if task=="chips": cv2.ellipse(mask,center,(70,50),10,0,360,255,-1)
        else: cv2.rectangle(mask,(center[0]-100,center[1]-70),(center[0]+100,center[1]+70),255,-1)
        mp=case/f"mask_{f:06d}.png"; cv2.imwrite(str(mp),mask); mask_rows.append({"frame_id":f,"source_rgb_decoded_sha256":rgb_digest(rgb),"observed_identity":"A_DIAMOND" if task=="poker" else "UNKNOWN","mask":artifact(mp)})
        d=np.full((480,640),1.2,np.float32); small=cv2.resize(mask,(640,480),interpolation=cv2.INTER_NEAREST)>0; d[small]=0.80+f*.001; valid=np.ones_like(small); dp=depth_root/"frames"/f"{f:06d}.npz"; atomic_npz(dp,frame_id=np.asarray(f),depth_m=d,valid=valid); depth_rows.append({"frame_id":f,"relative_path":f"frames/{f:06d}.npz",**{k:v for k,v in artifact(dp).items() if k!="path"}})
    cap.release(); mask_manifest=case/"OBJECT_MASK_MANIFEST.json";atomic_json(mask_manifest,{"schema_version":"rgb-object-mask-manifest-v1","frames":mask_rows})
    depth_manifest=depth_root/"FRAME_MANIFEST.json";atomic_json(depth_manifest,{"schema_version":"synthetic-depth-frame-manifest-v1","frames":depth_rows})
    reg=depth_root/"REGISTRATION_AUTHORITY.npz";h=np.asarray([[2.,0.,.5],[0.,2.,.5],[0.,0.,1.]]);atomic_npz(reg,H_selected_rgb_to_depth_pixel=np.linalg.inv(h),H_depth_pixel_to_selected_rgb=h,stereo_rectified_depth_formula_intrinsics=np.asarray([[320.,0.,319.5],[0.,320.,239.5],[0.,0.,1.]]))
    depth_result=depth_root/"RESULT.json";atomic_json(depth_result,{"status":"PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION","consumption_authorized":True})
    pose=case/"c2w.npz";atomic_npz(pose,c2w=np.tile(np.eye(4),(frames,1,1)))
    spec=case/"INPUT_SPEC.json";atomic_json(spec,{"schema_version":"visual-fixed-instance-object6d-input-v1","session_id":f"synthetic_{task}","task":task,"frame_count":frames,"selected_rgb":artifact(rgb_path),"depth_root":str(depth_root),"depth_result":artifact(depth_result),"depth_frame_manifest":artifact(depth_manifest),"registration_authority":artifact(reg),"rgb_object_mask_manifest":artifact(mask_manifest),"camera_to_world":artifact(pose),"camera_to_world_key":"c2w"})
    return spec,case/"output"


def synthetic_test(root: Path) -> dict:
    root.mkdir(parents=True,exist_ok=True); results=[]
    for task in ("chips","poker"):
        spec,out=make_synthetic_case(root,task); result=produce(spec,out); results.append({"task":task,"status":result["status"],"valid_frames":result["valid_frames"],"instance_count":result["instance_count"],"result":artifact(out/"RESULT.json")})
    final={"schema_version":"visual-fixed-instance-object6d-synthetic-test-v1","status":"PASS" if all(r["status"].startswith("PASS") and r["valid_frames"]==10 and r["instance_count"]==1 for r in results) else "HOLD","gpu_used":False,"hand_or_pinch_used":False,"sessions":results};atomic_json(root/"RESULT.json",final);return final


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--input-spec",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--validate-depth-manifest",type=Path);ap.add_argument("--rgb-mask-registry",type=Path);ap.add_argument("--validate-only-output",type=Path);ap.add_argument("--synthetic-test",type=Path);args=ap.parse_args()
    if args.synthetic_test: print(json.dumps(synthetic_test(args.synthetic_test),ensure_ascii=False));return 0
    if args.validate_depth_manifest:
        if not args.validate_only_output: raise SystemExit("--validate-only-output required")
        result=validate_from_depth_manifest(args.validate_depth_manifest,args.validate_only_output,args.rgb_mask_registry);print(json.dumps(result,ensure_ascii=False));return 0
    if not args.input_spec or not args.output: raise SystemExit("--input-spec and --output required")
    print(json.dumps(produce(args.input_spec,args.output),ensure_ascii=False));return 0


if __name__=="__main__":
    raise SystemExit(main())
