#!/usr/bin/env python3
"""Run FoundationStereo on a frame from a cropped PICO stereo clip.

The PICO recording is a side-by-side pair of equiDis62 fisheye images. This
entry point performs fisheye removal and stereo rectification before inference.
The fixed stereo rotation is estimated visually and cached by calibration hash;
the metric baseline comes from camera_params.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_DIR.parents[1]
DEFAULT_CKPT = (
    PROJECT_DIR
    / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
)
DEFAULT_CACHE = REPO_DIR / "calibrations"
DEFAULT_RUNTIME_CACHE = Path(
    os.environ.get(
        "FOUNDATION_STEREO_CACHE",
        str(PROJECT_DIR / "_run/caches/foundationstereo-py311-v1"),
    )
)


@dataclass(frozen=True)
class Eye:
    intrinsics: np.ndarray
    distortion: np.ndarray


def one_file(root: Path, patterns: tuple[str, ...]) -> Path:
    matches = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one of {patterns} in {root}, found {len(matches)}"
        )
    return matches[0]


def discover_clip(clip_dir: Path) -> tuple[Path, Path]:
    camera_params = one_file(clip_dir, ("camera_params.json",))
    source_dir = clip_dir / "source_stereo"
    if source_dir.is_dir():
        video = one_file(source_dir, ("CameraRecord_*_stereo.mp4",))
    else:
        video = one_file(clip_dir, ("CameraRecord_*_stereo.mp4",))
    return video, camera_params


def load_camera_params(path: Path) -> tuple[list[Eye], int, int, float]:
    params = json.loads(path.read_text(encoding="utf-8"))
    reference = params.get("reference_resolution")
    if isinstance(reference, dict):
        eye_width, eye_height = int(reference["w"]), int(reference["h"])
    else:
        # Acquisition-aligned 0909/0910 releases publish the same per-eye
        # source-domain contract as top-level width/height.  Accept that
        # canonical schema without fabricating a legacy field.
        eye_width, eye_height = int(params["width"]), int(params["height"])
    if eye_width <= 0 or eye_height <= 0:
        raise ValueError(f"Invalid PICO per-eye resolution: {eye_width}x{eye_height}")
    eyes: list[Eye] = []
    for name in ("left", "right"):
        data = params[name]
        if data["distortion"]["model"] != "equiDis62":
            raise ValueError(
                f"Unsupported {name} distortion model: "
                f"{data['distortion']['model']}"
            )
        intr = data["intrinsics"]
        eyes.append(
            Eye(
                intrinsics=np.asarray(
                    [intr["fx"], intr["fy"], intr["cx"], intr["cy"]],
                    dtype=np.float64,
                ),
                distortion=np.asarray(
                    data["distortion"]["coeffs"], dtype=np.float64
                ),
            )
        )

    left_ext = np.asarray(params["extrinsics"]["left"], dtype=np.float64)
    right_ext = np.asarray(params["extrinsics"]["right"], dtype=np.float64)
    if left_ext.shape != (4, 4) or right_ext.shape != (4, 4):
        raise ValueError("PICO stereo extrinsics must both be 4x4 matrices")
    left_center = -left_ext[:3, :3].T @ left_ext[:3, 3]
    right_center = -right_ext[:3, :3].T @ right_ext[:3, 3]
    baseline = float(np.linalg.norm(right_center - left_center))
    if not 0.02 <= baseline <= 0.20:
        raise ValueError(f"Implausible PICO stereo baseline: {baseline:.6f} m")
    return eyes, eye_width, eye_height, baseline


def load_source_indices(path: Path) -> tuple[int, int]:
    """Return SBS half indices in physical ``(left, right)`` eye order.

    PICO recordings are not required to encode the physical left eye in the
    first SBS half.  The acquisition contract records the routing explicitly
    as ``camera_params.<eye>.sourceIndex``.  Treating array order as source
    order silently pairs each image with the other eye's calibration.
    """

    params = json.loads(path.read_text(encoding="utf-8"))
    indices: list[int] = []
    for name in ("left", "right"):
        data = params.get(name)
        if not isinstance(data, dict) or "sourceIndex" not in data:
            raise ValueError(f"camera_params.{name}.sourceIndex is required")
        index = data["sourceIndex"]
        if not isinstance(index, int):
            raise ValueError(f"camera_params.{name}.sourceIndex must be an integer")
        indices.append(index)
    if sorted(indices) != [0, 1]:
        raise ValueError(
            "physical-eye sourceIndex must be an exact permutation of [0, 1], "
            f"got {indices}"
        )
    return indices[0], indices[1]


def virtual_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.asarray(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pinhole_rays(width: int, height: int, intrinsics: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    return np.column_stack(
        (
            ((u - intrinsics[0, 2]) / intrinsics[0, 0]).reshape(-1),
            ((v - intrinsics[1, 2]) / intrinsics[1, 1]).reshape(-1),
            np.ones(width * height, dtype=np.float64),
        )
    )


def project_equidis62(rays: np.ndarray, eye: Eye) -> np.ndarray:
    x, y, z = np.asarray(rays, dtype=np.float64).T
    radius_xy = np.hypot(x, y)
    theta = np.arctan2(radius_xy, z)
    direction_x = np.divide(
        x, radius_xy, out=np.zeros_like(x), where=radius_xy > 1e-12
    )
    direction_y = np.divide(
        y, radius_xy, out=np.zeros_like(y), where=radius_xy > 1e-12
    )
    theta2 = theta * theta
    radial = np.ones_like(theta)
    theta_power = theta2.copy()
    for coefficient in eye.distortion[:6]:
        radial += coefficient * theta_power
        theta_power *= theta2
    xd = theta * radial * direction_x
    yd = theta * radial * direction_y
    radius2 = xd * xd + yd * yd
    p1, p2 = eye.distortion[6:]
    xt = xd + 2.0 * p1 * xd * yd + p2 * (radius2 + 2.0 * xd * xd)
    yt = yd + p1 * (radius2 + 2.0 * yd * yd) + 2.0 * p2 * xd * yd
    fx, fy, cx, cy = eye.intrinsics
    return np.column_stack((fx * xt + cx, fy * yt + cy))


def make_map(
    eye: Eye,
    original_to_rectified_rotation: np.ndarray,
    width: int,
    height: int,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # rectified_ray -> original_ray = R.T @ rectified_ray. In row-vector
    # notation this is rectified_ray @ R.
    rays = pinhole_rays(width, height, intrinsics)
    original_rays = rays @ original_to_rectified_rotation
    pixels = project_equidis62(original_rays, eye)
    return (
        pixels[:, 0].reshape(height, width).astype(np.float32),
        pixels[:, 1].reshape(height, width).astype(np.float32),
    )


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read stereo frame {frame_index}")
    return frame


def remap_pair(
    stereo: np.ndarray,
    eye_width: int,
    maps: list[tuple[np.ndarray, np.ndarray]],
    source_indices: tuple[int, int] = (0, 1),
) -> list[np.ndarray]:
    if stereo.shape[1] != eye_width * 2:
        raise ValueError(
            f"Expected side-by-side width {eye_width * 2}, got {stereo.shape[1]}"
        )
    if tuple(sorted(source_indices)) != (0, 1):
        raise ValueError(
            "source_indices must map physical (left, right) to an exact "
            f"permutation of SBS halves [0, 1], got {source_indices}"
        )
    result = []
    for source_index, (map_x, map_y) in zip(source_indices, maps, strict=True):
        raw = stereo[
            :, source_index * eye_width : (source_index + 1) * eye_width
        ]
        result.append(
            cv2.remap(
                raw,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
    return result


def feature_matches(
    left: np.ndarray, right: np.ndarray, max_features: int = 6000
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.SIFT_create(max_features)
    key_left, desc_left = detector.detectAndCompute(left, None)
    key_right, desc_right = detector.detectAndCompute(right, None)
    if desc_left is None or desc_right is None:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_left, desc_right, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    points_left = np.asarray(
        [key_left[item.queryIdx].pt for item in good], dtype=np.float64
    )
    points_right = np.asarray(
        [key_right[item.trainIdx].pt for item in good], dtype=np.float64
    )
    if len(points_left):
        plausible = (
            (np.abs(points_left[:, 1] - points_right[:, 1]) < 80.0)
            & (np.abs(points_left[:, 0] - points_right[:, 0]) < 400.0)
        )
        points_left = points_left[plausible]
        points_right = points_right[plausible]
    return points_left, points_right


def estimate_stereo_rotation(
    video_path: Path,
    eyes: list[Eye],
    eye_width: int,
    eye_height: int,
    width: int,
    height: int,
    intrinsics: np.ndarray,
    sample_count: int,
    source_indices: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_maps = [
        make_map(eye, np.eye(3), width, height, intrinsics) for eye in eyes
    ]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.unique(
        np.linspace(
            max(0, int(frame_count * 0.05)),
            max(0, int(frame_count * 0.75)),
            max(1, sample_count),
            dtype=np.int64,
        )
    )
    all_left: list[np.ndarray] = []
    all_right: list[np.ndarray] = []
    per_frame: dict[str, int] = {}
    try:
        for index in indices:
            stereo = read_frame(capture, int(index))
            if stereo.shape[:2] != (eye_height, eye_width * 2):
                raise ValueError(
                    f"Video is {stereo.shape[1]}x{stereo.shape[0]}, expected "
                    f"{eye_width * 2}x{eye_height}"
                )
            left, right = remap_pair(
                stereo, eye_width, base_maps, source_indices=source_indices
            )
            points_left, points_right = feature_matches(left, right)
            per_frame[str(int(index))] = int(len(points_left))
            if len(points_left) >= 20:
                all_left.append(points_left)
                all_right.append(points_right)
    finally:
        capture.release()
    if not all_left:
        raise RuntimeError("Could not find enough stereo features for calibration")
    points_left = np.concatenate(all_left)
    points_right = np.concatenate(all_right)
    essential, mask = cv2.findEssentialMat(
        points_left,
        points_right,
        intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None:
        raise RuntimeError("Essential-matrix estimation failed")
    if essential.shape[0] > 3:
        essential = essential[:3]
    inliers, rotation, translation, _ = cv2.recoverPose(
        essential,
        points_left,
        points_right,
        intrinsics,
        mask=mask,
    )
    if inliers < 30:
        raise RuntimeError(f"Only {inliers} stereo calibration inliers")
    translation = translation.reshape(3)
    diagnostics = {
        "sample_frames": [int(index) for index in indices],
        "matches_by_frame": per_frame,
        "match_count": int(len(points_left)),
        "inlier_count": int(inliers),
        "inlier_ratio": float(inliers / len(points_left)),
    }
    return rotation, translation / np.linalg.norm(translation), diagnostics


def cache_path(
    camera_params: Path, width: int, height: int, horizontal_fov_deg: float
) -> Path:
    digest = hashlib.sha256(camera_params.read_bytes()).hexdigest()[:16]
    fov_tag = f"{horizontal_fov_deg:.3f}".replace(".", "p")
    return DEFAULT_CACHE / f"{digest}_{width}x{height}_fov{fov_tag}.json"


def validate_cached_calibration_identity(
    calibration: dict,
    camera_params: Path,
    source_indices: tuple[int, int],
) -> None:
    """Reject legacy/cross-session rectification before any remap or inference."""

    source = calibration.get("source_camera_params")
    if not isinstance(source, str) or Path(source).expanduser().resolve() != camera_params.resolve():
        raise ValueError(
            "Cached rectification is not bound to this same-session camera_params; "
            "cross-session calibration is forbidden"
        )
    expected = {"left": int(source_indices[0]), "right": int(source_indices[1])}
    if calibration.get("physical_eye_source_indices") != expected:
        raise ValueError(
            "Cached rectification predates or conflicts with physical-eye sourceIndex "
            "routing; rerun with --recalibrate"
        )


def build_rectification(
    args: argparse.Namespace,
    video: Path,
    camera_params: Path,
    eyes: list[Eye],
    eye_width: int,
    eye_height: int,
    baseline: float,
    source_indices: tuple[int, int] = (0, 1),
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict, Path]:
    initial_k = virtual_intrinsics(args.width, args.height, args.fov)
    calibration_file = (
        Path(args.calibration).expanduser().resolve()
        if args.calibration
        else cache_path(camera_params, args.width, args.height, args.fov)
    )
    if calibration_file.exists() and not args.recalibrate:
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        validate_cached_calibration_identity(
            calibration, camera_params, source_indices
        )
        rotation = np.asarray(calibration["right_from_left_rotation"])
        translation_unit = np.asarray(calibration["right_from_left_translation_unit"])
        diagnostics = calibration.get("estimation", {})
        logging.info("Using cached stereo rectification: %s", calibration_file)
    else:
        logging.info(
            "Estimating fixed stereo rotation from %d video frames",
            args.calibration_frames,
        )
        rotation, translation_unit, diagnostics = estimate_stereo_rotation(
            video,
            eyes,
            eye_width,
            eye_height,
            args.width,
            args.height,
            initial_k,
            args.calibration_frames,
            source_indices,
        )
    rotation_left, rotation_right, projection_left, projection_right, q, _, _ = (
        cv2.stereoRectify(
            initial_k,
            None,
            initial_k,
            None,
            (args.width, args.height),
            rotation,
            (translation_unit * baseline).reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
            newImageSize=(args.width, args.height),
        )
    )
    rectified_k = projection_left[:3, :3]
    calibration = {
        "version": "1.0",
        "source_camera_params": str(camera_params),
        "physical_eye_source_indices": {
            "left": int(source_indices[0]),
            "right": int(source_indices[1]),
        },
        "image_size": [args.width, args.height],
        "horizontal_fov_deg_before_stereo_rectify": args.fov,
        "baseline_m": baseline,
        "right_from_left_rotation": rotation.tolist(),
        "right_from_left_translation_unit": translation_unit.tolist(),
        "rectification_rotation_left": rotation_left.tolist(),
        "rectification_rotation_right": rotation_right.tolist(),
        "rectified_intrinsics": rectified_k.tolist(),
        "projection_left": projection_left.tolist(),
        "projection_right": projection_right.tolist(),
        "Q": q.tolist(),
        "estimation": diagnostics,
    }
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    maps = [
        make_map(eyes[0], rotation_left, args.width, args.height, rectified_k),
        make_map(eyes[1], rotation_right, args.width, args.height, rectified_k),
    ]
    return maps, rectified_k, calibration, calibration_file


def draw_epipolar_check(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    canvas = np.concatenate((left, right), axis=1)
    for y in range(40, canvas.shape[0], 80):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 255), 1)
    return canvas


def rectification_error(left: np.ndarray, right: np.ndarray) -> dict:
    points_left, points_right = feature_matches(left, right)
    if not len(points_left):
        return {"matches": 0, "median_vertical_error_px": None}
    vertical = np.abs(points_left[:, 1] - points_right[:, 1])
    return {
        "matches": int(len(vertical)),
        "median_vertical_error_px": float(np.median(vertical)),
        "p90_vertical_error_px": float(np.quantile(vertical, 0.90)),
    }


def colorize(values: np.ndarray, valid: np.ndarray, inverse: bool = False) -> np.ndarray:
    finite_values = values[valid]
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not finite_values.size:
        return output
    low, high = np.quantile(finite_values, [0.02, 0.98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if inverse:
        normalized = 1.0 - normalized
    colored = cv2.applyColorMap(
        np.nan_to_num(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    output[valid] = colored[valid]
    return output


def run_model(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    intrinsics: np.ndarray,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep timm/DINO source and backbone caches on persistent NAS instead of
    # the disposable container home directory.
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_RUNTIME_CACHE / "torch"))
    os.environ.setdefault(
        "HF_HOME", str(DEFAULT_RUNTIME_CACHE / "huggingface")
    )
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(REPO_DIR))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}\n"
            "Run ./setup_foundation_stereo.sh first."
        )
    cfg = OmegaConf.load(checkpoint.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = args.valid_iters
    model = FoundationStereo(cfg)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.cuda().eval()

    scale = args.scale
    if not 0.0 < scale <= 1.0:
        raise ValueError("--scale must be in (0, 1]")
    left = cv2.resize(left_bgr, None, fx=scale, fy=scale)
    right = cv2.resize(right_bgr, None, fx=scale, fy=scale)
    scaled_k = intrinsics.copy()
    scaled_k[:2] *= scale
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    tensor_left = (
        torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    )
    tensor_right = (
        torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    )
    padder = InputPadder(tensor_left.shape, divis_by=32, force_square=False)
    tensor_left, tensor_right = padder.pad(tensor_left, tensor_right)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        disparity = model.forward(
            tensor_left,
            tensor_right,
            iters=args.valid_iters,
            test_mode=True,
        )
    disparity = (
        padder.unpad(disparity.float()).cpu().numpy().reshape(left.shape[:2])
    )
    x_coordinates = np.arange(disparity.shape[1])[None, :]
    valid = (
        np.isfinite(disparity)
        & (disparity > args.min_disparity)
        & ((x_coordinates - disparity) >= 0)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = scaled_k[0, 0] * baseline / disparity[valid]
    valid &= (depth >= args.z_near) & (depth <= args.z_far)
    depth[~valid] = np.nan
    return disparity.astype(np.float32), depth, valid, scaled_k


def write_cloud(
    path: Path,
    depth: np.ndarray,
    color_bgr: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    stride: int,
) -> int:
    y, x = np.indices(depth.shape)
    select = valid & ((x % stride) == 0) & ((y % stride) == 0)
    z = depth[select]
    xyz = np.column_stack(
        (
            (x[select] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y[select] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)[select]
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
    return int(len(xyz))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rectify a PICO stereo frame and run FoundationStereo."
    )
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0, help="Zero-based video frame")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-frames", type=int, default=8)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid-iters", type=int, default=16)
    parser.add_argument("--min-disparity", type=float, default=0.25)
    parser.add_argument("--z-near", type=float, default=0.10)
    parser.add_argument("--z-far", type=float, default=5.0)
    parser.add_argument("--cloud-stride", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    clip_dir = args.clip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video, camera_params = discover_clip(clip_dir)
    eyes, eye_width, eye_height, baseline = load_camera_params(camera_params)
    source_indices = load_source_indices(camera_params)
    maps, rectified_k, _calibration, calibration_file = build_rectification(
        args,
        video,
        camera_params,
        eyes,
        eye_width,
        eye_height,
        baseline,
        source_indices,
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not 0 <= args.frame < frame_count:
        capture.release()
        raise ValueError(f"--frame must be in [0, {frame_count - 1}]")
    stereo = read_frame(capture, args.frame)
    capture.release()
    left, right = remap_pair(
        stereo, eye_width, maps, source_indices=source_indices
    )
    cv2.imwrite(str(output_dir / "left_rectified.png"), left)
    cv2.imwrite(str(output_dir / "right_rectified.png"), right)
    cv2.imwrite(
        str(output_dir / "rectification_check.png"),
        draw_epipolar_check(left, right),
    )
    quality = rectification_error(left, right)
    logging.info("Rectification quality: %s", quality)

    manifest = {
        "clip_dir": str(clip_dir),
        "source_video": str(video),
        "source_frame": args.frame,
        "frame_count": frame_count,
        "calibration_file": str(calibration_file),
        "baseline_m": baseline,
        "physical_eye_source_indices": {
            "left": int(source_indices[0]),
            "right": int(source_indices[1]),
        },
        "rectified_intrinsics": rectified_k.tolist(),
        "rectification_quality": quality,
        "model_ran": not args.prepare_only,
    }
    if not args.prepare_only:
        disparity, depth, valid, scaled_k = run_model(
            left, right, rectified_k, baseline, args
        )
        np.save(output_dir / "disparity_px.npy", disparity)
        np.save(output_dir / "depth_meter.npy", depth)
        cv2.imwrite(
            str(output_dir / "disparity_vis.png"), colorize(disparity, valid)
        )
        cv2.imwrite(
            str(output_dir / "depth_vis.png"),
            colorize(depth, valid, inverse=True),
        )
        scaled_left = cv2.resize(left, depth.shape[::-1])
        cloud_count = write_cloud(
            output_dir / "cloud.ply",
            depth,
            scaled_left,
            valid,
            scaled_k,
            args.cloud_stride,
        )
        manifest["inference"] = {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "scale": args.scale,
            "valid_iters": args.valid_iters,
            "depth_range_m": [args.z_near, args.z_far],
            "valid_depth_pixels": int(np.count_nonzero(valid)),
            "point_cloud_points": cloud_count,
            "scaled_intrinsics": scaled_k.tolist(),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Done: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
