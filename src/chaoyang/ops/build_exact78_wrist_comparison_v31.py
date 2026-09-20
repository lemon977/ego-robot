#!/usr/bin/env python3
"""Build the Exact78 V3.1 dual-wrist numeric product and review video.

The input NPZ is an explicit adapter boundary; this command never reads an old
Depth cache or derives a surface point from an arbitrary hand-mask pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import jsonschema
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
WRIST_RESULT_SCHEMA = PROJECT / "contracts/wrist_dual_representation_v1.schema.json"


COLORS = {
    "tracker": (30, 220, 80),
    "hawor": (30, 150, 255),
    "surface": (220, 60, 220),
    "fused": (255, 230, 40),
    "headset": (235, 235, 235),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_dual_wrist_npz(path: Path) -> dict[str, np.ndarray]:
    required = {
        "frame_id",
        "timestamp_s",
        "T_camera_controller_raw",
        "selected_T_controller_wrist",
        "static_T_camera_wrist",
        "observed_T_camera_wrist",
        "observed_valid",
        "visible_wrist_surface_point_camera",
        "visible_wrist_surface_source_pixel_uv",
        "visible_wrist_surface_valid",
        "visible_wrist_region_registration_only",
        "fused_T_camera_wrist",
        "fusion_valid",
        "correction_clipped",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"V3.1 wrist adapter input missing fields: {missing}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_dual_wrist_result(result_path: Path, npz_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    schema = json.loads(WRIST_RESULT_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result)
    reference = result["outputs"]["npz"]
    if (
        Path(reference["path"]).resolve(strict=True) != npz_path
        or int(reference["bytes"]) != npz_path.stat().st_size
        or str(reference["sha256"]) != sha256(npz_path)
    ):
        raise RuntimeError("dual-wrist RESULT does not bind the consumed NPZ")
    if any(result["authority"].values()):
        raise RuntimeError("dual-wrist authority exceeds development-only boundary")
    return result


def _distribution(values: np.ndarray, *, millimetres: bool = True) -> dict[str, Any]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if millimetres:
        finite = finite * 1000.0
    if not len(finite):
        return {"count": 0, "p50": None, "p95": None, "mean": None, "max": None}
    return {
        "count": int(len(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def comparison_metrics(product: dict[str, Any]) -> dict[str, Any]:
    tracker = np.asarray(product["static_T_camera_wrist"], np.float64)[..., :3, 3]
    hawor = np.asarray(product["observed_T_camera_wrist"], np.float64)[..., :3, 3]
    fused = np.asarray(product["fused_T_camera_wrist"], np.float64)[..., :3, 3]
    surface = np.asarray(product["visible_wrist_surface_point_camera"], np.float64)
    controller_valid = np.isfinite(tracker).all(-1)
    hawor_valid = np.asarray(product["observed_valid"], bool)
    fused_valid = np.asarray(product["fusion_valid"], bool)
    surface_valid = np.asarray(product["visible_wrist_surface_valid"], bool)
    rows: dict[str, Any] = {}
    for side, name in enumerate(("left", "right")):
        center_mask = controller_valid[:, side] & hawor_valid[:, side]
        fused_mask = controller_valid[:, side] & fused_valid[:, side]
        surface_tracker = surface_valid[:, side] & controller_valid[:, side]
        surface_hawor = surface_valid[:, side] & hawor_valid[:, side]
        rows[name] = {
            "tracker_vs_hawor_anatomical_center_distance_mm": _distribution(
                np.linalg.norm(tracker[:, side] - hawor[:, side], axis=-1)[center_mask]
            ),
            "tracker_vs_fused_anatomical_center_distance_mm": _distribution(
                np.linalg.norm(tracker[:, side] - fused[:, side], axis=-1)[fused_mask]
            ),
            "surface_to_tracker_mixed_semantic_distance_mm": _distribution(
                np.linalg.norm(surface[:, side] - tracker[:, side], axis=-1)[surface_tracker]
            ),
            "surface_to_hawor_mixed_semantic_distance_mm": _distribution(
                np.linalg.norm(surface[:, side] - hawor[:, side], axis=-1)[surface_hawor]
            ),
            "surface_observed_frames": int(surface_valid[:, side].sum()),
            "surface_region_only_frames": int(
                (
                    np.asarray(
                        product["visible_wrist_region_registration_only"], bool
                    )[:, side]
                    & ~surface_valid[:, side]
                ).sum()
            ),
            "fusion_bound_touched_frames": int(
                np.asarray(product["correction_clipped"], bool)[:, side].sum()
            ),
        }
    return {
        "schema_version": "EXACT78_WRIST_MULTI_SOURCE_METRICS_V31",
        "sides": rows,
        "interpretation": {
            "center_to_center": "internal agreement between stated anatomical wrist definitions",
            "surface_to_center": "mixed-semantic diagnostic, never anatomical wrist error",
            "fusion": "agreement with an input source is not independent accuracy evidence",
        },
    }


def project(point: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(point).all() or point[2] <= 1e-8:
        return None
    homogeneous = intrinsic @ point
    return int(round(homogeneous[0] / homogeneous[2])), int(round(homogeneous[1] / homogeneous[2]))


def transform_camera_points_to_world(
    T_world_camera: np.ndarray, points_camera: np.ndarray
) -> np.ndarray:
    transforms = np.asarray(T_world_camera, np.float64)
    points = np.asarray(points_camera, np.float64)
    if transforms.shape != (len(points), 4, 4) or points.shape[1:] != (2, 3):
        raise RuntimeError("T_world_camera/point shapes must be [T,4,4] and [T,2,3]")
    if (
        not np.isfinite(transforms).all()
        or not np.allclose(transforms[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise RuntimeError("T_world_camera is not a finite homogeneous transform")
    rotation = transforms[:, :3, :3]
    if not np.allclose(
        np.swapaxes(rotation, -1, -2) @ rotation,
        np.eye(3),
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError("T_world_camera rotation is not orthonormal")
    return np.einsum("tij,tsj->tsi", rotation, points) + transforms[:, None, :3, 3]


def _plot_orthographic(
    image: np.ndarray,
    rectangle: tuple[int, int, int, int],
    trajectories: list[tuple[np.ndarray, tuple[int, int, int], str]],
    frame: int,
    *,
    axes: tuple[int, int],
    title: str,
) -> None:
    x0, y0, x1, y1 = rectangle
    cv2.rectangle(image, (x0, y0), (x1, y1), (80, 80, 80), 1)
    valid_points = []
    for values, _, _ in trajectories:
        points = values[..., list(axes)].reshape(-1, 2)
        valid_points.append(points[np.isfinite(points).all(1)])
    merged = np.concatenate([value for value in valid_points if len(value)], axis=0)
    if not len(merged):
        cv2.putText(image, f"{title}: BLOCKED", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 120, 255), 2)
        return
    low = np.quantile(merged, 0.01, axis=0)
    high = np.quantile(merged, 0.99, axis=0)
    span = np.maximum(high - low, 0.05)
    low -= 0.1 * span
    high += 0.1 * span
    cv2.putText(image, title, (x0 + 12, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (230, 230, 230), 2)
    for values, color, label in trajectories:
        points = values[max(0, frame - 89) : frame + 1, list(axes)]
        pixels = []
        for point in points:
            if not np.isfinite(point).all():
                if len(pixels) > 1:
                    cv2.polylines(image, [np.asarray(pixels, np.int32)], False, color, 2, cv2.LINE_AA)
                pixels = []
                continue
            u = x0 + 15 + int((point[0] - low[0]) / (high[0] - low[0]) * (x1 - x0 - 30))
            v = y1 - 15 - int((point[1] - low[1]) / (high[1] - low[1]) * (y1 - y0 - 50))
            pixels.append((u, v))
        if len(pixels) > 1:
            cv2.polylines(image, [np.asarray(pixels, np.int32)], False, color, 2, cv2.LINE_AA)
        if pixels:
            cv2.circle(image, pixels[-1], 5, color, -1, cv2.LINE_AA)
    legend_x = x0 + 12
    for _, color, label in trajectories:
        cv2.putText(image, label, (legend_x, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        legend_x += 112


def render_review(
    *,
    raw_video: Path,
    video_output: Path,
    product: dict[str, Any],
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray | None,
    fps: float,
) -> None:
    tracker = np.asarray(product["static_T_camera_wrist"], np.float64)[..., :3, 3]
    hawor = np.asarray(product["observed_T_camera_wrist"], np.float64)[..., :3, 3]
    surface = np.asarray(product["visible_wrist_surface_point_camera"], np.float64)
    fused = np.asarray(product["fused_T_camera_wrist"], np.float64)[..., :3, 3]
    surface_uv = np.asarray(product["visible_wrist_surface_source_pixel_uv"], np.float64)
    count = len(tracker)
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics[None], (count, 3, 3))
    if intrinsics.shape != (count, 3, 3):
        raise RuntimeError("intrinsics must be [3,3] or [T,3,3]")
    world = None
    headset_world = None
    if T_world_camera is not None:
        transforms = np.asarray(T_world_camera, np.float64)
        if transforms.shape != (count, 4, 4):
            raise RuntimeError("T_world_camera must be [T,4,4]")
        world = {
            "tracker": transform_camera_points_to_world(transforms, tracker),
            "hawor": transform_camera_points_to_world(transforms, hawor),
            "surface": transform_camera_points_to_world(transforms, surface),
            "fused": transform_camera_points_to_world(transforms, fused),
        }
        headset_world = transforms[:, :3, 3]

    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open raw video: {raw_video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (1920, 1080)
    temporary = Path(tempfile.mkstemp(prefix="exact78_wrist_v31_", suffix=".mp4", dir="/tmp")[1])
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    try:
        for frame in range(count):
            ok, raw = capture.read()
            if not ok:
                raise RuntimeError(f"raw video ended at frame {frame}, expected {count}")
            canvas = np.full((1080, 1920, 3), 24, np.uint8)
            rgb = cv2.resize(raw, (960, 720), interpolation=cv2.INTER_AREA)
            scale = np.asarray([[960 / source_width, 0, 0], [0, 720 / source_height, 0], [0, 0, 1]], np.float64)
            k = scale @ intrinsics[frame]
            for side in range(2):
                for key, values, radius in (
                    ("tracker", tracker, 8),
                    ("hawor", hawor, 8),
                    ("fused", fused, 11),
                ):
                    uv = project(values[frame, side], k)
                    if uv is not None:
                        cv2.circle(rgb, uv, radius, COLORS[key], -1 if key == "fused" else 3, cv2.LINE_AA)
                if np.isfinite(surface_uv[frame, side]).all():
                    uv = tuple(np.rint(surface_uv[frame, side] * (960 / source_width, 720 / source_height)).astype(int))
                    cv2.drawMarker(rgb, uv, COLORS["surface"], cv2.MARKER_DIAMOND, 18, 3, cv2.LINE_AA)
            canvas[70:790, 20:980] = rgb
            cv2.putText(canvas, "2D physical selected-camera domain", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, .72, (235, 235, 235), 2)
            cv2.putText(canvas, f"frame {frame:04d}/{count-1:04d}", (1550, 42), cv2.FONT_HERSHEY_SIMPLEX, .72, (235, 235, 235), 2)

            camera_trajectories = []
            for side, suffix in enumerate(("L", "R")):
                for key, values in (("tracker", tracker), ("hawor", hawor), ("surface", surface), ("fused", fused)):
                    camera_trajectories.append((values[:, side], COLORS[key], f"{key}-{suffix}"))
            _plot_orthographic(
                canvas, (1000, 70, 1900, 500), camera_trajectories, frame,
                axes=(0, 2), title="Head-locked camera X-Z (m)",
            )
            world_trajectories = []
            if world is not None:
                for side, suffix in enumerate(("L", "R")):
                    for key in ("tracker", "hawor", "surface", "fused"):
                        world_trajectories.append((world[key][:, side], COLORS[key], f"{key}-{suffix}"))
                world_trajectories.append((headset_world, COLORS["headset"], "headset"))
            _plot_orthographic(
                canvas, (1000, 525, 1900, 955), world_trajectories, frame,
                axes=(0, 2), title="PICO tracking-world fixed X-Z (m)",
            )
            cv2.putText(
                canvas,
                "green=static tracker  orange=HaWoR anatomical  magenta=visible surface  yellow=fused",
                (30, 825), cv2.FONT_HERSHEY_SIMPLEX, .60, (225, 225, 225), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "surface-vs-center is mixed semantics; fusion is not independent accuracy proof",
                (30, 862), cv2.FONT_HERSHEY_SIMPLEX, .60, (90, 190, 255), 2, cv2.LINE_AA,
            )
            for side, suffix in enumerate(("L", "R")):
                y = 915 + side * 48
                values = []
                for key, array in (("T", tracker), ("H", hawor), ("S", surface), ("F", fused)):
                    z = array[frame, side, 2]
                    values.append(f"{key}={z*1000:.1f}mm" if np.isfinite(z) else f"{key}=NA")
                cv2.putText(canvas, f"{suffix} optical-Z  " + "  ".join(values), (30, y), cv2.FONT_HERSHEY_SIMPLEX, .62, (230, 230, 230), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        capture.release()
    video_output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, video_output)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(video_output), "-f", "null", "-"],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-wrist-result", type=Path, required=True)
    parser.add_argument("--dual-wrist-npz", type=Path, required=True)
    parser.add_argument(
        "--camera-adapter-npz",
        type=Path,
        required=True,
        help="frame_id, timestamp_s, intrinsics and optional T_world_camera",
    )
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    dual_result_path = args.dual_wrist_result.resolve(strict=True)
    dual_npz_path = args.dual_wrist_npz.resolve(strict=True)
    camera_path = args.camera_adapter_npz.resolve(strict=True)
    raw_video = args.raw_video.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh/no-clobber output required: {output}")
    output.mkdir(parents=True)
    dual_result = validate_dual_wrist_result(dual_result_path, dual_npz_path)
    product = load_dual_wrist_npz(dual_npz_path)
    with np.load(camera_path, allow_pickle=False) as archive:
        camera = {name: np.asarray(archive[name]) for name in archive.files}
    required_camera = {"frame_id", "timestamp_s", "intrinsics"}
    missing = sorted(required_camera - set(camera))
    if missing:
        raise RuntimeError(f"camera adapter missing fields: {missing}")
    if not np.array_equal(camera["frame_id"], product["frame_id"]):
        raise RuntimeError("camera adapter frame IDs differ from dual-wrist product")
    if not np.allclose(
        camera["timestamp_s"], product["timestamp_s"], atol=1e-9, rtol=0.0
    ):
        raise RuntimeError("camera adapter timestamps differ from dual-wrist product")
    metrics = comparison_metrics(product)
    metrics["dual_wrist_authority"] = dual_result["authority"]
    metrics["temporal_authority"] = dual_result["temporal_authority"]
    metrics["pico_world_coordinate_chain"] = (
        "AVAILABLE" if "T_world_camera" in camera else "BLOCKED_COORDINATE_CHAIN"
    )
    metrics_path = output / "METRICS.json"
    atomic_json(metrics_path, metrics)
    video = output / "CHIPS023_WRIST_2D_3D_DEPTH_REVIEW.mp4"
    render_review(
        raw_video=raw_video,
        video_output=video,
        product=product,
        intrinsics=camera["intrinsics"],
        T_world_camera=camera.get("T_world_camera"),
        fps=args.fps,
    )
    result = {
        "schema_version": "EXACT78_WRIST_COMPARISON_RESULT_V31",
        "status": (
            "PASS_DEVELOPMENT_MULTI_SOURCE_COMPARISON"
            if "T_world_camera" in camera
            else "PARTIAL_BLOCKED_PICO_WORLD_CHAIN"
        ),
        "inputs": {
            "dual_wrist_result": artifact(dual_result_path),
            "dual_wrist_npz": artifact(dual_npz_path),
            "camera_adapter_npz": artifact(camera_path),
            "raw_video": artifact(raw_video),
        },
        "outputs": {
            "metrics": artifact(metrics_path),
            "review_video": artifact(video),
        },
        "dual_wrist_schema_authority": "contracts/wrist_dual_representation_v1.schema.json",
        "dual_wrist_producer_authority": "build_wiyh_wrist_dual_representation_v1",
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Multi-source 3D consistency comparison; no input or fused result is "
            "independent external wrist truth."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
