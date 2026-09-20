#!/usr/bin/env python3
"""CPU-only three-part stereo preflight for an Exact78 encoded SBS video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from chaoyang.pipeline.exact78_stereo_preflight_v31 import (
    ImageDomainEvidenceV31,
    check_image_domain,
    check_metric_conversion,
    check_stereo_geometry,
    combine_preflight,
    mirrored_intrinsics,
)
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    frame_metrics,
    resize_only,
    robust_correspondences,
    split_source_index_eyes,
)


INPUT_SCHEMA = "EXACT78_STEREO_PREFLIGHT_INPUT_V31"


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != INPUT_SCHEMA:
        raise RuntimeError(f"{INPUT_SCHEMA} JSON object required")
    frames = value.get("frame_indices")
    if (
        not isinstance(frames, list)
        or len(frames) != 12
        or any(type(item) is not int or item < 0 for item in frames)
        or len(frames) != len(set(frames))
    ):
        raise RuntimeError("exactly 12 unique non-negative frame indices required")
    if not isinstance(value.get("image_domain"), dict) or not isinstance(
        value.get("metric_conversion"), dict
    ):
        raise RuntimeError("image_domain and metric_conversion objects required")
    return value


def decode_correspondences(
    video: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    domain = config["image_domain"]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open SBS video: {video}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows: list[dict[str, Any]] = []
    vertical: list[np.ndarray] = []
    disparity: list[np.ndarray] = []
    rgb_roundtrip_mismatch = 0
    try:
        for frame_id in config["frame_indices"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"SBS decode failed at frame {frame_id}")
            left, right = split_source_index_eyes(
                sbs,
                eye_width=int(domain["eye_width"]),
                eye_height=int(domain["eye_height"]),
                physical_left_source_index=int(domain["physical_left_source_index"]),
                physical_right_source_index=int(domain["physical_right_source_index"]),
            )
            left = resize_only(
                left, width=int(domain["model_width"]), height=int(domain["model_height"])
            )
            right = resize_only(
                right, width=int(domain["model_width"]), height=int(domain["model_height"])
            )
            reflected = bool(domain["horizontal_reflection_for_disparity_sign"])
            model_left = cv2.flip(left, 1) if reflected else left
            model_right = cv2.flip(right, 1) if reflected else right
            restored_left = cv2.flip(model_left, 1) if reflected else model_left
            restored_right = cv2.flip(model_right, 1) if reflected else model_right
            rgb_roundtrip_mismatch += int(
                np.any(restored_left != left, axis=2).sum()
                + np.any(restored_right != right, axis=2).sum()
            )
            points_left, points_right = robust_correspondences(left, right)
            row = frame_metrics(
                points_left,
                points_right,
                width=int(domain["model_width"]),
                height=int(domain["model_height"]),
            )
            row["frame_id"] = frame_id
            rows.append(row)
            vertical.append(np.abs(points_left[:, 1] - points_right[:, 1]))
            disparity.append(points_left[:, 0] - points_right[:, 0])
    finally:
        capture.release()
    return rows, vertical, disparity, {
        "decoded_sbs_width": source_width,
        "decoded_sbs_height": source_height,
        "mismatched_rgb_roundtrip_pixels": rgb_roundtrip_mismatch,
    }


def run(video: Path, config_path: Path, output: Path) -> dict[str, Any]:
    video = video.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh/no-clobber output required: {output}")
    config = load_config(config_path)
    rows, vertical, disparity, decoded = decode_correspondences(video, config)
    domain = config["image_domain"]
    image = check_image_domain(
        ImageDomainEvidenceV31(
            sbs_width=decoded["decoded_sbs_width"],
            sbs_height=decoded["decoded_sbs_height"],
            eye_width=int(domain["eye_width"]),
            eye_height=int(domain["eye_height"]),
            model_width=int(domain["model_width"]),
            model_height=int(domain["model_height"]),
            physical_left_source_index=int(domain["physical_left_source_index"]),
            physical_right_source_index=int(domain["physical_right_source_index"]),
            decoded_vst_already_undistorted=bool(
                domain["decoded_vst_already_undistorted"]
            ),
            lens_undistortion_applied=bool(domain["lens_undistortion_applied"]),
            horizontal_reflection_for_disparity_sign=bool(
                domain["horizontal_reflection_for_disparity_sign"]
            ),
            outputs_unflipped_to_physical_left=bool(
                domain["outputs_unflipped_to_physical_left"]
            ),
            maximum_coordinate_roundtrip_error_px=0.0,
            mismatched_rgb_roundtrip_pixels=int(
                decoded["mismatched_rgb_roundtrip_pixels"]
            ),
        )
    )
    geometry = check_stereo_geometry(
        rows,
        vertical,
        disparity,
        reflected_for_model=bool(domain["horizontal_reflection_for_disparity_sign"]),
    )
    metric_config = config["metric_conversion"]
    physical_left_k = np.asarray(metric_config["physical_left_k"], np.float64)
    physical_right_k = np.asarray(metric_config["physical_right_k"], np.float64)
    reflected = bool(domain["horizontal_reflection_for_disparity_sign"])
    metric = check_metric_conversion(
        physical_left_k=physical_left_k,
        physical_right_k=physical_right_k,
        model_left_k=(
            mirrored_intrinsics(physical_left_k, int(domain["model_width"]))
            if reflected
            else physical_left_k
        ),
        model_right_k=(
            mirrored_intrinsics(physical_right_k, int(domain["model_width"]))
            if reflected
            else physical_right_k
        ),
        projection_left=np.asarray(metric_config["projection_left"], np.float64),
        projection_right=np.asarray(metric_config["projection_right"], np.float64),
        baseline_m=float(metric_config["baseline_m"]),
        width=int(domain["model_width"]),
        height=int(domain["model_height"]),
        reflected_for_model=reflected,
        outputs_unflipped_to_physical_left=bool(
            domain["outputs_unflipped_to_physical_left"]
        ),
        calibration_pixel_domain=str(metric_config["calibration_pixel_domain"]),
        expected_pixel_domain=str(metric_config["expected_pixel_domain"]),
    )
    result = combine_preflight(image, geometry, metric)
    result["inputs"] = {"sbs_video": artifact(video), "config": artifact(config_path)}
    result["frame_rows"] = rows
    result["claim_limit"] = (
        "CPU correspondence and conversion admission only; no FoundationStereo "
        "inference and no external metric accuracy claim."
    )
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbs-video", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.sbs_video, args.config, args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["local_stereo_metric_dev"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
