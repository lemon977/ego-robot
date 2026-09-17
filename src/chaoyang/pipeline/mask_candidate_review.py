"""Build read-only MASK candidate integrity metrics and visual review media."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask(ref: dict[str, Any]) -> np.ndarray:
    path = Path(ref["path"])
    if _sha256(path) != ref["sha256"]:
        raise RuntimeError(f"MASK ref SHA mismatch: {path}")
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise RuntimeError(f"cannot decode MASK ref: {path}")
    return value > 0


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    alpha = 0.48
    color_image = np.empty_like(result)
    color_image[:] = color
    result[mask] = cv2.addWeighted(result[mask], 1.0 - alpha, color_image[mask], alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(result, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _diagnostic(image: np.ndarray, human: np.ndarray, obj: np.ndarray, contact: np.ndarray) -> np.ndarray:
    result = image.copy()
    layer = np.zeros_like(image)
    layer[human] = (0, 0, 255)
    layer[obj] = (0, 255, 0)
    layer[contact] = (255, 0, 0)
    active = human | obj | contact
    result[active] = cv2.addWeighted(result[active], 0.52, layer[active], 0.48, 0)
    return result


def _resize_panel(image: np.ndarray, width: int) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def build_review(
    source_manifest_path: Path,
    candidate_manifest_path: Path,
    baseline_manifest_path: Path,
    schema_path: Path,
    output_dir: Path,
    critical_frames: list[int],
) -> dict[str, Any]:
    source_manifest = _load_json(source_manifest_path)
    candidate = _load_json(candidate_manifest_path)
    baseline = _load_json(baseline_manifest_path)
    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(candidate)

    source_session = next(
        item for item in source_manifest["sessions"] if item["session_id"] == candidate["session_id"]
    )
    source_frames = source_session["frames"]
    if not (len(source_frames) == len(candidate["frames"]) == len(baseline["frames"])):
        raise RuntimeError("source/candidate/baseline frame counts disagree")

    output_dir.mkdir(parents=True, exist_ok=True)
    human_area: list[float] = []
    baseline_area: list[float] = []
    ious: list[float] = []
    extra_ratios: list[float] = []
    overlap_pixels = 0
    prompt_ref_sha_pass = 0
    independent_holds = {"left": 0, "right": 0}
    candidate_holds = {
        side: {part: 0 for part in ("hand_candidate", "forearm_candidate", "wrist_candidate")}
        for side in ("left", "right")
    }
    completion_pixels = {"left": [], "right": []}
    critical: dict[str, Any] = {}
    contact_rows: list[np.ndarray] = []

    video_path = output_dir / "MASK_V7_PREVIEW.mp4"
    video_width = 640 * 4
    video_height = 480
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{video_width}x{video_height}", "-r", str(source_session["fps"]),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    if ffmpeg.stdin is None:
        raise RuntimeError("ffmpeg preview pipe unavailable")

    for index, (source, frame, old_frame) in enumerate(
        zip(source_frames, candidate["frames"], baseline["frames"], strict=True)
    ):
        if int(frame["frame_index"]) != index or int(old_frame["frame_index"]) != index:
            raise RuntimeError(f"frame order mismatch at {index}")
        if frame["source_frame_sha256"] != source["image"]["sha256"]:
            raise RuntimeError(f"source identity mismatch at {index}")
        h_core = _mask(frame["h_core"])
        obj = _mask(frame["o_visible_core"])
        contact = _mask(frame["u_contact"])
        old_h = _mask(old_frame["h_core"])
        for key in ("instance_id", "confidence"):
            _mask(frame[key])
        prompt_path = Path(frame["prompt_provenance"]["path"])
        if _sha256(prompt_path) != frame["prompt_provenance"]["sha256"]:
            raise RuntimeError(f"prompt provenance SHA mismatch at {index}")
        prompt_ref_sha_pass += 1
        prompt = _load_json(prompt_path)
        for side in ("left", "right"):
            side_metrics = prompt["sides"][side]
            independent_holds[side] += int(side_metrics["independent_hold"] == 1.0)
            for part in candidate_holds[side]:
                candidate_holds[side][part] += int(side_metrics[part]["held"] == 1.0)
            completion_pixels[side].append(
                float(side_metrics["anatomy_bounds"]["analytic_corridor_completion_pixels"])
            )

        area = float(h_core.mean())
        old_area = float(old_h.mean())
        intersection = int(np.count_nonzero(h_core & old_h))
        union = int(np.count_nonzero(h_core | old_h))
        extra = int(np.count_nonzero(h_core & ~old_h))
        human_area.append(area)
        baseline_area.append(old_area)
        ious.append(float(intersection / max(union, 1)))
        extra_ratios.append(float(extra / max(np.count_nonzero(h_core), 1)))
        overlap_pixels += int(np.count_nonzero(h_core & obj))

        image_path = Path(source["image"]["path"])
        if _sha256(image_path) != source["image"]["sha256"]:
            raise RuntimeError(f"RAW image SHA mismatch at {index}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"RAW image decode failure at {index}")
        raw_panel = _label(_resize_panel(image, 640), f"RAW frame={index:05d}")
        old_panel = _label(_resize_panel(_overlay(image, old_h, (0, 0, 255)), 640), "v6 H_core")
        new_panel = _label(_resize_panel(_overlay(image, h_core, (0, 0, 255)), 640), "v7 H_core")
        diag_panel = _label(_resize_panel(_diagnostic(image, h_core, obj, contact), 640), "v7 H(red) O(green) U(blue)")
        row = np.concatenate([raw_panel, old_panel, new_panel, diag_panel], axis=1)
        if row.shape[:2] != (video_height, video_width):
            raise RuntimeError(f"unexpected preview shape at {index}: {row.shape}")
        ffmpeg.stdin.write(row.tobytes())

        if index in critical_frames:
            contact_rows.append(
                np.concatenate(
                    [
                        _label(_resize_panel(image, 320), f"RAW {index}"),
                        _label(_resize_panel(_overlay(image, old_h, (0, 0, 255)), 320), "v6"),
                        _label(_resize_panel(_overlay(image, h_core, (0, 0, 255)), 320), "v7"),
                        _label(_resize_panel(_diagnostic(image, h_core, obj, contact), 320), "v7 H/O/U"),
                    ],
                    axis=1,
                )
            )
            critical[str(index)] = {
                "v6_human_area_ratio": old_area,
                "v7_human_area_ratio": area,
                "v6_v7_iou": ious[-1],
                "v7_pixels_outside_v6_ratio": extra_ratios[-1],
                "left_completion_pixels": completion_pixels["left"][-1],
                "right_completion_pixels": completion_pixels["right"][-1],
            }

    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg preview encoder failed")
    if not contact_rows:
        raise RuntimeError("no critical review frames selected")
    comparison_path = output_dir / "critical_232_241_comparison.png"
    if not cv2.imwrite(str(comparison_path), np.concatenate(contact_rows, axis=0)):
        raise RuntimeError("critical comparison write failed")

    return {
        "schema_validation": "PASS",
        "frame_count": len(candidate["frames"]),
        "raw_sha_verified_frames": len(candidate["frames"]),
        "candidate_output_and_prompt_sha_verified_frames": prompt_ref_sha_pass,
        "h_core_object_core_overlap_pixels": overlap_pixels,
        "human_area_ratio": _quantiles(human_area),
        "baseline_human_area_ratio": _quantiles(baseline_area),
        "v6_v7_iou": _quantiles(ious),
        "v7_pixels_outside_v6_ratio": _quantiles(extra_ratios),
        "independent_hold_frames": independent_holds,
        "candidate_hold_frames": candidate_holds,
        "analytic_corridor_completion_pixels": {
            side: _quantiles(values) for side, values in completion_pixels.items()
        },
        "critical_frames": critical,
        "review_outputs": {
            "critical_comparison": {
                "path": str(comparison_path), "sha256": _sha256(comparison_path)
            },
            "full_preview": {"path": str(video_path), "sha256": _sha256(video_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--critical-start", type=int, default=232)
    parser.add_argument("--critical-end", type=int, default=241)
    args = parser.parse_args()
    metrics = build_review(
        args.source_manifest,
        args.candidate_manifest,
        args.baseline_manifest,
        args.schema,
        args.output_dir,
        list(range(args.critical_start, args.critical_end + 1)),
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
