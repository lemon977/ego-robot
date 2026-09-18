#!/usr/bin/env python3
"""Fill only tightly bounded internal gaps in a reviewed HaWoR MANO track.

The detector-backed ``observed`` mask is immutable.  A separate
``short_gap_inferred`` mask marks offline, acausal MANO-parameter interpolation
between same-side bracketing observations.  The resulting
``visual_continuity_valid`` mask may be used for development visualization and
motion continuity, while strict observation/contact consumers keep using
``observed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__:
    from . import run_hawor_bounded_parameter_successor as bounded
else:
    import run_hawor_bounded_parameter_successor as bounded


MAX_GAP_FRAMES = 2
MIN_ENDPOINT_CONFIDENCE = 0.4
MAX_WRIST_WORLD_STEP_MM = 20.0
MAX_JOINT_WORLD_STEP_MM = 25.0
MAX_JOINT_2D_STEP_PX = 40.0
SIDE_NAMES = ("left", "right")
CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
OBSERVED_COLORS = ((255, 255, 0), (255, 0, 255))
INFERRED_COLORS = ((0, 255, 255), (0, 165, 255))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def internal_false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open false runs that have true observations on both sides."""

    values = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist() + [True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            end = index
            if start > 0 and end < len(values) and values[start - 1] and values[end]:
                runs.append((start, end))
            start = None
    return runs


def pair_slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    shape = np.asarray(left).shape
    matrices = np.stack((left, right), axis=0).reshape(2, -1, 3, 3)
    output = np.empty(matrices.shape[1:], dtype=np.float64)
    for index in range(matrices.shape[1]):
        interpolation = Slerp((0.0, 1.0), Rotation.from_matrix(matrices[:, index]))
        output[index] = interpolation((fraction,)).as_matrix()[0]
    return output.reshape(shape)


def gap_motion_metrics(source: dict[str, np.ndarray], side: int, start: int, end: int) -> dict[str, float]:
    left = start - 1
    right = end
    steps = right - left
    world = source["joints_3d_world"][side]
    image = source["joints_2d"][side]
    return {
        "endpoint_confidence_min": float(min(
            source["detector_confidence"][side, left], source["detector_confidence"][side, right]
        )),
        "wrist_world_step_mm": float(np.linalg.norm(world[right, 0] - world[left, 0]) * 1000.0 / steps),
        "max_joint_world_step_mm": float(
            np.max(np.linalg.norm(world[right] - world[left], axis=-1)) * 1000.0 / steps
        ),
        "max_joint_2d_step_px": float(np.max(np.linalg.norm(image[right] - image[left], axis=-1)) / steps),
    }


def gap_is_eligible(start: int, end: int, metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if end - start > MAX_GAP_FRAMES:
        failures.append("GAP_TOO_LONG")
    if metrics["endpoint_confidence_min"] < MIN_ENDPOINT_CONFIDENCE:
        failures.append("ENDPOINT_CONFIDENCE_LOW")
    if metrics["wrist_world_step_mm"] > MAX_WRIST_WORLD_STEP_MM:
        failures.append("WRIST_WORLD_MOTION_HIGH")
    if metrics["max_joint_world_step_mm"] > MAX_JOINT_WORLD_STEP_MM:
        failures.append("JOINT_WORLD_MOTION_HIGH")
    if metrics["max_joint_2d_step_px"] > MAX_JOINT_2D_STEP_PX:
        failures.append("JOINT_IMAGE_MOTION_HIGH")
    return not failures, failures


def interpolate_gap(
    source: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    continuity_boxes: np.ndarray,
    side: int,
    start: int,
    end: int,
) -> None:
    left = start - 1
    right = end
    c2w = source["c2w"]
    left_world_translation = c2w[left, :3, :3] @ parameters["root_translation_camera"][side, left] + c2w[left, :3, 3]
    right_world_translation = c2w[right, :3, :3] @ parameters["root_translation_camera"][side, right] + c2w[right, :3, 3]
    left_world_rotation = c2w[left, :3, :3] @ parameters["root_orient_camera"][side, left]
    right_world_rotation = c2w[right, :3, :3] @ parameters["root_orient_camera"][side, right]
    for frame in range(start, end):
        fraction = (frame - left) / (right - left)
        world_translation = (1.0 - fraction) * left_world_translation + fraction * right_world_translation
        parameters["root_translation_camera"][side, frame] = (
            c2w[frame, :3, :3].T @ (world_translation - c2w[frame, :3, 3])
        )
        world_rotation = pair_slerp(left_world_rotation, right_world_rotation, fraction)
        parameters["root_orient_camera"][side, frame] = c2w[frame, :3, :3].T @ world_rotation
        parameters["hand_pose_rotmat"][side, frame] = pair_slerp(
            parameters["hand_pose_rotmat"][side, left],
            parameters["hand_pose_rotmat"][side, right],
            fraction,
        )
        parameters["betas"][side, frame] = (
            (1.0 - fraction) * parameters["betas"][side, left]
            + fraction * parameters["betas"][side, right]
        )
        continuity_boxes[side, frame] = (
            (1.0 - fraction) * source["detector_boxes_xyxy"][side, left]
            + fraction * source["detector_boxes_xyxy"][side, right]
        )


def longest_false_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def draw_hand(image: np.ndarray, joints: np.ndarray, color: tuple[int, int, int], *, inferred: bool) -> None:
    points = np.rint(joints).astype(np.int32)
    thickness = 2 if inferred else 3
    for chain in CHAINS:
        cv2.polylines(image, [points[np.asarray(chain)]], False, color, thickness, cv2.LINE_AA)
    radius = 3 if inferred else 4
    for point in points:
        cv2.circle(image, tuple(point), radius, color, -1, cv2.LINE_AA)


def render_review(
    video: Path,
    source: dict[str, np.ndarray],
    output_joints: dict[str, np.ndarray],
    inferred: np.ndarray,
    continuity_valid: np.ndarray,
    output: Path,
    session_id: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = source["observed"].shape[1]
    if (width, height) != (1280, 960) or abs(fps - float(source["fps"])) > 1e-5:
        raise RuntimeError("source video geometry/FPS differs from MANO source")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width * 2}x{height}",
        "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    decoded = 0
    try:
        for frame in range(frame_count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {frame}")
            baseline = image.copy()
            continuity = image.copy()
            for side in range(2):
                if source["observed"][side, frame]:
                    draw_hand(baseline, source["joints_2d"][side, frame], OBSERVED_COLORS[side], inferred=False)
                if continuity_valid[side, frame]:
                    is_inferred = bool(inferred[side, frame])
                    color = INFERRED_COLORS[side] if is_inferred else OBSERVED_COLORS[side]
                    draw_hand(continuity, output_joints["joints_2d"][side, frame], color, inferred=is_inferred)
            for panel, title in ((baseline, "bounded_v2: observed only"), (continuity, "short-gap continuity")):
                cv2.rectangle(panel, (0, 0), (width - 1, 68), (0, 0, 0), -1)
                cv2.putText(panel, f"{session_id} | frame {frame + 1}/{frame_count} | {title}",
                            (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
            baseline_state = " ".join(
                f"{name[0].upper()}={'OBS' if source['observed'][side, frame] else 'MISS'}"
                for side, name in enumerate(SIDE_NAMES)
            )
            continuity_state = " ".join(
                f"{name[0].upper()}={'INFERRED' if inferred[side, frame] else ('OBS' if source['observed'][side, frame] else 'MISS')}"
                for side, name in enumerate(SIDE_NAMES)
            )
            cv2.putText(baseline, baseline_state, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(continuity, continuity_state, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (80, 255, 255) if inferred[:, frame].any() else (220, 220, 220), 2, cv2.LINE_AA)
            assert process.stdin is not None
            process.stdin.write(np.hstack((baseline, continuity)).tobytes())
            decoded += 1
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    verify = cv2.VideoCapture(str(output))
    output_frames = 0
    while True:
        ok, _ = verify.read()
        if not ok:
            break
        output_frames += 1
    verify.release()
    if return_code != 0 or decoded != frame_count or output_frames != frame_count:
        raise RuntimeError(
            f"review full-decode failed: ffmpeg={return_code}, input={decoded}, "
            f"output={output_frames}, expected={frame_count}"
        )
    return {
        "input_frames": decoded,
        "output_frames": output_frames,
        "resolution": [width * 2, height],
        "fps": fps,
        "full_decode_pass": True,
    }


def main() -> int:
    runtime = bounded.validate_runtime_environment(Path(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded-npz", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.bounded_npz.resolve(strict=True)
    video_path = args.source_video.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"fresh output root required: {output_root}")
    stage = output_root.parent / f".{output_root.name}.stage.{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise RuntimeError(f"staging collision: {stage}")
    stage.mkdir(parents=True)
    started = time.time()
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            source = {name: np.asarray(archive[name]) for name in archive.files}
        required = {
            "observed", "detector_confidence", "detector_boxes_xyxy", "c2w", "intrinsics", "fps",
            "root_translation_camera", "root_orient_camera", "hand_pose_rotmat", "betas",
            "joints_2d", "joints_3d_camera", "joints_3d_world", "provenance",
        }
        missing = sorted(required - source.keys())
        if missing:
            raise RuntimeError(f"bounded source keys missing: {missing}")
        observed = np.asarray(source["observed"], dtype=bool)
        if observed.ndim != 2 or observed.shape[0] != 2:
            raise RuntimeError("observed mask must have shape (2,frames)")
        parameters = {
            name: source[name].copy()
            for name in ("root_translation_camera", "root_orient_camera", "hand_pose_rotmat", "betas")
        }
        continuity_boxes = source["detector_boxes_xyxy"].copy()
        inferred = np.zeros_like(observed)
        gaps: list[dict[str, Any]] = []
        for side, side_name in enumerate(SIDE_NAMES):
            for start, end in internal_false_runs(observed[side]):
                metrics = gap_motion_metrics(source, side, start, end)
                eligible, failures = gap_is_eligible(start, end, metrics)
                record = {
                    "side": side_name,
                    "start_frame_zero_based": start,
                    "end_frame_zero_based_inclusive": end - 1,
                    "length_frames": end - start,
                    "left_bracket_frame": start - 1,
                    "right_bracket_frame": end,
                    "eligible": eligible,
                    "failures": failures,
                    "metrics": metrics,
                }
                if eligible:
                    interpolate_gap(source, parameters, continuity_boxes, side, start, end)
                    inferred[side, start:end] = True
                gaps.append(record)
        continuity_valid = observed | inferred
        materialization_source = dict(source)
        materialization_source["observed"] = continuity_valid
        materialized = bounded.materialize(materialization_source, parameters)
        output_joints: dict[str, np.ndarray] = {}
        for name in ("joints_2d", "joints_3d_camera", "joints_3d_world"):
            output_joints[name] = source[name].copy()
            output_joints[name][inferred] = materialized[name][inferred]
        provenance = np.asarray(source["provenance"], dtype="U32")
        provenance[inferred] = "SHORT_GAP_MANO_INTERP"
        payload = dict(source)
        payload.update(parameters)
        payload.update(output_joints)
        payload.update({
            "observed": observed,
            "short_gap_inferred": inferred,
            "visual_continuity_valid": continuity_valid,
            "provenance": provenance,
            "continuity_boxes_xyxy": continuity_boxes,
            "continuity_method": np.asarray("OFFLINE_ACAUSAL_BOUNDED_MANO_PARAMETER_INTERPOLATION", dtype="U64"),
            "short_gap_max_frames": np.asarray(MAX_GAP_FRAMES, dtype=np.int32),
        })
        observed_exact = all(
            np.array_equal(output_joints[name][observed], source[name][observed], equal_nan=True)
            for name in output_joints
        )
        detector_evidence_unchanged = (
            np.array_equal(payload["detector_confidence"], source["detector_confidence"], equal_nan=True)
            and np.array_equal(payload["detector_boxes_xyxy"], source["detector_boxes_xyxy"], equal_nan=True)
        )
        npz_path = stage / "HAWOR_BOUNDED_V2_SHORT_GAP_CONTINUITY.npz"
        np.savez_compressed(npz_path, **payload)
        review_path = stage / "HAWOR_BOUNDED_V2_VS_SHORT_GAP_CONTINUITY.mp4"
        review = render_review(
            video_path, source, output_joints, inferred, continuity_valid, review_path, args.session_id
        )
        per_side = {}
        for side, name in enumerate(SIDE_NAMES):
            per_side[name] = {
                "observed_frames": int(observed[side].sum()),
                "short_gap_inferred_frames": int(inferred[side].sum()),
                "visual_continuity_valid_frames": int(continuity_valid[side].sum()),
                "remaining_missing_frames": int((~continuity_valid[side]).sum()),
                "longest_remaining_gap_frames": longest_false_run(continuity_valid[side]),
                "inferred_frame_indices_zero_based": np.flatnonzero(inferred[side]).tolist(),
                "bone_length_cv_continuity": bounded.bone_cv(output_joints["joints_3d_world"][side], continuity_valid[side]),
            }
        failures: list[str] = []
        if not inferred.any():
            failures.append("NO_ELIGIBLE_GAP_FILLED")
        if not observed_exact:
            failures.append("OBSERVED_GEOMETRY_CHANGED")
        if not detector_evidence_unchanged:
            failures.append("DETECTOR_EVIDENCE_CHANGED")
        if np.any(inferred & observed):
            failures.append("INFERRED_OVERLAPS_OBSERVED")
        if not np.isfinite(output_joints["joints_3d_camera"][continuity_valid]).all():
            failures.append("CONTINUITY_GEOMETRY_NONFINITE")
        status = "PASS_CONTINUITY_CANARY_NEEDS_HUMAN_REVIEW" if not failures else "HOLD_CONTINUITY_GATES"
        result = {
            "schema_version": "hawor-bounded-v2-short-gap-continuity-v1",
            "status": status,
            "session_id": args.session_id,
            "algorithm": {
                "input": "hawor_bounded_v2",
                "method": "offline acausal MANO parameter interpolation in world/camera-consistent coordinates",
                "max_gap_frames": MAX_GAP_FRAMES,
                "max_gap_ms_at_30fps": 1000.0 * MAX_GAP_FRAMES / 30.0,
                "endpoint_confidence_min": MIN_ENDPOINT_CONFIDENCE,
                "wrist_world_step_mm_max": MAX_WRIST_WORLD_STEP_MM,
                "joint_world_step_mm_max": MAX_JOINT_WORLD_STEP_MM,
                "joint_2d_step_px_max": MAX_JOINT_2D_STEP_PX,
                "edge_extrapolation": False,
                "cross_side_identity_matching": False,
                "pico_controller_consumed": False,
            },
            "claim_limit": (
                "Development visual/motion continuity only. observed remains detector-backed and immutable; "
                "SHORT_GAP_MANO_INTERP is not a HaWoR observation, contact truth, or causal online estimate."
            ),
            "failures": failures,
            "gaps": gaps,
            "metrics": {
                "sides": per_side,
                "bilateral_observed_frames": int(np.all(observed, axis=0).sum()),
                "bilateral_visual_continuity_frames": int(np.all(continuity_valid, axis=0).sum()),
                "observed_geometry_preserved_exactly": observed_exact,
                "observed_mask_preserved_exactly": True,
                "detector_evidence_preserved_exactly": detector_evidence_unchanged,
            },
            "runtime_environment": runtime,
            "producer": evidence(Path(__file__)),
            "dependencies": {
                "bounded_parameter_runtime": evidence(Path(bounded.__file__)),
            },
            "validation": {"gpu_calls": 0, "review_video": review},
            "inputs": {"bounded_npz": evidence(source_path), "source_video": evidence(video_path)},
            "outputs": {"continuity_npz": evidence(npz_path), "review_video": evidence(review_path)},
            "wall_seconds": time.time() - started,
        }
        atomic_json(stage / "RESULT.json", result)
        os.replace(stage, output_root)
        final_result = json.loads((output_root / "RESULT.json").read_text(encoding="utf-8"))
        for record in final_result["outputs"].values():
            record["path"] = str(output_root / Path(record["path"]).name)
        atomic_json(output_root / "RESULT.json", final_result)
        print(json.dumps(final_result, ensure_ascii=False))
        return 0 if status.startswith("PASS") else 2
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        if str(error).startswith("WRONG_RUNTIME_ENTRYPOINT"):
            print(str(error), file=sys.stderr)
            raise SystemExit(78) from error
        raise
