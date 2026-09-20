#!/usr/bin/env python3
"""Render a fail-closed Exact78 physical-mono plus hand-only FK review.

The KaiHand panels are wrist/root-relative diagrams.  They are deliberately
not overlaid on the camera image: the current evidence does not observe the
robot base, camera-to-base installation, or tool-to-hand mount.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCHEMA = "EXACT78_HAND_ONLY_VISUAL_REVIEW_V32"
FK_SCHEMA = "EXACT78_HAND_ONLY_Q22_FK_V32"
RGB_SCHEMA = "EXACT78_SAM31_HUMAN_MASK_CANARY_V1"
OUTPUT_SIZE = (1280, 600)
RAW_PANEL_SIZE = (800, 600)
HAND_PANEL_SIZE = (480, 300)
ARM_BLOCKER_STATUS = "BLOCKED_UNOBSERVED_INSTALLATION_GEOMETRY"
FINGER_COLORS = {
    "thumb": (80, 190, 255),
    "index": (80, 230, 120),
    "middle": (255, 190, 70),
    "ring": (220, 100, 220),
    "pinky": (100, 160, 255),
}


class HandOnlyReviewError(RuntimeError):
    """Raised when review lineage or decode closure is not exact."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HandOnlyReviewError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def reject_archive_inputs(paths: Sequence[Path]) -> None:
    for path in paths:
        resolved = path.resolve(strict=True)
        if "archive" in resolved.parts:
            raise HandOnlyReviewError(f"archive input forbidden: {resolved}")


def validate_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for field in (
        "control_ground_truth",
        "training_eligible",
        "physical_deployable",
        "external_metric_authority",
    ):
        if field in payload and payload[field] is not False:
            raise HandOnlyReviewError(f"{label} {field} must remain false")


def validate_inputs(
    *,
    session_id: str,
    rgb_result_path: Path,
    rgb_video_path: Path,
    fk_result_path: Path,
    fk_npz_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    reject_archive_inputs(
        (rgb_result_path, rgb_video_path, fk_result_path, fk_npz_path)
    )
    rgb = load_json(rgb_result_path)
    if (
        rgb.get("schema_version") != RGB_SCHEMA
        or rgb.get("status") != "PASS_DEVELOPMENT_MASKS"
        or rgb.get("session_id") != session_id
        or rgb.get("image_domain")
        != "CURRENT_MONO_1280X960_NO_ADDITIONAL_LENS_REMAP"
    ):
        raise HandOnlyReviewError("current physical-mono RGB lineage required")
    if rgb.get("inputs", {}).get("video") != ref(rgb_video_path):
        raise HandOnlyReviewError("RGB result does not bind the exact source video")
    claims = rgb.get("claims", {})
    if claims.get("training_eligible") is not False:
        raise HandOnlyReviewError("RGB input must remain ineligible for training")

    fk_result = load_json(fk_result_path)
    if (
        fk_result.get("schema_version") != FK_SCHEMA
        or fk_result.get("status") != "COMPLETED_HAND_ONLY_Q22_FK_ARM_BLOCKED"
        or fk_result.get("session_id") != session_id
        or fk_result.get("temporal_authority") != "OFFLINE_NONCAUSAL"
        or fk_result.get("arm_admission", {}).get("status") != ARM_BLOCKER_STATUS
    ):
        raise HandOnlyReviewError("current arm-blocked hand-only FK result required")
    validate_false_authority(fk_result, label="FK result")
    if fk_result.get("q22_fk") != ref(fk_npz_path):
        raise HandOnlyReviewError("FK result does not bind the exact q22/FK NPZ")

    with np.load(fk_npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "schema_version",
        "session_id",
        "source_frame",
        "timestamp_s",
        "physical_robot_side",
        "anatomical_side",
        "joint_names",
        "link_names",
        "q22",
        "q22_valid",
        "source_hawor_observed",
        "q22_inferred_not_observed",
        "projection_delta",
        "hard_limit_pass",
        "fk_root_relative",
        "fk_finite",
        "control_ground_truth",
        "physical_deployable",
    }
    if not required.issubset(arrays):
        raise HandOnlyReviewError(f"FK fields missing: {sorted(required - set(arrays))}")
    if str(arrays["schema_version"].item()) != FK_SCHEMA:
        raise HandOnlyReviewError("FK NPZ schema drift")
    if str(arrays["session_id"].item()) != session_id:
        raise HandOnlyReviewError("FK NPZ session mismatch")
    frame_count = int(rgb.get("frame_count", -1))
    if frame_count <= 0 or not np.array_equal(
        arrays["source_frame"], np.arange(frame_count)
    ):
        raise HandOnlyReviewError("FK source frame axis must be complete and contiguous")
    expected_shapes = {
        "q22": (frame_count, 2, 22),
        "q22_valid": (frame_count, 2),
        "source_hawor_observed": (frame_count, 2),
        "q22_inferred_not_observed": (frame_count, 2),
        "projection_delta": (frame_count, 2, 22),
        "hard_limit_pass": (frame_count, 2),
        "fk_root_relative": (frame_count, 2, 23, 4, 4),
        "fk_finite": (frame_count, 2),
        "link_names": (2, 23),
    }
    for field, shape in expected_shapes.items():
        if arrays[field].shape != shape:
            raise HandOnlyReviewError(f"{field} shape drift: {arrays[field].shape}")
    valid = np.asarray(arrays["q22_valid"], dtype=bool)
    if not np.array_equal(valid, arrays["fk_finite"].astype(bool)):
        raise HandOnlyReviewError("q22/FK validity mismatch")
    if not np.all(arrays["hard_limit_pass"][valid]):
        raise HandOnlyReviewError("valid hand state violates a hard joint limit")
    if bool(arrays["control_ground_truth"].item()) or bool(
        arrays["physical_deployable"].item()
    ):
        raise HandOnlyReviewError("FK NPZ authority drift")
    return rgb, fk_result, arrays


def finger_edges(link_names: np.ndarray) -> list[tuple[int, int, str]]:
    names = [str(value) for value in link_names.tolist()]
    if len(names) != 23 or "base_link" not in names[0]:
        raise HandOnlyReviewError("unexpected KaiHand link order")
    groups: dict[str, list[tuple[int, int]]] = {key: [] for key in FINGER_COLORS}
    pattern = re.compile(r"_(thumb|index|middle|ring|pinky)_link(\d+)$")
    for index, name in enumerate(names[1:], start=1):
        match = pattern.search(name)
        if match is None:
            raise HandOnlyReviewError(f"unrecognized KaiHand link: {name}")
        groups[match.group(1)].append((int(match.group(2)), index))
    edges: list[tuple[int, int, str]] = []
    for finger, rows in groups.items():
        ordered = [index for _, index in sorted(rows)]
        if not ordered:
            raise HandOnlyReviewError(f"missing {finger} links")
        edges.append((0, ordered[0], finger))
        edges.extend((parent, child, finger) for parent, child in zip(ordered, ordered[1:]))
    return edges


def text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    *,
    scale: float = 0.47,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_hand_panel(
    *,
    transforms: np.ndarray,
    link_names: np.ndarray,
    valid: bool,
    physical_side: str,
    anatomical_side: str,
    frame_id: int,
    projection_delta: np.ndarray,
) -> np.ndarray:
    width, height = HAND_PANEL_SIZE
    panel = np.full((height, width, 3), (30, 32, 38), dtype=np.uint8)
    text(
        panel,
        f"physical {physical_side.upper()} | anatomical {anatomical_side}",
        (12, 24),
        scale=0.48,
        color=(225, 225, 225),
    )
    text(
        panel,
        f"f{frame_id:05d} | Kai22 root-relative FK | INFERRED",
        (12, 47),
        scale=0.39,
        color=(165, 210, 255),
    )
    if not valid:
        text(
            panel,
            "UNKNOWN: no valid q22/FK (not filled)",
            (34, 160),
            scale=0.55,
            color=(80, 170, 255),
            thickness=2,
        )
        return panel

    positions = np.asarray(transforms[:, :3, 3], dtype=np.float64)
    if not np.isfinite(positions).all():
        raise HandOnlyReviewError("valid FK contains nonfinite coordinates")
    px = np.rint(width / 2 + positions[:, 0] / 0.22 * (width - 90)).astype(int)
    py = np.rint(68 + (-positions[:, 2]) / 0.18 * (height - 105)).astype(int)
    points = np.stack((px, py), axis=1)
    for parent, child, finger in finger_edges(link_names):
        cv2.line(
            panel,
            tuple(points[parent]),
            tuple(points[child]),
            FINGER_COLORS[finger],
            3,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(panel, tuple(point), 4, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(point), 4, (40, 40, 40), 1, cv2.LINE_AA)
    delta = np.asarray(projection_delta, dtype=np.float64)
    text(
        panel,
        f"projection |max dq|={np.degrees(np.max(np.abs(delta))):.2f} deg",
        (12, height - 14),
        scale=0.39,
        color=(190, 190, 190),
    )
    return panel


def decoded_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HandOnlyReviewError(f"video failed to open: {path}")
    digests: list[str] = []
    width = height = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if width < 0:
                height, width = frame.shape[:2]
            elif frame.shape[:2] != (height, width):
                raise HandOnlyReviewError("decoded video geometry changes mid-stream")
            digests.append(hashlib.sha256(frame.tobytes()).hexdigest())
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return {
        "decoded_frames": len(digests),
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_bgr_sha256_by_frame": digests,
    }


def render_review(
    *,
    session_id: str,
    rgb_result_path: Path,
    rgb_video_path: Path,
    fk_result_path: Path,
    fk_npz_path: Path,
    output_root: Path,
    fps: float | None = None,
) -> dict[str, Any]:
    rgb, fk_result, arrays = validate_inputs(
        session_id=session_id,
        rgb_result_path=rgb_result_path,
        rgb_video_path=rgb_video_path,
        fk_result_path=fk_result_path,
        fk_npz_path=fk_npz_path,
    )
    if output_root.exists() or output_root.is_symlink():
        raise HandOnlyReviewError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    frame_count = len(arrays["source_frame"])
    capture = cv2.VideoCapture(str(rgb_video_path))
    if not capture.isOpened():
        raise HandOnlyReviewError("source physical-mono video failed to open")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = float(fps if fps is not None else source_fps)
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise HandOnlyReviewError("invalid source/output fps")
    video_path = output_root / f"{session_id}_HAND_ONLY_ROOT_RELATIVE_REVIEW.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        capture.release()
        raise HandOnlyReviewError("review video writer failed to open")

    source_pixel_sha: list[str] = []
    try:
        for slot in range(frame_count):
            ok, raw = capture.read()
            if not ok:
                raise HandOnlyReviewError(f"source video ended before frame {slot}")
            if raw.shape[:2] != (960, 1280):
                raise HandOnlyReviewError(
                    f"source is not current 1280x960 mono: {raw.shape[:2]}"
                )
            source_pixel_sha.append(hashlib.sha256(raw.tobytes()).hexdigest())
            raw_panel = cv2.resize(raw, RAW_PANEL_SIZE, interpolation=cv2.INTER_AREA)
            text(
                raw_panel,
                f"{session_id} | frame {slot:05d} | physical mono RAW",
                (14, 25),
                scale=0.53,
                thickness=2,
            )
            text(
                raw_panel,
                "CURRENT_MONO_1280X960 | resize for review only | no lens remap",
                (14, 50),
                scale=0.43,
            )
            text(
                raw_panel,
                "NOT ROBOTIZED | NOT TRAINING INPUT | ARM BLOCKED",
                (14, 582),
                scale=0.53,
                color=(80, 170, 255),
                thickness=2,
            )
            panels = []
            for side in range(2):
                panels.append(
                    render_hand_panel(
                        transforms=arrays["fk_root_relative"][slot, side],
                        link_names=arrays["link_names"][side],
                        valid=bool(arrays["q22_valid"][slot, side]),
                        physical_side=str(arrays["physical_robot_side"][side]),
                        anatomical_side=str(arrays["anatomical_side"][side]),
                        frame_id=int(arrays["source_frame"][slot]),
                        projection_delta=arrays["projection_delta"][slot, side],
                    )
                )
            composite = np.hstack((raw_panel, np.vstack(panels)))
            if composite.shape[:2] != (OUTPUT_SIZE[1], OUTPUT_SIZE[0]):
                raise HandOnlyReviewError("internal review geometry drift")
            writer.write(composite)
        extra, _ = capture.read()
        if extra:
            raise HandOnlyReviewError("source video contains more frames than FK")
    finally:
        capture.release()
        writer.release()

    review_decode = decoded_video(video_path)
    if (
        review_decode["decoded_frames"] != frame_count
        or (review_decode["width"], review_decode["height"]) != OUTPUT_SIZE
    ):
        raise HandOnlyReviewError("review full-decode closure failed")
    if len(source_pixel_sha) != frame_count:
        raise HandOnlyReviewError("source full-decode closure failed")
    frame_manifest = {
        "schema_version": "EXACT78_HAND_ONLY_REVIEW_FRAME_SHA_V32",
        "session_id": session_id,
        "frame_count": frame_count,
        "source_frame": arrays["source_frame"].astype(int).tolist(),
        "timestamp_s": arrays["timestamp_s"].astype(float).tolist(),
        "source_decoded_bgr_sha256_by_frame": source_pixel_sha,
        "review_decoded_bgr_sha256_by_frame": review_decode.pop(
            "decoded_bgr_sha256_by_frame"
        ),
    }
    frame_manifest_path = output_root / "FRAME_SHA256.json"
    atomic_json(frame_manifest_path, frame_manifest)
    decode_receipt = {
        "schema_version": "EXACT78_HAND_ONLY_REVIEW_DECODE_RECEIPT_V32",
        "session_id": session_id,
        "status": "PASS_FULL_196_FRAME_DECODE"
        if frame_count == 196
        else "PASS_FULL_FRAME_DECODE_TEST_FIXTURE",
        "source": {
            **ref(rgb_video_path),
            "decoded_frames": frame_count,
            "width": 1280,
            "height": 960,
            "fps": source_fps,
        },
        "review": {**ref(video_path), **review_decode},
        "frame_sha256": ref(frame_manifest_path),
    }
    decode_receipt_path = output_root / "DECODE_RECEIPT.json"
    atomic_json(decode_receipt_path, decode_receipt)
    valid = np.asarray(arrays["q22_valid"], dtype=bool)
    result = {
        "schema_version": SCHEMA,
        "created_at": now(),
        "status": "COMPLETE_DEVELOPMENT_HAND_ONLY_REVIEW_ARM_BLOCKED",
        "session_id": session_id,
        "frame_count": frame_count,
        "temporal_authority": "OFFLINE_NONCAUSAL",
        "capability": "HAND_ONLY",
        "image_domain": rgb["image_domain"],
        "source_pixel_policy": "PHYSICAL_MONO_RAW_UNMODIFIED_EXCEPT_REVIEW_RESIZE",
        "robot_panel_policy": "KAIHAND_ROOT_RELATIVE_FK_NOT_CAMERA_OVERLAY",
        "inputs": {
            "rgb_result": ref(rgb_result_path),
            "rgb_video": ref(rgb_video_path),
            "fk_result": ref(fk_result_path),
            "q22_fk": ref(fk_npz_path),
            "producer_code": ref(Path(__file__)),
            "sam_masks_read": False,
            "clean_or_silver_read": False,
        },
        "artifacts": {
            "review_video": ref(video_path),
            "decode_receipt": ref(decode_receipt_path),
            "frame_sha256": ref(frame_manifest_path),
        },
        "decode": {
            "source_frames": frame_count,
            "review_frames": int(review_decode["decoded_frames"]),
            "review_width": int(review_decode["width"]),
            "review_height": int(review_decode["height"]),
            "full_decode_pass": True,
        },
        "hand_only_coverage": {
            "valid_side_frames": int(valid.sum()),
            "unknown_side_frames": int(valid.size - valid.sum()),
            "unknown_policy": "VISIBLE_UNKNOWN_PANEL_NO_FORWARD_FILL",
        },
        "arm_admission": fk_result["arm_admission"],
        "visual_review_status": "READY_FOR_HUMAN_REVIEW_NOT_YET_ADJUDICATED",
        "robotized_training_input": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "producer_registration_status": "PENDING_PUBLISHER_CONTRACT_REGISTRATION",
        "claim_limit": (
            "Complete development review of physical-mono Raw beside offline-noncausal "
            "root-relative KaiHand q22/FK. The panels are not an ego-camera Robotized "
            "render and are forbidden as Robotized training input. Arm/world/mount "
            "installation remains blocked and unobserved."
        ),
    }
    atomic_json(output_root / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--rgb-result", type=Path, required=True)
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--fk-result", type=Path, required=True)
    parser.add_argument("--fk-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    result = render_review(
        session_id=args.session,
        rgb_result_path=args.rgb_result.resolve(strict=True),
        rgb_video_path=args.rgb_video.resolve(strict=True),
        fk_result_path=args.fk_result.resolve(strict=True),
        fk_npz_path=args.fk_npz.resolve(strict=True),
        output_root=args.output_root.resolve(),
        fps=args.fps,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "session_id": result["session_id"],
                "frame_count": result["frame_count"],
                "result": str((args.output_root / "RESULT.json").resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
