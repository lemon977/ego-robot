"""Complete-frame visual review for raw Kai22 q22/full-FK evidence.

The renderer consumes the physical-left resize-only RGB, a reloadable full-FK
sidecar, the reconstructed historical clip delta, and the fail-closed tier
receipt.  It does not retarget, smooth, clip, or create a thumb candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence
import uuid

import cv2
import numpy as np

from chaoyang.pipeline.robot_renderer_cycles import UrdfModel


SCHEMA_VERSION = "KAI22_FULL_FK_REVIEW_V1"
OUTPUT_SIZE = (1280, 480)
RGB_SIZE = (640, 480)
PHYSICAL_SIDES = ("left", "right")


class Kai22ReviewError(RuntimeError):
    """Raised when a review would misrepresent or incompletely render evidence."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def full_decode(path: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise Kai22ReviewError("review video must contain exactly one video stream")
    stream = streams[0]
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    frames = int(stream.get("nb_read_frames", -1))
    if decoded.returncode or frames != expected_frames:
        raise Kai22ReviewError(
            f"review full-decode failed: returncode={decoded.returncode}, frames={frames}"
        )
    return {
        "status": "PASS_FULL_DECODE",
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": frames,
        "fps": str(stream["r_frame_rate"]),
    }


def _open_encoder(path: Path, fps: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def frame_annotation_rows(
    *,
    frame_id: int,
    timestamp_s: float,
    highest_tier: str,
    valid: np.ndarray,
    clip_delta: np.ndarray,
) -> list[str]:
    """Return the exact semantic text shown for one source frame."""

    valid_row = np.asarray(valid, dtype=bool)
    delta = np.asarray(clip_delta, dtype=np.float64)
    if valid_row.shape != (2,) or delta.shape != (2, 22):
        raise Kai22ReviewError("one frame must have [2] valid and [2,22] clip delta")
    rows = [
        f"frame {frame_id:04d} | t={timestamp_s:.3f}s",
        f"CURRENT HIGHEST TIER: {highest_tier}",
        "NOT DEVELOPMENT_R0 | control_ground_truth=false",
    ]
    for side, name in enumerate(PHYSICAL_SIDES):
        if not valid_row[side]:
            rows.append(f"physical {name}: valid=false | clip_delta=UNKNOWN")
            continue
        side_delta = delta[side]
        if not np.isfinite(side_delta).all():
            raise Kai22ReviewError("valid frame has unknown clip delta")
        clipped = int(np.sum(np.abs(side_delta) > 1e-12))
        maximum = float(np.max(np.abs(side_delta)))
        rows.append(
            f"physical {name}: valid=true | clipped={clipped}/22 | max|clip_delta|={maximum:.4f} rad"
        )
    return rows


def _projection_scales(fk: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    scales: list[float] = []
    for side in range(2):
        positions = fk[:, side, :, :3, 3]
        selected = positions[valid[:, side]]
        if selected.size == 0:
            scales.append(1000.0)
            continue
        radial = np.max(np.abs(selected[..., :2]))
        if not np.isfinite(radial) or radial <= 1e-9:
            scales.append(1000.0)
        else:
            scales.append(float(min(1800.0, 105.0 / radial)))
    return scales[0], scales[1]


def _draw_hand(
    panel: np.ndarray,
    *,
    model: UrdfModel,
    link_names: Sequence[str],
    transforms: np.ndarray,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    if tuple(link_names) != model.links:
        raise Kai22ReviewError("sidecar link order differs from pinned URDF")
    value = np.asarray(transforms, dtype=np.float64)
    if value.shape != (len(model.links), 4, 4) or not np.isfinite(value).all():
        raise Kai22ReviewError("valid root-relative FK is incomplete or non-finite")
    positions = {
        name: value[index, :3, 3] for index, name in enumerate(model.links)
    }
    projected = {
        name: (
            int(round(center[0] + scale * point[0])),
            int(round(center[1] - scale * point[1])),
        )
        for name, point in positions.items()
    }
    for joint in model.joints:
        cv2.line(
            panel,
            projected[joint.parent],
            projected[joint.child],
            color,
            3,
            cv2.LINE_AA,
        )
    for name, point in projected.items():
        radius = 5 if name == model.root_link else 3
        cv2.circle(panel, point, radius, (240, 240, 240), -1, cv2.LINE_AA)


def compose_review_frame(
    *,
    raw_frame: np.ndarray,
    session_id: str,
    frame_id: int,
    timestamp_s: float,
    highest_tier: str,
    valid: np.ndarray,
    clip_delta: np.ndarray,
    fk: np.ndarray,
    link_names: np.ndarray,
    models: Sequence[UrdfModel],
    scales: Sequence[float],
) -> np.ndarray:
    if highest_tier != "KINEMATIC_ONLY":
        raise Kai22ReviewError("V1 review is fixed to honest KINEMATIC_ONLY evidence")
    if len(models) != 2 or len(scales) != 2:
        raise Kai22ReviewError("two physical-side models/scales are required")
    raw = cv2.resize(np.asarray(raw_frame), RGB_SIZE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(raw, (0, 0), (640, 58), (0, 0, 0), -1)
    cv2.putText(
        raw,
        "RAW physical-left | sourceIndex=1 | resize-only | NO lens remap",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        raw,
        f"{session_id} | frame {frame_id:04d} | t={timestamp_s:.3f}s",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    panel = np.full((480, 640, 3), 24, dtype=np.uint8)
    cv2.putText(
        panel,
        "RAW R0 q22 -> root-relative Kai22 full FK",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "KINEMATIC_ONLY  |  NOT DEVELOPMENT_R0",
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (60, 170, 255),
        2,
        cv2.LINE_AA,
    )
    valid_row = np.asarray(valid, dtype=bool)
    delta = np.asarray(clip_delta, dtype=np.float64)
    fk_row = np.asarray(fk, dtype=np.float64)
    for side, name in enumerate(PHYSICAL_SIDES):
        box_x = 15 + side * 310
        cv2.rectangle(panel, (box_x, 75), (box_x + 295, 340), (75, 75, 75), 1)
        cv2.putText(
            panel,
            f"physical {name} | root-relative XY",
            (box_x + 8, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        if valid_row[side]:
            _draw_hand(
                panel,
                model=models[side],
                link_names=[str(value) for value in link_names[side]],
                transforms=fk_row[side],
                center=(box_x + 148, 210),
                scale=float(scales[side]),
                color=((255, 150, 50), (50, 130, 255))[side],
            )
            clipped = int(np.sum(np.abs(delta[side]) > 1e-12))
            maximum = float(np.max(np.abs(delta[side])))
            validity_text = "valid=true"
            clip_text = f"clip {clipped}/22 | max {maximum:.4f} rad"
        else:
            validity_text = "valid=false | NO FILL"
            clip_text = "clip_delta=UNKNOWN"
        cv2.putText(
            panel,
            validity_text,
            (box_x + 8, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (125, 230, 145) if valid_row[side] else (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            clip_text,
            (box_x + 8, 332),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 180, 255) if valid_row[side] and clipped else (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "Post-clip hard-limit PASS does not grant DEVELOPMENT_R0",
        (15, 376),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 170, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "No new thumb candidate | historical A6 rejection preserved",
        (15, 404),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "visual review only | control_ground_truth=false | physical_deployable=false",
        (15, 446),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )
    return np.hstack((raw, panel))


def render_kai22_full_fk_review(
    *,
    session_id: str,
    source_video: Path,
    full_fk_npz: Path,
    clip_npz: Path,
    tier_result: Path,
    destination: Path,
    models: Sequence[UrdfModel],
) -> dict[str, Any]:
    """Render and fully decode one immutable, complete-frame review video."""

    if destination.exists():
        raise FileExistsError(f"immutable review video exists: {destination}")
    tier = json.loads(tier_result.read_text(encoding="utf-8"))
    highest = tier.get("highest_admitted_level")
    if highest != "KINEMATIC_ONLY":
        raise Kai22ReviewError("review tier must be exactly KINEMATIC_ONLY")
    with np.load(full_fk_npz, allow_pickle=False) as sidecar, np.load(
        clip_npz, allow_pickle=False
    ) as clip:
        if str(sidecar["schema_version"].item()) != "KAI22_FULL_FK_SIDECAR_V1":
            raise Kai22ReviewError("unexpected full-FK sidecar schema")
        if str(sidecar["session_id"].item()) != session_id:
            raise Kai22ReviewError("sidecar session identity mismatch")
        frame_ids = sidecar["source_frame"].copy()
        timestamps = sidecar["timestamp_s"].copy()
        valid = sidecar["q22_valid"].copy()
        fk = sidecar["fk_root_relative"].copy()
        link_names = sidecar["link_names"].copy()
        delta = sidecar["clip_delta"].copy()
        if not np.array_equal(delta, clip["clip_delta"], equal_nan=True):
            raise Kai22ReviewError("clip/full-FK sidecars disagree")
        if not np.array_equal(frame_ids, np.arange(frame_ids.size)):
            raise Kai22ReviewError("review requires exact source frame axis 0..T-1")
        if (
            valid.shape != (frame_ids.size, 2)
            or delta.shape != (frame_ids.size, 2, 22)
            or fk.shape != (frame_ids.size, 2, 23, 4, 4)
        ):
            raise Kai22ReviewError("sidecar review axes are inconsistent")
        if np.any(valid & ~np.isfinite(delta).all(axis=2)):
            raise Kai22ReviewError("valid frame has unknown clip delta")

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise Kai22ReviewError(f"cannot open source video: {source_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise Kai22ReviewError("source video FPS is invalid")
    scales = _projection_scales(fk, valid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}.mp4"
    )
    encoder = _open_encoder(temporary, fps)
    if encoder.stdin is None:
        capture.release()
        raise Kai22ReviewError("ffmpeg encoder stdin is unavailable")
    frames = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            if frames >= frame_ids.size:
                raise Kai22ReviewError("source video has more frames than sidecar")
            rendered = compose_review_frame(
                raw_frame=raw,
                session_id=session_id,
                frame_id=int(frame_ids[frames]),
                timestamp_s=float(timestamps[frames]),
                highest_tier=str(highest),
                valid=valid[frames],
                clip_delta=delta[frames],
                fk=fk[frames],
                link_names=link_names,
                models=models,
                scales=scales,
            )
            encoder.stdin.write(rendered.tobytes())
            frames += 1
    finally:
        capture.release()
        encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    returncode = encoder.wait()
    if returncode or frames != frame_ids.size:
        temporary.unlink(missing_ok=True)
        raise Kai22ReviewError(
            f"review encode failed: returncode={returncode}, frames={frames}, stderr={stderr[-1000:]}"
        )
    decode = full_decode(temporary, frames)
    if (decode["width"], decode["height"]) != OUTPUT_SIZE:
        temporary.unlink(missing_ok=True)
        raise Kai22ReviewError("review video resolution mismatch")
    os.replace(temporary, destination)
    valid_delta = np.abs(delta[np.broadcast_to(valid[..., None], delta.shape)])
    clipped_joint = np.broadcast_to(valid[..., None], delta.shape) & (np.abs(delta) > 1e-12)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_COMPLETE_FRAME_REVIEW",
        "session_id": session_id,
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY_NO_LENS_REMAP",
        "frame_count": frames,
        "source_video": exact_ref(source_video),
        "full_fk_sidecar": exact_ref(full_fk_npz),
        "clip_sidecar": exact_ref(clip_npz),
        "tier_result": exact_ref(tier_result),
        "review_video": exact_ref(destination),
        "decode": decode,
        "displayed": {
            "raw_physical_left_rgb": True,
            "raw_r0_q22": True,
            "root_relative_full_fk": True,
            "clip_delta": True,
            "validity": True,
            "highest_tier": "KINEMATIC_ONLY",
            "not_development_r0": True,
        },
        "diagnostics": {
            "valid_side_frames": int(valid.sum()),
            "clipped_side_frames": int(np.any(clipped_joint, axis=2).sum()),
            "clipped_joint_values": int(clipped_joint.sum()),
            "max_abs_clip_delta_rad": float(valid_delta.max()),
        },
        "new_thumb_candidate_created": False,
        "historical_a6_rejection_preserved": True,
        "numeric_quality_pass": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Complete-frame visual review of frozen raw R0 q22, reconstructed clip delta, "
            "and root-relative pinned-URDF FK only; post-clip legality does not grant "
            "DEVELOPMENT_R0, control, external accuracy, or deployment authority."
        ),
    }
