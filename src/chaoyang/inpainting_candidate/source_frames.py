"""Raw selected-eye pixels and explicit native frame mapping for candidates.

The source declaration or explicitly adopted device/snapshot visual assumption
owns routing; neither implies verified physical eye. No old pixels, masks or
motion arrays are consumed. This stage cannot qualify projection or QPC truth.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from chaoyang.governance.common import atomic_json, artifact_ref, now_iso
from .contracts import canonical_sha, verify_ref
from .resources import PROJECT_ROOT


def decode_physical_eye(video: Path, sbs_index: int,
                        output_size: tuple[int, int] = (640, 480)) -> Iterator[tuple[np.ndarray, dict]]:
    import av
    import cv2
    if type(sbs_index) is not int or sbs_index not in (0, 1):
        raise ValueError("SOURCE_PHYSICAL_EYE_REQUIRED")
    width, height = output_size
    if (width, height) != (640, 480):
        raise ValueError("SOURCE_FROZEN_REFERENCE_SIZE_REQUIRED")
    cv2.setNumThreads(1)
    with av.open(str(video)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("SOURCE_ONE_VIDEO_STREAM_REQUIRED")
        container.streams.video[0].codec_context.thread_count = 2
        encoded_size = None
        for frame in container.decode(video=0):
            size = (frame.width, frame.height)
            if frame.width % 2 or frame.width <= 0 or frame.height <= 0:
                raise ValueError("SOURCE_SBS_DOMAIN_INVALID")
            if encoded_size is not None and encoded_size != size:
                raise ValueError("SOURCE_SBS_DOMAIN_CHANGED")
            encoded_size = size
            eye_width = frame.width // 2
            if eye_width * height != frame.height * width:
                raise ValueError("SOURCE_NONISOTROPIC_RESIZE_UNSUPPORTED")
            rgb = frame.to_ndarray(format="rgb24")
            crop_x = sbs_index * eye_width
            cropped = rgb[:, crop_x:crop_x + eye_width]
            pixels = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
            yield pixels, {"decoder_pts": frame.pts,
                           "decoder_time_base": str(frame.time_base) if frame.time_base else None,
                           "sbs_size": list(size), "crop": [crop_x, 0, eye_width, frame.height],
                           "resize": list(output_size), "resize_mode": "ISOTROPIC_AREA_NO_REMAP"}


def bind_decoded_frames(source_report: Mapping, decoded: Iterator[tuple[np.ndarray, dict]]) -> tuple[np.ndarray, list[dict]]:
    """All displayed pictures in order; no skipping, duplicating or -r adapter."""
    camera = source_report["camera"]
    if camera.get("physical_eye") != "right" or type(camera.get("sbs_index")) is not int:
        raise ValueError("SOURCE_PHYSICAL_EYE_UNRESOLVED")
    mapping = source_report["frame_mapping"]
    expected = source_report["native_sidecar_frames"]
    if len(mapping) != expected or expected <= 0:
        raise ValueError("SOURCE_FRAME_MAPPING_COUNT_REQUIRED")
    pixels, rows = [], []
    for index, (rgb, decode_info) in enumerate(decoded):
        if index >= expected:
            raise ValueError("SOURCE_MORE_DECODED_FRAMES_THAN_SIDECAR")
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError("SOURCE_RGB_DOMAIN_INVALID")
        row = dict(mapping[index])
        if row.get("native_frame_index") != index or row.get("sbs_index") != camera["sbs_index"]:
            raise ValueError("SOURCE_FRAME_MAPPING_IDENTITY_MISMATCH")
        row.update(decoded_display_index=index, sbs_source_frame=index, model_input_frame=index,
                   output_frame=index, pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                   decode=decode_info, mapping_qualification="RECORD_ORDER_COUNT_CHECKED_ACQUISITION_ID_UNKNOWN",
                   timebase_status=source_report["timebase_status"])
        pixels.append(rgb)
        rows.append(row)
    if len(pixels) != expected:
        raise ValueError("SOURCE_DECODE_SIDECAR_COUNT_MISMATCH")
    return np.stack(pixels), rows


def prepare_rgb_source(source_report: Mapping, output_dir: Path, *, component_signature: str) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or not output_dir.is_relative_to(PROJECT_ROOT) or output_dir == PROJECT_ROOT:
        raise ValueError("SOURCE_FRESH_PROJECT_OUTPUT_REQUIRED")
    if len(component_signature) != 64 or any(c not in "0123456789abcdef" for c in component_signature):
        raise ValueError("SOURCE_COMPONENT_SIGNATURE_REQUIRED")
    if source_report.get("status") != "SOURCE_PREFLIGHT_CANDIDATE":
        raise ValueError("SOURCE_PREFLIGHT_REQUIRED")
    refs = source_report["actual_inputs"]
    contracts = source_report.get("adopted_route_contracts", [])
    for ref in [*refs.values(), *contracts]:
        verify_ref(ref)
    video = Path(refs["video"]["path"])
    rgb, mapping = bind_decoded_frames(source_report, decode_physical_eye(video, source_report["camera"]["sbs_index"]))
    # Detect drift through the real decode, not just a cached source signature.
    for ref in [*refs.values(), *contracts]:
        verify_ref(ref)
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "RAW_RGB.npz", rgb=rgb)
    atomic_json(output_dir / "FRAME_MAP.json", mapping)
    atomic_json(output_dir / "SOURCE_REPORT.json", dict(source_report))
    receipt = {"schema": "inpainting-candidate-component-v1", "stage": "Source",
               "status": "SEALED_COMPLETE", "component_signature": component_signature,
               "session_id": source_report["session_id"], "native_frames": len(rgb),
               "actual_inputs": [*refs.values(), *contracts],
               "outputs": [artifact_ref(output_dir / name) for name in ("RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json")],
               "read_policy": "ORIGINAL_CAPTURE_ONLY_NO_OLD_DERIVED_PIXELS",
               "timebase_status": source_report["timebase_status"],
               "frame_mapping_qualified": False, "acquisition_frame_id": "UNKNOWN",
               "physical_eye_verified": False, "eye_route_status": source_report["camera"]["status"],
               "projection_qualified": False, "control_ground_truth": False,
               "training_eligible": False, "completed_at": now_iso()}
    atomic_json(output_dir / "RECEIPT.json", receipt)
    return receipt


def write_review_video(path: Path, frames_rgb: np.ndarray, mapping: list[dict]) -> dict:
    """Encode all frames at parent-declared host-QPC intervals, explicitly unverified.

    The rate argument is codec setup only. Each frame's PTS is set independently;
    verification checks decoded count and PTS, not a nominal FPS assumption.
    """
    import av
    path = Path(path).resolve()
    if path.exists() or not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("REVIEW_FRESH_PROJECT_OUTPUT_REQUIRED")
    if frames_rgb.dtype != np.uint8 or frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3 or len(mapping) != len(frames_rgb) or not len(mapping):
        raise ValueError("REVIEW_FRAMES_MAPPING_REQUIRED")
    qpc = np.array([row["qpc_declared_ns"] for row in mapping], dtype=np.int64)
    if np.any(np.diff(qpc) <= 0):
        raise ValueError("REVIEW_QPC_BACKWARD_OR_DUPLICATE")
    pts = (qpc - qpc[0] + 500) // 1000
    if np.any(np.diff(pts) <= 0):
        raise ValueError("REVIEW_PTS_QUANTIZATION_COLLISION")
    tick = Fraction(1, 1_000_000)
    height, width = frames_rgb.shape[1:3]
    with av.open(str(path), "w") as container:
        declared_rate = Fraction(1_000_000_000, int(np.median(np.diff(qpc)))) if len(qpc) > 1 else Fraction(30, 1)
        stream = container.add_stream("libx264", rate=declared_rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = stream.codec_context.time_base = tick
        stream.codec_context.thread_count = 2
        stream.options = {"crf": "12", "preset": "fast", "bf": "0"}
        for pixels, timestamp in zip(frames_rgb, pts):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = int(timestamp), tick
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    recovered = []
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for frame in container.decode(video=0):
            if frame.pts is None or (frame.width, frame.height) != (width, height):
                raise ValueError("REVIEW_DECODE_DOMAIN_OR_PTS_FAILED")
            recovered.append(float(frame.pts * frame.time_base))
        recovered_tick = float(container.streams.video[0].time_base)
    if len(recovered) != len(mapping):
        raise ValueError("REVIEW_DECODE_COUNT_FAILED")
    error = np.max(np.abs(np.asarray(recovered) - pts / 1e6))
    if error > max(float(tick), recovered_tick) + 1e-9:
        raise ValueError("REVIEW_DECODE_PTS_FAILED")
    return {"video": artifact_ref(path), "decoded_frames": len(recovered),
            "pts_roundtrip_max_error_seconds": float(error),
            "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
            "causal_training_input": False, "visual_quality": "NOT_REVIEWED"}
