"""Candidate successor for exact ProPainter MP4 structure validation.

This is an independent helper, not a change to the sealed fresh frontend v2.
Only encoded-CFR display timing is checked; no capture/QPC/world-time authority.
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, BinaryIO, Iterator

import cv2

from chaoyang.governance.common import artifact_ref


class Mp4AcceptanceError(RuntimeError):
    """A video cannot be sealed as one complete, correctly timed model output."""


def _boxes(stream: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    cursor = start
    while cursor + 8 <= end:
        stream.seek(cursor)
        header = stream.read(8)
        if len(header) != 8:
            raise Mp4AcceptanceError("MP4_BOX_HEADER_TRUNCATED")
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                raise Mp4AcceptanceError("MP4_EXTENDED_BOX_TRUNCATED")
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - cursor
        if size < header_size or cursor + size > end:
            raise Mp4AcceptanceError("MP4_BOX_BOUNDARY_INVALID")
        yield kind, cursor + header_size, cursor + size
        cursor += size
    if cursor != end:
        raise Mp4AcceptanceError("MP4_TRAILING_BOX_BYTES")


def _movie_header(path: Path) -> tuple[int, int]:
    """Return exact ``mvhd`` movie timescale and duration tick count."""
    with path.open("rb") as stream:
        total = path.stat().st_size
        movies = [(start, end) for kind, start, end in _boxes(stream, 0, total)
                  if kind == b"moov"]
        if len(movies) != 1:
            raise Mp4AcceptanceError("MP4_MOVIE_BOX_COUNT")
        headers = [(start, end) for kind, start, end in _boxes(stream, *movies[0])
                   if kind == b"mvhd"]
        if len(headers) != 1:
            raise Mp4AcceptanceError("MP4_MOVIE_HEADER_COUNT")
        start, end = headers[0]
        stream.seek(start)
        version_flags = stream.read(4)
        if len(version_flags) != 4:
            raise Mp4AcceptanceError("MP4_MOVIE_HEADER_TRUNCATED")
        version = version_flags[0]
        if version == 0:
            stream.seek(start + 12)
            length, width = 8, ">II"
            unknown_duration = 0xFFFFFFFF
        elif version == 1:
            stream.seek(start + 20)
            length, width = 12, ">IQ"
            unknown_duration = 0xFFFFFFFFFFFFFFFF
        else:
            raise Mp4AcceptanceError("MP4_MOVIE_HEADER_VERSION")
        if stream.tell() + length > end:
            raise Mp4AcceptanceError("MP4_MOVIE_HEADER_TRUNCATED")
        fields = stream.read(length)
        if len(fields) != length:
            raise Mp4AcceptanceError("MP4_MOVIE_HEADER_TRUNCATED")
        timescale, duration_ticks = struct.unpack(width, fields)
        if timescale <= 0 or duration_ticks == unknown_duration:
            raise Mp4AcceptanceError("MP4_MOVIE_TIMEBASE_INVALID")
        return timescale, duration_ticks


def _probe(path: Path) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,time_base,duration_ts,duration,nb_frames:format=duration:frame=best_effort_timestamp,pkt_duration",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise Mp4AcceptanceError(f"FFPROBE_FAILED:{result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout)
    except ValueError as error:
        raise Mp4AcceptanceError("FFPROBE_JSON_INVALID") from error
    if not isinstance(payload, dict):
        raise Mp4AcceptanceError("FFPROBE_JSON_INVALID")
    return payload


def _display_decimal_matches(value: str, exact: Fraction) -> bool:
    """Check FFprobe's decimal formatting, not an arbitrary seconds tolerance."""
    if not isinstance(value, str) or not value:
        return False
    precision = len(value.split(".", 1)[1]) if "." in value else 0
    if precision < 6:
        return False
    try:
        observed = Fraction(value)
    except ValueError:
        return False
    return abs(observed - exact) <= Fraction(1, 2 * 10**precision)


def verify_propainter_mp4(
    path: Path, frame_count: int, fps: int, *,
    expected_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require exact stream ticks, frame PTS, full decode and movie-time quantization.

    ``format.duration`` is checked against the parsed MP4 ``mvhd`` rational;
    ``mvhd`` itself must be the nearest movie tick to the exact stream duration.
    No frame may be inserted or dropped to make a video pass.
    """
    if (type(frame_count) is not int or frame_count <= 0
            or type(fps) is not int or fps <= 0):
        raise Mp4AcceptanceError("EXPECTED_FRAME_DOMAIN_INVALID")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Mp4AcceptanceError("MP4_MISSING_OR_LINK")
    try:
        record = _probe(path)
        streams, frames = record["streams"], record["frames"]
        if len(streams) != 1 or len(frames) != frame_count:
            raise Mp4AcceptanceError("STREAM_OR_FRAME_COUNT")
        video = streams[0]
        if (int(video["width"]), int(video["height"])) != (960, 720):
            raise Mp4AcceptanceError("FRAME_DIMENSIONS")
        if (Fraction(video["avg_frame_rate"]) != fps
                or Fraction(video["r_frame_rate"]) != fps):
            raise Mp4AcceptanceError("FRAME_RATE")
        if "nb_frames" in video and video["nb_frames"] != "N/A" and int(video["nb_frames"]) != frame_count:
            raise Mp4AcceptanceError("STREAM_FRAME_COUNT")
        time_base = Fraction(video["time_base"])
        if time_base <= 0:
            raise Mp4AcceptanceError("STREAM_TIMEBASE")
        period_ticks = Fraction(1, fps) / time_base
        if period_ticks.denominator != 1:
            raise Mp4AcceptanceError("FRAME_PERIOD_NOT_INTEGER_TICKS")
        period = int(period_ticks)
        duration_ticks = int(video["duration_ts"])
        if duration_ticks != frame_count * period:
            raise Mp4AcceptanceError("STREAM_DURATION_TICKS")
        expected_duration = Fraction(frame_count, fps)
        if not _display_decimal_matches(video["duration"], expected_duration):
            raise Mp4AcceptanceError("STREAM_DURATION_DECIMAL")
        for index, frame in enumerate(frames):
            if int(frame["best_effort_timestamp"]) != index * period:
                raise Mp4AcceptanceError(f"FRAME_PTS:{index}")
            if int(frame["pkt_duration"]) != period:
                raise Mp4AcceptanceError(f"FRAME_DURATION:{index}")
        movie_timescale, movie_ticks = _movie_header(path)
        movie_target = expected_duration * movie_timescale
        nearest_tick = int(movie_target + Fraction(1, 2))
        if movie_ticks != nearest_tick:
            raise Mp4AcceptanceError("MOVIE_DURATION_QUANTIZATION")
        movie_duration = Fraction(movie_ticks, movie_timescale)
        if not _display_decimal_matches(record["format"]["duration"], movie_duration):
            raise Mp4AcceptanceError("FORMAT_DURATION_DECIMAL")
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        raise Mp4AcceptanceError(f"MP4_PROBE_DOMAIN:{error}") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise Mp4AcceptanceError("MP4_DECODE_OPEN")
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape != (720, 960, 3):
                raise Mp4AcceptanceError(f"MP4_DECODE_FRAME_DOMAIN:{decoded}")
            decoded += 1
            if decoded > frame_count:
                raise Mp4AcceptanceError("MP4_DECODE_EXTRA_FRAME")
    finally:
        capture.release()
    if decoded != frame_count:
        raise Mp4AcceptanceError(f"MP4_DECODE_MISSING_FRAME:{decoded}")
    reference = artifact_ref(path)
    if expected_ref is not None and reference != expected_ref:
        raise Mp4AcceptanceError("MP4_SHA_OR_BYTES_DRIFT")
    return {"artifact": reference, "decoded_frames": decoded, "fps": fps,
            "width": 960, "height": 720, "stream_time_base": str(time_base),
            "frame_period_ticks": period, "stream_duration_ticks": duration_ticks,
            "movie_timescale": movie_timescale, "movie_duration_ticks": movie_ticks,
            "duration_rational": str(expected_duration),
            "format_duration_reported": record["format"]["duration"],
            "time_scope": "ENCODED_CFR_DISPLAY_ONLY_NOT_CAPTURE_TIME"}


__all__ = ["Mp4AcceptanceError", "verify_propainter_mp4"]
