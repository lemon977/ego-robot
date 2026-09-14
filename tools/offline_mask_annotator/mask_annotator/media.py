"""ffprobe frame cataloguing and deterministic exact-index extraction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeFrame:
    decoded_index: int
    pts: int | None
    pts_time: Decimal
    duration_time: Decimal | None
    key_frame: bool


@dataclass(frozen=True)
class ProbeResult:
    stream: dict[str, Any]
    frames: tuple[ProbeFrame, ...]
    ffprobe_version: str


def require_media_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MediaError("ffmpeg and ffprobe must be installed and available on PATH")
    return ffmpeg, ffprobe


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise MediaError(f"executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise MediaError(f"command failed ({command[0]}): {detail}") from exc


def _decimal(value: Any, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MediaError(f"invalid {name}: {value!r}") from exc


def probe_video(path: Path) -> ProbeResult:
    _, ffprobe = require_media_tools()
    version_line = _run([ffprobe, "-version"]).stdout.splitlines()[0]
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=index,width,height,codec_name,avg_frame_rate,r_frame_rate,time_base,duration,nb_frames:"
        "frame=media_type,best_effort_timestamp,best_effort_timestamp_time,pkt_duration_time,key_frame",
        "-show_frames",
        "-of", "json",
        str(path),
    ]
    payload = json.loads(_run(command).stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise MediaError(f"expected one selected video stream, got {len(streams)}")
    frames = []
    for index, item in enumerate(payload.get("frames", [])):
        if item.get("media_type", "video") != "video":
            continue
        pts_value = item.get("best_effort_timestamp")
        pts = int(pts_value) if pts_value not in (None, "N/A") else None
        pts_time_value = item.get("best_effort_timestamp_time")
        if pts_time_value in (None, "N/A"):
            raise MediaError(f"video frame {index} has no best-effort timestamp time")
        duration_value = item.get("pkt_duration_time")
        duration = None if duration_value in (None, "N/A") else _decimal(duration_value, "duration")
        frames.append(ProbeFrame(
            decoded_index=len(frames),
            pts=pts,
            pts_time=_decimal(pts_time_value, "PTS time"),
            duration_time=duration,
            key_frame=bool(int(item.get("key_frame", 0))),
        ))
    if not frames:
        raise MediaError("ffprobe returned no decoded video frames")
    return ProbeResult(stream=dict(streams[0]), frames=tuple(frames), ffprobe_version=version_line)


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    if total < 1 or count < 1:
        raise MediaError("total and count must be positive")
    if count > total:
        raise MediaError(f"requested {count} frames but video contains only {total}")
    if count == 1:
        return (total // 2,)
    indices = tuple((i * (total - 1) + (count - 1) // 2) // (count - 1) for i in range(count))
    if len(set(indices)) != count:
        raise MediaError("sampling produced duplicate decoded indices")
    return indices


def read_requested_timestamps(path: Path) -> tuple[Decimal, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp_seconds" not in reader.fieldnames:
            raise MediaError("timestamps CSV requires a timestamp_seconds column")
        values = []
        for row_number, row in enumerate(reader, start=2):
            value = _decimal(row.get("timestamp_seconds", ""), f"timestamp at CSV row {row_number}")
            if value < 0:
                raise MediaError(f"timestamp must be non-negative at CSV row {row_number}")
            values.append(value)
    if not values:
        raise MediaError("timestamps CSV is empty")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise MediaError("timestamps CSV must be strictly increasing")
    return tuple(values)


def resolve_timestamps(frames: tuple[ProbeFrame, ...], requested: Iterable[Decimal]) -> tuple[int, ...]:
    used: set[int] = set()
    selected = []
    for target in requested:
        candidates = sorted(frames, key=lambda frame: (abs(frame.pts_time - target), frame.decoded_index))
        match = next((frame for frame in candidates if frame.decoded_index not in used), None)
        if match is None:
            raise MediaError("not enough unique video frames for requested timestamps")
        selected.append(match.decoded_index)
        used.add(match.decoded_index)
    if any(b <= a for a, b in zip(selected, selected[1:])):
        raise MediaError("requested timestamps do not resolve to strictly increasing unique frames")
    return tuple(selected)


def select_frames(probe: ProbeResult, count: int, timestamps_csv: Path | None) -> tuple[ProbeFrame, ...]:
    if timestamps_csv is None:
        indices = evenly_spaced_indices(len(probe.frames), count)
    else:
        requested = read_requested_timestamps(timestamps_csv)
        if len(requested) != count:
            raise MediaError(
                f"timestamps CSV has {len(requested)} rows but sampling.frame_count is {count}"
            )
        indices = resolve_timestamps(probe.frames, requested)
    return tuple(probe.frames[index] for index in indices)


def extract_selected_frames(input_mp4: Path, selected: tuple[ProbeFrame, ...], output_pattern: Path) -> str:
    ffmpeg, _ = require_media_tools()
    output_pattern.parent.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{frame.decoded_index})" for frame in selected)
    command = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_mp4),
        "-vf", f"select={expression}",
        "-vsync", "0",
        "-start_number", "0",
        str(output_pattern),
    ]
    _run(command)
    version_line = _run([ffmpeg, "-version"]).stdout.splitlines()[0]
    return version_line
