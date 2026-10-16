#!/usr/bin/env python3
"""Exact78 HaWoR raw adapter preserving native source-frame identity.

Upstream HaWoR extracts every video at 30 fps.  That duplicates frames for
the 43 native-25-fps Exact78 sessions and breaks the frozen frame contract.
This adapter changes only extraction cadence: every decoded input frame is
written once.  Detector/tracker/model parameters and quality thresholds are
unchanged.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

from chaoyang.ops import run_exact78_hawor_raw_v1 as base


def extract_native_frames(video_path: str, output_folder: str) -> None:
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", video_path,
            "-vsync", "0", "-start_number", "0",
            str(destination / "%04d.jpg"),
        ],
        check=True,
    )


def main() -> int:
    sys.path.insert(0, str(base.HAWOR))
    detector = importlib.import_module(
        "scripts.scripts_test_video.detect_track_video"
    )
    detector.extract_frames = extract_native_frames
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
