#!/usr/bin/env python3
"""HaWoR recovery adapter normalizing loaded SLAM tensors to float32."""
from __future__ import annotations

import os

from chaoyang.ops.run_exact78_hawor_recovery_v3 import REPO_ROOT
from chaoyang.ops import run_exact78_hawor_recovery_v2 as base


def main() -> int:
    os.chdir(REPO_ROOT / "vendor/HaWoR")
    from scripts.scripts_test_video import hawor_video
    original = hawor_video.load_slam_cam

    def load_slam_cam(path):
        return tuple(value.float() for value in original(path))

    hawor_video.load_slam_cam = load_slam_cam
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
