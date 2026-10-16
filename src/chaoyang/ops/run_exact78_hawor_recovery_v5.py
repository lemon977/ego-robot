#!/usr/bin/env python3
"""HaWoR recovery adapter with float32 official-infiller camera input."""
from __future__ import annotations

import os

import numpy as np

from chaoyang.ops.run_exact78_hawor_recovery_v3 import REPO_ROOT
from chaoyang.ops import run_exact78_hawor_recovery_v2 as base


def main() -> int:
    os.chdir(REPO_ROOT / "vendor/HaWoR")
    original = base.official_infiller

    def official_infiller(arr, masks, staged_video, upstream_work, mano_functions, audit):
        original_c2w = arr["c2w"]
        arr["c2w"] = np.asarray(original_c2w, dtype=np.float32)
        try:
            return original(arr, masks, staged_video, upstream_work, mano_functions, audit)
        finally:
            arr["c2w"] = original_c2w

    base.official_infiller = official_infiller
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
