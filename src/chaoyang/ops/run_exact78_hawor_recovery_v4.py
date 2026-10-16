#!/usr/bin/env python3
"""HaWoR recovery adapter that also restores the vendor-relative MANO cwd."""
from __future__ import annotations

import os

from chaoyang.ops.run_exact78_hawor_recovery_v3 import REPO_ROOT
from chaoyang.ops.run_exact78_hawor_recovery_v2 import main as delegate


def main() -> int:
    os.chdir(REPO_ROOT / "vendor/HaWoR")
    return delegate()


if __name__ == "__main__":
    raise SystemExit(main())
