#!/usr/bin/env python3
"""Bounded recovery successor for the W0 HaWoR entry-point import failure."""

from __future__ import annotations

from chaoyang.ops import run_0915_robot15h_hawor_wave0_v1 as base


base.TASK_ID = "0915_robot15h_hawor_wave0_recovery_v1"
base.PHASE = "ROBOT15H_HAWOR_WAVE0_RESIZE_ONLY_RECOVERY"
base.OUTPUT = base.ROOT / "_run/current/0915_robot15h_hawor_wave0_recovery_v1/attempts/attempt_0001"
base.VISUAL = base.ROOT / "docs/current/visuals/0915_ROBOT15H_W0_HAWOR_RECOVERY_V1"
base.TERMINAL_RECEIPT = base.ROOT / "tasks/receipts/0915_ROBOT15H_HAWOR_WAVE0_RECOVERY_V1_RESULT.json"


def main() -> int:
    """Delegate to the shared runner after binding the recovery namespaces."""

    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
