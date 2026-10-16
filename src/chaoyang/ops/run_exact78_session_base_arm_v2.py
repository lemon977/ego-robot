#!/usr/bin/env python3
"""P1 Arm adapter: keep temporal-bound numeric failures frame-local."""
from __future__ import annotations

import numpy as np

from chaoyang.ops import run_exact78_session_base_arm_v1 as base


def main() -> int:
    original = base.arm.temporal.bounded_limits

    def bounded_limits(lower: np.ndarray, upper: np.ndarray, previous: np.ndarray,
                       previous_previous: np.ndarray | None, step_limit: float):
        clipped = np.clip(previous, lower, upper)
        older = None if previous_previous is None else np.clip(previous_previous, lower, upper)
        try:
            return original(lower, upper, clipped, older, step_limit)
        except base.arm.temporal.TemporalReviewError as error:
            # A contradictory acceleration intersection is a frame-local hold,
            # never a reason to terminate the whole session.
            print(f"ARM_TEMPORAL_BOUND_HOLD:{error}", flush=True)
            return clipped.copy(), clipped.copy()

    base.arm.temporal.bounded_limits = bounded_limits
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
