#!/usr/bin/env python3
"""Causal Arm postprocess with collapsed-bound-safe least squares."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from chaoyang.ops import run_exact78_arm_causal_visual_v2 as previous


def solve_one(assets, side, target_base, lower, upper, seeds,
              previous_q=None, lo=None, hi=None, **kwargs):
    if previous_q is None:
        previous_q = kwargs.get("previous")
    if lo is None:
        lo, hi = lower.copy(), upper.copy()
    attempts = []
    for seed_index, seed in enumerate(seeds):
        seed = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)

        def residual(x):
            pose = previous.base.arm.old.official._pose_residual(
                previous.base.arm.old.official._tool_fk(assets, side, x), target_base)
            branch = np.minimum(
                previous.base.arm.arm.margins(assets, side, x) - 0.003, 0.0
            ) / 0.001
            continuity = (
                np.empty(0) if previous_q is None
                else 0.01 * (x - previous_q)
            )
            return np.concatenate((pose, branch, continuity))

        free = hi - lo > 1e-12
        x = seed.copy()
        if np.any(free):
            def expand(values):
                expanded = x.copy()
                expanded[free] = values
                return expanded

            solved = least_squares(
                lambda values: residual(expand(values)),
                x[free], bounds=(lo[free], hi[free]), max_nfev=500,
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            x[free] = solved.x
        position, rotation = previous.base.arm.arm.error(
            assets, side, x, target_base)
        margins = previous.base.arm.arm.margins(assets, side, x)
        failed = not (
            position <= previous.base.POSITION_GATE_MM
            and rotation <= previous.base.ROTATION_GATE_DEG
            and float(np.min(margins)) >= -1e-8
        )
        score = (
            failed,
            max(0.0, -float(np.min(margins))),
            max(0.0, position - previous.base.POSITION_GATE_MM)
            + max(0.0, rotation - previous.base.ROTATION_GATE_DEG),
            position + rotation,
        )
        attempts.append((*score, x, seed_index, position, rotation, margins))
    return min(attempts, key=lambda row: row[:4])


def main() -> int:
    original = previous.base.arm.solve_one
    previous.base.arm.solve_one = solve_one
    try:
        return previous.main()
    finally:
        previous.base.arm.solve_one = original


if __name__ == "__main__":
    raise SystemExit(main())
