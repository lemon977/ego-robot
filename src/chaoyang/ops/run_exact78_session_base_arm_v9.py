#!/usr/bin/env python3
"""Repair v7 enrichment and collapsed-bound integration without gate changes."""

from __future__ import annotations

from chaoyang.ops import run_exact78_session_base_arm_v7 as previous
from chaoyang.ops.run_exact78_arm_causal_visual_v3 import solve_one as collapsed_safe_solve_one


def main() -> int:
    arm_module = previous.base.arm
    old_error = getattr(arm_module, "error", None)
    old_margins = getattr(arm_module, "margins", None)
    old_solve_one = arm_module.solve_one
    arm_module.error = arm_module.arm.error
    arm_module.margins = arm_module.arm.margins
    arm_module.solve_one = collapsed_safe_solve_one
    try:
        return previous.main()
    finally:
        arm_module.solve_one = old_solve_one
        if old_error is None:
            delattr(arm_module, "error")
        else:
            arm_module.error = old_error
        if old_margins is None:
            delattr(arm_module, "margins")
        else:
            arm_module.margins = old_margins


if __name__ == "__main__":
    raise SystemExit(main())
