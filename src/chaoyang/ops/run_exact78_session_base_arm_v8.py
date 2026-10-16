#!/usr/bin/env python3
"""Integration-only repair for the v7 causal Arm result enrichment."""

from __future__ import annotations

from chaoyang.ops import run_exact78_session_base_arm_v7 as previous


def main() -> int:
    # v7's post-solve enrichment accidentally addresses error/margins one
    # module too high.  Bind the already-pinned implementation at the name v7
    # expects; the solver, models and all numeric gates remain unchanged.
    arm_module = previous.base.arm
    old_error = getattr(arm_module, "error", None)
    old_margins = getattr(arm_module, "margins", None)
    arm_module.error = arm_module.arm.error
    arm_module.margins = arm_module.arm.margins
    try:
        return previous.main()
    finally:
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
