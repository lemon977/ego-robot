#!/usr/bin/env python3
"""Corrected P1 per-session arm wrapper.

This successor changes no solver, model, or numeric threshold.  It only binds
the pose-error helpers used by the v3 visual-trajectory postprocessor to the
actual low-level arm module.  Numeric misses remain per-frame statuses and do
not turn a completed session into a runtime failure.
"""
from __future__ import annotations

import json

from chaoyang.governance.common import atomic_json
from chaoyang.ops import run_exact78_session_base_arm_v3 as previous


def main() -> int:
    wrapper = previous.base.arm
    old_error = getattr(wrapper, "error", None)
    old_margins = getattr(wrapper, "margins", None)
    wrapper.error = wrapper.arm.error
    wrapper.margins = wrapper.arm.margins
    try:
        code = previous.main()
    finally:
        if old_error is None:
            del wrapper.error
        else:
            wrapper.error = old_error
        if old_margins is None:
            del wrapper.margins
        else:
            wrapper.margins = old_margins
    if code != 0:
        return code

    result_path = previous._output_root() / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema_version"] = "exact78-session-base-arm-v4"
    result["runtime_correction"] = {
        "changed": "postprocessor helper binding only",
        "numeric_thresholds_changed": False,
        "solver_changed": False,
        "model_changed": False,
        "frame_failure_semantics": "per-frame status; process exit remains zero",
    }
    atomic_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
