#!/usr/bin/env python3
"""P1 retry using float32-loaded SLAM tensors and frame-local Arm bounds."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_action_canary_v1 as base


def main() -> int:
    original_cli = base.cli

    def retry_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        replacements = {
            "run_exact78_hawor_recovery_v2": "run_exact78_hawor_recovery_v6",
            "run_exact78_session_base_arm_v1": "run_exact78_session_base_arm_v2",
        }
        return original_cli(replacements.get(operation, operation), *args, hawor=hawor)

    base.cli = retry_cli
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
