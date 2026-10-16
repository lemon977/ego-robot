#!/usr/bin/env python3
"""P1 fixed-set validation using the arm-v4 runtime correction."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_action_canary_v8 as previous


def main() -> int:
    original_cli = previous.base.cli

    def corrected_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        if operation == "run_exact78_session_base_arm_v3":
            operation = "run_exact78_session_base_arm_v4"
        return original_cli(operation, *args, hawor=hawor)

    previous.base.cli = corrected_cli
    try:
        return previous.main()
    finally:
        previous.base.cli = original_cli


if __name__ == "__main__":
    raise SystemExit(main())
