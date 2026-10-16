#!/usr/bin/env python3
"""P1 canary integration using the non-recursive mount adapter."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_action_canary_v11 as previous


def main() -> int:
    runner = previous.previous.previous.previous.base
    original_cli = runner.cli

    def corrected_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        if operation == "run_exact78_session_base_arm_v5":
            operation = "run_exact78_session_base_arm_v6"
        return original_cli(operation, *args, hawor=hawor)

    runner.cli = corrected_cli
    try:
        return previous.main()
    finally:
        runner.cli = original_cli


if __name__ == "__main__":
    raise SystemExit(main())
