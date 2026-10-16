#!/usr/bin/env python3
"""P1 canary with corrected mount frame, flange ring and joint-local bounds."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_action_canary_v10 as previous


def main() -> int:
    runner = previous.previous.previous.base
    original_cli = runner.cli

    def corrected_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        replacements = {
            "run_exact78_session_base_arm_v4": "run_exact78_session_base_arm_v5",
            "render_exact78_action_diagnostics_v1": "render_exact78_action_diagnostics_v2",
        }
        return original_cli(replacements.get(operation, operation), *args, hawor=hawor)

    runner.cli = corrected_cli
    try:
        return previous.main()
    finally:
        runner.cli = original_cli


if __name__ == "__main__":
    raise SystemExit(main())
