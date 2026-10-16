#!/usr/bin/env python3
"""Three-session P1 replay using the collapsed-bound-safe v9 Arm runner."""

from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_cross_session_replay_v1 as previous


def main() -> int:
    original_cli = previous.base.cli

    def corrected_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        if operation == "run_exact78_session_base_arm_v8":
            operation = "run_exact78_session_base_arm_v9"
        return original_cli(operation, *args, hawor=hawor)

    previous.base.cli = corrected_cli
    try:
        return previous.main()
    finally:
        previous.base.cli = original_cli


if __name__ == "__main__":
    raise SystemExit(main())
