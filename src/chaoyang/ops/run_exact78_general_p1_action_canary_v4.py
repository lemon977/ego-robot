#!/usr/bin/env python3
"""P1 integration retry using the cwd-correct HaWoR recovery adapter."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_action_canary_v1 as base


def main() -> int:
    original_cli = base.cli

    def retry_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        if operation == "run_exact78_hawor_recovery_v2":
            operation = "run_exact78_hawor_recovery_v4"
        return original_cli(operation, *args, hawor=hawor)

    base.cli = retry_cli
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
