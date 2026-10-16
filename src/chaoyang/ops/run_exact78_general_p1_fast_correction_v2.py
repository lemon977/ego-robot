#!/usr/bin/env python3
"""Fast P1 correction using collapsed-bound-safe causal Arm IK."""
from __future__ import annotations

from chaoyang.ops import run_exact78_general_p1_fast_correction_v1 as previous


def main() -> int:
    original_cli = previous.base.cli

    def corrected_cli(operation: str, *args, hawor: bool = False) -> list[str]:
        if operation == "run_exact78_arm_causal_visual_v2":
            operation = "run_exact78_arm_causal_visual_v3"
        return original_cli(operation, *args, hawor=hawor)

    previous.base.cli = corrected_cli
    try:
        return previous.main()
    finally:
        previous.base.cli = original_cli


if __name__ == "__main__":
    raise SystemExit(main())
