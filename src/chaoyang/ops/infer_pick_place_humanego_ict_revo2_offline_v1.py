"""Fenced HumanEgo offline inference only; never sends robot commands."""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from chaoyang.humanego_ict.derived_runtime import main as run
    from chaoyang.humanego_ict.derived_contract import INFER
    return run(INFER, argv)
