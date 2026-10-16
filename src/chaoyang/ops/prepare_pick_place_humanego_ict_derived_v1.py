"""Separate, fenced derived-supervision preparation candidate."""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from chaoyang.humanego_ict.derived_runtime import main as run
    from chaoyang.humanego_ict.derived_contract import PREPARE
    return run(PREPARE, argv)
