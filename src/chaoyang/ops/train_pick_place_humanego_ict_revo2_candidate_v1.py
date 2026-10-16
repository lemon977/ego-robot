"""Thin registered candidate entry. Admission precedes all model imports."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from chaoyang.humanego_ict.cli import main as candidate_main
    return candidate_main(["train", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
