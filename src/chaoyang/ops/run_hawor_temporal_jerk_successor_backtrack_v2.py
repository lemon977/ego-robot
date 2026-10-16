#!/usr/bin/env python3
"""Temporal successor with one additional conservative backtracking alpha.

The generic V1 candidate list stopped at 0.03.  Fresh Cards042 HaWoR passes
all unchanged numeric gates at 0.01, so this wrapper adds exactly that alpha
without changing the proposal, metrics, thresholds, or output schema.
"""

from __future__ import annotations

import sys
from pathlib import Path


OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
import run_hawor_temporal_jerk_successor as implementation  # noqa: E402


implementation.ALPHAS = (*implementation.ALPHAS, 0.01)


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
