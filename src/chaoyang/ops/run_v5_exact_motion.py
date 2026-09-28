"""Registered thin adapter for frozen 0902 exact-domain motion and Robot R0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.pipeline.v5_exact_motion import recover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.config), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
