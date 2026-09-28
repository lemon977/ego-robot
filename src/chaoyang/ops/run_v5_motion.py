"""Registered operation adapter for a SHA-pinned V5 motion recovery config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.pipeline.v5_motion import recover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = recover(arguments.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
