"""Single-session Cards004 candidate submit; execution requires a new v6 acceptance receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v6.exact78_cards004_submit_candidate_v1 import plan_one, submit_one
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v6")
    parser.add_argument("command", choices=("submit",))
    parser.add_argument("recipe", choices=("exact78-raw-to-robot",))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (plan_one(args.session, args.expected_input_sha256)
                  if args.check_only else
                  submit_one(args.session, args.expected_input_sha256))
    except (ColdStartInputError, RuntimeError) as error:
        print(json.dumps({"status": "REJECTED", "reason": str(error),
                          "general_operation": None, "scene_clean": "NOT_WIRED"},
                         sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]
