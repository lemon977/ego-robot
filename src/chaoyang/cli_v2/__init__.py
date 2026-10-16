"""Unwired, versioned Exact78 one-session candidate CLI.

This module does not replace ``chaoyang.cli`` or its installed entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v2.exact78_raw_to_robot_candidate import plan_one, submit_one
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v2")
    parser.add_argument("command", choices=("submit",))
    parser.add_argument("recipe", choices=("exact78-raw-to-robot",))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-input-sha256", required=True,
                        help="pre-frozen 64-character submission input SHA-256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        parser.error("MANIFEST_NOT_WIRED_ONE_SESSION_ONLY")
    assert args.session is not None
    try:
        result = (plan_one(args.session, args.expected_input_sha256) if args.check_only
                  else submit_one(args.session, args.expected_input_sha256))
    except ColdStartInputError as error:
        print(json.dumps({"schema_version": "exact78-newraw-submit-dag-candidate-v1",
                          "status": "REJECTED", "reason": str(error),
                          "general_operation": None, "scene_clean": "NOT_WIRED"},
                         ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]
