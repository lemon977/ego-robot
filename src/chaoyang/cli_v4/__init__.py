"""Unregistered manifest-fenced two-session Exact78 one-submit candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.cli_v4.exact78_raw_to_robot_manifest_candidate_v1 import (
    MANIFEST, plan_manifest, plan_one, submit_one,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chaoyang.cli_v4")
    parser.add_argument("command", choices=("submit",))
    parser.add_argument("recipe", choices=("exact78-raw-to-robot",))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        if args.manifest.resolve() != MANIFEST or args.expected_input_sha256 is not None:
            parser.error("ONLY_FROZEN_V4_MANIFEST_SUPPORTED")
        if not args.check_only:
            parser.error("MANIFEST_BATCH_DISPATCH_NOT_AUTHORIZED")
    elif args.expected_input_sha256 is None:
        parser.error("--expected-input-sha256 is required with --session")
    try:
        result = (plan_manifest() if args.manifest is not None else
                  plan_one(args.session, args.expected_input_sha256) if args.check_only else
                  submit_one(args.session, args.expected_input_sha256))
    except (ColdStartInputError, RuntimeError) as error:
        print(json.dumps({"schema_version": "exact78-newraw-submit-dag-candidate-v4",
                          "status": "REJECTED", "reason": str(error),
                          "general_operation": None, "scene_clean": "NOT_WIRED"},
                         ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]
